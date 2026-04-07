import argparse
import json
import logging
import pathlib
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE     = "data/train.jsonl"
VAL_FILE       = "data/val.jsonl"
OUTPUT_DIR     = "outputs/qwen7b-eporcha"
MAX_SEQ_LENGTH = 4096

LORA_R         = 64
LORA_ALPHA     = 128
LORA_DROPOUT   = 0.05

EPOCHS         = 3
LR             = 2e-4
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
WARMUP_RATIO   = 0.05
SAVE_STEPS     = 100

SYSTEM_PROMPT = (
    "আপনি e-porcha.com এর একজন সহায়ক AI assistant। "
    "প্রশ্ন বাংলায় হলে বাংলায় উত্তর দিন, ইংরেজিতে হলে ইংরেজিতে উত্তর দিন। "
    "প্রয়োজনে tools ব্যবহার করুন।"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply_template(examples, tokenizer):
    texts = []
    tools_list = examples.get("tools", [None] * len(examples["messages"]))
    for messages, tools in zip(examples["messages"], tools_list):
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


def truncate(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    return {
        "text": tokenizer.batch_decode(
            tokenized["input_ids"], skip_special_tokens=False
        )
    }

# ── GPU info ──────────────────────────────────────────────────────────────────

def print_gpu_info():
    if torch.cuda.is_available():
        log.info(f"GPU  : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        log.warning("No CUDA GPU found!")

# ── Train ─────────────────────────────────────────────────────────────────────

def train(resume=False, dry_run=False):
    print_gpu_info()

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    train_rows = load_jsonl(TRAIN_FILE)
    val_rows   = load_jsonl(VAL_FILE)
    train_dataset = Dataset.from_list(train_rows)
    val_dataset   = Dataset.from_list(val_rows)
    log.info(f"Train : {len(train_dataset)} samples")
    log.info(f"Val   : {len(val_dataset)} samples")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"

    # ── Apply chat template ───────────────────────────────────────────────────
    log.info("Applying chat template...")
    train_dataset = train_dataset.map(
        lambda ex: apply_template(ex, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda ex: apply_template(ex, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    # Truncate
    train_dataset = train_dataset.map(lambda ex: truncate(ex, tokenizer), batched=True)
    val_dataset   = val_dataset.map(lambda ex: truncate(ex, tokenizer),   batched=True)

    log.info(f"Sample: {train_dataset[0]['text'][:300]}")

    # ── Load model in 4-bit ───────────────────────────────────────────────────
    log.info(f"Loading model: {MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    log.info(f"Model loaded. Params: {model.num_parameters()/1e6:.0f}M")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        run_name="qwen7b-eporcha-v1",
        dataloader_num_workers=4,
        dataset_text_field="text",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    if dry_run:
        import wandb
        wandb.init(project="eporcha-finetune", name="dry-run-test")
        wandb.log({"test_metric": 1.0})
        wandb.finish()
        log.info("wandb connection OK")
        log.info("=== DRY RUN — skipping training ===")
        log.info(f"Train samples     : {len(train_dataset)}")
        log.info(f"Val samples       : {len(val_dataset)}")
        log.info(f"Steps per epoch   : {len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)}")
        log.info(f"Total steps       : {len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM) * EPOCHS}")
        log.info(f"Sample text (first 300 chars):")
        log.info(train_dataset[0]["text"][:300])
        log.info("=== DRY RUN COMPLETE — no issues found ===")
        return None, tokenizer
    else:
        trainer.train(resume_from_checkpoint=resume if resume else None)

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = f"{OUTPUT_DIR}/lora-adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    log.info(f"Adapter saved to {adapter_path}")

    # ── Quick inference test ──────────────────────────────────────────────────
    log.info("Running inference tests...")
    test_messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "খতিয়ান কী?"},
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "ঢাকা জেলার খতিয়ান নম্বর ১২৩ দেখাও"},
        ],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "What is the fee for a khatian copy?"},
        ],
    ]

    model.eval()
    for i, messages in enumerate(test_messages, 1):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        log.info(f"--- Test {i} ---")
        log.info(f"Q: {messages[-1]['content']}")
        log.info(f"A: {response}")

    return adapter_path, tokenizer





# ── Merge ─────────────────────────────────────────────────────────────────────

def merge(adapter_path=None, tokenizer=None):
    if adapter_path is None:
        adapter_path = f"{OUTPUT_DIR}/lora-adapter"
    if not pathlib.Path(adapter_path).exists():
        log.error(f"Adapter not found at {adapter_path}")
        sys.exit(1)

    log.info("Merging adapter into base model (on CPU)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = merged_model.merge_and_unload()

    merged_path = f"{OUTPUT_DIR}/merged"
    merged_model.save_pretrained(merged_path)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(merged_path)

    log.info(f"Merged model saved to {merged_path}")
    log.info("Next: convert to GGUF with llama.cpp for Ollama")





# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Test setup without training")
    parser.add_argument("--merge",  action="store_true", help="Merge adapter after training")
    args = parser.parse_args()

    adapter_path, tokenizer = train(resume=args.resume, dry_run=args.dry_run)

    if args.merge and not args.dry_run:
        merge(adapter_path, tokenizer)