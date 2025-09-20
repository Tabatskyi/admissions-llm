import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling

ADMISSIONS_INSTRUCTION = os.getenv(
    "ADMISSIONS_INSTRUCTION",
    "You are an admissions expert. Answer only in Ukrainian or English.",
)

MODEL_NAME = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./fine-tuned-mistral")
# defaults to <40xx gpus
USE_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() in {"1", "true", "yes"}
USE_FP16 = os.getenv("FP16", "true").lower() in {"1", "true", "yes"}
USE_BF16 = os.getenv("BF16", "false").lower() in {"1", "true", "yes"}
BNB_COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if token:
    try:
        login(token=token)
    except Exception:
        pass
else:
    raise ValueError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN env var for gated model access.")

os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "/root/.cache/huggingface"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

device_map = "auto" if USE_4BIT else None
dtype = (torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else None))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config if USE_4BIT else None,
    device_map=device_map,
    dtype=dtype,
    low_cpu_mem_usage=True,
)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

lora_r = int(os.getenv("LORA_R", "16"))
lora_alpha = int(os.getenv("LORA_ALPHA", "32"))
lora_dropout = float(os.getenv("LORA_DROPOUT", "0.05"))
target_modules = os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj").split(",")

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="admissions_dataset.jsonl", split="train")
def formatting_prompts_func(example):
    inst_str = f"[INST] {ADMISSIONS_INSTRUCTION} {example['prompt']}"
    formatted = f"<s>{inst_str} [/INST] {example['completion']} </s>"
    return {"text": formatted}

dataset = dataset.map(formatting_prompts_func)

max_length = int(os.getenv("MAX_LENGTH", "256"))

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
# defaults only good for smoke testing
num_train_epochs = float(os.getenv("NUM_EPOCHS", "1"))
per_device_train_batch_size = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "4"))
gradient_accumulation_steps = int(os.getenv("GRAD_ACCUM_STEPS", "2"))
warmup_steps = int(os.getenv("WARMUP_STEPS", "0"))
logging_steps = int(os.getenv("LOGGING_STEPS", "5"))
save_steps = int(os.getenv("SAVE_STEPS", "0"))
max_steps = int(os.getenv("MAX_STEPS", "-1"))
learning_rate = float(os.getenv("LEARNING_RATE", "2e-4"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "0.01"))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,
    logging_steps=logging_steps,
    save_steps=save_steps if save_steps > 0 else 10_000_000,  # effectively disable save by steps when 0
    save_total_limit=int(os.getenv("SAVE_TOTAL_LIMIT", "1")),
    fp16=USE_FP16 and not USE_BF16,
    bf16=USE_BF16,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    report_to=["none"],
    eval_strategy="no",
    save_strategy="steps" if save_steps > 0 else "no",
    remove_unused_columns=False,
    dataloader_num_workers=int(os.getenv("DATALOADER_WORKERS", "2")),
    max_steps=max_steps,
    optim=os.getenv("OPTIM", "paged_adamw_8bit"),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Fine-tuning complete! Adapter saved.")