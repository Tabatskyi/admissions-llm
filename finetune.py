import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

ADMISSIONS_INSTRUCTION = "You are an admissions expert. Answer only in Ukrainian or English."
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./fine-tuned-mistral"
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("Set HUGGING_FACE_HUB_TOKEN env var for gated model access.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="admissions_dataset.jsonl", split="train")
def formatting_prompts_func(example):
    inst_str = f"[INST] {ADMISSIONS_INSTRUCTION} {example['prompt']}"
    formatted = f"<s>{inst_str} [/INST] {example['completion']} </s>"
    return {"text": formatted}

dataset = dataset.map(formatting_prompts_func)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=2, per_device_train_batch_size=4,
    gradient_accumulation_steps=4, warmup_steps=100, logging_steps=10,
    save_steps=500, fp16=True, remove_unused_columns=False
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