import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from trl import SFTTrainer

ADMISSIONS_INSTRUCTION = "You are an admissions expert. Answer only in Ukrainian or English."
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./fine-tuned-mistral"
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("Set HUGGING_FACE_HUB_TOKEN env var for gated model access.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=2, per_device_train_batch_size=4,
    gradient_accumulation_steps=4, warmup_steps=100, logging_steps=10,
    save_steps=500, fp16=True, remove_unused_columns=False
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=512,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=training_args
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Fine-tuning complete! Adapter saved.")