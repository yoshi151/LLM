from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, TaskType

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)


peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]  # สำหรับ TinyLlama
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

from datasets import load_dataset

# ตัวอย่าง dataset
dataset = load_dataset("Abirate/english_quotes", split="train[:1%]")
print(dataset[0])
def tokenize(example):
    tokenized = tokenizer(
        example["quote"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    print(tokenized["input_ids"])
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True)
exit()
# Fine-Tuning
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    learning_rate=2e-4,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
trainer.save_model("./tinyllama-finetuned") 
