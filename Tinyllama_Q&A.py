from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# 1. Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#tokenizer.pad_token = tokenizer.eos_token  # Important for padding

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Load and format dataset
dataset = load_dataset("squad", split="train[:1%]")
print(dataset[0])
def format_qa(example):
    
    question = f"### Question: {example['question']}\n### Answer:"
    answer = f" {example['answers']['text'][0] if example['answers']['text'] else 'Not available'}"
    full_prompt = question + answer

    # Tokenize full prompt
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors=None
    )

    # Mask out question part in labels (no loss)
    q_len = len(tokenizer(question, truncation=True, max_length=128, padding=True, return_tensors=None)["input_ids"])
    labels = [-100] * q_len + tokenized["input_ids"][q_len:]

    #print(tokenized["input_ids"],"---", len(tokenized["input_ids"]))
    tokenized["labels"] = labels
    #print(tokenized["labels"], "---", len(tokenized["labels"]))
    
    return tokenized

tokenized_dataset = dataset.map(format_qa, remove_columns=dataset.column_names)

# 4. Use a proper collator to pad inputs
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM
)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama-qa",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    fp16=True,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none"
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 7. Train
trainer.train()

# 8. Save model
trainer.save_model("./tinyllama-qa")
