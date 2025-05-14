from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch


model_path = "./tinyllama-qa"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
# 2. โหลด config PEFT
peft_config = PeftConfig.from_pretrained(model_path)

# 3. โหลด base model และ tokenizer
#base_model = AutoModelForCausalLM.from_pretrained(
#    peft_config.base_model_name_or_path,
#    device_map="auto",
#    torch_dtype=torch.float16
#)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, use_fast=True)

# 4. โหลด LoRA adapter ที่ fine-tuned
model = PeftModel.from_pretrained(base_model, model_path)

# 5. ตั้งเป็น eval mode
model.eval()

# 6. ฟังก์ชันถาม-ตอบ
def answer_question(question):
    prompt = f"### Question: {question}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.9,
            temperature=0.1
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🔹 Model Output:\n", output_text)

# 7. ตัวอย่างการใช้งาน
if __name__ == "__main__":
    #question = "To whom did the Virgin Mary allegedly appear in Lourdes France?"
    question = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
    #question = "What is the capital of France?"
    answer_question(question)
