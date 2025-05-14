from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

# ========== กำหนด path ไปยังโมเดลที่ fine-tuned ==========
fine_tuned_model_path = "./tinyllama-finetuned"

# ========== โหลด PEFT config ==========
peft_config = PeftConfig.from_pretrained(fine_tuned_model_path)

# ========== โหลด tokenizer และ base model ==========
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
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

# ========== โหลด LoRA weights ==========
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model.eval()

# ========== ฟังก์ชันสำหรับทำ inference ==========
def generate_response(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ========== ทดสอบ ==========
prompt = "The greatest glory in living lies not in"
response = generate_response(prompt)
print("Generated:", response)
