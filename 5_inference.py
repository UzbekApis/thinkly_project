#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Thinkly AI - Model inference (test qilish)
# Google Colab uchun

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from peft import PeftModel
import threading
from IPython.display import display, Markdown

# Loyiha papkasini aniqlash
project_dir = "/content/drive/MyDrive/thinkly_project"
model_path = f"{project_dir}/qwen25_coder_7b"
trained_model_path = f"{project_dir}/thinkly_trained"

print("Thinkly AI test qilish boshlandi...")

# 1. Modelni yuklash
def load_model():
    print("Modelni yuklash...")
    
    # 4-bit quantization konfiguratsiyasi
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Base modelni yuklash
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Tokenizer yuklab olish
    tokenizer = AutoTokenizer.from_pretrained(
        trained_model_path, 
        trust_remote_code=True
    )
    
    # LoRA adapterni yuklash
    if os.path.exists(trained_model_path):
        print(f"LoRA adapter yuklanmoqda: {trained_model_path}")
        model = PeftModel.from_pretrained(base_model, trained_model_path)
    else:
        print("Diqqat: LoRA adapter topilmadi, base model ishlatiladi")
        model = base_model
    
    # Modelni eval rejimiga o'tkazish
    model.eval()
    
    return model, tokenizer

# 2. Matn generatsiya qilish
def generate_text(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    # Prompt formati
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenizatsiya
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Streamer yaratish
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generation parametrlari
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "streamer": streamer,
        "do_sample": True,
    }
    
    # Thread yaratish
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Natijani yig'ish
    generated_text = ""
    print("Thinkly AI javob bermoqda...")
    for new_text in streamer:
        generated_text += new_text
        print(new_text, end="", flush=True)
    
    print("\n")
    return generated_text

# 3. Test savollari
def run_tests(model, tokenizer):
    test_prompts = [
        "Sen qanday AI modelisan?",
        "O'zingni tanishtir",
        "Python da 1 dan 10 gacha sonlarni ekranga chiqaradigan kod yozing",
        "HTML da oddiy forma yarating",
        "PHP da MySQL ga ulanish uchun kod yozing",
        "JavaScript da oddiy kalkulyator yarating"
    ]
    
    print("Test savollari:")
    for i, prompt in enumerate(test_prompts):
        print(f"\n=== Test {i+1}: {prompt} ===")
        response = generate_text(model, tokenizer, prompt)
        # Markdown formatida ko'rsatish
        display(Markdown(f"**Savol:** {prompt}\n\n**Javob:**\n{response}"))

# 4. Interaktiv chat
def interactive_chat(model, tokenizer):
    print("\n=== Thinkly AI bilan suhbat ===")
    print("Chiqish uchun 'exit' yoki 'quit' yozing\n")
    
    chat_history = []
    
    while True:
        user_input = input("Savol: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = generate_text(model, tokenizer, user_input)
        chat_history.append((user_input, response))
        
        print("\n")

# Asosiy funksiya
def main():
    try:
        # Modelni yuklash
        model, tokenizer = load_model()
        
        # Test savollari
        run_tests(model, tokenizer)
        
        # Interaktiv chat
        interactive_chat(model, tokenizer)
        
        print("Thinkly AI test qilish tugadi!")
    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")

if __name__ == "__main__":
    main() 