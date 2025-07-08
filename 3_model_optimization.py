#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Thinkly AI - Model optimizatsiyasi
# Google Colab uchun

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Loyiha papkasini aniqlash
project_dir = "/content/drive/MyDrive/thinkly_project"
model_path = f"{project_dir}/qwen25_coder_7b"

print("Model optimizatsiyasi boshlandi...")

# 1. Xotira tekshirish
def check_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU mavjud: {torch.cuda.get_device_name(0)}")
        print(f"Umumiy VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Bo'sh VRAM: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("GPU mavjud emas!")
        return False
    return True

# 2. 4-bit quantization konfiguratsiyasi
def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return bnb_config

# 3. LoRA konfiguratsiyasi
def get_lora_config():
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora_config

# 4. Modelni yuklash va optimizatsiya qilish
def load_and_optimize_model():
    print("Modelni yuklash...")
    
    # Tokenizer yuklab olish
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # Model konfiguratsiyasi
    bnb_config = get_bnb_config()
    
    # Modelni yuklash (4-bit quantization bilan)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Gradient checkpointing yoqish
    model.gradient_checkpointing_enable()
    
    # Model nomini o'zgartirish
    model.config.model_type = "thinkly"
    
    # LoRA uchun modelni tayyorlash
    model = prepare_model_for_kbit_training(model)
    
    # LoRA qo'shish
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Parametrlar sonini ko'rsatish
    print(f"Modelning umumiy parametrlari: {model.num_parameters():,}")
    print(f"Train qilinadigan parametrlar: {model.num_parameters(only_trainable=True):,}")
    
    # Modelni saqlash
    print("Optimizatsiya qilingan modelni saqlash...")
    optimized_model_path = f"{project_dir}/thinkly_optimized"
    os.makedirs(optimized_model_path, exist_ok=True)
    
    # Faqat LoRA parametrlarini saqlash
    model.save_pretrained(optimized_model_path)
    tokenizer.save_pretrained(optimized_model_path)
    
    print(f"Optimizatsiya qilingan model saqlandi: {optimized_model_path}")
    return model, tokenizer

# 5. Xotira optimizatsiyasi
def cleanup_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Xotira tozalandi")

# Asosiy funksiya
def main():
    # GPU tekshirish
    if not check_gpu_memory():
        print("GPU mavjud emas, optimizatsiya to'xtatildi!")
        return
    
    # Modelni yuklash va optimizatsiya qilish
    model, tokenizer = load_and_optimize_model()
    
    # Xotira tozalash
    cleanup_memory()
    
    print("Model optimizatsiyasi tugadi!")

if __name__ == "__main__":
    main() 