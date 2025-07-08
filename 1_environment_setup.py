#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Thinkly AI - O'quv muhitini tayyorlash
# Google Colab uchun

# Kerakli kutubxonalarni o'rnatish
!pip install -q transformers==4.37.2
!pip install -q peft==0.7.1
!pip install -q accelerate==0.25.0
!pip install -q bitsandbytes==0.41.1
!pip install -q datasets==2.15.0
!pip install -q sentencepiece==0.1.99
!pip install -q huggingface_hub==0.20.2
!pip install -q einops==0.7.0
!pip install -q flash-attn==2.3.4

# Google Drive bilan bog'lanish
from google.colab import drive
drive.mount('/content/drive')

# Xotira va GPU ma'lumotlarini ko'rish
!nvidia-smi
!free -h

# Loyiha papkasini yaratish
import os
project_dir = "/content/drive/MyDrive/thinkly_project"
os.makedirs(project_dir, exist_ok=True)

# Modelni yuklab olish
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Qwen2.5-Coder-7B modelini yuklab olish boshlandi...")
model_path = f"{project_dir}/qwen25_coder_7b"
os.makedirs(model_path, exist_ok=True)

# Modelni yuklab olish (agar mavjud bo'lmasa)
if not os.path.exists(f"{model_path}/config.json"):
    snapshot_download(
        repo_id="Qwen/Qwen2.5-Coder-7B",
        local_dir=model_path,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
    )
    
    # Faqat zarur fayllarni yuklab olish
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B", 
        trust_remote_code=True
    )
    tokenizer.save_pretrained(model_path)
    
    print("Model konfiguratsiyasi va tokenizer yuklandi")
else:
    print("Model konfiguratsiyasi mavjud")

print("Muhit tayyorlash tugadi!") 