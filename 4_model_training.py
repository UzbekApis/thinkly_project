#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Thinkly AI - Model o'qitish
# Google Colab uchun

import os
import json
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Loyiha papkasini aniqlash
project_dir = "/content/drive/MyDrive/thinkly_project"
model_path = f"{project_dir}/qwen25_coder_7b"
dataset_dir = f"{project_dir}/datasets"
output_dir = f"{project_dir}/thinkly_trained"

# Xavfsizlik uchun tasodifiy sonni o'rnatish
set_seed(42)

print("Model o'qitish boshlandi...")

# 1. Dataset yuklash
def load_dataset_from_json(dataset_path_pattern):
    all_data = []
    
    # Barcha bo'laklarni topish
    import glob
    dataset_files = glob.glob(f"{dataset_path_pattern}_part*.json")
    
    if not dataset_files:
        raise ValueError(f"Dataset topilmadi: {dataset_path_pattern}_part*.json")
    
    print(f"Topilgan dataset fayllari: {len(dataset_files)}")
    
    # Har bir faylni yuklash
    for file_path in dataset_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
            print(f"Yuklandi: {file_path} ({len(data)} namunalar)")
        except Exception as e:
            print(f"Xatolik: {file_path} yuklanmadi - {e}")
    
    print(f"Jami yuklangan namunalar: {len(all_data)}")
    return all_data

# 2. Datasetni tokenizatsiya qilish
def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        # Alpaca formatidagi datasetni tokenizatsiya qilish
        prompt_template = """<|im_start|>user
{instruction} {input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i] if examples["input"][i] else ""
            output = examples["output"][i]
            
            # Prompt yaratish
            text = prompt_template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
            texts.append(text)
        
        # Tokenizatsiya
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Labels = input_ids (causal language modeling uchun)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Dataset yaratish
    dataset_dict = {
        "instruction": [item["instruction"] for item in dataset],
        "input": [item["input"] for item in dataset],
        "output": [item["output"] for item in dataset]
    }
    raw_dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenizatsiya
    print("Dataset tokenizatsiya qilinmoqda...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=1,
    )
    
    print(f"Tokenizatsiya tugadi: {len(tokenized_dataset)} namunalar")
    return tokenized_dataset

# 3. Training argumentlarini sozlash
def get_training_args():
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=3,
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        weight_decay=0.01,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    return training_args

# 4. Modelni o'qitish
def train_model():
    # Dataset yuklash
    dataset_path = f"{dataset_dir}/thinkly_dataset"
    raw_data = load_dataset_from_json(dataset_path)
    
    # Dataset hajmini cheklash (xotira tejash uchun)
    if len(raw_data) > 100000:
        print(f"Dataset hajmi katta ({len(raw_data)}), cheklanmoqda...")
        np.random.shuffle(raw_data)
        raw_data = raw_data[:100000]
    
    # Tokenizer yuklab olish
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # Dataset tokenizatsiya qilish
    tokenized_dataset = tokenize_dataset(raw_data, tokenizer)
    
    # 4-bit quantization konfiguratsiyasi
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Modelni yuklash
    print("Modelni yuklash...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Model nomini o'zgartirish
    model.config.model_type = "thinkly"
    
    # Gradient checkpointing yoqish
    model.gradient_checkpointing_enable()
    
    # LoRA uchun modelni tayyorlash
    model = prepare_model_for_kbit_training(model)
    
    # LoRA konfiguratsiyasi
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # LoRA qo'shish
    model = get_peft_model(model, lora_config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Training argumentlari
    training_args = get_training_args()
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # O'qitishni boshlash
    print("O'qitish boshlandi...")
    trainer.train()
    
    # Modelni saqlash
    print("O'qitilgan modelni saqlash...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Faqat LoRA parametrlarini saqlash
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"O'qitilgan model saqlandi: {output_dir}")
    return model, tokenizer

# 5. Xotira tozalash
def cleanup_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Xotira tozalandi")

# Asosiy funksiya
def main():
    try:
        # Modelni o'qitish
        model, tokenizer = train_model()
        
        # Xotira tozalash
        cleanup_memory()
        
        print("Model o'qitish tugadi!")
    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")

if __name__ == "__main__":
    main() 