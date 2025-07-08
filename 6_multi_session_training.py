#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Thinkly AI - Ko'p sessiyali o'qitish
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
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import argparse

# Loyiha papkasini aniqlash
project_dir = "/content/drive/MyDrive/thinkly_project"
model_path = f"{project_dir}/qwen25_coder_7b"
dataset_dir = f"{project_dir}/datasets"
output_dir = f"{project_dir}/thinkly_trained"
checkpoint_dir = f"{project_dir}/checkpoints"

# Xavfsizlik uchun tasodifiy sonni o'rnatish
set_seed(42)

# 1. Dataset bo'laklarini boshqarish
def get_dataset_chunk(dataset_path_pattern, chunk_id=0, chunks_total=3):
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
    
    # Datasetni bo'laklarga bo'lish
    np.random.shuffle(all_data)
    chunk_size = len(all_data) // chunks_total
    start_idx = chunk_id * chunk_size
    end_idx = start_idx + chunk_size if chunk_id < chunks_total - 1 else len(all_data)
    
    chunk_data = all_data[start_idx:end_idx]
    print(f"Tanlangan bo'lak {chunk_id+1}/{chunks_total}: {len(chunk_data)} namunalar")
    
    return chunk_data

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
def get_training_args(checkpoint_dir, resume_from_checkpoint=None):
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=1,  # Har bir sessiyada 1 epoch
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        weight_decay=0.01,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return training_args

# 4. Modelni o'qitish
def train_model(chunk_id=0, chunks_total=3, resume_from_checkpoint=None):
    # Dataset yuklash
    dataset_path = f"{dataset_dir}/thinkly_dataset"
    raw_data = get_dataset_chunk(dataset_path, chunk_id, chunks_total)
    
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
    
    # Checkpoint mavjud bo'lsa
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Checkpointdan davom etilmoqda: {resume_from_checkpoint}")
        
        # Base modelni yuklash
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # LoRA adapterni yuklash
        model = PeftModel.from_pretrained(base_model, resume_from_checkpoint)
    else:
        # Yangi model yaratish
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
    
    # Checkpoint papkasi
    current_checkpoint_dir = f"{checkpoint_dir}/chunk_{chunk_id}"
    os.makedirs(current_checkpoint_dir, exist_ok=True)
    
    # Training argumentlari
    training_args = get_training_args(current_checkpoint_dir, resume_from_checkpoint)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # O'qitishni boshlash
    print(f"Bo'lak {chunk_id+1}/{chunks_total} uchun o'qitish boshlandi...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Modelni saqlash
    print("Checkpointni saqlash...")
    trainer.save_model(current_checkpoint_dir)
    
    # Oxirgi bo'lak bo'lsa, yakuniy modelni saqlash
    if chunk_id == chunks_total - 1:
        print("Yakuniy modelni saqlash...")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Yakuniy model saqlandi: {output_dir}")
    
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
    # Argumentlarni olish
    parser = argparse.ArgumentParser(description="Ko'p sessiyali o'qitish")
    parser.add_argument("--chunk_id", type=int, default=0, help="Hozirgi bo'lak indeksi (0 dan boshlanadi)")
    parser.add_argument("--chunks_total", type=int, default=3, help="Jami bo'laklar soni")
    parser.add_argument("--resume", type=str, default=None, help="Checkpointdan davom ettirish")
    args = parser.parse_args()
    
    try:
        # Modelni o'qitish
        print(f"Bo'lak {args.chunk_id+1}/{args.chunks_total} uchun o'qitish boshlanmoqda...")
        model, tokenizer = train_model(args.chunk_id, args.chunks_total, args.resume)
        
        # Xotira tozalash
        cleanup_memory()
        
        print(f"Bo'lak {args.chunk_id+1}/{args.chunks_total} uchun o'qitish tugadi!")
        
        # Keyingi bo'lak uchun ko'rsatma
        if args.chunk_id < args.chunks_total - 1:
            next_chunk = args.chunk_id + 1
            next_checkpoint = f"{checkpoint_dir}/chunk_{args.chunk_id}"
            print(f"\nKeyingi sessiyada quyidagi buyruqni ishga tushiring:")
            print(f"!python 6_multi_session_training.py --chunk_id={next_chunk} --chunks_total={args.chunks_total} --resume={next_checkpoint}")
    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")

if __name__ == "__main__":
    main() 