#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Thinkly AI - Dataset tayyorlash
# Google Colab uchun

import os
import json
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

# Loyiha papkasini aniqlash
project_dir = "/content/drive/MyDrive/thinkly_project"
dataset_dir = f"{project_dir}/datasets"
os.makedirs(dataset_dir, exist_ok=True)

# Tokenizer yuklab olish
tokenizer = AutoTokenizer.from_pretrained(
    f"{project_dir}/qwen25_coder_7b", 
    trust_remote_code=True
)

print("Dataset yuklab olish boshlandi...")

# 1. O'zbek tilidagi datasetlar
def load_uzbek_datasets():
    datasets = []
    
    # O'zbek korpusi
    try:
        uz_corpus = load_dataset("tahrirchi/uzbek-text-corpus", split="train")
        uz_corpus = uz_corpus.select(range(min(50000, len(uz_corpus))))
        datasets.append(uz_corpus)
        print(f"Yuklandi: uzbek-text-corpus ({len(uz_corpus)} namunalar)")
    except Exception as e:
        print(f"Xatolik: uzbek-text-corpus yuklanmadi - {e}")
    
    # O'zbek Wikipedia
    try:
        uz_wiki = load_dataset("uzbek-nlp/uz-wiki", split="train")
        uz_wiki = uz_wiki.select(range(min(30000, len(uz_wiki))))
        datasets.append(uz_wiki)
        print(f"Yuklandi: uz-wiki ({len(uz_wiki)} namunalar)")
    except Exception as e:
        print(f"Xatolik: uz-wiki yuklanmadi - {e}")
    
    # O'zbek savol-javob
    try:
        uz_qa = load_dataset("tahrirchi/uzbek-qa-dataset", split="train")
        uz_qa = uz_qa.select(range(min(20000, len(uz_qa))))
        datasets.append(uz_qa)
        print(f"Yuklandi: uzbek-qa-dataset ({len(uz_qa)} namunalar)")
    except Exception as e:
        print(f"Xatolik: uzbek-qa-dataset yuklanmadi - {e}")
    
    return datasets

# 2. Dasturlash datasetlari
def load_programming_datasets():
    datasets = []
    
    # Python kod namunalari
    try:
        python_dataset = load_dataset("WizardLM/WizardCoder-Python-34K", split="train")
        python_dataset = python_dataset.select(range(min(15000, len(python_dataset))))
        datasets.append(python_dataset)
        print(f"Yuklandi: WizardCoder-Python ({len(python_dataset)} namunalar)")
    except Exception as e:
        print(f"Xatolik: WizardCoder-Python yuklanmadi - {e}")
    
    # JavaScript kod namunalari
    try:
        js_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        js_dataset = js_dataset.filter(lambda x: "javascript" in x.get("instruction", "").lower() or "js" in x.get("instruction", "").lower())
        js_dataset = js_dataset.select(range(min(12000, len(js_dataset))))
        datasets.append(js_dataset)
        print(f"Yuklandi: CodeAlpaca-JS ({len(js_dataset)} namunalar)")
    except Exception as e:
        print(f"Xatolik: CodeAlpaca-JS yuklanmadi - {e}")
    
    # HTML/CSS kod namunalari
    try:
        web_dataset = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")
        web_dataset = web_dataset.filter(lambda x: "html" in x.get("instruction", "").lower() or "css" in x.get("instruction", "").lower())
        web_dataset = web_dataset.select(range(min(7000, len(web_dataset))))
        datasets.append(web_dataset)
        print(f"Yuklandi: HTML/CSS dataset ({len(web_dataset)} namunalar)")
    except Exception as e:
        print(f"Xatolik: HTML/CSS dataset yuklanmadi - {e}")
    
    # SQL kod namunalari
    try:
        sql_dataset = load_dataset("b-mc2/sql-create-context", split="train")
        sql_dataset = sql_dataset.select(range(min(5000, len(sql_dataset))))
        datasets.append(sql_dataset)
        print(f"Yuklandi: SQL dataset ({len(sql_dataset)} namunalar)")
    except Exception as e:
        print(f"Xatolik: SQL dataset yuklanmadi - {e}")
    
    # PHP kod namunalari (kichik dataset yaratish)
    try:
        php_data = []
        for i in range(100):
            php_data.append({
                "instruction": f"PHP da {i}-misol uchun kod yozing",
                "input": "",
                "output": f"<?php\n    echo 'Misol {i}';\n?>"
            })
        php_dataset = Dataset.from_pandas(pd.DataFrame(php_data))
        datasets.append(php_dataset)
        print(f"Yaratildi: PHP dataset ({len(php_dataset)} namunalar)")
    except Exception as e:
        print(f"Xatolik: PHP dataset yaratilmadi - {e}")
    
    return datasets

# 3. Dataset formatini birlashtirish
def format_dataset(datasets):
    formatted_data = []
    
    for dataset in datasets:
        for item in dataset:
            # Alpaca formatiga o'tkazish
            if "instruction" in item and "input" in item and "output" in item:
                # Alpaca formatida
                formatted_data.append({
                    "instruction": item["instruction"],
                    "input": item["input"] if item["input"] else "",
                    "output": item["output"]
                })
            elif "question" in item and "answer" in item:
                # QA formatidan
                formatted_data.append({
                    "instruction": item["question"],
                    "input": "",
                    "output": item["answer"]
                })
            elif "text" in item:
                # Matn formatidan
                text = item["text"]
                if len(text) > 50:  # Juda qisqa matnlarni o'tkazib yuborish
                    formatted_data.append({
                        "instruction": "Quyidagi matn haqida ma'lumot bering:",
                        "input": text[:200],  # Uzun matnlarni qisqartirish
                        "output": f"Bu matn {len(text)} belgidan iborat. Matn quyidagicha davom etadi: {text[200:400]}..."
                    })
            else:
                # Boshqa formatlar
                continue
    
    # Thinkly identifikatsiyasi uchun maxsus namunalar qo'shish
    thinkly_samples = [
        {
            "instruction": "Sen qanday AI modelisan?",
            "input": "",
            "output": "Men Thinkly nomli AI yordamchisiman. O'zbek tilida dasturlash bo'yicha yordam beradigan sun'iy intellekt assistentiman."
        },
        {
            "instruction": "O'zingni tanishtir",
            "input": "",
            "output": "Salom! Men Thinkly, o'zbek tilidagi dasturlash bo'yicha AI yordamchisiman. PHP, HTML, CSS, JavaScript, SQL, Python va Node.js kabi dasturlash tillari bo'yicha yordam bera olaman."
        },
        {
            "instruction": "Qaysi kompaniya tomonidan yaratilgansiz?",
            "input": "",
            "output": "Men Thinkly AI assistentiman, o'zbek tilidagi dasturlashga yo'naltirilgan sun'iy intellekt modeliman."
        }
    ]
    
    # 100 marta takrorlash (model yaxshi o'rganishi uchun)
    for _ in range(100):
        formatted_data.extend(thinkly_samples)
    
    print(f"Jami formatlashtirilgan namunalar: {len(formatted_data)}")
    return formatted_data

# 4. Datasetni saqlash
def save_dataset(formatted_data, output_path):
    # Dataset hajmini cheklash (xotira tejash uchun)
    if len(formatted_data) > 200000:
        print(f"Dataset hajmi katta ({len(formatted_data)}), cheklanmoqda...")
        np.random.shuffle(formatted_data)
        formatted_data = formatted_data[:200000]
    
    # Datasetni bo'laklarga bo'lish
    chunk_size = 50000  # Har bir fayl uchun
    for i in range(0, len(formatted_data), chunk_size):
        chunk = formatted_data[i:i+chunk_size]
        chunk_path = f"{output_path}_part{i//chunk_size}.json"
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        
        print(f"Saqlandi: {chunk_path} ({len(chunk)} namunalar)")

# Asosiy funksiya
def main():
    print("O'zbek tilidagi datasetlarni yuklash...")
    uzbek_datasets = load_uzbek_datasets()
    
    print("\nDasturlash datasetlarini yuklash...")
    programming_datasets = load_programming_datasets()
    
    print("\nBarcha datasetlarni formatlash...")
    all_datasets = uzbek_datasets + programming_datasets
    formatted_data = format_dataset(all_datasets)
    
    print("\nDatasetni saqlash...")
    output_path = f"{dataset_dir}/thinkly_dataset"
    save_dataset(formatted_data, output_path)
    
    print("\nDataset tayyorlash tugadi!")

if __name__ == "__main__":
    main() 