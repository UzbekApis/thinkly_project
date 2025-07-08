# Thinkly AI - O'zbek tilidagi dasturlash assistenti

Thinkly AI - bu o'zbek tilida dasturlashga yo'naltirilgan sun'iy intellekt assistenti. Qwen2.5-Coder-7B modelini asosida yaratilgan bo'lib, PHP, HTML, CSS, JavaScript, SQL, Python va Node.js kabi dasturlash tillarida yordam beradi.

## Loyiha tuzilishi

```
thinkly_project/
├── 1_environment_setup.py    # Muhitni tayyorlash
├── 2_dataset_preparation.py  # Dataset tayyorlash
├── 3_model_optimization.py   # Model optimizatsiyasi
├── 4_model_training.py       # Model o'qitish
├── 5_inference.py            # Model test qilish
├── 6_multi_session_training.py # Ko'p sessiyali o'qitish
└── README.md                 # Ushbu hujjat
```

## Google Colab'da ishga tushirish

### 1. Muhitni tayyorlash

```python
# Google Colab'da yangi notebook yarating
# Kerakli kutubxonalarni o'rnatish va modelni yuklash
%cd /content
!git clone https://github.com/UzbekApis/thinkly_project.git
%cd thinkly_project
!python 1_environment_setup.py
```

### 2. Dataset tayyorlash

```python
# Datasetlarni yuklash va tayyorlash
!python 2_dataset_preparation.py
```

### 3. Model optimizatsiyasi

```python
# Modelni 4-bit quantization va LoRA bilan optimizatsiya qilish
!python 3_model_optimization.py
```

### 4. Model o'qitish

Kichik dataset uchun:

```python
# Oddiy o'qitish
!python 4_model_training.py
```

Katta dataset uchun:

```python
# 1-sessiya
!python 6_multi_session_training.py --chunk_id=0 --chunks_total=3

# Sessiya tugagandan so'ng, yangi sessiyada:
# 2-sessiya
!python 6_multi_session_training.py --chunk_id=1 --chunks_total=3 --resume=/content/drive/MyDrive/thinkly_project/checkpoints/chunk_0

# 3-sessiya
!python 6_multi_session_training.py --chunk_id=2 --chunks_total=3 --resume=/content/drive/MyDrive/thinkly_project/checkpoints/chunk_1
```

### 5. Model test qilish

```python
# O'qitilgan modelni test qilish
!python 5_inference.py
```

## Xotira optimizatsiyasi

Loyiha Google Colab Free versiyasida ishlashi uchun quyidagi optimizatsiyalar qo'llanilgan:

1. **4-bit quantization** - model hajmini 4 barobar kamaytiradi
2. **LoRA** - faqat kichik adapterni o'qitadi, asosiy modelni o'zgartirmaydi
3. **Gradient checkpointing** - xotirani tejash uchun
4. **Dataset streaming** - katta datasetlarni bo'laklarga bo'lib ishlash
5. **Flash Attention** - diqqat mexanizmini tezlashtirish va xotirani tejash

## Datasetlar

Loyihada quyidagi datasetlar ishlatiladi:

1. **O'zbek tilidagi datasetlar**:
   - tahrirchi/uzbek-text-corpus
   - uzbek-nlp/uz-wiki
   - tahrirchi/uzbek-qa-dataset

2. **Dasturlash datasetlari**:
   - WizardLM/WizardCoder-Python-34K
   - sahil2801/CodeAlpaca-20k
   - TokenBender/code_instructions_122k_alpaca_style
   - b-mc2/sql-create-context

## Muallif

Thinkly AI loyihasi [Muallif nomi] tomonidan yaratilgan.

## Litsenziya

MIT 
