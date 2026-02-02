# Java-Inline-LoRA

Проект для обучения LoRA-адаптеров на Java-коде с использованием моделей для code completion.

## Установка зависимостей

```bash
pip install -U "transformers>=4.41" datasets peft accelerate
```

Опционально для QLoRA:
```bash
pip install bitsandbytes
```

## Подготовка данных

### 1. Клонирование репозитория с данными

```bash
git clone https://github.com/itlemon/chatgpt4j data/repos/chatgpt4j
```

### 2. Создание датасета

```bash
python scripts/make_dataset.py \
  --repo ./data/repos/chatgpt4j \
  --include_header \
  --split_by_file \
  --max_prefix_lines 80 \
  --max_suffix_lines 80 \
  --max_samples_per_file 80 \
  --out_train data/train.jsonl \
  --out_val data/val.jsonl
```

## Обучение модели

```bash
python scripts/train_lora.py \
  --model Qwen/Qwen2.5-Coder-3B \
  --train data/train.jsonl \
  --val data/val.jsonl \
  --out adapters/chatgpt4j-qwen25coder3b-lora \
  --max_length 2048 \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 16
```

## Структура проекта

```
Java-Inline-LoRA/
├── adapters/           # Обученные LoRA-адаптеры
├── data/
│   ├── repos/         # Клонированные репозитории
│   ├── train.jsonl    # Обучающий датасет
│   └── val.jsonl      # Валидационный датасет
└── scripts/
    ├── make_dataset.py    # Скрипт для создания датасета
    └── train_lora.py      # Скрипт для обучения LoRA
```

