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

Примечание: для обучения используйте GPU с поддержкой bfloat16/float16; для QLoRA нужен сборка PyTorch с CUDA и bitsandbytes.

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

Формат записей JSONL: поля `prompt` (FIM-контекст с `<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`) и `completion` (строки, которые надо предсказать). Лосс считается только по `completion`.

### 3. Создание датасета с RAG-контекстом (опционально)

Базовый запуск с включённым RAG (retrieval вставляется в начало prefix в виде Java-комментария `/* RAG_CONTEXT ... */`):

```bash
python scripts/make_dataset.py \
  --repo ./data/repos/chatgpt4j \
  --include_header \
  --split_by_file \
  --max_prefix_lines 80 \
  --max_suffix_lines 80 \
  --max_samples_per_file 80 \
  --out_train data/train.jsonl \
  --out_val data/val.jsonl \
  --rag_enable \
  --rag_k 4 \
  --rag_max_chars 2000 \
  --rag_query_mode hybrid \
  --rag_exclude_same_file_window 120
```

Расширенный пример с явными настройками chunker/BM25 и доп. фильтрами:

```bash
python scripts/make_dataset.py \
  --repo ./data/repos/chatgpt4j \
  --bench_dir ./data/repoeval_samples \
  --include_header \
  --split_by_file \
  --max_prefix_lines 80 \
  --max_suffix_lines 80 \
  --max_samples_per_file 80 \
  --out_train data/train.jsonl \
  --out_val data/val.jsonl \
  --rag_enable \
  --rag_method bm25 \
  --rag_chunker lines \
  --rag_chunk_lines 30 \
  --rag_chunk_overlap 10 \
  --rag_k 4 \
  --rag_max_chars 2000 \
  --rag_max_snippet_chars 600 \
  --rag_query_mode hybrid \
  --rag_query_window_lines 20 \
  --rag_drop_stopwords \
  --rag_exclude_same_file_window 120 \
  --rag_exclude_bench_targets \
  --rag_exclude_completion_text \
  --rag_insert_location prefix_head \
  --bench_report_path data/bench_mask_report.json
```

В выходных записях при `--rag_enable` добавляются поля:
- `rag`: список фрагментов с путями и диапазонами строк,
- `rag_query`: текст запроса,
- `rag_k_used`: сколько сниппетов реально вставлено (учитывая фильтры и лимиты).

## Обучение модели

Быстрый тест без сохранения артефактов (строит модель+датасет, делает пару проходов вперёд и печатает VRAM):

```bash
python scripts/train_lora.py \
  --model Qwen/Qwen2.5-Coder-3B \
  --train data/train.jsonl \
  --val data/val.jsonl \
  --out adapters/debug \
  --dry_run
```

Стандартное обучение LoRA (bf16/auto, градиентный чекпойнтинг включён по умолчанию):

```bash
python scripts/train_lora.py \
  --model Qwen/Qwen2.5-Coder-3B \
  --train data/train.jsonl \
  --val data/val.jsonl \
  --out adapters/chatgpt4j-qwen25coder3b-lora \
  --max_length 2048 \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 16 \
  --lr 2e-4 \
  --warmup_ratio 0.03
```

QLoRA (4-bit), при наличии bitsandbytes:

```bash
python scripts/train_lora.py \
  --model Qwen/Qwen2.5-Coder-3B \
  --train data/train.jsonl \
  --val data/val.jsonl \
  --out adapters/chatgpt4j-qwen25coder3b-qlora \
  --use_4bit \
  --max_length 2048 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-4
```

Полезные флаги:
- `--dtype {auto,bf16,fp16}` — выбор типа (по умолчанию auto → bf16 если доступен).
- `--tf32` — включить TF32 на Ampere+.
- `--gradient_checkpointing/--no-gradient_checkpointing` — управление чекпойнтингом.
- `--warmup_steps` или `--warmup_ratio` — тёплый старт, ratio конвертируется в шаги.
- `--eval_steps/--save_steps/--logging_steps` — интервалы логов/оценки/сохранений.
- `--dry_run` — быстрая проверка без сохранений.

Артефакты сохраняются в `--out`: LoRA-адаптер, токенизатор и `run_config.json` (параметры запуска + рассчитанные величины).

## Подсчёт статистик по RepoEval

### Анализ всех тестовых файлов в директории

```bash
python scripts/repoeval_stats.py data/repoeval --pattern "*.test.jsonl"
```

### Анализ конкретных файлов с созданием графиков

```bash
python repoeval_stats.py api_level.java.test.jsonl live_level.java.test.jsonl --plots-dir ./reporeval_plots
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
