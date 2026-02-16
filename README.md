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

### 2.1. Использование готового 3-польного JSONL (prompt/expectedCode/completedCode)

Если у вас есть строки вида `{"prompt": ..., "expectedCode": ..., "completedCode": ...}`, можно конвертировать их в ожидаемую схему `prompt/completion` без make_dataset.py:

```bash
python scripts/convert_dataset.py \
  --train data/train_3fields.jsonl \
  --val data/val_3fields.jsonl \
  --dataset_format fim_expected_completed \
  --out_train data/converted_train.jsonl \
  --out_val data/converted_val.jsonl \
  --truncate_prompt_to_max_length \
  --max_seq_length 2048 \
  --truncate_policy drop_file_sep_prefix_then_left \
  --truncate_report_path data/truncate_report.json
```

Что делает команда:
- проверяет, что `prompt`, `expectedCode`, `completedCode` — строки;
- по умолчанию отбрасывает записи без всех FIM-токенов (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`); снять проверку можно флагом `--no-require_fim_tokens`;
- сохраняет `expectedCode` в поле `completion` (используется в лоссе), `completedCode` оставляет как метаданные (не участвует в лоссе);
- при `--truncate_prompt_to_max_length` гарантирует, что `prompt+completion+eos` умещаются в `--max_seq_length`, сначала удаляя префиксные блоки `<|file_sep|>...<|file_sep|>`, затем левой обрезкой; если сохранить FIM-структуру нельзя — образец дропается с явной причиной;
- пишет конвертированные файлы в указанные пути с `ensure_ascii=False`, китайские символы и `<|file_sep|>` сохраняются как есть.

Быстрая проверка длин/токенов без записи файлов:

```bash
python scripts/convert_dataset.py \
  --train data/train_3fields.jsonl \
  --val data/val_3fields.jsonl \
  --dataset_format fim_expected_completed \
  --example_count 2 \
  --out_train "" \
  --out_val "" \
  --truncate_prompt_to_max_length \
  --max_seq_length 2048 \
  --truncate_policy drop_file_sep_prefix_then_left
```

### Training on both No-RAG and RAG prompts (doubled dataset)

Конвертация успешных кейсов сразу в две выборки (чистый FIM и FIM с RAG-дополнением) — итоговый train/val может увеличиться до ~2x:

```bash
python scripts/convert_dataset.py \
  --dataset_format fim_expected_completed \
  --train data/train_success.jsonl \
  --val data/val_success.jsonl \
  --out_train data/train_doubled.jsonl \
  --out_val data/val_doubled.jsonl \
  --emit_both_rag_and_norag \
  --truncate_prompt_to_max_length \
  --max_seq_length 2048 \
  --truncate_policy drop_file_sep_prefix_then_left \
  --enforce_model_max_length \
  --report_path data/convert_report_doubled.json
```

- Вариант No-RAG полностью убирает все блоки `<|file_sep|>...<|file_sep|>`.
- Вариант RAG сохраняет `<|file_sep|>`-аугментацию, но безопасно обрезает префикс по заданному бюджету токенов, сохраняя FIM-токены.
- Оба варианта проходят независимую валидацию длины/структуры; при конфликте обрезок либо дропается только один вариант (по умолчанию), либо оба (если `--both_mode_on_conflict skip`).

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

Пример с политикой downweight для бенчмарков (снижает вес файлов из bench_dir в 4 раза):

```bash
python scripts/make_dataset.py \
  --repo ./data/repos/chatgpt4j \
  --bench_dir ./data/repoeval_samples \
  --include_header \
  --split_by_file \
  --max_prefix_lines 80 \
  --max_suffix_lines 80 \
  --max_samples_per_file 30 \
  --out_train data/train_rag.jsonl \
  --out_val data/val_rag.jsonl \
  --rag_enable \
  --rag_k 2 \
  --rag_max_chars 1000 \
  --rag_max_snippet_chars 500 \
  --rag_chunker lines \
  --rag_chunk_lines 30 \
  --rag_chunk_overlap 10 \
  --rag_query_mode hybrid \
  --rag_query_window_lines 20 \
  --rag_exclude_same_file_window 120 \
  --rag_exclude_bench_targets \
  --rag_exclude_completion_text \
  --leak_policy downweight_files \
  --bench_downweight_factor 0.25 \
  --bench_report_path data/bench_mask_report.json
```


В выходных записях при `--rag_enable` добавляются поля:
- `rag`: список фрагментов с путями и диапазонами строк,
- `rag_query`: текст запроса,
- `rag_k_used`: сколько сниппетов реально вставлено (учитывая фильтры и лимиты).

## Codefixes dataset + training (`/v1/chat/completions`)

Для задачи исправления ошибки в одной Java-строке добавлен отдельный изолированный пайплайн:
- `scripts/make_codefix_dataset.py` — генерация синтетических train/val сэмплов;
- `scripts/train_codefix_lora.py` — обучение отдельного LoRA-адаптера.

Существующие скрипты inline completion (`make_dataset.py`, `train_lora.py`, `convert_dataset.py`) не затрагиваются.

### Генерация 5000 codefix-сэмплов (4900/100, split-by-file)

```bash
python scripts/make_codefix_dataset.py \
  --repo ./data/repos/chatgpt4j \
  --target_total 5000 \
  --val_count 100 \
  --bm25_top_k 4 \
  --out_train data/codefix_train.jsonl \
  --out_val data/codefix_val.jsonl
```

Что делает скрипт:
- ищет одно-строчные Java statement-вызовы вида `...(...);`;
- синтетически портит строку одной ошибкой (`swap_args`, `missing_arg`, `extra_arg`);
- подбирает BM25-похожие корректные statement-строки (augmentations);
- формирует prompt в chat-формате (`system` + `user`) и `completion` как строго один ` ```java `-блок с исправленной строкой;
- собирает ровно `target_total` записей с fail-fast при нехватке кандидатов.

Формат строки JSONL:
- `prompt`, `completion`, `messages`;
- `mutation_type`, `original_line`, `broken_line`;
- `compiler_problems`, `augmentations`;
- `file`, `line_index`, `id`.

### Обучение отдельного LoRA-адаптера для codefix

```bash
python scripts/train_codefix_lora.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --train data/codefix_train.jsonl \
  --val data/codefix_val.jsonl \
  --out adapters/codefix-qwen25coder7b-bf16-lora \
  --dtype bf16 \
  --max_length 2048 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 16
```

Быстрый smoke-check без сохранения адаптера:

```bash
python scripts/train_codefix_lora.py \
  --train data/codefix_train.jsonl \
  --val data/codefix_val.jsonl \
  --out adapters/codefix-debug \
  --dry_run
```

## Фильтрация датасета teacher-моделью (`/v1/completions`)

Скрипт `scripts/filter_with_teacher.py` прогоняет FIM-образцы через teacher LLM и оставляет только те строки, где ответ teacher совпал с `completion` (режим сравнения настраивается). По умолчанию лимитов нет: обрабатывается весь вход, если не заданы `--max_keep`/`--max_samples`.

### Базовый детерминированный запуск

```bash
python scripts/filter_with_teacher.py \
  --in data/train.jsonl \
  --out data/train.filtered.teacher.jsonl \
  --endpoint http://everest.nsu.ru:9111/v1 \
  --model Qwen2.5-Coder-14B-GPTQ-Int4 \
  --temperature 0 \
  --top_p 1 \
  --max_tokens 64 \
  --stop "\n" \
  --match_mode ws_norm
```

### С повышенной параллельностью

```bash
python scripts/filter_with_teacher.py \
  --in data/train.jsonl \
  --out data/train.filtered.teacher.jsonl \
  --endpoint http://everest.nsu.ru:9111/v1 \
  --model Qwen2.5-Coder-14B-GPTQ-Int4 \
  --concurrency 16 \
  --qps 8 \
  --request_timeout 60 \
  --max_retries 5 \
  --retry_backoff_base 0.5
```

### С файлом отклонённых примеров

```bash
python scripts/filter_with_teacher.py \
  --in data/train.jsonl \
  --out data/train.filtered.teacher.jsonl \
  --rejected_out data/train.rejected.teacher.jsonl \
  --endpoint http://everest.nsu.ru:9111/v1 \
  --model Qwen2.5-Coder-14B-GPTQ-Int4 \
  --match_mode trimmed
```

### Ограничение количества (опционально)

```bash
# Остановиться после 20000 сохранённых примеров
python scripts/filter_with_teacher.py \
  --in data/train.jsonl \
  --out data/train.filtered.teacher.jsonl \
  --endpoint http://everest.nsu.ru:9111/v1 \
  --model Qwen2.5-Coder-14B-GPTQ-Int4 \
  --max_keep 20000

# Обработать только первые 50000 входных примеров
python scripts/filter_with_teacher.py \
  --in data/train.jsonl \
  --out data/train.filtered.teacher.jsonl \
  --endpoint http://everest.nsu.ru:9111/v1 \
  --model Qwen2.5-Coder-14B-GPTQ-Int4 \
  --max_samples 50000
```

Что пишет скрипт:
- в `--out` попадают только `teacher_filter.passed == true`;
- в `--rejected_out` (если задан) — mismatch/ошибки с метаданными `teacher_filter`;
- рядом с `--out` создаётся `*.run_manifest.json` (время запуска, аргументы, git commit, статистика);
- прогресс печатается каждые `--progress_every` обработанных образцов (по умолчанию 50).

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

### Пример обучения на 3-польном JSONL (prompt/expectedCode/completedCode)

```bash
python scripts/train_lora.py \
  --model Qwen/Qwen2.5-Coder-3B \
  --train data/train_3fields.jsonl \
  --val data/val_3fields.jsonl \
  --dataset_format fim_expected_completed \
  --out adapters/chatgpt4j-qwen25coder3b-lora \
  --max_length 2048 \
  --epochs 1.0
```

- `expectedCode` идёт в поле `completion` и участвует в лоссе, `completedCode` остаётся только как доп. поле (не влияет на обучение).
- По умолчанию включена проверка FIM-токенов в prompt; отключить можно `--no-require_fim_tokens`.

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
