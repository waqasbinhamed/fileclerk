# pi-doc-agent

A local document indexing and classification agent designed to run on a **Raspberry Pi 4 (4 GB RAM)**. It discovers documents from a local SSD or a Mac folder mounted via SSHFS, extracts text, embeds it with a lightweight sentence-transformer model, classifies each document using [Phi-3 mini](https://ollama.com/library/phi3) via [Ollama](https://ollama.com), and optionally moves files into an organised folder hierarchy — entirely offline, with no data leaving your network. Built for people who want smart document organisation without cloud dependency.

---

## Architecture

```
Documents (PDF, DOCX, XLSX, code, Markdown, CSV, …)
         │
         ▼
  ┌──────────────┐
  │  Extractor   │  Per-filetype text extraction
  │  extractor.py│  First ~1000 tokens of meaningful text
  └──────┬───────┘  PDF → pymupdf (OCR fallback via Tesseract)
         │          DOCX → python-docx  XLSX → openpyxl
         ▼          .ipynb → cell sources  plain text → direct read
  ┌──────────────┐
  │   Embedder   │  sentence-transformers: all-MiniLM-L6-v2
  │   indexer.py │  ~80 MB model, runs on Pi CPU
  └──────┬───────┘  SHA-256 hash → skip unchanged files
         │
         ▼
  ┌──────────────┐
  │   ChromaDB   │  Persistent cosine-similarity vector store
  │   (local)    │  Stored on SSD, never in-memory
  └──────┬───────┘
         │
         ▼
  ┌─────────────────────┐
  │     Classifier      │  Ollama + Phi-3 mini (runs on HOST)
  │   classifier.py     │  Returns JSON: category, confidence,
  └──────┬──────────────┘  suggested_folder, reason
         │
         ▼
  ┌──────────────┐
  │    Sorter    │  Dry-run by default
  │   sorter.py  │  --execute to move files
  └──────────────┘  Never overwrites · logs every action
```

---

## Quickstart (Docker — 3 commands)

```bash
# 1. Install Ollama on your machine and pull the model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini

# 2. Build the container (supports linux/arm64 for Pi and linux/amd64 for Mac/CI)
docker compose -f docker/docker-compose.dev.yml build

# 3. Index, classify, and preview the sort (dry-run — nothing is moved)
docker compose -f docker/docker-compose.dev.yml run agent sort \
  --path /data/input --output /data/output
```

To actually move files, add `--execute`:

```bash
docker compose -f docker/docker-compose.dev.yml run agent sort \
  --path /data/input --output /data/output --execute
```

---

## Mac remote access (SSHFS)

Run the agent on the Pi while it classifies documents that live on your Mac.

**On Mac** (one-time setup):

```bash
# Enable Remote Login
# System Settings → General → Sharing → Remote Login → On

# Verify SSH works from the Pi
ssh username@mac-local-ip
```

**On Pi** (mount the Mac folder):

```bash
sudo apt install sshfs
mkdir ~/mac_docs
sshfs username@mac-local-ip:/Users/username/Documents ~/mac_docs

# Run the agent against the mounted folder
python agent.py index --path ~/mac_docs
```

**Unmount when done:**

```bash
fusermount -u ~/mac_docs
```

---

## CLI reference

| Command | Description |
|---------|-------------|
| `python agent.py index --path <dir>` | Index all supported files in `<dir>` |
| `python agent.py sort --path <dir> --output <dir>` | Index + classify + preview sort (dry-run) |
| `python agent.py sort ... --execute` | Actually move files to `--output` |
| `python agent.py query "search terms"` | Semantic search over indexed documents |
| `python agent.py review` | List low-confidence files pending review |
| `python agent.py sync --path <dir>` | Re-index only changed files (hash check) |

### Examples

```bash
# Index all files on the SSD
python agent.py index --path /mnt/ssd/documents

# See what would happen (dry-run)
python agent.py sort --path /mnt/ssd/documents --output /mnt/ssd/sorted

# Execute the sort
python agent.py sort --path /mnt/ssd/documents --output /mnt/ssd/sorted --execute

# Find invoices semantically
python agent.py query "invoices from 2024"

# See what the classifier wasn't confident about
python agent.py review

# Fast incremental update after adding new files
python agent.py sync --path /mnt/ssd/documents
```

Use `--config` to point at a non-default config file:

```bash
python agent.py --config /path/to/config.yaml index --path /mnt/ssd/documents
```

---

## Configuration

Edit `config.yaml` before running. All values have sensible defaults.

```yaml
# Ollama model name — must be pulled with `ollama pull <model>`
ollama_model: phi3:mini

# Documents below this confidence score are marked 'needs_review' and NOT moved
confidence_threshold: 0.6

# Embedding batch size — lower this (e.g. 8) if you hit OOM on Pi 4 with 4 GB RAM
batch_size: 16

# Folder taxonomy — the LLM can only pick from these categories
taxonomy:
  - Finance
  - Finance/Invoices
  - Finance/Tax
  - Finance/Receipts
  - Work
  - Work/Reports
  - Work/Code
  - Work/Presentations
  - Research
  - Research/Papers
  - Personal
  - Personal/Health
  - Personal/Travel
  - Archive
  - Unsorted
```

**Adding categories:** append entries to `taxonomy` and re-run classification. Existing classifications are not automatically updated — re-index the affected folder with `sync`.

**Overriding Ollama host:** set `OLLAMA_HOST` environment variable (used automatically in Docker).

---

## Evaluation

Measure classifier accuracy against your own documents using the two-step harness.

### Step 1 — Label a sample

```bash
python eval/label_sample.py --n 50
```

Randomly picks 50 documents from the index and prompts you to enter the correct category for each. Saves to `eval/ground_truth.jsonl`.

### Step 2 — Compute metrics

```bash
python eval/eval.py
```

Runs the classifier over every labeled sample and prints per-category precision, recall, and F1. Results saved to `eval/results.jsonl` and `eval/results_metrics.json`.

### Example output

*Replace the table below with your own eval run results after labeling.*

| Category          | Precision | Recall | F1   |
|-------------------|-----------|--------|------|
| Finance/Invoices  | 0.91      | 0.88   | 0.89 |
| Work/Code         | 0.95      | 0.92   | 0.93 |
| Research/Papers   | 0.87      | 0.83   | 0.85 |
| Personal          | 0.82      | 0.79   | 0.80 |
| Personal/Health   | 0.78      | 0.74   | 0.76 |

---

## Performance on Pi 4 (4 GB RAM)

| Operation | Time | Peak RAM |
|-----------|------|----------|
| Load embedding model (first run) | ~10 s | ~400 MB |
| Embed batch of 16 documents | ~2–5 s | ~800 MB |
| Classify one document (Phi-3 mini) | 2–8 s | ~2.5 GB |
| OCR one PDF page (Tesseract fallback) | 15–30 s | ~300 MB |
| Index 1,000 text files | ~5–10 min | ~800 MB |
| Semantic query | <1 s | ~800 MB |

**Pi-specific tuning tips:**

- Ollama loads Phi-3 mini (~2.3 GB) into RAM on the first classification call; subsequent calls are fast.
- A 1-second delay between Ollama calls is built in to prevent memory pressure (`agent.py:cmd_sort`).
- If you have other workloads running alongside, reduce `batch_size` to `8` in `config.yaml`.
- OCR prints a warning before it runs — expected to be rare on text-based PDFs.
- RAM usage is logged after each major step (embedding, classification) at `DEBUG` log level.

---

## Deployment on Pi

```bash
# Prerequisites on Pi
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini

# Clone the repo and configure
git clone <repo-url> pi-doc-agent
cd pi-doc-agent/docker

# Build (arm64 is detected automatically)
docker compose build

# Initial full index + classify + dry-run preview
docker compose run agent sort --path /data/input --output /data/output

# Execute (moves files)
docker compose run agent sort --path /data/input --output /data/output --execute

# Incremental sync after new files arrive
docker compose run agent sync --path /data/input

# Semantic search
docker compose run agent query "tax return 2023"

# Folder watcher (auto-index on file change)
# python watcher.py is not wired into docker-compose; run directly on Pi:
python watcher.py   # edit watcher.py to call watch(path, indexer)
```

The `docker-compose.yml` mounts documents **read-only** (`ro`) so the container can never modify your source files. Only the `--output` volume is writable.

---

## Contributing / local dev

### VS Code devcontainer

Open the repo in VS Code and select **Reopen in Container**. Docker Desktop builds the image, mounts the workspace, installs dev dependencies, and sets `OLLAMA_HOST` automatically. No local Python installation needed.

Extensions installed automatically: `ms-python.python`, `ms-python.ruff`, `ms-python.black-formatter`.

### Running tests locally

```bash
# Install dev deps
pip install -r requirements-dev.txt

# Run all tests (Ollama is mocked — no LLM required)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Lint
ruff check .

# Format
black .
```

### File type support

| Extension | Library | Notes |
|-----------|---------|-------|
| `.pdf` | `pymupdf` | OCR fallback via `pytesseract` if text < 100 chars |
| `.docx` | `python-docx` | Paragraphs + table cells |
| `.xlsx` | `openpyxl` | First sheet, first 20 rows |
| `.py` `.js` `.ts` `.sql` `.sh` | plain read | — |
| `.ipynb` | json parse | Cell sources only |
| `.md` `.txt` `.csv` | plain read | — |
| anything else | — | Logged to `skipped.log` |

### Supported Ollama models

Any model that outputs valid JSON works. Tested with:

- `phi3:mini` — recommended for Pi 4 (~2.3 GB RAM)
- `llama3.2:3b` — slightly larger, better accuracy
- `gemma2:2b` — fast alternative

Change `ollama_model` in `config.yaml` and re-run classification.
