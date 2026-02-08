# Project Context — MiniCPM-o 4.5 + ACE-Step 1.5

Transfer this file to a new chat session to restore full context.

---

## Project: MiniCPM-o 4.5 inference with tool-calling routing
**Workspace:** `/home/ubuntu/github/minicpm-o-4_5/`

### Python Environments

| Env name | Python | Purpose | Key deps |
|----------|--------|---------|----------|
| `local-llm` | 3.12.11 | MiniCPM-o 4.5 inference, Gradio UI, model server | transformers==4.51.0, torch 2.8.0+cu128, gradio 6.5.1, fastapi, uvicorn, httpx |
| `music-gen` | 3.11.12 | ACE-Step 1.5 music generation | transformers 4.57.6, torch 2.10.0+cu128, ace-step 1.5.0 (editable from repo) |

Activate with: `eval "$(pyenv init -)" && pyenv activate <env-name>`

### Model Paths (external disk)

| Model | Path | Size |
|-------|------|------|
| MiniCPM-o 4.5 | `/home/ubuntu/karthik-ragunath-ananda-kumar-utah/MiniCPM-o-4_5` | ~17 GB |
| ACE-Step 1.5 | `/home/ubuntu/karthik-ragunath-ananda-kumar-utah/Ace-Step1.5` | 9.4 GB |

### Files in workspace (`/home/ubuntu/github/minicpm-o-4_5/`)

| File | Description |
|------|-------------|
| `inference_tool_calling.py` | Core module: keyword classifier (`classify_query`), tool registry (`called_cat`, `call_dog`), background thread dispatch, model loading, system prompt, interactive CLI chat loop. Exports `MAX_CONTEXT_TURNS=10` for sliding window. |
| `model_server.py` | FastAPI server that loads MiniCPM-o 4.5 once and exposes `POST /chat` and `GET /health` endpoints on port 8000. Keeps model in GPU memory persistently. |
| `app.py` | Gradio 6 web chat UI on port 7860. Calls `model_server.py` via HTTP (no model loaded locally). Tool routing + background dispatch still works. Sliding window context (10 turns). Uses `_extract_text()` helper for Gradio 6 message format compatibility. |
| `test_tool_calling.py` | Automated tests: (1) classifier, (2) background dispatch, (3) full model inference with 3 queries. Pass `--no-model` to skip GPU tests. All tests pass. |
| `inference_ace_step.py` | ACE-Step 1.5 text-to-music inference script. Uses `acestep.handler.AceStepHandler` + `acestep.llm_inference.LLMHandler`. Supports `--caption`, `--lyrics`, `--instrumental`, `--duration`, `--demo` flags. Requires `music-gen` env. |
| `setup_ace_step.sh` | Setup script to create the `music-gen` pyenv virtualenv and install ACE-Step. Already run — env exists. |
| `start.sh` | Launcher: `./start.sh` (both), `./start.sh server`, `./start.sh ui` |
| `commands.txt` | Terminal commands for model server + Gradio UI |
| `requirements.txt` | Deps for `local-llm` env (transformers, torch, gradio, fastapi, uvicorn, httpx, huggingface_hub) |
| `.vscode/launch.json` | Debug configs: Model Server, Web Chat (Gradio UI), Web Chat + share, Test Tool Calling (full), Test Tool Calling (no model), Interactive Chat (CLI) |

### External repos

| Repo | Path | Purpose |
|------|------|---------|
| ACE-Step-1.5 | `/home/ubuntu/github/ACE-Step-1.5/` | Cloned from `github.com/ace-step/ACE-Step-1.5`. Installed as editable in `music-gen` env. Contains pipeline code (`acestep/` package). |

### Architecture: Split model server + UI

```
model_server.py (port 8000)          app.py (port 7860)
┌────────────────────────┐           ┌────────────────────────┐
│ MiniCPM-o 4.5 on GPU   │◄─HTTP───►│ Gradio 6 chat UI       │
│ POST /chat              │           │ Tool routing (keywords)│
│ GET /health             │           │ Background dispatch    │
│ Loads once, stays up    │           │ Sliding window (10 t.) │
└────────────────────────┘           └────────────────────────┘
  Start once, keep running             Restart freely for debugging
```

### How to run

**MiniCPM-o 4.5 (chat with tool calling):**
```bash
# Terminal 1 — model server (leave running)
pyenv activate local-llm
cd /home/ubuntu/github/minicpm-o-4_5
python model_server.py

# Terminal 2 — Gradio UI (restart freely)
pyenv activate local-llm
cd /home/ubuntu/github/minicpm-o-4_5
python app.py
```

**ACE-Step 1.5 (music generation):**
```bash
pyenv activate music-gen
cd /home/ubuntu/github/minicpm-o-4_5
python inference_ace_step.py --caption "calm ambient piano" --duration 30
python inference_ace_step.py --demo   # run 3 built-in demos
```

### Current status

- MiniCPM-o 4.5: Fully working. All tests pass. Model server + Gradio UI validated end-to-end.
- ACE-Step 1.5: `music-gen` env created with Python 3.11.12, ace-step installed (editable), torch 2.10.0+cu128 installed. Some optional deps missing (torchvision, torchao, torchcodec, uvicorn) but core inference deps are present. Inference script written but not yet test-run.

### Hardware

- GPU: NVIDIA H100 PCIe
- Local disk: ~231 GB free on `/`
- External disk: 1.9 PB at `/home/ubuntu/karthik-ragunath-ananda-kumar-utah/`

### Git

- Repo initialized, branch: `master`, 1 commit (`24adcfd dev - initial commit`)
- Remote: `github.com:Karthik-Ragunath/minicpm-o-4_5.git` (not yet pushed — use `git push -u origin master` or rename branch to `main` first)
