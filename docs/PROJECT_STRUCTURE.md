# Project Structure

```
Multimodal-risk-assessment/
├─ api/
│  └─ main.py                 # FastAPI entry: POST /verify-video
├─ backend/
│  ├─ unified_engine.py       # UnifiedVideoIntelligenceEngine coordinator
│  ├─ scoring.py              # Final confidence + label rules
│  └─ models/
│     ├─ nlp_intent.py        # HF model + rule cues for intent
│     └─ voice_stress.py      # Lightweight voice stress score
├─ multimodal_coercion/
│  ├─ core/                   # config, logging, registry, persistence
│  ├─ engine/
│  │  └─ engine.py            # Delegates to backend.unified_engine
│  ├─ facial_emotion/         # video frames + TF emotion
│  ├─ speech/                 # Whisper STT and classifier (for reuse)
│  ├─ ui/
│  │  └─ app.py               # Streamlit demo (single video upload)
│  ├─ configs/                # models.yaml, thresholds.yaml, etc.
│  └─ requirements.txt
├─ frontend/                  # React Vite app (optional UI)
├─ .venv/                     # Active virtualenv (kept)
└─ docs/
   └─ PROJECT_STRUCTURE.md    # This file
```

Notes
- Single upload path: UI → /verify-video → unified_engine runs STT, NLP, emotion, voice stress → unified JSON.
- Keep `.venv/` only; duplicate `venv/` removed.
- Root `package-lock.json` removed (frontend has its own lockfile).

Run
- Streamlit UI:
  - `streamlit run multimodal_coercion/ui/app.py`
- API:
  - `uvicorn api.main:app --host 0.0.0.0 --port 8000`

Minimal Python deps for classifier UI
- `streamlit`, `torch`, `transformers`, `sentencepiece`, `tokenizers`

