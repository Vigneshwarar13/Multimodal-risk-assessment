from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from backend.unified_engine import verify_video


app = FastAPI(title="Multimodal Willingness Verification Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/verify-video")
async def verify_video_route(file: UploadFile = File(...)):
    suffix = ".mp4"
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = verify_video(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return result
