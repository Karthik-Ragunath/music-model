"""
ACE-Step 1.5 — Model Server (FastAPI)

Loads the ACE-Step model (DiT + LLM + VAE) once at startup and serves
music generation requests via a REST API.

Usage:
    pyenv activate music-gen
    python model_server.py                        # default: port 8190
    python model_server.py --port 8190 --host 0.0.0.0

Endpoints:
    POST /generate      Generate music from caption + lyrics → returns FLAC audio
    GET  /health        Health check
"""

import argparse
import io
import os
import sys
import time
import tempfile
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# Ensure the ACE-Step repo is importable
REPO_DIR = "/home/ubuntu/github/ACE-Step-1.5"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

MODEL_DIR = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/Ace-Step1.5"


# =============================================================================
# Request / Response schemas
# =============================================================================

class GenerateRequest(BaseModel):
    """Request body for /generate endpoint."""
    caption: str = Field(..., description="Text description of desired music (e.g. 'upbeat pop song')")
    lyrics: str = Field(default="", description="Song lyrics. Use '[Instrumental]' for no vocals.")
    instrumental: bool = Field(default=False, description="If True, force instrumental output")
    duration: float = Field(default=30.0, ge=5.0, le=600.0, description="Duration in seconds")
    bpm: Optional[int] = Field(default=None, ge=30, le=300, description="BPM (auto if not set)")
    seed: int = Field(default=-1, description="Random seed (-1 = random)")
    batch_size: int = Field(default=1, ge=1, le=4, description="Number of variations")
    inference_steps: int = Field(default=8, ge=1, le=100, description="Diffusion steps (8 for turbo)")
    shift: float = Field(default=3.0, description="Timestep shift (3.0 for turbo)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_name: str


# =============================================================================
# Global model state
# =============================================================================

dit_handler = None
llm_handler = None
_model_loaded = False
_device = "cuda"


def load_models(device: str = "cuda"):
    """Load DiT + LLM handlers. Called once at startup."""
    global dit_handler, llm_handler, _model_loaded, _device
    _device = device

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    logger.info("Loading DiT handler ...")
    t0 = time.time()
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=MODEL_DIR,
        config_path="acestep-v15-turbo",
        device=device,
    )
    logger.info(f"  DiT ready in {time.time() - t0:.1f}s")

    logger.info("Loading LLM handler ...")
    t0 = time.time()
    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=MODEL_DIR,
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="transformers",
        device=device,
    )
    logger.info(f"  LLM ready in {time.time() - t0:.1f}s")

    _model_loaded = True
    logger.info("All models loaded and ready to serve.")


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(
    title="ACE-Step 1.5 Model Server",
    description="Music generation API — send caption + lyrics, receive FLAC audio",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return HealthResponse(
        status="ok" if _model_loaded else "loading",
        model_loaded=_model_loaded,
        device=_device,
        gpu_name=gpu_name,
    )


@app.post("/generate")
def generate_music_endpoint(req: GenerateRequest):
    """
    Generate music from caption + lyrics.

    Returns FLAC audio as binary response with Content-Type: audio/flac.
    Also includes metadata in response headers:
        X-Sample-Rate, X-Seed, X-Duration, X-Generation-Time
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading. Try again shortly.")

    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    # Build generation params
    lyrics = req.lyrics
    if req.instrumental and not lyrics:
        lyrics = "[Instrumental]"

    params = GenerationParams(
        task_type="text2music",
        caption=req.caption,
        lyrics=lyrics,
        instrumental=req.instrumental,
        duration=req.duration,
        bpm=req.bpm,
        inference_steps=req.inference_steps,
        shift=req.shift,
        infer_method="ode",
        seed=req.seed,
        thinking=True,
    )

    config = GenerationConfig(
        batch_size=req.batch_size,
        audio_format="flac",
    )

    # Create a temp directory for saving audio files
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"Generating: caption='{req.caption[:80]}', duration={req.duration}s, seed={req.seed}")
        t0 = time.time()

        result = generate_music(
            dit_handler,
            llm_handler,
            params,
            config,
            save_dir=tmpdir,
        )
        elapsed = time.time() - t0

        if not result.success:
            raise HTTPException(status_code=500, detail=f"Generation failed: {result.error}")

        if not result.audios:
            raise HTTPException(status_code=500, detail="No audio generated")

        # Read the first generated audio file
        audio_info = result.audios[0]
        audio_path = audio_info.get("path", "")

        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Audio file not found on disk")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Extract metadata
        sample_rate = audio_info.get("sample_rate", 48000)
        seed_used = audio_info.get("params", {}).get("seed", -1)
        time_costs = result.extra_outputs.get("time_costs", {})

        logger.info(f"  Done in {elapsed:.1f}s, file size: {len(audio_bytes)/1024:.0f} KB")

        # Return audio as binary with safe metadata in headers
        headers = {
            "X-Sample-Rate": str(sample_rate),
            "X-Seed": str(seed_used),
            "X-Duration": str(req.duration),
            "X-Generation-Time": f"{elapsed:.2f}",
            "Content-Disposition": 'attachment; filename="ace_step_output.flac"',
        }

        # Log the status message server-side (don't put it in headers — too fragile)
        if result.status_message:
            logger.info(f"  Status: {result.status_message}")

        return Response(
            content=audio_bytes,
            media_type="audio/flac",
            headers=headers,
        )


@app.post("/generate_json")
def generate_music_json(req: GenerateRequest):
    """
    Same as /generate, but returns JSON with base64-encoded audio.
    Useful for clients that can't easily handle binary responses.
    """
    import base64

    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading. Try again shortly.")

    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    lyrics = req.lyrics
    if req.instrumental and not lyrics:
        lyrics = "[Instrumental]"

    params = GenerationParams(
        task_type="text2music",
        caption=req.caption,
        lyrics=lyrics,
        instrumental=req.instrumental,
        duration=req.duration,
        bpm=req.bpm,
        inference_steps=req.inference_steps,
        shift=req.shift,
        infer_method="ode",
        seed=req.seed,
        thinking=True,
    )

    config = GenerationConfig(
        batch_size=req.batch_size,
        audio_format="flac",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"Generating (JSON): caption='{req.caption[:80]}', duration={req.duration}s")
        t0 = time.time()

        result = generate_music(
            dit_handler,
            llm_handler,
            params,
            config,
            save_dir=tmpdir,
        )
        elapsed = time.time() - t0

        if not result.success:
            raise HTTPException(status_code=500, detail=f"Generation failed: {result.error}")

        if not result.audios:
            raise HTTPException(status_code=500, detail="No audio generated")

        # Encode all audio files as base64
        audio_results = []
        for audio_info in result.audios:
            audio_path = audio_info.get("path", "")
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            else:
                audio_b64 = ""

            audio_results.append({
                "audio_base64": audio_b64,
                "sample_rate": audio_info.get("sample_rate", 48000),
                "seed": audio_info.get("params", {}).get("seed", -1),
                "format": "flac",
            })

        time_costs = result.extra_outputs.get("time_costs", {})

        return JSONResponse({
            "success": True,
            "generation_time": round(elapsed, 2),
            "audios": audio_results,
            "status_message": result.status_message,
            "time_costs": {k: round(v, 3) if isinstance(v, float) else v for k, v in time_costs.items()},
        })


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ACE-Step 1.5 Model Server")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=8190, help="Bind port (default 8190)")
    p.add_argument("--device", default="cuda", help="Device (default cuda)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load models before starting the server
    load_models(device=args.device)

    # Start uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        workers=1,  # Must be 1 — models are in-process
    )
