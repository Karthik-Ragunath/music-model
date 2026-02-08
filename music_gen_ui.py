"""
ACE-Step 1.5 — Gradio Client UI

A browser-based UI for generating music. Connects to the model server
(model_server.py) running on a separate process/port.

Usage:
    # First, start the model server:
    python model_server.py --port 8190

    # Then, start this Gradio UI:
    python gradio_client.py                          # default: port 7860
    python gradio_client.py --port 7860 --server-url http://localhost:8190

Open http://localhost:7860 in your browser.
"""

import argparse
import base64
import os
import tempfile
import time
from pathlib import Path

import requests
import gradio as gr


# =============================================================================
# Server communication
# =============================================================================

DEFAULT_SERVER_URL = "http://localhost:8190"


def check_server_health(server_url: str) -> dict:
    """Check if the model server is up and models are loaded."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


def generate_music(
    caption: str,
    lyrics: str,
    instrumental: bool,
    duration: float,
    bpm: int | None,
    seed: int,
    server_url: str,
) -> tuple[str | None, str]:
    """
    Call the model server to generate music.

    Returns:
        (audio_file_path, status_text)
    """
    if not caption.strip():
        return None, "⚠️ Please provide a caption describing the music."

    # Build request payload
    payload = {
        "caption": caption.strip(),
        "lyrics": lyrics.strip() if lyrics else "",
        "instrumental": instrumental,
        "duration": duration,
        "seed": seed,
        "batch_size": 1,
        "inference_steps": 8,
        "shift": 3.0,
    }
    if bpm and bpm > 0:
        payload["bpm"] = bpm

    # Call server
    status_lines = [f"🎵 Generating {duration}s of music..."]
    try:
        t0 = time.time()
        resp = requests.post(
            f"{server_url}/generate",
            json=payload,
            timeout=max(300, duration * 10),  # generous timeout
        )
        elapsed = time.time() - t0

        if resp.status_code != 200:
            error_detail = "Unknown error"
            try:
                error_detail = resp.json().get("detail", resp.text[:500])
            except Exception:
                error_detail = resp.text[:500]
            return None, f"❌ Server error ({resp.status_code}): {error_detail}"

        # Save the binary FLAC response to a temp file
        tmpdir = tempfile.mkdtemp(prefix="ace_step_ui_")
        seed_used = resp.headers.get("X-Seed", "?")
        gen_time = resp.headers.get("X-Generation-Time", f"{elapsed:.1f}")
        sample_rate = resp.headers.get("X-Sample-Rate", "48000")

        audio_path = os.path.join(tmpdir, "generated.flac")
        with open(audio_path, "wb") as f:
            f.write(resp.content)

        file_size_kb = len(resp.content) / 1024
        status_lines.append(f"✅ Done in {gen_time}s")
        status_lines.append(f"📊 Sample rate: {sample_rate} Hz | Seed: {seed_used} | File: {file_size_kb:.0f} KB")

        return audio_path, "\n".join(status_lines)

    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to model server. Is it running?\n   Start it with: python model_server.py --port 8190"
    except requests.exceptions.Timeout:
        return None, "❌ Request timed out. The model may be overloaded."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# =============================================================================
# Gradio UI
# =============================================================================

DEMO_EXAMPLES = [
    # [caption, lyrics, instrumental, duration, bpm, seed]
    [
        "calm ambient music with soft piano and gentle strings, relaxing and dreamy",
        "[Instrumental]",
        True, 30, 0, -1,
    ],
    [
        "upbeat electronic dance music with heavy bass and synthesizer leads",
        "",
        True, 30, 128, -1,
    ],
    [
        "acoustic folk song with warm vocals and gentle guitar picking",
        "[Verse 1]\nWalking down the morning road\nSunlight breaking through the trees\nEvery step a story told\nCarried softly on the breeze\n\n[Chorus]\nWe are the dreamers of the dawn\nSinging until the night is gone",
        False, 45, 0, -1,
    ],
    [
        "female vocalist singing a catchy pop melody with soft acoustic guitar accompaniment, warm and intimate, clear vocals, English",
        "[Verse]\nHello hello\nHello you stranger\n\n[Chorus]\nHello hello\nHello you stranger",
        False, 30, 0, -1,
    ],
]


def build_ui(server_url: str) -> gr.Blocks:
    """Build the Gradio Blocks UI."""

    with gr.Blocks(
        title="ACE-Step 1.5 — Music Generator",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="blue",
        ),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        .generate-btn { min-height: 50px !important; font-size: 1.1em !important; }
        """,
    ) as demo:

        # Header
        gr.HTML("""
            <h1 class="main-title">🎵 ACE-Step 1.5 — Music Generator</h1>
            <p class="subtitle">Generate music from text descriptions and lyrics</p>
        """)

        with gr.Row():
            # ── Left column: Inputs ──
            with gr.Column(scale=1):
                caption_input = gr.Textbox(
                    label="🎼 Music Description (Caption)",
                    placeholder="e.g. 'upbeat pop song with catchy melody and warm vocals'",
                    lines=3,
                    max_lines=5,
                )

                lyrics_input = gr.Textbox(
                    label="📝 Lyrics",
                    placeholder="Use [Verse], [Chorus] tags for structure.\nLeave empty or type [Instrumental] for instrumental.",
                    lines=8,
                    max_lines=20,
                )

                with gr.Row():
                    instrumental_toggle = gr.Checkbox(
                        label="🎹 Instrumental",
                        value=False,
                    )
                    duration_slider = gr.Slider(
                        label="⏱️ Duration (seconds)",
                        minimum=5, maximum=120, step=5, value=30,
                    )

                with gr.Row():
                    bpm_input = gr.Number(
                        label="💓 BPM (0 = auto)",
                        value=0,
                        precision=0,
                    )
                    seed_input = gr.Number(
                        label="🎲 Seed (-1 = random)",
                        value=-1,
                        precision=0,
                    )

                generate_btn = gr.Button(
                    "🚀 Generate Music",
                    variant="primary",
                    elem_classes=["generate-btn"],
                )

            # ── Right column: Output ──
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="🔊 Generated Audio",
                    type="filepath",
                    interactive=False,
                )

                status_output = gr.Textbox(
                    label="📋 Status",
                    lines=4,
                    interactive=False,
                )

                # Server health indicator
                server_status = gr.Textbox(
                    label="🖥️ Server Status",
                    interactive=False,
                    lines=1,
                )

        # Examples
        gr.Examples(
            examples=DEMO_EXAMPLES,
            inputs=[caption_input, lyrics_input, instrumental_toggle, duration_slider, bpm_input, seed_input],
            label="💡 Example Prompts (click to fill)",
        )

        # ── Event handlers ──
        def on_generate(caption, lyrics, instrumental, duration, bpm, seed):
            bpm_val = int(bpm) if bpm and int(bpm) > 0 else None
            seed_val = int(seed) if seed else -1
            audio_path, status = generate_music(
                caption=caption,
                lyrics=lyrics,
                instrumental=instrumental,
                duration=duration,
                bpm=bpm_val,
                seed=seed_val,
                server_url=server_url,
            )
            return audio_path, status

        generate_btn.click(
            fn=on_generate,
            inputs=[caption_input, lyrics_input, instrumental_toggle, duration_slider, bpm_input, seed_input],
            outputs=[audio_output, status_output],
        )

        def on_load():
            health = check_server_health(server_url)
            if health.get("status") == "ok":
                return f"✅ Connected — {health.get('gpu_name', 'GPU')} | Model loaded"
            elif health.get("status") == "loading":
                return "⏳ Server is up but models are still loading..."
            else:
                return f"❌ Cannot reach server at {server_url} — start it with: python model_server.py"

        demo.load(fn=on_load, outputs=[server_status])

    return demo


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ACE-Step 1.5 Gradio Client UI")
    p.add_argument("--port", type=int, default=7860, help="Gradio UI port (default 7860)")
    p.add_argument("--host", default="0.0.0.0", help="Gradio bind host (default 0.0.0.0)")
    p.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Model server URL (default http://localhost:8190)")
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"🎵 ACE-Step 1.5 Gradio Client")
    print(f"   Model server: {args.server_url}")
    print(f"   UI:           http://{args.host}:{args.port}")
    print()

    demo = build_ui(server_url=args.server_url)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
