"""
ACE-Step 1.5 — Text-to-Music Inference Script

Generates music from text prompts using the ACE-Step 1.5 model.

Requirements:
    - Run setup_ace_step.sh first to create the ace-step virtualenv
    - pyenv activate ace-step

Usage:
    python inference_ace_step.py                          # default demo
    python inference_ace_step.py --caption "jazz piano"   # custom prompt
    python inference_ace_step.py --lyrics lyrics.txt      # with lyrics file
    python inference_ace_step.py --instrumental            # no vocals

Model: /home/ubuntu/karthik-ragunath-ananda-kumar-utah/Ace-Step1.5
Repo:  /home/ubuntu/github/ACE-Step-1.5
"""

import argparse
import os
import sys
import time

# Ensure the ACE-Step repo is importable
REPO_DIR = "/home/ubuntu/github/ACE-Step-1.5"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

MODEL_DIR = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/Ace-Step1.5"
OUTPUT_DIR = "/home/ubuntu/github/minicpm-o-4_5/output_music"


def run_generation(
    caption: str,
    lyrics: str = "",
    instrumental: bool = False,
    duration: float = 30.0,
    bpm: int | None = None,
    batch_size: int = 1,
    seed: int = -1,
    device: str = "cuda",
):
    """Generate music using the ACE-Step 1.5 pipeline."""
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- Initialize DiT handler (main generation model) -----
    print("Initializing DiT handler ...")
    t0 = time.time()
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=MODEL_DIR,
        config_path="acestep-v15-turbo",
        device=device,
    )
    print(f"  DiT ready in {time.time() - t0:.1f}s")

    # ----- Initialize LLM handler (song planner / CoT) -----
    print("Initializing LLM handler ...")
    t0 = time.time()
    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=MODEL_DIR,
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="transformers",   # use HF transformers (no vllm needed)
        device=device,
    )
    print(f"  LLM ready in {time.time() - t0:.1f}s")

    # ----- Set up generation params -----
    if instrumental and not lyrics:
        lyrics = "[Instrumental]"

    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        instrumental=instrumental,
        duration=duration,
        bpm=bpm,
        inference_steps=8,       # turbo model sweet spot
        shift=3.0,               # recommended for turbo
        infer_method="ode",
        seed=seed,
        thinking=True,           # enable LM chain-of-thought
    )

    config = GenerationConfig(
        batch_size=batch_size,
        audio_format="flac",
    )

    # ----- Generate! -----
    print(f"\nGenerating music ...")
    print(f"  Caption:      {caption}")
    print(f"  Lyrics:       {'(instrumental)' if instrumental else (lyrics[:80] + '...' if len(lyrics) > 80 else lyrics) or '(none — LM will generate)'}")
    print(f"  Duration:     {duration}s")
    print(f"  BPM:          {bpm or 'auto'}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Seed:         {seed if seed >= 0 else 'random'}")
    print()

    t0 = time.time()
    result = generate_music(
        dit_handler,
        llm_handler,
        params,
        config,
        save_dir=OUTPUT_DIR,
    )
    elapsed = time.time() - t0

    # ----- Report results -----
    if result.success:
        print(f"\nGeneration succeeded in {elapsed:.1f}s")
        for i, audio in enumerate(result.audios, 1):
            print(f"  [{i}] {audio['path']}")
            print(f"      Sample rate: {audio['sample_rate']} Hz")
            print(f"      Seed: {audio['params'].get('seed', '?')}")

        # Print time breakdown
        tc = result.extra_outputs.get("time_costs", {})
        if tc:
            print(f"\n  Time breakdown:")
            for k, v in tc.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.2f}s")
    else:
        print(f"\nGeneration FAILED: {result.error}")
        print(f"  Status: {result.status_message}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMO_PROMPTS = [
    {
        "caption": "calm ambient music with soft piano and gentle strings, relaxing and dreamy",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "duration": 30.0,
    },
    {
        "caption": "upbeat electronic dance music with heavy bass and synthesizer leads",
        "lyrics": "",
        "instrumental": True,
        "duration": 30.0,
    },
    {
        "caption": "acoustic folk song with warm vocals and gentle guitar picking",
        "lyrics": (
            "[Verse 1]\n"
            "Walking down the morning road\n"
            "Sunlight breaking through the trees\n"
            "Every step a story told\n"
            "Carried softly on the breeze\n"
            "\n"
            "[Chorus]\n"
            "We are the dreamers of the dawn\n"
            "Singing until the night is gone\n"
        ),
        "instrumental": False,
        "duration": 45.0,
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="ACE-Step 1.5 Music Generation")
    p.add_argument("--caption", type=str, default=None, help="Text description of desired music")
    p.add_argument("--lyrics", type=str, default=None, help="Lyrics text or path to a .txt file")
    p.add_argument("--instrumental", action="store_true", help="Generate instrumental (no vocals)")
    p.add_argument("--duration", type=float, default=30.0, help="Duration in seconds (default 30)")
    p.add_argument("--bpm", type=int, default=None, help="Beats per minute (auto if not set)")
    p.add_argument("--batch-size", type=int, default=1, help="Number of variations (default 1)")
    p.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    p.add_argument("--device", default="cuda", help="Device (default cuda)")
    p.add_argument("--demo", action="store_true", help="Run all 3 built-in demo prompts")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        for i, prompt in enumerate(DEMO_PROMPTS, 1):
            print("=" * 60)
            print(f"  DEMO {i}/{len(DEMO_PROMPTS)}")
            print("=" * 60)
            run_generation(
                caption=prompt["caption"],
                lyrics=prompt["lyrics"],
                instrumental=prompt["instrumental"],
                duration=prompt["duration"],
                batch_size=args.batch_size,
                seed=args.seed,
                device=args.device,
            )
            print()
        return

    # Custom prompt
    caption = args.caption or "calm ambient music with soft piano and gentle strings"
    lyrics = ""
    if args.lyrics:
        if os.path.isfile(args.lyrics):
            with open(args.lyrics) as f:
                lyrics = f.read()
        else:
            lyrics = args.lyrics

    run_generation(
        caption=caption,
        lyrics=lyrics,
        instrumental=args.instrumental,
        duration=args.duration,
        bpm=args.bpm,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
