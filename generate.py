from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import time

import numpy as np
import soundfile as sf
import torch

from vhdiff.inference import generate_batch, load_model

LABELS = ("AS", "AR", "MS", "MR", "TR")
ROOT = Path(__file__).resolve().parent


def parse_condition(text):
    text = text.strip().upper()
    if len(text) == 5 and set(text) <= {"0", "1"}:
        return text
    if text in ("NORMAL", "N"):
        return "00000"
    disease_names = [name.strip() for name in text.replace("+", ",").split(",")]
    if not disease_names or any(name not in LABELS for name in disease_names):
        raise ValueError(f"Unknown condition {text!r}. Use Normal, AS,AR,MS,MR,TR, or a five bit code.")
    return "".join("1" if label in disease_names else "0" for label in LABELS)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate three second heart sounds for selected valve disease configurations.')
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/vhdiff.pt")
    parser.add_argument("--condition", action="append", help="Normal, MS, AS+MS, AS,MR,TR, or 10100. Repeat for several requests.")
    parser.add_argument("--all", action="store_true", help="Generate every one of the 32 complete configurations")
    parser.add_argument("--num-samples", type=int, default=2, help="Clips per requested configuration")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=None, help="Optional replay token. Omit for independent fresh random noise.")
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.75)
    args = parser.parse_args(argv)
    if args.num_samples < 1 or not 1 <= args.batch_size <= 32:
        parser.error("num-samples must be positive and batch-size must be between 1 and 32")
    if args.all and args.condition:
        parser.error("Choose --all or --condition")
    if not args.all and not args.condition:
        parser.error("Specify --condition Normal, --condition AS+MS, or --all")
    if args.seed is not None and not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    try:
        conditions = [f"{i:05b}" for i in range(32)] if args.all else list(dict.fromkeys(map(parse_condition, args.condition)))
    except ValueError as error:
        parser.error(str(error))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model, diffusion, sampler = load_model(args.checkpoint, device)
    sampler.update(batch_size=args.batch_size, guidance=args.guidance, eta=args.eta)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata, request_index = [], 0
    started = time.monotonic()
    used_tokens = set()
    for code in conditions:
        name = "_".join(label for label, bit in zip(LABELS, code) if bit == "1") or "Normal"
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        for first in range(0, args.num_samples, args.batch_size):
            count = min(args.batch_size, args.num_samples-first)
            requests = []
            for index in range(count):
                token = (args.seed + request_index) % 2**63 if args.seed is not None else secrets.randbits(63)
                while token in used_tokens:
                    token = secrets.randbits(63)
                used_tokens.add(token)
                requests.append(dict(condition_id=code, values=[int(v) for v in code], known=[1]*5,
                                     seed=token, loser_seed=token))
                request_index += 1
            waves, diagnostics = generate_batch(model, diffusion, requests, sampler, device)
            if waves.shape != (count, 6000) or not np.isfinite(waves).all():
                raise RuntimeError("Generation produced an invalid waveform. No invalid audio is saved.")
            for offset, (wave, row, diagnostic) in enumerate(zip(waves, requests, diagnostics), first+1):
                path = folder / f"{offset:04d}.wav"
                sf.write(path, wave, 2000, subtype="PCM_16")
                metadata.append(dict(file=str(path.relative_to(args.output)), condition=code,
                                     diseases=[label for label, bit in zip(LABELS, code) if bit == "1"],
                                     noise_token=row["seed"], sample_rate=2000, duration_seconds=3,
                                     sampler=sampler, final_peak=diagnostic["final_abs_max_physical"]))
            print(f"{name}: {first+count}/{args.num_samples} saved", flush=True)
    (args.output / "generation.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(f"Generated {len(metadata)} clips on {device} in {time.monotonic()-started:.1f} s")


if __name__ == "__main__":
    main()
