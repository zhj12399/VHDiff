from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import torch

from contextlib import contextmanager
import random
from .composition import CompositePredictor


@contextmanager
def _frozen_inference(model, device):

    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()

    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.eval().requires_grad_(False)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        for module, training in modes:
            module.training = training
        for parameter, requires_grad in gradients:
            parameter.requires_grad_(requires_grad)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)

NUM_SAMPLES = 6000
DEFAULT_SAMPLER = {
    "steps": 100, "eta": 0.0, "guidance": 1.0, "batch_size": 4,
    "precision": "fp32", "clip_x0": True,
    "time_grid": "uniform", "grid_power": 1.0,
}
NOISE_SEED_SCHEME = "splitmix64-v1(seed xor E2DD1A5E53C09B71 xor ((timestep+1)*D1342543DE82EF95 mod 2^64)); low63"


def transition_seed(seed: int, timestep: int) -> int:

    mask = (1 << 64) - 1


    time_key = ((timestep + 1) * 0xD1342543DE82EF95) & mask
    value = (seed ^ 0xE2DD1A5E53C09B71 ^ time_key) & mask
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


def _seed(row):
    seed = row.get("seed", row.get("loser_seed"))
    if "seed" in row and "loser_seed" in row and row["seed"] != row["loser_seed"]:
        raise ValueError("candidate seed and loser_seed disagree")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("candidate seed must be an integer in [0, 2**63)")
    return seed


def _validate_grid_options(time_grid, grid_power):
    if not isinstance(time_grid, str) or time_grid not in ("uniform", "power"):
        raise ValueError("time_grid must be 'uniform' or 'power'")
    if (isinstance(grid_power, bool) or not isinstance(grid_power, (int, float))
            or not math.isfinite(grid_power) or grid_power <= 0):
        raise ValueError("grid_power must be a finite positive number")
    if time_grid == "uniform" and grid_power != 1.0:
        raise ValueError("uniform time_grid requires grid_power=1.0; do not silently ignore an exponent")


def sampling_timesteps(diffusion, steps, time_grid="uniform", grid_power=1.0):

    _validate_grid_options(time_grid, grid_power)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= diffusion.timesteps:
        raise ValueError("steps must be a positive integer within the diffusion schedule")
    if time_grid == "uniform":
        raw = list(diffusion.ddim_timesteps(steps))
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
            raise ValueError("DDIM time grid must contain integer indices")
        indices = [int(value) for value in raw]
    else:
        if steps < 2:
            raise ValueError("power time_grid needs at least two steps to include both endpoints")
        indices = [int(round((diffusion.timesteps - 1) * (index / (steps - 1)) ** grid_power))
                   for index in range(steps - 1, -1, -1)]
    if (not indices or len(indices) != steps
            or any(not 0 <= value < diffusion.timesteps for value in indices)
            or any(left <= right for left, right in zip(indices, indices[1:]))):
        raise ValueError("DDIM time grid must contain exactly the requested number of strictly descending unique valid steps; rounding duplicates are not removed")
    if indices[0] != diffusion.timesteps - 1 or (steps > 1 and indices[-1] != 0):
        raise ValueError("DDIM time grid must include the highest noisy index and end at zero")
    return indices


def _validate(model, diffusion, candidates, sampler_config, device):
    if not isinstance(sampler_config, dict):
        raise ValueError("sampler_config must be a dictionary")
    unknown = set(sampler_config) - set(DEFAULT_SAMPLER)
    if unknown:
        raise ValueError(f"unknown sampler options: {sorted(unknown)}")
    config = {**DEFAULT_SAMPLER, **sampler_config}
    _validate_grid_options(config["time_grid"], config["grid_power"])
    for key in ("steps", "batch_size"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["steps"] > diffusion.timesteps:
        raise ValueError("steps cannot exceed the diffusion schedule")
    for key, low, high in (("eta", 0.0, 1.0), ("guidance", 1.0, 3.0)):
        value = config[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high):
            raise ValueError(f"{key} must be finite and in [{low}, {high}]")
    if config["precision"] != "fp32" or config["clip_x0"] is not True:
        raise ValueError("Stage E improvement requires fp32 and clip_x0=True")
    if diffusion.prediction_type != "v" or diffusion.data_scale != 1.0:
        raise ValueError("Stage E improvement requires the selected v/scale1 generator")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= config["batch_size"]:
        raise ValueError("candidates must be a nonempty list no larger than batch_size")
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError("each candidate must be a dictionary")
        for key in ("values", "known"):
            if not isinstance(row.get(key), (list, tuple)) or len(row[key]) != 5:
                raise ValueError(f"candidate {key} must contain exactly five entries")
        if any(value not in (0, 1) for value in row["values"]):
            raise ValueError("candidate values must be binary")
        if any(flag != 1 for flag in row["known"]):
            raise ValueError("only complete known five-label conditions are allowed")
        _seed(row)
    device = torch.device(device)
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type != device.type or (device.index is not None and tensor.device.index != device.index):
            raise ValueError("reference model must already be on the sampling device")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError("reference model floating tensors must already be fp32")
    return config, device


def ddim_sigma(alpha, previous_alpha, eta):

    variance = (1 - previous_alpha) / (1 - alpha) * (1 - alpha / previous_alpha)
    return eta * variance.clamp_min(0).sqrt()


def _normal_rows(seeds, device, timestep=None):
    rows = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed if timestep is None else transition_seed(seed, timestep))
        rows.append(torch.randn((1, 1, NUM_SAMPLES), generator=generator,
                                device=device, dtype=torch.float32))
    return torch.cat(rows, dim=0)


def sample_batch(model, diffusion, candidates: list[dict], sampler_config: dict, device):

    config, device = _validate(model, diffusion, candidates, sampler_config, device)
    count = len(candidates)
    indices = sampling_timesteps(diffusion, config["steps"], config["time_grid"], config["grid_power"])
    seeds = [_seed(row) for row in candidates]
    values = torch.tensor([row["values"] for row in candidates], device=device, dtype=torch.long)
    known = torch.ones_like(values)
    null_values = torch.zeros_like(values)
    null_known = torch.zeros_like(known)
    predictor = CompositePredictor(model, values, known)
    with _frozen_inference(model, device):
        x = _normal_rows(seeds, device)
        coefficients = diffusion.alpha_bar.to(device=device, dtype=torch.float32)
        if (coefficients.ndim != 1 or len(coefficients) != diffusion.timesteps
                or not bool(torch.isfinite(coefficients).all() & (coefficients > 0).all() & (coefficients < 1).all())
                or not bool((coefficients[:-1] > coefficients[1:]).all())):
            raise ValueError("diffusion alpha_bar must be strictly decreasing finite values in (0,1)")
        invalid = torch.zeros(count, device=device, dtype=torch.bool)
        first_step = torch.full((count,), -1, device=device, dtype=torch.long)
        first_timestep = first_step.clone()
        outside_count = torch.zeros(count, device=device, dtype=torch.int64)
        preclip_peak = torch.zeros(count, device=device, dtype=torch.float32)
        clip_fraction_by_step = torch.full(
            (count, len(indices)), float("nan"), device=device, dtype=torch.float64,
        )
        preclip_peak_by_step = torch.full_like(clip_fraction_by_step, float("nan"))
        sigmas = []
        for position, index in enumerate(indices):
            t = torch.full((count,), index, device=device, dtype=torch.long)
            prediction = predictor(x, t).float()
            if config["guidance"] != 1.0:
                null = model(x, t, null_values, null_known).float()
                prediction = null + config["guidance"] * (prediction - null)
            if prediction.shape != x.shape:
                raise ValueError("reference prediction changed the requested waveform shape")
            alpha = coefficients[index]

            proposal_x0 = alpha.sqrt() * x - (1 - alpha).sqrt() * prediction
            epsilon = (1 - alpha).sqrt() * x + alpha.sqrt() * prediction
            finite = (torch.isfinite(prediction).flatten(1).all(1)
                      & torch.isfinite(proposal_x0).flatten(1).all(1)
                      & torch.isfinite(epsilon).flatten(1).all(1))
            newly_invalid = ~finite & ~invalid
            first_step[newly_invalid] = position + 1
            first_timestep[newly_invalid] = index
            invalid |= ~finite
            live = ~invalid
            safe_proposal = torch.where(live[:, None, None], proposal_x0, torch.zeros_like(proposal_x0))
            step_outside_count = (safe_proposal.abs() > 1).flatten(1).sum(1)
            outside_count += step_outside_count
            clip_fraction_by_step[live, position] = step_outside_count[live].double() / NUM_SAMPLES
            step_peak = safe_proposal.abs().flatten(1).amax(1)
            preclip_peak_by_step[live, position] = step_peak[live].double()
            preclip_peak = torch.maximum(preclip_peak, step_peak)
            predicted_x0 = safe_proposal.clamp(-1, 1)
            epsilon = (x - alpha.sqrt() * predicted_x0) / (1 - alpha).sqrt()
            previous_index = indices[position + 1] if position + 1 < len(indices) else -1
            if previous_index < 0:
                sigma = torch.zeros((), device=device, dtype=torch.float32)
                x = predicted_x0
            else:
                previous_alpha = coefficients[previous_index]
                if config["eta"] == 0.0:
                    sigma = torch.zeros((), device=device, dtype=torch.float32)
                    x = previous_alpha.sqrt() * predicted_x0 + (1 - previous_alpha).sqrt() * epsilon
                else:
                    sigma = ddim_sigma(alpha, previous_alpha, config["eta"])
                    direction = (1 - previous_alpha - sigma.square()).clamp_min(0).sqrt()
                    noise = _normal_rows(seeds, device, timestep=index)
                    x = previous_alpha.sqrt() * predicted_x0 + direction * epsilon + sigma * noise
            sigmas.append(float(sigma.cpu()))
            finite_update = torch.isfinite(x).flatten(1).all(1)
            newly_invalid = ~finite_update & ~invalid
            first_step[newly_invalid] = position + 1
            first_timestep[newly_invalid] = index
            invalid |= ~finite_update
            clip_fraction_by_step[invalid, position] = float("nan")
            preclip_peak_by_step[invalid, position] = float("nan")
            x = torch.where(invalid[:, None, None], torch.zeros_like(x), x)

        waves = x[:, 0].cpu().numpy().copy().astype(np.float32, copy=False)
        invalid_rows = invalid.cpu().tolist()
        peaks = preclip_peak.cpu().tolist()
        clipped = outside_count.cpu().tolist()
        bad_steps, bad_timesteps = first_step.cpu().tolist(), first_timestep.cpu().tolist()
        clip_histories = clip_fraction_by_step.cpu().tolist()
        peak_histories = preclip_peak_by_step.cpu().tolist()
        result = []
        for row_index, seed in enumerate(seeds):
            failed = bool(invalid_rows[row_index])
            if failed:
                waves[row_index] = np.nan
            peak = None if failed else float(peaks[row_index])
            final_peak = None if failed else float(np.abs(waves[row_index]).max())
            result.append({
                "loser_seed": seed, "seed": seed,
                "nonfinite_failure": failed,
                "failure_reason": "nonfinite_reference_prediction_or_ddim_state" if failed else None,
                "first_nonfinite_step": bad_steps[row_index] if failed else None,
                "first_nonfinite_timestep": bad_timesteps[row_index] if failed else None,
                "sampling_steps": len(indices),
                "requested_sampling_steps": config["steps"],
                "time_grid": config["time_grid"], "grid_power": float(config["grid_power"]),
                "model_forward_calls": len(indices),
                "composite_forward_calls": len(indices),
                "native_branch_forward_calls": (6 + int(config["guidance"] != 1.0)) * len(indices),
                "additional_guidance_null_forward_calls": int(config["guidance"] != 1.0) * len(indices),
                "forward_count_scope": "per candidate scheduled steps; failed rows are not retried",
                "clip_x0": True,
                "x0_clip_fraction": None if failed else float(clipped[row_index] / (NUM_SAMPLES * len(indices))),
                "ddim_timesteps": list(indices),
                "x0_clip_fraction_per_step": [float(v) if math.isfinite(v) else None for v in clip_histories[row_index]],
                "preclip_x0_abs_max_per_step": [float(v) if math.isfinite(v) else None for v in peak_histories[row_index]],
                "preclip_x0_abs_max": peak,
                "preclip_x0_abs_max_model": peak,
                "preclip_x0_abs_max_physical": peak,
                "final_abs_max_model": final_peak,
                "final_abs_max_physical": final_peak,
                "prediction_type": "v", "data_scale": 1.0, "precision": "fp32",
                "eta": float(config["eta"]), "guidance": float(config["guidance"]),
                "sigma_per_step": sigmas,
                "transition_noise_seeds": [
                    transition_seed(seed, index) if position + 1 < len(indices) and config["eta"] > 0 else None
                    for position, index in enumerate(indices)
                ],
                "transition_noise_seed_scheme": NOISE_SEED_SCHEME,
                "guidance_formula": "v_null + s * (v_composite - v_null); s=1 directly calls CompositePredictor",
            })
    return waves, result
