"""Peak GPU memory and step time of fine-tuning at long sequence lengths, under a V100-like memory cap.

Token classification, forward + backward + AdamW step on full-length inputs, for:
  * v3-poscopy:   LayoutLMv3-base with 1D position embeddings tiled to the length (--extend_positions)
  * ernie-extrap: ERNIE-Layout with the port's RoPE extrapolation (--extrapolation)
  * v3-window / ernie-window: one 512-token window (what --long_docs window trains on), for reference

Precision: fp16 autocast with a GradScaler (V100 has no bf16) and fp32. The cap (default 15 GB, to leave room under a
16 GB V100) is enforced with torch.cuda.set_per_process_memory_fraction, and SDPA is limited to the kernels V100 runs
(memory-efficient and math), so a configuration that needs more fails with OOM.

Example:
    CUDA_VISIBLE_DEVICES=3 ERNIE_LAYOUT_PYTORCH=../ernie-layout-pytorch python scripts/bench_train_memory.py \
        --out results/long_docs/train_memory.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def batch_for(model_family, length, batch_size, num_labels, device):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(5, 30000, (batch_size, length), generator=g)
    xy = torch.randint(0, 900, (batch_size, length, 2), generator=g)
    bbox = torch.cat(
        [xy, xy + torch.randint(10, 90, (batch_size, length, 2), generator=g)], -1
    ).clamp(max=1000)
    pix = torch.rand(batch_size, 3, 224, 224, generator=g)
    pix = pix * 255 if model_family == "ernie" else pix * 2 - 1
    return {
        "input_ids": ids.to(device),
        "bbox": bbox.to(device),
        "attention_mask": torch.ones_like(ids).to(device),
        "pixel_values": pix.to(device),
        "labels": torch.randint(0, num_labels, (batch_size, length), generator=g).to(
            device
        ),
    }


def build(method, length, num_labels):
    if method.startswith("ernie"):
        from layoutlmv3_pretraining import ernie_layout

        _, model, _ = ernie_layout.load(
            num_labels, extrapolation=method == "ernie-extrap", use_sdpa=True
        )
        return model, "ernie"
    from transformers import LayoutLMv3ForTokenClassification

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", num_labels=num_labels
    )
    if method == "v3-poscopy" and length > 512:
        from layoutlmv3_pretraining.long_docs import extend_position_embeddings

        extend_position_embeddings(model.layoutlmv3, length)
    return model, "v3"


def measure(
    method, length, batch_size, precision, steps, device="cuda", sdpa=False, ckpt=False
):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from layoutlmv3_pretraining.long_docs import (
        checkpoint_encoder_layers,
        use_sdpa_attention,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, family = build(method, length, num_labels=7)
    if sdpa and family == "v3":
        use_sdpa_attention(model)
    if ckpt:  # transformers' LayoutLMv3ForTokenClassification does not support gradient_checkpointing_enable()
        checkpoint_encoder_layers(model)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    batch = batch_for(family, length, batch_size, 7, device)
    times = []
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        for _ in range(steps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=precision == "fp16"
            ):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt, batch
    return round(peak, 2), round(sum(times[1:]) / max(1, len(times) - 1), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["v3-window", "v3-poscopy", "ernie-window", "ernie-extrap"],
    )
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--precisions", nargs="+", default=["fp16", "fp32"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["base"],
        help="base, sdpa (LayoutLMv3 SDPA attention), ckpt (gradient checkpointing), sdpa+ckpt",
    )
    ap.add_argument("--cap_gb", type=float, default=15.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gb * 2**30 / total))
    rows = []
    for method in a.methods:
        lengths = [512] if method.endswith("window") else a.lengths
        for length in lengths:
            for variant in a.variants:
                if "sdpa" in variant and not method.startswith("v3"):
                    continue
                for precision in a.precisions:
                    for bs in a.batch_sizes:
                        row = {
                            "method": method,
                            "variant": variant,
                            "length": length,
                            "precision": precision,
                            "batch_size": bs,
                        }
                        try:
                            peak, sec = measure(
                                method,
                                length,
                                bs,
                                precision,
                                a.steps,
                                sdpa="sdpa" in variant,
                                ckpt="ckpt" in variant,
                            )
                            row.update(
                                peak_gib=peak, seconds_per_step=sec, fits_cap=True
                            )
                        except torch.cuda.OutOfMemoryError:
                            row.update(
                                peak_gib=None,
                                fits_cap=False,
                                status=f"OOM above {a.cap_gb} GB cap",
                            )
                        except Exception as e:  # noqa: BLE001  (record and continue with the other settings)
                            row.update(
                                status=f"error: {type(e).__name__}: {str(e)[:200]}"
                            )
                        torch.cuda.empty_cache()
                        print(json.dumps(row), flush=True)
                        rows.append(row)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                {
                    "cap_gb": a.cap_gb,
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "rows": rows,
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
