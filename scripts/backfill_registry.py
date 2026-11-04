"""Add registry entries for runs that finished before experiment tracking existed (idempotent).

Every entry is marked ``"backfilled": true``. Fields are filled only from what was recorded at the time
(results JSON, run_info.json args, launch scripts kept in the repo or in runs/, file modification times,
and launch times noted during the session); anything else is left null. Entries whose result files do
not exist yet are skipped, so the script can be re-run as runs finish.

    python scripts/backfill_registry.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from layoutlmv3_pretraining.tracking import append_record, read_registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# runs launched in the warm-up session used this environment (same .venv, A100-SXM4-80GB)
SESSION_ENV = {"python": "3.11.7", "torch": "2.6.0+cu124", "cuda": "12.4", "transformers": "4.57.6",
               "hostname": "hgx2.novalocal"}
GPU = "NVIDIA A100-SXM4-80GB"


def mtime(path) -> str | None:
    p = ROOT / path
    return dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds") if p.exists() else None


def load(path):
    return json.loads((ROOT / path).read_text())


def ft_entry(rid, path, *, command=None, command_source=None, git=None, env=None, start=None, tags=(), notes=None):
    r = load(path)
    if "task" not in r:  # FUNSD-only script of the sanity run: hyperparameters are not in the file, but the
        # README documents them and they were the script's defaults at the time (6aa8569)
        r.update(task="funsd", steps=1000, lr=1e-5, batch_size=2)
        notes = (notes or "") + "steps/lr/batch_size from the README (script defaults at 6aa8569), not the file."
    return {
        "run_id": rid, "kind": f"finetune_{r['task']}", "status": "completed", "backfilled": True,
        "tags": ["backfilled", *tags], "start_utc": start, "end_utc": mtime(path),
        "git": git or {"commit": None, "dirty": None, "diff_sha256": None},
        "command": command, "command_source": command_source, "argv": None, "cwd": str(ROOT) if command else None,
        "args": {"task": r["task"], "model": r["model"], "seeds": r["seeds"], "steps": r["steps"], "lr": r["lr"],
                 "batch_size": r["batch_size"], "out": path},
        "config": {"task": r["task"], "model": r["model"], "seeds": r["seeds"], "steps": r["steps"], "lr": r["lr"],
                   "batch_size": r["batch_size"],
                   "dataset": "nielsr/funsd-layoutlmv3" if r["task"] == "funsd" else "naver-clova-ix/cord-v2"},
        "environment": env, "outputs": {"results": path},
        "metrics": {"f1": r["f1"], "mean": r["mean"], "std": r["std"], "minutes": r.get("minutes"),
                    "per_seed": dict(zip(map(str, r["seeds"]), r["f1"]))},
        "error": None, "notes": notes}


def pt_entry(rid, log_path, info_path=None, *, command=None, command_source=None, git=None, env=None, start=None,
             tags=(), notes=None, outputs=None, end_path=None):
    log = load(log_path)
    info = load(info_path) if info_path else {}
    args = info.get("args")
    warm = (args or {}).get("head_warmup_steps", 0)
    joint = [r for r in log[1:] if r.get("phase", "joint") == "joint"]
    after = [r for r in log if warm and r["step"] == warm and r.get("phase") == "head_warmup"]
    config = None
    if args:
        config = {"model": args["model"], "init": "random" if args.get("random_init") else "pretrained",
                  "wpa_head": args.get("wpa_head", "linear"), "image_tokenizer": args["image_tokenizer"],
                  "image_tokenizer_path": args.get("dalle_encoder") if args["image_tokenizer"] == "dalle" else None,
                  "image_vocab_size": info.get("image_vocab_size"), "corpus": args["corpus"],
                  "train_shards": args.get("idl_shards") if args["corpus"] == "idl" else None,
                  "val_shards": args.get("idl_val_shards") if args["corpus"] == "idl" else None,
                  "seed": args["seed"], "head_warmup_steps": warm, "head_lr": args.get("head_lr") if warm else None,
                  "max_steps": args["max_steps"], "lr": args["lr"], "layer_decay": args.get("layer_decay", 1.0),
                  "effective_batch": args["batch_size"] * args["grad_accum"],
                  "corpus_info": {k: v for k, v in info.items() if k not in ("args", "loading_info")},
                  "loading_info": info.get("loading_info")}
    return {
        "run_id": rid, "kind": "pretrain", "status": "completed", "backfilled": True,
        "tags": ["backfilled", *tags, *(["probe"] if warm and args and args["max_steps"] == 0 else [])],
        "start_utc": start, "end_utc": mtime(end_path or log_path),
        "git": git or {"commit": None, "dirty": None, "diff_sha256": None},
        "command": command, "command_source": command_source, "argv": None, "cwd": str(ROOT) if command else None,
        "args": args, "config": config, "environment": env,
        "outputs": {"log": log_path, **({"run_info": info_path} if info_path else {}), **(outputs or {})},
        "metrics": {"step0": log[0], "after_head_warmup": after[0] if after else None, "final": log[-1],
                    "best_joint": min(joint, key=lambda r: r["loss"]) if joint else None},
        "error": None, "notes": notes}


def entries():
    """Zero-argument builders, one per run (a builder raises FileNotFoundError if its results are missing)."""
    e = []
    # ---------------------------------------------------------------- FUNSD-only sanity run (k-means)
    src = "scripts/run_experiment.sh (GPU unknown)"
    e.append(partial(pt_entry, "backfill-funsd_pt-pretrain", "results/pretrain_log.json", tags=["funsd_sanity"],
                      command="python scripts/pretrain.py --output_dir runs/funsd_pt --max_steps 3000 --eval_every 250",
                      command_source=src,
                      notes="FUNSD train, k-means 512 targets, linear WPA head; last step used (README). "
                            "Args other than those on the command line were script defaults at the time; not recorded."))
    e.append(partial(ft_entry, "backfill-funsd_base_3seeds", "results/funsd_base.json", tags=["funsd_sanity"],
                      command="python scripts/finetune_funsd.py --model microsoft/layoutlmv3-base --seeds 1 2 3 "
                              "--out results/funsd_base.json", command_source=src))
    e.append(partial(ft_entry, "backfill-funsd_continued_3seeds", "results/funsd_continued.json", tags=["funsd_sanity"],
                      command="python scripts/finetune_funsd.py --model runs/funsd_pt/backbone --seeds 1 2 3 "
                              "--out results/funsd_continued.json", command_source=src))
    # ---------------------------------------------------------------- IDL continued pre-training (no warm-up)
    for size in ("base", "large"):
        e.append(partial(pt_entry, f"backfill-idl_dalle_{size}-pretrain", f"results/idl/pretrain_{size}_log.json",
                          f"results/idl/pretrain_{size}_run_info.json", tags=["idl", size, "no_warmup"],
                          outputs={"backbone": f"runs/idl_dalle_{size}/backbone"},
                          notes="Exact command not recorded; args are the parsed args from run_info.json "
                                "(output_dir was runs/idl_dalle, renamed afterwards)."))
        for task in ("funsd", "cord"):
            e.append(partial(ft_entry, f"backfill-idl-{task}_{size}_original", f"results/idl/{task}_{size}_original.json",
                              tags=["idl", size, "original"]))
            chain = "runs/chain_base.sh" if size == "base" else "runs/chain_large2.sh (or chain_large.sh)"
            lr = " --lr 2e-5" if (task == "cord" and size == "large") else ""
            e.append(partial(ft_entry, f"backfill-idl-{task}_{size}_continued", f"results/idl/{task}_{size}_continued.json",
                              tags=["idl", size, "continued"],
                              command=f"python -u scripts/finetune_funsd.py --task {task}{lr} --model "
                                      f"runs/idl_dalle_{size}/backbone --seeds 1 2 3 4 5 --out "
                                      f"results/idl/{task}_{size}_continued.json", command_source=chain))
    # ---------------------------------------------------------------- head warm-up session (this commit series)
    code = {"commit": None, "dirty": True, "diff_sha256": None,
            "note": "launched from an uncommitted working tree whose pretrain.py/modeling.py were committed "
                    "unchanged as c9d8774"}
    ft_code = {"commit": None, "dirty": True, "diff_sha256": None,
               "note": "finetune_funsd.py was unchanged from d3a97e3 at launch"}
    for init, gpu, start in (("pretrained", "3", "2026-11-04T21:33:41+00:00"), ("scratch", "4", "2026-11-04T21:37:20+00:00")):
        lr = "1e-4" if init == "scratch" else "2e-5"
        decay = "1.0" if init == "scratch" else "0.9"
        e.append(partial(pt_entry, f"backfill-idl_warmup_base_{init}-pretrain",
                          f"results/idl_warmup/pretrain_base_{init}_log.json",
                          f"results/idl_warmup/pretrain_base_{init}_run_info.json", tags=["idl_warmup", "base", init],
                          git=code, env={**SESSION_ENV, "cuda_visible_devices": gpu, "gpu": [GPU]}, start=start,
                          command=f"python scripts/pretrain.py --model microsoft/layoutlmv3-base"
                                  f"{' --random_init' if init == 'scratch' else ''} --corpus idl --idl_shards 0-26 "
                                  f"--idl_val_shards 2999 --val_docs 512 --pages_per_doc 2 --exclude_funsd_test "
                                  f"--image_tokenizer dalle --dalle_encoder runs/dalle/encoder.pkl --head_warmup_steps 1000 "
                                  f"--head_lr 1e-3 --max_steps 6000 --lr {lr} --layer_decay {decay} --batch_size 16 "
                                  f"--grad_accum 2 --num_workers 12 --eval_every 250 --patience 8 --save_last "
                                  f"--output_dir runs/idl_warmup_base_{init}",
                          command_source=f"scripts/run_idl_warmup_experiment.sh (GPU={gpu} SIZE=base INIT={init})",
                          outputs={"backbone": f"runs/idl_warmup_base_{init}/backbone",
                                   "warmup_backbone": f"runs/idl_warmup_base_{init}/warmup_backbone"},
                          end_path=f"runs/idl_warmup_base_{init}/log.json"))
    e.append(partial(ft_entry, "backfill-idl_warmup-funsd_base_scratch", "results/idl_warmup/funsd_base_scratch.json",
                      tags=["idl_warmup", "base", "scratch"], git=ft_code,
                      env={**SESSION_ENV, "cuda_visible_devices": "4", "gpu": [GPU]},
                      command="python scripts/finetune_funsd.py --task funsd --model runs/idl_warmup_base_scratch/backbone "
                              "--seeds 1 2 --out results/idl_warmup/funsd_base_scratch.json",
                      command_source="scripts/run_idl_warmup_experiment.sh"))
    e.append(partial(ft_entry, "backfill-idl_warmup-funsd_base_warmup_only_seed1",
                      "results/idl_warmup/funsd_base_warmup_only_seed1.json", tags=["idl_warmup", "base", "warmup_only"],
                      git=ft_code, env={**SESSION_ENV, "cuda_visible_devices": "3", "gpu": [GPU]},
                      command="python scripts/finetune_funsd.py --task funsd --model "
                              "runs/idl_warmup_base_pretrained/warmup_backbone --seeds 1 "
                              "--out results/idl_warmup/funsd_base_warmup_only_seed1.json",
                      command_source="run by hand in the session"))
    for task, lr in (("funsd", ""), ("cord", " --lr 5e-5")):
        e.append(partial(ft_entry, f"backfill-idl_warmup-{task}_base_warmup_continued",
                          f"results/idl_warmup/{task}_base_warmup_continued.json",
                          tags=["idl_warmup", "base", "warmup_continued"], git=ft_code,
                          env={**SESSION_ENV, "cuda_visible_devices": "3", "gpu": [GPU]},
                          command=f"python scripts/finetune_funsd.py --task {task}{lr} --model "
                                  f"runs/idl_warmup_base_pretrained/backbone --seeds 1 2 3 4 5 --out "
                                  f"results/idl_warmup/{task}_base_warmup_continued.json",
                          command_source="scripts/run_idl_warmup_experiment.sh"))
    # ---------------------------------------------------------------- phase-2 runs started before tracking
    pre = {"commit": None, "dirty": True, "diff_sha256": None,
           "note": "code as in 7287ff5 except that pretrain.py/finetune_dvae.py had no tracking wrapper yet"}
    for init in ("pretrained", "random"):
        name = f"base_{init}_wpa-linear_mim-dalle"
        e.append(partial(pt_entry, f"backfill-probe-{name}", f"results/probes/{name}_log.json",
                          f"results/probes/{name}_run_info.json", tags=["probes", "base", init, "wpa-linear", "mim-dalle"],
                          git=pre, env={**SESSION_ENV, "cuda_visible_devices": "4", "gpu": [GPU]},
                          command=f"python scripts/pretrain.py --model microsoft/layoutlmv3-base"
                                  f"{' --random_init' if init == 'random' else ''} --corpus idl --idl_shards 0-26 "
                                  "--idl_val_shards 2999 --val_docs 512 --pages_per_doc 2 --exclude_funsd_test "
                                  "--image_tokenizer dalle --dalle_encoder runs/dalle/encoder.pkl --wpa_head linear "
                                  "--head_warmup_steps 1000 --head_lr 1e-3 --max_steps 0 --batch_size 16 --grad_accum 2 "
                                  f"--num_workers 12 --eval_every 250 --output_dir runs/probes/{name}",
                          command_source="scripts/run_probes.sh (GPU=4)"))
    if (ROOT / "runs/dvae_doc/results.json").exists():
        r = load("runs/dvae_doc/results.json")
        rec = {"run_id": "backfill-dvae_doc", "kind": "finetune_dvae", "status": "completed", "backfilled": True,
                  "tags": ["backfilled", "dvae"], "start_utc": "2026-11-04T23:00:46+00:00",
                  "end_utc": mtime("runs/dvae_doc/results.json"), "git": pre,
                  "command": "python scripts/finetune_dvae.py --output_dir runs/dvae_doc --steps 6000 --eval_every 1000 "
                             "--val_pages 256 --exclude_funsd_test --kl_weight 0",
                  "command_source": "run by hand in the session (GPU=3)", "argv": None, "cwd": str(ROOT),
                  "args": r["args"], "config": {"encoder": r["args"]["encoder"], "decoder": r["args"]["decoder"],
                                                "train_shards": r["args"]["idl_shards"], "val_shards": r["args"]["val_shards"],
                                                "val_pages": r["val_pages"], "view_size": 112},
                  "environment": {**SESSION_ENV, "cuda_visible_devices": "3", "gpu": [GPU]},
                  "outputs": {"encoder": "runs/dvae_doc/encoder.pt", "decoder": "runs/dvae_doc/decoder.pt",
                              "results": "runs/dvae_doc/results.json"},
                  "metrics": {"before": r["before"], "after": r["after"], "minutes": r["minutes"]},
                  "error": None, "notes": None}
        e.append(lambda rec=rec: rec)
    return e


def main():
    have = {r["run_id"] for r in read_registry()}
    added = skipped = 0
    for build in entries():
        try:
            rec = build()
        except FileNotFoundError:
            skipped += 1                                   # results not there yet
            continue
        if rec["run_id"] not in have:
            append_record(rec); added += 1
    print(f"added {added} backfilled entries ({skipped} skipped: results not there yet)")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
