"""
Download and load MacrOData datasets (OddBench, OvRBench).

Each dataset is a single .npz file on HuggingFace:
    https://huggingface.co/datasets/MacrOData-CMU/<bench>/resolve/main/<subset>/<name>.npz

Inside the file:
    train        (n_train, d)  normal points only
    train_labels (n_train,)    all 0
    test         (n_test, d)   normal points + anomalies
    test_labels  (n_test,)     0 = normal, 1 = anomaly
    + metadata (feature_names, anomaly_fraction, tags, origin, ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

BENCHES = ("OddBench", "OvRBench")
SUBSETS = ("representative", "public")


@dataclass
class Dataset:
    name: str
    bench: str
    X_train: np.ndarray      # normal points only
    X_test: np.ndarray
    y_test: np.ndarray       # 0 normal, 1 anomaly
    meta: dict

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def anomaly_fraction(self) -> float:
        return float(self.y_test.mean())


def list_remote(bench: str, subset: str = "representative") -> list[str]:
    """Names of the datasets available on HuggingFace for this bench/subset."""
    api = HfApi()
    files = api.list_repo_files(f"MacrOData-CMU/{bench}", repo_type="dataset")
    prefix = f"{subset}/"
    return sorted(f[len(prefix):-4] for f in files if f.startswith(prefix) and f.endswith(".npz"))


def download(bench: str, subset: str = "representative", names: list[str] | None = None) -> list[Path]:
    """Download the .npz files into data/<bench>/<subset>/. Skips files already there."""
    names = names or list_remote(bench, subset)
    out_dir = DATA_DIR / bench / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        target = out_dir / f"{name}.npz"
        if not target.exists():
            got = hf_hub_download(
                repo_id=f"MacrOData-CMU/{bench}",
                repo_type="dataset",
                filename=f"{subset}/{name}.npz",
                local_dir=DATA_DIR / bench,
            )
            # hf_hub_download already writes to local_dir/<subset>/<name>.npz
            assert Path(got).resolve() == target.resolve(), (got, target)
        paths.append(target)
    return paths


def load(path: Path | str) -> Dataset:
    """Load one .npz file into a Dataset object."""
    path = Path(path)
    raw = np.load(path, allow_pickle=True)
    meta = {}
    for key in raw.files:
        if key in ("train", "test", "train_labels", "test_labels", "train_groups", "test_groups"):
            continue
        value = raw[key]
        meta[key] = value.tolist() if value.ndim > 0 else value.item()
    bench = path.parents[1].name if path.parents[1].name in BENCHES else "unknown"
    return Dataset(
        name=path.stem,
        bench=bench,
        X_train=raw["train"].astype(np.float64),
        X_test=raw["test"].astype(np.float64),
        y_test=raw["test_labels"].astype(int),
        meta=meta,
    )


def load_all(bench: str, subset: str = "representative") -> list[Dataset]:
    """Load every dataset already downloaded for this bench/subset."""
    folder = DATA_DIR / bench / subset
    return [load(p) for p in sorted(folder.glob("*.npz"))]


def summary_table(datasets: list[Dataset]) -> "list[dict]":
    """Small per-dataset summary, handy for the report."""
    rows = []
    for d in datasets:
        rows.append({
            "bench": d.bench,
            "dataset": d.name,
            "n_train": len(d.X_train),
            "n_test": len(d.X_test),
            "n_features": d.n_features,
            "anomaly_fraction": round(d.anomaly_fraction, 4),
            "rf_auc": d.meta.get("rf_auc"),
            "ifor_auc": d.meta.get("ifor_auc"),
            "origin": d.meta.get("origin"),
        })
    return rows


if __name__ == "__main__":
    # `python src/data.py` downloads the two representative subsets and prints a summary
    for bench in BENCHES:
        paths = download(bench, "representative")
        print(f"{bench}: {len(paths)} files in {DATA_DIR / bench / 'representative'}")
    rows = summary_table(load_all("OddBench") + load_all("OvRBench"))
    print(json.dumps(rows[:3], indent=2))
    print(f"... {len(rows)} datasets total")
