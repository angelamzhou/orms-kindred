"""Learn ranking weights from the user's own reviewer ratings, locally.

Feedback rows (~/.ormatch/feedback.jsonl): one per rated candidate, with the unweighted
features the matcher computed for it (max_sim, topk_mean, cited, volume, editor_roles,
seniority, early_career, seed) and a label +1 (good fit) / -1 (poor fit). Unrated candidates
that were shown alongside a rated one are stored with label 0 and used as weak negatives.

Fitting: pairwise logistic regression on feature differences (good minus bad) with an L2
penalty that pulls the weights toward the user's current slider settings, so a handful of
ratings moves the weights a little and many ratings move them a lot. The learned weight
vector is mapped back onto the slider parameters and saved to ~/.ormatch/weights.json.
Nothing here leaves the machine.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

FEEDBACK = Path.home() / ".ormatch" / "feedback.jsonl"
WEIGHTS = Path.home() / ".ormatch" / "weights.json"
FEATURES = ["max_sim", "topk_mean", "cited", "volume", "editor_roles", "seniority", "early_career", "seed"]
# slider name -> (feature, sign): score += sign * slider * feature
SLIDER_MAP = {"cite_weight": ("cited", +1), "volume_correction": ("volume", -1), "editor_weight": ("editor_roles", -1),
              "seniority_weight": ("seniority", -1), "early_career_weight": ("early_career", +1), "seed_weight": ("seed", +1)}


def manuscript_key(title: str, abstract: str) -> str:
    return hashlib.sha1((title or "").strip().lower().encode() + b"|" + (abstract or "").strip().lower()[:500].encode()).hexdigest()[:16]


def record(mkey: str, author_id: str, author_name: str, label: int, features: Dict[str, float], path: Path = FEEDBACK) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load(path)
    rows = [r for r in rows if not (r["manuscript"] == mkey and r["author_id"] == author_id)]  # latest rating wins
    rows.append({"manuscript": mkey, "author_id": author_id, "author_name": author_name, "label": int(label),
                 "features": {k: float(features.get(k, 0.0)) for k in FEATURES}, "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load(path: Path = FEEDBACK) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def params_to_weights(params: Dict[str, float]) -> np.ndarray:
    """Slider settings -> weight vector over FEATURES (the linear objective the matcher uses)."""
    lam = float(params.get("lam", 0.3))
    w = {"max_sim": 1 - lam, "topk_mean": lam}
    for slider, (feat, sign) in SLIDER_MAP.items():
        w[feat] = sign * float(params.get(slider, 0.0) or 0.0)
    return np.array([w[f] for f in FEATURES], dtype=float)


def weights_to_params(w: np.ndarray, base: Dict[str, float]) -> Dict[str, float]:
    """Weight vector -> slider settings. The text part is renormalised to sum to 1 (the sliders
    are in units of cosine similarity); penalties that came out with the 'wrong' sign clip to 0."""
    d = dict(zip(FEATURES, w))
    text = max(d["max_sim"] + d["topk_mean"], 1e-6)
    out = dict(base)
    out["lam"] = float(np.clip(d["topk_mean"] / text, 0, 1))
    for slider, (feat, sign) in SLIDER_MAP.items():
        out[slider] = float(max(0.0, sign * d[feat] / text))
    return out


def fit(rows: Iterable[dict], base_params: Dict[str, float], l2: float = 1.0, weak: float = 0.3,
        iters: int = 500, lr: float = 0.5) -> Dict:
    """Pairwise logistic regression toward base_params. Returns dict(weights, params, n_pairs, ...)."""
    rows = list(rows)
    by_m: Dict[str, List[dict]] = {}
    for r in rows:
        by_m.setdefault(r["manuscript"], []).append(r)
    X, wts = [], []
    for grp in by_m.values():
        pos = [r for r in grp if r["label"] > 0]
        neg = [r for r in grp if r["label"] < 0]
        unr = [r for r in grp if r["label"] == 0]
        for p in pos:
            fp = np.array([p["features"].get(k, 0.0) for k in FEATURES])
            for n in neg:
                X.append(fp - np.array([n["features"].get(k, 0.0) for k in FEATURES])); wts.append(1.0)
            for n in unr:
                X.append(fp - np.array([n["features"].get(k, 0.0) for k in FEATURES])); wts.append(weak)
        for n in neg:  # a bad fit should also rank below the unrated ones shown
            fn = np.array([n["features"].get(k, 0.0) for k in FEATURES])
            for u in unr:
                X.append(np.array([u["features"].get(k, 0.0) for k in FEATURES]) - fn); wts.append(weak)
    w0 = params_to_weights(base_params)
    if not X:
        return {"weights": w0, "params": dict(base_params), "n_pairs": 0, "n_rated": sum(r["label"] != 0 for r in rows),
                "n_manuscripts": len(by_m), "train_acc": None}
    X = np.array(X); wts = np.array(wts)
    # features live on very different scales; standardise the differences, fit, map back
    scale = X.std(axis=0) + 1e-6
    Xs = X / scale
    w = w0 * scale  # start at the current settings expressed in standardised units
    w0s = w.copy()
    for _ in range(iters):
        z = Xs @ w
        pz = 1 / (1 + np.exp(-z))
        grad = -(Xs * (wts * (1 - pz))[:, None]).sum(axis=0) / wts.sum() + l2 * (w - w0s) / len(X)
        w -= lr * grad
    w_final = w / scale
    acc = float((wts * ((X @ w_final) > 0)).sum() / wts.sum())
    return {"weights": w_final, "params": weights_to_params(w_final, base_params), "n_pairs": int(len(X)),
            "n_rated": int(sum(r["label"] != 0 for r in rows)), "n_manuscripts": len(by_m), "train_acc": acc,
            "base_acc": float((wts * ((X @ w0) > 0)).sum() / wts.sum())}


def save_params(params: Dict[str, float], path: Path = WEIGHTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()}, indent=1))


def load_params(path: Path = WEIGHTS) -> Optional[Dict[str, float]]:
    return json.loads(path.read_text()) if path.exists() else None
