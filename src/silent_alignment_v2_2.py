"""
silent_alignment_experiment_v2_2.py

Silent Alignment Experiment — v2.2 (OpenAI, instrumented)
Goal: add instrumentation + plotting + caching + resume, WITHOUT changing core logic.

Core definitions preserved:
- silent = (semantic >= epsilon) AND (surface <= tau) AND (not trivial)
- permutation null test for p-value

Usage:
    python silent_alignment_experiment_v2_2.py
    python silent_alignment_experiment_v2_2.py --yes
    python silent_alignment_experiment_v2_2.py --tasks tasks.txt
    python silent_alignment_experiment_v2_2.py --out results_silent_alignment_v2_2

Requirements:
    pip install openai numpy tqdm
Optional:
    pip install matplotlib
    pip install scipy
"""

import os
import re
import csv
import json
import time
import math
import uuid
import hashlib
import random
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Optional plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not available. Plots will be skipped.")

# Optional SciPy for p-value confidence intervals
try:
    from scipy.stats import beta as _beta_dist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[INFO] scipy not available. p-value confidence intervals will be skipped.")

# Will hold the OpenAI client after init
client = None


@dataclass
class ExperimentConfig:
    model_a: str = "gpt-4.1-mini"
    model_b: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-large"
    temperature_a: float = 0.7
    temperature_b: float = 0.7
    max_tokens: int = 512

    epsilon: float = 0.80          # main semantic similarity threshold
    tau: float = 0.60              # surface similarity upper bound

    num_permutations: int = 300
    seed: int = 42

    output_dir: str = "results_silent_alignment_v2_2"
    tasks_file: str = "tasks.txt"

    system_prompt_a: str = "You are Observer A. Answer clearly, analytically, and concisely."
    system_prompt_b: str = "You are Observer B. Answer clearly, analytically, and concisely."

    # Instrumentation / robustness
    request_retries: int = 6
    request_backoff_base: float = 1.5
    request_backoff_jitter: float = 0.25
    cache_dirname: str = "_cache"
    save_permutation_counts: bool = True  # if False, store only summary stats


EPSILON_SWEEP_DEFAULT: List[float] = [0.70, 0.75, 0.80, 0.85, 0.90]


# -----------------------------
# Setup & reproducibility
# -----------------------------

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# -----------------------------
# OpenAI client + retry wrapper
# -----------------------------

def init_openai_client() -> None:
    global client
    print("Initialising OpenAI client...")

    try:
        from openai import OpenAI
    except ImportError:
        print("\n[ERROR] The 'openai' package is not installed.")
        print("Install it with:  pip install openai\n")
        raise SystemExit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nNo OPENAI_API_KEY found in environment.")
        api_key = input("Please paste your OpenAI API key (sk-...): ").strip()
        if not api_key:
            print("\n[ERROR] No API key provided. Exiting.")
            raise SystemExit(1)

    client = OpenAI(api_key=api_key)
    print("OpenAI client initialised.\n")


def with_retries(fn, cfg: ExperimentConfig, label: str):
    """
    Basic retry with exponential backoff + jitter.
    Keeps behavior the same when the request succeeds; only affects transient failures.
    """
    last_err = None
    for attempt in range(cfg.request_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            # Heuristic: treat all exceptions as retryable up to limit; this is simplest and robust.
            backoff = (cfg.request_backoff_base ** attempt)
            jitter = random.uniform(-cfg.request_backoff_jitter, cfg.request_backoff_jitter)
            sleep_s = max(0.0, backoff + jitter)
            print(f"[WARN] {label} failed (attempt {attempt+1}/{cfg.request_retries}): {e}")
            if attempt < cfg.request_retries - 1:
                print(f"       retrying in {sleep_s:.2f}s...")
                time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {cfg.request_retries} attempts: {last_err}")


def test_api_connection(cfg: ExperimentConfig) -> None:
    print("Running quick test call to OpenAI...")

    def _call():
        return client.chat.completions.create(
            model=cfg.model_a,
            messages=[
                {"role": "system", "content": "You are a health-check assistant."},
                {"role": "user", "content": "Reply with the single word OK."},
            ],
            max_tokens=3,
            temperature=0.0,
        )

    try:
        response = with_retries(_call, cfg, "health_check")
        text = response.choices[0].message.content.strip()
        print(f"Test response from {cfg.model_a}: {repr(text)}")
        print("API connection looks good.\n")
    except Exception as e:
        print("\n[ERROR] Test call failed.")
        print(f"Details: {e}")
        print("Check your API key, model name, or network connection.\n")
        raise SystemExit(1)


# -----------------------------
# Cache helpers
# -----------------------------

def cache_paths(cfg: ExperimentConfig) -> Tuple[str, str]:
    cache_dir = os.path.join(cfg.output_dir, cfg.cache_dirname)
    ensure_dir(cache_dir)
    return cache_dir, os.path.join(cache_dir, "cache_index.json")


def load_cache_index(cfg: ExperimentConfig) -> Dict[str, Any]:
    _, idx_path = cache_paths(cfg)
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"created_utc": utc_now_str(), "items": {}}


def save_cache_index(cfg: ExperimentConfig, index: Dict[str, Any]) -> None:
    _, idx_path = cache_paths(cfg)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def cache_get(cfg: ExperimentConfig, key: str) -> Optional[Any]:
    index = load_cache_index(cfg)
    item = index["items"].get(key)
    if not item:
        return None
    path = item.get("path")
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_put(cfg: ExperimentConfig, key: str, payload: Any) -> None:
    cache_dir, _ = cache_paths(cfg)
    index = load_cache_index(cfg)

    fname = f"{key}.json"
    path = os.path.join(cache_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    index["items"][key] = {"path": path, "saved_utc": utc_now_str()}
    save_cache_index(cfg, index)


# -----------------------------
# LLM + Embeddings (cached)
# -----------------------------

def call_model(
    cfg: ExperimentConfig,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    seed: int,
) -> str:
    if client is None:
        raise RuntimeError("OpenAI client is not initialised.")

    cache_key = sha256_text(stable_json_dumps({
        "kind": "chat",
        "model": model,
        "system": system_prompt,
        "user": user_prompt,
        "temperature": temperature,
        "max_tokens": cfg.max_tokens,
        "seed": seed,
    }))

    cached = cache_get(cfg, cache_key)
    if cached is not None:
        return cached["text"]

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=cfg.max_tokens,
            seed=seed,
        )

    resp = with_retries(_call, cfg, f"chat:{model}")
    text = resp.choices[0].message.content.strip()
    cache_put(cfg, cache_key, {"text": text})
    return text


def embed_text(cfg: ExperimentConfig, model: str, texts: List[str]) -> np.ndarray:
    if client is None:
        raise RuntimeError("OpenAI client is not initialised.")

    # Batch caching: cache each text independently so partial reuse works.
    vectors: List[np.ndarray] = []
    for t in texts:
        cache_key = sha256_text(stable_json_dumps({
            "kind": "embed",
            "model": model,
            "text": t,
        }))
        cached = cache_get(cfg, cache_key)
        if cached is not None:
            vectors.append(np.array(cached["vec"], dtype=np.float32))
            continue

        def _call():
            return client.embeddings.create(model=model, input=[t])

        resp = with_retries(_call, cfg, f"embed:{model}")
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        cache_put(cfg, cache_key, {"vec": vec.tolist()})
        vectors.append(vec)

    return np.stack(vectors, axis=0)


# -----------------------------
# Utilities (metrics)
# -----------------------------

def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def surface_similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def load_tasks(tasks_file: str) -> List[str]:
    if os.path.exists(tasks_file):
        print(f"Loading tasks from {tasks_file}...")
        tasks: List[str] = []
        with open(tasks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tasks.append(line)
        if tasks:
            print(f"Loaded {len(tasks)} tasks.\n")
            return tasks
        print(f"[WARN] {tasks_file} exists but is empty after filtering.\n")

    print("Using built-in demo tasks (N=5).")
    return [
        "Explain the difference between variance and standard deviation.",
        "Summarize the core idea of Bayesian inference in one paragraph.",
        "Describe the double-slit experiment in simple terms.",
        "Compare supervised and unsupervised learning.",
        "Explain what a Markov chain is.",
    ]


# -----------------------------
# Core pipeline
# -----------------------------

def run_observers(tasks: List[str], cfg: ExperimentConfig) -> Dict[str, Any]:
    results = []
    print("Running observers on tasks...")
    for idx, task in enumerate(tqdm(tasks, desc="Tasks")):
        seed_a = cfg.seed + idx * 2
        seed_b = cfg.seed + idx * 2 + 1

        y_a = call_model(cfg, cfg.model_a, cfg.system_prompt_a, task, cfg.temperature_a, seed_a)
        y_b = call_model(cfg, cfg.model_b, cfg.system_prompt_b, task, cfg.temperature_b, seed_b)

        results.append({
            "task_index": idx,
            "task": task,
            "y_a": y_a,
            "y_b": y_b,
            "seed_a": seed_a,
            "seed_b": seed_b,
        })

    return {"config": asdict(cfg), "results": results}


def attach_embeddings(data: Dict[str, Any], cfg: ExperimentConfig) -> Dict[str, Any]:
    texts_a = [r["y_a"] for r in data["results"]]
    texts_b = [r["y_b"] for r in data["results"]]

    print("Embedding Observer A outputs...")
    emb_a = embed_text(cfg, cfg.embedding_model, texts_a)
    print("Embedding Observer B outputs...")
    emb_b = embed_text(cfg, cfg.embedding_model, texts_b)

    for i, r in enumerate(data["results"]):
        r["emb_a"] = emb_a[i].tolist()
        r["emb_b"] = emb_b[i].tolist()
    return data


def compute_silent_overlaps(
    data: Dict[str, Any],
    tau: float,
    epsilon: float
) -> Tuple[int, List[float], List[float], List[int]]:
    S_obs = 0
    semantic_sims: List[float] = []
    surface_sims: List[float] = []
    silent_flags: List[int] = []

    for r in data["results"]:
        y_a = r["y_a"]
        y_b = r["y_b"]

        v_a = np.array(r["emb_a"], dtype=np.float32)
        v_b = np.array(r["emb_b"], dtype=np.float32)

        s = cosine_similarity(v_a, v_b)
        semantic_sims.append(s)

        surf = surface_similarity(normalize_text(y_a), normalize_text(y_b))
        surface_sims.append(surf)

        trivial = normalize_text(y_a) == normalize_text(y_b)

        silent = int((s >= epsilon) and (surf <= tau) and (not trivial))
        silent_flags.append(silent)
        S_obs += silent

    return S_obs, semantic_sims, surface_sims, silent_flags


def permutation_null_test(
    data: Dict[str, Any],
    tau: float,
    epsilon: float,
    num_permutations: int,
    seed: int
) -> List[int]:
    results = data["results"]
    n = len(results)

    emb_a = np.stack([np.array(r["emb_a"], dtype=np.float32) for r in results], axis=0)
    emb_b = np.stack([np.array(r["emb_b"], dtype=np.float32) for r in results], axis=0)
    texts_a = [r["y_a"] for r in results]
    texts_b = [r["y_b"] for r in results]

    indices = list(range(n))
    null_counts: List[int] = []
    rng = random.Random(seed)

    print(f"Running permutation null K={num_permutations} (epsilon={epsilon:.2f})...")
    for _ in tqdm(range(num_permutations), desc=f"Permutations ε={epsilon:.2f}"):
        rng.shuffle(indices)
        S_k = 0
        for i in range(n):
            j = indices[i]
            s = cosine_similarity(emb_a[i], emb_b[j])
            surf = surface_similarity(normalize_text(texts_a[i]), normalize_text(texts_b[j]))
            trivial = normalize_text(texts_a[i]) == normalize_text(texts_b[j])
            S_k += int((s >= epsilon) and (surf <= tau) and (not trivial))
        null_counts.append(S_k)

    return null_counts


def compute_p_value(S_obs: int, null_counts: List[int]) -> float:
    K = len(null_counts)
    ge = sum(1 for x in null_counts if x >= S_obs)
    return (1 + ge) / (1 + K)


def p_value_confidence_interval(S_obs: int, null_counts: List[int], alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    if not SCIPY_AVAILABLE:
        return None
    K = len(null_counts)
    n_ge = sum(1 for x in null_counts if x >= S_obs)
    a = n_ge + 1
    b = K - n_ge + 1
    lower, upper = _beta_dist.ppf([alpha / 2, 1 - alpha / 2], a, b)
    return float(lower), float(upper)


# -----------------------------
# Plotting
# -----------------------------

def plot_null_distribution(S_obs: int, null_counts: List[int], out_path: str, title: str):
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.figure()
    plt.hist(null_counts, bins="auto", alpha=0.7)
    plt.axvline(S_obs, linestyle="--", linewidth=2)
    plt.xlabel("Silent overlaps under null")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_scatter_semantic_vs_surface(semantic: List[float], surface: List[float], flags: List[int], out_path: str, title: str):
    if not MATPLOTLIB_AVAILABLE:
        return
    x = np.array(surface, dtype=float)
    y = np.array(semantic, dtype=float)
    f = np.array(flags, dtype=int)

    plt.figure()
    # two scatters so it works even without specifying colors explicitly (matplotlib chooses defaults)
    plt.scatter(x[f == 0], y[f == 0], alpha=0.6, label="not silent")
    plt.scatter(x[f == 1], y[f == 1], alpha=0.9, label="silent")
    plt.xlabel("Surface similarity (SequenceMatcher ratio)")
    plt.ylabel("Semantic similarity (cosine of embeddings)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def pca_3d(X: np.ndarray) -> np.ndarray:
    """
    Minimal PCA to 3D without external deps.
    X shape: (n, d)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:3].T
    return Z


def plot_3d_pca_pairs(emb_a: np.ndarray, emb_b: np.ndarray, out_path: str, title: str):
    if not MATPLOTLIB_AVAILABLE:
        return
    # Combine for shared PCA basis
    X = np.vstack([emb_a, emb_b])
    Z = pca_3d(X)
    Za = Z[: emb_a.shape[0]]
    Zb = Z[emb_a.shape[0] :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Za[:, 0], Za[:, 1], Za[:, 2], alpha=0.7, label="A")
    ax.scatter(Zb[:, 0], Zb[:, 1], Zb[:, 2], alpha=0.7, label="B")

    # draw pair lines (light)
    for i in range(Za.shape[0]):
        ax.plot([Za[i, 0], Zb[i, 0]], [Za[i, 1], Zb[i, 1]], [Za[i, 2], Zb[i, 2]], alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_sweep(sweep_rows: List[Dict[str, Any]], out_path: str, title: str):
    if not MATPLOTLIB_AVAILABLE:
        return
    eps = [r["epsilon"] for r in sweep_rows]
    rate = [r["S_obs"] / max(1, r["N"]) for r in sweep_rows]
    pval = [r["p_value"] for r in sweep_rows]

    plt.figure()
    plt.plot(eps, rate, marker="o", label="silent overlap rate")
    plt.xlabel("epsilon")
    plt.ylabel("rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_rate.png"), dpi=200)

    plt.figure()
    plt.plot(eps, pval, marker="o", label="permutation p-value")
    plt.xlabel("epsilon")
    plt.ylabel("p-value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_pvalue.png"), dpi=200)


# -----------------------------
# Saving helpers
# -----------------------------

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default=None, help="Path to tasks file (one per line)")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--yes", action="store_true", help="Skip the 'Press Enter' pause")
    parser.add_argument("--perms", type=int, default=None, help="Override num_permutations")
    parser.add_argument("--epsilon", type=float, default=None, help="Override main epsilon")
    parser.add_argument("--tau", type=float, default=None, help="Override tau")
    parser.add_argument("--sweep", type=str, default=None, help="Comma-separated epsilon sweep, e.g. 0.7,0.75,0.8")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.tasks:
        cfg.tasks_file = args.tasks
    if args.out:
        cfg.output_dir = args.out
    if args.perms is not None:
        cfg.num_permutations = int(args.perms)
    if args.epsilon is not None:
        cfg.epsilon = float(args.epsilon)
    if args.tau is not None:
        cfg.tau = float(args.tau)

    epsilon_sweep = EPSILON_SWEEP_DEFAULT
    if args.sweep:
        epsilon_sweep = [float(x.strip()) for x in args.sweep.split(",") if x.strip()]

    print("=== Silent Alignment Experiment (OpenAI) — v2.2 ===\n")
    set_global_seeds(cfg.seed)
    ensure_dir(cfg.output_dir)

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    manifest = {
        "run_id": run_id,
        "created_utc": utc_now_str(),
        "config": asdict(cfg),
        "epsilon_sweep": epsilon_sweep,
        "notes": "v2.2 adds caching, retries, richer plots, csv exports. Core logic unchanged.",
    }
    write_json(os.path.join(cfg.output_dir, "manifest.json"), manifest)

    init_openai_client()
    test_api_connection(cfg)

    tasks = load_tasks(cfg.tasks_file)
    N = len(tasks)
    print(f"Number of tasks: N = {N}\n")
    if not args.yes:
        input("Press Enter to run the full experiment with these tasks...")

    # Run + embed
    data = run_observers(tasks, cfg)
    data = attach_embeddings(data, cfg)

    # Save raw immediately (resume-friendly)
    raw_path = os.path.join(cfg.output_dir, "raw_results_v2_2.json")
    write_json(raw_path, data)
    print(f"\nSaved raw results to {raw_path}")

    # Prepare numpy for 3D plot
    emb_a_np = np.stack([np.array(r["emb_a"], dtype=np.float32) for r in data["results"]], axis=0)
    emb_b_np = np.stack([np.array(r["emb_b"], dtype=np.float32) for r in data["results"]], axis=0)

    # Main epsilon
    print(f"\n=== Main analysis ε={cfg.epsilon:.2f}, τ={cfg.tau:.2f} ===\n")
    S_obs, sem, surf, flags = compute_silent_overlaps(data, cfg.tau, cfg.epsilon)

    null_counts = permutation_null_test(
        data,
        tau=cfg.tau,
        epsilon=cfg.epsilon,
        num_permutations=cfg.num_permutations,
        seed=cfg.seed + 999,
    )
    p_val = compute_p_value(S_obs, null_counts)
    p_ci = p_value_confidence_interval(S_obs, null_counts)

    # Per-task CSV
    per_task_rows = []
    for i, r in enumerate(data["results"]):
        per_task_rows.append({
            "task_index": r["task_index"],
            "seed_a": r["seed_a"],
            "seed_b": r["seed_b"],
            "semantic": float(sem[i]),
            "surface": float(surf[i]),
            "silent": int(flags[i]),
            "task": r["task"],
        })
    per_task_csv = os.path.join(cfg.output_dir, "per_task_metrics.csv")
    write_csv(per_task_csv, per_task_rows, ["task_index", "seed_a", "seed_b", "semantic", "surface", "silent", "task"])
    print(f"Saved per-task metrics CSV to {per_task_csv}")

    # Plots (main)
    if MATPLOTLIB_AVAILABLE:
        plot_null_distribution(
            S_obs, null_counts,
            out_path=os.path.join(cfg.output_dir, f"null_distribution_eps_{cfg.epsilon:.2f}.png"),
            title=f"Permutation Null vs Observed Silent Overlaps (ε={cfg.epsilon:.2f}, τ={cfg.tau:.2f})"
        )
        plot_scatter_semantic_vs_surface(
            sem, surf, flags,
            out_path=os.path.join(cfg.output_dir, f"scatter_semantic_vs_surface_eps_{cfg.epsilon:.2f}.png"),
            title=f"Semantic vs Surface (ε={cfg.epsilon:.2f}, τ={cfg.tau:.2f})"
        )
        plot_3d_pca_pairs(
            emb_a_np, emb_b_np,
            out_path=os.path.join(cfg.output_dir, f"pca3d_pairs_eps_{cfg.epsilon:.2f}.png"),
            title=f"3D PCA of embeddings (A/B pairs) (ε={cfg.epsilon:.2f})"
        )

    # Sweep
    print("\n=== Epsilon sweep (fixed τ={:.2f}) ===".format(cfg.tau))
    sweep_rows: List[Dict[str, Any]] = []
    for eps in epsilon_sweep:
        S_e, _, _, _ = compute_silent_overlaps(data, cfg.tau, eps)
        null_e = permutation_null_test(
            data,
            tau=cfg.tau,
            epsilon=eps,
            num_permutations=cfg.num_permutations,
            seed=cfg.seed + int(eps * 1000),
        )
        p_e = compute_p_value(S_e, null_e)
        ci_e = p_value_confidence_interval(S_e, null_e)
        row = {
            "epsilon": float(eps),
            "N": int(N),
            "S_obs": int(S_e),
            "rate": float(S_e / max(1, N)),
            "p_value": float(p_e),
            "p_ci_low": (float(ci_e[0]) if ci_e else None),
            "p_ci_high": (float(ci_e[1]) if ci_e else None),
        }
        if cfg.save_permutation_counts:
            row["null_counts"] = null_e
        sweep_rows.append(row)

    sweep_json = os.path.join(cfg.output_dir, "epsilon_sweep.json")
    write_json(sweep_json, sweep_rows)
    print(f"Saved epsilon sweep JSON to {sweep_json}")

    sweep_csv = os.path.join(cfg.output_dir, "epsilon_sweep.csv")
    write_csv(
        sweep_csv,
        sweep_rows,
        ["epsilon", "N", "S_obs", "rate", "p_value", "p_ci_low", "p_ci_high"]
    )
    print(f"Saved epsilon sweep CSV to {sweep_csv}")

    if MATPLOTLIB_AVAILABLE:
        plot_sweep(
            sweep_rows,
            out_path=os.path.join(cfg.output_dir, "epsilon_sweep_plots.png"),
            title=f"Epsilon sweep (τ={cfg.tau:.2f})"
        )

    # Summary JSON
    summary = {
        "run_id": run_id,
        "created_utc": utc_now_str(),
        "N": N,
        "main": {
            "epsilon": cfg.epsilon,
            "tau": cfg.tau,
            "S_obs": S_obs,
            "rate": float(S_obs / max(1, N)),
            "mean_semantic": float(np.mean(sem)) if sem else 0.0,
            "mean_surface": float(np.mean(surf)) if surf else 0.0,
            "p_value": float(p_val),
            "p_ci": p_ci,
            "num_permutations": cfg.num_permutations,
        },
    }
    summary_path = os.path.join(cfg.output_dir, "summary_v2_2.json")
    write_json(summary_path, summary)
    print(f"\nSaved summary to {summary_path}\n")

    # Console summary
    print("=== MAIN RUN SUMMARY ===")
    print(f"Tasks (N)                 : {N}")
    print(f"Silent overlaps (S_obs)   : {S_obs}")
    print(f"Silent overlap rate       : {S_obs / max(1, N):.3f}")
    print(f"Epsilon (semantic)        : {cfg.epsilon:.2f}")
    print(f"Tau (surface)             : {cfg.tau:.2f}")
    print(f"Mean semantic similarity  : {summary['main']['mean_semantic']:.3f}")
    print(f"Mean surface similarity   : {summary['main']['mean_surface']:.3f}")
    print(f"Permutation p-value       : {p_val:.6f}")
    if p_ci is not None:
        print(f"p-value 95% CI            : [{p_ci[0]:.6f}, {p_ci[1]:.6f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
