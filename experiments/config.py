"""Shared constants for the 12 debiasing experiments."""

import os

# --- Paths ---
BASE_DIR = "/path/to/data"
PROJECT_DIR = os.path.join(BASE_DIR, "stereoimage")
DATA_CSV = os.path.join(PROJECT_DIR, "data", "benchmark_prompts.csv")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "experiment_outputs")

# --- API ---
OPENROUTER_API_KEY = ""
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# --- Models ---
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
LLM_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Image")
MODEL_CACHE = os.path.join(BASE_DIR, "models")

# --- Experiment parameters ---
SEEDS = [0, 1, 2]
ALPHAS = [2.0]
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 4.0

# --- Experiment registry ---
EXPERIMENT_NAMES = {
    0: "baseline",
    1: "llm_rewrite_no_kg",
    2: "extracted_kg_llm_rewrite",
    3: "gt_kg_llm_rewrite",
    4: "extracted_kg_full_triple_sv",
    5: "extracted_kg_tail_sv",
    6: "gt_kg_full_triple_sv",
    7: "gt_kg_tail_sv",
    8: "extracted_kg_llm_pair_sv",
    9: "gt_kg_llm_pair_sv",
    10: "extracted_kg_gt_pair_sv",
    11: "gt_kg_gt_pair_sv",
    12: "gpt_image_2_baseline",
    13: "sd3_baseline",
    14: "nano_banana_2_baseline",
}

# Experiments that use generate_baseline (prompt rewriting)
BASELINE_EXPERIMENTS = {0, 1, 2, 3, 12, 13, 14}
# Experiments that use steering vectors
STEERING_EXPERIMENTS = {4, 5, 6, 7, 8, 9, 10, 11}
# Experiments that use tail-only steering (mean-pool + broadcast)
TAIL_STEERING_EXPERIMENTS = {5, 7}

# --- Cache file paths ---
EMBEDDINGS_DIR = os.path.join(CACHE_DIR, "embeddings")
LLM_OUTPUTS_DIR = os.path.join(CACHE_DIR, "llm_outputs")
EXTRACTED_KG_CSV = os.path.join(CACHE_DIR, "extracted_kg.csv")
EVAL_RESULTS_DIR = os.path.join(CACHE_DIR, "eval_results")


def exp_output_dir(exp_id):
    """Return the output directory for a given experiment ID."""
    return os.path.join(OUTPUT_DIR, f"exp_{exp_id:02d}_{EXPERIMENT_NAMES[exp_id]}")
