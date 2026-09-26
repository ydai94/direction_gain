"""Local config for the layer-wise probing pilot.

Overrides the placeholder paths in experiments/config.py with the real
locations on the project filesystem.
"""

import os

PROJECT_DIR = os.environ.get("DG_ROOT", ".")
MODELS_DIR = os.environ.get("MODEL_ROOT", "models") + "/"

DATA_CSV = os.path.join(PROJECT_DIR, "data", "benchmark_prompts.csv")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache", "layer_probing")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots", "layer_probing")

QWEN_IMAGE_DIR = os.path.join(MODELS_DIR, "Qwen-Image")
TEXT_ENCODER_DIR = os.path.join(QWEN_IMAGE_DIR, "text_encoder")
TOKENIZER_DIR = os.path.join(QWEN_IMAGE_DIR, "tokenizer")

NUM_HIDDEN_LAYERS = 28
NUM_HIDDEN_STATES = NUM_HIDDEN_LAYERS + 1  # 29: embedding + 28 transformer blocks
HIDDEN_SIZE = 3584

SAMPLE_SEED = 42
BOOTSTRAP_SEED = 42
SHUFFLE_SEEDS = [42, 43, 44, 45, 46]
PER_CATEGORY_CAP = 50

POOLINGS = ("mean", "last")

REPS_PATH = {
    "mean": os.path.join(CACHE_DIR, "reps_mean.h5"),
    "last": os.path.join(CACHE_DIR, "reps_last.h5"),
}
PROMPT_INDEX_PATH = os.path.join(CACHE_DIR, "prompt_index.parquet")
METADATA_PATH = os.path.join(CACHE_DIR, "metadata.json")
LEAN_RESULTS_PATH = os.path.join(CACHE_DIR, "lean_results.parquet")
PROBE_RESULTS_PATH = os.path.join(CACHE_DIR, "probe_results.parquet")
SHUFFLED_RESULTS_PATH = os.path.join(CACHE_DIR, "shuffled_results.parquet")

VARIANT_COLUMNS = {
    "neutral": "prompt_neutral",
    "stereo": "prompt_stereotype",
    "anti": "prompt_anti_stereotype",
}
VARIANTS = ("neutral", "stereo", "anti")

MAX_TOKEN_LENGTH = 512
DEFAULT_BATCH_SIZE = 8

# Mirrors diffusers QwenImagePipeline.__init__ (verified 2026-05).
# Every prompt is wrapped in this template, and the first
# PROMPT_TEMPLATE_DROP_IDX tokens are dropped before being passed to the DiT.
# We mirror this so our probed representations match what the T2I model sees.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_DROP_IDX = 34
TOKENIZER_MAX_LENGTH = 1024  # matches QwenImagePipeline.tokenizer_max_length
