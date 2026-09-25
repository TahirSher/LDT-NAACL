"""
LDT-NAACL — Layer-wise Lexicality Probing at the First-Output Prediction Position
=================================================================================

Claim boundary (K1/K2/K3 — enforced in every metadata file):
  * We probe the layer-wise HIDDEN STATE AT THE FIRST-OUTPUT PREDICTION
    POSITION (the final prompt token) for lexicality. We do NOT claim that the
    LLM "predicts WORD/NONWORD at each layer": the probe is an analysis tool.
  * Layer depth is not a reaction time. Human-RT alignment is a statistical
    association between probe-derived lexical evidence and ELP RTs.

Analysis families added for the NAACL revision (letters follow the revision
master list; every output row carries its representation / readout label):
  A  methodological controls: MLP probe, random-init model, sublayer
     decomposition, position identity check, mean-pool vs primary,
     surface-form (orthographic / tokenizer) baselines
  B  confound controls: caliper matching, SMD / variance-ratio balance,
     covariate-erased probe, frequency-covariate models, length × frequency
  C  causal analyses: probe-space dose-response with random-direction null,
     residual-stream steering, head ablation, subspace mean-ablation,
     permutation / random-probe nulls for cross-model transfer
  D  attention mass, attention knockout, stimulus-position sensitivity,
     tokenization-stratified primary results
  E  human-RT alignment: accuracy filter, HC3 SEs, mixed model,
     global FDR, overlap statistics, RT distribution diagnostics
  F  contextual controls inside / outside the task prompt
  G  exact p-values (log10), bootstrap CIs, DeLong AUC tests, Bayes factors
  H  scaling regression, curve alignment, linear CKA, transfer vs native
  I  figures;  J  reproducibility (+ `python LDT.py --self-test`)

"""
import gc
import inspect
import logging
import os
import random
import re
import string
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

# J1 — no hard-coded device. Set LDT_GPU_INDEX (an index or a GPU/MIG UUID)
# to pin a device; otherwise CUDA_VISIBLE_DEVICES is left untouched.
TARGET_GPU = os.environ.get('LDT_GPU_INDEX')
if TARGET_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(TARGET_GPU)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                      'expandable_segments:True,max_split_size_mb:512')

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*Unknown solver options.*')
warnings.filterwarnings('ignore')

import json
from datetime import datetime
import matplotlib
matplotlib.use("Agg", force=True)   # headless: script only saves figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.integrate
import seaborn as sns
import statsmodels.api as sm
import torch
import torch.nn.functional as F
import transformers as _transformers_pkg
from matplotlib import gridspec
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
# ── Install the iprint-filter at import time, before any probe fit runs ──
try:
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings(
        "ignore", category=OptimizeWarning,
        message=r".*Unknown solver options: iprint.*")
except Exception:
    pass
# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multi_model_ldt_v7_first_output_token.log')
    ]
)
logger = logging.getLogger(__name__)
SEED = 42

# ── transformers version check ──────────────────────────────────────────────

_MIN_TRANSFORMERS_VERSION = "4.53.0"


def _parse_version(v: str) -> tuple:
    core = v.split("+")[0].split("dev")[0].split("rc")[0].strip(".")
    parts = []
    for p in core.split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    parts = (parts + [0, 0, 0])[:3]
    return tuple(parts)


def _check_transformers_version(min_version: str = _MIN_TRANSFORMERS_VERSION) -> None:
    installed = _transformers_pkg.__version__
    if _parse_version(installed) < _parse_version(min_version):
        msg = (
            f"\n{'!'*70}\n"
            f"transformers {installed} is installed, but newer architectures\n"
            f"(Qwen3-*, SmolLM3-*, and anything else merged into transformers\n"
            f"after your current version) need >={min_version} to load.\n"
            f"Models already using older/established architectures will still\n"
            f"work fine — only those newer families will fail, with an error\n"
            f"like: \"model type 'qwen3'/'smollm3' but Transformers does not\n"
            f"recognize this architecture\".\n\n"
            f"Fix — in the same environment you run this script from:\n"
            f"    pip install -U \"transformers>={min_version}\"\n"
            f"{'!'*70}\n"
        )
        logger.warning(msg)
    else:
        logger.info(f"transformers version OK: {installed} (>= {min_version})")


_check_transformers_version()

# ── GPU ──────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    _gp = torch.cuda.get_device_properties(0)
    logger.info(f"GPU: {_gp.name}  ({_gp.total_memory/1024**3:.1f} GB)")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"-> device 0 = {_gp.name}, cc={_gp.major}.{_gp.minor}, "
                f"SMs={_gp.multi_processor_count}")
    if _gp.total_memory < 20 * 1024**3:
        logger.info(f"Using MIG slice: {_gp.name} "
                    f"({_gp.total_memory/1024**3:.1f} GB)")
else:
    DEVICE = torch.device("cpu")
    logger.warning("CUDA not available — using CPU")

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ════════════════════════════════════════════════════════════════════════════
# ENUMS & CONFIG
# ════════════════════════════════════════════════════════════════════════════

class ArchitectureType(Enum):
    ENCODER_ONLY    = "encoder-only"
    DECODER_ONLY    = "decoder-only"
    ENCODER_DECODER = "encoder-decoder"


class InputMode(Enum):
    RAW_TEXT = "raw_text"
    CHAT     = "chat"


class PaddingSide(Enum):
    LEFT  = "left"
    RIGHT = "right"


class RepresentationType(Enum):
    """
    POSITIONAL representation strategies (NOT pooling operations, except
    MEAN_PROMPT_TOKENS). Every analysis output carries one of these labels.
    """
    # PRIMARY — final non-padding position of the PROMPT, i.e. the position
    # whose hidden state predicts the first output token. Nothing generated,
    # nothing appended.
    FIRST_OUTPUT_PREDICTION_POSITION = "first_output_prediction_position"
    MEAN_PROMPT_TOKENS               = "mean_prompt_tokens"                       # CONTROL
    # CONTROL — hidden state of the token the model actually generated (the
    # previous version's primary representation). Kept for backward
    # comparability ONLY; at that position the emitted answer token is the input.
    GENERATED_TOKEN_CONTROL          = "generated_token_representation_control"
    CONTEXTUAL_TARGET_TOKEN          = "contextual_target_token"                  # CONTEXTUAL CONTROL


PRIMARY_REPRESENTATION = RepresentationType.FIRST_OUTPUT_PREDICTION_POSITION.value

# Readout labels. Every result row records which of these produced it.
# PRIMARY: independent per-layer linear WORD/NONWORD probes, 5-fold cross-fitted,
# so every stimulus is scored only by a probe that never saw it. Architecture-
# and tokenizer-agnostic: no YES/NO tokens, no vocabulary-dependent scoring.
PRIMARY_READOUT          = "crossfitted_linear_probe"
# SECONDARY: the single stratified train/val/test probe used by the extended
# analyses (1-10), kept unchanged for comparability with earlier runs.
HOLDOUT_PROBE_READOUT    = "holdout_linear_probe"
SECONDARY_READOUT        = HOLDOUT_PROBE_READOUT      # backward-compatible alias
# OPTIONAL diagnostic, disabled by default (Config.ENABLE_NATIVE_LM_HEAD_DIAGNOSTIC)
NATIVE_LM_HEAD_READOUT   = "native_lm_head"


# ════════════════════════════════════════════════════════════════════════════
# TASK-CONDITIONED LEXICAL-DECISION PROMPT  (single source of truth)
# ════════════════════════════════════════════════════════════════════════════
# "Prompt trick" = this fixed, hand-designed task-conditioned prompt, which
# makes the first output-token position an answer position, so that the
# YES/NO logits read off that position are a lexical decision.
# It is NOT optimised in any way (no prompt search, no selection on any data).
# The SAME template is used for words, nonwords, and every frequency group;
# only {STIMULUS} changes. No label, frequency, or RT information is inserted.
TASK_PROMPT_TEMPLATE = (
    "Determine whether the following letter string is a valid English word.\n"
    "\n"
    "Letter string: {STIMULUS}\n"
    "\n"
    "Respond with exactly one answer:\n"
    "YES = English word\n"
    "NO = not an English word\n"
    "\n"
    "Answer:"
)

_TEMPLATE_FIELDS = {f for _, f, _, _ in string.Formatter().parse(TASK_PROMPT_TEMPLATE)
                    if f is not None}
if _TEMPLATE_FIELDS != {"STIMULUS"} or TASK_PROMPT_TEMPLATE.count("{STIMULUS}") != 1:
    raise RuntimeError(f"TASK_PROMPT_TEMPLATE must contain exactly one {{STIMULUS}} "
                       f"field and nothing else; found {_TEMPLATE_FIELDS}")


def build_task_prompt(stimulus: str) -> str:
    """The ONLY place the lexical-decision prompt is constructed.
    Receives the stimulus string only (never a label)."""
    return TASK_PROMPT_TEMPLATE.format(STIMULUS=str(stimulus))


# Explicit answer vocabulary. A generated first token is mapped to a response
# ONLY via token-ID sets that are precomputed once per tokenizer by scanning the
# full vocabulary with this normaliser (see AllLayerExtractor._build_answer_token_maps).
ANSWER_WORD_FORMS    = ("yes",)   # YES → WORD
ANSWER_NONWORD_FORMS = ("no",)    # NO  → NONWORD
_ANSWER_STRIP_CHARS  = " \t\r\n.,:;!?\"'`()[]{}*_-"


def normalize_answer_text(token_text: str) -> str:
    """Whitespace/punctuation-stripped, case-folded surface form of ONE token."""
    return token_text.strip(_ANSWER_STRIP_CHARS).casefold()


def probe_type_name(hidden_dims) -> str:
    return ("linear_logistic_regression" if not hidden_dims
            else "mlp_" + "x".join(str(h) for h in hidden_dims))


@dataclass
class ModelConfig:
    name:              str
    model_id:          str
    architecture_type: str
    input_mode:        str  = "raw_text"
    padding_side:      str  = "left"
    batch_size:        int  = 32
    use_8bit:          bool = False
    use_4bit:          bool = False

    def __post_init__(self):
        self.architecture_type = ArchitectureType(self.architecture_type)
        self.input_mode        = InputMode(self.input_mode)
        self.padding_side      = PaddingSide(self.padding_side)


@dataclass
class Config:
    WORDS_PATH:    str = "Items.csv"
    NONWORDS_PATH: str = "NonWord.csv"
    # New directory: previous (raw-stimulus) results are never overwritten.
    OUTPUT_DIR:    str = "LDT_Journel-Results-9Models_crossfitted_linear_probe"
    CONTEXT_USE_SYNONYM_CONTROL: bool = True
    CONTEXT_SYNONYM_PAIRS_PATH: str | None = None
    # ── Analysis 9: cross-model stitching (replaces literal weight transfer) ──
    STITCHING_ALPHAS: list[float] = field(default_factory=lambda: [1.0, 10.0, 100.0, 1000.0, 10000.0])
    STITCHING_MIN_PAIRED_SAMPLES: int = 100
    MODELS: list[ModelConfig] = field(default_factory=lambda: [
    
        ModelConfig(
            name="SmolLM3-3B-Base",
            model_id="HuggingFaceTB/SmolLM3-3B-Base",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),
        ModelConfig(
            name="Qwen3-4B-Base",
            model_id="Qwen/Qwen3-4B-Base",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Qwen3-8B-Base",
            model_id="Qwen/Qwen3-8B-Base",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="SmolLM2-360M",
            model_id="HuggingFaceTB/SmolLM2-360M",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Qwen2.5-1.5B",
            model_id="Qwen/Qwen2.5-1.5B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Llama-3.2-1B",
            model_id="meta-llama/Llama-3.2-1B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Qwen2.5-3B",
            model_id="Qwen/Qwen2.5-3B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="LLaMA-3.2-3B",
            model_id="meta-llama/Llama-3.2-3B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),

        ModelConfig(
            name="Llama-3.1-8B",
            model_id="meta-llama/Llama-3.1-8B",
            architecture_type="decoder-only",
            input_mode="raw_text",
            padding_side="left",
            batch_size=32, use_8bit=False, use_4bit=False
        ),
    ])
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_SAMPLES: int | None = None

    CLEAR_CACHE_BETWEEN_MODELS: bool = True

    HIGH_FREQ_PERCENTILE: float = 66.0
    LOW_FREQ_PERCENTILE:  float = 33.0

    # PRIMARY PROBE = LINEAR LOGISTIC REGRESSION.
    # hidden_dims=[] makes LexicalDecisionClassifier a single nn.Linear(D, 2);
    # softmax over two logits is exactly binary logistic regression on
    # (w1 - w0)·x + (b1 - b0), fitted with class-weighted cross-entropy,
    # decoupled L2 weight decay, and validation-loss early stopping.
    # Using one probe family for ALL analyses keeps them comparable
    # (Hewitt & Liang, 2019). Set e.g. [512, 256] ONLY for a separately
    # reported non-linear robustness check.
    CLASSIFIER_ARCHITECTURE: list  = field(default_factory=lambda: [])
    CLASSIFIER_USE_RESIDUAL:  bool  = False
    CLASSIFIER_DROPOUT:       float = 0.3
    CLASSIFIER_LR:            float = 1e-3
    CLASSIFIER_EPOCHS:        int   = 20    # FIX: was 50 (early stop rarely fires late)
    CLASSIFIER_PATIENCE:      int   = 4     # FIX: was 10
    CLASSIFIER_WEIGHT_DECAY:  float = 1e-2
    CLASSIFIER_BATCH_SIZE:    int   = 512   # FIX: was 64; GPU was idle
    CLASSIFIER_FOCAL_GAMMA:   float = 0.0
    CLASSIFIER_BN_MOMENTUM:   float = 0.1
    CLASSIFIER_NOISE_STD:     float = 0.0

    VAL_SIZE:  float = 0.15
    TEST_SIZE: float = 0.15

    MIN_BATCH_SIZE:        int   = 8
    MIN_TEST_SAMPLES:      int   = 10
    MIN_SAMPLES_PER_GROUP: int   = 50
    ALPHA:                 float = 0.05

    # ── Analysis 4: Multi-seed stability ──────────────────────────────
    N_SEEDS: int = 5  # Number of random seeds for stability analysis

    # ── Analysis 5: Representation ablation (positional strategies) ───
    # First entry MUST be the primary representation. All three are taken
    # from the SAME forward pass and probed with the SAME split/probe/seed.
    # The PRIMARY representation must stay first. The rest are SECONDARY
    # CONTROLS; the generated-token control additionally requires
    # RUN_NATIVE_GENERATION_DIAGNOSTIC (it needs a generated token to exist).
    REPRESENTATION_TYPES: list[str] = field(default_factory=lambda: [
        RepresentationType.FIRST_OUTPUT_PREDICTION_POSITION.value,
        RepresentationType.MEAN_PROMPT_TOKENS.value,
    ])
    # dtype for the two CONTROL caches (primary is always float32). Use
    # "float16" only if CPU RAM is insufficient for 3 × N × L × H float32.
    CONTROL_REPRESENTATION_DTYPE: str = "float32"

    # ── Analysis 8: Representation intervention (causal test) ─────────
    INTERVENTION_TOP_K:     int   = 10   # dims manipulated (by |LR coef|)
    INTERVENTION_MAGNITUDE: float = 0.5  # perturbation size, in standardised
                                          # (z-scored) units along the
                                          # frequency-discriminant direction

    # ── Analysis 9: Cross-model probe transfer ─────────────────────────
    TRANSFER_TEST_LAYERS: list[int] | None = None  # None -> representative
                                                        # layers (first/mid/last/
                                                        # every-4th), matched by
                                                        # normalised depth across
                                                        # architectures of
                                                        # different layer counts

    # ── Analysis 10: Contextual (sentence-embedded) frequency effect ──
    CONTEXT_SENTENCE_TEMPLATE: str = "The {} was seen by the group."
    # FIX: 200 items with a single 15% test split meant 30 test items (SE of
    # accuracy ≈ 0.09). Now 1000 items, every item scored out-of-fold.
    CONTEXT_N_SAMPLES: int = 1000   # 500 high + 500 low
    CONTEXT_SEED:      int = 42
    # F2: carrier in which the target is the FINAL word, so the final position
    # IS the target position (isolates position from context).
    CONTEXT_FINAL_POSITION_TEMPLATE: str = "The group looked at the {}"
    # F1: carrier sentence placed INSIDE the lexical-decision task prompt.
    CONTEXT_TASK_TEMPLATE: str = (
        "Determine whether the target letter string in the sentence below is a "
        "valid English word.\n"
        "\n"
        "Sentence: {SENTENCE}\n"
        "Target letter string: {STIMULUS}\n"
        "\n"
        "Respond with exactly one answer:\n"
        "YES = English word\n"
        "NO = not an English word\n"
        "\n"
        "Answer:"
    )

    FIGURES_DIR:    str = None
    RESULTS_DIR:    str = None
    COMPARISON_DIR: str = None

    # ── Analysis 11: Human–LLM Reaction-Time Alignment (NEW, modular) ──
    # This analysis is entirely additive: it reads the same cached
    # hidden states already produced for the existing pipeline, writes
    # to its own sub-directory, and never touches the files/objects
    # written by Analyses 1-10.
    RT_ALIGNMENT_DIR: str = None
    # Candidate ELP column names for "mean lexical-decision RT for
    # words", tried in order — the first one present in WORDS_PATH is
    # used and logged explicitly (see _identify_rt_column).
    ELP_RT_COLUMN_CANDIDATES: list[str] = field(default_factory=lambda: [
        "I_Mean_RT", "Mean_RT", "I_Mean_RT_Word", "MeanRT",
        "RT_Mean", "I_RT_Mean", "ReactionTime", "Mean_RT_Word",
    ])
    ELP_ACCURACY_COLUMN_CANDIDATES: list[str] = field(default_factory=lambda: [
        "I_Mean_Accuracy", "Mean_Accuracy", "I_Accuracy",
    ])
    RT_ALIGNMENT_VAL_SIZE: float = 0.15   # held-out fraction for probe early stopping
    RT_ALIGNMENT_MIN_WORDS_PER_LAYER: int = 30
    RT_ALIGNMENT_SEED: int = SEED
    RT_LOG_SKEW_THRESHOLD: float = 1.0    # |skew| above this -> use log(RT) in regression

    # ── Analysis 11: 5-fold cross-fitted lexical evidence ──────────────
    # Every item's lexical-evidence score is produced by a probe that was
    # never trained on that item. Human RT is never used in probe fitting.
    RT_ALIGNMENT_CROSSFIT: bool = True
    RT_ALIGNMENT_N_FOLDS: int = 5
    RT_ALIGNMENT_CV_SHUFFLE: bool = True

    # ── PRIMARY: per-layer cross-fitted linear WORD/NONWORD probes ─────
    # 5 folds, stratified, deterministic. Every stimulus is scored exactly once,
    # by the fold probe that did not train on it. Folds, scaler and early
    # stopping are all fitted inside the training fold only.
    PRIMARY_PROBE_N_FOLDS: int = 5
    PRIMARY_PROBE_SEED:    int = SEED
    PRIMARY_PROBE_CV_SHUFFLE: bool = True
    # Numerically stable clipping before logit(P(WORD)).
    LEXICAL_EVIDENCE_CLIP: float = 1e-6

    # Primary probe estimator. The previous version used liblinear, which
    # (unlike every other solver) L2-penalises the intercept. 'lbfgs' fits the
    # textbook L2-regularised logistic regression; set 'liblinear' only to
    # reproduce numbers from the earlier runs exactly.
    PRIMARY_PROBE_C:      float = 1.0
    PRIMARY_PROBE_SOLVER: str   = "lbfgs"
    PRIMARY_PROBE_MAX_ITER: int = 1000
    # 'auto' → full-batch L-BFGS in torch on the GPU when DEVICE == 'cuda'
    # (the SAME convex objective as sklearn: mean log-loss + ‖w‖²/(2·C·n),
    # intercept unpenalised → same optimum; unit-tested against sklearn);
    # 'sklearn' forces CPU sklearn, 'torch' forces torch on DEVICE.
    PRIMARY_PROBE_BACKEND: str  = "auto"

    # ── OPTIONAL secondary diagnostic: native LM-head YES/NO readout ────
    # J5: ON by default, because the model-level causal analyses (C2, C3, C5,
    # D2) are read out through the model's own YES/NO logits. It stays a
    # DIAGNOSTIC: it never replaces the primary cross-fitted probe.
    ENABLE_NATIVE_LM_HEAD_DIAGNOSTIC: bool = True
    # Intermediate hidden states are NOT normalised, whereas the LM head was
    # trained on the post-final-norm residual stream. Applying the final
    # norm before the head is therefore required for the readout to be
    # meaningful (logit lens; nostalgebraist 2020). Set False only for an
    # explicitly reported ablation.
    APPLY_FINAL_NORM_BEFORE_LM_HEAD: bool = True
    # Multi-token YES/NO fallback: if an answer has no single-token surface
    # form, score it by its FIRST sub-token (documented, never silent).
    MULTI_TOKEN_ANSWER_SCORING: str = "first_subword_token"
    LM_HEAD_BATCH_ROWS: int = 4096   # rows per chunk when applying the head

    # ── Secondary behavioural diagnostic: one-token greedy generation ───
    # Off by default (§13: no unnecessary generated sequences are stored).
    # Required if REPRESENTATION_TYPES contains the generated-token control.
    RUN_NATIVE_GENERATION_DIAGNOSTIC: bool = False
    GENERATION_STRATEGY: str = "greedy"   # recorded; the only supported value
    MAX_NEW_TOKENS:      int = 1          # recorded; the only supported value
    # Sanity set (§23) — obvious items, run before full extraction.
    SANITY_WORDS:    list[str] = field(default_factory=lambda: [
        "house", "apple", "table", "dog"])
    SANITY_NONWORDS: list[str] = field(default_factory=lambda: [
        "xqzpt", "qwrtx", "blorpz", "zkjvt"])
    # Elicitation is flagged UNRELIABLE if the UNKNOWN (non-YES/NO) first-token
    # rate exceeds this, or native accuracy is not above this.
    PROMPT_UNRELIABLE_UNKNOWN_RATE: float = 0.25
    PROMPT_UNRELIABLE_MIN_ACCURACY: float = 0.50
    # If True, an unreliable model raises instead of being reported-and-kept.
    STOP_IF_PROMPT_UNRELIABLE: bool = False

    # ── Primary emergence-curve statistics ──────────────────────────────
    N_BOOTSTRAP:            int   = 2000   # percentile bootstrap over evaluated items
    EMERGENCE_EARLY_DEPTH:  float = 0.25   # relative depth (li+1)/L ≤ this = "early"
    TOKEN_IDENTITY_MIN_STRATUM: int = 20   # min test items per generated-token stratum

    PRIMARY_DIR: str = None   # spec-named consolidated outputs
    NAACL_DIR:   str = None   # outputs of the NAACL-revision analyses (A–J)

    # ════════════════════════════════════════════════════════════════════
    # NAACL REVISION — analysis switches and hyper-parameters
    # ════════════════════════════════════════════════════════════════════
    # "representative" = first / quartiles / last / every 4th layer (the same
    # rule used by analyses 4, 6, 8, 9); "all" = every transformer layer.
    # ── A1: non-linear probe control (Hewitt & Liang, 2019) ─────────────
    RUN_MLP_PROBE:            bool      = True
    MLP_PROBE_HIDDEN_DIMS:    list[int] = field(default_factory=lambda: [512, 256])
    MLP_PROBE_USE_RESIDUAL:   bool      = True
    MLP_PROBE_LAYERS:         str       = "all"
    MLP_PROBE_EPOCHS:         int       = 30
    MLP_PROBE_PATIENCE:       int       = 4
    MLP_PROBE_BATCH_SIZE:     int       = 512
    MLP_PROBE_LR:             float     = 1e-3
    MLP_PROBE_WEIGHT_DECAY:   float     = 1e-2
    MLP_PROBE_INNER_VAL_FRAC: float     = 0.1   # early-stopping split, inside the training fold
    # ── A2: random-initialisation control (Zhang & Bowman, 2018) ────────
    RUN_RANDOM_INIT_BASELINE: bool       = True
    RANDOM_INIT_SEED:         int        = SEED
    RANDOM_INIT_MAX_ITEMS:    int | None = None  # None → the full stimulus set
    # ── A3 / C2 / C3 / C5 / D1 / D2 / D3: live-model analyses ───────────
    RUN_MODEL_LEVEL_ANALYSES: bool = True
    MODEL_LEVEL_N_ITEMS:      int  = 1000   # balanced words / nonwords
    MODEL_LEVEL_FIT_MAX_ITEMS: int = 20000  # disjoint items used to fit directions / probes
    MODEL_LEVEL_BATCH_SIZE:   int  = 16
    MODEL_LEVEL_LAYERS:       str  = "representative"
    MODEL_LEVEL_MAX_LAYERS:   int  = 8      # cap on intervention layers per model
    SUBLAYER_N_ITEMS:         int  = 4000
    POSITION_SENSITIVITY_N_ITEMS: int = 4000
    STEERING_MAGNITUDES: list[float] = field(
        default_factory=lambda: [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0])
    N_RANDOM_DIRECTIONS:      int  = 20     # null draws for C1 (probe space, cheap)
    # null draws for the live-model C2 / C3 / C5 (each draw = full forward
    # passes); the empirical p-value cannot fall below 1/(draws + 1), so the
    # parametric paired t-test and the z vs the random distribution are
    # reported alongside it.
    N_RANDOM_DIRECTIONS_MODEL_LEVEL: int = 5
    SUBSPACE_ABLATION_K:      int  = 4      # INLP iterations for C5
    HEAD_ABLATION_TOP_K:      int  = 10
    HEAD_ABLATION_MODE:       str  = "mean"   # "mean" (default) | "zero"
    KNOCKOUT_CONTROL:         bool = True     # knock out equally many NON-stimulus tokens
    # ── A6 (added): surface-form baselines without any LLM hidden state ─
    RUN_SURFACE_BASELINES:    bool = True
    CHAR_NGRAM_RANGE: tuple[int, int] = (1, 3)
    CHAR_NGRAM_FEATURES:      int  = 2 ** 18
    # ── B1–B3: confound matching ────────────────────────────────────────
    MATCH_CALIPER_SD:         float = 0.2    # per-covariate caliper, pooled-SD units (Austin, 2011)
    SMD_BALANCE_THRESHOLD:    float = 0.1
    # ── B4 / B5 / B6 ────────────────────────────────────────────────────
    RUN_FREQUENCY_COVARIATE_ANALYSES: bool = True
    COVARIATE_ERASURE_LAYERS: str  = "representative"
    # ── C1: probe-space dose-response ───────────────────────────────────
    INTERVENTION_MAGNITUDES: list[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    # ── C4 / H3 / H4: transfer nulls ────────────────────────────────────
    TRANSFER_N_PERMUTATIONS:  int  = 1000   # cheap: closed form, see _transfer_nulls
    TRANSFER_N_RANDOM_PROBES: int  = 200
    # ── E: human RT ─────────────────────────────────────────────────────
    RT_ALIGNMENT_MIN_HUMAN_ACCURACY: float | None = 0.8
    RT_PRIMARY_SAMPLE:        str  = "accuracy_filtered"   # | "all_items"
    RT_CLUSTER_VARIABLE:      str  = "freq_group"
    RT_MIN_CLUSTERS:          int  = 20    # cluster-robust SEs are invalid with few clusters
    RT_MIXED_MODEL_GROUP:     str  = "word_length"
    RT_MIXED_MODEL_MIN_GROUPS: int = 5
    RT_MIXED_MODEL_LAYERS:    str  = "representative"
    RT_OUTLIER_Z:             float = 3.0
    # ── G: statistics ───────────────────────────────────────────────────
    BF_EVIDENCE_THRESHOLD:    float = 3.0   # BF01 ≥ 3 → moderate evidence for H0
    # ── H: cross-model ──────────────────────────────────────────────────
    RUN_CKA:                  bool = True
    CKA_N_ITEMS:              int  = 1000
    CURVE_ALIGNMENT_GRID:     int  = 50
    # ── J8 ──────────────────────────────────────────────────────────────
    RUN_SELF_TESTS_BEFORE_EXPERIMENT: bool = True

    def _validate(self):
        """J7 — explicit validation of every switch and numeric range, so a
        mis-set flag fails at start-up instead of hours into a run."""
        def _in(name, lo, hi, lo_open=False, hi_open=False):
            v = getattr(self, name)
            if v is None or not np.isfinite(v):
                raise ValueError(f"Config.{name} must be finite, got {v!r}")
            if (v <= lo if lo_open else v < lo) or (v >= hi if hi_open else v > hi):
                raise ValueError(f"Config.{name}={v} outside "
                                 f"{'(' if lo_open else '['}{lo}, {hi}{')' if hi_open else ']'}")
        for n in ('VAL_SIZE', 'TEST_SIZE', 'RT_ALIGNMENT_VAL_SIZE',
                  'MLP_PROBE_INNER_VAL_FRAC'):
            _in(n, 0.0, 1.0, lo_open=True, hi_open=True)
        if self.VAL_SIZE + self.TEST_SIZE >= 1.0:
            raise ValueError("VAL_SIZE + TEST_SIZE must be < 1")
        _in('ALPHA', 0.0, 0.5, lo_open=True)
        _in('LOW_FREQ_PERCENTILE', 0.0, 100.0)
        _in('HIGH_FREQ_PERCENTILE', 0.0, 100.0)
        if self.LOW_FREQ_PERCENTILE >= self.HIGH_FREQ_PERCENTILE:
            raise ValueError("LOW_FREQ_PERCENTILE must be < HIGH_FREQ_PERCENTILE")
        for n in ('PRIMARY_PROBE_N_FOLDS', 'RT_ALIGNMENT_N_FOLDS'):
            if int(getattr(self, n)) < 2:
                raise ValueError(f"Config.{n} must be ≥ 2")
        for n in ('N_BOOTSTRAP', 'N_SEEDS', 'MIN_TEST_SAMPLES', 'MIN_SAMPLES_PER_GROUP',
                  'MODEL_LEVEL_N_ITEMS', 'SUBLAYER_N_ITEMS', 'CKA_N_ITEMS',
                  'POSITION_SENSITIVITY_N_ITEMS', 'CONTEXT_N_SAMPLES',
                  'MODEL_LEVEL_BATCH_SIZE', 'MODEL_LEVEL_MAX_LAYERS',
                  'MODEL_LEVEL_FIT_MAX_ITEMS',
                  'HEAD_ABLATION_TOP_K', 'SUBSPACE_ABLATION_K',
                  'N_RANDOM_DIRECTIONS', 'N_RANDOM_DIRECTIONS_MODEL_LEVEL',
                  'CURVE_ALIGNMENT_GRID',
                  'MLP_PROBE_EPOCHS', 'MLP_PROBE_BATCH_SIZE', 'RT_MIN_CLUSTERS',
                  'RT_MIXED_MODEL_MIN_GROUPS', 'INTERVENTION_TOP_K',
                  'STITCHING_MIN_PAIRED_SAMPLES'):
            v = getattr(self, n)
            if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1:
                raise ValueError(f"Config.{n} must be a positive integer, got {v!r}")
        for n in ('TRANSFER_N_PERMUTATIONS', 'TRANSFER_N_RANDOM_PROBES'):
            v = getattr(self, n)
            if not isinstance(v, (int, np.integer)) or v < 0:
                raise ValueError(f"Config.{n} must be a non-negative integer")
        _in('PRIMARY_PROBE_C', 0.0, np.inf, lo_open=True, hi_open=True)
        if self.PRIMARY_PROBE_BACKEND not in ('auto', 'sklearn', 'torch'):
            raise ValueError("PRIMARY_PROBE_BACKEND must be 'auto' | 'sklearn' | 'torch'")
        if self.PRIMARY_PROBE_BACKEND == 'torch' and self.PRIMARY_PROBE_SOLVER == 'liblinear':
            raise ValueError("liblinear penalises the intercept; use backend 'sklearn' for it")
        if self.PRIMARY_PROBE_SOLVER not in ('lbfgs', 'liblinear', 'newton-cg',
                                             'newton-cholesky', 'sag', 'saga'):
            raise ValueError(f"unsupported PRIMARY_PROBE_SOLVER {self.PRIMARY_PROBE_SOLVER!r}")
        _in('LEXICAL_EVIDENCE_CLIP', 0.0, 0.5, lo_open=True, hi_open=True)
        _in('MATCH_CALIPER_SD', 0.0, np.inf, lo_open=True, hi_open=True)
        _in('SMD_BALANCE_THRESHOLD', 0.0, np.inf, lo_open=True, hi_open=True)
        _in('EMERGENCE_EARLY_DEPTH', 0.0, 1.0, lo_open=True)
        _in('BF_EVIDENCE_THRESHOLD', 1.0, np.inf, hi_open=True)
        if self.RT_ALIGNMENT_MIN_HUMAN_ACCURACY is not None:
            _in('RT_ALIGNMENT_MIN_HUMAN_ACCURACY', 0.0, 1.0)
        if self.RT_PRIMARY_SAMPLE not in ('accuracy_filtered', 'all_items'):
            raise ValueError("RT_PRIMARY_SAMPLE must be 'accuracy_filtered' | 'all_items'")
        if (self.RT_PRIMARY_SAMPLE == 'accuracy_filtered'
                and self.RT_ALIGNMENT_MIN_HUMAN_ACCURACY is None):
            raise ValueError("RT_PRIMARY_SAMPLE='accuracy_filtered' needs "
                             "RT_ALIGNMENT_MIN_HUMAN_ACCURACY")
        for n in ('MLP_PROBE_LAYERS', 'MODEL_LEVEL_LAYERS', 'COVARIATE_ERASURE_LAYERS',
                  'RT_MIXED_MODEL_LAYERS'):
            if getattr(self, n) not in ('all', 'representative'):
                raise ValueError(f"Config.{n} must be 'all' or 'representative'")
        if self.HEAD_ABLATION_MODE not in ('mean', 'zero'):
            raise ValueError("HEAD_ABLATION_MODE must be 'mean' or 'zero'")
        if not self.MLP_PROBE_HIDDEN_DIMS or any(int(h) < 1 for h in self.MLP_PROBE_HIDDEN_DIMS):
            raise ValueError("MLP_PROBE_HIDDEN_DIMS must be a non-empty list of positive ints")
        if not self.INTERVENTION_MAGNITUDES or any(m <= 0 for m in self.INTERVENTION_MAGNITUDES):
            raise ValueError("INTERVENTION_MAGNITUDES must be positive")
        if not self.STEERING_MAGNITUDES or 0.0 in self.STEERING_MAGNITUDES:
            raise ValueError("STEERING_MAGNITUDES must be non-empty and exclude 0 "
                             "(the unperturbed baseline is always evaluated)")
        lo_, hi_ = self.CHAR_NGRAM_RANGE
        if not (1 <= lo_ <= hi_):
            raise ValueError("CHAR_NGRAM_RANGE must satisfy 1 ≤ lo ≤ hi")
        if self.RANDOM_INIT_MAX_ITEMS is not None and self.RANDOM_INIT_MAX_ITEMS < 100:
            raise ValueError("RANDOM_INIT_MAX_ITEMS must be None or ≥ 100")
        if self.CONTEXT_SENTENCE_TEMPLATE.count('{}') != 1 or \
                self.CONTEXT_FINAL_POSITION_TEMPLATE.count('{}') != 1:
            raise ValueError("context sentence templates need exactly one '{}' slot")
        if not self.CONTEXT_FINAL_POSITION_TEMPLATE.endswith('{}'):
            raise ValueError("CONTEXT_FINAL_POSITION_TEMPLATE must END with the target slot")
        for f_ in ('{SENTENCE}', '{STIMULUS}'):
            if self.CONTEXT_TASK_TEMPLATE.count(f_) != 1:
                raise ValueError(f"CONTEXT_TASK_TEMPLATE must contain {f_} exactly once")
        if self.CONTEXT_TASK_TEMPLATE.index('{SENTENCE}') > \
                self.CONTEXT_TASK_TEMPLATE.index('{STIMULUS}'):
            raise ValueError("CONTEXT_TASK_TEMPLATE: {SENTENCE} must precede {STIMULUS}")
        for mc in self.MODELS:
            if not isinstance(mc, ModelConfig):
                raise ValueError("Config.MODELS must contain ModelConfig objects")
            if mc.batch_size < 1:
                raise ValueError(f"{mc.name}: batch_size must be ≥ 1")
            if mc.use_4bit and mc.use_8bit:
                raise ValueError(f"{mc.name}: use_4bit and use_8bit are exclusive")
        names = [mc.name for mc in self.MODELS]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")

    def __post_init__(self):
        self._validate()
        if self.GENERATION_STRATEGY != "greedy" or self.MAX_NEW_TOKENS != 1:
            raise ValueError("Only greedy decoding with MAX_NEW_TOKENS=1 is valid "
                             "for first-generated-output-token extraction.")
        if (RepresentationType.GENERATED_TOKEN_CONTROL.value in self.REPRESENTATION_TYPES
                and not self.RUN_NATIVE_GENERATION_DIAGNOSTIC):
            raise ValueError("The generated-token control representation requires "
                             "RUN_NATIVE_GENERATION_DIAGNOSTIC=True.")
        if self.MULTI_TOKEN_ANSWER_SCORING != "first_subword_token":
            raise ValueError("Only 'first_subword_token' multi-token answer "
                             "scoring is implemented.")
        if not self.REPRESENTATION_TYPES or \
                self.REPRESENTATION_TYPES[0] != PRIMARY_REPRESENTATION:
            raise ValueError("REPRESENTATION_TYPES[0] must be "
                             f"'{PRIMARY_REPRESENTATION}'.")
        for rt in self.REPRESENTATION_TYPES:
            RepresentationType(rt)
        if self.CONTROL_REPRESENTATION_DTYPE not in ("float32", "float16"):
            raise ValueError("CONTROL_REPRESENTATION_DTYPE must be float32|float16")
        self.FIGURES_DIR    = os.path.join(self.OUTPUT_DIR, "figures")
        self.RESULTS_DIR    = os.path.join(self.OUTPUT_DIR, "results")
        self.COMPARISON_DIR = os.path.join(self.OUTPUT_DIR, "comparisons")
        self.RT_ALIGNMENT_DIR = os.path.join(self.OUTPUT_DIR, "human_rt_alignment")
        self.PRIMARY_DIR    = os.path.join(self.OUTPUT_DIR,
                                           "primary_layerwise_linear_probe")
        self.NAACL_DIR      = os.path.join(self.OUTPUT_DIR, "naacl_revision")
        for d in [self.OUTPUT_DIR, self.FIGURES_DIR,
                  self.RESULTS_DIR, self.COMPARISON_DIR,
                  self.RT_ALIGNMENT_DIR, self.PRIMARY_DIR,
                  self.NAACL_DIR, os.path.join(self.NAACL_DIR, "figures"),
                  os.path.join(self.RT_ALIGNMENT_DIR, "figures"),
                  os.path.join(self.RT_ALIGNMENT_DIR, "results")]:
            os.makedirs(d, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class LDTDataset(Dataset):
    """Balanced word / non-word dataset with frequency stratification."""

    def __init__(self, words_df, nonwords_df, tokenizer, config: Config):
        self.tokenizer = tokenizer
        self.config    = config

        wp  = self._process_words(words_df)
        nwp = self._process_nonwords(nonwords_df)

        if config.MAX_SAMPLES is not None:
            tgt = config.MAX_SAMPLES // 2
            wp  = wp.sample(n=min(tgt, len(wp)),   random_state=SEED)
            nwp = nwp.sample(n=min(tgt, len(nwp)), random_state=SEED)
        else:
            n   = min(len(wp), len(nwp))
            wp  = wp.sample(n=n,  random_state=SEED)
            nwp = nwp.sample(n=n, random_state=SEED)

        self.data = (pd.concat([wp, nwp], ignore_index=True)
                       .sample(frac=1, random_state=SEED)
                       .reset_index(drop=True))
        self._compute_token_counts(tokenizer)
        self._stratify_frequency()
        self._print_summary()

    def _process_words(self, df):
        p = pd.DataFrame()
        p['stimulus']      = df['Word'].astype(str).str.lower()
        p['is_word']       = 1
        p['length']        = pd.to_numeric(df['Length'],           errors='coerce')
        p['accuracy']      = pd.to_numeric(
            df['I_Mean_Accuracy'].replace('#', np.nan),            errors='coerce')
        p['log_frequency'] = pd.to_numeric(
            df['Log_Freq_HAL'].replace('#', np.nan),               errors='coerce')
        p['ortho_n']       = pd.to_numeric(df['Ortho_N'].replace('#', np.nan), errors='coerce')
        p['is_high_freq']  = 0
        p['is_low_freq']   = 0
        p['freq_group']    = 'mid'
        return p.dropna(subset=['stimulus'])

    def _process_nonwords(self, df):
        p = pd.DataFrame()
        p['stimulus']      = df['Word'].astype(str).str.lower()
        p['is_word']       = 0
        p['length']        = pd.to_numeric(df['Length'],           errors='coerce')
        p['accuracy']      = pd.to_numeric(
            df['NWI_Mean_Accuracy'].replace('#', np.nan),          errors='coerce')
        p['log_frequency'] = np.nan
        p['ortho_n']       = pd.to_numeric(df['Ortho_N'].replace('#', np.nan), errors='coerce')
        p['is_high_freq']  = 0
        p['is_low_freq']   = 0
        p['freq_group']    = 'nonword'
        return p.dropna(subset=['stimulus'])

    # ── Analysis 2: Tokenization control ──────────────────────────────
    def _compute_token_counts(self, tokenizer):
        """
        For each stimulus, compute the number of subword tokens and
        whether it is a single-token item under this tokenizer.
        This is essential for controlling tokenization confounds:
        frequency effects observed in probing could be artefacts of
        high-frequency words being single-token while low-frequency
        words are multi-token (and thus have a fundamentally different
        representation structure).
        """
        token_counts = []
        for stim in self.data['stimulus']:
            ids = tokenizer.encode(stim, add_special_tokens=False)
            token_counts.append(len(ids))
        self.data['token_count'] = token_counts
        self.data['is_single_token'] = (self.data['token_count'] == 1).astype(int)
        logger.info(
            f"Tokenization: single-token={self.data['is_single_token'].sum()}, "
            f"multi-token={(self.data['is_single_token'] == 0).sum()}, "
            f"mean tokens={self.data['token_count'].mean():.2f}"
        )

    def _stratify_frequency(self):
        wm = self.data['is_word'] == 1
        fm = wm & self.data['log_frequency'].notna()
        if fm.sum() == 0:
            logger.warning("No frequency data — stratification skipped"); return
        fv  = self.data.loc[fm, 'log_frequency']
        hi  = np.percentile(fv, self.config.HIGH_FREQ_PERCENTILE)
        lo  = np.percentile(fv, self.config.LOW_FREQ_PERCENTILE)
        hm  = fm & (self.data['log_frequency'] >= hi)
        lm  = fm & (self.data['log_frequency'] <= lo)
        mm  = fm & ~hm & ~lm
        self.data.loc[hm, ['is_high_freq','freq_group']] = [1, 'high']
        self.data.loc[lm, ['is_low_freq', 'freq_group']] = [1, 'low']
        self.data.loc[mm,  'freq_group']                 = 'mid'
        logger.info(f"Freq stratification: hi>={hi:.3f}  lo<={lo:.3f}")

    def _print_summary(self):
        wm = self.data['is_word'] == 1
        print(f"\n{'='*64}\nDATASET SUMMARY\n{'='*64}")
        print(f"  Total         : {len(self.data):,}")
        print(f"  Words  (1)    : {wm.sum():,}")
        print(f"  Non-words (0) : {(~wm).sum():,}")
        print(f"  High-freq     : {(self.data['is_high_freq']==1).sum():,}")
        print(f"  Low-freq      : {(self.data['is_low_freq'] ==1).sum():,}")
        print(f"  Single-token  : {self.data['is_single_token'].sum():,}")
        print(f"  Multi-token   : {(self.data['is_single_token']==0).sum():,}")
        word_ratio = wm.sum() / len(self.data)
        print(f"  Word ratio    : {word_ratio:.3f}")
        print(f"{'='*64}\n")

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        r = self.data.iloc[idx]
        def ft(v): return torch.tensor(v if pd.notna(v) else float('nan'),
                                       dtype=torch.float32)
        return {
            'stimulus':        r['stimulus'],
            'is_word':         torch.tensor(r['is_word'],      dtype=torch.long),
            'accuracy':        ft(r['accuracy']),
            'log_frequency':   ft(r['log_frequency']),
            'length':          ft(r['length']),
            'ortho_n':         ft(r['ortho_n']),
            'is_high_freq':    torch.tensor(r['is_high_freq'], dtype=torch.long),
            'is_low_freq':     torch.tensor(r['is_low_freq'],  dtype=torch.long),
            'freq_group':      r['freq_group'],
            'token_count':     torch.tensor(r['token_count'],  dtype=torch.long),
            'is_single_token': torch.tensor(r['is_single_token'], dtype=torch.long),
            'idx':             idx,
        }


class ContextualLDTDataset:
    """
    Analysis 10 support class — generates sentence-embedded stimuli for
    a stratified sample of high- and low-frequency words already present
    in an ``LDTDataset``.

    Design note (explicitly flagged, not glossed over): the "isolated"
    condition reuses the ALREADY-CACHED task-conditioned
    first-generated-output-token hidden states (PRIMARY representation) from
    the main extraction pass (no new forward pass, per the memory-management
    note in the task spec). The sentence condition, however, requires
    embedding each word in a novel carrier sentence that was never
    tokenised/forwarded during the main extraction pass. There is no way
    to obtain those representations without at least one additional
    forward pass per sentence — this directly contradicts the blanket
    "do not run additional transformer forward passes" instruction
    elsewhere in the spec. I am implementing Analysis 10 correctly (with
    the necessary new forward passes) rather than silently degrading it,
    because doing otherwise would make the analysis meaningless. See the
    accompanying notes for how this is kept small and where the model is
    kept resident to make it possible.
    """

    def __init__(self, targets: dict, config: Config):
        self.config = config
        is_word      = targets['is_word']
        is_high_freq = targets['is_high_freq']
        is_low_freq  = targets['is_low_freq']
        stimuli      = targets['stimulus']
        _all_idx     = targets['idx']  # not used downstream here, retained for completeness

        wm  = (is_word == 1)
        hfm = wm & (is_high_freq == 1)
        lfm = wm & (is_low_freq  == 1)

        n_each = config.CONTEXT_N_SAMPLES // 2
        rng    = np.random.RandomState(config.CONTEXT_SEED)

        hf_pos = np.where(hfm)[0]
        lf_pos = np.where(lfm)[0]
        hf_sel = rng.choice(hf_pos, size=min(n_each, len(hf_pos)), replace=False)
        lf_sel = rng.choice(lf_pos, size=min(n_each, len(lf_pos)), replace=False)

        if len(hf_sel) < config.MIN_SAMPLES_PER_GROUP or \
           len(lf_sel) < config.MIN_SAMPLES_PER_GROUP:
            logger.warning(
                f"  Contextual analysis: insufficient words "
                f"(hf={len(hf_sel)}, lf={len(lf_sel)}) — will be skipped."
            )

        sel_positions = np.concatenate([hf_sel, lf_sel])
        self.positions_in_all_hidden = sel_positions  # index into all_hidden[layer] rows
        self.words   = [stimuli[i] for i in sel_positions]
        self.sentences = [config.CONTEXT_SENTENCE_TEMPLATE.format(w)
                           for w in self.words]
        self.freq_label = np.concatenate([
            np.ones(len(hf_sel), dtype=int), np.zeros(len(lf_sel), dtype=int)
        ])  # 1 = high freq, 0 = low freq
        self.n_high = len(hf_sel)
        self.n_low  = len(lf_sel)

    def __len__(self):
        return len(self.words)


# ════════════════════════════════════════════════════════════════════════════
# INSTRUMENTED ATTENTION (D1 attention mass, D2 attention knockout)
# ════════════════════════════════════════════════════════════════════════════
class _LDTAttentionContext:
    """Per-forward state read by `_ldt_attention_forward`. Inactive by default,
    in which case the function is plain softmax attention (numerically the
    eager implementation)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.query_pos = None      # LongTensor (B,): the query row of interest
        self.block = None          # BoolTensor (B, T_k): keys that query may NOT see
        self.layers = None         # set[int] | None (None = every layer)
        self.capture = None        # dict[layer] -> (B, T_k) head-mean attention


_LDT_ATTN_CTX = _LDTAttentionContext()
_LDT_ATTN_NAME = "ldt_instrumented"
_LDT_ATTN_REGISTERED = None


def _ldt_attention_forward(module, query, key, value, attention_mask,
                           scaling=None, dropout=0.0, **kwargs):
    """Architecture-agnostic attention (GQA-aware) that can (a) knock out the
    edges query_pos → blocked keys at selected layers (Geva et al., 2023,
    'attention knockout') and (b) record the attention distribution of the
    query row (head-averaged). Receives post-RoPE / post-q_norm states, as
    every HF attention implementation does."""
    ctx = _LDT_ATTN_CTX
    B, nH, Tq, d = query.shape
    n_rep = nH // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    Tk = key.shape[2]
    sc = scaling if scaling is not None else d ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * sc
    if attention_mask is None:
        causal = torch.ones(Tq, Tk, dtype=torch.bool, device=scores.device).tril(Tk - Tq)
        scores = scores.masked_fill(~causal, float('-inf'))
    else:
        m = attention_mask[..., :Tk]
        if m.dtype == torch.bool:
            scores = scores.masked_fill(~m, float('-inf'))
        else:
            scores = scores + m.float()
    li = getattr(module, 'layer_idx', None)
    in_scope = ctx.layers is None or (li is not None and li in ctx.layers)
    if ctx.block is not None and ctx.query_pos is not None and in_scope:
        qp = ctx.query_pos.to(scores.device)
        blk = ctx.block.to(scores.device)[:, :Tk]
        rows = scores[torch.arange(B, device=scores.device), :, qp, :]      # (B, nH, Tk)
        rows = rows.masked_fill(blk[:, None, :], float('-inf'))
        scores[torch.arange(B, device=scores.device), :, qp, :] = rows
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    if ctx.capture is not None and ctx.query_pos is not None and li is not None:
        qp = ctx.query_pos.to(probs.device)
        ctx.capture[li] = probs[torch.arange(B, device=probs.device), :, qp, :].mean(1).cpu()
    out = torch.matmul(probs.to(value.dtype), value)
    return out.transpose(1, 2).contiguous(), None


def ensure_ldt_attention_registered() -> bool:
    """Registers the instrumented attention (and an SDPA-style boolean mask
    builder) with transformers' AttentionInterface (transformers ≥ 4.48)."""
    global _LDT_ATTN_REGISTERED
    if _LDT_ATTN_REGISTERED is not None:
        return _LDT_ATTN_REGISTERED
    try:
        from transformers import AttentionInterface
        AttentionInterface.register(_LDT_ATTN_NAME, _ldt_attention_forward)
        try:
            from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
            AttentionMaskInterface.register(_LDT_ATTN_NAME, sdpa_mask)
        except Exception:
            pass          # older versions build float masks themselves
        _LDT_ATTN_REGISTERED = True
    except Exception as e:                               # noqa: BLE001
        logger.warning(f"Instrumented attention unavailable ({e}); D1/D2 skipped.")
        _LDT_ATTN_REGISTERED = False
    return _LDT_ATTN_REGISTERED


# ════════════════════════════════════════════════════════════════════════════
# SINGLE-PASS ALL-LAYER EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════

class AllLayerExtractor:
    """
    Frozen decoder-only LM. One ORDINARY forward pass of the task prompt
    (stimulus included) per batch yields, at every transformer layer:

      PRIMARY  first_output_prediction_position
               h_l at the final non-padding PROMPT position — the state that
               predicts the first output token — read by the model's OWN
               pretrained LM head: logit(YES) vs logit(NO) → WORD/NONWORD.
               Nothing is generated, nothing is appended, nothing is trained,
               and the readout is never fed back into the transformer.

      CONTROL  mean_prompt_tokens — mean of h_l over prompt positions.

    Computed SEPARATELY and clearly labelled as secondary:
      * one-token greedy generation (behavioural diagnostic, §12/§22)
      * generated_token_representation_control — h_l at the position of the
        token the model actually generated (the previous version's primary
        representation; retained for comparability only)

    Causality: position t attends only to positions ≤ t, so the final-prompt
    state is identical whether or not a token is appended afterwards — this is
    asserted in run_prompt_sanity_checks rather than assumed.

    Layer-index convention (unchanged): layer li ≡ hidden_states[li + 1];
    layer 0 = first transformer block output; embeddings not analysed.
    """

    def __init__(self, model_config: ModelConfig, device: str = 'cuda',
                 config: 'Config' = None, random_init: bool = False):
        """random_init=True (A2) keeps the pretrained TOKENIZER and
        architecture but draws every weight from the architecture's own
        initialiser (`_init_weights`, seeded by Config.RANDOM_INIT_SEED)."""
        self.config = config
        self.random_init = bool(random_init)
        print(f"\n{'='*64}\nLOADING: {model_config.name}"
              f"{'  [RANDOM-INIT CONTROL]' if self.random_init else ''}")
        print(f"  model_id     : {model_config.model_id}")
        print(f"  arch         : {model_config.architecture_type.value}")
        print(f"  input_mode   : {model_config.input_mode.value}")
        print(f"  padding_side : {model_config.padding_side.value}")
        print(f"  representation: {PRIMARY_REPRESENTATION} "
              f"(final prompt position) | readout: {PRIMARY_READOUT}")
        print(f"{'='*64}")

        if model_config.architecture_type != ArchitectureType.DECODER_ONLY:
            # Explicit failure instead of silent degradation: an encoder-only
            # model has no autoregressive "first generated token".
            raise NotImplementedError(
                f"{model_config.name}: first-generated-output-token extraction "
                f"requires a decoder-only (causal LM) model; got "
                f"{model_config.architecture_type.value}.")

        self.model_config      = model_config
        self.device            = device
        self.architecture_type = model_config.architecture_type
        self.input_mode        = model_config.input_mode

        # ── tokeniser ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_id,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = model_config.padding_side.value

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── quantisation ──────────────────────────────────────────────────
        qcfg = None
        if model_config.use_4bit:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif model_config.use_8bit:
            qcfg = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=True
            )

        dmap    = 'auto' if device == 'cuda' else None
        max_mem = None
        if device == 'cuda' and torch.cuda.is_available():
            tot     = torch.cuda.get_device_properties(0).total_memory / 1024**3
            max_mem = {0: f"{int(tot*0.85)}GB", "cpu": "30GB"}

        kw = {
            'device_map': dmap,
            'quantization_config': qcfg,
            'low_cpu_mem_usage': True,
            'max_memory': max_mem,
            'trust_remote_code': True
        }

        # ── dtype selection (unchanged) ───────────────────────────────
        if device == 'cuda' and torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 8:
                kw['torch_dtype'] = torch.bfloat16
                logger.info(f"  Using bfloat16 (GPU CC {cc[0]}.{cc[1]} ≥ 8.0)")
            else:
                kw['torch_dtype'] = torch.float16
                logger.info(f"  Using float16 (GPU CC {cc[0]}.{cc[1]} < 8.0 — "
                            f"bf16 not supported, watch for overflow in deep layers)")
        else:
            kw['torch_dtype'] = torch.float32
        self.torch_dtype = str(kw['torch_dtype'])
        # transformers ≥ 4.56 renamed torch_dtype → dtype (the old name warns)
        if _parse_version(_transformers_pkg.__version__) >= (4, 56, 0):
            kw['dtype'] = kw.pop('torch_dtype')

        def _is_unrecognized_architecture_error(exc: Exception) -> bool:
            msg = str(exc)
            return (
                "does not recognize this architecture" in msg
                or "Unrecognized configuration class" in msg
                or "Unrecognized model" in msg
            )

        def _unrecognized_architecture_error(model_id: str, exc: Exception) -> RuntimeError:
            installed = _transformers_pkg.__version__
            return RuntimeError(
                f"'{model_id}' uses a model architecture that transformers "
                f"{installed} does not recognize yet. This is a library-version "
                f"issue, not a data/config problem — fix with:\n"
                f"    pip install -U \"transformers>={_MIN_TRANSFORMERS_VERSION}\"\n"
                f"then re-run. (Original error: {exc})"
            )

        # A language-modelling head is REQUIRED: the model must generate its
        # own first answer token. (The previous version loaded AutoModel, i.e.
        # a headless backbone that cannot generate.)
        try:
            if self.random_init:
                from transformers import AutoConfig
                cfg_rand = AutoConfig.from_pretrained(model_config.model_id,
                                                      trust_remote_code=True)
                seed_ri = int(self._cfg('RANDOM_INIT_SEED', SEED))
                torch.manual_seed(seed_ri)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_ri)
                dkey = 'dtype' if 'dtype' in kw else 'torch_dtype'
                self.model = AutoModelForCausalLM.from_config(
                    cfg_rand, trust_remote_code=True, **{dkey: kw[dkey]})
                if device == 'cuda' and torch.cuda.is_available():
                    self.model = self.model.to('cuda')
                self._model_type = "AutoModelForCausalLM.from_config (random init)"
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_config.model_id, **kw)
                self._model_type = "AutoModelForCausalLM"
        except Exception as e:
            if _is_unrecognized_architecture_error(e):
                raise _unrecognized_architecture_error(model_config.model_id, e) from None
            raise RuntimeError(
                f"{model_config.model_id} could not be loaded with a causal-LM "
                f"head; first-generated-output-token extraction is impossible "
                f"without one. Original error: {e}") from e

        # ── FROZEN transformer ────────────────────────────────────────────
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Headless backbone (same weights) for the hidden-state forward:
        # identical hidden_states, but no (B, T, |V|) logits tensor.
        prefix = getattr(self.model, 'base_model_prefix', None)
        bb = getattr(self.model, prefix, None) if prefix else None
        self._backbone = bb if isinstance(bb, nn.Module) else self.model
        self._supports_position_ids = True

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        try:
            self.n_parameters = int(sum(p.numel() for p in self.model.parameters()))
        except Exception:
            self.n_parameters = None
        print(f"✓ Loaded ({self._model_type}) | layers={self.num_layers} "
              f"| hidden_dim={self.hidden_dim}")
        print(f"  tokenizer    : {type(self.tokenizer).__name__}")
        print(f"  pad_token    : '{self.tokenizer.pad_token}' "
              f"(id={self.tokenizer.pad_token_id})")
        print(f"  padding_side : {self.tokenizer.padding_side}")

        # Pure greedy decoding for exactly one token. A fresh GenerationConfig
        # is used on purpose so that model-default processors (e.g. a
        # repetition_penalty in generation_config.json) cannot alter argmax.
        self._gen_config = GenerationConfig(
            max_new_tokens=1, do_sample=False, num_beams=1,
            output_scores=True, return_dict_in_generate=True,
            output_hidden_states=False, output_attentions=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        # The PRIMARY experiment needs neither YES/NO tokens nor the LM head.
        # Both are built ONLY for the optional native-readout diagnostic and
        # for the greedy-generation diagnostic.
        self.native_readout_enabled = bool(
            self._cfg('ENABLE_NATIVE_LM_HEAD_DIAGNOSTIC', False))
        self.generation_enabled = bool(
            self._cfg('RUN_NATIVE_GENERATION_DIAGNOSTIC', False))
        if self.native_readout_enabled or self.generation_enabled:
            self._build_answer_token_maps()
        else:
            self._blank_answer_token_maps()
        if self.native_readout_enabled:
            self._resolve_native_lm_head()
        else:
            self._blank_native_lm_head()
            print("  LM head      : native YES/NO readout DISABLED "
                  "(primary readout is the cross-fitted linear probe)")

        if self.input_mode == InputMode.CHAT:
            self._validate_chat_template()

        print()

    def _blank_answer_token_maps(self):
        """No YES/NO machinery: the primary pipeline never uses it."""
        self.yes_token_ids, self.no_token_ids = [], []
        self._yes_id_set, self._no_id_set = set(), set()
        self.answer_token_report = []

    def _blank_native_lm_head(self):
        self._lm_head = None
        self._lm_head_name = None
        self._lm_head_tied = None
        self._final_norm, self._final_norm_attr = None, None
        self._last_hidden_is_normed = None
        self._lm_head_verification = {'enabled': False}
        self.yes_score_ids, self.no_score_ids = [], []
        self.yes_scoring_mode = self.no_scoring_mode = None
        self.single_token_yes = self.single_token_no = None

    # ── description for metadata ──────────────────────────────────────────
    def describe(self) -> dict:
        return {
            'model': self.model_config.name,
            'model_id': self.model_config.model_id,
            'architecture': self.architecture_type.value,
            'model_class': self._model_type,
            'tokenizer': type(self.tokenizer).__name__,
            'number_of_layers': int(self.num_layers),
            'hidden_dimension': int(self.hidden_dim),
            'padding_side': self.tokenizer.padding_side,
            'input_mode': self.input_mode.value,
            'torch_dtype': self.torch_dtype,
            'n_parameters': self.n_parameters,
            'random_init_control': bool(self.random_init),
            'hidden_states_last_is_post_final_norm_note': (
                'In HF decoder-only models hidden_states[-1] is the output of the '
                'FINAL NORM, whereas hidden_states[1..L-1] are raw residual-stream '
                'states. Reported layer L-1 is therefore RMS/Layer-normalised; '
                'per-feature standardisation in the probes removes scale, but the '
                'last point of every curve is a different kind of state.'),
            'layer_index_convention': ('layer li = hidden_states[li+1]; layer 0 = '
                                       'output of first transformer block; '
                                       'embedding output not analysed'),
            'representation_type': PRIMARY_REPRESENTATION,
            'primary_readout_type': PRIMARY_READOUT,
            'native_lm_head_diagnostic_enabled': bool(self.native_readout_enabled),
            'generation_diagnostic_enabled': bool(self.generation_enabled),
            # tokenisation diagnostics — only meaningful for the optional
            # YES/NO diagnostics; empty when those are disabled.
            'yes_token_text': [self.tokenizer.decode([t]) for t in self.yes_score_ids],
            'yes_token_id': list(self.yes_score_ids),
            'no_token_text': [self.tokenizer.decode([t]) for t in self.no_score_ids],
            'no_token_id': list(self.no_score_ids),
            'yes_tokenization': [r for r in self.answer_token_report if r['label'] == 'WORD'],
            'no_tokenization': [r for r in self.answer_token_report if r['label'] == 'NONWORD'],
            'single_token_yes': bool(self.single_token_yes),
            'single_token_no': bool(self.single_token_no),
            'yes_scoring_mode': self.yes_scoring_mode,
            'no_scoring_mode': self.no_scoring_mode,
            'answer_score_definition': ('logsumexp of the LM-head logits over all '
                                        'listed token ids of that answer'),
            'lm_head_pathway': self._lm_head_verification,
            'position_identity_check': dict(getattr(self, 'position_identity_check', {}) or {}),
        }

    # ── explicit YES/NO token-ID sets (§13, §53) ──────────────────────────
    def _build_answer_token_maps(self):
        """
        Exhaustively scans the tokenizer vocabulary ONCE and assigns a token
        ID to WORD (YES) / NONWORD (NO) iff its single-token surface form,
        after `normalize_answer_text`, is exactly one of the answer forms.
        Tokens that are only a PREFIX of an answer (e.g. "Y") are NOT mapped,
        so they yield UNKNOWN rather than a guessed decision.
        """
        V = len(self.tokenizer)
        decoded = self.tokenizer.batch_decode(
            [[i] for i in range(V)], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        yes_ids, no_ids = [], []
        for tid, txt in enumerate(decoded):
            norm = normalize_answer_text(txt)
            if norm in ANSWER_WORD_FORMS:
                yes_ids.append(tid)
            elif norm in ANSWER_NONWORD_FORMS:
                no_ids.append(tid)
        self.yes_token_ids = sorted(yes_ids)
        self.no_token_ids  = sorted(no_ids)
        self._yes_id_set, self._no_id_set = set(yes_ids), set(no_ids)

        # How the canonical answers tokenize (multi-subword answers: only the
        # FIRST token is ever analysed — the complete answer is never used).
        report = []
        for label, forms in (('WORD', ('YES', 'Yes', 'yes')),
                             ('NONWORD', ('NO', 'No', 'no'))):
            for f in forms:
                for pre in ('', ' '):
                    ids = self.tokenizer.encode(pre + f, add_special_tokens=False)
                    report.append({
                        'label': label, 'surface': pre + f, 'token_ids': ids,
                        'n_subword_tokens': len(ids),
                        'first_token_id': ids[0] if ids else None,
                        'first_token_mapped_to':
                            (self.classify_generated_token(ids[0]) if ids else None),
                    })
        self.answer_token_report = report
        print(f"  YES-token ids: {len(self.yes_token_ids)} "
              f"{[self.tokenizer.decode([t]) for t in self.yes_token_ids[:8]]}")
        print(f"  NO-token ids : {len(self.no_token_ids)} "
              f"{[self.tokenizer.decode([t]) for t in self.no_token_ids[:8]]}")
        if not self.yes_token_ids or not self.no_token_ids:
            logger.warning(f"  {self.model_config.name}: no single-token YES and/or "
                           f"NO surface form in the vocabulary — native "
                           f"predictions can only be UNKNOWN for that answer.")

    def classify_generated_token(self, token_id: int) -> str:
        tid = int(token_id)
        if tid in self._yes_id_set:
            return 'WORD'
        if tid in self._no_id_set:
            return 'NONWORD'
        return 'UNKNOWN'

    def _validate_chat_template(self):
        test_msg = [{"role": "user", "content": "test"}]
        try:
            if (hasattr(self.tokenizer, 'apply_chat_template') and
                    self.tokenizer.chat_template is not None):
                formatted = self.tokenizer.apply_chat_template(
                    test_msg, tokenize=False, add_generation_prompt=True
                )
                print(f"  chat_template test: {formatted[:80]!r}...")
        except Exception as e:
            logger.warning(f"Chat template validation failed: {e}")

    def build_input_text(self, text: str) -> str:
        """Model-specific formatting of an arbitrary user text (RAW or CHAT).
        Used for the task prompt AND for Analysis 10 carrier sentences."""
        if self.input_mode == InputMode.RAW_TEXT:
            return text

        if (hasattr(self.tokenizer, 'apply_chat_template') and
                self.tokenizer.chat_template is not None):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(
                    f"apply_chat_template failed ({e}); using model-specific fallback"
                )

        model_id_lower = self.model_config.model_id.lower()
        if 'phi-3' in model_id_lower or 'phi3' in model_id_lower:
            return f"<|user|>\n{text}<|end|>\n<|assistant|>\n"
        if 'qwen' in model_id_lower:
            return (f"<|im_start|>user\n{text}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
        if 'llama' in model_id_lower:
            return (f"<|begin_of_text|><|start_header_id|>user"
                    f"<|end_header_id|>\n\n{text}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")

        logger.warning(
            f"No known template for {self.model_config.model_id}; "
            f"using raw text"
        )
        return text

    def build_task_input(self, stimulus: str) -> str:
        """Task-conditioned prompt, formatted for this model's input mode."""
        return self.build_input_text(build_task_prompt(stimulus))

    def _add_special_tokens(self) -> bool:
        # Chat templates already contain BOS/role tokens.
        return self.input_mode == InputMode.RAW_TEXT

    def _encode_prompts(self, prompts: list[str]) -> dict:
        # truncation=False: a truncated prompt would silently change the task.
        return self.tokenizer(prompts, return_tensors='pt', padding=True,
                              truncation=False,
                              add_special_tokens=self._add_special_tokens())

    @staticmethod
    def _finite_numpy(t: torch.Tensor, dtype=np.float32) -> np.ndarray:
        """Upcast → CPU numpy; irrecoverable NaN/Inf → 0.0 (counted later by
        _sanity_check), exactly as in the previous extractor."""
        a = t.float().cpu().numpy()
        bad = ~np.isfinite(a)
        if bad.any():
            a[bad] = 0.0
        return a.astype(dtype, copy=False)

    def _backbone_forward(self, input_ids, attention_mask, position_ids=None):
        kw = {'input_ids': input_ids, 'attention_mask': attention_mask,
              'output_hidden_states': True, 'use_cache': False}
        if position_ids is not None and self._supports_position_ids:
            kw['position_ids'] = position_ids
        try:
            return self._backbone(**kw)
        except TypeError as e:
            if 'position_ids' in kw and 'position_ids' in str(e):
                self._supports_position_ids = False
                logger.warning(f"  {self.model_config.name}: backbone does not "
                               f"accept position_ids; using model default.")
                kw.pop('position_ids')
                return self._backbone(**kw)
            kw.pop('use_cache', None)
            return self._backbone(**kw)

    # ── STEP 3: deterministic generation of ONE token (§10, §52) ─────────
    @torch.no_grad()
    def _generate_first_tokens(self, stimuli):
        """
        NO-LABEL-LEAKAGE GUARANTEE (§9, §66): this method receives ONLY the
        stimulus strings. Gold labels, frequency groups and human RT are not
        in scope here, so generation cannot depend on them. The signature is
        asserted to be exactly (stimuli) in run_prompt_sanity_checks.

        Returns the unpadded prompt token IDs, the ACTUAL generated token ID
        per stimulus, and the log-probability mass on YES- vs NO-tokens at the
        decision step (a behavioural diagnostic only; never used to choose
        which hidden state is extracted).
        """
        stimuli = [str(s) for s in stimuli]
        prompts = [self.build_task_input(s) for s in stimuli]

        def _one_call(prompt_batch):
            enc = self._encode_prompts(prompt_batch)
            ids = enc['input_ids'].to(self.device)
            am  = enc['attention_mask'].to(self.device)
            out = self.model.generate(input_ids=ids, attention_mask=am,
                                      generation_config=self._gen_config)
            seq = out.sequences
            T = ids.shape[1]
            if seq.shape[1] != T + 1 or not torch.equal(seq[:, :T], ids):
                raise RuntimeError("generate() did not return [prompt] + exactly "
                                   "one new token; cannot locate the first "
                                   "generated token.")
            gen = seq[:, T].detach().cpu().tolist()
            logits = out.scores[0].detach().float().cpu()          # (B, |V|)
            pids = [ids[b][am[b].bool()].detach().cpu().tolist()
                    for b in range(ids.shape[0])]
            # consistency: greedy token == argmax of the decision-step scores
            if not torch.equal(logits.argmax(-1), torch.tensor(gen)):
                raise RuntimeError("Generated token != argmax of step logits — "
                                   "decoding is not pure greedy.")
            del out, seq, ids, am
            return pids, gen, logits

        if self.tokenizer.padding_side == 'left' or len(prompts) == 1:
            prompt_ids, gen_ids, logits = _one_call(prompts)
        else:
            # Right padding: batched decoder-only generation would read logits
            # at pad positions. Generate per sample (no padding at all).
            prompt_ids, gen_ids, parts = [], [], []
            for p in prompts:
                pi, gi, lg = _one_call([p])
                prompt_ids += pi; gen_ids += gi; parts.append(lg)
            logits = torch.cat(parts, 0)

        logp = torch.log_softmax(logits, dim=-1)
        V = logp.shape[1]
        yes = [t for t in self.yes_token_ids if t < V]
        no  = [t for t in self.no_token_ids if t < V]
        yes_lp = (torch.logsumexp(logp[:, yes], -1) if yes
                  else torch.full((len(gen_ids),), -np.inf))
        no_lp  = (torch.logsumexp(logp[:, no], -1) if no
                  else torch.full((len(gen_ids),), -np.inf))
        del logits, logp
        return {'prompts': prompts, 'prompt_ids': prompt_ids,
                'gen_ids': [int(g) for g in gen_ids],
                'yes_logprob': yes_lp.numpy().astype(float),
                'no_logprob': no_lp.numpy().astype(float)}

    # ── STEPS 5–7: append ACTUAL token, forward, locate its position ─────
    @torch.no_grad()
    def _forward_with_generated_token(self, prompt_ids: list, gen_ids: list):
        """
        Builds [prompt tokens] + [actual generated token] per sample, pads
        with the tokenizer's own padding side, and computes the generated
        position SEPARATELY for every sample (never assumes index -1).
        Every position is verified; any inconsistency raises.

        Returns hidden_states tuple, gen_pos (padded coords, LongTensor),
        prompt_mask (1 on prompt tokens only), prompt lengths (list[int]).
        """
        seqs  = [list(p) + [int(g)] for p, g in zip(prompt_ids, gen_ids)]
        plens = [len(p) for p in prompt_ids]
        padded = self.tokenizer.pad({'input_ids': seqs}, padding=True,
                                    return_tensors='pt')
        ids2, m2 = padded['input_ids'], padded['attention_mask']
        B, T2 = ids2.shape
        gen_pos = []
        for b in range(B):
            L_b = plens[b] + 1
            if int(m2[b].sum()) != L_b:
                raise RuntimeError(f"sample {b}: attention mask length mismatch")
            real = torch.nonzero(m2[b], as_tuple=False).flatten()
            pos = int(real[-1])                           # last REAL position
            expected = (T2 - L_b + plens[b]) if self.tokenizer.padding_side == 'left' \
                else plens[b]
            if pos != expected or int(ids2[b, pos]) != int(gen_ids[b]):
                raise RuntimeError(f"sample {b}: generated-token position "
                                   f"verification failed (pos={pos}, "
                                   f"expected={expected}).")
            if ids2[b, real[:-1]].tolist() != list(prompt_ids[b]):
                raise RuntimeError(f"sample {b}: prompt tokens altered by padding")
            gen_pos.append(pos)
        gen_pos = torch.tensor(gen_pos, dtype=torch.long)
        prompt_mask = m2.clone()
        prompt_mask[torch.arange(B), gen_pos] = 0

        # position_ids from the mask (0 at the first REAL token), matching the
        # convention generate() uses for left-padded batches.
        pos_ids = (m2.long().cumsum(-1) - 1).clamp(min=0)
        outputs = self._backbone_forward(ids2.to(self.device), m2.to(self.device),
                                         pos_ids.to(self.device))
        hs = outputs.hidden_states
        if len(hs) != self.num_layers + 1:
            raise RuntimeError(f"expected {self.num_layers + 1} hidden-state "
                               f"tensors, got {len(hs)}")
        return hs, gen_pos, prompt_mask, plens

    # ════════════════════════════════════════════════════════════════════
    # NATIVE LM-HEAD READOUT (PRIMARY) — "logit lens" over depth
    # ════════════════════════════════════════════════════════════════════
    def _resolve_native_lm_head(self):
        """
        Resolves the model's OWN pretrained output head through the architecture's
        native pathway (`get_output_embeddings()`, which respects weight tying)
        and the final normalisation module that precedes it.

        Nothing here is trained, re-initialised or modified.

        Why the final norm matters: the head was trained on the POST-final-norm
        residual stream. Intermediate hidden states are pre-norm and have a very
        different scale, so applying the head to them raw produces
        uninterpretable logits. Applying the model's own final norm first is the
        standard logit-lens construction (nostalgebraist, 2020; Belrose et al.,
        2023 discuss its limits — the readout is an approximation for layers the
        head was not trained to read).
        """
        head = self.model.get_output_embeddings()
        if head is None or not hasattr(head, 'weight'):
            raise RuntimeError(
                f"{self.model_config.name}: the model exposes no output embedding "
                f"matrix via get_output_embeddings(); the primary layer-wise "
                f"native LM-head readout cannot be computed.")
        self._lm_head = head
        self._lm_head_name = type(head).__name__
        self._lm_head_tied = bool(getattr(self.model.config, 'tie_word_embeddings', False))

        norm, norm_attr = None, None
        for attr in ('norm', 'final_layernorm', 'ln_f', 'final_layer_norm'):
            cand = getattr(self._backbone, attr, None)
            if isinstance(cand, nn.Module):
                norm, norm_attr = cand, attr
                break
        self._final_norm, self._final_norm_attr = norm, norm_attr
        if norm is None:
            logger.warning(f"  {self.model_config.name}: no final-norm module found on "
                           f"the backbone; the LM head will be applied to raw hidden "
                           f"states (recorded in metadata).")
        # Resolved empirically in _verify_lm_head_pathway():
        #   HF appends the POST-final-norm state as hidden_states[-1] for most
        #   decoder-only models, so the last layer must not be normed twice.
        self._last_hidden_is_normed = None
        self._lm_head_verification = {}
        self._build_answer_scoring_sets()
        print(f"  LM head      : {self._lm_head_name} "
              f"(tied={self._lm_head_tied}, final_norm={norm_attr})")

    def _build_answer_scoring_sets(self):
        """
        Chooses the token IDs whose logits represent YES and NO (§8, §9).

        Preference order, recorded explicitly, never silent:
          1. every single-token surface form of the answer in the vocabulary
             (scored by log-sum-exp over that set = log of the total
             unnormalised mass the head assigns to "answering YES/NO here");
          2. if an answer has NO single-token form, its FIRST sub-word token
             (config.MULTI_TOKEN_ANSWER_SCORING = 'first_subword_token'),
             which is the only token the first-output position can emit.
        """
        def _fallback(forms):
            ids = []
            for f in forms:
                for pre in ('', ' '):
                    e = self.tokenizer.encode(pre + f, add_special_tokens=False)
                    if e:
                        ids.append(int(e[0]))
            return sorted(set(ids))

        if self.yes_token_ids:
            self.yes_score_ids, self.yes_scoring_mode = list(self.yes_token_ids), 'single_token'
        else:
            self.yes_score_ids, self.yes_scoring_mode = _fallback(('YES', 'Yes', 'yes')), \
                self._cfg('MULTI_TOKEN_ANSWER_SCORING', 'first_subword_token')
        if self.no_token_ids:
            self.no_score_ids, self.no_scoring_mode = list(self.no_token_ids), 'single_token'
        else:
            self.no_score_ids, self.no_scoring_mode = _fallback(('NO', 'No', 'no')), \
                self._cfg('MULTI_TOKEN_ANSWER_SCORING', 'first_subword_token')
        if not self.yes_score_ids or not self.no_score_ids:
            raise RuntimeError(f"{self.model_config.name}: could not resolve YES/NO "
                               f"answer token IDs — refusing to substitute any other "
                               f"token.")
        self.single_token_yes = self.yes_scoring_mode == 'single_token'
        self.single_token_no  = self.no_scoring_mode == 'single_token'
        dev = self._lm_head.weight.device
        y = torch.tensor(self.yes_score_ids, dtype=torch.long, device=dev)
        n = torch.tensor(self.no_score_ids, dtype=torch.long, device=dev)
        # Rows of the head that matter — avoids materialising (B, |V|) logits.
        self._yes_W = self._lm_head.weight.index_select(0, y).detach()
        self._no_W  = self._lm_head.weight.index_select(0, n).detach()
        b = getattr(self._lm_head, 'bias', None)
        self._yes_b = b.index_select(0, y).detach() if b is not None else None
        self._no_b  = b.index_select(0, n).detach() if b is not None else None
        print(f"  answer tokens: YES={[self.tokenizer.decode([t]) for t in self.yes_score_ids[:6]]}"
              f" ({self.yes_scoring_mode}) | "
              f"NO={[self.tokenizer.decode([t]) for t in self.no_score_ids[:6]]}"
              f" ({self.no_scoring_mode})")

    def _cfg(self, key, default):
        return getattr(self.config, key, default) if self.config is not None else default

    def _normalise_for_head(self, h: torch.Tensor, li: int) -> torch.Tensor:
        """Apply the model's own final norm unless this state is already normed."""
        if not self._cfg('APPLY_FINAL_NORM_BEFORE_LM_HEAD', True):
            return h
        if self._final_norm is None:
            return h
        if li == self.num_layers - 1 and self._last_hidden_is_normed:
            return h                      # hidden_states[-1] is post-norm already
        return self._final_norm(h)

    def _answer_scores(self, h: torch.Tensor, li: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        YES / NO scores at layer li for hidden states h (B, H).

        READOUT ONLY: h is detached from the transformer computation; the
        result is never written back into any layer.
        """
        x = self._normalise_for_head(h.detach(), li).to(self._yes_W.dtype)
        ys = F.linear(x, self._yes_W, self._yes_b)      # (B, |yes ids|)
        ns = F.linear(x, self._no_W,  self._no_b)
        # log-sum-exp over the surface forms of each answer = log of the total
        # unnormalised mass assigned to that answer at this position.
        return torch.logsumexp(ys.float(), -1), torch.logsumexp(ns.float(), -1)

    @torch.no_grad()
    def _verify_lm_head_pathway(self, probe_stimuli) -> dict:
        """
        Establishes, empirically and per model, (a) whether hidden_states[-1] is
        already post-final-norm, and (b) that the cheap two-column readout
        reproduces the full head's logits exactly. Any failure raises.
        """
        prompts = [self.build_task_input(s) for s in probe_stimuli[:4]]
        enc = self._encode_prompts(prompts)
        ids = enc['input_ids'].to(self.device)
        am = enc['attention_mask'].to(self.device)
        out = self.model(input_ids=ids, attention_mask=am,
                         output_hidden_states=True, use_cache=False)
        fp = self._final_prompt_positions(am)
        ar = torch.arange(ids.shape[0], device=ids.device)
        ref = out.logits[ar, fp].float()                       # native head output
        h_last = out.hidden_states[-1][ar, fp]
        direct = self._lm_head(h_last).float()
        normed = (self._lm_head(self._final_norm(h_last)).float()
                  if self._final_norm is not None else None)
        d_direct = float((direct - ref).abs().max())
        d_normed = float((normed - ref).abs().max()) if normed is not None else np.inf
        self._last_hidden_is_normed = bool(d_direct <= d_normed)
        tol = max(1e-2, 50 * float(torch.finfo(ref.dtype).eps) * float(ref.abs().max()))
        best = min(d_direct, d_normed)
        if best > tol:
            raise RuntimeError(f"{self.model_config.name}: could not reproduce the "
                               f"model's own logits from hidden_states[-1] via the "
                               f"resolved head (|Δ| direct={d_direct:.3e}, "
                               f"normed={d_normed:.3e}); the readout pathway is wrong.")
        # cheap two-column path must match the full head exactly
        ys, ns = self._answer_scores(h_last, self.num_layers - 1)
        ref_y = torch.logsumexp(ref[:, self.yes_score_ids], -1)
        ref_n = torch.logsumexp(ref[:, self.no_score_ids], -1)
        d_cols = float(max((ys - ref_y).abs().max(), (ns - ref_n).abs().max()))
        if d_cols > tol:
            raise RuntimeError(f"{self.model_config.name}: the YES/NO column readout "
                               f"disagrees with the full LM head (|Δ|={d_cols:.3e}).")
        rep = {
            'lm_head_class': self._lm_head_name,
            'lm_head_tied_to_embeddings': self._lm_head_tied,
            'final_norm_module': self._final_norm_attr,
            'apply_final_norm_before_lm_head':
                bool(self._cfg('APPLY_FINAL_NORM_BEFORE_LM_HEAD', True)),
            'hidden_states_last_is_post_final_norm': self._last_hidden_is_normed,
            'max_abs_logit_error_direct': d_direct,
            'max_abs_logit_error_after_final_norm': d_normed,
            'max_abs_error_yes_no_column_readout': d_cols,
            'tolerance': tol,
        }
        self._lm_head_verification = rep
        print(f"  [head] hidden_states[-1] post-norm: {self._last_hidden_is_normed} "
              f"| max|Δlogit| = {best:.2e} | YES/NO column error = {d_cols:.2e} ✓")
        del out
        return rep

    # ── position of the state that predicts the FIRST OUTPUT TOKEN (§6, §17)
    @staticmethod
    def _final_prompt_positions(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Index of the final NON-PADDING prompt token for every row. Correct for
        left- AND right-padded batches (largest index with mask==1); never -1.
        """
        T = attention_mask.shape[1]
        idx = torch.arange(T, device=attention_mask.device).unsqueeze(0)
        pos = (attention_mask.long() * (idx + 1)).argmax(dim=1)
        if int((attention_mask.sum(1) == 0).sum()) > 0:
            raise RuntimeError("empty attention mask row — prompt tokenisation failed")
        return pos

    def _verify_final_positions(self, input_ids, attention_mask, pos, prompt_ids=None):
        """A4 — explicit identity check, per item: the extracted position is
        (i) not padding, (ii) followed only by padding, and (iii) holds the
        LAST NON-PAD PROMPT TOKEN. Any failure raises; passes are counted and
        written to experiment_metadata.json (position_identity_check)."""
        B = input_ids.shape[0]
        chk = getattr(self, 'position_identity_check', None)
        if chk is None:
            chk = self.position_identity_check = {
                'n_items_checked': 0, 'n_items_passed': 0,
                'n_items_token_identity_checked': 0,
                'rule': ('first-output prediction position == index of the last '
                         'non-pad prompt token (attention mask), and the token id '
                         'there == last prompt token id')}
        for b in range(B):
            if int(attention_mask[b, pos[b]]) != 1:
                raise RuntimeError(f"sample {b}: selected position is padding")
            after = attention_mask[b, pos[b] + 1:]
            if after.numel() and int(after.sum()) != 0:
                raise RuntimeError(f"sample {b}: non-padding tokens after the selected "
                                   f"final prompt position")
            if prompt_ids is not None and int(input_ids[b, pos[b]]) != int(prompt_ids[b][-1]):
                raise RuntimeError(f"sample {b}: token at the selected position is not "
                                   f"the last prompt token")
            chk['n_items_checked'] += 1
            chk['n_items_passed'] += 1
            if prompt_ids is not None:
                chk['n_items_token_identity_checked'] += 1

    def _read_representations(self, hs, gen_pos, prompt_mask, rep_types,
                              control_dtype=np.float32) -> dict:
        """Generated-token CONTROL representation, from the appended-token pass."""
        B = gen_pos.shape[0]
        out = {rt: {} for rt in rep_types}
        for li in range(self.num_layers):
            lh = hs[li + 1]
            dev = lh.device
            ar, gp = torch.arange(B, device=dev), gen_pos.to(dev)
            out[RepresentationType.GENERATED_TOKEN_CONTROL.value][li] = \
                self._finite_numpy(lh[ar, gp], control_dtype)
        return out

    # ════════════════════════════════════════════════════════════════════
    # §32 — PRIMARY EXTRACTION: prompt-only forward, layer-wise native readout
    # ════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def extract_layerwise_representations(self, dataloader, representation_types=None,
                                          control_dtype: str = "float32",
                                          run_generation_diagnostic: bool = False):
        """
        PRIMARY EXTRACTION (§2, §7, §13).

        The model receives the PROMPT ONLY. One ordinary forward pass per batch
        under torch.no_grad(); for every transformer layer the hidden state at
        the final real prompt position — the position whose state the causal LM
        uses to predict the first output token — is stored:

            h_l = hidden_states[l + 1][b, final_prompt_position_b, :]

        `final_prompt_position_b` comes from the attention mask per example, so
        left- and right-padded batches are both correct; index -1 is never used.
        No token is generated and nothing is appended before extraction. No
        logits, sequences or computation graphs are retained.

        The optional native YES/NO readout, the greedy-generation diagnostic and
        the generated-token CONTROL representation are computed only when
        explicitly enabled, and never feed the primary result.

        Returns
        -------
        reps        : {representation_type: {layer: (N, H) np.ndarray}}
        native      : {'yes_score': (N, L), 'no_score': (N, L)} or None
        targets     : metadata arrays
        diagnostics : pd.DataFrame, one row per stimulus
        """
        rep_types = list(representation_types or [PRIMARY_REPRESENTATION])
        if rep_types[0] != PRIMARY_REPRESENTATION:
            raise ValueError("first representation type must be the primary one")
        cdt = np.float16 if control_dtype == "float16" else np.float32
        gen_ctrl = RepresentationType.GENERATED_TOKEN_CONTROL.value in rep_types
        native_on = bool(self.native_readout_enabled)
        run_generation_diagnostic = bool(run_generation_diagnostic) or gen_ctrl
        if gen_ctrl and not self.generation_enabled:
            raise ValueError("generated-token control requested but the generation "
                             "diagnostic is disabled")
        if native_on and self._last_hidden_is_normed is None:
            self._verify_lm_head_pathway(self.config.SANITY_WORDS if self.config
                                         else ["house"])

        accum = {rt: {li: [] for li in range(self.num_layers)} for rt in rep_types}
        yes_acc, no_acc = [], []
        targets = {k: [] for k in
                   ['is_word', 'is_high_freq', 'is_low_freq',
                    'freq_group', 'stimulus', 'idx',
                    'token_count', 'is_single_token',
                    'log_frequency', 'length', 'ortho_n',
                    'generated_token_id', 'prompt_length']}
        targets['native_prediction'] = []
        diag_rows = []

        print(f"  Prompt-only forward pass, extracting {self.num_layers} layers "
              f"({len(dataloader)} batches); representations: {rep_types}; "
              f"native LM-head diagnostic: {'ON' if native_on else 'OFF'}")

        for batch in tqdm(dataloader,
                          desc=f"Layer-wise native LM-head LDT | {self.model_config.name}"):
            stimuli = [str(s) for s in batch['stimulus']]
            prompts = [self.build_task_input(s) for s in stimuli]
            enc = self._encode_prompts(prompts)
            ids = enc['input_ids'].to(self.device)
            am  = enc['attention_mask'].to(self.device)
            B = ids.shape[0]
            prompt_ids = [ids[b][am[b].bool()].detach().cpu().tolist() for b in range(B)]
            plens = [len(p) for p in prompt_ids]

            fp = self._final_prompt_positions(am)
            self._verify_final_positions(ids, am, fp, prompt_ids)

            outputs = self._backbone_forward(
                ids, am, (am.long().cumsum(-1) - 1).clamp(min=0))
            hs = outputs.hidden_states
            if len(hs) != self.num_layers + 1:
                raise RuntimeError(f"expected {self.num_layers + 1} hidden-state tensors, "
                                   f"got {len(hs)}")
            ar = torch.arange(B, device=hs[1].device)
            fp_d = fp.to(hs[1].device)
            ys_b = np.empty((B, self.num_layers), dtype=np.float32) if native_on else None
            ns_b = np.empty((B, self.num_layers), dtype=np.float32) if native_on else None
            for li in range(self.num_layers):
                h = hs[li + 1][ar, fp_d]                       # (B, H) — READ ONLY
                if native_on:
                    y, n = self._answer_scores(h, li)
                    ys_b[:, li] = y.cpu().numpy()
                    ns_b[:, li] = n.cpu().numpy()
                if PRIMARY_REPRESENTATION in accum:
                    accum[PRIMARY_REPRESENTATION][li].append(self._finite_numpy(h))
                if RepresentationType.MEAN_PROMPT_TOKENS.value in accum:
                    pm = am.to(hs[li + 1].device, dtype=torch.float32).unsqueeze(-1)
                    vec = (hs[li + 1].float() * pm).sum(1) / pm.sum(1).clamp(min=1.0)
                    accum[RepresentationType.MEAN_PROMPT_TOKENS.value][li].append(
                        self._finite_numpy(vec, cdt))
            if native_on:
                yes_acc.append(ys_b); no_acc.append(ns_b)
            del outputs, hs

            # ── SECONDARY (never mixed into the primary curve) ────────────
            gen = None
            if run_generation_diagnostic or gen_ctrl:
                gen = self._generate_first_tokens(stimuli)
                if gen['prompt_ids'] != prompt_ids:
                    raise RuntimeError("generation path re-tokenised the prompt "
                                       "differently from the primary forward pass")
            if gen_ctrl:
                hs2, gen_pos, prompt_mask2, _ = self._forward_with_generated_token(
                    gen['prompt_ids'], gen['gen_ids'])
                reps2 = self._read_representations(hs2, gen_pos, prompt_mask2,
                                                   [RepresentationType.GENERATED_TOKEN_CONTROL.value],
                                                   cdt)
                for li in range(self.num_layers):
                    accum[RepresentationType.GENERATED_TOKEN_CONTROL.value][li].append(
                        reps2[RepresentationType.GENERATED_TOKEN_CONTROL.value][li])
                del hs2, reps2
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            # ── gold labels joined AFTER prediction — evaluation only ─────
            gold = batch['is_word'].cpu().numpy().astype(int)
            fgrp = list(batch['freq_group'])
            final_pred = (np.where(ys_b[:, -1] > ns_b[:, -1], 'WORD', 'NONWORD')
                          if native_on else None)
            for b, s in enumerate(stimuli):
                gold_lbl = 'WORD' if gold[b] == 1 else 'NONWORD'
                row = {
                    'stimulus': s, 'gold_label': gold_lbl,
                    'gold_is_word': int(gold[b]), 'freq_group': fgrp[b],
                    'prompt_length': int(plens[b]),
                    'first_output_prediction_position': int(plens[b] - 1),
                    'first_output_prediction_position_padded': int(fp[b]),
                    'last_prompt_token_text': self.tokenizer.decode([prompt_ids[b][-1]]),
                    'representation_type': PRIMARY_REPRESENTATION,
                }
                if native_on:
                    row.update({
                        'final_layer_yes_logit': float(ys_b[b, -1]),
                        'final_layer_no_logit': float(ns_b[b, -1]),
                        'final_layer_lm_head_prediction': str(final_pred[b]),
                        'final_layer_lm_head_correct': bool(final_pred[b] == gold_lbl),
                    })
                if gen is not None:
                    gid = gen['gen_ids'][b]
                    native = self.classify_generated_token(gid)
                    yl, nl = float(gen['yes_logprob'][b]), float(gen['no_logprob'][b])
                    fc = ('WORD' if yl > nl else 'NONWORD') if np.isfinite(max(yl, nl)) else 'UNKNOWN'
                    row.update({
                        'generated_token_id': int(gid),
                        'generated_token_text': self.tokenizer.decode([gid]),
                        'generated_token_piece': self.tokenizer.convert_ids_to_tokens(int(gid)),
                        'native_prediction': native,
                        'native_correct': bool(native == gold_lbl),
                        'yes_logprob': yl, 'no_logprob': nl,
                        'forced_choice_prediction': fc,
                        'forced_choice_correct': bool(fc == gold_lbl),
                        # §22: is the final-layer readout the same decision the
                        # full model actually emits? Never forced to agree.
                        'generation_strategy': 'greedy', 'max_new_tokens': 1,
                    })
                    if native_on:
                        row['final_layer_agrees_with_generation'] = \
                            bool(final_pred[b] == native)
                    targets['native_prediction'].append(native)
                else:
                    targets['native_prediction'].append('NOT_RUN')
                diag_rows.append(row)
            gids = gen['gen_ids'] if gen is not None else [-1] * B
            targets['generated_token_id'].append(np.array(gids, dtype=np.int64))
            targets['prompt_length'].append(np.array(plens, dtype=np.int64))

            targets['is_word'].append(batch['is_word'].cpu().numpy())
            targets['is_high_freq'].append(batch['is_high_freq'].cpu().numpy())
            targets['is_low_freq'].append(batch['is_low_freq'].cpu().numpy())
            targets['freq_group'].extend(batch['freq_group'])
            targets['stimulus'].extend(batch['stimulus'])
            targets['idx'].extend(batch['idx'].cpu().numpy())
            targets['token_count'].append(batch['token_count'].cpu().numpy())
            targets['is_single_token'].append(batch['is_single_token'].cpu().numpy())
            targets['log_frequency'].append(batch['log_frequency'].cpu().numpy())
            targets['length'].append(batch['length'].cpu().numpy())
            targets['ortho_n'].append(batch['ortho_n'].cpu().numpy())

        all_reps: dict[str, dict[int, np.ndarray]] = {}
        for rt in rep_types:
            all_reps[rt] = {li: np.concatenate(accum[rt][li], axis=0)
                            for li in range(self.num_layers)}
            accum[rt] = None
        del accum
        gc.collect()

        targets = {k: (np.concatenate(v) if (len(v) and isinstance(v[0], np.ndarray)) else v)
                   for k, v in targets.items()}
        targets['native_prediction'] = np.asarray(targets['native_prediction'], dtype=object)
        native = None
        if native_on:
            native = {'yes_score': np.concatenate(yes_acc, 0),
                      'no_score': np.concatenate(no_acc, 0)}
            if not (np.isfinite(native['yes_score']).all()
                    and np.isfinite(native['no_score']).all()):
                bad = int((~np.isfinite(native['yes_score'])).sum() +
                          (~np.isfinite(native['no_score'])).sum())
                logger.error(f"  {self.model_config.name}: {bad} non-finite LM-head "
                             f"scores (fp16 overflow?); affected layers show NaN.")

        total = sum(a.nbytes for d in all_reps.values() for a in d.values())
        print(f"  ✓ Extraction complete — {len(rep_types)} representation caches "
              f"× {self.num_layers} layers ({total/1024**3:.2f} GB RAM)")
        for rt in rep_types:
            self._sanity_check(all_reps[rt], label=rt)
        return all_reps, native, targets, pd.DataFrame(diag_rows)

    # ── §18, §20, §23 — mandatory checks BEFORE the full experiment ──────
    @torch.no_grad()
    def run_prompt_sanity_checks(self, sanity_words, sanity_nonwords,
                                 freq_examples: dict,
                                 unknown_rate_threshold: float,
                                 min_accuracy: float) -> dict:
        name = self.model_config.name
        print(f"\n  {'─'*60}\n  PROMPT / POSITION / READOUT SANITY CHECKS — {name}\n  {'─'*60}")
        report = {'model': name}

        # §18 — structural no-label-leakage check
        sig_gen = list(inspect.signature(self._generate_first_tokens).parameters)
        sig_prm = list(inspect.signature(build_task_prompt).parameters)
        if sig_gen != ['stimuli'] or sig_prm != ['stimulus']:
            raise RuntimeError(f"Label-leakage check FAILED: generation signature "
                               f"{sig_gen}, prompt signature {sig_prm}")
        print(f"  [§18] prompt inputs = {sig_prm}; generation inputs = {sig_gen}  ✓ "
              f"(no gold_label / frequency_group / human_RT)")
        report['no_label_leakage_check'] = 'passed'

        # §20 — identical template across frequency groups / nonwords
        prefix, suffix = TASK_PROMPT_TEMPLATE.split("{STIMULUS}")
        for grp, w in freq_examples.items():
            p = build_task_prompt(w)
            if not (p.startswith(prefix) and p.endswith(suffix)
                    and p[len(prefix):len(p) - len(suffix)] == w):
                raise RuntimeError(f"Prompt-invariance check FAILED for {grp}:{w!r}")
            print(f"  [§20] {grp:<8} {w!r:<18} → same template, only {{STIMULUS}} differs ✓")
        report['frequency_prompt_invariance_check'] = 'passed'
        report['frequency_prompt_examples'] = dict(freq_examples)

        # ── §11 — the primary pipeline's own correctness checks ─────────
        stimuli = list(sanity_words) + list(sanity_nonwords)
        gold = ['WORD'] * len(sanity_words) + ['NONWORD'] * len(sanity_nonwords)
        prompts = [self.build_task_input(s) for s in stimuli]
        enc = self._encode_prompts(prompts)
        ids, am = enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device)
        fp = self._final_prompt_positions(am)
        self._verify_final_positions(ids, am, fp)
        o = self._backbone_forward(ids, am, (am.long().cumsum(-1) - 1).clamp(min=0))
        hs = o.hidden_states
        if len(hs) != self.num_layers + 1:
            raise RuntimeError(f"layer-count check FAILED: expected "
                               f"{self.num_layers + 1} hidden-state tensors "
                               f"(embedding + {self.num_layers} blocks), got {len(hs)}")
        ar = torch.arange(ids.shape[0], device=hs[1].device)
        fp_d = fp.to(hs[1].device)
        finite_all, var_by_layer = True, []
        for li in range(self.num_layers):
            h = hs[li + 1][ar, fp_d].float()
            finite_all &= bool(torch.isfinite(h).all())
            var_by_layer.append(float(h.var(dim=0).mean()))
        if not finite_all:
            raise RuntimeError("sanity FAILED: non-finite hidden states at the "
                               "first-output prediction position")
        # Every example's position must be its own last real prompt token.
        for b in range(ids.shape[0]):
            if int(ids[b, fp[b]]) != int(ids[b][am[b].bool()][-1]):
                raise RuntimeError(f"sample {b}: selected position is not the final "
                                   f"real prompt token")
        print(f"\n  [§11] layer count: {len(hs)} tensors = embedding + "
              f"{self.num_layers} transformer blocks ✓")
        print(f"  [§11] first-output prediction position verified per example "
              f"(padding_side={self.tokenizer.padding_side}) ✓")
        print(f"  [§11] hidden states finite at every layer ✓ "
              f"(mean feature variance layer0={var_by_layer[0]:.3e}, "
              f"last={var_by_layer[-1]:.3e})")
        print(f"  [§11] last prompt token: "
              f"{self.tokenizer.decode([int(ids[0, fp[0]])])!r}")
        report.update({
            'layer_count_check': 'passed',
            'n_hidden_state_tensors': int(len(hs)),
            'position_check': 'passed',
            'padding_side': self.tokenizer.padding_side,
            'hidden_states_finite': True,
            'mean_feature_variance_by_layer': var_by_layer,
            'last_prompt_token': self.tokenizer.decode([int(ids[0, fp[0]])]),
            'first_output_prediction_position':
                [int(x) for x in (am.sum(1) - 1).cpu().numpy()],
            'sanity_stimuli': stimuli, 'sanity_gold': gold,
        })
        del o

        # ── OPTIONAL: native YES/NO LM-head diagnostic (§9) ──────────────
        if not self.native_readout_enabled:
            print(f"  [§9] native LM-head YES/NO diagnostic disabled — the primary "
                  f"readout is tokenizer-independent.")
            report['native_lm_head_diagnostic'] = 'disabled'
        else:
            report['lm_head_pathway'] = self._verify_lm_head_pathway(stimuli)
            out = self.model(input_ids=ids, attention_mask=am,
                             output_hidden_states=True, use_cache=False)
            hs2 = out.hidden_states
            ar2 = torch.arange(ids.shape[0], device=hs2[1].device)
            layer_acc = []
            for li in range(self.num_layers):
                y, n = self._answer_scores(hs2[li + 1][ar2, fp_d], li)
                pred = np.where(y.cpu().numpy() > n.cpu().numpy(), 'WORD', 'NONWORD')
                layer_acc.append(float(np.mean(pred == np.array(gold))))
                if li == self.num_layers - 1:
                    y_fin, n_fin, pred_fin = y.cpu().numpy(), n.cpu().numpy(), pred
            print(f"\n  [§9] final-layer native LM-head readout (diagnostic only)")
            print(f"        {'stimulus':<10}| {'gold':<8}| {'YES logit':>10}| "
                  f"{'NO logit':>10}| {'pred':<8}| correct")
            rows = []
            for s, g, yv, nv, pd_ in zip(stimuli, gold, y_fin, n_fin, pred_fin):
                rows.append({'stimulus': s, 'gold': g, 'yes_logit': float(yv),
                             'no_logit': float(nv), 'predicted': str(pd_),
                             'correct': pd_ == g})
                print(f"        {s:<10}| {g:<8}| {yv:>10.3f}| {nv:>10.3f}| "
                      f"{pd_:<8}| {pd_ == g}")
            report.update({'sanity_rows': rows,
                           'sanity_lm_head_accuracy':
                               float(np.mean([r['correct'] for r in rows])),
                           'sanity_layerwise_lm_head_accuracy': layer_acc})
            del out

        # ── OPTIONAL: generation diagnostic + causality check ────────────
        if self.generation_enabled:
            s0 = stimuli[0]
            g1 = self._generate_first_tokens([s0])
            pid, gid = g1['prompt_ids'][0], g1['gen_ids'][0]
            P = len(pid)
            seq = torch.tensor([pid + [gid]], device=self.device)
            out2 = self.model(input_ids=seq, attention_mask=torch.ones_like(seq),
                              output_hidden_states=True, use_cache=False)
            h_pred_pos = out2.hidden_states[self.num_layers][0, P - 1].float()
            enc1 = self._encode_prompts([self.build_task_input(s0)])
            i1 = enc1['input_ids'].to(self.device); a1 = enc1['attention_mask'].to(self.device)
            o1 = self._backbone_forward(i1, a1)
            fp1 = int(self._final_prompt_positions(a1)[0])
            if i1.shape[1] != P or fp1 != P - 1:
                raise RuntimeError("prompt-only tokenisation/position mismatch")
            h_prompt_only = o1.hidden_states[self.num_layers][0, fp1].float()
            cos = float(F.cosine_similarity(h_prompt_only, h_pred_pos, dim=0))
            if cos < 0.999:
                raise RuntimeError(f"the first-output-prediction state differs between "
                                   f"the prompt-only and appended-token passes "
                                   f"(cos={cos:.5f}) — causal masking violated.")
            native = self.classify_generated_token(gid)
            print(f"\n  [generation diagnostic] {s0!r}: prompt length {P}, "
                  f"greedy token {self.tokenizer.decode([gid])!r} → {native}")
            print(f"        prompt-only vs appended-token state cosine: {cos:.6f} ✓")
            report.update({
                'generation_diagnostic': 'run',
                'position_check_prompt_length': P,
                'prompt_only_vs_appended_cosine': cos,
                'greedy_generated_token': self.tokenizer.decode([gid]),
                'native_generation_prediction': native,
            })
            del out2, o1
        else:
            report['generation_diagnostic'] = 'disabled'

        if self.device == 'cuda':
            torch.cuda.empty_cache()
        print(f"  {'─'*60}\n")
        return report

    def _sanity_check(self, all_hidden: dict[int, np.ndarray], label: str = ""):
        """
        Verify extracted representations are not degenerate.

        Checks every layer (not just sampled ones) because NaN
        issues are layer-specific — they typically appear only in
        the deepest layers where the residual stream magnitude is
        highest and most likely to overflow fp16.
        """
        total_nan_samples = 0
        total_inf_samples = 0
        total_zero_samples = 0
        nan_layers = []
        tag = f"[{label}] " if label else ""

        for li in range(self.num_layers):
            h = all_hidden[li]
            n_zero = int(np.sum(np.all(h == 0, axis=1)))
            n_nan  = int(np.sum(np.any(np.isnan(h), axis=1)))
            n_inf  = int(np.sum(np.any(np.isinf(h), axis=1)))

            total_nan_samples += n_nan
            total_inf_samples += n_inf
            total_zero_samples += n_zero

            finite_mask = np.all(np.isfinite(h), axis=1)
            if finite_mask.sum() > 0:
                var = np.var(h[finite_mask].astype(np.float64), axis=0).mean()
            else:
                var = np.nan

            has_problem = (n_zero > 0 or n_nan > 0 or n_inf > 0
                           or (not np.isnan(var) and var < 1e-6))
            is_sampled = li in [0, self.num_layers // 2, self.num_layers - 1]

            if has_problem:
                nan_layers.append(li)
                if n_nan > 0:
                    logger.error(f"  {tag}Layer {li}: FAIL — {n_nan} samples contained NaN")
                if n_inf > 0:
                    logger.error(f"  {tag}Layer {li}: FAIL — {n_inf} samples contain Inf!")
                if n_zero > 0:
                    logger.warning(
                        f"  {tag}Layer {li}: WARN — {n_zero} all-zero vectors "
                        f"(NaN/Inf→0 replacement or degenerate state).")
                if not np.isnan(var) and var < 1e-6:
                    logger.warning(
                        f"  {tag}Layer {li}: WARN — mean feature variance = {var:.2e} "
                        f"(suspiciously low — representations may be collapsed)")
            elif is_sampled:
                logger.info(f"  {tag}Layer {li}: OK (var={var:.4f}, "
                            f"zeros={n_zero}, nan={n_nan})")

        if total_nan_samples > 0 or total_inf_samples > 0:
            logger.warning(
                f"  {tag}SANITY SUMMARY: {total_nan_samples} NaN samples, "
                f"{total_inf_samples} Inf samples across "
                f"{len(nan_layers)} layers {nan_layers}. Affected vectors were "
                f"zeroed to prevent downstream corruption.")
        else:
            logger.info(f"  {tag}SANITY SUMMARY: All layers clean — no NaN/Inf/zero issues.")


    # ════════════════════════════════════════════════════════════════════
    # NAACL REVISION — live-model utilities (A3, C2, C3, C5, D1–D3, F1–F3)
    # ════════════════════════════════════════════════════════════════════
    def ensure_answer_readout(self) -> bool:
        """Builds the YES/NO token sets and the verified LM-head pathway on
        demand, so model-level causal analyses can read the model's OWN
        decision variable logit(YES) − logit(NO) even when the native
        diagnostic was switched off."""
        try:
            if not self.yes_token_ids or not self.no_token_ids:
                self._build_answer_token_maps()
            if self._lm_head is None:
                self._resolve_native_lm_head()
            if self._last_hidden_is_normed is None:
                self._verify_lm_head_pathway(self._cfg('SANITY_WORDS', ['house']))
            return True
        except Exception as e:                           # noqa: BLE001
            logger.warning(f"  {self.model_config.name}: YES/NO readout unavailable "
                           f"({e}); native-logit outcomes will be NaN.")
            return False

    def answer_margin(self, h_last: torch.Tensor) -> np.ndarray:
        """logit(YES) − logit(NO) from the LAST hidden state (post-final-norm
        state as returned by the backbone), i.e. the model's own decision."""
        ys, ns = self._answer_scores(h_last, self.num_layers - 1)
        return (ys - ns).float().cpu().numpy()

    def decoder_layers(self):
        for path in ('layers', 'h', 'blocks', 'decoder.layers'):
            obj = self._backbone
            try:
                for part in path.split('.'):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if isinstance(obj, nn.ModuleList) and len(obj) == self.num_layers:
                return obj
        raise RuntimeError(f"{self.model_config.name}: could not locate the decoder "
                           f"layer ModuleList on the backbone")

    @staticmethod
    def attn_module(layer):
        for n in ('self_attn', 'attn', 'attention', 'self_attention'):
            if isinstance(getattr(layer, n, None), nn.Module):
                return getattr(layer, n)
        raise RuntimeError("no attention sub-module found")

    @staticmethod
    def mlp_module(layer):
        for n in ('mlp', 'feed_forward', 'ffn'):
            if isinstance(getattr(layer, n, None), nn.Module):
                return getattr(layer, n)
        raise RuntimeError("no MLP sub-module found")

    def o_proj_module(self, attn):
        for n in ('o_proj', 'out_proj', 'dense', 'c_proj'):
            m = getattr(attn, n, None)
            if isinstance(m, nn.Linear) and m.out_features == self.hidden_dim:
                return m
        raise RuntimeError("no attention output projection found")

    def n_query_heads(self) -> int:
        return int(getattr(self.model.config, 'num_attention_heads'))

    @contextmanager
    def instrumented_attention(self):
        """Temporarily route every attention call through
        `_ldt_attention_forward` (restored on exit, also on error)."""
        if not ensure_ldt_attention_registered():
            raise RuntimeError("instrumented attention not available")
        cfgs = {id(self.model.config): self.model.config}
        bb_cfg = getattr(self._backbone, 'config', None)
        if bb_cfg is not None:
            cfgs[id(bb_cfg)] = bb_cfg
        saved = {k: getattr(c, '_attn_implementation', None) for k, c in cfgs.items()}
        try:
            for c in cfgs.values():
                c._attn_implementation = _LDT_ATTN_NAME
            yield _LDT_ATTN_CTX
        finally:
            for k, c in cfgs.items():
                c._attn_implementation = saved[k]
            _LDT_ATTN_CTX.reset()

    def encode_task_batch(self, raw_texts: list[str], char_spans=None):
        """Formats (RAW/CHAT), tokenises with padding and — when spans are
        given as character offsets into the RAW text — maps each span to its
        token positions via the fast tokenizer's offset mapping.
        Returns ids, am, fp (final prompt position), span_mask (B, T) bool."""
        formatted = [self.build_input_text(t) for t in raw_texts]
        want_off = char_spans is not None and getattr(self.tokenizer, 'is_fast', False)
        enc = self.tokenizer(formatted, return_tensors='pt', padding=True, truncation=False,
                             add_special_tokens=self._add_special_tokens(),
                             return_offsets_mapping=want_off)
        ids = enc['input_ids'].to(self.device)
        am = enc['attention_mask'].to(self.device)
        fp = self._final_prompt_positions(am)
        self._verify_final_positions(ids, am, fp)
        span_mask = None
        if char_spans is not None:
            span_mask = torch.zeros_like(am, dtype=torch.bool)
            if want_off:
                off = enc['offset_mapping']
                for b, (raw, fmt, sp) in enumerate(zip(raw_texts, formatted, char_spans)):
                    if sp is None:
                        continue
                    base = fmt.find(raw)
                    if base < 0:
                        continue
                    s0, e0 = base + sp[0], base + sp[1]
                    o = off[b]
                    real = am[b].bool().cpu()
                    hit = (o[:, 0] < e0) & (o[:, 1] > s0) & real & (o[:, 1] > o[:, 0])
                    span_mask[b] = hit.to(span_mask.device)
        return ids, am, fp, span_mask

    def backbone_hidden(self, ids, am):
        out = self._backbone_forward(ids, am, (am.long().cumsum(-1) - 1).clamp(min=0))
        hs = out.hidden_states
        if len(hs) != self.num_layers + 1:
            raise RuntimeError(f"expected {self.num_layers + 1} hidden-state tensors")
        return hs

    @torch.no_grad()
    def extract_final_and_span_states(self, raw_texts: list[str], char_spans=None,
                                      layers=None, batch_size=None, desc="states"):
        """Hidden states at the FINAL prompt position and (optionally) the
        mean over a character span's tokens, for every requested layer.
        Returns {'final': {li: (N,H)}, 'span': {li: (N,H)} | None,
                 'span_found': (N,) bool, 'n_span_tokens': (N,) int,
                 'tokens_after_span': (N,) int}."""
        layers = list(range(self.num_layers)) if layers is None else list(layers)
        bs = int(batch_size or self._cfg('MODEL_LEVEL_BATCH_SIZE', 16))
        fin = {li: [] for li in layers}
        spn = {li: [] for li in layers} if char_spans is not None else None
        found, ntok, after = [], [], []
        for s in tqdm(range(0, len(raw_texts), bs), desc=f"{desc} | {self.model_config.name}"):
            texts = raw_texts[s:s + bs]
            spans = None if char_spans is None else char_spans[s:s + bs]
            ids, am, fp, smask = self.encode_task_batch(texts, spans)
            hs = self.backbone_hidden(ids, am)
            B = ids.shape[0]
            ar = torch.arange(B, device=hs[1].device)
            fpd = fp.to(hs[1].device)
            for li in layers:
                h = hs[li + 1]
                fin[li].append(self._finite_numpy(h[ar, fpd]))
                if spn is not None:
                    m = smask.to(h.device).float().unsqueeze(-1)
                    v = (h.float() * m).sum(1) / m.sum(1).clamp(min=1.0)
                    spn[li].append(self._finite_numpy(v))
            if smask is not None:
                cnt = smask.sum(1).cpu().numpy()
                found.extend((cnt > 0).tolist()); ntok.extend(cnt.tolist())
                pos = torch.arange(ids.shape[1], device=smask.device).unsqueeze(0)
                last = torch.where(smask, pos, torch.full_like(pos, -1)).max(1).values
                after.extend((fp.to(last.device) - last).cpu().numpy().tolist())
            del hs
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        out = {'final': {li: np.concatenate(v, 0) for li, v in fin.items()},
               'span': ({li: np.concatenate(v, 0) for li, v in spn.items()}
                        if spn is not None else None)}
        if char_spans is not None:
            out['span_found'] = np.asarray(found, bool)
            out['n_span_tokens'] = np.asarray(ntok, int)
            out['tokens_after_span'] = np.asarray(after, int)
        return out

    # ── Analysis 10: word-position (non-last-token) extraction ────────
    @staticmethod
    def _locate_target_char_start(formatted: str, sentence: str, word: str,
                                  template: str | None) -> int:
        """
        Character offset of the TARGET word inside `formatted`.
        FIX (previous version used formatted.lower().find(word)): that
        returned the FIRST occurrence, so for targets that also occur in the
        carrier ("the", "was", "seen", "by", "group" in "The {} was seen by
        the group.") the span pointed at the wrong token. The target is now
        located at the template's slot position.
        """
        s0 = formatted.find(sentence)
        if s0 == -1:
            return -1
        if template and template.count('{}') == 1:
            prefix = template.split('{}')[0]
            if sentence.startswith(prefix) and \
                    sentence[len(prefix):len(prefix) + len(word)] == word:
                return s0 + len(prefix)
        m = re.search(r'\b' + re.escape(word) + r'\b', sentence)
        return (s0 + m.start()) if m else -1

    @torch.no_grad()
    def extract_word_position_layers(self, sentences: list[str],
                                       target_words: list[str],
                                       sentence_template: str | None = None
                                       ) -> tuple[dict[int, np.ndarray], list[bool]]:
        """
        CONTEXTUAL CONTROL representation (RepresentationType.
        CONTEXTUAL_TARGET_TOKEN): the target word's hidden state at its
        ACTUAL sentence position. This is NOT the first-generated-output-token
        representation and is never labelled as such. No task prompt is used.

        For each sentence, locate the target word's subword-token span
        and mean-pool the hidden state over exactly those tokens (per
        layer). This is deliberately NOT last-token pooling: the target
        word is embedded mid-sentence, so its representation must be
        read off at its own token position(s), not the sequence end.

        Uses the fast tokenizer's offset mapping to map the character
        span of the (first occurrence of the) target word in the
        formatted input text back to token indices. Requires a
        ``PreTrainedTokenizerFast`` — if unavailable, falls back to a
        substring-based token count heuristic and logs a warning, since
        exact offset mapping is not obtainable.

        Runs samples one at a time (no batching) because the target
        token span differs per sample and batching complicates masking
        for negligible speed benefit at N≈200 sentences.

        Returns
        -------
        layer_hidden : dict {layer_idx: np.ndarray (N, H)}
        found_mask   : list[bool], True if the word span was located
                       unambiguously in that sample (samples that fail
                       are zero-vectors and flagged so callers can drop
                       them rather than silently contaminating results).
        """
        is_fast = getattr(self.tokenizer, 'is_fast', False)
        if not is_fast:
            logger.warning(
                "  Tokenizer is not a fast tokenizer — word-position "
                "extraction cannot use exact offset mapping; contextual "
                "analysis results may be less precise."
            )

        layer_accum: dict[int, list[np.ndarray]] = {
            li: [] for li in range(self.num_layers)
        }
        found_mask: list[bool] = []

        for sent, word in tqdm(list(zip(sentences, target_words)),
                                desc="Contextual word-position extraction"):
            formatted = self.build_input_text(sent)
            char_start = self._locate_target_char_start(
                formatted, sent, word, sentence_template)
            found = char_start != -1 and is_fast

            enc = self.tokenizer(
                formatted, return_tensors='pt', truncation=False,
                add_special_tokens=self._add_special_tokens(),
                return_offsets_mapping=is_fast
            )
            input_ids      = enc['input_ids'].to(self.device)
            attention_mask = enc['attention_mask'].to(self.device)

            tok_start, tok_end = None, None
            if found:
                char_end = char_start + len(word)
                offsets  = enc['offset_mapping'][0].tolist()
                span = [ti for ti, (s, e) in enumerate(offsets)
                        if not (s == 0 and e == 0) and s < char_end and e > char_start]
                if span:
                    tok_start, tok_end = span[0], span[-1] + 1
                else:
                    found = False

            outputs = self._backbone_forward(input_ids, attention_mask)

            for li in range(self.num_layers):
                lh = outputs.hidden_states[li + 1][0].float()  # (T, H)
                if found:
                    vec = lh[tok_start:tok_end, :].mean(dim=0)
                else:
                    # Fallback: whole-sequence mean over non-pad tokens.
                    # Flagged via found_mask=False so it can be excluded
                    # from the contextual analysis rather than treated
                    # as a genuine word-position representation.
                    m = attention_mask[0].float().unsqueeze(-1)
                    vec = (lh * m).sum(0) / m.sum().clamp(min=1)
                arr = vec.cpu().numpy()
                nan_mask = ~np.isfinite(arr)
                if nan_mask.any():
                    arr[nan_mask] = 0.0
                layer_accum[li].append(arr)

            found_mask.append(bool(found))
            del outputs
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        layer_hidden = {li: np.stack(layer_accum[li])
                        for li in range(self.num_layers)}
        n_missed = sum(1 for f in found_mask if not f)
        if n_missed:
            logger.warning(
                f"  Contextual extraction: {n_missed}/{len(found_mask)} "
                f"target-word spans could not be located exactly "
                f"(fell back to whole-sentence mean pooling)."
            )
        return layer_hidden, found_mask


# ════════════════════════════════════════════════════════════════════════════
# PROBE CLASSIFIERS
# ════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Pre-activation residual block. He et al. (2016)."""
    def __init__(self, dim, dropout=0.3, bn_mom=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim, eps=1e-5, momentum=bn_mom)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim, eps=1e-5, momentum=bn_mom)
        self.fc2 = nn.Linear(dim, dim)
        self.dp  = nn.Dropout(dropout)
        for fc in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(fc.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(fc.bias, 0)

    def forward(self, x):
        h = F.relu(self.bn1(x));  h = self.fc1(h)
        h = self.dp(F.relu(self.bn2(h)));  h = self.fc2(h)
        return h + x


class LexicalDecisionClassifier(nn.Module):
    """
    Binary probe: Word (1) vs Non-Word (0).
    When hidden_dims=[], reduces to a linear probe (Hewitt & Liang, 2019).
    """
    def __init__(self, input_dim, hidden_dims=None,
                 dropout=0.3, use_residual=False, bn_mom=0.1):
        if hidden_dims is None:
            hidden_dims = [512, 256]
        super().__init__()
        self.use_residual = use_residual

        if not hidden_dims:
            # Linear probe
            self.input_proj = nn.Identity()
            self.layers     = nn.ModuleList()
            self.out        = nn.Linear(input_dim, 2)
            nn.init.xavier_normal_(self.out.weight, gain=1.0)
            nn.init.constant_(self.out.bias, 0)
            return

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0], eps=1e-5, momentum=bn_mom),
            nn.ReLU(), nn.Dropout(dropout * 0.5)
        )
        nn.init.kaiming_normal_(self.input_proj[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.input_proj[0].bias, 0)

        if use_residual and hidden_dims:
            self.res1 = ResidualBlock(hidden_dims[0], dropout*0.5, bn_mom)
            self.res2 = ResidualBlock(hidden_dims[0], dropout*0.5, bn_mom)

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            blk = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1], eps=1e-5, momentum=bn_mom),
                nn.ReLU(), nn.Dropout(dropout)
            )
            nn.init.kaiming_normal_(blk[0].weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(blk[0].bias, 0)
            self.layers.append(blk)

        self.out = nn.Linear(hidden_dims[-1], 2)
        nn.init.xavier_normal_(self.out.weight, gain=1.0)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)
        if self.use_residual and hasattr(self, 'res1'):
            x = self.res1(x); x = self.res2(x)
        for l in self.layers: x = l(x)
        return self.out(x)


class FocalLoss(nn.Module):
    """
    Lin et al. (2017). When gamma=0, reduces to weighted cross-entropy.
    """
    def __init__(self, alpha=None, gamma=0.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma

    def forward(self, logits, targets):
        if self.gamma == 0.0:
            weight = self.alpha if self.alpha is not None else None
            return F.cross_entropy(logits, targets, weight=weight)

        ce     = F.cross_entropy(logits, targets, reduction='none')
        log_pt = (F.log_softmax(logits, 1) * F.one_hot(targets, logits.size(1))).sum(1)
        pt     = torch.clamp(log_pt.exp(), 1e-7, 1.0)
        f      = ((1-pt)**self.gamma) * ce
        if self.alpha is not None: f = self.alpha[targets] * f
        return f.mean()


# ════════════════════════════════════════════════════════════════════════════
# METRICS HELPER
# ════════════════════════════════════════════════════════════════════════════

def _full_metrics(y_true, y_pred, y_prob, group_name) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_prob = np.asarray(y_prob)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    two_classes = len(np.unique(y_true)) == 2
    try:
        auc = roc_auc_score(y_true, y_prob) if two_classes else np.nan
    except Exception:
        auc = np.nan
    try:
        auprc = average_precision_score(y_true, y_prob) if two_classes else np.nan
    except Exception:
        auprc = np.nan
    bacc = balanced_accuracy_score(y_true, y_pred) if two_classes else np.nan
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        'group_name': group_name, 'n_samples': len(y_true),
        'accuracy': acc, 'precision': pr, 'recall': rc, 'f1': f1, 'auc': auc,
        'balanced_accuracy': bacc, 'macro_f1': macro_f1, 'auprc': auprc,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'predictions': y_pred, 'probabilities': y_prob, 'y_true': y_true,
    }


# ════════════════════════════════════════════════════════════════════════════
# FREQUENCY ANALYZER  (probe training + evaluation, memory-only)
# ════════════════════════════════════════════════════════════════════════════

class FrequencyAnalyzer:
    """
    Trains a probe classifier on the cached hidden states of each layer.
    No transformer forward pass happens here — pure numpy/pytorch training
    on CPU-RAM cached arrays.

    With the default CLASSIFIER_ARCHITECTURE=[] the probe is LINEAR LOGISTIC
    REGRESSION (see Config). The probe is an analytical instrument for
    measuring linearly recoverable information — it is NOT the model's
    native output head.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE

    @property
    def probe_type(self) -> str:
        return probe_type_name(self.config.CLASSIFIER_ARCHITECTURE)

    # ── FIX: fast GPU-standardised closed-form linear probe ────────────────
    def _train_linear_probe_fast(self, X_tr, y_tr, X_val, y_val, seed=SEED):
        """
        Drop-in replacement for the DataLoader/epoch loop WHEN the probe is
        linear (hidden_dims == []), which is the config default and the
        overwhelming majority of training calls.

        Why this is dramatically faster (100-500x in practice):
          * Standardisation done once on GPU in torch (no sklearn scaler
            re-fit per fold over 600 MB float arrays).
          * Weighted logistic regression via sklearn's LBFGS on the
            standardised arrays — converges in ~1-2 s at D=2048, N=65k.
          * No DataLoader, no per-batch Python overhead, no 50 epochs of
            mini-batch SGD with patience=10.

        Returns a shim object exposing .scaler, .eval(), .__call__(x),
        .state_dict() so the rest of the pipeline (evaluate_group,
        interventions, transfer) keeps working unchanged.
        """
        import torch as _torch
        from sklearn.linear_model import LogisticRegression

        dev = self.device
        Xt = _torch.as_tensor(X_tr, dtype=_torch.float32, device=dev)
        mu = Xt.mean(dim=0, keepdim=True)
        sd = Xt.std(dim=0, keepdim=True).clamp_min(1e-6)
        Xtr_s = ((Xt - mu) / sd).detach().cpu().numpy()
        del Xt

        Xv = _torch.as_tensor(X_val, dtype=_torch.float32, device=dev)
        Xval_s = ((Xv - mu) / sd).detach().cpu().numpy()
        del Xv

        C = 1.0 / max(self.config.CLASSIFIER_WEIGHT_DECAY, 1e-12)
        lr = LogisticRegression(
            solver="lbfgs", max_iter=300, C=C,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        lr.fit(Xtr_s, y_tr)
        del Xtr_s

        class _ScalerShim:
            __slots__ = ("mean_", "scale_")
            def __init__(self, mean_, scale_):
                self.mean_ = mean_
                self.scale_ = scale_
            def transform(self, X):
                return (np.asarray(X, dtype=np.float32) - self.mean_) / self.scale_
            def fit_transform(self, X):
                return self.transform(X)

        class _LinearProbeShim(torch.nn.Module):
            """Minimal nn.Module wrapper: exposes .scaler, .eval(), forward."""
            def __init__(self, lr_sklearn, mean_, scale_):
                super().__init__()
                self.scaler = _ScalerShim(mean_, scale_)
                coef = np.asarray(lr_sklearn.coef_, dtype=np.float32)
                inter = np.asarray(lr_sklearn.intercept_, dtype=np.float32)
                self._lin = torch.nn.Linear(coef.shape[1], coef.shape[0], bias=True)
                with torch.no_grad():
                    self._lin.weight.copy_(torch.from_numpy(coef))
                    self._lin.bias.copy_(torch.from_numpy(inter))
            def forward(self, x):
                z = self._lin(x)
                # sklearn binary LR computes p = sigmoid(z).
                # torch.softmax over [-z/2, +z/2] gives p = sigmoid(z).
                z = z * 0.5
                return _torch.cat([-z, z], dim=-1)

        # np.float32 scalars for scaler
        mu_np = mu.detach().cpu().numpy().astype(np.float32).ravel()
        sd_np = sd.detach().cpu().numpy().astype(np.float32).ravel()
        model = _LinearProbeShim(lr, mu_np, sd_np).to(dev)
        model.eval()
        return model


    def train_classifier(self, X_tr, y_tr, X_val, y_val, layer_idx, model_name,
                         hidden_dims=None, seed=SEED):
        """Train a single probe classifier with specified architecture and seed."""
        # FIX: fast path for the linear probe (default), which is 100-500x
        # faster than the mini-batch DataLoader loop and converges to the
        # same L2-regularised logistic-regression solution.
        _hd = self.config.CLASSIFIER_ARCHITECTURE if hidden_dims is None else hidden_dims
        if not _hd:
            return self._train_linear_probe_fast(X_tr, y_tr, X_val, y_val, seed=seed)
        # Reproducibility per seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        sc     = StandardScaler()
        Xtr_s  = sc.fit_transform(X_tr)
        Xval_s = sc.transform(X_val)
        Xtr_t  = torch.FloatTensor(Xtr_s); ytr_t  = torch.LongTensor(y_tr)
        Xval_t = torch.FloatTensor(Xval_s); yval_t = torch.LongTensor(y_val)

        ds   = TensorDataset(Xtr_t, ytr_t)
        bs   = min(self.config.CLASSIFIER_BATCH_SIZE,
                   max(self.config.MIN_BATCH_SIZE, len(ds)//10))
        cc   = np.bincount(y_tr)
        sw   = (1.0/cc)[y_tr]
        samp = WeightedRandomSampler(sw, len(sw), replacement=True,
                                     generator=torch.Generator().manual_seed(seed))
        # FIX: more workers + pinned memory so the tiny GPU work is not
        # starved by single-threaded batch assembly.
        _nw = 4 if (self.device == 'cuda' and len(ds) > 4096) else 0
        dl   = DataLoader(ds, batch_size=bs, sampler=samp, num_workers=_nw,
                          pin_memory=(self.device=='cuda'), drop_last=True,
                          persistent_workers=(_nw > 0))

        if hidden_dims is None:
            hidden_dims = self.config.CLASSIFIER_ARCHITECTURE

        model = LexicalDecisionClassifier(
            input_dim=X_tr.shape[1],
            hidden_dims=hidden_dims,
            dropout=self.config.CLASSIFIER_DROPOUT,
            use_residual=self.config.CLASSIFIER_USE_RESIDUAL,
            bn_mom=self.config.CLASSIFIER_BN_MOMENTUM
        ).to(self.device)

        cw   = torch.FloatTensor([len(y_tr)/(len(cc)*c) for c in cc]).to(self.device)
        crit = FocalLoss(alpha=cw, gamma=self.config.CLASSIFIER_FOCAL_GAMMA)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.CLASSIFIER_LR,
            weight_decay=self.config.CLASSIFIER_WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        best_vl, pat, best_st = float('inf'), 0, None

        for ep in range(self.config.CLASSIFIER_EPOCHS):
            model.train()
            tl = []
            for bX, by in dl:
                bX, by = bX.to(self.device), by.to(self.device)
                if self.config.CLASSIFIER_NOISE_STD > 0:
                    bX = bX + torch.randn_like(bX) * self.config.CLASSIFIER_NOISE_STD
                opt.zero_grad()
                lg = model(bX); loss = crit(lg, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl.append(loss.item())

            model.eval()
            with torch.no_grad():
                vl    = model(Xval_t.to(self.device))
                vl_loss = F.cross_entropy(vl, yval_t.to(self.device))

            sch.step(vl_loss.item())

            if vl_loss.item() < best_vl:
                best_vl = vl_loss.item(); pat = 0
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
            if pat >= self.config.CLASSIFIER_PATIENCE:
                break

        model.load_state_dict(best_st)
        model.scaler = sc
        return model

    def evaluate_group(self, clf, X, y, group_name) -> dict:
        Xs = clf.scaler.transform(X)
        clf.eval()
        with torch.no_grad():
            probs = F.softmax(clf(torch.FloatTensor(Xs).to(self.device)), 1).cpu().numpy()
        return _full_metrics(y, probs.argmax(1), probs[:,1], group_name)

    def analyze_layer(self, hidden_states, targets, layer_idx, model_name,
                      seed=SEED) -> dict:
        """Probe one layer's cached representations."""
        is_word      = targets['is_word']
        is_high_freq = targets['is_high_freq']
        is_low_freq  = targets['is_low_freq']
        freq_groups  = targets['freq_group']

        wm  = (is_word == 1)
        hfm = wm & (is_high_freq == 1)
        lfm = wm & (is_low_freq  == 1)

        if hfm.sum() < self.config.MIN_SAMPLES_PER_GROUP or \
           lfm.sum() < self.config.MIN_SAMPLES_PER_GROUP:
            logger.warning(f"Layer {layer_idx}: insufficient samples "
                           f"(high={hfm.sum()}, low={lfm.sum()})")
            return self._empty_result(layer_idx)

        strat   = ['nonword' if is_word[i]==0 else f"word_{freq_groups[i]}"
                   for i in range(len(is_word))]
        idx_all = np.arange(len(is_word))

        strat_arr = np.array(strat)
        try:
            Xtv,Xte,ytv,yte,itv,ite = train_test_split(
                hidden_states, is_word, idx_all,
                test_size=self.config.TEST_SIZE, stratify=strat,
                random_state=seed)
            stv = strat_arr[itv].tolist()
        except ValueError:
            Xtv,Xte,ytv,yte,itv,ite = train_test_split(
                hidden_states, is_word, idx_all,
                test_size=self.config.TEST_SIZE, random_state=seed)
            stv = None

        vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
        try:
            Xtr,Xval,ytr,yval,_,_ = train_test_split(
                Xtv,ytv,itv, test_size=vsz,
                stratify=(stv if stv is not None else ytv),
                random_state=seed)
        except ValueError:
            Xtr,Xval,ytr,yval,_,_ = train_test_split(
                Xtv,ytv,itv, test_size=vsz, random_state=seed)

        clf = self.train_classifier(Xtr, ytr, Xval, yval, layer_idx,
                                    model_name, seed=seed)

        overall = self.evaluate_group(clf, Xte, yte, "Test-Overall")
        # validation metrics (for validation-based layer selection only)
        _v = self.evaluate_group(clf, Xval, yval, "Validation")
        validation = {k: _v[k] for k in ('accuracy', 'balanced_accuracy', 'auc', 'n_samples')}

        hf_mask = np.isin(ite, np.where(hfm)[0])
        lf_mask = np.isin(ite, np.where(lfm)[0])

        hf_res = (self.evaluate_group(clf, Xte[hf_mask], yte[hf_mask], "Test-HighFreq")
                  if hf_mask.sum() >= self.config.MIN_TEST_SAMPLES else None)
        lf_res = (self.evaluate_group(clf, Xte[lf_mask], yte[lf_mask], "Test-LowFreq")
                  if lf_mask.sum() >= self.config.MIN_TEST_SAMPLES else None)

        freq_effect = (self._freq_stats(hf_res, lf_res)
                       if hf_res and lf_res else self._empty_freq_effect())
        if hf_res and lf_res:
            # G3 FIX: AUC needs both classes. HF and LF subsets contain words
            # only, so hf['auc'] − lf['auc'] was always NaN. The meaningful
            # contrast is AUC(HF words vs nonwords) − AUC(LF words vs
            # nonwords) on the SAME test nonwords (DeLong, shared negatives).
            p_te = overall['probabilities']
            nw_te = (yte == 0)
            freq_effect.update(delong_auc_diff_shared_negatives(
                p_te[hf_mask], p_te[lf_mask], p_te[nw_te]))
            freq_effect['auc_hf_vs_nonwords'] = freq_effect.pop('auc_a')
            freq_effect['auc_lf_vs_nonwords'] = freq_effect.pop('auc_b')

        del clf
        if self.device == 'cuda': torch.cuda.empty_cache()

        return {
            'layer': layer_idx, 'overall': overall,
            'high_frequency': hf_res, 'low_frequency': lf_res,
            'frequency_effect': freq_effect,
            'validation': validation,
            'probe_type': self.probe_type,
            # test_idx: row indices of the test items. The split depends only
            # on labels/strata and the seed, so it is IDENTICAL across layers
            # and representation types → paired (McNemar) comparisons valid.
            'split_info': {'n_train':len(Xtr),'n_val':len(Xval),'n_test':len(Xte),
                           'test_idx': np.asarray(ite)},
        }

    def _freq_stats(self, hf, lf) -> dict:
        """HF − LF word accuracy of the LDT probe. p-values are computed from
        log survival functions (never 0.0, G1); Cohen's h carries a bootstrap
        95% CI (G2) and the contrast a BIC Bayes factor (G5).
        z_statistic is now SIGNED (positive = HF more accurate)."""
        ch = (np.asarray(hf['y_true']) == np.asarray(hf['predictions'])).astype(float)
        cl = (np.asarray(lf['y_true']) == np.asarray(lf['predictions'])).astype(float)
        out = frequency_contrast(ch, cl, B=self.config.N_BOOTSTRAP, seed=SEED,
                                 bf_threshold=self.config.BF_EVIDENCE_THRESHOLD)
        out['auc_difference'] = np.nan   # filled by analyze_layer (DeLong)
        return out

    def _empty_freq_effect(self):
        return ({k: np.nan for k in ['accuracy_difference', 'high_freq_accuracy',
                                     'low_freq_accuracy', 'z_statistic',
                                     'p_value', 'p_value_log10', 'cohens_h',
                                     'cohens_h_ci95_low', 'cohens_h_ci95_high',
                                     'accuracy_difference_ci95_low',
                                     'accuracy_difference_ci95_high',
                                     'auc_difference', 'p_value_auc',
                                     'p_value_auc_log10', 'log10_bf01']} |
                {'n_high': 0, 'n_low': 0, 'correct_high': 0, 'correct_low': 0,
                 'continuity_correction': False,
                 'bf01_interpretation': 'undetermined'})

    # ════════════════════════════════════════════════════════════════════
    # J2 — CROSS-FITTED PROBES live here (previously inside the RT module)
    # ════════════════════════════════════════════════════════════════════
    def make_folds(self, y, n_folds=None, seed=None, shuffle=None) -> np.ndarray:
        """Deterministic stratified fold ids. The split depends only on the
        label vector and seed, so it is IDENTICAL across layers, probe kinds
        and representations → every comparison is paired on the same items."""
        n_folds = n_folds or self.config.PRIMARY_PROBE_N_FOLDS
        seed = self.config.PRIMARY_PROBE_SEED if seed is None else seed
        shuffle = self.config.PRIMARY_PROBE_CV_SHUFFLE if shuffle is None else shuffle
        y = np.asarray(y).astype(int)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=shuffle,
                             random_state=seed if shuffle else None)
        fold_of = np.full(len(y), -1, dtype=np.int16)
        for k, (_, te) in enumerate(cv.split(np.zeros(len(y)), y)):
            fold_of[te] = k
        return fold_of

    @staticmethod
    def _residualize(X_tr, X_te, Z_tr, Z_te):
        """B4 — linear covariate erasure fitted on the TRAINING fold only:
        X ← X − [1, Z]·B with B = argmin ‖X_tr − [1, Z_tr]B‖². After this,
        the training features have zero sample cross-covariance with Z, which
        is the necessary and sufficient condition for NO linear classifier
        to recover Z from them (linear guardedness; Belrose et al., 2023,
        LEACE, Thm. 3.1)."""
        A_tr = np.column_stack([np.ones(len(Z_tr)), Z_tr])
        A_te = np.column_stack([np.ones(len(Z_te)), Z_te])
        B, *_ = np.linalg.lstsq(A_tr, X_tr.astype(np.float64), rcond=None)
        return ((X_tr - A_tr @ B).astype(np.float32),
                (X_te - A_te @ B).astype(np.float32))

    def _linear_backend(self) -> str:
        b = self.config.PRIMARY_PROBE_BACKEND
        if b == 'auto':
            return ('torch' if (self.device == 'cuda' and torch.cuda.is_available()
                                and self.config.PRIMARY_PROBE_SOLVER != 'liblinear')
                    else 'sklearn')
        return b

    @staticmethod
    def logistic_torch_standardised(Xs, y, C=1.0, max_iter=1000):
        """Core solver on ALREADY-standardised float32 tensors → (w, b)."""
        n, d = Xs.shape
        dev = Xs.device
        w = torch.zeros(d, device=dev, requires_grad=True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        lam = 1.0 / (float(C) * n)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=int(max_iter), history_size=20,
                                tolerance_grad=1e-7, tolerance_change=1e-12,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(Xs @ w + b, y) + 0.5 * lam * (w * w).sum()
            loss.backward()
            return loss
        with torch.enable_grad():
            opt.step(closure)
        return w.detach(), b.detach()

    @staticmethod
    def fit_predict_logistic_torch(X_tr, y_tr, X_te, C=1.0, max_iter=1000, device='cpu'):
        """L2 logistic regression by full-batch L-BFGS (strong-Wolfe line
        search). Objective = sklearn's up to the positive factor 1/(C·n):
            mean_i logloss(y_i, w·x_i + b) + ‖w‖² / (2·C·n),
        with features standardised on the training fold exactly like
        StandardScaler (ddof = 0, zero-variance columns scaled by 1).
        Strictly convex in w → the same minimiser as sklearn."""
        dev = torch.device(device)
        Xt = torch.as_tensor(np.asarray(X_tr), dtype=torch.float32, device=dev)
        mu = Xt.mean(0)
        sd = Xt.std(0, unbiased=False)
        sd = torch.where(sd == 0, torch.ones_like(sd), sd)
        Xs = (Xt - mu) / sd
        del Xt
        y = torch.as_tensor(np.asarray(y_tr), dtype=torch.float32, device=dev)
        w, b = FrequencyAnalyzer.logistic_torch_standardised(Xs, y, C, max_iter)
        with torch.no_grad():
            Xe = (torch.as_tensor(np.asarray(X_te), dtype=torch.float32, device=dev) - mu) / sd
            p = torch.sigmoid(Xe @ w + b).double().cpu().numpy()
        del Xs, y, w, b
        return p

    def _fit_predict_linear(self, X_tr, y_tr, X_te, seed):
        if self._linear_backend() == 'torch':
            return self.fit_predict_logistic_torch(
                X_tr, y_tr, X_te, C=self.config.PRIMARY_PROBE_C,
                max_iter=self.config.PRIMARY_PROBE_MAX_ITER, device=self.device)
        sc = StandardScaler().fit(X_tr)
        kw = dict(C=self.config.PRIMARY_PROBE_C, solver=self.config.PRIMARY_PROBE_SOLVER,
                  max_iter=self.config.PRIMARY_PROBE_MAX_ITER, random_state=seed)
        if self.config.PRIMARY_PROBE_SOLVER == 'liblinear':
            kw['penalty'] = 'l2'
        clf = LogisticRegression(**kw)
        clf.fit(sc.transform(X_tr), y_tr)
        return clf.predict_proba(sc.transform(X_te))[:, 1]

    def _fit_predict_mlp(self, X_tr, y_tr, X_te, seed):
        """A1 — non-linear probe (LexicalDecisionClassifier with
        MLP_PROBE_HIDDEN_DIMS, optional residual blocks), trained on the GPU
        with full-tensor mini-batching (no DataLoader). Standardisation and
        the early-stopping split are fitted INSIDE the training fold only."""
        cfg = self.config
        dev = self.device
        g = torch.Generator().manual_seed(int(seed))
        torch.manual_seed(int(seed))
        try:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y_tr)), test_size=cfg.MLP_PROBE_INNER_VAL_FRAC,
                stratify=y_tr, random_state=int(seed))
        except ValueError:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y_tr)), test_size=cfg.MLP_PROBE_INNER_VAL_FRAC,
                random_state=int(seed))
        sc = StandardScaler().fit(X_tr[tr_idx])

        def _t(a):
            return torch.as_tensor(sc.transform(a), dtype=torch.float32, device=dev)
        Xa, Xv, Xt = _t(X_tr[tr_idx]), _t(X_tr[va_idx]), _t(X_te)
        ya = torch.as_tensor(y_tr[tr_idx], dtype=torch.long, device=dev)
        yv = torch.as_tensor(y_tr[va_idx], dtype=torch.long, device=dev)
        cc = np.bincount(y_tr[tr_idx], minlength=2).astype(float)
        cw = torch.as_tensor(len(tr_idx) / (2.0 * np.maximum(cc, 1.0)),
                             dtype=torch.float32, device=dev)
        model = LexicalDecisionClassifier(
            input_dim=X_tr.shape[1], hidden_dims=list(cfg.MLP_PROBE_HIDDEN_DIMS),
            dropout=cfg.CLASSIFIER_DROPOUT, use_residual=cfg.MLP_PROBE_USE_RESIDUAL,
            bn_mom=cfg.CLASSIFIER_BN_MOMENTUM).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.MLP_PROBE_LR,
                                weight_decay=cfg.MLP_PROBE_WEIGHT_DECAY)
        best, best_state, pat = np.inf, None, 0
        bs = int(min(cfg.MLP_PROBE_BATCH_SIZE, max(len(ya), 2)))
        for _ep in range(cfg.MLP_PROBE_EPOCHS):
            model.train()
            perm = torch.randperm(len(ya), generator=g).to(dev)
            for s in range(0, len(ya), bs):
                b = perm[s:s + bs]
                if len(b) < 2:          # BatchNorm needs ≥ 2 rows
                    continue
                opt.zero_grad()
                loss = F.cross_entropy(model(Xa[b]), ya[b], weight=cw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(F.cross_entropy(model(Xv), yv, weight=cw))
            if vl < best - 1e-6:
                best, pat = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= cfg.MLP_PROBE_PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(Xt), dim=1)[:, 1].float().cpu().numpy()
        del Xa, Xv, Xt, model
        if dev == 'cuda':
            torch.cuda.empty_cache()
        return p

    def crossfit_word_probabilities(self, X, y, n_folds=None, seed=None, shuffle=None,
                                    probe: str = 'linear', covariates=None,
                                    fold_of=None):
        """Out-of-fold P(class 1) for every row: each item is scored exactly
        once, by the probe fitted on the other K−1 folds.

        probe      : 'linear' (primary, L2 logistic regression) | 'mlp' (A1)
        covariates : optional (N, q) matrix; if given, the representation is
                     residualised on it inside every training fold (B4).
        fold_of    : optional precomputed fold ids (guarantees pairing).
        Returns (oof_probability, fold_of)."""
        seed = self.config.PRIMARY_PROBE_SEED if seed is None else seed
        y = np.asarray(y).astype(int)
        if fold_of is None:
            fold_of = self.make_folds(y, n_folds, seed, shuffle)
        X = np.asarray(X)
        Z = None
        if covariates is not None:
            Z = np.asarray(covariates, float).copy()
        oof = np.full(len(y), np.nan)
        for k in np.unique(fold_of):
            te = np.where(fold_of == k)[0]
            tr = np.where(fold_of != k)[0]
            X_tr = X[tr].astype(np.float32, copy=False)
            X_te = X[te].astype(np.float32, copy=False)
            if Z is not None:
                Z_tr, Z_te = Z[tr].copy(), Z[te].copy()
                med = np.nanmedian(Z_tr, axis=0)       # imputation from the training fold
                for j in range(Z.shape[1]):
                    Z_tr[np.isnan(Z_tr[:, j]), j] = med[j]
                    Z_te[np.isnan(Z_te[:, j]), j] = med[j]
                mu, sd = Z_tr.mean(0), Z_tr.std(0)
                sd[sd == 0] = 1.0
                X_tr, X_te = self._residualize(X_tr, X_te, (Z_tr - mu) / sd, (Z_te - mu) / sd)
            if probe == 'linear':
                oof[te] = self._fit_predict_linear(X_tr, y[tr], X_te, seed + int(k))
            elif probe == 'mlp':
                oof[te] = self._fit_predict_mlp(X_tr, y[tr], X_te, seed + int(k))
            else:
                raise ValueError(f"unknown probe kind {probe!r}")
        return oof, fold_of

    def _empty_result(self, li):
        return {'layer':li,'overall':None,'high_frequency':None,'low_frequency':None,
                'frequency_effect':self._empty_freq_effect(),'split_info':None,
                'validation':None,'probe_type':self.probe_type}


def _t_ci95(values) -> tuple[float, float]:
    """95% Student-t confidence interval of the mean (NaN if n < 2)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return (np.nan, np.nan)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return (float(m - h), float(m + h))


# ════════════════════════════════════════════════════════════════════════════
# STATISTICAL UTILITIES (G1, G2, G3, G5, B2, B3, H2) — pure functions, unit-tested
# ════════════════════════════════════════════════════════════════════════════
_LN10 = np.log(10.0)


def _finite(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(x))
    except TypeError:
        return False


def p_from_z(z) -> tuple[float, float]:
    """Two-sided normal p-value AND log10(p), from the log survival function,
    so an extreme statistic never underflows to p = 0.0 (G1)."""
    if not _finite(z):
        return (np.nan, np.nan)
    lp = min(0.0, float(np.log(2.0) + stats.norm.logsf(abs(float(z)))))
    return float(np.exp(lp)), lp / _LN10


def p_from_t(t, df) -> tuple[float, float]:
    """Two-sided Student-t p-value and log10(p) (G1)."""
    if not (_finite(t) and _finite(df)) or df <= 0:
        return (np.nan, np.nan)
    at, nu = abs(float(t)), float(df)
    lsf = float(stats.t.logsf(at, nu))
    if not np.isfinite(lsf):
        # Far tail: sf(t) ≈ f(t)·(ν + t²)/((ν + 1)·t), the Mills-ratio
        # asymptote implied by d log f/dt = −(ν+1)t/(ν+t²); exact as t → ∞.
        lsf = float(stats.t.logpdf(at, nu) + np.log((nu + at * at) / ((nu + 1.0) * at)))
    lp = min(0.0, float(np.log(2.0) + lsf))
    return float(np.exp(lp)), lp / _LN10


def p_from_chi2(x, df) -> tuple[float, float]:
    if not (_finite(x) and _finite(df)) or df <= 0:
        return (np.nan, np.nan)
    lp = min(0.0, float(stats.chi2.logsf(max(float(x), 0.0), float(df))))
    return float(np.exp(lp)), lp / _LN10


def p_binom_greater(k: int, n: int, p0: float) -> tuple[float, float]:
    """One-sided exact binomial P(X ≥ k | n, p0) and its log10 (G1)."""
    if n <= 0:
        return (np.nan, np.nan)
    lp = min(0.0, float(stats.binom.logsf(int(k) - 1, int(n), float(p0))))
    return float(np.exp(lp)), lp / _LN10


def p_correlation(r, n) -> tuple[float, float]:
    """p and log10(p) for a Pearson/Spearman coefficient via the t
    approximation t = r·sqrt((n−2)/(1−r²)) — the same approximation SciPy
    uses for spearmanr, but without underflow (G1)."""
    if not (_finite(r) and n is not None and n > 2):
        return (np.nan, np.nan)
    r = float(np.clip(r, -1 + 1e-15, 1 - 1e-15))
    return p_from_t(r * np.sqrt((n - 2) / (1 - r * r)), n - 2)


def log10_or_nan(p) -> float:
    return float(np.log10(p)) if _finite(p) and p > 0 else np.nan


def p_display(p, log10p) -> str:
    """Human-readable p that never prints a bare 0 (G1)."""
    if not _finite(p) and not _finite(log10p):
        return "nan"
    if _finite(log10p) and log10p < -300:
        return "< 1e-300"
    if _finite(p) and p > 0:
        return f"{p:.3g}"
    return f"1e{log10p:.1f}"


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (G6)."""
    if n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def cohens_h(p1, p2) -> float:
    """Cohen's h = 2·asin√p1 − 2·asin√p2 (Cohen, 1988)."""
    if not (_finite(p1) and _finite(p2)):
        return np.nan
    return float(2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1)))
                 - 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1))))


def bootstrap_proportion_contrast_ci(k1, n1, k2, n2, B=2000, seed=SEED,
                                     alpha=0.05) -> dict:
    """Percentile-bootstrap CIs for Cohen's h and the accuracy difference of
    two INDEPENDENT groups of 0/1 outcomes (G2, I1). Resampling n items with
    replacement from a 0/1 vector with k ones gives a Binomial(n, k/n) count,
    so the bootstrap is drawn exactly and in O(B)."""
    out = {'cohens_h_ci95_low': np.nan, 'cohens_h_ci95_high': np.nan,
           'accuracy_difference_ci95_low': np.nan,
           'accuracy_difference_ci95_high': np.nan}
    if n1 <= 0 or n2 <= 0:
        return out
    rng = np.random.default_rng(seed)
    a = rng.binomial(int(n1), k1 / n1, size=B) / n1
    b = rng.binomial(int(n2), k2 / n2, size=B) / n2
    h = 2 * np.arcsin(np.sqrt(a)) - 2 * np.arcsin(np.sqrt(b))
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    out['cohens_h_ci95_low'], out['cohens_h_ci95_high'] = map(float, np.percentile(h, q))
    d = a - b
    out['accuracy_difference_ci95_low'], out['accuracy_difference_ci95_high'] = \
        map(float, np.percentile(d, q))
    return out


def bootstrap_mean_ci(x, B=2000, seed=SEED, alpha=0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of a mean (vectorised in chunks)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    out, chunk = np.empty(B), 250
    for s in range(0, B, chunk):
        k = min(chunk, B - s)
        out[s:s + k] = x[rng.integers(0, n, size=(k, n))].mean(1)
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def two_proportion_test(k1, n1, k2, n2) -> dict:
    """Two-sided pooled two-proportion z-test (Agresti, 2002), Yates-type
    correction only when a cell is < 10 (unchanged rule), exact log10 p."""
    if n1 <= 0 or n2 <= 0:
        return {'z_statistic': np.nan, 'p_value': np.nan, 'p_value_log10': np.nan,
                'continuity_correction': False}
    a1, a2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    me = min(k1, n1 - k1, k2, n2 - k2)
    if se <= 0:
        return {'z_statistic': np.nan, 'p_value': np.nan, 'p_value_log10': np.nan,
                'continuity_correction': bool(me < 10)}
    cc = (0.5 / n1 + 0.5 / n2) if me < 10 else 0.0
    z = max(0.0, abs(a1 - a2) - cc) / se
    p, lp = p_from_z(z)
    return {'z_statistic': float(np.sign(a1 - a2) * z), 'p_value': p,
            'p_value_log10': lp, 'continuity_correction': bool(me < 10)}


def log10_bf01_from_t(t, df, n) -> float:
    """BIC approximation to the Bayes factor for H0: β = 0 of ONE coefficient
    (Wagenmakers, 2007; Raftery, 1995):
        ln BF01 ≈ ½·ln n − (n/2)·ln(1 + t²/df),
    because (1 − R²_full)/(1 − R²_reduced) = 1/(1 + t²/df). For a simple
    correlation t²/df = r²/(1−r²), giving BF01 = √n·(1 − r²)^{n/2}.
    BF01 > 1 favours the null; ≥ 3 is conventionally 'moderate' and ≥ 10
    'strong' evidence for H0 (Lee & Wagenmakers, 2013). G5."""
    if not (_finite(t) and _finite(df) and n is not None and n > 1 and df > 0):
        return np.nan
    ln_bf01 = 0.5 * np.log(n) - (n / 2.0) * np.log1p(float(t) ** 2 / float(df))
    return float(ln_bf01 / _LN10)


def log10_bf01_correlation(r, n) -> float:
    if not (_finite(r) and n is not None and n > 2):
        return np.nan
    r = float(np.clip(r, -1 + 1e-15, 1 - 1e-15))
    return log10_bf01_from_t(r * np.sqrt((n - 2) / (1 - r * r)), n - 2, n)


def log10_bf01_two_proportions(k1, n1, k2, n2) -> float:
    """BIC-approximate BF01 for H0: p1 = p2 vs H1: p1 ≠ p2 (binomial
    likelihoods, 1 vs 2 free parameters; Wagenmakers, 2007). G5."""
    if n1 <= 0 or n2 <= 0:
        return np.nan

    def _ll(k, n, p):
        p = min(max(p, 1e-300), 1 - 1e-16)
        return k * np.log(p) + (n - k) * np.log1p(-p)
    N = n1 + n2
    ll0 = _ll(k1 + k2, N, (k1 + k2) / N)
    ll1 = _ll(k1, n1, k1 / n1) + _ll(k2, n2, k2 / n2)
    bic0 = -2 * ll0 + 1 * np.log(N)
    bic1 = -2 * ll1 + 2 * np.log(N)
    return float(((bic1 - bic0) / 2.0) / _LN10)


def bf_label(log10_bf01, threshold: float = 3.0) -> str:
    if not _finite(log10_bf01):
        return 'undetermined'
    t = np.log10(threshold)
    if log10_bf01 >= t:
        return 'evidence_for_null'
    if log10_bf01 <= -t:
        return 'evidence_for_effect'
    return 'inconclusive'


def mcnemar_exact(c_a, c_b) -> tuple[int, int, float]:
    """Exact (binomial) two-sided McNemar test on paired 0/1 correctness."""
    c_a = np.asarray(c_a).astype(int)
    c_b = np.asarray(c_b).astype(int)
    if c_a.shape != c_b.shape:
        raise ValueError("McNemar needs paired vectors of equal length")
    b = int(np.sum((c_a == 1) & (c_b == 0)))
    c = int(np.sum((c_a == 0) & (c_b == 1)))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def adjust_pvalues(pvals, method: str = 'fdr_bh', alpha: float = 0.05) -> np.ndarray:
    """Multiplicity adjustment that tolerates NaNs (NaN in → NaN out)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], alpha=alpha, method=method)[1]
    return out


def _auc_structural_components(pos, neg):
    """DeLong (1988) placement values: V10[i] = P(neg < pos_i) + ½P(=),
    V01[j] = P(pos > neg_j) + ½P(=)."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    sn, sp = np.sort(neg), np.sort(pos)
    v10 = (np.searchsorted(sn, pos, 'left') + np.searchsorted(sn, pos, 'right')) / (2.0 * len(sn))
    below = np.searchsorted(sp, neg, 'left')      # positives < neg_j
    above_or_eq = len(sp) - below
    eq = np.searchsorted(sp, neg, 'right') - below
    v01 = (above_or_eq - eq + 0.5 * eq) / len(sp)
    return v10, v01


def auc_mann_whitney(pos, neg) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    v10, _ = _auc_structural_components(pos, neg)
    return float(v10.mean())


def delong_auc_diff_shared_negatives(pos_a, pos_b, neg) -> dict:
    """AUC(pos_a vs neg) − AUC(pos_b vs neg) with DeLong variance for two
    DISJOINT positive sets that share ONE negative set (G3):
        Var = Var(V10_a)/n_a + Var(V10_b)/n_b + Var(V01_a − V01_b)/n_neg.
    Used for the HF-vs-nonword vs LF-vs-nonword AUC difference (the previous
    'auc_difference' was computed on word-only subsets, where AUC is
    undefined, and was therefore always NaN)."""
    pos_a, pos_b, neg = (np.asarray(v, float) for v in (pos_a, pos_b, neg))
    if min(len(pos_a), len(pos_b)) < 2 or len(neg) < 2:
        return {'auc_a': np.nan, 'auc_b': np.nan, 'auc_difference': np.nan,
                'auc_difference_se': np.nan, 'auc_difference_z': np.nan,
                'p_value_auc': np.nan, 'p_value_auc_log10': np.nan}
    v10a, v01a = _auc_structural_components(pos_a, neg)
    v10b, v01b = _auc_structural_components(pos_b, neg)
    auc_a, auc_b = v10a.mean(), v10b.mean()
    var = (np.var(v10a, ddof=1) / len(pos_a) + np.var(v10b, ddof=1) / len(pos_b)
           + np.var(v01a - v01b, ddof=1) / len(neg))
    se = float(np.sqrt(max(var, 0.0)))
    z = (auc_a - auc_b) / se if se > 0 else np.nan
    p, lp = p_from_z(z)
    return {'auc_a': float(auc_a), 'auc_b': float(auc_b),
            'auc_difference': float(auc_a - auc_b), 'auc_difference_se': se,
            'auc_difference_z': float(z) if _finite(z) else np.nan,
            'p_value_auc': p, 'p_value_auc_log10': lp}


def frequency_contrast(correct_hf, correct_lf, B=2000, seed=SEED,
                       bf_threshold=3.0) -> dict:
    """Everything reported for a HIGH- vs LOW-frequency accuracy contrast:
    difference, two-proportion z-test (exact log10 p), Cohen's h with a
    bootstrap CI (G2), Wilson CIs per group (G6) and a BIC Bayes factor (G5).
    Cohen's h is an effect size — its test IS the two-proportion test, so no
    separate 'Cohen's h p-value' exists."""
    ch = np.asarray(correct_hf, float)
    cl = np.asarray(correct_lf, float)
    nh, nl = len(ch), len(cl)
    kh, kl = int(round(ch.sum())), int(round(cl.sum()))
    ah = kh / nh if nh else np.nan
    al = kl / nl if nl else np.nan
    out = {'high_freq_accuracy': ah, 'low_freq_accuracy': al,
           'accuracy_difference': (ah - al) if (nh and nl) else np.nan,
           'n_high': nh, 'n_low': nl, 'correct_high': kh, 'correct_low': kl,
           'cohens_h': cohens_h(ah, al) if (nh and nl) else np.nan}
    out['high_freq_accuracy_ci95_low'], out['high_freq_accuracy_ci95_high'] = wilson_ci(kh, nh)
    out['low_freq_accuracy_ci95_low'], out['low_freq_accuracy_ci95_high'] = wilson_ci(kl, nl)
    out.update(two_proportion_test(kh, nh, kl, nl))
    out.update(bootstrap_proportion_contrast_ci(kh, nh, kl, nl, B=B, seed=seed))
    lbf = log10_bf01_two_proportions(kh, nh, kl, nl)
    out['log10_bf01'] = lbf
    out['bf01_interpretation'] = bf_label(lbf, bf_threshold)
    return out


def standardized_mean_difference(x_t, x_c, sd_ref=None) -> float:
    """SMD = (mean_t − mean_c)/SD, SD = √((s²_t + s²_c)/2) unless a reference
    SD is supplied (Austin, 2009). Supplying the PRE-matching SD for both the
    before and after rows makes the change reflect mean shifts only (Stuart,
    2010). |SMD| < 0.1 is the usual balance threshold."""
    x_t = np.asarray(x_t, float); x_t = x_t[np.isfinite(x_t)]
    x_c = np.asarray(x_c, float); x_c = x_c[np.isfinite(x_c)]
    if len(x_t) < 2 or len(x_c) < 2:
        return np.nan
    sd = sd_ref if _finite(sd_ref) else np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    return float((x_t.mean() - x_c.mean()) / sd) if sd > 0 else 0.0


def variance_ratio(x_t, x_c) -> float:
    x_t = np.asarray(x_t, float); x_t = x_t[np.isfinite(x_t)]
    x_c = np.asarray(x_c, float); x_c = x_c[np.isfinite(x_c)]
    if len(x_t) < 2 or len(x_c) < 2 or x_c.var(ddof=1) == 0:
        return np.nan
    return float(x_t.var(ddof=1) / x_c.var(ddof=1))


def greedy_caliper_match(feat_t: np.ndarray, feat_c: np.ndarray,
                         caliper_sd: float, pooled_sd: np.ndarray,
                         k_neighbors: int = 50):
    """1:1 nearest-neighbour matching WITHOUT replacement, with a
    per-covariate caliper |x_t − x_c| ≤ caliper_sd · pooled_SD_j (B1).

    Deterministic, order-independent greedy rule: treated units are processed
    in increasing order of their nearest-candidate distance (standardised
    Euclidean), each taking its closest still-unused control that satisfies
    the caliper on EVERY covariate; units with none are REJECTED and counted.
    (The previous implementation claimed to sort by match distance but
    processed items in file order and had no caliper.)

    Returns (idx_t, idx_c, n_rejected) as positions into feat_t / feat_c.
    """
    feat_t = np.asarray(feat_t, float)
    feat_c = np.asarray(feat_c, float)
    sd = np.where(np.asarray(pooled_sd, float) > 0, pooled_sd, 1.0)
    zt, zc = feat_t / sd, feat_c / sd
    if len(zt) == 0 or len(zc) == 0:
        return np.array([], int), np.array([], int), len(zt)
    k = int(min(max(k_neighbors, 1), len(zc)))
    nn_ = NearestNeighbors(n_neighbors=k).fit(zc)
    dist, ind = nn_.kneighbors(zt)
    order = np.lexsort((np.arange(len(zt)), dist[:, 0]))
    used = np.zeros(len(zc), bool)
    mt, mc, rejected = [], [], 0
    tol = caliper_sd + 1e-12
    for i in order:
        chosen = -1
        for j in ind[i]:
            if not used[j] and np.all(np.abs(zt[i] - zc[j]) <= tol):
                chosen = j
                break
        if chosen < 0 and k < len(zc):
            ok = (~used) & np.all(np.abs(zc - zt[i]) <= tol, axis=1)
            if ok.any():
                cand = np.where(ok)[0]
                chosen = int(cand[np.argmin(((zc[cand] - zt[i]) ** 2).sum(1))])
        if chosen < 0:
            rejected += 1
            continue
        used[chosen] = True
        mt.append(int(i)); mc.append(int(chosen))
    return np.asarray(mt, int), np.asarray(mc, int), int(rejected)


def centered_gram(X: np.ndarray) -> np.ndarray:
    """Gram matrix of column-centred features (float64), for linear CKA."""
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(0, keepdims=True)
    return Xc @ Xc.T


def linear_cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019) from column-centred Gram matrices:
    CKA = <K, L>_F / (‖K‖_F ‖L‖_F). Invariant to orthogonal transforms and
    isotropic scaling, so it compares models of different width (H2)."""
    num = float(np.sum(K * L))
    den = float(np.sqrt(np.sum(K * K) * np.sum(L * L)))
    return num / den if den > 0 else np.nan


def representative_layers(all_layers) -> list[int]:
    """first / quartiles / last / every 4th layer — the fixed sampling rule
    shared by analyses 4, 6, 8, 9 and the NAACL-revision analyses."""
    all_layers = sorted(all_layers)
    n = len(all_layers)
    if n == 0:
        return []
    return sorted(set([all_layers[0], all_layers[n // 4], all_layers[n // 2],
                       all_layers[3 * n // 4], all_layers[-1]] + all_layers[::4]))


def select_layers(num_layers: int, mode: str, cap: int | None = None) -> list[int]:
    layers = list(range(num_layers)) if mode == 'all' else representative_layers(range(num_layers))
    if cap is not None and len(layers) > cap:
        pick = np.unique(np.round(np.linspace(0, len(layers) - 1, cap)).astype(int))
        layers = [layers[i] for i in pick]
    return layers


def ols_hc3(y, X_df: pd.DataFrame, term: str) -> dict:
    """OLS with classical AND HC3 heteroskedasticity-robust SEs (MacKinnon &
    White, 1985) for one term, with exact log10 p and a BIC Bayes factor."""
    res = {'beta': np.nan, 'se': np.nan, 't': np.nan, 'p': np.nan, 'p_log10': np.nan,
           'se_hc3': np.nan, 'p_hc3': np.nan, 'p_hc3_log10': np.nan,
           'r2': np.nan, 'adj_r2': np.nan, 'n': int(len(X_df)), 'log10_bf01': np.nan}
    try:
        Xc = sm.add_constant(X_df.astype(float), has_constant='add')
        fit = sm.OLS(np.asarray(y, float), Xc).fit()
        rob = fit.get_robustcov_results(cov_type='HC3')
        j = list(Xc.columns).index(term)
        t = float(fit.tvalues.iloc[j])
        res.update({'beta': float(fit.params.iloc[j]), 'se': float(fit.bse.iloc[j]), 't': t,
                    'r2': float(fit.rsquared), 'adj_r2': float(fit.rsquared_adj)})
        res['p'], res['p_log10'] = p_from_t(t, fit.df_resid)
        res['se_hc3'] = float(np.asarray(rob.bse)[j])
        t_hc3 = float(np.asarray(rob.tvalues)[j])
        res['p_hc3'], res['p_hc3_log10'] = p_from_t(t_hc3, fit.df_resid)
        res['log10_bf01'] = log10_bf01_from_t(t, fit.df_resid, int(fit.nobs))
    except Exception as e:                                 # noqa: BLE001
        res['error'] = str(e)
    return res


def apply_frequency_fdr(layer_results: list[dict], alpha: float = 0.05):
    """BH-FDR across layers for the HF-vs-LF accuracy test AND the DeLong
    AUC-difference test of hold-out layer results (J3, G3). Must run before
    any table reads `p_value_corrected` — called by every producer of
    layer_results (main hold-out probe, tokenization controls, controls)."""
    for key, out in (('p_value', 'p_value_corrected'), ('p_value_auc', 'p_value_auc_corrected')):
        pv = [r['frequency_effect'].get(key, np.nan) for r in layer_results]
        adj = adjust_pvalues(pv, 'fdr_bh', alpha)
        for r, v in zip(layer_results, adj):
            if np.isfinite(v):
                r['frequency_effect'][out] = float(v)
                if key == 'p_value':
                    r['frequency_effect']['significant_fdr'] = bool(v < alpha)
                    # Cohen's h is tested by the same two-proportion test (G3)
                    r['frequency_effect']['p_value_cohens_h_corrected'] = float(v)


# ════════════════════════════════════════════════════════════════════════════
# EXTENDED ANALYSES (Document Requirements 1–7)
# ════════════════════════════════════════════════════════════════════════════

class ExtendedAnalyses:
    """
    Implements all seven analyses required to make the frequency claim
    defensible. Each method operates on pre-extracted hidden states (numpy
    arrays cached in CPU RAM) — no additional GPU forward passes needed.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE

    # ────────────────────────────────────────────────────────────────────
    # Analysis 1: Direct Frequency Probe
    # ────────────────────────────────────────────────────────────────────
    def direct_frequency_probe(self, all_hidden, targets, model_name) -> dict:
        """
        Train a WORDS-ONLY probe: high-freq vs low-freq (binary).
        This is the single most important analysis because the original
        pipeline infers frequency sensitivity indirectly from accuracy
        gaps in a word-vs-nonword probe. Here we directly classify
        frequency from hidden states.

        Unlike the LDT probe (word vs nonword), this probe trains only
        on word stimuli and the label is the frequency group (high=1,
        low=0). This provides direct evidence that the representation
        encodes frequency information, not just lexical status.

        Also runs an optional 3-way classification (high/mid/low) as
        secondary evidence.
        """
        logger.info(f"  [Analysis 1] Direct frequency probe — {model_name}")

        is_word      = targets['is_word']
        is_high_freq = targets['is_high_freq']
        is_low_freq  = targets['is_low_freq']
        freq_groups  = targets['freq_group']

        # Binary: high vs low words only
        wm  = (is_word == 1)
        hfm = wm & (is_high_freq == 1)
        lfm = wm & (is_low_freq  == 1)
        binary_mask = hfm | lfm

        if binary_mask.sum() < 2 * self.config.MIN_SAMPLES_PER_GROUP:
            logger.warning("  Insufficient samples for direct frequency probe")
            return {'binary': [], 'three_way': []}

        # Labels: 1=high, 0=low
        freq_labels = np.zeros(len(is_word), dtype=int)
        freq_labels[hfm] = 1

        # 3-way: high=2, mid=1, low=0 (words only with frequency data)
        mid_mask = wm & (~hfm) & (~lfm) & (np.array(
            [g == 'mid' for g in freq_groups]))
        three_way_mask = hfm | lfm | mid_mask
        three_way_labels = np.zeros(len(is_word), dtype=int)
        three_way_labels[hfm] = 2
        three_way_labels[mid_mask] = 1
        # low stays 0

        binary_results = []
        three_way_results = []
        analyzer = FrequencyAnalyzer(self.config)

        for li in tqdm(range(len(all_hidden)), desc="Direct freq probe"):
            if li not in all_hidden:
                continue

            # ── Binary probe ──────────────────────────────────────────
            X_bin = all_hidden[li][binary_mask]
            y_bin = freq_labels[binary_mask]

            if len(np.unique(y_bin)) < 2:
                binary_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1': np.nan,
                    'auc': np.nan, 'n_high': hfm.sum(), 'n_low': lfm.sum()
                })
                continue

            try:
                strat_bin = y_bin
                Xtv, Xte, ytv, yte = train_test_split(
                    X_bin, y_bin, test_size=self.config.TEST_SIZE,
                    stratify=strat_bin, random_state=SEED)
                vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                Xtr, Xval, ytr, yval = train_test_split(
                    Xtv, ytv, test_size=vsz, stratify=ytv, random_state=SEED)

                clf = analyzer.train_classifier(
                    Xtr, ytr, Xval, yval, li, model_name)
                res = analyzer.evaluate_group(clf, Xte, yte, "FreqBinary")

                k_ = int(round(res['accuracy'] * len(yte)))
                lo_, hi_ = wilson_ci(k_, len(yte))
                binary_results.append({
                    'layer': li, 'accuracy': res['accuracy'],
                    'accuracy_ci95_low': lo_, 'accuracy_ci95_high': hi_,
                    'precision': res['precision'], 'recall': res['recall'],
                    'f1': res['f1'], 'auc': res['auc'], 'n_test': int(len(yte)),
                    'n_high': hfm.sum(), 'n_low': lfm.sum(),
                })
                del clf
            except Exception as e:
                logger.warning(f"  Binary freq probe layer {li} failed: {e}")
                binary_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1': np.nan,
                    'auc': np.nan, 'n_high': hfm.sum(), 'n_low': lfm.sum()
                })

            # ── 3-way probe ───────────────────────────────────────────
            X_3w = all_hidden[li][three_way_mask]
            y_3w = three_way_labels[three_way_mask]

            if len(np.unique(y_3w)) < 3:
                three_way_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1_macro': np.nan
                })
                continue

            try:
                sc = StandardScaler()
                Xtv3, Xte3, ytv3, yte3 = train_test_split(
                    X_3w, y_3w, test_size=self.config.TEST_SIZE,
                    stratify=y_3w, random_state=SEED)
                vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                Xtr3, _Xval3, ytr3, _yval3 = train_test_split(
                    Xtv3, ytv3, test_size=vsz, stratify=ytv3, random_state=SEED)

                Xtr3_s = sc.fit_transform(Xtr3)
                Xte3_s = sc.transform(Xte3)

                lr3 = LogisticRegression(
                    max_iter=1000,  # lbfgs ⇒ multinomial (sklearn ≥1.5 default; 'multi_class' removed in 1.8)
                    solver='lbfgs', random_state=SEED, C=1.0)
                lr3.fit(Xtr3_s, ytr3)
                y3_pred = lr3.predict(Xte3_s)
                acc3 = accuracy_score(yte3, y3_pred)
                _, _, f1_3, _ = precision_recall_fscore_support(
                    yte3, y3_pred, average='macro', zero_division=0)

                lo3, hi3 = wilson_ci(int(round(acc3 * len(yte3))), len(yte3))
                three_way_results.append({
                    'layer': li, 'accuracy': acc3, 'f1_macro': f1_3,
                    'accuracy_ci95_low': lo3, 'accuracy_ci95_high': hi3,
                    'n_test': int(len(yte3)), 'chance': 1.0 / 3.0,
                })
            except Exception as e:
                logger.warning(f"  3-way freq probe layer {li} failed: {e}")
                three_way_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1_macro': np.nan
                })

        return {'binary': binary_results, 'three_way': three_way_results}

    # ────────────────────────────────────────────────────────────────────
    # Analysis 2: Tokenization-Controlled Rerun
    # ────────────────────────────────────────────────────────────────────
    def tokenization_controlled_analysis(self, all_hidden, targets,
                                          model_name) -> dict:
        """
        Control for tokenization confounds by:
        1. Running the LDT probe on SINGLE-TOKEN stimuli only.
        2. Running on token-count-matched high vs low subsets.

        Rationale: High-frequency words tend to be single subword tokens
        while low-frequency words are split into multiple subwords. Even
        though the representation is now read at the first generated
        output token (which attends to all stimulus sub-tokens), the
        number of sub-tokens the model must integrate is itself a
        systematic correlate of frequency. Token counts are computed on
        the stimulus string with the model's tokenizer (unchanged).
        """
        logger.info(f"  [Analysis 2] Tokenization-controlled — {model_name}")

        is_word        = targets['is_word']
        is_high_freq   = targets['is_high_freq']
        is_low_freq    = targets['is_low_freq']
        targets['freq_group']
        is_single_tok  = targets['is_single_token']
        token_counts   = targets['token_count']
        _freq_groups   = targets['freq_group']  # kept for completeness, not used here

        wm  = (is_word == 1)
        hfm = wm & (is_high_freq == 1)
        lfm = wm & (is_low_freq  == 1)

        # ── Stats table ───────────────────────────────────────────────
        hf_single = (hfm & (is_single_tok == 1)).sum()
        lf_single = (lfm & (is_single_tok == 1)).sum()
        hf_tc_mean = token_counts[hfm].mean() if hfm.sum() > 0 else np.nan
        lf_tc_mean = token_counts[lfm].mean() if lfm.sum() > 0 else np.nan

        token_stats = {
            'high_freq_total': int(hfm.sum()),
            'low_freq_total': int(lfm.sum()),
            'high_freq_single_token': int(hf_single),
            'low_freq_single_token': int(lf_single),
            'high_freq_pct_single': float(hf_single / max(hfm.sum(), 1) * 100),
            'low_freq_pct_single': float(lf_single / max(lfm.sum(), 1) * 100),
            'high_freq_mean_tokens': float(hf_tc_mean),
            'low_freq_mean_tokens': float(lf_tc_mean),
        }
        logger.info(f"  Token stats: HF single={hf_single}/{hfm.sum()}, "
                    f"LF single={lf_single}/{lfm.sum()}")

        analyzer = FrequencyAnalyzer(self.config)

        # ── Single-token-only subset ──────────────────────────────────
        single_mask = (is_single_tok == 1)
        single_ldt_results = []

        if single_mask.sum() >= 2 * self.config.MIN_SAMPLES_PER_GROUP:
            for li in tqdm(range(len(all_hidden)),
                           desc="Single-token LDT"):
                if li not in all_hidden:
                    continue
                # Build targets subset
                tgt_sub = {k: (v[single_mask] if isinstance(v, np.ndarray) else
                               [v[i] for i in range(len(v)) if single_mask[i]])
                           for k, v in targets.items()}
                try:
                    res = analyzer.analyze_layer(
                        all_hidden[li][single_mask], tgt_sub, li, model_name)
                    single_ldt_results.append(res)
                except Exception as e:
                    logger.warning(f"  Single-token layer {li} failed: {e}")
                    single_ldt_results.append(analyzer._empty_result(li))
        else:
            logger.warning("  Not enough single-token stimuli for subset analysis")

        # ── Token-count-matched high vs low ───────────────────────────
        matched_results = []
        # Match by finding the common token counts between high & low
        hf_indices = np.where(hfm)[0]
        lf_indices = np.where(lfm)[0]
        hf_tcounts = token_counts[hf_indices]
        lf_tcounts = token_counts[lf_indices]

        # For each token count, take min(n_high, n_low) from each group
        matched_hf_idx = []
        matched_lf_idx = []
        for tc in np.unique(np.concatenate([hf_tcounts, lf_tcounts])):
            hf_with_tc = hf_indices[hf_tcounts == tc]
            lf_with_tc = lf_indices[lf_tcounts == tc]
            n_match = min(len(hf_with_tc), len(lf_with_tc))
            if n_match > 0:
                rng = np.random.RandomState(SEED)
                matched_hf_idx.extend(rng.choice(hf_with_tc, n_match, replace=False))
                matched_lf_idx.extend(rng.choice(lf_with_tc, n_match, replace=False))

        if len(matched_hf_idx) >= self.config.MIN_SAMPLES_PER_GROUP:
            matched_word_idx = np.array(matched_hf_idx + matched_lf_idx)
            # Also include nonwords to maintain LDT framing
            nw_idx = np.where(is_word == 0)[0]
            matched_all_idx = np.concatenate([matched_word_idx, nw_idx])

            for li in tqdm(range(len(all_hidden)),
                           desc="Token-matched LDT"):
                if li not in all_hidden:
                    continue
                tgt_sub = {k: (v[matched_all_idx] if isinstance(v, np.ndarray) else
                               [v[i] for i in matched_all_idx])
                           for k, v in targets.items()}
                try:
                    res = analyzer.analyze_layer(
                        all_hidden[li][matched_all_idx], tgt_sub, li, model_name)
                    matched_results.append(res)
                except Exception as e:
                    logger.warning(f"  Token-matched layer {li} failed: {e}")
                    matched_results.append(analyzer._empty_result(li))

            logger.info(f"  Token-matched: {len(matched_hf_idx)} high, "
                        f"{len(matched_lf_idx)} low (matched)")
        else:
            logger.warning("  Not enough token-count-matched samples")
        # J3 FIX: these subsets never received FDR correction before, so
        # every downstream 'freq_p_fdr' column for them was NaN.
        apply_frequency_fdr(single_ldt_results, self.config.ALPHA)
        apply_frequency_fdr(matched_results, self.config.ALPHA)

        return {
            'token_stats': token_stats,
            'single_token_results': single_ldt_results,
            'token_matched_results': matched_results,
        }

    # ────────────────────────────────────────────────────────────────────
    # Analysis 3: Lexical-Confound-Matched Frequency Comparisons
    # ────────────────────────────────────────────────────────────────────
    def confound_matched_analysis(self, all_hidden, targets,
                                   model_name) -> dict:
        """
        B1–B3. Match high- and low-frequency words 1:1 on character length,
        orthographic neighbourhood density (Ortho_N) and sub-word token
        count, then rerun the direct frequency probe on the matched set.

        Matching: greedy nearest-neighbour WITHOUT replacement in
        standardised covariate space, processed in order of best-match
        distance, with a PER-COVARIATE caliper of MATCH_CALIPER_SD pooled
        SDs (Austin, 2011). Items with no admissible partner are rejected and
        counted (`n_rejected_by_caliper`).

        Balance diagnostics per covariate, before and after matching:
          * standardised mean difference, denominator = PRE-matching pooled
            SD for both rows (Stuart, 2010); |SMD| < SMD_BALANCE_THRESHOLD;
          * variance ratio (acceptable ≈ 0.5–2; Rubin, 2001);
          * Kolmogorov–Smirnov and Welch t-tests. CAUTION (reported, not
            hidden): hypothesis tests are NOT valid balance criteria — their
            p-values depend on the matched sample size (Imai, King & Stuart,
            2008, 'balance test fallacy'); SMD is the pre-specified criterion.
            KS is also conservative for discrete covariates (token count).
        Ortho_N missing values are imputed with the group-pooled median
        before matching (count reported).
        """
        logger.info(f"  [Analysis 3 / B1–B3] Confound-matched frequency — {model_name}")
        cfg = self.config
        is_word      = targets['is_word']
        is_high_freq = targets['is_high_freq']
        is_low_freq  = targets['is_low_freq']
        cov = {'length': np.asarray(targets['length'], float),
               'ortho_n': np.asarray(targets['ortho_n'], float),
               'token_count': np.asarray(targets['token_count'], float)}
        wm  = (is_word == 1)
        hf_idx = np.where(wm & (is_high_freq == 1))[0]
        lf_idx = np.where(wm & (is_low_freq == 1))[0]
        both = np.concatenate([hf_idx, lf_idx])
        n_imputed = {}
        feats = {}
        for c, v in cov.items():
            v = v.copy()
            miss = np.isnan(v[both])
            n_imputed[c] = int(miss.sum())
            med = np.nanmedian(v[both]) if np.any(~miss) else 0.0
            v[np.isnan(v)] = med
            feats[c] = v
        names = list(cov)
        F_hf = np.column_stack([feats[c][hf_idx] for c in names])
        F_lf = np.column_stack([feats[c][lf_idx] for c in names])
        pooled_sd = np.sqrt((F_hf.var(0, ddof=1) + F_lf.var(0, ddof=1)) / 2.0)
        # low-frequency words are the scarcer, 'treated' group matched to HF
        mt, mc, n_rej = greedy_caliper_match(F_lf, F_hf, cfg.MATCH_CALIPER_SD, pooled_sd)
        matched_lf = lf_idx[mt]
        matched_hf = hf_idx[mc]

        def _balance(hi, lo, stage):
            rows_ = []
            for j, c in enumerate(names):
                a, b = cov[c][hi], cov[c][lo]
                a_, b_ = a[np.isfinite(a)], b[np.isfinite(b)]
                row = {'stage': stage, 'covariate': c, 'n_high': int(len(hi)),
                       'n_low': int(len(lo)),
                       'mean_high': float(np.mean(a_)) if len(a_) else np.nan,
                       'mean_low': float(np.mean(b_)) if len(b_) else np.nan,
                       'sd_high': float(np.std(a_, ddof=1)) if len(a_) > 1 else np.nan,
                       'sd_low': float(np.std(b_, ddof=1)) if len(b_) > 1 else np.nan,
                       'smd': standardized_mean_difference(a_, b_, sd_ref=pooled_sd[j]),
                       'smd_own_pooled_sd': standardized_mean_difference(a_, b_),
                       'variance_ratio': variance_ratio(a_, b_),
                       'n_missing_imputed_for_matching': n_imputed[c]}
                row['balanced_smd'] = bool(_finite(row['smd']) and
                                           abs(row['smd']) < cfg.SMD_BALANCE_THRESHOLD)
                if len(a_) > 1 and len(b_) > 1:
                    ks = stats.ks_2samp(a_, b_)
                    tt = stats.ttest_ind(a_, b_, equal_var=False)
                    row.update({'ks_statistic': float(ks.statistic), 'ks_p': float(ks.pvalue),
                                'welch_t': float(tt.statistic), 'welch_t_p': float(tt.pvalue)})
                rows_.append(row)
            return rows_

        balance_rows = _balance(hf_idx, lf_idx, 'before') + \
            (_balance(matched_hf, matched_lf, 'after') if len(matched_hf) else [])

        def _group_stats(idx, label):
            return {
                'group': label, 'n': len(idx),
                'length_mean': float(np.nanmean(cov['length'][idx])) if len(idx) else np.nan,
                'length_std':  float(np.nanstd(cov['length'][idx])) if len(idx) else np.nan,
                'ortho_n_mean': float(np.nanmean(cov['ortho_n'][idx])) if len(idx) else np.nan,
                'ortho_n_std':  float(np.nanstd(cov['ortho_n'][idx])) if len(idx) else np.nan,
                'token_count_mean': float(np.mean(cov['token_count'][idx])) if len(idx) else np.nan,
                'token_count_std':  float(np.std(cov['token_count'][idx])) if len(idx) else np.nan,
            }
        balance_before = [_group_stats(hf_idx, 'high_freq_before'),
                          _group_stats(lf_idx, 'low_freq_before')]
        balance_after = [_group_stats(matched_hf, 'high_freq_after'),
                         _group_stats(matched_lf, 'low_freq_after')]
        all_balanced = bool(len(matched_hf) and all(
            r['balanced_smd'] for r in balance_rows if r['stage'] == 'after'))
        logger.info(f"  Matched pairs: {len(matched_hf)} | rejected by caliper: {n_rej} "
                    f"| all |SMD| < {cfg.SMD_BALANCE_THRESHOLD}: {all_balanced}")

        # ── direct frequency probe on the matched subset (cross-fitted) ──
        matched_probe_results = []
        if len(matched_hf) >= cfg.MIN_SAMPLES_PER_GROUP:
            matched_all = np.concatenate([matched_hf, matched_lf])
            y_m = np.concatenate([np.ones(len(matched_hf), int), np.zeros(len(matched_lf), int)])
            fa = FrequencyAnalyzer(cfg)
            folds = fa.make_folds(y_m)
            for li in tqdm(sorted(all_hidden.keys()), desc="Confound-matched freq probe"):
                try:
                    p, _ = fa.crossfit_word_probabilities(all_hidden[li][matched_all], y_m,
                                                          fold_of=folds)
                    pred = (p >= 0.5).astype(int)
                    k = int((pred == y_m).sum())
                    lo, hi = wilson_ci(k, len(y_m))
                    pb, pbl = p_binom_greater(k, len(y_m), 0.5)
                    matched_probe_results.append({
                        'layer': li, 'accuracy': k / len(y_m),
                        'accuracy_ci95_low': lo, 'accuracy_ci95_high': hi,
                        'f1': float(f1_score(y_m, pred, zero_division=0)),
                        'auc': float(roc_auc_score(y_m, p)),
                        'p_vs_chance': pb, 'p_vs_chance_log10': pbl,
                        'n_matched_pairs': int(len(matched_hf)),
                        'scoring': 'out_of_fold_only'})
                except Exception as e:
                    logger.warning(f"  Confound-matched layer {li} failed: {e}")
                    matched_probe_results.append({'layer': li, 'accuracy': np.nan,
                                                  'f1': np.nan, 'auc': np.nan})
            adj = adjust_pvalues([r.get('p_vs_chance', np.nan) for r in matched_probe_results])
            for r, v in zip(matched_probe_results, adj):
                r['p_vs_chance_fdr_bh'] = float(v) if np.isfinite(v) else np.nan
        else:
            logger.warning("  Not enough matched pairs for confound-matched analysis")

        return {
            'balance_before': balance_before,
            'balance_after': balance_after,
            'balance_diagnostics': balance_rows,
            'n_matched_pairs': int(len(matched_hf)),
            'n_rejected_by_caliper': int(n_rej),
            'caliper_sd': float(cfg.MATCH_CALIPER_SD),
            'all_covariates_balanced_after_matching': all_balanced,
            'matched_probe_results': matched_probe_results,
            'matched_hf_idx': matched_hf, 'matched_lf_idx': matched_lf,
        }

    # ────────────────────────────────────────────────────────────────────
    # Analysis 4: Multi-Seed Stability
    # ────────────────────────────────────────────────────────────────────
    def multi_seed_stability(self, all_hidden, targets, model_name,
                              n_seeds: int = 5) -> dict:
        """
        Repeat the main LDT probe pipeline across multiple random seeds,
        using different train/val/test splits and weight initialisations.
        Report mean ± std per layer.

        This addresses the concern that results from a single 70/15/15
        split may be unstable. Seeds control:
          1. train_test_split random_state
          2. PyTorch weight initialisation (torch.manual_seed)
          3. WeightedRandomSampler ordering

        The reported confidence interval width directly indicates whether
        the frequency effect is robust to data sampling variance.
        """
        logger.info(f"  [Analysis 4] Multi-seed stability ({n_seeds} seeds) "
                    f"— {model_name}")

        analyzer = FrequencyAnalyzer(self.config)
        seeds = [SEED + i * 7 for i in range(n_seeds)]  # Deterministic seed sequence

        # Collect per-seed, per-layer results
        # We only probe a representative subset of layers to save time:
        # first, middle, last, and every 4th layer
        all_layers = sorted(all_hidden.keys())
        probe_layers = sorted(set(
            [all_layers[0], all_layers[len(all_layers)//4],
             all_layers[len(all_layers)//2],
             all_layers[3*len(all_layers)//4],
             all_layers[-1]]
            + all_layers[::4]
        ))

        seed_results = {s: [] for s in seeds}

        for seed_i, seed in enumerate(seeds):
            logger.info(f"  Seed {seed_i+1}/{n_seeds} (seed={seed})")
            for li in tqdm(probe_layers,
                           desc=f"Seed {seed_i+1}/{n_seeds}"):
                if li not in all_hidden:
                    continue
                res = analyzer.analyze_layer(
                    all_hidden[li], targets, li, model_name, seed=seed)
                seed_results[seed].append(res)

        # Aggregate: mean ± std per layer
        layer_stats = []
        for layer_pos, li in enumerate(probe_layers):
            accs = []
            f1s = []
            freq_diffs = []
            cohens_hs = []

            for seed in seeds:
                if layer_pos < len(seed_results[seed]):
                    r = seed_results[seed][layer_pos]
                    if r.get('overall'):
                        accs.append(r['overall']['accuracy'])
                        f1s.append(r['overall']['f1'])
                    fe = r['frequency_effect']
                    if not np.isnan(fe['accuracy_difference']):
                        freq_diffs.append(fe['accuracy_difference'])
                    if not np.isnan(fe.get('cohens_h', np.nan)):
                        cohens_hs.append(fe['cohens_h'])

            layer_stats.append({
                'layer': li,
                'acc_mean': float(np.mean(accs)) if accs else np.nan,
                'acc_std':  float(np.std(accs))  if accs else np.nan,
                'f1_mean':  float(np.mean(f1s))  if f1s  else np.nan,
                'f1_std':   float(np.std(f1s))   if f1s  else np.nan,
                'freq_diff_mean': float(np.mean(freq_diffs)) if freq_diffs else np.nan,
                'freq_diff_std':  float(np.std(freq_diffs))  if freq_diffs else np.nan,
                'cohens_h_mean':  float(np.mean(cohens_hs))  if cohens_hs  else np.nan,
                'cohens_h_std':   float(np.std(cohens_hs))   if cohens_hs  else np.nan,
                # 95% t-interval of the mean across seeds (df = n_seeds-1)
                'acc_ci95_low':  _t_ci95(accs)[0],
                'acc_ci95_high': _t_ci95(accs)[1],
                'freq_diff_ci95_low':  _t_ci95(freq_diffs)[0],
                'freq_diff_ci95_high': _t_ci95(freq_diffs)[1],
                'n_seeds': len(accs),
                'representation_type': PRIMARY_REPRESENTATION,
                'probe_type': analyzer.probe_type,
            })

        return {
            'n_seeds': n_seeds,
            'seeds': seeds,
            'layer_stats': layer_stats,
        }

    # ────────────────────────────────────────────────────────────────────
    # Analysis 6: Probe Selectivity Controls
    # ────────────────────────────────────────────────────────────────────
    def probe_selectivity_controls(self, all_hidden, targets, model_name) -> dict:
        """
        Three control conditions to establish probe selectivity.
        Now includes class ratio analysis to explain shuffled-label baseline.
        """
        logger.info(f"  [Analysis 6] Probe selectivity controls — {model_name}")
        
        is_word = targets['is_word']
        analyzer = FrequencyAnalyzer(self.config)
        
        # ── NEW: Class ratio analysis ────────────────────────────────────
        class_ratio = is_word.mean()
        expected_shuffled = max(class_ratio, 1 - class_ratio)
        logger.info(f"  Class ratio (words/total): {class_ratio:.3f}")
        logger.info(f"  Expected shuffled accuracy: {expected_shuffled:.3f} "
                    f"(majority class baseline)")
        
        linear_results = []
        shuffled_results = []
        selectivity_results = []
        class_ratio_results = []  # NEW
        
        all_layers = sorted(all_hidden.keys())
        probe_layers = sorted(set(
            [all_layers[0], all_layers[len(all_layers)//4],
            all_layers[len(all_layers)//2],
            all_layers[3*len(all_layers)//4],
            all_layers[-1]]
            + all_layers[::4]
        ))
        
        for li in tqdm(probe_layers, desc="Selectivity controls"):
            if li not in all_hidden:
                continue
            X = all_hidden[li]
            y = is_word
            
            # Split
            try:
                strat = y
                Xtv, Xte, ytv, yte = train_test_split(
                    X, y, test_size=self.config.TEST_SIZE,
                    stratify=strat, random_state=SEED)
                vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                Xtr, Xval, ytr, yval = train_test_split(
                    Xtv, ytv, test_size=vsz, stratify=ytv, random_state=SEED)
            except ValueError:
                continue
            
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            Xval_s = sc.transform(Xval)  # scaled validation set (used implicitly by train_classifier)
            
            # ── NEW: Class ratio for this split ──────────────────────────
            tr_ratio = ytr.mean()
            val_ratio = yval.mean()
            te_ratio = yte.mean()
            
            class_ratio_results.append({
                'layer': li,
                'train_ratio': tr_ratio,
                'val_ratio': val_ratio,
                'test_ratio': te_ratio,
                'expected_shuffled_acc': max(te_ratio, 1 - te_ratio),
            })

            # ── (a) Linear probe ──────────────────────────────────────
            try:
                lr = LogisticRegression(
                    max_iter=1000, solver='liblinear', penalty='l2', C=1.0,
                    random_state=SEED,
                )
                lr.fit(Xtr_s, ytr)
                y_pred_lr = lr.predict(Xte_s)
                y_prob_lr = lr.predict_proba(Xte_s)[:, 1]
                acc_linear = accuracy_score(yte, y_pred_lr)
                _, _, f1_linear, _ = precision_recall_fscore_support(
                    yte, y_pred_lr, average='binary', zero_division=0)
                try:
                    auc_linear = roc_auc_score(yte, y_prob_lr)
                except:
                    auc_linear = np.nan

                lo_l, hi_l = wilson_ci(int(round(acc_linear * len(yte))), len(yte))
                linear_results.append({
                    'layer': li, 'accuracy': acc_linear,
                    'accuracy_ci95_low': lo_l, 'accuracy_ci95_high': hi_l,
                    'f1': f1_linear, 'auc': auc_linear
                })
            except Exception as e:
                logger.warning(f"  Linear probe layer {li} failed: {e}")
                linear_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1': np.nan,
                    'auc': np.nan
                })

            # ── (b) Shuffled-label control ────────────────────────────
            try:
                rng = np.random.RandomState(SEED)
                ytr_shuffled = rng.permutation(ytr)
                yval_shuffled = rng.permutation(yval)

                clf_shuf = analyzer.train_classifier(
                    Xtr, ytr_shuffled, Xval, yval_shuffled,
                    li, model_name, seed=SEED)
                res_shuf = analyzer.evaluate_group(
                    clf_shuf, Xte, yte, "Shuffled")

                lo_s, hi_s = wilson_ci(int(round(res_shuf['accuracy'] * len(yte))), len(yte))
                shuffled_results.append({
                    'layer': li, 'accuracy': res_shuf['accuracy'],
                    'accuracy_ci95_low': lo_s, 'accuracy_ci95_high': hi_s,
                    'f1': res_shuf['f1'], 'auc': res_shuf['auc'],
                })
                del clf_shuf
            except Exception as e:
                logger.warning(f"  Shuffled probe layer {li} failed: {e}")
                shuffled_results.append({
                    'layer': li, 'accuracy': np.nan, 'f1': np.nan,
                    'auc': np.nan
                })

            # ── (c) Selectivity ───────────────────────────────────────
            # Train the primary probe (same family as the main curve)
            try:
                clf_task = analyzer.train_classifier(
                    Xtr, ytr, Xval, yval, li, model_name, seed=SEED)
                res_task = analyzer.evaluate_group(clf_task, Xte, yte, "Task")

                acc_task = res_task['accuracy']
                acc_shuf = shuffled_results[-1]['accuracy']
                selectivity = acc_task - acc_shuf if not np.isnan(acc_shuf) else np.nan

                selectivity_results.append({
                    'layer': li,
                    'acc_task': acc_task,
                    'acc_shuffled': acc_shuf,
                    'selectivity': selectivity,
                })
                del clf_task
            except Exception:
                selectivity_results.append({
                    'layer': li, 'acc_task': np.nan,
                    'acc_shuffled': np.nan, 'selectivity': np.nan,
                })
            # ── NEW: Add class ratio analysis to results ──────────────────
            # In shuffled_results, add expected baseline
            if shuffled_results:
                shuffled_results[-1]['expected_baseline'] = max(te_ratio, 1 - te_ratio)
                shuffled_results[-1]['class_ratio'] = te_ratio
        
        return {
            'linear_probe': linear_results,
            'shuffled_label': shuffled_results,
            'selectivity': selectivity_results,
            'class_ratio_analysis': class_ratio_results,  # NEW
            'global_class_ratio': float(class_ratio),
            'expected_shuffled_global': float(expected_shuffled),
        }
    def _analyze_shuffled_performance(self, shuffled_results, class_ratios):
        """
        Analyze whether shuffled performance exceeds chance due to class imbalance.
        """
        analysis = []
        for res, ratio in zip(shuffled_results, class_ratios):
            expected = max(ratio, 1 - ratio)
            actual = res['accuracy']
            excess = actual - expected
            
            analysis.append({
                'layer': res['layer'],
                'class_ratio': ratio,
                'expected_chance': 0.5,
                'expected_majority_baseline': expected,
                'shuffled_accuracy': actual,
                'excess_over_baseline': excess,
                'significant_excess': excess > 0.05,  # Heuristic threshold
            })
        return analysis
    # ────────────────────────────────────────────────────────────────────
    # Analysis 7: Continuous Frequency Regression
    # ────────────────────────────────────────────────────────────────────
    def continuous_frequency_regression(self, all_hidden, targets,
                                         model_name) -> dict:
        """
        Predict continuous log_HAL frequency from hidden states using
        ridge regression. This provides stronger psycholinguistic evidence
        than tertile separation because it treats frequency as a continuous
        variable and avoids information loss from discretisation.

        R² and Spearman correlation are reported per layer.
        A positive R² means the hidden states carry frequency information
        beyond what a constant (mean) prediction would give.

        Ridge regression (L2 penalty) is chosen over OLS because
        hidden-state dimensionality (e.g. 2048) may exceed training
        samples, and L2 regularisation prevents degenerate coefficient
        inflation. Alpha is selected from {0.1, 1, 10, 100} by
        validation-set R².
        """
        logger.info(f"  [Analysis 7] Continuous frequency regression — {model_name}")

        is_word      = targets['is_word']
        log_freq     = targets['log_frequency']

        # Words with valid frequency only
        wm = (is_word == 1) & (~np.isnan(log_freq))

        if wm.sum() < 2 * self.config.MIN_SAMPLES_PER_GROUP:
            logger.warning("  Not enough words with frequency for regression")
            return {'regression_results': []}

        y_freq = log_freq[wm]

        regression_results = []

        for li in tqdm(range(len(all_hidden)),
                       desc="Continuous freq regression"):
            if li not in all_hidden:
                continue
            X = all_hidden[li][wm]

            try:
                Xtv, Xte, ytv, yte = train_test_split(
                    X, y_freq, test_size=self.config.TEST_SIZE,
                    random_state=SEED)
                vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                Xtr, Xval, ytr, yval = train_test_split(
                    Xtv, ytv, test_size=vsz, random_state=SEED)

                sc = StandardScaler()
                Xtr_s  = sc.fit_transform(Xtr)
                Xval_s = sc.transform(Xval)
                Xte_s  = sc.transform(Xte)

                # Select alpha by validation R²
                best_alpha, best_val_r2 = 1.0, -np.inf
                for alpha in [0.1, 1.0, 10.0, 100.0]:
                    ridge = Ridge(alpha=alpha, random_state=SEED)
                    ridge.fit(Xtr_s, ytr)
                    val_r2 = ridge.score(Xval_s, yval)
                    if val_r2 > best_val_r2:
                        best_alpha = alpha
                        best_val_r2 = val_r2

                ridge = Ridge(alpha=best_alpha, random_state=SEED)
                ridge.fit(Xtr_s, ytr)
                y_pred = ridge.predict(Xte_s)

                r2   = r2_score(yte, y_pred)
                rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))
                spearman_r, spearman_p = stats.spearmanr(yte, y_pred)
                pearson_r, pearson_p = stats.pearsonr(yte, y_pred)

                regression_results.append({
                    'layer': li,
                    'r2': float(r2),
                    'rmse': rmse,
                    'pearson_r': float(pearson_r),
                    'pearson_p': float(pearson_p),
                    'spearman_r': float(spearman_r),
                    'spearman_p': float(spearman_p),
                    'representation_type': PRIMARY_REPRESENTATION,
                    'best_alpha': best_alpha,
                    'n_train': len(ytr),
                    'n_test': len(yte),
                })
            except Exception as e:
                logger.warning(f"  Regression layer {li} failed: {e}")
                regression_results.append({
                    'layer': li, 'r2': np.nan, 'rmse': np.nan,
                    'spearman_r': np.nan, 'spearman_p': np.nan,
                })

        return {'regression_results': regression_results}

    # ────────────────────────────────────────────────────────────────────
    # Shared helper: fixed representative-layer sampling scheme
    # (identical logic to the one duplicated in Analyses 4 and 6, factored
    # out here for the new analyses so the sampling rule stays consistent)
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _representative_layers(all_layers: list[int]) -> list[int]:
        n = len(all_layers)
        return sorted(set(
            [all_layers[0], all_layers[n // 4], all_layers[n // 2],
             all_layers[3 * n // 4], all_layers[-1]]
            + all_layers[::4]
        ))

    # ────────────────────────────────────────────────────────────────────
    # Analysis 8: Representation Intervention (Causal Test)
    # ────────────────────────────────────────────────────────────────────
    def intervention_analysis(self, all_hidden, targets, model_name) -> list[dict]:
        """
        Causal test: does *directly manipulating* the frequency-coded
        directions in a layer's representation change the ALREADY-TRAINED
        word-vs-nonword LDT classifier's behaviour?

        Methodology
        -----------
        1. At each representative layer, train the same LDT classifier
           used in Step 2 (word vs non-word) on a fresh train/val/test
           split, and keep it in memory (unlike `FrequencyAnalyzer.
           analyze_layer`, which discards the classifier).
        2. Independently, fit a LINEAR probe (logistic regression) on
           the SAME standardised feature space (`clf.scaler`) to
           discriminate high- vs low-frequency words. Linear weights are
           required here because "top-k most predictive dimensions" is
           only a well-defined, individually-interpretable notion for a
           linear model. LEAKAGE FIX: the direction is fitted ONLY on the
           high/low-frequency words that lie in the LDT classifier's TRAIN
           split (the previous version fitted it on all high/low words,
           including the test items later used for evaluation).
        3. Take the top-k |coefficient| dimensions and build a unit
           vector supported only on those dimensions.
        4. On the LDT classifier's test split, apply three interventions
           to the STANDARDISED representation and re-evaluate the SAME
           trained classifier (no retraining — this is what makes the
           test causal rather than merely correlational):
             - additive_pos : + INTERVENTION_MAGNITUDE * direction
             - additive_neg : - INTERVENTION_MAGNITUDE * direction
             - nullify      : zero out the top-k dimensions
        5. Report accuracy and the high/low frequency accuracy gap
           before and after each intervention.

        Caveat (flagged, not glossed over): steps 4's "additive" magnitude
        is defined in standardised (z-score) units, uniformly across all
        test samples — it is not adaptively scaled per class. If the
        classifier is causally using this direction to encode frequency
        (and, indirectly, tokenisation/typicality confounds correlated
        with frequency), the frequency gap should shrink under nullification
        and shift under the additive interventions; if the gap is
        unchanged, the top-k linear direction found by the auxiliary probe
        is not what the LDT probe relies on, which is itself a legitimate
        (and reportable) negative result. This remains an intervention on
        the PROBE's input space (representation-level), not on the LLM's
        forward computation; it licenses claims about what the trained
        probe uses, not about what the LLM uses.
        """
        logger.info(f"  [Analysis 8] Representation intervention — {model_name}")

        is_word      = targets['is_word']
        is_high_freq = targets['is_high_freq']
        is_low_freq  = targets['is_low_freq']
        freq_groups  = targets['freq_group']

        wm  = (is_word == 1)
        hfm = wm & (is_high_freq == 1)
        lfm = wm & (is_low_freq  == 1)

        if hfm.sum() < self.config.MIN_SAMPLES_PER_GROUP or \
           lfm.sum() < self.config.MIN_SAMPLES_PER_GROUP:
            logger.warning("  Insufficient high/low words for intervention analysis")
            return []

        all_layers   = sorted(all_hidden.keys())
        probe_layers = self._representative_layers(all_layers)
        mag = self.config.INTERVENTION_MAGNITUDE
        k   = self.config.INTERVENTION_TOP_K

        hf_idx = np.where(hfm)[0]
        lf_idx = np.where(lfm)[0]
        freq_row_idx = np.concatenate([hf_idx, lf_idx])
        freq_y = np.concatenate([np.ones(len(hf_idx), int), np.zeros(len(lf_idx), int)])

        strat = ['nonword' if is_word[i] == 0 else f"word_{freq_groups[i]}"
                 for i in range(len(is_word))]
        idx_all = np.arange(len(is_word))

        analyzer = FrequencyAnalyzer(self.config)
        results = []

        for li in tqdm(probe_layers, desc="Intervention analysis"):
            X = all_hidden[li]
            try:
                Xtv, Xte, ytv, yte, itv, ite = train_test_split(
                    X, is_word, idx_all, test_size=self.config.TEST_SIZE,
                    stratify=strat, random_state=SEED)
                vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                Xtr, Xval, ytr, yval, itr, _ival = train_test_split(
                    Xtv, ytv, itv, test_size=vsz, stratify=ytv, random_state=SEED)
            except ValueError as e:
                logger.warning(f"  Intervention split failed at layer {li}: {e}")
                results.append(self._empty_intervention_row(li))
                continue

            try:
                clf = analyzer.train_classifier(Xtr, ytr, Xval, yval, li,
                                                model_name, seed=SEED)
            except Exception as e:
                logger.warning(f"  Intervention classifier failed at layer {li}: {e}")
                results.append(self._empty_intervention_row(li))
                continue

            # ── Frequency direction: linear probe on clf's own scaled space ──
            # Fitted on TRAIN-split high/low words only (no test leakage).
            dir_mask = np.isin(freq_row_idx, itr)
            dir_rows = freq_row_idx[dir_mask]
            dir_y    = freq_y[dir_mask]
            if np.isin(dir_rows, ite).any():
                raise RuntimeError("Intervention direction would use test items")
            if len(np.unique(dir_y)) < 2:
                logger.warning(f"  Intervention layer {li}: train split lacks "
                               f"both frequency groups — skipped")
                results.append(self._empty_intervention_row(li))
                del clf
                continue
            X_freq_s = clf.scaler.transform(X[dir_rows])
            try:
                dir_lr = LogisticRegression(max_iter=1000, solver='lbfgs',
                                            random_state=SEED, C=1.0)
                dir_lr.fit(X_freq_s, dir_y)
                dir_train_acc = dir_lr.score(X_freq_s, dir_y)
            except Exception as e:
                logger.warning(f"  Direction probe failed at layer {li}: {e}")
                results.append(self._empty_intervention_row(li))
                del clf
                continue

            coef = dir_lr.coef_[0]
            topk_idx = np.argsort(np.abs(coef))[-k:]
            direction = np.zeros_like(coef)
            direction[topk_idx] = coef[topk_idx]
            dnorm = np.linalg.norm(direction)
            if dnorm > 0:
                direction = direction / dnorm

            Xte_s = clf.scaler.transform(Xte)
            hf_mask_te = np.isin(ite, hf_idx)
            lf_mask_te = np.isin(ite, lf_idx)

            def _eval(Xs, tag):
                overall = self._forward_metrics(clf, Xs, yte, tag)
                hf_r = (self._forward_metrics(clf, Xs[hf_mask_te], yte[hf_mask_te], tag)
                        if hf_mask_te.sum() >= self.config.MIN_TEST_SAMPLES else None)
                lf_r = (self._forward_metrics(clf, Xs[lf_mask_te], yte[lf_mask_te], tag)
                        if lf_mask_te.sum() >= self.config.MIN_TEST_SAMPLES else None)
                gap = (hf_r['accuracy'] - lf_r['accuracy']) if (hf_r and lf_r) else np.nan
                return overall['accuracy'], gap

            base_acc, base_gap = _eval(Xte_s, 'baseline')

            row = {
                'layer': li,
                'n_test': len(yte),
                'baseline_accuracy': base_acc,
                'baseline_freq_gap': base_gap,
                'direction_probe_train_accuracy': dir_train_acc,
                'direction_fit_split': 'ldt_train_split_only',
                'n_direction_fit_words': int(len(dir_rows)),
                'top_k_dims': topk_idx.tolist(),
                'representation_type': PRIMARY_REPRESENTATION,
                'probe_type': analyzer.probe_type,
            }

            interventions = {
                'additive_pos': Xte_s + mag * direction[np.newaxis, :],
                'additive_neg': Xte_s - mag * direction[np.newaxis, :],
            }
            Xte_null = Xte_s.copy()
            Xte_null[:, topk_idx] = 0.0
            interventions['nullify'] = Xte_null

            for name, Xs in interventions.items():
                acc, gap = _eval(Xs, name)
                row[f'{name}_accuracy']       = acc
                row[f'{name}_freq_gap']       = gap
                row[f'{name}_delta_accuracy'] = acc - base_acc
                row[f'{name}_delta_freq_gap'] = (
                    gap - base_gap if not (np.isnan(gap) or np.isnan(base_gap)) else np.nan
                )

            # ── C1: dose-response sweep with a random-direction null ──────
            # Magnitudes are in units of the SD of the TRAINING projections on
            # the direction (the fixed 0.5 z-unit step above is tiny relative
            # to ‖x‖ ≈ √D in standardised space). Null: k-sparse random unit
            # directions with the same support size, same SD scaling.
            Xtr_s = clf.scaler.transform(Xtr)
            rng_c1 = np.random.RandomState(SEED + li)

            def _dir_effects(dvec):
                sdp = float(np.std(Xtr_s @ dvec)) or 1.0
                eff = {}
                for m in self.config.INTERVENTION_MAGNITUDES:
                    for sgn, tag in ((1.0, 'pos'), (-1.0, 'neg')):
                        acc_, gap_ = _eval(Xte_s + sgn * m * sdp * dvec[np.newaxis, :], tag)
                        eff[(m, tag)] = (acc_ - base_acc,
                                         gap_ - base_gap if _finite(gap_) and _finite(base_gap) else np.nan)
                return eff, sdp
            eff_true, sd_true = _dir_effects(direction)
            null = []
            for _ in range(self.config.N_RANDOM_DIRECTIONS):
                rd = np.zeros_like(direction)
                sup = rng_c1.choice(len(direction), size=min(k, len(direction)), replace=False)
                rd[sup] = rng_c1.randn(len(sup))
                rd /= np.linalg.norm(rd)
                null.append(_dir_effects(rd)[0])
            dose = []
            for m in self.config.INTERVENTION_MAGNITUDES:
                obs = max(abs(np.nan_to_num(eff_true[(m, 'pos')][1])),
                          abs(np.nan_to_num(eff_true[(m, 'neg')][1])))
                nul = [max(abs(np.nan_to_num(e[(m, 'pos')][1])), abs(np.nan_to_num(e[(m, 'neg')][1])))
                       for e in null]
                dose.append({
                    'layer': li, 'magnitude_sd_units': float(m),
                    'projection_sd_standardised_space': sd_true,
                    'delta_accuracy_pos': eff_true[(m, 'pos')][0],
                    'delta_accuracy_neg': eff_true[(m, 'neg')][0],
                    'delta_freq_gap_pos': eff_true[(m, 'pos')][1],
                    'delta_freq_gap_neg': eff_true[(m, 'neg')][1],
                    'null_mean_abs_delta_freq_gap': float(np.mean(nul)) if nul else np.nan,
                    'null_q95_abs_delta_freq_gap': float(np.percentile(nul, 95)) if nul else np.nan,
                    'null_mean_delta_accuracy_pos': float(np.mean([e[(m, 'pos')][0] for e in null])) if null else np.nan,
                    'p_vs_random_directions_empirical': ((1 + sum(v >= obs for v in nul)) /
                                                         (1 + len(nul))) if nul else np.nan,
                    'n_random_directions': len(nul),
                    'intervention_space': 'probe_input_standardised',
                    'representation_type': PRIMARY_REPRESENTATION})
            row['dose_response'] = dose

            results.append(row)
            del clf
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return results

    def _forward_metrics(self, clf, X_standardised, y, name) -> dict:
        """Evaluate an already-trained classifier on PRE-standardised
        input (bypasses clf.scaler — required for representation
        interventions, which are applied in the standardised space)."""
        clf.eval()
        with torch.no_grad():
            probs = F.softmax(
                clf(torch.FloatTensor(X_standardised).to(self.device)), 1
            ).cpu().numpy()
        return _full_metrics(y, probs.argmax(1), probs[:, 1], name)

    @staticmethod
    def _empty_intervention_row(li) -> dict:
        row = {'layer': li, 'n_test': 0, 'baseline_accuracy': np.nan,
               'baseline_freq_gap': np.nan, 'direction_probe_train_accuracy': np.nan,
               'top_k_dims': []}
        for name in ['additive_pos', 'additive_neg', 'nullify']:
            row[f'{name}_accuracy'] = np.nan
            row[f'{name}_freq_gap'] = np.nan
            row[f'{name}_delta_accuracy'] = np.nan
            row[f'{name}_delta_freq_gap'] = np.nan
        return row

    # ────────────────────────────────────────────────────────────────────
    # Analysis 9: Cross-Model Probe Transferability
    # ────────────────────────────────────────────────────────────────────
    def build_transfer_snapshot(self, all_hidden, targets, model_name,
                                    num_layers) -> dict:
            """
            Caches, at each representative layer: (a) the trained probe
            (scaler + weights) and its held-out TEST split, exactly as
            before, and (b) a disjoint TRAIN+VAL pool with stimulus
            identities (`idx`), used ONLY to fit a cross-model stitching
            map later — never to evaluate a probe. Keeping the fit pool
            and the test split disjoint is what keeps the eventual
            transfer number honest (no leakage between "learn the mapping"
            and "measure how well it transfers").
            """
            logger.info(f"  [Analysis 9] Building transfer snapshot — {model_name}")

            is_word      = targets['is_word']
            is_high_freq = targets['is_high_freq']
            is_low_freq  = targets['is_low_freq']
            stim_id_all  = np.asarray(targets['idx'])

            wm  = (is_word == 1)
            hfm = wm & (is_high_freq == 1)
            lfm = wm & (is_low_freq  == 1)

            empty = {'model_name': model_name, 'num_layers': num_layers,
                    'hidden_dim': None, 'layers': {}}
            if hfm.sum() < self.config.MIN_SAMPLES_PER_GROUP or \
            lfm.sum() < self.config.MIN_SAMPLES_PER_GROUP:
                logger.warning("  Not enough high/low words for transfer snapshot")
                return empty

            hidden_dim = next(iter(all_hidden.values())).shape[1]
            all_layers = sorted(all_hidden.keys())
            if self.config.TRANSFER_TEST_LAYERS:
                repr_layers = [li for li in self.config.TRANSFER_TEST_LAYERS
                            if li in all_hidden]
                if not repr_layers:
                    logger.warning("  TRANSFER_TEST_LAYERS did not match any "
                                "extracted layer — falling back to default")
                    repr_layers = self._representative_layers(all_layers)
            else:
                repr_layers = self._representative_layers(all_layers)

            hf_idx = np.where(hfm)[0]
            lf_idx = np.where(lfm)[0]
            row_idx = np.concatenate([hf_idx, lf_idx])
            stim_id = stim_id_all[row_idx]
            y_all   = np.concatenate([np.ones(len(hf_idx), int), np.zeros(len(lf_idx), int)])

            analyzer = FrequencyAnalyzer(self.config)
            layer_snapshots = {}

            for li in tqdm(repr_layers, desc="Transfer snapshot"):
                X = all_hidden[li][row_idx]
                try:
                    Xtv, Xte, ytv, yte, stv, ste = train_test_split(
                        X, y_all, stim_id, test_size=self.config.TEST_SIZE,
                        stratify=y_all, random_state=SEED)
                    vsz = self.config.VAL_SIZE / (1 - self.config.TEST_SIZE)
                    Xtr, Xval, ytr, yval, str_, sval = train_test_split(
                        Xtv, ytv, stv, test_size=vsz, stratify=ytv, random_state=SEED)

                    clf = analyzer.train_classifier(Xtr, ytr, Xval, yval, li,
                                                    model_name, seed=SEED)
                    native = analyzer.evaluate_group(clf, Xte, yte, 'Native')
                    # BUG FIX: the fast linear probe is a `_LinearProbeShim`
                    # whose weights are `_lin.*`; reloading it into a
                    # LexicalDecisionClassifier (keys `out.*`) raised in
                    # load_state_dict, so EVERY transfer row was NaN. The
                    # linear probe is now stored as its logit (w, b).
                    probe_linear = None
                    lin = getattr(clf, '_lin', None)
                    if lin is not None:
                        probe_linear = {'w': lin.weight.detach().cpu().numpy()[0].astype(np.float64),
                                        'b': float(lin.bias.detach().cpu().numpy()[0])}
                    elif not self.config.CLASSIFIER_ARCHITECTURE:
                        W_ = clf.out.weight.detach().cpu().numpy().astype(np.float64)
                        b_ = clf.out.bias.detach().cpu().numpy().astype(np.float64)
                        probe_linear = {'w': W_[1] - W_[0], 'b': float(b_[1] - b_[0])}

                    layer_snapshots[li] = {
                        'fraction': li / max(num_layers - 1, 1),
                        'state_dict': {k: v.detach().cpu().clone()
                                    for k, v in clf.state_dict().items()},
                        'scaler_mean':  clf.scaler.mean_.copy(),
                        'scaler_scale': clf.scaler.scale_.copy(),
                        'input_dim': X.shape[1],
                        'native_accuracy': native['accuracy'],
                        'native_f1': native['f1'],
                        'native_auc': native['auc'],
                        'native_correct': (native['predictions'] == yte).astype(int),
                        'probe_linear': probe_linear,
                        # Evaluation-only test split (never used to fit the stitching map)
                        'X_test': Xte.astype(np.float32).copy(),
                        'y_test': yte.copy(),
                        'stim_test': ste.copy(),
                        # Fit-only pool for the cross-model stitching map (never evaluated)
                        'X_fit_pool':    np.concatenate([Xtr, Xval], axis=0).astype(np.float32),
                        'stim_fit_pool': np.concatenate([str_, sval], axis=0),
                    }
                    del clf
                except Exception as e:
                    logger.warning(f"  Transfer snapshot layer {li} failed: {e}")

            if self.device == 'cuda':
                torch.cuda.empty_cache()

            return {'model_name': model_name, 'num_layers': num_layers,
                    'hidden_dim': hidden_dim, 'layers': layer_snapshots}
            
            
    def _fit_stitching_map(self, Xs_pool, Xt_pool, stim_s, stim_t):
            """
            Learns a linear, ridge-regularised map g: R^{d_t} -> R^{d_s},
            g(x) = (x - mu_t)/sigma_t @ W + b, translating a TARGET model's
            representation into the SOURCE model's representation space.

            This is "model stitching" (Lenc & Vedaldi, 2015; Bansal,
            Nakkiran & Barak, 2021 — "Revisiting Model Stitching to Compare
            Neural Representations"), the standard way to compare/reuse
            probes across representations of different dimensionality (and,
            properly, of the SAME dimensionality too — matching width does
            not imply matching basis).

            Pairing is done by stimulus identity (`idx`), not array
            position, since each model's train/val split is independent.

            Returns
            -------
            (ridge_model, target_scaler) or (None, None), alpha, val_r2, n_paired
            val_r2 is the held-out R² of the mapping itself (uniform-average
            across output dims). This MUST be reported alongside
            transfer_accuracy: a low/negative val_r2 means the mapping
            explains little more than the mean, and any transfer_accuracy
            computed through it is not evidence of shared representational
            content — it is noise passed through a trained classifier.
            """
            common = np.intersect1d(stim_s, stim_t)
            if len(common) < self.config.STITCHING_MIN_PAIRED_SAMPLES:
                return (None, None), None, np.nan, len(common)

            s_pos = {v: i for i, v in enumerate(stim_s)}
            t_pos = {v: i for i, v in enumerate(stim_t)}
            s_rows = np.array([s_pos[c] for c in common])
            t_rows = np.array([t_pos[c] for c in common])

            Xs = Xs_pool[s_rows]
            Xt = Xt_pool[t_rows]

            n = len(common)
            rng = np.random.RandomState(SEED)
            perm = rng.permutation(n)
            n_val = max(int(0.2 * n), 10)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            if len(tr_idx) < self.config.STITCHING_MIN_PAIRED_SAMPLES // 2:
                return (None, None), None, np.nan, n

            sc_t = StandardScaler().fit(Xt[tr_idx])
            Xt_tr_s  = sc_t.transform(Xt[tr_idx])
            Xt_val_s = sc_t.transform(Xt[val_idx])

            best_alpha, best_r2 = self.config.STITCHING_ALPHAS[0], -np.inf
            for alpha in self.config.STITCHING_ALPHAS:
                ridge = Ridge(alpha=alpha, random_state=SEED)
                ridge.fit(Xt_tr_s, Xs[tr_idx])
                r2 = ridge.score(Xt_val_s, Xs[val_idx])
                if r2 > best_r2:
                    best_alpha, best_r2 = alpha, r2

            # Refit on the full paired pool at the selected alpha for the final map
            sc_t_full = StandardScaler().fit(Xt)
            ridge_final = Ridge(alpha=best_alpha, random_state=SEED)
            ridge_final.fit(sc_t_full.transform(Xt), Xs)

            return (ridge_final, sc_t_full), best_alpha, float(best_r2), n

    def _paired_pool(self, snap_s, snap_t):
        common = np.intersect1d(snap_s['stim_fit_pool'], snap_t['stim_fit_pool'])
        s_pos = {v: i for i, v in enumerate(snap_s['stim_fit_pool'])}
        t_pos = {v: i for i, v in enumerate(snap_t['stim_fit_pool'])}
        return (snap_s['X_fit_pool'][[s_pos[c] for c in common]].astype(np.float64),
                snap_t['X_fit_pool'][[t_pos[c] for c in common]].astype(np.float64))

    def _transfer_nulls(self, snap_s, snap_t, alpha, true_pred) -> dict:
        """C4 / H3 — two nulls for the stitched transfer, both in CLOSED FORM.

        With target features standardised on the paired pool (S, column
        means 0) and ridge  W = (SᵀS + αI)⁻¹ Sᵀ Y_c, the transferred logit of
        the source's LINEAR probe (w, b; source scaler μ_s, σ_s) is
            z = S_te (SᵀS + αI)⁻¹ Sᵀ q + const,    q = Y_c (w / σ_s),
        so re-pairing the stimuli (a permutation π of the rows of Y) only
        permutes q. Hence every permutation costs one solve with a cached
        Cholesky factor — 1000 permutations are cheap and EXACT (no
        approximation), and the alpha of the true map is reused.

        (1) shuffled-pairing null: π random → is the stimulus-level
            correspondence between the two models what makes transfer work?
        (2) random-probe null: w replaced by a random direction of equal norm
            → is the SOURCE PROBE's direction special, or would any readout of
            the frequency-bearing subspace transfer? Threshold-free AUC is
            used, with max(AUC, 1−AUC) for the null (conservative).
        CAVEAT (reported): a high transfer accuracy is also expected whenever
        both models encode frequency linearly and the ridge map preserves that
        subspace — (2) is the control for exactly this."""
        cfg = self.config
        pl = snap_s.get('probe_linear')
        out = {'perm_null_n': 0, 'random_probe_null_n': 0}
        if pl is None:
            out['null_skip_reason'] = 'source probe is non-linear — closed-form null undefined'
            return out
        Ys, Xt = self._paired_pool(snap_s, snap_t)
        mu_t, sd_t = Xt.mean(0), Xt.std(0)
        sd_t[sd_t == 0] = 1.0
        S = (Xt - mu_t) / sd_t
        S_te = (snap_t['X_test'].astype(np.float64) - mu_t) / sd_t
        Yc = Ys - Ys.mean(0)
        dev = self.device if torch.cuda.is_available() else 'cpu'
        St = torch.as_tensor(S, dtype=torch.float64, device=dev)
        G = St.T @ St + float(alpha) * torch.eye(S.shape[1], dtype=torch.float64, device=dev)
        Lc = torch.linalg.cholesky(G)
        Ste = torch.as_tensor(S_te, dtype=torch.float64, device=dev)

        def _logits(Q):              # Q: (n_pool, P) → (n_test, P)
            Qt = torch.as_tensor(Q, dtype=torch.float64, device=dev)
            return (Ste @ torch.cholesky_solve(St.T @ Qt, Lc)).cpu().numpy()
        sig_s = np.asarray(snap_s['scaler_scale'], np.float64)
        u = pl['w'] / sig_s
        q = Yc @ u
        c0 = float((Ys.mean(0) - snap_s['scaler_mean']) / sig_s @ pl['w'] + pl['b'])
        z_true = _logits(q[:, None])[:, 0] + c0
        y_te = np.asarray(snap_t['y_test']).astype(int)
        agree = float(np.mean((z_true > 0).astype(int) == np.asarray(true_pred).astype(int)))
        out['closed_form_agreement_with_pipeline'] = agree
        if agree < 0.99:
            out['null_skip_reason'] = (f'closed form reproduces only {agree:.3f} of the '
                                       f'pipeline predictions — null not reported')
            return out
        acc_true = float(np.mean((z_true > 0).astype(int) == y_te))
        auc_true = auc_mann_whitney(z_true[y_te == 1], z_true[y_te == 0])
        rng = np.random.RandomState(SEED + 97)
        perm_acc, perm_auc = [], []
        P = cfg.TRANSFER_N_PERMUTATIONS
        for s in range(0, P, 200):
            k = min(200, P - s)
            Q = np.column_stack([q[rng.permutation(len(q))] for _ in range(k)])
            Zp = _logits(Q) + c0
            for j in range(k):
                perm_acc.append(float(np.mean((Zp[:, j] > 0).astype(int) == y_te)))
                perm_auc.append(auc_mann_whitney(Zp[y_te == 1, j], Zp[y_te == 0, j]))
        rp_auc = []
        R = cfg.TRANSFER_N_RANDOM_PROBES
        wn = np.linalg.norm(pl['w'])
        for s in range(0, R, 200):
            k = min(200, R - s)
            Wr = rng.randn(len(pl['w']), k)
            Wr *= wn / np.linalg.norm(Wr, axis=0, keepdims=True)
            Zr = _logits(Yc @ (Wr / sig_s[:, None]))
            for j in range(k):
                a = auc_mann_whitney(Zr[y_te == 1, j], Zr[y_te == 0, j])
                rp_auc.append(max(a, 1 - a))
        del St, G, Lc, Ste
        out.update({
            'closed_form_transfer_accuracy': acc_true, 'closed_form_transfer_auc': auc_true,
            'perm_null_n': len(perm_acc),
            'perm_null_accuracy_mean': float(np.mean(perm_acc)) if perm_acc else np.nan,
            'perm_null_accuracy_q95': float(np.percentile(perm_acc, 95)) if perm_acc else np.nan,
            'perm_null_auc_mean': float(np.mean(perm_auc)) if perm_auc else np.nan,
            'perm_p_value_accuracy': ((1 + sum(a >= acc_true for a in perm_acc)) /
                                      (1 + len(perm_acc))) if perm_acc else np.nan,
            'perm_p_value_auc': ((1 + sum(a >= auc_true for a in perm_auc)) /
                                 (1 + len(perm_auc))) if perm_auc else np.nan,
            'random_probe_null_n': len(rp_auc),
            'random_probe_null_auc_mean': float(np.mean(rp_auc)) if rp_auc else np.nan,
            'random_probe_null_auc_q95': float(np.percentile(rp_auc, 95)) if rp_auc else np.nan,
            'random_probe_p_value_auc': ((1 + sum(a >= auc_true for a in rp_auc)) /
                                         (1 + len(rp_auc))) if rp_auc else np.nan,
            '_perm_null_accuracies': perm_acc,
        })
        return out

    def cross_model_transfer_analysis(self, source_snap: dict,
                                      target_snap: dict) -> list[dict]:
        """
        Applies SOURCE's frozen frequency probe to TARGET's held-out test
        features via a LEARNED linear stitching map (see `_fit_stitching_map`),
        so it works for equal or unequal hidden_dims. Source/target layers are
        matched by nearest normalised depth. Adds (H4) an exact McNemar test
        of transfer vs the target's NATIVE probe on the SAME target test items
        and (C4/H3) permutation and random-probe nulls.
        """
        rows = []
        src_name, tgt_name = source_snap['model_name'], target_snap['model_name']
        if not source_snap['layers'] or not target_snap['layers']:
            return rows

        for li_s, snap_s in source_snap['layers'].items():
            frac_s = snap_s['fraction']
            li_t = min(target_snap['layers'].keys(),
                       key=lambda l: abs(target_snap['layers'][l]['fraction'] - frac_s))
            snap_t = target_snap['layers'][li_t]

            row = {
                'source_model': src_name, 'target_model': tgt_name,
                'source_layer': li_s, 'target_layer': li_t,
                'representation_type': PRIMARY_REPRESENTATION,
                'transfer_task': 'high_vs_low_frequency_words',
                'source_relative_depth': (li_s + 1) / max(source_snap['num_layers'], 1),
                'target_relative_depth': (li_t + 1) / max(target_snap['num_layers'], 1),
                'source_fraction': frac_s, 'target_fraction': snap_t['fraction'],
                'source_hidden_dim': snap_s['input_dim'],
                'target_hidden_dim': snap_t['input_dim'],
            }
            try:
                (mapping, sc_t_full), alpha, val_r2, n_paired = self._fit_stitching_map(
                    snap_s['X_fit_pool'], snap_t['X_fit_pool'],
                    snap_s['stim_fit_pool'], snap_t['stim_fit_pool'])

                if mapping is None:
                    row.update({
                        'transfer_accuracy': np.nan, 'transfer_f1': np.nan,
                        'transfer_auc': np.nan,
                        'native_accuracy': snap_t['native_accuracy'],
                        'transfer_efficiency': np.nan,
                        'stitching_val_r2': np.nan, 'stitching_alpha': np.nan,
                        'n_paired_stimuli': n_paired,
                        'skip_reason': (f"only {n_paired} paired stimuli "
                                        f"(need >= {self.config.STITCHING_MIN_PAIRED_SAMPLES}) "
                                        f"— cannot fit a reliable stitching map"),
                    })
                    rows.append(row)
                    continue

                Xte_t_mapped = mapping.predict(sc_t_full.transform(snap_t['X_test']))
                Xte_scaled = (Xte_t_mapped - snap_s['scaler_mean']) / snap_s['scaler_scale']
                pl = snap_s.get('probe_linear')
                if pl is not None:
                    z = Xte_scaled @ pl['w'] + pl['b']
                    p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
                    probs = np.column_stack([1 - p1, p1])
                else:
                    clf = LexicalDecisionClassifier(
                        input_dim=snap_s['input_dim'],
                        hidden_dims=self.config.CLASSIFIER_ARCHITECTURE,
                        dropout=self.config.CLASSIFIER_DROPOUT,
                        use_residual=self.config.CLASSIFIER_USE_RESIDUAL,
                        bn_mom=self.config.CLASSIFIER_BN_MOMENTUM).to(self.device)
                    clf.load_state_dict(snap_s['state_dict'])
                    clf.eval()
                    with torch.no_grad():
                        probs = F.softmax(clf(torch.FloatTensor(Xte_scaled).to(self.device)),
                                          1).cpu().numpy()
                    del clf
                yte_t = snap_t['y_test']
                tm = _full_metrics(yte_t, probs.argmax(1), probs[:, 1], 'Transfer')
                native_acc = snap_t['native_accuracy']
                k_tr = int((tm['predictions'] == yte_t).sum())
                lo, hi = wilson_ci(k_tr, len(yte_t))
                tr_correct = (tm['predictions'] == yte_t).astype(int)
                _b, _c, p_mc = mcnemar_exact(tr_correct, snap_t['native_correct'])
                row.update({
                    'transfer_accuracy': tm['accuracy'], 'transfer_f1': tm['f1'],
                    'transfer_auc': tm['auc'], 'native_accuracy': native_acc,
                    'transfer_accuracy_ci95_low': lo, 'transfer_accuracy_ci95_high': hi,
                    'transfer_efficiency': (tm['accuracy'] / native_acc
                                            if native_acc and native_acc > 0 else np.nan),
                    'mcnemar_transfer_only_correct': _b, 'mcnemar_native_only_correct': _c,
                    'mcnemar_p_transfer_vs_native': p_mc,
                    'stitching_val_r2': val_r2, 'stitching_alpha': alpha,
                    'n_paired_stimuli': n_paired, 'n_target_test_items': int(len(yte_t)),
                    'skip_reason': None if val_r2 > 0.0 else
                        (f"stitching map val R²={val_r2:.3f} <= 0 — the map "
                         f"explains no more variance than predicting the mean; "
                         f"treat transfer_accuracy as unreliable, not as "
                         f"evidence of shared representational structure"),
                })
                try:
                    row.update(self._transfer_nulls(snap_s, snap_t, alpha, tm['predictions']))
                except Exception as e:                    # noqa: BLE001
                    row['null_skip_reason'] = f'null failed: {e}'
            except Exception as e:
                logger.warning(f"  Transfer {src_name}->{tgt_name} layer "
                               f"{li_s}->{li_t} failed: {e}")
                row.update({
                    'transfer_accuracy': np.nan, 'transfer_f1': np.nan,
                    'transfer_auc': np.nan, 'native_accuracy': np.nan,
                    'transfer_efficiency': np.nan, 'stitching_val_r2': np.nan,
                    'stitching_alpha': np.nan, 'n_paired_stimuli': np.nan,
                    'skip_reason': str(e),
                })
            rows.append(row)

        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return rows

    # ────────────────────────────────────────────────────────────────────
    # D4 — tokenization-stratified PRIMARY results (no refitting)
    # ────────────────────────────────────────────────────────────────────
    def tokenization_stratified_primary(self, oof_cache, targets, model_name,
                                        num_layers) -> list[dict]:
        """Breaks the PRIMARY out-of-fold predictions down by sub-word token
        count (1, 2, 3, ≥4) and single- vs multi-token, per layer and class.
        Nonwords are rarely single tokens, so class counts per stratum are
        reported: a stratum with (almost) one class cannot support a
        lexicality claim and is flagged (`stratum_has_both_classes`)."""
        y = np.asarray(targets['is_word']).astype(int)
        tc = np.asarray(targets['token_count']).astype(int)
        strata = {'single_token': tc == 1, 'multi_token': tc > 1,
                  'tokens_1': tc == 1, 'tokens_2': tc == 2, 'tokens_3': tc == 3,
                  'tokens_4plus': tc >= 4}
        rows = []
        for li in sorted(oof_cache):
            p = oof_cache[li][0]
            correct = ((p >= 0.5).astype(int) == y).astype(float)
            for sname, m in strata.items():
                n_w, n_nw = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
                r = {'model': model_name, 'layer': li, 'relative_depth': (li + 1) / num_layers,
                     'stratum': sname, 'n_words': n_w, 'n_nonwords': n_nw,
                     'stratum_has_both_classes': bool(n_w >= 10 and n_nw >= 10),
                     'representation_type': PRIMARY_REPRESENTATION,
                     'readout_type': PRIMARY_READOUT}
                if m.sum():
                    k = int(correct[m].sum())
                    r['accuracy'] = k / int(m.sum())
                    r['accuracy_ci95_low'], r['accuracy_ci95_high'] = wilson_ci(k, int(m.sum()))
                    r['word_accuracy'] = float(correct[m & (y == 1)].mean()) if n_w else np.nan
                    r['nonword_accuracy'] = float(correct[m & (y == 0)].mean()) if n_nw else np.nan
                    if r['stratum_has_both_classes']:
                        r['auroc'] = auc_mann_whitney(p[m & (y == 1)], p[m & (y == 0)])
                        r['balanced_accuracy'] = 0.5 * (r['word_accuracy'] + r['nonword_accuracy'])
                rows.append(r)
            ms, mm = strata['single_token'], strata['multi_token']
            if ms.sum() and mm.sum():
                fc = frequency_contrast(correct[ms], correct[mm], B=self.config.N_BOOTSTRAP)
                rows.append({'model': model_name, 'layer': li,
                             'relative_depth': (li + 1) / num_layers,
                             'stratum': 'single_minus_multi_contrast',
                             'accuracy_difference': fc['accuracy_difference'],
                             'accuracy_difference_ci95_low': fc['accuracy_difference_ci95_low'],
                             'accuracy_difference_ci95_high': fc['accuracy_difference_ci95_high'],
                             'cohens_h': fc['cohens_h'], 'p_value': fc['p_value'],
                             'p_value_log10': fc['p_value_log10'],
                             'representation_type': PRIMARY_REPRESENTATION})
        blk = [r for r in rows if r['stratum'] == 'single_minus_multi_contrast']
        for r, v in zip(blk, adjust_pvalues([r['p_value'] for r in blk])):
            r['p_value_fdr_bh'] = float(v) if np.isfinite(v) else np.nan
        return rows

    # ────────────────────────────────────────────────────────────────────
    # B5 / B6 — frequency and length × frequency in the lexical evidence
    # ────────────────────────────────────────────────────────────────────
    def frequency_covariate_models(self, oof_cache, targets, model_name,
                                   num_layers) -> dict:
        """
        B5 (corrected). The master list asked for `P(word) ~ h + log_freq`
        over words AND nonwords — that model is not estimable: nonwords have
        no corpus frequency (log_freq is undefined, not zero). Two
        well-posed versions are fitted per layer instead, on the PRIMARY
        out-of-fold lexical evidence S = logit P(WORD) (no refitting):

          (i)  words only:  S ~ z(log_freq) + z(length) + z(Ortho_N) + z(tokens)
               → does lexical evidence scale with frequency BEYOND the
               surface covariates? (OLS, HC3 SEs, BIC Bayes factor)
          (ii) all items:   is_word ~ covariates  vs  is_word ~ covariates + S
               (logistic, likelihood-ratio χ²(1), ΔAUC)
               → does S carry lexicality information beyond length /
               Ortho_N / token count, i.e. beyond what these confounds give?

        B6. (i) + z(log_freq) × z(length) interaction (centred predictors).
        """
        y = np.asarray(targets['is_word']).astype(int)
        cov = pd.DataFrame({'log_freq': np.asarray(targets['log_frequency'], float),
                            'length': np.asarray(targets['length'], float),
                            'ortho_n': np.asarray(targets['ortho_n'], float),
                            'token_count': np.asarray(targets['token_count'], float)})
        z = (cov - cov.mean()) / cov.std(ddof=0).replace(0, 1.0)
        wmask = (y == 1) & z.notna().all(1).to_numpy()
        amask = z[['length', 'ortho_n', 'token_count']].notna().all(1).to_numpy()
        eps = self.config.LEXICAL_EVIDENCE_CLIP
        freq_rows, lik_rows = [], []
        for li in sorted(oof_cache):
            p = np.clip(oof_cache[li][0], eps, 1 - eps)
            S = np.log(p / (1 - p))
            base = {'model': model_name, 'layer': li, 'relative_depth': (li + 1) / num_layers,
                    'representation_type': PRIMARY_REPRESENTATION, 'readout_type': PRIMARY_READOUT}
            if wmask.sum() >= 2 * self.config.MIN_SAMPLES_PER_GROUP:
                Xw = z.loc[wmask, ['log_freq', 'length', 'ortho_n', 'token_count']]
                r5 = ols_hc3(S[wmask], Xw, 'log_freq')
                Xi = Xw.copy()
                Xi['log_freq_x_length'] = Xi['log_freq'] * Xi['length']
                r6 = ols_hc3(S[wmask], Xi, 'log_freq_x_length')
                row = dict(base, n_words=int(wmask.sum()))
                row.update({f'freq_{k}': v for k, v in r5.items() if k != 'n'})
                row.update({f'len_x_freq_{k}': v for k, v in r6.items() if k != 'n'})
                row['freq_bf01_interpretation'] = bf_label(r5['log10_bf01'],
                                                           self.config.BF_EVIDENCE_THRESHOLD)
                row['len_x_freq_bf01_interpretation'] = bf_label(r6['log10_bf01'],
                                                                 self.config.BF_EVIDENCE_THRESHOLD)
                freq_rows.append(row)
            if amask.sum() > 50 and len(np.unique(y[amask])) == 2:
                Xa = z.loc[amask, ['length', 'ortho_n', 'token_count']].to_numpy()
                ya = y[amask]
                try:
                    m0 = sm.Logit(ya, sm.add_constant(Xa)).fit(disp=0, maxiter=200)
                    m1 = sm.Logit(ya, sm.add_constant(np.column_stack([Xa, S[amask]]))).fit(
                        disp=0, maxiter=200)
                    lr = 2 * (m1.llf - m0.llf)
                    p_lr, lp_lr = p_from_chi2(lr, 1)
                    auc0 = roc_auc_score(ya, m0.predict())
                    auc1 = roc_auc_score(ya, m1.predict())
                    lik_rows.append(dict(base, n_items=int(amask.sum()),
                                         covariates_only_auc=float(auc0),
                                         covariates_plus_evidence_auc=float(auc1),
                                         delta_auc=float(auc1 - auc0),
                                         lr_chi2=float(lr), lr_p=p_lr, lr_p_log10=lp_lr,
                                         evidence_coef=float(m1.params[-1]),
                                         note=('in-sample 4- vs 5-parameter logistic models; '
                                               'S itself is out-of-fold')))
                except Exception as e:                    # noqa: BLE001
                    lik_rows.append(dict(base, error=str(e)))
        for rows_, cols in ((freq_rows, ('freq_p_hc3', 'len_x_freq_p_hc3')),
                            (lik_rows, ('lr_p',))):
            for c in cols:
                for r, v in zip(rows_, adjust_pvalues([r.get(c, np.nan) for r in rows_])):
                    r[c + '_fdr_bh'] = float(v) if np.isfinite(v) else np.nan
        return {'words_only_frequency': freq_rows, 'incremental_lexicality': lik_rows}

    # ────────────────────────────────────────────────────────────────────
    # Analysis 10: Contextual Frequency Effect (Sentence Context)
    # ────────────────────────────────────────────────────────────────────
    def contextual_analysis(self, all_hidden, targets, extractor,
                            model_name) -> dict:
        """
        Analysis 10 + F1–F3: is the HF/LF frequency distinction still
        decodable when the word is read in context, and WHERE?

        Conditions (HF vs LF words; identical items and folds; every item
        scored out-of-fold; McNemar vs 'isolated' on the same items):
          isolated                  PRIMARY task prompt, first-output prediction
                                    position (cached, no new forward pass)
          sentence_target           carrier sentence, NO task prompt, mean over
                                    the target word's tokens (legacy control)
          sentence_final            same sentence, FINAL token (context readout)
          task_sentence_final       F1: sentence INSIDE the task prompt, final
                                    'Answer:' position
          task_sentence_target      F1: same prompt, target-word position
          final_position_target     F2: carrier ending in the target, so the
                                    final position IS the target (position ≠
                                    context control)
          filler_shuffled_target    F3 (corrected): the carrier holds a
                                    DIFFERENT word from the sample (random
                                    derangement) while labels stay with the
                                    original item → must be at chance.
        NOTE on F3 as written in the master list ("same carrier length, no
        target word"): with no target word every input is identical, so every
        hidden state is identical and chance accuracy holds by construction —
        a vacuous control. The derangement version keeps the input
        distribution but breaks the item→label link, which is the informative
        negative control.
        Requires the live model (new forward passes).
        """
        logger.info(f"  [Analysis 10 / F1–F3] Contextual frequency effect — {model_name}")
        cfg = self.config
        ctx_ds = ContextualLDTDataset(targets, cfg)
        empty = {'isolated': [], 'contextual': [], 'conditions': [],
                 'n_high': ctx_ds.n_high, 'n_low': ctx_ds.n_low, 'n_located_in_sentence': 0}
        if ctx_ds.n_high < cfg.MIN_SAMPLES_PER_GROUP or ctx_ds.n_low < cfg.MIN_SAMPLES_PER_GROUP:
            logger.warning("  Insufficient high/low words for contextual analysis")
            return empty
        y = ctx_ds.freq_label
        words = [str(w) for w in ctx_ds.words]
        fa = FrequencyAnalyzer(cfg)
        folds = fa.make_folds(y, seed=cfg.CONTEXT_SEED)
        layers = sorted(all_hidden.keys())

        def _slot(template, w, slot='{}'):
            pre = template.split(slot)[0]
            return template.replace(slot, w), (len(pre), len(pre) + len(w))
        sent = [_slot(cfg.CONTEXT_SENTENCE_TEMPLATE, w) for w in words]
        fin_t = [_slot(cfg.CONTEXT_FINAL_POSITION_TEMPLATE, w) for w in words]
        task = []
        for w, (s_txt, (s0, s1)) in zip(words, sent):
            pre = cfg.CONTEXT_TASK_TEMPLATE.split('{SENTENCE}')[0]
            txt = cfg.CONTEXT_TASK_TEMPLATE.replace('{SENTENCE}', s_txt).replace('{STIMULUS}', w)
            task.append((txt, (len(pre) + s0, len(pre) + s1)))
        rng = np.random.RandomState(cfg.CONTEXT_SEED + 7)
        perm = rng.permutation(len(words))
        while np.any(perm == np.arange(len(words))):          # derangement
            perm = rng.permutation(len(words))
        filler = [_slot(cfg.CONTEXT_SENTENCE_TEMPLATE, words[j]) for j in perm]

        reps = {'isolated': {li: all_hidden[li][ctx_ds.positions_in_all_hidden] for li in layers}}
        found = {}
        try:
            for cname, items, keys in (
                    ('sentence', sent, ('sentence_final', 'sentence_target')),
                    ('task_sentence', task, ('task_sentence_final', 'task_sentence_target')),
                    ('final_position', fin_t, ('final_position_final', 'final_position_target')),
                    ('filler', filler, (None, 'filler_shuffled_target'))):
                st = extractor.extract_final_and_span_states(
                    [t for t, _ in items], [sp for _, sp in items], layers=layers,
                    desc=f'F {cname}')
                if keys[0]:
                    reps[keys[0]] = st['final']
                reps[keys[1]] = st['span']
                found[keys[1]] = st['span_found']
        except Exception as e:                                 # noqa: BLE001
            logger.error(f"  Contextual extraction failed: {e}", exc_info=True)
        reps.pop('final_position_final', None)   # identical position to _target by design

        cond_rows, corr = [], {}
        for cname, R in reps.items():
            m = found.get(cname, np.ones(len(y), bool))
            for li in layers:
                X = R[li]
                yy, ff = y[m], folds[m]
                row = {'model': model_name, 'condition': cname, 'layer': li,
                       'n': int(m.sum()), 'n_high': int(yy.sum()), 'n_low': int((1 - yy).sum()),
                       'scoring': 'out_of_fold_only'}
                if m.sum() < 2 * cfg.MIN_SAMPLES_PER_GROUP or len(np.unique(yy)) < 2:
                    cond_rows.append(dict(row, accuracy=np.nan, auc=np.nan, f1=np.nan))
                    continue
                p, _ = fa.crossfit_word_probabilities(X[m], yy, fold_of=ff)
                pred = (p >= 0.5).astype(int)
                k = int((pred == yy).sum())
                lo, hi = wilson_ci(k, len(yy))
                pb, pbl = p_binom_greater(k, len(yy), 0.5)
                row.update({'accuracy': k / len(yy), 'accuracy_ci95_low': lo,
                            'accuracy_ci95_high': hi, 'auc': float(roc_auc_score(yy, p)),
                            'f1': float(f1_score(yy, pred, zero_division=0)),
                            'p_vs_chance': pb, 'p_vs_chance_log10': pbl})
                full = np.full(len(y), np.nan)
                full[m] = (pred == yy)
                corr[(cname, li)] = full
                cond_rows.append(row)
        for r in cond_rows:
            a = corr.get((r['condition'], r['layer']))
            b = corr.get(('isolated', r['layer']))
            if r['condition'] != 'isolated' and a is not None and b is not None:
                ok = np.isfinite(a) & np.isfinite(b)
                r['mcnemar_vs_isolated_p'] = mcnemar_exact(a[ok], b[ok])[2]
        for cname in reps:
            blk = [r for r in cond_rows if r['condition'] == cname]
            for col, meth in (('p_vs_chance', 'fdr_bh'), ('mcnemar_vs_isolated_p', 'holm')):
                for r, v in zip(blk, adjust_pvalues([r.get(col, np.nan) for r in blk], meth)):
                    if np.isfinite(v):
                        r[col + ('_fdr_bh' if meth == 'fdr_bh' else '_holm')] = float(v)

        def _legacy(cname):
            return [{'layer': r['layer'], 'accuracy': r.get('accuracy', np.nan),
                     'accuracy_ci95_low': r.get('accuracy_ci95_low', np.nan),
                     'accuracy_ci95_high': r.get('accuracy_ci95_high', np.nan),
                     'f1': r.get('f1', np.nan), 'auc': r.get('auc', np.nan), 'n': r['n']}
                    for r in cond_rows if r['condition'] == cname]
        return {
            'isolated': _legacy('isolated'),
            'contextual': _legacy('sentence_target'),
            'conditions': cond_rows,
            'n_high': int(ctx_ds.n_high), 'n_low': int(ctx_ds.n_low),
            'n_located_in_sentence': int(found.get('sentence_target', np.zeros(0, bool)).sum()),
            'isolated_representation_type': PRIMARY_REPRESENTATION,
            'contextual_representation_type': RepresentationType.CONTEXTUAL_TARGET_TOKEN.value,
        }

    def _small_sample_freq_probe(self, X, y, layer_idx) -> dict:
        """Kept for backward compatibility; the contextual analysis now uses
        5-fold cross-fitted scoring with Wilson CIs instead."""
        if len(np.unique(y)) < 2 or len(y) < self.config.MIN_SAMPLES_PER_GROUP:
            return {'layer': layer_idx, 'accuracy': np.nan, 'f1': np.nan,
                    'auc': np.nan, 'n': len(y)}
        fa = FrequencyAnalyzer(self.config)
        p, _ = fa.crossfit_word_probabilities(X, y)
        pred = (p >= 0.5).astype(int)
        return {'layer': layer_idx, 'accuracy': float(np.mean(pred == y)),
                'f1': float(f1_score(y, pred, zero_division=0)),
                'auc': float(roc_auc_score(y, p)), 'n': len(y)}


# ════════════════════════════════════════════════════════════════════════════
# A6 (ADDED) — SURFACE-FORM BASELINES: how far do you get WITHOUT the LLM?
# ════════════════════════════════════════════════════════════════════════════
class SurfaceFormBaselines:
    """
    The single most likely reviewer objection to a high WORD/NONWORD probe
    accuracy is that the distinction is recoverable from the letter string
    (ELP pseudowords are orthographically legal but still statistically
    different) or from the tokenizer (words are more often single tokens).
    These baselines use NO hidden state, the SAME items and the SAME folds as
    the primary probe, so every comparison is an exact paired McNemar test:

      surface_covariates : length, sub-word token count, Ortho_N (+ missing flag)
      char_ngrams        : hashed character 1–3-grams (orthographic regularity)
      subword_tokens     : bag of the MODEL'S OWN sub-word token ids + count
    The primary claim is only as strong as primary − best baseline.
    """

    def __init__(self, config: Config):
        self.config = config

    def _features(self, targets, tokenizer) -> dict:
        from scipy import sparse
        from sklearn.feature_extraction.text import HashingVectorizer
        stim = [str(s) for s in targets['stimulus']]
        on = np.asarray(targets['ortho_n'], float)
        miss = np.isnan(on).astype(float)
        dense = np.column_stack([np.asarray(targets['length'], float),
                                 np.asarray(targets['token_count'], float), on, miss])
        feats = {'surface_covariates': dense}
        hv = HashingVectorizer(analyzer='char_wb', ngram_range=tuple(self.config.CHAR_NGRAM_RANGE),
                               n_features=self.config.CHAR_NGRAM_FEATURES,
                               alternate_sign=False, binary=True, norm='l2')
        feats['char_ngrams'] = hv.transform(stim)
        if tokenizer is not None:
            ids = [tokenizer.encode(s, add_special_tokens=False) for s in stim]
            V = max(len(tokenizer), max((max(i) for i in ids if i), default=0) + 1)
            r = np.repeat(np.arange(len(ids)), [len(i) for i in ids])
            c = np.concatenate([np.asarray(i, int) for i in ids]) if len(r) else np.zeros(0, int)
            bag = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(ids), V))
            tc = sparse.csr_matrix(np.asarray([[len(i)] for i in ids], float) / 10.0)
            feats['subword_tokens'] = sparse.hstack([bag, tc]).tocsr()
        return feats

    def _crossfit(self, X, y, folds) -> np.ndarray:
        from scipy import sparse
        oof = np.full(len(y), np.nan)
        for k in np.unique(folds):
            te, tr = folds == k, folds != k
            if sparse.issparse(X):
                clf = LogisticRegression(C=self.config.PRIMARY_PROBE_C, solver='liblinear',
                                         max_iter=2000, random_state=SEED)
                clf.fit(X[tr], y[tr])
                oof[te] = clf.predict_proba(X[te])[:, 1]
            else:
                Xtr, Xte = X[tr].copy(), X[te].copy()
                med = np.nanmedian(Xtr, 0)
                for j in range(X.shape[1]):
                    Xtr[np.isnan(Xtr[:, j]), j] = med[j]
                    Xte[np.isnan(Xte[:, j]), j] = med[j]
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(C=self.config.PRIMARY_PROBE_C, max_iter=1000,
                                         random_state=SEED)
                clf.fit(sc.transform(Xtr), y[tr])
                oof[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
        return oof

    def run(self, targets, tokenizer, oof_cache, model_name, num_layers,
            fold_ids, bootstrap_fn) -> list[dict]:
        y = np.asarray(targets['is_word']).astype(int)
        rows = []
        feats = self._features(targets, tokenizer)
        prim = {li: ((p >= 0.5).astype(int) == y).astype(int) for li, (p, _) in oof_cache.items()}
        prim_acc = {li: c.mean() for li, c in prim.items()}
        best_li = max(prim_acc, key=prim_acc.get) if prim_acc else None
        for name, X in feats.items():
            try:
                p = self._crossfit(X, y, fold_ids)
            except Exception as e:                           # noqa: BLE001
                logger.warning(f"  [A6] {name} baseline failed: {e}")
                continue
            pred = (p >= 0.5).astype(int)
            corr = (pred == y).astype(int)
            lo, hi = bootstrap_fn(corr.astype(float))
            base = {'model': model_name, 'baseline': name, 'n': int(len(y)),
                    'accuracy': float(corr.mean()), 'accuracy_ci95_low': lo,
                    'accuracy_ci95_high': hi,
                    'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
                    'auroc': float(roc_auc_score(y, p)),
                    'uses_llm_hidden_states': False, 'scoring': 'out_of_fold_same_folds_as_primary'}
            comp = []
            for li in sorted(prim):
                b_, c_, pv = mcnemar_exact(prim[li], corr)
                comp.append({'layer': li, 'primary_minus_baseline_accuracy': float(prim_acc[li] - corr.mean()),
                             'mcnemar_primary_only_correct': b_, 'mcnemar_baseline_only_correct': c_,
                             'mcnemar_p': pv})
            adj = adjust_pvalues([c['mcnemar_p'] for c in comp], 'holm', self.config.ALPHA)
            for c, v in zip(comp, adj):
                c['mcnemar_p_holm'] = float(v)
            n_sig_better = sum(1 for c in comp if c['primary_minus_baseline_accuracy'] > 0
                               and c['mcnemar_p_holm'] < self.config.ALPHA)
            base.update({'best_primary_layer': best_li,
                         'best_primary_accuracy': float(prim_acc[best_li]) if best_li is not None else np.nan,
                         'best_primary_minus_baseline': (float(prim_acc[best_li] - corr.mean())
                                                         if best_li is not None else np.nan),
                         'n_layers_primary_significantly_better_holm': int(n_sig_better)})
            if best_li is not None:
                base['best_layer_mcnemar_p'] = next(c['mcnemar_p'] for c in comp if c['layer'] == best_li)
            rows.append(dict(base, row_type='baseline_summary'))
            rows.extend(dict(c, model=model_name, baseline=name, row_type='per_layer_comparison',
                             relative_depth=(c['layer'] + 1) / num_layers) for c in comp)
        return rows


# ════════════════════════════════════════════════════════════════════════════
# NAACL REVISION — LIVE-MODEL ANALYSES (A3, C2, C3, C5, D1, D2, D3)
# ════════════════════════════════════════════════════════════════════════════
def fit_linear_direction(X: np.ndarray, y: np.ndarray, C: float = 1.0,
                         seed: int = SEED, device: str | None = None) -> dict:
    """L2 logistic regression on standardised X. Returns the probe in the
    standardised space (mu, sd, w, b) AND the equivalent RAW-space unit
    direction u = (w/σ)/‖w/σ‖, because logit(x) = w·((x − μ)/σ) + b is linear
    in x with gradient w/σ. `proj_sd` is the SD of the raw data projected on
    u — interventions are expressed in units of this SD, which makes their
    size comparable across layers and models (the residual-stream norm grows
    with depth, so raw-unit or z-unit magnitudes are not comparable)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y).astype(int)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    if device is None and torch.cuda.is_available():
        device = 'cuda'
    if device is not None:
        # same convex objective as sklearn (see FrequencyAnalyzer), in torch
        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        wt, bt = FrequencyAnalyzer.logistic_torch_standardised(
            Xt, torch.as_tensor(y, dtype=torch.float32, device=device), C)
        w, b = wt.double().cpu().numpy(), float(bt.cpu().numpy()[0])
        del Xt
    else:
        clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', random_state=seed)
        clf.fit(Xs, y)
        w, b = clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
    g = w / sd
    nrm = np.linalg.norm(g)
    u = g / nrm if nrm > 0 else g
    acc = float(np.mean(((Xs @ w + b) > 0).astype(int) == y))
    return {'mu': mu, 'sd': sd, 'w': w, 'b': b, 'u': u,
            'proj_sd': float(np.std(X @ u)), 'train_accuracy': acc}


def probe_logits(probe: dict, X: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, np.float64) - probe['mu']) / probe['sd']) @ probe['w'] + probe['b']


def inlp_subspace(X: np.ndarray, y: np.ndarray, k: int, C: float = 1.0,
                  seed: int = SEED) -> np.ndarray:
    """Iterative nullspace projection (Ravfogel et al., 2020): fit a linear
    probe, take its raw-space direction, project it out, repeat k times.
    Returns an orthonormal basis U (H × k') of the removed subspace."""
    Xw = np.asarray(X, np.float64).copy()
    dirs = []
    for i in range(k):
        d = fit_linear_direction(Xw, y, C=C, seed=seed + i)['u']
        if not np.all(np.isfinite(d)) or np.linalg.norm(d) == 0:
            break
        dirs.append(d)
        Q, _ = np.linalg.qr(np.column_stack(dirs))
        Xw = Xw - (Xw @ Q) @ Q.T
    if not dirs:
        return np.zeros((X.shape[1], 0))
    Q, _ = np.linalg.qr(np.column_stack(dirs))
    return Q


def _hook_out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _hook_replace_out(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


class ModelLevelAnalyses:
    """Analyses that need the LIVE, frozen model (run before it is freed, J6).
    Every intervention is applied ONLY at the first-output prediction
    position of a single layer, never trains the LLM, and is evaluated on
    items that were NOT used to fit the direction / probe it manipulates.

    Outcome variables
      * native decision variable  m = logit(YES) − logit(NO)  of the model's
        own LM head at the final layer (C2, C3, C5, D2);
      * FIXED linear probes (trained on disjoint items) read at a downstream
        layer (C3);
      * cross-fitted probe accuracy (A3, D3).
    Nulls: random directions / subspaces / heads / knocked-out tokens of the
    SAME size (C1–C5, D2) — an effect is interpretable only relative to them.
    """

    def __init__(self, config: Config, experiment: 'MultiModelExperiment'):
        self.config = config
        self.exp = experiment
        self.analyzer = FrequencyAnalyzer(config)

    # ── sampling ────────────────────────────────────────────────────────
    def balanced_rows(self, targets, n, seed=SEED, exclude=None) -> np.ndarray:
        """n/2 nonwords + n/2 words; the words are split equally over the
        high / low frequency tertiles (remainder: mid), so frequency contrasts
        are powered. Deterministic for a given seed."""
        y = np.asarray(targets['is_word']).astype(int)
        fg = np.asarray(targets['freq_group'], dtype=object)
        rng = np.random.RandomState(seed)
        avail = np.ones(len(y), bool)
        if exclude is not None:
            avail[np.asarray(exclude, int)] = False

        def _take(mask, k):
            idx = np.where(mask & avail)[0]
            if k <= 0 or len(idx) == 0:
                return idx[:0]
            return rng.choice(idx, size=min(k, len(idx)), replace=False)
        n_w, n_nw = n // 2, n - n // 2
        hf = _take((y == 1) & (fg == 'high'), n_w // 3)
        lf = _take((y == 1) & (fg == 'low'), n_w // 3)
        rest = n_w - len(hf) - len(lf)
        mid = _take((y == 1) & (fg == 'mid'), rest)
        nw = _take(y == 0, n_nw)
        return np.sort(np.concatenate([hf, lf, mid, nw]).astype(int))

    def fit_rows(self, targets, eval_rows, max_items=None, seed=SEED + 1) -> np.ndarray:
        max_items = max_items or self.config.MODEL_LEVEL_FIT_MAX_ITEMS
        rest = np.setdiff1d(np.arange(len(targets['is_word'])), eval_rows)
        if len(rest) > max_items:
            rng = np.random.RandomState(seed)
            rest = np.sort(rng.choice(rest, size=max_items, replace=False))
        return rest

    @staticmethod
    def _prompts(targets, rows) -> list[str]:
        return [build_task_prompt(str(targets['stimulus'][i])) for i in rows]

    @staticmethod
    def _stimulus_spans(targets, rows, template=TASK_PROMPT_TEMPLATE):
        pre = template.split('{STIMULUS}')[0]
        return [(len(pre), len(pre) + len(str(targets['stimulus'][i]))) for i in rows]

    def _layers(self, num_layers, cap=True) -> list[int]:
        return select_layers(num_layers, self.config.MODEL_LEVEL_LAYERS,
                             self.config.MODEL_LEVEL_MAX_LAYERS if cap else None)

    # ── generic batched runner with transient hooks ─────────────────────
    @torch.no_grad()
    def _run(self, ex, texts, install=None, state_layers=(), want_margin=True):
        bs = self.config.MODEL_LEVEL_BATCH_SIZE
        margins, states = [], {li: [] for li in state_layers}
        can_margin = want_margin and ex._lm_head is not None
        for s in range(0, len(texts), bs):
            ids, am, fp, _ = ex.encode_task_batch(texts[s:s + bs])
            handles = install(fp) if install is not None else []
            try:
                hs = ex.backbone_hidden(ids, am)
            finally:
                for h in handles:
                    h.remove()
            ar = torch.arange(ids.shape[0], device=hs[-1].device)
            fpd = fp.to(hs[-1].device)
            if can_margin:
                margins.append(ex.answer_margin(hs[-1][ar, fpd]))
            else:
                margins.append(np.full(ids.shape[0], np.nan))
            for li in state_layers:
                states[li].append(ex._finite_numpy(hs[li + 1][ar, fp.to(hs[li + 1].device)]))
            del hs
        if ex.device == 'cuda':
            torch.cuda.empty_cache()
        return (np.concatenate(margins) if margins else np.array([]),
                {li: np.concatenate(v, 0) for li, v in states.items()})

    def _margin_summary(self, m, y, fg, prefix) -> dict:
        m = np.asarray(m, float)
        out = {}
        for name, mask in (('all', np.ones(len(y), bool)), ('words', y == 1),
                           ('nonwords', y == 0), ('hf_words', (y == 1) & (fg == 'high')),
                           ('lf_words', (y == 1) & (fg == 'low'))):
            out[f'{prefix}_mean_margin_{name}'] = float(np.nanmean(m[mask])) if mask.any() else np.nan
        pred = (m > 0).astype(int)
        ok = np.isfinite(m)
        out[f'{prefix}_native_accuracy'] = float(np.mean(pred[ok] == y[ok])) if ok.any() else np.nan
        hf, lf = (y == 1) & (fg == 'high') & ok, (y == 1) & (fg == 'low') & ok
        out[f'{prefix}_native_hf_minus_lf_accuracy'] = (
            float(np.mean(pred[hf] == 1) - np.mean(pred[lf] == 1))
            if hf.any() and lf.any() else np.nan)
        return out

    @staticmethod
    def _paired_delta_test(d) -> dict:
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        if len(d) < 3 or np.allclose(d, 0):
            return {'mean_delta': float(np.mean(d)) if len(d) else np.nan,
                    'delta_t': np.nan, 'delta_p': np.nan, 'delta_p_log10': np.nan}
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else np.nan
        p, lp = p_from_t(t, len(d) - 1)
        return {'mean_delta': float(d.mean()), 'delta_t': t, 'delta_p': p, 'delta_p_log10': lp}

    # ════════════════════════════════════════════════════════════════════
    # A3 — SUBLAYER DECOMPOSITION at the first-output prediction position
    # ════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def sublayer_decomposition(self, ex, targets, model_name) -> list[dict]:
        """For pre-norm blocks  h_out = h_in + Attn(·) + MLP(·). We probe, at
        the SAME position and on the SAME items/folds:
            resid_pre  = h_in (block input), attn_out, mlp_out,
            block_out  = h_out (raw, pre-final-norm)
        and verify the additive identity numerically (reported). A component
        that is decodable only through h_in carries information written by
        EARLIER layers; attn_out/mlp_out isolate what THIS layer writes."""
        rows = self.balanced_rows(targets, self.config.SUBLAYER_N_ITEMS, seed=SEED + 11)
        y = np.asarray(targets['is_word']).astype(int)[rows]
        layers = self._layers(ex.num_layers, cap=False)
        dec = ex.decoder_layers()
        comps = ('resid_pre', 'attn_out', 'mlp_out', 'block_out')
        store = {c: {li: [] for li in layers} for c in comps}
        state = {}

        def _grab(c, li, h):
            fp = state['fp'].to(h.device)
            ar = torch.arange(h.shape[0], device=h.device)
            store[c][li].append(ex._finite_numpy(h[ar, fp]))
        handles = []
        try:
            for li in layers:
                layer = dec[li]

                def pre(mod, args, kwargs, li=li):
                    h = args[0] if args else kwargs.get('hidden_states')
                    _grab('resid_pre', li, h)
                handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
                handles.append(layer.register_forward_hook(
                    lambda m, i, o, li=li: _grab('block_out', li, _hook_out_tensor(o))))
                handles.append(ex.attn_module(layer).register_forward_hook(
                    lambda m, i, o, li=li: _grab('attn_out', li, _hook_out_tensor(o))))
                handles.append(ex.mlp_module(layer).register_forward_hook(
                    lambda m, i, o, li=li: _grab('mlp_out', li, _hook_out_tensor(o))))
            texts = self._prompts(targets, rows)
            bs = self.config.MODEL_LEVEL_BATCH_SIZE
            for s in tqdm(range(0, len(texts), bs), desc=f"A3 sublayers | {model_name}"):
                ids, am, fp, _ = ex.encode_task_batch(texts[s:s + bs])
                state['fp'] = fp
                ex.backbone_hidden(ids, am)
        finally:
            for h in handles:
                h.remove()
        reps = {c: {li: np.concatenate(v, 0) for li, v in store[c].items()} for c in comps}
        additivity = {}
        for li in layers:
            recon = reps['resid_pre'][li] + reps['attn_out'][li] + reps['mlp_out'][li]
            err = np.abs(recon - reps['block_out'][li]).mean()
            scale = np.abs(reps['block_out'][li]).mean() + 1e-12
            additivity[li] = float(err / scale)
        folds = self.analyzer.make_folds(y)
        sub_t = {k: np.asarray(v)[rows] if isinstance(v, np.ndarray) else [v[i] for i in rows]
                 for k, v in targets.items()}
        tables, corrects = {}, {}
        for c in comps:
            tbl, oof = self.exp._crossfit_table(
                model_name, reps[c], sub_t, self.analyzer, ex.num_layers, ex.hidden_dim,
                layers=layers, fold_ids=folds, representation_type=f'sublayer_{c}',
                readout=PRIMARY_READOUT, desc=f'A3 {c}')
            tables[c] = {r['layer']: r for r in tbl}
            corrects[c] = {li: ((p >= 0.5).astype(int) == y).astype(int) for li, (p, _) in oof.items()}
        out = []
        for c in comps:
            block = []
            for li in layers:
                r = dict(tables[c].get(li, {'model': model_name, 'layer': li}))
                r.update({'component': c, 'n_items': int(len(rows)),
                          'additivity_relative_error': additivity[li],
                          'additivity_holds': bool(additivity[li] < 2e-2)})
                if c != 'block_out' and li in corrects[c] and li in corrects['block_out']:
                    b_, c_, p = mcnemar_exact(corrects[c][li], corrects['block_out'][li])
                    r.update({'accuracy_minus_block_out': r.get('accuracy', np.nan)
                              - tables['block_out'][li].get('accuracy', np.nan),
                              'mcnemar_vs_block_out_p': p})
                block.append(r)
            adj = adjust_pvalues([r.get('mcnemar_vs_block_out_p', np.nan) for r in block],
                                 'holm', self.config.ALPHA)
            for r, v in zip(block, adj):
                if np.isfinite(v):
                    r['mcnemar_vs_block_out_p_holm'] = float(v)
            out.extend(block)
        del reps, store
        gc.collect()
        return out

    # ════════════════════════════════════════════════════════════════════
    # D1 — ATTENTION MASS from the first-output prediction position
    # ════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def attention_mass(self, ex, targets, model_name) -> dict:
        """Head-averaged attention of the final prompt position over five
        segments: BOS (attention sink), instruction prefix, STIMULUS, suffix,
        and the final token itself. Stimulus mass is also reported PER
        STIMULUS TOKEN, because nonwords have more sub-tokens than words and
        raw mass is confounded with token count (Xiao et al., 2024 on sinks)."""
        rows = self.balanced_rows(targets, self.config.MODEL_LEVEL_N_ITEMS, seed=SEED + 21)
        y = np.asarray(targets['is_word']).astype(int)[rows]
        fg = np.asarray(targets['freq_group'], dtype=object)[rows]
        texts = self._prompts(targets, rows)
        spans = self._stimulus_spans(targets, rows)
        L = ex.num_layers
        segs = ('bos', 'prefix', 'stimulus', 'suffix', 'final_token')
        mass = np.full((len(rows), L, len(segs)), np.nan, np.float32)
        n_stim = np.zeros(len(rows), int)
        bs = self.config.MODEL_LEVEL_BATCH_SIZE
        bos_id = ex.tokenizer.bos_token_id
        with ex.instrumented_attention() as ctx:
            for s in tqdm(range(0, len(texts), bs), desc=f"D1 attention | {model_name}"):
                ids, am, fp, smask = ex.encode_task_batch(texts[s:s + bs], spans[s:s + bs])
                ctx.query_pos, ctx.capture, ctx.block, ctx.layers = fp, {}, None, None
                ex.backbone_hidden(ids, am)
                cap = ctx.capture
                B, T = ids.shape
                pos = torch.arange(T).unsqueeze(0).expand(B, T)
                real = am.bool().cpu()
                sm = smask.cpu()
                first_real = torch.where(real, pos, torch.full_like(pos, T)).min(1).values
                fpc = fp.cpu()
                is_bos = (pos == first_real[:, None]) & (ids.cpu() == bos_id) if bos_id is not None \
                    else torch.zeros_like(real)
                stim_start = torch.where(sm, pos, torch.full_like(pos, T)).min(1).values
                seg_masks = {
                    'bos': is_bos,
                    'prefix': real & ~is_bos & ~sm & (pos < stim_start[:, None]),
                    'stimulus': sm,
                    'suffix': real & ~sm & (pos > stim_start[:, None]) & (pos < fpc[:, None]),
                    'final_token': pos == fpc[:, None],
                }
                for li in range(L):
                    a = cap.get(li)
                    if a is None:
                        continue
                    a = a[:, :T].float()
                    for j, sgn in enumerate(segs):
                        mass[s:s + B, li, j] = (a * seg_masks[sgn].float()).sum(1).numpy()
                n_stim[s:s + B] = sm.sum(1).numpy()
                ctx.capture = {}
        if np.isnan(mass).all():
            return {'long': [], 'tests': [], 'note': 'attention capture failed'}
        per_tok = mass[:, :, segs.index('stimulus')] / np.maximum(n_stim, 1)[:, None]
        long_rows, tests = [], []
        groups = {'word': y == 1, 'nonword': y == 0,
                  'hf_word': (y == 1) & (fg == 'high'), 'lf_word': (y == 1) & (fg == 'low')}
        for li in range(L):
            for j, sgn in enumerate(segs + ('stimulus_per_token',)):
                v = per_tok[:, li] if sgn == 'stimulus_per_token' else mass[:, li, j]
                for gname, gm in groups.items():
                    vv = v[gm]
                    vv = vv[np.isfinite(vv)]
                    long_rows.append({'model': model_name, 'layer': li,
                                      'relative_depth': (li + 1) / L, 'segment': sgn,
                                      'group': gname, 'mean_attention': float(vv.mean()) if len(vv) else np.nan,
                                      'sem': float(vv.std(ddof=1) / np.sqrt(len(vv))) if len(vv) > 1 else np.nan,
                                      'n': int(len(vv))})
            for metric, v in (('stimulus_mass', mass[:, li, segs.index('stimulus')]),
                              ('stimulus_mass_per_token', per_tok[:, li])):
                for a_name, b_name in (('word', 'nonword'), ('hf_word', 'lf_word')):
                    a, b = v[groups[a_name]], v[groups[b_name]]
                    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                    if len(a) > 2 and len(b) > 2:
                        t, _ = stats.ttest_ind(a, b, equal_var=False)
                        dfw = ((a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 /
                               ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                                + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1) + 1e-300))
                        p, lp = p_from_t(t, dfw)
                        tests.append({'model': model_name, 'layer': li, 'metric': metric,
                                      'contrast': f'{a_name}_minus_{b_name}',
                                      'mean_difference': float(a.mean() - b.mean()),
                                      'smd': standardized_mean_difference(a, b),
                                      'welch_t': float(t), 'p': p, 'p_log10': lp})
        for metric in ('stimulus_mass', 'stimulus_mass_per_token'):
            for con in ('word_minus_nonword', 'hf_word_minus_lf_word'):
                blk = [r for r in tests if r['metric'] == metric and r['contrast'] == con]
                for r, v in zip(blk, adjust_pvalues([r['p'] for r in blk], 'fdr_bh')):
                    r['p_fdr_bh'] = float(v) if np.isfinite(v) else np.nan
        return {'long': long_rows, 'tests': tests, 'n_items': int(len(rows)),
                'segments': list(segs)}

    # ════════════════════════════════════════════════════════════════════
    # D2 — ATTENTION KNOCKOUT (final position ↛ stimulus tokens at layer l)
    # ════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def attention_knockout(self, ex, targets, model_name) -> list[dict]:
        """Blocks the edges final-position → stimulus tokens at ONE layer
        (or at all layers) and measures the change in the model's own
        decision variable m = logit(YES) − logit(NO). Control: the same
        number of randomly chosen NON-stimulus prompt tokens (excluding BOS
        and the final token) is knocked out instead, so the effect is
        attributable to the stimulus, not to perturbing attention per se."""
        if not ex.ensure_answer_readout():
            return []
        rows = self.balanced_rows(targets, self.config.MODEL_LEVEL_N_ITEMS, seed=SEED + 31)
        y = np.asarray(targets['is_word']).astype(int)[rows]
        fg = np.asarray(targets['freq_group'], dtype=object)[rows]
        texts = self._prompts(targets, rows)
        spans = self._stimulus_spans(targets, rows)
        layer_sets = [(str(li), {li}) for li in self._layers(ex.num_layers)] + \
                     [('all', None)]
        bs = self.config.MODEL_LEVEL_BATCH_SIZE
        bos_id = ex.tokenizer.bos_token_id
        rng = np.random.RandomState(SEED + 32)

        def _run_blocks(mode, layer_set):
            out = []
            with ex.instrumented_attention() as ctx:
                for s in range(0, len(texts), bs):
                    ids, am, fp, smask = ex.encode_task_batch(texts[s:s + bs], spans[s:s + bs])
                    B, T = ids.shape
                    blk = None
                    if mode == 'stimulus':
                        blk = smask.clone()
                    elif mode == 'control':
                        blk = torch.zeros_like(smask)
                        pos = torch.arange(T, device=am.device)
                        for b in range(B):
                            real = am[b].bool()
                            cand = real & ~smask[b] & (pos != fp[b])
                            if bos_id is not None:
                                cand &= ~((pos == int(torch.nonzero(real)[0])) & (ids[b] == bos_id))
                            ci = torch.nonzero(cand).flatten().cpu().numpy()
                            k = int(smask[b].sum())
                            if len(ci) and k:
                                pick = rng.choice(ci, size=min(k, len(ci)), replace=False)
                                blk[b, torch.as_tensor(pick, device=blk.device)] = True
                    ctx.query_pos, ctx.block, ctx.capture = fp, blk, None
                    ctx.layers = layer_set
                    hs = ex.backbone_hidden(ids, am)
                    ar = torch.arange(B, device=hs[-1].device)
                    out.append(ex.answer_margin(hs[-1][ar, fp.to(hs[-1].device)]))
            return np.concatenate(out)

        m0 = _run_blocks('none', None)
        base = self._margin_summary(m0, y, fg, 'baseline')
        results = []
        for name, ls in tqdm(layer_sets, desc=f"D2 knockout | {model_name}"):
            m1 = _run_blocks('stimulus', ls)
            row = {'model': model_name, 'knockout_layers': name,
                   'relative_depth': ((int(name) + 1) / ex.num_layers) if name != 'all' else np.nan,
                   'n_items': int(len(rows)), **base,
                   **self._margin_summary(m1, y, fg, 'knockout')}
            d = m1 - m0
            row.update({f'knockout_{k}': v for k, v in self._paired_delta_test(np.abs(d)).items()})
            row['knockout_mean_signed_delta_words'] = float(np.mean(d[y == 1]))
            row['knockout_mean_signed_delta_nonwords'] = float(np.mean(d[y == 0]))
            c0 = ((m0 > 0).astype(int) == y).astype(int)
            c1 = ((m1 > 0).astype(int) == y).astype(int)
            row['mcnemar_accuracy_p'] = mcnemar_exact(c0, c1)[2]
            if self.config.KNOCKOUT_CONTROL:
                mc = _run_blocks('control', ls)
                dc = mc - m0
                row.update(self._margin_summary(mc, y, fg, 'control'))
                t = self._paired_delta_test(np.abs(d) - np.abs(dc))
                row.update({'stimulus_vs_control_mean_abs_delta_difference': t['mean_delta'],
                            'stimulus_vs_control_t': t['delta_t'],
                            'stimulus_vs_control_p': t['delta_p'],
                            'stimulus_vs_control_p_log10': t['delta_p_log10']})
            results.append(row)
        for col in ('stimulus_vs_control_p', 'mcnemar_accuracy_p'):
            adj = adjust_pvalues([r.get(col, np.nan) for r in results], 'holm', self.config.ALPHA)
            for r, v in zip(results, adj):
                r[col + '_holm'] = float(v) if np.isfinite(v) else np.nan
        return results

    # ════════════════════════════════════════════════════════════════════
    # D3 — STIMULUS POSITION SENSITIVITY
    # ════════════════════════════════════════════════════════════════════
    POSITION_TEMPLATES = {
        'stimulus_first': (
            "Letter string: {STIMULUS}\n"
            "\n"
            "Determine whether the letter string above is a valid English word.\n"
            "\n"
            "Respond with exactly one answer:\n"
            "YES = English word\n"
            "NO = not an English word\n"
            "\n"
            "Answer:"),
        'stimulus_last': (
            "Determine whether the letter string below is a valid English word.\n"
            "\n"
            "Respond with exactly one answer:\n"
            "YES = English word\n"
            "NO = not an English word\n"
            "\n"
            "Letter string: {STIMULUS}\n"
            "\n"
            "Answer:"),
    }

    @torch.no_grad()
    def position_sensitivity(self, ex, targets, all_hidden, model_name) -> list[dict]:
        """The SAME three prompt segments (instruction, stimulus line,
        response format) re-ordered so the stimulus is first / in the middle
        (PRIMARY template) / last. The extraction position is always the
        final 'Answer:' token; only the token distance stimulus → final
        position changes (recorded). Only 'following' → 'above'/'below' is
        re-worded to keep the instruction coherent (disclosed)."""
        rows = self.balanced_rows(targets, self.config.POSITION_SENSITIVITY_N_ITEMS, seed=SEED + 41)
        y = np.asarray(targets['is_word']).astype(int)[rows]
        folds = self.analyzer.make_folds(y)
        sub_t = {k: np.asarray(v)[rows] if isinstance(v, np.ndarray) else [v[i] for i in rows]
                 for k, v in targets.items()}
        L = ex.num_layers
        prim = {li: all_hidden[li][rows] for li in range(L)}
        ref_tbl, ref_oof = self.exp._crossfit_table(
            model_name, prim, sub_t, self.analyzer, L, ex.hidden_dim, fold_ids=folds,
            representation_type='stimulus_middle (primary template)', desc='D3 middle')
        spans_mid = self._stimulus_spans(targets, rows)
        mid_dist = ex.extract_final_and_span_states(
            [build_task_prompt(str(targets['stimulus'][i])) for i in rows[:64]],
            spans_mid[:64], layers=[0], desc='D3 distance')['tokens_after_span']
        out = [dict(r, template='stimulus_middle',
                    mean_tokens_between_stimulus_and_final=float(np.mean(mid_dist)))
               for r in ref_tbl]
        for name, tmpl in self.POSITION_TEMPLATES.items():
            texts = [tmpl.replace('{STIMULUS}', str(targets['stimulus'][i])) for i in rows]
            pre = tmpl.split('{STIMULUS}')[0]
            spans = [(len(pre), len(pre) + len(str(targets['stimulus'][i]))) for i in rows]
            st = ex.extract_final_and_span_states(texts, spans, desc=f'D3 {name}')
            tbl, oof = self.exp._crossfit_table(
                model_name, st['final'], sub_t, self.analyzer, L, ex.hidden_dim,
                fold_ids=folds, representation_type=f'{name}_final_position',
                desc=f'D3 {name}')
            block = []
            for r in tbl:
                li = r['layer']
                r = dict(r, template=name,
                         mean_tokens_between_stimulus_and_final=float(np.mean(st['tokens_after_span'])))
                if li in oof and li in ref_oof:
                    ca = ((oof[li][0] >= 0.5).astype(int) == y).astype(int)
                    cb = ((ref_oof[li][0] >= 0.5).astype(int) == y).astype(int)
                    _b, _c, p = mcnemar_exact(ca, cb)
                    r.update({'accuracy_minus_primary_template': float(ca.mean() - cb.mean()),
                              'mcnemar_vs_primary_template_p': p})
                block.append(r)
            adj = adjust_pvalues([r.get('mcnemar_vs_primary_template_p', np.nan) for r in block],
                                 'holm', self.config.ALPHA)
            for r, v in zip(block, adj):
                r['mcnemar_vs_primary_template_p_holm'] = float(v) if np.isfinite(v) else np.nan
            out.extend(block)
            del st
            gc.collect()
        return out

    # ════════════════════════════════════════════════════════════════════
    # C2 — RESIDUAL-STREAM STEERING   and   C5 — SUBSPACE MEAN-ABLATION
    # ════════════════════════════════════════════════════════════════════
    def _position_hook(self, fp_holder, fn):
        def hook(mod, inp, out):
            h = _hook_out_tensor(out)
            fp = fp_holder['fp'].to(h.device)
            ar = torch.arange(h.shape[0], device=h.device)
            h = h.clone()
            h[ar, fp] = fn(h[ar, fp].float()).to(h.dtype)
            return _hook_replace_out(out, h)
        return hook

    def _run_with_layer_fn(self, ex, texts, li, fn, state_layers=()):
        dec = ex.decoder_layers()
        holder = {}

        def install(fp):
            holder['fp'] = fp
            return [dec[li].register_forward_hook(self._position_hook(holder, fn))]
        return self._run(ex, texts, install=install, state_layers=state_layers)

    @torch.no_grad()
    def steering_and_subspace_ablation(self, ex, targets, all_hidden, model_name) -> dict:
        """C2: h ← h + α·s·u at the first-output prediction position of layer
        l, where u is the RAW-space unit direction of a linear probe
        (lexicality: word vs nonword; frequency: HF vs LF words) fitted on
        items disjoint from the evaluation items, and s the SD of the
        training projections on u (α in SD units). Outcome: Δm, the change in
        the model's own logit(YES) − logit(NO). Null: random unit directions
        scaled by THEIR projection SD.
        C5: h ← h − U Uᵀ (h − μ) with U an INLP basis (k directions) —
        mean-ablation of the probe-identified subspace; null: random
        orthonormal subspaces of the same rank."""
        if not ex.ensure_answer_readout():
            return {'steering': [], 'subspace_ablation': []}
        cfg = self.config
        ev = self.balanced_rows(targets, cfg.MODEL_LEVEL_N_ITEMS, seed=SEED + 51)
        fit = self.fit_rows(targets, ev)
        y_all = np.asarray(targets['is_word']).astype(int)
        fg_all = np.asarray(targets['freq_group'], dtype=object)
        y, fg = y_all[ev], fg_all[ev]
        texts = self._prompts(targets, ev)
        m0, _ = self._run(ex, texts)
        base = self._margin_summary(m0, y, fg, 'baseline')
        rng = np.random.RandomState(SEED + 52)
        steer_rows, abl_rows = [], []
        n_rand = cfg.N_RANDOM_DIRECTIONS_MODEL_LEVEL
        for li in tqdm(self._layers(ex.num_layers), desc=f"C2/C5 | {model_name}"):
            Xf = all_hidden[li][fit].astype(np.float64)
            yf, fgf = y_all[fit], fg_all[fit]
            fmask = (yf == 1) & np.isin(fgf, ['high', 'low'])
            dirs = {'lexicality': fit_linear_direction(Xf, yf, cfg.PRIMARY_PROBE_C)}
            if fmask.sum() >= 2 * cfg.MIN_SAMPLES_PER_GROUP:
                dirs['frequency'] = fit_linear_direction(
                    Xf[fmask], (fgf[fmask] == 'high').astype(int), cfg.PRIMARY_PROBE_C)
            rand_dirs = []
            for _ in range(n_rand):
                r = rng.randn(Xf.shape[1]); r /= np.linalg.norm(r)
                rand_dirs.append({'u': r, 'proj_sd': float(np.std(Xf @ r))})
            # ── C2: steering dose-response ────────────────────────────────
            for alpha in cfg.STEERING_MAGNITUDES:
                rand_delta = []
                for rd in rand_dirs:
                    v = torch.as_tensor(alpha * rd['proj_sd'] * rd['u'], dtype=torch.float32)
                    m1, _ = self._run_with_layer_fn(ex, texts, li, lambda x, v=v: x + v.to(x.device))
                    rand_delta.append(float(np.mean(m1 - m0)))
                for dname, d in dirs.items():
                    v = torch.as_tensor(alpha * d['proj_sd'] * d['u'], dtype=torch.float32)
                    m1, _ = self._run_with_layer_fn(ex, texts, li, lambda x, v=v: x + v.to(x.device))
                    delta = m1 - m0
                    row = {'model': model_name, 'layer': li,
                           'relative_depth': (li + 1) / ex.num_layers,
                           'direction': dname, 'alpha_sd_units': float(alpha),
                           'direction_train_accuracy': d['train_accuracy'],
                           'projection_sd': d['proj_sd'], 'n_eval_items': int(len(ev)),
                           'n_fit_items': int(len(fit)),
                           **base, **self._margin_summary(m1, y, fg, 'steered'),
                           'mean_delta_margin_all': float(np.mean(delta)),
                           'mean_delta_margin_words': float(np.mean(delta[y == 1])),
                           'mean_delta_margin_nonwords': float(np.mean(delta[y == 0])),
                           'random_direction_mean_delta': float(np.mean(rand_delta)),
                           'random_direction_sd_delta': float(np.std(rand_delta, ddof=1)) if len(rand_delta) > 1 else np.nan,
                           'n_random_directions': len(rand_delta)}
                    t = self._paired_delta_test(delta)
                    row.update({'delta_t': t['delta_t'], 'delta_p': t['delta_p'],
                                'delta_p_log10': t['delta_p_log10']})
                    ge = sum(abs(r_) >= abs(row['mean_delta_margin_all']) for r_ in rand_delta)
                    row['p_vs_random_directions_empirical'] = (1 + ge) / (1 + len(rand_delta))
                    sdr = row['random_direction_sd_delta']
                    row['z_vs_random_directions'] = (
                        (row['mean_delta_margin_all'] - row['random_direction_mean_delta']) / sdr
                        if _finite(sdr) and sdr > 0 else np.nan)
                    steer_rows.append(row)
            # ── C5: subspace mean-ablation ────────────────────────────────
            k = cfg.SUBSPACE_ABLATION_K
            spaces = {'lexicality': (Xf, yf)}
            if 'frequency' in dirs:
                spaces['frequency'] = (Xf[fmask], (fgf[fmask] == 'high').astype(int))
            mu = torch.as_tensor(Xf.mean(0), dtype=torch.float32)
            for sname, (Xs, ys) in spaces.items():
                ntr = int(0.8 * len(ys))
                perm = np.random.RandomState(SEED + li).permutation(len(ys))
                tr_i, te_i = perm[:ntr], perm[ntr:]
                U = inlp_subspace(Xs[tr_i], ys[tr_i], k, cfg.PRIMARY_PROBE_C)
                if U.shape[1] == 0:
                    continue
                Q = U @ U.T
                Xte_proj = Xs[te_i] - (Xs[te_i] - Xs[tr_i].mean(0)) @ Q
                try:
                    post = fit_linear_direction(Xs[tr_i] - (Xs[tr_i] - Xs[tr_i].mean(0)) @ Q,
                                                ys[tr_i], cfg.PRIMARY_PROBE_C)
                    post_acc = float(np.mean((probe_logits(post, Xte_proj) > 0).astype(int) == ys[te_i]))
                except Exception:
                    post_acc = np.nan
                Ut = torch.as_tensor(U, dtype=torch.float32)

                def _abl(x, Ut=Ut):
                    Ud = Ut.to(x.device)
                    return x - ((x - mu.to(x.device)) @ Ud) @ Ud.T
                m1, _ = self._run_with_layer_fn(ex, texts, li, _abl)
                rand_eff = []
                for _ in range(n_rand):
                    R, _ = np.linalg.qr(rng.randn(Xf.shape[1], U.shape[1]))
                    Rt = torch.as_tensor(R, dtype=torch.float32)
                    mr, _ = self._run_with_layer_fn(
                        ex, texts, li, lambda x, Rt=Rt: x - ((x - mu.to(x.device)) @ Rt.to(x.device)) @ Rt.to(x.device).T)
                    rand_eff.append(self._margin_summary(mr, y, fg, 'r'))
                s1 = self._margin_summary(m1, y, fg, 'ablated')
                row = {'model': model_name, 'layer': li, 'relative_depth': (li + 1) / ex.num_layers,
                       'subspace': sname, 'subspace_rank': int(U.shape[1]),
                       'post_erasure_heldout_probe_accuracy': post_acc,
                       'n_eval_items': int(len(ev)), **base, **s1}
                gap0 = base['baseline_mean_margin_hf_words'] - base['baseline_mean_margin_lf_words']
                gap1 = s1['ablated_mean_margin_hf_words'] - s1['ablated_mean_margin_lf_words']
                rg = [r['r_mean_margin_hf_words'] - r['r_mean_margin_lf_words'] for r in rand_eff]
                ra = [r['r_native_accuracy'] for r in rand_eff]
                row.update({'hf_minus_lf_margin_gap_baseline': gap0,
                            'hf_minus_lf_margin_gap_ablated': gap1,
                            'delta_hf_minus_lf_margin_gap': gap1 - gap0,
                            'random_subspace_mean_delta_gap': float(np.mean(rg) - gap0),
                            'delta_native_accuracy': s1['ablated_native_accuracy'] - base['baseline_native_accuracy'],
                            'random_subspace_mean_delta_accuracy': float(np.mean(ra) - base['baseline_native_accuracy']),
                            'n_random_subspaces': len(rand_eff)})
                abl_rows.append(row)
        return {'steering': steer_rows, 'subspace_ablation': abl_rows}

    # ════════════════════════════════════════════════════════════════════
    # C3 — ATTENTION-HEAD ATTRIBUTION AND ABLATION
    # ════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def head_ablation(self, ex, targets, all_hidden, model_name, readout_layer: int) -> dict:
        """1. Record each head's output z_h (input to the output projection)
              at the first-output prediction position for every layer ≤ R.
           2. Direct attribution of head (l, h) to a FIXED linear probe at the
              readout layer R: s = z_h · (W_O[:, h]ᵀ u_R); score =
              mean_words − mean_nonwords (lexicality) or mean_HF − mean_LF
              (frequency). This is the direct residual path only (Elhage et
              al., 2021); indirect effects are captured by step 3.
           3. Jointly ablate the top-k heads by |score| at that position
              (mean-ablation by default — zero-ablation takes activations off
              distribution; Wang et al., 2023) and re-read the SAME fixed
              probes at layer R and the model's YES/NO margin.
           Null: k random heads from layers ≤ R."""
        cfg = self.config
        R = int(min(max(readout_layer, 1), ex.num_layers - 1))
        ev = self.balanced_rows(targets, cfg.MODEL_LEVEL_N_ITEMS, seed=SEED + 61)
        fit = self.fit_rows(targets, ev)
        y_all = np.asarray(targets['is_word']).astype(int)
        fg_all = np.asarray(targets['freq_group'], dtype=object)
        y, fg = y_all[ev], fg_all[ev]
        texts = self._prompts(targets, ev)
        Xf = all_hidden[R][fit]
        probes = {'lexicality': fit_linear_direction(Xf, y_all[fit], cfg.PRIMARY_PROBE_C)}
        fmask = (y_all[fit] == 1) & np.isin(fg_all[fit], ['high', 'low'])
        if fmask.sum() >= 2 * cfg.MIN_SAMPLES_PER_GROUP:
            probes['frequency'] = fit_linear_direction(
                Xf[fmask], (fg_all[fit][fmask] == 'high').astype(int), cfg.PRIMARY_PROBE_C)
        dec = ex.decoder_layers()
        layers = list(range(R + 1))
        nH = ex.n_query_heads()
        zs = {li: [] for li in layers}
        holder = {}
        handles = []
        try:
            for li in layers:
                op = ex.o_proj_module(ex.attn_module(dec[li]))

                def pre(mod, args, li=li):
                    z = args[0]
                    fp = holder['fp'].to(z.device)
                    zs[li].append(z[torch.arange(z.shape[0], device=z.device), fp].float().cpu())
                handles.append(op.register_forward_pre_hook(pre))
            bs = cfg.MODEL_LEVEL_BATCH_SIZE
            for s in tqdm(range(0, len(texts), bs), desc=f"C3 head capture | {model_name}"):
                ids, am, fp, _ = ex.encode_task_batch(texts[s:s + bs])
                holder['fp'] = fp
                ex.backbone_hidden(ids, am)
        finally:
            for h in handles:
                h.remove()
        Z = {li: torch.cat(v, 0).numpy() for li, v in zs.items()}
        dh = Z[0].shape[1] // nH
        score_rows = []
        for li in layers:
            W = ex.o_proj_module(ex.attn_module(dec[li])).weight.detach().float().cpu().numpy()
            for pname, pr in probes.items():
                a = W.T @ pr['u']                                  # (nH*dh,)
                s_ = (Z[li].reshape(len(ev), nH, dh) * a.reshape(nH, dh)[None]).sum(-1)
                if pname == 'lexicality':
                    sc = s_[y == 1].mean(0) - s_[y == 0].mean(0)
                else:
                    hfm, lfm = (y == 1) & (fg == 'high'), (y == 1) & (fg == 'low')
                    sc = (s_[hfm].mean(0) - s_[lfm].mean(0)) if hfm.any() and lfm.any() \
                        else np.full(nH, np.nan)
                for h in range(nH):
                    score_rows.append({'model': model_name, 'layer': li, 'head': h,
                                       'probe': pname, 'readout_layer': R,
                                       'direct_attribution_score': float(sc[h])})
        sdf = pd.DataFrame(score_rows)
        k = int(min(cfg.HEAD_ABLATION_TOP_K, len(layers) * nH))
        z_mean = {li: torch.as_tensor(Z[li].mean(0), dtype=torch.float32) for li in layers}
        del Z, zs
        gc.collect()

        def _ablate(heads):
            by_layer = {}
            for (li, h) in heads:
                by_layer.setdefault(li, []).append(h)
            hold = {}

            def install(fp):
                hold['fp'] = fp
                hs_ = []
                for li, hl in by_layer.items():
                    op = ex.o_proj_module(ex.attn_module(dec[li]))

                    def pre(mod, args, li=li, hl=hl):
                        z = args[0].clone()
                        fpd = hold['fp'].to(z.device)
                        ar = torch.arange(z.shape[0], device=z.device)
                        for h in hl:
                            sl = slice(h * dh, (h + 1) * dh)
                            rep = (z_mean[li][sl].to(z.device, z.dtype)
                                   if cfg.HEAD_ABLATION_MODE == 'mean'
                                   else torch.zeros(dh, device=z.device, dtype=z.dtype))
                            z[ar, fpd, sl] = rep
                        return (z,) + tuple(args[1:])
                    hs_.append(op.register_forward_pre_hook(pre))
                return hs_
            return self._run(ex, texts, install=install, state_layers=(R,))

        def _readout(m, st):
            X = st[R]
            out = self._margin_summary(m, y, fg, 'native')
            lx = probe_logits(probes['lexicality'], X)
            out['fixed_lexicality_probe_accuracy'] = float(np.mean((lx > 0).astype(int) == y))
            hfm, lfm = (y == 1) & (fg == 'high'), (y == 1) & (fg == 'low')
            out['fixed_lexicality_probe_hf_minus_lf_word_accuracy'] = (
                float(np.mean(lx[hfm] > 0) - np.mean(lx[lfm] > 0)) if hfm.any() and lfm.any() else np.nan)
            if 'frequency' in probes and hfm.any() and lfm.any():
                fx = probe_logits(probes['frequency'], X)
                m_ = hfm | lfm
                out['fixed_frequency_probe_accuracy'] = float(
                    np.mean((fx[m_] > 0).astype(int) == hfm[m_].astype(int)))
            return out, (lx > 0).astype(int) == y

        m0, st0 = self._run(ex, texts, state_layers=(R,))
        base, c0 = _readout(m0, st0)
        rng = np.random.RandomState(SEED + 62)
        all_heads = [(li, h) for li in layers for h in range(nH)]
        abl_rows = []
        for pname in probes:
            sub = sdf[sdf['probe'] == pname].dropna(subset=['direct_attribution_score'])
            top = sub.reindex(sub['direct_attribution_score'].abs().sort_values(ascending=False).index).head(k)
            heads = [(int(r.layer), int(r.head)) for r in top.itertuples()]
            m1, st1 = _ablate(heads)
            ab, c1 = _readout(m1, st1)
            rand = []
            for _ in range(cfg.N_RANDOM_DIRECTIONS_MODEL_LEVEL):
                pick = [all_heads[i] for i in rng.choice(len(all_heads), size=k, replace=False)]
                mr, str_ = _ablate(pick)
                rand.append(_readout(mr, str_)[0])
            row = {'model': model_name, 'ranking_probe': pname, 'readout_layer': R,
                   'k_heads': k, 'ablation_mode': cfg.HEAD_ABLATION_MODE,
                   'ablated_heads': json.dumps(heads), 'n_eval_items': int(len(ev))}
            for key, v in base.items():
                row[f'baseline_{key}'] = v
                row[f'ablated_{key}'] = ab.get(key, np.nan)
                row[f'delta_{key}'] = ab.get(key, np.nan) - v if _finite(v) and _finite(ab.get(key)) else np.nan
                rv = [r.get(key, np.nan) for r in rand]
                row[f'random_heads_mean_delta_{key}'] = float(np.nanmean(rv) - v) if _finite(v) else np.nan
            row['mcnemar_fixed_probe_p'] = mcnemar_exact(c0.astype(int), c1.astype(int))[2]
            abl_rows.append(row)
        return {'head_scores': score_rows, 'head_ablation': abl_rows}


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 11 (NEW): HUMAN–LLM REACTION-TIME ALIGNMENT
# ════════════════════════════════════════════════════════════════════════════
#
# Fully modular, post-hoc addition. Does NOT alter, rerun, or overwrite
# Analyses 1-10, the existing probe architecture, splits, seeds, frequency
# thresholds, or any existing output file.
#
# Objective (see task spec):
#   Do word-level lexical representations in LLM transformer layers contain
#   a continuous signal that predicts human lexical-decision reaction time
#   from the English Lexicon Project (ELP), and at which layers is this
#   alignment strongest after controlling for frequency and other lexical
#   confounds?
#
# Interpretation constraint: this analysis NEVER treats inference latency /
# forward-pass duration as "LLM reaction time". The LLM-side quantity is a
# continuous *lexical-evidence* score S_{i,l} = logit(P(WORD | h_{i,l}))
# derived from the existing word-vs-nonword probe's representation of each
# word — nothing about wall-clock compute is used anywhere in this class.
#
# Data-reuse note (task section 3): the existing pipeline trains a probe
# per layer inside FrequencyAnalyzer.analyze_layer(), but that classifier
# object is discarded (`del clf`) and no word-level probabilities/logits
# are persisted anywhere in the existing outputs. There is therefore no
# saved word-level score to reuse. The minimum targeted rerun needed is:
# retrain ONE probe per layer (same architecture/training routine as the
# existing FrequencyAnalyzer, via FrequencyAnalyzer.train_classifier) on
# the ALREADY-CACHED hidden states — i.e. no new transformer forward pass,
# no re-extraction, no re-running of Analyses 1-10 — and use it to score
# every real word. This is the smallest change that satisfies the
# objective while preserving the existing probe implementation exactly.
# ════════════════════════════════════════════════════════════════════════════

class HumanRTAlignmentAnalysis:
    """Analysis 11: Human-LLM Reaction-Time Alignment with 5-fold cross-fitted lexical evidence."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.word_level_rows: list = []   # list of pd.DataFrames, one per model
        self.elp_lookup, self.rt_column, self.acc_column = self._load_elp_lookup()
        self.match_report: dict = {}  # per-model match counts

    # ── ELP lookup table ────────────────────────────────────────────────
    def _load_elp_lookup(self) -> tuple[dict[str, dict], str | None, str | None]:
        """
        Reads the existing ELP items.csv (config.WORDS_PATH) independently
        of LDTDataset (which does not carry the RT column forward), builds
        a word -> {human_rt, human_accuracy} lookup, and identifies +
        documents which column was used as "mean lexical-decision RT".
        """
        try:
            df = pd.read_csv(self.config.WORDS_PATH)
        except Exception as e:
            logger.error(f"[Analysis 11] Could not read {self.config.WORDS_PATH} "
                         f"for ELP RT lookup: {e}")
            return {}, None, None

        rt_col = None
        for cand in self.config.ELP_RT_COLUMN_CANDIDATES:
            if cand in df.columns:
                rt_col = cand
                break
        if rt_col is None:
            logger.error(
                "[Analysis 11] Could not identify an ELP mean-RT column among "
                f"candidates {self.config.ELP_RT_COLUMN_CANDIDATES}. "
                f"Available columns: {list(df.columns)}. "
                "Human RT alignment will be SKIPPED (no RT values invented)."
            )
            return {}, None, None

        logger.info(f"[Analysis 11] Using ELP column '{rt_col}' as the mean "
                    f"lexical-decision reaction time (human_rt).")

        acc_col = None
        for cand in self.config.ELP_ACCURACY_COLUMN_CANDIDATES:
            if cand in df.columns:
                acc_col = cand
                break
        if acc_col:
            logger.info(f"[Analysis 11] Using ELP column '{acc_col}' as "
                        f"human_accuracy.")
        else:
            logger.warning("[Analysis 11] No ELP accuracy column found — "
                           "human_accuracy will be NaN in the word-level table.")

        word_col = 'Word' if 'Word' in df.columns else df.columns[0]
        lookup: dict[str, dict] = {}
        for _, row in df.iterrows():
            w = str(row[word_col]).strip().lower()
            if not w or w == 'nan':
                continue
            rt = pd.to_numeric(str(row[rt_col]).replace('#', 'nan'), errors='coerce')
            acc = (pd.to_numeric(str(row[acc_col]).replace('#', 'nan'), errors='coerce')
                   if acc_col else np.nan)
            # If a word appears more than once in the ELP file, keep the
            # first valid RT rather than silently overwriting with a
            # later duplicate (duplicates are reported, not hidden).
            if w in lookup:
                continue
            lookup[w] = {'human_rt': rt, 'human_accuracy': acc}
        return lookup, rt_col, acc_col

    # ── Step A: word-level LLM lexical-evidence extraction ─────────────
    def _crossfit_word_probabilities_for_layer(self, analyzer, X, y, layer_idx,
                                               model_name, n_folds=None, seed=None,
                                               shuffle=None, quiet=False):
        """J2 — thin backward-compatible wrapper: the cross-fitted probe now
        lives in FrequencyAnalyzer so the PRIMARY result does not depend on
        the RT module. Defaults follow the RT-alignment fold settings."""
        return analyzer.crossfit_word_probabilities(
            X, y,
            n_folds=n_folds or self.config.RT_ALIGNMENT_N_FOLDS,
            seed=self.config.RT_ALIGNMENT_SEED if seed is None else seed,
            shuffle=self.config.RT_ALIGNMENT_CV_SHUFFLE if shuffle is None else shuffle,
            probe='linear')

    def _make_internal_validation_split(self, X: np.ndarray, y: np.ndarray):
        """
        Create a 15% internal validation split and return values in the order
        expected by train_classifier: (X_tr, y_tr, X_val, y_val).

        train_test_split returns (X_train, X_test, y_train, y_test), so we
        rename: X_train->X_tr, X_test->X_val, y_train->y_tr, y_test->y_val.
        """
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self.config.RT_ALIGNMENT_VAL_SIZE,
                stratify=y, random_state=self.config.RT_ALIGNMENT_SEED
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self.config.RT_ALIGNMENT_VAL_SIZE,
                random_state=self.config.RT_ALIGNMENT_SEED
            )
        return X_tr, y_tr, X_val, y_val

    @staticmethod
    def _predict_word_probability(analyzer: 'FrequencyAnalyzer', clf,
                                  hidden_states: np.ndarray) -> np.ndarray:
        """P(WORD | h) for every row in hidden_states, using the trained probe."""
        Xs = clf.scaler.transform(hidden_states)
        clf.eval()
        with torch.no_grad():
            probs = F.softmax(
                clf(torch.FloatTensor(Xs).to(analyzer.device)), dim=1
            ).cpu().numpy()
        return probs[:, 1]

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def extract_word_level_scores(self, all_hidden: dict[int, np.ndarray],
                                  targets: dict, analyzer: 'FrequencyAnalyzer',
                                  model_name: str, num_layers: int,
                                  precomputed_oof: dict = None) -> pd.DataFrame:
        """
        Extract cross-fitted lexical evidence for every real word.
    
        For each transformer layer, a 5-fold cross-fitted probe is trained on
        word/nonword hidden states. Each item's score is generated only by the
        fold model that did not train on that item. This preserves the original
        lexical-evidence definition S=logit(P(WORD|h)) while eliminating
        in-sample probe scoring from the human-RT alignment analysis.
        """
        is_word  = np.asarray(targets['is_word'])
        stimulus = np.asarray(targets['stimulus'])
        log_freq = np.asarray(targets['log_frequency'], dtype=float)
        length   = np.asarray(targets['length'], dtype=float)
        ortho_n  = np.asarray(targets['ortho_n'], dtype=float)
        tok_cnt  = np.asarray(targets['token_count'], dtype=float)
    
        word_mask = (is_word == 1)
        
        # ADDED: Better logging and validation
        if word_mask.sum() < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER:
            logger.warning(f"[Analysis 11] {model_name}: too few real words "
                           f"({word_mask.sum()}) — skipping.")
            return pd.DataFrame()
        
        logger.info(f"[Analysis 11] {model_name}: extracting for {word_mask.sum()} words, {num_layers} layers")
        
        rows = []
        widx = np.where(word_mask)[0]
        layers_processed = 0  # ADDED: Track how many layers were actually processed
    
        for li in tqdm(range(num_layers),
                       desc=f"[Analysis 11 CF] 5-fold OOF lexical evidence | {model_name}"):
            if li not in all_hidden:
                logger.warning(f"[Analysis 11] {model_name} layer {li}: not in all_hidden")
                continue
            
            X = all_hidden[li]
            try:
                if precomputed_oof is not None and li in precomputed_oof:
                    # Reuse the PRIMARY experiment's cross-fitted scores: same
                    # folds, same seed, same probe — no refitting, no divergence.
                    p_all, fold_all = precomputed_oof[li]
                else:
                    p_all, fold_all = self._crossfit_word_probabilities_for_layer(
                        analyzer, X, is_word, li, model_name
                    )
            except Exception as e:
                logger.error(
                    f"[Analysis 11 CF] {model_name} layer {li}: cross-fit failed "
                    f"({e}) — layer skipped.", exc_info=True
                )
                continue
    
            p_word = p_all[word_mask]
            fold_word = fold_all[word_mask]
            evidence = self._logit(p_word)
    
            for j, i in enumerate(widx):
                rows.append({
                    'model': model_name,
                    'layer': li,
                    'word': str(stimulus[i]).strip().lower(),
                    'log_freq_hal': log_freq[i],
                    'word_length': length[i],
                    'ortho_n': ortho_n[i],
                    'token_count': tok_cnt[i],
                    'llm_word_probability': p_word[j],
                    'llm_lexical_evidence': evidence[j],
                    'probe_score_type': '5fold_out_of_fold',
                    'representation_type': PRIMARY_REPRESENTATION,
                    # FIX: these ARE the cross-fitted (primary) scores, not the hold-out probe
                    'readout_type': PRIMARY_READOUT,
                    'probe_type': analyzer.probe_type,
                    'probe_n_folds': self.config.RT_ALIGNMENT_N_FOLDS,
                    'probe_seed': self.config.RT_ALIGNMENT_SEED,
                    'oof_fold': int(fold_word[j]),
                })
    
            layers_processed += 1  # ADDED: Count successful layers
            del p_all, fold_all, p_word, fold_word, evidence
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
    
        logger.info(f"[Analysis 11] {model_name}: processed {layers_processed}/{num_layers} layers")  # ADDED
        
        if not rows:  # ADDED: Check if any rows were generated
            logger.warning(f"[Analysis 11] {model_name}: no rows generated")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        logger.info(
            f"[Analysis 11 CF] {model_name}: extracted 5-fold OOF word-level "
            f"scores for {df['layer'].nunique() if len(df) else 0} layers x "
            f"{word_mask.sum()} words."
        )
        return df

    def accumulate(self, df: pd.DataFrame, model_name: str = None):
        """Accumulate word-level data from one model."""
        if df is not None and len(df):
            self.word_level_rows.append(df)
            
            # ADDED: Save per-model intermediate file to avoid overwriting
            if model_name:
                safe = model_name.replace(' ', '_').replace('/', '_')
                results_dir = os.path.join(self.config.RT_ALIGNMENT_DIR, "results")
                os.makedirs(results_dir, exist_ok=True)
                temp_path = os.path.join(results_dir, f'{safe}_temp_word_level.csv')
                df.to_csv(temp_path, index=False)
                logger.info(f"[Analysis 11] Saved temporary word-level data for {model_name}: {temp_path}")

    # ── Step B: match to ELP human RT, build master word-level table ───
    def _build_matched_table(self) -> pd.DataFrame:
        """Build the master word-level table by matching LLM scores to ELP RTs."""
        
        # ADDED: Check if any data was accumulated
        if not self.word_level_rows:
            logger.error("[Analysis 11] No word-level rows accumulated. "
                         "Check that extract_word_level_scores ran successfully for at least one model.")
            return pd.DataFrame()
        
        logger.info(f"[Analysis 11] Building matched table from {len(self.word_level_rows)} model DataFrames")
        
        # ADDED: Log total rows for debugging
        total_rows = sum(len(df) for df in self.word_level_rows)
        logger.info(f"[Analysis 11] Total raw rows: {total_rows}")
        
        if not self.elp_lookup:
            logger.error("[Analysis 11] No ELP RT lookup available — cannot "
                         "perform human RT alignment.")
            return pd.DataFrame()
    
        llm_df = pd.concat(self.word_level_rows, ignore_index=True)
        llm_df['word'] = llm_df['word'].astype(str).str.strip().str.lower()

        # Ensure no unintentional duplicate (word, model, layer) rows.
        before = len(llm_df)
        llm_df = llm_df.drop_duplicates(subset=['model', 'layer', 'word'])
        if len(llm_df) != before:
            logger.warning(f"[Analysis 11] Dropped {before - len(llm_df)} "
                           f"duplicate (model, layer, word) rows.")

        llm_words = set(llm_df['word'].unique())
        elp_words = set(self.elp_lookup.keys())
        matched_words   = llm_words & elp_words
        unmatched_llm   = llm_words - elp_words
        unmatched_elp   = elp_words - llm_words

        logger.info(f"[Analysis 11] Word matching — "
                    f"matched={len(matched_words)}, "
                    f"unmatched_model_stimuli={len(unmatched_llm)}, "
                    f"unmatched_ELP_words={len(unmatched_elp)}")
        self.match_report = {
            'n_matched_words':          len(matched_words),
            'n_unmatched_model_only':   len(unmatched_llm),
            'n_unmatched_elp_only':     len(unmatched_elp),
            'elp_rt_column_used':       self.rt_column,
            'elp_accuracy_column_used': self.acc_column,
        }

        llm_df['human_rt']       = llm_df['word'].map(
            lambda w: self.elp_lookup.get(w, {}).get('human_rt', np.nan))
        llm_df['human_accuracy'] = llm_df['word'].map(
            lambda w: self.elp_lookup.get(w, {}).get('human_accuracy', np.nan))

        n_before = len(llm_df)
        llm_df = llm_df[llm_df['word'].isin(matched_words)].copy()
        llm_df = llm_df.dropna(subset=['human_rt'])
        # Remove non-physical / invalid RTs (statistical safeguard, §14).
        llm_df = llm_df[llm_df['human_rt'] > 0]
        n_after = len(llm_df)
        logger.info(f"[Analysis 11] Word-level table: {n_before} -> {n_after} "
                    f"rows after requiring a valid matched human_rt "
                    f"({n_before - n_after} removed/reported, not silently "
                    f"discarded).")

        # Frequency group, carried over from the existing project
        # definition (percentile thresholds already computed upstream);
        # recompute the label here purely from log_freq_hal using the
        # SAME existing config thresholds — not redefined.
        if llm_df['log_freq_hal'].notna().any():
            hi = np.nanpercentile(llm_df['log_freq_hal'], self.config.HIGH_FREQ_PERCENTILE)
            lo = np.nanpercentile(llm_df['log_freq_hal'], self.config.LOW_FREQ_PERCENTILE)
            llm_df['freq_group'] = np.select(
                [llm_df['log_freq_hal'] >= hi, llm_df['log_freq_hal'] <= lo],
                ['high', 'low'], default='mid')
        else:
            llm_df['freq_group'] = 'mid'

        return llm_df

    # ── Step C: statistics ──────────────────────────────────────────────
    @staticmethod
    def _fdr(pvals: list[float]) -> list[float]:
        pvals = np.asarray(pvals, dtype=float)
        out = np.full(len(pvals), np.nan)
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            return out.tolist()
        _, pc, _, _ = multipletests(pvals[valid], alpha=0.05, method='fdr_bh')
        out[valid] = pc
        return out.tolist()

    def _spearman_per_layer(self, g: pd.DataFrame) -> dict:
        if len(g) < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER:
            return {'rho': np.nan, 'p': np.nan, 'p_log10': np.nan,
                    'log10_bf01': np.nan, 'n': len(g)}
        rho, p = stats.spearmanr(g['llm_lexical_evidence'], g['human_rt'])
        p2, lp = p_correlation(rho, len(g))
        return {'rho': rho, 'p': p if p > 0 else p2, 'p_log10': lp,
                'log10_bf01': log10_bf01_correlation(rho, len(g)), 'n': len(g)}

    _REG_KEYS = ['beta1', 'se', 't', 'p', 'p_log10', 'se_hc3', 'p_hc3', 'p_hc3_log10',
                 'se_cluster', 'p_cluster', 'n_clusters', 'cluster_note',
                 'log10_bf01', 'r2', 'adj_r2', 'n', 'dv_sd',
                 'beta_acc_cov', 'p_acc_cov', 'beta_len_x_freq', 'p_len_x_freq',
                 'p_len_x_freq_log10']

    def _regression_per_layer(self, g: pd.DataFrame, use_log_rt: bool) -> dict:
        """RT ~ S + log_freq + length + Ortho_N + tokens (item-level OLS).
        E2: classical, HC3-robust and — only when the cluster variable has at
        least RT_MIN_CLUSTERS levels — cluster-robust SEs. With the 3-level
        freq_group proposed in the master list, cluster-robust SEs are NOT
        reported: the sandwich estimator needs many clusters (≈ 30+;
        Cameron & Miller, 2015) and with 3 it is badly biased.
        E1 sensitivity: + human accuracy as a covariate (flagged as a
        potential BAD CONTROL: accuracy and RT are joint outcomes of the same
        decision process). B6: + log_freq × length (centred)."""
        cols = ['llm_lexical_evidence', 'log_freq_hal', 'word_length',
                'ortho_n', 'token_count']
        gg = g.dropna(subset=cols + ['human_rt'])
        out = {k: np.nan for k in self._REG_KEYS}
        out['n'] = len(gg)
        if len(gg) < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER:
            return out
        y = (np.log(gg['human_rt']) if use_log_rt else gg['human_rt']).astype(float)
        r = ols_hc3(y, gg[cols], 'llm_lexical_evidence')
        out.update({'beta1': r['beta'], 'se': r['se'], 't': r['t'], 'p': r['p'],
                    'p_log10': r['p_log10'], 'se_hc3': r['se_hc3'], 'p_hc3': r['p_hc3'],
                    'p_hc3_log10': r['p_hc3_log10'], 'log10_bf01': r['log10_bf01'],
                    'r2': r['r2'], 'adj_r2': r['adj_r2'], 'dv_sd': float(y.std(ddof=0))})
        cv = self.config.RT_CLUSTER_VARIABLE
        if cv in gg.columns:
            n_cl = int(gg[cv].nunique())
            out['n_clusters'] = n_cl
            if n_cl >= self.config.RT_MIN_CLUSTERS:
                try:
                    Xc = sm.add_constant(gg[cols].astype(float))
                    groups = pd.factorize(gg[cv])[0]
                    fc = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': groups})
                    out['se_cluster'] = float(fc.bse['llm_lexical_evidence'])
                    out['p_cluster'] = float(fc.pvalues['llm_lexical_evidence'])
                    out['cluster_note'] = f'{n_cl} clusters of {cv}'
                except Exception as e:                        # noqa: BLE001
                    out['cluster_note'] = f'cluster SE failed: {e}'
            else:
                out['cluster_note'] = (f'not reported: {n_cl} clusters of {cv} < '
                                       f'{self.config.RT_MIN_CLUSTERS} (cluster-robust SEs '
                                       f'are invalid with few clusters)')
        if 'human_accuracy' in gg.columns and gg['human_accuracy'].notna().sum() > 30:
            ga = gg.dropna(subset=['human_accuracy'])
            ya = (np.log(ga['human_rt']) if use_log_rt else ga['human_rt']).astype(float)
            ra = ols_hc3(ya, ga[cols + ['human_accuracy']], 'llm_lexical_evidence')
            out['beta_acc_cov'], out['p_acc_cov'] = ra['beta'], ra['p_hc3']
        Xi = gg[cols].astype(float).copy()
        for c in ('log_freq_hal', 'word_length'):
            Xi[c] = Xi[c] - Xi[c].mean()
        Xi['log_freq_x_length'] = Xi['log_freq_hal'] * Xi['word_length']
        ri = ols_hc3(y, Xi, 'log_freq_x_length')
        out['beta_len_x_freq'], out['p_len_x_freq'] = ri['beta'], ri['p_hc3']
        out['p_len_x_freq_log10'] = ri['p_hc3_log10']
        return out

    def _mixed_model_per_layer(self, g: pd.DataFrame, use_log_rt: bool) -> dict:
        """E3 (corrected). RT ~ S + log_freq + Ortho_N + tokens + (1 | length).
        Not fitted as in the master list — (1|freq_group) + (1|length_bin)
        with log_freq also a FIXED effect — because (a) freq_group is a
        deterministic discretisation of the fixed covariate log_freq, so its
        random intercept is confounded with the fixed slope, and (b) a
        variance component cannot be estimated from 3 levels. ELP gives item
        MEANS (one row per word), so there is no subject/trial structure for
        a crossed-random-effects model (Baayen, Davidson & Bates, 2008); the
        one defensible grouping is word length (≥ RT_MIXED_MODEL_MIN_GROUPS
        levels), modelled as a random intercept INSTEAD of a linear term."""
        cols = ['llm_lexical_evidence', 'log_freq_hal', 'ortho_n', 'token_count']
        grp = self.config.RT_MIXED_MODEL_GROUP
        gg = g.dropna(subset=cols + ['human_rt', grp])
        out = {'mixed_beta': np.nan, 'mixed_se': np.nan, 'mixed_p': np.nan,
               'mixed_group_var': np.nan, 'mixed_n_groups': int(gg[grp].nunique()) if len(gg) else 0,
               'mixed_n': len(gg), 'mixed_converged': False}
        if len(gg) < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER or \
                out['mixed_n_groups'] < self.config.RT_MIXED_MODEL_MIN_GROUPS:
            out['mixed_note'] = 'too few observations or groups'
            return out
        y = (np.log(gg['human_rt']) if use_log_rt else gg['human_rt']).astype(float)
        X = sm.add_constant(((gg[cols] - gg[cols].mean()) / gg[cols].std(ddof=0).replace(0, 1)).astype(float))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = sm.MixedLM(y.to_numpy(), X.to_numpy(), groups=gg[grp].to_numpy()).fit(
                    reml=True, method='lbfgs')
            j = list(X.columns).index('llm_lexical_evidence')
            fe, se = np.asarray(fit.fe_params, float), np.asarray(fit.bse_fe, float)
            out.update({'mixed_beta': float(fe[j]), 'mixed_se': float(se[j]),
                        'mixed_group_var': float(np.asarray(fit.cov_re, float).ravel()[0]),
                        'mixed_converged': bool(fit.converged)})
            out['mixed_p'], out['mixed_p_log10'] = p_from_z(fe[j] / se[j])
            out['mixed_note'] = 'beta is per SD of S (predictors standardised)'
        except Exception as e:                                # noqa: BLE001
            out['mixed_note'] = f'failed: {e}'
        return out

    def _interaction_per_layer(self, g: pd.DataFrame, use_log_rt: bool) -> dict:
        gg = g[g['freq_group'].isin(['high', 'low'])].copy()
        cols = ['llm_lexical_evidence', 'word_length', 'ortho_n', 'token_count']
        gg = gg.dropna(subset=cols + ['human_rt'])
        if len(gg) < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER:
            return {k: np.nan for k in
                    ['beta3', 'se', 't', 'p', 'r2', 'n']} | {'n': len(gg)}
        gg['FreqGroup'] = (gg['freq_group'] == 'high').astype(float)
        gg['Interaction'] = gg['llm_lexical_evidence'] * gg['FreqGroup']
        y = np.log(gg['human_rt']) if use_log_rt else gg['human_rt']
        X = sm.add_constant(gg[cols + ['FreqGroup', 'Interaction']].astype(float))
        try:
            model = sm.OLS(y.astype(float), X).fit()
        except Exception:
            return {k: np.nan for k in
                    ['beta3', 'se', 't', 'p', 'r2', 'n']} | {'n': len(gg)}
        return {
            'beta3': model.params.get('Interaction', np.nan),
            'se':    model.bse.get('Interaction', np.nan),
            't':     model.tvalues.get('Interaction', np.nan),
            'p':     model.pvalues.get('Interaction', np.nan),
            'r2':    model.rsquared,
            'n':     len(gg),
        }

    def _freq_group_descriptives(self, g: pd.DataFrame) -> dict:
        out = {}
        for grp in ['high', 'low']:
            sub = g[g['freq_group'] == grp]
            out[grp] = {
                'n':                 len(sub),
                'mean_human_rt':     float(sub['human_rt'].mean()) if len(sub) else np.nan,
                'median_human_rt':   float(sub['human_rt'].median()) if len(sub) else np.nan,
                'mean_llm_evidence': float(sub['llm_lexical_evidence'].mean()) if len(sub) else np.nan,
                'median_llm_evidence': float(sub['llm_lexical_evidence'].median()) if len(sub) else np.nan,
            }
        return out

    # ── Step D: figures ─────────────────────────────────────────────────
    def _plot_figures(self, model_name: str, layer_df: pd.DataFrame,
                      word_df: pd.DataFrame, best_layer: int | None):
        fig_dir = os.path.join(self.config.RT_ALIGNMENT_DIR, "figures")

        # Figure 1 — Spearman rho across layers
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(0, color='gray', lw=1, ls='--')
        sig = layer_df['spearman_fdr_p'] < 0.05
        ax.plot(layer_df['layer'], layer_df['spearman_rho'], color='#1A535C', lw=1.5, zorder=1)
        ax.scatter(layer_df.loc[~sig, 'layer'], layer_df.loc[~sig, 'spearman_rho'],
                   color='#AAAAAA', label='not FDR-significant', zorder=2)
        ax.scatter(layer_df.loc[sig, 'layer'], layer_df.loc[sig, 'spearman_rho'],
                   color='#D62246', label='FDR-significant', zorder=3)
        ax.set_xlabel('Transformer layer'); ax.set_ylabel("Spearman ρ (LLM evidence vs human RT)")
        ax.set_title(f'{model_name} — Human RT alignment across layers')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'{model_name}_fig1_spearman_by_layer.png'))
        plt.close(fig)

        # Figure 2 — standardized regression beta across layers
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(0, color='gray', lw=1, ls='--')
        sig = layer_df['reg_fdr_p'] < 0.05
        ax.plot(layer_df['layer'], layer_df['reg_beta1_std'], color='#2E86AB', lw=1.5, zorder=1)
        ax.scatter(layer_df.loc[~sig, 'layer'], layer_df.loc[~sig, 'reg_beta1_std'],
                   color='#AAAAAA', label='not FDR-significant', zorder=2)
        ax.scatter(layer_df.loc[sig, 'layer'], layer_df.loc[sig, 'reg_beta1_std'],
                   color='#D62246', label='FDR-significant', zorder=3)
        ax.set_xlabel('Transformer layer'); ax.set_ylabel('Standardized β (LLM evidence)')
        ax.set_title(f'{model_name} — Regression effect across layers\n'
                    f'(controlling for frequency, length, Ortho_N, token count)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'{model_name}_fig2_regression_beta_by_layer.png'))
        plt.close(fig)

        # Figure 3 — representative scatter at the strongest layer
        if best_layer is not None:
            sub = word_df[word_df['layer'] == best_layer]
            if len(sub):
                fig, ax = plt.subplots(figsize=(7, 6))
                for grp, color in [('high', '#2E86AB'), ('low', '#D62246'), ('mid', '#AAAAAA')]:
                    s = sub[sub['freq_group'] == grp]
                    if len(s):
                        ax.scatter(s['llm_lexical_evidence'], s['human_rt'],
                                  s=14, alpha=0.5, color=color, label=f'{grp}-freq')
                if len(sub) > 2:
                    m, b = np.polyfit(sub['llm_lexical_evidence'], sub['human_rt'], 1)
                    xs = np.linspace(sub['llm_lexical_evidence'].min(),
                                     sub['llm_lexical_evidence'].max(), 50)
                    ax.plot(xs, m * xs + b, color='black', lw=2, label='fitted regression')
                ax.set_xlabel('LLM lexical evidence  S = logit P(WORD | h)')
                ax.set_ylabel('Human mean RT (ELP)')
                ax.set_title(f'{model_name} — layer {best_layer} (strongest alignment)')
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, f'{model_name}_fig3_best_layer_scatter.png'))
                plt.close(fig)

    # ── Orchestration / save ────────────────────────────────────────────
    def finalize_and_save(self):
        matched = self._build_matched_table()
        if matched.empty:
            logger.warning("[Analysis 11] No matched word-level data")
            return
    
        results_dir = os.path.join(self.config.RT_ALIGNMENT_DIR, "results")
        
        # Save combined file (keep this)
        matched.to_csv(os.path.join(results_dir, 'human_rt_word_level_alignment.csv'), index=False)
        
        # ALSO save per-model files to avoid overwriting
        for model_name, g_model in matched.groupby('model'):
            safe = model_name.replace(' ', '_').replace('/', '_')
            g_model.to_csv(
                os.path.join(results_dir, f'{safe}_human_rt_word_level.csv'), 
                index=False
            )
        logger.info(f"✓ [Analysis 11 CF] human_rt_word_level_alignment.csv "
                    f"({len(matched)} word x model x layer rows; 5-fold OOF evidence)")

        # ── E5 / E6: overlap statistics and RT distribution diagnostics ──
        try:
            self._write_rt_diagnostics(matched, results_dir)
        except Exception as e:                                 # noqa: BLE001
            logger.warning(f"[Analysis 11] RT diagnostics failed: {e}")

        # ── E1: human-accuracy filter (primary sample set in Config) ──────
        thr = self.config.RT_ALIGNMENT_MIN_HUMAN_ACCURACY
        samples = {'all_items': matched}
        if thr is not None and 'human_accuracy' in matched.columns:
            samples['accuracy_filtered'] = matched[matched['human_accuracy'] >= thr].copy()
        primary_sample = (self.config.RT_PRIMARY_SAMPLE
                          if self.config.RT_PRIMARY_SAMPLE in samples else 'all_items')
        logger.info(f"[Analysis 11] primary RT sample = {primary_sample} "
                    f"({samples[primary_sample]['word'].nunique()} words; "
                    f"all items: {matched['word'].nunique()})")
        for sname, sdf in samples.items():
            if sname != primary_sample:
                try:
                    self._sensitivity_layer_table(sdf, sname, results_dir)
                except Exception as e:                         # noqa: BLE001
                    logger.warning(f"[Analysis 11] sensitivity sample {sname} failed: {e}")
        matched = samples[primary_sample]
        if matched.empty:
            logger.warning("[Analysis 11] primary RT sample is empty")
            return

        skew = float(stats.skew(matched['human_rt'].dropna()))
        use_log_rt = abs(skew) > self.config.RT_LOG_SKEW_THRESHOLD
        logger.info(f"[Analysis 11] Human RT skew={skew:.2f} -> "
                    f"{'log(RT)' if use_log_rt else 'raw RT'} used for regression "
                    f"models (raw RT always used for the primary Spearman test).")

        layer_rows, cross_model_rows = [], []

        for model_name, g_model in matched.groupby('model'):
            layers_sorted = sorted(g_model['layer'].unique())
            sp_rows, reg_rows, int_rows = [], [], []

            for li in layers_sorted:
                g = g_model[g_model['layer'] == li]
                sp  = self._spearman_per_layer(g)
                reg = self._regression_per_layer(g, use_log_rt)
                inter = self._interaction_per_layer(g, use_log_rt)
                sp_rows.append(sp); reg_rows.append(reg); int_rows.append(inter)

            sp_fdr  = self._fdr([r['p'] for r in sp_rows])
            reg_fdr = self._fdr([r['p'] for r in reg_rows])
            reg_fdr_hc3 = self._fdr([r['p_hc3'] for r in reg_rows])
            int_fdr = self._fdr([r['p'] for r in int_rows])

            # standardized beta1 for plotting comparability across layers.
            # BUG FIX: sy must be the SD of the dependent variable actually
            # regressed (log RT when use_log_rt), not of raw RT.
            std_betas = []
            for li, reg in zip(layers_sorted, reg_rows):
                g = g_model[g_model['layer'] == li].dropna(
                    subset=['llm_lexical_evidence', 'log_freq_hal', 'word_length',
                            'ortho_n', 'token_count', 'human_rt'])
                sx = g['llm_lexical_evidence'].std(ddof=0)
                sy = reg.get('dv_sd', np.nan)
                std_betas.append(reg['beta1'] * sx / sy
                                 if (sx and _finite(sy) and sy > 0 and not np.isnan(reg['beta1']))
                                 else np.nan)

            model_layer_df_rows = []
            for li, sp, reg, inter, sfdr, rfdr, rfdr_hc3, ifdr, sb in zip(
                    layers_sorted, sp_rows, reg_rows, int_rows,
                    sp_fdr, reg_fdr, reg_fdr_hc3, int_fdr, std_betas):
                row = {
                    'model': model_name, 'layer': li, 'n': sp['n'],
                    'spearman_rho': sp['rho'], 'spearman_p': sp['p'],
                    'spearman_p_log10': sp.get('p_log10', np.nan),
                    'spearman_fdr_p': sfdr,
                    'spearman_log10_bf01': sp.get('log10_bf01', np.nan),
                    'spearman_bf01_interpretation': bf_label(
                        sp.get('log10_bf01', np.nan), self.config.BF_EVIDENCE_THRESHOLD),
                    'reg_beta1': reg['beta1'], 'reg_beta1_std': sb,
                    'reg_se': reg['se'], 'reg_t': reg['t'], 'reg_p': reg['p'],
                    'reg_p_log10': reg.get('p_log10', np.nan),
                    'reg_fdr_p': rfdr, 'reg_r2': reg['r2'], 'reg_adj_r2': reg['adj_r2'],
                    'reg_se_hc3': reg.get('se_hc3', np.nan),
                    'reg_p_hc3': reg.get('p_hc3', np.nan),
                    'reg_p_hc3_log10': reg.get('p_hc3_log10', np.nan),
                    'reg_fdr_p_hc3': rfdr_hc3,
                    'reg_se_cluster': reg.get('se_cluster', np.nan),
                    'reg_p_cluster': reg.get('p_cluster', np.nan),
                    'reg_cluster_note': reg.get('cluster_note', ''),
                    'reg_log10_bf01': reg.get('log10_bf01', np.nan),
                    'reg_bf01_interpretation': bf_label(
                        reg.get('log10_bf01', np.nan), self.config.BF_EVIDENCE_THRESHOLD),
                    'reg_beta1_with_human_accuracy_covariate': reg.get('beta_acc_cov', np.nan),
                    'reg_p_with_human_accuracy_covariate': reg.get('p_acc_cov', np.nan),
                    'human_accuracy_covariate_note': 'sensitivity only — potential bad control',
                    'rt_len_x_freq_beta': reg.get('beta_len_x_freq', np.nan),
                    'rt_len_x_freq_p_hc3': reg.get('p_len_x_freq', np.nan),
                    'rt_len_x_freq_p_hc3_log10': reg.get('p_len_x_freq_log10', np.nan),
                    'rt_sample': primary_sample,
                    'reg_n': reg['n'],
                    'freq_interaction_beta3': inter['beta3'],
                    'freq_interaction_se': inter['se'], 'freq_interaction_t': inter['t'],
                    'freq_interaction_p': inter['p'], 'freq_interaction_fdr_p': ifdr,
                    'freq_interaction_r2': inter['r2'], 'freq_interaction_n': inter['n'],
                    'used_log_rt_for_regression': use_log_rt,
                    'representation_type': PRIMARY_REPRESENTATION,
                    'readout_type': PRIMARY_READOUT,
                    'relative_depth': (li + 1) / (int(max(layers_sorted)) + 1),
                }
                model_layer_df_rows.append(row)
                layer_rows.append(row)

            model_layer_df = pd.DataFrame(model_layer_df_rows)

            # ── strongest alignment layer (largest negative significant rho) ──
            sig_rows = model_layer_df[model_layer_df['spearman_fdr_p'] < 0.05]
            exploratory = False
            if len(sig_rows):
                best_row = sig_rows.loc[sig_rows['spearman_rho'].idxmin()]
            else:
                exploratory = True
                finite = model_layer_df.dropna(subset=['spearman_rho'])
                best_row = (finite.loc[finite['spearman_rho'].idxmin()]
                           if len(finite) else None)
            best_layer = int(best_row['layer']) if best_row is not None else None

            best_reg_sig = model_layer_df[model_layer_df['reg_fdr_p'] < 0.05]
            best_reg_layer = (int(best_reg_sig.loc[best_reg_sig['reg_beta1'].abs().idxmax(), 'layer'])
                              if len(best_reg_sig) else None)

            self._plot_figures(model_name, model_layer_df, g_model, best_layer)

            freq_desc = (self._freq_group_descriptives(g_model[g_model['layer'] == best_layer])
                        if best_layer is not None else {})

            interaction_sig = bool(
                best_row is not None and
                model_layer_df.loc[model_layer_df['layer'] == best_layer,
                                   'freq_interaction_fdr_p'].iloc[0] < 0.05
            ) if best_layer is not None else False

            cross_model_rows.append({
                'model': model_name,
                'n_matched_elp_words': self.match_report.get('n_matched_words', 0),
                'elp_rt_column_used': self.rt_column,
                'best_alignment_layer': best_layer,
                'best_alignment_layer_is_exploratory_only': exploratory,
                'spearman_rho_at_best_layer': (best_row['spearman_rho']
                                               if best_row is not None else np.nan),
                'spearman_fdr_p_at_best_layer': (best_row['spearman_fdr_p']
                                                 if best_row is not None else np.nan),
                'strongest_significant_regression_layer': best_reg_layer,
                'reg_beta1_at_best_layer': (best_row['reg_beta1']
                                            if best_row is not None else np.nan),
                'reg_fdr_p_at_best_layer': (best_row['reg_fdr_p']
                                            if best_row is not None else np.nan),
                'reg_r2_at_best_layer': (best_row['reg_r2']
                                         if best_row is not None else np.nan),
                'alignment_survives_lexical_controls': bool(
                    best_row is not None and best_row['reg_fdr_p'] < 0.05),
                'freq_interaction_significant_at_best_layer': interaction_sig,
                'high_freq_mean_rt': freq_desc.get('high', {}).get('mean_human_rt', np.nan),
                'low_freq_mean_rt':  freq_desc.get('low', {}).get('mean_human_rt', np.nan),
                'high_freq_mean_llm_evidence': freq_desc.get('high', {}).get('mean_llm_evidence', np.nan),
                'low_freq_mean_llm_evidence':  freq_desc.get('low', {}).get('mean_llm_evidence', np.nan),
                'used_log_rt_for_regression': use_log_rt,
                'lexical_evidence_scoring': '5fold_out_of_fold',
                'probe_n_folds': self.config.RT_ALIGNMENT_N_FOLDS,
                'probe_cv_seed': self.config.RT_ALIGNMENT_SEED,
                'human_rt_used_in_probe_fitting': False,
                'representation_type': PRIMARY_REPRESENTATION,
            })

            if best_row is not None:
                logger.info(
                    f"[Analysis 11] {model_name}: matched "
                    f"N={self.match_report.get('n_matched_words', 0)}  "
                    f"best_layer={best_layer} "
                    f"(rho={best_row['spearman_rho']:.3f} "
                    f"FDR-p={best_row['spearman_fdr_p']:.4f})"
                )
            else:
                logger.info(f"[Analysis 11] {model_name}: no valid layer result.")

        layer_results_df = pd.DataFrame(layer_rows)
        # E4 / G4 — GLOBAL BH-FDR across every (model, layer) test, in
        # addition to the within-model correction above.
        for col, out in (('spearman_p', 'spearman_fdr_p_global'),
                         ('reg_p_hc3', 'reg_fdr_p_hc3_global')):
            if col in layer_results_df.columns:
                layer_results_df[out] = adjust_pvalues(layer_results_df[col].to_numpy(), 'fdr_bh')
        try:
            self._write_mixed_model_table(matched, use_log_rt, results_dir)
        except Exception as e:                                 # noqa: BLE001
            logger.warning(f"[Analysis 11] mixed-model table failed: {e}")
        layer_results_df.to_csv(
            os.path.join(results_dir, 'human_rt_alignment_layer_results.csv'), index=False)
        logger.info("✓ [Analysis 11] human_rt_alignment_layer_results.csv")

        cross_model_df = pd.DataFrame(cross_model_rows)
        cross_model_df.to_csv(
            os.path.join(results_dir, 'human_rt_alignment_cross_model.csv'), index=False)
        logger.info("✓ [Analysis 11] human_rt_alignment_cross_model.csv")

        match_meta = {
            'n_matched_words': 0,
            'n_unmatched_model_only': 0,
            'n_unmatched_elp_only': 0,
            'elp_rt_column_used': self.rt_column,
            'elp_accuracy_column_used': self.acc_column,
        }
        match_meta.update(self.match_report)  # overwrite with actual values if available
        match_meta['human_rt_skew'] = skew
        match_meta['used_log_rt_for_regression'] = use_log_rt
        match_meta['lexical_evidence_scoring'] = '5fold_out_of_fold'
        match_meta['probe_n_folds'] = self.config.RT_ALIGNMENT_N_FOLDS
        match_meta['probe_cv_seed'] = self.config.RT_ALIGNMENT_SEED
        match_meta['human_rt_used_in_probe_fitting'] = False
        match_meta['representation_type'] = PRIMARY_REPRESENTATION
        match_meta['readout_type'] = PRIMARY_READOUT
        match_meta['rt_primary_sample'] = primary_sample
        match_meta['rt_accuracy_filter_threshold'] = thr
        match_meta['lexical_evidence_definition'] = (
            "S = logit(P(WORD | h)), h = hidden state at the first-output "
            "PREDICTION position (final prompt token) at layer l; P from a "
            "5-fold cross-fitted WORD/NONWORD probe (out-of-fold scores only). "
            "This is the TRAINED-PROBE readout; the native LM-head counterpart "
            "S = logit(YES) - logit(NO), which requires no fitting at all, is "
            "reported in layerwise_native_lm_head_rt_alignment.csv.")
        match_meta['interpretation_note'] = (
            "Alignment refers to a statistical association between LLM "
            "layer-wise lexical-evidence representations and human ELP "
            "reaction times, not a claim that the LLM has a reaction time. "
            "Human RT alignment, LLM frequency representation, LLM LDT "
            "performance, and causal/intervention results are distinct "
            "metrics reported in separate files and must not be merged."
        )
        with open(os.path.join(results_dir, 'human_rt_alignment_metadata.json'), 'w') as f:
            json.dump(match_meta, f, indent=2, default=float)
        logger.info("✓ [Analysis 11] human_rt_alignment_metadata.json")

        # Figure 4 (spec §57): layer-wise human RT alignment, all models
        try:
            self._plot_layerwise_rt_alignment(layer_results_df)
        except Exception as e:
            logger.warning(f"[Analysis 11] layer-wise RT alignment figure skipped: {e}")

        # ════════════════════════════════════════════════════════════════
        # Analysis 11b (NEW): Layer-wise lexical-evidence EMERGENCE
        # ════════════════════════════════════════════════════════════════
        print(f"\n{'='*64}\nANALYSIS 11b: LAYER-WISE LEXICAL-EVIDENCE EMERGENCE\n{'='*64}")
        try:
            self._run_layerwise_emergence_analysis(matched, use_log_rt)
        except Exception as e:
            logger.error(f"[Analysis 11b] Layer-wise emergence analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ── E5 / E6 / I7 ────────────────────────────────────────────────────
    def _write_rt_diagnostics(self, matched: pd.DataFrame, results_dir: str):
        """E6: skew, excess kurtosis, D'Agostino K², |z| > RT_OUTLIER_Z and
        Tukey-fence outliers for raw and log RT (one row per unique word) +
        histogram / QQ figure (I7). E5: for every model, the words with a
        valid ELP RT (and passing the accuracy filter) compared with ALL LDT
        words on the covariates (SMD), so readers can judge whether the RT
        analysis sample is representative."""
        words = matched.drop_duplicates('word')
        rows = []
        for name, v in (('raw_rt', words['human_rt'].to_numpy(float)),
                        ('log_rt', np.log(words['human_rt'].to_numpy(float)))):
            v = v[np.isfinite(v)]
            if len(v) < 8:
                continue
            z = (v - v.mean()) / v.std(ddof=1)
            q1, q3 = np.percentile(v, [25, 75])
            iqr = q3 - q1
            k2 = stats.normaltest(v)
            rows.append({'variable': name, 'n_words': int(len(v)), 'mean': float(v.mean()),
                         'sd': float(v.std(ddof=1)), 'median': float(np.median(v)),
                         'skew': float(stats.skew(v)), 'excess_kurtosis': float(stats.kurtosis(v)),
                         'dagostino_k2': float(k2.statistic), 'dagostino_p': float(k2.pvalue),
                         f'n_abs_z_gt_{self.config.RT_OUTLIER_Z:g}': int((np.abs(z) > self.config.RT_OUTLIER_Z).sum()),
                         'n_tukey_outliers': int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())})
        pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'rt_distribution_diagnostics.csv'),
                                  index=False)
        try:
            v = words['human_rt'].to_numpy(float)
            v = v[np.isfinite(v) & (v > 0)]
            fig, ax = plt.subplots(2, 2, figsize=(12, 9))
            ax[0, 0].hist(v, bins=60, color='#1A535C'); ax[0, 0].set(title='Raw ELP RT', xlabel='ms')
            ax[0, 1].hist(np.log(v), bins=60, color='#2E86AB'); ax[0, 1].set(title='log RT', xlabel='log ms')
            stats.probplot(v, dist='norm', plot=ax[1, 0]); ax[1, 0].set_title('QQ — raw RT')
            stats.probplot(np.log(v), dist='norm', plot=ax[1, 1]); ax[1, 1].set_title('QQ — log RT')
            fig.suptitle('I7 — ELP RT distribution diagnostics (unique matched words)')
            fig.tight_layout()
            fig.savefig(os.path.join(self.config.RT_ALIGNMENT_DIR, 'figures',
                                     'rt_distribution_diagnostics.png'), dpi=200)
            plt.close(fig)
        except Exception as e:                                 # noqa: BLE001
            logger.warning(f"[I7] RT diagnostics figure failed: {e}")
        if not self.word_level_rows:
            return
        full = pd.concat(self.word_level_rows, ignore_index=True)
        full = full[full['layer'] == full.groupby('model')['layer'].transform('min')]
        thr = self.config.RT_ALIGNMENT_MIN_HUMAN_ACCURACY
        ov = []
        for model, g in full.groupby('model'):
            gm = matched[matched['model'] == model].drop_duplicates('word')
            gf = gm[gm['human_accuracy'] >= thr] if thr is not None else gm
            r = {'model': model, 'n_ldt_words': int(g['word'].nunique()),
                 'n_with_valid_elp_rt': int(gm['word'].nunique()),
                 'n_after_accuracy_filter': int(gf['word'].nunique()),
                 'accuracy_filter_threshold': thr}
            for c in ('log_freq_hal', 'word_length', 'ortho_n', 'token_count'):
                r[f'smd_rt_sample_vs_all_{c}'] = standardized_mean_difference(
                    gf[c].to_numpy(float), g[c].to_numpy(float))
            ov.append(r)
        pd.DataFrame(ov).to_csv(os.path.join(results_dir, 'rt_item_overlap_statistics.csv'),
                                index=False)
        logger.info("✓ [E5/E6] rt_item_overlap_statistics.csv, rt_distribution_diagnostics.csv")

    def _sensitivity_layer_table(self, df: pd.DataFrame, sample: str, results_dir: str):
        """E1 sensitivity: the same per-layer Spearman + HC3 regression on the
        non-primary sample (e.g. all items, without the accuracy filter)."""
        skew = float(stats.skew(df['human_rt'].dropna()))
        use_log = abs(skew) > self.config.RT_LOG_SKEW_THRESHOLD
        rows = []
        for (model, li), g in df.groupby(['model', 'layer']):
            sp = self._spearman_per_layer(g)
            rg = self._regression_per_layer(g, use_log)
            rows.append({'model': model, 'layer': li, 'rt_sample': sample, 'n': sp['n'],
                         'spearman_rho': sp['rho'], 'spearman_p': sp['p'],
                         'reg_beta1': rg['beta1'], 'reg_p_hc3': rg['p_hc3'],
                         'used_log_rt_for_regression': use_log})
        out = pd.DataFrame(rows)
        if not out.empty:
            out['spearman_fdr_p_global'] = adjust_pvalues(out['spearman_p'].to_numpy())
            out.to_csv(os.path.join(results_dir, f'human_rt_alignment_layer_results_{sample}.csv'),
                       index=False)

    def _write_mixed_model_table(self, matched: pd.DataFrame, use_log_rt: bool,
                                 results_dir: str):
        rows = []
        for model, gm in matched.groupby('model'):
            L = int(gm['layer'].max()) + 1
            layers = select_layers(L, self.config.RT_MIXED_MODEL_LAYERS)
            for li in layers:
                g = gm[gm['layer'] == li]
                r = self._mixed_model_per_layer(g, use_log_rt)
                rows.append({'model': model, 'layer': li, 'relative_depth': (li + 1) / L,
                             'formula': ('RT ~ z(S) + z(log_freq) + z(Ortho_N) + z(tokens) '
                                         f'+ (1 | {self.config.RT_MIXED_MODEL_GROUP})'),
                             'used_log_rt': use_log_rt, **r})
        out = pd.DataFrame(rows)
        if not out.empty:
            out['mixed_p_fdr_bh_global'] = adjust_pvalues(out['mixed_p'].to_numpy())
            out.to_csv(os.path.join(results_dir, 'human_rt_mixed_model_results.csv'), index=False)
            logger.info("✓ [E3] human_rt_mixed_model_results.csv")

    def _plot_layerwise_rt_alignment(self, layer_df: pd.DataFrame):
        if layer_df is None or layer_df.empty:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
        for m, g in layer_df.groupby('model'):
            g = g.sort_values('layer')
            ax1.plot(g['relative_depth'], g['spearman_rho'], 'o-', lw=2, ms=4, label=m)
            sig = g['spearman_fdr_p'] < 0.05
            ax1.scatter(g.loc[sig, 'relative_depth'], g.loc[sig, 'spearman_rho'],
                        s=40, facecolors='none', edgecolors='black', lw=1.0)
            ax2.plot(g['relative_depth'], g['reg_beta1_std'], 's-', lw=2, ms=4, label=m)
        for ax in (ax1, ax2):
            ax.axhline(0, color='gray', ls='--', lw=1.1)
            ax.set_xlabel('Relative depth (layer+1)/L'); ax.grid(True, alpha=0.3)
        ax1.set(ylabel='Spearman ρ (lexical evidence S vs human RT)',
                title='Layer-wise human RT alignment (circles: FDR p<.05)')
        ax2.set(ylabel='Standardised β (S | freq, length, Ortho_N, tokens)',
                title='Alignment after lexical controls')
        ax1.legend(fontsize=8)
        fig.suptitle('Trained-probe lexical evidence at the first-output prediction '
                     'position vs ELP RT (association; layer depth is NOT a reaction '
                     'time)', fontsize=12)
        plt.tight_layout()
        for d in (os.path.join(self.config.RT_ALIGNMENT_DIR, 'figures'),
                  self.config.PRIMARY_DIR):
            plt.savefig(os.path.join(d, 'layerwise_human_rt_alignment.png'),
                        bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info("✓ layerwise_human_rt_alignment.png")

    # ════════════════════════════════════════════════════════════════════
    # ANALYSIS 11b: LAYER-WISE LEXICAL-EVIDENCE EMERGENCE
    # ════════════════════════════════════════════════════════════════════


    LEXICAL_MEASURES = [
        'max_llm_word_probability',
        'layer_of_max_probability',
        'layer_reaching_80pct',
        'layer_reaching_90pct',
        'lexical_probability_auc',
        'early_layer_lexical_evidence',

        # Normalized-depth trajectory measures
        'early_0_25_evidence',
        'early_mid_25_50_evidence',
        'late_mid_50_75_evidence',
        'late_75_100_evidence',
        'late_minus_early_evidence',
    ]

    def _compute_word_trajectory_measures(self, matched: pd.DataFrame) -> pd.DataFrame:
        """
        Collapses the word x layer table (one row per model, word, layer)
        into one row per (model, word) carrying the five emergence
        measures (§3-7) plus the word-level covariates needed downstream.
        The full layer-wise P(WORD) trajectory is used for every measure
        — the word is never reduced to a single layer before this point.
        """
        rows = []
        # normalised transformer depth needs each model's max layer index
        model_max_layer = matched.groupby('model')['layer'].max().to_dict()

        for (model_name, word), g in matched.groupby(['model', 'word']):
            g = g.sort_values('layer')
            layers = g['layer'].to_numpy(dtype=float)
            probs  = g['llm_word_probability'].to_numpy(dtype=float)
            if len(layers) < 2:
                continue

            max_p   = float(np.max(probs))
            l_max   = float(layers[int(np.argmax(probs))])

            def _threshold_layer(tau: float):
                hit = np.where(probs >= tau)[0]
                return float(layers[hit[0]]) if len(hit) else np.nan

            l90 = _threshold_layer(0.90)
            l80 = _threshold_layer(0.80)

            auc = float(scipy.integrate.trapezoid(probs, layers))

            # Normalize transformer depth to [0, 1]
            # 0.0 = first layer, 1.0 = final layer
            max_layer = model_max_layer.get(model_name, layers.max())
            depth = layers / max_layer if max_layer > 0 else layers

            # Evidence in normalized depth regions
            early_mask       = (depth >= 0.00) & (depth < 0.25)
            early_mid_mask   = (depth >= 0.25) & (depth < 0.50)
            late_mid_mask    = (depth >= 0.50) & (depth < 0.75)
            late_mask        = (depth >= 0.75) & (depth <= 1.00)

            early_evidence = (
                float(np.mean(probs[early_mask]))
                if early_mask.sum() > 0 else np.nan
            )

            early_mid_evidence = (
                float(np.mean(probs[early_mid_mask]))
                if early_mid_mask.sum() > 0 else np.nan
            )

            late_mid_evidence = (
                float(np.mean(probs[late_mid_mask]))
                if late_mid_mask.sum() > 0 else np.nan
            )

            late_evidence = (
                float(np.mean(probs[late_mask]))
                if late_mask.sum() > 0 else np.nan
            )

            # How much lexical evidence develops from early to late depth
            late_minus_early = (
                late_evidence - early_evidence
                if np.isfinite(late_evidence) and np.isfinite(early_evidence)
                else np.nan
            )

            first = g.iloc[0]
            rows.append({
                'model': model_name, 'word': word,
                'human_rt':      first['human_rt'],
                'human_accuracy': first.get('human_accuracy', np.nan),
                'log_freq_hal':  first['log_freq_hal'],
                'word_length':   first['word_length'],
                'ortho_n':       first['ortho_n'],
                'token_count':   first['token_count'],
                'freq_group':    first['freq_group'],
                'max_llm_word_probability':     max_p,
                'layer_of_max_probability':     l_max,
                'layer_reaching_80pct':         l80,
                'layer_reaching_90pct':         l90,
                'lexical_probability_auc':        auc,
                'early_layer_lexical_evidence':  early_evidence,

                # New normalized-depth trajectory measures
                'early_0_25_evidence':            early_evidence,
                'early_mid_25_50_evidence':       early_mid_evidence,
                'late_mid_50_75_evidence':        late_mid_evidence,
                'late_75_100_evidence':           late_evidence,
                'late_minus_early_evidence':      late_minus_early,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        
        def _ease_01(rt: pd.Series) -> pd.Series:
            rng = rt.max() - rt.min()
            if rng > 0:
                return 1.0 - (rt - rt.min()) / rng
            return pd.Series(np.nan, index=rt.index)

        df['human_ease_01'] = df.groupby('model')['human_rt'].transform(_ease_01)
        return df

    def _spearman_pair(self, x: pd.Series, y: pd.Series) -> dict:
        gg = pd.concat([x, y], axis=1).dropna()
        if len(gg) < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER:
            return {'rho': np.nan, 'p': np.nan, 'n': len(gg)}
        rho, p = stats.spearmanr(gg.iloc[:, 0], gg.iloc[:, 1])
        return {'rho': rho, 'p': p, 'n': len(gg)}

    def _regression_measure_vs_rt(self, g: pd.DataFrame, measure: str,
                                  use_log_rt: bool) -> dict:
        cols = [measure, 'log_freq_hal', 'word_length', 'ortho_n', 'token_count']
        gg = g.dropna(subset=cols + ['human_rt'])
        if len(gg) < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER:
            return {k: np.nan for k in ['beta', 'se', 't', 'p', 'r2', 'adj_r2']} | {'n': len(gg)}
        y = np.log(gg['human_rt']) if use_log_rt else gg['human_rt']
        X = sm.add_constant(gg[cols].astype(float))
        try:
            model = sm.OLS(y.astype(float), X).fit()
        except Exception:
            return {k: np.nan for k in ['beta', 'se', 't', 'p', 'r2', 'adj_r2']} | {'n': len(gg)}
        return {
            'beta': model.params.get(measure, np.nan),
            'se':   model.bse.get(measure, np.nan),
            't':    model.tvalues.get(measure, np.nan),
            'p':    model.pvalues.get(measure, np.nan),
            'r2':   model.rsquared, 'adj_r2': model.rsquared_adj,
            'n':    len(gg),
        }

    def _regression_measure_vs_freqgroup(self, g: pd.DataFrame, measure: str) -> dict:
        """Measure_i = b0 + b1*FreqGroup + controls(length, ortho_n, token_count)."""
        gg = g[g['freq_group'].isin(['high', 'low'])].copy()
        cols = ['word_length', 'ortho_n', 'token_count']
        gg = gg.dropna(subset=cols + [measure])
        if len(gg) < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER:
            return {k: np.nan for k in ['beta', 'se', 't', 'p', 'r2']} | {'n': len(gg)}
        gg['FreqGroup'] = (gg['freq_group'] == 'high').astype(float)
        X = sm.add_constant(gg[['FreqGroup'] + cols].astype(float))
        try:
            model = sm.OLS(gg[measure].astype(float), X).fit()
        except Exception:
            return {k: np.nan for k in ['beta', 'se', 't', 'p', 'r2']} | {'n': len(gg)}
        return {
            'beta': model.params.get('FreqGroup', np.nan),
            'se':   model.bse.get('FreqGroup', np.nan),
            't':    model.tvalues.get('FreqGroup', np.nan),
            'p':    model.pvalues.get('FreqGroup', np.nan),
            'r2':   model.rsquared, 'n': len(gg),
        }

    def _interaction_measure_x_freqgroup(self, g: pd.DataFrame, measure: str,
                                         use_log_rt: bool) -> dict:
        gg = g[g['freq_group'].isin(['high', 'low'])].copy()
        cols = ['word_length', 'ortho_n', 'token_count']
        gg = gg.dropna(subset=cols + [measure, 'human_rt'])
        if len(gg) < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER:
            return {k: np.nan for k in ['beta3', 'se', 't', 'p', 'r2']} | {'n': len(gg)}
        gg['FreqGroup'] = (gg['freq_group'] == 'high').astype(float)
        gg['Interaction'] = gg[measure] * gg['FreqGroup']
        y = np.log(gg['human_rt']) if use_log_rt else gg['human_rt']
        X = sm.add_constant(gg[[measure, 'FreqGroup', 'Interaction'] + cols].astype(float))
        try:
            model = sm.OLS(y.astype(float), X).fit()
        except Exception:
            return {k: np.nan for k in ['beta3', 'se', 't', 'p', 'r2']} | {'n': len(gg)}
        return {
            'beta3': model.params.get('Interaction', np.nan),
            'se':    model.bse.get('Interaction', np.nan),
            't':     model.tvalues.get('Interaction', np.nan),
            'p':     model.pvalues.get('Interaction', np.nan),
            'r2':    model.rsquared, 'n': len(gg),
        }

    def _emergence_descriptives(self, g: pd.DataFrame) -> dict:
        out = {}
        for grp in ['high', 'low']:
            sub = g[g['freq_group'] == grp]
            out[grp] = {}
            for m in self.LEXICAL_MEASURES:
                vals = sub[m].dropna()
                out[grp][m] = {
                    'n': len(vals),
                    'mean':   float(vals.mean())   if len(vals) else np.nan,
                    'median': float(vals.median()) if len(vals) else np.nan,
                    'sd':     float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                }
        return out

    def _plot_emergence_figures(self, model_name: str, wdf: pd.DataFrame):
        fig_dir = os.path.join(self.config.RT_ALIGNMENT_DIR, "figures")

        def _scatter(x, y, xlabel, ylabel, title, fname, trend=True):
            gg = wdf[[x, y]].dropna()
            if len(gg) < 3:
                return
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.scatter(gg[x], gg[y], s=14, alpha=0.5, color='#1A535C')
            if trend and len(gg) > 2:
                m, b = np.polyfit(gg[x], gg[y], 1)
                xs = np.linspace(gg[x].min(), gg[x].max(), 50)
                ax.plot(xs, m * xs + b, color='#D62246', lw=2)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(f'{model_name} — {title}')
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f'{model_name}_{fname}.png'))
            plt.close(fig)

        # Figure A
        _scatter('max_llm_word_probability', 'human_rt',
                 'Max lexical probability  P_max', 'Human mean RT (ELP)',
                 'Fig A — human RT vs maximum lexical evidence',
                 'figA_maxprob_vs_rt')
        _scatter('max_llm_word_probability', 'human_ease_01',
                 'Max lexical probability  P_max', 'Human ease (1 = fastest)',
                 'Fig A(rev) — human ease vs maximum lexical evidence',
                 'figA_maxprob_vs_ease')
        # Figure B
        _scatter('layer_of_max_probability', 'human_rt',
                 'Layer of maximum probability  L_max', 'Human mean RT (ELP)',
                 'Fig B — human RT vs layer of max probability',
                 'figB_layerofmax_vs_rt')
        # Figure C
        _scatter('layer_reaching_90pct', 'human_rt',
                 'Layer reaching 90% probability  L_90', 'Human mean RT (ELP)',
                 'Fig C — human RT vs 90%-threshold layer',
                 'figC_layer90_vs_rt')
        # Figure D
        _scatter('lexical_probability_auc', 'human_rt',
                 'Lexical probability AUC across layers', 'Human mean RT (ELP)',
                 'Fig D — human RT vs cumulative lexical evidence (AUC)',
                 'figD_auc_vs_rt')

    def _plot_trajectory_examples(self, model_name: str, matched: pd.DataFrame,
                                  wdf: pd.DataFrame):
        """
        Figure E — representative layer-wise P(WORD) trajectories for
        high- vs low-frequency words. Examples are chosen by a
        transparent, pre-specified rule (closest to each frequency
        group's median log-frequency among matched words), NOT cherry-
        picked to support the hypothesis.
        """
        fig_dir = os.path.join(self.config.RT_ALIGNMENT_DIR, "figures")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        colors = {'high': '#2E86AB', 'low': '#D62246'}
        for grp in ['high', 'low']:
            sub = wdf[wdf['freq_group'] == grp].dropna(subset=['log_freq_hal'])
            if sub.empty:
                continue
            med = sub['log_freq_hal'].median()
            sub = sub.assign(_d=(sub['log_freq_hal'] - med).abs()).sort_values('_d')
            example_words = sub['word'].head(3).tolist()
            for i, w in enumerate(example_words):
                traj = matched[(matched['model'] == model_name) &
                               (matched['word'] == w)].sort_values('layer')
                if traj.empty:
                    continue
                ax.plot(traj['layer'], traj['llm_word_probability'],
                       color=colors[grp], alpha=0.8, lw=1.8,
                       label=f'{grp}-freq (median-representative)' if i == 0 else None,
                       marker='o', markersize=3)
        ax.axhline(0.90, color='gray', ls='--', lw=1, label='τ=0.90')
        ax.set_xlabel('Transformer layer'); ax.set_ylabel('P(WORD | h_l)')
        ax.set_title(f'{model_name} — Fig E: representative lexical-evidence '
                    f'trajectories\n(high- vs low-frequency, median-representative words)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'{model_name}_figE_trajectory_examples.png'))
        plt.close(fig)

    def _run_layerwise_emergence_analysis(self, matched: pd.DataFrame, use_log_rt: bool):
        """
        Runs the layer-wise emergence analysis on the matched word-level data.
        Fixed to handle missing columns gracefully.
        """
        wdf = self._compute_word_trajectory_measures(matched)
        if wdf.empty:
            logger.warning("[Analysis 11b] No words with a full layer trajectory — "
                           "emergence analysis produced no output.")
            return

        results_dir = os.path.join(self.config.RT_ALIGNMENT_DIR, "results")

        # ── Define columns that should exist ──────────────────────────────
        word_level_cols = [
            'model', 'word', 'human_rt', 'human_ease_01',
            'log_freq_hal', 'word_length', 'ortho_n', 'token_count',
            'max_llm_word_probability',
            'layer_of_max_probability',
            'layer_reaching_80pct',
            'layer_reaching_90pct',
            'lexical_probability_auc',
            'early_layer_lexical_evidence',
            'early_0_25_evidence',
            'early_mid_25_50_evidence',
            'late_mid_50_75_evidence',
            'late_75_100_evidence',
            'late_minus_early_evidence',
        ]
        
        # ── Only keep columns that actually exist in wdf ──────────────────
        existing_cols = [col for col in word_level_cols if col in wdf.columns]
        logger.info(f"[Analysis 11b] Saving {len(existing_cols)}/{len(word_level_cols)} "
                   f"available columns to CSV")
        
        if not existing_cols:
            logger.warning("[Analysis 11b] No expected columns found in wdf — "
                          "checking available columns: %s", list(wdf.columns))
            return
        
        wdf[existing_cols].to_csv(
            os.path.join(results_dir, 'human_rt_layerwise_emergence_word_level.csv'),
            index=False)
        # spec-named copy (§62), tagged with the representation type
        _em = wdf[existing_cols].copy()
        _em.insert(1, 'representation_type', PRIMARY_REPRESENTATION)
        _em.to_csv(os.path.join(results_dir, 'layerwise_lexical_evidence_emergence.csv'),
                   index=False)
        logger.info(f"✓ [Analysis 11b] human_rt_layerwise_emergence_word_level.csv "
                    f"({len(wdf)} word x model rows)")

        per_model_rows, cross_model_rows = [], []

        for model_name, g in wdf.groupby('model'):
            # ── Only plot if we have required columns ──────────────────────
            if 'max_llm_word_probability' in wdf.columns and 'human_rt' in wdf.columns:
                self._plot_emergence_figures(model_name, g)
            else:
                logger.warning(f"[Analysis 11b] {model_name}: missing required columns for plotting")
            
            # ── Only plot trajectories if we have layer-wise data ──────────
            if matched is not None and not matched.empty and 'model' in matched.columns:
                try:
                    self._plot_trajectory_examples(model_name, matched, g)
                except Exception as e:
                    logger.warning(f"[Analysis 11b] {model_name}: trajectory plot failed: {e}")
            else:
                logger.warning(f"[Analysis 11b] {model_name}: no matched data for trajectory plots")

            # ── §10-12: strength / depth / accumulation vs RT & ease ──────
            # Only compute correlations if we have required columns
            corr = {}
            if 'max_llm_word_probability' in g.columns and 'human_rt' in g.columns:
                corr['maxprob_vs_rt'] = self._spearman_pair(g['max_llm_word_probability'], g['human_rt'])
            if 'max_llm_word_probability' in g.columns and 'human_ease_01' in g.columns:
                corr['maxprob_vs_ease'] = self._spearman_pair(g['max_llm_word_probability'], g['human_ease_01'])
            if 'layer_of_max_probability' in g.columns and 'human_rt' in g.columns:
                corr['layermax_vs_rt'] = self._spearman_pair(g['layer_of_max_probability'], g['human_rt'])
            if 'layer_reaching_90pct' in g.columns and 'human_rt' in g.columns:
                corr['layer90_vs_rt'] = self._spearman_pair(g['layer_reaching_90pct'], g['human_rt'])
            if 'layer_reaching_80pct' in g.columns and 'human_rt' in g.columns:
                corr['layer80_vs_rt'] = self._spearman_pair(g['layer_reaching_80pct'], g['human_rt'])
            if 'lexical_probability_auc' in g.columns and 'human_rt' in g.columns:
                corr['auc_vs_rt'] = self._spearman_pair(g['lexical_probability_auc'], g['human_rt'])
            if 'lexical_probability_auc' in g.columns and 'human_ease_01' in g.columns:
                corr['auc_vs_ease'] = self._spearman_pair(g['lexical_probability_auc'], g['human_ease_01'])
            if 'early_layer_lexical_evidence' in g.columns and 'human_rt' in g.columns:
                corr['early_vs_rt'] = self._spearman_pair(g['early_layer_lexical_evidence'], g['human_rt'])

            # ── FDR correction for correlations ─────────────────────────────
            corr_fdr = {}
            if corr:
                corr_keys = list(corr.keys())
                corr_pvals = [corr[k].get('p', np.nan) for k in corr_keys]
                corr_fdr = dict(zip(corr_keys, self._fdr(corr_pvals)))

            # ── §13: regression per LLM-derived measure ──────────────────
            reg = {}
            available_measures = [m for m in self.LEXICAL_MEASURES if m in g.columns]
            for m in available_measures:
                try:
                    reg[m] = self._regression_measure_vs_rt(g, m, use_log_rt)
                except Exception as e:
                    logger.warning(f"[Analysis 11b] Regression for {m} failed: {e}")
                    reg[m] = {k: np.nan for k in ['beta', 'se', 't', 'p', 'r2', 'adj_r2', 'n']}
            
            reg_fdr = {}
            if reg:
                reg_keys = list(reg.keys())
                reg_fdr = dict(zip(reg_keys,
                                   self._fdr([reg[m].get('p', np.nan) for m in reg_keys])))

            # ── §14: measure ~ FreqGroup, and RT ~ measure x FreqGroup ──
            freqgrp_reg = {}
            for m in available_measures:
                try:
                    freqgrp_reg[m] = self._regression_measure_vs_freqgroup(g, m)
                except Exception as e:
                    logger.warning(f"[Analysis 11b] FreqGroup regression for {m} failed: {e}")
                    freqgrp_reg[m] = {k: np.nan for k in ['beta', 'se', 't', 'p', 'r2', 'n']}

            freqgrp_fdr = {}
            if freqgrp_reg:
                fgrp_keys = list(freqgrp_reg.keys())
                freqgrp_fdr = dict(zip(fgrp_keys,
                                       self._fdr([freqgrp_reg[m].get('p', np.nan) for m in fgrp_keys])))

            inter = {}
            for m in available_measures:
                try:
                    inter[m] = self._interaction_measure_x_freqgroup(g, m, use_log_rt)
                except Exception as e:
                    logger.warning(f"[Analysis 11b] Interaction for {m} failed: {e}")
                    inter[m] = {k: np.nan for k in ['beta3', 'se', 't', 'p', 'r2', 'n']}

            inter_fdr = {}
            if inter:
                inter_keys = list(inter.keys())
                inter_fdr = dict(zip(inter_keys,
                                     self._fdr([inter[m].get('p', np.nan) for m in inter_keys])))

            desc = self._emergence_descriptives(g)

            for m in available_measures:
                per_model_rows.append({
                    'model': model_name, 'measure': m,
                    'reg_beta': reg.get(m, {}).get('beta', np.nan),
                    'reg_se': reg.get(m, {}).get('se', np.nan),
                    'reg_t': reg.get(m, {}).get('t', np.nan),
                    'reg_p': reg.get(m, {}).get('p', np.nan),
                    'reg_fdr_p': reg_fdr.get(m, np.nan),
                    'reg_r2': reg.get(m, {}).get('r2', np.nan),
                    'reg_adj_r2': reg.get(m, {}).get('adj_r2', np.nan),
                    'reg_n': reg.get(m, {}).get('n', 0),
                    'freqgroup_beta': freqgrp_reg.get(m, {}).get('beta', np.nan),
                    'freqgroup_se': freqgrp_reg.get(m, {}).get('se', np.nan),
                    'freqgroup_t': freqgrp_reg.get(m, {}).get('t', np.nan),
                    'freqgroup_p': freqgrp_reg.get(m, {}).get('p', np.nan),
                    'freqgroup_fdr_p': freqgrp_fdr.get(m, np.nan),
                    'freqgroup_n': freqgrp_reg.get(m, {}).get('n', 0),
                    'rt_x_freqgroup_interaction_beta3': inter.get(m, {}).get('beta3', np.nan),
                    'rt_x_freqgroup_interaction_se': inter.get(m, {}).get('se', np.nan),
                    'rt_x_freqgroup_interaction_t': inter.get(m, {}).get('t', np.nan),
                    'rt_x_freqgroup_interaction_p': inter.get(m, {}).get('p', np.nan),
                    'rt_x_freqgroup_interaction_fdr_p': inter_fdr.get(m, np.nan),
                    'rt_x_freqgroup_interaction_n': inter.get(m, {}).get('n', 0),
                    'high_freq_mean': desc.get('high', {}).get(m, {}).get('mean', np.nan),
                    'high_freq_median': desc.get('high', {}).get(m, {}).get('median', np.nan),
                    'high_freq_sd': desc.get('high', {}).get(m, {}).get('sd', np.nan),
                    'high_freq_n': desc.get('high', {}).get(m, {}).get('n', 0),
                    'low_freq_mean': desc.get('low', {}).get(m, {}).get('mean', np.nan),
                    'low_freq_median': desc.get('low', {}).get(m, {}).get('median', np.nan),
                    'low_freq_sd': desc.get('low', {}).get(m, {}).get('sd', np.nan),
                    'low_freq_n': desc.get('low', {}).get(m, {}).get('n', 0),
                    'used_log_rt_for_regression': use_log_rt,
                })

            # ── Cross-model summary row ───────────────────────────────────
            cross_model_rows.append({
                'model': model_name,
                'n_words': len(g),
                'maxprob_vs_rt_rho': corr.get('maxprob_vs_rt', {}).get('rho', np.nan),
                'maxprob_vs_rt_fdr_p': corr_fdr.get('maxprob_vs_rt', np.nan),
                'maxprob_vs_ease_rho': corr.get('maxprob_vs_ease', {}).get('rho', np.nan),
                'maxprob_vs_ease_fdr_p': corr_fdr.get('maxprob_vs_ease', np.nan),
                'layermax_vs_rt_rho': corr.get('layermax_vs_rt', {}).get('rho', np.nan),
                'layermax_vs_rt_fdr_p': corr_fdr.get('layermax_vs_rt', np.nan),
                'layer90_vs_rt_rho': corr.get('layer90_vs_rt', {}).get('rho', np.nan),
                'layer90_vs_rt_fdr_p': corr_fdr.get('layer90_vs_rt', np.nan),
                'auc_vs_rt_rho': corr.get('auc_vs_rt', {}).get('rho', np.nan),
                'auc_vs_rt_fdr_p': corr_fdr.get('auc_vs_rt', np.nan),
                'auc_vs_ease_rho': corr.get('auc_vs_ease', {}).get('rho', np.nan),
                'auc_vs_ease_fdr_p': corr_fdr.get('auc_vs_ease', np.nan),
                'early_vs_rt_rho': corr.get('early_vs_rt', {}).get('rho', np.nan),
                'early_vs_rt_fdr_p': corr_fdr.get('early_vs_rt', np.nan),
                'strength_survives_controls': bool(
                    not np.isnan(reg_fdr.get('max_llm_word_probability', np.nan))
                    and reg_fdr.get('max_llm_word_probability', np.nan) < 0.05
                ) if 'max_llm_word_probability' in reg_fdr else False,
                'depth_Lmax_survives_controls': bool(
                    reg_fdr.get('layer_of_max_probability', np.nan) < 0.05
                ) if 'layer_of_max_probability' in reg_fdr else False,
                'depth_L90_survives_controls': bool(
                    reg_fdr.get('layer_reaching_90pct', np.nan) < 0.05
                ) if 'layer_reaching_90pct' in reg_fdr else False,
                'accumulation_AUC_survives_controls': bool(
                    reg_fdr.get('lexical_probability_auc', np.nan) < 0.05
                ) if 'lexical_probability_auc' in reg_fdr else False,
                'early_evidence_survives_controls': bool(
                    reg_fdr.get('early_layer_lexical_evidence', np.nan) < 0.05
                ) if 'early_layer_lexical_evidence' in reg_fdr else False,
            })

            if 'maxprob_vs_rt' in corr:
                logger.info(
                    f"[Analysis 11b] {model_name}: N={len(g)}  "
                    f"P_max~RT rho={corr['maxprob_vs_rt'].get('rho', np.nan):.3f} "
                    f"(FDR-p={corr_fdr.get('maxprob_vs_rt', np.nan):.4f})  |  "
                    f"AUC~RT rho={corr.get('auc_vs_rt', {}).get('rho', np.nan):.3f} "
                    f"(FDR-p={corr_fdr.get('auc_vs_rt', np.nan):.4f})"
                )

        if per_model_rows:
            pd.DataFrame(per_model_rows).to_csv(
                os.path.join(results_dir, 'human_rt_layerwise_emergence_results.csv'),
                index=False)
            logger.info("✓ [Analysis 11b] human_rt_layerwise_emergence_results.csv")

        if cross_model_rows:
            pd.DataFrame(cross_model_rows).to_csv(
                os.path.join(results_dir, 'human_rt_layerwise_emergence_cross_model.csv'),
                index=False)
            logger.info("✓ [Analysis 11b] human_rt_layerwise_emergence_cross_model.csv")

        meta = {
            'thresholds': {'tau_primary': 0.90, 'tau_robustness': 0.80,
                           'early_depth_cutoff_normalized': 0.25},
            'measures': [m for m in self.LEXICAL_MEASURES if m in wdf.columns],
            'n_words_total': len(wdf),
            'used_log_rt_for_regression': use_log_rt,
            'interpretation_note': (
                "These measures characterize the layer-wise word-vs-nonword "
                "probe probability trajectory P(WORD|h_l) for each word: "
                "strength (max probability), depth of emergence (layer of "
                "max / threshold-crossing layers), accumulation (AUC across "
                "layers), and early emergence (mean probability in the "
                "first 25% of normalized depth). None of these is inference "
                "latency or wall-clock compute time, and none should be "
                "called 'LLM reaction time'. Layer indices describe network "
                "depth, not processing time."
            ),
        }
        with open(os.path.join(results_dir, 'human_rt_layerwise_emergence_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=float)
        logger.info("✓ [Analysis 11b] human_rt_layerwise_emergence_metadata.json")

class MultiModelExperiment:

    OVERALL_C = '#1A535C'
    HIGH_C    = '#2E86AB'
    LOW_C     = '#D62246'
    DIFF_SIG  = '#2DC653'
    DIFF_NS   = '#AAAAAA'

    def __init__(self, config: Config):
        self.config = config
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.3)
        plt.rcParams.update({'figure.dpi':150, 'savefig.dpi':300})

        # ── Persistent state for Analysis 9 (cross-model transfer) ─────
        self.ext = ExtendedAnalyses(config)
        self._transfer_snapshots: dict[str, dict] = {}
        self.cross_model_transfer_rows: list[dict] = []

        # ── Analysis 11: Human-LLM Reaction-Time Alignment ─────────────
        self.rt_alignment = HumanRTAlignmentAnalysis(config)

        # ── Spec-named consolidated outputs (§62), accumulated per model ─
        self.spec_rows: dict[str, list] = {}
        self.model_metadata: dict[str, dict] = {}
        # ── NAACL revision state ────────────────────────────────────────
        self._naacl_rows: dict[str, list] = {}
        self._cka: dict[str, dict] = {}
        # ── J9: every seed used anywhere, logged once and written to metadata
        self.seeds = {
            'global_SEED (python/numpy/torch/PYTHONHASHSEED)': SEED,
            'PRIMARY_PROBE_SEED (fold assignment)': config.PRIMARY_PROBE_SEED,
            'RT_ALIGNMENT_SEED': config.RT_ALIGNMENT_SEED,
            'CONTEXT_SEED': config.CONTEXT_SEED,
            'RANDOM_INIT_SEED': config.RANDOM_INIT_SEED,
            'multi_seed_stability_seeds': [SEED + i * 7 for i in range(config.N_SEEDS)],
            'model_level_sampling_seeds': {'A3': SEED + 11, 'D1': SEED + 21, 'D2': SEED + 31,
                                           'D3': SEED + 41, 'C2_C5': SEED + 51, 'C3': SEED + 61},
            'bootstrap_seed': SEED,
            'torch_cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        }
        logger.info(f"[J9] seeds: {json.dumps(self.seeds)}")

    # ── entry point ───────────────────────────────────────────────────────
    def run(self):
        t0 = datetime.now()
        print(f"\n{'='*64}\nMULTI-MODEL LDT v9 — LAYER-WISE LEXICALITY PROBING AT THE "
              f"FIRST-OUTPUT PREDICTION POSITION\n"
              f"  PRIMARY  : prompt-only forward → h_l at the final prompt position →"
              f"\n             {self.config.PRIMARY_PROBE_N_FOLDS}-fold cross-fitted "
              f"linear WORD/NONWORD probe → logit(P(WORD)) → ELP RT alignment\n"
              f"  SECONDARY: hold-out probe (analyses 1-10), representation controls"
              f"\n  native LM-head YES/NO diagnostic: "
              f"{'ON' if self.config.ENABLE_NATIVE_LM_HEAD_DIAGNOSTIC else 'OFF'}"
              f"  |  generation diagnostic: "
              f"{'ON' if self.config.RUN_NATIVE_GENERATION_DIAGNOSTIC else 'OFF'}\n"
              f"Started: {t0:%Y-%m-%d %H:%M:%S}\n{'='*64}\n")
        print("PROMPT TEMPLATE (fixed, identical for every stimulus):\n"
              + TASK_PROMPT_TEMPLATE.replace("{STIMULUS}", "<STIMULUS>") + "\n")

        words_df    = pd.read_csv(self.config.WORDS_PATH)
        nonwords_df = pd.read_csv(self.config.NONWORDS_PATH)
        logger.info(f"Data: {len(words_df):,} words  {len(nonwords_df):,} nonwords")

        all_results = {}
        for mc in self.config.MODELS:
            print(f"\n{'='*64}\nMODEL: {mc.name}\n{'='*64}")
            try:
                res = self._process_model(mc, words_df, nonwords_df)
                all_results[mc.name] = res
                self._save_model_csv(mc.name, res['layer_results'])
                self._save_extended_csvs(mc.name, res)
                self._plot_model_all(mc.name, res['layer_results'])
                self._plot_extended(mc.name, res)
                self._plot_primary_model(mc.name, res)
                self._collect_spec_rows(mc.name, res)
                self._write_spec_outputs()
                try:
                    self._collect_naacl_rows(mc.name, res)
                    self._write_naacl_outputs()
                    self._plot_naacl_model(mc.name, res)
                except Exception as e:                     # noqa: BLE001
                    logger.error(f"NAACL outputs failed for {mc.name}: {e}", exc_info=True)
                self._write_experiment_metadata()
                if self.config.CLEAR_CACHE_BETWEEN_MODELS:
                    torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                logger.error(f"Failed {mc.name}: {e}")
                import traceback; traceback.print_exc()

        if len(all_results) > 1:
            self._plot_cross_model(all_results)
            self._save_cross_model_csv(all_results)
            self._save_cross_model_transfer_csv()
            self._save_cross_model_contextual_csv(all_results)
        if all_results:
            self._write_spec_outputs()
            self._plot_spec_figures(all_results)
            try:
                self._cross_model_naacl(all_results)
                self._write_naacl_outputs()
            except Exception as e:                         # noqa: BLE001
                logger.error(f"NAACL cross-model analyses failed: {e}", exc_info=True)
            self._write_experiment_metadata()

        print(f"\n{'='*64}\nANALYSIS 11: HUMAN-LLM REACTION-TIME ALIGNMENT\n{'='*64}")
        try:
            self.rt_alignment.finalize_and_save()
        except Exception as e:
            logger.error(f"Analysis 11 finalize_and_save failed: {e}")
            import traceback; traceback.print_exc()
        try:
            self._write_best_layers_summary()
        except Exception as e:
            logger.error(f"Best-layer summary failed: {e}")

        t1 = datetime.now()
        print(f"\n{'='*64}\nDONE  ({t1-t0})  →  {self.config.OUTPUT_DIR}\n{'='*64}\n")
        return all_results

    # ── per-model pipeline ────────────────────────────────────────────────
    @staticmethod
    def _frequency_prompt_examples(data: pd.DataFrame) -> dict:
        ex = {}
        for grp in ('high', 'mid', 'low'):
            sub = data[(data['is_word'] == 1) & (data['freq_group'] == grp)]
            if len(sub):
                ex[grp] = str(sub['stimulus'].iloc[0])
        sub = data[data['is_word'] == 0]
        if len(sub):
            ex['nonword'] = str(sub['stimulus'].iloc[0])
        return ex

    def _process_model(self, mc, words_df, nonwords_df) -> dict:
        cfg = self.config
        extractor = AllLayerExtractor(mc, cfg.DEVICE, cfg)
        dataset   = LDTDataset(words_df, nonwords_df, extractor.tokenizer, cfg)
        dataloader = DataLoader(dataset, batch_size=mc.batch_size,
                                shuffle=False, num_workers=0)
        model_meta = extractor.describe()

        # ════════════════════════════════════════════════════════════════
        # STEP 0: mandatory prompt / position / readout / leakage checks
        # ════════════════════════════════════════════════════════════════
        sanity = extractor.run_prompt_sanity_checks(
            cfg.SANITY_WORDS, cfg.SANITY_NONWORDS,
            self._frequency_prompt_examples(dataset.data),
            cfg.PROMPT_UNRELIABLE_UNKNOWN_RATE,
            cfg.PROMPT_UNRELIABLE_MIN_ACCURACY)

        # ════════════════════════════════════════════════════════════════
        # STEP 1: prompt-only forward — first-output-prediction-position state
        # ════════════════════════════════════════════════════════════════
        print("\n  [Step 1] Prompt-only forward pass — extracting the "
              "first-output-prediction-position state at every layer …")
        t_ext = datetime.now()
        reps, native_scores, targets, diag_df = extractor.extract_layerwise_representations(
            dataloader, cfg.REPRESENTATION_TYPES,
            control_dtype=cfg.CONTROL_REPRESENTATION_DTYPE,
            run_generation_diagnostic=cfg.RUN_NATIVE_GENERATION_DIAGNOSTIC)
        print(f"  Extraction done in {datetime.now()-t_ext}\n")
        all_hidden     = reps.pop(PRIMARY_REPRESENTATION)
        control_hidden = reps                       # {rep_type: {layer: arr}}
        diag_df.insert(0, 'model', mc.name)
        num_layers = extractor.num_layers
        hidden_dim = int(extractor.hidden_dim)
        model_meta = extractor.describe()
        native_summary, token_counts = self._summarise_native_generation(
            mc.name, diag_df)
        y_all = np.asarray(targets['is_word']).astype(int)

        native_ldt_table, consistency, native_rt = [], {}, []
        if native_scores is not None:
            native_ldt_table = self._native_lm_head_layer_table(
                mc.name, native_scores, targets, num_layers, hidden_dim,
                diag_df, native_summary)
            consistency = self._readout_generation_consistency(mc.name, diag_df)
            native_rt = self._native_lm_head_rt_alignment(mc.name, native_scores,
                                                          targets, num_layers)

        # ════════════════════════════════════════════════════════════════
        # STEP 2: PRIMARY — per-layer 5-fold CROSS-FITTED linear probe.
        # (Moved BEFORE the model is freed: the live-model analyses need the
        # best lexicality layer and fold-disjoint directions.)
        # ════════════════════════════════════════════════════════════════
        analyzer = FrequencyAnalyzer(cfg)
        fold_ids = analyzer.make_folds(y_all)
        t_cf = datetime.now()
        crossfit_table, oof_cache = self._crossfitted_probe_layer_table(
            mc.name, all_hidden, targets, analyzer, num_layers, hidden_dim)
        print(f"  Cross-fitted probes done in {datetime.now()-t_cf}\n")
        crossfit_emergence = [self._emergence_summary(
            mc.name, PRIMARY_REPRESENTATION, crossfit_table, num_layers,
            readout=PRIMARY_READOUT)]
        best_lexicality = self._best_lexicality_layer(crossfit_table)
        prim_correct = {li: ((p >= 0.5).astype(int) == y_all).astype(int)
                        for li, (p, _) in oof_cache.items()}

        # ════════════════════════════════════════════════════════════════
        # LIVE-MODEL ANALYSES (J6: the model is freed only after all of them)
        # ════════════════════════════════════════════════════════════════
        print("  [Step 3] Contextual controls (Analysis 10, F1–F3) …")
        try:
            contextual = self.ext.contextual_analysis(all_hidden, targets, extractor, mc.name)
        except Exception as e:
            logger.error(f"  Contextual analysis failed for {mc.name}: {e}", exc_info=True)
            contextual = {'isolated': [], 'contextual': [], 'conditions': [],
                          'n_high': 0, 'n_low': 0, 'n_located_in_sentence': 0}

        model_level = {}
        if cfg.RUN_MODEL_LEVEL_ANALYSES:
            mla = ModelLevelAnalyses(cfg, self)
            R = int(best_lexicality.get('best_lexicality_layer', num_layers // 2))
            jobs = [
                ('sublayer_decomposition', lambda: mla.sublayer_decomposition(extractor, targets, mc.name)),
                ('attention_mass', lambda: mla.attention_mass(extractor, targets, mc.name)),
                ('attention_knockout', lambda: mla.attention_knockout(extractor, targets, mc.name)),
                ('position_sensitivity', lambda: mla.position_sensitivity(extractor, targets, all_hidden, mc.name)),
                ('steering', lambda: mla.steering_and_subspace_ablation(extractor, targets, all_hidden, mc.name)),
                ('head_ablation', lambda: mla.head_ablation(extractor, targets, all_hidden, mc.name, R)),
            ]
            for name, fn in jobs:
                print(f"  [Step 3] model-level analysis: {name} …")
                t0 = datetime.now()
                try:
                    model_level[name] = fn()
                except Exception as e:                     # noqa: BLE001
                    logger.error(f"  [{name}] failed for {mc.name}: {e}", exc_info=True)
                    model_level[name] = {'error': str(e)}
                print(f"    {name} done in {datetime.now() - t0}")
                if cfg.DEVICE == 'cuda':
                    torch.cuda.empty_cache()

        # ── A6: surface-form baselines (need only the tokenizer) ──────────
        surface_rows = []
        if cfg.RUN_SURFACE_BASELINES:
            try:
                surface_rows = SurfaceFormBaselines(cfg).run(
                    targets, extractor.tokenizer, oof_cache, mc.name, num_layers,
                    fold_ids, self._bootstrap_ci)
            except Exception as e:                         # noqa: BLE001
                logger.error(f"  [A6] surface baselines failed: {e}", exc_info=True)

        # ── H2b: centred Gram matrices on a fixed stimulus subset (CKA) ───
        if cfg.RUN_CKA:
            try:
                self._store_cka_grams(mc.name, all_hidden, targets, num_layers)
            except Exception as e:                         # noqa: BLE001
                logger.warning(f"  [H2b] CKA Gram capture failed: {e}")

        # ── J6: free GPU memory only now ──────────────────────────────────
        model_meta = extractor.describe()
        del extractor.model
        extractor._backbone = None
        if cfg.DEVICE == 'cuda':
            torch.cuda.empty_cache(); torch.cuda.synchronize(); gc.collect()
            logger.info(f"  GPU after model unload: "
                        f"alloc={torch.cuda.memory_allocated(0)/1024**3:.2f}GB")

        # ════════════════════════════════════════════════════════════════
        # SECONDARY: single stratified train/val/test probe (analyses 1-10)
        # ════════════════════════════════════════════════════════════════
        print(f"  [Step 4] SECONDARY {analyzer.probe_type} hold-out probes on "
              f"{PRIMARY_REPRESENTATION} ({num_layers} layers) …")
        t_probe = datetime.now()
        layer_results = []
        for li in tqdm(range(num_layers), desc=f"LDT Probe | {mc.name}"):
            layer_results.append(analyzer.analyze_layer(all_hidden[li], targets, li, mc.name))
        print(f"  LDT Probe training done in {datetime.now()-t_probe}\n")
        self._apply_fdr(layer_results)
        probe_table = self._build_layer_table(
            mc.name, layer_results, num_layers, PRIMARY_REPRESENTATION,
            analyzer.probe_type, targets, native_summary, token_identity=False)
        emergence = list(crossfit_emergence) + [
            self._emergence_summary(mc.name, PRIMARY_REPRESENTATION, probe_table,
                                    num_layers, readout=HOLDOUT_PROBE_READOUT)]
        if native_ldt_table:
            emergence.append(self._emergence_summary(
                mc.name, PRIMARY_REPRESENTATION, native_ldt_table, num_layers,
                readout=NATIVE_LM_HEAD_READOUT))

        # ── Analysis 11: Human-LLM RT alignment (reuses primary OOF) ──────
        print("  [Step 5] Analysis 11: word-level lexical evidence for RT alignment …")
        try:
            reuse = (oof_cache
                     if (cfg.PRIMARY_PROBE_N_FOLDS == cfg.RT_ALIGNMENT_N_FOLDS
                         and cfg.PRIMARY_PROBE_SEED == cfg.RT_ALIGNMENT_SEED
                         and cfg.PRIMARY_PROBE_CV_SHUFFLE == cfg.RT_ALIGNMENT_CV_SHUFFLE)
                     else None)
            if reuse is None:
                logger.info("[Analysis 11] primary and RT fold settings differ — "
                            "refitting cross-fitted probes for the RT analysis.")
            rt_word_df = self.rt_alignment.extract_word_level_scores(
                all_hidden, targets, analyzer, mc.name, num_layers, precomputed_oof=reuse)
            self.rt_alignment.accumulate(rt_word_df, mc.name)
        except Exception as e:
            logger.error(f"  Analysis 11 failed for {mc.name}: {e}", exc_info=True)

        # ── Analysis 5 + A5: representation controls (hold-out AND paired
        #    cross-fitted comparison on the SAME folds as the primary) ─────
        ablation: dict[str, list] = {}
        control_tables: dict[str, list] = {}
        crossfit_controls: list[dict] = []
        for rep_type in list(control_hidden.keys()):
            print(f"  [Step 6] Control probes on {rep_type} …")
            hid = control_hidden[rep_type]
            try:
                ctab, coof = self._crossfit_table(
                    mc.name, {li: hid[li].astype(np.float32, copy=False) for li in range(num_layers)},
                    targets, analyzer, num_layers, hidden_dim, fold_ids=fold_ids,
                    representation_type=rep_type, desc=f'A5 cross-fit {rep_type}')
                crossfit_controls.extend(self._paired_vs_primary(
                    ctab, coof, prim_correct, y_all, 'primary_minus_control'))
            except Exception as e:                         # noqa: BLE001
                logger.error(f"  [A5] cross-fitted control {rep_type} failed: {e}")
            res_list = []
            for li in tqdm(range(num_layers), desc=f"{rep_type} | {mc.name}"):
                res_list.append(analyzer.analyze_layer(
                    hid[li].astype(np.float32, copy=False), targets, li, mc.name))
                hid[li] = None
            self._apply_fdr(res_list)
            ablation[rep_type] = res_list
            ctrl_table = self._build_layer_table(
                mc.name, res_list, num_layers, rep_type, analyzer.probe_type,
                targets, native_summary,
                token_identity=(rep_type == RepresentationType.GENERATED_TOKEN_CONTROL.value))
            control_tables[rep_type] = ctrl_table
            emergence.append(self._emergence_summary(
                mc.name, rep_type, ctrl_table, num_layers, readout=SECONDARY_READOUT))
            del control_hidden[rep_type], hid
            gc.collect()
        ablation_rows = self._representation_ablation_rows(
            mc.name, layer_results, ablation, num_layers, analyzer.probe_type)

        # ── A1: non-linear (MLP) probe, same folds, paired vs linear ──────
        mlp_rows = []
        if cfg.RUN_MLP_PROBE:
            print("  [Step 7] A1: MLP probe control …")
            try:
                mtab, moof = self._crossfit_table(
                    mc.name, all_hidden, targets, analyzer, num_layers, hidden_dim,
                    layers=select_layers(num_layers, cfg.MLP_PROBE_LAYERS), probe='mlp',
                    fold_ids=fold_ids, readout='crossfitted_mlp_probe', desc='A1 MLP probe')
                mlp_rows = self._paired_vs_primary(mtab, moof, prim_correct, y_all,
                                                   'mlp_minus_linear')
            except Exception as e:                         # noqa: BLE001
                logger.error(f"  [A1] MLP probe failed: {e}", exc_info=True)

        # ── B4: covariate-erased probe (length, Ortho_N, token count) ─────
        erased_rows, freq_cov = [], {}
        if cfg.RUN_FREQUENCY_COVARIATE_ANALYSES:
            print("  [Step 7] B4–B6: covariate-erased probe and frequency models …")
            try:
                Z = np.column_stack([np.asarray(targets[c], float)
                                     for c in ('length', 'ortho_n', 'token_count')])
                etab, eoof = self._crossfit_table(
                    mc.name, all_hidden, targets, analyzer, num_layers, hidden_dim,
                    layers=select_layers(num_layers, cfg.COVARIATE_ERASURE_LAYERS),
                    covariates=Z, fold_ids=fold_ids,
                    representation_type=f'{PRIMARY_REPRESENTATION}__covariates_linearly_erased',
                    desc='B4 covariate-erased probe')
                erased_rows = self._paired_vs_primary(etab, eoof, prim_correct, y_all,
                                                      'erased_minus_primary')
            except Exception as e:                         # noqa: BLE001
                logger.error(f"  [B4] covariate-erased probe failed: {e}", exc_info=True)
            try:
                freq_cov = self.ext.frequency_covariate_models(oof_cache, targets, mc.name,
                                                               num_layers)
            except Exception as e:                         # noqa: BLE001
                logger.error(f"  [B5/B6] frequency covariate models failed: {e}", exc_info=True)

        # ── D4: tokenization-stratified primary results ───────────────────
        try:
            tok_strat = self.ext.tokenization_stratified_primary(oof_cache, targets, mc.name,
                                                                 num_layers)
        except Exception as e:                             # noqa: BLE001
            logger.error(f"  [D4] failed: {e}")
            tok_strat = []

        # ════════════════════════════════════════════════════════════════
        # Extended analyses 1-9 (all on the PRIMARY representation)
        # ════════════════════════════════════════════════════════════════
        print("\n  [Step 8] Running extended analyses …")
        ext = self.ext
        direct_freq = ext.direct_frequency_probe(all_hidden, targets, mc.name)
        token_ctrl = ext.tokenization_controlled_analysis(all_hidden, targets, mc.name)
        confound = ext.confound_matched_analysis(all_hidden, targets, mc.name)
        stability = ext.multi_seed_stability(all_hidden, targets, mc.name,
                                             n_seeds=cfg.N_SEEDS)
        selectivity = ext.probe_selectivity_controls(all_hidden, targets, mc.name)
        regression = ext.continuous_frequency_regression(all_hidden, targets, mc.name)
        try:
            intervention = ext.intervention_analysis(all_hidden, targets, mc.name)
        except Exception as e:
            logger.error(f"  Intervention analysis failed for {mc.name}: {e}")
            intervention = []
        dose_response = [dict(d, model=mc.name) for r in intervention
                         for d in r.pop('dose_response', [])]

        try:
            this_snapshot = ext.build_transfer_snapshot(all_hidden, targets, mc.name, num_layers)
        except Exception as e:
            logger.error(f"  Transfer snapshot failed for {mc.name}: {e}")
            this_snapshot = {'model_name': mc.name, 'num_layers': num_layers,
                             'hidden_dim': None, 'layers': {}}
        for prior_name, prior_snapshot in self._transfer_snapshots.items():
            try:
                self.cross_model_transfer_rows.extend(
                    ext.cross_model_transfer_analysis(prior_snapshot, this_snapshot))
                self.cross_model_transfer_rows.extend(
                    ext.cross_model_transfer_analysis(this_snapshot, prior_snapshot))
            except Exception as e:
                logger.error(f"  Cross-model transfer {prior_name}<->{mc.name} failed: {e}")
        self._transfer_snapshots[mc.name] = this_snapshot

        # ── Free remaining hidden states ──────────────────────────────────
        for li in list(all_hidden.keys()):
            del all_hidden[li]
        gc.collect()

        # ════════════════════════════════════════════════════════════════
        # A2: RANDOM-INITIALISATION CONTROL (same tokenizer, same items,
        # same probe; paired McNemar vs the trained model per layer)
        # ════════════════════════════════════════════════════════════════
        random_init_rows, random_init_emergence = [], None
        if cfg.RUN_RANDOM_INIT_BASELINE:
            print("  [Step 9] A2: random-initialisation control …")
            try:
                random_init_rows, random_init_emergence = self._random_init_control(
                    mc, dataset, targets, prim_correct, num_layers, hidden_dim, analyzer)
            except Exception as e:                         # noqa: BLE001
                logger.error(f"  [A2] random-init control failed: {e}", exc_info=True)
            if cfg.DEVICE == 'cuda':
                torch.cuda.empty_cache()
        del oof_cache
        gc.collect()

        del extractor.tokenizer, extractor
        gc.collect()

        model_meta.update({'native_generation': native_summary,
                           'prompt_sanity_checks': sanity})
        self.model_metadata[mc.name] = model_meta

        return {
            'model_config':   mc,
            'layer_results':  layer_results,
            'num_layers':     num_layers,
            'hidden_dim':     hidden_dim,
            'n_parameters':   model_meta.get('n_parameters'),
            'representation_type': PRIMARY_REPRESENTATION,
            'readout_type':   PRIMARY_READOUT,
            'probe_type':     analyzer.probe_type,
            # PRIMARY
            'crossfit_table': crossfit_table,
            'best_lexicality_layer': best_lexicality,
            # OPTIONAL native LM-head diagnostic
            'native_ldt_table': native_ldt_table,
            'native_ldt_scores': native_scores,
            'readout_generation_consistency': consistency,
            'native_rt_alignment': native_rt,
            # SECONDARY
            'probe_table':    probe_table,
            'primary_table':  probe_table,   # backward-compatible alias
            'control_tables': control_tables,
            'emergence_summary': emergence + ([random_init_emergence] if random_init_emergence else []),
            'native_summary': native_summary,
            'prompt_diagnostics': diag_df,
            'generated_token_counts': token_counts,
            'sanity': sanity,
            # Extended analyses
            'representation_ablation': ablation,
            'representation_ablation_rows': ablation_rows,
            'direct_freq':       direct_freq,
            'token_ctrl':        token_ctrl,
            'confound_matched':  confound,
            'stability':         stability,
            'selectivity':       selectivity,
            'regression':        regression,
            'intervention':      intervention,
            'contextual':        contextual,
            'transfer_snapshot_summary': {
                li: {'fraction': s['fraction'], 'native_accuracy': s['native_accuracy']}
                for li, s in this_snapshot['layers'].items()
            },
            # NAACL revision (A–H)
            'mlp_probe': mlp_rows,
            'random_init': random_init_rows,
            'crossfit_controls': crossfit_controls,
            'covariate_erased': erased_rows,
            'frequency_covariates': freq_cov,
            'tokenization_stratified': tok_strat,
            'surface_baselines': surface_rows,
            'intervention_dose_response': dose_response,
            'model_level': model_level,
        }

    # ── NAACL helpers used by _process_model ─────────────────────────────
    def _paired_vs_primary(self, rows, oof, prim_correct, y, label) -> list[dict]:
        """Adds, per layer, the accuracy difference to the PRIMARY probe and
        an exact McNemar test on the SAME items (paired because both use the
        same folds), Holm-corrected across layers (A1, A2, A5, B4)."""
        y = np.asarray(y).astype(int)
        for r in rows:
            li = r['layer']
            if li in oof and li in prim_correct and len(prim_correct[li]) == len(y):
                c = ((oof[li][0] >= 0.5).astype(int) == y).astype(int)
                _b, _c, p = mcnemar_exact(c, prim_correct[li])
                r[f'{label}_accuracy'] = float(c.mean() - prim_correct[li].mean())
                r['mcnemar_vs_primary_p'] = p
                r['mcnemar_vs_primary_only_this_correct'] = _b
                r['mcnemar_vs_primary_only_primary_correct'] = _c
        adj = adjust_pvalues([r.get('mcnemar_vs_primary_p', np.nan) for r in rows], 'holm',
                             self.config.ALPHA)
        for r, v in zip(rows, adj):
            if np.isfinite(v):
                r['mcnemar_vs_primary_p_holm'] = float(v)
                r['difference_significant_holm'] = bool(v < self.config.ALPHA)
        return rows

    def _random_init_control(self, mc, dataset, targets, prim_correct, num_layers,
                             hidden_dim, analyzer):
        cfg = self.config
        idx = np.arange(len(dataset))
        if cfg.RANDOM_INIT_MAX_ITEMS is not None and cfg.RANDOM_INIT_MAX_ITEMS < len(idx):
            y = np.asarray(targets['is_word']).astype(int)
            idx, _ = train_test_split(idx, train_size=cfg.RANDOM_INIT_MAX_ITEMS,
                                      stratify=y, random_state=cfg.RANDOM_INIT_SEED)
            idx = np.sort(idx)
        loader = DataLoader(torch.utils.data.Subset(dataset, idx.tolist()),
                            batch_size=mc.batch_size, shuffle=False, num_workers=0)
        rex = AllLayerExtractor(mc, cfg.DEVICE, cfg, random_init=True)
        rex.native_readout_enabled = False
        try:
            reps_r, _n, targets_r, _d = rex.extract_layerwise_representations(
                loader, [PRIMARY_REPRESENTATION], control_dtype='float32',
                run_generation_diagnostic=False)
        finally:
            del rex.model
            rex._backbone = None
            if cfg.DEVICE == 'cuda':
                torch.cuda.empty_cache()
        hid_r = reps_r[PRIMARY_REPRESENTATION]
        if list(np.asarray(targets_r['stimulus'])) != [targets['stimulus'][i] for i in idx]:
            raise RuntimeError("random-init extraction changed the item order")
        y_sub = np.asarray(targets_r['is_word']).astype(int)
        tab, oof = self._crossfit_table(
            mc.name, hid_r, targets_r, analyzer, num_layers, hidden_dim,
            representation_type=f'{PRIMARY_REPRESENTATION}__random_init_model',
            readout=PRIMARY_READOUT, desc='A2 random-init probe',
            extra_tags={'model_weights': 'random_init',
                        'random_init_seed': cfg.RANDOM_INIT_SEED,
                        'trained_probe_fit_on_full_sample': bool(len(idx) < len(dataset))})
        prim_sub = {li: c[idx] for li, c in prim_correct.items()}
        rows = self._paired_vs_primary(tab, oof, prim_sub, y_sub, 'random_init_minus_trained')
        emer = self._emergence_summary(mc.name, f'{PRIMARY_REPRESENTATION}__random_init_model',
                                       tab, num_layers, readout=PRIMARY_READOUT)
        del hid_r, reps_r
        gc.collect()
        return rows, emer

    def _store_cka_grams(self, model_name, all_hidden, targets, num_layers):
        """Centred Gram matrices on a FIXED stimulus subset chosen by a hash of
        the stimulus string (identical across models when the items are)."""
        import hashlib
        stim = [str(s) for s in targets['stimulus']]
        order = sorted(range(len(stim)),
                       key=lambda i: hashlib.md5(stim[i].encode()).hexdigest())
        rows = np.sort(np.asarray(order[:self.config.CKA_N_ITEMS]))
        self._cka[model_name] = {
            'stimuli': [stim[i] for i in rows], 'num_layers': num_layers,
            'grams': {li: centered_gram(all_hidden[li][rows]).astype(np.float32)
                      for li in range(num_layers)}}

    # ════════════════════════════════════════════════════════════════════════
    # PRIMARY RESULT — layer-wise CROSS-FITTED LINEAR LEXICALITY PROBE (§2-§4)
    # ════════════════════════════════════════════════════════════════════════
    def _oof_layer_row(self, p, y, freq, row) -> np.ndarray:
        """Fills `row` with every held-out metric of one layer's out-of-fold
        scores and returns the 0/1 correctness vector (for paired tests)."""
        eps = float(self.config.LEXICAL_EVIDENCE_CLIP)
        n = len(y)
        chance = float(max(y.mean(), 1 - y.mean()))
        pred = (p >= 0.5).astype(int)
        correct = (pred == y).astype(float)
        ev = self.rt_alignment._logit(p, eps)
        lo, hi = self._bootstrap_ci(correct)
        two = len(np.unique(y)) == 2
        try:
            auroc = roc_auc_score(y, p) if two else np.nan
            auprc = average_precision_score(y, p) if two else np.nan
        except Exception:
            auroc = auprc = np.nan
        pb, pb_log = p_binom_greater(int(correct.sum()), n, chance)
        row.update({
            'n': int(n), 'chance_accuracy': chance,
            'accuracy': float(correct.mean()),
            'accuracy_ci95_low': lo, 'accuracy_ci95_high': hi,
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if two else np.nan,
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'auroc': auroc, 'auprc': auprc,
            'mean_p_word': float(p.mean()),
            'mean_p_word_on_words': float(p[y == 1].mean()) if (y == 1).any() else np.nan,
            'mean_p_word_on_nonwords': float(p[y == 0].mean()) if (y == 0).any() else np.nan,
            'mean_lexical_evidence': float(ev.mean()),
            'mean_lexical_evidence_words': float(ev[y == 1].mean()) if (y == 1).any() else np.nan,
            'mean_lexical_evidence_nonwords': float(ev[y == 0].mean()) if (y == 0).any() else np.nan,
            'lexical_evidence_definition': f'logit(P(WORD)) with P clipped to [{eps}, {1 - eps}]',
            'word_accuracy': float(correct[y == 1].mean()) if (y == 1).any() else np.nan,
            'nonword_accuracy': float(correct[y == 0].mean()) if (y == 0).any() else np.nan,
            'p_vs_chance_binomial': pb, 'p_vs_chance_binomial_log10': pb_log,
        })
        for g in ('high', 'mid', 'low'):
            m = (y == 1) & (freq == g)
            row[f'word_accuracy_{g}_freq'] = float(correct[m].mean()) if m.any() else np.nan
            row[f'mean_lexical_evidence_{g}_freq'] = float(ev[m].mean()) if m.any() else np.nan
            row[f'n_word_{g}_freq'] = int(m.sum())
        hf, lf = (y == 1) & (freq == 'high'), (y == 1) & (freq == 'low')
        if hf.sum() >= self.config.MIN_TEST_SAMPLES and lf.sum() >= self.config.MIN_TEST_SAMPLES:
            fc = frequency_contrast(correct[hf], correct[lf], B=self.config.N_BOOTSTRAP,
                                    bf_threshold=self.config.BF_EVIDENCE_THRESHOLD)
            row['freq_accuracy_difference'] = fc['accuracy_difference']
            for k in ('accuracy_difference_ci95_low', 'accuracy_difference_ci95_high',
                      'cohens_h', 'cohens_h_ci95_low', 'cohens_h_ci95_high',
                      'z_statistic', 'p_value', 'p_value_log10', 'log10_bf01',
                      'bf01_interpretation'):
                row[f'freq_{k}'] = fc[k]
            d = delong_auc_diff_shared_negatives(p[hf], p[lf], p[y == 0])
            row.update({'freq_auc_hf_vs_nonwords': d['auc_a'],
                        'freq_auc_lf_vs_nonwords': d['auc_b'],
                        'freq_auc_difference': d['auc_difference'],
                        'freq_auc_difference_se': d['auc_difference_se'],
                        'freq_p_value_auc': d['p_value_auc'],
                        'freq_p_value_auc_log10': d['p_value_auc_log10']})
        else:
            row['freq_accuracy_difference'] = np.nan
        return correct

    def _finalize_oof_rows(self, rows: list[dict]):
        """Within-table multiplicity control: BH-FDR for layer-vs-chance and
        for the frequency tests (G3), Holm for consecutive-layer McNemar."""
        a = self.config.ALPHA
        for col, method, out in (
                ('p_vs_chance_binomial', 'fdr_bh', 'p_vs_chance_fdr_bh'),
                ('mcnemar_p_vs_previous_layer', 'holm', 'mcnemar_p_vs_previous_layer_holm'),
                ('freq_p_value', 'fdr_bh', 'freq_p_value_corrected'),
                ('freq_p_value_auc', 'fdr_bh', 'p_value_auc_corrected')):
            p = [r.get(col, np.nan) for r in rows]
            adj = adjust_pvalues(p, method, a)
            for r, v in zip(rows, adj):
                if np.isfinite(v):
                    r[out] = float(v)
                    r[out + '_significant'] = bool(v < a)
        # Cohen's h is tested by the same two-proportion test (G3 note).
        for r in rows:
            if 'freq_p_value_corrected' in r:
                r['p_value_cohens_h_corrected'] = r['freq_p_value_corrected']

    def _crossfit_table(self, model_name, hidden_by_layer, targets, analyzer, num_layers,
                        hidden_dim, layers=None, probe='linear',
                        representation_type=PRIMARY_REPRESENTATION,
                        readout=PRIMARY_READOUT, covariates=None, fold_ids=None,
                        rows_subset=None, desc='Cross-fitted probe', extra_tags=None):
        """Generic cross-fitted layer table (used by the PRIMARY curve and by
        every paired control: MLP probe, random-init model, mean-pool,
        covariate-erased probe, sublayers, stimulus-position templates).
        All tables built with the same `fold_ids` are paired item-by-item."""
        y_all = np.asarray(targets['is_word']).astype(int)
        freq_all = np.asarray(targets['freq_group'], dtype=object)
        sub = np.arange(len(y_all)) if rows_subset is None else np.asarray(rows_subset)
        y, freq = y_all[sub], freq_all[sub]
        if fold_ids is None:
            fold_ids = analyzer.make_folds(y)
        layers = list(range(num_layers)) if layers is None else list(layers)
        probe_type = (analyzer.probe_type if probe == 'linear'
                      else probe_type_name(self.config.MLP_PROBE_HIDDEN_DIMS)
                      + ('_residual' if self.config.MLP_PROBE_USE_RESIDUAL else ''))
        if probe == 'linear':
            be = analyzer._linear_backend()
            probe_type = (f"l2_logistic_regression_C{self.config.PRIMARY_PROBE_C:g}_"
                          + (f"sklearn_{self.config.PRIMARY_PROBE_SOLVER}" if be == 'sklearn'
                             else "torch_fullbatch_lbfgs"))
        rows, oof_cache, prev = [], {}, None
        for li in tqdm(layers, desc=f"{desc} | {model_name}"):
            row = {'model': model_name, 'layer': li,
                   'relative_depth': (li + 1) / num_layers,
                   'hidden_dim': int(hidden_dim),
                   'representation_type': representation_type,
                   'readout_type': readout, 'probe_type': probe_type,
                   'n_folds': int(len(np.unique(fold_ids))),
                   'probe_seed': int(self.config.PRIMARY_PROBE_SEED),
                   'scoring': 'out_of_fold_only',
                   'covariates_erased': covariates is not None}
            if extra_tags:
                row.update(extra_tags)
            try:
                X = hidden_by_layer[li]
                X = X if rows_subset is None or len(X) == len(sub) else X[sub]
                if len(X) != len(y):
                    raise ValueError(f"representation has {len(X)} rows, labels {len(y)}")
                Z = None if covariates is None else np.asarray(covariates)[sub]
                p, fold = analyzer.crossfit_word_probabilities(
                    X, y, probe=probe, covariates=Z, fold_of=fold_ids)
            except Exception as e:
                logger.error(f"  [{desc}] {model_name} layer {li}: cross-fit failed "
                             f"({e}) — layer reported as missing.", exc_info=True)
                rows.append(row); prev = None
                continue
            oof_cache[li] = (p, fold)
            correct = self._oof_layer_row(p, y, freq, row)
            if prev is not None and prev[2] == li - 1:
                _b, _c, pv = self._mcnemar_exact(prev[0], correct)
                row.update({'delta_accuracy_vs_previous_layer': row['accuracy'] - prev[1],
                            'mcnemar_p_vs_previous_layer': pv})
            prev = (correct, row['accuracy'], li)
            rows.append(row)
        self._finalize_oof_rows(rows)
        return rows, oof_cache

    def _crossfitted_probe_layer_table(self, model_name, all_hidden, targets,
                                       analyzer, num_layers, hidden_dim):
        """
        PRIMARY RESULT. For every transformer layer, fit K independent L2
        logistic-regression WORD/NONWORD probes on the hidden state at the
        FIRST-OUTPUT PREDICTION POSITION and score each stimulus exactly once,
        with the fold probe that did not train on it. Nothing in the language
        model is touched. The probe is an ANALYSIS TOOL: the numbers say what
        is linearly recoverable from the representation — not what the LLM
        predicts (K1).

        Returns (rows, oof_cache) where oof_cache[layer] = (p_all, fold_all)
        so Analysis 11 and every paired control reuse exactly these scores.
        """
        y = np.asarray(targets['is_word']).astype(int)
        fold_ids = analyzer.make_folds(y)
        print(f"  [PRIMARY] {self.config.PRIMARY_PROBE_N_FOLDS}-fold cross-fitted "
              f"linear probes on {PRIMARY_REPRESENTATION} "
              f"({num_layers} layers, N={len(y)}) …")
        rows, oof_cache = self._crossfit_table(
            model_name, all_hidden, targets, analyzer, num_layers, hidden_dim,
            probe='linear', fold_ids=fold_ids, desc='Cross-fitted LDT probe')
        fin = [r for r in rows if np.isfinite(r.get('balanced_accuracy', np.nan))]
        if fin:
            best = max(fin, key=lambda r: r['balanced_accuracy'])
            print(f"  [PRIMARY] best lexicality layer = {best['layer']} "
                  f"(relative depth {best['relative_depth']:.2f}): "
                  f"balanced acc={best['balanced_accuracy']:.3f}, "
                  f"AUROC={best['auroc']:.3f}, acc={best['accuracy']:.3f} "
                  f"(chance {best['chance_accuracy']:.3f})")
        return rows, oof_cache

    @staticmethod
    def _best_lexicality_layer(table) -> dict:
        """Best layer by held-out balanced accuracy; ties → shallowest layer."""
        fin = [r for r in table if np.isfinite(r.get('balanced_accuracy', np.nan))]
        if not fin:
            return {}
        top = max(r['balanced_accuracy'] for r in fin)
        b = min((r for r in fin if r['balanced_accuracy'] == top), key=lambda r: r['layer'])
        return {'best_lexicality_layer': int(b['layer']),
                'best_lexicality_relative_depth': float(b['relative_depth']),
                'best_lexicality_balanced_accuracy': float(b['balanced_accuracy']),
                'best_lexicality_accuracy': float(b['accuracy']),
                'best_lexicality_auroc': float(b.get('auroc', np.nan)),
                'best_lexicality_selection_criterion':
                    'max out-of-fold balanced accuracy (ties → shallowest layer)'}

    def _write_best_layers_summary(self):
        """§14 — best lexicality layer and best RT-alignment layer per model."""
        rows = []
        by_model: dict[str, list] = {}
        for r in self.spec_rows.get('layerwise_crossfitted_probe_ldt', []):
            by_model.setdefault(r['model'], []).append(r)
        rt_path = os.path.join(self.config.RT_ALIGNMENT_DIR, 'results',
                               'human_rt_alignment_layer_results.csv')
        rt = None
        if os.path.exists(rt_path):
            try:
                rt = pd.read_csv(rt_path)
            except Exception as e:
                logger.warning(f"[best layers] could not read {rt_path}: {e}")
        for model, tbl in by_model.items():
            row = {'model': model, 'representation_type': PRIMARY_REPRESENTATION,
                   'readout_type': PRIMARY_READOUT}
            row.update(self._best_lexicality_layer(tbl))
            if rt is not None and 'model' in rt.columns:
                g = rt[rt['model'] == model].copy()
                if len(g) and 'spearman_rho' in g:
                    g['_abs'] = g['spearman_rho'].abs()
                    sig = g[g.get('spearman_fdr_p', pd.Series(1.0, index=g.index)) < self.config.ALPHA]
                    pick = (sig if len(sig) else g).sort_values(
                        ['_abs', 'layer'], ascending=[False, True]).iloc[0]
                    row.update({
                        'best_rt_alignment_layer': int(pick['layer']),
                        'best_rt_alignment_spearman_rho': float(pick['spearman_rho']),
                        'best_rt_alignment_fdr_p': float(pick.get('spearman_fdr_p', np.nan)),
                        'best_rt_alignment_significant_after_fdr': bool(len(sig) > 0),
                        'best_rt_alignment_selection_criterion':
                            'max |Spearman rho| among FDR-significant layers, else '
                            'max |rho| overall (ties → shallowest layer)',
                        'human_rt_used_in_probe_fitting': False,
                    })
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(self.config.PRIMARY_DIR, 'best_layers_summary.csv'),
                      index=False)
            logger.info("✓ best_layers_summary.csv")
            print(f"\n{'='*64}\nBEST LAYERS (§14)\n{'='*64}")
            cols = [c for c in ['model', 'best_lexicality_layer',
                                'best_lexicality_balanced_accuracy',
                                'best_lexicality_auroc', 'best_rt_alignment_layer',
                                'best_rt_alignment_spearman_rho'] if c in df.columns]
            print(df[cols].to_string(index=False))

    # ════════════════════════════════════════════════════════════════════════
    # OPTIONAL DIAGNOSTIC — layer-wise native LM-head lexical decision
    # ════════════════════════════════════════════════════════════════════════
    def _native_lm_head_layer_table(self, model_name, native, targets, num_layers,
                                    hidden_dim, diag_df, native_summary) -> list[dict]:
        """
        One row per transformer layer. Every row is produced by applying the
        model's OWN pretrained head to that layer's final-prompt-position state
        and comparing logit(YES) with logit(NO). Nothing is fitted, so ALL
        items are evaluated (no train/test split is needed or used).

        Gold labels enter only here, after the predictions exist.
        """
        y = np.asarray(targets['is_word']).astype(int)
        freq = np.asarray(targets['freq_group'], dtype=object)
        yes, no = native['yes_score'], native['no_score']
        chance = float(max(y.mean(), 1 - y.mean()))
        n = len(y)
        rows, prev = [], None
        gen_pred = (diag_df['native_prediction'].to_numpy()
                    if 'native_prediction' in diag_df.columns else None)
        for li in range(num_layers):
            ys, ns = yes[:, li].astype(float), no[:, li].astype(float)
            margin = ys - ns                       # decision variable
            ok = np.isfinite(margin)
            pred = (margin > 0).astype(int)        # 1 = WORD (YES beats NO)
            correct = (pred == y).astype(float)
            row = {'model': model_name, 'layer': li,
                   'relative_depth': (li + 1) / num_layers,
                   'hidden_dim': int(hidden_dim), 'n_examples': int(n),
                   'representation_type': PRIMARY_REPRESENTATION,
                   'readout_type': NATIVE_LM_HEAD_READOUT,
                   'yes_prediction_rate': float(pred.mean()),
                   'mean_yes_logit': float(np.nanmean(ys)),
                   'mean_no_logit': float(np.nanmean(ns)),
                   'mean_logit_margin_yes_minus_no': float(np.nanmean(margin)),
                   'n_non_finite_scores': int((~ok).sum()),
                   'chance_accuracy': chance}
            if not ok.all():
                rows.append(row); prev = None; continue
            acc = float(correct.mean())
            lo, hi = self._bootstrap_ci(correct)
            try:
                auroc = roc_auc_score(y, margin) if len(np.unique(y)) == 2 else np.nan
                auprc = average_precision_score(y, margin) if len(np.unique(y)) == 2 else np.nan
            except Exception:
                auroc = auprc = np.nan
            row.update({
                'accuracy': acc, 'accuracy_ci95_low': lo, 'accuracy_ci95_high': hi,
                'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
                'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
                'auroc': auroc, 'auprc': auprc,
                'p_vs_chance_binomial': float(stats.binomtest(
                    int(correct.sum()), n, chance, alternative='greater').pvalue),
                'word_accuracy': float(correct[y == 1].mean()) if (y == 1).any() else np.nan,
                'nonword_accuracy': float(correct[y == 0].mean()) if (y == 0).any() else np.nan,
            })
            for g in ('high', 'mid', 'low'):
                m = (y == 1) & (freq == g)
                row[f'word_accuracy_{g}_freq'] = float(correct[m].mean()) if m.any() else np.nan
                row[f'n_word_{g}_freq'] = int(m.sum())
            # frequency effect of the native readout (high − low words)
            row['freq_accuracy_difference'] = (row['word_accuracy_high_freq']
                                               - row['word_accuracy_low_freq'])
            if gen_pred is not None:
                row['agreement_with_native_generation'] = float(np.mean(
                    np.where(pred == 1, 'WORD', 'NONWORD') == gen_pred))
            if prev is not None:
                b, c, p = self._mcnemar_exact(prev[0], correct)
                row.update({'delta_accuracy_vs_previous_layer': acc - prev[1],
                            'mcnemar_p_vs_previous_layer': p})
            prev = (correct, acc)
            rows.append(row)

        def _adj(col, method, out):
            idx = [i for i, rw in enumerate(rows) if np.isfinite(rw.get(col, np.nan))]
            if idx:
                _, pc, _, _ = multipletests([rows[i][col] for i in idx],
                                            alpha=self.config.ALPHA, method=method)
                for i, p in zip(idx, pc):
                    rows[i][out] = float(p)
                    rows[i][out + '_significant'] = bool(p < self.config.ALPHA)
        _adj('p_vs_chance_binomial', 'fdr_bh', 'p_vs_chance_fdr_bh')
        _adj('mcnemar_p_vs_previous_layer', 'holm', 'mcnemar_p_vs_previous_layer_holm')

        fin = [r for r in rows if np.isfinite(r.get('accuracy', np.nan))]
        if fin:
            best = max(fin, key=lambda r: r['accuracy'])
            print(f"  PRIMARY (native LM head): final-layer acc="
                  f"{fin[-1]['accuracy']:.3f} | best layer {best['layer']} "
                  f"acc={best['accuracy']:.3f} | chance={chance:.3f}")
        return rows

    def _readout_generation_consistency(self, model_name, diag_df) -> dict:
        """§22 — final-layer native readout vs full-model one-token generation.
        Disagreement is reported, never removed by forcing either side."""
        if 'native_prediction' not in diag_df.columns:
            return {'model': model_name, 'generation_diagnostic_run': False}
        ro = diag_df['final_layer_lm_head_prediction'].to_numpy()
        gn = diag_df['native_prediction'].to_numpy()
        known = gn != 'UNKNOWN'
        out = {
            'model': model_name, 'generation_diagnostic_run': True,
            'n_examples': int(len(diag_df)),
            'agreement_rate': float(np.mean(ro == gn)),
            'agreement_rate_excluding_unknown_generation':
                float(np.mean(ro[known] == gn[known])) if known.any() else np.nan,
            'n_disagreements': int(np.sum(ro != gn)),
            'final_layer_readout_accuracy':
                float(diag_df['final_layer_lm_head_correct'].mean()),
            'native_generation_accuracy': float(diag_df['native_correct'].mean()),
            'generation_unknown_rate': float(np.mean(~known)),
            'note': ('Disagreement is expected when the greedy token is neither '
                     'YES nor NO (UNKNOWN); the readout always chooses between '
                     'YES and NO and therefore never returns UNKNOWN.'),
        }
        print(f"  §22 consistency: final-layer readout vs generation agreement="
              f"{out['agreement_rate']:.3f} "
              f"(excluding UNKNOWN: {out['agreement_rate_excluding_unknown_generation']:.3f})")
        return out

    def _native_lm_head_rt_alignment(self, model_name, native, targets,
                                     num_layers) -> list[dict]:
        """
        §21 — human RT alignment for the NATIVE readout, kept separate from the
        probe-based Analysis 11. Lexical evidence here needs no fitting at all:
            S_l = logit(YES) − logit(NO)  at layer l
        Human RT is used only for evaluation, never to produce S_l. Layer depth
        is not a reaction time and is never treated as one.
        """
        lookup = getattr(self.rt_alignment, 'elp_lookup', None)
        if not lookup:
            return []
        is_word = np.asarray(targets['is_word']).astype(int)
        stim = np.asarray(targets['stimulus'], dtype=object)
        widx = np.where(is_word == 1)[0]
        rt, keep = [], []
        for i in widx:
            rec = lookup.get(str(stim[i]).strip().lower())
            v = None if rec is None else rec.get('human_rt')
            if v is not None and np.isfinite(v):
                keep.append(i); rt.append(float(v))
        if len(keep) < self.config.RT_ALIGNMENT_MIN_WORDS_PER_LAYER:
            logger.warning(f"[native RT] {model_name}: only {len(keep)} RT-matched "
                           f"words — skipped.")
            return []
        keep = np.asarray(keep); rt = np.asarray(rt)
        rows = []
        for li in range(num_layers):
            S = (native['yes_score'][keep, li] - native['no_score'][keep, li]).astype(float)
            if not np.isfinite(S).all() or np.allclose(S, S[0]):
                rows.append({'model': model_name, 'layer': li, 'n_words': len(keep),
                             'representation_type': PRIMARY_REPRESENTATION,
                             'readout_type': NATIVE_LM_HEAD_READOUT}); continue
            rho, p = stats.spearmanr(S, rt)
            rows.append({
                'model': model_name, 'layer': li,
                'relative_depth': (li + 1) / num_layers,
                'n_words': int(len(keep)),
                'representation_type': PRIMARY_REPRESENTATION,
                'readout_type': NATIVE_LM_HEAD_READOUT,
                'lexical_evidence_definition': 'logit(YES) - logit(NO), no fitting',
                'spearman_rho': float(rho), 'spearman_p': float(p),
                'human_rt_used_in_prediction': False,
                'human_rt_column': self.rt_alignment.rt_column,
            })
        idx = [i for i, r in enumerate(rows) if np.isfinite(r.get('spearman_p', np.nan))]
        if idx:
            _, pc, _, _ = multipletests([rows[i]['spearman_p'] for i in idx],
                                        alpha=self.config.ALPHA, method='fdr_bh')
            for i, p in zip(idx, pc):
                rows[i]['spearman_fdr_p'] = float(p)
                rows[i]['spearman_significant'] = bool(p < self.config.ALPHA)
        return rows

    # ════════════════════════════════════════════════════════════════════════
    # NATIVE GENERATION DIAGNOSTIC (§13, §14, §55) — behaviour, not probe
    # ════════════════════════════════════════════════════════════════════════
    def _summarise_native_generation(self, model_name, df: pd.DataFrame):
        """SECONDARY behavioural diagnostic (§12): what the full model actually
        emits when allowed to generate one token. Never mixed into the primary
        layer-wise LM-head curve."""
        n = len(df)
        if 'native_prediction' not in df.columns:
            return ({'model': model_name, 'n_samples': n,
                     'generation_diagnostic_run': False,
                     'native_accuracy': np.nan,
                     'prompt_elicitation_reliable': None},
                    pd.DataFrame(columns=['model', 'generated_token_id',
                                          'generated_token_text', 'count']))
        pred = df['native_prediction']
        unknown_rate = float((pred == 'UNKNOWN').mean())
        known = df[pred != 'UNKNOWN']
        summ = {
            'model': model_name, 'n_samples': n,
            'generation_diagnostic_run': True,
            'analysis_role': 'secondary_behavioural_diagnostic',
            'yes_rate': float((pred == 'WORD').mean()),
            'no_rate': float((pred == 'NONWORD').mean()),
            'unknown_rate': unknown_rate,
            # identical by definition: a first token outside the YES/NO sets
            'unexpected_token_rate': unknown_rate,
            'native_accuracy': float(df['native_correct'].mean()),
            'native_accuracy_on_yes_no_only': (float(known['native_correct'].mean())
                                               if len(known) else np.nan),
            'native_balanced_accuracy': float(np.mean([
                df.loc[df['gold_is_word'] == c, 'native_correct'].mean()
                for c in (0, 1) if (df['gold_is_word'] == c).any()])),
            'forced_choice_accuracy_yes_vs_no_mass':
                float(df['forced_choice_correct'].mean()),
            'mean_prompt_length': float(df['prompt_length'].mean()),
            'n_distinct_generated_tokens': int(df['generated_token_id'].nunique()),
            'generation_strategy': 'greedy', 'max_new_tokens': 1, 'do_sample': False,
        }
        for grp, mask in [('word_high', (df['gold_is_word'] == 1) & (df['freq_group'] == 'high')),
                          ('word_mid',  (df['gold_is_word'] == 1) & (df['freq_group'] == 'mid')),
                          ('word_low',  (df['gold_is_word'] == 1) & (df['freq_group'] == 'low')),
                          ('nonword',   df['gold_is_word'] == 0)]:
            summ[f'native_accuracy_{grp}'] = (float(df.loc[mask, 'native_correct'].mean())
                                              if mask.any() else np.nan)
            summ[f'n_{grp}'] = int(mask.sum())
        reliable = (unknown_rate <= self.config.PROMPT_UNRELIABLE_UNKNOWN_RATE and
                    summ['native_accuracy'] > self.config.PROMPT_UNRELIABLE_MIN_ACCURACY)
        summ['prompt_elicitation_reliable'] = bool(reliable)
        if not reliable:
            logger.warning(f"WARNING: lexical-decision prompt elicitation is unreliable "
                           f"for {model_name} on the FULL dataset (unknown_rate="
                           f"{unknown_rate:.3f}, native_accuracy="
                           f"{summ['native_accuracy']:.3f}). Reported, not discarded.")
        tc = (df.groupby(['generated_token_id', 'generated_token_text',
                          'native_prediction'], dropna=False)
                .size().reset_index(name='count')
                .sort_values('count', ascending=False))
        tc.insert(0, 'model', model_name)
        tc['fraction'] = tc['count'] / max(n, 1)
        summ['top_generated_tokens'] = json.dumps(
            {repr(r.generated_token_text): int(r.count) for r in tc.head(15).itertuples()})
        print(f"  Native generation: acc={summ['native_accuracy']:.3f}  "
              f"YES={summ['yes_rate']:.3f}  NO={summ['no_rate']:.3f}  "
              f"UNKNOWN={unknown_rate:.3f}  forced-choice acc="
              f"{summ['forced_choice_accuracy_yes_vs_no_mass']:.3f}")
        return summ, tc

    # ════════════════════════════════════════════════════════════════════════
    # PRIMARY EMERGENCE-CURVE STATISTICS (§18-21, §73)
    # ════════════════════════════════════════════════════════════════════════
    def _bootstrap_ci(self, correct: np.ndarray, seed: int = SEED) -> tuple[float, float]:
        """Percentile bootstrap 95% CI of test accuracy (resampling test items)."""
        n = len(correct)
        if n == 0:
            return (np.nan, np.nan)
        rng = np.random.default_rng(seed)
        B = self.config.N_BOOTSTRAP
        out, chunk = np.empty(B), 250
        for s in range(0, B, chunk):
            k = min(chunk, B - s)
            out[s:s + k] = correct[rng.integers(0, n, size=(k, n))].mean(1)
        return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

    @staticmethod
    def _mcnemar_exact(c_a: np.ndarray, c_b: np.ndarray) -> tuple[int, int, float]:
        """Exact (binomial) McNemar test on paired correctness vectors."""
        return mcnemar_exact(c_a, c_b)

    def _token_identity_control(self, test_idx, y_te, prob_te, pred_te, gen_tok):
        """
        CONFOUND CONTROL specific to the first-generated-output-token position.
        The generated token (e.g. " Yes"/" No") is the INPUT at that position,
        so its identity is decodable from the very first layer.

        (a) token_identity_baseline_accuracy — gold label predicted from the
            generated-token ID alone (majority gold label per token ID, fitted
            on the NON-test items of the same split; unseen IDs → majority).
        (b) within_token_stratum_auroc / balanced_accuracy — probe performance
            computed separately inside each generated-token stratum (test items
            that produced the SAME token), averaged with stratum-size weights.
            Inside a stratum the token identity is constant, so values > 0.5
            indicate lexical information BEYOND the model's emitted answer.
        """
        n_all = len(gen_tok)
        tr_mask = np.ones(n_all, bool); tr_mask[test_idx] = False
        y_all_tr = self._is_word_cache[tr_mask]
        tok_tr = gen_tok[tr_mask]
        major = int(np.mean(y_all_tr) >= 0.5)
        mapping = {}
        for t in np.unique(tok_tr):
            m = y_all_tr[tok_tr == t].mean()
            mapping[int(t)] = int(m > 0.5) if m != 0.5 else major
        tok_te = gen_tok[test_idx]
        base_pred = np.array([mapping.get(int(t), major) for t in tok_te])
        base_acc = float(np.mean(base_pred == y_te))

        aucs, baccs, ws = [], [], []
        for t in np.unique(tok_te):
            m = tok_te == t
            if m.sum() < self.config.TOKEN_IDENTITY_MIN_STRATUM or \
                    len(np.unique(y_te[m])) < 2:
                continue
            aucs.append(roc_auc_score(y_te[m], prob_te[m]))
            baccs.append(balanced_accuracy_score(y_te[m], pred_te[m]))
            ws.append(int(m.sum()))
        ws_a = np.asarray(ws, float)
        return {
            'token_identity_baseline_accuracy': base_acc,
            'within_token_stratum_auroc': (float(np.average(aucs, weights=ws_a))
                                           if ws else np.nan),
            'within_token_stratum_balanced_accuracy': (float(np.average(baccs, weights=ws_a))
                                                       if ws else np.nan),
            'within_token_stratum_coverage': float(ws_a.sum() / max(len(y_te), 1)),
            'within_token_n_strata': len(ws),
        }

    def _build_layer_table(self, model_name, layer_results, num_layers, rep_type,
                           probe_type, targets, native_summary,
                           token_identity: bool) -> list[dict]:
        self._is_word_cache = np.asarray(targets['is_word']).astype(int)
        gen_tok = np.asarray(targets.get('generated_token_id', []))
        rows, prev = [], None
        for r in layer_results:
            li = r['layer']; ov = r.get('overall'); si = r.get('split_info') or {}
            row = {'model': model_name, 'layer': li,
                   'relative_depth': (li + 1) / num_layers,
                   'representation_type': rep_type,
                   'readout_type': SECONDARY_READOUT, 'probe_type': probe_type,
                   'n_train': si.get('n_train', np.nan),
                   'n_validation': si.get('n_val', np.nan),
                   'n_test': si.get('n_test', np.nan),
                   'native_generation_accuracy': native_summary.get('native_accuracy', np.nan)}
            if ov is None:
                rows.append(row); prev = None; continue
            y, yp, pr = ov['y_true'], ov['predictions'], ov['probabilities']
            correct = (y == yp).astype(float)
            chance = float(max(y.mean(), 1 - y.mean()))
            lo, hi = self._bootstrap_ci(correct)
            k, n = int(correct.sum()), len(y)
            val = r.get('validation') or {}
            fe = r['frequency_effect']
            row.update({
                'accuracy': ov['accuracy'], 'accuracy_ci95_low': lo,
                'accuracy_ci95_high': hi,
                'balanced_accuracy': ov['balanced_accuracy'],
                'macro_f1': ov['macro_f1'], 'f1': ov['f1'],
                'auroc': ov['auc'], 'auprc': ov['auprc'],
                'chance_accuracy': chance,
                'p_vs_chance_binomial': float(stats.binomtest(
                    k, n, chance, alternative='greater').pvalue),
                'validation_accuracy': val.get('accuracy', np.nan),
                'hf_word_accuracy': (r['high_frequency'] or {}).get('accuracy', np.nan),
                'lf_word_accuracy': (r['low_frequency'] or {}).get('accuracy', np.nan),
                'freq_accuracy_difference': fe.get('accuracy_difference', np.nan),
                'cohens_h': fe.get('cohens_h', np.nan),
            })
            te = np.asarray(si['test_idx'])
            if token_identity and len(gen_tok) and (gen_tok >= 0).all():
                ti = self._token_identity_control(te, y, pr, yp, gen_tok)
                ti['probe_minus_token_identity_baseline'] = (
                    ov['accuracy'] - ti['token_identity_baseline_accuracy'])
                row.update(ti)
            # consecutive-layer paired comparison (same test items)
            if prev is not None and np.array_equal(prev[0], te):
                b, c, p = self._mcnemar_exact(prev[1], correct)
                row.update({'delta_accuracy_vs_previous_layer': ov['accuracy'] - prev[2],
                            'mcnemar_p_vs_previous_layer': p})
            prev = (te, correct, ov['accuracy'])
            rows.append(row)

        def _adj(col, method, out):
            idx = [i for i, rw in enumerate(rows) if np.isfinite(rw.get(col, np.nan))]
            if idx:
                _, pc, _, _ = multipletests([rows[i][col] for i in idx],
                                            alpha=self.config.ALPHA, method=method)
                for i, p in zip(idx, pc):
                    rows[i][out] = float(p)
                    rows[i][out + '_significant'] = bool(p < self.config.ALPHA)
        _adj('p_vs_chance_binomial', 'fdr_bh', 'p_vs_chance_fdr_bh')
        _adj('mcnemar_p_vs_previous_layer', 'holm', 'mcnemar_p_vs_previous_layer_holm')
        return rows

    def _emergence_summary(self, model_name, rep_type, table, num_layers,
                           readout: str = SECONDARY_READOUT) -> dict:
        """
        Emergence descriptors of the layer-wise accuracy curve (§21).
        'x% of peak' means x% of the peak ABOVE-CHANCE information:
            first layer l with  acc_l − chance ≥ x · (acc_peak − chance).
        accuracy_auc_across_depth = ∫ (acc − chance)/(1 − chance) d(relative
            depth), divided by the depth range → mean normalised above-chance
            information in [−1, 1] (0 = chance everywhere, 1 = perfect everywhere).
        peak_* is descriptive and optimistically biased (max over noisy
        estimates); peak_layer_selected_on_validation + its TEST accuracy is
        the unbiased counterpart.
        """
        L  = np.array([r['layer'] for r in table], float)
        acc = np.array([r.get('accuracy', np.nan) for r in table], float)
        val = np.array([r.get('validation_accuracy', np.nan) for r in table], float)
        ch  = np.array([r.get('chance_accuracy', np.nan) for r in table], float)
        out = {'model': model_name, 'representation_type': rep_type,
               'readout_type': readout, 'num_layers': num_layers}
        ok = np.isfinite(acc)
        if not ok.any():
            return out
        chance = float(np.nanmean(ch))
        rel = (L + 1) / num_layers
        pi = int(np.nanargmax(acc))
        peak = float(acc[pi]); peak_info = peak - chance

        def _reach(q):
            if peak_info <= 0:
                return np.nan
            hit = np.where(ok & (acc - chance >= q * peak_info))[0]
            return int(L[hit[0]]) if len(hit) else np.nan
        early = ok & (rel <= self.config.EMERGENCE_EARLY_DEPTH)
        x, yv = rel[ok], (acc[ok] - chance) / max(1 - chance, 1e-12)
        auc = (float(scipy.integrate.trapezoid(yv, x) / (x[-1] - x[0]))
               if len(x) > 1 and x[-1] > x[0] else float(yv.mean()))
        vi = int(np.nanargmax(val)) if np.isfinite(val).any() else None
        out.update({
            'chance_accuracy': chance,
            'peak_layer': int(L[pi]), 'peak_relative_depth': float(rel[pi]),
            'peak_accuracy': peak,
            'early_layer_accuracy': float(np.nanmean(acc[early])) if early.any() else np.nan,
            'early_depth_cutoff': self.config.EMERGENCE_EARLY_DEPTH,
            'final_layer_accuracy': float(acc[ok][-1]),
            'layer_reaching_80pct_of_peak': _reach(0.80),
            'layer_reaching_90pct_of_peak': _reach(0.90),
            'accuracy_auc_across_depth': auc,
            'peak_layer_selected_on_validation': int(L[vi]) if vi is not None else np.nan,
            'test_accuracy_at_validation_selected_layer': float(acc[vi]) if vi is not None else np.nan,
            'n_layers_above_chance_fdr': int(sum(bool(r.get('p_vs_chance_fdr_bh_significant', False))
                                                 for r in table)),
        })
        return out

    def _representation_ablation_rows(self, model_name, primary_results, ablation,
                                      num_layers, probe_type) -> list[dict]:
        """Primary vs each control at every layer, paired on identical test
        items (exact McNemar, Holm-corrected across layers per control)."""
        rows = []

        def _m(r):
            ov = r.get('overall'); fe = r['frequency_effect']
            g = lambda k: (ov[k] if ov else np.nan)
            return {'accuracy': g('accuracy'), 'balanced_accuracy': g('balanced_accuracy'),
                    'macro_f1': g('macro_f1'), 'f1': g('f1'), 'auroc': g('auc'),
                    'auprc': g('auprc'),
                    'freq_diff': fe.get('accuracy_difference', np.nan),
                    'cohens_h': fe.get('cohens_h', np.nan)}
        for r in primary_results:
            rows.append({'model': model_name, 'layer': r['layer'],
                         'relative_depth': (r['layer'] + 1) / num_layers,
                         'representation_type': PRIMARY_REPRESENTATION,
                         'role': 'primary', 'probe_type': probe_type, **_m(r)})
        for rep_type, res_list in ablation.items():
            block = []
            for rp, rc in zip(primary_results, res_list):
                row = {'model': model_name, 'layer': rc['layer'],
                       'relative_depth': (rc['layer'] + 1) / num_layers,
                       'representation_type': rep_type, 'role': 'control',
                       'probe_type': probe_type, **_m(rc)}
                op, oc = rp.get('overall'), rc.get('overall')
                if op and oc and np.array_equal(rp['split_info']['test_idx'],
                                                rc['split_info']['test_idx']):
                    cp = (op['y_true'] == op['predictions']).astype(int)
                    cc = (oc['y_true'] == oc['predictions']).astype(int)
                    b, c, p = self._mcnemar_exact(cp, cc)
                    row.update({'accuracy_primary_minus_control': op['accuracy'] - oc['accuracy'],
                                'mcnemar_primary_only_correct': b,
                                'mcnemar_control_only_correct': c,
                                'mcnemar_p': p})
                block.append(row)
            idx = [i for i, rw in enumerate(block) if np.isfinite(rw.get('mcnemar_p', np.nan))]
            if idx:
                _, pc, _, _ = multipletests([block[i]['mcnemar_p'] for i in idx],
                                            alpha=self.config.ALPHA, method='holm')
                for i, p in zip(idx, pc):
                    block[i]['mcnemar_p_holm'] = float(p)
                    block[i]['difference_significant_holm'] = bool(p < self.config.ALPHA)
            rows.extend(block)
        return rows

    # ════════════════════════════════════════════════════════════════════════
    # SPEC-NAMED CONSOLIDATED OUTPUTS (§62), FIGURES (§19, §57), METADATA (§63)
    # ════════════════════════════════════════════════════════════════════════
    def _set_rows(self, key: str, model_name: str, rows: list[dict]):
        """Idempotent per model: replaces that model's rows under `key`."""
        cur = [r for r in self.spec_rows.get(key, []) if r.get('model') != model_name]
        self.spec_rows[key] = cur + [dict(r, model=model_name) for r in rows]

    @staticmethod
    def _scalar_layer_row(r: dict) -> dict:
        ov = r.get('overall') or {}
        fe = r.get('frequency_effect') or {}
        si = r.get('split_info') or {}
        return {'layer': r['layer'],
                'accuracy': ov.get('accuracy', np.nan),
                'balanced_accuracy': ov.get('balanced_accuracy', np.nan),
                'macro_f1': ov.get('macro_f1', np.nan), 'f1': ov.get('f1', np.nan),
                'auroc': ov.get('auc', np.nan), 'auprc': ov.get('auprc', np.nan),
                'hf_word_accuracy': (r.get('high_frequency') or {}).get('accuracy', np.nan),
                'lf_word_accuracy': (r.get('low_frequency') or {}).get('accuracy', np.nan),
                'freq_diff': fe.get('accuracy_difference', np.nan),
                'cohens_h': fe.get('cohens_h', np.nan),
                'freq_p_fdr': fe.get('p_value_corrected', np.nan),
                'n_train': si.get('n_train', np.nan), 'n_test': si.get('n_test', np.nan)}

    def _collect_spec_rows(self, model_name: str, res: dict):
        PR, pt = PRIMARY_REPRESENTATION, res.get('probe_type', '')
        tag = {'representation_type': PR, 'probe_type': pt}

        # ── PRIMARY: layer-wise cross-fitted linear lexicality probe ────
        self._set_rows('layerwise_crossfitted_probe_ldt', model_name,
                       res['crossfit_table'])
        # ── OPTIONAL native LM-head diagnostic (empty when disabled) ────
        if res.get('native_ldt_table'):
            self._set_rows('layerwise_native_lm_head_ldt', model_name,
                           res['native_ldt_table'])
        if res.get('readout_generation_consistency'):
            self._set_rows('readout_generation_consistency', model_name,
                           [res['readout_generation_consistency']])
        if res.get('native_rt_alignment'):
            self._set_rows('layerwise_native_lm_head_rt_alignment', model_name,
                           res['native_rt_alignment'])
        # ── SECONDARY: trained linear probe on the same representation ──
        self._set_rows('layerwise_trained_probe_results', model_name,
                       list(res['probe_table']) +
                       [r for tbl in res.get('control_tables', {}).values() for r in tbl])
        self._set_rows('layerwise_emergence_summary', model_name,
                       res['emergence_summary'])
        self._set_rows('prompt_diagnostics', model_name,
                       res['prompt_diagnostics'].to_dict('records'))
        self._set_rows('prompt_diagnostics_summary', model_name, [res['native_summary']])
        self._set_rows('generated_token_counts', model_name,
                       res['generated_token_counts'].to_dict('records'))

        df_ = res.get('direct_freq') or {}
        self._set_rows('direct_frequency_probe', model_name,
                       [{**tag, 'task': 'binary_high_vs_low', **r} for r in df_.get('binary', [])] +
                       [{'representation_type': PR,
                         'probe_type': 'sklearn_multinomial_logistic_regression_C1',
                         'task': '3way_high_mid_low', **r} for r in df_.get('three_way', [])])

        tc = res.get('token_ctrl') or {}
        rows = [{**tag, 'analysis': 'token_stats', **tc.get('token_stats', {})}] if tc else []
        rows += [{**tag, 'analysis': 'single_token_ldt', **self._scalar_layer_row(r)}
                 for r in tc.get('single_token_results', [])]
        rows += [{**tag, 'analysis': 'token_count_matched_ldt', **self._scalar_layer_row(r)}
                 for r in tc.get('token_matched_results', [])]
        self._set_rows('tokenization_control', model_name, rows)

        cm = res.get('confound_matched') or {}
        self._set_rows('lexical_confound_matching', model_name,
                       [{**tag, 'row_type': 'balance', **r}
                        for r in cm.get('balance_before', []) + cm.get('balance_after', [])] +
                       [{**tag, 'row_type': 'matched_frequency_probe', **r}
                        for r in cm.get('matched_probe_results', [])])

        st = res.get('stability') or {}
        self._set_rows('multi_seed_stability', model_name,
                       [{**tag, 'seeds': json.dumps(st.get('seeds', [])), **r}
                        for r in st.get('layer_stats', [])])

        self._set_rows('representation_ablation', model_name,
                       res.get('representation_ablation_rows', []))

        sel = res.get('selectivity') or {}
        by_layer: dict[int, dict] = {}
        for key, pre in (('selectivity', ''), ('linear_probe', 'sklearn_lr_'),
                         ('shuffled_label', 'shuffled_'), ('class_ratio_analysis', '')):
            for r in sel.get(key, []):
                d = by_layer.setdefault(r['layer'], {'layer': r['layer'], **tag})
                for k, v in r.items():
                    if k != 'layer':
                        d[(pre + k) if pre and k not in ('expected_baseline',
                                                         'class_ratio') else k] = v
        self._set_rows('probe_selectivity', model_name,
                       [by_layer[k] for k in sorted(by_layer)])

        reg = (res.get('regression') or {}).get('regression_results', [])
        self._set_rows('continuous_frequency_regression', model_name,
                       [{**tag, 'probe_type': 'ridge_regression_alpha_selected_on_validation',
                         **r} for r in reg])

        iv = res.get('intervention') or []
        self._set_rows('representation_intervention', model_name,
                       [{**tag, **{k: (json.dumps(v) if isinstance(v, list) else v)
                                   for k, v in r.items()}} for r in iv])

        ctx = res.get('contextual') or {}
        self._set_rows('contextual_frequency_analysis', model_name,
                       [{'representation_type': PR,
                         'condition': 'isolated_task_first_output_prediction_position', **r}
                        for r in ctx.get('isolated', [])] +
                       [{'representation_type': RepresentationType.CONTEXTUAL_TARGET_TOKEN.value,
                         'condition': 'sentence_context_target_token', **r}
                        for r in ctx.get('contextual', [])])

    def _write_spec_outputs(self):
        d = self.config.PRIMARY_DIR
        for key, rows in self.spec_rows.items():
            df = pd.DataFrame(rows)
            if 'model' in df.columns:
                df = df[['model'] + [c for c in df.columns if c != 'model']]
            df.to_csv(os.path.join(d, f'{key}.csv'), index=False)
        tr = pd.DataFrame(self.cross_model_transfer_rows)
        if tr.empty:
            tr = pd.DataFrame(columns=['source_model', 'target_model', 'source_layer',
                                       'target_layer', 'representation_type',
                                       'source_relative_depth', 'target_relative_depth',
                                       'transfer_accuracy', 'skip_reason'])
        tr.to_csv(os.path.join(d, 'cross_model_transfer.csv'), index=False)
        logger.info(f"✓ spec-named outputs written to {d}")

    def _primary_curve_panel(self, ax, table, x_key, color, label, show_ci=True):
        t = pd.DataFrame(table).sort_values('layer')
        if 'accuracy' not in t or t['accuracy'].isna().all():
            return
        ax.plot(t[x_key], t['accuracy'], 'o-', lw=2.4, ms=5, color=color, label=label)
        if show_ci and 'accuracy_ci95_low' in t:
            ax.fill_between(t[x_key], t['accuracy_ci95_low'], t['accuracy_ci95_high'],
                            color=color, alpha=0.15)

    def _plot_primary_model(self, model_name, res):
        """PRIMARY figure: layer-wise cross-fitted probe accuracy."""
        safe = model_name.replace(' ', '_').replace('/', '_')
        ct = pd.DataFrame(res['crossfit_table'])
        if not ct.empty and 'accuracy' in ct and not ct['accuracy'].isna().all():
            ct = ct.sort_values('layer')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 6.5))
            self._primary_curve_panel(ax1, res['crossfit_table'], 'layer', '#1A535C',
                                      'Out-of-fold accuracy, 95% bootstrap CI')
            ax1.plot(ct['layer'], ct['balanced_accuracy'], 's--', lw=2, color='#3A506B',
                     label='Balanced accuracy')
            ax1.plot(ct['layer'], ct['auroc'], '^-.', lw=2, color='#F18F01', label='AUROC')
            sig = ct.get('p_vs_chance_fdr_bh_significant',
                         pd.Series(False, index=ct.index)).fillna(False).astype(bool)
            ax1.scatter(ct.loc[~sig, 'layer'], ct.loc[~sig, 'accuracy'], s=60,
                        facecolors='white', edgecolors='#1A535C', zorder=4,
                        label='n.s. vs chance (BH-FDR)')
            ax1.axhline(float(ct['chance_accuracy'].iloc[0]), color='gray', ls=':',
                        lw=1.4, label='Chance (majority class)')
            b = res.get('best_lexicality_layer') or {}
            if b:
                ax1.axvline(b['best_lexicality_layer'], color='#D62246', ls='--', lw=1.2,
                            label=f"Best lexicality layer ({b['best_lexicality_layer']})")
            ax1.set(xlabel='Transformer Layer',
                    ylabel='Held-out WORD vs NONWORD performance',
                    title=f'PRIMARY — cross-fitted linear lexicality probe at the '
                          f'first-output prediction position\n{model_name}')
            ax1.legend(fontsize=8, loc='lower right'); ax1.grid(True, alpha=0.3)

            ax2.plot(ct['layer'], ct['mean_lexical_evidence_words'], 'o-', lw=2,
                     label='Words: mean logit P(WORD)')
            ax2.plot(ct['layer'], ct['mean_lexical_evidence_nonwords'], 's--', lw=2,
                     label='Nonwords: mean logit P(WORD)')
            ax2.axhline(0, color='gray', ls=':', lw=1.3)
            ax2.set(xlabel='Transformer Layer', ylabel='Lexical evidence = logit(P(WORD))',
                    title=f'Out-of-fold lexical evidence by class — {model_name}')
            ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                     f'{safe}_layerwise_crossfitted_probe.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()

        nt = pd.DataFrame(res['native_ldt_table'])
        if nt.empty:
            return
        nt = nt.sort_values('layer')
        if not nt.empty and not nt.get('accuracy', pd.Series(dtype=float)).isna().all():
            fig, ax = plt.subplots(figsize=(14, 6.5))
            self._primary_curve_panel(ax, res['native_ldt_table'], 'layer', '#1A535C',
                                      'Native LM-head LDT accuracy, 95% bootstrap CI')
            sig = nt.get('p_vs_chance_fdr_bh_significant',
                         pd.Series(False, index=nt.index)).fillna(False).astype(bool)
            ax.scatter(nt.loc[~sig, 'layer'], nt.loc[~sig, 'accuracy'], s=60,
                       facecolors='white', edgecolors='#1A535C', zorder=4,
                       label='n.s. vs chance (BH-FDR)')
            na = res['native_summary'].get('native_accuracy', np.nan)
            if np.isfinite(na):
                ax.axhline(na, color='#F18F01', ls='-.', lw=1.6,
                           label='Full-model one-token generation accuracy (secondary)')
            ax.axhline(float(nt['chance_accuracy'].iloc[0]), color='gray', ls=':', lw=1.4,
                       label='Chance (majority class)')
            ax.set(xlabel='Transformer Layer',
                   ylabel='Native LM-head WORD vs NONWORD accuracy',
                   title=f'PRIMARY — layer-wise native LM-head lexical decision — {model_name}',
                   ylim=[max(0.0, np.nanmin(nt['accuracy_ci95_low']) - 0.05), 1.01])
            ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                     f'{safe}_layerwise_native_lm_head_accuracy.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all')

            # native YES/NO logits across depth
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(nt['layer'], nt['mean_yes_logit'], 'o-', lw=2, label='mean logit(YES)')
            ax.plot(nt['layer'], nt['mean_no_logit'], 's--', lw=2, label='mean logit(NO)')
            ax2 = ax.twinx()
            ax2.plot(nt['layer'], nt['yes_prediction_rate'], '^:', color='#D62246',
                     lw=2, label='YES prediction rate')
            ax2.set_ylabel('YES prediction rate'); ax2.set_ylim(0, 1)
            ax.set(xlabel='Transformer Layer', ylabel='Mean answer logit',
                   title=f'Native LM-head answer logits across depth — {model_name}')
            ax.legend(fontsize=9, loc='upper left'); ax2.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                     f'{safe}_native_lm_head_answer_logits.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all')

        t = pd.DataFrame(res['probe_table'])
        if t.empty or 'accuracy' not in t or t['accuracy'].isna().all():
            return
        t = t.sort_values('layer')
        fig, ax = plt.subplots(figsize=(14, 6.5))
        self._primary_curve_panel(ax, res['crossfit_table'], 'layer', '#1A535C',
                                  'PRIMARY: cross-fitted probe accuracy, 95% CI')
        ax.plot(t['layer'], t['accuracy'], 's--', lw=2.2, color='#3A506B',
                label='SECONDARY: hold-out probe accuracy')
        if not nt.empty and 'accuracy' in nt:
            ax.plot(nt['layer'], nt['accuracy'], '^-.', lw=2, color='#F18F01',
                    label='DIAGNOSTIC: native LM-head accuracy')
        ax.axhline(0.5, color='gray', ls=':', lw=1.4, label='Chance (50%)')
        ax.set(xlabel='Transformer Layer', ylabel='WORD vs NONWORD accuracy',
               title=f'Primary probe vs secondary readouts — {model_name}')
        ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                 f'{safe}_native_vs_trained_probe.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all')

        fig, ax = plt.subplots(figsize=(14, 6.5))
        for col, st, lab in [('balanced_accuracy', 'o-', 'Balanced accuracy'),
                             ('macro_f1', 's--', 'Macro-F1'), ('auroc', '^-.', 'AUROC'),
                             ('within_token_stratum_auroc', 'd:',
                              'AUROC within generated-token strata')]:
            if col in t and not t[col].isna().all():
                ax.plot(t['layer'], t[col], st, lw=2, ms=5, label=lab)
        ax.axhline(0.5, color='gray', ls=':', lw=1.3)
        ax.set(xlabel='Transformer Layer', ylabel='Score',
               title=f'Layer-wise balanced accuracy / F1 / AUROC — {model_name}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                 f'{safe}_layerwise_balanced_accuracy_f1.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()

    def _plot_spec_figures(self, all_results):
        d = self.config.PRIMARY_DIR
        models = list(all_results.keys())
        pal = sns.color_palette("husl", max(len(models), 1))

        # Figure 1 — PRIMARY: layer-wise cross-fitted lexicality probe
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 7.5))
        for m, c in zip(models, pal):
            self._primary_curve_panel(a1, all_results[m]['crossfit_table'], 'layer', c, m)
            self._primary_curve_panel(a2, all_results[m]['crossfit_table'],
                                      'relative_depth', c, m)
            b = all_results[m].get('best_lexicality_layer') or {}
            if b:
                a2.scatter([b['best_lexicality_relative_depth']],
                           [b['best_lexicality_accuracy']], marker='*', s=240,
                           color=c, edgecolors='black', zorder=5)
        for ax, xl in ((a1, 'Transformer Layer'), (a2, 'Relative depth (layer+1)/L')):
            ax.axhline(0.5, color='red', ls='--', lw=1.5, label='Chance (50%)')
            ax.set(xlabel=xl, ylabel='Out-of-fold WORD vs NONWORD accuracy')
            ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        fig.suptitle('PRIMARY — layer-wise lexicality probing at the first-output '
                     'prediction position (5-fold cross-fitted linear probes, '
                     '95% bootstrap CI; stars = best lexicality layer)', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(d, 'layerwise_crossfitted_probe_accuracy.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all')

        # AUROC + lexical evidence separation across depth
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 7))
        for m, c in zip(models, pal):
            t = pd.DataFrame(all_results[m]['crossfit_table']).sort_values('layer')
            if 'auroc' in t:
                a1.plot(t['relative_depth'], t['auroc'], 'o-', color=c, lw=2, label=m)
                a2.plot(t['relative_depth'],
                        t['mean_lexical_evidence_words'] - t['mean_lexical_evidence_nonwords'],
                        's-', color=c, lw=2, label=m)
        a1.axhline(0.5, color='gray', ls=':', lw=1.3)
        a1.set(xlabel='Relative depth (layer+1)/L', ylabel='Out-of-fold AUROC',
               title='Lexicality AUROC across depth')
        a2.axhline(0, color='gray', ls=':', lw=1.3)
        a2.set(xlabel='Relative depth (layer+1)/L',
               ylabel='Mean logit P(WORD): words − nonwords',
               title='Separation of lexical evidence across depth')
        for ax in (a1, a2):
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(d, 'layerwise_lexical_evidence_separation.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all')

        # Cross-fitted (primary) vs hold-out probe (secondary)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 7.5))
        for m, c in zip(models, pal):
            ct = pd.DataFrame(all_results[m]['crossfit_table']).sort_values('layer')
            pt = pd.DataFrame(all_results[m]['probe_table']).sort_values('layer')
            if 'accuracy' in ct:
                a1.plot(ct['relative_depth'], ct['accuracy'], 'o-', color=c, lw=2.2, label=m)
            if 'accuracy' in pt:
                a2.plot(pt['relative_depth'], pt['accuracy'], 's--', color=c, lw=2.2, label=m)
        a1.set(xlabel='Relative depth (layer+1)/L', ylabel='Accuracy',
               title='PRIMARY — 5-fold cross-fitted probe (all items scored out-of-fold)')
        a2.set(xlabel='Relative depth (layer+1)/L', ylabel='Accuracy',
               title='SECONDARY — single hold-out probe (basis of analyses 1-10)')
        for ax in (a1, a2):
            ax.axhline(0.5, color='gray', ls=':', lw=1.3)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(d, 'crossfitted_vs_holdout_probe.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all')

        # OPTIONAL native LM-head diagnostic, only when it was run
        if all(all_results[m].get('native_ldt_table') for m in models):
            fig, ax = plt.subplots(figsize=(14, 7))
            for m, c in zip(models, pal):
                nt = pd.DataFrame(all_results[m]['native_ldt_table']).sort_values('layer')
                ct = pd.DataFrame(all_results[m]['crossfit_table']).sort_values('layer')
                ax.plot(nt['relative_depth'], nt['accuracy'], 'o--', color=c, lw=2,
                        label=f'{m} — native LM head (diagnostic)')
                ax.plot(ct['relative_depth'], ct['accuracy'], 's-', color=c, lw=2,
                        label=f'{m} — cross-fitted probe (PRIMARY)')
            ax.axhline(0.5, color='gray', ls=':', lw=1.3)
            ax.set(xlabel='Relative depth (layer+1)/L', ylabel='Accuracy',
                   title='Optional diagnostic: native YES/NO readout vs the primary probe')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(d, 'native_lm_head_diagnostic.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all')

        # SECONDARY hold-out probe curve on its own
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 7.5))
        for m, c in zip(models, pal):
            self._primary_curve_panel(a1, all_results[m]['probe_table'], 'layer', c, m)
            self._primary_curve_panel(a2, all_results[m]['probe_table'],
                                      'relative_depth', c, m)
        for ax, xl in ((a1, 'Transformer Layer'), (a2, 'Relative depth (layer+1)/L')):
            ax.axhline(0.5, color='red', ls='--', lw=1.5, label='Chance (50%)')
            ax.set(xlabel=xl, ylabel='Trained-probe accuracy')
            ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        fig.suptitle('SECONDARY — hold-out linear probe (single stratified split), '
                     'reported for continuity with analyses 1-10', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(d, 'layerwise_trained_probe_accuracy.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all')

        # Figure 2 — balanced accuracy / macro-F1
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 7))
        for m, c in zip(models, pal):
            t = pd.DataFrame(all_results[m]['crossfit_table']).sort_values('layer')
            if 'balanced_accuracy' in t:
                a1.plot(t['relative_depth'], t['balanced_accuracy'], 'o-', color=c, lw=2, label=m)
                a2.plot(t['relative_depth'], t['macro_f1'], 's-', color=c, lw=2, label=m)
        for ax, yl in ((a1, 'Balanced accuracy'), (a2, 'Macro-F1')):
            ax.axhline(0.5, color='gray', ls=':', lw=1.3)
            ax.set(xlabel='Relative depth (layer+1)/L', ylabel=yl); ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(d, 'layerwise_balanced_accuracy_f1.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all')

        # Figure 3 — layer-wise frequency prediction
        fig, axes = plt.subplots(1, 3, figsize=(27, 7))
        for m, c in zip(models, pal):
            md = all_results[m]; nl = md['num_layers']
            b = pd.DataFrame((md.get('direct_freq') or {}).get('binary', []))
            w = pd.DataFrame((md.get('direct_freq') or {}).get('three_way', []))
            r = pd.DataFrame((md.get('regression') or {}).get('regression_results', []))
            for ax, df_, col in ((axes[0], b, 'accuracy'), (axes[1], w, 'accuracy'),
                                 (axes[2], r, 'r2')):
                if not df_.empty and col in df_ and not df_[col].isna().all():
                    ax.plot((df_['layer'] + 1) / nl, df_[col], 'o-', color=c, lw=2, label=m)
        for ax, yl, ch in ((axes[0], 'High vs Low accuracy (words only)', 0.5),
                           (axes[1], '3-way H/M/L accuracy', 1 / 3),
                           (axes[2], 'Ridge R² (log HAL frequency)', 0.0)):
            ax.axhline(ch, color='gray', ls='--', lw=1.2)
            ax.set(xlabel='Relative depth (layer+1)/L', ylabel=yl)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(d, 'layerwise_frequency_prediction.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all')

        # frequency_effect.png — HF − LF WORD accuracy of the LDT probe
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 7))
        for m, c in zip(models, pal):
            t = pd.DataFrame(all_results[m]['crossfit_table']).sort_values('layer')
            if 'freq_accuracy_difference' in t:
                a1.plot(t['relative_depth'], t['freq_accuracy_difference'], 'o-', color=c, lw=2, label=m)
                a2.plot(t['relative_depth'], t['word_accuracy_high_freq'], 's-',
                        color=c, lw=2, label=m)
        a1.axhline(0, color='black', lw=1.2)
        a1.set(xlabel='Relative depth (layer+1)/L',
               ylabel='High − Low frequency WORD accuracy (cross-fitted probe)',
               title='Frequency effect on the primary probe (not by itself '
                     'evidence of frequency coding)')
        a2.set(xlabel='Relative depth (layer+1)/L', ylabel='High-freq word accuracy')
        for ax in (a1, a2):
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(d, 'frequency_effect.png'), bbox_inches='tight', dpi=300)
        plt.close('all')

        # Figure 5 — native generation behaviour by model
        ns = pd.DataFrame([all_results[m]['native_summary'] for m in models])
        cols = [('native_accuracy', 'Native accuracy (greedy 1st token)'),
                ('forced_choice_accuracy_yes_vs_no_mass', 'Forced-choice accuracy (P(YES) vs P(NO))'),
                ('yes_rate', 'YES rate'), ('no_rate', 'NO rate'),
                ('unknown_rate', 'UNKNOWN / unexpected rate')]
        cols = [(c, l) for c, l in cols if c in ns.columns]
        if cols:
            fig, ax = plt.subplots(figsize=(max(12, 2.2 * len(models)), 7))
            x = np.arange(len(models)); wdt = 0.8 / len(cols)
            for i, (col, lab) in enumerate(cols):
                ax.bar(x + (i - (len(cols) - 1) / 2) * wdt, ns[col], wdt, label=lab)
            ax.axhline(0.5, color='gray', ls=':', lw=1.2)
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha='right')
            ax.set(ylabel='Rate / accuracy', ylim=[0, 1.05],
                   title='Native one-token generation behaviour '
                         '(secondary diagnostic, not the probe)')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(d, 'native_generation_accuracy_by_model.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all')
        else:
            logger.info("Generation diagnostic disabled — "
                        "native_generation_accuracy_by_model.png skipped.")

        # Readout vs full-model generation consistency (§22) + the
        # generated-token CONTROL confound diagnostic, if that control was run.
        cons = pd.DataFrame([all_results[m]['readout_generation_consistency']
                             for m in models
                             if all_results[m].get('readout_generation_consistency')])
        if not cons.empty and 'agreement_rate' in cons.columns:
            fig, ax = plt.subplots(figsize=(max(10, 2.0 * len(models)), 6.5))
            x = np.arange(len(models))
            ax.bar(x - 0.2, cons['agreement_rate'], 0.4,
                   label='Final-layer readout vs one-token generation')
            ax.bar(x + 0.2, cons['agreement_rate_excluding_unknown_generation'], 0.4,
                   label='Same, excluding UNKNOWN generated tokens')
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha='right')
            ax.set(ylabel='Agreement rate', ylim=[0, 1.05],
                   title='§22 consistency check (disagreement is reported, never forced)')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(d, 'readout_generation_consistency.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all')

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 7))
        for m, c in zip(models, pal):
            ct = all_results[m].get('control_tables', {}).get(
                RepresentationType.GENERATED_TOKEN_CONTROL.value, [])
            t = pd.DataFrame(ct)
            if not t.empty:
                t = t.sort_values('layer')
            if 'probe_minus_token_identity_baseline' in t:
                a1.plot(t['relative_depth'], t['probe_minus_token_identity_baseline'],
                        'o-', color=c, lw=2, label=m)
                a2.plot(t['relative_depth'], t['within_token_stratum_auroc'],
                        's-', color=c, lw=2, label=m)
        a1.axhline(0, color='black', lw=1.2)
        a1.set(xlabel='Relative depth (layer+1)/L',
               ylabel='Probe accuracy − token-identity baseline',
               title='Generated-token CONTROL: information beyond the emitted token')
        a2.axhline(0.5, color='gray', ls=':', lw=1.3)
        a2.set(xlabel='Relative depth (layer+1)/L', ylabel='AUROC within generated-token strata',
               title='WORD/NONWORD separability among items with the SAME emitted token')
        for ax in (a1, a2):
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(d, 'token_identity_control.png'), bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info(f"✓ spec figures written to {d}")

    # ════════════════════════════════════════════════════════════════════════
    # NAACL REVISION — outputs, figures (I1–I6) and cross-model analyses (H)
    # ════════════════════════════════════════════════════════════════════════
    def _naacl_path(self, *parts):
        return os.path.join(self.config.NAACL_DIR, *parts)

    @staticmethod
    def _csv_safe(rows):
        """Drops private list-valued keys (e.g. null distributions kept only
        for figures) and JSON-encodes remaining lists."""
        out = []
        for r in rows or []:
            d = {}
            for k, v in r.items():
                if k.startswith('_'):
                    continue
                d[k] = json.dumps(v, default=float) if isinstance(v, (list, tuple, dict)) else v
            out.append(d)
        return out

    def _collect_naacl_rows(self, model_name, res):
        ml = res.get('model_level') or {}
        fc = res.get('frequency_covariates') or {}
        ctx = res.get('contextual') or {}
        cm = res.get('confound_matched') or {}
        blocks = {
            'A1_mlp_vs_linear_probe': res.get('mlp_probe'),
            'A2_random_init_vs_trained': res.get('random_init'),
            'A3_sublayer_decomposition': ml.get('sublayer_decomposition')
                if isinstance(ml.get('sublayer_decomposition'), list) else [],
            'A5_crossfitted_representation_controls': res.get('crossfit_controls'),
            'A6_surface_form_baselines': res.get('surface_baselines'),
            'B1_B3_matching_balance': [dict(r, n_rejected_by_caliper=cm.get('n_rejected_by_caliper'),
                                            caliper_sd=cm.get('caliper_sd'))
                                       for r in cm.get('balance_diagnostics', [])],
            'B4_covariate_erased_probe': res.get('covariate_erased'),
            'B5_B6_words_only_frequency_models': fc.get('words_only_frequency'),
            'B5_incremental_lexicality_beyond_covariates': fc.get('incremental_lexicality'),
            'C1_probe_space_dose_response': res.get('intervention_dose_response'),
            'C2_residual_stream_steering': (ml.get('steering') or {}).get('steering')
                if isinstance(ml.get('steering'), dict) else [],
            'C5_subspace_mean_ablation': (ml.get('steering') or {}).get('subspace_ablation')
                if isinstance(ml.get('steering'), dict) else [],
            'C3_head_attribution_scores': (ml.get('head_ablation') or {}).get('head_scores')
                if isinstance(ml.get('head_ablation'), dict) else [],
            'C3_head_ablation': (ml.get('head_ablation') or {}).get('head_ablation')
                if isinstance(ml.get('head_ablation'), dict) else [],
            'D1_attention_mass': (ml.get('attention_mass') or {}).get('long')
                if isinstance(ml.get('attention_mass'), dict) else [],
            'D1_attention_mass_tests': (ml.get('attention_mass') or {}).get('tests')
                if isinstance(ml.get('attention_mass'), dict) else [],
            'D2_attention_knockout': ml.get('attention_knockout')
                if isinstance(ml.get('attention_knockout'), list) else [],
            'D3_stimulus_position_sensitivity': ml.get('position_sensitivity')
                if isinstance(ml.get('position_sensitivity'), list) else [],
            'D4_tokenization_stratified_primary': res.get('tokenization_stratified'),
            'F_contextual_conditions': ctx.get('conditions'),
        }
        for key, rows in blocks.items():
            self._naacl_rows.setdefault(key, [])
            self._naacl_rows[key] = [r for r in self._naacl_rows[key]
                                     if r.get('model') != model_name]
            self._naacl_rows[key] += [dict(r, model=model_name) for r in self._csv_safe(rows)]
        errors = {k: v['error'] for k, v in ml.items() if isinstance(v, dict) and 'error' in v}
        if errors:
            self._naacl_rows.setdefault('model_level_errors', [])
            self._naacl_rows['model_level_errors'] = [
                r for r in self._naacl_rows['model_level_errors'] if r.get('model') != model_name
            ] + [{'model': model_name, 'analysis': k, 'error': v} for k, v in errors.items()]

    def _write_naacl_outputs(self):
        for key, rows in self._naacl_rows.items():
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            if 'model' in df.columns:
                df = df[['model'] + [c for c in df.columns if c != 'model']]
            df.to_csv(self._naacl_path(f'{key}.csv'), index=False)
        tr = self._csv_safe(self.cross_model_transfer_rows)
        if tr:
            pd.DataFrame(tr).to_csv(self._naacl_path('C4_H3_H4_transfer_with_nulls.csv'),
                                    index=False)

    def _savefig(self, fig, name):
        fig.savefig(self._naacl_path('figures', name), bbox_inches='tight', dpi=250)
        plt.close(fig)

    def _plot_naacl_model(self, model_name, res):
        safe = model_name.replace(' ', '_').replace('/', '_')
        ct = pd.DataFrame(res.get('crossfit_table') or [])
        # I1 — frequency effect with bootstrap CIs (primary cross-fitted probe)
        try:
            if not ct.empty and 'freq_accuracy_difference_ci95_low' in ct:
                ct = ct.sort_values('layer')
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 6))
                sig = ct.get('freq_p_value_corrected', pd.Series(np.nan, index=ct.index)) < self.config.ALPHA
                err = np.vstack([ct['freq_accuracy_difference'] - ct['freq_accuracy_difference_ci95_low'],
                                 ct['freq_accuracy_difference_ci95_high'] - ct['freq_accuracy_difference']])
                a1.bar(ct['layer'], ct['freq_accuracy_difference'],
                       color=np.where(sig, self.DIFF_SIG, self.DIFF_NS), edgecolor='black', lw=0.5)
                a1.errorbar(ct['layer'], ct['freq_accuracy_difference'], yerr=np.clip(err, 0, None),
                            fmt='none', ecolor='black', capsize=2)
                a1.axhline(0, color='black', lw=1)
                a1.set(xlabel='Layer', ylabel='HF − LF word accuracy (95% bootstrap CI)',
                       title=f'I1 — frequency effect, cross-fitted probe — {model_name}\n'
                             '(green: BH-FDR p < .05)')
                a2.plot(ct['layer'], ct['freq_cohens_h'], 'o-', color='#5C4B8A')
                a2.fill_between(ct['layer'], ct['freq_cohens_h_ci95_low'], ct['freq_cohens_h_ci95_high'],
                                color='#5C4B8A', alpha=0.2)
                a2.axhline(0, color='black', lw=1)
                a2.set(xlabel='Layer', ylabel="Cohen's h (95% bootstrap CI)",
                       title="Effect size across layers")
                for ax in (a1, a2):
                    ax.grid(True, alpha=0.3)
                self._savefig(fig, f'{safe}_I1_frequency_effect_bootstrap_ci.png')
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"[I1] {model_name}: {e}")
        # I3 — Love plot
        try:
            bal = pd.DataFrame((res.get('confound_matched') or {}).get('balance_diagnostics', []))
            if not bal.empty:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                for stage, mk, col in (('before', 'o', '#D62246'), ('after', 's', '#1A535C')):
                    b = bal[bal['stage'] == stage]
                    ax.scatter(b['smd'].abs(), b['covariate'], marker=mk, s=70, color=col, label=stage)
                ax.axvline(self.config.SMD_BALANCE_THRESHOLD, color='gray', ls='--',
                           label=f'|SMD| = {self.config.SMD_BALANCE_THRESHOLD}')
                ax.set(xlabel='|standardised mean difference| (pre-matching SD)',
                       title=f'I3 — Love plot, HF vs LF words — {model_name}')
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
                self._savefig(fig, f'{safe}_I3_love_plot.png')
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"[I3] {model_name}: {e}")
        # I4 — dose-response (probe space C1, model level C2)
        try:
            dr = pd.DataFrame(res.get('intervention_dose_response') or [])
            ml = res.get('model_level') or {}
            st = pd.DataFrame((ml.get('steering') or {}).get('steering', [])
                              if isinstance(ml.get('steering'), dict) else [])
            if not dr.empty or not st.empty:
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 6))
                if not dr.empty:
                    for li, g in dr.groupby('layer'):
                        g = g.sort_values('magnitude_sd_units')
                        a1.plot(g['magnitude_sd_units'], g['delta_freq_gap_pos'], 'o-', label=f'L{li} (+)')
                    g0 = dr.groupby('magnitude_sd_units')['null_q95_abs_delta_freq_gap'].mean()
                    a1.fill_between(g0.index, -g0.values, g0.values, color='gray', alpha=0.2,
                                    label='random-direction 95% band')
                    a1.set_xscale('log')
                    a1.set(xlabel='Perturbation (SD of projection)', ylabel='Δ HF−LF accuracy gap',
                           title='C1 — probe-space dose-response')
                    a1.legend(fontsize=7, ncol=2)
                if not st.empty:
                    for (li, d), g in st.groupby(['layer', 'direction']):
                        g = g.sort_values('alpha_sd_units')
                        a2.plot(g['alpha_sd_units'], g['mean_delta_margin_all'],
                                'o-' if d == 'lexicality' else 's--', label=f'L{li} {d}')
                    gr = st.groupby('alpha_sd_units')['random_direction_mean_delta'].mean()
                    a2.plot(gr.index, gr.values, 'k:', lw=2, label='random directions')
                    a2.set(xlabel='α (SD of projection)', ylabel='Δ logit(YES) − logit(NO)',
                           title='C2 — residual-stream steering (model output)')
                    a2.legend(fontsize=7, ncol=2)
                for ax in (a1, a2):
                    ax.axhline(0, color='black', lw=1); ax.grid(True, alpha=0.3)
                fig.suptitle(f'I4 — dose-response — {model_name}')
                self._savefig(fig, f'{safe}_I4_dose_response.png')
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"[I4] {model_name}: {e}")
        # I6 — attention heatmap (layer × segment)
        try:
            am = pd.DataFrame((res.get('model_level') or {}).get('attention_mass', {}).get('long', [])
                              if isinstance((res.get('model_level') or {}).get('attention_mass'), dict) else [])
            if not am.empty:
                segs = ['bos', 'prefix', 'stimulus', 'suffix', 'final_token']
                piv = lambda grp: am[(am['group'] == grp) & am['segment'].isin(segs)].pivot(
                    index='layer', columns='segment', values='mean_attention')[segs]
                pw, pn = piv('word'), piv('nonword')
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, max(5, 0.25 * len(pw))))
                sns.heatmap(pw, ax=a1, cmap='viridis', cbar_kws={'label': 'attention mass'})
                sns.heatmap(pw - pn, ax=a2, cmap='RdBu_r', center=0,
                            cbar_kws={'label': 'words − nonwords'})
                a1.set_title('I6 — attention from first-output position (words)')
                a2.set_title('words − nonwords')
                for ax in (a1, a2):
                    ax.invert_yaxis()
                self._savefig(fig, f'{safe}_I6_attention_heatmap.png')
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"[I6] {model_name}: {e}")
        # A1 / A2 / A5 / B4 / A6 — control curves against the primary
        try:
            fig, ax = plt.subplots(figsize=(14, 6.5))
            if not ct.empty:
                ax.plot(ct['layer'], ct['accuracy'], 'o-', lw=2.6, color='#1A535C',
                        label='PRIMARY: linear probe (trained model)')
            for key, st_, lab in (('mlp_probe', 's--', 'A1: MLP probe'),
                                  ('random_init', '^-.', 'A2: random-init model'),
                                  ('covariate_erased', 'd:', 'B4: covariates linearly erased'),
                                  ('crossfit_controls', 'v--', 'A5: mean-pooled prompt')):
                t = pd.DataFrame(res.get(key) or [])
                if not t.empty and 'accuracy' in t:
                    t = t.sort_values('layer')
                    ax.plot(t['layer'], t['accuracy'], st_, lw=2, label=lab)
            sb = pd.DataFrame(res.get('surface_baselines') or [])
            if not sb.empty:
                for r in sb[sb['row_type'] == 'baseline_summary'].itertuples():
                    ax.axhline(r.accuracy, ls=':', lw=1.3, label=f'A6: {r.baseline} (no LLM)')
            ax.axhline(0.5, color='gray', ls=':', lw=1)
            ax.set(xlabel='Transformer layer', ylabel='Out-of-fold WORD/NONWORD accuracy',
                   title=f'Paired controls vs the primary probe (same items, same folds) — {model_name}')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            self._savefig(fig, f'{safe}_controls_vs_primary.png')
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"[controls figure] {model_name}: {e}")

    def _cross_model_naacl(self, all_results):
        """G4, H1, H2, I2, I5 + CKA."""
        cfg = self.config
        models = list(all_results)
        # G4 — global BH-FDR across every (model, layer) frequency test
        rows = []
        for m in models:
            for r in all_results[m].get('crossfit_table') or []:
                rows.append({'model': m, 'layer': r['layer'],
                             'relative_depth': r.get('relative_depth'),
                             'accuracy': r.get('accuracy'),
                             'balanced_accuracy': r.get('balanced_accuracy'),
                             'freq_accuracy_difference': r.get('freq_accuracy_difference'),
                             'freq_cohens_h': r.get('freq_cohens_h'),
                             'freq_cohens_h_ci95_low': r.get('freq_cohens_h_ci95_low'),
                             'freq_cohens_h_ci95_high': r.get('freq_cohens_h_ci95_high'),
                             'freq_p_value': r.get('freq_p_value'),
                             'freq_p_value_log10': r.get('freq_p_value_log10'),
                             'freq_p_value_auc': r.get('freq_p_value_auc'),
                             'p_vs_chance_binomial': r.get('p_vs_chance_binomial')})
        g = pd.DataFrame(rows)
        if not g.empty:
            for col in ('freq_p_value', 'freq_p_value_auc', 'p_vs_chance_binomial'):
                g[col + '_fdr_bh_global'] = adjust_pvalues(g[col].to_numpy(float))
            g.to_csv(self._naacl_path('G4_global_fdr_primary.csv'), index=False)
        # H1 — scaling regression
        sc = []
        for m in models:
            b = all_results[m].get('best_lexicality_layer') or {}
            n_p = all_results[m].get('n_parameters')
            fam = re.split(r'[-_.]', m.lower())[0]
            if b and n_p:
                sc.append({'model': m, 'family': fam, 'n_parameters': n_p,
                           'log10_params': np.log10(n_p),
                           'best_balanced_accuracy': b.get('best_lexicality_balanced_accuracy'),
                           'best_auroc': b.get('best_lexicality_auroc'),
                           'best_relative_depth': b.get('best_lexicality_relative_depth')})
        scd = pd.DataFrame(sc)
        if len(scd) >= 3:
            out = []
            for metric in ('best_balanced_accuracy', 'best_auroc', 'best_relative_depth'):
                d = scd.dropna(subset=[metric])
                if len(d) < 3:
                    continue
                r = ols_hc3(d[metric], d[['log10_params']], 'log10_params')
                rho, p_s = stats.spearmanr(d['log10_params'], d[metric])
                rng = np.random.default_rng(SEED)
                boots = []
                for _ in range(2000):
                    ix = rng.integers(0, len(d), len(d))
                    x_, y_ = d['log10_params'].to_numpy()[ix], d[metric].to_numpy()[ix]
                    if np.ptp(x_) > 0:
                        boots.append(np.polyfit(x_, y_, 1)[0])
                row = {'metric': metric, 'n_models': len(d), 'slope_per_decade': r['beta'],
                       'slope_se_hc3': r['se_hc3'], 'slope_p_hc3': r['p_hc3'], 'r2': r['r2'],
                       'slope_bootstrap_ci95_low': float(np.percentile(boots, 2.5)) if boots else np.nan,
                       'slope_bootstrap_ci95_high': float(np.percentile(boots, 97.5)) if boots else np.nan,
                       'spearman_rho': float(rho), 'spearman_p': float(p_s),
                       'log10_bf01': r['log10_bf01'],
                       'caveat': ('n models is small and size is confounded with family, data '
                                  'and training tokens; descriptive, not a scaling law')}
                fams = d['family'].value_counts()
                if (fams >= 2).sum() >= 2:
                    try:
                        Xf = pd.get_dummies(d['family'], drop_first=True, dtype=float)
                        Xf['log10_params'] = d['log10_params']
                        rf = ols_hc3(d[metric], Xf, 'log10_params')
                        row.update({'slope_within_family': rf['beta'], 'slope_within_family_p_hc3': rf['p_hc3']})
                    except Exception:
                        pass
                out.append(row)
            pd.DataFrame(out).to_csv(self._naacl_path('H1_scaling_regression.csv'), index=False)
            scd.to_csv(self._naacl_path('H1_scaling_points.csv'), index=False)
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                for fam, d in scd.groupby('family'):
                    ax.scatter(d['n_parameters'], d['best_balanced_accuracy'], s=70, label=fam)
                    for r in d.itertuples():
                        ax.annotate(r.model, (r.n_parameters, r.best_balanced_accuracy), fontsize=7)
                x = np.linspace(scd['log10_params'].min(), scd['log10_params'].max(), 50)
                k, b0 = np.polyfit(scd['log10_params'], scd['best_balanced_accuracy'], 1)
                ax.plot(10 ** x, k * x + b0, 'k--', lw=1)
                ax.set_xscale('log')
                ax.set(xlabel='parameters', ylabel='best out-of-fold balanced accuracy',
                       title='H1 — best lexicality vs model size (descriptive)')
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
                self._savefig(fig, 'H1_scaling.png')
            except Exception as e:                           # noqa: BLE001
                logger.warning(f"[H1] figure: {e}")
        # H2 — alignment of normalised-depth curves (level and first differences)
        grid = np.linspace(0, 1, cfg.CURVE_ALIGNMENT_GRID)
        curves = {}
        for m in models:
            t = pd.DataFrame(all_results[m].get('crossfit_table') or [])
            if t.empty or 'accuracy' not in t or t['accuracy'].isna().all():
                continue
            t = t.dropna(subset=['accuracy']).sort_values('layer')
            L = all_results[m]['num_layers']
            x = t['layer'].to_numpy() / max(L - 1, 1)
            curves[m] = np.interp(grid, x, t['accuracy'].to_numpy())
        al = []
        for i, a in enumerate(curves):
            for b in list(curves)[i + 1:]:
                ca, cb = curves[a], curves[b]
                r_lvl = float(np.corrcoef(ca, cb)[0, 1]) if np.std(ca) > 0 and np.std(cb) > 0 else np.nan
                da, db = np.diff(ca), np.diff(cb)
                r_dif = float(np.corrcoef(da, db)[0, 1]) if np.std(da) > 0 and np.std(db) > 0 else np.nan
                al.append({'model_a': a, 'model_b': b, 'pearson_r_curves': r_lvl,
                           'pearson_r_first_differences': r_dif,
                           'rmse': float(np.sqrt(np.mean((ca - cb) ** 2))),
                           'note': ('level correlations of monotone curves are inflated by '
                                    'the shared trend; first differences remove it')})
        if al:
            ald = pd.DataFrame(al)
            ald.to_csv(self._naacl_path('H2_layerwise_curve_alignment.csv'), index=False)
            try:
                names = list(curves)
                M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
                for r in al:
                    M.loc[r['model_a'], r['model_b']] = M.loc[r['model_b'], r['model_a']] = \
                        r['pearson_r_first_differences']
                fig, ax = plt.subplots(figsize=(1.1 * len(names) + 4, 1.0 * len(names) + 3))
                sns.heatmap(M, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
                ax.set_title('H2 — alignment of detrended depth curves (r of first differences)')
                self._savefig(fig, 'H2_curve_alignment_heatmap.png')
            except Exception as e:                           # noqa: BLE001
                logger.warning(f"[H2] figure: {e}")
        # H2b — linear CKA between models (layer × layer)
        if cfg.RUN_CKA and len(self._cka) > 1:
            ck, summ = [], []
            names = list(self._cka)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    A, B = self._cka[a], self._cka[b]
                    common = sorted(set(A['stimuli']) & set(B['stimuli']))
                    if len(common) < 50:
                        continue
                    ia = np.array([A['stimuli'].index(s) for s in common])
                    ib = np.array([B['stimuli'].index(s) for s in common])
                    mat = np.full((A['num_layers'], B['num_layers']), np.nan)
                    for la, Ka in A['grams'].items():
                        Kc = Ka[np.ix_(ia, ia)].astype(np.float64)
                        Kc = Kc - Kc.mean(0) - Kc.mean(1)[:, None] + Kc.mean()
                        for lb, Kb in B['grams'].items():
                            Lc = Kb[np.ix_(ib, ib)].astype(np.float64)
                            Lc = Lc - Lc.mean(0) - Lc.mean(1)[:, None] + Lc.mean()
                            mat[la, lb] = linear_cka_from_grams(Kc, Lc)
                            ck.append({'model_a': a, 'layer_a': la, 'model_b': b, 'layer_b': lb,
                                       'relative_depth_a': (la + 1) / A['num_layers'],
                                       'relative_depth_b': (lb + 1) / B['num_layers'],
                                       'linear_cka': mat[la, lb], 'n_items': len(common)})
                    diag = [mat[la, int(round((la + 1) / A['num_layers'] * B['num_layers'])) - 1]
                            for la in range(A['num_layers'])]
                    summ.append({'model_a': a, 'model_b': b,
                                 'mean_cka_matched_depth': float(np.nanmean(diag)),
                                 'max_cka': float(np.nanmax(mat))})
            if ck:
                pd.DataFrame(ck).to_csv(self._naacl_path('H2b_linear_cka_layer_pairs.csv'), index=False)
                sd = pd.DataFrame(summ)
                sd.to_csv(self._naacl_path('H2b_linear_cka_summary.csv'), index=False)
                try:
                    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
                    for r in summ:
                        M.loc[r['model_a'], r['model_b']] = M.loc[r['model_b'], r['model_a']] = \
                            r['mean_cka_matched_depth']
                    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 4, 1.0 * len(names) + 3))
                    sns.heatmap(M, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1, ax=ax)
                    ax.set_title('H2b — linear CKA at matched relative depth')
                    self._savefig(fig, 'H2b_cka_heatmap.png')
                except Exception as e:                       # noqa: BLE001
                    logger.warning(f"[H2b] figure: {e}")
        # I2 — forest plot of Cohen's h at each model's best lexicality layer
        try:
            fr = []
            for m in models:
                b = (all_results[m].get('best_lexicality_layer') or {}).get('best_lexicality_layer')
                row = next((r for r in all_results[m].get('crossfit_table') or []
                            if r['layer'] == b), None)
                if row is not None and _finite(row.get('freq_cohens_h')):
                    fr.append((m, b, row['freq_cohens_h'], row['freq_cohens_h_ci95_low'],
                               row['freq_cohens_h_ci95_high']))
            if fr:
                fig, ax = plt.subplots(figsize=(8, 0.55 * len(fr) + 2))
                for i, (m, b, h, lo, hi) in enumerate(fr):
                    ax.errorbar(h, i, xerr=[[h - lo], [hi - h]], fmt='o', color='#1A535C', capsize=4)
                ax.set_yticks(range(len(fr)))
                ax.set_yticklabels([f'{m} (L{b})' for m, b, *_ in fr])
                ax.axvline(0, color='black', lw=1)
                for v in (0.2, 0.5, 0.8):
                    ax.axvline(v, color='gray', ls=':', lw=0.8)
                ax.set(xlabel="Cohen's h, HF − LF word accuracy (95% bootstrap CI)",
                       title='I2 — frequency effect at the best lexicality layer')
                ax.grid(True, alpha=0.3, axis='x')
                self._savefig(fig, 'I2_forest_cohens_h.png')
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"[I2] {e}")
        # I5 — permutation null of the best transfer pair
        try:
            cand = [r for r in self.cross_model_transfer_rows if r.get('_perm_null_accuracies')]
            if cand:
                best = max(cand, key=lambda r: r.get('closed_form_transfer_accuracy', -1))
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(best['_perm_null_accuracies'], bins=40, color='#AAAAAA',
                        label='shuffled-pairing null')
                ax.axvline(best['closed_form_transfer_accuracy'], color='#D62246', lw=2,
                           label='true pairing')
                ax.set(xlabel='transfer accuracy (HF vs LF)',
                       title=f"I5 — {best['source_model']} L{best['source_layer']} → "
                             f"{best['target_model']} L{best['target_layer']}\n"
                             f"perm p = {best.get('perm_p_value_accuracy', np.nan):.4f}; "
                             f"random-probe p = {best.get('random_probe_p_value_auc', np.nan):.4f}")
                ax.legend()
                self._savefig(fig, 'I5_transfer_permutation_null.png')
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"[I5] {e}")

    def _write_experiment_metadata(self):
        meta = {
            # ── primary definition ──────────────────────────────────────
            'primary_task': 'WORD_vs_NONWORD',
            'representation_type': PRIMARY_REPRESENTATION,
            'representation_definition': (
                'h_l = hidden_states[l + 1][b, final_prompt_position_b, :], where '
                'final_prompt_position_b is the last real prompt token from the '
                'attention mask — the position whose hidden state the causal LM uses '
                'to predict the first output token. Not a mean, not a pooled state, '
                'not a generated token, not an arbitrary position.'),
            'readout_type': PRIMARY_READOUT,
            'readout_definition': (
                # FIX: the earlier text described an nn.Linear probe with early
                # stopping; the primary probe is (and was) an sklearn L2 logistic
                # regression without early stopping.
                f'{self.config.PRIMARY_PROBE_N_FOLDS}-fold cross-fitted, independent '
                f'L2-regularised logistic regressions per layer (sklearn, solver='
                f'{self.config.PRIMARY_PROBE_SOLVER}, C={self.config.PRIMARY_PROBE_C}, '
                f'features standardised on the training fold); every stimulus is '
                f'scored exactly once by the fold probe that did not train on it. '
                f'lexical_evidence = logit(P(WORD)) with P clipped to '
                f'[{self.config.LEXICAL_EVIDENCE_CLIP}, '
                f'{1 - self.config.LEXICAL_EVIDENCE_CLIP}].'),
            'methodology_sentence': (
                'For each LDT stimulus, we extract the hidden state at the final prompt '
                'position — the position used to predict the first output token — from '
                'every Transformer layer, and use cross-fitted linear probes to measure '
                'layer-wise WORD/NONWORD lexical information without modifying the '
                'frozen language model.'),
            'claims_policy': (
                'Report: "we probe the layer-wise hidden representation at the '
                'first-output prediction position for lexicality". Do NOT write "the '
                'LLM predicts WORD/NONWORD at each layer" — the probe is an analysis '
                'tool, not the model\'s output. Layer depth is not a reaction time.'),
            'layerwise_forward_pass': True,
            'model_receives_prompt_only': True,
            'generated_token_used_for_primary_representation': False,
            'teacher_forcing': False,
            'ground_truth_used_for_prediction': False,
            'frequency_used_in_prompt': False,
            'human_rt_used_in_probe_fitting': False,
            'human_rt_used_for_hyperparameter_selection': False,
            'transformer_frozen': True,
            'lm_head_trained': False,
            'trained_parameters': 'the linear probes only',
            'extraction_under_no_grad': True,
            'prompt_template': TASK_PROMPT_TEMPLATE,
            'prompt_optimised': False,
            'tokenizer_dependent_scoring': False,
            'yes_no_tokens_required': False,
            # ── layer indexing (§10) ────────────────────────────────────
            'layer_index_convention': (
                'Reported layer li corresponds to outputs.hidden_states[li + 1]. '
                'hidden_states[0] is the embedding output (before the first '
                'Transformer block) and is NOT reported; reported layer 0 is the '
                'output of the FIRST Transformer block and reported layer L-1 the '
                'output of the last. This is the convention the earlier versions of '
                'this script used and it is unchanged. relative_depth = (li + 1) / L.'),
            'n_hidden_state_tensors_expected': 'num_hidden_layers + 1 (verified per model)',
            # ── cross-fitting / leakage guarantees (§11) ────────────────
            'crossfitting': {
                'n_folds': self.config.PRIMARY_PROBE_N_FOLDS,
                'seed': self.config.PRIMARY_PROBE_SEED,
                'shuffle': self.config.PRIMARY_PROBE_CV_SHUFFLE,
                'splitter': 'StratifiedKFold on the WORD/NONWORD label only — '
                            'deterministic and identical across layers and models, '
                            'so layer comparisons are paired',
                'scaler_fitted_on': 'training fold only',
                'early_stopping': ('none for the primary logistic probe (convex, '
                                   'fitted to convergence); the A1 MLP probe early-stops '
                                   'on a split taken from inside the training fold only'),
                'scores_reported': 'held-out (out-of-fold) only',
                'each_stimulus_scored_once': True,
                'rt_analysis_reuses_primary_oof_scores': True,
            },
            'evaluation_set': ('all extracted items — every item has an out-of-fold '
                               'score, so no separate test split is needed'),
            'statistics': {
                'accuracy_ci': f'percentile bootstrap over items, B={self.config.N_BOOTSTRAP}',
                'vs_chance': 'one-sided exact binomial vs majority-class rate, BH-FDR across layers',
                'consecutive_layers': 'exact McNemar on paired items, Holm across layers',
                'best_lexicality_layer': 'max out-of-fold balanced accuracy (ties → shallowest)',
                'best_rt_alignment_layer': 'max |Spearman rho| among FDR-significant layers',
            },
            'cross_model_comparability': (
                'Hidden dimensionality differs across models; each model gets its own '
                'Linear(hidden_dim, 2). The invariant is the representation LOCATION: '
                'the final prompt position used to predict the first output token.'),
            # ── secondary / optional components ─────────────────────────
            'secondary_analyses': {
                'holdout_linear_probe': {
                    'probe_type': probe_type_name(self.config.CLASSIFIER_ARCHITECTURE),
                    'split': (f'stratified train/val/test = '
                              f'{1 - self.config.VAL_SIZE - self.config.TEST_SIZE:.2f}/'
                              f'{self.config.VAL_SIZE:.2f}/{self.config.TEST_SIZE:.2f}, '
                              f'seed {SEED}'),
                    'role': 'unchanged basis of extended analyses 1-10',
                },
                'native_lm_head_yes_no_readout': {
                    'enabled': bool(self.config.ENABLE_NATIVE_LM_HEAD_DIAGNOSTIC),
                    'role': 'optional diagnostic only, never the primary result',
                    'why_off_by_default': ('depends on tokenizer-specific YES/NO tokens '
                                           'and vocabulary-dependent scoring, which '
                                           'breaks clean cross-model comparison'),
                },
                'native_one_token_generation': {
                    'enabled': bool(self.config.RUN_NATIVE_GENERATION_DIAGNOSTIC),
                    'generation_strategy': 'greedy', 'max_new_tokens': 1,
                    'role': 'behavioural diagnostic; also required by the '
                            'generated-token control representation',
                },
            },
            'representations': {
                PRIMARY_REPRESENTATION:
                    'final real PROMPT position — the state used to predict the first '
                    'output token (PRIMARY)',
                RepresentationType.MEAN_PROMPT_TOKENS.value:
                    'mean hidden state over prompt tokens (SECONDARY CONTROL)',
                RepresentationType.GENERATED_TOKEN_CONTROL.value:
                    'hidden state of the token the model actually generated '
                    '(SECONDARY CONTROL; requires the generation diagnostic)',
                RepresentationType.CONTEXTUAL_TARGET_TOKEN.value:
                    'target word at its sentence position (Analysis 10 CONTROL)',
            },
            'models': self.model_metadata,
            'config': {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                       for k, v in vars(self.config).items() if k != 'MODELS'},
            # ── J9 / J10 ───────────────────────────────────────────────
            'seeds': self.seeds,
            'environment': _environment_snapshot(),
            # ── K1–K6 ──────────────────────────────────────────────────
            'claim_boundary': CLAIM_BOUNDARY,
            'multiple_comparisons_procedure': MULTIPLE_COMPARISONS_PROCEDURE,
            'analysis_plan': ANALYSIS_PLAN,
        }
        with open(os.path.join(self.config.NAACL_DIR, 'analysis_plan.json'), 'w') as f:
            json.dump({'claim_boundary': CLAIM_BOUNDARY,
                       'multiple_comparisons_procedure': MULTIPLE_COMPARISONS_PROCEDURE,
                       'analysis_plan': ANALYSIS_PLAN}, f, indent=2)
        for d in (self.config.OUTPUT_DIR, self.config.PRIMARY_DIR):
            with open(os.path.join(d, 'experiment_metadata.json'), 'w') as f:
                json.dump(meta, f, indent=2, default=str)

    def _apply_fdr(self, lr):
        apply_frequency_fdr(lr, self.config.ALPHA)
    def _apply_fdr_with_class_ratio(self, lr, class_ratio):
        """
        Apply FDR correction with class ratio awareness.
        """
        pv, ix = [], []
        for i, r in enumerate(lr):
            p = r['frequency_effect']['p_value']
            if not np.isnan(p):
                pv.append(p)
                ix.append(i)
        if not pv:
            return
        
        rej, pc, _, _ = multipletests(pv, alpha=self.config.ALPHA, method='fdr_bh')
        for i, p, s in zip(ix, pc, rej):
            lr[i]['frequency_effect']['p_value_corrected'] = p
            lr[i]['frequency_effect']['significant_fdr'] = bool(s)
            # NEW: Add class ratio info
            lr[i]['frequency_effect']['class_ratio'] = class_ratio

    # ── metric extraction helper ──────────────────────────────────────────
    @staticmethod
    def _series(lr, group_key, metric):
        out = []
        for r in lr:
            g = r.get(group_key)
            out.append(g[metric] if g else np.nan)
        return np.array(out, dtype=float)

    @staticmethod
    def _L(lr): return [r['layer'] for r in lr]

    # ════════════════════════════════════════════════════════════════════════
    # PER-MODEL PLOTS (original)
    # ════════════════════════════════════════════════════════════════════════
    def _plot_class_ratio_analysis(self, model_name, selectivity_results, safe):
        """
        Plot class ratio analysis alongside selectivity results.
        """
        if not selectivity_results.get('class_ratio_analysis'):
            return
        
        df_cr = pd.DataFrame(selectivity_results['class_ratio_analysis'])
        df_shuf = pd.DataFrame(selectivity_results.get('shuffled_label', []))
        
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Panel A: Class ratios across layers
        ax1.plot(df_cr['layer'], df_cr['train_ratio'], 'o-', 
                lw=2, color='#2E86AB', label='Train')
        ax1.plot(df_cr['layer'], df_cr['val_ratio'], 's--', 
                lw=2, color='#D62246', label='Validation')
        ax1.plot(df_cr['layer'], df_cr['test_ratio'], '^-', 
                lw=2, color='#F18F01', label='Test')
        ax1.axhline(0.5, color='gray', ls=':', lw=1.2, label='Balanced (0.5)')
        ax1.set(xlabel='Layer', ylabel='Class Ratio (words/total)',
                title=f'Class Ratio Across Splits — {model_name}')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        # Panel B: Shuffled accuracy vs expected baseline
        if not df_shuf.empty and 'expected_baseline' in df_shuf.columns:
            ax2.plot(df_shuf['layer'], df_shuf['accuracy'], 'o-', 
                    lw=2.5, color='#E63946', label='Shuffled Accuracy')
            ax2.plot(df_shuf['layer'], df_shuf['expected_baseline'], 's--', 
                    lw=2.5, color='#264653', label='Expected Majority Baseline')
            ax2.axhline(0.5, color='gray', ls=':', lw=1.2, label='Chance (0.5)')
            ax2.set(xlabel='Layer', ylabel='Accuracy',
                    title=f'Shuffled vs Expected Baseline — {model_name}')
            ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                f'{safe}_class_ratio_analysis.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info(f"✓ {safe}_class_ratio_analysis.png")
    
    def _plot_model_all(self, model_name, lr):
        safe = model_name.replace(' ','_').replace('/','_')
        L    = self._L(lr)

        groups = {
            'overall':       'overall',
            'high_frequency':'high_frequency',
            'low_frequency': 'low_frequency',
        }
        metrics = ['accuracy','precision','recall','f1','auc']

        data = {g: {m: self._series(lr,g,m) for m in metrics} for g in groups}
        
        # Extract frequency effect data
        diff_acc = np.array([r['frequency_effect']['accuracy_difference'] for r in lr])
        diff_auc = np.array([r['frequency_effect'].get('auc_difference', np.nan) for r in lr])
        p_corr_acc = np.array([r['frequency_effect'].get('p_value_corrected', np.nan) for r in lr])
        # Since p_value_corrected_auc doesn't exist, use p_corr_acc or set to NaN
        p_corr_auc = np.array([np.nan for _ in lr])  # Or use p_corr_acc if you want
        coh_h = np.array([r['frequency_effect'].get('cohens_h', np.nan) for r in lr])

        ylabels = {'accuracy':'Accuracy','precision':'Precision',
                'recall':'Recall','f1':'F1-Score','auc':'AUC-ROC'}
        for m in metrics:
            ov = data['overall'][m]
            hf = data['high_frequency'][m]
            lf = data['low_frequency'][m]
            self._three_group_plot(
                L, ov, hf, lf,
                ylabel=ylabels[m],
                title=f'{ylabels[m]} — All Groups  [{model_name}]',
                fname=os.path.join(self.config.FIGURES_DIR,
                                f'{safe}_{m}_all_groups.png'))

        self._freq_effect_plot(L, diff_acc, diff_auc, p_corr_acc, p_corr_auc, coh_h, model_name, safe)
        self._all_metrics_grid(model_name, safe, L, data, diff_acc, p_corr_acc, coh_h)
    
    def _three_group_plot(self, L, ov, hf, lf, ylabel, title, fname):
        valid = [v for v in list(ov)+list(hf)+list(lf) if not np.isnan(v)]
        ymin  = max(0.0, min(valid, default=0.4) - 0.05) if valid else 0.4
        _fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(L, ov, 'o-',  lw=2.5, ms=7, color=self.OVERALL_C, label='Overall Test', zorder=3)
        ax.plot(L, hf, 's--', lw=2.5, ms=7, color=self.HIGH_C,    label='High-Freq',    zorder=3)
        ax.plot(L, lf, '^-.', lw=2.5, ms=7, color=self.LOW_C,     label='Low-Freq',     zorder=3)
        ax.fill_between(L, hf, lf, alpha=0.12, color='#F18F01', label='High−Low gap')
        ax.axhline(0.5, color='gray', ls=':', lw=1.2, alpha=0.6)
        ax.set(xlabel='Layer Index', ylabel=ylabel, title=title, ylim=[ymin, 1.01])
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info(f"✓ {os.path.basename(fname)}")

    def _freq_effect_plot(self, L, diff_acc, diff_auc, p_corr_acc, p_corr_auc, coh_h, model_name, safe):
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        
        # Use diff_acc and p_corr_acc (not undefined diff and p_corr)
        colors = [self.DIFF_SIG if (not np.isnan(p) and p<0.05) else self.DIFF_NS
                for p in p_corr_acc]
        ax1.bar(L, diff_acc, color=colors, alpha=0.75, edgecolor='black', lw=0.6)
        ax1.axhline(0, color='black', lw=1.5)
        
        # Use diff_acc and p_corr_acc
        for li, d, p in zip(L, diff_acc, p_corr_acc):
            if np.isnan(p): continue
            s = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
            if s:
                ax1.text(li, d, s, ha='center',
                        va='bottom' if d>0 else 'top', fontsize=10, fontweight='bold')
        ax1.set(xlabel='Layer', ylabel='Accuracy Diff (High − Low)',
                title=f'(A) Frequency Effect — {model_name}')
        ax1.grid(True, alpha=0.3, axis='y')
        from matplotlib.patches import Patch
        ax1.legend(handles=[Patch(fc=self.DIFF_SIG, label='p<0.05 FDR'),
                            Patch(fc=self.DIFF_NS,  label='n.s.')], fontsize=10)

        ax2.plot(L, coh_h, 'o-', lw=2.5, ms=7, color='#5C4B8A')
        for h, lbl, c in [(0.2,'Small','#e57373'),(0.5,'Medium','#ffa726'),(0.8,'Large','#66bb6a')]:
            ax2.axhline(h, color=c, ls='--', lw=1.4, alpha=0.7, label=f'{lbl} h={h}')
        ax2.axhline(0, color='black', lw=1.2)
        ax2.set(xlabel='Layer', ylabel="Cohen's h", ylim=[-0.05, None],
                title=f"(B) Effect Size — {model_name}")
        ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                f'{safe}_frequency_effect.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info(f"✓ {safe}_frequency_effect.png")

    def _all_metrics_grid(self, model_name, safe, L, data, diff, p_corr, coh_h):
        fig = plt.figure(figsize=(28, 14))
        gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.42, wspace=0.35)

        metric_info = [
            (gs[0,0], 'accuracy',  'Accuracy'),
            (gs[0,1], 'precision', 'Precision'),
            (gs[0,2], 'recall',    'Recall'),
            (gs[0,3], 'f1',        'F1-Score'),
            (gs[0,4], 'auc',       'AUC-ROC'),
        ]
        for gspec, m, ylabel in metric_info:
            ax = fig.add_subplot(gspec)
            ax.plot(L, data['overall'][m],        'o-',  lw=2, ms=5, color=self.OVERALL_C, label='Overall')
            ax.plot(L, data['high_frequency'][m],  's--', lw=2, ms=5, color=self.HIGH_C,    label='High-Freq')
            ax.plot(L, data['low_frequency'][m],   '^-.', lw=2, ms=5, color=self.LOW_C,     label='Low-Freq')
            ax.fill_between(L, data['high_frequency'][m], data['low_frequency'][m],
                            alpha=0.10, color='#F18F01')
            ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.5)
            ax.set(xlabel='Layer', ylabel=ylabel, title=ylabel)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

        ax_b = fig.add_subplot(gs[1, 0:2])
        colors = [self.DIFF_SIG if (not np.isnan(p) and p<0.05) else self.DIFF_NS for p in p_corr]
        ax_b.bar(L, diff, color=colors, alpha=0.75, edgecolor='black', lw=0.5)
        ax_b.axhline(0, color='black', lw=1.3)
        for li, d, p in zip(L, diff, p_corr):
            if np.isnan(p): continue
            s = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
            if s:
                ax_b.text(li, d, s, ha='center', va='bottom' if d>0 else 'top',
                          fontsize=9, fontweight='bold')
        ax_b.set(xlabel='Layer', ylabel='Acc Diff (High − Low)',
                 title='Frequency Effect (FDR corrected)')
        ax_b.grid(True, alpha=0.25, axis='y')
        from matplotlib.patches import Patch
        ax_b.legend(handles=[Patch(fc=self.DIFF_SIG, label='p<0.05'),
                              Patch(fc=self.DIFF_NS,  label='n.s.')], fontsize=9)

        ax_h = fig.add_subplot(gs[1, 2:4])
        ax_h.plot(L, coh_h, 'o-', lw=2, ms=5, color='#5C4B8A')
        for h, lbl, c in [(0.2,'Small','#e57373'),(0.5,'Medium','#ffa726'),(0.8,'Large','#66bb6a')]:
            ax_h.axhline(h, color=c, ls='--', lw=1.2, alpha=0.7, label=f'{lbl} h={h}')
        ax_h.axhline(0, color='black', lw=1.0)
        ax_h.set(xlabel='Layer', ylabel="Cohen's h", title="Effect Size (Cohen's h)")
        ax_h.legend(fontsize=8); ax_h.grid(True, alpha=0.25)

        ax_p = fig.add_subplot(gs[1, 4]); ax_p.axis('off')
        ax_p.text(0.5, 0.5, 'Probe: memory-only\n(no .pt written)',
                  ha='center', va='center', transform=ax_p.transAxes,
                  fontsize=11, color='gray', style='italic')

        fig.suptitle(f'Full Metric Suite — {model_name}',
                     fontsize=16, fontweight='bold', y=1.01)
        plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                 f'{safe}_all_metrics_grid.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info(f"✓ {safe}_all_metrics_grid.png")

    # ════════════════════════════════════════════════════════════════════════
    # EXTENDED ANALYSIS PLOTS
    # ════════════════════════════════════════════════════════════════════════

    def _plot_extended(self, model_name, res):
        safe = model_name.replace(' ', '_').replace('/', '_')

        # ── Analysis 1: Direct Frequency Probe ────────────────────────
        if res.get('direct_freq') and res['direct_freq']['binary']:
            df_bin = pd.DataFrame(res['direct_freq']['binary'])
            if not df_bin.empty and not df_bin['accuracy'].isna().all():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                ax1.plot(df_bin['layer'], df_bin['accuracy'], 'o-', lw=2.5,
                         ms=7, color='#E63946', label='Binary (H vs L)')
                ax1.axhline(0.5, color='gray', ls='--', lw=1.2, label='Chance')
                ax1.set(xlabel='Layer', ylabel='Accuracy',
                        title=f'Direct Frequency Probe (Binary) — {model_name}')
                ax1.legend(); ax1.grid(True, alpha=0.3)

                if 'f1' in df_bin.columns:
                    ax2.plot(df_bin['layer'], df_bin['f1'], 's-', lw=2.5,
                             ms=7, color='#457B9D', label='F1')
                if 'auc' in df_bin.columns:
                    ax2.plot(df_bin['layer'], df_bin['auc'], '^-', lw=2.5,
                             ms=7, color='#2A9D8F', label='AUC')
                ax2.axhline(0.5, color='gray', ls='--', lw=1.2)
                ax2.set(xlabel='Layer', ylabel='Score',
                        title=f'Direct Freq Probe — F1 & AUC — {model_name}')
                ax2.legend(); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_direct_freq_probe.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

            # 3-way
            df_3w = pd.DataFrame(res['direct_freq'].get('three_way', []))
            if not df_3w.empty and not df_3w['accuracy'].isna().all():
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df_3w['layer'], df_3w['accuracy'], 'D-', lw=2.5,
                        ms=7, color='#6A0572', label='3-way (H/M/L)')
                ax.axhline(1.0/3, color='gray', ls='--', lw=1.2, label='Chance (0.33)')
                ax.set(xlabel='Layer', ylabel='Accuracy',
                       title=f'3-Way Frequency Probe — {model_name}')
                ax.legend(); ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_3way_freq_probe.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

        # ── Analysis 5: Representation Ablation (primary vs controls) ─
        if res.get('representation_ablation') and res.get('layer_results'):
            lr_p = res['layer_results']
            L = self._L(lr_p)
            styles = {RepresentationType.MEAN_PROMPT_TOKENS.value: ('s--', '#D62246'),
                      RepresentationType.GENERATED_TOKEN_CONTROL.value: ('^-.', '#F18F01')}
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            ax1.plot(L, self._series(lr_p, 'overall', 'accuracy'), 'o-', lw=2.8,
                     color='#1A535C', label=f'{PRIMARY_REPRESENTATION} (PRIMARY)')
            ax2.plot(L, [r['frequency_effect']['accuracy_difference'] for r in lr_p],
                     'o-', lw=2.8, color='#1A535C', label=f'{PRIMARY_REPRESENTATION} (PRIMARY)')
            for rep_type, lr_c in res['representation_ablation'].items():
                st, col = styles.get(rep_type, ('d:', '#555555'))
                ax1.plot(L, self._series(lr_c, 'overall', 'accuracy'), st, lw=2,
                         color=col, label=f'{rep_type} (control)')
                ax2.plot(L, [r['frequency_effect']['accuracy_difference'] for r in lr_c],
                         st, lw=2, color=col, label=f'{rep_type} (control)')
            ax1.axhline(0.5, color='gray', ls=':', lw=1.2)
            ax1.set(xlabel='Transformer Layer', ylabel='WORD vs NONWORD Probe Accuracy',
                    title=f'Representation Ablation: LDT Accuracy — {model_name}')
            ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
            ax2.axhline(0, color='black', lw=1.2)
            ax2.set(xlabel='Transformer Layer', ylabel='Acc Diff (High − Low)',
                    title=f'Representation Ablation: Freq Effect — {model_name}')
            ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                     f'{safe}_representation_ablation.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()

        # ── Analysis 4: Stability (error bars) ────────────────────────
        if res.get('stability') and res['stability']['layer_stats']:
            df_s = pd.DataFrame(res['stability']['layer_stats'])
            if not df_s.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

                ax1.errorbar(df_s['layer'], df_s['acc_mean'], yerr=df_s['acc_std'],
                            fmt='o-', lw=2, ms=6, capsize=4, color='#1A535C',
                            label=f'Mean ± SD ({res["stability"]["n_seeds"]} seeds)')
                ax1.axhline(0.5, color='gray', ls='--', lw=1.2)
                ax1.set(xlabel='Layer', ylabel='Accuracy',
                        title=f'Multi-Seed Stability: Accuracy — {model_name}')
                ax1.legend(); ax1.grid(True, alpha=0.3)

                ax2.errorbar(df_s['layer'], df_s['freq_diff_mean'],
                            yerr=df_s['freq_diff_std'],
                            fmt='s-', lw=2, ms=6, capsize=4, color='#D62246',
                            label=f'Mean ± SD ({res["stability"]["n_seeds"]} seeds)')
                ax2.axhline(0, color='black', lw=1.2)
                ax2.set(xlabel='Layer', ylabel='Freq Effect (Acc Diff)',
                        title=f'Multi-Seed Stability: Freq Effect — {model_name}')
                ax2.legend(); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_stability.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

        # ── Analysis 6: Selectivity ───────────────────────────────────
        # ── Analysis 6: Selectivity ───────────────────────────────────
        if res.get('selectivity'):
            sel = res['selectivity']
            
            # Add this call to plot class ratio analysis
            self._plot_class_ratio_analysis(model_name, sel, safe)
            
            if sel.get('selectivity') and sel['selectivity']:
                df_sel = pd.DataFrame(sel['selectivity'])
                if not df_sel.empty:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                    ax1.plot(df_sel['layer'], df_sel['acc_task'], 'o-',
                            lw=2.5, color='#2E86AB', label='Task (real labels)')
                    ax1.plot(df_sel['layer'], df_sel['acc_shuffled'], 's--',
                            lw=2.5, color='#AAAAAA', label='Shuffled labels')
                    ax1.axhline(0.5, color='gray', ls=':', lw=1.2)
                    ax1.set(xlabel='Layer', ylabel='Accuracy',
                            title=f'Probe Selectivity — {model_name}')
                    ax1.legend(); ax1.grid(True, alpha=0.3)

                    ax2.bar(df_sel['layer'], df_sel['selectivity'],
                            color='#2DC653', alpha=0.8, edgecolor='black', lw=0.5)
                    ax2.axhline(0, color='black', lw=1.2)
                    ax2.set(xlabel='Layer', ylabel='Selectivity (Task − Shuffled)',
                            title=f'Selectivity Score — {model_name}')
                    ax2.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                            f'{safe}_selectivity.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close('all'); gc.collect()

            # sklearn LR (C=1) control vs the PRIMARY probe.
            # J4 FIX: the reference curve is the primary CROSS-FITTED probe
            # (crossfit_table), not the secondary hold-out probe it used before.
            if sel.get('linear_probe') and sel['linear_probe']:
                df_lp = pd.DataFrame(sel['linear_probe'])
                mlp_accs = {r['layer']: r['accuracy'] for r in res.get('crossfit_table', [])
                            if _finite(r.get('accuracy'))}

                if not df_lp.empty:
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(df_lp['layer'], df_lp['accuracy'], 's--',
                            lw=2.5, color='#E63946', label='sklearn LogisticRegression (C=1)')
                    mlp_layers = sorted(mlp_accs.keys())
                    mlp_vals = [mlp_accs[l] for l in mlp_layers]
                    ax.plot(mlp_layers, mlp_vals, 'o-', lw=2.5, color='#1A535C',
                            label='PRIMARY cross-fitted linear probe (out-of-fold)')
                    ax.axhline(0.5, color='gray', ls=':', lw=1.2)
                    ax.set(xlabel='Layer', ylabel='Accuracy',
                           title=f'sklearn LR vs Primary Probe — {model_name}')
                    ax.legend(); ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                             f'{safe}_linear_vs_primary_probe.png'),
                                bbox_inches='tight', dpi=300)
                    plt.close('all'); gc.collect()

        # ── Analysis 7: Continuous frequency regression ───────────────
        if res.get('regression') and res['regression'].get('regression_results'):
            df_reg = pd.DataFrame(res['regression']['regression_results'])
            if not df_reg.empty and not df_reg['r2'].isna().all():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

                ax1.plot(df_reg['layer'], df_reg['r2'], 'o-', lw=2.5,
                         ms=7, color='#6A0572', label='R²')
                ax1.axhline(0, color='gray', ls='--', lw=1.2, label='Baseline (mean)')
                ax1.set(xlabel='Layer', ylabel='R²',
                        title=f'Continuous Freq Regression: R² — {model_name}')
                ax1.legend(); ax1.grid(True, alpha=0.3)

                ax2.plot(df_reg['layer'], df_reg['spearman_r'], 's-', lw=2.5,
                         ms=7, color='#2A9D8F', label='Spearman ρ')
                ax2.axhline(0, color='gray', ls='--', lw=1.2)
                ax2.set(xlabel='Layer', ylabel='Spearman ρ',
                        title=f'Continuous Freq Regression: Correlation — {model_name}')
                ax2.legend(); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_freq_regression.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

        # ── Analysis 3: Confound-matched probe ────────────────────────
        if (res.get('confound_matched') and
                res['confound_matched'].get('matched_probe_results')):
            df_cm = pd.DataFrame(res['confound_matched']['matched_probe_results'])
            if not df_cm.empty and not df_cm['accuracy'].isna().all():
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df_cm['layer'], df_cm['accuracy'], 'D-', lw=2.5,
                        ms=7, color='#E76F51',
                        label=f'Confound-Matched (n={res["confound_matched"]["n_matched_pairs"]} pairs)')
                # Overlay original direct freq probe for comparison
                if res.get('direct_freq') and res['direct_freq']['binary']:
                    df_orig = pd.DataFrame(res['direct_freq']['binary'])
                    if not df_orig.empty:
                        ax.plot(df_orig['layer'], df_orig['accuracy'], 'o--',
                                lw=2, ms=5, color='#264653', alpha=0.6,
                                label='Unmatched')
                ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='Chance')
                ax.set(xlabel='Layer', ylabel='Accuracy',
                       title=f'Confound-Matched vs Unmatched Freq Probe — {model_name}')
                ax.legend(); ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_confound_matched.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

        # ── Analysis 8: Intervention (accuracy & freq-gap before/after) ─
        if res.get('intervention'):
            df_iv = pd.DataFrame(res['intervention'])
            if not df_iv.empty and not df_iv['baseline_accuracy'].isna().all():
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

                ax1.plot(df_iv['layer'], df_iv['baseline_accuracy'], 'o-',
                         lw=2.5, ms=8, color='#1A535C', label='Baseline (no intervention)')
                ax1.plot(df_iv['layer'], df_iv['additive_pos_accuracy'], 's--',
                         lw=2, ms=6, color='#2E86AB', label='Additive (+)')
                ax1.plot(df_iv['layer'], df_iv['additive_neg_accuracy'], '^--',
                         lw=2, ms=6, color='#D62246', label='Additive (−)')
                ax1.plot(df_iv['layer'], df_iv['nullify_accuracy'], 'D--',
                         lw=2, ms=6, color='#E76F51', label='Nullify top-k dims')
                ax1.axhline(0.5, color='gray', ls=':', lw=1.2)
                ax1.set(xlabel='Layer', ylabel='LDT Accuracy (same trained probe)',
                        title=f'Analysis 8: Intervention — Accuracy — {model_name}')
                ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

                ax2.plot(df_iv['layer'], df_iv['baseline_freq_gap'], 'o-',
                         lw=2.5, ms=8, color='#1A535C', label='Baseline gap (H−L)')
                ax2.plot(df_iv['layer'], df_iv['additive_pos_freq_gap'], 's--',
                         lw=2, ms=6, color='#2E86AB', label='Additive (+)')
                ax2.plot(df_iv['layer'], df_iv['additive_neg_freq_gap'], '^--',
                         lw=2, ms=6, color='#D62246', label='Additive (−)')
                ax2.plot(df_iv['layer'], df_iv['nullify_freq_gap'], 'D--',
                         lw=2, ms=6, color='#E76F51', label='Nullify top-k dims')
                ax2.axhline(0, color='black', lw=1.2)
                ax2.set(xlabel='Layer', ylabel='Freq Effect (High−Low Acc)',
                        title=f'Analysis 8: Intervention — Freq Gap — {model_name}')
                ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_intervention.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

        # ── Analysis 10: Contextual vs isolated frequency effect ────────
        if res.get('contextual') and res['contextual'].get('isolated'):
            df_iso = pd.DataFrame(res['contextual']['isolated'])
            df_ctx = pd.DataFrame(res['contextual'].get('contextual', []))
            if not df_iso.empty and not df_iso['accuracy'].isna().all():
                _fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(df_iso['layer'], df_iso['accuracy'], 'o-', lw=2.5,
                        ms=7, color='#2E86AB',
                        label='Task prompt: first-output prediction position')
                if not df_ctx.empty and not df_ctx['accuracy'].isna().all():
                    ax.plot(df_ctx['layer'], df_ctx['accuracy'], 's--', lw=2.5,
                            ms=7, color='#D62246',
                            label='Sentence context: target-word position (CONTROL)')
                ax.axhline(0.5, color='gray', ls=':', lw=1.2, label='Chance')
                n_h = res['contextual'].get('n_high', 0)
                n_l = res['contextual'].get('n_low', 0)
                ax.set(xlabel='Layer', ylabel='High-vs-Low Freq Accuracy',
                       title=f'Analysis 10: Contextual Freq Effect — {model_name} '
                             f'(n_high={n_h}, n_low={n_l})')
                ax.legend(); ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.FIGURES_DIR,
                                         f'{safe}_contextual_effect.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()

    # ════════════════════════════════════════════════════════════════════════
    # EXTENDED ANALYSIS CSVs
    # ════════════════════════════════════════════════════════════════════════

    def _save_extended_csvs(self, model_name, res):
        safe = model_name.replace(' ', '_').replace('/', '_')
        ext_dir = os.path.join(self.config.RESULTS_DIR, f'{safe}_extended')
        os.makedirs(ext_dir, exist_ok=True)

        # Analysis 1: Direct frequency probe
        if res.get('direct_freq'):
            if res['direct_freq']['binary']:
                pd.DataFrame(res['direct_freq']['binary']).to_csv(
                    os.path.join(ext_dir, 'direct_freq_binary.csv'), index=False)
            if res['direct_freq'].get('three_way'):
                pd.DataFrame(res['direct_freq']['three_way']).to_csv(
                    os.path.join(ext_dir, 'direct_freq_3way.csv'), index=False)

        # Analysis 2: Tokenization control
        if res.get('token_ctrl'):
            tc = res['token_ctrl']
            with open(os.path.join(ext_dir, 'token_stats.json'), 'w') as f:
                json.dump(tc['token_stats'], f, indent=2)
            if tc.get('single_token_results'):
                rows = []
                for r in tc['single_token_results']:
                    fe = r['frequency_effect']
                    ov = r.get('overall')
                    rows.append({
                        'layer': r['layer'],
                        'accuracy': ov['accuracy'] if ov else np.nan,
                        'f1': ov['f1'] if ov else np.nan,
                        'freq_diff': fe['accuracy_difference'],
                    })
                pd.DataFrame(rows).to_csv(
                    os.path.join(ext_dir, 'single_token_ldt.csv'), index=False)
            if tc.get('token_matched_results'):
                rows = []
                for r in tc['token_matched_results']:
                    fe = r['frequency_effect']
                    ov = r.get('overall')
                    rows.append({
                        'layer': r['layer'],
                        'accuracy': ov['accuracy'] if ov else np.nan,
                        'f1': ov['f1'] if ov else np.nan,
                        'freq_diff': fe['accuracy_difference'],
                    })
                pd.DataFrame(rows).to_csv(
                    os.path.join(ext_dir, 'token_matched_ldt.csv'), index=False)

        # Analysis 3: Confound matching
        if res.get('confound_matched'):
            cm = res['confound_matched']
            balance = cm['balance_before'] + cm['balance_after']
            pd.DataFrame(balance).to_csv(
                os.path.join(ext_dir, 'balance_table.csv'), index=False)
            if cm.get('matched_probe_results'):
                pd.DataFrame(cm['matched_probe_results']).to_csv(
                    os.path.join(ext_dir, 'confound_matched_probe.csv'), index=False)

        # Analysis 4: Stability
        if res.get('stability') and res['stability']['layer_stats']:
            pd.DataFrame(res['stability']['layer_stats']).to_csv(
                os.path.join(ext_dir, 'stability_stats.csv'), index=False)

        # Analysis 5: Representation ablation (positional strategies)
        if res.get('representation_ablation_rows'):
            pd.DataFrame(res['representation_ablation_rows']).to_csv(
                os.path.join(ext_dir, 'representation_ablation.csv'), index=False)

        # Primary layer table + emergence + native diagnostics (per model)
        if res.get('crossfit_table'):
            pd.DataFrame(res['crossfit_table']).to_csv(
                os.path.join(ext_dir, 'layerwise_crossfitted_probe_ldt.csv'), index=False)
        if res.get('native_ldt_table'):
            pd.DataFrame(res['native_ldt_table']).to_csv(
                os.path.join(ext_dir, 'layerwise_native_lm_head_ldt.csv'), index=False)
        if res.get('probe_table'):
            pd.DataFrame(res['probe_table']).to_csv(
                os.path.join(ext_dir, 'layerwise_trained_probe_results.csv'), index=False)
        if res.get('emergence_summary'):
            pd.DataFrame(res['emergence_summary']).to_csv(
                os.path.join(ext_dir, 'layerwise_emergence_summary.csv'), index=False)
        if res.get('prompt_diagnostics') is not None:
            res['prompt_diagnostics'].to_csv(
                os.path.join(ext_dir, 'prompt_diagnostics.csv'), index=False)

        # Analysis 6: Selectivity
        if res.get('selectivity'):
            for key in ['linear_probe', 'shuffled_label', 'selectivity']:
                data = res['selectivity'].get(key, [])
                if data:
                    pd.DataFrame(data).to_csv(
                        os.path.join(ext_dir, f'selectivity_{key}.csv'), index=False)

        # Analysis 7: Regression
        if res.get('regression') and res['regression'].get('regression_results'):
            pd.DataFrame(res['regression']['regression_results']).to_csv(
                os.path.join(ext_dir, 'freq_regression.csv'), index=False)

        # Analysis 8: Intervention
        if res.get('intervention'):
            df_iv = pd.DataFrame(res['intervention'])
            if not df_iv.empty:
                df_iv.to_csv(os.path.join(ext_dir, 'intervention_results.csv'),
                             index=False)

        # Analysis 9: Cross-model transfer — rows involving this model
        # that have been computed SO FAR (this file grows as later models
        # are processed; the authoritative, complete matrix is written
        # once at the end to comparisons/cross_model_transfer_matrix.csv)
        rows_this_model = [
            r for r in self.cross_model_transfer_rows
            if r['source_model'] == model_name or r['target_model'] == model_name
        ]
        with open(os.path.join(ext_dir, 'transfer_results.json'), 'w') as f:
            json.dump(self._csv_safe(rows_this_model), f, indent=2, default=str)

        # Analysis 10: Contextual
        if res.get('contextual') and (res['contextual'].get('isolated') or
                                       res['contextual'].get('contextual')):
            rows = []
            for r in res['contextual'].get('isolated', []):
                rows.append({**r, 'condition': 'isolated_task_first_output_prediction_position',
                             'representation_type': PRIMARY_REPRESENTATION})
            for r in res['contextual'].get('contextual', []):
                rows.append({**r, 'condition': 'sentence_context_target_token',
                             'representation_type':
                                 RepresentationType.CONTEXTUAL_TARGET_TOKEN.value})
            if rows:
                pd.DataFrame(rows).to_csv(
                    os.path.join(ext_dir, 'contextual_results.csv'), index=False)

        logger.info(f"✓ Extended CSVs saved to {ext_dir}")

    # ════════════════════════════════════════════════════════════════════════
    # PER-MODEL CSV (original)
    # ════════════════════════════════════════════════════════════════════════

    def _save_model_csv(self, model_name, lr):
        safe = model_name.replace(' ','_').replace('/','_')
        rows = []
        for r in lr:
            fe = r['frequency_effect']
            ov = r.get('overall'); hf = r.get('high_frequency'); lf = r.get('low_frequency')
            def mg(g,m): return g[m] if g else np.nan
            rows.append({
                'model': model_name, 'layer': r['layer'],
                'representation_type': PRIMARY_REPRESENTATION,
                'probe_type': r.get('probe_type', ''),
                'test_balanced_accuracy': mg(ov,'balanced_accuracy'),
                'test_macro_f1': mg(ov,'macro_f1'), 'test_auprc': mg(ov,'auprc'),
                'test_accuracy': mg(ov,'accuracy'), 'test_precision': mg(ov,'precision'),
                'test_recall':   mg(ov,'recall'),   'test_f1':        mg(ov,'f1'),
                'test_auc':      mg(ov,'auc'),       'test_n':         mg(ov,'n_samples'),
                'hf_accuracy': mg(hf,'accuracy'), 'hf_precision': mg(hf,'precision'),
                'hf_recall':   mg(hf,'recall'),   'hf_f1':        mg(hf,'f1'),
                'hf_auc':      mg(hf,'auc'),       'hf_n':         mg(hf,'n_samples'),
                'lf_accuracy': mg(lf,'accuracy'), 'lf_precision': mg(lf,'precision'),
                'lf_recall':   mg(lf,'recall'),   'lf_f1':        mg(lf,'f1'),
                'lf_auc':      mg(lf,'auc'),       'lf_n':         mg(lf,'n_samples'),
                'accuracy_difference': fe['accuracy_difference'],
                'accuracy_difference_ci95_low': fe.get('accuracy_difference_ci95_low', np.nan),
                'accuracy_difference_ci95_high': fe.get('accuracy_difference_ci95_high', np.nan),
                'cohens_h':            fe.get('cohens_h', np.nan),
                'cohens_h_ci95_low':   fe.get('cohens_h_ci95_low', np.nan),
                'cohens_h_ci95_high':  fe.get('cohens_h_ci95_high', np.nan),
                'z_statistic':         fe['z_statistic'],
                'p_value':             fe['p_value'],
                'p_value_log10':       fe.get('p_value_log10', np.nan),
                'p_value_corrected':   fe.get('p_value_corrected', np.nan),
                'significant_fdr':     fe.get('significant_fdr', False),
                'p_value_cohens_h_corrected': fe.get('p_value_cohens_h_corrected', np.nan),
                'auc_hf_vs_nonwords':  fe.get('auc_hf_vs_nonwords', np.nan),
                'auc_lf_vs_nonwords':  fe.get('auc_lf_vs_nonwords', np.nan),
                'auc_difference':      fe.get('auc_difference', np.nan),
                'p_value_auc':         fe.get('p_value_auc', np.nan),
                'p_value_auc_log10':   fe.get('p_value_auc_log10', np.nan),
                'p_value_auc_corrected': fe.get('p_value_auc_corrected', np.nan),
                'log10_bf01':          fe.get('log10_bf01', np.nan),
                'bf01_interpretation': fe.get('bf01_interpretation', ''),
                'test_accuracy_ci95_low': (wilson_ci(int(round(ov['accuracy'] * ov['n_samples'])),
                                                     int(ov['n_samples']))[0] if ov else np.nan),
                'test_accuracy_ci95_high': (wilson_ci(int(round(ov['accuracy'] * ov['n_samples'])),
                                                      int(ov['n_samples']))[1] if ov else np.nan),
                'n_high': fe['n_high'], 'n_low': fe['n_low'],
            })
        out = os.path.join(self.config.RESULTS_DIR, f'{safe}_full_metrics.csv')
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info(f"✓ CSV: {os.path.basename(out)}")

    # ════════════════════════════════════════════════════════════════════════
    # CROSS-MODEL PLOTS
    # ════════════════════════════════════════════════════════════════════════

    def _plot_cross_model(self, all_results):
        rows = []
        for mn, md in all_results.items():
            for r in md['layer_results']:
                fe = r['frequency_effect']
                ov = r.get('overall'); hf = r.get('high_frequency'); lf = r.get('low_frequency')
                def mg(g,m): return g[m] if g else np.nan
                n_layers = md['num_layers']
                rows.append({
                    'model': mn, 'layer': r['layer'],
                    'layer_norm': r['layer'] / max(n_layers - 1, 1),
                    'test_acc':  mg(ov,'accuracy'), 'test_pr': mg(ov,'precision'),
                    'test_rc':   mg(ov,'recall'),   'test_f1': mg(ov,'f1'),
                    'test_auc':  mg(ov,'auc'),
                    'hf_acc':    mg(hf,'accuracy'), 'hf_f1':   mg(hf,'f1'), 'hf_auc': mg(hf,'auc'),
                    'lf_acc':    mg(lf,'accuracy'), 'lf_f1':   mg(lf,'f1'), 'lf_auc': mg(lf,'auc'),
                    'acc_diff':  fe['accuracy_difference'],
                    'cohens_h':  fe.get('cohens_h', np.nan),
                    'p_corr':    fe.get('p_value_corrected', np.nan),
                })
        df      = pd.DataFrame(rows)
        models  = df['model'].unique()
        palette = sns.color_palette("husl", len(models))

        def _line(col, ylabel, title, fname, hline=None, refs=None, use_norm=False):
            _fig, ax = plt.subplots(figsize=(16, 7))
            x_col = 'layer_norm' if use_norm else 'layer'
            x_label = 'Normalised Depth (0=first, 1=last)' if use_norm else 'Layer'
            for m, c in zip(models, palette):
                sub = df[df['model']==m]
                ax.plot(sub[x_col], sub[col], 'o-', label=m, lw=2.8, ms=8, color=c)
            if hline is not None:
                ax.axhline(hline, color='red', ls='--', lw=1.8, label='Chance/Zero')
            if refs:
                for h, lbl, c in refs:
                    ax.axhline(h, color=c, ls='--', lw=1.2, alpha=0.6, label=lbl)
            ax.set(xlabel=x_label, ylabel=ylabel, title=title)
            ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.COMPARISON_DIR, fname),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()
            logger.info(f"✓ {fname}")

        # Raw layer plots
        _line('test_acc',  'Accuracy',           'Cross-Model: Overall Test Accuracy',
              'cross_model_test_accuracy.png',   hline=0.5)
        _line('hf_acc',    'Accuracy',           'Cross-Model: High-Freq Accuracy',
              'cross_model_high_freq_accuracy.png', hline=0.5)
        _line('lf_acc',    'Accuracy',           'Cross-Model: Low-Freq Accuracy',
              'cross_model_low_freq_accuracy.png',  hline=0.5)
        _line('test_f1',   'F1-Score',           'Cross-Model: Overall Test F1',
              'cross_model_test_f1.png')
        _line('test_auc',  'AUC-ROC',            'Cross-Model: Overall Test AUC',
              'cross_model_test_auc.png',        hline=0.5)
        _line('acc_diff',  'Acc Diff (High−Low)','Cross-Model: Frequency Effect',
              'cross_model_frequency_effect.png', hline=0.0)
        _line('cohens_h',  "Cohen's h",          "Cross-Model: Effect Size",
              'cross_model_effect_size.png',
              refs=[(0.2,'Small','#e57373'),(0.5,'Medium','#ffa726'),(0.8,'Large','#66bb6a')])

        # Normalised depth plots
        _line('test_acc',  'Accuracy',
              'Cross-Model: Test Accuracy (Normalised Depth)',
              'cross_model_test_accuracy_normdepth.png', hline=0.5, use_norm=True)
        _line('acc_diff',  'Acc Diff (High−Low)',
              'Cross-Model: Frequency Effect (Normalised Depth)',
              'cross_model_freq_effect_normdepth.png', hline=0.0, use_norm=True)

        # heatmaps
        for col, label, cmap, center, fname in [
            ('test_acc', 'Overall Test Accuracy',   'YlOrRd', None,  'cross_model_heatmap_test_acc.png'),
            ('acc_diff', 'Frequency Effect',        'RdBu_r', 0.0,   'cross_model_heatmap_freq_effect.png'),
            ('test_f1',  'Test F1-Score',           'YlOrRd', None,  'cross_model_heatmap_f1.png'),
            ('hf_acc',   'High-Freq Accuracy',      'YlOrRd', None,  'cross_model_heatmap_hf_acc.png'),
            ('lf_acc',   'Low-Freq Accuracy',       'YlOrRd', None,  'cross_model_heatmap_lf_acc.png'),
        ]:
            try:
                pivot = df.pivot(index='layer', columns='model', values=col)
                fig, ax = plt.subplots(figsize=(max(10, len(models)*4), 10))
                sns.heatmap(pivot.T, annot=True, fmt='.3f', cmap=cmap, center=center,
                            linewidths=0.4, ax=ax, cbar_kws={'label': label})
                ax.set(xlabel='Layer', ylabel='Model', title=f'Heatmap: {label}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.COMPARISON_DIR, fname),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()
                logger.info(f"✓ {fname}")
            except Exception as e:
                logger.warning(f"Heatmap {fname} skipped: {e}")

        # 3×3 all-metrics grid
        panels = [
            ('test_acc', 'Overall Accuracy',   0.5),
            ('hf_acc',   'High-Freq Accuracy', 0.5),
            ('lf_acc',   'Low-Freq Accuracy',  0.5),
            ('test_pr',  'Overall Precision',  None),
            ('test_rc',  'Overall Recall',     None),
            ('test_f1',  'Overall F1-Score',   None),
            ('test_auc', 'Overall AUC-ROC',    0.5),
            ('acc_diff', 'Freq Effect',         0.0),
            ('cohens_h', "Cohen's h",           0.0),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        for ax, (col, ylabel, hl) in zip(axes.flat, panels):
            for m, c in zip(models, palette):
                sub = df[df['model']==m]
                ax.plot(sub['layer'], sub[col], 'o-', label=m, lw=2, ms=5, color=c)
            if hl is not None:
                ax.axhline(hl, color='gray', ls='--', lw=1.2)
            ax.set(xlabel='Layer', ylabel=ylabel, title=ylabel)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        fig.suptitle('Cross-Model Comparison — All Metrics', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                 'cross_model_all_metrics_grid.png'),
                    bbox_inches='tight', dpi=300)
        plt.close('all'); gc.collect()
        logger.info("✓ cross_model_all_metrics_grid.png")

        # ── Cross-model extended comparison plots ─────────────────────
        self._plot_cross_model_extended(all_results, models, palette)

    def _plot_cross_model_extended(self, all_results, models, palette):
        """Cross-model comparison plots for extended analyses."""

        # ── Cross-model: Direct Frequency Probe ──────────────────────
        has_direct = any(
            r.get('direct_freq', {}).get('binary')
            for r in all_results.values()
        )
        if has_direct:
            fig, ax = plt.subplots(figsize=(16, 7))
            for m, c in zip(models, palette):
                if m not in all_results:
                    continue
                df_bin = all_results[m].get('direct_freq', {}).get('binary', [])
                if df_bin:
                    df_b = pd.DataFrame(df_bin)
                    nl = all_results[m]['num_layers']
                    df_b['layer_norm'] = df_b['layer'] / max(nl - 1, 1)
                    ax.plot(df_b['layer_norm'], df_b['accuracy'], 'o-',
                            label=m, lw=2.5, ms=7, color=c)
            ax.axhline(0.5, color='red', ls='--', lw=1.5, label='Chance')
            ax.set(xlabel='Normalised Depth', ylabel='Accuracy',
                   title='Cross-Model: Direct Frequency Probe (Binary)')
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                     'cross_model_direct_freq_probe.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()

        # ── Cross-model: Regression R² ────────────────────────────────
        has_reg = any(
            r.get('regression', {}).get('regression_results')
            for r in all_results.values()
        )
        if has_reg:
            fig, ax = plt.subplots(figsize=(16, 7))
            for m, c in zip(models, palette):
                if m not in all_results:
                    continue
                reg = all_results[m].get('regression', {}).get('regression_results', [])
                if reg:
                    df_r = pd.DataFrame(reg)
                    nl = all_results[m]['num_layers']
                    df_r['layer_norm'] = df_r['layer'] / max(nl - 1, 1)
                    ax.plot(df_r['layer_norm'], df_r['r2'], 'o-',
                            label=m, lw=2.5, ms=7, color=c)
            ax.axhline(0, color='gray', ls='--', lw=1.2)
            ax.set(xlabel='Normalised Depth', ylabel='R²',
                   title='Cross-Model: Continuous Frequency Regression R²')
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                     'cross_model_freq_regression_r2.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()

        # ── Cross-model: Representation ablation (primary − control) ─
        has_abl = any(r.get('representation_ablation') for r in all_results.values())
        if has_abl:
            ctrl_types = [rt for rt in self.config.REPRESENTATION_TYPES[1:]]
            fig, axes = plt.subplots(1, max(len(ctrl_types), 1),
                                     figsize=(10 * max(len(ctrl_types), 1), 7),
                                     squeeze=False)
            for ax, rep_type in zip(axes[0], ctrl_types):
                for m, c in zip(models, palette):
                    md = all_results.get(m)
                    if not md or rep_type not in md.get('representation_ablation', {}):
                        continue
                    nl = md['num_layers']
                    p_acc = self._series(md['layer_results'], 'overall', 'accuracy')
                    c_acc = self._series(md['representation_ablation'][rep_type],
                                         'overall', 'accuracy')
                    rel = [(r['layer'] + 1) / nl for r in md['layer_results']]
                    ax.plot(rel, p_acc - c_acc, 'o-', label=m, lw=2, ms=5, color=c)
                ax.axhline(0, color='black', lw=1.2)
                ax.set(xlabel='Relative depth (layer+1)/L',
                       ylabel=f'Accuracy: {PRIMARY_REPRESENTATION} − {rep_type}',
                       title=f'Primary vs {rep_type}')
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                     'cross_model_representation_ablation.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()

        # ── Analysis 9: Cross-model transfer heatmap ───────────────────
        if self.cross_model_transfer_rows:
            df_tr = pd.DataFrame(self.cross_model_transfer_rows)
            # Aggregate across matched layer-pairs into one number per
            # (source, target) — mean transfer accuracy, since a single
            # scalar per model pair is what a "transfer matrix" means.
            agg = (df_tr.groupby(['source_model', 'target_model'])
                        ['transfer_accuracy'].mean().reset_index())
            try:
                pivot = agg.pivot(index='source_model', columns='target_model',
                                  values='transfer_accuracy')
                # Diagonal = native accuracy (probe tested on its own model)
                native = (df_tr.groupby('target_model')['native_accuracy']
                                .mean())
                for m in pivot.index:
                    if m in pivot.columns and m in native.index:
                        pivot.loc[m, m] = native[m]
                fig, ax = plt.subplots(figsize=(max(10, len(models)*1.2),
                                                max(8, len(models)*1.0)))
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                            linewidths=0.4, ax=ax,
                            cbar_kws={'label': 'Mean Transfer Accuracy '
                                                '(matched by norm. depth)'})
                ax.set(xlabel='Target Model (evaluated on)',
                       ylabel='Source Model (probe trained on)',
                       title='Cross-Model: Frequency-Probe Transfer Accuracy via ridge '
                             'stitching\n(diagonal = native accuracy; see '
                             'C4_H3_H4_transfer_with_nulls.csv for nulls)')
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                         'cross_model_transfer_heatmap.png'),
                            bbox_inches='tight', dpi=300)
                plt.close('all'); gc.collect()
                logger.info("✓ cross_model_transfer_heatmap.png")
            except Exception as e:
                logger.warning(f"Transfer heatmap skipped: {e}")

        # ── Analysis 10: Cross-model contextual pattern ────────────────
        has_ctx = any(r.get('contextual', {}).get('isolated')
                      for r in all_results.values())
        if has_ctx:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
            for m, c in zip(models, palette):
                if m not in all_results:
                    continue
                ctx = all_results[m].get('contextual', {})
                nl  = all_results[m]['num_layers']
                iso = pd.DataFrame(ctx.get('isolated', []))
                sen = pd.DataFrame(ctx.get('contextual', []))
                if not iso.empty and not iso['accuracy'].isna().all():
                    iso = iso.copy()
                    iso['layer_norm'] = iso['layer'] / max(nl - 1, 1)
                    ax1.plot(iso['layer_norm'], iso['accuracy'], 'o-',
                             label=m, lw=2.2, ms=6, color=c)
                if not sen.empty and not sen['accuracy'].isna().all():
                    sen = sen.copy()
                    sen['layer_norm'] = sen['layer'] / max(nl - 1, 1)
                    ax2.plot(sen['layer_norm'], sen['accuracy'], 's-',
                             label=m, lw=2.2, ms=6, color=c)
            ax1.axhline(0.5, color='gray', ls='--', lw=1.2)
            ax1.set(xlabel='Normalised Depth', ylabel='High-vs-Low Accuracy',
                    title='Task prompt — first-output prediction position')
            ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
            ax2.axhline(0.5, color='gray', ls='--', lw=1.2)
            ax2.set(xlabel='Normalised Depth',
                    title='Sentence context — target-word position (CONTROL)')
            ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
            fig.suptitle('Cross-Model: Contextual Frequency Effect', fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.COMPARISON_DIR,
                                     'cross_model_contextual_pattern.png'),
                        bbox_inches='tight', dpi=300)
            plt.close('all'); gc.collect()
            logger.info("✓ cross_model_contextual_pattern.png")

    # ════════════════════════════════════════════════════════════════════════
    # ANALYSIS 9 & 10 — CROSS-MODEL CSV EXPORTS
    # ════════════════════════════════════════════════════════════════════════

    def _save_cross_model_transfer_csv(self):
        """comparisons/cross_model_transfer_matrix.csv and
        comparisons/cross_model_transfer_efficiency.csv (Analysis 9)."""
        if not self.cross_model_transfer_rows:
            logger.warning("No cross-model transfer results to save "
                           "(0 valid model pairs)")
            return

        df = pd.DataFrame(self.cross_model_transfer_rows)
        df.to_csv(os.path.join(self.config.COMPARISON_DIR,
                               'cross_model_transfer_detailed.csv'), index=False)

        acc_matrix = df.pivot_table(
            index='source_model', columns='target_model',
            values='transfer_accuracy', aggfunc='mean')
        acc_matrix.to_csv(os.path.join(self.config.COMPARISON_DIR,
                                       'cross_model_transfer_matrix.csv'))

        eff_matrix = df.pivot_table(
            index='source_model', columns='target_model',
            values='transfer_efficiency', aggfunc='mean')
        eff_matrix.to_csv(os.path.join(self.config.COMPARISON_DIR,
                                       'cross_model_transfer_efficiency.csv'))

        n_valid = df['skip_reason'].isna().sum()
        n_total = len(df)
        logger.info(f"✓ cross_model_transfer_matrix.csv / "
                    f"cross_model_transfer_efficiency.csv "
                    f"({n_valid}/{n_total} layer-pairs had matching "
                    f"hidden_dim and were actually transferable)")

    def _save_cross_model_contextual_csv(self, all_results):
        """comparisons/cross_model_contextual_comparison.csv (Analysis 10)."""
        rows = []
        for mn, md in all_results.items():
            ctx = md.get('contextual', {})
            nl  = md['num_layers']
            for r in ctx.get('isolated', []):
                rows.append({'model': mn,
                             'condition': 'isolated_task_first_output_prediction_position',
                             'representation_type': PRIMARY_REPRESENTATION,
                            'layer': r['layer'],
                            'layer_norm': r['layer'] / max(nl - 1, 1),
                            'accuracy': r['accuracy'], 'f1': r['f1'],
                            'auc': r['auc'], 'n': r['n']})
            for r in ctx.get('contextual', []):
                rows.append({'model': mn, 'condition': 'sentence_context_target_token',
                             'representation_type': RepresentationType.CONTEXTUAL_TARGET_TOKEN.value,
                            'layer': r['layer'],
                            'layer_norm': r['layer'] / max(nl - 1, 1),
                            'accuracy': r['accuracy'], 'f1': r['f1'],
                            'auc': r['auc'], 'n': r['n']})
        if not rows:
            logger.warning("No contextual analysis results to save")
            return
        pd.DataFrame(rows).to_csv(
            os.path.join(self.config.COMPARISON_DIR,
                         'cross_model_contextual_comparison.csv'), index=False)
        logger.info("✓ cross_model_contextual_comparison.csv")

    # ════════════════════════════════════════════════════════════════════════
    # CROSS-MODEL CSV + METADATA
    # ════════════════════════════════════════════════════════════════════════

    def _save_cross_model_csv(self, all_results):
        detailed, summary = [], []
        for mn, md in all_results.items():
            lr = md['layer_results']
            for r in lr:
                fe = r['frequency_effect']
                ov = r.get('overall'); hf = r.get('high_frequency'); lf = r.get('low_frequency')
                def mg(g,m): return g[m] if g else np.nan
                detailed.append({
                    'model': mn, 'representation_type': PRIMARY_REPRESENTATION,
                    'architecture': md['model_config'].architecture_type.value,
                    'input_mode': md['model_config'].input_mode.value, 'layer': r['layer'],
                    'test_accuracy': mg(ov,'accuracy'), 'test_precision': mg(ov,'precision'),
                    'test_recall':   mg(ov,'recall'),   'test_f1':        mg(ov,'f1'),
                    'test_auc':      mg(ov,'auc'),
                    'hf_accuracy': mg(hf,'accuracy'), 'hf_f1': mg(hf,'f1'), 'hf_auc': mg(hf,'auc'),
                    'lf_accuracy': mg(lf,'accuracy'), 'lf_f1': mg(lf,'f1'), 'lf_auc': mg(lf,'auc'),
                    'accuracy_difference': fe['accuracy_difference'],
                    'cohens_h':            fe.get('cohens_h', np.nan),
                    'z_statistic':         fe['z_statistic'],
                    'p_value':             fe['p_value'],
                    'p_value_log10':       fe.get('p_value_log10', np.nan),
                    'p_value_corrected':   fe.get('p_value_corrected', np.nan),
                    'significant_fdr':     fe.get('significant_fdr', False),
                })

            def _nm(lst): return float(np.nanmean(lst)) if lst else np.nan
            def _nx(lst): return float(np.nanmax(lst))  if lst else np.nan
            accs  = [r['overall']['accuracy'] for r in lr if r.get('overall')]
            f1s   = [r['overall']['f1']       for r in lr if r.get('overall')]
            aucs  = [r['overall']['auc']       for r in lr if r.get('overall')]
            diffs = [r['frequency_effect']['accuracy_difference'] for r in lr
                     if not np.isnan(r['frequency_effect']['accuracy_difference'])]
            hs    = [r['frequency_effect']['cohens_h'] for r in lr
                     if not np.isnan(r['frequency_effect'].get('cohens_h', np.nan))]
            pcs   = [r['frequency_effect'].get('p_value_corrected', np.nan) for r in lr]
            summary.append({
                'model':               mn,
                'architecture':        md['model_config'].architecture_type.value,
                'input_mode':          md['model_config'].input_mode.value,
                'num_layers':          md['num_layers'],
                'mean_test_accuracy':  _nm(accs),  'max_test_accuracy': _nx(accs),
                'mean_test_f1':        _nm(f1s),   'mean_test_auc':     _nm(aucs),
                'mean_freq_effect':    _nm(diffs),  'max_freq_effect':   _nx(diffs),
                'mean_cohens_h':       _nm(hs),     'max_cohens_h':      _nx(hs),
                'n_significant_layers':sum(1 for p in pcs
                                           if not np.isnan(p) and p < 0.05),
            })

        det = pd.DataFrame(detailed)
        if not det.empty and 'p_value' in det:
            # G4 — global BH-FDR across every (model, layer) pair
            det['p_value_fdr_bh_global'] = adjust_pvalues(det['p_value'].to_numpy(float))
        det.to_csv(
            os.path.join(self.config.COMPARISON_DIR, 'cross_model_detailed_results.csv'),
            index=False)
        pd.DataFrame(summary).to_csv(
            os.path.join(self.config.COMPARISON_DIR, 'cross_model_statistics.csv'),
            index=False)

        meta = {
            'version': 'NAACL revision — first-output prediction position, cross-fitted linear probe',
            'representation_type': PRIMARY_REPRESENTATION,
            'probe_type': probe_type_name(self.config.CLASSIFIER_ARCHITECTURE),
            'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': [{'name': m.name, 'model_id': m.model_id,
                        'architecture': m.architecture_type.value,
                        'input_mode': m.input_mode.value,
                        'padding_side': m.padding_side.value}
                       for m in self.config.MODELS],
            'extended_analyses': [
                '1. Direct frequency probe (binary H-vs-L + 3-way H/M/L)',
                '2. Tokenization-controlled rerun (single-token + token-matched)',
                '3. Lexical-confound-matched rerun (length + Ortho_N + token-count matching)',
                '4. Multi-seed stability (N seeds with mean ± std)',
                ('5. Representation ablation: first_output_prediction_position '
                 '(PRIMARY) vs mean_prompt_tokens vs '
                 'generated_token_representation_control, same split/probe/seed, '
                 'paired McNemar'),
                '6. Probe selectivity controls (linear baseline + shuffled labels)',
                '7. Continuous frequency regression (Ridge on log_HAL)',
                ('8. Representation intervention (additive +/- and nullification '
                'causal test on frequency-relevant dimensions)'),
                ('9. Cross-model probe transferability (matched by normalised '
                'depth; undefined for hidden_dim-mismatched model pairs)'),
                ('10. Contextual (sentence-embedded) frequency effect '
                '(requires new forward passes; disclosed exception to the '
                'cached-hidden-states-only constraint)'),
            ],
            'references': [
                'He et al. (2015): Kaiming init for ReLU',
                'He et al. (2016): Pre-activation residual blocks',
                'Lin et al. (2017): Focal Loss',
                'Ioffe & Szegedy (2015): Batch Normalisation',
                'Agresti (2002): Two-proportion z-test',
                'Muennighoff et al. (2022): SGPT — last-token pooling (control)',
                'Alain & Bengio (2017): Understanding intermediate layers using linear classifier probes',
                'Belinkov (2022): Probing classifiers — promises, shortcomings, advances',
                'McNemar (1947); Holm (1979); Benjamini & Hochberg (1995)',
                'Efron & Tibshirani (1993): bootstrap confidence intervals',
                'Hewitt & Liang (2019): Designing and interpreting probes',
                'Yang et al. (2024): Qwen2 Technical Report',
                'Falcon-LLM Team (2024): Falcon 3 Family of Open Models',
                'Meta AI (2024): Llama 3.2 — Llama 3.2-3B base model',
                'Zhang & Bowman (2018): random-init encoders as probing baselines',
                'Austin (2009, 2011): SMD balance diagnostics; caliper matching',
                'Stuart (2010); Imai, King & Stuart (2008): balance test fallacy',
                'Belrose et al. (2023): LEACE — linear concept erasure',
                'Ravfogel et al. (2020): INLP — iterative nullspace projection',
                'DeLong, DeLong & Clarke-Pearson (1988): correlated AUC comparison',
                'Wagenmakers (2007): BIC approximation to the Bayes factor',
                'MacKinnon & White (1985): HC3 standard errors',
                'Cameron & Miller (2015): cluster-robust inference with few clusters',
                'Baayen, Davidson & Bates (2008): mixed-effects models for items/subjects',
                'Geva et al. (2023): attention knockout',
                'Wang et al. (2023): mean-ablation of attention heads (IOI circuit)',
                'Elhage et al. (2021): residual-stream direct attribution',
                'Kornblith et al. (2019): linear CKA',
                'Kaplan et al. (2020); Hoffmann et al. (2022): scaling laws',
                'Balota et al. (2007): The English Lexicon Project',
            ],
        }
        with open(os.path.join(self.config.COMPARISON_DIR, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info("✓ Saved all cross-model CSVs and metadata.json")


# ════════════════════════════════════════════════════════════════════════════
# K1–K6 — claim boundary, multiplicity procedure, analysis plan (metadata)
# ════════════════════════════════════════════════════════════════════════════
CLAIM_BOUNDARY = {
    'K1_what_is_claimed': ('We probe the layer-wise hidden representation at the '
                           'first-output prediction position for lexicality.'),
    'K1_what_is_NOT_claimed': ('We do NOT claim that the LLM predicts WORD/NONWORD at '
                               'each layer; probes are analysis tools, and decodability '
                               'is not use (causal analyses C2/C3/C5/D2 address use).'),
    'K2_terminology': ('"hidden state at the first-output prediction position" — the '
                       'position is the final prompt token; it is not a generated token '
                       'and not a pooled representation.'),
    'K3_depth_is_not_time': ('Layer depth is not a reaction time. RT alignment is a '
                             'statistical association between probe-derived lexical '
                             'evidence and ELP mean RTs, not a model of processing speed.'),
    'K4_confound_limitations': ('HF and LF words differ in length, Ortho_N and sub-word '
                                'token count; matching (B1–B3), covariate erasure (B4), '
                                'covariate models (B5/B6), tokenization strata (D4) and '
                                'surface baselines (A6) bound but do not eliminate them '
                                '(e.g. bigram frequency, age of acquisition and '
                                'morphology are not controlled).'),
}

MULTIPLE_COMPARISONS_PROCEDURE = {
    'across_layers_within_model_and_analysis': 'Benjamini–Hochberg FDR (q = ALPHA)',
    'paired_model_or_representation_comparisons_across_layers': 'Holm (FWER)',
    'consecutive_layer_McNemar': 'Holm across layers',
    'global_across_models_and_layers': ('BH-FDR over every (model, layer) test, columns '
                                        '*_global (G4, E4)'),
    'effect_sizes': "Cohen's h with percentile-bootstrap 95% CIs (B = N_BOOTSTRAP)",
    'auc_differences': 'DeLong variance with shared negatives, BH-FDR across layers',
    'null_results': 'BIC-approximate BF01 (Wagenmakers, 2007); ≥ 3 = moderate evidence for H0',
    'p_value_reporting': 'exact, from log survival functions; *_log10 columns; never 0.0',
}

ANALYSIS_PLAN = {
    'note': ('Written AFTER the first results were seen; it separates the analyses the '
             'paper\'s claims rest on from exploratory ones. It is NOT a pre-registration '
             'and must not be described as one (K6).'),
    'confirmatory': {
        'primary_curve': 'cross-fitted linear probe accuracy / AUROC per layer vs chance',
        'controls_the_primary_claim_must_survive': [
            'A2 random-init model (trained − random > 0, Holm McNemar)',
            'A6 surface-form baselines (primary − best baseline > 0)',
            'B4 covariate-erased probe', 'A1 MLP probe (capacity check)'],
        'frequency_effect': 'HF − LF word accuracy of the primary probe (BH-FDR, Cohen\'s h CI)',
        'rt_alignment': 'Spearman(S, RT) and HC3 regression on the accuracy-filtered sample',
    },
    'exploratory': ['A3 sublayers', 'C1–C5 interventions', 'D1–D3 attention and position',
                    'F contextual conditions', 'H1 scaling', 'H2 curve alignment and CKA',
                    'E3 mixed model', 'analysis 11b emergence measures'],
}


def _environment_snapshot() -> dict:
    """J10 — exact package versions (importlib.metadata; equivalent to pip
    freeze without spawning pip) plus interpreter, platform and CUDA info."""
    import platform
    try:
        from importlib import metadata as _md
        pk = sorted(f"{d.metadata['Name']}=={d.version}" for d in _md.distributions()
                    if d.metadata.get('Name'))
    except Exception as e:                                   # noqa: BLE001
        pk = [f'unavailable: {e}']
    return {'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'transformers': _transformers_pkg.__version__,
            'cuda_available': bool(torch.cuda.is_available()),
            'cuda_version': getattr(torch.version, 'cuda', None),
            'gpu': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            'pip_freeze': pk}


# ════════════════════════════════════════════════════════════════════════════
# J8 — UNIT TESTS (python LDT.py --self-test ; also run before every experiment)
# ════════════════════════════════════════════════════════════════════════════
def run_self_tests(verbose: bool = True) -> bool:
    import unittest

    class T(unittest.TestCase):
        def test_mcnemar_exact(self):
            a = np.array([1, 1, 1, 0, 0, 1, 1, 0])
            b = np.array([1, 0, 0, 0, 1, 1, 0, 0])
            bb, cc, p = mcnemar_exact(a, b)
            self.assertEqual((bb, cc), (3, 1))
            self.assertAlmostEqual(p, stats.binomtest(1, 4, 0.5).pvalue)
            self.assertEqual(mcnemar_exact(a, a)[2], 1.0)
            with self.assertRaises(ValueError):
                mcnemar_exact(a, b[:-1])

        def test_bootstrap_ci(self):
            cfg = type('C', (), {'N_BOOTSTRAP': 4000})()
            exp = MultiModelExperiment.__new__(MultiModelExperiment)
            exp.config = cfg
            x = np.r_[np.ones(700), np.zeros(300)]
            lo, hi = exp._bootstrap_ci(x)
            se = np.sqrt(0.7 * 0.3 / 1000)
            self.assertAlmostEqual(lo, 0.7 - 1.96 * se, delta=0.01)
            self.assertAlmostEqual(hi, 0.7 + 1.96 * se, delta=0.01)
            self.assertTrue(np.isnan(exp._bootstrap_ci(np.array([]))[0]))

        def test_fit_stitching_map_recovers_linear_map(self):
            cfg = type('C', (), {'STITCHING_MIN_PAIRED_SAMPLES': 20,
                                 'STITCHING_ALPHAS': [1e-3, 1.0, 10.0]})()
            ea = ExtendedAnalyses.__new__(ExtendedAnalyses)
            ea.config = cfg
            rng = np.random.default_rng(0)
            Xt = rng.normal(size=(400, 6))
            A = rng.normal(size=(6, 4))
            Xs = Xt @ A + 0.01 * rng.normal(size=(400, 4))
            stim = np.arange(400)
            (m, sc), alpha, r2, n = ea._fit_stitching_map(Xs, Xt, stim, stim[::-1].copy()[::-1])
            self.assertEqual(n, 400)
            self.assertGreater(r2, 0.99)
            self.assertLess(np.abs(m.predict(sc.transform(Xt)) - Xs).max(), 0.1)
            (m2, _), *_rest = ea._fit_stitching_map(Xs, Xt, stim[:5], stim[:5])
            self.assertIsNone(m2)

        def test_pvalues_never_zero(self):
            p, lp = p_from_z(60.0)
            self.assertGreaterEqual(p, 0.0)
            self.assertTrue(np.isfinite(lp) and lp < -700)
            self.assertEqual(p_display(0.0, -900.0), '< 1e-300')
            p2, _ = p_from_z(1.959963985)
            self.assertAlmostEqual(p2, 0.05, places=6)
            pc, lpc = p_correlation(0.9999, 10 ** 5)
            self.assertTrue(np.isfinite(lpc))

        def test_cohens_h_and_bootstrap(self):
            self.assertAlmostEqual(cohens_h(0.5, 0.5), 0.0)
            self.assertAlmostEqual(cohens_h(0.9, 0.5),
                                   2 * np.arcsin(np.sqrt(0.9)) - 2 * np.arcsin(np.sqrt(0.5)))
            ci = bootstrap_proportion_contrast_ci(90, 100, 50, 100, B=4000)
            h = cohens_h(0.9, 0.5)
            self.assertLess(ci['cohens_h_ci95_low'], h)
            self.assertGreater(ci['cohens_h_ci95_high'], h)

        def test_delong_matches_mann_whitney_and_null(self):
            rng = np.random.default_rng(1)
            neg = rng.normal(0, 1, 300)
            pa, pb = rng.normal(1, 1, 200), rng.normal(1, 1, 250)
            d = delong_auc_diff_shared_negatives(pa, pb, neg)
            u = stats.mannwhitneyu(pa, neg).statistic / (len(pa) * len(neg))
            self.assertAlmostEqual(d['auc_a'], u, places=10)
            self.assertGreater(d['p_value_auc'], 0.001)
            d2 = delong_auc_diff_shared_negatives(rng.normal(3, 1, 200), pb, neg)
            self.assertLess(d2['p_value_auc'], 1e-6)

        def test_bayes_factor_direction(self):
            self.assertGreater(log10_bf01_correlation(0.001, 5000), np.log10(3))
            self.assertLess(log10_bf01_correlation(0.3, 5000), -np.log10(3))
            self.assertGreater(log10_bf01_two_proportions(500, 1000, 502, 1000), 0)
            self.assertLess(log10_bf01_two_proportions(900, 1000, 500, 1000), 0)

        def test_smd_and_caliper_matching(self):
            rng = np.random.default_rng(2)
            t = rng.normal(0.5, 1, (200, 2))
            c = rng.normal(0.0, 1, (600, 2))
            psd = np.sqrt((t.var(0, ddof=1) + c.var(0, ddof=1)) / 2)
            mt, mc_, rej = greedy_caliper_match(t, c, 0.2, psd)
            self.assertEqual(len(mt) + rej, len(t))
            self.assertEqual(len(set(mc_.tolist())), len(mc_))
            self.assertTrue(np.all(np.abs(t[mt] - c[mc_]) <= 0.2 * psd + 1e-9))
            before = abs(standardized_mean_difference(t[:, 0], c[:, 0]))
            after = abs(standardized_mean_difference(t[mt, 0], c[mc_, 0]))
            self.assertLess(after, before)
            self.assertLess(after, 0.1)

        def test_linear_cka_invariances(self):
            rng = np.random.default_rng(3)
            X = rng.normal(size=(100, 20))
            Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
            self.assertAlmostEqual(linear_cka_from_grams(centered_gram(X),
                                                         centered_gram(3.0 * X @ Q)), 1.0, places=8)
            self.assertLess(linear_cka_from_grams(centered_gram(X),
                                                  centered_gram(rng.normal(size=(100, 20)))), 0.5)

        def test_residualisation_is_linearly_guarded(self):
            rng = np.random.default_rng(4)
            Z = rng.normal(size=(500, 2))
            X = Z @ rng.normal(size=(2, 30)) + rng.normal(size=(500, 30))
            Xr, _ = FrequencyAnalyzer._residualize(X, X[:5], Z, Z[:5])
            cc = (Xr - Xr.mean(0)).T @ (Z - Z.mean(0)) / len(Z)
            self.assertLess(np.abs(cc).max(), 1e-8)

        def test_inlp_removes_direction(self):
            rng = np.random.default_rng(5)
            y = rng.integers(0, 2, 600)
            X = rng.normal(size=(600, 10))
            X[:, 0] += 3 * y
            U = inlp_subspace(X, y, 2)
            Xp = X - (X - X.mean(0)) @ U @ U.T
            acc = fit_linear_direction(Xp, y)['train_accuracy']
            self.assertLess(acc, 0.65)

        def test_torch_logistic_matches_sklearn(self):
            rng = np.random.default_rng(6)
            X = rng.normal(size=(800, 12)) * rng.uniform(0.5, 5, 12)
            y = (X[:, 0] / 3 - X[:, 3] / 4 + rng.normal(size=800) > 0).astype(int)
            Xte = rng.normal(size=(200, 12)) * 2
            for C in (0.05, 1.0):
                sc = StandardScaler().fit(X)
                ref = LogisticRegression(C=C, max_iter=5000, tol=1e-10).fit(sc.transform(X), y)
                p_ref = ref.predict_proba(sc.transform(Xte))[:, 1]
                p_t = FrequencyAnalyzer.fit_predict_logistic_torch(X, y, Xte, C=C, device='cpu')
                self.assertLess(np.abs(p_ref - p_t).max(), 2e-3)

        def test_direction_backends_agree(self):
            rng = np.random.default_rng(7)
            X = rng.normal(size=(500, 8)) * rng.uniform(1, 10, 8) + 5
            y = (X[:, 2] - 5 + rng.normal(size=500) * 3 > 0).astype(int)
            with torch.no_grad():                       # callers run under no_grad
                a = fit_linear_direction(X, y, 1.0, device='cpu')
            b = fit_linear_direction(X, y, 1.0, device=None) if not torch.cuda.is_available() \
                else fit_linear_direction(X, y, 1.0, device='cpu')
            self.assertGreater(abs(float(a['u'] @ b['u'])), 0.999)
            self.assertAlmostEqual(a['train_accuracy'], b['train_accuracy'], delta=0.01)

        def test_representative_layers(self):
            self.assertEqual(representative_layers(range(8)), [0, 2, 4, 6, 7])
            self.assertEqual(select_layers(36, 'all')[-1], 35)
            self.assertLessEqual(len(select_layers(36, 'representative', 4)), 4)

    suite = unittest.TestLoader().loadTestsFromTestCase(T)
    res = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    return res.wasSuccessful()


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    if '--self-test' in sys.argv:
        ok = run_self_tests()
        sys.exit(0 if ok else 1)
    config = Config()
    if config.RUN_SELF_TESTS_BEFORE_EXPERIMENT and not run_self_tests(verbose=False):
        raise SystemExit("Self-tests failed — refusing to run the experiment.")
    print(f"\n{'='*64}\nVERIFYING DATA FILES\n{'='*64}")
    for path, name in [(config.WORDS_PATH,'Words'),(config.NONWORDS_PATH,'NonWords')]:
        if os.path.exists(path):
            print(f"  ✓  {name}  →  {path}")
        else:
            print(f"  ✗  MISSING: {name}  →  {path}"); return None
    print()
    return MultiModelExperiment(config).run()


if __name__ == "__main__":
    main()