"""
LDT-NAACL — Lexicality at the decision site of a task-conditioned lexical
decision task in decoder-only LLMs
=========================================================================

OBJECTIVE
  Where across depth, and how strongly, does lexicality (word vs. ELP
  pseudoword) become linearly decodable at the DECISION SITE — the final
  prompt token, whose hidden state predicts the first answer token — of a
  task-conditioned lexical decision task (LDT)? Does this internal lexical
  evidence show the human word-frequency effect, predict human LDT reaction
  times beyond standard lexical covariates, and is it USED for the model's
  own YES/NO decision?

RESEARCH QUESTIONS  (confirmatory = headline claims; everything else exploratory)
  RQ1 [confirmatory]  lexicality is decodable at the decision site
        H1a  out-of-fold AUROC > chance
        H1b  trained model > identically-probed random-initialisation model
        H1c  decision-site probe > best probe that never sees a hidden state
             (character n-grams, the model's own sub-word ids, surface
             covariates, the model's own unconditional word log-probability)
  RQ2 [confirmatory]  frequency effect in internal lexical evidence
        H2   AUROC(HF words vs nonwords) − AUROC(LF words vs nonwords) > 0
             (discriminability, not a hit rate; DeLong, shared negatives).
             Effect size: continuous slope of lexical evidence on
             log-frequency, words only, controlling length / Ortho_N /
             token count / model log-probability (no extreme-group inflation).
             Robustness: caliper-matched HF/LF sets, covariate-erased probe,
             single- vs multi-token strata.
             All word covariates also include every supplied lexical norm
             (morpheme count, morphological family size, AoA, concreteness …).
  RQ3 [confirmatory]  incremental validity for human LDT RTs
        H3   ΔR² of item-level lexical evidence S over a covariate model
             (HAL and, if present, SUBTLEX frequency; length; Ortho_N; token
             count; the model's OUTPUT-level word log-probability; bigram
             frequency and lexical norms if present). The output-level
             log-probability is also tested on its own, so internal evidence is
             contrasted with the output-probability baseline of surprisal work.
  RQ4 [confirmatory H4 + exploratory]  use
        H4   antisymmetric steering along the lexicality direction at the
             selected layer, ½[m(+α) − m(−α)] of the model's own
             m = logit(YES) − logit(NO), exceeds that of random directions
             (direction fitted on the selection half, tested on the evaluation half).
        exploratory: dose-response, INLP subspace ablation, head ablation,
             attention mass / knockout, each against same-size random nulls.
  RQ5 [exploratory]   generality: normalised-depth alignment, linear CKA,
                      lexicality-probe transfer through stitching maps, scale,
                      and BASE vs INSTRUCT counterparts (chat-template input)
  RQ6 [exploratory]   task frame: carrier sentences inside / outside the prompt
  Robustness [exploratory]  prompt paraphrases, swapped answer order, stimulus
                      position; nonword RTs; layer-trajectory measures;
                      frequency DECODING beyond the LM-internal baseline.

  Layer selection for confirmatory tests is made on a stratified SELECTION
  half of the items; the confirmatory statistics are computed on the disjoint
  EVALUATION half, so "best layer" choice cannot inflate them.

CLAIM BOUNDARY
  We probe the hidden state at the first-output prediction position for
  lexicality; we do not claim that the LLM predicts WORD/NONWORD at each
  layer (decodability ≠ use; RQ4 addresses use). Layer depth is not a
  reaction time; RT alignment is a statistical association. The analysis
  plan was fixed after first results were seen and is NOT a pre-registration.

USAGE
  python LDT-NAACL.py                      # all models, then aggregation
  python LDT-NAACL.py --models A,B         # only these models (resumable)
  python LDT-NAACL.py --aggregate-only     # cross-model stage from saved results
  python LDT-NAACL.py --force              # recompute models already finished
  python LDT-NAACL.py --self-test          # unit tests of the statistics
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import string
import sys
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime

# Device pinning only when requested (index or GPU/MIG UUID).
if os.environ.get('LDT_GPU_INDEX'):
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LDT_GPU_INDEX']
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import seaborn as sns
import statsmodels.api as sm
import torch
import torch.nn.functional as F
import transformers as _transformers_pkg
from scipy import sparse, stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('ldt')

SEED = 42
MIN_TRANSFORMERS = (4, 53, 0)


def _version_tuple(v: str) -> tuple:
    parts = []
    for p in v.split('+')[0].split('.')[:3]:
        digits = ''.join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple((parts + [0, 0, 0])[:3])


TRANSFORMERS_VERSION = _version_tuple(_transformers_pkg.__version__)
if TRANSFORMERS_VERSION < MIN_TRANSFORMERS:
    logger.warning(f"transformers {_transformers_pkg.__version__} < 4.53: Qwen3 / SmolLM3 "
                   f"will not load and instrumented attention may be unavailable.")


def seed_everything(seed: int = SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

# ════════════════════════════════════════════════════════════════════════════
# TASK PROMPTS
# ════════════════════════════════════════════════════════════════════════════
# The PRIMARY template is fixed and hand-written (never optimised). Only
# {STIMULUS} varies; no label, frequency or RT information enters the prompt.
# Every template ends in "Answer:", so the decision site is the same token.
PROMPT_TEMPLATES = {
    'primary': (
        "Determine whether the following letter string is a valid English word.\n\n"
        "Letter string: {STIMULUS}\n\n"
        "Respond with exactly one answer:\nYES = English word\nNO = not an English word\n\n"
        "Answer:"),
    # ── robustness templates (I3) ───────────────────────────────────────────
    'paraphrase_question': (
        "Is the following string of letters a real English word?\n\n"
        "String: {STIMULUS}\n\n"
        "Reply YES if it is a real English word and NO if it is not.\n\n"
        "Answer:"),
    'paraphrase_terse': (
        "Lexical decision task.\nItem: {STIMULUS}\n"
        "Question: Is this item an English word? Reply YES or NO.\n"
        "Answer:"),
    'answer_order_swapped': (
        "Determine whether the following letter string is a valid English word.\n\n"
        "Letter string: {STIMULUS}\n\n"
        "Respond with exactly one answer:\nNO = not an English word\nYES = English word\n\n"
        "Answer:"),
    'stimulus_first': (
        "Letter string: {STIMULUS}\n\n"
        "Determine whether the letter string above is a valid English word.\n\n"
        "Respond with exactly one answer:\nYES = English word\nNO = not an English word\n\n"
        "Answer:"),
    'stimulus_last': (
        "Determine whether the letter string below is a valid English word.\n\n"
        "Respond with exactly one answer:\nYES = English word\nNO = not an English word\n\n"
        "Letter string: {STIMULUS}\n\n"
        "Answer:"),
}
for _name, _tpl in PROMPT_TEMPLATES.items():
    _fields = {f for _, f, _, _ in string.Formatter().parse(_tpl) if f is not None}
    if _fields != {'STIMULUS'} or _tpl.count('{STIMULUS}') != 1 or not _tpl.endswith('Answer:'):
        raise RuntimeError(f"template {_name!r} must contain one {{STIMULUS}} and end in 'Answer:'")


def build_prompt(stimulus: str, template: str = 'primary') -> tuple[str, tuple[int, int]]:
    """Prompt text and the character span of the stimulus inside it."""
    tpl = PROMPT_TEMPLATES[template]
    pre = tpl.split('{STIMULUS}')[0]
    s = str(stimulus)
    return tpl.replace('{STIMULUS}', s), (len(pre), len(pre) + len(s))


ANSWER_WORD_FORMS = ('yes',)
ANSWER_NONWORD_FORMS = ('no',)
_ANSWER_STRIP = " \t\r\n.,:;!?\"'`()[]{}*_-"


def normalize_answer_text(token_text: str) -> str:
    return token_text.strip(_ANSWER_STRIP).casefold()


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    name: str
    model_id: str
    batch_size: int = 32
    family: str = ''
    variant: str = 'base'           # 'base' | 'instruct'
    base_of: str | None = None      # for instruct variants: name of the base counterpart
    chat: bool = False              # wrap every input in the tokenizer's chat template

    def __post_init__(self):
        if not self.family:
            self.family = self.name.lower().split('-')[0].rstrip('0123456789.')


def default_models() -> list[ModelConfig]:
    base = [('SmolLM2-360M', 'HuggingFaceTB/SmolLM2-360M', 'smollm'),
            ('SmolLM3-3B-Base', 'HuggingFaceTB/SmolLM3-3B-Base', 'smollm'),
            ('Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B', 'qwen'),
            ('Qwen2.5-3B', 'Qwen/Qwen2.5-3B', 'qwen'),
            ('Qwen3-4B-Base', 'Qwen/Qwen3-4B-Base', 'qwen'),
            ('Qwen3-8B-Base', 'Qwen/Qwen3-8B-Base', 'qwen'),
            ('Llama-3.2-1B', 'meta-llama/Llama-3.2-1B', 'llama'),
            ('Llama-3.2-3B', 'meta-llama/Llama-3.2-3B', 'llama'),
            ('Llama-3.1-8B', 'meta-llama/Llama-3.1-8B', 'llama')]
    instruct = [('SmolLM2-360M-Instruct', 'HuggingFaceTB/SmolLM2-360M-Instruct', 'smollm', 'SmolLM2-360M'),
                ('Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'qwen', 'Qwen2.5-1.5B'),
                ('Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'qwen', 'Qwen2.5-3B'),
                ('Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct', 'llama', 'Llama-3.2-1B'),
                ('Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct', 'llama', 'Llama-3.2-3B')]
    return ([ModelConfig(n, m, family=f) for n, m, f in base] +
            [ModelConfig(n, m, family=f, variant='instruct', base_of=b, chat=True)
             for n, m, f, b in instruct])


@dataclass
class Config:
    # ── data (ELP; Balota et al., 2007) ─────────────────────────────────────
    WORDS_PATH: str = 'Items.csv'
    NONWORDS_PATH: str = 'NonWord.csv'
    # Lexical norms that are WORD properties (undefined for pseudowords). They
    # enter every word-level covariate set when ≥ NORM_MIN_COVERAGE of the
    # sampled words have a value. ELP columns are renamed; external files are
    # merged on the lower-cased word, e.g.
    #   {'path': 'MorphoLEX_en.csv', 'word_column': 'Word',
    #    'columns': {'PFMF': 'morph_family_frequency', 'FamSize': 'morph_family_size'}}
    #   {'path': 'AoA_Kuperman2012.csv', 'word_column': 'Word', 'columns': {'Rating.Mean': 'aoa'}}
    #   {'path': 'Concreteness_Brysbaert2014.csv', 'word_column': 'Word',
    #    'columns': {'Conc.M': 'concreteness'}}
    ELP_NORM_COLUMNS: dict = field(default_factory=lambda: {'NMorph': 'n_morphemes'})
    LEXICAL_NORMS: list[dict] = field(default_factory=list)
    NORM_MIN_COVERAGE: float = 0.8
    MATCH_ON_NORMS: list[str] = field(default_factory=lambda: ['n_morphemes', 'morph_family_size'])
    MAX_ITEMS: int | None = None         # None → all items (balanced words / nonwords)
    WORD_RT_COLUMN: str = 'I_Mean_RT'
    WORD_ACC_COLUMN: str = 'I_Mean_Accuracy'
    NONWORD_RT_COLUMN: str = 'NWI_Mean_RT'
    NONWORD_ACC_COLUMN: str = 'NWI_Mean_Accuracy'
    HAL_COLUMN: str = 'Log_Freq_HAL'
    SUBTLEX_COLUMNS: list[str] = field(default_factory=lambda: ['LgSUBTLWF', 'SUBTLWF'])
    BIGRAM_COLUMN: str = 'BG_Mean'
    HIGH_FREQ_PERCENTILE: float = 100 * 2 / 3
    LOW_FREQ_PERCENTILE: float = 100 / 3
    MODELS: list[ModelConfig] = field(default_factory=default_models)
    OUTPUT_DIR: str = 'LDT_NAACL_results'
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    REP_DTYPE: str = 'float32'           # cached hidden states; 'float16' halves RAM

    # ── probes ──────────────────────────────────────────────────────────────
    N_FOLDS: int = 5
    PROBE_C: float = 1.0                 # L2 logistic regression, intercept unpenalised
    PROBE_MAX_ITER: int = 1000
    PROBE_BACKEND: str = 'auto'          # 'auto' (torch on CUDA) | 'torch' | 'sklearn'
    MLP_HIDDEN_DIMS: list[int] = field(default_factory=lambda: [512, 256])
    MLP_EPOCHS: int = 30
    MLP_PATIENCE: int = 4
    MLP_BATCH_SIZE: int = 512
    MLP_LR: float = 1e-3
    MLP_WEIGHT_DECAY: float = 1e-2
    MLP_DROPOUT: float = 0.1
    LEXICAL_EVIDENCE_CLIP: float = 1e-6

    # ── statistics ──────────────────────────────────────────────────────────
    ALPHA: float = 0.05
    N_BOOTSTRAP: int = 2000
    SELECTION_FRACTION: float = 0.5      # items used ONLY to choose the confirmatory layer
    BF_THRESHOLD: float = 3.0
    SMD_THRESHOLD: float = 0.1
    MATCH_CALIPER_SD: float = 0.2        # Austin (2011)
    MIN_GROUP: int = 30

    # ── analysis switches ───────────────────────────────────────────────────
    RUN_MLP_PROBE: bool = True                    # A1
    RUN_RANDOM_INIT: bool = True                  # H1b
    RUN_MEAN_POOL_CONTROL: bool = True            # position control
    RUN_PROMPT_ROBUSTNESS: bool = True            # I3
    RUN_CAUSAL: bool = True                       # RQ4
    RUN_CONTEXTUAL: bool = True                   # RQ6
    RUN_INSTRUCT_VARIANTS: bool = True            # RQ5: instruct counterparts of base models
    LAYER_MODE_MLP: str = 'all'                   # 'all' | 'representative'
    LAYER_MODE_ERASURE: str = 'representative'
    LAYER_MODE_CONTROLS: str = 'representative'   # label permutation

    # ── sub-sample sizes for the expensive analyses ─────────────────────────
    PROMPT_ROBUSTNESS_N_ITEMS: int = 4000
    CAUSAL_N_ITEMS: int = 512
    CAUSAL_FIT_MAX_ITEMS: int = 20000
    CAUSAL_MAX_LAYERS: int = 4
    CAUSAL_BATCH_SIZE: int = 16
    CAUSAL_N_RANDOM: int = 50                     # I6: empirical p floor = 1/51 (exploratory)
    H4_ALPHA: float = 2.0                         # confirmatory steering magnitude (SD units)
    H4_N_NULL: int = 300                          # random directions → p floor 1/301
    STEERING_MAGNITUDES: list[float] = field(default_factory=lambda: [-4.0, -2.0, 2.0, 4.0])
    SUBSPACE_RANK: int = 4
    HEAD_TOP_K: int = 10
    CONTEXT_N_ITEMS: int = 1000
    CONTEXT_SENTENCE: str = "The {} was seen by the group."
    CONTEXT_FINAL_SENTENCE: str = "The group looked at the {}"
    CONTEXT_TASK_TEMPLATE: str = (
        "Determine whether the target letter string in the sentence below is a valid "
        "English word.\n\nSentence: {SENTENCE}\nTarget letter string: {STIMULUS}\n\n"
        "Respond with exactly one answer:\nYES = English word\nNO = not an English word\n\n"
        "Answer:")
    TRANSFER_N_ITEMS: int = 6000
    TRANSFER_TEST_FRACTION: float = 0.3
    TRANSFER_DEPTHS: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    TRANSFER_ALPHAS: list[float] = field(default_factory=lambda: [1.0, 10.0, 100.0, 1000.0, 1e4])
    TRANSFER_N_PERMUTATIONS: int = 1000
    TRANSFER_N_RANDOM_PROBES: int = 200
    CKA_N_ITEMS: int = 1000
    CURVE_GRID: int = 50

    # ── human RT (RQ3) ──────────────────────────────────────────────────────
    RT_MIN_ACCURACY: float = 0.8                  # items with ELP accuracy below are excluded
    RT_LOG_SKEW_THRESHOLD: float = 1.0
    RT_MIN_ITEMS: int = 50
    RT_MIXED_GROUP_MIN: int = 5

    def validate(self):
        def need(cond, msg):
            if not cond:
                raise ValueError(f"Config: {msg}")
        need(self.N_FOLDS >= 2, "N_FOLDS ≥ 2")
        need(self.PROBE_C > 0, "PROBE_C > 0")
        need(self.PROBE_BACKEND in ('auto', 'torch', 'sklearn'), "PROBE_BACKEND")
        need(0 < self.ALPHA < 0.5, "ALPHA in (0, 0.5)")
        need(0.1 <= self.SELECTION_FRACTION <= 0.9, "SELECTION_FRACTION in [0.1, 0.9]")
        need(0 < self.LOW_FREQ_PERCENTILE < self.HIGH_FREQ_PERCENTILE < 100, "frequency percentiles")
        need(0 < self.LEXICAL_EVIDENCE_CLIP < 0.5, "LEXICAL_EVIDENCE_CLIP")
        need(self.REP_DTYPE in ('float32', 'float16'), "REP_DTYPE")
        need(0 <= self.RT_MIN_ACCURACY < 1, "RT_MIN_ACCURACY in [0, 1)")
        need(0 < self.NORM_MIN_COVERAGE <= 1, "NORM_MIN_COVERAGE in (0, 1]")
        need(self.H4_ALPHA > 0, "H4_ALPHA > 0")
        for nd in self.LEXICAL_NORMS:
            need(isinstance(nd, dict) and {'path', 'word_column', 'columns'} <= set(nd),
                 "LEXICAL_NORMS entries need path, word_column, columns")
        need(0 < self.TRANSFER_TEST_FRACTION < 0.9, "TRANSFER_TEST_FRACTION")
        need(all(0 < d <= 1 for d in self.TRANSFER_DEPTHS), "TRANSFER_DEPTHS in (0, 1]")
        need(0 not in self.STEERING_MAGNITUDES and len(self.STEERING_MAGNITUDES) > 0,
             "STEERING_MAGNITUDES non-empty, no 0")
        for n in ('LAYER_MODE_MLP', 'LAYER_MODE_ERASURE', 'LAYER_MODE_CONTROLS'):
            need(getattr(self, n) in ('all', 'representative'), f"{n} in all|representative")
        for n in ('N_BOOTSTRAP', 'MIN_GROUP', 'PROMPT_ROBUSTNESS_N_ITEMS', 'CAUSAL_N_ITEMS',
                  'CAUSAL_MAX_LAYERS', 'CAUSAL_BATCH_SIZE', 'CAUSAL_N_RANDOM', 'SUBSPACE_RANK',
                  'HEAD_TOP_K', 'CONTEXT_N_ITEMS', 'TRANSFER_N_ITEMS', 'CKA_N_ITEMS',
                  'CURVE_GRID', 'RT_MIN_ITEMS', 'MLP_EPOCHS', 'MLP_BATCH_SIZE', 'H4_N_NULL'):
            v = getattr(self, n)
            need(isinstance(v, int) and not isinstance(v, bool) and v >= 1, f"{n} positive int")
        need(self.CONTEXT_SENTENCE.count('{}') == 1, "CONTEXT_SENTENCE needs one {}")
        need(self.CONTEXT_FINAL_SENTENCE.endswith('{}'), "CONTEXT_FINAL_SENTENCE must end with {}")
        need(self.CONTEXT_TASK_TEMPLATE.count('{SENTENCE}') == 1 and
             self.CONTEXT_TASK_TEMPLATE.count('{STIMULUS}') == 1, "CONTEXT_TASK_TEMPLATE fields")
        names = [m.name for m in self.MODELS]
        need(len(names) == len(set(names)) and names, "unique, non-empty model names")
        for m in self.MODELS:
            need(m.variant in ('base', 'instruct'), f"{m.name}: variant base|instruct")
            need(m.base_of is None or m.base_of in names, f"{m.name}: base_of not in MODELS")
        return self

    def dirs(self) -> dict:
        d = {'root': self.OUTPUT_DIR, 'models': os.path.join(self.OUTPUT_DIR, 'models'),
             'aggregate': os.path.join(self.OUTPUT_DIR, 'aggregate'),
             'figures': os.path.join(self.OUTPUT_DIR, 'figures')}
        for p in d.values():
            os.makedirs(p, exist_ok=True)
        return d

    def model_dir(self, name: str) -> str:
        p = os.path.join(self.OUTPUT_DIR, 'models', safe_name(name))
        os.makedirs(p, exist_ok=True)
        return p


def safe_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-._' else '_' for ch in str(name))


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS  (pure functions; unit-tested by --self-test)
# ════════════════════════════════════════════════════════════════════════════
_LN10 = np.log(10.0)


def finite(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(x))
    except TypeError:
        return False


def p_from_z(z) -> tuple[float, float]:
    """Two-sided normal p and log10 p from the log survival function (never 0)."""
    if not finite(z):
        return np.nan, np.nan
    lp = min(0.0, float(np.log(2.0) + stats.norm.logsf(abs(float(z)))))
    return float(np.exp(lp)), lp / _LN10


def p_from_t(t, df) -> tuple[float, float]:
    """Two-sided Student-t p and log10 p; Mills-ratio asymptote in the far tail."""
    if not (finite(t) and finite(df)) or df <= 0:
        return np.nan, np.nan
    at, nu = abs(float(t)), float(df)
    lsf = float(stats.t.logsf(at, nu))
    if not np.isfinite(lsf):
        lsf = float(stats.t.logpdf(at, nu) + np.log((nu + at * at) / ((nu + 1.0) * at)))
    lp = min(0.0, float(np.log(2.0) + lsf))
    return float(np.exp(lp)), lp / _LN10


def p_from_f(f, df1, df2) -> tuple[float, float]:
    if not (finite(f) and df1 > 0 and df2 > 0):
        return np.nan, np.nan
    lsf = float(stats.f.logsf(max(float(f), 0.0), df1, df2))
    if not np.isfinite(lsf) and df1 == 1:
        return p_from_t(np.sqrt(max(float(f), 0.0)), df2)
    lp = min(0.0, lsf)
    return float(np.exp(lp)), lp / _LN10


def p_binom_greater(k: int, n: int, p0: float) -> tuple[float, float]:
    """One-sided exact binomial P(X ≥ k)."""
    if n <= 0:
        return np.nan, np.nan
    lp = min(0.0, float(stats.binom.logsf(int(k) - 1, int(n), float(p0))))
    return float(np.exp(lp)), lp / _LN10


def p_correlation(r, n) -> tuple[float, float]:
    if not (finite(r) and n is not None and n > 2):
        return np.nan, np.nan
    r = float(np.clip(r, -1 + 1e-15, 1 - 1e-15))
    return p_from_t(r * np.sqrt((n - 2) / (1 - r * r)), n - 2)


def log10_bf01_from_t(t, df, n) -> float:
    """BIC approximation to BF01 for one regression coefficient (Wagenmakers,
    2007): ln BF01 ≈ ½ ln n − (n/2) ln(1 + t²/df). > log10(3) = moderate
    evidence for the null."""
    if not (finite(t) and finite(df) and n is not None and n > 1 and df > 0):
        return np.nan
    return float((0.5 * np.log(n) - (n / 2.0) * np.log1p(float(t) ** 2 / float(df))) / _LN10)


def log10_bf01_correlation(r, n) -> float:
    if not (finite(r) and n is not None and n > 2):
        return np.nan
    r = float(np.clip(r, -1 + 1e-15, 1 - 1e-15))
    return log10_bf01_from_t(r * np.sqrt((n - 2) / (1 - r * r)), n - 2, n)


def bf_label(log10_bf01, threshold: float = 3.0) -> str:
    if not finite(log10_bf01):
        return 'undetermined'
    t = np.log10(threshold)
    return ('evidence_for_null' if log10_bf01 >= t else
            'evidence_for_effect' if log10_bf01 <= -t else 'inconclusive')


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def bootstrap_mean_ci(x, B=2000, seed=SEED, alpha=0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of a mean (items resampled)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for s in range(0, B, 250):
        k = min(250, B - s)
        out[s:s + k] = x[rng.integers(0, len(x), size=(k, len(x)))].mean(1)
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def mcnemar_exact(c_a, c_b) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired 0/1 correctness."""
    c_a, c_b = np.asarray(c_a).astype(int), np.asarray(c_b).astype(int)
    if c_a.shape != c_b.shape:
        raise ValueError("McNemar needs paired vectors of equal length")
    b = int(np.sum((c_a == 1) & (c_b == 0)))
    c = int(np.sum((c_a == 0) & (c_b == 1)))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def adjust_pvalues(pvals, method: str = 'fdr_bh', alpha: float = 0.05) -> np.ndarray:
    """NaN-tolerant multiplicity adjustment (NaN in → NaN out)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], alpha=alpha, method=method)[1]
    return out


# ── AUROC with DeLong (1988) placement values ──────────────────────────────
def _placements(pos, neg):
    """V10[i] = P(neg < pos_i) + ½P(=);  V01[j] = P(pos > neg_j) + ½P(=)."""
    sn, sp = np.sort(np.asarray(neg, float)), np.sort(np.asarray(pos, float))
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    v10 = (np.searchsorted(sn, pos, 'left') + np.searchsorted(sn, pos, 'right')) / (2.0 * len(sn))
    v01 = (2 * len(sp) - np.searchsorted(sp, neg, 'left')
           - np.searchsorted(sp, neg, 'right')) / (2.0 * len(sp))
    return v10, v01


def auc_with_ci(pos, neg, alpha=0.05) -> dict:
    """AUROC and its DeLong standard error / Wald CI."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return {'auroc': np.nan, 'auroc_se': np.nan,
                'auroc_ci95_low': np.nan, 'auroc_ci95_high': np.nan}
    v10, v01 = _placements(pos, neg)
    auc = float(v10.mean())
    se = float(np.sqrt(np.var(v10, ddof=1) / len(pos) + np.var(v01, ddof=1) / len(neg)))
    z = stats.norm.ppf(1 - alpha / 2)
    return {'auroc': auc, 'auroc_se': se,
            'auroc_ci95_low': max(0.0, auc - z * se), 'auroc_ci95_high': min(1.0, auc + z * se)}


def delong_paired(score_a, score_b, y) -> dict:
    """AUROC(a) − AUROC(b) for two scorers on the SAME items (DeLong, 1988)."""
    y = np.asarray(y).astype(int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    pos, neg = y == 1, y == 0
    if pos.sum() < 2 or neg.sum() < 2:
        return {'auroc_a': np.nan, 'auroc_b': np.nan, 'auroc_difference': np.nan,
                'auroc_difference_se': np.nan, 'p_delong': np.nan, 'p_delong_log10': np.nan}
    v10a, v01a = _placements(a[pos], a[neg])
    v10b, v01b = _placements(b[pos], b[neg])
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = ((s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / pos.sum()
           + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / neg.sum())
    se = float(np.sqrt(max(var, 0.0)))
    d = float(v10a.mean() - v10b.mean())
    p, lp = p_from_z(d / se) if se > 0 else (np.nan, np.nan)
    return {'auroc_a': float(v10a.mean()), 'auroc_b': float(v10b.mean()), 'auroc_difference': d,
            'auroc_difference_se': se, 'p_delong': p, 'p_delong_log10': lp}


def delong_shared_negatives(pos_a, pos_b, neg) -> dict:
    """AUROC(pos_a vs neg) − AUROC(pos_b vs neg): DISJOINT positive sets sharing
    ONE negative set. Var = Var(V10_a)/n_a + Var(V10_b)/n_b + Var(V01_a − V01_b)/n_neg.
    This is the H2 statistic: frequency effect on DISCRIMINABILITY."""
    pos_a, pos_b, neg = (np.asarray(v, float) for v in (pos_a, pos_b, neg))
    out = {'auroc_hf': np.nan, 'auroc_lf': np.nan, 'delta_auroc': np.nan,
           'delta_auroc_se': np.nan, 'delta_auroc_ci95_low': np.nan,
           'delta_auroc_ci95_high': np.nan, 'p_delta_auroc': np.nan,
           'p_delta_auroc_log10': np.nan, 'n_hf': len(pos_a), 'n_lf': len(pos_b),
           'n_nonwords': len(neg)}
    if min(len(pos_a), len(pos_b)) < 2 or len(neg) < 2:
        return out
    v10a, v01a = _placements(pos_a, neg)
    v10b, v01b = _placements(pos_b, neg)
    d = float(v10a.mean() - v10b.mean())
    se = float(np.sqrt(np.var(v10a, ddof=1) / len(pos_a) + np.var(v10b, ddof=1) / len(pos_b)
                       + np.var(v01a - v01b, ddof=1) / len(neg)))
    p, lp = p_from_z(d / se) if se > 0 else (np.nan, np.nan)
    out.update({'auroc_hf': float(v10a.mean()), 'auroc_lf': float(v10b.mean()),
                'delta_auroc': d, 'delta_auroc_se': se,
                'delta_auroc_ci95_low': d - 1.959964 * se,
                'delta_auroc_ci95_high': d + 1.959964 * se,
                'p_delta_auroc': p, 'p_delta_auroc_log10': lp})
    return out


# ── regression ──────────────────────────────────────────────────────────────
def _design(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X.astype(float), has_constant='add')


def ols_term(y, X: pd.DataFrame, term: str) -> dict:
    """OLS coefficient of `term` with classical and HC3 SEs (MacKinnon &
    White, 1985), exact log10 p and a BIC Bayes factor."""
    out = {'beta': np.nan, 'se': np.nan, 'se_hc3': np.nan, 'p_hc3': np.nan,
           'p_hc3_log10': np.nan, 'r2': np.nan, 'n': int(len(X)), 'log10_bf01': np.nan}
    Xd = _design(X)
    fit = sm.OLS(np.asarray(y, float), Xd).fit()
    rob = fit.get_robustcov_results(cov_type='HC3')
    j = list(Xd.columns).index(term)
    t_hc3 = float(np.asarray(rob.tvalues)[j])
    p, lp = p_from_t(t_hc3, fit.df_resid)
    out.update({'beta': float(fit.params.iloc[j]), 'se': float(fit.bse.iloc[j]),
                'se_hc3': float(np.asarray(rob.bse)[j]), 'p_hc3': p, 'p_hc3_log10': lp,
                'r2': float(fit.rsquared),
                'log10_bf01': log10_bf01_from_t(float(fit.tvalues.iloc[j]), fit.df_resid,
                                                int(fit.nobs))})
    return out


def nested_ols(y, X_base: pd.DataFrame, X_add: pd.DataFrame) -> dict:
    """Incremental validity: R² of base vs base + added predictors, partial F
    test, and for a single added predictor its HC3 coefficient test and BIC
    Bayes factor."""
    y = np.asarray(y, float)
    if (X_add.std(ddof=0) == 0).any():
        return {'n': int(len(y)), 'delta_r2': np.nan, 'p_f': np.nan, 'p_f_log10': np.nan,
                'note': 'added predictor is constant — model not identifiable'}
    f0 = sm.OLS(y, _design(X_base)).fit()
    Xf = pd.concat([X_base, X_add], axis=1)
    f1 = sm.OLS(y, _design(Xf)).fit()
    q = X_add.shape[1]
    F = ((f0.ssr - f1.ssr) / q) / (f1.ssr / f1.df_resid)
    p, lp = p_from_f(F, q, f1.df_resid)
    out = {'n': int(f1.nobs), 'r2_base': float(f0.rsquared), 'r2_full': float(f1.rsquared),
           'delta_r2': float(f1.rsquared - f0.rsquared), 'f_statistic': float(F),
           'p_f': p, 'p_f_log10': lp}
    if q == 1:
        t = ols_term(y, Xf, X_add.columns[0])
        out.update({'beta_std': t['beta'], 'beta_se_hc3': t['se_hc3'], 'p_hc3': t['p_hc3'],
                    'p_hc3_log10': t['p_hc3_log10'], 'log10_bf01': t['log10_bf01']})
    return out


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    sd = df.std(ddof=0).replace(0, 1.0)
    return (df - df.mean()) / sd


# ── balance and matching (B1–B3) ────────────────────────────────────────────
def standardized_mean_difference(x_t, x_c, sd_ref=None) -> float:
    """(mean_t − mean_c)/SD with SD = pooled pre-matching SD when given
    (Austin, 2009; Stuart, 2010). |SMD| < 0.1 is the balance criterion."""
    x_t = np.asarray(x_t, float); x_t = x_t[np.isfinite(x_t)]
    x_c = np.asarray(x_c, float); x_c = x_c[np.isfinite(x_c)]
    if len(x_t) < 2 or len(x_c) < 2:
        return np.nan
    sd = sd_ref if finite(sd_ref) else np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    return float((x_t.mean() - x_c.mean()) / sd) if sd > 0 else 0.0


def variance_ratio(x_t, x_c) -> float:
    x_t = np.asarray(x_t, float); x_t = x_t[np.isfinite(x_t)]
    x_c = np.asarray(x_c, float); x_c = x_c[np.isfinite(x_c)]
    if len(x_t) < 2 or len(x_c) < 2 or x_c.var(ddof=1) == 0:
        return np.nan
    return float(x_t.var(ddof=1) / x_c.var(ddof=1))


def greedy_caliper_match(feat_t, feat_c, caliper_sd: float, pooled_sd, k_neighbors: int = 50):
    """1:1 nearest-neighbour matching without replacement with a per-covariate
    caliper |Δx_j| ≤ caliper·SD_j. Treated units are processed in increasing
    order of their nearest-candidate distance (order-independent); units with
    no admissible control are rejected. Returns (idx_t, idx_c, n_rejected)."""
    feat_t, feat_c = np.asarray(feat_t, float), np.asarray(feat_c, float)
    sd = np.where(np.asarray(pooled_sd, float) > 0, pooled_sd, 1.0)
    zt, zc = feat_t / sd, feat_c / sd
    if len(zt) == 0 or len(zc) == 0:
        return np.array([], int), np.array([], int), len(zt)
    k = int(min(k_neighbors, len(zc)))
    dist, ind = NearestNeighbors(n_neighbors=k).fit(zc).kneighbors(zt)
    used = np.zeros(len(zc), bool)
    mt, mc, rejected = [], [], 0
    tol = caliper_sd + 1e-12
    for i in np.lexsort((np.arange(len(zt)), dist[:, 0])):
        chosen = next((j for j in ind[i] if not used[j] and np.all(np.abs(zt[i] - zc[j]) <= tol)), -1)
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
    return np.asarray(mt, int), np.asarray(mc, int), rejected


# ── representational similarity (H2) ────────────────────────────────────────
def centered_gram(X) -> np.ndarray:
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(0, keepdims=True)
    return Xc @ Xc.T


def double_center(K) -> np.ndarray:
    K = np.asarray(K, np.float64)
    return K - K.mean(0) - K.mean(1)[:, None] + K.mean()


def linear_cka(K, L) -> float:
    """Linear CKA (Kornblith et al., 2019) from double-centred Gram matrices."""
    num = float(np.sum(K * L))
    den = float(np.sqrt(np.sum(K * K) * np.sum(L * L)))
    return num / den if den > 0 else np.nan


# ── layers ──────────────────────────────────────────────────────────────────
def representative_layers(n_layers: int) -> list[int]:
    if n_layers <= 0:
        return []
    L = list(range(n_layers))
    return sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1] + L[::4]))


def select_layers(n_layers: int, mode: str, cap: int | None = None) -> list[int]:
    layers = list(range(n_layers)) if mode == 'all' else representative_layers(n_layers)
    if cap is not None and len(layers) > cap:
        pick = np.unique(np.round(np.linspace(0, len(layers) - 1, cap)).astype(int))
        layers = [layers[i] for i in pick]
    return layers


def layer_at_depth(n_layers: int, depth: float) -> int:
    """Layer index whose relative depth (l+1)/L is closest to `depth`."""
    return int(np.clip(round(depth * n_layers) - 1, 0, n_layers - 1))


def logit(p, eps):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def stimulus_hash(s: str) -> str:
    return hashlib.md5(str(s).encode()).hexdigest()


# ════════════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════════════
def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(',', '', regex=False)
                         .replace({'#': np.nan, 'nan': np.nan, '': np.nan}), errors='coerce')


def load_items(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """One row per stimulus: ELP words and ELP pseudowords (balanced, seeded),
    lexical covariates, human RT / accuracy for BOTH classes, frequency
    tertiles (words only) and the confirmatory SELECTION / EVALUATION split."""
    wdf, ndf = pd.read_csv(cfg.WORDS_PATH), pd.read_csv(cfg.NONWORDS_PATH)
    info = {}

    def _frame(df, is_word):
        out = pd.DataFrame({'stimulus': df['Word'].astype(str).str.strip().str.lower()})
        out['is_word'] = int(is_word)
        out['length'] = _numeric(df['Length']) if 'Length' in df else out['stimulus'].str.len()
        out['ortho_n'] = _numeric(df['Ortho_N']) if 'Ortho_N' in df else np.nan
        rt_col, acc_col = ((cfg.WORD_RT_COLUMN, cfg.WORD_ACC_COLUMN) if is_word
                           else (cfg.NONWORD_RT_COLUMN, cfg.NONWORD_ACC_COLUMN))
        out['human_rt'] = _numeric(df[rt_col]) if rt_col in df else np.nan
        out['human_accuracy'] = _numeric(df[acc_col]) if acc_col in df else np.nan
        out['log_freq_hal'] = _numeric(df[cfg.HAL_COLUMN]) if (is_word and cfg.HAL_COLUMN in df) else np.nan
        out['subtlex'] = np.nan
        if is_word:
            for c in cfg.SUBTLEX_COLUMNS:
                if c in df:
                    v = _numeric(df[c])
                    # SUBTLWF is per-million: Zipf-like log10(fpm + 1/51M·1e6) + 3
                    out['subtlex'] = v if c != 'SUBTLWF' else np.log10(v + 0.0196) + 3
                    info['subtlex_column'] = c
                    break
        out['bigram_mean'] = _numeric(df[cfg.BIGRAM_COLUMN]) if cfg.BIGRAM_COLUMN in df else np.nan
        for src, dst in cfg.ELP_NORM_COLUMNS.items():
            out[dst] = _numeric(df[src]) if (is_word and src in df) else np.nan
        return out[out['stimulus'].str.fullmatch(r'[a-z]+', na=False)]

    words, nonwords = _frame(wdf, True), _frame(ndf, False)
    words = words.drop_duplicates('stimulus')
    nonwords = nonwords.drop_duplicates('stimulus')
    overlap = set(words['stimulus']) & set(nonwords['stimulus'])
    nonwords = nonwords[~nonwords['stimulus'].isin(overlap)]
    info.update({'n_words_file': len(words), 'n_nonwords_file': len(nonwords),
                 'n_word_nonword_string_overlap_removed': len(overlap),
                 'subtlex_column': info.get('subtlex_column')})
    norm_names = list(cfg.ELP_NORM_COLUMNS.values())
    for nd in cfg.LEXICAL_NORMS:
        ext = pd.read_csv(nd['path'])
        ext = pd.DataFrame({'stimulus': ext[nd['word_column']].astype(str).str.strip().str.lower(),
                            **{dst: _numeric(ext[src]) for src, dst in nd['columns'].items()}})
        words = words.drop(columns=[c for c in nd['columns'].values() if c in words])
        words = words.merge(ext.drop_duplicates('stimulus'), on='stimulus', how='left')
        norm_names += [c for c in nd['columns'].values() if c not in norm_names]
    for c in norm_names:
        nonwords[c] = np.nan                       # word properties: undefined for pseudowords

    n = min(len(words), len(nonwords))
    if cfg.MAX_ITEMS is not None:
        n = min(n, cfg.MAX_ITEMS // 2)
    items = pd.concat([words.sample(n=n, random_state=SEED),
                       nonwords.sample(n=n, random_state=SEED)], ignore_index=True)
    items = items.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    items['freq_group'] = np.where(items['is_word'] == 1, 'mid', 'nonword')
    wf = items.loc[items['is_word'] == 1, 'log_freq_hal']
    hi, lo = np.nanpercentile(wf, cfg.HIGH_FREQ_PERCENTILE), np.nanpercentile(wf, cfg.LOW_FREQ_PERCENTILE)
    items.loc[(items['is_word'] == 1) & (items['log_freq_hal'] >= hi), 'freq_group'] = 'high'
    items.loc[(items['is_word'] == 1) & (items['log_freq_hal'] <= lo), 'freq_group'] = 'low'
    items.loc[(items['is_word'] == 1) & items['log_freq_hal'].isna(), 'freq_group'] = 'unknown'

    strata = items['is_word'].astype(str) + '_' + items['freq_group'].replace('unknown', 'mid')
    sel, _ = train_test_split(np.arange(len(items)), train_size=cfg.SELECTION_FRACTION,
                              stratify=strata, random_state=SEED)
    items['split'] = 'evaluation'
    items.loc[sel, 'split'] = 'selection'
    wmask = items['is_word'] == 1
    info['word_norms_available'] = [c for c in norm_names
                                    if items.loc[wmask, c].notna().mean() >= cfg.NORM_MIN_COVERAGE]
    info['word_norms_coverage'] = {c: float(items.loc[wmask, c].notna().mean()) for c in norm_names}
    info.update({'n_items': len(items), 'n_per_class': n,
                 'hal_tertile_cutoffs': {'high_ge': float(hi), 'low_le': float(lo)},
                 'n_high': int((items['freq_group'] == 'high').sum()),
                 'n_low': int((items['freq_group'] == 'low').sum()),
                 'n_selection': int((items['split'] == 'selection').sum())})
    logger.info(f"Items: {len(items)} ({n} words + {n} pseudowords); HF={info['n_high']} "
                f"LF={info['n_low']}; selection/evaluation = "
                f"{info['n_selection']}/{len(items) - info['n_selection']}")
    return items, info


# ════════════════════════════════════════════════════════════════════════════
# INSTRUMENTED ATTENTION (attention mass and attention knockout)
# ════════════════════════════════════════════════════════════════════════════
class _AttentionContext:
    """State read by `_instrumented_attention`. Inactive → plain softmax attention."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.query_pos = None    # LongTensor (B,)   query row of interest (decision site)
        self.block = None        # BoolTensor (B, T) keys that query may NOT attend to
        self.layers = None       # set[int] | None   layers where blocking applies (None = all)
        self.capture = None      # dict[layer] -> (B, T) head-mean attention of the query row


_ATTN = _AttentionContext()
_ATTN_NAME = 'ldt_instrumented'
_ATTN_OK = None


def _instrumented_attention(module, query, key, value, attention_mask,
                            scaling=None, dropout=0.0, **kwargs):
    """GQA-aware attention over post-RoPE states (as every HF attention
    implementation receives them) that can knock out query→key edges at
    selected layers (Geva et al., 2023) and record the query row."""
    B, nH, Tq, d = query.shape
    rep = nH // key.shape[1]
    if rep > 1:
        key, value = key.repeat_interleave(rep, 1), value.repeat_interleave(rep, 1)
    Tk = key.shape[2]
    scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * (scaling or d ** -0.5)
    if attention_mask is None:
        causal = torch.ones(Tq, Tk, dtype=torch.bool, device=scores.device).tril(Tk - Tq)
        scores = scores.masked_fill(~causal, float('-inf'))
    elif attention_mask.dtype == torch.bool:
        scores = scores.masked_fill(~attention_mask[..., :Tk], float('-inf'))
    else:
        scores = scores + attention_mask[..., :Tk].float()
    li = getattr(module, 'layer_idx', None)
    ar = torch.arange(B, device=scores.device)
    if _ATTN.block is not None and (_ATTN.layers is None or li in _ATTN.layers):
        qp = _ATTN.query_pos.to(scores.device)
        rows = scores[ar, :, qp, :].masked_fill(_ATTN.block.to(scores.device)[:, None, :Tk],
                                                float('-inf'))
        scores[ar, :, qp, :] = rows
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    if _ATTN.capture is not None and li is not None:
        _ATTN.capture[li] = probs[ar, :, _ATTN.query_pos.to(probs.device), :].mean(1).cpu()
    out = torch.matmul(probs.to(value.dtype), value)
    return out.transpose(1, 2).contiguous(), None


def register_instrumented_attention() -> bool:
    global _ATTN_OK
    if _ATTN_OK is None:
        try:
            from transformers import AttentionInterface
            AttentionInterface.register(_ATTN_NAME, _instrumented_attention)
            try:
                from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
                AttentionMaskInterface.register(_ATTN_NAME, sdpa_mask)
            except ImportError:
                pass
            _ATTN_OK = True
        except Exception as e:                                       # noqa: BLE001
            logger.warning(f"instrumented attention unavailable ({e})")
            _ATTN_OK = False
    return _ATTN_OK


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_out(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


# ════════════════════════════════════════════════════════════════════════════
# FROZEN LANGUAGE MODEL
# ════════════════════════════════════════════════════════════════════════════
class LanguageModel:
    """Frozen decoder-only LM. Layer convention: layer l ≡ hidden_states[l + 1]
    (l = 0 is the first block's output; the embedding output is not analysed).
    In HF decoder-only models hidden_states[-1] is the output of the FINAL
    NORM; earlier entries are raw residual-stream states."""

    def __init__(self, mc: ModelConfig, cfg: Config, random_init: bool = False):
        self.mc, self.cfg, self.random_init = mc, cfg, bool(random_init)
        self.device = cfg.DEVICE
        tok = AutoTokenizer.from_pretrained(mc.model_id, trust_remote_code=True)
        tok.padding_side = 'left'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        if self.device == 'cuda':
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        self.dtype = dtype
        dkey = 'dtype' if TRANSFORMERS_VERSION >= (4, 56, 0) else 'torch_dtype'
        if mc.chat and not tok.chat_template:
            raise RuntimeError(f"{mc.name}: chat=True but the tokenizer has no chat template")
        if self.random_init:
            seed_everything(SEED)
            model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(mc.model_id, trust_remote_code=True),
                trust_remote_code=True, **{dkey: dtype})
            model = model.to(self.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                mc.model_id, trust_remote_code=True, low_cpu_mem_usage=True,
                device_map=('auto' if self.device == 'cuda' else None), **{dkey: dtype})
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        prefix = getattr(model, 'base_model_prefix', None)
        bb = getattr(model, prefix, None) if prefix else None
        self.backbone = bb if isinstance(bb, nn.Module) else model
        self.num_layers = int(model.config.num_hidden_layers)
        self.hidden_dim = int(model.config.hidden_size)
        self.n_parameters = int(sum(p.numel() for p in model.parameters()))
        self.position_checks = 0
        self._build_answer_readout()
        logger.info(f"Loaded {mc.name}{' [RANDOM INIT]' if random_init else ''}: "
                    f"{self.num_layers} layers, d={self.hidden_dim}, "
                    f"{self.n_parameters / 1e9:.2f}B params, {dtype}")

    # ── YES/NO readout through the model's OWN head ─────────────────────────
    def _build_answer_readout(self):
        tok = self.tokenizer
        decoded = tok.batch_decode([[i] for i in range(len(tok))], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        self.yes_ids = [i for i, t in enumerate(decoded) if normalize_answer_text(t) in ANSWER_WORD_FORMS]
        self.no_ids = [i for i, t in enumerate(decoded) if normalize_answer_text(t) in ANSWER_NONWORD_FORMS]
        if not self.yes_ids or not self.no_ids:     # first sub-token fallback, recorded
            def _first(forms):
                return sorted({tok.encode(p + f, add_special_tokens=False)[0]
                               for f in forms for p in ('', ' ')})
            self.yes_ids = self.yes_ids or _first(('YES', 'Yes', 'yes'))
            self.no_ids = self.no_ids or _first(('NO', 'No', 'no'))
            self.answer_scoring = 'first_subword_fallback'
        else:
            self.answer_scoring = 'all_single_token_surface_forms'
        head = self.model.get_output_embeddings()
        self.final_norm = next((getattr(self.backbone, a) for a in
                                ('norm', 'final_layernorm', 'ln_f', 'final_layer_norm')
                                if isinstance(getattr(self.backbone, a, None), nn.Module)), None)
        W = head.weight.detach()
        b = getattr(head, 'bias', None)
        dev = W.device
        self._yes_W = W.index_select(0, torch.tensor(self.yes_ids, device=dev)).float()
        self._no_W = W.index_select(0, torch.tensor(self.no_ids, device=dev)).float()
        self._yes_b = b.detach().index_select(0, torch.tensor(self.yes_ids, device=dev)).float() \
            if b is not None else None
        self._no_b = b.detach().index_select(0, torch.tensor(self.no_ids, device=dev)).float() \
            if b is not None else None
        self._head = head
        self.last_hidden_is_normed = True
        self.readout_check = self._verify_readout()

    def _norm_for_head(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Logit lens (nostalgebraist, 2020): apply the model's final norm to an
        intermediate residual state before the head; skip it for the last
        layer when hidden_states[-1] is already post-norm."""
        if self.final_norm is None or (layer == self.num_layers - 1 and self.last_hidden_is_normed):
            return h.float()
        dt = next(self.final_norm.parameters()).dtype
        return self.final_norm(h.to(dt)).float()

    def answer_margin(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """logit-sum-exp(YES ids) − logit-sum-exp(NO ids) at `layer`."""
        x = self._norm_for_head(h, layer).to(self._yes_W.device)
        ys = torch.logsumexp(F.linear(x, self._yes_W, self._yes_b), -1)
        ns = torch.logsumexp(F.linear(x, self._no_W, self._no_b), -1)
        return ys - ns

    @torch.no_grad()
    def _verify_readout(self) -> dict:
        """Establishes empirically whether hidden_states[-1] is post-norm and that
        the two-row YES/NO readout reproduces the model's own logits. All
        comparisons in float32 with RELATIVE error, so bf16 rounding of the
        reference logits (|logit| ≈ 30) cannot fail a correct pathway."""
        prompts = [build_prompt(w)[0] for w in ('house', 'table', 'blorpz', 'qwrtx')]
        ids, am, fp, _ = self.encode(prompts)
        out = self.model(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False)
        ar = torch.arange(ids.shape[0], device=ids.device)
        ref = out.logits[ar, fp].float()
        h = out.hidden_states[-1][ar, fp]
        W = self._head.weight.float()
        b = self._head.bias.float() if getattr(self._head, 'bias', None) is not None else None
        direct = F.linear(h.float().to(W.device), W, b)
        rel = lambda a: float((a - ref.to(a.device)).norm() / ref.norm().clamp_min(1e-6))
        e_direct = rel(direct)
        e_normed = np.inf
        if self.final_norm is not None:
            dt = next(self.final_norm.parameters()).dtype
            e_normed = rel(F.linear(self.final_norm(h.to(dt)).float().to(W.device), W, b))
        self.last_hidden_is_normed = bool(e_direct <= e_normed)
        best = min(e_direct, e_normed)
        if best > 0.05:
            raise RuntimeError(f"{self.mc.name}: LM head does not reproduce the model's logits "
                               f"(relative error direct={e_direct:.3g}, normed={e_normed:.3g})")
        m = self.answer_margin(h, self.num_layers - 1).float().cpu()
        ref_m = (torch.logsumexp(ref[:, self.yes_ids], -1) - torch.logsumexp(ref[:, self.no_ids], -1)).cpu()
        e_cols = float((m - ref_m).abs().max() / ref_m.abs().max().clamp_min(1.0))
        if e_cols > 0.05:
            raise RuntimeError(f"{self.mc.name}: YES/NO readout disagrees with the head ({e_cols:.3g})")
        return {'hidden_states_last_is_post_norm': self.last_hidden_is_normed,
                'relative_error_direct': e_direct, 'relative_error_after_norm': e_normed,
                'relative_error_yes_no_margin': e_cols, 'answer_scoring': self.answer_scoring,
                'yes_tokens': [self.tokenizer.decode([i]) for i in self.yes_ids[:10]],
                'no_tokens': [self.tokenizer.decode([i]) for i in self.no_ids[:10]]}

    # ── tokenisation and forward ────────────────────────────────────────────
    def format(self, text: str, span=None):
        """Instruct variants: the task text becomes the user turn of the model's
        own chat template with the generation prompt appended, so the decision
        site is the first assistant position. Spans are shifted accordingly."""
        if not self.mc.chat:
            return text, span
        f = self.tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                               tokenize=False, add_generation_prompt=True)
        off = f.find(text)
        if off < 0:
            raise RuntimeError(f"{self.mc.name}: chat template altered the task text")
        return f, (None if span is None else (span[0] + off, span[1] + off))

    def encode(self, texts: list[str], spans=None):
        """Left-padded batch; `fp` = decision site (last non-pad token), verified
        per row; `span_mask` marks the tokens overlapping each character span."""
        want = spans is not None
        pairs = [self.format(t, None if spans is None else spans[i]) for i, t in enumerate(texts)]
        texts = [p[0] for p in pairs]
        spans = [p[1] for p in pairs] if want else None
        enc = self.tokenizer(list(texts), return_tensors='pt', padding=True, truncation=False,
                             add_special_tokens=not self.mc.chat, return_offsets_mapping=want)
        ids = enc['input_ids'].to(self.device)
        am = enc['attention_mask'].to(self.device)
        T = am.shape[1]
        fp = (am.long() * (torch.arange(T, device=am.device) + 1)).argmax(1)
        if (am.sum(1) == 0).any() or bool((am[torch.arange(len(fp)), fp] != 1).any()):
            raise RuntimeError("decision-site position check failed")
        if bool((am.long().cumsum(1)[torch.arange(len(fp)), fp] != am.sum(1)).any()):
            raise RuntimeError("non-pad tokens after the decision site")
        self.position_checks += len(fp)
        mask = None
        if want:
            off = enc['offset_mapping']
            mask = torch.zeros_like(am, dtype=torch.bool)
            for i, sp in enumerate(spans):
                if sp is None:
                    continue
                o = off[i]
                hit = (o[:, 0] < sp[1]) & (o[:, 1] > sp[0]) & (o[:, 1] > o[:, 0]) & am[i].bool().cpu()
                mask[i] = hit.to(mask.device)
        return ids, am, fp, mask

    def hidden_states(self, ids, am):
        pos = (am.long().cumsum(-1) - 1).clamp(min=0)
        hs = self.backbone(input_ids=ids, attention_mask=am, position_ids=pos,
                           output_hidden_states=True, use_cache=False).hidden_states
        if len(hs) != self.num_layers + 1:
            raise RuntimeError(f"expected {self.num_layers + 1} hidden-state tensors, got {len(hs)}")
        return hs

    @torch.no_grad()
    def extract(self, texts: list[str], spans=None, layers=None, batch_size=None,
                mean_pool=False, desc='extract') -> dict:
        """Decision-site hidden state at every requested layer, the logit-lens
        YES/NO margin at every layer, optionally the mean over prompt tokens
        and the mean over a character span (e.g. the stimulus)."""
        layers = list(range(self.num_layers)) if layers is None else list(layers)
        dt = np.float16 if self.cfg.REP_DTYPE == 'float16' else np.float32
        bs = batch_size or self.mc.batch_size
        N = len(texts)
        reps = {li: np.empty((N, self.hidden_dim), dt) for li in layers}
        pool = {li: np.empty((N, self.hidden_dim), np.float16) for li in layers} if mean_pool else None
        span = {li: np.empty((N, self.hidden_dim), dt) for li in layers} if spans is not None else None
        margin = np.empty((N, self.num_layers), np.float32)
        n_span = np.zeros(N, int)
        for s in tqdm(range(0, N, bs), desc=f"{desc} | {self.mc.name}", leave=False):
            ids, am, fp, smask = self.encode(texts[s:s + bs], None if spans is None else spans[s:s + bs])
            hs = self.hidden_states(ids, am)
            ar = torch.arange(ids.shape[0], device=hs[1].device)
            fpd = fp.to(hs[1].device)
            for li in range(self.num_layers):
                h = hs[li + 1][ar, fpd]
                margin[s:s + len(fp), li] = self.answer_margin(h, li).cpu().numpy()
                if li in reps:
                    reps[li][s:s + len(fp)] = torch.nan_to_num(h.float()).cpu().numpy()
                    if pool is not None:
                        m = am.to(hs[li + 1].device).unsqueeze(-1).float()
                        pool[li][s:s + len(fp)] = ((hs[li + 1].float() * m).sum(1) / m.sum(1)).cpu().numpy()
                    if span is not None:
                        m = smask.to(hs[li + 1].device).unsqueeze(-1).float()
                        span[li][s:s + len(fp)] = ((hs[li + 1].float() * m).sum(1)
                                                   / m.sum(1).clamp(min=1)).cpu().numpy()
            if smask is not None:
                n_span[s:s + len(fp)] = smask.sum(1).cpu().numpy()
            del hs
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return {'reps': reps, 'mean_pool': pool, 'span': span, 'margin': margin, 'n_span_tokens': n_span}

    @torch.no_grad()
    def word_logprob(self, stimuli, batch_size=64) -> np.ndarray:
        """Model-internal frequency proxy (I4): log P(' ' + word | document start),
        summed over its sub-word tokens, with BOS (or EOS as document
        separator) as the only context. No task prompt."""
        tok = self.tokenizer
        start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        out = np.empty(len(stimuli))
        for s in range(0, len(stimuli), batch_size):
            seqs = [[start] + tok.encode(' ' + str(w), add_special_tokens=False)
                    for w in stimuli[s:s + batch_size]]
            T = max(map(len, seqs))
            ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
            am = torch.zeros_like(ids)
            for i, q in enumerate(seqs):
                ids[i, :len(q)] = torch.tensor(q)
                am[i, :len(q)] = 1
            ids, am = ids.to(self.device), am.to(self.device)
            lp = torch.log_softmax(self.model(input_ids=ids, attention_mask=am,
                                              use_cache=False).logits.float(), -1)
            tgt = lp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0] * am[:, 1:]
            out[s:s + len(seqs)] = tgt.sum(1).cpu().numpy()
        return out

    def token_counts(self, stimuli) -> np.ndarray:
        return np.array([len(self.tokenizer.encode(str(w), add_special_tokens=False))
                         for w in stimuli])

    # ── module access for interventions ─────────────────────────────────────
    def decoder_layers(self):
        for path in ('layers', 'h', 'blocks', 'decoder.layers'):
            obj = self.backbone
            try:
                for part in path.split('.'):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if isinstance(obj, nn.ModuleList) and len(obj) == self.num_layers:
                return obj
        raise RuntimeError("decoder layer list not found")

    def o_proj(self, layer: int) -> nn.Linear:
        blk = self.decoder_layers()[layer]
        attn = next(getattr(blk, n) for n in ('self_attn', 'attn', 'attention') if hasattr(blk, n))
        return next(getattr(attn, n) for n in ('o_proj', 'out_proj', 'dense', 'c_proj')
                    if isinstance(getattr(attn, n, None), nn.Linear))

    @contextmanager
    def instrumented(self):
        if not register_instrumented_attention():
            raise RuntimeError("instrumented attention not available")
        cfgs = {id(c): c for c in (self.model.config, getattr(self.backbone, 'config', None)) if c is not None}
        saved = {k: getattr(c, '_attn_implementation', None) for k, c in cfgs.items()}
        try:
            for c in cfgs.values():
                c._attn_implementation = _ATTN_NAME
            yield _ATTN
        finally:
            for k, c in cfgs.items():
                c._attn_implementation = saved[k]
            _ATTN.reset()

    def describe(self) -> dict:
        return {'model': self.mc.name, 'model_id': self.mc.model_id, 'family': self.mc.family,
                'variant': self.mc.variant, 'base_of': self.mc.base_of, 'chat_template_input': self.mc.chat,
                'random_init': self.random_init, 'num_layers': self.num_layers,
                'hidden_dim': self.hidden_dim, 'n_parameters': self.n_parameters,
                'dtype': str(self.dtype), 'tokenizer': type(self.tokenizer).__name__,
                'decision_site_position_checks_passed': self.position_checks,
                'readout_verification': self.readout_check}

    def free(self):
        del self.model, self.backbone
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
# PROBES
# ════════════════════════════════════════════════════════════════════════════
def logistic_torch(Xs: torch.Tensor, y: torch.Tensor, C: float, max_iter: int):
    """L2 logistic regression by full-batch L-BFGS on standardised features.
    Objective = sklearn's divided by C·n:  mean logloss + ‖w‖²/(2·C·n),
    intercept unpenalised → the same (unique) minimiser. Returns (w, b, converged)."""
    n, d = Xs.shape
    w = torch.zeros(d, device=Xs.device, requires_grad=True)
    b = torch.zeros(1, device=Xs.device, requires_grad=True)
    lam = 1.0 / (C * n)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-7, tolerance_change=1e-12, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xs @ w + b, y) + 0.5 * lam * (w * w).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xs @ w + b, y) + 0.5 * lam * (w * w).sum()
        loss.backward()
        converged = bool(torch.cat([w.grad, b.grad]).abs().max() < 1e-4)
    return w.detach(), b.detach(), converged


def standardize_fit(X: np.ndarray):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


class Prober:
    """Cross-fitted probes. All tables built with the same fold vector are
    paired item by item."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.n_fits = 0
        self.n_not_converged = 0

    @property
    def backend(self) -> str:
        if self.cfg.PROBE_BACKEND == 'auto':
            return 'torch' if (self.device == 'cuda' and torch.cuda.is_available()) else 'sklearn'
        return self.cfg.PROBE_BACKEND

    def folds(self, y, seed: int = SEED) -> np.ndarray:
        y = np.asarray(y).astype(int)
        fold = np.full(len(y), -1, np.int16)
        cv = StratifiedKFold(self.cfg.N_FOLDS, shuffle=True, random_state=seed)
        for k, (_, te) in enumerate(cv.split(np.zeros(len(y)), y)):
            fold[te] = k
        return fold

    def fit_linear(self, X, y):
        """Fit on X (raw) → dict(mu, sd, w, b) in the standardised space."""
        X = np.asarray(X, np.float32)
        y = np.asarray(y).astype(int)
        mu, sd = standardize_fit(X)
        self.n_fits += 1
        if self.backend == 'torch':
            dev = self.device if torch.cuda.is_available() else 'cpu'
            Xs = torch.as_tensor((X - mu) / sd, device=dev)
            w, b, ok = logistic_torch(Xs, torch.as_tensor(y, dtype=torch.float32, device=dev),
                                      self.cfg.PROBE_C, self.cfg.PROBE_MAX_ITER)
            w, b = w.double().cpu().numpy(), float(b.cpu().numpy()[0])
            del Xs
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                clf = LogisticRegression(C=self.cfg.PROBE_C, solver='lbfgs',
                                         max_iter=self.cfg.PROBE_MAX_ITER).fit((X - mu) / sd, y)
            ok = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
            w, b = clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
        self.n_not_converged += int(not ok)
        return {'mu': mu, 'sd': sd, 'w': w, 'b': b}

    @staticmethod
    def decision(probe: dict, X) -> np.ndarray:
        return ((np.asarray(X, np.float64) - probe['mu']) / probe['sd']) @ probe['w'] + probe['b']

    def _fit_predict_mlp(self, X_tr, y_tr, X_te, seed):
        """Non-linear capacity check (A1): MLP with residual blocks; scaler and
        early-stopping split fitted INSIDE the training fold."""
        cfg, dev = self.cfg, self.device
        torch.manual_seed(seed)
        tr, va = train_test_split(np.arange(len(y_tr)), test_size=0.1, stratify=y_tr, random_state=seed)
        mu, sd = standardize_fit(X_tr[tr])
        T = lambda a: torch.as_tensor((np.asarray(a, np.float32) - mu) / sd, device=dev)
        Xa, Xv, Xt = T(X_tr[tr]), T(X_tr[va]), T(X_te)
        ya = torch.as_tensor(y_tr[tr], device=dev)
        yv = torch.as_tensor(y_tr[va], device=dev)
        dims = [X_tr.shape[1]] + list(cfg.MLP_HIDDEN_DIMS)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU(), nn.Dropout(cfg.MLP_DROPOUT)]
        net = nn.Sequential(*layers, nn.Linear(dims[-1], 2)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.MLP_LR, weight_decay=cfg.MLP_WEIGHT_DECAY)
        g = torch.Generator().manual_seed(seed)
        best, state, wait = np.inf, None, 0
        with torch.enable_grad():
            for _ in range(cfg.MLP_EPOCHS):
                net.train()
                perm = torch.randperm(len(ya), generator=g).to(dev)
                for s in range(0, len(ya), cfg.MLP_BATCH_SIZE):
                    idx = perm[s:s + cfg.MLP_BATCH_SIZE]
                    opt.zero_grad()
                    F.cross_entropy(net(Xa[idx]), ya[idx]).backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vl = float(F.cross_entropy(net(Xv), yv))
                if vl < best - 1e-6:
                    best, wait = vl, 0
                    state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= cfg.MLP_PATIENCE:
                        break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            return torch.softmax(net(Xt), 1)[:, 1].cpu().numpy()

    @staticmethod
    def residualize(X_tr, X_te, Z_tr, Z_te):
        """Linear covariate erasure fitted on the training fold: X − [1,Z]B.
        Zero training cross-covariance with Z ⇔ no linear classifier can use Z
        (linear guardedness; Belrose et al., NeurIPS 2023, LEACE Thm. 3.1)."""
        A_tr = np.column_stack([np.ones(len(Z_tr)), Z_tr])
        A_te = np.column_stack([np.ones(len(Z_te)), Z_te])
        B, *_ = np.linalg.lstsq(A_tr, X_tr.astype(np.float64), rcond=None)
        return (X_tr - A_tr @ B).astype(np.float32), (X_te - A_te @ B).astype(np.float32)

    def crossfit(self, X, y, folds, kind='linear', covariates=None) -> np.ndarray:
        """Out-of-fold P(class 1) for every row."""
        X, y = np.asarray(X), np.asarray(y).astype(int)
        oof = np.full(len(y), np.nan)
        for k in np.unique(folds):
            te, tr = np.where(folds == k)[0], np.where(folds != k)[0]
            X_tr, X_te = X[tr].astype(np.float32), X[te].astype(np.float32)
            if covariates is not None:
                Z = np.asarray(covariates, float)
                Z_tr, Z_te = Z[tr].copy(), Z[te].copy()
                med = np.nanmedian(Z_tr, 0)
                for j in range(Z.shape[1]):
                    Z_tr[np.isnan(Z_tr[:, j]), j] = med[j]
                    Z_te[np.isnan(Z_te[:, j]), j] = med[j]
                zm, zs = standardize_fit(Z_tr)
                X_tr, X_te = self.residualize(X_tr, X_te, (Z_tr - zm) / zs, (Z_te - zm) / zs)
            if kind == 'linear':
                pr = self.fit_linear(X_tr, y[tr])
                oof[te] = 1.0 / (1.0 + np.exp(-np.clip(self.decision(pr, X_te), -50, 50)))
            elif kind == 'mlp':
                oof[te] = self._fit_predict_mlp(X_tr, y[tr], X_te, SEED + int(k))
            else:
                raise ValueError(kind)
        return oof

    def crossfit_ridge_r2(self, X, y, folds, baseline, alphas=(1.0, 10.0, 100.0, 1e3, 1e4)) -> dict:
        """Cross-fitted decoding of a continuous target (log frequency).
          baseline : OLS on the baseline covariates (fitted per training fold)
          hidden   : ridge on the hidden state alone
          combined : baseline OLS + ridge on h fitted to the baseline RESIDUAL
        Ridge alpha is chosen on an inner 20% split of each training fold.
        ΔR² = R²(combined) − R²(baseline) on out-of-fold predictions, tested
        with a paired t-test on per-item squared-error reductions."""
        X, y = np.asarray(X, np.float64), np.asarray(y, float)
        Zall = np.asarray(baseline, float)
        pred_h, pred_b, pred_c = (np.full(len(y), np.nan) for _ in range(3))
        for k in np.unique(folds):
            te, tr = np.where(folds == k)[0], np.where(folds != k)[0]
            mu, sd = standardize_fit(X[tr])
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            zm, zs = standardize_fit(Zall[tr])
            A_tr = np.column_stack([np.ones(len(tr)), (Zall[tr] - zm) / zs])
            A_te = np.column_stack([np.ones(len(te)), (Zall[te] - zm) / zs])
            beta, *_ = np.linalg.lstsq(A_tr, y[tr], rcond=None)
            pred_b[te] = A_te @ beta
            resid = y[tr] - A_tr @ beta
            pred_h[te] = self._ridge_fit_predict(Xtr, y[tr], Xte, alphas, int(k))
            pred_c[te] = pred_b[te] + self._ridge_fit_predict(Xtr, resid, Xte, alphas, int(k))
        ss = np.sum((y - y.mean()) ** 2)
        r2 = lambda p: float(1 - np.sum((y - p) ** 2) / ss)
        gain = (y - pred_b) ** 2 - (y - pred_c) ** 2
        p, lp = p_from_t(gain.mean() / (gain.std(ddof=1) / np.sqrt(len(gain))), len(gain) - 1)
        return {'n': int(len(y)), 'r2_baseline': r2(pred_b), 'r2_hidden': r2(pred_h),
                'r2_combined': r2(pred_c), 'delta_r2_beyond_baseline': r2(pred_c) - r2(pred_b),
                'spearman_hidden': float(stats.spearmanr(pred_h, y)[0]),
                'p_beyond_baseline': p, 'p_beyond_baseline_log10': lp}

    @staticmethod
    def _ridge_fit_predict(Xtr, ytr, Xte, alphas, seed) -> np.ndarray:
        """Ridge on standardised features; alpha chosen on an inner 20% split."""
        ym = ytr.mean()
        inner = np.random.default_rng(SEED + seed).permutation(len(ytr))
        a_, b_ = inner[len(ytr) // 5:], inner[:len(ytr) // 5]

        def solve(A, t, alpha):
            return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ t)
        best = min(alphas, key=lambda al: np.mean(
            (Xtr[b_] @ solve(Xtr[a_], ytr[a_] - ym, al) + ym - ytr[b_]) ** 2))
        return Xte @ solve(Xtr, ytr - ym, best) + ym


# ════════════════════════════════════════════════════════════════════════════
# LAYER METRICS
# ════════════════════════════════════════════════════════════════════════════
def frequency_metrics(p, y, freq) -> dict:
    """H2 statistic on one score vector: ΔAUROC (HF vs nonwords − LF vs
    nonwords, DeLong shared negatives) + HF/LF hit rates as SECONDARY."""
    p, y, freq = np.asarray(p, float), np.asarray(y).astype(int), np.asarray(freq, object)
    hf, lf, nw = (y == 1) & (freq == 'high'), (y == 1) & (freq == 'low'), y == 0
    d = delong_shared_negatives(p[hf], p[lf], p[nw])
    out = {f'freq_{k}': v for k, v in d.items()}
    pred = (p >= 0.5).astype(int)
    for g, m in (('hf', hf), ('lf', lf)):
        k, n = int(pred[m].sum()), int(m.sum())
        out[f'freq_hit_rate_{g}'] = k / n if n else np.nan
        out[f'freq_hit_rate_{g}_ci95_low'], out[f'freq_hit_rate_{g}_ci95_high'] = wilson_ci(k, n)
    out['freq_hit_rate_difference'] = out['freq_hit_rate_hf'] - out['freq_hit_rate_lf']
    return out


def lexicality_metrics(p, y, freq, cfg: Config) -> dict:
    """Held-out lexicality metrics of one out-of-fold score vector."""
    p, y = np.asarray(p, float), np.asarray(y).astype(int)
    pred = (p >= 0.5).astype(int)
    corr = (pred == y).astype(float)
    chance = float(max(y.mean(), 1 - y.mean()))
    lo, hi = bootstrap_mean_ci(corr, cfg.N_BOOTSTRAP)
    pb, pbl = p_binom_greater(int(corr.sum()), len(y), chance)
    ev = logit(p, cfg.LEXICAL_EVIDENCE_CLIP)
    out = {'n': int(len(y)), 'accuracy': float(corr.mean()), 'accuracy_ci95_low': lo,
           'accuracy_ci95_high': hi, 'chance_accuracy': chance,
           'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
           'p_vs_chance': pb, 'p_vs_chance_log10': pbl,
           'mean_evidence_words': float(ev[y == 1].mean()),
           'mean_evidence_nonwords': float(ev[y == 0].mean())}
    out.update(auc_with_ci(p[y == 1], p[y == 0]))
    out.update(frequency_metrics(p, y, freq))
    return out


def layer_table(oof: dict, y, freq, cfg: Config, tags: dict, num_layers: int) -> list[dict]:
    """One row per layer + BH-FDR across layers (vs chance; ΔAUROC)."""
    rows = []
    prev = None
    for li in sorted(oof):
        r = {**tags, 'layer': li, 'relative_depth': (li + 1) / num_layers}
        r.update(lexicality_metrics(oof[li], y, freq, cfg))
        c = ((oof[li] >= 0.5).astype(int) == np.asarray(y)).astype(int)
        if prev is not None and prev[0] == li - 1:
            r['mcnemar_p_vs_previous_layer'] = mcnemar_exact(c, prev[1])[2]
        prev = (li, c)
        rows.append(r)
    for col in ('p_vs_chance', 'freq_p_delta_auroc', 'mcnemar_p_vs_previous_layer'):
        adj = adjust_pvalues([r.get(col, np.nan) for r in rows],
                             'holm' if col.startswith('mcnemar') else 'fdr_bh', cfg.ALPHA)
        for r, v in zip(rows, adj):
            r[col + '_adj'] = v
    return rows


def paired_vs_reference(oof: dict, ref_oof: dict, y, cfg: Config) -> list[dict]:
    """Per layer: accuracy McNemar and AUROC paired DeLong of a control vs the
    reference (primary) probe on the SAME items; Holm across layers."""
    y = np.asarray(y).astype(int)
    rows = []
    for li in sorted(set(oof) & set(ref_oof)):
        ca = ((oof[li] >= 0.5).astype(int) == y).astype(int)
        cb = ((ref_oof[li] >= 0.5).astype(int) == y).astype(int)
        b_, c_, p_mc = mcnemar_exact(ca, cb)
        d = delong_paired(oof[li], ref_oof[li], y)
        rows.append({'layer': li, 'accuracy_minus_reference': float(ca.mean() - cb.mean()),
                     'mcnemar_p': p_mc, 'auroc_minus_reference': d['auroc_difference'],
                     'auroc_difference_se': d['auroc_difference_se'], 'p_delong': d['p_delong']})
    for col in ('mcnemar_p', 'p_delong'):
        for r, v in zip(rows, adjust_pvalues([r[col] for r in rows], 'holm', cfg.ALPHA)):
            r[col + '_holm'] = v
    return rows


# ════════════════════════════════════════════════════════════════════════════
# NO-HIDDEN-STATE BASELINES (H1c)
# ════════════════════════════════════════════════════════════════════════════
def baseline_features(items: pd.DataFrame, tokenizer, word_logprob, char_range=(1, 3),
                      n_features=2 ** 18) -> dict:
    """Feature sets that never touch a hidden state at the decision site:
      surface_covariates : length, token count, Ortho_N (+ missing flag)
      char_ngrams        : hashed character 1–3-grams (orthographic regularity)
      subword_tokens     : bag of the model's own sub-word ids
      word_logprob       : the model's unconditional log P(word) + length + tokens
      all_combined       : all of the above (the strongest such baseline)"""
    stim = items['stimulus'].tolist()
    on = items['ortho_n'].to_numpy(float)
    tc = items['token_count'].to_numpy(float)
    ln = items['length'].to_numpy(float)
    dense = np.column_stack([ln, tc, np.nan_to_num(on, nan=np.nanmedian(on)), np.isnan(on)])
    chars = HashingVectorizer(analyzer='char_wb', ngram_range=char_range, n_features=n_features,
                              alternate_sign=False, binary=True, norm='l2').transform(stim)
    ids = [tokenizer.encode(s, add_special_tokens=False) for s in stim]
    r = np.repeat(np.arange(len(ids)), [len(i) for i in ids])
    c = np.concatenate([np.asarray(i, int) for i in ids])
    bag = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(ids), len(tokenizer)))
    lp = np.column_stack([word_logprob, ln, tc])
    zd = lambda a: sparse.csr_matrix((a - a.mean(0)) / np.where(a.std(0) > 0, a.std(0), 1))
    return {'surface_covariates': dense, 'char_ngrams': chars, 'subword_tokens': bag,
            'word_logprob': lp,
            'all_combined': sparse.hstack([chars, bag, zd(dense), zd(lp)]).tocsr()}


def crossfit_baseline(X, y, folds, C: float) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for k in np.unique(folds):
        te, tr = folds == k, folds != k
        if sparse.issparse(X):
            # liblinear penalises the intercept; a large intercept_scaling makes
            # that penalty negligible (sparse inputs exclude lbfgs-on-dense).
            clf = LogisticRegression(C=C, solver='liblinear', intercept_scaling=100.0,
                                     max_iter=2000).fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        else:
            mu, sd = standardize_fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
            oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return oof


# ════════════════════════════════════════════════════════════════════════════
# RQ4 — DOES THE MODEL USE THE DECISION-SITE LEXICALITY REPRESENTATION?
# ════════════════════════════════════════════════════════════════════════════
def inlp_basis(prober: Prober, X, y, k: int) -> np.ndarray:
    """Iterative nullspace projection (Ravfogel et al., ACL 2020): fit, take the
    raw-space probe direction, project it out, repeat. Orthonormal (H × k')."""
    Xw = np.asarray(X, np.float64).copy()
    dirs = []
    for _ in range(k):
        pr = prober.fit_linear(Xw, y)
        g = pr['w'] / pr['sd']
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) == 0:
            break
        dirs.append(g / np.linalg.norm(g))
        Q, _ = np.linalg.qr(np.column_stack(dirs))
        Xw = Xw - (Xw @ Q) @ Q.T
    return np.linalg.qr(np.column_stack(dirs))[0] if dirs else np.zeros((X.shape[1], 0))


def raw_direction(probe: dict) -> np.ndarray:
    """logit(x) = w·((x − μ)/σ) + b has raw-space gradient w/σ → unit direction."""
    g = probe['w'] / probe['sd']
    return g / np.linalg.norm(g)


class CausalAnalyses:
    """Interventions at the DECISION SITE of ONE layer. Directions / probes are
    fitted on SELECTION-half items, effects are measured on EVALUATION-half
    items; outcome is the model's own final-layer m = logit(YES) − logit(NO).
    Every effect is compared with same-size random directions / subspaces /
    heads / tokens."""

    def __init__(self, lm: LanguageModel, items: pd.DataFrame, reps: dict, prober: Prober,
                 cfg: Config, layers: list[int], readout_layer: int):
        self.lm, self.items, self.reps, self.prober, self.cfg = lm, items, reps, prober, cfg
        self.layers, self.R = layers, readout_layer
        rng = np.random.RandomState(SEED + 101)
        y, fg = items['is_word'].to_numpy(), items['freq_group'].to_numpy()

        def take(mask, k):
            idx = np.where(mask)[0]
            return rng.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else idx
        n = cfg.CAUSAL_N_ITEMS
        is_ev = (items['split'] == 'evaluation').to_numpy()
        ev = np.concatenate([take(is_ev & (y == 1) & (fg == 'high'), n // 6),
                             take(is_ev & (y == 1) & (fg == 'low'), n // 6),
                             take(is_ev & (y == 1) & (fg == 'mid'), n // 2 - 2 * (n // 6)),
                             take(is_ev & (y == 0), n - n // 2)])
        self.ev = np.sort(ev)
        sel = np.where(~is_ev)[0]
        self.fit = np.sort(rng.choice(sel, size=min(cfg.CAUSAL_FIT_MAX_ITEMS, len(sel)), replace=False))
        self.y, self.fg = y[self.ev], fg[self.ev]
        self.texts = [build_prompt(s)[0] for s in items['stimulus'].to_numpy()[self.ev]]
        self.spans = [build_prompt(s)[1] for s in items['stimulus'].to_numpy()[self.ev]]
        self.rng = np.random.RandomState(SEED + 102)

    # ── batched forward with transient hooks; returns final-layer margins ──
    @torch.no_grad()
    def _margins(self, install=None, state_layer=None):
        lm, bs = self.lm, self.cfg.CAUSAL_BATCH_SIZE
        m_out, st_out = [], []
        for s in range(0, len(self.texts), bs):
            ids, am, fp, _ = lm.encode(self.texts[s:s + bs])
            handles = install(fp) if install else []
            try:
                hs = lm.hidden_states(ids, am)
            finally:
                for h in handles:
                    h.remove()
            ar = torch.arange(len(fp), device=hs[-1].device)
            m_out.append(lm.answer_margin(hs[-1][ar, fp.to(hs[-1].device)], lm.num_layers - 1).cpu().numpy())
            if state_layer is not None:
                st_out.append(hs[state_layer + 1][ar, fp.to(hs[-1].device)].float().cpu().numpy())
        m = np.concatenate(m_out)
        return (m, np.concatenate(st_out)) if state_layer is not None else m

    def _layer_hook(self, layer, fn):
        dec = self.lm.decoder_layers()

        def install(fp):
            def hook(mod, inp, out):
                h = _out_tensor(out).clone()
                ar = torch.arange(h.shape[0], device=h.device)
                f = fp.to(h.device)
                h[ar, f] = fn(h[ar, f].float()).to(h.dtype)
                return _replace_out(out, h)
            return [dec[layer].register_forward_hook(hook)]
        return install

    def _summary(self, m) -> dict:
        y, fg = self.y, self.fg
        hf, lf = (y == 1) & (fg == 'high'), (y == 1) & (fg == 'low')
        return {'accuracy': float(np.mean((m > 0).astype(int) == y)),
                'auroc': auc_with_ci(m[y == 1], m[y == 0])['auroc'],
                'mean_margin_words': float(m[y == 1].mean()), 'mean_margin_nonwords': float(m[y == 0].mean()),
                'hf_minus_lf_margin': float(m[hf].mean() - m[lf].mean()) if hf.any() and lf.any() else np.nan}

    @staticmethod
    def _vs_null(obs: float, null: list[float]) -> dict:
        null = np.asarray([v for v in null if finite(v)])
        if len(null) < 2:
            return {'null_mean': np.nan, 'null_sd': np.nan, 'z_vs_null': np.nan, 'p_empirical': np.nan}
        sd = null.std(ddof=1)
        return {'null_mean': float(null.mean()), 'null_sd': float(sd),
                'null_q025': float(np.percentile(null, 2.5)), 'null_q975': float(np.percentile(null, 97.5)),
                'z_vs_null': float((obs - null.mean()) / sd) if sd > 0 else np.nan,
                'p_empirical': float((1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())))
                                     / (1 + len(null)))}

    def _directions(self, layer):
        X = self.reps[layer][self.fit].astype(np.float64)
        y = self.items['is_word'].to_numpy()[self.fit]
        fg = self.items['freq_group'].to_numpy()[self.fit]
        out = {'lexicality': (X, y)}
        fm = (y == 1) & np.isin(fg, ['high', 'low'])
        if fm.sum() >= 2 * self.cfg.MIN_GROUP:
            out['frequency'] = (X[fm], (fg[fm] == 'high').astype(int))
        return X, out

    def steering(self) -> list[dict]:
        """h ← h + α·s·u at the decision site of layer l; u = raw-space probe
        direction, s = SD of training projections on u (α in SD units)."""
        m0 = self._margins()
        base = self._summary(m0)
        rows = []
        for li in tqdm(self.layers, desc=f"RQ4 steering | {self.lm.mc.name}", leave=False):
            X, spaces = self._directions(li)
            dirs = {k: raw_direction(self.prober.fit_linear(Xs, ys)) for k, (Xs, ys) in spaces.items()}
            rand = [r / np.linalg.norm(r) for r in self.rng.randn(self.cfg.CAUSAL_N_RANDOM, X.shape[1])]
            for alpha in self.cfg.STEERING_MAGNITUDES:
                def shift(u):
                    v = torch.as_tensor(alpha * float(np.std(X @ u)) * u, dtype=torch.float32)
                    return self._margins(self._layer_hook(li, lambda h: h + v.to(h.device)))
                null = [float(np.mean(shift(u) - m0)) for u in rand]
                for name, u in dirs.items():
                    d = shift(u) - m0
                    rows.append({'layer': li, 'relative_depth': (li + 1) / self.lm.num_layers,
                                 'direction': name, 'alpha_sd': alpha, 'n_items': len(d),
                                 'baseline_accuracy': base['accuracy'],
                                 'mean_delta_margin': float(d.mean()),
                                 'mean_delta_margin_words': float(d[self.y == 1].mean()),
                                 'mean_delta_margin_nonwords': float(d[self.y == 0].mean()),
                                 **self._vs_null(float(d.mean()), null)})
        return rows

    def confirmatory_steering(self) -> dict:
        """H4. At the confirmatory layer R, the antisymmetric steering effect
            T(u) = mean_i ½[m_i(h + α s u) − m_i(h − α s u)]
        for the lexicality direction u (positive = towards WORD) versus
        H4_N_NULL random unit directions scaled by their own projection SD.
        The antisymmetric contrast cancels any direction-independent effect of
        perturbing the residual stream. Because u and −u are equally likely
        under the random-direction null, that null is symmetric about 0, so
        p = (1 + #{|T_null| ≥ |T|}) / (1 + N). No item-level t-test is reported:
        it only tests T ≠ 0, which ANY direction satisfies, not specificity."""
        R, a = self.R, self.cfg.H4_ALPHA
        X, spaces = self._directions(R)
        u = raw_direction(self.prober.fit_linear(*spaces['lexicality']))

        def contrast(v):
            s = float(np.std(X @ v))
            vt = torch.as_tensor(a * s * v, dtype=torch.float32)
            mp = self._margins(self._layer_hook(R, lambda h: h + vt.to(h.device)))
            mn = self._margins(self._layer_hook(R, lambda h: h - vt.to(h.device)))
            return 0.5 * (mp - mn)
        d = contrast(u)
        null = []
        for _ in tqdm(range(self.cfg.H4_N_NULL), desc=f"H4 null | {self.lm.mc.name}", leave=False):
            r = self.rng.randn(X.shape[1])
            null.append(float(contrast(r / np.linalg.norm(r)).mean()))
        null = np.asarray(null)
        T = float(d.mean())
        lo, hi = bootstrap_mean_ci(d, self.cfg.N_BOOTSTRAP)
        return {'layer': R, 'alpha_sd': a, 'n_items': int(len(d)), 'n_null': int(len(null)),
                'T': T, 'T_ci95_low': lo, 'T_ci95_high': hi,
                'T_words': float(d[self.y == 1].mean()), 'T_nonwords': float(d[self.y == 0].mean()),
                'null_mean': float(null.mean()), 'null_sd': float(null.std(ddof=1)),
                'z_vs_null': float((T - null.mean()) / null.std(ddof=1)) if null.std(ddof=1) > 0 else np.nan,
                'p_random_direction': float((1 + np.sum(np.abs(null) >= abs(T))) / (1 + len(null)))}

    def subspace_ablation(self) -> list[dict]:
        """h ← h − UUᵀ(h − μ): mean-ablation of a rank-k INLP subspace; null =
        random orthonormal subspaces of the same rank. Erasure is checked on
        held-out fit items (post-erasure probe accuracy)."""
        m0 = self._margins()
        base = self._summary(m0)
        rows = []
        for li in tqdm(self.layers, desc=f"RQ4 subspace | {self.lm.mc.name}", leave=False):
            X, spaces = self._directions(li)
            mu = torch.as_tensor(X.mean(0), dtype=torch.float32)

            def ablate(U):
                Ut = torch.as_tensor(U, dtype=torch.float32)
                return self._margins(self._layer_hook(
                    li, lambda h: h - ((h - mu.to(h.device)) @ Ut.to(h.device)) @ Ut.to(h.device).T))
            for name, (Xs, ys) in spaces.items():
                perm = np.random.RandomState(SEED + li).permutation(len(ys))
                tr, te = perm[:int(0.8 * len(ys))], perm[int(0.8 * len(ys)):]
                U = inlp_basis(self.prober, Xs[tr], ys[tr], self.cfg.SUBSPACE_RANK)
                if U.shape[1] == 0:
                    continue
                P = lambda A: A - (A - Xs[tr].mean(0)) @ U @ U.T
                post = self.prober.fit_linear(P(Xs[tr]), ys[tr])
                post_acc = float(np.mean((Prober.decision(post, P(Xs[te])) > 0).astype(int) == ys[te]))
                s1 = self._summary(ablate(U))
                nulls = [self._summary(ablate(np.linalg.qr(self.rng.randn(X.shape[1], U.shape[1]))[0]))
                         for _ in range(self.cfg.CAUSAL_N_RANDOM)]
                key = 'accuracy' if name == 'lexicality' else 'hf_minus_lf_margin'
                rows.append({'layer': li, 'relative_depth': (li + 1) / self.lm.num_layers,
                             'subspace': name, 'rank': int(U.shape[1]),
                             'post_erasure_heldout_probe_accuracy': post_acc,
                             'outcome': key, 'baseline': base[key], 'ablated': s1[key],
                             'delta': s1[key] - base[key],
                             **self._vs_null(s1[key] - base[key], [n[key] - base[key] for n in nulls])})
        return rows

    def head_ablation(self) -> tuple[list[dict], list[dict]]:
        """Direct attribution of each head (layers ≤ R) to a FIXED lexicality
        probe at readout layer R (Elhage et al., 2021, direct path), then
        joint MEAN-ablation of the top-k heads at the decision site; outcome:
        fixed-probe accuracy at R and the model's margin; null: k random heads."""
        lm, R = self.lm, self.R
        nH = int(lm.model.config.num_attention_heads)
        y_fit = self.items['is_word'].to_numpy()[self.fit]
        probe = self.prober.fit_linear(self.reps[R][self.fit], y_fit)
        u = raw_direction(probe)
        zs = {li: [] for li in range(R + 1)}
        holder = {}

        def capture(fp):
            holder['fp'] = fp
            hs_ = []
            for li in range(R + 1):
                def pre(mod, args, li=li):
                    z = args[0]
                    zs[li].append(z[torch.arange(z.shape[0], device=z.device),
                                    holder['fp'].to(z.device)].float().cpu())
                hs_.append(lm.o_proj(li).register_forward_pre_hook(pre))
            return hs_
        self._margins(capture)
        Z = {li: torch.cat(v).numpy() for li, v in zs.items()}
        dh = Z[0].shape[1] // nH
        scores = []
        for li in range(R + 1):
            a = lm.o_proj(li).weight.detach().float().cpu().numpy().T @ u
            s = (Z[li].reshape(len(self.ev), nH, dh) * a.reshape(nH, dh)[None]).sum(-1)
            sc = s[self.y == 1].mean(0) - s[self.y == 0].mean(0)
            scores += [{'layer': li, 'head': h, 'readout_layer': R, 'direct_attribution': float(sc[h])}
                       for h in range(nH)]
        z_mean = {li: torch.as_tensor(Z[li].mean(0)) for li in Z}
        Z = None
        zs.clear()

        def ablate(heads):
            by = {}
            for li, h in heads:
                by.setdefault(li, []).append(h)

            def install(fp):
                hs_ = []
                for li, hl in by.items():
                    def pre(mod, args, li=li, hl=hl):
                        z = args[0].clone()
                        ar = torch.arange(z.shape[0], device=z.device)
                        f = fp.to(z.device)
                        for h in hl:
                            z[ar, f, h * dh:(h + 1) * dh] = z_mean[li][h * dh:(h + 1) * dh].to(z.device, z.dtype)
                        return (z,) + tuple(args[1:])
                    hs_.append(lm.o_proj(li).register_forward_pre_hook(pre))
                return hs_
            m, st = self._margins(install, state_layer=R)
            return {'probe_accuracy': float(np.mean((Prober.decision(probe, st) > 0).astype(int) == self.y)),
                    **self._summary(m)}
        k = min(self.cfg.HEAD_TOP_K, (R + 1) * nH)
        top = sorted(scores, key=lambda r: -abs(r['direct_attribution']))[:k]
        base = ablate([])
        abl = ablate([(r['layer'], r['head']) for r in top])
        allh = [(li, h) for li in range(R + 1) for h in range(nH)]
        nulls = [ablate([allh[i] for i in self.rng.choice(len(allh), k, replace=False)])
                 for _ in range(self.cfg.CAUSAL_N_RANDOM)]
        rows = []
        for key in ('probe_accuracy', 'accuracy', 'mean_margin_words', 'mean_margin_nonwords'):
            rows.append({'readout_layer': R, 'k_heads': k, 'ablation': 'mean', 'outcome': key,
                         'heads': json.dumps([(r['layer'], r['head']) for r in top]),
                         'baseline': base[key], 'ablated': abl[key], 'delta': abl[key] - base[key],
                         **self._vs_null(abl[key] - base[key], [n[key] - base[key] for n in nulls])})
        return scores, rows

    def attention(self) -> tuple[list[dict], list[dict]]:
        """(a) head-averaged attention mass from the decision site onto prompt
        segments (stimulus mass also per stimulus token — nonwords have more
        tokens); (b) knockout of decision-site → stimulus edges at one layer (or
        all) vs knockout of equally many non-stimulus, non-BOS tokens."""
        lm, L = self.lm, self.lm.num_layers
        bos = lm.tokenizer.bos_token_id
        mass = np.zeros((len(self.ev), L, 3))
        n_stim = np.zeros(len(self.ev))
        bs = self.cfg.CAUSAL_BATCH_SIZE
        with lm.instrumented() as ctx:
            for s in range(0, len(self.texts), bs):
                ids, am, fp, smask = lm.encode(self.texts[s:s + bs], self.spans[s:s + bs])
                ctx.query_pos, ctx.capture, ctx.block, ctx.layers = fp, {}, None, None
                lm.hidden_states(ids, am)
                T = ids.shape[1]
                pos = torch.arange(T)[None].expand(len(fp), T)
                sm = smask.cpu()
                fin = pos == fp.cpu()[:, None]
                rest = am.cpu().bool() & ~sm & ~fin
                if bos is not None:
                    rest &= ids.cpu() != bos
                for li in range(L):
                    a = ctx.capture[li][:, :T].float()
                    mass[s:s + len(fp), li] = np.column_stack([(a * sm).sum(1), (a * rest).sum(1),
                                                               (a * fin).sum(1)])
                n_stim[s:s + len(fp)] = sm.sum(1).numpy()

            mass_rows = []
            for li in range(L):
                for g, gm in (('word', self.y == 1), ('nonword', self.y == 0)):
                    for j, seg in enumerate(('stimulus', 'other_prompt_tokens', 'decision_site_self')):
                        mass_rows.append({'layer': li, 'relative_depth': (li + 1) / L, 'group': g,
                                          'segment': seg, 'mean_attention': float(mass[gm, li, j].mean())})
                    mass_rows.append({'layer': li, 'relative_depth': (li + 1) / L, 'group': g,
                                      'segment': 'stimulus_per_token',
                                      'mean_attention': float((mass[gm, li, 0] / np.maximum(n_stim[gm], 1)).mean())})

            def run_block(mode, layer_set):
                out = []
                for s in range(0, len(self.texts), bs):
                    ids, am, fp, smask = lm.encode(self.texts[s:s + bs], self.spans[s:s + bs])
                    blk = smask.clone() if mode == 'stimulus' else None
                    if mode == 'control':
                        blk = torch.zeros_like(smask)
                        for b in range(len(fp)):
                            cand = am[b].bool() & ~smask[b]
                            cand[fp[b]] = False
                            if bos is not None:
                                cand &= ids[b] != bos
                            ci = torch.nonzero(cand).flatten().cpu().numpy()
                            k = int(smask[b].sum())
                            if len(ci) and k:
                                blk[b, torch.as_tensor(self.rng.choice(ci, min(k, len(ci)), replace=False),
                                                       device=blk.device)] = True
                    ctx.query_pos, ctx.block, ctx.capture, ctx.layers = fp, blk, None, layer_set
                    hs = lm.hidden_states(ids, am)
                    ar = torch.arange(len(fp), device=hs[-1].device)
                    out.append(lm.answer_margin(hs[-1][ar, fp.to(hs[-1].device)], L - 1).cpu().numpy())
                ctx.block = None
                return np.concatenate(out)

            m0 = run_block('none', set())
            ko_rows = []
            for name, ls in [(str(li), {li}) for li in self.layers] + [('all', None)]:
                ms, mc = run_block('stimulus', ls), run_block('control', ls)
                ds, dc = np.abs(ms - m0), np.abs(mc - m0)
                diff = ds - dc
                p, lp = p_from_t(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))), len(diff) - 1) \
                    if diff.std(ddof=1) > 0 else (np.nan, np.nan)
                ko_rows.append({'knockout_layers': name, 'baseline_accuracy': self._summary(m0)['accuracy'],
                                'stimulus_knockout_accuracy': self._summary(ms)['accuracy'],
                                'control_knockout_accuracy': self._summary(mc)['accuracy'],
                                'mean_abs_delta_margin_stimulus': float(ds.mean()),
                                'mean_abs_delta_margin_control': float(dc.mean()),
                                'stimulus_minus_control': float(diff.mean()),
                                'p_paired_t': p, 'p_paired_t_log10': lp})
            for r, v in zip(ko_rows, adjust_pvalues([r['p_paired_t'] for r in ko_rows], 'holm')):
                r['p_paired_t_holm'] = v
        return mass_rows, ko_rows


# ════════════════════════════════════════════════════════════════════════════
# PER-MODEL STUDY
# ════════════════════════════════════════════════════════════════════════════
class ModelStudy:
    """Runs every per-model analysis and writes it to OUTPUT_DIR/models/<model>/.
    A model is complete when its DONE file exists (resumable)."""

    def __init__(self, mc: ModelConfig, cfg: Config, items: pd.DataFrame, word_norms: list[str]):
        self.mc, self.cfg = mc, cfg
        self.items = items.copy()
        self.word_norms = list(word_norms)
        self.dir = cfg.model_dir(mc.name)
        self.prober = Prober(cfg)
        self.summary: dict = {'model': mc.name, 'family': mc.family, 'variant': mc.variant,
                              'base_of': mc.base_of, 'word_norms_used': self.word_norms}

    # ── helpers ─────────────────────────────────────────────────────────────
    @property
    def y(self):
        return self.items['is_word'].to_numpy().astype(int)

    @property
    def fg(self):
        return self.items['freq_group'].to_numpy(object)

    def _save(self, name, rows):
        pd.DataFrame(rows).to_csv(os.path.join(self.dir, f'{name}.csv'), index=False)

    def _tag(self, rows, **kw):
        return [{'model': self.mc.name, **kw, **r} for r in rows]

    def _crossfit_layers(self, reps, layers, kind='linear', covariates=None, desc=''):
        oof = {}
        for li in tqdm(layers, desc=f"{desc} | {self.mc.name}", leave=False):
            oof[li] = self.prober.crossfit(reps[li], self.y, self.folds, kind, covariates)
        return oof

    def _evidence(self, p):
        return logit(p, self.cfg.LEXICAL_EVIDENCE_CLIP)

    # ── pipeline ────────────────────────────────────────────────────────────
    def run(self) -> dict:
        cfg, t0 = self.cfg, datetime.now()
        lm = LanguageModel(self.mc, cfg)
        L = self.L = lm.num_layers
        stim = self.items['stimulus'].tolist()
        self.items['token_count'] = lm.token_counts(stim)
        self.items['word_logprob'] = lm.word_logprob(stim)
        prompts, spans = zip(*[build_prompt(s) for s in stim])
        ext = lm.extract(list(prompts), list(spans), mean_pool=cfg.RUN_MEAN_POOL_CONTROL, desc='primary')
        reps, self.margin = ext['reps'], ext['margin']
        self.items['n_stimulus_tokens_in_prompt'] = ext['n_span_tokens']
        self.folds = self.prober.folds(self.y)

        # RQ1 — primary curve and confirmatory layer selection (selection half)
        self.oof = self._crossfit_layers(reps, range(L), desc='primary probe')
        self._save('layers_primary', self._tag(layer_table(self.oof, self.y, self.fg, cfg, {
            'readout': 'crossfitted_linear_probe'}, L)))
        sel = (self.items['split'] == 'selection').to_numpy()
        bal = {li: balanced_accuracy_score(self.y[sel], (p[sel] >= 0.5).astype(int))
               for li, p in self.oof.items()}
        self.best = min(li for li in bal if bal[li] == max(bal.values()))
        self.summary.update({'num_layers': L, 'hidden_dim': lm.hidden_dim,
                             'n_parameters': lm.n_parameters, 'confirmatory_layer': self.best,
                             'confirmatory_layer_relative_depth': (self.best + 1) / L,
                             'selection_balanced_accuracy': bal[self.best]})
        logger.info(f"[{self.mc.name}] confirmatory layer (selection half) = {self.best} "
                    f"(depth {(self.best + 1) / L:.2f}, bal.acc {bal[self.best]:.3f})")
        self._native_curve()
        self._baselines(lm.tokenizer)

        # live-model analyses — the model is freed only after all of them
        self._causal(lm, reps)
        if cfg.RUN_PROMPT_ROBUSTNESS:
            self._prompt_robustness(lm, reps)
        if cfg.RUN_CONTEXTUAL:
            self._contextual(lm, reps)
        self._save_artifacts(reps)
        describe = lm.describe()
        lm.free()

        self._controls(reps, ext['mean_pool'])
        self._frequency(reps)
        del reps, ext
        gc.collect()
        self._rt()
        if cfg.RUN_RANDOM_INIT:
            self._random_init(list(prompts))
        self._confirmatory()

        self.items.to_csv(os.path.join(self.dir, 'items_with_model_covariates.csv'), index=False)
        np.savez_compressed(os.path.join(self.dir, 'oof_primary.npz'),
                            stimulus=np.asarray(stim), **{f'L{li}': p.astype(np.float32)
                                                          for li, p in self.oof.items()},
                            native_margin=self.margin)
        self.summary.update({'model_description': describe, 'probe_backend': self.prober.backend,
                             'n_probe_fits': self.prober.n_fits,
                             'n_probe_fits_not_converged': self.prober.n_not_converged,
                             'runtime': str(datetime.now() - t0)})
        with open(os.path.join(self.dir, 'summary.json'), 'w') as f:
            json.dump(self.summary, f, indent=2, default=float)
        open(os.path.join(self.dir, 'DONE'), 'w').write(datetime.now().isoformat())
        return self.summary

    # ── RQ1 diagnostics ─────────────────────────────────────────────────────
    def _native_curve(self):
        """Calibration-free counterpart: the model's own logit-lens YES/NO margin
        at every layer (no probe is fitted)."""
        rows = []
        for li in range(self.L):
            m = self.margin[:, li]
            r = {'layer': li, 'relative_depth': (li + 1) / self.L, 'readout': 'native_logit_lens_margin',
                 'accuracy_margin_gt_0': float(np.mean((m > 0).astype(int) == self.y)),
                 'yes_rate': float(np.mean(m > 0))}
            r.update(auc_with_ci(m[self.y == 1], m[self.y == 0]))
            r.update(frequency_metrics(m, self.y, self.fg))
            rows.append(r)
        self._save('layers_native_logit_lens', self._tag(rows))

    def _baselines(self, tokenizer):
        """H1c — probes that never see a decision-site hidden state, same folds."""
        feats = baseline_features(self.items, tokenizer, self.items['word_logprob'].to_numpy())
        self.baseline_oof = {k: crossfit_baseline(X, self.y, self.folds, self.cfg.PROBE_C)
                             for k, X in feats.items()}
        rows = []
        best_p = self.oof[self.best]
        for k, p in self.baseline_oof.items():
            r = {'baseline': k, 'uses_decision_site_hidden_state': False}
            r.update(lexicality_metrics(p, self.y, self.fg, self.cfg))
            d = delong_paired(best_p, p, self.y)
            r.update({'primary_minus_baseline_auroc_full_sample': d['auroc_difference'],
                      'p_delong_full_sample': d['p_delong']})
            rows.append(r)
        self._save('baselines_no_hidden_state', self._tag(rows))

    # ── RQ4 / robustness / RQ6 (need the live model) ────────────────────────
    def _causal(self, lm, reps):
        layers = sorted(set(select_layers(self.L, 'representative', self.cfg.CAUSAL_MAX_LAYERS - 1))
                        | {self.best})
        ca = CausalAnalyses(lm, self.items, reps, self.prober, self.cfg, layers, self.best)
        self.h4 = ca.confirmatory_steering()
        self._save('confirmatory_h4_steering', self._tag([self.h4]))
        if not self.cfg.RUN_CAUSAL:
            return
        for name, fn in (('causal_steering', ca.steering), ('causal_subspace_ablation', ca.subspace_ablation)):
            try:
                self._save(name, self._tag(fn()))
            except Exception as e:                                   # noqa: BLE001
                logger.error(f"[{self.mc.name}] {name} failed: {e}", exc_info=True)
        try:
            scores, abl = ca.head_ablation()
            self._save('causal_head_attribution', self._tag(scores))
            self._save('causal_head_ablation', self._tag(abl))
        except Exception as e:                                       # noqa: BLE001
            logger.error(f"[{self.mc.name}] head ablation failed: {e}", exc_info=True)
        try:
            mass, ko = ca.attention()
            self._save('attention_mass', self._tag(mass))
            self._save('causal_attention_knockout', self._tag(ko))
        except Exception as e:                                       # noqa: BLE001
            logger.error(f"[{self.mc.name}] attention analyses failed: {e}", exc_info=True)

    def _subsample(self, n, seed):
        strata = self.items['is_word'].astype(str) + '_' + self.items['freq_group'].replace('unknown', 'mid')
        if n >= len(self.items):
            return np.arange(len(self.items))
        idx, _ = train_test_split(np.arange(len(self.items)), train_size=n, stratify=strata,
                                  random_state=seed)
        return np.sort(idx)

    def _prompt_robustness(self, lm, reps):
        """I3 — same items, every template; curves on identical folds; paired
        comparison with the primary template; across-template spread."""
        idx = self._subsample(self.cfg.PROMPT_ROBUSTNESS_N_ITEMS, SEED + 7)
        y, fg = self.y[idx], self.fg[idx]
        folds = self.prober.folds(y, SEED + 7)
        stim = self.items['stimulus'].to_numpy()[idx]
        oofs, rows = {}, []
        for name in PROMPT_TEMPLATES:
            if name == 'primary':
                R = {li: reps[li][idx] for li in range(self.L)}
            else:
                tx, sp = zip(*[build_prompt(s, name) for s in stim])
                R = lm.extract(list(tx), list(sp), desc=f'template {name}')['reps']
            oofs[name] = {li: self.prober.crossfit(R[li], y, folds) for li in range(self.L)}
            rows += layer_table(oofs[name], y, fg, self.cfg, {'template': name}, self.L)
            del R
        comp = []
        for name in PROMPT_TEMPLATES:
            if name != 'primary':
                comp += [{'template': name, **r} for r in
                         paired_vs_reference(oofs[name], oofs['primary'], y, self.cfg)]
        self._save('prompt_robustness_layers', self._tag(rows))
        self._save('prompt_robustness_vs_primary', self._tag(comp))
        df = pd.DataFrame(rows)
        spread = df.groupby('layer').agg(auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'),
                                         auroc_min=('auroc', 'min'), auroc_max=('auroc', 'max'),
                                         delta_auroc_freq_mean=('freq_delta_auroc', 'mean'),
                                         delta_auroc_freq_sd=('freq_delta_auroc', 'std')).reset_index()
        best = df.loc[df.groupby('template')['balanced_accuracy'].idxmax(), ['template', 'layer']]
        self.summary['prompt_robustness_best_relative_depth_by_template'] = {
            r.template: (r.layer + 1) / self.L for r in best.itertuples()}
        self._save('prompt_robustness_spread', self._tag(spread.to_dict('records')))

    def _contextual(self, lm, reps):
        """RQ6 — HF vs LF words read in carrier sentences outside / inside the
        task prompt; F3 = derangement control (labels stay, words shuffled)."""
        cfg = self.cfg
        y, fg = self.y, self.fg
        rng = np.random.RandomState(SEED + 11)
        hf, lf = np.where((y == 1) & (fg == 'high'))[0], np.where((y == 1) & (fg == 'low'))[0]
        k = min(cfg.CONTEXT_N_ITEMS // 2, len(hf), len(lf))
        if k < cfg.MIN_GROUP:
            return
        idx = np.concatenate([rng.choice(hf, k, replace=False), rng.choice(lf, k, replace=False)])
        lab = np.r_[np.ones(k, int), np.zeros(k, int)]
        words = self.items['stimulus'].to_numpy()[idx].tolist()
        perm = rng.permutation(len(words))
        while np.any(perm == np.arange(len(words))):
            perm = rng.permutation(len(words))

        def slot(tpl, w):
            pre = tpl.split('{}')[0]
            return tpl.replace('{}', w), (len(pre), len(pre) + len(w))

        def task(w):
            s, (a, b) = slot(cfg.CONTEXT_SENTENCE, w)
            pre = cfg.CONTEXT_TASK_TEMPLATE.split('{SENTENCE}')[0]
            return (cfg.CONTEXT_TASK_TEMPLATE.replace('{SENTENCE}', s).replace('{STIMULUS}', w),
                    (len(pre) + a, len(pre) + b))
        conds = {'sentence': [slot(cfg.CONTEXT_SENTENCE, w) for w in words],
                 'task_sentence': [task(w) for w in words],
                 'sentence_ending_in_target': [slot(cfg.CONTEXT_FINAL_SENTENCE, w) for w in words],
                 'derangement_control': [slot(cfg.CONTEXT_SENTENCE, words[j]) for j in perm]}
        R = {'isolated_task_decision_site': {li: reps[li][idx] for li in range(self.L)}}
        for name, items in conds.items():
            e = lm.extract([t for t, _ in items], [s for _, s in items], desc=f'context {name}')
            if name in ('sentence', 'task_sentence'):
                R[f'{name}_final_position'] = e['reps']
            R[f'{name}_target_span'] = e['span']
        folds = self.prober.folds(lab, SEED + 11)
        rows = []
        for cname, reps_c in R.items():
            for li in range(self.L):
                p = self.prober.crossfit(reps_c[li], lab, folds)
                kc = int(((p >= 0.5).astype(int) == lab).sum())
                lo, hi = wilson_ci(kc, len(lab))
                r = {'condition': cname, 'layer': li, 'relative_depth': (li + 1) / self.L,
                     'task': 'high_vs_low_frequency_words', 'n': len(lab), 'accuracy': kc / len(lab),
                     'accuracy_ci95_low': lo, 'accuracy_ci95_high': hi}
                r.update(auc_with_ci(p[lab == 1], p[lab == 0]))
                rows.append(r)
        for cname in R:
            blk = [r for r in rows if r['condition'] == cname]
            pv = [p_from_z((r['auroc'] - 0.5) / r['auroc_se'])[0] if finite(r['auroc_se']) and r['auroc_se'] > 0
                  else np.nan for r in blk]
            for r, p, q in zip(blk, pv, adjust_pvalues(pv)):
                r['p_auroc_vs_chance'], r['p_auroc_vs_chance_fdr'] = p, q
        self._save('contextual_frequency', self._tag(rows))

    def _save_artifacts(self, reps):
        """Inputs of the cross-model stage: decision-site features at fixed
        relative depths for a hash-chosen item subset (transfer) and centred
        Gram matrices for every layer on another hash-chosen subset (CKA)."""
        h = self.items['stimulus'].map(stimulus_hash).to_numpy()
        order = np.argsort(h)
        tr_idx = np.sort(order[:self.cfg.TRANSFER_N_ITEMS])
        layers = sorted({layer_at_depth(self.L, d) for d in self.cfg.TRANSFER_DEPTHS})
        np.savez_compressed(os.path.join(self.dir, 'transfer_features.npz'),
                            stimulus=self.items['stimulus'].to_numpy()[tr_idx], y=self.y[tr_idx],
                            layers=np.asarray(layers), num_layers=self.L,
                            **{f'L{li}': reps[li][tr_idx].astype(np.float16) for li in layers})
        ck = np.sort(order[:self.cfg.CKA_N_ITEMS])
        np.savez_compressed(os.path.join(self.dir, 'cka_grams.npz'),
                            stimulus=self.items['stimulus'].to_numpy()[ck],
                            **{f'L{li}': centered_gram(reps[li][ck]).astype(np.float32)
                               for li in range(self.L)})

    # ── RQ1 controls (no live model) ────────────────────────────────────────
    def _controls(self, reps, mean_pool):
        cfg = self.cfg
        if mean_pool is not None:
            oof = self._crossfit_layers(mean_pool, range(self.L), desc='mean-pool control')
            self._save('layers_mean_pool_control', self._tag(layer_table(
                oof, self.y, self.fg, cfg, {'readout': 'crossfitted_linear_probe',
                                            'representation': 'mean_over_prompt_tokens'}, self.L)))
            self._save('paired_mean_pool_vs_primary', self._tag(paired_vs_reference(oof, self.oof, self.y, cfg)))
        if cfg.RUN_MLP_PROBE:
            layers = select_layers(self.L, cfg.LAYER_MODE_MLP)
            oof = self._crossfit_layers(reps, layers, 'mlp', desc='MLP probe')
            self._save('layers_mlp_probe', self._tag(layer_table(
                oof, self.y, self.fg, cfg, {'readout': 'crossfitted_mlp_probe'}, self.L)))
            self._save('paired_mlp_vs_linear', self._tag(paired_vs_reference(oof, self.oof, self.y, cfg)))
        # label-permutation control (Hewitt & Liang, 2019): must be at chance
        perm_y = np.random.RandomState(SEED + 3).permutation(self.y)
        rows = []
        for li in select_layers(self.L, cfg.LAYER_MODE_CONTROLS):
            p = self.prober.crossfit(reps[li], perm_y, self.folds)
            rows.append({'layer': li, 'relative_depth': (li + 1) / self.L,
                         'accuracy_on_permuted_labels': float(np.mean((p >= 0.5) == perm_y)),
                         **auc_with_ci(p[perm_y == 1], p[perm_y == 0]),
                         'primary_auroc': auc_with_ci(self.oof[li][self.y == 1], self.oof[li][self.y == 0])['auroc']})
        self._save('label_permutation_control', self._tag(rows))

    # ── RQ2 — frequency ─────────────────────────────────────────────────────
    def _frequency(self, reps):
        cfg, it, y, fg = self.cfg, self.items, self.y, self.fg
        # word-level covariates: surface form, tokenisation, the model's own
        # output-level log-probability and every available lexical norm
        # (morphology, AoA, concreteness …) — G2.
        cov_names = ['length', 'ortho_n', 'token_count', 'word_logprob'] + self.word_norms
        words = (y == 1) & it[['log_freq_hal'] + cov_names].notna().all(1).to_numpy()
        # (a) continuous slope — PRIMARY effect size (no extreme-group inflation)
        rows = []
        for li in range(self.L):
            S = self._evidence(self.oof[li])[words]
            Z = zscore_frame(it.loc[words, ['log_freq_hal'] + cov_names].reset_index(drop=True))
            r = {'layer': li, 'relative_depth': (li + 1) / self.L, 'n_words': int(words.sum())}
            t = ols_term(S, Z, 'log_freq_hal')
            r.update({f'hal_{k}': v for k, v in t.items() if k != 'n'})
            Zi = Z.assign(hal_x_length=Z['log_freq_hal'] * Z['length'])
            t2 = ols_term(S, Zi, 'hal_x_length')
            r.update({'hal_x_length_beta': t2['beta'], 'hal_x_length_p_hc3': t2['p_hc3'],
                      'hal_x_length_log10_bf01': t2['log10_bf01']})
            sm_ = words & it['subtlex'].notna().to_numpy()
            if sm_.sum() > 0.8 * words.sum():
                Zs = zscore_frame(it.loc[sm_, ['subtlex'] + cov_names].reset_index(drop=True))
                t3 = ols_term(self._evidence(self.oof[li])[sm_], Zs, 'subtlex')
                r.update({'subtlex_beta': t3['beta'], 'subtlex_p_hc3': t3['p_hc3']})
            rows.append(r)
        for col in ('hal_p_hc3', 'hal_x_length_p_hc3'):
            for r, q in zip(rows, adjust_pvalues([r[col] for r in rows])):
                r[col + '_fdr'] = q
        self._save('frequency_continuous_slope', self._tag(rows))

        # (b) caliper matching on length / Ortho_N / token count and, when
        #     supplied, morphology (B1–B3). NOT on word_logprob: it is itself a
        #     frequency estimate. Erasure (c) uses only covariates defined for
        #     pseudowords too.
        ecov = ['length', 'ortho_n', 'token_count']
        mcov = ecov + [n for n in cfg.MATCH_ON_NORMS if n in self.word_norms]
        hfi, lfi = np.where(fg == 'high')[0], np.where(fg == 'low')[0]
        feat = it[mcov].to_numpy(float)
        med = np.nanmedian(feat[np.r_[hfi, lfi]], 0)
        feat = np.where(np.isnan(feat), med, feat)
        psd = np.sqrt((feat[hfi].var(0, ddof=1) + feat[lfi].var(0, ddof=1)) / 2)
        mt, mc, rej = greedy_caliper_match(feat[lfi], feat[hfi], cfg.MATCH_CALIPER_SD, psd)
        m_lf, m_hf = lfi[mt], hfi[mc]
        bal = []
        for stage, a, b in (('before', hfi, lfi), ('after', m_hf, m_lf)):
            for j, c in enumerate(mcov):
                xa, xb = it[c].to_numpy(float)[a], it[c].to_numpy(float)[b]
                ks = stats.ks_2samp(xa[np.isfinite(xa)], xb[np.isfinite(xb)]) if len(a) > 1 else None
                smd = standardized_mean_difference(xa, xb, psd[j])
                bal.append({'stage': stage, 'covariate': c, 'n_pairs_or_groups': len(a), 'smd': smd,
                            'balanced': bool(finite(smd) and abs(smd) < cfg.SMD_THRESHOLD),
                            'variance_ratio': variance_ratio(xa, xb),
                            'ks_p': float(ks.pvalue) if ks is not None else np.nan,
                            'n_rejected_by_caliper': rej, 'caliper_sd': cfg.MATCH_CALIPER_SD})
        self._save('frequency_matching_balance', self._tag(bal))
        mrows = []
        if len(m_hf) >= cfg.MIN_GROUP:
            for li in range(self.L):
                p = self.oof[li]
                d = delong_shared_negatives(p[m_hf], p[m_lf], p[y == 0])
                mrows.append({'layer': li, 'relative_depth': (li + 1) / self.L, **d})
            for r, q in zip(mrows, adjust_pvalues([r['p_delta_auroc'] for r in mrows])):
                r['p_delta_auroc_fdr'] = q
        self._save('frequency_matched_delta_auroc', self._tag(mrows))

        # (c) covariate-erased probe (B4): lexicality and ΔAUROC without any
        #     linear information about length / Ortho_N / token count.
        Zc = it[ecov].to_numpy(float)
        layers = select_layers(self.L, cfg.LAYER_MODE_ERASURE) + [self.best]
        oof_e = self._crossfit_layers(reps, sorted(set(layers)), covariates=Zc, desc='covariate-erased')
        self._save('layers_covariate_erased', self._tag(layer_table(
            oof_e, y, fg, cfg, {'representation': 'decision_site_minus_linear_covariates'}, self.L)))
        self._save('paired_erased_vs_primary', self._tag(paired_vs_reference(oof_e, self.oof, y, cfg)))

        # (d) token-count strata (D4): ΔAUROC among single-token and among
        #     multi-token words (negatives: all nonwords).
        srows = []
        tc = it['token_count'].to_numpy()
        for li in range(self.L):
            p = self.oof[li]
            for name, m in (('single_token_words', tc == 1), ('multi_token_words', tc > 1)):
                d = delong_shared_negatives(p[(fg == 'high') & m], p[(fg == 'low') & m], p[y == 0])
                srows.append({'layer': li, 'relative_depth': (li + 1) / self.L, 'stratum': name, **d})
        self._save('frequency_token_strata', self._tag(srows))

        # (e) frequency DECODING from the decision site, and BEYOND the
        #     LM-internal baseline (G5): does h carry frequency information that
        #     the model's own output log-probability plus surface covariates and
        #     norms do not?
        drows = []
        wl = words & it['log_freq_hal'].notna().to_numpy()
        hal = it.loc[wl, 'log_freq_hal']
        f_w = self.prober.folds((hal > hal.median()).astype(int).to_numpy())
        Zb = it.loc[wl, cov_names].to_numpy(float)
        for li in select_layers(self.L, cfg.LAYER_MODE_CONTROLS):
            drows.append({'layer': li, 'relative_depth': (li + 1) / self.L,
                          'baseline_covariates': '+'.join(cov_names),
                          **self.prober.crossfit_ridge_r2(reps[li][wl], hal.to_numpy(), f_w, baseline=Zb)})
        for r, q in zip(drows, adjust_pvalues([r['p_beyond_baseline'] for r in drows])):
            r['p_beyond_baseline_fdr'] = q
        self._save('frequency_decoding_beyond_lm_baseline', self._tag(drows))

    # ── RQ3 — human RT ──────────────────────────────────────────────────────
    def _rt_covariates(self, word: bool) -> list[str]:
        it = self.items[self.items['is_word'] == int(word)]
        base = ['length', 'ortho_n', 'token_count', 'word_logprob']
        opt = (['log_freq_hal', 'subtlex', 'bigram_mean'] + self.word_norms) if word else ['bigram_mean']
        return base + [c for c in opt if it[c].notna().mean() >= 0.8]

    def _rt_sample(self, word: bool, covs: list[str]) -> np.ndarray:
        it = self.items
        m = ((it['is_word'] == int(word)) & (it['human_rt'] > 0)
             & (it['human_accuracy'] >= self.cfg.RT_MIN_ACCURACY) & it[covs].notna().all(1))
        return m.to_numpy()

    def _log_rt(self, word: bool) -> bool:
        """log RT iff |skew| of the class's valid RTs exceeds the threshold —
        decided once per class from human data only (model-independent)."""
        it = self.items
        rt = it.loc[(it['is_word'] == int(word)) & (it['human_rt'] > 0)
                    & (it['human_accuracy'] >= self.cfg.RT_MIN_ACCURACY), 'human_rt']
        return bool(abs(stats.skew(rt.to_numpy(float))) > self.cfg.RT_LOG_SKEW_THRESHOLD)

    def _rt_outcome(self, m):
        rt = self.items.loc[m, 'human_rt'].to_numpy(float)
        use_log = self._log_rt(bool(self.items.loc[m, 'is_word'].iloc[0]))
        return (np.log(rt) if use_log else rt), use_log

    def _incremental(self, m, covs, predictor: np.ndarray) -> dict:
        y_rt, use_log = self._rt_outcome(m)
        Xb = zscore_frame(self.items.loc[m, covs].reset_index(drop=True))
        Xa = pd.DataFrame({'S': predictor[m]})
        out = nested_ols(y_rt, Xb, Xa if Xa['S'].std() == 0 else zscore_frame(Xa))
        if not finite(out['delta_r2']):
            return {**out, 'spearman_rho': np.nan, 'log_rt': use_log}
        rho, _ = stats.spearmanr(predictor[m], self.items.loc[m, 'human_rt'])
        out.update({'spearman_rho': float(rho), 'log_rt': use_log,
                    'log10_bf01_label': bf_label(out.get('log10_bf01'), self.cfg.BF_THRESHOLD)})
        return out

    def _rt(self):
        cfg = self.cfg
        wc, nc = self._rt_covariates(True), self._rt_covariates(False)
        wm, nm = self._rt_sample(True, wc), self._rt_sample(False, nc)
        self.summary.update({'rt_word_covariates': wc, 'rt_nonword_covariates': nc,
                             'rt_n_words': int(wm.sum()), 'rt_n_nonwords': int(nm.sum())})
        rows = []
        for li in range(self.L):
            S, M = self._evidence(self.oof[li]), self.margin[:, li]
            for target, m, covs in (('words', wm, wc), ('nonwords', nm, nc)):
                if m.sum() < cfg.RT_MIN_ITEMS:
                    continue
                for pred_name, pred in (('probe_evidence_S', S), ('native_margin', M)):
                    rows.append({'layer': li, 'relative_depth': (li + 1) / self.L, 'items': target,
                                 'predictor': pred_name, 'covariates': '+'.join(covs),
                                 **self._incremental(m, covs, pred)})
        for key in {(r['items'], r['predictor']) for r in rows}:
            blk = [r for r in rows if (r['items'], r['predictor']) == key]
            for r, q in zip(blk, adjust_pvalues([r['p_f'] for r in blk])):
                r['p_f_fdr_within_model'] = q
        self._save('rt_incremental_validity', self._tag(rows))

        # internal evidence vs the OUTPUT-level probability baseline (G5): the
        # model's unconditional word log-probability (the quantity surprisal
        # studies use) against the same covariates without it, next to the
        # internal evidence S at the confirmatory layer with and without it.
        S_best = self._evidence(self.oof[self.best])
        lp = self.items['word_logprob'].to_numpy(float)
        comp = []
        for target, m, covs in (('words', wm, wc), ('nonwords', nm, nc)):
            if m.sum() < cfg.RT_MIN_ITEMS:
                continue
            c0 = [c for c in covs if c != 'word_logprob']
            for label, cv, pred in (('output_logprob_over_covariates', c0, lp),
                                    ('internal_S_over_covariates', c0, S_best),
                                    ('internal_S_over_covariates_and_output_logprob', covs, S_best)):
                comp.append({'items': target, 'comparison': label, 'layer': self.best,
                             'covariates': '+'.join(cv), **self._incremental(m, cv, pred)})
        self._save('rt_internal_vs_output_probability', self._tag(comp))

        # trajectory measures over NORMALISED depth (I2; exploratory)
        d = (np.arange(self.L) + 1) / self.L
        S_all = np.column_stack([self._evidence(self.oof[li]) for li in range(self.L)])
        traj_rows = []
        for src, A in (('probe_evidence_S', S_all), ('native_margin', self.margin)):
            meas = {'area_over_depth': scipy.integrate.trapezoid(A, d, axis=1),
                    'peak_depth': d[np.argmax(A, axis=1)],
                    'first_positive_depth': np.where((A > 0).any(1), d[np.argmax(A > 0, axis=1)], np.nan)}
            for mname, v in meas.items():
                m = wm & np.isfinite(v)
                if m.sum() >= cfg.RT_MIN_ITEMS:
                    traj_rows.append({'source': src, 'measure': mname, **self._incremental(m, wc, v)})
        for r, q in zip(traj_rows, adjust_pvalues([r['p_f'] for r in traj_rows])):
            r['p_f_fdr'] = q
        self._save('rt_trajectory_measures', self._tag(traj_rows))

        # mixed-model sensitivity at the confirmatory layer: (1 | word length)
        S = self._evidence(self.oof[self.best])
        g = self.items.loc[wm].assign(S=S[wm])
        y_rt, use_log = self._rt_outcome(wm)
        fixed = [c for c in wc if c != 'length'] + ['S']
        if g['length'].nunique() >= cfg.RT_MIXED_GROUP_MIN:
            X = sm.add_constant(zscore_frame(g[fixed].reset_index(drop=True)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = sm.MixedLM(y_rt, X.to_numpy(), groups=g['length'].to_numpy()).fit(reml=True)
            j = list(X.columns).index('S')
            fe, se = np.asarray(fit.fe_params), np.asarray(fit.bse_fe)
            p, lp = p_from_z(fe[j] / se[j])
            self._save('rt_mixed_model', self._tag([{
                'layer': self.best, 'formula': f"RT ~ {' + '.join(fixed)} + (1 | length)",
                'beta_S_std': float(fe[j]), 'se': float(se[j]), 'p': p, 'p_log10': lp,
                'group_variance': float(np.asarray(fit.cov_re).ravel()[0]),
                'converged': bool(fit.converged), 'log_rt': use_log, 'n': int(wm.sum())}]))

    # ── H1b — random-initialisation control ─────────────────────────────────
    def _random_init(self, prompts):
        lm = LanguageModel(self.mc, self.cfg, random_init=True)
        reps = lm.extract(prompts, desc='random-init')['reps']
        lm.free()
        self.oof_random = self._crossfit_layers(reps, range(self.L), desc='random-init probe')
        del reps
        self._save('layers_random_init', self._tag(layer_table(
            self.oof_random, self.y, self.fg, self.cfg, {'weights': 'random_init'}, self.L)))
        self._save('paired_trained_vs_random_init', self._tag(
            paired_vs_reference(self.oof, self.oof_random, self.y, self.cfg)))

    # ── confirmatory tests on the EVALUATION half at the selected layer ──────
    def _confirmatory(self):
        cfg, it = self.cfg, self.items
        ev = (it['split'] == 'evaluation').to_numpy()
        sel = ~ev
        y, fg, Lb = self.y, self.fg, self.best
        p = self.oof[Lb]
        rows = []
        a = auc_with_ci(p[ev & (y == 1)], p[ev & (y == 0)])
        pz, lz = p_from_z((a['auroc'] - 0.5) / a['auroc_se'])
        rows.append({'hypothesis': 'H1a', 'test': 'AUROC > 0.5 (DeLong z)', 'estimate': a['auroc'],
                     'ci95_low': a['auroc_ci95_low'], 'ci95_high': a['auroc_ci95_high'], 'p': pz, 'p_log10': lz})
        if getattr(self, 'oof_random', None) is not None:
            d = delong_paired(p[ev], self.oof_random[Lb][ev], y[ev])
            rows.append({'hypothesis': 'H1b', 'test': 'AUROC trained − random-init (paired DeLong)',
                         'estimate': d['auroc_difference'],
                         'ci95_low': d['auroc_difference'] - 1.959964 * d['auroc_difference_se'],
                         'ci95_high': d['auroc_difference'] + 1.959964 * d['auroc_difference_se'],
                         'p': d['p_delong'], 'p_log10': d['p_delong_log10']})
        best_b = max(self.baseline_oof, key=lambda k: auc_with_ci(
            self.baseline_oof[k][sel & (y == 1)], self.baseline_oof[k][sel & (y == 0)])['auroc'])
        d = delong_paired(p[ev], self.baseline_oof[best_b][ev], y[ev])
        rows.append({'hypothesis': 'H1c', 'test': f'AUROC primary − best no-hidden-state baseline ({best_b})',
                     'estimate': d['auroc_difference'],
                     'ci95_low': d['auroc_difference'] - 1.959964 * d['auroc_difference_se'],
                     'ci95_high': d['auroc_difference'] + 1.959964 * d['auroc_difference_se'],
                     'p': d['p_delong'], 'p_log10': d['p_delong_log10']})
        h = delong_shared_negatives(p[ev & (fg == 'high')], p[ev & (fg == 'low')], p[ev & (y == 0)])
        rows.append({'hypothesis': 'H2', 'test': 'AUROC(HF vs NW) − AUROC(LF vs NW) (DeLong)',
                     'estimate': h['delta_auroc'], 'ci95_low': h['delta_auroc_ci95_low'],
                     'ci95_high': h['delta_auroc_ci95_high'], 'p': h['p_delta_auroc'],
                     'p_log10': h['p_delta_auroc_log10']})
        wc = self._rt_covariates(True)
        m = self._rt_sample(True, wc) & ev
        if m.sum() >= cfg.RT_MIN_ITEMS:
            r = self._incremental(m, wc, self._evidence(p))
            rows.append({'hypothesis': 'H3', 'test': f"ΔR² of S over {'+'.join(wc)} (partial F)",
                         'estimate': r['delta_r2'], 'ci95_low': np.nan, 'ci95_high': np.nan,
                         'p': r['p_f'], 'p_log10': r['p_f_log10'], 'beta_std_S': r.get('beta_std'),
                         'p_hc3_S': r.get('p_hc3'), 'log10_bf01': r.get('log10_bf01'),
                         'bf01_label': r.get('log10_bf01_label'), 'n': r['n']})
        h4 = self.h4
        rows.append({'hypothesis': 'H4', 'test': (f"antisymmetric steering ½[m(+{h4['alpha_sd']}σu) − "
                                                  f"m(−{h4['alpha_sd']}σu)] vs {h4['n_null']} random directions"),
                     'estimate': h4['T'], 'ci95_low': h4['T_ci95_low'], 'ci95_high': h4['T_ci95_high'],
                     'p': h4['p_random_direction'], 'p_log10': np.log10(h4['p_random_direction']),
                     'z_vs_null': h4['z_vs_null'], 'n': h4['n_items']})
        for r, q in zip(rows, adjust_pvalues([r['p'] for r in rows], 'holm', cfg.ALPHA)):
            r.update({'layer': Lb, 'relative_depth': (Lb + 1) / self.L, 'items': 'evaluation_half',
                      'p_holm_within_model': q, 'supported': bool(finite(q) and q < cfg.ALPHA
                                                                  and r['estimate'] > 0)})
        self._save('confirmatory_tests', self._tag(rows))
        self.summary['confirmatory'] = {r['hypothesis']: {'estimate': r['estimate'], 'p_holm': r['p_holm_within_model'],
                                                          'supported': r['supported']} for r in rows}


# ════════════════════════════════════════════════════════════════════════════
# CROSS-MODEL STAGE (reads only what the per-model stage saved)
# ════════════════════════════════════════════════════════════════════════════
class Aggregator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.d = cfg.dirs()
        self.models = [m for m in cfg.MODELS
                       if os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'models', safe_name(m.name), 'DONE'))]
        self.summaries = {m.name: json.load(open(os.path.join(cfg.model_dir(m.name), 'summary.json')))
                          for m in self.models}

    def table(self, name) -> pd.DataFrame:
        parts = [pd.read_csv(p) for m in self.models
                 if os.path.exists(p := os.path.join(self.cfg.model_dir(m.name), f'{name}.csv'))
                 and os.path.getsize(p) > 1]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def save(self, df: pd.DataFrame, name: str):
        df.to_csv(os.path.join(self.d['aggregate'], f'{name}.csv'), index=False)

    def run(self):
        if not self.models:
            logger.warning("aggregation: no finished models")
            return
        logger.info(f"aggregating {len(self.models)} models: {[m.name for m in self.models]}")
        steps = [('confirmatory', self.confirmatory), ('global FDR', self.global_fdr),
                 ('scaling', self.scaling), ('base vs instruct', self.base_vs_instruct),
                 ('curve alignment', self.curve_alignment),
                 ('CKA', self.cka), ('transfer', self.transfer), ('figures', self.figures)]
        for name, fn in steps:
            try:
                fn()
            except Exception as e:                                       # noqa: BLE001
                logger.error(f"aggregation step {name} failed: {e}", exc_info=True)

    # ── confirmatory family across models ───────────────────────────────────
    def variant(self, model: str) -> str:
        return self.summaries[model].get('variant', 'base')

    def confirmatory(self):
        """Holm across models per hypothesis, separately for base models (the
        headline family) and instruct variants (robustness family)."""
        c = self.table('confirmatory_tests')
        if c.empty:
            return
        c['variant'] = c['model'].map(self.variant)
        c['p_holm_across_models'] = np.nan
        for _, g in c.groupby(['variant', 'hypothesis']):
            c.loc[g.index, 'p_holm_across_models'] = adjust_pvalues(g['p'].to_numpy(), 'holm', self.cfg.ALPHA)
        c['supported_across_models'] = (c['p_holm_across_models'] < self.cfg.ALPHA) & (c['estimate'] > 0)
        self.save(c, 'confirmatory_tests_all_models')
        s = c.groupby(['variant', 'hypothesis']).agg(n_models=('model', 'nunique'),
                                        n_supported_within_model=('supported', 'sum'),
                                        n_supported_across_models=('supported_across_models', 'sum'),
                                        median_estimate=('estimate', 'median')).reset_index()
        self.save(s, 'confirmatory_summary')
        logger.info("confirmatory summary:\n" + s.to_string(index=False))

    # ── G4/E4 — global BH-FDR over every (model, layer) test ────────────────
    def global_fdr(self):
        lp = self.table('layers_primary')
        if not lp.empty:
            lp['freq_p_delta_auroc_fdr_global'] = adjust_pvalues(lp['freq_p_delta_auroc'].to_numpy())
            lp['p_vs_chance_fdr_global'] = adjust_pvalues(lp['p_vs_chance'].to_numpy())
            self.save(lp, 'layers_primary_all_models')
        rt = self.table('rt_incremental_validity')
        if not rt.empty:
            rt['p_f_fdr_global'] = np.nan
            for _, g in rt.groupby(['items', 'predictor']):
                rt.loc[g.index, 'p_f_fdr_global'] = adjust_pvalues(g['p_f'].to_numpy())
            self.save(rt, 'rt_incremental_validity_all_models')
        for name in ('frequency_continuous_slope', 'frequency_matched_delta_auroc',
                     'frequency_decoding_beyond_lm_baseline', 'rt_internal_vs_output_probability',
                     'confirmatory_h4_steering',
                     'layers_random_init', 'baselines_no_hidden_state', 'causal_steering',
                     'causal_subspace_ablation', 'causal_head_ablation', 'causal_attention_knockout',
                     'prompt_robustness_spread', 'contextual_frequency', 'rt_trajectory_measures',
                     'layers_native_logit_lens', 'layers_mlp_probe', 'label_permutation_control'):
            t = self.table(name)
            if not t.empty:
                self.save(t, f'{name}_all_models')

    # ── H1/RQ5 — scale (descriptive) ────────────────────────────────────────
    def scaling(self):
        c = self.table('confirmatory_tests')
        if c.empty:
            return
        a = c[c['hypothesis'] == 'H1a'].set_index('model')
        rows = [{'model': m, 'family': s['family'], 'log10_params': np.log10(s['n_parameters']),
                 'evaluation_auroc': a.loc[m, 'estimate'] if m in a.index else np.nan,
                 'confirmatory_relative_depth': s['confirmatory_layer_relative_depth']}
                for m, s in self.summaries.items() if self.variant(m) == 'base']
        d = pd.DataFrame(rows)
        self.save(d, 'scaling_points')
        if len(d) < 4:
            return
        out = []
        for metric in ('evaluation_auroc', 'confirmatory_relative_depth'):
            t = ols_term(d[metric], d[['log10_params']], 'log10_params')
            rho, p = stats.spearmanr(d['log10_params'], d[metric])
            r = {'metric': metric, 'n_models': len(d), 'slope_per_decade': t['beta'],
                 'se_hc3': t['se_hc3'], 'p_hc3': t['p_hc3'], 'spearman_rho': rho, 'spearman_p': p,
                 'note': 'descriptive: few models, size confounded with family / data / tokens'}
            if (d['family'].value_counts() >= 2).sum() >= 2:
                X = pd.get_dummies(d['family'], drop_first=True, dtype=float).assign(log10_params=d['log10_params'])
                tf = ols_term(d[metric], X, 'log10_params')
                r.update({'slope_within_family': tf['beta'], 'p_hc3_within_family': tf['p_hc3']})
            out.append(r)
        self.save(pd.DataFrame(out), 'scaling_regression')

    # ── RQ5 — base vs instruct counterparts (I9) ────────────────────────────
    def base_vs_instruct(self):
        """For every instruct variant whose base model finished: difference of
        each confirmatory estimate, of the peak AUROC and of the depth of the
        peak, plus the correlation of the two normalised-depth AUROC curves."""
        c, lp = self.table('confirmatory_tests'), self.table('layers_primary')
        if c.empty or lp.empty:
            return
        rows = []
        for m, s in self.summaries.items():
            b = s.get('base_of')
            if self.variant(m) != 'instruct' or b not in self.summaries:
                continue
            r = {'instruct_model': m, 'base_model': b}
            for h in sorted(c['hypothesis'].unique()):
                ei = c[(c['model'] == m) & (c['hypothesis'] == h)]['estimate']
                eb = c[(c['model'] == b) & (c['hypothesis'] == h)]['estimate']
                if len(ei) and len(eb):
                    r.update({f'{h}_base': float(eb.iloc[0]), f'{h}_instruct': float(ei.iloc[0]),
                              f'{h}_instruct_minus_base': float(ei.iloc[0] - eb.iloc[0])})
            gi = lp[lp['model'] == m].sort_values('layer')
            gb = lp[lp['model'] == b].sort_values('layer')
            grid = np.linspace(0, 1, self.cfg.CURVE_GRID)
            ci = np.interp(grid, gi['relative_depth'], gi['auroc'])
            cb = np.interp(grid, gb['relative_depth'], gb['auroc'])
            r.update({'peak_auroc_base': float(gb['auroc'].max()), 'peak_auroc_instruct': float(gi['auroc'].max()),
                      'peak_depth_base': float(gb.loc[gb['auroc'].idxmax(), 'relative_depth']),
                      'peak_depth_instruct': float(gi.loc[gi['auroc'].idxmax(), 'relative_depth']),
                      'curve_r_first_differences': float(np.corrcoef(np.diff(ci), np.diff(cb))[0, 1])})
            rows.append(r)
        if rows:
            self.save(pd.DataFrame(rows), 'base_vs_instruct')

    # ── RQ5 — alignment of normalised-depth curves ──────────────────────────
    def curve_alignment(self):
        lp = self.table('layers_primary')
        if lp.empty or lp['model'].nunique() < 2:
            return
        grid = np.linspace(0, 1, self.cfg.CURVE_GRID)
        curves = {m: np.interp(grid, (g['layer'] / max(g['layer'].max(), 1)).to_numpy(), g['auroc'].to_numpy())
                  for m, g in lp.sort_values('layer').groupby('model')}
        names = list(curves)
        rows = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                da, db = np.diff(curves[a]), np.diff(curves[b])
                rows.append({'model_a': a, 'model_b': b,
                             'r_levels': float(np.corrcoef(curves[a], curves[b])[0, 1]),
                             'r_first_differences': float(np.corrcoef(da, db)[0, 1]),
                             'rmse': float(np.sqrt(np.mean((curves[a] - curves[b]) ** 2)))})
        self.save(pd.DataFrame(rows), 'curve_alignment')

    # ── RQ5 — linear CKA between models ─────────────────────────────────────
    def cka(self):
        grams = {}
        for m in self.models:
            p = os.path.join(self.cfg.model_dir(m.name), 'cka_grams.npz')
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                grams[m.name] = (list(z['stimulus']), {int(k[1:]): z[k] for k in z.files if k.startswith('L')})
        rows, summ = [], []
        names = list(grams)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa, ga = grams[a]
                sb, gb = grams[b]
                common = sorted(set(sa) & set(sb))
                ia = np.array([sa.index(s) for s in common])
                ib = np.array([sb.index(s) for s in common])
                La, Lb = len(ga), len(gb)
                Kb = {lb: double_center(gb[lb][np.ix_(ib, ib)]) for lb in gb}
                mat = np.zeros((La, Lb))
                for la in ga:
                    Ka = double_center(ga[la][np.ix_(ia, ia)])
                    for lb in gb:
                        mat[la, lb] = linear_cka(Ka, Kb[lb])
                        rows.append({'model_a': a, 'layer_a': la, 'model_b': b, 'layer_b': lb,
                                     'depth_a': (la + 1) / La, 'depth_b': (lb + 1) / Lb, 'cka': mat[la, lb]})
                diag = [mat[la, layer_at_depth(Lb, (la + 1) / La)] for la in range(La)]
                summ.append({'model_a': a, 'model_b': b, 'n_items': len(common),
                             'mean_cka_matched_depth': float(np.mean(diag)), 'max_cka': float(mat.max())})
        if rows:
            self.save(pd.DataFrame(rows), 'cka_layer_pairs')
            self.save(pd.DataFrame(summ), 'cka_summary')

    # ── RQ5 — lexicality-probe transfer through stitching maps ──────────────
    def transfer(self):
        """For ordered model pairs and matched relative depths: a ridge map
        target → source is fitted on paired FIT items, the SOURCE lexicality
        probe (fit items) is applied to mapped TARGET test items, and compared
        with the target's own probe on the SAME test items (McNemar, paired
        DeLong). Nulls (closed form, exact): shuffled pairing, and random
        probe directions of equal norm (is the source probe's direction
        special, or would any readout of the shared subspace transfer?)."""
        cfg = self.cfg
        feats = {}
        for m in self.models:
            p = os.path.join(cfg.model_dir(m.name), 'transfer_features.npz')
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                L = int(z['num_layers'])
                feats[m.name] = {'stim': list(z['stimulus']), 'y': z['y'], 'L': L,
                                 'X': {(int(k[1:]) + 1) / L: z[k].astype(np.float64)
                                       for k in z.files if k.startswith('L') and k != 'layers'}}
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        prober = Prober(cfg)
        rows, rng = [], np.random.RandomState(SEED + 97)
        names = list(feats)
        for src in names:
            for tgt in names:
                if src == tgt:
                    continue
                A, B = feats[src], feats[tgt]
                common = sorted(set(A['stim']) & set(B['stim']))
                ia = np.array([A['stim'].index(s) for s in common])
                ib = np.array([B['stim'].index(s) for s in common])
                y = A['y'][ia]
                is_test = np.array([int(stimulus_hash(s)[:8], 16) % 1000 < 1000 * cfg.TRANSFER_TEST_FRACTION
                                    for s in common])
                fit, te = ~is_test, is_test
                for depth in cfg.TRANSFER_DEPTHS:
                    da = min(A['X'], key=lambda k: abs(k - depth))
                    db = min(B['X'], key=lambda k: abs(k - depth))
                    Xs, Xt = A['X'][da][ia], B['X'][db][ib]
                    r = self._transfer_one(Xs, Xt, y, fit, te, prober, dev, rng)
                    rows.append({'source_model': src, 'target_model': tgt, 'requested_depth': depth,
                                 'source_depth': da, 'target_depth': db, 'n_fit': int(fit.sum()),
                                 'n_test': int(te.sum()), **r})
        if rows:
            df = pd.DataFrame(rows)
            for col in ('p_permutation', 'p_random_probe', 'mcnemar_p_transfer_vs_native'):
                df[col + '_fdr'] = adjust_pvalues(df[col].to_numpy())
            self.save(df, 'transfer_lexicality_probe')

    def _transfer_one(self, Xs, Xt, y, fit, te, prober, dev, rng) -> dict:
        cfg = self.cfg
        src_probe = prober.fit_linear(Xs[fit], y[fit])
        nat_probe = prober.fit_linear(Xt[fit], y[fit])
        z_nat = Prober.decision(nat_probe, Xt[te])
        mu_t, sd_t = standardize_fit(Xt[fit])
        T = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
        S, S_te = T((Xt[fit] - mu_t) / sd_t), T((Xt[te] - mu_t) / sd_t)
        Ybar = Xs[fit].mean(0)
        Yc = T(Xs[fit] - Ybar)
        n = S.shape[0]
        perm = rng.permutation(n)
        tr, va = perm[: int(0.8 * n)], perm[int(0.8 * n):]
        lam, V = torch.linalg.eigh(S[tr].T @ S[tr])
        proj_tr = V.T @ (S[tr].T @ (Yc[tr] - Yc[tr].mean(0)))

        def r2(alpha):
            W = V @ (proj_tr / (lam + alpha)[:, None])
            pred = S[va] @ W + Yc[tr].mean(0)
            return float(1 - ((pred - Yc[va]) ** 2).sum() / ((Yc[va] - Yc[va].mean(0)) ** 2).sum())
        r2s = {a: r2(a) for a in cfg.TRANSFER_ALPHAS}
        alpha = max(r2s, key=r2s.get)
        lam, V = torch.linalg.eigh(S.T @ S)
        H = S_te @ V @ torch.diag(1.0 / (lam + alpha)) @ V.T @ S.T          # (n_test, n_fit)
        u = T(src_probe['w'] / src_probe['sd'])
        c0 = float(((Ybar - src_probe['mu']) / src_probe['sd']) @ src_probe['w'] + src_probe['b'])
        q = Yc @ u
        z_tr = (H @ q).cpu().numpy() + c0
        yt = y[te]
        auc = lambda z: auc_with_ci(z[yt == 1], z[yt == 0])['auroc']
        a_tr, a_nat = auc(z_tr), auc(z_nat)
        c_tr, c_nat = ((z_tr > 0).astype(int) == yt), ((z_nat > 0).astype(int) == yt)
        d = delong_paired(z_tr, z_nat, yt)
        perm_auc = []
        for s in range(0, cfg.TRANSFER_N_PERMUTATIONS, 200):
            k = min(200, cfg.TRANSFER_N_PERMUTATIONS - s)
            Q = torch.stack([q[torch.as_tensor(rng.permutation(n), device=q.device)] for _ in range(k)], 1)
            Z = (H @ Q).cpu().numpy()
            perm_auc += [auc(Z[:, j]) for j in range(k)]
        rp_auc = []
        wn = np.linalg.norm(src_probe['w'])
        for s in range(0, cfg.TRANSFER_N_RANDOM_PROBES, 200):
            k = min(200, cfg.TRANSFER_N_RANDOM_PROBES - s)
            Wr = rng.randn(len(src_probe['w']), k)
            Wr *= wn / np.linalg.norm(Wr, axis=0, keepdims=True)
            Z = (H @ (Yc @ T(Wr / src_probe['sd'][:, None]))).cpu().numpy()
            rp_auc += [max(auc(Z[:, j]), 1 - auc(Z[:, j])) for j in range(k)]
        return {'stitching_alpha': alpha, 'stitching_val_r2': r2s[alpha],
                'transfer_auroc': a_tr, 'native_auroc': a_nat,
                'transfer_accuracy': float(c_tr.mean()), 'native_accuracy': float(c_nat.mean()),
                'transfer_minus_native_auroc': d['auroc_difference'], 'p_delong_transfer_vs_native': d['p_delong'],
                'mcnemar_p_transfer_vs_native': mcnemar_exact(c_tr, c_nat)[2],
                'permutation_null_auroc_mean': float(np.mean(perm_auc)),
                'p_permutation': float((1 + np.sum(np.asarray(perm_auc) >= a_tr)) / (1 + len(perm_auc))),
                'random_probe_null_auroc_mean': float(np.mean(rp_auc)),
                'random_probe_null_auroc_q95': float(np.percentile(rp_auc, 95)),
                'p_random_probe': float((1 + np.sum(np.asarray(rp_auc) >= a_tr)) / (1 + len(rp_auc)))}


    # ── figures ─────────────────────────────────────────────────────────────
    def figures(self):
        fd = self.d['figures']
        pal = dict(zip([m.name for m in self.models],
                       sns.color_palette('husl', max(len(self.models), 1))))
        lp_all, rnd = self.table('layers_primary'), self.table('layers_random_init')
        base = self.table('baselines_no_hidden_state')
        lp = lp_all[lp_all['model'].map(self.variant) == 'base'] if not lp_all.empty else lp_all
        depth = 'relative_depth'
        if not lp.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            for m, g in lp.groupby('model'):
                g = g.sort_values('layer')
                ax.plot(g[depth], g['auroc'], '-', color=pal[m], lw=2, label=m)
                ax.fill_between(g[depth], g['auroc_ci95_low'], g['auroc_ci95_high'], color=pal[m], alpha=.15)
                r = rnd[rnd['model'] == m].sort_values('layer') if not rnd.empty else pd.DataFrame()
                if not r.empty:
                    ax.plot(r[depth], r['auroc'], ':', color=pal[m], lw=1.5)
                b = base[base['model'] == m] if not base.empty else pd.DataFrame()
                if not b.empty:
                    ax.axhline(b['auroc'].max(), color=pal[m], ls='--', lw=0.8, alpha=.7)
            ax.axhline(0.5, color='grey', lw=0.8)
            ax.set(xlabel='relative depth (layer + 1) / L', ylabel='out-of-fold AUROC (words vs pseudowords)',
                   title='Lexicality at the decision site\nsolid: trained · dotted: random init · '
                         'dashed: best no-hidden-state baseline')
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout(); fig.savefig(os.path.join(fd, 'F1_lexicality_by_depth.png'), dpi=250); plt.close(fig)

            fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
            slope = self.table('frequency_continuous_slope')
            for m, g in lp.groupby('model'):
                g = g.sort_values('layer')
                a1.plot(g[depth], g['freq_delta_auroc'], color=pal[m], lw=2, label=m)
                a1.fill_between(g[depth], g['freq_delta_auroc_ci95_low'], g['freq_delta_auroc_ci95_high'],
                                color=pal[m], alpha=.15)
                s = slope[slope['model'] == m].sort_values('layer') if not slope.empty else pd.DataFrame()
                if not s.empty:
                    a2.plot(s[depth], s['hal_beta'], color=pal[m], lw=2)
                    a2.fill_between(s[depth], s['hal_beta'] - 1.96 * s['hal_se_hc3'],
                                    s['hal_beta'] + 1.96 * s['hal_se_hc3'], color=pal[m], alpha=.15)
            for ax in (a1, a2):
                ax.axhline(0, color='grey', lw=0.8)
                ax.set_xlabel('relative depth')
            a1.set(ylabel='AUROC(HF vs NW) − AUROC(LF vs NW)', title='H2: frequency effect on discriminability')
            a2.set(ylabel='β log-frequency on lexical evidence (words; HC3 95% CI)',
                   title='Continuous frequency slope, covariate-adjusted')
            a1.legend(fontsize=7, ncol=2)
            fig.tight_layout(); fig.savefig(os.path.join(fd, 'F2_frequency_effect.png'), dpi=250); plt.close(fig)

        rt = self.table('rt_incremental_validity')
        if not rt.empty:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
            for ax, items in zip(axes, ('words', 'nonwords')):
                for m, g in rt[(rt['items'] == items) & (rt['predictor'] == 'probe_evidence_S')].groupby('model'):
                    g = g.sort_values('layer')
                    ax.plot(g[depth], g['delta_r2'], color=pal[m], lw=2, label=m)
                ax.set(xlabel='relative depth', ylabel='ΔR² of lexical evidence over covariates',
                       title=f'H3: incremental validity for ELP RT ({items})')
            axes[0].legend(fontsize=7, ncol=2)
            fig.tight_layout(); fig.savefig(os.path.join(fd, 'F3_rt_incremental_validity.png'), dpi=250); plt.close(fig)

        c = self.table('confirmatory_tests')
        c = c[c['model'].map(self.variant) == 'base'] if not c.empty else c
        if not c.empty:
            hyps = sorted(c['hypothesis'].unique())
            fig, axes = plt.subplots(1, len(hyps), figsize=(4 * len(hyps), 0.45 * c['model'].nunique() + 2))
            for ax, h in zip(np.atleast_1d(axes), hyps):
                g = c[c['hypothesis'] == h].reset_index(drop=True)
                lo = (g['estimate'] - g['ci95_low']).clip(lower=0).fillna(0)
                hi = (g['ci95_high'] - g['estimate']).clip(lower=0).fillna(0)
                ax.errorbar(g['estimate'], np.arange(len(g)), xerr=[lo, hi], fmt='o',
                            color='#1A535C', capsize=3)
                ax.set_yticks(np.arange(len(g)))
                ax.set_yticklabels(g['model'], fontsize=7)
                ax.axvline(0.5 if h == 'H1a' else 0, color='grey', lw=0.8)
                ax.set_title(h, fontsize=10)
            fig.suptitle('Confirmatory estimates (evaluation half, selected layer, 95% CI)')
            fig.tight_layout(); fig.savefig(os.path.join(fd, 'F4_confirmatory_forest.png'), dpi=250); plt.close(fig)

        st = self.table('causal_steering')
        if not st.empty:
            st = st[st['direction'] == 'lexicality']
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for m, g in st.groupby('model'):
                best = self.summaries[m]['confirmatory_layer']
                g = g[g['layer'] == best].sort_values('alpha_sd')
                ax.plot(g['alpha_sd'], g['mean_delta_margin'], 'o-', color=pal[m], label=m)
                ax.fill_between(g['alpha_sd'], g['null_q025'], g['null_q975'], color=pal[m], alpha=.12)
            ax.axhline(0, color='grey', lw=0.8)
            ax.set(xlabel='steering magnitude α (SD of projection)', ylabel='Δ logit(YES) − logit(NO)',
                   title='RQ4: steering along the lexicality direction (band: random-direction 95%)')
            ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(fd, 'F5_steering.png'), dpi=250); plt.close(fig)

        pr = self.table('prompt_robustness_spread')
        if not pr.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            for m, g in pr.groupby('model'):
                g = g.sort_values('layer')
                x = (g['layer'] + 1) / self.summaries[m]['num_layers']
                ax.plot(x, g['auroc_mean'], color=pal[m], lw=2, label=m)
                ax.fill_between(x, g['auroc_min'], g['auroc_max'], color=pal[m], alpha=.15)
            ax.set(xlabel='relative depth', ylabel='AUROC (mean; band = min–max over templates)',
                   title='Robustness to prompt paraphrase, answer order and stimulus position')
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout(); fig.savefig(os.path.join(fd, 'F6_prompt_robustness.png'), dpi=250); plt.close(fig)

        pairs = [(m, s['base_of']) for m, s in self.summaries.items()
                 if self.variant(m) == 'instruct' and s.get('base_of') in self.summaries]
        if pairs and not lp_all.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            for m, b in pairs:
                for name, ls in ((b, '-'), (m, '--')):
                    g = lp_all[lp_all['model'] == name].sort_values('layer')
                    ax.plot(g[depth], g['auroc'], ls, color=pal[b], lw=2, label=name)
            ax.axhline(0.5, color='grey', lw=0.8)
            ax.set(xlabel='relative depth', ylabel='out-of-fold AUROC',
                   title='Base (solid) vs instruct counterpart (dashed, chat-template input)')
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout(); fig.savefig(os.path.join(fd, 'F8_base_vs_instruct.png'), dpi=250); plt.close(fig)

        p = os.path.join(self.d['aggregate'], 'cka_summary.csv')
        if os.path.exists(p):
            s = pd.read_csv(p)
            names = sorted(set(s['model_a']) | set(s['model_b']))
            M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
            for r in s.itertuples():
                M.loc[r.model_a, r.model_b] = M.loc[r.model_b, r.model_a] = r.mean_cka_matched_depth
            fig, ax = plt.subplots(figsize=(1.0 * len(names) + 4, 0.9 * len(names) + 3))
            sns.heatmap(M, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1, ax=ax)
            ax.set_title('Linear CKA at matched relative depth')
            fig.tight_layout(); fig.savefig(os.path.join(fd, 'F7_cka.png'), dpi=250); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# METADATA
# ════════════════════════════════════════════════════════════════════════════
ANALYSIS_PLAN = {
    'status': ('Fixed after first results were seen; NOT a pre-registration. Headline claims '
               'come only from the confirmatory family.'),
    'confirmatory': {
        'H1a': 'AUROC > 0.5 at the selected layer (evaluation half)',
        'H1b': 'AUROC trained − random-init > 0 (paired DeLong)',
        'H1c': 'AUROC primary − best no-hidden-state baseline > 0 (paired DeLong; baseline chosen on selection half)',
        'H2': 'AUROC(HF vs NW) − AUROC(LF vs NW) > 0 (DeLong, shared negatives)',
        'H3': 'ΔR² of lexical evidence over the covariate model > 0 (partial F; accuracy-filtered words)',
        'H4': ('antisymmetric steering along the lexicality direction at the selected layer > random '
               'directions (direction fitted on the selection half; effect on evaluation-half items)'),
        'layer_selection': 'max balanced accuracy on the SELECTION half (ties → shallowest)',
        'multiplicity': ('Holm within model across H1a–H4; Holm across models per hypothesis, separately '
                         'for base models (headline) and instruct variants (robustness)'),
    },
    'exploratory': ['all layer-wise curves (BH-FDR within model, global BH-FDR across models)',
                    'continuous frequency slope (effect size), matching, erasure, token strata',
                    'frequency decoding beyond the LM-internal baseline',
                    'internal evidence vs output log-probability for RT; nonword RTs; native margin; '
                    'trajectory measures; mixed model',
                    'MLP and mean-pool controls, label permutation',
                    'RQ4 dose-response, subspace / head ablation, attention knockout',
                    'RQ5 (incl. base vs instruct)', 'RQ6', 'prompt robustness'],
}


def claim_boundary(info: dict) -> dict:
    """What can and cannot be claimed, given the lexical norms actually supplied."""
    used = info.get('word_norms_available', [])
    standard = {'n_morphemes': 'morpheme count', 'morph_family_size': 'morphological family size',
                'aoa': 'age of acquisition', 'concreteness': 'concreteness'}
    return {
        'claimed': 'lexicality is (or is not) linearly decodable from the hidden state at the first-output '
                   'prediction position, with the stated controls',
        'not_claimed': ['that the LLM predicts WORD/NONWORD at each layer outside the H4 intervention',
                        'that layer depth is a processing time',
                        'generality to other languages, to prompts beyond the tested templates, or to '
                        'instruct models beyond the tested base/instruct pairs'],
        'word_covariates_controlled': ['length', 'Ortho_N', 'sub-word token count',
                                       'model output log-probability', 'HAL frequency'] + used,
        'confounds_not_controlled': [v for k, v in standard.items() if k not in used],
    }


def environment_snapshot() -> dict:
    from importlib import metadata as md
    import platform
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'transformers': _transformers_pkg.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'packages': sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()
                               if d.metadata.get('Name'))}


def write_metadata(cfg: Config, data_info: dict):
    meta = {'written': datetime.now().isoformat(), 'seed': SEED, 'config': asdict(cfg),
            'data': data_info, 'prompt_templates': PROMPT_TEMPLATES, 'analysis_plan': ANALYSIS_PLAN,
            'claim_boundary': claim_boundary(data_info), 'environment': environment_snapshot()}
    with open(os.path.join(cfg.OUTPUT_DIR, 'experiment_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)


# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS  (python LDT-NAACL.py --self-test; also run before every experiment)
# ════════════════════════════════════════════════════════════════════════════
def run_self_tests(verbose: bool = True) -> bool:
    import unittest
    rng0 = np.random.default_rng

    class T(unittest.TestCase):
        def test_mcnemar(self):
            a, b = np.array([1, 1, 1, 0, 0, 1, 1, 0]), np.array([1, 0, 0, 0, 1, 1, 0, 0])
            self.assertEqual(mcnemar_exact(a, b)[:2], (3, 1))
            self.assertAlmostEqual(mcnemar_exact(a, b)[2], stats.binomtest(1, 4, 0.5).pvalue)

        def test_p_values_never_zero(self):
            self.assertTrue(np.isfinite(p_from_z(60)[1]) and p_from_z(60)[1] < -700)
            self.assertTrue(np.isfinite(p_correlation(0.9999, 10 ** 5)[1]))
            self.assertAlmostEqual(p_from_z(1.959963985)[0], 0.05, places=6)
            self.assertAlmostEqual(p_from_f(4.0, 1, 50)[0], p_from_t(2.0, 50)[0], places=8)

        def test_auc_and_delong(self):
            r = rng0(1)
            y = r.integers(0, 2, 600)
            a, b = r.normal(size=600) + y, r.normal(size=600) + 0.5 * y
            self.assertAlmostEqual(auc_with_ci(a[y == 1], a[y == 0])['auroc'], roc_auc_score(y, a), places=10)
            d = delong_paired(a, b, y)
            self.assertAlmostEqual(d['auroc_difference'], roc_auc_score(y, a) - roc_auc_score(y, b), places=10)
            self.assertLess(d['p_delong'], 0.01)
            self.assertAlmostEqual(delong_paired(a, a, y)['auroc_difference'], 0.0)
            neg = r.normal(0, 1, 400)
            self.assertGreater(delong_shared_negatives(r.normal(1, 1, 300), r.normal(1, 1, 300), neg)['p_delta_auroc'], 1e-3)
            self.assertLess(delong_shared_negatives(r.normal(2, 1, 300), r.normal(1, 1, 300), neg)['p_delta_auroc'], 1e-6)

        def test_nested_ols(self):
            r = rng0(2)
            X = pd.DataFrame(r.normal(size=(500, 3)), columns=['a', 'b', 'c'])
            y = X['a'] + 0.3 * X['c'] + r.normal(size=500)
            out = nested_ols(y, X[['a', 'b']], X[['c']])
            f0 = sm.OLS(y, sm.add_constant(X[['a', 'b']])).fit()
            f1 = sm.OLS(y, sm.add_constant(X)).fit()
            self.assertAlmostEqual(out['delta_r2'], f1.rsquared - f0.rsquared, places=10)
            self.assertAlmostEqual(out['p_f'], f1.compare_f_test(f0)[1], places=8)
            self.assertLess(out['log10_bf01'], 0)

        def test_bayes_factor_direction(self):
            self.assertGreater(log10_bf01_correlation(0.001, 5000), np.log10(3))
            self.assertLess(log10_bf01_correlation(0.3, 5000), -np.log10(3))

        def test_caliper_matching(self):
            r = rng0(3)
            t, c = r.normal(0.5, 1, (200, 2)), r.normal(0, 1, (600, 2))
            psd = np.sqrt((t.var(0, ddof=1) + c.var(0, ddof=1)) / 2)
            mt, mc, rej = greedy_caliper_match(t, c, 0.2, psd)
            self.assertEqual(len(mt) + rej, len(t))
            self.assertEqual(len(set(mc)), len(mc))
            self.assertTrue(np.all(np.abs(t[mt] - c[mc]) <= 0.2 * psd + 1e-9))
            self.assertLess(abs(standardized_mean_difference(t[mt, 0], c[mc, 0])), 0.1)

        def test_cka(self):
            r = rng0(4)
            X = r.normal(size=(100, 20))
            Q, _ = np.linalg.qr(r.normal(size=(20, 20)))
            self.assertAlmostEqual(linear_cka(double_center(centered_gram(X)),
                                              double_center(centered_gram(3 * X @ Q))), 1.0, places=8)

        def test_residualisation_guarded(self):
            r = rng0(5)
            Z = r.normal(size=(500, 2))
            X = Z @ r.normal(size=(2, 30)) + r.normal(size=(500, 30))
            Xr, _ = Prober.residualize(X, X[:5], Z, Z[:5])
            self.assertLess(np.abs((Xr - Xr.mean(0)).T @ (Z - Z.mean(0))).max() / len(Z), 1e-8)

        def test_torch_logistic_equals_sklearn(self):
            r = rng0(6)
            X = r.normal(size=(800, 12)) * r.uniform(0.5, 5, 12)
            y = (X[:, 0] / 3 - X[:, 3] / 4 + r.normal(size=800) > 0).astype(int)
            mu, sd = standardize_fit(X)
            ref = LogisticRegression(C=0.5, max_iter=5000, tol=1e-10).fit((X - mu) / sd, y)
            w, b, ok = logistic_torch(torch.as_tensor((X - mu) / sd, dtype=torch.float32),
                                      torch.as_tensor(y, dtype=torch.float32), 0.5, 1000)
            self.assertTrue(ok)
            self.assertLess(np.abs(w.numpy() - ref.coef_[0]).max(), 2e-3)

        def test_inlp(self):
            r = rng0(7)
            y = r.integers(0, 2, 600)
            X = r.normal(size=(600, 10))
            X[:, 0] += 3 * y
            pr = Prober(Config(DEVICE='cpu', PROBE_BACKEND='sklearn'))
            U = inlp_basis(pr, X, y, 2)
            Xp = X - (X - X.mean(0)) @ U @ U.T
            p = pr.fit_linear(Xp, y)
            self.assertLess(np.mean((Prober.decision(p, Xp) > 0) == y), 0.65)

        def test_transfer_closed_form_equals_ridge(self):
            from sklearn.linear_model import Ridge
            r = rng0(8)
            n, y = 600, r.integers(0, 2, 600)
            Xt = r.normal(size=(n, 15)) + y[:, None] * r.normal(size=15)
            Xs = Xt @ r.normal(size=(15, 10)) + 0.3 * r.normal(size=(n, 10))
            fit, te = np.arange(n) < 400, np.arange(n) >= 400
            cfg = Config(DEVICE='cpu', PROBE_BACKEND='sklearn', TRANSFER_N_PERMUTATIONS=20,
                         TRANSFER_N_RANDOM_PROBES=20, MODELS=[ModelConfig('x', 'x')])
            ag = Aggregator.__new__(Aggregator)
            ag.cfg = cfg
            pr = Prober(cfg)
            out = ag._transfer_one(Xs, Xt, y, fit, te, pr, 'cpu', np.random.RandomState(0))
            mu, sd = standardize_fit(Xt[fit])
            rid = Ridge(alpha=out['stitching_alpha']).fit((Xt[fit] - mu) / sd, Xs[fit])
            sp = pr.fit_linear(Xs[fit], y[fit])
            z = Prober.decision(sp, rid.predict((Xt[te] - mu) / sd))
            self.assertAlmostEqual(out['transfer_auroc'], roc_auc_score(y[te], z), places=6)

        def test_layers(self):
            self.assertEqual(representative_layers(8), [0, 2, 4, 6, 7])
            self.assertEqual(layer_at_depth(36, 1.0), 35)
            self.assertEqual(layer_at_depth(36, 0.5), 17)
            self.assertLessEqual(len(select_layers(36, 'representative', 4)), 4)

        def test_decoding_beyond_baseline(self):
            r = rng0(9)
            n = 900
            Z = r.normal(size=(n, 2))
            X = r.normal(size=(n, 20))
            folds = np.arange(n) % 5
            pr = Prober(Config(DEVICE='cpu', PROBE_BACKEND='sklearn'))
            y_null = Z @ np.array([1.0, -0.5]) + 0.5 * r.normal(size=n)
            out0 = pr.crossfit_ridge_r2(X, y_null, folds, baseline=Z)
            self.assertLess(abs(out0['delta_r2_beyond_baseline']), 0.02)
            y_alt = y_null + X[:, 0]
            out1 = pr.crossfit_ridge_r2(X, y_alt, folds, baseline=Z)
            self.assertGreater(out1['delta_r2_beyond_baseline'], 0.2)
            self.assertLess(out1['p_beyond_baseline'], 1e-6)

        def test_chat_format_shifts_span(self):
            class Tok:
                @staticmethod
                def apply_chat_template(msgs, tokenize, add_generation_prompt):
                    return f"<s>[USER] {msgs[0]['content']} [/USER][ASSISTANT]"
            lm = LanguageModel.__new__(LanguageModel)
            lm.tokenizer, lm.mc = Tok(), ModelConfig('m', 'm', chat=True)
            txt, sp = build_prompt('house')
            f, sp2 = lm.format(txt, sp)
            self.assertEqual(f[sp2[0]:sp2[1]], 'house')
            lm.mc = ModelConfig('m', 'm')
            self.assertEqual(lm.format(txt, sp), (txt, sp))

        def test_lexical_norms_merge(self):
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                w = pd.DataFrame({'Word': [f'w{c}{k}' for c in 'abcdefghij' for k in 'abcdefghij'],
                                  'Length': 3, 'Log_Freq_HAL': np.linspace(1, 12, 100), 'Ortho_N': 2,
                                  'I_Mean_RT': 600.0, 'I_Mean_Accuracy': 0.9, 'NMorph': 1})
                w['Word'] = w['Word'].str.replace(r'\d', '', regex=True)
                nw = pd.DataFrame({'Word': [f'q{c}{k}z' for c in 'abcdefghij' for k in 'abcdefghij'],
                                   'Length': 4, 'Ortho_N': 1, 'NWI_Mean_RT': 700.0, 'NWI_Mean_Accuracy': 0.9})
                ext = pd.DataFrame({'Word': w['Word'].str.upper(), 'Conc.M': np.arange(100.0)})
                for name, df in (('w.csv', w), ('n.csv', nw), ('c.csv', ext)):
                    df.to_csv(os.path.join(d, name), index=False)
                cfg = Config(WORDS_PATH=os.path.join(d, 'w.csv'), NONWORDS_PATH=os.path.join(d, 'n.csv'),
                             LEXICAL_NORMS=[{'path': os.path.join(d, 'c.csv'), 'word_column': 'Word',
                                             'columns': {'Conc.M': 'concreteness'}}])
                items, info = load_items(cfg)
                self.assertEqual(set(info['word_norms_available']), {'n_morphemes', 'concreteness'})
                self.assertTrue(items.loc[items['is_word'] == 0, 'concreteness'].isna().all())
                self.assertTrue(items.loc[items['is_word'] == 1, 'concreteness'].notna().all())
                self.assertIn('concreteness', claim_boundary(info)['word_covariates_controlled'])

        def test_prompts(self):
            for name in PROMPT_TEMPLATES:
                txt, (a, b) = build_prompt('house', name)
                self.assertEqual(txt[a:b], 'house')
                self.assertTrue(txt.endswith('Answer:'))

    res = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(
        unittest.TestLoader().loadTestsFromTestCase(T))
    return res.wasSuccessful()


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--models', help='comma-separated model names (default: all in Config.MODELS)')
    ap.add_argument('--aggregate-only', action='store_true')
    ap.add_argument('--force', action='store_true', help='recompute models already finished')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--output-dir')
    ap.add_argument('--max-items', type=int)
    args = ap.parse_args(argv)
    if args.self_test:
        sys.exit(0 if run_self_tests() else 1)
    cfg = Config()
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.max_items:
        cfg.MAX_ITEMS = args.max_items
    if args.models:
        wanted = [s.strip() for s in args.models.split(',') if s.strip()]
        unknown = set(wanted) - {m.name for m in cfg.MODELS}
        if unknown:
            raise SystemExit(f"unknown model names: {sorted(unknown)}")
        run_models = [m for m in cfg.MODELS if m.name in wanted]
    else:
        run_models = list(cfg.MODELS)
    cfg.validate()
    cfg.dirs()
    run_experiment(cfg, run_models, aggregate_only=args.aggregate_only, force=args.force)


def run_experiment(cfg: Config, run_models: list[ModelConfig], aggregate_only=False, force=False):
    fh = logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, 'run.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    if not run_self_tests(verbose=False):
        raise SystemExit("self-tests failed — refusing to run")
    items, info = load_items(cfg)
    write_metadata(cfg, info)
    if not cfg.RUN_INSTRUCT_VARIANTS:
        run_models = [m for m in run_models if m.variant == 'base']
    if not aggregate_only:
        for mc in run_models:
            done = os.path.join(cfg.model_dir(mc.name), 'DONE')
            if os.path.exists(done) and not force:
                logger.info(f"[{mc.name}] already finished — skipped (use --force to recompute)")
                continue
            if os.path.exists(done):
                os.remove(done)
            try:
                ModelStudy(mc, cfg, items, info['word_norms_available']).run()
            except Exception as e:                                       # noqa: BLE001
                logger.error(f"[{mc.name}] failed: {e}", exc_info=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    Aggregator(cfg).run()
    write_metadata(cfg, info)
    logger.info(f"done → {cfg.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
