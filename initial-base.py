# %%
"""
================================================================================
BLOCK 0: CENTRAL CONFIGURATION & UTILITY FUNCTIONS
Run this block FIRST in every notebook session
================================================================================
"""

import os, gc, time, pickle, random, json, warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, jaccard_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Activation, Add, BatchNormalization,
    SpatialDropout2D, concatenate,
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision

from deap import base, creator, tools, algorithms
import optuna
from tqdm import tqdm

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 80)
print("  🚀 BLOCK-BY-BLOCK CHANGE DETECTION PIPELINE v3.0")
print("  ✅ Fixed: DNA constraints, parameter budget, OOM prevention")
print("=" * 80)

# ============================================================================
# CENTRAL CONFIGURATION - MODIFY ALL SETTINGS HERE
# ============================================================================

CONFIG = {
    # ── PATHS ──────────────────────────────────────────────────────────────
    'dataset_path':   Path('/kaggle/input/datasets/mdrifaturrahman33/levir-cd-change-detection/LEVIR-CD+'),
    'checkpoint_dir': Path('/kaggle/working/checkpoints'),
    'stats_dir':      Path('/kaggle/working/stats'),
    'plots_dir':      Path('/kaggle/working/plots'),
    
    # ── REPRODUCIBILITY ────────────────────────────────────────────────────
    'random_seed': 42,
    
    # ── RESOLUTIONS ────────────────────────────────────────────────────────
    'resolution': {
        'proxy':   (128, 128),   # For evolution
        'tuning':  (256, 256),   # For Optuna
        'train256':(256, 256),   # Stage 1 training
        'train512':(512, 512),   # Stage 2 training
    },
    
    # ── BATCH SIZES ────────────────────────────────────────────────────────
    'batch_size': {
        'proxy':    8,
        'tuning':   4,
        'train256': 4,
        'train512': 2,
    },
    
    # ── ✅ FIXED DNA SEARCH SPACE (CONSTRAINED FOR MEMORY) ─────────────────
    'dna_space': {
        'base_filters':     {'type': 'cat',  'choices': [16, 20, 24, 28]},       # ← Max 28 (was 48)
        'growth_ratio':     {'type': 'cont', 'min': 1.0, 'max': 1.8},           # ← Max 1.8 (was 2.5)
        'blocks_per_level': {'type': 'cat',  'choices': [[1,2,2,3,3], [2,2,3,3,3]]},
        'kernel_size':      {'type': 'cat',  'choices': [3, 5]},
        'dropout_rate':     {'type': 'cont', 'min': 0.0, 'max': 0.3},
        'use_mixed_kernels':{'type': 'cat',  'choices': [False, True]},
    },
    'dna_gene_order': ['base_filters', 'growth_ratio', 'blocks_per_level',
                       'kernel_size', 'dropout_rate', 'use_mixed_kernels'],
    
    # ── EVOLUTION PARAMETERS (5 GENS PER BLOCK) ────────────────────────────
    'evolution': {
        'population_size':  6,        # Population per generation
        'generations_per_block': 5,   # ← Each block runs 5 generations
        'crossover_prob':   0.8,
        'mutation_prob':    0.3,
        'tournament_size':  3,
        'proxy_epochs':     3,        # ← Reduced for speed
        'proxy_samples':    100,      # ← Reduced to prevent OOM
        'proxy_val_samples': 25,
        'proxy_levels':     5,        # CRITICAL: Must match full model
        'primary_metric':   'iou',
        'fitness_weights':  (1.0, 0.8, -1.0),  # (IoU, F1, -params)
        'max_params_M':     50.0,     # ← Max 50M params (budget constraint)
    },
    
    # ── OPTUNA PARAMETERS ──────────────────────────────────────────────────
    'optuna': {
        'n_trials':      8,           # ← Reduced (better completion rate)
        'tuning_epochs':  4,
        'metric':         'val_iou',
        'direction':      'maximize',
        'pruner_warmup':   2,
    },
    
    # ── TRAINING PARAMETERS ────────────────────────────────────────────────
    'training': {
        'epochs_256':               25,
        'epochs_512':               15,
        'early_stopping_patience':   8,
        'reduce_lr_patience':        4,
        'reduce_lr_factor':          0.5,
        'min_lr':                    1e-7,
        'num_outputs':               5,
        'monitor':                   'val_output_5_iou',
    },
    
    # ── DATA SPLIT ─────────────────────────────────────────────────────────
    'data_split': {'train_ratio': 0.85, 'val_ratio': 0.15},
    
    # ── MEMORY MANAGEMENT ──────────────────────────────────────────────────
    'memory': {
        'mixed_precision': True,
        'cleanup_every_n_images': 10,
    },
}

# Create directories
for d in [CONFIG['checkpoint_dir'], CONFIG['stats_dir'], CONFIG['plots_dir']]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\n📋 Configuration Summary:")
print(f"   DNA Search Space: {len(CONFIG['dna_space'])} genes")
print(f"   Max Parameters:   {CONFIG['evolution']['max_params_M']}M")
print(f"   Gens per Block:   {CONFIG['evolution']['generations_per_block']}")
print(f"   Population Size:  {CONFIG['evolution']['population_size']}")
print(f"   Primary Metric:   {CONFIG['evolution']['primary_metric'].upper()}")

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================

def set_seeds(seed=None):
    s = seed or CONFIG['random_seed']
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)

set_seeds()
print(f"✅ Random seed set: {CONFIG['random_seed']}")

# ============================================================================
# GPU & MIXED PRECISION SETUP
# ============================================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"✅ GPU memory growth enabled: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️  GPU config error: {e}")
else:
    print("⚠️  No GPU detected")

if CONFIG['memory']['mixed_precision'] and gpus:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"✅ Mixed precision: {policy.compute_dtype}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cleanup_memory():
    """Deep memory cleanup"""
    K.clear_session()
    try:
        tf.compat.v1.reset_default_graph()
    except:
        pass
    gc.collect()
    time.sleep(0.1)

def save_json(obj, fname):
    p = CONFIG['stats_dir'] / fname
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"   💾 {fname}")

def save_csv(df, fname):
    p = CONFIG['stats_dir'] / fname
    df.to_csv(p, index=False)
    print(f"   💾 {fname}")

def save_checkpoint(data, fname):
    p = CONFIG['checkpoint_dir'] / fname
    with open(p, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"   💾 Checkpoint: {fname}")

def load_checkpoint(fname):
    p = CONFIG['checkpoint_dir'] / fname
    if p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None

def iou_from_cm(tp, fp, fn):
    return float(tp) / (float(tp + fp + fn) + 1e-7)

def f1_from_cm(tp, fp, fn):
    p = float(tp) / (float(tp + fp) + 1e-7)
    r = float(tp) / (float(tp + fn) + 1e-7)
    return 2 * p * r / (p + r + 1e-7), p, r

print("\n✅ Utility functions loaded")

# ============================================================================
# DNA OPERATIONS
# ============================================================================

def create_random_dna():
    """Create random DNA dict"""
    dna = {}
    for g, cfg in CONFIG['dna_space'].items():
        if cfg['type'] == 'cat':
            dna[g] = random.choice(cfg['choices'])
        else:
            dna[g] = random.uniform(cfg['min'], cfg['max'])
    return dna

def encode_dna(dna):
    """DNA dict → flat list (DEAP format)"""
    enc = []
    for g in CONFIG['dna_gene_order']:
        cfg = CONFIG['dna_space'][g]
        v = dna[g]
        if cfg['type'] == 'cat':
            choices = cfg['choices']
            enc.append(float(choices.index(v) if v in choices else 0))
        else:
            enc.append(float(v))
    return enc

def decode_individual(ind):
    """Flat list → DNA dict"""
    dna = {}
    for i, g in enumerate(CONFIG['dna_gene_order']):
        cfg = CONFIG['dna_space'][g]
        raw = float(ind[i])
        if cfg['type'] == 'cat':
            choices = cfg['choices']
            idx = int(round(raw)) % len(choices)
            dna[g] = choices[idx]
        else:
            dna[g] = float(np.clip(raw, cfg['min'], cfg['max']))
    return dna

def dna_to_filters(dna, half=False):
    """Compute 5-level filter counts"""
    base = dna['base_filters']
    ratio = dna['growth_ratio']
    if half:
        base = max(8, base // 2)
    return [int(base * (ratio ** i)) for i in range(5)]

def dna_summary(dna):
    """Compact DNA description"""
    return (f"base={dna['base_filters']}, ratio={dna['growth_ratio']:.2f}, "
            f"ks={dna['kernel_size']}, drop={dna['dropout_rate']:.2f}, "
            f"mixed={dna['use_mixed_kernels']}")

print("✅ DNA operations loaded")

# ============================================================================
# ✅ PARAMETER BUDGET VALIDATION
# ============================================================================

def estimate_full_params(proxy_params):
    """Estimate full model params from proxy model"""
    # Proxy uses half filters → full uses ~4x params
    return proxy_params * 4

def validate_param_budget(params, max_params_M=None):
    """Check if model fits within parameter budget"""
    max_p = (max_params_M or CONFIG['evolution']['max_params_M']) * 1e6
    return params <= max_p

print("✅ Parameter validation loaded")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset():
    """Load and split LEVIR-CD+ dataset"""
    base = CONFIG['dataset_path']
    
    train_a = sorted((base / 'train' / 'A').glob('*.png'))
    train_b = sorted((base / 'train' / 'B').glob('*.png'))
    train_m = sorted((base / 'train' / 'label').glob('*.png'))
    test_a  = sorted((base / 'test' / 'A').glob('*.png'))
    test_b  = sorted((base / 'test' / 'B').glob('*.png'))
    test_m  = sorted((base / 'test' / 'label').glob('*.png'))
    
    assert len(train_a) == len(train_b) == len(train_m)
    assert len(test_a) == len(test_b) == len(test_m)
    
    # Split train into train/val
    train_a, val_a, train_b, val_b, train_m, val_m = train_test_split(
        train_a, train_b, train_m,
        test_size=CONFIG['data_split']['val_ratio'],
        random_state=CONFIG['random_seed'],
        shuffle=True,
    )
    
    return {
        'train_a': train_a, 'train_b': train_b, 'train_m': train_m,
        'val_a': val_a, 'val_b': val_b, 'val_m': val_m,
        'test_a': test_a, 'test_b': test_b, 'test_m': test_m,
    }

print("\n📂 Loading dataset...")
DATASET = load_dataset()
print(f"✅ Dataset loaded:")
print(f"   Train: {len(DATASET['train_a'])} samples")
print(f"   Val:   {len(DATASET['val_a'])} samples")
print(f"   Test:  {len(DATASET['test_a'])} samples")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_simple(pa, pb, pm, img_size):
    """Fast preprocessing (proxy/tuning)"""
    def _load_rgb(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    a = cv2.resize(_load_rgb(pa), img_size, interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(_load_rgb(pb), img_size, interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(cv2.imread(str(pm), cv2.IMREAD_GRAYSCALE), img_size,
                   interpolation=cv2.INTER_NEAREST)
    
    x = np.concatenate([a.astype(np.float32) / 255.0,
                        b.astype(np.float32) / 255.0], axis=-1)
    y = (m > 127).astype(np.float32)[..., np.newaxis]
    return x, y

def preprocess_clahe(pa, pb, pm, img_size):
    """CLAHE preprocessing (final training)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def _load_clahe(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for c in range(3):
            img[:, :, c] = clahe.apply(img[:, :, c])
        return img
    
    a = cv2.resize(_load_clahe(pa), img_size, interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(_load_clahe(pb), img_size, interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(cv2.imread(str(pm), cv2.IMREAD_GRAYSCALE), img_size,
                   interpolation=cv2.INTER_NEAREST)
    
    x = np.concatenate([a.astype(np.float32) / 255.0,
                        b.astype(np.float32) / 255.0], axis=-1)
    y = (m > 127).astype(np.float32)[..., np.newaxis]
    return x, y

def numpy_batches(paths_a, paths_b, paths_m, img_size, batch_size,
                  max_samples=None, shuffle=True, use_clahe=False):
    """Memory-efficient numpy batch loader (no TF graph accumulation)"""
    fn = preprocess_clahe if use_clahe else preprocess_simple
    pa, pb, pm = list(paths_a), list(paths_b), list(paths_m)
    
    if max_samples and max_samples < len(pa):
        idx = random.sample(range(len(pa)), max_samples)
        pa = [pa[i] for i in idx]
        pb = [pb[i] for i in idx]
        pm = [pm[i] for i in idx]
    
    if shuffle:
        combined = list(zip(pa, pb, pm))
        random.shuffle(combined)
        pa, pb, pm = zip(*combined)
    
    batches = []
    for start in range(0, len(pa), batch_size):
        end = min(start + batch_size, len(pa))
        xs, ys = [], []
        for i in range(start, end):
            try:
                x, y = fn(pa[i], pb[i], pm[i], img_size)
                xs.append(x)
                ys.append(y)
            except:
                pass
        if xs:
            batches.append((np.stack(xs).astype(np.float32),
                           np.stack(ys).astype(np.float32)))
    return batches

print("✅ Data preprocessing loaded")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def _res_block(x, nb_filter, tag, ks=3, drop=0.0):
    """Residual block"""
    shortcut = x
    x = Conv2D(nb_filter, ks, padding='same', use_bias=False,
               kernel_regularizer=l2(1e-4), name=f'{tag}_c1')(x)
    x = BatchNormalization(name=f'{tag}_bn1')(x)
    x = Activation('selu', name=f'{tag}_a1')(x)
    if drop > 0:
        x = SpatialDropout2D(drop, name=f'{tag}_dp')(x)
    x = Conv2D(nb_filter, ks, padding='same', use_bias=False,
               kernel_regularizer=l2(1e-4), name=f'{tag}_c2')(x)
    x = BatchNormalization(name=f'{tag}_bn2')(x)
    if K.int_shape(shortcut)[-1] != nb_filter:
        shortcut = Conv2D(nb_filter, 1, use_bias=False, name=f'{tag}_skip')(shortcut)
    x = Add(name=f'{tag}_add')([x, shortcut])
    return Activation('selu', name=f'{tag}_a2')(x)

def _mixed_block(x, nb_filter, tag, drop=0.0):
    """Mixed kernel block"""
    half = max(nb_filter // 2, 1)
    x3 = Conv2D(half, 3, padding='same', use_bias=False,
                kernel_regularizer=l2(1e-4), name=f'{tag}_m3')(x)
    x5 = Conv2D(half, 5, padding='same', use_bias=False,
                kernel_regularizer=l2(1e-4), name=f'{tag}_m5')(x)
    xm = concatenate([x3, x5], name=f'{tag}_mcat')
    if half * 2 != nb_filter:
        xm = Conv2D(nb_filter, 1, use_bias=False, name=f'{tag}_malign')(xm)
    xm = BatchNormalization(name=f'{tag}_mbn')(xm)
    shortcut = x
    if K.int_shape(shortcut)[-1] != nb_filter:
        shortcut = Conv2D(nb_filter, 1, use_bias=False, name=f'{tag}_mskip')(shortcut)
    xm = Add(name=f'{tag}_madd')([xm, shortcut])
    return Activation('selu', name=f'{tag}_ma2')(xm)

def _enc_stack(x, nb_filter, n_blocks, tag, ks=3, drop=0.0, mixed=False):
    """Stack of blocks"""
    block_fn = _mixed_block if mixed else _res_block
    for i in range(n_blocks):
        kw = dict(tag=f'{tag}_b{i}', drop=drop)
        if mixed:
            x = block_fn(x, nb_filter, **kw)
        else:
            x = block_fn(x, nb_filter, ks=ks, **kw)
    return x

def build_proxy_unetpp(dna, img_size=(128, 128)):
    """5-level proxy (memory-efficient)"""
    bpl = dna.get('blocks_per_level', [1, 2, 2, 3, 3])
    ks = dna.get('kernel_size', 3)
    drop = dna.get('dropout_rate', 0.1)
    mixed = dna.get('use_mixed_kernels', False)
    nf = dna_to_filters(dna, half=True)
    
    inp = Input(shape=(*img_size, 6), name='input')
    
    # Encoder
    e0 = _enc_stack(inp, nf[0], bpl[0], 'e0', ks, drop, mixed)
    e1 = _enc_stack(MaxPooling2D((2, 2))(e0), nf[1], bpl[1], 'e1', ks, drop, mixed)
    e2 = _enc_stack(MaxPooling2D((2, 2))(e1), nf[2], bpl[2], 'e2', ks, drop, mixed)
    e3 = _enc_stack(MaxPooling2D((2, 2))(e2), nf[3], bpl[3], 'e3', ks, drop, mixed)
    e4 = _enc_stack(MaxPooling2D((2, 2))(e3), nf[4], bpl[4], 'e4', ks, drop, mixed)
    
    # Simple decoder
    d3 = _enc_stack(concatenate([UpSampling2D()(e4), e3]), nf[3], 1, 'pd3', ks, drop, mixed)
    d2 = _enc_stack(concatenate([UpSampling2D()(d3), e2]), nf[2], 1, 'pd2', ks, drop, mixed)
    d1 = _enc_stack(concatenate([UpSampling2D()(d2), e1]), nf[1], 1, 'pd1', ks, drop, mixed)
    d0 = _enc_stack(concatenate([UpSampling2D()(d1), e0]), nf[0], 1, 'pd0', ks, drop, mixed)
    
    out = Conv2D(1, 1, activation='sigmoid', dtype='float32', name='output')(d0)
    return Model(inp, out, name='ProxyUNetPP_5L')

def build_full_unetpp(dna, img_size=(512, 512), deep_supervision=True):
    """Full 5-level UNet++"""
    bpl = dna.get('blocks_per_level', [1, 2, 2, 3, 3])
    ks = dna.get('kernel_size', 3)
    drop = dna.get('dropout_rate', 0.1)
    mixed = dna.get('use_mixed_kernels', False)
    nf = dna_to_filters(dna, half=False)
    
    inp = Input(shape=(*img_size, 6), name='input')
    bn_axis = 3
    
    # Encoder
    x00 = _enc_stack(inp, nf[0], bpl[0], 'x00', ks, drop, mixed)
    x10 = _enc_stack(MaxPooling2D((2, 2))(x00), nf[1], bpl[1], 'x10', ks, drop, mixed)
    x20 = _enc_stack(MaxPooling2D((2, 2))(x10), nf[2], bpl[2], 'x20', ks, drop, mixed)
    x30 = _enc_stack(MaxPooling2D((2, 2))(x20), nf[3], bpl[3], 'x30', ks, drop, mixed)
    x40 = _enc_stack(MaxPooling2D((2, 2))(x30), nf[4], bpl[4], 'x40', ks, drop, mixed)
    
    # Decoder column 1
    x01 = _enc_stack(concatenate([UpSampling2D()(x10), x00], bn_axis), nf[0], bpl[0], 'x01', ks, drop, mixed)
    x11 = _enc_stack(concatenate([UpSampling2D()(x20), x10], bn_axis), nf[1], bpl[1], 'x11', ks, drop, mixed)
    x21 = _enc_stack(concatenate([UpSampling2D()(x30), x20], bn_axis), nf[2], bpl[2], 'x21', ks, drop, mixed)
    x31 = _enc_stack(concatenate([UpSampling2D()(x40), x30], bn_axis), nf[3], bpl[3], 'x31', ks, drop, mixed)
    
    # Decoder column 2
    x02 = _enc_stack(concatenate([UpSampling2D()(x11), x00, x01], bn_axis), nf[0], bpl[0], 'x02', ks, drop, mixed)
    x12 = _enc_stack(concatenate([UpSampling2D()(x21), x10, x11], bn_axis), nf[1], bpl[1], 'x12', ks, drop, mixed)
    x22 = _enc_stack(concatenate([UpSampling2D()(x31), x20, x21], bn_axis), nf[2], bpl[2], 'x22', ks, drop, mixed)
    
    # Decoder column 3
    x03 = _enc_stack(concatenate([UpSampling2D()(x12), x00, x01, x02], bn_axis), nf[0], bpl[0], 'x03', ks, drop, mixed)
    x13 = _enc_stack(concatenate([UpSampling2D()(x22), x10, x11, x12], bn_axis), nf[1], bpl[1], 'x13', ks, drop, mixed)
    
    # Decoder column 4
    x04 = _enc_stack(concatenate([UpSampling2D()(x13), x00, x01, x02, x03], bn_axis),
                     nf[0], bpl[0], 'x04', ks, drop, mixed)
    
    # Output heads
    def _head(feat, name):
        return Conv2D(1, 1, activation='sigmoid', dtype='float32', name=name)(feat)
    
    o1 = _head(x01, 'output_1')
    o2 = _head(x02, 'output_2')
    o3 = _head(x03, 'output_3')
    o4 = _head(x04, 'output_4')
    o5 = _head(concatenate([x01, x02, x03, x04], bn_axis, name='msof'), 'output_5')
    
    outputs = [o1, o2, o3, o4, o5] if deep_supervision else o5
    return Model(inp, outputs, name='FullUNetPP_5L')

print("✅ Model architecture loaded")

# ============================================================================
# LOSS & METRICS
# ============================================================================

class IoUMetric(keras.metrics.Metric):
    """Streaming IoU metric (Keras 3 compatible)"""
    def __init__(self, name='iou', threshold=0.5, **kw):
        super().__init__(name=name, **kw)
        self.thr = threshold
        self.tp = self.add_weight(name='tp', shape=(), initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', shape=(), initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', shape=(), initializer='zeros', dtype=tf.float32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true > 0.5, tf.float32)
        yp = tf.cast(y_pred > self.thr, tf.float32)
        self.tp.assign_add(tf.reduce_sum(yt * yp))
        self.fp.assign_add(tf.reduce_sum((1 - yt) * yp))
        self.fn.assign_add(tf.reduce_sum(yt * (1 - yp)))
    
    def result(self):
        return self.tp / (self.tp + self.fp + self.fn + K.epsilon())
    
    def reset_state(self):
        self.tp.assign(0.)
        self.fp.assign(0.)
        self.fn.assign(0.)

def make_hybrid_loss(dice_w=1.0):
    """Hybrid BCE + Dice loss"""
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Weighted BCE
        pos_w = tf.reduce_mean(1.0 - y_true)
        bce = K.binary_crossentropy(y_true, y_pred)
        wbce = tf.reduce_mean(bce * (pos_w * y_true + (1 - pos_w) * (1 - y_true)))
        # Dice
        inter = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
        return wbce + dice_w * dice
    return _loss

print("✅ Loss & metrics loaded")

print("\n" + "=" * 80)
print("✅ BLOCK 0 COMPLETE - Configuration & utilities ready")
print("=" * 80)
print("\n💡 Next steps:")
print("   1. Run Block 1 to start evolution (Gen 1-5)")
print("   2. Run Block 2 to continue evolution (Gen 6-10)")
print("   3. Run Block 3 for Optuna HPO")
print("   4. Run Block 4 for progressive training")
print("   5. Run Block 5 for evaluation & visualization")

# %%
"""
================================================================================
BLOCK 1: EVOLUTIONARY SEARCH - GENERATIONS 1-5
Run after Block 0
================================================================================
"""

# Ensure Block 0 is loaded
assert 'CONFIG' in globals(), "❌ Run Block 0 first!"

from deap import base, creator, tools, algorithms

print("=" * 80)
print("  BLOCK 1: Evolution Generations 1-5")
print("=" * 80)

# ============================================================================
# DEAP SETUP
# ============================================================================

for attr in ['FitnessMulti', 'Individual']:
    if hasattr(creator, attr):
        delattr(creator, attr)

creator.create('FitnessMulti', base.Fitness,
               weights=tuple(CONFIG['evolution']['fitness_weights']))
creator.create('Individual', list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def _make_individual():
    dna = create_random_dna()
    return creator.Individual(encode_dna(dna))

toolbox.register('individual', _make_individual)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

def _cx_dna(ind1, ind2):
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def _mut_dna(ind, indpb=0.3):
    for i, g in enumerate(CONFIG['dna_gene_order']):
        if random.random() > indpb:
            continue
        cfg = CONFIG['dna_space'][g]
        if cfg['type'] == 'cat':
            choices = cfg['choices']
            cur_idx = int(round(ind[i])) % len(choices)
            others = [j for j in range(len(choices)) if j != cur_idx]
            if others:
                ind[i] = float(random.choice(others))
        else:
            std = (cfg['max'] - cfg['min']) * 0.2
            ind[i] = float(np.clip(ind[i] + random.gauss(0, std),
                                   cfg['min'], cfg['max']))
    return ind,

toolbox.register('mate', _cx_dna)
toolbox.register('mutate', _mut_dna, indpb=CONFIG['evolution']['mutation_prob'])
toolbox.register('select', tools.selNSGA2)

print("✅ DEAP toolbox configured")

# ============================================================================
# ✅ PROXY EVALUATION WITH PARAMETER BUDGET
# ============================================================================

def evaluate_proxy(individual):
    """Evaluate individual with parameter budget enforcement"""
    cleanup_memory()
    
    try:
        cfg = CONFIG['evolution']
        dna = decode_individual(individual)
        model = build_proxy_unetpp(dna, img_size=CONFIG['resolution']['proxy'])
        params = model.count_params()
        
        # ✅ PARAMETER BUDGET CHECK
        estimated_full = estimate_full_params(params)
        if not validate_param_budget(estimated_full):
            print(f"   ⚠️  Model too large: {estimated_full/1e6:.1f}M params (max: {cfg['max_params_M']}M)")
            del model
            cleanup_memory()
            return (0.0, 0.0, 999.0)  # Worst fitness
        
        # Compile
        loss_fn = make_hybrid_loss(1.0)
        model.compile(
            optimizer=Adam(1e-3),
            loss=loss_fn,
            metrics=[IoUMetric(name='iou')],
        )
        
        # Load data
        bs = CONFIG['batch_size']['proxy']
        res = CONFIG['resolution']['proxy']
        trn_bat = numpy_batches(
            DATASET['train_a'], DATASET['train_b'], DATASET['train_m'],
            res, bs, max_samples=cfg['proxy_samples']
        )
        val_bat = numpy_batches(
            DATASET['val_a'], DATASET['val_b'], DATASET['val_m'],
            res, bs, max_samples=cfg['proxy_val_samples'], shuffle=False
        )
        
        # Train
        best_iou = 0.0
        for epoch in range(cfg['proxy_epochs']):
            random.shuffle(trn_bat)
            for xb, yb in trn_bat:
                model.train_on_batch(xb, yb)
            
            # Validation
            tp_e = fp_e = fn_e = 0.0
            for xb, yb in val_bat:
                yp = model(xb, training=False).numpy()
                tp_e += float(np.sum((yb > 0.5) * (yp > 0.5)))
                fp_e += float(np.sum((yb < 0.5) * (yp > 0.5)))
                fn_e += float(np.sum((yb > 0.5) * (yp < 0.5)))
            
            epoch_iou = tp_e / (tp_e + fp_e + fn_e + 1e-7)
            best_iou = max(best_iou, epoch_iou)
        
        # Final F1
        all_yt, all_yp = [], []
        for xb, yb in val_bat:
            yp = model(xb, training=False).numpy()
            all_yt.append(yb)
            all_yp.append(yp)
        
        yt_flat = (np.concatenate(all_yt).flatten() > 0.5).astype(np.int32)
        yp_flat = (np.concatenate(all_yp).flatten() > 0.5).astype(np.int32)
        cm = confusion_matrix(yt_flat, yp_flat, labels=[0, 1])
        tn_v, fp_v, fn_v, tp_v = cm.ravel()
        val_f1, _, _ = f1_from_cm(tp_v, fp_v, fn_v)
        
        del model, trn_bat, val_bat
        cleanup_memory()
        
        return (float(best_iou), float(val_f1), float(estimated_full / 1e6))
        
    except (tf.errors.ResourceExhaustedError, MemoryError) as oom:
        print(f"   ⚠️  OOM: {oom}")
        cleanup_memory()
        return (0.0, 0.0, 999.0)
    except Exception as exc:
        print(f"   ⚠️  Error: {exc}")
        cleanup_memory()
        return (0.0, 0.0, 999.0)

print("✅ Evaluation function ready")

# ============================================================================
# EVOLUTION RUNNER
# ============================================================================

def run_evolution_block(start_gen, end_gen, resume_pop=None, resume_hof=None):
    """Run evolution for a block of generations"""
    cfg = CONFIG['evolution']
    pop = resume_pop if resume_pop else toolbox.population(n=cfg['population_size'])
    hof = resume_hof if resume_hof else tools.ParetoFront()
    
    eval_rows = []
    gen_rows = []
    
    print(f"\n🧬 Evolution: Gen {start_gen}-{end_gen}")
    print(f"   Population: {cfg['population_size']}")
    print(f"   Proxy: {CONFIG['resolution']['proxy']} @ {cfg['proxy_epochs']} epochs")
    print(f"   Samples: {cfg['proxy_samples']} train / {cfg['proxy_val_samples']} val")
    
    for gen in range(start_gen, end_gen + 1):
        t0 = time.time()
        print(f"\n{'─' * 70}")
        print(f"  Generation {gen}")
        
        # Evaluate
        invalid = [ind for ind in pop if not ind.fitness.valid]
        print(f"  Evaluating {len(invalid)} individuals...")
        
        for ind in tqdm(invalid, desc=f"Gen {gen}", leave=False):
            ind.fitness.values = evaluate_proxy(ind)
            dna = decode_individual(ind)
            eval_rows.append({
                'generation': gen,
                'base_filters': dna['base_filters'],
                'growth_ratio': round(dna['growth_ratio'], 3),
                'kernel_size': dna['kernel_size'],
                'dropout_rate': round(dna['dropout_rate'], 3),
                'use_mixed': dna['use_mixed_kernels'],
                'iou': round(ind.fitness.values[0], 4),
                'f1': round(ind.fitness.values[1], 4),
                'params_M': round(ind.fitness.values[2], 3),
            })
        
        # Genetic ops
        offspring = algorithms.varAnd(pop, toolbox,
                                      cxpb=cfg['crossover_prob'],
                                      mutpb=cfg['mutation_prob'])
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate_proxy(ind)
                dna = decode_individual(ind)
                eval_rows.append({
                    'generation': gen,
                    'base_filters': dna['base_filters'],
                    'growth_ratio': round(dna['growth_ratio'], 3),
                    'kernel_size': dna['kernel_size'],
                    'dropout_rate': round(dna['dropout_rate'], 3),
                    'use_mixed': dna['use_mixed_kernels'],
                    'iou': round(ind.fitness.values[0], 4),
                    'f1': round(ind.fitness.values[1], 4),
                    'params_M': round(ind.fitness.values[2], 3),
                })
        
        pop = toolbox.select(pop + offspring, cfg['population_size'])
        hof.update(pop)
        
        # Stats
        iou_vals = [ind.fitness.values[0] for ind in pop]
        f1_vals = [ind.fitness.values[1] for ind in pop]
        p_vals = [abs(ind.fitness.values[2]) for ind in pop]
        
        gen_stats = {
            'generation': gen,
            'max_iou': round(max(iou_vals), 4),
            'mean_iou': round(float(np.mean(iou_vals)), 4),
            'max_f1': round(max(f1_vals), 4),
            'min_params_M': round(min(p for p in p_vals if p < 900), 3) if any(p < 900 for p in p_vals) else 999.0,
            'pareto_size': len(hof),
            'time_min': round((time.time() - t0) / 60, 2),
        }
        gen_rows.append(gen_stats)
        
        print(f"  IoU: max={gen_stats['max_iou']:.4f}  mean={gen_stats['mean_iou']:.4f}")
        print(f"  F1:  max={gen_stats['max_f1']:.4f}  Pareto={len(hof)}")
        print(f"  Time: {gen_stats['time_min']:.1f} min")
    
    return pop, hof, pd.DataFrame(eval_rows), pd.DataFrame(gen_rows)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "=" * 80)
print("  Starting Evolution: Generations 1-5")
print("=" * 80)

# Check for resume
resume_data = load_checkpoint('evo_block1.pkl')
if resume_data:
    print("\n♻️  Found checkpoint - resuming from Gen 5")
    pop_final = resume_data['pop']
    hof_final = resume_data['hof']
    eval_df = resume_data['eval_df']
    gen_df = resume_data['gen_df']
else:
    pop_final, hof_final, eval_df, gen_df = run_evolution_block(1, 5)
    
    # Save checkpoint
    save_checkpoint({
        'pop': pop_final,
        'hof': hof_final,
        'eval_df': eval_df,
        'gen_df': gen_df,
    }, 'evo_block1.pkl')
    
    # Save stats
    save_csv(eval_df, 'evo_block1_evaluations.csv')
    save_csv(gen_df, 'evo_block1_gen_stats.csv')

# Display results
print("\n" + "=" * 80)
print("  BLOCK 1 COMPLETE: Generations 1-5")
print("=" * 80)
print(f"\n📊 Results:")
print(f"   Pareto front: {len(hof_final)} individuals")
print(f"   Best IoU:     {gen_df['max_iou'].max():.4f}")
print(f"   Best F1:      {gen_df['max_f1'].max():.4f}")

# Show top 3 Pareto individuals
print(f"\n🏆 Top 3 Pareto-Optimal Architectures:")
sorted_hof = sorted(hof_final, key=lambda x: x.fitness.values[0], reverse=True)
for i, ind in enumerate(sorted_hof[:3], 1):
    dna = decode_individual(ind)
    iou, f1, params = ind.fitness.values
    print(f"\n   Rank {i}:")
    print(f"      IoU:    {iou:.4f}")
    print(f"      F1:     {f1:.4f}")
    print(f"      Params: {abs(params):.1f}M")
    print(f"      DNA:    {dna_summary(dna)}")

print("\n💡 Next: Run Block 2 to continue to Gen 6-10")

# %%
# ✅ CHECKPOINT VERIFICATION CELL
import os
from pathlib import Path

checkpoint_dir = Path('/kaggle/working/checkpoints')
checkpoint_file = checkpoint_dir / 'evo_block1.pkl'

print("=" * 80)
print("  CHECKPOINT VERIFICATION")
print("=" * 80)

# Check if directory exists
if checkpoint_dir.exists():
    print(f"\n✅ Checkpoint directory exists: {checkpoint_dir}")
    
    # List all files
    all_files = list(checkpoint_dir.glob('*'))
    print(f"\n📂 Files in checkpoint directory ({len(all_files)}):")
    for f in all_files:
        size = f.stat().st_size / (1024 * 1024)  # MB
        print(f"   • {f.name:30s} - {size:.2f} MB")
    
    # Check specific checkpoint
    if checkpoint_file.exists():
        print(f"\n✅ Block 1 checkpoint EXISTS!")
        print(f"   Path: {checkpoint_file}")
        print(f"   Size: {checkpoint_file.stat().st_size / (1024*1024):.2f} MB")
        
        # Try to load it
        try:
            import pickle
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n✅ Checkpoint is VALID and loadable!")
            print(f"   Contains keys: {list(data.keys())}")
            print(f"   Population size: {len(data['pop'])}")
            print(f"   Pareto front size: {len(data['hof'])}")
            print(f"   Generations: {data['gen_df']['generation'].unique().tolist()}")
            print(f"   Best IoU: {data['gen_df']['max_iou'].max():.4f}")
        except Exception as e:
            print(f"\n❌ Error loading checkpoint: {e}")
    else:
        print(f"\n❌ Block 1 checkpoint NOT FOUND!")
        print(f"   Expected at: {checkpoint_file}")
else:
    print(f"\n❌ Checkpoint directory does not exist: {checkpoint_dir}")

print("\n" + "=" * 80)

# %%
"""
================================================================================
💾 SAVE CHECKPOINTS AS KAGGLE DATASET
Run this after completing any block to save progress
================================================================================
"""

import shutil
from pathlib import Path

print("=" * 80)
print("  💾 CREATING KAGGLE DATASET FROM CHECKPOINTS")
print("=" * 80)

# Create a directory to bundle everything
output_bundle = Path('/kaggle/working/output_bundle')
output_bundle.mkdir(exist_ok=True)

# Copy all checkpoint files
checkpoint_dir = Path('/kaggle/working/checkpoints')
stats_dir = Path('/kaggle/working/stats')
plots_dir = Path('/kaggle/working/plots')

print("\n📦 Copying files to output bundle...")

# Copy checkpoints
if checkpoint_dir.exists():
    output_ckpt = output_bundle / 'checkpoints'
    output_ckpt.mkdir(exist_ok=True)
    for f in checkpoint_dir.glob('*'):
        if f.is_file():
            shutil.copy2(f, output_ckpt / f.name)
            print(f"   ✅ {f.name}")

# Copy stats (optional, for reference)
if stats_dir.exists():
    output_stats = output_bundle / 'stats'
    output_stats.mkdir(exist_ok=True)
    for f in stats_dir.glob('*.csv'):
        shutil.copy2(f, output_stats / f.name)
    for f in stats_dir.glob('*.json'):
        shutil.copy2(f, output_stats / f.name)

print(f"\n✅ Bundle created at: {output_bundle}")
print(f"\n📋 Contents:")

def show_tree(path, prefix=""):
    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        if item.is_file():
            size_mb = item.stat().st_size / (1024*1024)
            print(f"{prefix}{connector}{item.name} ({size_mb:.2f} MB)")
        else:
            print(f"{prefix}{connector}{item.name}/")
            ext = "    " if is_last else "│   "
            show_tree(item, prefix + ext)

show_tree(output_bundle)

print("\n" + "=" * 80)
print("  📤 NEXT STEPS:")
print("=" * 80)
print("""
1. Click 'Save Version' button (top right)
2. Wait for notebook to finish running
3. Go to 'Output' tab (right sidebar)
4. Click 'Save Output as Dataset'
5. Name it: 'change-detection-checkpoints-block1' (or appropriate block number)
6. Click 'Create'

This dataset can now be added as input to future notebook sessions!
""")


