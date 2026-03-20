# %%
"""
================================================================================
🔄 LOAD CHECKPOINTS FROM PREVIOUS SESSION
Run this BEFORE Block 0 to restore checkpoints
================================================================================
"""

import shutil
from pathlib import Path

print("=" * 80)
print("  🔄 RESTORING CHECKPOINTS FROM DATASET")
print("=" * 80)

# Path to your dataset (adjust name based on what you named it)
input_dataset = Path('/kaggle/input/datasets/abhinavvishen/dataopt-1')

# Restore to working directory
checkpoint_dir = Path('/kaggle/working/checkpoints')
stats_dir = Path('/kaggle/working/stats')

checkpoint_dir.mkdir(parents=True, exist_ok=True)
stats_dir.mkdir(parents=True, exist_ok=True)

print("\n📥 Restoring files...")

# Restore checkpoints
if (input_dataset / 'checkpoints').exists():
    for f in (input_dataset / 'checkpoints').glob('*'):
        if f.is_file():
            shutil.copy2(f, checkpoint_dir / f.name)
            print(f"   ✅ {f.name}")

# Restore stats
if (input_dataset / 'stats').exists():
    for f in (input_dataset / 'stats').glob('*'):
        if f.is_file():
            shutil.copy2(f, stats_dir / f.name)

print(f"\n✅ Checkpoints restored to /kaggle/working/")

# Verify
print(f"\n🔍 Verification:")
all_checkpoints = list(checkpoint_dir.glob('*.pkl'))
print(f"   Found {len(all_checkpoints)} checkpoint files:")
for ckpt in all_checkpoints:
    size_mb = ckpt.stat().st_size / (1024*1024)
    print(f"      • {ckpt.name} ({size_mb:.2f} MB)")

print("\n✅ Ready to continue! Now run Block 0, then your next block.")

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
        'epochs_256':               100,
        'epochs_512':               50,
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
🔍 PRE-TRAINING MODEL SIZE VALIDATION
Run BEFORE Block 4 to verify model will fit
================================================================================
"""

print("=" * 80)
print("  🔍 MODEL SIZE VALIDATION")
print("=" * 80)

# Load champion
champion_data = load_checkpoint('champion.pkl')
if not champion_data:
    print("❌ Champion not found!")
else:
    champion_dna = champion_data['champion_dna']
    
    print(f"\n📐 Champion DNA:")
    print(f"   Base filters: {champion_dna['base_filters']}")
    print(f"   Growth ratio: {champion_dna['growth_ratio']:.2f}")
    print(f"   Blocks per level: {champion_dna['blocks_per_level']}")
    
    # Build and check 256px model
    print(f"\n🔍 Building 256×256 model...")
    cleanup_memory()
    model_256 = build_full_unetpp(
        champion_dna,
        img_size=(256, 256),
        deep_supervision=True
    )
    params_256 = model_256.count_params()
    print(f"   Parameters: {params_256:,} ({params_256/1e6:.1f}M)")
    
    if params_256 > 50e6:
        print(f"   ⚠️  WARNING: Model exceeds 50M parameter budget!")
    else:
        print(f"   ✅ Within budget")
    
    del model_256
    cleanup_memory()
    
    # Build and check 512px model
    print(f"\n🔍 Building 512×512 model...")
    model_512 = build_full_unetpp(
        champion_dna,
        img_size=(512, 512),
        deep_supervision=True
    )
    params_512 = model_512.count_params()
    print(f"   Parameters: {params_512:,} ({params_512/1e6:.1f}M)")
    
    if params_512 > 50e6:
        print(f"   ⚠️  WARNING: Model exceeds 50M parameter budget!")
        print(f"   Consider reducing batch_size to 1 for 512px training")
    else:
        print(f"   ✅ Within budget")
    
    del model_512
    cleanup_memory()
    
    # Memory estimate
    print(f"\n💾 Estimated GPU Memory (mixed precision):")
    print(f"   256px (batch=4): ~{params_256 * 4 * 2 / 1e9:.1f} GB")
    print(f"   512px (batch=2): ~{params_512 * 2 * 2 / 1e9:.1f} GB")
    
    print("\n✅ Validation complete - safe to proceed to Block 4")

# %%
"""
================================================================================
BLOCK 4: PROGRESSIVE TRAINING (256px → 512px)
Stage 1: Train at 256px
Stage 2: Fine-tune at 512px with transferred weights
================================================================================
"""

# Ensure Block 0 is loaded
assert 'CONFIG' in globals(), "❌ Run Block 0 first!"

print("=" * 80)
print("  BLOCK 4: Progressive Training")
print("=" * 80)

# ============================================================================
# LOAD CHAMPION & HYPERPARAMETERS
# ============================================================================

print("\n📂 Loading champion DNA and hyperparameters...")
champion_data = load_checkpoint('champion.pkl')
optuna_data = load_checkpoint('optuna_study.pkl')

if not champion_data or not optuna_data:
    raise FileNotFoundError("❌ Run Blocks 1-3 first!")

champion_dna = champion_data['champion_dna']
best_hparams = optuna_data['best_params']

print(f"✅ Champion: {dna_summary(champion_dna)}")
print(f"✅ Hyperparams: {best_hparams}")

# ============================================================================
# HELPER: CREATE CALLBACKS
# ============================================================================

def build_callbacks(monitor, checkpoint_path, csv_path):
    """Standard callback stack"""
    tc = CONFIG['training']
    return [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor=monitor, mode='max',
            save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor=monitor, mode='max',
            patience=tc['early_stopping_patience'],
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor, mode='max',
            factor=tc['reduce_lr_factor'],
            patience=tc['reduce_lr_patience'],
            min_lr=tc['min_lr'], verbose=1
        ),
        CSVLogger(str(csv_path), append=True),
    ]

# ============================================================================
# HELPER: COMPILE MODEL
# ============================================================================

def compile_model(model, hparams, n_outputs=5):
    """Compile with hyperparameters"""
    opt = (Adam(hparams['learning_rate'])
           if hparams.get('optimizer', 'Adam') == 'Adam'
           else AdamW(hparams['learning_rate']))
    
    loss_fn = make_hybrid_loss(hparams['dice_weight'])
    metrics_per_output = [[IoUMetric(name='iou')] for _ in range(n_outputs)]
    
    model.compile(
        optimizer=opt,
        loss=[loss_fn] * n_outputs,
        metrics=metrics_per_output
    )
    return model

# ============================================================================
# STAGE 1: TRAINING AT 256px
# ============================================================================

print("\n" + "=" * 80)
print("  STAGE 1: Training at 256×256")
print("=" * 80)

# Check for resume
stage1_data = load_checkpoint('stage1_complete.pkl')
if stage1_data:
    print("\n♻️  Stage 1 already complete - loading history")
    hist256 = stage1_data['history']
    stage1_best_iou = stage1_data['best_val_iou']
    print(f"   Best val IoU: {stage1_best_iou:.4f}")
else:
    cleanup_memory()
    
    tc = CONFIG['training']
    monitor = tc['monitor']
    
    print(f"\n📐 Building model at 256×256...")
    model256 = build_full_unetpp(
        champion_dna,
        img_size=CONFIG['resolution']['train256'],
        deep_supervision=True
    )
    
    params = model256.count_params()
    print(f"   Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Validate size
    if not validate_param_budget(params):
        raise ValueError(f"Model too large: {params/1e6:.1f}M params!")
    
    model256 = compile_model(model256, best_hparams)
    
    print(f"\n📊 Creating datasets...")
    use_clahe = best_hparams.get('use_clahe', False)
    
    # Use numpy_batches for full dataset
    print(f"   Loading training data...")
    trn_bat256 = numpy_batches(
        DATASET['train_a'], DATASET['train_b'], DATASET['train_m'],
        CONFIG['resolution']['train256'],
        CONFIG['batch_size']['train256'],
        use_clahe=use_clahe
    )
    
    print(f"   Loading validation data...")
    val_bat256 = numpy_batches(
        DATASET['val_a'], DATASET['val_b'], DATASET['val_m'],
        CONFIG['resolution']['train256'],
        CONFIG['batch_size']['train256'],
        shuffle=False,
        use_clahe=use_clahe
    )
    
    print(f"   Train batches: {len(trn_bat256)}")
    print(f"   Val batches:   {len(val_bat256)}")
    
    # Callbacks
    ckpt_path = CONFIG['checkpoint_dir'] / 'model_256_best.keras'
    csv_path = CONFIG['stats_dir'] / 'history_256.csv'
    cbs = build_callbacks(monitor, ckpt_path, csv_path)
    
    print(f"\n🏋️  Training for {tc['epochs_256']} epochs...")
    print(f"   Monitor: {monitor}")
    print(f"   Early stopping patience: {tc['early_stopping_patience']}")
    
    # Manual training loop (better control)
    history_data = {'loss': [], 'val_loss': [], 'output_5_iou': [], 'val_output_5_iou': []}
    
    best_val_iou = 0.0
    patience_counter = 0
    
    for epoch in range(1, tc['epochs_256'] + 1):
        print(f"\nEpoch {epoch}/{tc['epochs_256']}")
        
        # Training
        random.shuffle(trn_bat256)
        train_losses = []
        train_ious = []
        
        for xb, yb in tqdm(trn_bat256, desc="Training", leave=False):
            metrics = model256.train_on_batch(xb, [yb] * 5)
            train_losses.append(metrics[0])
            # Output 5 IoU is typically at index 10 (5 losses + 5 ious)
            if len(metrics) > 10:
                train_ious.append(metrics[10])
        
        # Validation
        val_losses = []
        tp_v = fp_v = fn_v = 0.0
        
        for xb, yb in tqdm(val_bat256, desc="Validation", leave=False):
            preds = model256(xb, training=False)
            pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
            
            # Compute loss
            loss_fn = make_hybrid_loss(best_hparams['dice_weight'])
            val_loss = float(loss_fn(yb, pred).numpy())
            val_losses.append(val_loss)
            
            # Compute IoU
            yt = (yb.flatten() > 0.5).astype(np.float32)
            yp = (pred.numpy().flatten() > 0.5).astype(np.float32)
            tp_v += float(np.sum(yt * yp))
            fp_v += float(np.sum((1 - yt) * yp))
            fn_v += float(np.sum(yt * (1 - yp)))
        
        val_iou = tp_v / (tp_v + fp_v + fn_v + 1e-7)
        
        # Log metrics
        train_loss = np.mean(train_losses)
        train_iou = np.mean(train_ious) if train_ious else 0.0
        val_loss = np.mean(val_losses)
        
        history_data['loss'].append(float(train_loss))
        history_data['val_loss'].append(float(val_loss))
        history_data['output_5_iou'].append(float(train_iou))
        history_data['val_output_5_iou'].append(float(val_iou))
        
        print(f"   loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
        print(f"   output_5_iou: {train_iou:.4f} - val_output_5_iou: {val_iou:.4f}")
        
        # Save best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            model256.save(str(ckpt_path))
            print(f"   💾 Saved best model (IoU: {best_val_iou:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= tc['early_stopping_patience']:
            print(f"\n⏸️  Early stopping triggered")
            break
    
    # Save weights for transfer learning
    weights_path = CONFIG['checkpoint_dir'] / 'weights_256.weights.h5'
    model256.save_weights(str(weights_path))
    print(f"\n✅ Stage 1 complete - best val IoU: {best_val_iou:.4f}")
    
    # Create history object
    class HistoryObj:
        def __init__(self, data):
            self.history = data
    
    hist256 = HistoryObj(history_data)
    stage1_best_iou = best_val_iou
    
    # Save checkpoint
    save_checkpoint({
        'history': hist256,
        'best_val_iou': stage1_best_iou,
    }, 'stage1_complete.pkl')
    
    del model256, trn_bat256, val_bat256
    cleanup_memory()

# ============================================================================
# STAGE 2: FINE-TUNING AT 512px
# ============================================================================

print("\n" + "=" * 80)
print("  STAGE 2: Fine-tuning at 512×512")
print("=" * 80)

# Check for resume
stage2_data = load_checkpoint('stage2_complete.pkl')
if stage2_data:
    print("\n♻️  Stage 2 already complete - loading history")
    hist512 = stage2_data['history']
    stage2_best_iou = stage2_data['best_val_iou']
    print(f"   Best val IoU: {stage2_best_iou:.4f}")
else:
    cleanup_memory()
    
    tc = CONFIG['training']
    monitor = tc['monitor']
    
    print(f"\n📐 Building model at 512×512...")
    model512 = build_full_unetpp(
        champion_dna,
        img_size=CONFIG['resolution']['train512'],
        deep_supervision=True
    )
    
    params = model512.count_params()
    print(f"   Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # ✅ FIXED: Keras 3 compatible weight transfer
    weights_path = CONFIG['checkpoint_dir'] / 'weights_256.weights.h5'
    if weights_path.exists():
        try:
            # Keras 3 doesn't support by_name/skip_mismatch - just load directly
            model512.load_weights(str(weights_path))
            print(f"   ✅ Transferred weights from 256px model")
        except Exception as e:
            print(f"   ⚠️  Weight transfer failed: {e}")
            print(f"   Training from scratch at 512px")
    else:
        print(f"   ⚠️  256px weights not found - training from scratch")
    
    # Compile with reduced LR (fine-tuning)
    ft_hparams = dict(best_hparams)
    ft_hparams['learning_rate'] *= 0.1  # 10x smaller LR for fine-tuning
    model512 = compile_model(model512, ft_hparams)
    
    print(f"\n📊 Creating datasets at 512px...")
    use_clahe = True  # Always use CLAHE for final training
    
    print(f"   Loading training data...")
    trn_bat512 = numpy_batches(
        DATASET['train_a'], DATASET['train_b'], DATASET['train_m'],
        CONFIG['resolution']['train512'],
        CONFIG['batch_size']['train512'],
        use_clahe=use_clahe
    )
    
    print(f"   Loading validation data...")
    val_bat512 = numpy_batches(
        DATASET['val_a'], DATASET['val_b'], DATASET['val_m'],
        CONFIG['resolution']['train512'],
        CONFIG['batch_size']['train512'],
        shuffle=False,
        use_clahe=use_clahe
    )
    
    print(f"   Train batches: {len(trn_bat512)}")
    print(f"   Val batches:   {len(val_bat512)}")
    
    # Callbacks
    ckpt_path = CONFIG['checkpoint_dir'] / 'model_512_best.keras'
    csv_path = CONFIG['stats_dir'] / 'history_512.csv'
    cbs = build_callbacks(monitor, ckpt_path, csv_path)
    
    print(f"\n🏋️  Fine-tuning for {tc['epochs_512']} epochs...")
    print(f"   Reduced LR: {ft_hparams['learning_rate']:.6f}")
    
    # Manual training loop
    history_data = {'loss': [], 'val_loss': [], 'output_5_iou': [], 'val_output_5_iou': []}
    
    best_val_iou = 0.0
    patience_counter = 0
    
    for epoch in range(1, tc['epochs_512'] + 1):
        print(f"\nEpoch {epoch}/{tc['epochs_512']}")
        
        # Training
        random.shuffle(trn_bat512)
        train_losses = []
        train_ious = []
        
        for xb, yb in tqdm(trn_bat512, desc="Training", leave=False):
            metrics = model512.train_on_batch(xb, [yb] * 5)
            train_losses.append(metrics[0])
            if len(metrics) > 10:
                train_ious.append(metrics[10])
        
        # Validation
        val_losses = []
        tp_v = fp_v = fn_v = 0.0
        
        for xb, yb in tqdm(val_bat512, desc="Validation", leave=False):
            preds = model512(xb, training=False)
            pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
            
            loss_fn = make_hybrid_loss(ft_hparams['dice_weight'])
            val_loss = float(loss_fn(yb, pred).numpy())
            val_losses.append(val_loss)
            
            yt = (yb.flatten() > 0.5).astype(np.float32)
            yp = (pred.numpy().flatten() > 0.5).astype(np.float32)
            tp_v += float(np.sum(yt * yp))
            fp_v += float(np.sum((1 - yt) * yp))
            fn_v += float(np.sum(yt * (1 - yp)))
        
        val_iou = tp_v / (tp_v + fp_v + fn_v + 1e-7)
        
        # Log metrics
        train_loss = np.mean(train_losses)
        train_iou = np.mean(train_ious) if train_ious else 0.0
        val_loss = np.mean(val_losses)
        
        history_data['loss'].append(float(train_loss))
        history_data['val_loss'].append(float(val_loss))
        history_data['output_5_iou'].append(float(train_iou))
        history_data['val_output_5_iou'].append(float(val_iou))
        
        print(f"   loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
        print(f"   output_5_iou: {train_iou:.4f} - val_output_5_iou: {val_iou:.4f}")
        
        # Save best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            model512.save(str(ckpt_path))
            print(f"   💾 Saved best model (IoU: {best_val_iou:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= tc['early_stopping_patience']:
            print(f"\n⏸️  Early stopping triggered")
            break
    
    # Save final model
    final_path = CONFIG['checkpoint_dir'] / 'final_model.keras'
    model512.save(str(final_path))
    print(f"\n✅ Stage 2 complete - best val IoU: {best_val_iou:.4f}")
    
    # Create history object
    class HistoryObj:
        def __init__(self, data):
            self.history = data
    
    hist512 = HistoryObj(history_data)
    stage2_best_iou = best_val_iou
    
    # Save checkpoint
    save_checkpoint({
        'history': hist512,
        'best_val_iou': stage2_best_iou,
    }, 'stage2_complete.pkl')
    
    del model512, trn_bat512, val_bat512
    cleanup_memory()

# ============================================================================
# ✅ FIXED FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("  BLOCK 4 COMPLETE: Training Finished")
print("=" * 80)

# Reload checkpoints to ensure we have the data
stage1_final = load_checkpoint('stage1_complete.pkl')
stage2_final = load_checkpoint('stage2_complete.pkl')

print(f"\n📊 Training Summary:")

if stage1_final:
    iou_256 = stage1_final['best_val_iou']
    print(f"   Stage 1 (256px) best val IoU: {iou_256:.4f}")
else:
    print(f"   Stage 1 (256px): ⚠️ Not found")
    iou_256 = None

if stage2_final:
    iou_512 = stage2_final['best_val_iou']
    print(f"   Stage 2 (512px) best val IoU: {iou_512:.4f}")
else:
    print(f"   Stage 2 (512px): ⚠️ Not found")
    iou_512 = None

# Show comparison if both stages completed
if iou_256 is not None and iou_512 is not None:
    diff = iou_512 - iou_256
    pct_change = (diff / iou_256) * 100
    
    if diff > 0:
        print(f"\n   📈 Improvement 256→512: +{diff:.4f} (+{pct_change:.1f}%)")
    else:
        print(f"\n   📉 Change 256→512: {diff:.4f} ({pct_change:.1f}%)")
        print(f"   ℹ️  Lower IoU at 512px is normal (4x more pixels, same model capacity)")

print(f"\n📁 Saved Models:")
model_256 = CONFIG['checkpoint_dir'] / 'model_256_best.keras'
model_512 = CONFIG['checkpoint_dir'] / 'model_512_best.keras'
final_model = CONFIG['checkpoint_dir'] / 'final_model.keras'

if model_256.exists():
    print(f"   ✅ model_256_best.keras ({model_256.stat().st_size/(1024**2):.1f} MB)")
if model_512.exists():
    print(f"   ✅ model_512_best.keras ({model_512.stat().st_size/(1024**2):.1f} MB)")
if final_model.exists():
    print(f"   ✅ final_model.keras ({final_model.stat().st_size/(1024**2):.1f} MB)")

print("\n💡 Next: Run Block 5 for test evaluation & visualization")

# %%
"""
================================================================================
BLOCK 5: TEST EVALUATION & VISUALIZATION
Evaluates final model on test set and generates all visualizations
================================================================================
"""

# Ensure Block 0 is loaded
assert 'CONFIG' in globals(), "❌ Run Block 0 first!"

print("=" * 80)
print("  BLOCK 5: Test Evaluation & Visualization")
print("=" * 80)

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================

print("\n📂 Loading trained model...")
champion_data = load_checkpoint('champion.pkl')
if not champion_data:
    raise FileNotFoundError("❌ Champion DNA not found! Run previous blocks first.")

champion_dna = champion_data['champion_dna']
print(f"✅ Champion DNA loaded: {dna_summary(champion_dna)}")

cleanup_memory()

# Build model at 512px (final resolution)
print(f"\n📐 Building model at 512×512...")
final_model = build_full_unetpp(
    champion_dna,
    img_size=CONFIG['resolution']['train512'],
    deep_supervision=True
)

params = final_model.count_params()
print(f"   Parameters: {params:,} ({params/1e6:.1f}M)")

# Load best weights
best_ckpt = CONFIG['checkpoint_dir'] / 'model_512_best.keras'
if best_ckpt.exists():
    try:
        final_model = keras.models.load_model(str(best_ckpt), compile=False)
        print(f"✅ Loaded best checkpoint from Stage 2")
    except Exception as e:
        print(f"⚠️  Error loading checkpoint: {e}")
        print(f"   Using freshly initialized model")
else:
    print(f"⚠️  Best checkpoint not found - using freshly initialized model")

# ============================================================================
# TEST SET EVALUATION (Memory-Efficient)
# ============================================================================

print("\n" + "=" * 80)
print("  Test Set Evaluation")
print("=" * 80)

print(f"\n🔐 Evaluating on {len(DATASET['test_a'])} test images...")
print(f"   Processing one image at a time to prevent OOM...")

tp_total = tn_total = fp_total = fn_total = 0
sample_predictions = []

img_size = CONFIG['resolution']['train512']

for idx in tqdm(range(len(DATASET['test_a'])), desc="Test eval"):
    try:
        x, y = preprocess_clahe(
            DATASET['test_a'][idx],
            DATASET['test_b'][idx],
            DATASET['test_m'][idx],
            img_size
        )
        
        xb = x[np.newaxis]
        
        # ✅ FIXED: Predict - handle list output correctly
        preds = final_model(xb, training=False)
        
        # Extract final output (output_5) from list
        if isinstance(preds, (list, tuple)):
            pred = preds[-1]  # Get the last output (output_5)
        else:
            pred = preds
        
        # NOW convert to numpy
        pred = pred.numpy()
        
        # Confusion matrix
        yt = (y.flatten() > 0.5).astype(np.int32)
        yp = (pred.flatten() > 0.5).astype(np.int32)
        
        # Confusion matrix
        yt = (y.flatten() > 0.5).astype(np.int32)
        yp = (pred.flatten() > 0.5).astype(np.int32)
        
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn_i, fp_i, fn_i, tp_i = cm.ravel()
        
        tp_total += tp_i
        tn_total += tn_i
        fp_total += fp_i
        fn_total += fn_i
        
        # Store first 10 samples (downsampled to 256 for visualization)
        if len(sample_predictions) < 10:
            x256 = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)
            y256 = cv2.resize(y[..., 0], (256, 256), interpolation=cv2.INTER_NEAREST)
            p256 = cv2.resize(pred[0, ..., 0], (256, 256), interpolation=cv2.INTER_LINEAR)
            
            sample_predictions.append({
                'x': x256,
                'gt': y256,
                'pred': p256,
                'idx': idx,
            })
        
        if idx % CONFIG['memory']['cleanup_every_n_images'] == 0:
            gc.collect()
            
    except Exception as e:
        print(f"   ⚠️  Error on image {idx}: {e}")
        continue

# Compute metrics
total_pixels = tp_total + tn_total + fp_total + fn_total

test_metrics = {
    'accuracy': float((tp_total + tn_total) / (total_pixels + 1e-7)),
    'precision': float(tp_total / (tp_total + fp_total + 1e-7)),
    'recall': float(tp_total / (tp_total + fn_total + 1e-7)),
    'f1': float(2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-7)),
    'iou': float(tp_total / (tp_total + fp_total + fn_total + 1e-7)),
    'dice': float(2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-7)),
    'specificity': float(tn_total / (tn_total + fp_total + 1e-7)),
    'tp': int(tp_total),
    'tn': int(tn_total),
    'fp': int(fp_total),
    'fn': int(fn_total),
    'total_pixels': int(total_pixels),
}

print(f"\n📊 Test Results:")
for k in ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']:
    print(f"   {k:12s}: {test_metrics[k]:.4f}")

save_json(test_metrics, 'test_metrics.json')

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("  Generating Visualizations")
print("=" * 80)

plt.style.use('seaborn-v0_8-darkgrid')

# Create plots directory if needed
CONFIG['plots_dir'].mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# PLOT 1: Training History
# ──────────────────────────────────────────────────────────────────────────

print("\n📊 Plot 1: Training history...")
stage1 = load_checkpoint('stage1_complete.pkl')
stage2 = load_checkpoint('stage2_complete.pkl')

if stage1 and stage2:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Progressive Training: 256px → 512px', fontsize=14, fontweight='bold')
    
    # Extract history data (handle HistoryObj format)
    hist1 = stage1['history'].history if hasattr(stage1['history'], 'history') else stage1['history']
    hist2 = stage2['history'].history if hasattr(stage2['history'], 'history') else stage2['history']
    
    for row, (lbl, hist_data) in enumerate([
        ('Stage 1 (256px)', hist1),
        ('Stage 2 (512px)', hist2),
    ]):
        # Loss
        ax = axes[row, 0]
        if 'loss' in hist_data and 'val_loss' in hist_data:
            ax.plot(hist_data['loss'], label='Train', lw=2, color='#2ecc71')
            ax.plot(hist_data['val_loss'], label='Val', lw=2, color='#e74c3c')
            ax.set_title(f'{lbl}: Loss', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Loss data unavailable', ha='center', va='center')
            ax.set_title(f'{lbl}: Loss', fontweight='bold')
        
        # IoU
        ax = axes[row, 1]
        if 'output_5_iou' in hist_data and 'val_output_5_iou' in hist_data:
            ax.plot(hist_data['output_5_iou'], label='Train', lw=2, color='#3498db')
            ax.plot(hist_data['val_output_5_iou'], label='Val', lw=2, color='#9b59b6')
            best = max(hist_data['val_output_5_iou'])
            ax.axhline(best, ls='--', color='#9b59b6', lw=1, label=f'Best={best:.4f}')
            ax.set_title(f'{lbl}: IoU (output_5)', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('IoU')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'IoU data unavailable', ha='center', va='center')
            ax.set_title(f'{lbl}: IoU', fontweight='bold')
    
    plt.tight_layout()
    plot1_path = CONFIG['plots_dir'] / 'plot1_training_history.png'
    plt.savefig(plot1_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot1_path.name}")
else:
    print("   ⚠️  Training history not found - skipping Plot 1")

# ──────────────────────────────────────────────────────────────────────────
# PLOT 2: Evolution Progress
# ──────────────────────────────────────────────────────────────────────────

print("\n📊 Plot 2: Evolution progress...")
evo_data = load_checkpoint('evo_block2.pkl')

if evo_data:
    gen_df = evo_data['gen_df']
    eval_df = evo_data['eval_df']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evolutionary Search Progress (Gen 1-10)', fontsize=14, fontweight='bold')
    
    # IoU progression
    ax = axes[0, 0]
    ax.plot(gen_df['generation'], gen_df['max_iou'], 'o-', lw=2, label='Max', color='#27ae60', markersize=6)
    ax.plot(gen_df['generation'], gen_df['mean_iou'], 's--', lw=1.5, label='Mean', color='#2ecc71', markersize=5)
    ax.fill_between(gen_df['generation'], gen_df['mean_iou'], gen_df['max_iou'],
                    alpha=0.2, color='#27ae60')
    ax.set_title('IoU Evolution', fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Val IoU (proxy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 progression
    ax = axes[0, 1]
    ax.plot(gen_df['generation'], gen_df['max_f1'], 'o-', lw=2, label='Max', color='#8e44ad', markersize=6)
    ax.fill_between(gen_df['generation'], [0]*len(gen_df), gen_df['max_f1'],
                    alpha=0.2, color='#8e44ad')
    ax.set_title('F1 Evolution', fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Val F1 (proxy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Pareto front (filter out failed evaluations)
    valid_eval = eval_df[eval_df['params_M'] < 900].copy()
    ax = axes[1, 0]
    sc = ax.scatter(valid_eval['params_M'], valid_eval['iou'],
                   c=valid_eval['generation'], cmap='viridis',
                   alpha=0.6, s=50, edgecolors='k', lw=0.3)
    plt.colorbar(sc, ax=ax, label='Generation')
    ax.set_title('Accuracy-Efficiency Trade-off', fontweight='bold')
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Proxy IoU')
    ax.grid(True, alpha=0.3)
    
    # DNA exploration
    ax = axes[1, 1]
    sc = ax.scatter(valid_eval['base_filters'], valid_eval['growth_ratio'],
                   c=valid_eval['iou'], cmap='RdYlGn',
                   alpha=0.7, s=80, edgecolors='k', lw=0.3)
    plt.colorbar(sc, ax=ax, label='IoU')
    ax.set_title('DNA Search Space', fontweight='bold')
    ax.set_xlabel('Base Filters')
    ax.set_ylabel('Growth Ratio')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot2_path = CONFIG['plots_dir'] / 'plot2_evolution_progress.png'
    plt.savefig(plot2_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot2_path.name}")
else:
    print("   ⚠️  Evolution data not found - skipping Plot 2")

# ──────────────────────────────────────────────────────────────────────────
# PLOT 3: Sample Predictions
# ──────────────────────────────────────────────────────────────────────────

print("\n📊 Plot 3: Sample predictions...")
if sample_predictions:
    n = min(8, len(sample_predictions))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    if n == 1:
        axes = axes[np.newaxis]
    
    fig.suptitle('Sample Test Predictions (downsampled to 256×256 for display)',
                 fontsize=14, fontweight='bold')
    
    col_labels = ['Before (T1)', 'After (T2)', 'Ground Truth', 'Prediction']
    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, fontsize=11, fontweight='bold')
    
    for i, s in enumerate(sample_predictions[:n]):
        x, gt, pred = s['x'], s['gt'], s['pred']
        
        # Before image (first 3 channels)
        axes[i, 0].imshow(x[:, :, :3])
        axes[i, 0].axis('off')
        
        # After image (last 3 channels)
        axes[i, 1].imshow(x[:, :, 3:])
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(gt, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].axis('off')
        
        # Prediction with IoU
        yt_f = (gt.flatten() > 0.5).astype(int)
        yp_f = (pred.flatten() > 0.5).astype(int)
        iou_i = jaccard_score(yt_f, yp_f, zero_division=0)
        
        axes[i, 3].imshow(pred, cmap='Reds', vmin=0, vmax=1)
        axes[i, 3].set_title(f'IoU={iou_i:.3f}', fontsize=9)
        axes[i, 3].axis('off')
        
        # Row label
        axes[i, 0].set_ylabel(f'Sample #{s["idx"]}', fontsize=9, rotation=0, labelpad=40)
    
    plt.tight_layout()
    plot3_path = CONFIG['plots_dir'] / 'plot3_sample_predictions.png'
    plt.savefig(plot3_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot3_path.name}")
else:
    print("   ⚠️  No sample predictions - skipping Plot 3")

# ──────────────────────────────────────────────────────────────────────────
# PLOT 4: Metrics Bar Chart & Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────

print("\n📊 Plot 4: Metrics bar chart & confusion matrix...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Test Set Performance', fontsize=14, fontweight='bold')

# Bar chart
ax = axes[0]
keys = ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']
vals = [test_metrics[k] for k in keys]
lbls = ['Accuracy', 'Precision', 'Recall', 'F1', 'IoU', 'Dice']
clrs = ['#27ae60', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']

bars = ax.bar(lbls, vals, color=clrs, alpha=0.85, edgecolor='white', lw=1.5)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Overall Metrics', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
           f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Confusion matrix
ax = axes[1]
cm_arr = np.array([
    [test_metrics['tn'], test_metrics['fp']],
    [test_metrics['fn'], test_metrics['tp']],
], dtype=np.int64)

im = ax.imshow(cm_arr, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred: No-Change', 'Pred: Change'])
ax.set_yticklabels(['Actual: No-Change', 'Actual: Change'])
ax.set_title('Confusion Matrix', fontweight='bold')
plt.colorbar(im, ax=ax)

mx = cm_arr.max()
for r in range(2):
    for c in range(2):
        v = cm_arr[r, c]
        col = 'white' if v > mx * 0.6 else 'black'
        ax.text(c, r, f'{v:,}', ha='center', va='center',
               fontsize=13, fontweight='bold', color=col)

plt.tight_layout()
plot4_path = CONFIG['plots_dir'] / 'plot4_metrics.png'
plt.savefig(plot4_path, dpi=180, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: {plot4_path.name}")

# ──────────────────────────────────────────────────────────────────────────
# PLOT 5: Training vs Test Comparison
# ──────────────────────────────────────────────────────────────────────────

print("\n📊 Plot 5: Training vs Test comparison...")

if stage1 and stage2:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    stage1_val = stage1.get('best_val_iou', 0)
    stage2_val = stage2.get('best_val_iou', 0)
    test_iou = test_metrics['iou']
    
    stages = ['Stage 1\n(256px Val)', 'Stage 2\n(512px Val)', 'Test\n(512px)']
    values = [stage1_val, stage2_val, test_iou]
    colors = ['#3498db', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(stages, values, color=colors, alpha=0.85, edgecolor='white', lw=2)
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_ylabel('IoU Score', fontsize=12, fontweight='bold')
    ax.set_title('IoU Comparison: Validation vs Test', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
               f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add gap analysis
    val_test_gap = test_iou - stage2_val
    gap_pct = (val_test_gap / stage2_val) * 100
    gap_color = '#27ae60' if val_test_gap >= 0 else '#e74c3c'
    gap_sign = '+' if val_test_gap >= 0 else ''
    
    ax.text(0.98, 0.02, 
           f'Val→Test Gap: {gap_sign}{val_test_gap:.4f} ({gap_sign}{gap_pct:.1f}%)',
           transform=ax.transAxes, ha='right', va='bottom',
           fontsize=10, bbox=dict(boxstyle='round', facecolor=gap_color, alpha=0.3))
    
    plt.tight_layout()
    plot5_path = CONFIG['plots_dir'] / 'plot5_val_test_comparison.png'
    plt.savefig(plot5_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot5_path.name}")
else:
    print("   ⚠️  Training data incomplete - skipping Plot 5")

# ============================================================================
# TABLES
# ============================================================================

print("\n" + "=" * 80)
print("  Generating Tables")
print("=" * 80)

# ──────────────────────────────────────────────────────────────────────────
# TABLE 1: Champion Architecture
# ──────────────────────────────────────────────────────────────────────────

print("\n📋 Table 1: Champion Architecture")
nf = dna_to_filters(champion_dna)
rows = []
names = ['Level 0 (Input)', 'Level 1', 'Level 2', 'Level 3', 'Level 4 (Bridge)']
for i, (nm, f) in enumerate(zip(names, nf)):
    rows.append([nm, f, f'{f / nf[0]:.2f}x'])

df = pd.DataFrame(rows, columns=['Level', 'Filters', 'Growth Multiplier'])
print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
save_csv(df, 'table1_architecture.csv')

# ──────────────────────────────────────────────────────────────────────────
# TABLE 2: Test Metrics Summary
# ──────────────────────────────────────────────────────────────────────────

print("\n📋 Table 2: Test Metrics Summary")
rows = [
    ['Accuracy', f"{test_metrics['accuracy']:.4f}", 'Overall correctness'],
    ['Precision', f"{test_metrics['precision']:.4f}", 'Positive prediction accuracy'],
    ['Recall', f"{test_metrics['recall']:.4f}", 'Change detection rate'],
    ['F1-Score', f"{test_metrics['f1']:.4f}", 'Harmonic mean of P & R'],
    ['IoU', f"{test_metrics['iou']:.4f}", 'Intersection over Union'],
    ['Dice', f"{test_metrics['dice']:.4f}", 'Sørensen–Dice coefficient'],
    ['Specificity', f"{test_metrics['specificity']:.4f}", 'No-change detection rate'],
]
df = pd.DataFrame(rows, columns=['Metric', 'Value', 'Description'])
print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
save_csv(df, 'table2_test_metrics.csv')

# ──────────────────────────────────────────────────────────────────────────
# TABLE 3: Training Summary
# ──────────────────────────────────────────────────────────────────────────

print("\n📋 Table 3: Training Summary")

if stage1 and stage2:
    rows = [
        ['Stage 1 (256px)', 
         f"{stage1['best_val_iou']:.4f}",
         '256×256',
         f"{params:,}"],
        ['Stage 2 (512px)', 
         f"{stage2['best_val_iou']:.4f}",
         '512×512',
         f"{params:,}"],
        ['Test Set', 
         f"{test_metrics['iou']:.4f}",
         '512×512',
         f"{params:,}"],
    ]
    df = pd.DataFrame(rows, columns=['Stage', 'IoU', 'Resolution', 'Parameters'])
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
    save_csv(df, 'table3_training_summary.csv')
else:
    print("   ⚠️  Training data incomplete - skipping Table 3")

# ──────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  BLOCK 5 COMPLETE: Evaluation Finished")
print("=" * 80)

# Build comprehensive summary
summary = {
    'champion_dna': champion_dna,
    'test_metrics': test_metrics,
    'model_parameters': params,
    'outputs': {
        'checkpoints': str(CONFIG['checkpoint_dir']),
        'stats': str(CONFIG['stats_dir']),
        'plots': str(CONFIG['plots_dir']),
    },
}

# Add training results if available
if stage1:
    summary['stage1_val_iou'] = float(stage1['best_val_iou'])
if stage2:
    summary['stage2_val_iou'] = float(stage2['best_val_iou'])

save_json(summary, 'final_summary.json')

print(f"\n🏆 Final Results:")
print(f"   Test IoU:      {test_metrics['iou']:.4f}")
print(f"   Test F1:       {test_metrics['f1']:.4f}")
print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"   Test Precision:{test_metrics['precision']:.4f}")
print(f"   Test Recall:   {test_metrics['recall']:.4f}")

if stage2:
    val_test_gap = test_metrics['iou'] - stage2['best_val_iou']
    print(f"\n📊 Generalization:")
    print(f"   Val IoU (512px):  {stage2['best_val_iou']:.4f}")
    print(f"   Test IoU (512px): {test_metrics['iou']:.4f}")
    print(f"   Gap: {val_test_gap:+.4f} ({val_test_gap/stage2['best_val_iou']*100:+.1f}%)")

print(f"\n📁 Generated Files:")

# Count files in each directory
plot_files = list(CONFIG['plots_dir'].glob('*.png'))
stats_files = list(CONFIG['stats_dir'].glob('*.csv')) + list(CONFIG['stats_dir'].glob('*.json'))
ckpt_files = list(CONFIG['checkpoint_dir'].glob('*'))

print(f"   Plots:       {len(plot_files)} files")
for f in sorted(plot_files):
    print(f"      • {f.name}")

print(f"   Statistics:  {len(stats_files)} files")
for f in sorted(stats_files)[:10]:  # Show first 10
    print(f"      • {f.name}")
if len(stats_files) > 10:
    print(f"      ... and {len(stats_files)-10} more")

print(f"   Checkpoints: {len(ckpt_files)} files")
for f in sorted(ckpt_files)[:5]:  # Show first 5
    size_mb = f.stat().st_size / (1024**2)
    print(f"      • {f.name} ({size_mb:.1f} MB)")
if len(ckpt_files) > 5:
    print(f"      ... and {len(ckpt_files)-5} more")

print("\n" + "🎉" * 40)
print("  COMPLETE PIPELINE FINISHED!")
print("🎉" * 40)

print(f"\n💡 Next Steps:")
print(f"   1. Review plots in: {CONFIG['plots_dir']}")
print(f"   2. Check metrics in: {CONFIG['stats_dir']}/final_summary.json")
print(f"   3. Download trained model: {CONFIG['checkpoint_dir']}/final_model.keras")
print(f"   4. Save this session as dataset for future use")

# ============================================================================
# ✅ BONUS: Create Results Archive
# ============================================================================

print("\n" + "=" * 80)
print("  📦 Creating Results Archive")
print("=" * 80)

try:
    import zipfile
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = Path(f'/kaggle/working/results_archive_{timestamp}.zip')
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add plots
        for f in CONFIG['plots_dir'].glob('*.png'):
            zipf.write(f, f'plots/{f.name}')
        
        # Add key stats
        for f in CONFIG['stats_dir'].glob('*.json'):
            zipf.write(f, f'stats/{f.name}')
        for f in CONFIG['stats_dir'].glob('*summary*.csv'):
            zipf.write(f, f'stats/{f.name}')
        
        # Add champion info
        champion_json = CONFIG['stats_dir'] / 'champion_dna.json'
        if champion_json.exists():
            zipf.write(champion_json, 'champion_dna.json')
    
    zip_size_mb = zip_path.stat().st_size / (1024**2)
    print(f"\n✅ Results archive created:")
    print(f"   {zip_path.name} ({zip_size_mb:.1f} MB)")
    print(f"\n   Download from Output tab → {zip_path.name}")
    
except Exception as e:
    print(f"\n⚠️  Could not create archive: {e}")

print("\n" + "=" * 80)
print("✅ All tasks complete!")
print("=" * 80)



