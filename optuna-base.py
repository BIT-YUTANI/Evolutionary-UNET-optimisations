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
input_dataset = Path('/kaggle/input/datasets/abhinavvishen/datagen-6-10')

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
        'n_trials':      16,           # ← Reduced (better completion rate)
        'tuning_epochs':  30,
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
✅ PRE-BLOCK 3 VERIFICATION
Ensures everything is ready for Optuna
================================================================================
"""

from pathlib import Path
import pickle

print("=" * 80)
print("  ✅ PRE-OPTUNA VERIFICATION")
print("=" * 80)

all_good = True

# Check 1: CONFIG loaded
if 'CONFIG' in globals():
    print("\n✅ Block 0 (CONFIG) loaded")
else:
    print("\n❌ Block 0 not loaded - RUN BLOCK 0 FIRST")
    all_good = False

# Check 2: Dataset loaded
if 'DATASET' in globals():
    print(f"✅ Dataset loaded: {len(DATASET['train_a'])} train samples")
else:
    print("❌ Dataset not loaded - RUN BLOCK 0 FIRST")
    all_good = False

# Check 3: Champion checkpoint exists
champion_file = Path('/kaggle/working/checkpoints/champion.pkl')
if champion_file.exists():
    print(f"✅ Champion checkpoint exists ({champion_file.stat().st_size/1024:.1f} KB)")
    
    # Verify it's loadable
    try:
        with open(champion_file, 'rb') as f:
            champion_data = pickle.load(f)
        
        if 'champion_dna' in champion_data:
            dna = champion_data['champion_dna']
            print(f"   Champion DNA: base_filters={dna['base_filters']}, growth_ratio={dna['growth_ratio']:.2f}")
        else:
            print("   ⚠️  Champion DNA key missing")
            all_good = False
    except Exception as e:
        print(f"   ❌ Error loading champion: {e}")
        all_good = False
else:
    print(f"❌ Champion checkpoint missing at: {champion_file}")
    print("   Run the champion fix cell first!")
    all_good = False

# Check 4: Evolution results
evo_file = Path('/kaggle/working/checkpoints/evo_block2.pkl')
if evo_file.exists():
    print(f"✅ Evolution checkpoint exists ({evo_file.stat().st_size/(1024*1024):.1f} MB)")
else:
    print(f"⚠️  Evolution checkpoint missing (optional for Block 3)")

# Check 5: GPU available
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {len(gpus)} device(s)")
else:
    print("⚠️  No GPU detected - training will be slow")

# Check 6: Memory status
import psutil
mem = psutil.virtual_memory()
print(f"\n💾 Memory Status:")
print(f"   Total: {mem.total/(1024**3):.1f} GB")
print(f"   Available: {mem.available/(1024**3):.1f} GB")
print(f"   Used: {mem.percent:.1f}%")

if mem.percent > 80:
    print("   ⚠️  High memory usage - consider restarting kernel")

print("\n" + "=" * 80)
if all_good:
    print("  ✅ ALL CHECKS PASSED - READY FOR BLOCK 3")
else:
    print("  ❌ SOME CHECKS FAILED - FIX ISSUES ABOVE")
print("=" * 80)

# %%
"""
================================================================================
BLOCK 3: OPTUNA HYPERPARAMETER OPTIMIZATION
Finds optimal learning rate, optimizer, loss weights for champion DNA
================================================================================
"""

# Ensure Block 0 is loaded
assert 'CONFIG' in globals(), "❌ Run Block 0 first!"

import optuna
from optuna.pruners import MedianPruner

print("=" * 80)
print("  BLOCK 3: Optuna Hyperparameter Optimization")
print("=" * 80)

# ============================================================================
# LOAD CHAMPION DNA
# ============================================================================

print("\n📂 Loading champion DNA...")
champion_data = load_checkpoint('champion.pkl')

if not champion_data:
    raise FileNotFoundError("❌ Champion DNA not found! Run Block 1-2 first.")

champion_dna = champion_data['champion_dna']
print(f"✅ Champion loaded: {dna_summary(champion_dna)}")

# ============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================================

def optuna_objective(trial):
    """Optuna objective with parameter validation"""
    cleanup_memory()
    
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    opt_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    dice_w = trial.suggest_float('dice_weight', 0.3, 0.8)
    use_clahe = trial.suggest_categorical('use_clahe', [False, True])
    
    cfg = CONFIG['optuna']
    
    try:
        # Build model
        model = build_full_unetpp(
            champion_dna,
            img_size=CONFIG['resolution']['tuning'],
            deep_supervision=True
        )
        
        # Validate size
        params = model.count_params()
        print(f"   Trial model: {params/1e6:.1f}M params")
        
        # Compile
        loss_fn = make_hybrid_loss(dice_w)
        opt = Adam(lr) if opt_name == 'Adam' else AdamW(lr)
        metrics = [[IoUMetric(name='iou')] for _ in range(5)]
        model.compile(optimizer=opt, loss=[loss_fn] * 5, metrics=metrics)
        
        # Create datasets (small subsets)
        trn_bat = numpy_batches(
            DATASET['train_a'], DATASET['train_b'], DATASET['train_m'],
            CONFIG['resolution']['tuning'], CONFIG['batch_size']['tuning'],
            max_samples=cfg['n_trials'] * 10, use_clahe=use_clahe
        )
        val_bat = numpy_batches(
            DATASET['val_a'], DATASET['val_b'], DATASET['val_m'],
            CONFIG['resolution']['tuning'], CONFIG['batch_size']['tuning'],
            max_samples=30, shuffle=False, use_clahe=use_clahe
        )
        
        # Train
        for epoch in range(cfg['tuning_epochs']):
            random.shuffle(trn_bat)
            for xb, yb in trn_bat:
                # Deep supervision: replicate labels
                model.train_on_batch(xb, [yb] * 5)
        
        # Compute validation IoU
        tp_s = fp_s = fn_s = 0.0
        for xb, yb in val_bat:
            preds = model(xb, training=False)
            pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
            yt = (yb.flatten() > 0.5).astype(np.float32)
            yp = (pred.numpy().flatten() > 0.5).astype(np.float32)
            tp_s += float(np.sum(yt * yp))
            fp_s += float(np.sum((1 - yt) * yp))
            fn_s += float(np.sum(yt * (1 - yp)))
        
        val_iou = tp_s / (tp_s + fp_s + fn_s + 1e-7)
        
        del model, trn_bat, val_bat
        cleanup_memory()
        
        return float(val_iou)
        
    except (tf.errors.ResourceExhaustedError, MemoryError) as oom:
        print(f"   ⚠️  OOM in trial")
        cleanup_memory()
        raise optuna.TrialPruned()
    except Exception as exc:
        print(f"   ⚠️  Trial failed: {exc}")
        cleanup_memory()
        raise optuna.TrialPruned()

# ============================================================================
# RUN OPTUNA
# ============================================================================

print("\n" + "=" * 80)
print("  Starting Optuna Optimization")
print("=" * 80)

cfg = CONFIG['optuna']
print(f"\n⚙️  Settings:")
print(f"   Trials: {cfg['n_trials']}")
print(f"   Tuning epochs: {cfg['tuning_epochs']}")
print(f"   Resolution: {CONFIG['resolution']['tuning']}")

# Check for resume
optuna_data = load_checkpoint('optuna_study.pkl')
if optuna_data:
    print("\n♻️  Found saved study - using best params")
    best_params = optuna_data['best_params']
    trials_df = optuna_data['trials_df']
else:
    study = optuna.create_study(
        direction=cfg['direction'],
        pruner=MedianPruner(n_warmup_steps=cfg['pruner_warmup']),
    )
    
    study.optimize(
        optuna_objective,
        n_trials=cfg['n_trials'],
        gc_after_trial=True,
        show_progress_bar=True,
    )
    
    # Handle all-pruned case
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed:
        print("\n⚠️  All trials pruned - using fallback hyperparameters")
        best_params = {
            'learning_rate': 1e-4,
            'optimizer': 'Adam',
            'dice_weight': 0.5,
            'use_clahe': False,
        }
    else:
        best_params = study.best_params
        print(f"\n✅ Best val IoU: {study.best_value:.4f}")
    
    trials_df = study.trials_dataframe()
    
    # Save
    save_checkpoint({
        'best_params': best_params,
        'trials_df': trials_df,
    }, 'optuna_study.pkl')
    
    save_json(best_params, 'best_hyperparams.json')
    save_csv(trials_df, 'optuna_trials.csv')

print("\n" + "=" * 80)
print("  BLOCK 3 COMPLETE: Hyperparameters Optimized")
print("=" * 80)

print(f"\n🎯 Best Hyperparameters:")
for k, v in best_params.items():
    print(f"   {k:20s}: {v}")

print("\n💡 Next: Run Block 4 for progressive training (256px → 512px)")

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


