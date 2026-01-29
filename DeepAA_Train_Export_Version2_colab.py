# @title
import json
import gc
import math
import time
import datetime
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# 0. Colab 环境设置与工具函数
# ============================================================================

def setup_colab_env():
    """Colab 环境初始化"""
    # 挂载 Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive 已挂载")
        IN_COLAB = True
    except:
        print("⚠ 非 Colab 环境或 Drive 挂载失败")
        IN_COLAB = False
    
    return IN_COLAB


def copy_data_to_local(drive_data_dir: str, local_data_dir: str):
    """
    将 Drive 数据复制到 Colab 本地（加速训练）
    
    关键文件：train_data.npz, metadata.json, scaler.pkl
    """
    drive_path = Path(drive_data_dir)
    local_path = Path(local_data_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    required_files = ['train_data.npz', 'metadata.json', 'scaler.pkl']
    
    print(f"\n复制数据从 Drive 到本地...")
    for fname in required_files:
        src = drive_path / fname
        dst = local_path / fname
        if src.exists():
            if not dst.exists():
                print(f"  复制: {fname}")
                shutil.copy(src, dst)
            else:
                print(f"  跳过 (已存在): {fname}")
        else:
            print(f"  ⚠ 缺失: {fname}")
    
    print(f"✓ 数据已准备到: {local_path}\n")
    return str(local_path)


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        print("⚠ 使用 CPU (训练会很慢)")
        return torch.device('cpu')


def set_seed(seed=42):
    """设置随机种子（包括保存/恢复 rng state）"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rng_state():
    """保存随机状态"""
    return {
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def load_rng_state(rng_state):
    """恢复随机状态"""
    np.random.set_state(rng_state['numpy'])
    torch.set_rng_state(rng_state['torch'])
    if torch.cuda.is_available() and rng_state['cuda'] is not None:
        torch.cuda.set_rng_state_all(rng_state['cuda'])


def safe_log1p(x):
    """安全的log1p变换,处理负值（与第一份代码严格一致）"""
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_log1p(y):
    """log1p的逆变换（与第一份代码严格一致）"""
    return np.sign(y) * (np.exp(np.abs(y)) - 1)


class ScalerAdapter:
    """Scaler适配器：对齐第一份代码的scaler.pkl格式"""
    
    def __init__(self, scaler_path: str):
        scaler_dict = joblib.load(scaler_path)
        
        self.mean = scaler_dict['mean']
        self.std = scaler_dict['std']
        self.log1p_dims = scaler_dict.get('log1p_dims', [0, 1, 4, 7, 10])
        self.fitted = scaler_dict.get('fitted', True)
        
    def _apply_log1p(self, X: np.ndarray) -> np.ndarray:
        X_transformed = X.copy()
        for dim in self.log1p_dims:
            X_transformed[:, :, dim] = safe_log1p(X_transformed[:, :, dim])
        return X_transformed
    
    def _inverse_log1p(self, X: np.ndarray) -> np.ndarray:
        X_original = X.copy()
        for dim in self.log1p_dims:
            X_original[:, :, dim] = inverse_log1p(X_original[:, :, dim])
        return X_original
    
    def transform(self, X_tokens: np.ndarray) -> np.ndarray:
        X_log = self._apply_log1p(X_tokens)
        X_scaled = (X_log - self.mean) / self.std
        return X_scaled
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        X_log = X_scaled * self.std + self.mean
        X_original = self._inverse_log1p(X_log)
        return X_original


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.log_file = output_dir / 'train_log.csv'
        self.logs = []
        
        if not self.log_file.exists():
            pd.DataFrame(columns=[
                'timestamp', 'stage', 'epoch', 'loss', 
                'loss_rec', 'loss_balance', 'loss_sep', 'loss_sparse',
                'loss_sep_phys', 'loss_kl', 'loss_entropy'
            ]).to_csv(self.log_file, index=False)
        else:
            # 恢复已有日志
            existing_logs = pd.read_csv(self.log_file)
            self.logs = existing_logs.to_dict('records')
    
    def log(self, stage: str, epoch: int, loss_dict: Dict[str, float]):
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'stage': stage,
            'epoch': epoch,
            'loss': loss_dict.get('total', 0.0),
            'loss_rec': loss_dict.get('rec', 0.0),
            'loss_balance': loss_dict.get('balance', 0.0),
            'loss_sep': loss_dict.get('sep', 0.0),
            'loss_sparse': loss_dict.get('sparse', 0.0),
            'loss_sep_phys': loss_dict.get('sep_phys', 0.0),
            'loss_kl': loss_dict.get('kl', 0.0),
            'loss_entropy': loss_dict.get('entropy', 0.0)
        }
        
        self.logs.append(log_entry)
        
        pd.DataFrame([log_entry]).to_csv(
            self.log_file, 
            mode='a', 
            header=False, 
            index=False
        )
    
    def save_config(self, config: Dict):
        config_file = self.output_dir / 'run_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def plot_training_curves(self):
        if not self.logs:
            return
        
        df = pd.DataFrame(self.logs)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Loss Curves', fontsize=16)
        
        for stage in df['stage'].unique():
            stage_df = df[df['stage'] == stage]
            
            axes[0, 0].plot(stage_df['epoch'], stage_df['loss'], 
                           label=f'Stage {stage}', marker='o', markersize=3)
            
            if stage_df['loss_rec'].sum() > 0:
                axes[0, 1].plot(stage_df['epoch'], stage_df['loss_rec'],
                               label=f'Stage {stage}', marker='o', markersize=3)
            
            if stage_df['loss_balance'].sum() > 0:
                axes[1, 0].plot(stage_df['epoch'], stage_df['loss_balance'],
                               label=f'Stage {stage} Balance', marker='o', markersize=3)
            if stage_df['loss_sep'].sum() > 0:
                axes[1, 0].plot(stage_df['epoch'], stage_df['loss_sep'],
                               label=f'Stage {stage} Sep', marker='o', markersize=3, linestyle='--')
            
            if stage_df['loss_sep_phys'].sum() > 0:
                axes[1, 1].plot(stage_df['epoch'], stage_df['loss_sep_phys'],
                               label=f'Stage {stage} Phys Sep', marker='o', markersize=3)
            if stage_df['loss_kl'].sum() > 0:
                axes[1, 1].plot(stage_df['epoch'], stage_df['loss_kl'],
                               label=f'Stage {stage} KL', marker='o', markersize=3)
        
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].set_title('Balance & Separation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('Physical Sep / Distillation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300)
        plt.close()
        
        print(f"✓ 训练曲线已保存: {self.output_dir / 'training_curves.png'}")


# ============================================================================
# 1. 数据加载与分层采样 (Colab优化)
# ============================================================================

class DeepAATokenDataset(Dataset):
    """DeepAA Token数据集"""
    
    def __init__(self, 
                 X_tokens: np.ndarray,
                 aez_labels: Optional[np.ndarray] = None,
                 sample_ids: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X_tokens)
        self.aez = aez_labels
        self.ids = sample_ids
        self.N = len(X_tokens)
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        item = {'X': self.X[idx]}
        if self.aez is not None:
            item['aez'] = self.aez[idx]
        if self.ids is not None:
            item['id'] = self.ids[idx]
        return item


class StratifiedBatchSampler(Sampler):
    """
    分层均衡批次采样器 (Colab优化版)
    
    关键改进：
    - n_batches_per_epoch 控制每个epoch步数
    - 循环采样避免被最小AEZ卡住
    - 支持 AEZ + climate_zone 组合分层
    """
    
    def __init__(self, 
                 aez_labels: np.ndarray,
                 batch_size: int,
                 samples_per_stratum: Optional[int] = None,
                 shuffle: bool = True,
                 n_batches_per_epoch: Optional[int] = None):
        """
        参数:
            aez_labels: (N,) 分层标签 (可以是 AEZ 或 AEZ*100+climate_zone)
            batch_size: 批次大小
            samples_per_stratum: 每层每批抽样数
            shuffle: 是否打乱
            n_batches_per_epoch: 每个epoch的批次数 (关键:控制训练时间)
        """
        self.aez_labels = aez_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 按分层分组索引
        self.stratum_indices = defaultdict(list)
        for idx, aez in enumerate(aez_labels):
            self.stratum_indices[aez].append(idx)
        
        self.strata = list(self.stratum_indices.keys())
        self.n_strata = len(self.strata)
        
        # 每层每批抽样数
        if samples_per_stratum is None:
            self.samples_per_stratum = max(1, batch_size // self.n_strata)
        else:
            self.samples_per_stratum = samples_per_stratum
        
        # 每epoch批次数 (关键：控制训练时间)
        if n_batches_per_epoch is not None:
            self.n_batches = n_batches_per_epoch
        else:
            # 默认：基于总样本数
            total_samples = len(aez_labels)
            self.n_batches = total_samples // batch_size
        
        print(f"  分层采样: {self.n_strata} 层, 每epoch {self.n_batches} 批次, 每批次目标大小: {self.batch_size}")
        
    def __iter__(self):
        # 为每层创建循环迭代器
        stratum_iters = {}
        for aez, indices in self.stratum_indices.items():
            indices_copy = indices.copy()
            if self.shuffle:
                np.random.shuffle(indices_copy)
            stratum_iters[aez] = iter(self._cycle_indices(indices_copy, self.shuffle))
        
        # 生成批次
        batches_yielded = 0
        for batch_idx in range(self.n_batches):
            batch = []
            
            # 从每层采样固定数量
            for aez in self.strata:
                for _ in range(self.samples_per_stratum):
                    idx = next(stratum_iters[aez])  # _cycle_indices 保证不会 StopIteration
                    batch.append(idx)
            
            # 补充到 batch_size (循环采样保证一定能填满)
            while len(batch) < self.batch_size:
                for aez in self.strata:
                    idx = next(stratum_iters[aez])
                    batch.append(idx)
                    if len(batch) >= self.batch_size:
                        break
            
            # 打乱并截取
            if self.shuffle:
                np.random.shuffle(batch)
            
            yield batch[:self.batch_size]
            batches_yielded += 1
        
        # 验证：确保生成了正确数量的批次
        if batches_yielded != self.n_batches:
            print(f"  ⚠ 警告: 期望 {self.n_batches} 批次, 实际生成 {batches_yielded} 批次")
    
    def _cycle_indices(self, indices, shuffle):
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for idx in indices:
                yield idx
    
    def __len__(self):
        return self.n_batches


# ============================================================================
# 2. Entmax激活函数 (优化: n_iter可配置)
# ============================================================================

class Entmax15(nn.Module):
    """Entmax 1.5激活函数 (n_iter可配置)"""
    
    def __init__(self, dim=-1, n_iter=15):
        super().__init__()
        self.dim = dim
        self.n_iter = n_iter  # 默认15，比50快很多
    
    def forward(self, logits):
        return entmax15(logits, dim=self.dim, n_iter=self.n_iter)


def entmax15(logits, dim=-1, n_iter=15):
    """Entmax 1.5的简化实现 (n_iter可配置)"""
    alpha = F.softmax(logits, dim=dim)
    
    for _ in range(n_iter):
        alpha_sqrt = torch.sqrt(torch.clamp(alpha, min=1e-12))
        tau = (torch.sum(alpha_sqrt, dim=dim, keepdim=True) - 1) / \
              torch.sum(1.0 / alpha_sqrt, dim=dim, keepdim=True)
        
        alpha_new = torch.clamp(
            (alpha_sqrt + tau) ** 2,
            min=0.0
        )
        
        alpha = alpha_new / torch.sum(alpha_new, dim=dim, keepdim=True)
    
    return alpha


# ============================================================================
# 3. 模型定义 (支持AMP)
# ============================================================================

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim=11, d_model=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 n_heads=8, 
                 n_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
    def forward(self, x, mask=None):
        if mask is not None:
            src_key_padding_mask = mask
        else:
            src_key_padding_mask = None
        
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out


class PrototypeMixingHead(nn.Module):
    def __init__(self, d_model=256, n_prototypes=50, use_entmax=True, entmax_n_iter=15):
        super().__init__()
        
        self.pooling = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.to_logits = nn.Linear(d_model, n_prototypes)
        
        self.use_entmax = use_entmax
        if use_entmax:
            self.activation = Entmax15(dim=-1, n_iter=entmax_n_iter)
        else:
            self.activation = nn.Softmax(dim=-1)
    
    def forward(self, z):
        z_global = torch.mean(z, dim=1)
        z_global = self.pooling(z_global)
        logits = self.to_logits(z_global)
        w = self.activation(logits)
        return w, logits


class GenerativeDecoder(nn.Module):
    def __init__(self, d_model=256, output_dim=11):
        super().__init__()
        
        self.expand = nn.Linear(d_model, 11 * d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.to_tokens = nn.Linear(d_model, output_dim)
        
    def forward(self, z_hat):
        B = z_hat.size(0)
        x = self.expand(z_hat)
        x = x.view(B, 11, -1)
        x = self.decoder(x)
        X_hat = self.to_tokens(x)
        return X_hat


class DeepAABigModel(nn.Module):
    """DeepAA大模型 (支持AMP)"""
    
    def __init__(self,
                 n_prototypes=50,
                 d_model=256,
                 n_heads=8,
                 n_encoder_layers=4,
                 use_entmax=True,
                 entmax_n_iter=15):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.d_model = d_model
        
        self.token_embedding = TokenEmbedding(input_dim=11, d_model=d_model)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers
        )
        self.prototype_head = PrototypeMixingHead(
            d_model=d_model,
            n_prototypes=n_prototypes,
            use_entmax=use_entmax,
            entmax_n_iter=entmax_n_iter
        )
        self.decoder = GenerativeDecoder(d_model=d_model, output_dim=11)
        
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model))
        nn.init.orthogonal_(self.prototypes)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, 11))
        self.mask_decoder_head = nn.Linear(d_model, 11)
    
    def forward(self, X, mask=None, stage='B'):
        B = X.size(0)
        
        if mask is not None:
            X = X.clone()
            num_masked = mask.sum().item()
            X[mask] = self.mask_token.squeeze(0).expand(num_masked, -1)
        
        z = self.token_embedding(X)
        z = self.encoder(z, mask=mask)
        
        if stage == 'A':
            token_preds = self.mask_decoder_head(z)
            return {'X_hat': token_preds, 'z': z}
        else:
            w, logits = self.prototype_head(z)
            z_hat = torch.matmul(w, self.prototypes)
            X_hat = self.decoder(z_hat)
            
            return {
                'X_hat': X_hat,
                'w': w,
                'logits': logits,
                'z': z,
                'z_hat': z_hat
            }
    
    def generate_from_prototype(self, k):
        with torch.no_grad():
            p_k = self.prototypes[k:k+1]
            X_hat_k = self.decoder(p_k)
        return X_hat_k.squeeze(0)


# ============================================================================
# 4. 训练阶段 (带AMP和Checkpoint)
# ============================================================================

class MaskedReconstructionTrainer:
    """阶段A训练器: Masked Reconstruction (带AMP)"""
    
    def __init__(self, 
                 model: DeepAABigModel,
                 device: torch.device,
                 mask_ratio: float = 0.3,
                 use_amp: bool = True):
        self.model = model
        self.device = device
        self.mask_ratio = mask_ratio
        self.use_amp = use_amp and device.type == 'cuda'
        
        if self.use_amp:
            self.scaler = GradScaler()
    
    def create_random_mask(self, batch_size, n_tokens=11):
        mask = torch.rand(batch_size, n_tokens) < self.mask_ratio
        mask[mask.sum(dim=1) == 0, 0] = True
        return mask.to(self.device)
    
    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        n_batches = 0
        expected_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Stage A]", total=expected_batches)
        
        for batch in pbar:
            X = batch['X'].to(self.device)
            B = X.size(0)
            
            mask = self.create_random_mask(B)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(X, mask=mask, stage='A')
                    X_hat = outputs['X_hat']
                    loss = F.mse_loss(X_hat[mask], X[mask])
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(X, mask=mask, stage='A')
                X_hat = outputs['X_hat']
                loss = F.mse_loss(X_hat[mask], X[mask])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'batch': f'{n_batches}/{expected_batches}'})
        
        # 验证实际批次数
        if n_batches != expected_batches:
            print(f"\n  ⚠ 警告: 期望 {expected_batches} 批次, 实际训练 {n_batches} 批次")
        
        return total_loss / n_batches


class PrototypeMixingTrainer:
    """阶段B训练器: 原型混合 (带AMP)"""
    
    def __init__(self,
                 model: DeepAABigModel,
                 device: torch.device,
                 lambda_balance: float = 0.01,
                 lambda_sep_latent: float = 0.001,
                 lambda_sparse: float = 0.0,  # 关闭 sparse loss，避免过度平滑
                 warmup_epochs: int = 5,
                 use_amp: bool = True):
        self.model = model
        self.device = device
        self.lambda_balance = lambda_balance
        self.lambda_sep_latent = lambda_sep_latent
        self.lambda_sparse = lambda_sparse
        self.warmup_epochs = warmup_epochs
        self.use_amp = use_amp and device.type == 'cuda'
        
        if self.use_amp:
            self.scaler = GradScaler()
    
    def compute_balance_loss(self, w):
        """使用对称 KL 散度鼓励原型使用均衡，但不强制完全均匀"""
        mean_w = w.mean(dim=0)
        uniform = torch.ones_like(mean_w) / self.model.n_prototypes
        
        # 对称 KL: (KL(mean_w || uniform) + KL(uniform || mean_w)) / 2
        kl_forward = F.kl_div(
            (mean_w + 1e-8).log(),
            uniform,
            reduction='batchmean'
        )
        kl_backward = F.kl_div(
            uniform.log(),
            mean_w + 1e-8,
            reduction='batchmean'
        )
        loss_balance = (kl_forward + kl_backward) / 2
        return loss_balance
    
    def compute_separation_loss_latent(self):
        """鼓励原型在潜在空间分离，但不要求完全正交"""
        P = self.model.prototypes
        K = P.size(0)
        P_norm = F.normalize(P, dim=1)
        sim_matrix = torch.matmul(P_norm, P_norm.t())
        mask = ~torch.eye(K, dtype=torch.bool, device=self.device)
        similarities = sim_matrix[mask]
        
        # 降低 margin：0.7 对应约 45° 角度，比 0.5 (60°) 更宽松
        margin = 0.7
        loss_sep = torch.clamp(similarities - margin, min=0).mean()
        return loss_sep
    
    def compute_sparse_loss(self, w):
        """鼓励适度稀疏，但不要求极端稀疏（避免只用少数原型）
        
        返回负的归一化熵，最小化它等于最大化熵（更平滑）
        注意：原实现是错误的，会导致极端稀疏
        """
        entropy = -(w * (w + 1e-8).log()).sum(dim=1).mean()
        max_entropy = math.log(self.model.n_prototypes)
        
        # 返回负值：最小化 -entropy 等于最大化 entropy（鼓励平滑分布）
        # 如果想要稀疏，应该设置 lambda_sparse 为负值
        normalized_entropy = entropy / max_entropy
        return -normalized_entropy  # 最小化负熵 = 最大化熵
    
    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        loss_dict = defaultdict(float)
        n_batches = 0
        expected_batches = len(dataloader)
        
        if epoch < self.warmup_epochs:
            alpha_sep = self.lambda_sep_latent * (epoch / self.warmup_epochs)
            alpha_sparse = self.lambda_sparse * (epoch / self.warmup_epochs)
        else:
            alpha_sep = self.lambda_sep_latent
            alpha_sparse = self.lambda_sparse
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Stage B]", total=expected_batches)
        
        for batch in pbar:
            X = batch['X'].to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(X, stage='B')
                    X_hat = outputs['X_hat']
                    w = outputs['w']
                    
                    loss_rec = F.mse_loss(X_hat, X)
                    loss_balance = self.compute_balance_loss(w)
                    loss_sep = self.compute_separation_loss_latent()
                    loss_sparse = self.compute_sparse_loss(w)
                    
                    loss = (loss_rec + 
                           self.lambda_balance * loss_balance +
                           alpha_sep * loss_sep +
                           alpha_sparse * loss_sparse)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(X, stage='B')
                X_hat = outputs['X_hat']
                w = outputs['w']
                
                loss_rec = F.mse_loss(X_hat, X)
                loss_balance = self.compute_balance_loss(w)
                loss_sep = self.compute_separation_loss_latent()
                loss_sparse = self.compute_sparse_loss(w)
                
                loss = (loss_rec + 
                       self.lambda_balance * loss_balance +
                       alpha_sep * loss_sep +
                       alpha_sparse * loss_sparse)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            loss_dict['rec'] += loss_rec.item()
            loss_dict['balance'] += loss_balance.item()
            loss_dict['sep'] += loss_sep.item()
            loss_dict['sparse'] += loss_sparse.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{loss_rec.item():.4f}',
                'batch': f'{n_batches}/{expected_batches}'
            })
        
        # 验证实际批次数
        if n_batches != expected_batches:
            print(f"\n  ⚠ 警告: 期望 {expected_batches} 批次, 实际训练 {n_batches} 批次")
        
        avg_loss = total_loss / n_batches
        avg_loss_dict = {k: v / n_batches for k, v in loss_dict.items()}
        
        return avg_loss, avg_loss_dict


class PhysicalSeparationTrainer:
    """阶段C训练器: 物理空间去重 (带AMP) - 修复版"""
    
    def __init__(self,
                 model: DeepAABigModel,
                 device: torch.device,
                 scaler_path: str,
                 lambda_sep_phys: float = 0.05,  # 降低权重避免过度惩罚
                 margin_phys: float = 3.0,  # 降低 margin 要求
                 top_k_neighbors: int = 3,  # 只关注最近的3个邻居
                 warmup_epochs: int = 3,  # 添加 warmup
                 update_proto_every_n_batches: int = 50,
                 use_amp: bool = True):
        self.model = model
        self.device = device
        self.lambda_sep_phys = lambda_sep_phys
        self.margin_phys = margin_phys
        self.top_k_neighbors = top_k_neighbors
        self.warmup_epochs = warmup_epochs
        self.update_proto_every_n_batches = update_proto_every_n_batches
        self.use_amp = use_amp and device.type == 'cuda'
        
        if self.use_amp:
            self.scaler = GradScaler()
        
        self.scaler_adapter = ScalerAdapter(scaler_path)
        self.cached_proto_phys = None
        self.batch_counter = 0
    
    def generate_physical_prototypes_with_grad(self):
        """生成物理空间原型 - 保留梯度版本"""
        K = self.model.n_prototypes
        
        # 直接使用原型参数，不调用 generate_from_prototype (它有 no_grad)
        # 而是直接通过 decoder 前向传播
        X_proto = self.model.decoder(self.model.prototypes)  # (K, 11, 11)
        
        # 转换到物理空间 (这里需要保持梯度，用 torch 操作)
        # 简化：直接在标准化空间计算距离，然后乘以一个近似scale
        # 或者：保持在标准化空间计算 sep_phys
        
        # 为了简单且保留梯度，我们直接用 decoder 输出的标准化空间
        X_proto_flat = X_proto.view(K, -1)  # (K, 121)
        
        return X_proto_flat
    
    def compute_physical_separation_loss(self):
        """计算物理分离损失 - 修复版：保留梯度"""
        # 每 N 个 batch 更新一次（但这次保留梯度）
        X_proto = self.generate_physical_prototypes_with_grad()
        K = X_proto.size(0)
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(X_proto, X_proto, p=2)
        
        # 排除对角线
        mask = ~torch.eye(K, dtype=torch.bool, device=self.device)
        dist_matrix_masked = dist_matrix.clone()
        dist_matrix_masked[~mask] = 1e6  # 对角线设为大值
        
        # 找最近的 k 个邻居
        topk_dists, _ = torch.topk(
            dist_matrix_masked,
            k=min(self.top_k_neighbors, K-1),
            dim=1,
            largest=False
        )
        
        # Hinge loss: 希望最近邻距离 > margin
        loss_sep = torch.clamp(self.margin_phys - topk_dists, min=0).mean()
        
        return loss_sep
    
    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        loss_dict = defaultdict(float)
        n_batches = 0
        expected_batches = len(dataloader)
        
        # Warmup 调度：逐渐增加物理分离损失的权重
        if epoch <= self.warmup_epochs:
            alpha_phys = self.lambda_sep_phys * (epoch / self.warmup_epochs)
        else:
            alpha_phys = self.lambda_sep_phys
        
        self.batch_counter = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Stage C]", total=expected_batches)
        
        for batch in pbar:
            X = batch['X'].to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(X, stage='B')  # Stage C 用 B 的前向
                    X_hat = outputs['X_hat']
                    loss_rec = F.mse_loss(X_hat, X)
                    
                    # 关键修复：sep_phys 现在有梯度了
                    loss_sep_phys = self.compute_physical_separation_loss()
                    
                    loss = loss_rec + alpha_phys * loss_sep_phys
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(X, stage='B')
                X_hat = outputs['X_hat']
                loss_rec = F.mse_loss(X_hat, X)
                
                loss_sep_phys = self.compute_physical_separation_loss()
                
                loss = loss_rec + alpha_phys * loss_sep_phys
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            
            self.batch_counter += 1
            
            total_loss += loss.item()
            loss_dict['rec'] += loss_rec.item()
            loss_dict['sep_phys'] += loss_sep_phys.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rec': f'{loss_rec.item():.4f}',
                'sep_phys': f'{loss_sep_phys.item():.4f}',
                'batch': f'{n_batches}/{expected_batches}'
            })
        
        if n_batches != expected_batches:
            print(f"\n  ⚠ 警告: 期望 {expected_batches} 批次, 实际训练 {n_batches} 批次")
        
        avg_loss = total_loss / n_batches
        avg_loss_dict = {k: v / n_batches for k, v in loss_dict.items()}
        
        return avg_loss, avg_loss_dict


# ============================================================================
# 5. Checkpoint管理 (Colab断点续训核心)
# ============================================================================

class CheckpointManager:
    """Checkpoint管理器: 支持断点续训"""
    
    def __init__(self, output_dir: Path):
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_path = self.checkpoint_dir / 'last.pt'
    
    def save(self, 
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             stage: str,
             epoch: int,
             loss: float,
             config: Dict,
             scaler: Optional[GradScaler] = None):
        """保存完整恢复状态"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stage': stage,
            'epoch': epoch,
            'loss': loss,
            'config': config,
            'rng_state': save_rng_state(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # 保存阶段checkpoint
        stage_path = self.checkpoint_dir / f'stage{stage}_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, stage_path)
        
        # 保存last.pt (覆盖)
        torch.save(checkpoint, self.last_checkpoint_path)
        
        print(f"  ✓ Checkpoint saved: {stage_path.name}")
    
    def load_last(self,
                  model: nn.Module,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  scaler: Optional[GradScaler] = None,
                  device: torch.device = torch.device('cpu')) -> Optional[Dict]:
        """加载last.pt并恢复状态"""
        if not self.last_checkpoint_path.exists():
            return None
        
        print(f"\n发现断点: {self.last_checkpoint_path}")
        checkpoint = torch.load(self.last_checkpoint_path, map_location=device, weights_only=False)
        
        # 恢复模型
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复scaler
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 恢复随机状态
        if 'rng_state' in checkpoint:
            load_rng_state(checkpoint['rng_state'])
        
        print(f"  ✓ 从 Stage {checkpoint['stage']} Epoch {checkpoint['epoch']} 恢复")
        print(f"  ✓ 保存时间: {checkpoint['timestamp']}")
        
        return checkpoint
    
    def get_resume_info(self) -> Optional[Dict]:
        """获取恢复信息 (不加载模型)"""
        if not self.last_checkpoint_path.exists():
            return None
        
        checkpoint = torch.load(self.last_checkpoint_path, map_location='cpu', weights_only=False)
        return {
            'stage': checkpoint['stage'],
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'timestamp': checkpoint['timestamp']
        }


# ============================================================================
# 6. 导出器
# ============================================================================

class PrototypeExporter:
    """原型解释包导出器"""
    
    def __init__(self, 
                 model: DeepAABigModel,
                 scaler_path: str,
                 metadata_path: str,
                 output_dir: str,
                 device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = ScalerAdapter(scaler_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.token_order = self.metadata['token_order']
        self.feature_names = self.metadata['feature_names']
    
    def export_prototype_tokens(self):
        K = self.model.n_prototypes
        X_proto_list = []
        
        for k in range(K):
            X_hat_k = self.model.generate_from_prototype(k)
            X_proto_list.append(X_hat_k.cpu().numpy())
        
        X_proto = np.stack(X_proto_list, axis=0)
        X_proto_phys = self.scaler.inverse_transform(X_proto)
        
        rows = []
        for k in range(K):
            row = {'prototype_id': k}
            for t_idx, token_name in enumerate(self.token_order):
                for f_idx, feat_name in enumerate(self.feature_names):
                    col_name = f"{token_name}_{feat_name}"
                    row[col_name] = X_proto_phys[k, t_idx, f_idx]
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / 'prototype_tokens.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"✓ 导出原型token特征: {csv_path}")
        return df, X_proto_phys
    
    def export_prototype_curves(self, X_proto_phys):
        K = self.model.n_prototypes
        n_days = 365
        t = np.linspace(0, 2*np.pi, n_days)
        
        curves_dir = self.output_dir / 'prototype_curves'
        curves_dir.mkdir(exist_ok=True)
        
        for k in range(K):
            curves_k = {}
            
            for t_idx, token_name in enumerate(self.token_order):
                a0 = X_proto_phys[k, t_idx, 0]
                A1, u1, v1 = X_proto_phys[k, t_idx, 1:4]
                A2, u2, v2 = X_proto_phys[k, t_idx, 4:7]
                A3, u3, v3 = X_proto_phys[k, t_idx, 7:10]
                
                c1, s1 = A1 * u1, A1 * v1
                c2, s2 = A2 * u2, A2 * v2
                c3, s3 = A3 * u3, A3 * v3
                
                curve = (a0 + 
                        c1 * np.cos(t) + s1 * np.sin(t) +
                        c2 * np.cos(2*t) + s2 * np.sin(2*t) +
                        c3 * np.cos(3*t) + s3 * np.sin(3*t))
                
                curves_k[token_name] = curve
            
            np.savez(curves_dir / f'prototype_{k:03d}.npz', **curves_k, days=np.arange(n_days))
        
        print(f"✓ 导出原型曲线: {curves_dir}")
        return curves_dir
    
    def export_physical_distance_matrix(self, X_proto_phys):
        K = self.model.n_prototypes
        X_flat = X_proto_phys.reshape(K, -1)
        
        from scipy.spatial.distance import pdist, squareform
        D_phys = squareform(pdist(X_flat, metric='euclidean'))
        
        npy_path = self.output_dir / 'D_phys.npy'
        np.save(npy_path, D_phys)
        
        csv_path = self.output_dir / 'D_phys.csv'
        pd.DataFrame(D_phys).to_csv(csv_path, index=False)
        
        print(f"✓ 导出物理距离矩阵: {npy_path}")
        return D_phys
    
    def export_all(self):
        print("\n" + "="*80)
        print("导出原型解释包")
        print("="*80 + "\n")
        
        df_tokens, X_proto_phys = self.export_prototype_tokens()
        curves_dir = self.export_prototype_curves(X_proto_phys)
        D_phys = self.export_physical_distance_matrix(X_proto_phys)
        
        print("\n✓ 原型解释包导出完成!\n")
        
        return {
            'tokens_df': df_tokens,
            'tokens_phys': X_proto_phys,
            'curves_dir': curves_dir,
            'D_phys': D_phys
        }


class DistillationMLPStudent(nn.Module):
    """蒸馏学生模型"""
    
    def __init__(self,
                 input_dim=121,
                 hidden_dims=[256, 128],
                 output_dim=50,
                 activation='tanh',
                 temperature=1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation
        self.temperature = temperature
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.network(x)
        probs = F.softmax(logits / self.temperature, dim=-1)
        return probs, logits


class DistillationTrainer:
    """蒸馏训练器 - 修复KL方向"""
    
    def __init__(self,
                 teacher_model: DeepAABigModel,
                 student_model: DistillationMLPStudent,
                 device: torch.device,
                 lambda_entropy: float = 0.001,
                 use_amp: bool = True):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.device = device
        self.lambda_entropy = lambda_entropy
        self.use_amp = use_amp and device.type == 'cuda'
        
        if self.use_amp:
            self.scaler = GradScaler()
    
    @torch.no_grad()
    def generate_soft_labels(self, X):
        if self.use_amp:
            with autocast():
                outputs = self.teacher(X, stage='B')
        else:
            outputs = self.teacher(X, stage='B')
        return outputs['w']
    
    def train_epoch(self, dataloader, optimizer, epoch, max_batches: Optional[int] = None):
        self.student.train()
        total_loss = 0
        total_kl = 0
        total_entropy = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Distill Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            X = batch['X'].to(self.device)
            B = X.size(0)
            X_flat = X.view(B, -1)
            
            w_teacher = self.generate_soft_labels(X)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    w_student, _ = self.student(X_flat)
                    
                    # 修复KL散度计算方向
                    # KL(P_teacher || P_student) = sum(P_teacher * log(P_teacher / P_student))
                    # PyTorch的kl_div期望输入是log概率，目标是概率
                    # kl_div(log_student, teacher) = teacher * (log(teacher) - log_student)
                    
                    # 正确写法：student 取 log，teacher 作为目标
                    loss_kl = F.kl_div(
                        (w_student + 1e-8).log(),  # log(student)
                        w_teacher,                  # teacher (概率)
                        reduction='batchmean',
                        log_target=False            # teacher 不是 log 形式
                    )
                    
                    # 熵正则化：鼓励学生输出更集中
                    entropy = -(w_student * (w_student + 1e-8).log()).sum(dim=1).mean()
                    
                    # 最小化 KL，同时最小化熵（更稀疏）
                    loss = loss_kl + self.lambda_entropy * entropy
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                w_student, _ = self.student(X_flat)
                
                loss_kl = F.kl_div(
                    (w_student + 1e-8).log(),
                    w_teacher,
                    reduction='batchmean',
                    log_target=False
                )
                
                entropy = -(w_student * (w_student + 1e-8).log()).sum(dim=1).mean()
                loss = loss_kl + self.lambda_entropy * entropy
                
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            total_kl += loss_kl.item()
            total_entropy += entropy.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'kl': f'{loss_kl.item():.4f}',
                'entropy': f'{entropy.item():.4f}'
            })
        
        avg_loss = total_loss / n_batches
        avg_kl = total_kl / n_batches
        avg_entropy = total_entropy / n_batches
        
        return avg_loss, {'kl': avg_kl, 'entropy': avg_entropy}


class DistillationExporter:
    """蒸馏模型导出器 (GEE格式)"""
    
    def __init__(self, student_model: DistillationMLPStudent, output_dir: str, metadata: Dict):
        self.student = student_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
    
    def export_weights(self):
        weights_dict = {}
        layer_idx = 0
        
        for name, param in self.student.named_parameters():
            if 'weight' in name:
                weights_dict[f'layer_{layer_idx}_weight'] = param.detach().cpu().numpy().tolist()
            elif 'bias' in name:
                weights_dict[f'layer_{layer_idx}_bias'] = param.detach().cpu().numpy().tolist()
                layer_idx += 1
        
        json_path = self.output_dir / 'distill_weights.json'
        with open(json_path, 'w') as f:
            json.dump(weights_dict, f, indent=2)
        
        print(f"✓ 导出蒸馏权重: {json_path}")
        return json_path
    
    def export_config(self):
        config = {
            'input_dim': self.student.input_dim,
            'hidden_dims': self.student.hidden_dims,
            'output_dim': self.student.output_dim,
            'activation': self.student.activation_name,
            'temperature': self.student.temperature,
            'token_order': self.metadata['token_order'],
            'feature_names': self.metadata['feature_names'],
            'flatten_order': 'row-major'
        }
        
        json_path = self.output_dir / 'distill_config.json'
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ 导出蒸馏配置: {json_path}")
        return json_path
    
    def export_all(self):
        print("\n" + "="*80)
        print("导出蒸馏推断包 (GEE部署用)")
        print("="*80 + "\n")
        
        weights_path = self.export_weights()
        config_path = self.export_config()
        
        print("\n✓ 蒸馏推断包导出完成!\n")
        
        return {'weights_path': weights_path, 'config_path': config_path}


# ============================================================================
# 7. 主训练管道 (Colab优化版)
# ============================================================================

class DeepAATrainingPipelineColab:
    """DeepAA完整训练管道 (Colab优化版) - 修复断点续训"""
    
    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 n_prototypes: int = 50,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_encoder_layers: int = 2,
                 entmax_n_iter: int = 15,
                 use_amp: bool = True,
                 device: Optional[torch.device] = None,
                 seed: int = 42):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_prototypes = n_prototypes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.entmax_n_iter = entmax_n_iter
        self.use_amp = use_amp
        self.device = device or get_device()
        self.seed = seed
        
        set_seed(seed)
        
        print(f"\n{'='*80}")
        print(f"DeepAA Training Pipeline (Colab Optimized)")
        print(f"{'='*80}")
        print(f"设备: {self.device}")
        print(f"AMP: {'启用' if self.use_amp and self.device.type == 'cuda' else '禁用'}")
        print(f"原型数: {n_prototypes}")
        print(f"模型维度: {d_model}")
        print(f"Encoder层数: {n_encoder_layers}")
        print(f"{'='*80}\n")
        
        # 加载元数据
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # 模型
        self.model = None
        self.student_model = None
        
        # Checkpoint管理器
        self.checkpoint_mgr = CheckpointManager(self.output_dir)
        
        # 日志记录器
        self.logger = TrainingLogger(self.output_dir)
        
        # 保存运行配置
        self._save_run_config()
    
    def _save_run_config(self):
        config = {
            'timestamp': datetime.datetime.now().isoformat(),
            'seed': self.seed,
            'device': str(self.device),
            'n_prototypes': self.n_prototypes,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_encoder_layers': self.n_encoder_layers,
            'entmax_n_iter': self.entmax_n_iter,
            'use_amp': self.use_amp,
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'pytorch_version': torch.__version__
        }
        self.logger.save_config(config)
    
    def load_data(self, 
                  batch_size=512,
                  use_stratified=True,
                  num_workers=2,
                  n_batches_per_epoch: Optional[int] = None,
                  samples_per_stratum: Optional[int] = None):
        """加载数据 (支持分层采样和每epoch步数控制)"""
        print("加载训练数据...")
        
        train_data = np.load(self.data_dir / 'train_data.npz')
        X_train = train_data['X']
        aez_train = train_data['aez'] if 'aez' in train_data and len(train_data['aez']) > 0 else None
        ids_train = train_data['ids'] if 'ids' in train_data and len(train_data['ids']) > 0 else None
        
        # 检查是否有 climate_zone 用于组合分层
        if 'climate_zone' in train_data and aez_train is not None:
            cz_train = train_data['climate_zone']
            # 组合标签: aez * 100 + climate_zone
            stratum_labels = aez_train * 100 + cz_train
            print(f"  使用 AEZ + Climate Zone 组合分层")
        else:
            stratum_labels = aez_train
        
        print(f"  训练集: {len(X_train):,} 样本")
        
        train_dataset = DeepAATokenDataset(X_train, stratum_labels, ids_train)
        
        if use_stratified and stratum_labels is not None:
            sampler = StratifiedBatchSampler(
                stratum_labels,
                batch_size=batch_size,
                samples_per_stratum=samples_per_stratum,
                shuffle=True,
                n_batches_per_epoch=n_batches_per_epoch
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=True if num_workers > 0 else False
            )
        else:
            print("使用随机采样")
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=True if num_workers > 0 else False
            )
        
        return train_loader
    
    def build_model(self):
        """构建模型"""
        print("\n构建DeepAA Big Model...")
        
        self.model = DeepAABigModel(
            n_prototypes=self.n_prototypes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            use_entmax=True,
            entmax_n_iter=self.entmax_n_iter
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  参数量: {n_params:,}")
        
        return self.model
    
    def _load_best_checkpoint_for_stage(self, target_stage: str):
        """
        为指定阶段加载最佳 checkpoint
        
        关键修复：确保 Stage C 从 Stage B 的最佳状态开始
        """
        checkpoint_dir = self.output_dir / 'checkpoints'
        
        # 确定前置阶段
        if target_stage == 'B':
            prev_stage = 'A'
        elif target_stage == 'C':
            prev_stage = 'B'
        else:
            return None
        
        # 查找前置阶段的所有 checkpoint
        pattern = f'stage{prev_stage}_epoch_*.pt'
        checkpoints = list(checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            print(f"  ⚠ 未找到 Stage {prev_stage} 的 checkpoint")
            return None
        
        # 按 epoch 排序，取最后一个（最新/最好的）
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        best_ckpt = checkpoints[-1]
        
        print(f"\n  📥 加载 Stage {prev_stage} 的最佳权重: {best_ckpt.name}")
        
        checkpoint = torch.load(best_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 验证加载成功
        print(f"     Stage: {checkpoint['stage']}, Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
        
        return checkpoint
    
    def train_stage_A(self, train_loader, n_epochs=10, lr=1e-3, start_epoch=1):
        """训练阶段A: Masked Reconstruction"""
        print("\n" + "="*80)
        print("阶段A: Masked Reconstruction 预训练")
        print("="*80 + "\n")
        
        trainer = MaskedReconstructionTrainer(
            self.model, self.device, use_amp=self.use_amp
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # 如果是恢复训练，加载优化器状态
        if start_epoch > 1:
            ckpt = self.checkpoint_mgr.load_last(self.model, optimizer, 
                                                  trainer.scaler if self.use_amp else None,
                                                  self.device)
        
        for epoch in range(start_epoch, n_epochs + 1):
            loss = trainer.train_epoch(train_loader, optimizer, epoch)
            print(f"Epoch {epoch}/{n_epochs} - Loss: {loss:.4f}")
            
            self.logger.log('A', epoch, {'total': loss, 'rec': loss})
            
            # 每epoch保存checkpoint
            self.checkpoint_mgr.save(
                self.model, optimizer, 'A', epoch, loss,
                {'n_epochs': n_epochs, 'lr': lr},
                trainer.scaler if self.use_amp else None
            )
        
        print("\n✓ 阶段A完成\n")
    
    def train_stage_B(self, train_loader, n_epochs=20, lr=5e-4, start_epoch=1):
        """训练阶段B: 原型混合 - 修复版"""
        print("\n" + "="*80)
        print("阶段B: Entmax原型混合 + 生成式Decoder")
        print("="*80 + "\n")
        
        # 关键修复：如果是从头开始 Stage B，先加载 Stage A 的最佳权重
        if start_epoch == 1:
            self._load_best_checkpoint_for_stage('B')
        
        trainer = PrototypeMixingTrainer(
            self.model, self.device, use_amp=self.use_amp
        )
        
        # 分层学习率
        optimizer = torch.optim.AdamW([
            {'params': self.model.token_embedding.parameters(), 'lr': lr * 0.5},
            {'params': self.model.encoder.parameters(), 'lr': lr * 0.5},
            {'params': self.model.prototypes, 'lr': lr * 2.0},
            {'params': self.model.decoder.parameters(), 'lr': lr * 2.0},
            {'params': self.model.prototype_head.parameters(), 'lr': lr}
        ])
        
        # 如果是恢复训练（start_epoch > 1），加载完整状态
        if start_epoch > 1:
            ckpt = self.checkpoint_mgr.load_last(self.model, optimizer,
                                                  trainer.scaler if self.use_amp else None,
                                                  self.device)
        
        for epoch in range(start_epoch, n_epochs + 1):
            loss, loss_dict = trainer.train_epoch(train_loader, optimizer, epoch)
            print(f"Epoch {epoch}/{n_epochs} - "
                  f"Loss: {loss:.4f} | "
                  f"Rec: {loss_dict['rec']:.4f} | "
                  f"Balance: {loss_dict['balance']:.4f}")
            
            loss_dict['total'] = loss
            self.logger.log('B', epoch, loss_dict)
            
            self.checkpoint_mgr.save(
                self.model, optimizer, 'B', epoch, loss,
                {'n_epochs': n_epochs, 'lr': lr},
                trainer.scaler if self.use_amp else None
            )
        
        print("\n✓ 阶段B完成\n")
    
    def train_stage_C(self, train_loader, n_epochs=10, lr=1e-4, start_epoch=1):
        """训练阶段C: 物理空间分离 - 修复版"""
        print("\n" + "="*80)
        print("阶段C: 物理空间原型分离")
        print("="*80 + "\n")
        
        # 关键修复：如果是从头开始 Stage C，先加载 Stage B 的最佳权重
        if start_epoch == 1:
            self._load_best_checkpoint_for_stage('C')
        
        scaler_path = self.data_dir / 'scaler.pkl'
        trainer = PhysicalSeparationTrainer(
            self.model, self.device, str(scaler_path), use_amp=self.use_amp
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        if start_epoch > 1:
            ckpt = self.checkpoint_mgr.load_last(self.model, optimizer,
                                                  trainer.scaler if self.use_amp else None,
                                                  self.device)
        
        for epoch in range(start_epoch, n_epochs + 1):
            loss, loss_dict = trainer.train_epoch(train_loader, optimizer, epoch)
            print(f"Epoch {epoch}/{n_epochs} - "
                  f"Loss: {loss:.4f} | "
                  f"Rec: {loss_dict['rec']:.4f} | "
                  f"Sep_Phys: {loss_dict['sep_phys']:.4f}")
            
            loss_dict['total'] = loss
            self.logger.log('C', epoch, loss_dict)
            
            self.checkpoint_mgr.save(
                self.model, optimizer, 'C', epoch, loss,
                {'n_epochs': n_epochs, 'lr': lr},
                trainer.scaler if self.use_amp else None
            )
        
        # 保存最终模型
        torch.save(self.model.state_dict(), 
                  self.output_dir / 'checkpoint_stageC_final.pt')
        print("\n✓ 阶段C完成\n")
    
    def export_prototypes(self):
        """导出原型解释包"""
        exporter = PrototypeExporter(
            self.model,
            str(self.data_dir / 'scaler.pkl'),
            str(self.data_dir / 'metadata.json'),
            str(self.output_dir / 'prototype_export'),
            self.device
        )
        return exporter.export_all()
    
    def distill_and_export(self, 
                          train_loader,
                          hidden_dims=[256, 128],
                          temperature=1.5,
                          n_epochs=20,
                          lr=1e-3,
                          max_distill_batches: Optional[int] = 1000,
                          save_distill_dataset=False):
        """蒸馏学生模型并导出 - 修复KL计算"""
        print("\n" + "="*80)
        print("蒸馏轻量MLP学生模型")
        print("="*80 + "\n")
        
        # 生成蒸馏数据集 (可选，且只抽子集)
        if save_distill_dataset:
            print(f"生成蒸馏数据集 (最多 {max_distill_batches} 批次)...")
            distill_dir = self.output_dir / 'distillation_dataset'
            distill_dir.mkdir(exist_ok=True)
            
            X_list = []
            w_teacher_list = []
            
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(train_loader, desc="生成蒸馏数据")):
                    if max_distill_batches is not None and batch_idx >= max_distill_batches:
                        break
                    
                    X = batch['X'].to(self.device)
                    if self.use_amp:
                        with autocast():
                            outputs = self.model(X, stage='B')
                    else:
                        outputs = self.model(X, stage='B')
                    w_teacher = outputs['w']
                    
                    X_list.append(X.cpu().numpy())
                    w_teacher_list.append(w_teacher.cpu().numpy())
            
            X_distill = np.vstack(X_list)
            w_distill = np.vstack(w_teacher_list)
            
            np.savez_compressed(
                distill_dir / 'distill_data.npz',
                X=X_distill,
                w_teacher=w_distill
            )
            
            print(f"  ✓ 蒸馏数据集已保存: {X_distill.shape[0]:,} 样本")
        
        # 构建学生模型
        self.student_model = DistillationMLPStudent(
            input_dim=121,
            hidden_dims=hidden_dims,
            output_dim=self.n_prototypes,
            activation='tanh',
            temperature=temperature
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.student_model.parameters())
        print(f"学生模型参数量: {n_params:,}")
        
        # 蒸馏训练
        trainer = DistillationTrainer(
            self.model, self.student_model, self.device, use_amp=self.use_amp
        )
        
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=lr)
        
        for epoch in range(1, n_epochs + 1):
            loss, loss_components = trainer.train_epoch(
                train_loader, optimizer, epoch,
                max_batches=max_distill_batches
            )
            print(f"Epoch {epoch}/{n_epochs} - Loss: {loss:.4f} | "
                  f"KL: {loss_components['kl']:.4f} | Entropy: {loss_components['entropy']:.4f}")
            self.logger.log('Distill', epoch, {
                'total': loss, 
                'kl': loss_components['kl'],
                'entropy': loss_components['entropy']
            })
        
        # 保存学生模型
        torch.save(self.student_model.state_dict(),
                  self.output_dir / 'distill_student.pt')
        
        # 导出GEE格式
        exporter = DistillationExporter(
            self.student_model,
            str(self.output_dir / 'gee_deploy'),
            self.metadata
        )
        
        results = exporter.export_all()
        print("\n✓ 蒸馏完成\n")
        
        return results
    
    def _export_gee_consistency_package(self):
        """导出GEE一致性包"""
        print("\n" + "="*80)
        print("导出GEE一致性包")
        print("="*80 + "\n")
        
        gee_dir = self.output_dir / 'gee_deploy'
        gee_dir.mkdir(exist_ok=True)
        
        # 1. scaler.json
        scaler_pkl = self.data_dir / 'scaler.pkl'
        if scaler_pkl.exists():
            scaler_dict = joblib.load(scaler_pkl)
            scaler_json = {
                'mean': scaler_dict['mean'].tolist(),
                'std': scaler_dict['std'].tolist(),
                'log1p_dims': scaler_dict.get('log1p_dims', [0, 1, 4, 7, 10]),
                'fitted': scaler_dict.get('fitted', True)
            }
            with open(gee_dir / 'scaler.json', 'w') as f:
                json.dump(scaler_json, f, indent=2)
            print(f"  ✓ scaler.json")
        
        # 2. feature_spec.json
        feature_spec = {
            'token_order': self.metadata['token_order'],
            'feature_names': self.metadata['feature_names'],
            'n_tokens': 11,
            'n_features_per_token': 11,
            'total_features': 121,
            'log1p_dims': [0, 1, 4, 7, 10]
        }
        with open(gee_dir / 'feature_spec.json', 'w') as f:
            json.dump(feature_spec, f, indent=2)
        print(f"  ✓ feature_spec.json")
        
        # 3. 复制 D_phys
        proto_export_dir = self.output_dir / 'prototype_export'
        for fname in ['D_phys.npy', 'D_phys.csv']:
            src = proto_export_dir / fname
            if src.exists():
                shutil.copy(src, gee_dir / fname)
                print(f"  ✓ {fname}")
        
        print("\n✓ GEE一致性包导出完成\n")
    
    def run_full_pipeline(self,
                         batch_size=512,
                         n_batches_per_epoch_A=3000,
                         n_batches_per_epoch_B=3000,
                         n_batches_per_epoch_C=1000,
                         stage_A_epochs=5,
                         stage_B_epochs=15,
                         stage_C_epochs=8,
                         distill_epochs=15,
                         max_distill_batches=1000,
                         auto_resume=True):
        """
        运行完整训练管道 - 修复版
        
        关键修复:
        1. Stage B/C 开始时正确加载前一阶段的最佳权重
        2. 避免优化器状态与模型状态不匹配
        """
        print("\n" + "="*80)
        print("DeepAA 完整训练管道 (Colab优化版)")
        print("="*80)
        
        # 检查是否需要恢复
        resume_info = None
        if auto_resume:
            resume_info = self.checkpoint_mgr.get_resume_info()
            if resume_info:
                print(f"\n⚡ 检测到断点:")
                print(f"   Stage: {resume_info['stage']}")
                print(f"   Epoch: {resume_info['epoch']}")
                print(f"   时间: {resume_info['timestamp']}")
        
        # 构建模型
        self.build_model()
        
        # 确定起始阶段和epoch
        start_stage = 'A'
        start_epoch_A = 1
        start_epoch_B = 1
        start_epoch_C = 1
        
        if resume_info:
            if resume_info['stage'] == 'A':
                start_stage = 'A'
                start_epoch_A = resume_info['epoch'] + 1
                if start_epoch_A > stage_A_epochs:
                    start_stage = 'B'
                    start_epoch_A = stage_A_epochs + 1
            elif resume_info['stage'] == 'B':
                start_stage = 'B'
                start_epoch_B = resume_info['epoch'] + 1
                if start_epoch_B > stage_B_epochs:
                    start_stage = 'C'
                    start_epoch_B = stage_B_epochs + 1
            elif resume_info['stage'] == 'C':
                start_stage = 'C'
                start_epoch_C = resume_info['epoch'] + 1
                if start_epoch_C > stage_C_epochs:
                    start_stage = 'DONE'
            
            # 关键修复：只有在恢复同一阶段时才加载 last.pt
            # 跨阶段时由各 train_stage_X 函数负责加载正确的前置权重
            if start_stage == resume_info['stage']:
                self.checkpoint_mgr.load_last(self.model, device=self.device)
        
        # 阶段A
        if start_stage == 'A':
            train_loader_A = self.load_data(
                batch_size=batch_size,
                use_stratified=True,
                num_workers=2,
                n_batches_per_epoch=n_batches_per_epoch_A
            )
            self.train_stage_A(train_loader_A, n_epochs=stage_A_epochs, 
                              start_epoch=start_epoch_A)
            start_stage = 'B'
            start_epoch_B = 1
            del train_loader_A
            gc.collect()
        
        # 阶段B
        if start_stage == 'B':
            train_loader_B = self.load_data(
                batch_size=batch_size,
                use_stratified=True,
                num_workers=2,
                n_batches_per_epoch=n_batches_per_epoch_B
            )
            self.train_stage_B(train_loader_B, n_epochs=stage_B_epochs,
                              start_epoch=start_epoch_B)
            start_stage = 'C'
            start_epoch_C = 1
            del train_loader_B
            gc.collect()
        
        # 阶段C
        if start_stage == 'C':
            train_loader_C = self.load_data(
                batch_size=batch_size,
                use_stratified=True,
                num_workers=2,
                n_batches_per_epoch=n_batches_per_epoch_C
            )
            self.train_stage_C(train_loader_C, n_epochs=stage_C_epochs,
                              start_epoch=start_epoch_C)
            del train_loader_C
            gc.collect()
        
        # 导出原型
        print("\n开始导出...")
        proto_results = self.export_prototypes()
        
        # 蒸馏并导出
        train_loader_distill = self.load_data(
            batch_size=batch_size,
            use_stratified=True,
            num_workers=2,
            n_batches_per_epoch=max_distill_batches
        )
        
        distill_results = self.distill_and_export(
            train_loader_distill,
            n_epochs=distill_epochs,
            max_distill_batches=max_distill_batches,
            save_distill_dataset=False
        )
        
        # 导出GEE一致性包
        self._export_gee_consistency_package()
        
        # 绘制训练曲线
        self.logger.plot_training_curves()
        
        print("\n" + "="*80)
        print("✓✓✓ 完整训练管道完成! ✓✓✓")
        print("="*80)
        print("\n导出文件位置:")
        print(f"  原型解释包: {self.output_dir / 'prototype_export'}")
        print(f"  GEE部署包: {self.output_dir / 'gee_deploy'}")
        print(f"  训练日志: {self.output_dir / 'train_log.csv'}")
        print(f"  训练曲线: {self.output_dir / 'training_curves.png'}")
        print("="*80 + "\n")
        
        return {
            'prototype_export': proto_results,
            'distill_export': distill_results
        }
    

# @title
# ============================================================================
# 8. 主程序入口 (Colab专用)
# ============================================================================

def main_colab():
    """Colab主函数"""
    
    # ==================== 1. 环境设置 ====================
    IN_COLAB = setup_colab_env()
    
    # ==================== 2. 路径配置 ====================
    
    # Drive上的数据源路径 (请根据你的实际路径修改)
    DRIVE_DATA_DIR = "/content/drive/MyDrive/DeepAA_TokenData"
    
    # 本地数据路径 (训练时从这里读取，速度快)
    LOCAL_DATA_DIR = "/content/DeepAA_TokenData"
    
    # 输出路径 (直接写到Drive，防止断连丢失)
    OUTPUT_DIR = "/content/drive/MyDrive/DeepAA/DeepAA_Training_Run1"
    
    # ==================== 3. 数据准备 ====================
    
    # 将Drive数据复制到本地 (关键：加速训练)
    if IN_COLAB:
        DATA_DIR = copy_data_to_local(DRIVE_DATA_DIR, LOCAL_DATA_DIR)
    else:
        # 本地测试时直接用源路径
        DATA_DIR = DRIVE_DATA_DIR
    
    # ==================== 4. 训练参数配置 ====================
    
    # 🚀 批次大小 (根据GPU显存调整, T4可用1024-2048)
    BATCH_SIZE = 4096
    
    # 🚀 每epoch步数 (关键：控制训练时间)
    # 773万样本全跑一遍约15000步，这里只跑子集
    N_BATCHES_PER_EPOCH_A = 800    # 阶段A: ~300万样本/epoch
    N_BATCHES_PER_EPOCH_B = 800    # 阶段B: ~300万样本/epoch
    N_BATCHES_PER_EPOCH_C = 300    # 阶段C: ~100万样本/epoch
    
    # 🚀 训练轮数
    STAGE_A_EPOCHS = 5    # 预训练
    STAGE_B_EPOCHS = 10   # 原型混合
    STAGE_C_EPOCHS = 5    # 物理分离
    DISTILL_EPOCHS = 10   # 蒸馏
    
    # 🚀 蒸馏子集 (不需要全量数据)
    MAX_DISTILL_BATCHES = 100  # ~50万样本
    
    # 🚀 模型结构 (平衡速度和精度)
    D_MODEL = 256           # 128快/256平衡/512慢
    N_HEADS = 8             # 注意力头数
    N_ENCODER_LAYERS = 4    # Encoder层数
    ENTMAX_N_ITER = 15      # Entmax迭代次数 (15够用，50太慢)
    
    # 原型数量
    N_PROTOTYPES = 60
    
    # 数据加载
    NUM_WORKERS = 4         # Colab建议2-4
    
    # AMP混合精度
    USE_AMP = True
    
    # 随机种子
    SEED = 42
    
    # ==================== 5. 运行训练 ====================
    
    print("\n" + "="*80)
    print("⚡ Colab训练配置")
    print("="*80)
    print(f"批次大小: {BATCH_SIZE}")
    print(f"每epoch步数: A={N_BATCHES_PER_EPOCH_A}, B={N_BATCHES_PER_EPOCH_B}, C={N_BATCHES_PER_EPOCH_C}")
    print(f"训练轮数: A={STAGE_A_EPOCHS}, B={STAGE_B_EPOCHS}, C={STAGE_C_EPOCHS}, Distill={DISTILL_EPOCHS}")
    print(f"模型: d={D_MODEL}, heads={N_HEADS}, layers={N_ENCODER_LAYERS}")
    print(f"原型数: {N_PROTOTYPES}")
    print(f"AMP: {USE_AMP}")
    print("="*80 + "\n")
    
    # 创建管道
    pipeline = DeepAATrainingPipelineColab(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        n_prototypes=N_PROTOTYPES,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        entmax_n_iter=ENTMAX_N_ITER,
        use_amp=USE_AMP,
        seed=SEED
    )
    
    # 运行完整流程 (自动断点续训)
    results = pipeline.run_full_pipeline(
        batch_size=BATCH_SIZE,
        n_batches_per_epoch_A=N_BATCHES_PER_EPOCH_A,
        n_batches_per_epoch_B=N_BATCHES_PER_EPOCH_B,
        n_batches_per_epoch_C=N_BATCHES_PER_EPOCH_C,
        stage_A_epochs=STAGE_A_EPOCHS,
        stage_B_epochs=STAGE_B_EPOCHS,
        stage_C_epochs=STAGE_C_EPOCHS,
        distill_epochs=DISTILL_EPOCHS,
        max_distill_batches=MAX_DISTILL_BATCHES,
        auto_resume=True  # 自动检测并恢复断点
    )
    return results


# 如果直接运行此脚本（在Colab中）
if __name__ == '__main__':
    results = main_colab()