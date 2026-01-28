"""
DeepAA 数据准备与 Token 构建管道
Data & Token Pipeline for DeepAA Model

功能:
1. 读取与字段分组 (11个token: b01-b07 + PV/NPV/BS/DA)
2. 生成token内11维特征 [a0, A1, u1, v1, A2, u2, v2, A3, u3, v3, RMSE]
3. 清洗规则与缺失处理
4. 离线标准化 (分层增量拟合)
5. 数据集切分与打包 (按AEZ分层)
"""

import json
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil
import joblib
from sklearn.model_selection import train_test_split


# ============================================================================
# 辅助工具函数
# ============================================================================

def get_memory_usage():
    """获取当前进程内存使用量(MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def safe_log1p(x):
    """安全的log1p变换,处理负值"""
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_log1p(y):
    """log1p的逆变换"""
    return np.sign(y) * (np.exp(np.abs(y)) - 1)


# ============================================================================
# 1. Token映射与字段分组
# ============================================================================

class TokenMapper:
    """Token字段映射管理器"""
    
    def __init__(self):
        # 定义11个token的名称和顺序
        self.spectral_tokens = ['b01', 'b02', 'b03', 'b04', 'b05', 'b06', 'b07']
        self.structural_tokens = ['PV', 'NPV', 'BS', 'DA']
        self.all_tokens = self.spectral_tokens + self.structural_tokens
        
        # 每个token需要的字段后缀
        self.required_suffixes = ['a0', 'cos1t', 'sin1t', 'cos2t', 'sin2t', 
                                  'cos3t', 'sin3t', 'RMSE']
        
        # 11维特征的名称
        self.feature_names = ['a0', 'A1', 'u1', 'v1', 'A2', 'u2', 'v2', 
                             'A3', 'u3', 'v3', 'RMSE']
        
    def build_token_map(self, available_columns: List[str]) -> Dict[str, Dict[str, str]]:
        """
        构建token到列名的映射
        
        参数:
            available_columns: CSV中可用的列名
        
        返回:
            token_map: {token_name: {field_type: column_name}}
        """
        token_map = {}
        available_cols_set = set(available_columns)
        
        for token in self.all_tokens:
            token_fields = {}
            
            # 光谱token使用 sur_refl_bXX_ 前缀
            if token in self.spectral_tokens:
                prefix = f"sur_refl_{token}_"
                
                for suffix in self.required_suffixes:
                    col_name = prefix + suffix
                    if col_name in available_cols_set:
                        token_fields[suffix] = col_name
                    else:
                        raise ValueError(f"缺失必需列: {col_name}")
            
            # 结构token直接使用名称
            else:
                for suffix in self.required_suffixes:
                    col_name = f"{token}_{suffix}"
                    if col_name in available_cols_set:
                        token_fields[suffix] = col_name
                    else:
                        raise ValueError(f"缺失必需列: {col_name}")
            
            token_map[token] = token_fields
        
        return token_map
    
    def get_all_required_columns(self, token_map: Dict) -> List[str]:
        """获取所有需要读取的列"""
        all_cols = []
        for token in self.all_tokens:
            all_cols.extend(token_map[token].values())
        return all_cols


# ============================================================================
# 2. Token特征工程 (11维特征构建)
# ============================================================================

class TokenFeatureBuilder:
    """Token内11维特征构建器"""
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        
    def compute_token_features(self, 
                              a0: np.ndarray,
                              c1: np.ndarray, s1: np.ndarray,
                              c2: np.ndarray, s2: np.ndarray,
                              c3: np.ndarray, s3: np.ndarray,
                              rmse: np.ndarray) -> np.ndarray:
        """
        计算单个token的11维特征
        
        特征顺序: [a0, A1, u1, v1, A2, u2, v2, A3, u3, v3, RMSE]
        
        其中:
            A_k = sqrt(c_k^2 + s_k^2)  振幅
            u_k = c_k / (A_k + eps)     归一化余弦
            v_k = s_k / (A_k + eps)     归一化正弦
        
        参数:
            a0: 基础项 (N,)
            c1,s1,c2,s2,c3,s3: 谐波系数 (N,)
            rmse: 拟合误差 (N,)
        
        返回:
            features: (N, 11) 特征矩阵
        """
        N = len(a0)
        features = np.zeros((N, 11), dtype=np.float32)
        
        # 第0维: a0
        features[:, 0] = a0
        
        # 第1-3维: A1, u1, v1
        A1 = np.sqrt(c1**2 + s1**2)
        features[:, 1] = A1
        features[:, 2] = c1 / (A1 + self.epsilon)  # u1
        features[:, 3] = s1 / (A1 + self.epsilon)  # v1
        
        # 第4-6维: A2, u2, v2
        A2 = np.sqrt(c2**2 + s2**2)
        features[:, 4] = A2
        features[:, 5] = c2 / (A2 + self.epsilon)  # u2
        features[:, 6] = s2 / (A2 + self.epsilon)  # v2
        
        # 第7-9维: A3, u3, v3
        A3 = np.sqrt(c3**2 + s3**2)
        features[:, 7] = A3
        features[:, 8] = c3 / (A3 + self.epsilon)  # u3
        features[:, 9] = s3 / (A3 + self.epsilon)  # v3
        
        # 第10维: RMSE
        features[:, 10] = rmse
        
        return features
    
    def build_all_tokens(self, 
                        data_chunk: pd.DataFrame, 
                        token_map: Dict[str, Dict[str, str]],
                        token_order: List[str]) -> np.ndarray:
        """
        构建所有token的特征矩阵
        
        参数:
            data_chunk: 数据块
            token_map: token字段映射
            token_order: token顺序
        
        返回:
            X_tokens: (N, 11, 11) - N样本, 11个token, 每个11维特征
        """
        N = len(data_chunk)
        X_tokens = np.zeros((N, 11, 11), dtype=np.float32)
        
        for token_idx, token_name in enumerate(token_order):
            fields = token_map[token_name]
            
            # 读取原始字段
            a0 = data_chunk[fields['a0']].values
            c1 = data_chunk[fields['cos1t']].values
            s1 = data_chunk[fields['sin1t']].values
            c2 = data_chunk[fields['cos2t']].values
            s2 = data_chunk[fields['sin2t']].values
            c3 = data_chunk[fields['cos3t']].values
            s3 = data_chunk[fields['sin3t']].values
            rmse = data_chunk[fields['RMSE']].values
            
            # 计算11维特征
            token_features = self.compute_token_features(
                a0, c1, s1, c2, s2, c3, s3, rmse
            )
            
            X_tokens[:, token_idx, :] = token_features
        
        return X_tokens


# ============================================================================
# 3. 数据清洗与质量控制
# ============================================================================

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, 
                 value_range=(-1e6, 1e6),
                 min_amplitude=1e-10):
        self.value_range = value_range
        self.min_amplitude = min_amplitude
        
    def clean_tokens(self, X_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        清洗token特征
        
        处理:
        1. 无效值 (nan/inf)
        2. 极端异常值
        3. 振幅过小时的u,v处理
        4. RMSE缺失填充
        
        参数:
            X_tokens: (N, 11, 11) 原始特征
        
        返回:
            X_clean: (N, 11, 11) 清洗后特征
            valid_mask: (N,) 有效样本mask
        """
        N, T, F = X_tokens.shape
        X_clean = X_tokens.copy()
        
        # 1. 标记无效值
        has_nan = np.isnan(X_clean).any(axis=(1, 2))
        has_inf = np.isinf(X_clean).any(axis=(1, 2))
        
        # 2. 标记极端异常值
        has_extreme = ((X_clean < self.value_range[0]) | 
                      (X_clean > self.value_range[1])).any(axis=(1, 2))
        
        # 3. 处理振幅过小的情况
        # 维度索引: [a0, A1, u1, v1, A2, u2, v2, A3, u3, v3, RMSE]
        #           [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,   10]
        for i in range(N):
            # 检查A1, A2, A3 (索引1, 4, 7)
            for amp_idx, u_idx, v_idx in [(1, 2, 3), (4, 5, 6), (7, 8, 9)]:
                A = X_clean[i, :, amp_idx]  # 所有token的该振幅
                
                # 如果振幅过小,将对应的u,v设为0
                small_amp_mask = A < self.min_amplitude
                X_clean[i, small_amp_mask, u_idx] = 0
                X_clean[i, small_amp_mask, v_idx] = 0
        
        # 4. RMSE缺失填充 (使用中位数)
        rmse_values = X_clean[:, :, 10].flatten()
        valid_rmse = rmse_values[np.isfinite(rmse_values)]
        if len(valid_rmse) > 0:
            rmse_median = np.median(valid_rmse)
            rmse_nan_mask = np.isnan(X_clean[:, :, 10])
            X_clean[:, :, 10][rmse_nan_mask] = rmse_median
        
        # 最终有效样本mask
        valid_mask = ~(has_nan | has_inf | has_extreme)
        
        return X_clean, valid_mask


# ============================================================================
# 4. 分层增量标准化
# ============================================================================

class StratifiedScaler:
    """分层增量标准化器 (按AEZ分层抽样拟合)"""
    
    def __init__(self, 
                 log1p_dims: Optional[List[int]] = None,
                 samples_per_stratum: int = 10000):
        """
        参数:
            log1p_dims: 需要log1p变换的维度索引 (建议: [0,1,4,7,10] 即a0,A1,A2,A3,RMSE)
            samples_per_stratum: 每层抽样数量
        """
        self.log1p_dims = log1p_dims or [0, 1, 4, 7, 10]
        self.samples_per_stratum = samples_per_stratum
        
        # 统计量 (shape: (11 tokens, 11 features))
        self.mean = None
        self.std = None
        self.fitted = False
        
    def _apply_log1p(self, X: np.ndarray) -> np.ndarray:
        """对指定维度应用log1p"""
        X_transformed = X.copy()
        for dim in self.log1p_dims:
            X_transformed[:, :, dim] = safe_log1p(X_transformed[:, :, dim])
        return X_transformed
    
    def _inverse_log1p(self, X: np.ndarray) -> np.ndarray:
        """对指定维度应用log1p逆变换"""
        X_original = X.copy()
        for dim in self.log1p_dims:
            X_original[:, :, dim] = inverse_log1p(X_original[:, :, dim])
        return X_original
    
    def fit(self, 
            X_tokens: np.ndarray, 
            aez_labels: Optional[np.ndarray] = None,
            stratified: bool = True):
        """
        拟合标准化参数
        
        参数:
            X_tokens: (N, 11, 11) 特征数据
            aez_labels: (N,) AEZ标签 (如果stratified=True)
            stratified: 是否分层抽样
        """
        print("\n" + "="*80)
        print("拟合标准化器")
        print("="*80)
        
        # 应用log1p
        X_log = self._apply_log1p(X_tokens)
        
        # 分层抽样
        if stratified and aez_labels is not None:
            print(f"分层抽样模式: 每层抽取 {self.samples_per_stratum} 样本")
            unique_aez = np.unique(aez_labels)
            sampled_indices = []
            
            for aez in unique_aez:
                aez_indices = np.where(aez_labels == aez)[0]
                n_samples = min(self.samples_per_stratum, len(aez_indices))
                sampled = np.random.choice(aez_indices, size=n_samples, replace=False)
                sampled_indices.extend(sampled)
            
            X_sample = X_log[sampled_indices]
            print(f"  原始样本数: {len(X_tokens):,}")
            print(f"  AEZ层数: {len(unique_aez)}")
            print(f"  抽样后样本数: {len(X_sample):,}")
        else:
            print("全局抽样模式")
            n_samples = min(self.samples_per_stratum * 10, len(X_log))
            sampled_indices = np.random.choice(len(X_log), size=n_samples, replace=False)
            X_sample = X_log[sampled_indices]
            print(f"  抽样样本数: {len(X_sample):,}")
        
        # 计算均值和标准差 (沿样本维度)
        # X_sample: (M, 11, 11)
        self.mean = np.mean(X_sample, axis=0)  # (11, 11)
        self.std = np.std(X_sample, axis=0)    # (11, 11)
        
        # 防止除零
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        
        # u,v维度 (2,3,5,6,8,9) 已经在[-1,1],可选择不标准化
        # 这里仍标准化,但会给出警告
        uv_dims = [2, 3, 5, 6, 8, 9]
        print(f"\n  ℹ️  u,v维度 (索引{uv_dims}) 已归一化,标准化可选")
        
        self.fitted = True
        
        print(f"\n  ✓ 标准化器拟合完成")
        print(f"  log1p变换维度: {self.log1p_dims}")
        print("="*80 + "\n")
        
    def transform(self, X_tokens: np.ndarray) -> np.ndarray:
        """标准化转换"""
        if not self.fitted:
            raise ValueError("Scaler未拟合,请先调用fit()")
        
        X_log = self._apply_log1p(X_tokens)
        X_scaled = (X_log - self.mean) / self.std
        return X_scaled
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """逆标准化"""
        if not self.fitted:
            raise ValueError("Scaler未拟合")
        
        X_log = X_scaled * self.std + self.mean
        X_original = self._inverse_log1p(X_log)
        return X_original
    
    def save(self, path: str):
        """保存scaler"""
        joblib.dump({
            'mean': self.mean,
            'std': self.std,
            'log1p_dims': self.log1p_dims,
            'samples_per_stratum': self.samples_per_stratum,
            'fitted': self.fitted
        }, path)
        
    def load(self, path: str):
        """加载scaler"""
        data = joblib.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.log1p_dims = data['log1p_dims']
        self.samples_per_stratum = data['samples_per_stratum']
        self.fitted = data['fitted']


# ============================================================================
# 5. 数据集管理与切分
# ============================================================================

class DeepAADataset:
    """DeepAA数据集管理器"""
    
    def __init__(self, 
                 X_tokens: np.ndarray,
                 y_target: Optional[np.ndarray] = None,
                 aez_labels: Optional[np.ndarray] = None,
                 sample_ids: Optional[np.ndarray] = None):
        """
        参数:
            X_tokens: (N, 11, 11) 标准化后的token特征
            y_target: (N,) 目标变量 (如果有)
            aez_labels: (N,) AEZ标签
            sample_ids: (N,) 样本ID
        """
        self.X = X_tokens
        self.y = y_target
        self.aez = aez_labels
        self.ids = sample_ids
        self.N = len(X_tokens)
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        item = {'X': self.X[idx]}
        if self.y is not None:
            item['y'] = self.y[idx]
        if self.aez is not None:
            item['aez'] = self.aez[idx]
        if self.ids is not None:
            item['id'] = self.ids[idx]
        return item
    
    def stratified_split(self, 
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        random_state: int = 42) -> Tuple['DeepAADataset', 
                                                          'DeepAADataset', 
                                                          'DeepAADataset']:
        """
        按AEZ分层切分数据集
        
        返回:
            train_dataset, val_dataset, test_dataset
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        print("\n" + "="*80)
        print("数据集分层切分")
        print("="*80)
        
        if self.aez is None:
            print("  警告: 无AEZ标签,使用随机切分")
            indices = np.arange(self.N)
            np.random.seed(random_state)
            np.random.shuffle(indices)
            
            n_train = int(self.N * train_ratio)
            n_val = int(self.N * val_ratio)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]
        else:
            print(f"  按AEZ分层切分 (train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%})")
            # 先切分train和temp
            train_idx, temp_idx = train_test_split(
                np.arange(self.N),
                test_size=(1-train_ratio),
                stratify=self.aez,
                random_state=random_state
            )
            
            # 再切分val和test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=(1-val_size),
                stratify=self.aez[temp_idx],
                random_state=random_state
            )
        
        print(f"\n  训练集: {len(train_idx):,} 样本")
        print(f"  验证集: {len(val_idx):,} 样本")
        print(f"  测试集: {len(test_idx):,} 样本")
        print("="*80 + "\n")
        
        # 创建子数据集
        train_ds = DeepAADataset(
            self.X[train_idx],
            self.y[train_idx] if self.y is not None else None,
            self.aez[train_idx] if self.aez is not None else None,
            self.ids[train_idx] if self.ids is not None else None
        )
        
        val_ds = DeepAADataset(
            self.X[val_idx],
            self.y[val_idx] if self.y is not None else None,
            self.aez[val_idx] if self.aez is not None else None,
            self.ids[val_idx] if self.ids is not None else None
        )
        
        test_ds = DeepAADataset(
            self.X[test_idx],
            self.y[test_idx] if self.y is not None else None,
            self.aez[test_idx] if self.aez is not None else None,
            self.ids[test_idx] if self.ids is not None else None
        )
        
        return train_ds, val_ds, test_ds


# ============================================================================
# 6. 主数据管道
# ============================================================================

class DeepAADataPipeline:
    """DeepAA完整数据管道"""
    
    def __init__(self, 
                 csv_path: str,
                 output_dir: str,
                 target_column: Optional[str] = None,
                 aez_column: str = 'AEZ_ID',
                 id_column: str = 'system:index',
                 chunksize: int = 1_000_000):
        """
        参数:
            csv_path: CSV文件路径或目录
            output_dir: 输出目录
            target_column: 目标变量列名 (如果有监督学习)
            aez_column: AEZ列名
            id_column: 样本ID列名
            chunksize: 分块大小
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_column = target_column
        self.aez_column = aez_column
        self.id_column = id_column
        self.chunksize = chunksize
        
        # 组件
        self.token_mapper = TokenMapper()
        self.feature_builder = TokenFeatureBuilder()
        self.cleaner = DataCleaner()
        self.scaler = StratifiedScaler()
        
        # 映射
        self.token_map = None
        self.meta = None
        
    def process(self, 
                fit_scaler: bool = True,
                save_datasets: bool = True,
                stratified_scaling: bool = True):
        """
        执行完整管道
        
        参数:
            fit_scaler: 是否拟合scaler (False时需要已有scaler)
            save_datasets: 是否保存数据集
            stratified_scaling: 是否分层拟合scaler
        """
        print("\n" + "="*80)
        print("DeepAA 数据管道启动")
        print("="*80)
        print(f"输入: {self.csv_path}")
        print(f"输出: {self.output_dir}")
        print(f"初始内存: {get_memory_usage():.1f} MB")
        print("="*80 + "\n")
        
        # 阶段1: 读取与字段分组
        print("【阶段1】读取与字段分组")
        self._build_token_mapping()
        
        # 阶段2: 构建token特征
        print("\n【阶段2】构建token特征 (11维×11token)")
        X_tokens_raw, y_targets, aez_labels, sample_ids = self._build_token_features()
        
        # 阶段3: 数据清洗
        print("\n【阶段3】数据清洗")
        X_tokens_clean, valid_mask = self.cleaner.clean_tokens(X_tokens_raw)
        
        print(f"  有效样本: {valid_mask.sum():,} / {len(valid_mask):,} ({valid_mask.mean():.2%})")
        
        # 过滤无效样本
        X_tokens_clean = X_tokens_clean[valid_mask]
        if y_targets is not None:
            y_targets = y_targets[valid_mask]
        if aez_labels is not None:
            aez_labels = aez_labels[valid_mask]
        if sample_ids is not None:
            sample_ids = sample_ids[valid_mask]
        
        # 阶段4: 标准化
        print("\n【阶段4】标准化")
        if fit_scaler:
            self.scaler.fit(X_tokens_clean, aez_labels, stratified=stratified_scaling)
            self.scaler.save(self.output_dir / 'scaler.pkl')
            print(f"  ✓ Scaler已保存")
        else:
            self.scaler.load(self.output_dir / 'scaler.pkl')
            print(f"  ✓ Scaler已加载")
        
        X_tokens_scaled = self.scaler.transform(X_tokens_clean)
        
        # 阶段5: 数据集切分
        print("\n【阶段5】数据集切分")
        dataset = DeepAADataset(X_tokens_scaled, y_targets, aez_labels, sample_ids)
        train_ds, val_ds, test_ds = dataset.stratified_split()
        
        # 阶段6: 保存
        if save_datasets:
            print("\n【阶段6】保存数据集")
            self._save_datasets(train_ds, val_ds, test_ds)
        
        # 保存元信息
        self._save_metadata()
        
        print("\n" + "="*80)
        print("✓ 数据管道完成!")
        print(f"最终内存: {get_memory_usage():.1f} MB")
        print("="*80 + "\n")
        
        return train_ds, val_ds, test_ds
    
    def _build_token_mapping(self):
        """构建token映射"""
        import glob
        
        # 获取文件列表
        if self.csv_path.is_file():
            csv_files = [self.csv_path]
        else:
            csv_files = sorted(glob.glob(str(self.csv_path / "*.csv")))
        
        if not csv_files:
            raise ValueError(f"未找到CSV文件: {self.csv_path}")
        
        # 读取第一个文件的列名
        first_df = pd.read_csv(csv_files[0], nrows=1)
        available_columns = list(first_df.columns)
        
        print(f"  检测到列数: {len(available_columns)}")
        
        # 构建映射
        self.token_map = self.token_mapper.build_token_map(available_columns)
        
        print(f"  ✓ 映射完成: 11个token")
        print(f"  Token顺序: {', '.join(self.token_mapper.all_tokens)}")
        
    def _build_token_features(self):
        """构建token特征"""
        import glob
        
        if self.csv_path.is_file():
            csv_files = [self.csv_path]
        else:
            csv_files = sorted(glob.glob(str(self.csv_path / "*.csv")))
        
        # 需要读取的列
        required_cols = self.token_mapper.get_all_required_columns(self.token_map)
        extra_cols = []
        if self.target_column:
            extra_cols.append(self.target_column)
        if self.aez_column:
            extra_cols.append(self.aez_column)
        if self.id_column:
            extra_cols.append(self.id_column)
        
        all_cols = required_cols + extra_cols
        
        # 计算总行数
        total_lines = sum(sum(1 for _ in open(f, encoding='utf-8')) - 1 
                         for f in csv_files)
        
        print(f"  处理 {len(csv_files)} 个文件, {total_lines:,} 行")
        
        # 分块处理
        all_X_tokens = []
        all_y = []
        all_aez = []
        all_ids = []
        
        pbar = tqdm(total=total_lines, desc="构建Token", unit="rows")
        
        for csv_file in csv_files:
            for chunk in pd.read_csv(csv_file, usecols=all_cols, chunksize=self.chunksize):
                # 构建token特征
                X_chunk = self.feature_builder.build_all_tokens(
                    chunk, self.token_map, self.token_mapper.all_tokens
                )
                all_X_tokens.append(X_chunk)
                
                # 提取其他字段
                if self.target_column and self.target_column in chunk.columns:
                    all_y.append(chunk[self.target_column].values)
                if self.aez_column and self.aez_column in chunk.columns:
                    all_aez.append(chunk[self.aez_column].values)
                if self.id_column and self.id_column in chunk.columns:
                    all_ids.append(chunk[self.id_column].values)
                
                pbar.update(len(chunk))
                pbar.set_postfix({"mem": f"{get_memory_usage():.0f}MB"})
                
                del chunk
                gc.collect()
        
        pbar.close()
        
        # 合并
        X_tokens = np.vstack(all_X_tokens)
        y_targets = np.concatenate(all_y) if all_y else None
        aez_labels = np.concatenate(all_aez) if all_aez else None
        sample_ids = np.concatenate(all_ids) if all_ids else None
        
        print(f"  ✓ Token特征构建完成: {X_tokens.shape}")
        
        return X_tokens, y_targets, aez_labels, sample_ids
    
    def _save_datasets(self, train_ds, val_ds, test_ds):
        """保存数据集"""
        np.savez_compressed(
            self.output_dir / 'train_data.npz',
            X=train_ds.X,
            y=train_ds.y if train_ds.y is not None else np.array([]),
            aez=train_ds.aez if train_ds.aez is not None else np.array([]),
            ids=train_ds.ids if train_ds.ids is not None else np.array([])
        )
        
        np.savez_compressed(
            self.output_dir / 'val_data.npz',
            X=val_ds.X,
            y=val_ds.y if val_ds.y is not None else np.array([]),
            aez=val_ds.aez if val_ds.aez is not None else np.array([]),
            ids=val_ds.ids if val_ds.ids is not None else np.array([])
        )
        
        np.savez_compressed(
            self.output_dir / 'test_data.npz',
            X=test_ds.X,
            y=test_ds.y if test_ds.y is not None else np.array([]),
            aez=test_ds.aez if test_ds.aez is not None else np.array([]),
            ids=test_ds.ids if test_ds.ids is not None else np.array([])
        )
        
        print(f"  ✓ 数据集已保存 (.npz格式)")
        print(f"    - train_data.npz: {len(train_ds):,} 样本")
        print(f"    - val_data.npz: {len(val_ds):,} 样本")
        print(f"    - test_data.npz: {len(test_ds):,} 样本")
    
    def _save_metadata(self):
        """保存元信息"""
        meta = {
            "version": "1.0",
            "created_at": "2026-01-28",
            "token_order": self.token_mapper.all_tokens,
            "feature_names": self.token_mapper.feature_names,
            "n_tokens": len(self.token_mapper.all_tokens),
            "n_features_per_token": len(self.token_mapper.feature_names),
            "total_dims": (len(self.token_mapper.all_tokens), 
                          len(self.token_mapper.feature_names)),
            "log1p_dims": self.scaler.log1p_dims,
            "token_map": {token: {k: str(v) for k, v in fields.items()} 
                         for token, fields in self.token_map.items()},
            "target_column": self.target_column,
            "aez_column": self.aez_column,
            "id_column": self.id_column,
        }
        
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 元信息已保存: metadata.json")


# ============================================================================
# 7. 数据加载器 (供训练使用)
# ============================================================================

def load_deepaa_dataset(data_dir: str, split: str = 'train') -> DeepAADataset:
    """
    加载已处理的DeepAA数据集
    
    参数:
        data_dir: 数据目录
        split: 'train', 'val', 或 'test'
    
    返回:
        DeepAADataset实例
    """
    data_dir = Path(data_dir)
    data_file = data_dir / f'{split}_data.npz'
    
    if not data_file.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_file}")
    
    data = np.load(data_file)
    
    X = data['X']
    y = data['y'] if len(data['y']) > 0 else None
    aez = data['aez'] if len(data['aez']) > 0 else None
    ids = data['ids'] if len(data['ids']) > 0 else None
    
    return DeepAADataset(X, y, aez, ids)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # ========== 配置参数 ==========
    
    # 数据路径 (与DeepAA_PCA.py相同)
    CSV_PATH = r"G:\0论文\06作物多样性\CROPGRIDS\cropdiversity\Cal_CROPDIVERSITY\Samples\Global Samples"
    OUTPUT_DIR = r"G:\0论文\06作物多样性\CROPGRIDS\cropdiversity\Cal_CROPDIVERSITY\Samples\DeepAA_TokenData"
    
    # 数据列配置
    TARGET_COLUMN = None  # 如果有目标变量,填写列名 (例如: 'crop_diversity')
    AEZ_COLUMN = 'AEZ_ID'  # AEZ分层列名
    ID_COLUMN = 'system:index'  # 样本ID列名
    
    # 处理参数
    CHUNKSIZE = 5_000_000  # 分块大小
    FIT_SCALER = True  # 是否拟合新的scaler (False时加载已有)
    STRATIFIED_SCALING = True  # 是否分层拟合scaler
    SAVE_DATASETS = True  # 是否保存处理后的数据集
    
    # 数据集划分比例
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 运行模式
    MODE = "process"  # "process": 完整处理 | "analyze": 仅分析元数据
    
    # ==============================
    
    if MODE == "process":
        # 创建管道
        pipeline = DeepAADataPipeline(
            csv_path=CSV_PATH,
            output_dir=OUTPUT_DIR,
            target_column=TARGET_COLUMN,
            aez_column=AEZ_COLUMN,
            id_column=ID_COLUMN,
            chunksize=CHUNKSIZE
        )
        
        # 执行处理
        import time
        start_time = time.time()
        
        try:
            train_ds, val_ds, test_ds = pipeline.process(
                fit_scaler=FIT_SCALER,
                save_datasets=SAVE_DATASETS,
                stratified_scaling=STRATIFIED_SCALING
            )
            
            elapsed = time.time() - start_time
            print(f"\n[OK] 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
            
            print("\n" + "="*80)
            print("【下一步】使用数据集")
            print("="*80)
            print("方式1 - 直接使用返回的Dataset对象:")
            print("  train_ds, val_ds, test_ds = pipeline.process(...)")
            print("  for batch in train_ds:")
            print("      X, y = batch['X'], batch['y']")
            print()
            print("方式2 - 从文件加载:")
            print("  from DeepAA_DataPipeline import load_deepaa_dataset")
            print(f"  train_ds = load_deepaa_dataset('{OUTPUT_DIR}', 'train')")
            print()
            print("数据格式:")
            print(f"  X.shape = (N, 11, 11)  # N样本 × 11token × 11特征")
            print("  Token顺序: b01-b07, PV, NPV, BS, DA")
            print("  特征顺序: [a0, A1, u1, v1, A2, u2, v2, A3, u3, v3, RMSE]")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n[ERROR] 错误: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    elif MODE == "analyze":
        # 分析元数据
        meta_file = Path(OUTPUT_DIR) / 'metadata.json'
        if not meta_file.exists():
            print(f"错误: 未找到元数据文件 {meta_file}")
            print("请先运行 MODE='process' 处理数据")
        else:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            print("\n" + "="*80)
            print("DeepAA Token数据元信息")
            print("="*80)
            print(f"版本: {meta['version']}")
            print(f"创建时间: {meta['created_at']}")
            print(f"\nToken数量: {meta['n_tokens']}")
            print(f"Token顺序: {', '.join(meta['token_order'])}")
            print(f"\n每Token特征数: {meta['n_features_per_token']}")
            print(f"特征顺序: {', '.join(meta['feature_names'])}")
            print(f"\n数据维度: {meta['total_dims']}")
            print(f"log1p变换维度索引: {meta['log1p_dims']}")
            print(f"\n目标变量: {meta['target_column'] or 'None'}")
            print(f"AEZ列: {meta['aez_column']}")
            print(f"ID列: {meta['id_column']}")
            print("="*80 + "\n")
            
            # 检查数据文件
            data_dir = Path(OUTPUT_DIR)
            for split in ['train', 'val', 'test']:
                data_file = data_dir / f'{split}_data.npz'
                if data_file.exists():
                    data = np.load(data_file)
                    print(f"{split:5s} 数据: {len(data['X']):,} 样本, "
                          f"大小: {data_file.stat().st_size / 1024 / 1024:.1f} MB")
            print()
    
    else:
        print(f"错误: 未知模式 '{MODE}'")
        print("请设置 MODE 为 'process' 或 'analyze'")
