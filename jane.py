import warnings
warnings.filterwarnings('ignore')


import os
import gc
import pandas as pd
import numpy as np
import json
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

import cloudpickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger

from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr, correlation_contribution
from sklearn.model_selection import TimeSeriesSplit

os.environ["MallocStackLogging"] = "0"
os.environ["MallocStackLoggingNoCompact"] = "0"


# ============================================================================
# Configuration
# ============================================================================
DATA_VERSION = "v5.2"
FEATURE_SET = "small"  # or "medium" or "all"
MAIN_TARGET = "target_ender_20"
TARGET_CANDIDATES = [
    "target_ender_20",
    "target_victor_20",
    "target_xerxes_20",
    "target_teager2b_20"
]

PARAMS = {
    'hidden_units': [96, 96, 896, 448, 448, 256],
    'dropout_rates': [0.035, 0.038, 0.424, 0.104, 0.492, 0.320, 0.272, 0.438],
    'learning_rate': 1e-3,
    'label_smoothing': 0.0,
    'weight_decay': 1e-5,
    'batch_size': 4096,
    'n_splits': 5,
    'max_epochs': 100,
}


# ============================================================================
# Dataset Class
# ============================================================================
class NumeraiDataset(Dataset):
    """Dataset for Numerai data with era support"""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 eras: np.ndarray = None, weights: np.ndarray = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.eras = eras
        self.weights = torch.FloatTensor(weights) if weights is not None else torch.ones(len(features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': self.targets[idx],
            'weights': self.weights[idx]
        }


def save_model_pkl(model: pl.LightningModule, path: str):
    model.eval()
    model.cpu()

    def predict_fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            _, _, preds = model(x)
            return torch.sigmoid(preds).numpy()

    with open(path, "wb") as f:
        cloudpickle.dump(predict_fn, f)

def get_accelerator():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"

# ============================================================================
# Model Architecture (Autoencoder + MLP inspired by Jane Street)
# ============================================================================
class NumeraiAutoEncoderMLP(pl.LightningModule):
    """
    Autoencoder + Multi-Layer Perceptron for Numerai
    Adapted from Jane Street competition architecture
    """

    def __init__(self, num_features: int, num_targets: int,
                 hidden_units: List[int], dropout_rates: List[float],
                 learning_rate: float = 1e-3, label_smoothing: float = 0.0,
                 weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.num_features = num_features
        self.num_targets = num_targets
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.weight_decay = weight_decay

        # Input normalization
        self.input_bn = nn.BatchNorm1d(num_features)

        # Encoder
        self.encoder_noise = nn.Dropout(dropout_rates[0])
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.SiLU()  # Swish activation
        )

        # Decoder (reconstruction)
        self.decoder = nn.Sequential(
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_units[0], num_features)
        )

        # Auxiliary prediction from decoder
        self.ae_predictor = nn.Sequential(
            nn.Linear(num_features, hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(hidden_units[1], num_targets)
        )

        # Main prediction network (concatenate input + encoder)
        concat_size = num_features + hidden_units[0]
        layers = [
            nn.BatchNorm1d(concat_size),
            nn.Dropout(dropout_rates[3])
        ]

        current_size = concat_size
        for i, units in enumerate(hidden_units[2:], start=2):
            layers.extend([
                nn.Linear(current_size, units),
                nn.BatchNorm1d(units),
                nn.SiLU(),
                nn.Dropout(dropout_rates[i + 2])
            ])
            current_size = units

        layers.append(nn.Linear(current_size, num_targets))
        self.main_predictor = nn.Sequential(*layers)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # Input normalization
        x_norm = self.input_bn(x)

        # Encoder
        encoder_out = self.encoder(self.encoder_noise(x_norm))

        # Decoder (reconstruction)
        decoder_out = self.decoder(encoder_out)

        # Auxiliary predictions
        ae_pred = self.ae_predictor(decoder_out)

        # Main predictions (concatenate original input + encoder)
        concat = torch.cat([x_norm, encoder_out], dim=1)
        main_pred = self.main_predictor(concat)

        return decoder_out, ae_pred, main_pred

    def training_step(self, batch, batch_idx):
        features = batch['features']
        targets = batch['targets']
        weights = batch['weights']

        decoder_out, ae_pred, main_pred = self(features)

        # Reconstruction loss
        recon_loss = self.mse_loss(decoder_out, features)

        # Auxiliary prediction loss (using targets as binary classification)
        # Convert Numerai targets to binary (above/below 0.5)
        binary_targets = (targets > 0.5).float()
        ae_loss = self.bce_loss(ae_pred, binary_targets)

        # Main prediction loss (regression for Numerai targets)
        main_loss = F.mse_loss(torch.sigmoid(main_pred), targets, reduction='none')
        main_loss = (main_loss * weights.unsqueeze(1)).mean()

        # Combined loss
        loss = recon_loss + ae_loss + main_loss

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/recon_loss', recon_loss)
        self.log('train/ae_loss', ae_loss)
        self.log('train/main_loss', main_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        features = batch['features']
        targets = batch['targets']
        weights = batch['weights']

        decoder_out, ae_pred, main_pred = self(features)

        # Losses
        recon_loss = self.mse_loss(decoder_out, features)
        binary_targets = (targets > 0.5).float()
        ae_loss = self.bce_loss(ae_pred, binary_targets)
        main_loss = F.mse_loss(torch.sigmoid(main_pred), targets, reduction='none')
        main_loss = (main_loss * weights.unsqueeze(1)).mean()
        loss = recon_loss + ae_loss + main_loss

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/recon_loss', recon_loss)
        self.log('val/ae_loss', ae_loss)
        self.log('val/main_loss', main_loss)

        return {
            'loss': loss,
            'predictions': torch.sigmoid(main_pred),
            'targets': targets
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }

    def predict_step(self, batch, batch_idx):
        features = batch['features']
        _, _, main_pred = self(features)
        return torch.sigmoid(main_pred)


# ============================================================================
# Weighted Average (Donate et al.'s formula)
# ============================================================================
def weighted_average(scores):
    """Weighted average for ensemble of folds"""
    n = len(scores)
    weights = []
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        weights.append(1 / (2**(n + 1 - j)))
    return np.average(scores, weights=weights)


# ============================================================================
# Era-based Cross-Validation
# ============================================================================
def era_based_split(train_df, n_splits=3, embargo_gap=4):
    unique_eras = np.sort(train_df['era'].unique())
    n_eras = len(unique_eras)
    era_size = n_eras // n_splits

    splits = []

    for i in range(n_splits):
        val_start = i * era_size
        val_end = val_start + era_size if i < n_splits - 1 else n_eras

        train_end = val_start - embargo_gap
        if train_end <= 0:
            continue  # train set пуст — такой фолд недопустим

        train_eras = unique_eras[:train_end]
        val_eras = unique_eras[val_start:val_end]

        train_mask = train_df['era'].isin(train_eras).values
        val_mask   = train_df['era'].isin(val_eras).values

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))

    return splits


# ============================================================================
# Numerai Correlation Metric
# ============================================================================
def compute_numerai_metrics(predictions, targets, eras):
    """Compute per-era correlation (Numerai's primary metric)"""
    df = pd.DataFrame({
        'prediction': predictions.flatten(),
        'target': targets.flatten(),
        'era': eras
    })

    per_era_corr = df.groupby('era').apply(
        lambda x: numerai_corr(x[['prediction']], x['target'])
    )

    return {
        'mean': per_era_corr.mean().values[0],
        'std': per_era_corr.std().values[0],
        'sharpe': (per_era_corr.mean() / per_era_corr.std()).values[0]
    }


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    print("=" * 80)
    print("Numerai PyTorch Lightning Training")
    print("=" * 80)

    # Initialize API
    napi = NumerAPI()

    # Download data
    print("\n[1/5] Downloading data...")
    napi.download_dataset(f"{DATA_VERSION}/train.parquet")
    napi.download_dataset(f"{DATA_VERSION}/features.json")

    # Load features metadata
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_cols = feature_metadata["feature_sets"][FEATURE_SET]

    # Load training data
    print("[2/5] Loading training data...")
    train = pd.read_parquet(
        f"{DATA_VERSION}/train.parquet",
        columns=["era"] + feature_cols + TARGET_CANDIDATES
    )

    # Downsample for faster training (optional)
    train = train.reset_index(drop=True)

    SUBSET_ERAS = train["era"].unique()[::8]  # 12.5%

    train = train[train["era"].isin(SUBSET_ERAS)].reset_index(drop=True)

    print(f"Training data shape: {train.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Number of targets: {len(TARGET_CANDIDATES)}")

    # Prepare data
    X = train[feature_cols].values
    y = train[TARGET_CANDIDATES].values
    eras = train['era'].values

    # Create era-based splits
    print("\n[3/5] Creating era-based CV splits...")
    splits = era_based_split(train, n_splits=PARAMS['n_splits'])

    # Train models for each fold
    print("\n[4/5] Training models...")
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{PARAMS['n_splits']}")
        print(f"{'=' * 60}")

        # Create datasets
        train_dataset = NumeraiDataset(X[train_idx], y[train_idx], eras[train_idx])
        val_dataset = NumeraiDataset(X[val_idx], y[val_idx], eras[val_idx])

        wandb_logger = WandbLogger(
            project="numerai-ae-mlp",
            name=f"fold_{fold}",
            log_model=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=PARAMS['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=PARAMS['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Create model
        model = NumeraiAutoEncoderMLP(
            num_features=len(feature_cols),
            num_targets=len(TARGET_CANDIDATES),
            hidden_units=PARAMS['hidden_units'],
            dropout_rates=PARAMS['dropout_rates'],
            learning_rate=PARAMS['learning_rate'],
            label_smoothing=PARAMS['label_smoothing'],
            weight_decay=PARAMS['weight_decay']
        )

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/fold_{fold}',
            filename='numerai-{epoch:02d}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=1
        )

        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            patience=10,
            mode='min',
            verbose=True
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Trainer
        trainer = pl.Trainer(
            max_epochs=PARAMS['max_epochs'],
            accelerator=get_accelerator(),
            devices=1,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            logger=CSVLogger('logs', name=f'fold_{fold}'),
            gradient_clip_val=1.0,
            deterministic=True
        )

        # Train
        trainer.fit(model, train_loader, val_loader)
        fold_pkl_path = f"checkpoints/fold_{fold}/model_fold_{fold}.pkl"
        save_model_pkl(model, fold_pkl_path)

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                preds = model.predict_step(batch, 0)
                val_predictions.append(preds.cpu().numpy())
                val_targets.append(batch['targets'].cpu().numpy())

        val_predictions = np.vstack(val_predictions)
        val_targets = np.vstack(val_targets)

        # Compute Numerai correlation for main target
        metrics = compute_numerai_metrics(
            val_predictions[:, 0],  # Main target predictions
            val_targets[:, 0],      # Main target actual
            eras[val_idx]
        )

        print(f"\nFold {fold + 1} Results:")
        print(f"  Mean Correlation: {metrics['mean']:.6f}")
        print(f"  Std Correlation:  {metrics['std']:.6f}")
        print(f"  Sharpe:           {metrics['sharpe']:.6f}")

        fold_scores.append(metrics['mean'])

        # Clean up
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    def load_fold_models(paths):
      models = []
      for p in paths:


        with open(p, "rb") as f:
          models.append(cloudpickle.load(f))
      return models


    fold_model_paths = [
      f"checkpoints/fold_{i}/model_fold_{i}.pkl"
      for i in range(len(splits))
    ]

    fold_predict_fns = load_fold_models(fold_model_paths)


    def ensemble_predict_fn(x: np.ndarray) -> np.ndarray:
      preds = [fn(x) for fn in fold_predict_fns]
      preds = np.stack(preds, axis=0)  # (n_folds, N, targets)
      return weighted_average(preds)


    with open("final_ensemble_model.pkl", "wb") as f:
      cloudpickle.dump(ensemble_predict_fn, f)

    # Final results
    print("\n" + "=" * 80)
    print("[5/5] Final Results")
    print("=" * 80)
    print(f"Fold Scores: {[f'{s:.6f}' for s in fold_scores]}")
    print(f"Mean Score: {np.mean(fold_scores):.6f}")
    print(f"Weighted Average Score: {weighted_average(fold_scores):.6f}")
    print("=" * 80)


if __name__ == '__main__':
    pl.seed_everything(42)
    main()
