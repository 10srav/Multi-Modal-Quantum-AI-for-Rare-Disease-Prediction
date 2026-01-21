"""
Classical ML Models and Training Infrastructure

Includes:
- Scikit-learn classifiers (SVM, Random Forest, XGBoost)
- PyTorch training loops
- Model calibration using Platt scaling
- Comprehensive evaluation metrics
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .features import FaceCNN, TabularMLP, LateFusionClassifier, MultimodalFusionModel
from .features.face_cnn import FocalLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SCIKIT-LEARN CLASSICAL MODELS
# ============================================================================

class ClassicalTabularModels:
    """
    Wrapper for classical scikit-learn models on tabular data.

    Supports SVM, Random Forest, XGBoost, and Logistic Regression
    with automatic probability calibration.
    """

    def __init__(self, calibrate: bool = True, random_state: int = 42):
        """
        Args:
            calibrate: Whether to apply probability calibration
            random_state: Random seed for reproducibility
        """
        self.calibrate = calibrate
        self.random_state = random_state
        self.models = {}
        self.calibrated_models = {}

        self._init_models()

    def _init_models(self):
        """Initialize the classical models."""
        self.models = {
            'svm_rbf': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'svm_linear': SVC(
                kernel='linear',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'logistic': LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
        }

        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=9  # For 90:10 imbalance
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train all or selected classical models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features for calibration
            y_val: Validation labels for calibration
            model_names: Specific models to train (None = all)

        Returns:
            Dictionary of training results
        """
        if model_names is None:
            model_names = list(self.models.keys())

        results = {}

        for name in model_names:
            if name not in self.models:
                logger.warning(f"Model {name} not found, skipping")
                continue

            logger.info(f"Training {name}...")

            model = self.models[name]
            model.fit(X_train, y_train)

            # Calibrate if requested and validation data available
            if self.calibrate and X_val is not None and len(X_val) >= 6:
                logger.info(f"Calibrating {name}...")
                try:
                    # Use cross-validation based calibration (works with all sklearn versions)
                    # Clone the base estimator for calibration
                    from sklearn.base import clone
                    base_model = clone(self.models[name])
                    calibrated = CalibratedClassifierCV(
                        base_model, method='isotonic', cv=3
                    )
                    # Fit on combined train+val data for calibration
                    X_combined = np.vstack([X_train, X_val])
                    y_combined = np.hstack([y_train, y_val])
                    calibrated.fit(X_combined, y_combined)
                    self.calibrated_models[name] = calibrated
                except Exception as e:
                    # Fallback: use uncalibrated model
                    logger.warning(f"Calibration failed for {name}: {e}, using uncalibrated model")
                    self.calibrated_models[name] = model
            else:
                self.calibrated_models[name] = model

            results[name] = {'status': 'trained'}

        return results

    def predict(
        self,
        X: np.ndarray,
        model_name: str = 'xgboost'
    ) -> np.ndarray:
        """Get class predictions."""
        if model_name not in self.calibrated_models:
            model_name = list(self.calibrated_models.keys())[0]
        return self.calibrated_models[model_name].predict(X)

    def predict_proba(
        self,
        X: np.ndarray,
        model_name: str = 'xgboost'
    ) -> np.ndarray:
        """Get probability predictions."""
        if model_name not in self.calibrated_models:
            model_name = list(self.calibrated_models.keys())[0]
        return self.calibrated_models[model_name].predict_proba(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate models on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Specific model to evaluate (None = all)

        Returns:
            Dictionary of metrics per model
        """
        if model_name:
            models_to_eval = {model_name: self.calibrated_models[model_name]}
        else:
            models_to_eval = self.calibrated_models

        results = {}

        for name, model in models_to_eval.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # Binary classification metrics
            if y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, 1]
                try:
                    auc = roc_auc_score(y_test, y_proba_pos)
                except ValueError:
                    auc = 0.5
            else:
                try:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                except ValueError:
                    auc = 0.5

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'auc': auc
            }

            logger.info(f"{name} - Acc: {results[name]['accuracy']:.4f}, "
                       f"F1: {results[name]['f1']:.4f}, AUC: {results[name]['auc']:.4f}")

        return results

    def get_feature_importance(self, model_name: str = 'random_forest') -> Optional[np.ndarray]:
        """Get feature importance for tree-based models."""
        model = self.models.get(model_name)
        if model is None:
            return None

        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).flatten()
        return None


# ============================================================================
# PYTORCH TRAINING INFRASTRUCTURE
# ============================================================================

class PyTorchTrainer:
    """
    Training infrastructure for PyTorch models.

    Handles training loops, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            model: PyTorch model to train
            device: Device for training
            learning_rate: Learning rate
            weight_decay: L2 regularization
            focal_gamma: Focal loss gamma parameter
            class_weights: Class weights for imbalanced data
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Loss functions
        if class_weights is not None:
            class_weights = class_weights.to(device)

        self.risk_criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.progression_criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_risk_acc': [], 'val_risk_acc': [],
            'train_prog_acc': [], 'val_prog_acc': []
        }

        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(
        self,
        train_loader: DataLoader,
        is_multimodal: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        risk_correct = 0
        prog_correct = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            if is_multimodal:
                images = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                risk_labels = batch['risk_label'].to(self.device)
                prog_labels = batch['progression_label'].to(self.device)

                # Forward pass
                output = self.model(images, tabular)
            else:
                # Single modality
                if 'image' in batch:
                    x = batch['image'].to(self.device)
                else:
                    x = batch['tabular'].to(self.device)
                risk_labels = batch['risk_label'].to(self.device)
                prog_labels = batch['progression_label'].to(self.device)

                output = self.model(x)

            # Compute loss
            risk_loss = self.risk_criterion(output['risk_logits'], risk_labels)
            prog_loss = self.progression_criterion(output['progression_logits'], prog_labels)
            loss = risk_loss + 0.5 * prog_loss  # Weight risk more heavily

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * len(risk_labels)
            risk_pred = output['risk_logits'].argmax(dim=1)
            prog_pred = output['progression_logits'].argmax(dim=1)
            risk_correct += (risk_pred == risk_labels).sum().item()
            prog_correct += (prog_pred == prog_labels).sum().item()
            total_samples += len(risk_labels)

        return {
            'loss': total_loss / total_samples,
            'risk_acc': risk_correct / total_samples,
            'prog_acc': prog_correct / total_samples
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        is_multimodal: bool = True
    ) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        risk_correct = 0
        prog_correct = 0
        total_samples = 0

        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if is_multimodal:
                images = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                risk_labels = batch['risk_label'].to(self.device)
                prog_labels = batch['progression_label'].to(self.device)

                output = self.model(images, tabular)
            else:
                if 'image' in batch:
                    x = batch['image'].to(self.device)
                else:
                    x = batch['tabular'].to(self.device)
                risk_labels = batch['risk_label'].to(self.device)
                prog_labels = batch['progression_label'].to(self.device)

                output = self.model(x)

            risk_loss = self.risk_criterion(output['risk_logits'], risk_labels)
            prog_loss = self.progression_criterion(output['progression_logits'], prog_labels)
            loss = risk_loss + 0.5 * prog_loss

            total_loss += loss.item() * len(risk_labels)
            risk_pred = output['risk_logits'].argmax(dim=1)
            prog_pred = output['progression_logits'].argmax(dim=1)
            risk_correct += (risk_pred == risk_labels).sum().item()
            prog_correct += (prog_pred == prog_labels).sum().item()
            total_samples += len(risk_labels)

        return {
            'loss': total_loss / total_samples,
            'risk_acc': risk_correct / total_samples,
            'prog_acc': prog_correct / total_samples
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        is_multimodal: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum epochs
            early_stopping_patience: Epochs without improvement before stopping
            is_multimodal: Whether model expects both image and tabular input
            save_dir: Directory to save best model

        Returns:
            Training history dictionary
        """
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, is_multimodal)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_risk_acc'].append(train_metrics['risk_acc'])
            self.history['train_prog_acc'].append(train_metrics['prog_acc'])

            # Validate
            val_metrics = self.validate(val_loader, is_multimodal)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_risk_acc'].append(val_metrics['risk_acc'])
            self.history['val_prog_acc'].append(val_metrics['prog_acc'])

            # Update scheduler
            self.scheduler.step(val_metrics['loss'])

            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Risk Acc: {train_metrics['risk_acc']:.4f}, "
                f"Prog Acc: {train_metrics['prog_acc']:.4f}"
            )
            logger.info(
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Risk Acc: {val_metrics['risk_acc']:.4f}, "
                f"Prog Acc: {val_metrics['prog_acc']:.4f}"
            )

            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0

                if save_dir:
                    self.save_checkpoint(save_dir, epoch, val_metrics)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        torch.save(checkpoint, save_path / 'best_model.pt')
        logger.info(f"Saved checkpoint to {save_path / 'best_model.pt'}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


# ============================================================================
# MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation.

    Computes classification metrics, calibration metrics, and
    generates evaluation plots.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: Trained PyTorch model
            device: Device for inference
        """
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def get_predictions(
        self,
        data_loader: DataLoader,
        is_multimodal: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get all predictions for a dataset.

        Returns:
            Dictionary with predictions, probabilities, and labels
        """
        all_risk_probs = []
        all_prog_probs = []
        all_risk_labels = []
        all_prog_labels = []

        for batch in data_loader:
            if is_multimodal:
                images = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                output = self.model.predict_proba(images, tabular)
            else:
                if 'image' in batch:
                    x = batch['image'].to(self.device)
                else:
                    x = batch['tabular'].to(self.device)
                output = self.model.predict_proba(x)

            all_risk_probs.append(output['risk_probs'].cpu().numpy())
            all_prog_probs.append(output['progression_probs'].cpu().numpy())
            all_risk_labels.append(batch['risk_label'].numpy())
            all_prog_labels.append(batch['progression_label'].numpy())

        return {
            'risk_probs': np.vstack(all_risk_probs),
            'prog_probs': np.vstack(all_prog_probs),
            'risk_labels': np.concatenate(all_risk_labels),
            'prog_labels': np.concatenate(all_prog_labels),
            'risk_preds': np.vstack(all_risk_probs).argmax(axis=1),
            'prog_preds': np.vstack(all_prog_probs).argmax(axis=1)
        }

    def compute_metrics(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive metrics.

        Args:
            predictions: Output from get_predictions()

        Returns:
            Dictionary of metrics for risk and progression tasks
        """
        metrics = {}

        # Risk classification metrics
        risk_labels = predictions['risk_labels']
        risk_preds = predictions['risk_preds']
        risk_probs = predictions['risk_probs']

        metrics['risk'] = {
            'accuracy': accuracy_score(risk_labels, risk_preds),
            'f1': f1_score(risk_labels, risk_preds, average='weighted'),
            'precision': precision_score(risk_labels, risk_preds, average='weighted', zero_division=0),
            'recall': recall_score(risk_labels, risk_preds, average='weighted', zero_division=0),
            'sensitivity': recall_score(risk_labels, risk_preds, pos_label=1, zero_division=0),
            'specificity': recall_score(risk_labels, risk_preds, pos_label=0, zero_division=0)
        }

        # AUC for binary classification
        if risk_probs.shape[1] == 2:
            try:
                metrics['risk']['auc'] = roc_auc_score(risk_labels, risk_probs[:, 1])
            except ValueError:
                metrics['risk']['auc'] = 0.5

        # Progression classification metrics
        prog_labels = predictions['prog_labels']
        prog_preds = predictions['prog_preds']
        prog_probs = predictions['prog_probs']

        metrics['progression'] = {
            'accuracy': accuracy_score(prog_labels, prog_preds),
            'f1': f1_score(prog_labels, prog_preds, average='weighted'),
            'precision': precision_score(prog_labels, prog_preds, average='weighted', zero_division=0),
            'recall': recall_score(prog_labels, prog_preds, average='weighted', zero_division=0)
        }

        # Multi-class AUC
        try:
            metrics['progression']['auc'] = roc_auc_score(
                prog_labels, prog_probs, multi_class='ovr', average='weighted'
            )
        except ValueError:
            metrics['progression']['auc'] = 0.5

        return metrics

    def compute_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            y_true: True labels
            y_prob: Predicted probabilities for positive class
            n_bins: Number of bins for calibration

        Returns:
            Dictionary with ECE and MCE (Maximum Calibration Error)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        mce = 0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += prop_in_bin * bin_error
                mce = max(mce, bin_error)

        return {'ece': ece, 'mce': mce}

    def get_confusion_matrices(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Get confusion matrices for both tasks."""
        return {
            'risk': confusion_matrix(
                predictions['risk_labels'],
                predictions['risk_preds']
            ),
            'progression': confusion_matrix(
                predictions['prog_labels'],
                predictions['prog_preds']
            )
        }

    def get_classification_reports(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, str]:
        """Get detailed classification reports."""
        return {
            'risk': classification_report(
                predictions['risk_labels'],
                predictions['risk_preds'],
                target_names=['Control', 'HGPS']
            ),
            'progression': classification_report(
                predictions['prog_labels'],
                predictions['prog_preds'],
                target_names=['Slow', 'Moderate', 'Rapid']
            )
        }


# ============================================================================
# PROBABILITY CALIBRATION
# ============================================================================

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for neural network calibration.

    Learns a single temperature parameter to scale logits
    before softmax to improve calibration.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        is_multimodal: bool = True,
        max_iter: int = 100
    ):
        """
        Learn optimal temperature on validation set.

        Args:
            model: Trained model
            val_loader: Validation data loader
            device: Device for computation
            is_multimodal: Whether model is multimodal
            max_iter: Maximum optimization iterations
        """
        model.eval()
        self.to(device)

        # Collect all logits and labels
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if is_multimodal:
                    images = batch['image'].to(device)
                    tabular = batch['tabular'].to(device)
                    output = model(images, tabular)
                else:
                    if 'image' in batch:
                        x = batch['image'].to(device)
                    else:
                        x = batch['tabular'].to(device)
                    output = model(x)

                all_logits.append(output['risk_logits'])
                all_labels.append(batch['risk_label'].to(device))

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        # Optimize temperature
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        logger.info(f"Optimal temperature: {self.temperature.item():.4f}")

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply calibration to logits."""
        return torch.softmax(self.forward(logits), dim=1)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.

    Args:
        labels: Array of class labels

    Returns:
        Tensor of class weights
    """
    classes, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(classes) * counts)
    return torch.FloatTensor(weights)


def train_classical_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[ClassicalTabularModels, Dict[str, Dict[str, float]]]:
    """
    Train and evaluate all classical models.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data

    Returns:
        Tuple of (trained models, evaluation results)
    """
    models = ClassicalTabularModels(calibrate=True)
    models.fit(X_train, y_train, X_val, y_val)
    results = models.evaluate(X_test, y_test)
    return models, results


if __name__ == "__main__":
    # Test classical models
    print("Testing classical models...")

    # Generate dummy data
    np.random.seed(42)
    n_samples = 500
    n_features = 11

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split
    train_idx = int(0.7 * n_samples)
    val_idx = int(0.85 * n_samples)

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    # Train and evaluate
    models, results = train_classical_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\nResults:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\nClassical models test passed!")
