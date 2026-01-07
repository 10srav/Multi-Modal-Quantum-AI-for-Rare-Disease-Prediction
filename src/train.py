"""
Training script for HGPS Prediction Models.

This script trains all models (Classical ML, CNN, QML) and saves them
for production deployment.

Usage:
    python -m src.train --all
    python -m src.train --classical
    python -m src.train --qml
    python -m src.train --cnn
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data import (
    TabularPreprocessor,
    create_data_splits,
    generate_hgps_tabular_data,
    get_qml_features,
)
from src.models import ClassicalTabularModels

# Ensure directories exist
settings.paths.ensure_dirs()

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.paths.logs_dir / "training.log"),
    ],
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates training of all model types."""

    def __init__(self):
        self.settings = settings
        self.settings.paths.ensure_dirs()
        self.results = {}
        self.device = self._get_device()

    def _get_device(self) -> str:
        if self.settings.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.settings.device

    def generate_training_data(self) -> tuple:
        """Generate and split training data."""
        logger.info("Generating synthetic training data...")

        df = generate_hgps_tabular_data(
            n_hgps=self.settings.model.n_hgps_train,
            n_controls=self.settings.model.n_controls_train,
            random_state=self.settings.model.random_state,
        )

        train_df, val_df, test_df = create_data_splits(
            df,
            test_size=self.settings.model.test_size,
            val_size=self.settings.model.val_size,
            random_state=self.settings.model.random_state,
        )

        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Save data splits
        train_df.to_csv(self.settings.paths.data_dir / "train.csv", index=False)
        val_df.to_csv(self.settings.paths.data_dir / "val.csv", index=False)
        test_df.to_csv(self.settings.paths.data_dir / "test.csv", index=False)

        return train_df, val_df, test_df

    def train_preprocessor(self, train_df: pd.DataFrame) -> TabularPreprocessor:
        """Fit and save the preprocessor."""
        logger.info("Fitting preprocessor...")

        preprocessor = TabularPreprocessor()
        preprocessor.fit(train_df)

        # Save preprocessor
        preprocessor_path = self.settings.paths.models_dir / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        return preprocessor

    def train_classical_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        preprocessor: TabularPreprocessor,
    ) -> dict:
        """Train and evaluate classical ML models."""
        logger.info("=" * 50)
        logger.info("Training Classical ML Models")
        logger.info("=" * 50)

        # Prepare features
        X_train = preprocessor.transform(train_df)
        X_val = preprocessor.transform(val_df)
        X_test = preprocessor.transform(test_df)

        y_train = train_df["risk_label"].values
        y_val = val_df["risk_label"].values
        y_test = test_df["risk_label"].values

        # Train models using ClassicalTabularModels
        start_time = time.time()
        classical_models = ClassicalTabularModels(
            calibrate=True,
            random_state=self.settings.model.random_state
        )
        classical_models.fit(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time

        logger.info(f"Classical models trained in {training_time:.2f}s")

        # Evaluate on test set
        results = classical_models.evaluate(X_test, y_test)

        # Save the trained models
        models_path = self.settings.paths.models_dir / "classical_models.joblib"
        joblib.dump(classical_models, models_path)
        logger.info(f"Classical models saved to {models_path}")

        # Log results
        for name, metrics in results.items():
            logger.info(f"{name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

        self.results["classical"] = results
        return results

    def train_qml_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        preprocessor: TabularPreprocessor,
    ) -> dict:
        """Train and evaluate Quantum ML models."""
        logger.info("=" * 50)
        logger.info("Training Quantum ML Models")
        logger.info("=" * 50)

        try:
            from src.qml import QuantumSVM, QuantumNeuralNetwork

            # Get QML features (uses default 6 features for quantum circuits)
            X_train, y_train, _ = get_qml_features(train_df, preprocessor)
            X_val, y_val, _ = get_qml_features(val_df, preprocessor)
            X_test, y_test, _ = get_qml_features(test_df, preprocessor)

            results = {}

            # Train QSVM
            logger.info("Training Quantum SVM...")
            start_time = time.time()

            qsvm = QuantumSVM(
                num_features=self.settings.model.qml_feature_dim,
                feature_map_reps=2,
            )
            qsvm.fit(X_train, y_train)

            qsvm_time = time.time() - start_time
            logger.info(f"QSVM trained in {qsvm_time:.2f}s")

            # Evaluate QSVM
            y_pred_qsvm = qsvm.predict(X_test)
            results["QSVM"] = {
                "accuracy": accuracy_score(y_test, y_pred_qsvm),
                "precision": precision_score(y_test, y_pred_qsvm, zero_division=0),
                "recall": recall_score(y_test, y_pred_qsvm, zero_division=0),
                "f1": f1_score(y_test, y_pred_qsvm, zero_division=0),
            }
            logger.info(f"QSVM: Accuracy={results['QSVM']['accuracy']:.4f}, F1={results['QSVM']['f1']:.4f}")

            # Save QSVM
            qsvm_path = self.settings.paths.models_dir / "qsvm.joblib"
            joblib.dump(qsvm, qsvm_path)

            # Train QNN
            logger.info("Training Quantum Neural Network...")
            start_time = time.time()

            qnn = QuantumNeuralNetwork(
                num_features=self.settings.model.qml_feature_dim,
                ansatz_reps=2,
            )
            qnn.fit(X_train, y_train)

            qnn_time = time.time() - start_time
            logger.info(f"QNN trained in {qnn_time:.2f}s")

            # Evaluate QNN
            y_pred_qnn = qnn.predict(X_test)
            results["QNN"] = {
                "accuracy": accuracy_score(y_test, y_pred_qnn),
                "precision": precision_score(y_test, y_pred_qnn, zero_division=0),
                "recall": recall_score(y_test, y_pred_qnn, zero_division=0),
                "f1": f1_score(y_test, y_pred_qnn, zero_division=0),
            }
            logger.info(f"QNN: Accuracy={results['QNN']['accuracy']:.4f}, F1={results['QNN']['f1']:.4f}")

            # Save QNN
            qnn_path = self.settings.paths.models_dir / "qnn.joblib"
            joblib.dump(qnn, qnn_path)

            self.results["qml"] = results
            return results

        except ImportError as e:
            logger.warning(f"QML dependencies not available: {e}")
            return {}

    def train_cnn_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        """Train and evaluate CNN model for face analysis."""
        logger.info("=" * 50)
        logger.info("Training CNN Face Model")
        logger.info("=" * 50)

        try:
            from src.features.face_cnn import FaceCNN

            # Create CNN model
            cnn = FaceCNN(
                pretrained=self.settings.model.cnn_pretrained,
                embedding_dim=256,
                num_risk_classes=2,
                num_progression_classes=3,
            )

            # For now, we save the pre-trained backbone
            # In production, this would be fine-tuned on real face data
            cnn_path = self.settings.paths.models_dir / "face_cnn.pt"
            torch.save(cnn.state_dict(), cnn_path)
            logger.info(f"CNN model saved to {cnn_path}")

            results = {
                "CNN": {
                    "status": "pretrained_backbone_saved",
                    "backbone": "resnet18",
                    "note": "Fine-tune on real facial data for production",
                }
            }

            self.results["cnn"] = results
            return results

        except Exception as e:
            logger.error(f"CNN training failed: {e}")
            return {}

    def save_training_report(self):
        """Save comprehensive training report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "n_hgps": self.settings.model.n_hgps_train,
                "n_controls": self.settings.model.n_controls_train,
                "random_state": self.settings.model.random_state,
                "device": self.device,
            },
            "results": self.results,
        }

        report_path = self.settings.paths.results_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Training report saved to {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        if "classical" in self.results:
            print("\nClassical ML Models:")
            for name, metrics in self.results["classical"].items():
                print(f"  {name}: F1={metrics['f1']:.4f}, AUC={metrics.get('roc_auc', 'N/A')}")

        if "qml" in self.results:
            print("\nQuantum ML Models:")
            for name, metrics in self.results["qml"].items():
                print(f"  {name}: F1={metrics['f1']:.4f}")

        print("\n" + "=" * 60)

    def train_all(self):
        """Train all model types."""
        logger.info("Starting full training pipeline...")

        # Generate data
        train_df, val_df, test_df = self.generate_training_data()

        # Train preprocessor
        preprocessor = self.train_preprocessor(train_df)

        # Train all model types
        self.train_classical_models(train_df, val_df, test_df, preprocessor)
        self.train_qml_models(train_df, val_df, test_df, preprocessor)
        self.train_cnn_model(train_df, val_df, test_df)

        # Save report
        self.save_training_report()

        logger.info("Training pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train HGPS Prediction Models")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--classical", action="store_true", help="Train classical ML models only")
    parser.add_argument("--qml", action="store_true", help="Train QML models only")
    parser.add_argument("--cnn", action="store_true", help="Train CNN model only")

    args = parser.parse_args()

    trainer = ModelTrainer()

    if args.all or not any([args.classical, args.qml, args.cnn]):
        trainer.train_all()
    else:
        # Generate data first
        train_df, val_df, test_df = trainer.generate_training_data()
        preprocessor = trainer.train_preprocessor(train_df)

        if args.classical:
            trainer.train_classical_models(train_df, val_df, test_df, preprocessor)

        if args.qml:
            trainer.train_qml_models(train_df, val_df, test_df, preprocessor)

        if args.cnn:
            trainer.train_cnn_model(train_df, val_df, test_df)

        trainer.save_training_report()


if __name__ == "__main__":
    main()
