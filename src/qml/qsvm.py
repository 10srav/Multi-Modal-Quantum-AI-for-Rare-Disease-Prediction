"""
Quantum Support Vector Machine (QSVM) for HGPS Detection

Implements QSVM with quantum kernels for binary classification,
comparing against classical SVM baselines.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
import logging
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

logger = logging.getLogger(__name__)

# Qiskit imports with fallback
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit.primitives import Sampler
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    logger.warning("Qiskit not available. QSVM will use classical fallback.")


class QuantumKernel:
    """
    Quantum kernel computation using quantum feature maps.

    Computes the kernel matrix K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
    where phi is the quantum feature map.
    """

    def __init__(
        self,
        num_qubits: int,
        feature_map_reps: int = 2,
        entanglement: str = 'full'
    ):
        """
        Args:
            num_qubits: Number of qubits (must match feature dimension)
            feature_map_reps: Repetitions of the feature map circuit
            entanglement: Entanglement pattern for the feature map
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required for quantum kernels")

        self.num_qubits = num_qubits
        self.feature_map_reps = feature_map_reps
        self.entanglement = entanglement

        # Create the quantum feature map
        self.feature_map = ZZFeatureMap(
            feature_dimension=num_qubits,
            reps=feature_map_reps,
            entanglement=entanglement
        )

        # Create the quantum kernel
        self.sampler = Sampler()
        self.kernel = FidelityQuantumKernel(
            feature_map=self.feature_map
        )

        logger.info(
            f"QuantumKernel initialized: {num_qubits} qubits, "
            f"{feature_map_reps} reps, {entanglement} entanglement"
        )

    def compute_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the quantum kernel matrix.

        Args:
            X1: First set of data points (n1, num_features)
            X2: Second set of data points (n2, num_features), None for self-kernel

        Returns:
            Kernel matrix of shape (n1, n2) or (n1, n1)
        """
        if X2 is None:
            return self.kernel.evaluate(X1)
        else:
            return self.kernel.evaluate(X1, X2)


class QuantumSVM:
    """
    Quantum Support Vector Machine for binary classification.

    Uses a quantum kernel for computing similarities between data points,
    which can capture complex feature interactions that classical kernels miss.
    """

    def __init__(
        self,
        num_features: int = 6,
        feature_map_reps: int = 2,
        entanglement: str = 'full',
        C: float = 1.0,
        use_quantum: bool = True
    ):
        """
        Args:
            num_features: Number of input features
            feature_map_reps: Repetitions in quantum feature map
            entanglement: Entanglement pattern
            C: SVM regularization parameter
            use_quantum: If False, use classical SVM (for comparison)
        """
        self.num_features = num_features
        self.use_quantum = use_quantum and HAS_QISKIT
        self.C = C

        if self.use_quantum:
            # Create quantum feature map
            self.feature_map = ZZFeatureMap(
                feature_dimension=num_features,
                reps=feature_map_reps,
                entanglement=entanglement
            )

            # Create QSVC (Quantum SVC)
            self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
            self.model = QSVC(quantum_kernel=self.kernel, C=C)

            logger.info(f"Initialized Quantum SVM with {num_features} qubits")
        else:
            # Classical SVM fallback
            self.model = SVC(
                kernel='rbf',
                C=C,
                probability=True,
                class_weight='balanced'
            )
            logger.info("Initialized Classical SVM (RBF kernel)")

        self.is_fitted = False
        self.train_data = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> 'QuantumSVM':
        """
        Train the QSVM on the training data.

        Args:
            X_train: Training features (n_samples, num_features)
            y_train: Training labels

        Returns:
            self
        """
        # Ensure data is properly scaled to [0, 2*pi] for quantum encoding
        X_scaled = self._scale_features(X_train)

        logger.info(f"Training {'Quantum' if self.use_quantum else 'Classical'} SVM "
                   f"on {len(X_train)} samples...")

        self.model.fit(X_scaled, y_train)
        self.train_data = X_scaled
        self.is_fitted = True

        logger.info("Training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self._scale_features(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Note: QSVC doesn't natively support probabilities.
        For QSVM, we use decision function as a proxy.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self._scale_features(X)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # Use decision function for QSVC
            decision = self.model.decision_function(X_scaled)
            # Convert to pseudo-probabilities via sigmoid
            prob_positive = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - prob_positive, prob_positive])

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features to [0, 2*pi] range for quantum encoding.

        Args:
            X: Input features (assumed to be in [0, 1])

        Returns:
            Scaled features
        """
        # Scale from [0, 1] to [0, 2*pi]
        return X * 2 * np.pi

    def get_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the kernel matrix for visualization/analysis.

        Args:
            X1: First data points
            X2: Second data points (optional)

        Returns:
            Kernel matrix
        """
        if not self.use_quantum:
            from sklearn.metrics.pairwise import rbf_kernel
            return rbf_kernel(X1, X2)

        X1_scaled = self._scale_features(X1)
        X2_scaled = self._scale_features(X2) if X2 is not None else None

        return self.kernel.evaluate(X1_scaled, X2_scaled)


def train_qsvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_features: int = 6,
    feature_map_reps: int = 2,
    entanglement: str = 'full',
    C: float = 1.0,
    use_quantum: bool = True
) -> QuantumSVM:
    """
    Convenience function to train a QSVM.

    Args:
        X_train: Training features
        y_train: Training labels
        num_features: Number of features
        feature_map_reps: Feature map repetitions
        entanglement: Entanglement pattern
        C: Regularization parameter
        use_quantum: Whether to use quantum kernel

    Returns:
        Trained QuantumSVM
    """
    qsvm = QuantumSVM(
        num_features=num_features,
        feature_map_reps=feature_map_reps,
        entanglement=entanglement,
        C=C,
        use_quantum=use_quantum
    )
    qsvm.fit(X_train, y_train)
    return qsvm


def evaluate_qsvm(
    model: QuantumSVM,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate QSVM performance.

    Args:
        model: Trained QSVM
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
    }

    # AUC for binary classification
    if len(np.unique(y_test)) == 2:
        try:
            metrics['auc'] = roc_auc_score(y_test, y_proba[:, 1])
        except ValueError:
            metrics['auc'] = 0.5

    return metrics


def compare_quantum_classical_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_features: int = 6,
    feature_map_reps: int = 2
) -> Dict[str, Dict[str, float]]:
    """
    Compare quantum and classical SVM performance.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        num_features: Number of features
        feature_map_reps: Feature map repetitions

    Returns:
        Dictionary with metrics for both models
    """
    results = {}

    # Train and evaluate classical SVM
    logger.info("Training Classical SVM...")
    classical_svm = train_qsvm(
        X_train, y_train,
        num_features=num_features,
        use_quantum=False
    )
    results['classical_svm'] = evaluate_qsvm(classical_svm, X_test, y_test)

    # Train and evaluate quantum SVM
    if HAS_QISKIT:
        logger.info("Training Quantum SVM...")
        quantum_svm = train_qsvm(
            X_train, y_train,
            num_features=num_features,
            feature_map_reps=feature_map_reps,
            use_quantum=True
        )
        results['quantum_svm'] = evaluate_qsvm(quantum_svm, X_test, y_test)
    else:
        logger.warning("Qiskit not available, skipping quantum SVM")

    return results


class QSVMExperiment:
    """
    Experiment wrapper for QSVM analysis under data scarcity.

    Evaluates QSVM vs classical SVM at different sample sizes
    to demonstrate quantum advantage regimes.
    """

    def __init__(
        self,
        num_features: int = 6,
        feature_map_reps: int = 2,
        entanglement: str = 'full',
        random_state: int = 42
    ):
        self.num_features = num_features
        self.feature_map_reps = feature_map_reps
        self.entanglement = entanglement
        self.random_state = random_state
        self.results = {}

    def run_sample_size_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_sizes: list = [20, 50, 100, 200],
        test_size: float = 0.3,
        n_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Run experiments at different training sample sizes.

        Args:
            X: Full feature matrix
            y: Full labels
            sample_sizes: List of training set sizes to evaluate
            test_size: Fraction of data for testing
            n_runs: Number of repeated runs per size

        Returns:
            Results dictionary with metrics at each sample size
        """
        from sklearn.model_selection import train_test_split

        results = {
            'sample_sizes': sample_sizes,
            'classical': {size: [] for size in sample_sizes},
            'quantum': {size: [] for size in sample_sizes}
        }

        for size in sample_sizes:
            logger.info(f"\nEvaluating with {size} training samples...")

            for run in range(n_runs):
                # Split data
                if size >= len(X):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=self.random_state + run
                    )
                else:
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=self.random_state + run
                    )
                    # Subsample training set
                    indices = np.random.RandomState(self.random_state + run).choice(
                        len(X_temp), size=min(size, len(X_temp)), replace=False
                    )
                    X_train, y_train = X_temp[indices], y_temp[indices]

                # Classical SVM
                classical_svm = train_qsvm(
                    X_train, y_train,
                    num_features=self.num_features,
                    use_quantum=False
                )
                classical_metrics = evaluate_qsvm(classical_svm, X_test, y_test)
                results['classical'][size].append(classical_metrics)

                # Quantum SVM (if available)
                if HAS_QISKIT:
                    quantum_svm = train_qsvm(
                        X_train, y_train,
                        num_features=self.num_features,
                        feature_map_reps=self.feature_map_reps,
                        use_quantum=True
                    )
                    quantum_metrics = evaluate_qsvm(quantum_svm, X_test, y_test)
                    results['quantum'][size].append(quantum_metrics)

        self.results = results
        return results

    def summarize_results(self) -> Dict[str, Any]:
        """Compute mean and std of metrics across runs."""
        if not self.results:
            return {}

        summary = {'sample_sizes': self.results['sample_sizes']}

        for model_type in ['classical', 'quantum']:
            if not self.results.get(model_type):
                continue

            summary[model_type] = {}
            for size, runs in self.results[model_type].items():
                if not runs:
                    continue

                metrics_names = runs[0].keys()
                summary[model_type][size] = {}

                for metric in metrics_names:
                    values = [r[metric] for r in runs]
                    summary[model_type][size][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }

        return summary


if __name__ == "__main__":
    print("Testing QSVM module...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 6

    # Create separable data
    X_class0 = np.random.randn(n_samples // 2, n_features) * 0.5
    X_class1 = np.random.randn(n_samples // 2, n_features) * 0.5 + 1.5
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Normalize to [0, 1]
    X = (X - X.min()) / (X.max() - X.min())

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Compare models
    print("\nComparing Quantum vs Classical SVM:")
    results = compare_quantum_classical_svm(
        X_train, y_train, X_test, y_test,
        num_features=n_features,
        feature_map_reps=2
    )

    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\nQSVM test complete!")
