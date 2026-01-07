"""
Quantum Neural Network (Variational Quantum Classifier) for HGPS Detection

Implements VQC with parameterized quantum circuits that can be
optimized using gradient-based methods.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable, Any
import logging

logger = logging.getLogger(__name__)

# Qiskit imports with fallback
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
    from qiskit_machine_learning.algorithms import VQC, NeuralNetworkClassifier
    from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
    from qiskit.primitives import Sampler, Estimator
    from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM, L_BFGS_B
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    logger.warning("Qiskit not available. QNN will use classical MLP fallback.")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


class QuantumNeuralNetwork:
    """
    Variational Quantum Classifier (VQC) for binary classification.

    Uses a parameterized quantum circuit with:
    - Feature map: encodes classical data into quantum states
    - Ansatz: variational circuit with trainable parameters
    - Measurement: extracts classical predictions
    """

    def __init__(
        self,
        num_features: int = 6,
        feature_map_type: str = 'zz',
        feature_map_reps: int = 2,
        ansatz_type: str = 'real_amplitudes',
        ansatz_reps: int = 3,
        entanglement: str = 'full',
        optimizer: str = 'cobyla',
        max_iter: int = 100,
        use_quantum: bool = True
    ):
        """
        Args:
            num_features: Number of input features (qubits)
            feature_map_type: Type of quantum feature map
            feature_map_reps: Repetitions in feature map
            ansatz_type: Type of variational ansatz
            ansatz_reps: Repetitions in ansatz
            entanglement: Entanglement pattern
            optimizer: Classical optimizer ('cobyla', 'spsa', 'adam', 'l_bfgs_b')
            max_iter: Maximum optimization iterations
            use_quantum: If False, use classical MLP (for comparison)
        """
        self.num_features = num_features
        self.use_quantum = use_quantum and HAS_QISKIT
        self.max_iter = max_iter

        if self.use_quantum:
            # Create feature map
            self.feature_map = self._create_feature_map(
                feature_map_type, num_features, feature_map_reps, entanglement
            )

            # Create ansatz
            self.ansatz = self._create_ansatz(
                ansatz_type, num_features, ansatz_reps, entanglement
            )

            # Create optimizer
            self.optimizer = self._create_optimizer(optimizer, max_iter)

            # Build the VQC
            self.model = VQC(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=self.optimizer,
                sampler=Sampler()
            )

            self.num_parameters = self.ansatz.num_parameters

            logger.info(
                f"Initialized Quantum Neural Network: {num_features} qubits, "
                f"{self.num_parameters} trainable parameters"
            )
        else:
            # Classical MLP fallback
            self.model = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
            self.num_parameters = None
            logger.info("Initialized Classical MLP (fallback)")

        self.is_fitted = False
        self.training_history = []

    def _create_feature_map(
        self,
        fm_type: str,
        num_qubits: int,
        reps: int,
        entanglement: str
    ) -> 'QuantumCircuit':
        """Create the quantum feature map."""
        if fm_type == 'zz':
            return ZZFeatureMap(
                feature_dimension=num_qubits,
                reps=reps,
                entanglement=entanglement
            )
        else:
            return ZZFeatureMap(
                feature_dimension=num_qubits,
                reps=reps,
                entanglement=entanglement
            )

    def _create_ansatz(
        self,
        ansatz_type: str,
        num_qubits: int,
        reps: int,
        entanglement: str
    ) -> 'QuantumCircuit':
        """Create the variational ansatz."""
        if ansatz_type == 'real_amplitudes':
            return RealAmplitudes(
                num_qubits=num_qubits,
                reps=reps,
                entanglement=entanglement
            )
        elif ansatz_type == 'efficient_su2':
            return EfficientSU2(
                num_qubits=num_qubits,
                reps=reps,
                entanglement=entanglement
            )
        else:
            return RealAmplitudes(
                num_qubits=num_qubits,
                reps=reps,
                entanglement=entanglement
            )

    def _create_optimizer(self, optimizer_name: str, max_iter: int):
        """Create the classical optimizer."""
        optimizers = {
            'cobyla': COBYLA(maxiter=max_iter),
            'spsa': SPSA(maxiter=max_iter),
            'adam': ADAM(maxiter=max_iter),
            'l_bfgs_b': L_BFGS_B(maxiter=max_iter)
        }
        return optimizers.get(optimizer_name.lower(), COBYLA(maxiter=max_iter))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        callback: Optional[Callable] = None
    ) -> 'QuantumNeuralNetwork':
        """
        Train the quantum neural network.

        Args:
            X_train: Training features
            y_train: Training labels
            callback: Optional callback function for monitoring

        Returns:
            self
        """
        # Scale features for quantum encoding
        X_scaled = self._scale_features(X_train)

        logger.info(f"Training {'Quantum' if self.use_quantum else 'Classical'} Neural Network "
                   f"on {len(X_train)} samples...")

        if self.use_quantum and callback:
            # Wrap callback for monitoring
            self.training_history = []

            def wrapped_callback(weights, obj_func_eval):
                self.training_history.append({
                    'iteration': len(self.training_history),
                    'objective': obj_func_eval
                })
                if callback:
                    callback(weights, obj_func_eval)

            # Note: VQC callback signature may differ
            self.model.fit(X_scaled, y_train)
        else:
            self.model.fit(X_scaled, y_train)

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

        For VQC, we compute probabilities from the measurement outcomes.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self._scale_features(X)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For VQC without native probability support
            predictions = self.model.predict(X_scaled)
            # Convert to one-hot style probabilities
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions.astype(int)] = 1.0
            return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features to [0, 2*pi] for quantum encoding."""
        return X * 2 * np.pi

    def get_circuit(self) -> Optional['QuantumCircuit']:
        """Get the full quantum circuit (feature map + ansatz)."""
        if not self.use_quantum:
            return None

        # Compose feature map and ansatz
        full_circuit = QuantumCircuit(self.num_features)
        full_circuit.compose(self.feature_map, inplace=True)
        full_circuit.compose(self.ansatz, inplace=True)
        return full_circuit


def train_qnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_features: int = 6,
    ansatz_reps: int = 3,
    max_iter: int = 100,
    optimizer: str = 'cobyla',
    use_quantum: bool = True
) -> QuantumNeuralNetwork:
    """
    Convenience function to train a QNN.

    Args:
        X_train: Training features
        y_train: Training labels
        num_features: Number of features
        ansatz_reps: Ansatz repetitions
        max_iter: Maximum iterations
        optimizer: Optimizer name
        use_quantum: Whether to use quantum circuit

    Returns:
        Trained QuantumNeuralNetwork
    """
    qnn = QuantumNeuralNetwork(
        num_features=num_features,
        ansatz_reps=ansatz_reps,
        max_iter=max_iter,
        optimizer=optimizer,
        use_quantum=use_quantum
    )
    qnn.fit(X_train, y_train)
    return qnn


def evaluate_qnn(
    model: QuantumNeuralNetwork,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate QNN performance.

    Args:
        model: Trained QNN
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


def compare_quantum_classical_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_features: int = 6,
    ansatz_reps: int = 3,
    max_iter: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Compare quantum and classical neural network performance.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        num_features: Number of features
        ansatz_reps: Ansatz repetitions
        max_iter: Maximum iterations

    Returns:
        Dictionary with metrics for both models
    """
    results = {}

    # Train and evaluate classical MLP
    logger.info("Training Classical MLP...")
    classical_nn = train_qnn(
        X_train, y_train,
        num_features=num_features,
        max_iter=max_iter,
        use_quantum=False
    )
    results['classical_mlp'] = evaluate_qnn(classical_nn, X_test, y_test)

    # Train and evaluate quantum neural network
    if HAS_QISKIT:
        logger.info("Training Quantum Neural Network...")
        quantum_nn = train_qnn(
            X_train, y_train,
            num_features=num_features,
            ansatz_reps=ansatz_reps,
            max_iter=max_iter,
            use_quantum=True
        )
        results['quantum_nn'] = evaluate_qnn(quantum_nn, X_test, y_test)
    else:
        logger.warning("Qiskit not available, skipping quantum neural network")

    return results


class QNNExperiment:
    """
    Experiment wrapper for QNN analysis under various conditions.

    Studies QNN performance vs classical models with different:
    - Sample sizes (data scarcity)
    - Circuit depths (ansatz reps)
    - Number of features (qubits)
    """

    def __init__(
        self,
        num_features: int = 6,
        ansatz_reps: int = 3,
        max_iter: int = 100,
        random_state: int = 42
    ):
        self.num_features = num_features
        self.ansatz_reps = ansatz_reps
        self.max_iter = max_iter
        self.random_state = random_state
        self.results = {}

    def run_depth_experiment(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        depths: List[int] = [1, 2, 3, 4, 5]
    ) -> Dict[str, Any]:
        """
        Evaluate QNN at different circuit depths.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            depths: List of ansatz repetitions to test

        Returns:
            Results dictionary
        """
        results = {
            'depths': depths,
            'quantum': {},
            'classical': None
        }

        # Classical baseline (depth-independent)
        logger.info("Training classical baseline...")
        classical_nn = train_qnn(
            X_train, y_train,
            num_features=self.num_features,
            max_iter=self.max_iter,
            use_quantum=False
        )
        results['classical'] = evaluate_qnn(classical_nn, X_test, y_test)

        # Quantum at different depths
        if HAS_QISKIT:
            for depth in depths:
                logger.info(f"Training QNN with depth={depth}...")
                quantum_nn = train_qnn(
                    X_train, y_train,
                    num_features=self.num_features,
                    ansatz_reps=depth,
                    max_iter=self.max_iter,
                    use_quantum=True
                )
                results['quantum'][depth] = evaluate_qnn(quantum_nn, X_test, y_test)

        self.results = results
        return results

    def run_sample_size_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_sizes: List[int] = [20, 50, 100, 200],
        test_size: float = 0.3,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate QNN at different training sample sizes.

        This is key for demonstrating potential quantum advantage
        in data-scarce scenarios.
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
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state + run
                )

                # Subsample training set
                if size < len(X_temp):
                    indices = np.random.RandomState(self.random_state + run).choice(
                        len(X_temp), size=size, replace=False
                    )
                    X_train, y_train = X_temp[indices], y_temp[indices]
                else:
                    X_train, y_train = X_temp, y_temp

                # Classical MLP
                classical_nn = train_qnn(
                    X_train, y_train,
                    num_features=self.num_features,
                    max_iter=self.max_iter,
                    use_quantum=False
                )
                results['classical'][size].append(
                    evaluate_qnn(classical_nn, X_test, y_test)
                )

                # Quantum NN
                if HAS_QISKIT:
                    quantum_nn = train_qnn(
                        X_train, y_train,
                        num_features=self.num_features,
                        ansatz_reps=self.ansatz_reps,
                        max_iter=self.max_iter,
                        use_quantum=True
                    )
                    results['quantum'][size].append(
                        evaluate_qnn(quantum_nn, X_test, y_test)
                    )

        self.results = results
        return results

    def summarize_results(self) -> Dict[str, Any]:
        """Compute summary statistics from experiments."""
        if not self.results:
            return {}

        summary = {}

        if 'sample_sizes' in self.results:
            summary['sample_sizes'] = self.results['sample_sizes']

            for model_type in ['classical', 'quantum']:
                if isinstance(self.results.get(model_type), dict):
                    summary[model_type] = {}
                    for size, runs in self.results[model_type].items():
                        if not runs:
                            continue
                        summary[model_type][size] = {}
                        for metric in runs[0].keys():
                            values = [r[metric] for r in runs]
                            summary[model_type][size][metric] = {
                                'mean': np.mean(values),
                                'std': np.std(values)
                            }

        elif 'depths' in self.results:
            summary['depths'] = self.results['depths']
            summary['classical'] = self.results.get('classical', {})
            summary['quantum'] = self.results.get('quantum', {})

        return summary


class HybridClassicalQuantumModel:
    """
    Hybrid model combining classical preprocessing with quantum classification.

    Uses classical neural network for feature extraction/reduction,
    then quantum circuit for final classification.
    """

    def __init__(
        self,
        input_dim: int,
        quantum_dim: int = 6,
        ansatz_reps: int = 3,
        max_iter: int = 100
    ):
        """
        Args:
            input_dim: Original feature dimension
            quantum_dim: Reduced dimension for quantum circuit
            ansatz_reps: Ansatz repetitions
            max_iter: Maximum optimization iterations
        """
        self.input_dim = input_dim
        self.quantum_dim = quantum_dim

        # Classical dimensionality reduction
        from sklearn.decomposition import PCA
        self.reducer = PCA(n_components=quantum_dim)

        # Quantum classifier
        self.qnn = QuantumNeuralNetwork(
            num_features=quantum_dim,
            ansatz_reps=ansatz_reps,
            max_iter=max_iter,
            use_quantum=HAS_QISKIT
        )

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HybridClassicalQuantumModel':
        """Train the hybrid model."""
        # Reduce dimensions
        X_reduced = self.reducer.fit_transform(X)

        # Normalize to [0, 1]
        X_normalized = (X_reduced - X_reduced.min(axis=0)) / (
            X_reduced.max(axis=0) - X_reduced.min(axis=0) + 1e-8
        )

        # Train quantum classifier
        self.qnn.fit(X_normalized, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_reduced = self.reducer.transform(X)
        X_normalized = (X_reduced - X_reduced.min(axis=0)) / (
            X_reduced.max(axis=0) - X_reduced.min(axis=0) + 1e-8
        )
        return self.qnn.predict(X_normalized)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_reduced = self.reducer.transform(X)
        X_normalized = (X_reduced - X_reduced.min(axis=0)) / (
            X_reduced.max(axis=0) - X_reduced.min(axis=0) + 1e-8
        )
        return self.qnn.predict_proba(X_normalized)


if __name__ == "__main__":
    print("Testing QNN module...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 6

    # Create XOR-like data (harder to separate)
    X = np.random.randn(n_samples, n_features)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)

    # Normalize to [0, 1]
    X = (X - X.min()) / (X.max() - X.min())

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Compare models
    print("\nComparing Quantum vs Classical Neural Networks:")
    results = compare_quantum_classical_nn(
        X_train, y_train, X_test, y_test,
        num_features=n_features,
        ansatz_reps=3,
        max_iter=50  # Reduced for testing
    )

    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Test hybrid model
    print("\n\nTesting Hybrid Classical-Quantum Model:")
    hybrid = HybridClassicalQuantumModel(
        input_dim=n_features,
        quantum_dim=4,
        ansatz_reps=2,
        max_iter=30
    )
    hybrid.fit(X_train, y_train)
    hybrid_acc = accuracy_score(y_test, hybrid.predict(X_test))
    print(f"Hybrid model accuracy: {hybrid_acc:.4f}")

    print("\nQNN tests complete!")
