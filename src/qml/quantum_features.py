"""
Quantum Feature Maps and Ansatzes for QML

Implements quantum circuits for encoding classical data
and variational ansatzes for quantum classifiers.
"""

import numpy as np
from typing import Optional, List, Callable
import logging

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.circuit.library import ZZFeatureMap as QiskitZZFeatureMap
    from qiskit.circuit.library import RealAmplitudes as QiskitRealAmplitudes
    from qiskit.circuit.library import EfficientSU2, TwoLocal
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    QuantumCircuit = None

logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE MAPS
# ============================================================================

class ZZFeatureMap:
    """
    ZZ Feature Map for encoding classical data into quantum states.

    Implements second-order Pauli-Z evolution for feature encoding
    with entanglement between qubits.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 2,
        entanglement: str = 'full',
        data_map_func: Optional[Callable] = None
    ):
        """
        Args:
            num_qubits: Number of qubits (must match number of features)
            reps: Number of repetitions of the feature map
            entanglement: Entanglement pattern ('full', 'linear', 'circular')
            data_map_func: Function to preprocess data before encoding
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required for quantum feature maps")

        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.data_map_func = data_map_func or (lambda x: x)

        # Use Qiskit's built-in ZZFeatureMap
        self.circuit = QiskitZZFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            entanglement=entanglement,
            data_map_func=self.data_map_func
        )

        logger.info(
            f"ZZFeatureMap: {num_qubits} qubits, {reps} reps, {entanglement} entanglement"
        )

    def get_circuit(self) -> 'QuantumCircuit':
        """Return the quantum circuit."""
        return self.circuit

    @property
    def num_parameters(self) -> int:
        """Number of parameters (features) in the feature map."""
        return self.num_qubits


class PauliFeatureMap:
    """
    Custom Pauli Feature Map with configurable Pauli gates.

    Allows more flexible encoding strategies.
    """

    def __init__(
        self,
        num_qubits: int,
        paulis: List[str] = ['Z', 'ZZ'],
        reps: int = 2,
        entanglement: str = 'full'
    ):
        """
        Args:
            num_qubits: Number of qubits
            paulis: List of Pauli strings to use ('Z', 'ZZ', 'X', 'Y')
            reps: Number of repetitions
            entanglement: Entanglement pattern
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required")

        self.num_qubits = num_qubits
        self.paulis = paulis
        self.reps = reps

        # Create parameter vector for features
        self.x = ParameterVector('x', num_qubits)

        # Build custom circuit
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> 'QuantumCircuit':
        """Build the parameterized quantum circuit."""
        qc = QuantumCircuit(self.num_qubits)

        for _ in range(self.reps):
            # Hadamard layer
            for i in range(self.num_qubits):
                qc.h(i)

            # Single-qubit rotations
            if 'Z' in self.paulis or 'ZZ' in self.paulis:
                for i in range(self.num_qubits):
                    qc.p(2 * self.x[i], i)

            # Two-qubit entanglement
            if 'ZZ' in self.paulis:
                for i in range(self.num_qubits - 1):
                    for j in range(i + 1, self.num_qubits):
                        qc.cx(i, j)
                        qc.p(2 * (np.pi - self.x[i]) * (np.pi - self.x[j]), j)
                        qc.cx(i, j)

        return qc

    def get_circuit(self) -> 'QuantumCircuit':
        """Return the quantum circuit."""
        return self.circuit


# ============================================================================
# ANSATZES (VARIATIONAL FORMS)
# ============================================================================

class RealAmplitudes:
    """
    RealAmplitudes ansatz for variational quantum circuits.

    Consists of alternating layers of Y-rotations and entangling gates.
    Suitable for real-valued problems.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 3,
        entanglement: str = 'full',
        skip_final_rotation: bool = False
    ):
        """
        Args:
            num_qubits: Number of qubits
            reps: Number of repetition layers
            entanglement: Entanglement pattern
            skip_final_rotation: Whether to skip final rotation layer
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required")

        self.num_qubits = num_qubits
        self.reps = reps

        # Use Qiskit's built-in RealAmplitudes
        self.circuit = QiskitRealAmplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation
        )

        self._num_parameters = self.circuit.num_parameters

        logger.info(
            f"RealAmplitudes: {num_qubits} qubits, {reps} reps, "
            f"{self._num_parameters} parameters"
        )

    def get_circuit(self) -> 'QuantumCircuit':
        """Return the quantum circuit."""
        return self.circuit

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters."""
        return self._num_parameters


class EfficientSU2Ansatz:
    """
    EfficientSU2 ansatz for variational circuits.

    More expressive than RealAmplitudes, using full SU(2) rotations.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 3,
        entanglement: str = 'full',
        su2_gates: List[str] = ['ry', 'rz']
    ):
        """
        Args:
            num_qubits: Number of qubits
            reps: Number of repetition layers
            entanglement: Entanglement pattern
            su2_gates: Single-qubit rotation gates to use
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required")

        self.num_qubits = num_qubits
        self.reps = reps

        self.circuit = EfficientSU2(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            su2_gates=su2_gates
        )

        self._num_parameters = self.circuit.num_parameters

        logger.info(
            f"EfficientSU2: {num_qubits} qubits, {reps} reps, "
            f"{self._num_parameters} parameters"
        )

    def get_circuit(self) -> 'QuantumCircuit':
        """Return the quantum circuit."""
        return self.circuit

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters."""
        return self._num_parameters


class HardwareEfficientAnsatz:
    """
    Hardware-efficient ansatz optimized for NISQ devices.

    Uses only native gates available on most quantum hardware.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 2,
        rotation_gates: List[str] = ['ry'],
        entanglement_gates: str = 'cx',
        entanglement: str = 'linear'
    ):
        """
        Args:
            num_qubits: Number of qubits
            reps: Number of layers
            rotation_gates: Single-qubit rotation gates
            entanglement_gates: Two-qubit entangling gate
            entanglement: Entanglement connectivity
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required")

        self.num_qubits = num_qubits
        self.reps = reps

        self.circuit = TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_gates,
            entanglement_blocks=entanglement_gates,
            entanglement=entanglement,
            reps=reps
        )

        self._num_parameters = self.circuit.num_parameters

        logger.info(
            f"HardwareEfficientAnsatz: {num_qubits} qubits, {reps} reps, "
            f"{self._num_parameters} parameters"
        )

    def get_circuit(self) -> 'QuantumCircuit':
        """Return the quantum circuit."""
        return self.circuit

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters."""
        return self._num_parameters


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_feature_map(
    num_qubits: int,
    feature_map_type: str = 'zz',
    reps: int = 2,
    entanglement: str = 'full',
    **kwargs
) -> 'QuantumCircuit':
    """
    Factory function to create quantum feature maps.

    Args:
        num_qubits: Number of qubits/features
        feature_map_type: Type of feature map ('zz', 'pauli')
        reps: Number of repetitions
        entanglement: Entanglement pattern
        **kwargs: Additional arguments for specific feature maps

    Returns:
        Quantum circuit for feature encoding
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit is required for quantum feature maps")

    if feature_map_type == 'zz':
        fm = ZZFeatureMap(num_qubits, reps=reps, entanglement=entanglement)
    elif feature_map_type == 'pauli':
        paulis = kwargs.get('paulis', ['Z', 'ZZ'])
        fm = PauliFeatureMap(num_qubits, paulis=paulis, reps=reps, entanglement=entanglement)
    else:
        raise ValueError(f"Unknown feature map type: {feature_map_type}")

    return fm.get_circuit()


def create_ansatz(
    num_qubits: int,
    ansatz_type: str = 'real_amplitudes',
    reps: int = 3,
    entanglement: str = 'full',
    **kwargs
) -> 'QuantumCircuit':
    """
    Factory function to create variational ansatzes.

    Args:
        num_qubits: Number of qubits
        ansatz_type: Type of ansatz ('real_amplitudes', 'efficient_su2', 'hardware_efficient')
        reps: Number of repetition layers
        entanglement: Entanglement pattern
        **kwargs: Additional arguments

    Returns:
        Quantum circuit ansatz
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit is required for quantum ansatzes")

    if ansatz_type == 'real_amplitudes':
        ans = RealAmplitudes(num_qubits, reps=reps, entanglement=entanglement)
    elif ansatz_type == 'efficient_su2':
        su2_gates = kwargs.get('su2_gates', ['ry', 'rz'])
        ans = EfficientSU2Ansatz(num_qubits, reps=reps, entanglement=entanglement, su2_gates=su2_gates)
    elif ansatz_type == 'hardware_efficient':
        ans = HardwareEfficientAnsatz(num_qubits, reps=reps, entanglement=entanglement)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")

    return ans.get_circuit()


def get_num_ansatz_parameters(
    num_qubits: int,
    ansatz_type: str = 'real_amplitudes',
    reps: int = 3
) -> int:
    """
    Calculate the number of parameters for an ansatz.

    Args:
        num_qubits: Number of qubits
        ansatz_type: Type of ansatz
        reps: Number of layers

    Returns:
        Number of trainable parameters
    """
    if ansatz_type == 'real_amplitudes':
        # (reps + 1) layers of Y-rotations
        return num_qubits * (reps + 1)
    elif ansatz_type == 'efficient_su2':
        # 2 rotations per qubit per layer (reps + 1 layers)
        return 2 * num_qubits * (reps + 1)
    elif ansatz_type == 'hardware_efficient':
        return num_qubits * (reps + 1)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")


if __name__ == "__main__":
    if not HAS_QISKIT:
        print("Qiskit not installed. Install with: pip install qiskit qiskit-machine-learning")
    else:
        print("Testing quantum features...")

        num_qubits = 6

        # Test feature maps
        print("\n1. Testing ZZFeatureMap:")
        zz_fm = create_feature_map(num_qubits, 'zz', reps=2)
        print(f"   Circuit depth: {zz_fm.depth()}")
        print(f"   Num parameters: {zz_fm.num_parameters}")

        # Test ansatzes
        print("\n2. Testing RealAmplitudes ansatz:")
        ra_ans = create_ansatz(num_qubits, 'real_amplitudes', reps=3)
        print(f"   Circuit depth: {ra_ans.depth()}")
        print(f"   Num parameters: {ra_ans.num_parameters}")

        print("\n3. Testing EfficientSU2 ansatz:")
        esu2_ans = create_ansatz(num_qubits, 'efficient_su2', reps=2)
        print(f"   Circuit depth: {esu2_ans.depth()}")
        print(f"   Num parameters: {esu2_ans.num_parameters}")

        print("\n4. Testing HardwareEfficient ansatz:")
        he_ans = create_ansatz(num_qubits, 'hardware_efficient', reps=2)
        print(f"   Circuit depth: {he_ans.depth()}")
        print(f"   Num parameters: {he_ans.num_parameters}")

        print("\nAll quantum feature tests passed!")
