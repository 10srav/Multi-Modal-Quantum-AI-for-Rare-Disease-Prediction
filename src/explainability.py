"""
SHAP-based Explainability Module for HGPS Risk Prediction.

Provides feature importance explanations using SHAP (SHapley Additive exPlanations)
for both classical ML models and the overall prediction system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import SHAP, fall back gracefully if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class SHAPExplainer:
    """SHAP-based explainability for HGPS risk prediction models."""

    FEATURE_NAMES = [
        'age', 'height_cm', 'weight_kg', 'bmi',
        'height_z_score', 'weight_z_score',
        'small_jaw', 'prominent_eyes', 'thin_skin',
        'hair_loss', 'lmna_mut'
    ]

    FEATURE_DESCRIPTIONS = {
        'age': 'Patient age in years',
        'height_cm': 'Height in centimeters',
        'weight_kg': 'Weight in kilograms',
        'bmi': 'Body Mass Index',
        'height_z_score': 'Height deviation from normal (z-score)',
        'weight_z_score': 'Weight deviation from normal (z-score)',
        'small_jaw': 'Micrognathia (small jaw) present',
        'prominent_eyes': 'Prominent eyes phenotype',
        'thin_skin': 'Thin, aged-appearing skin',
        'hair_loss': 'Alopecia (hair loss)',
        'lmna_mut': 'Known LMNA gene mutation'
    }

    def __init__(self, model: Any = None, model_type: str = 'tree'):
        """
        Initialize SHAP explainer.

        Args:
            model: The trained model to explain
            model_type: Type of model ('tree', 'linear', 'kernel')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.background_data = None

        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - explanations will use fallback method")

    def fit(self, X_background: np.ndarray, max_samples: int = 100) -> 'SHAPExplainer':
        """
        Fit the SHAP explainer with background data.

        Args:
            X_background: Background dataset for SHAP calculations
            max_samples: Maximum number of background samples to use

        Returns:
            self
        """
        if not SHAP_AVAILABLE or self.model is None:
            return self

        # Subsample if needed
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            X_background = X_background[indices]

        self.background_data = X_background

        try:
            if self.model_type == 'tree':
                # For tree-based models (XGBoost, Random Forest)
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'linear':
                # For linear models (Logistic Regression, SVM linear)
                self.explainer = shap.LinearExplainer(self.model, X_background)
            else:
                # Kernel SHAP for any model (slower but universal)
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    shap.sample(X_background, min(50, len(X_background)))
                )
            logger.info(f"SHAP explainer fitted with {len(X_background)} background samples")
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            self.explainer = None

        return self

    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Generate SHAP explanations for input samples.

        Args:
            X: Input features to explain (n_samples, n_features)

        Returns:
            Dictionary containing SHAP values and feature importance
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_features = X.shape[1]
        feature_names = self.FEATURE_NAMES[:n_features]

        # Use SHAP if available
        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X)

                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # Multi-class: use positive class
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

                return self._format_explanation(X, shap_values, feature_names)
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}, using fallback")

        # Fallback: use model feature importance or coefficient-based
        return self._fallback_explanation(X, feature_names)

    def _format_explanation(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Format SHAP values into explanation dictionary."""

        # Mean absolute SHAP values for feature importance
        if shap_values.ndim > 1:
            importance = np.abs(shap_values).mean(axis=0)
        else:
            importance = np.abs(shap_values)

        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]

        # Build feature contributions
        contributions = []
        for i, idx in enumerate(sorted_idx):
            feature = feature_names[idx]
            value = float(X[0, idx]) if X.ndim > 1 else float(X[idx])
            shap_val = float(shap_values[0, idx]) if shap_values.ndim > 1 else float(shap_values[idx])

            contributions.append({
                'rank': i + 1,
                'feature': feature,
                'description': self.FEATURE_DESCRIPTIONS.get(feature, feature),
                'value': value,
                'shap_value': shap_val,
                'impact': 'increases' if shap_val > 0 else 'decreases',
                'importance': float(importance[idx])
            })

        # Top risk factors
        risk_factors = [c for c in contributions if c['shap_value'] > 0][:5]
        protective_factors = [c for c in contributions if c['shap_value'] < 0][:3]

        return {
            'method': 'shap',
            'contributions': contributions,
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'feature_importance': {
                feature_names[i]: float(importance[i])
                for i in range(len(feature_names))
            }
        }

    def _fallback_explanation(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Fallback explanation using feature values and domain knowledge."""

        # HGPS-specific feature weights based on medical literature
        hgps_weights = {
            'lmna_mut': 0.95,      # LMNA mutation is definitive
            'small_jaw': 0.75,     # Key phenotypic marker
            'prominent_eyes': 0.70,
            'hair_loss': 0.65,
            'thin_skin': 0.60,
            'height_z_score': -0.50,  # Negative z-score indicates growth failure
            'weight_z_score': -0.45,
            'bmi': -0.30,
            'age': 0.20,
            'height_cm': -0.15,
            'weight_kg': -0.10
        }

        contributions = []
        for i, feature in enumerate(feature_names):
            value = float(X[0, i]) if X.ndim > 1 else float(X[i])
            weight = hgps_weights.get(feature, 0.1)

            # Calculate pseudo-importance based on value and weight
            if feature in ['small_jaw', 'prominent_eyes', 'thin_skin', 'hair_loss', 'lmna_mut']:
                impact = value * weight  # Binary features
            elif feature in ['height_z_score', 'weight_z_score']:
                impact = -value * abs(weight)  # Negative z-scores increase risk
            else:
                impact = value * weight * 0.1  # Continuous features

            contributions.append({
                'rank': 0,
                'feature': feature,
                'description': self.FEATURE_DESCRIPTIONS.get(feature, feature),
                'value': value,
                'shap_value': impact,
                'impact': 'increases' if impact > 0 else 'decreases',
                'importance': abs(impact)
            })

        # Sort and rank
        contributions.sort(key=lambda x: x['importance'], reverse=True)
        for i, c in enumerate(contributions):
            c['rank'] = i + 1

        risk_factors = [c for c in contributions if c['shap_value'] > 0][:5]
        protective_factors = [c for c in contributions if c['shap_value'] < 0][:3]

        return {
            'method': 'domain_knowledge',
            'contributions': contributions,
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'feature_importance': {c['feature']: c['importance'] for c in contributions}
        }

    def get_summary_text(self, explanation: Dict[str, Any]) -> str:
        """Generate human-readable explanation summary."""

        lines = ["## Risk Factor Analysis\n"]

        if explanation['risk_factors']:
            lines.append("### Factors Increasing Risk:")
            for factor in explanation['risk_factors'][:5]:
                lines.append(
                    f"- **{factor['feature']}** ({factor['description']}): "
                    f"value={factor['value']:.2f}, impact={factor['importance']:.3f}"
                )

        if explanation['protective_factors']:
            lines.append("\n### Protective Factors:")
            for factor in explanation['protective_factors'][:3]:
                lines.append(
                    f"- **{factor['feature']}** ({factor['description']}): "
                    f"value={factor['value']:.2f}"
                )

        lines.append(f"\n*Analysis method: {explanation['method']}*")

        return "\n".join(lines)


def explain_prediction(
    model: Any,
    X: np.ndarray,
    X_background: Optional[np.ndarray] = None,
    model_type: str = 'tree'
) -> Dict[str, Any]:
    """
    Convenience function to explain a single prediction.

    Args:
        model: Trained model
        X: Input features to explain
        X_background: Background data for SHAP (optional)
        model_type: Type of model

    Returns:
        Explanation dictionary
    """
    explainer = SHAPExplainer(model, model_type)

    if X_background is not None:
        explainer.fit(X_background)

    return explainer.explain(X)


def get_feature_importance_plot_data(explanation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get data formatted for plotting feature importance.

    Returns dict with 'features', 'importance', 'colors' for bar chart.
    """
    contributions = explanation.get('contributions', [])

    features = [c['feature'] for c in contributions[:10]]
    importance = [c['importance'] for c in contributions[:10]]
    colors = ['#e74c3c' if c['shap_value'] > 0 else '#27ae60' for c in contributions[:10]]

    return {
        'features': features,
        'importance': importance,
        'colors': colors,
        'title': 'Feature Importance for HGPS Risk Prediction'
    }
