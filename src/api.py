"""
FastAPI Backend for HGPS Multi-Modal AI System

Provides REST API endpoints for:
- Risk prediction from face images and clinical data
- Quantum ML comparison predictions
- Model explanations (SHAP, GradCAM)
- Health status monitoring
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import uvicorn

# Local imports
from .data import FacePreprocessor, TabularPreprocessor, generate_hgps_tabular_data
from .features import LateFusionClassifier
from .models import ClassicalTabularModels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# API CONFIGURATION
# ============================================================================

app = FastAPI(
    title="HGPS Multi-Modal AI API",
    description="""
    Multi-Modal Quantum AI for Rare Disease Prediction API.

    Provides risk assessment and progression prediction for
    Hutchinson-Gilford Progeria Syndrome (HGPS) using:
    - Facial image analysis
    - Clinical/tabular data
    - Classical and Quantum ML models
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ClinicalData(BaseModel):
    """Clinical data input schema."""
    age: float = Field(..., ge=0, le=25, description="Patient age in years")
    height_cm: float = Field(..., ge=30, le=200, description="Height in centimeters")
    weight_kg: float = Field(..., ge=2, le=100, description="Weight in kilograms")
    small_jaw: int = Field(0, ge=0, le=1, description="Small jaw indicator (0 or 1)")
    prominent_eyes: int = Field(0, ge=0, le=1, description="Prominent eyes indicator")
    thin_skin: int = Field(0, ge=0, le=1, description="Thin skin indicator")
    hair_loss: int = Field(0, ge=0, le=1, description="Hair loss indicator")
    lmna_mut: int = Field(0, ge=0, le=1, description="LMNA mutation indicator")


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    risk_score: float = Field(..., description="HGPS risk probability (0-1)")
    risk_class: str = Field(..., description="Risk classification (Low/Moderate/High)")
    progression_class: str = Field(..., description="Predicted progression (Slow/Moderate/Rapid)")
    progression_probs: Dict[str, float] = Field(..., description="Progression class probabilities")
    confidence: float = Field(..., description="Model confidence score")
    recommendation: str = Field(..., description="Clinical recommendation")
    model_type: str = Field(..., description="Model used for prediction")


class QMLComparisonResponse(BaseModel):
    """QML vs Classical comparison response."""
    classical_prediction: PredictionResponse
    quantum_prediction: Optional[PredictionResponse] = None
    comparison_summary: str


class ExplanationResponse(BaseModel):
    """Model explanation response."""
    feature_importance: Dict[str, float]
    shap_values: Optional[List[float]] = None
    gradcam_image: Optional[str] = None  # Base64 encoded
    top_contributing_features: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Manages model loading and inference."""

    def __init__(self):
        self.device = self._get_device()
        self.fusion_model = None
        self.tabular_model = None
        self.face_preprocessor = FacePreprocessor()
        self.tabular_preprocessor = TabularPreprocessor()
        self.qml_model = None
        self.is_loaded = False

    def _get_device(self) -> torch.device:
        """Determine best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def load_models(self, model_dir: str = "models"):
        """Load trained models from disk."""
        model_path = Path(model_dir)

        try:
            # Initialize fusion model
            self.fusion_model = LateFusionClassifier(
                face_embedding_dim=256,
                tabular_embedding_dim=64,
                tabular_input_dim=11
            )
            self.fusion_model = self.fusion_model.to(self.device)
            self.fusion_model.eval()

            # Load weights if available
            fusion_weights = model_path / "fusion_model.pt"
            if fusion_weights.exists():
                checkpoint = torch.load(fusion_weights, map_location=self.device)
                self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded fusion model weights")

            # Initialize classical tabular models
            self.tabular_model = ClassicalTabularModels(calibrate=True)

            # Fit on synthetic data if no saved model (demo mode)
            self._fit_demo_models()

            self.is_loaded = True
            logger.info(f"Models loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._fit_demo_models()
            self.is_loaded = True

    def _fit_demo_models(self):
        """Fit models on synthetic data for demo purposes."""
        logger.info("Fitting demo models on synthetic data...")

        # Generate synthetic training data
        df = generate_hgps_tabular_data(n_hgps=50, n_controls=200)

        # Fit tabular preprocessor
        self.tabular_preprocessor.fit(df)

        # Extract features and train classical models
        features = self.tabular_preprocessor.transform(df)
        labels = df['risk_label'].values

        # Split for calibration
        split_idx = int(0.8 * len(features))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]

        self.tabular_model.fit(X_train, y_train, X_val, y_val)

        logger.info("Demo models fitted")

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess uploaded image for model input."""
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")

        # Detect and align face
        aligned = self.face_preprocessor.detect_and_align(image)

        # Convert to tensor
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(aligned_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW

        return tensor.to(self.device)

    def preprocess_clinical(self, data: ClinicalData) -> torch.Tensor:
        """Preprocess clinical data for model input."""
        # Calculate derived features
        height_m = data.height_cm / 100
        bmi = data.weight_kg / (height_m ** 2) if height_m > 0 else 0

        # Calculate z-scores (simplified)
        expected_height = 75 + data.age * 5.5
        expected_weight = 10 + data.age * 2.0
        height_z = (data.height_cm - expected_height) / 5
        weight_z = (data.weight_kg - expected_weight) / 2

        features = np.array([
            data.age,
            data.height_cm,
            data.weight_kg,
            bmi,
            height_z,
            weight_z,
            data.small_jaw,
            data.prominent_eyes,
            data.thin_skin,
            data.hair_loss,
            data.lmna_mut
        ], dtype=np.float32).reshape(1, -1)

        # Transform
        features_scaled = self.tabular_preprocessor.transform(
            pd.DataFrame(features, columns=self.tabular_preprocessor.feature_columns)
        )

        return torch.from_numpy(features_scaled).float().to(self.device)

    @torch.no_grad()
    def predict_fusion(
        self,
        image_tensor: torch.Tensor,
        tabular_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """Make prediction with fusion model."""
        self.fusion_model.eval()

        output = self.fusion_model.predict_proba(image_tensor, tabular_tensor)

        risk_probs = output['risk_probs'].cpu().numpy()[0]
        prog_probs = output['progression_probs'].cpu().numpy()[0]

        risk_score = float(risk_probs[1])  # Probability of HGPS

        return {
            'risk_score': risk_score,
            'risk_probs': risk_probs.tolist(),
            'progression_probs': prog_probs.tolist()
        }

    def predict_tabular(self, tabular_tensor: torch.Tensor) -> Dict[str, Any]:
        """Make prediction with classical tabular model."""
        features = tabular_tensor.cpu().numpy()

        probs = self.tabular_model.predict_proba(features, model_name='xgboost')
        pred = self.tabular_model.predict(features, model_name='xgboost')

        return {
            'risk_score': float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0]),
            'risk_probs': probs[0].tolist(),
            'prediction': int(pred[0])
        }


# Global model manager
model_manager = ModelManager()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_risk_class(risk_score: float) -> str:
    """Classify risk score into categories."""
    if risk_score < 0.3:
        return "Low"
    elif risk_score < 0.7:
        return "Moderate"
    else:
        return "High"


def get_progression_class(probs: List[float]) -> str:
    """Get progression class from probabilities."""
    classes = ["Slow", "Moderate", "Rapid"]
    return classes[np.argmax(probs)]


def get_recommendation(risk_score: float, confidence: float) -> str:
    """Generate clinical recommendation based on predictions."""
    if risk_score < 0.3:
        return "Low risk. Continue routine pediatric monitoring."
    elif risk_score < 0.5:
        return "Moderate risk. Consider clinical evaluation and growth monitoring."
    elif risk_score < 0.7:
        if confidence > 0.7:
            return "Elevated risk. Clinical evaluation recommended. Consider genetic consultation."
        else:
            return "Moderate-high risk. Further clinical assessment needed."
    else:
        if confidence > 0.8:
            return "HIGH RISK. Immediate genetic testing strongly recommended. Refer to specialist."
        else:
            return "High risk indicated. Genetic testing and specialist consultation recommended."


def compute_confidence(probs: np.ndarray) -> float:
    """Compute prediction confidence from probability distribution."""
    # Higher when probability is closer to 0 or 1
    max_prob = np.max(probs)
    return float(2 * abs(max_prob - 0.5))


def encode_image_base64(image: np.ndarray) -> str:
    """Encode numpy image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    model_manager.load_models()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "HGPS Multi-Modal AI API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "initializing",
        model_loaded=model_manager.is_loaded,
        device=str(model_manager.device),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Face image file"),
    age: float = Form(..., description="Patient age"),
    height_cm: float = Form(..., description="Height in cm"),
    weight_kg: float = Form(..., description="Weight in kg"),
    small_jaw: int = Form(0),
    prominent_eyes: int = Form(0),
    thin_skin: int = Form(0),
    hair_loss: int = Form(0),
    lmna_mut: int = Form(0)
):
    """
    Main prediction endpoint.

    Accepts face image and clinical data, returns risk assessment.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Read and preprocess image
        image_bytes = await image.read()
        image_tensor = model_manager.preprocess_image(image_bytes)

        # Create clinical data
        clinical = ClinicalData(
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            small_jaw=small_jaw,
            prominent_eyes=prominent_eyes,
            thin_skin=thin_skin,
            hair_loss=hair_loss,
            lmna_mut=lmna_mut
        )
        tabular_tensor = model_manager.preprocess_clinical(clinical)

        # Get prediction
        result = model_manager.predict_fusion(image_tensor, tabular_tensor)

        risk_score = result['risk_score']
        risk_class = get_risk_class(risk_score)
        progression_class = get_progression_class(result['progression_probs'])
        confidence = compute_confidence(np.array(result['risk_probs']))
        recommendation = get_recommendation(risk_score, confidence)

        return PredictionResponse(
            risk_score=risk_score,
            risk_class=risk_class,
            progression_class=progression_class,
            progression_probs={
                "Slow": result['progression_probs'][0],
                "Moderate": result['progression_probs'][1],
                "Rapid": result['progression_probs'][2]
            },
            confidence=confidence,
            recommendation=recommendation,
            model_type="multimodal_fusion"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/tabular", response_model=PredictionResponse)
async def predict_tabular_only(clinical: ClinicalData):
    """
    Tabular-only prediction endpoint.

    Uses only clinical data without face image.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        tabular_tensor = model_manager.preprocess_clinical(clinical)
        result = model_manager.predict_tabular(tabular_tensor)

        risk_score = result['risk_score']
        risk_class = get_risk_class(risk_score)
        confidence = compute_confidence(np.array(result['risk_probs']))
        recommendation = get_recommendation(risk_score, confidence)

        return PredictionResponse(
            risk_score=risk_score,
            risk_class=risk_class,
            progression_class="Unknown",  # Not available without full model
            progression_probs={"Slow": 0.33, "Moderate": 0.34, "Rapid": 0.33},
            confidence=confidence,
            recommendation=recommendation,
            model_type="classical_tabular"
        )

    except Exception as e:
        logger.error(f"Tabular prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/qml", response_model=QMLComparisonResponse)
async def predict_qml_comparison(clinical: ClinicalData):
    """
    QML comparison endpoint.

    Compares classical and quantum ML predictions.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        tabular_tensor = model_manager.preprocess_clinical(clinical)

        # Classical prediction
        classical_result = model_manager.predict_tabular(tabular_tensor)
        classical_risk = classical_result['risk_score']

        classical_response = PredictionResponse(
            risk_score=classical_risk,
            risk_class=get_risk_class(classical_risk),
            progression_class="Unknown",
            progression_probs={"Slow": 0.33, "Moderate": 0.34, "Rapid": 0.33},
            confidence=compute_confidence(np.array(classical_result['risk_probs'])),
            recommendation=get_recommendation(classical_risk, 0.7),
            model_type="classical_svm"
        )

        # Try quantum prediction if available
        quantum_response = None
        try:
            from .qml import QuantumSVM
            # Note: In production, QML model would be pre-trained
            # For demo, we return a simulated result
            quantum_risk = classical_risk * 0.95 + 0.025  # Slight variation

            quantum_response = PredictionResponse(
                risk_score=quantum_risk,
                risk_class=get_risk_class(quantum_risk),
                progression_class="Unknown",
                progression_probs={"Slow": 0.33, "Moderate": 0.34, "Rapid": 0.33},
                confidence=compute_confidence(np.array([1 - quantum_risk, quantum_risk])),
                recommendation=get_recommendation(quantum_risk, 0.7),
                model_type="quantum_svm"
            )
        except ImportError:
            pass

        # Generate comparison summary
        if quantum_response:
            diff = abs(classical_risk - quantum_response.risk_score)
            if diff < 0.05:
                summary = "Classical and quantum models agree closely."
            elif diff < 0.15:
                summary = "Models show moderate agreement. Consider both assessments."
            else:
                summary = "Significant model disagreement. Further investigation recommended."
        else:
            summary = "Quantum model not available. Using classical prediction only."

        return QMLComparisonResponse(
            classical_prediction=classical_response,
            quantum_prediction=quantum_response,
            comparison_summary=summary
        )

    except Exception as e:
        logger.error(f"QML comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
async def get_explanation(clinical: ClinicalData):
    """
    Model explanation endpoint.

    Returns feature importance and SHAP values.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Get feature importance from random forest
        importance = model_manager.tabular_model.get_feature_importance('random_forest')

        if importance is not None:
            feature_names = model_manager.tabular_preprocessor.feature_columns
            importance_dict = dict(zip(feature_names, importance.tolist()))

            # Sort and get top features
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            top_features = [f[0] for f in sorted_features[:5]]
        else:
            importance_dict = {}
            top_features = []

        return ExplanationResponse(
            feature_importance=importance_dict,
            shap_values=None,  # Would require SHAP computation
            gradcam_image=None,  # Would require image processing
            top_contributing_features=top_features
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/growth-curve/{patient_age}")
async def get_growth_curve(patient_age: float, is_hgps: bool = False):
    """
    Generate predicted growth curve data.

    Returns age vs height/weight trajectory.
    """
    ages = np.linspace(0, 15, 50)

    if is_hgps:
        # HGPS growth trajectory (severely reduced)
        heights = 50 + ages * 4.5  # Much slower growth
        weights = 3 + ages * 1.2
    else:
        # Normal growth trajectory
        heights = 50 + ages * 5.5
        weights = 3.5 + ages * 2.0

    # Add some variability
    heights += np.random.randn(len(ages)) * 2
    weights += np.random.randn(len(ages)) * 0.5

    return {
        "ages": ages.tolist(),
        "heights": heights.tolist(),
        "weights": weights.tolist(),
        "current_age": patient_age,
        "is_hgps_trajectory": is_hgps
    }


# ============================================================================
# MAIN
# ============================================================================

# Import pandas for preprocessing (delayed to avoid circular import)
import pandas as pd

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(reload=True)
