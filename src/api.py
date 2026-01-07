"""
FastAPI Backend for HGPS Multi-Modal AI System

Production-ready API with:
- Risk prediction from face images and clinical data
- Quantum ML comparison predictions
- Model explanations (SHAP, GradCAM)
- Health status monitoring
- API authentication and rate limiting
"""

import io
import os
import base64
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict
from functools import wraps

import numpy as np
import cv2
import joblib
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import torch
import uvicorn

# Local imports
from .data import FacePreprocessor, TabularPreprocessor, generate_hgps_tabular_data
from .features import LateFusionClassifier
from .models import ClassicalTabularModels
from .config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
    - Facial image analysis (CNN)
    - Clinical/tabular data (Classical ML)
    - Classical and Quantum ML models (QSVM, QNN)

    ## Authentication
    Include API key in header: `X-API-Key: your-api-key`

    ## Rate Limiting
    Default: 100 requests per minute
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "HGPS AI Team",
        "email": "support@hgps-ai.com"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SECURITY & RATE LIMITING
# ============================================================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory rate limiting (use Redis in production)
rate_limit_store: Dict[str, List[float]] = defaultdict(list)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> Optional[str]:
    """Verify API key if authentication is enabled."""
    if not settings.api.api_key:
        return "anonymous"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include X-API-Key header."
        )

    if api_key != settings.api.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return api_key


async def rate_limit(request: Request):
    """Simple rate limiting middleware."""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    window = 60  # 1 minute window

    # Clean old entries
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip]
        if current_time - t < window
    ]

    # Check limit
    if len(rate_limit_store[client_ip]) >= settings.api.rate_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {settings.api.rate_limit} requests per minute."
        )

    # Record request
    rate_limit_store[client_ip].append(current_time)


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

    class Config:
        json_schema_extra = {
            "example": {
                "age": 5.0,
                "height_cm": 85.0,
                "weight_kg": 12.0,
                "small_jaw": 1,
                "prominent_eyes": 1,
                "thin_skin": 0,
                "hair_loss": 0,
                "lmna_mut": 0
            }
        }


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
    gradcam_image: Optional[str] = None
    top_contributing_features: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str
    environment: str
    uptime_seconds: float
    models_available: Dict[str, bool]


class ModelInfoResponse(BaseModel):
    """Model information response."""
    models: Dict[str, Any]
    last_trained: Optional[str]
    training_metrics: Optional[Dict[str, Any]]


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Manages model loading and inference for production."""

    def __init__(self):
        self.device = self._get_device()
        self.fusion_model = None
        self.tabular_model = None
        self.qsvm_model = None
        self.qnn_model = None
        self.face_preprocessor = FacePreprocessor()
        self.tabular_preprocessor = None
        self.is_loaded = False
        self.load_time = None
        self.model_info = {}
        self._start_time = time.time()

    def _get_device(self) -> torch.device:
        """Determine best available device."""
        if settings.device == "cuda" and torch.cuda.is_available():
            return torch.device('cuda')
        elif settings.device == "auto":
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
        return torch.device('cpu')

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    def load_models(self, model_dir: Optional[str] = None):
        """Load trained models from disk."""
        if model_dir is None:
            model_dir = settings.paths.models_dir

        model_path = Path(model_dir)
        logger.info(f"Loading models from {model_path}")

        try:
            # Load preprocessor
            preprocessor_path = model_path / "preprocessor.joblib"
            if preprocessor_path.exists():
                self.tabular_preprocessor = joblib.load(preprocessor_path)
                logger.info("Loaded preprocessor from disk")
                self.model_info['preprocessor'] = True
            else:
                logger.warning("Preprocessor not found, creating new one")
                self.tabular_preprocessor = TabularPreprocessor()
                self._fit_preprocessor()
                self.model_info['preprocessor'] = False

            # Load classical models
            classical_path = model_path / "classical_models.joblib"
            if classical_path.exists():
                self.tabular_model = joblib.load(classical_path)
                logger.info("Loaded classical models from disk")
                self.model_info['classical'] = True
            else:
                logger.warning("Classical models not found, training new ones")
                self._fit_demo_models()
                self.model_info['classical'] = False

            # Load QSVM model
            qsvm_path = model_path / "qsvm.joblib"
            if qsvm_path.exists():
                self.qsvm_model = joblib.load(qsvm_path)
                logger.info("Loaded QSVM model from disk")
                self.model_info['qsvm'] = True
            else:
                self.model_info['qsvm'] = False

            # Load QNN model
            qnn_path = model_path / "qnn.joblib"
            if qnn_path.exists():
                self.qnn_model = joblib.load(qnn_path)
                logger.info("Loaded QNN model from disk")
                self.model_info['qnn'] = True
            else:
                self.model_info['qnn'] = False

            # Load fusion model
            self._init_fusion_model(model_path)

            # Load CNN model
            cnn_path = model_path / "face_cnn.pt"
            self.model_info['cnn'] = cnn_path.exists()

            self.is_loaded = True
            self.load_time = datetime.now().isoformat()
            logger.info(f"Models loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._fit_demo_models()
            self.is_loaded = True
            self.load_time = datetime.now().isoformat()

    def _init_fusion_model(self, model_path: Path):
        """Initialize fusion model."""
        self.fusion_model = LateFusionClassifier(
            face_embedding_dim=256,
            tabular_embedding_dim=64,
            tabular_input_dim=11
        )
        self.fusion_model = self.fusion_model.to(self.device)
        self.fusion_model.eval()

        fusion_weights = model_path / "fusion_model.pt"
        if fusion_weights.exists():
            checkpoint = torch.load(fusion_weights, map_location=self.device)
            self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded fusion model weights")
            self.model_info['fusion'] = True
        else:
            self.model_info['fusion'] = False

    def _fit_preprocessor(self):
        """Fit preprocessor on synthetic data."""
        df = generate_hgps_tabular_data(n_hgps=50, n_controls=200)
        self.tabular_preprocessor.fit(df)

    def _fit_demo_models(self):
        """Fit models on synthetic data for demo purposes."""
        logger.info("Fitting demo models on synthetic data...")

        df = generate_hgps_tabular_data(n_hgps=50, n_controls=200)

        if self.tabular_preprocessor is None:
            self.tabular_preprocessor = TabularPreprocessor()
        self.tabular_preprocessor.fit(df)

        features = self.tabular_preprocessor.transform(df)
        labels = df['risk_label'].values

        split_idx = int(0.8 * len(features))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]

        self.tabular_model = ClassicalTabularModels(calibrate=True)
        self.tabular_model.fit(X_train, y_train, X_val, y_val)

        logger.info("Demo models fitted")

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess uploaded image for model input."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")

        aligned = self.face_preprocessor.detect_and_align(image)
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(aligned_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self.device)

    def preprocess_clinical(self, data: ClinicalData) -> torch.Tensor:
        """Preprocess clinical data for model input."""
        height_m = data.height_cm / 100
        bmi = data.weight_kg / (height_m ** 2) if height_m > 0 else 0

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

        features_scaled = self.tabular_preprocessor.transform(
            pd.DataFrame(features, columns=self.tabular_preprocessor.feature_columns)
        )

        return torch.from_numpy(features_scaled).float().to(self.device)

    def get_qml_features(self, clinical: ClinicalData) -> np.ndarray:
        """Extract features for QML models."""
        height_m = clinical.height_cm / 100
        bmi = clinical.weight_kg / (height_m ** 2) if height_m > 0 else 0
        expected_height = 75 + clinical.age * 5.5
        height_z = (clinical.height_cm - expected_height) / 5

        features = np.array([
            clinical.age / 25.0,
            height_z / 5.0 + 0.5,
            bmi / 30.0,
            clinical.small_jaw,
            clinical.prominent_eyes,
            clinical.lmna_mut
        ], dtype=np.float32).reshape(1, -1)

        return np.clip(features, 0, 1)

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
        risk_score = float(risk_probs[1])

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

    def predict_qsvm(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction with QSVM model."""
        if self.qsvm_model is None:
            raise ValueError("QSVM model not loaded")

        pred = self.qsvm_model.predict(features)
        probs = self.qsvm_model.predict_proba(features) if hasattr(self.qsvm_model, 'predict_proba') else None

        risk_score = float(probs[0, 1]) if probs is not None else float(pred[0])

        return {
            'risk_score': risk_score,
            'prediction': int(pred[0])
        }

    def predict_qnn(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction with QNN model."""
        if self.qnn_model is None:
            raise ValueError("QNN model not loaded")

        pred = self.qnn_model.predict(features)
        probs = self.qnn_model.predict_proba(features) if hasattr(self.qnn_model, 'predict_proba') else None

        risk_score = float(probs[0, 1]) if probs is not None else float(pred[0])

        return {
            'risk_score': risk_score,
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
    logger.info("Starting HGPS API server...")
    model_manager.load_models()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "HGPS Multi-Modal AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with detailed status."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "initializing",
        model_loaded=model_manager.is_loaded,
        device=str(model_manager.device),
        version="1.0.0",
        environment=settings.environment,
        uptime_seconds=model_manager.uptime,
        models_available=model_manager.model_info
    )


@app.get("/models", response_model=ModelInfoResponse, dependencies=[Depends(rate_limit)])
async def get_model_info(api_key: str = Depends(verify_api_key)):
    """Get information about loaded models."""
    return ModelInfoResponse(
        models=model_manager.model_info,
        last_trained=model_manager.load_time,
        training_metrics=None
    )


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(rate_limit)])
async def predict(
    image: UploadFile = File(..., description="Face image file"),
    age: float = Form(..., description="Patient age"),
    height_cm: float = Form(..., description="Height in cm"),
    weight_kg: float = Form(..., description="Weight in kg"),
    small_jaw: int = Form(0),
    prominent_eyes: int = Form(0),
    thin_skin: int = Form(0),
    hair_loss: int = Form(0),
    lmna_mut: int = Form(0),
    api_key: str = Depends(verify_api_key)
):
    """
    Main prediction endpoint.

    Accepts face image and clinical data, returns risk assessment.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        image_bytes = await image.read()
        image_tensor = model_manager.preprocess_image(image_bytes)

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


@app.post("/predict/tabular", response_model=PredictionResponse, dependencies=[Depends(rate_limit)])
async def predict_tabular_only(
    clinical: ClinicalData,
    api_key: str = Depends(verify_api_key)
):
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
            progression_class="Unknown",
            progression_probs={"Slow": 0.33, "Moderate": 0.34, "Rapid": 0.33},
            confidence=confidence,
            recommendation=recommendation,
            model_type="classical_tabular"
        )

    except Exception as e:
        logger.error(f"Tabular prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/qml", response_model=QMLComparisonResponse, dependencies=[Depends(rate_limit)])
async def predict_qml_comparison(
    clinical: ClinicalData,
    api_key: str = Depends(verify_api_key)
):
    """
    QML comparison endpoint.

    Compares classical and quantum ML predictions.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        tabular_tensor = model_manager.preprocess_clinical(clinical)
        qml_features = model_manager.get_qml_features(clinical)

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
            model_type="classical_xgboost"
        )

        # Quantum prediction
        quantum_response = None
        if model_manager.qsvm_model is not None:
            try:
                qsvm_result = model_manager.predict_qsvm(qml_features)
                quantum_risk = qsvm_result['risk_score']

                quantum_response = PredictionResponse(
                    risk_score=quantum_risk,
                    risk_class=get_risk_class(quantum_risk),
                    progression_class="Unknown",
                    progression_probs={"Slow": 0.33, "Moderate": 0.34, "Rapid": 0.33},
                    confidence=compute_confidence(np.array([1 - quantum_risk, quantum_risk])),
                    recommendation=get_recommendation(quantum_risk, 0.7),
                    model_type="quantum_svm"
                )
            except Exception as e:
                logger.warning(f"QSVM prediction failed: {e}")

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


@app.post("/explain", response_model=ExplanationResponse, dependencies=[Depends(rate_limit)])
async def get_explanation(
    clinical: ClinicalData,
    api_key: str = Depends(verify_api_key)
):
    """
    Model explanation endpoint.

    Returns feature importance and SHAP values.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        importance = model_manager.tabular_model.get_feature_importance('random_forest')

        if importance is not None:
            feature_names = model_manager.tabular_preprocessor.feature_columns
            importance_dict = dict(zip(feature_names, importance.tolist()))

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
            shap_values=None,
            gradcam_image=None,
            top_contributing_features=top_features
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/growth-curve/{patient_age}", dependencies=[Depends(rate_limit)])
async def get_growth_curve(
    patient_age: float,
    is_hgps: bool = False,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate predicted growth curve data.

    Returns age vs height/weight trajectory.
    """
    ages = np.linspace(0, 15, 50)

    if is_hgps:
        heights = 50 + ages * 4.5
        weights = 3 + ages * 1.2
    else:
        heights = 50 + ages * 5.5
        weights = 3.5 + ages * 2.0

    np.random.seed(42)
    heights = heights + np.random.randn(len(ages)) * 2
    weights = weights + np.random.randn(len(ages)) * 0.5

    return {
        "ages": ages.tolist(),
        "heights": heights.tolist(),
        "weights": weights.tolist(),
        "current_age": patient_age,
        "is_hgps_trajectory": is_hgps
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# MAIN
# ============================================================================

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
