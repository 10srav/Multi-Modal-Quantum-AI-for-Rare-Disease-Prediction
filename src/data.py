"""
Data Module for HGPS Multi-Modal AI System

Handles:
- Kaggle dataset downloading
- Synthetic HGPS data generation
- Face preprocessing and landmark extraction
- Tabular data normalization
- PyTorch Dataset classes for multimodal training
"""

import os
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS - Based on HGPS Medical Literature
# ============================================================================

# HGPS Growth Statistics (from published clinical data)
HGPS_STATS = {
    "height_z_score_mean": -4.0,  # Severe short stature
    "height_z_score_std": 0.8,
    "weight_z_score_mean": -3.5,
    "weight_z_score_std": 0.7,
    "typical_height_range": (80, 115),  # cm at various ages
    "typical_weight_range": (10, 20),   # kg
    "lifespan_mean": 14.6,  # years
}

# Control (normal) growth statistics
CONTROL_STATS = {
    "height_z_score_mean": 0.0,
    "height_z_score_std": 1.0,
    "weight_z_score_mean": 0.0,
    "weight_z_score_std": 1.0,
}

# Craniofacial feature ratios for HGPS
HGPS_FACIAL_FEATURES = {
    "head_to_face_ratio": 1.3,      # Larger head relative to face
    "jaw_reduction_factor": 0.8,    # Micrognathia (small jaw)
    "nose_sharpening": 1.2,         # Pinched nose
    "eye_prominence": 1.15,         # Prominent eyes
}


# ============================================================================
# KAGGLE DATA DOWNLOADING
# ============================================================================

def check_kaggle_setup() -> bool:
    """Check if Kaggle API is properly configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        logger.warning(
            "Kaggle API key not found. Please:\n"
            "1. Go to https://www.kaggle.com/settings\n"
            "2. Create new API token\n"
            "3. Place kaggle.json in ~/.kaggle/"
        )
        return False

    # Set permissions on Unix systems
    if os.name != 'nt':
        os.chmod(kaggle_json, 0o600)

    return True


def download_kaggle_datasets(data_dir: str = "data/raw") -> Dict[str, Path]:
    """
    Download required Kaggle datasets for the project.

    Returns:
        Dictionary mapping dataset names to local paths
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    datasets = {
        "genetic_disorders": "aibuzz/predict-the-genetic-disorders-datasetof-genomes",
        "skin_diseases": "ismailpromus/skin-diseases-image-dataset",
    }

    downloaded = {}

    if not check_kaggle_setup():
        logger.warning("Skipping Kaggle downloads - API not configured")
        return downloaded

    try:
        import kaggle

        for name, dataset_id in datasets.items():
            dest_path = data_path / name
            dest_path.mkdir(exist_ok=True)

            logger.info(f"Downloading {name} from Kaggle...")
            try:
                kaggle.api.dataset_download_files(
                    dataset_id,
                    path=str(dest_path),
                    unzip=True,
                    quiet=False
                )
                downloaded[name] = dest_path
                logger.info(f"Successfully downloaded {name}")
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")

    except ImportError:
        logger.error("Kaggle package not installed. Run: pip install kaggle")

    return downloaded


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_hgps_tabular_data(
    n_hgps: int = 50,
    n_controls: int = 450,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic tabular data mimicking HGPS and control patients.

    Based on published HGPS clinical characteristics:
    - Severe growth retardation (height/weight z-scores -3 to -5)
    - Characteristic craniofacial features
    - LMNA gene mutation presence

    Args:
        n_hgps: Number of HGPS samples to generate
        n_controls: Number of control samples
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with synthetic patient data
    """
    np.random.seed(random_state)

    records = []
    patient_id = 0

    # Generate HGPS samples
    for _ in range(n_hgps):
        age = np.random.uniform(1, 15)

        # HGPS growth patterns - severe growth retardation
        height_z = np.random.normal(
            HGPS_STATS["height_z_score_mean"],
            HGPS_STATS["height_z_score_std"]
        )
        weight_z = np.random.normal(
            HGPS_STATS["weight_z_score_mean"],
            HGPS_STATS["weight_z_score_std"]
        )

        # Convert z-scores to actual measurements using CDC growth charts approximation
        # Normal height at age (simplified linear approximation)
        expected_height = 75 + age * 5.5  # Rough normal trajectory
        expected_weight = 10 + age * 2.0

        height_cm = expected_height + height_z * 5  # 5cm per SD
        weight_kg = max(5, expected_weight + weight_z * 2)  # 2kg per SD

        # BMI calculation
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0

        # Phenotypic features - high probability in HGPS
        small_jaw = int(np.random.random() < 0.92)
        prominent_eyes = int(np.random.random() < 0.88)
        thin_skin = int(np.random.random() < 0.95)
        hair_loss = int(np.random.random() < 0.90)

        # LMNA mutation - high probability in HGPS (some cases pending confirmation)
        lmna_mut = int(np.random.random() < 0.70)

        # Progression based on age and severity
        severity_score = abs(height_z) + abs(weight_z)
        if age < 5:
            progression = 0  # Early - slow apparent progression
        elif severity_score > 8:
            progression = 2  # Rapid
        else:
            progression = 1  # Moderate

        records.append({
            "patient_id": f"HGPS_{patient_id:04d}",
            "age": round(age, 1),
            "height_cm": round(height_cm, 1),
            "weight_kg": round(weight_kg, 1),
            "bmi": round(bmi, 1),
            "height_z_score": round(height_z, 2),
            "weight_z_score": round(weight_z, 2),
            "small_jaw": small_jaw,
            "prominent_eyes": prominent_eyes,
            "thin_skin": thin_skin,
            "hair_loss": hair_loss,
            "lmna_mut": lmna_mut,
            "risk_label": 1,
            "progression_label": progression,
        })
        patient_id += 1

    # Generate control samples
    for _ in range(n_controls):
        age = np.random.uniform(1, 15)

        # Normal growth patterns
        height_z = np.random.normal(
            CONTROL_STATS["height_z_score_mean"],
            CONTROL_STATS["height_z_score_std"]
        )
        weight_z = np.random.normal(
            CONTROL_STATS["weight_z_score_mean"],
            CONTROL_STATS["weight_z_score_std"]
        )

        expected_height = 75 + age * 5.5
        expected_weight = 10 + age * 2.0

        height_cm = expected_height + height_z * 5
        weight_kg = max(5, expected_weight + weight_z * 2)

        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0

        # Phenotypic features - low probability in controls
        small_jaw = int(np.random.random() < 0.05)
        prominent_eyes = int(np.random.random() < 0.08)
        thin_skin = int(np.random.random() < 0.03)
        hair_loss = int(np.random.random() < 0.02)

        # LMNA mutation - very low probability in controls (rare variants)
        lmna_mut = int(np.random.random() < 0.03)

        records.append({
            "patient_id": f"CTRL_{patient_id:04d}",
            "age": round(age, 1),
            "height_cm": round(height_cm, 1),
            "weight_kg": round(weight_kg, 1),
            "bmi": round(bmi, 1),
            "height_z_score": round(height_z, 2),
            "weight_z_score": round(weight_z, 2),
            "small_jaw": small_jaw,
            "prominent_eyes": prominent_eyes,
            "thin_skin": thin_skin,
            "hair_loss": hair_loss,
            "lmna_mut": lmna_mut,
            "risk_label": 0,
            "progression_label": 0,  # N/A for controls
        })
        patient_id += 1

    df = pd.DataFrame(records)

    # Shuffle the data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(f"Generated {n_hgps} HGPS and {n_controls} control samples")
    return df


def generate_synthetic_face_image(
    is_hgps: bool,
    base_image: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (224, 224),
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate a synthetic face image with HGPS-like characteristics.

    If no base image provided, creates a simple geometric face placeholder.
    For real use, apply transforms to actual face images.

    Args:
        is_hgps: Whether to apply HGPS transformations
        base_image: Optional base face image to transform
        image_size: Output image size
        random_state: Random seed

    Returns:
        Synthetic face image as numpy array
    """
    if random_state is not None:
        np.random.seed(random_state)

    if base_image is not None:
        # Apply HGPS-like transformations to existing face
        img = cv2.resize(base_image, image_size)

        if is_hgps:
            # Simulate HGPS facial features through image transforms
            h, w = img.shape[:2]

            # Enlarge head (scale up slightly)
            scale = HGPS_FACIAL_FEATURES["head_to_face_ratio"]
            M_scale = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            img = cv2.warpAffine(img, M_scale, (w, h))

            # Reduce jaw area (compress bottom portion)
            jaw_factor = HGPS_FACIAL_FEATURES["jaw_reduction_factor"]
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([
                [0, 0], [w, 0],
                [w*0.1, h*jaw_factor], [w*0.9, h*jaw_factor]
            ])
            M_jaw = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M_jaw, (w, h))

            # Add slight skin tone adjustment (paler appearance)
            img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)

        return img

    else:
        # Create placeholder geometric face
        img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 240

        h, w = image_size
        center_x, center_y = w // 2, h // 2

        # Face outline
        if is_hgps:
            # HGPS: Larger head, smaller jaw
            face_width = int(w * 0.4)
            face_height = int(h * 0.5)
            jaw_width = int(face_width * 0.6)
        else:
            # Normal proportions
            face_width = int(w * 0.35)
            face_height = int(h * 0.45)
            jaw_width = int(face_width * 0.9)

        # Draw face oval
        cv2.ellipse(
            img, (center_x, center_y),
            (face_width, face_height), 0, 0, 360,
            (220, 200, 180), -1
        )

        # Eyes
        eye_y = center_y - int(face_height * 0.2)
        eye_spacing = int(face_width * 0.4)
        eye_size = 8 if is_hgps else 6  # Prominent eyes in HGPS

        cv2.circle(img, (center_x - eye_spacing, eye_y), eye_size, (50, 50, 50), -1)
        cv2.circle(img, (center_x + eye_spacing, eye_y), eye_size, (50, 50, 50), -1)

        # Nose
        nose_y = center_y + int(face_height * 0.1)
        nose_width = 4 if is_hgps else 6  # Pinched nose in HGPS
        cv2.line(img, (center_x, eye_y + 15), (center_x, nose_y), (180, 160, 140), nose_width)

        # Mouth/jaw region
        mouth_y = center_y + int(face_height * 0.35)
        cv2.ellipse(
            img, (center_x, mouth_y),
            (15, 5), 0, 0, 180,
            (180, 130, 130), 2
        )

        # Add some noise for realism
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img


def generate_synthetic_image_dataset(
    tabular_df: pd.DataFrame,
    output_dir: str = "data/synthetic/images",
    use_base_images: bool = False,
    base_image_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate synthetic face images for all patients in tabular data.

    Args:
        tabular_df: DataFrame with patient data
        output_dir: Directory to save images
        use_base_images: Whether to use real base images for augmentation
        base_image_dir: Directory containing base face images

    Returns:
        Updated DataFrame with image paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base images if available
    base_images = []
    if use_base_images and base_image_dir:
        base_path = Path(base_image_dir)
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            base_images.extend(list(base_path.glob(ext)))

    image_paths = []

    for idx, row in tqdm(tabular_df.iterrows(), total=len(tabular_df), desc="Generating images"):
        is_hgps = row['risk_label'] == 1

        # Select base image if available
        base_img = None
        if base_images:
            base_img_path = np.random.choice(base_images)
            base_img = cv2.imread(str(base_img_path))

        # Generate synthetic face
        face_img = generate_synthetic_face_image(
            is_hgps=is_hgps,
            base_image=base_img,
            random_state=idx
        )

        # Save image
        img_filename = f"{row['patient_id']}.png"
        img_path = output_path / img_filename
        cv2.imwrite(str(img_path), face_img)

        image_paths.append(str(img_path))

    tabular_df = tabular_df.copy()
    tabular_df['image_path'] = image_paths

    logger.info(f"Generated {len(image_paths)} synthetic face images")
    return tabular_df


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

class FacePreprocessor:
    """
    Face detection, alignment, and feature extraction pipeline.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.face_cascade = None
        self._init_detector()

    def _init_detector(self):
        """Initialize face detection model."""
        # Use OpenCV's Haar Cascade as fallback
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Try to initialize MediaPipe for better detection
        try:
            import mediapipe as mp
            # Handle both old and new MediaPipe API versions
            if hasattr(mp, 'solutions'):
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face_mesh = mp.solutions.face_mesh
                self.use_mediapipe = True
                logger.info("Using MediaPipe for face detection")
            else:
                # Newer MediaPipe versions have different API
                self.use_mediapipe = False
                logger.info("MediaPipe version not compatible, using OpenCV Haar Cascade")
        except (ImportError, AttributeError):
            self.use_mediapipe = False
            logger.info("MediaPipe not available, using OpenCV Haar Cascade")

    def detect_and_align(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face, align, and crop to target size.

        Args:
            image: Input BGR image

        Returns:
            Aligned and cropped face image, or None if no face detected
        """
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            # No face detected, return resized original
            return cv2.resize(image, self.target_size)

        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add padding around face
        pad = int(0.2 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        face_crop = image[y1:y2, x1:x2]

        # Resize to target size
        face_resized = cv2.resize(face_crop, self.target_size)

        return face_resized

    def extract_landmarks(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Extract facial landmarks and compute craniofacial ratios.

        Returns dictionary of computed ratios useful for HGPS detection.
        """
        if not self.use_mediapipe:
            # Return placeholder ratios if MediaPipe not available
            return self._compute_basic_ratios(image)

        try:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            ) as face_mesh:

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_image)

                if not results.multi_face_landmarks:
                    return self._compute_basic_ratios(image)

                landmarks = results.multi_face_landmarks[0]
                h, w = image.shape[:2]

                # Extract key landmarks
                # Landmark indices based on MediaPipe face mesh
                nose_tip = landmarks.landmark[1]
                chin = landmarks.landmark[152]
                forehead = landmarks.landmark[10]
                left_eye = landmarks.landmark[33]
                right_eye = landmarks.landmark[263]
                left_ear = landmarks.landmark[234]
                right_ear = landmarks.landmark[454]

                # Compute ratios
                face_height = abs(forehead.y - chin.y) * h
                face_width = abs(left_ear.x - right_ear.x) * w
                eye_distance = abs(left_eye.x - right_eye.x) * w

                # Upper face to lower face ratio (forehead to nose vs nose to chin)
                upper_face = abs(forehead.y - nose_tip.y) * h
                lower_face = abs(nose_tip.y - chin.y) * h

                ratios = {
                    "face_height_width_ratio": face_height / face_width if face_width > 0 else 1.0,
                    "upper_lower_face_ratio": upper_face / lower_face if lower_face > 0 else 1.0,
                    "eye_face_width_ratio": eye_distance / face_width if face_width > 0 else 0.4,
                    "jaw_prominence": lower_face / face_height if face_height > 0 else 0.5,
                }

                return ratios

        except Exception as e:
            logger.warning(f"Landmark extraction failed: {e}")
            return self._compute_basic_ratios(image)

    def _compute_basic_ratios(self, image: np.ndarray) -> Dict[str, float]:
        """Compute basic image-based ratios when landmarks unavailable."""
        h, w = image.shape[:2]

        # Simple heuristic ratios based on image analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Analyze upper vs lower half intensity (proxy for face structure)
        upper_half = gray[:h//2, :]
        lower_half = gray[h//2:, :]

        upper_intensity = np.mean(upper_half)
        lower_intensity = np.mean(lower_half)

        return {
            "face_height_width_ratio": h / w,
            "upper_lower_face_ratio": upper_intensity / lower_intensity if lower_intensity > 0 else 1.0,
            "eye_face_width_ratio": 0.4,  # Default
            "jaw_prominence": 0.5,  # Default
        }


# ============================================================================
# TABULAR DATA PREPROCESSING
# ============================================================================

class TabularPreprocessor:
    """
    Preprocessing pipeline for tabular clinical data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age', 'height_cm', 'weight_kg', 'bmi',
            'height_z_score', 'weight_z_score',
            'small_jaw', 'prominent_eyes', 'thin_skin', 'hair_loss', 'lmna_mut'
        ]
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'TabularPreprocessor':
        """Fit preprocessor on training data."""
        numeric_cols = ['age', 'height_cm', 'weight_kg', 'bmi', 'height_z_score', 'weight_z_score']
        self.scaler.fit(df[numeric_cols])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform tabular data to normalized feature array."""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        df = df.copy()

        # Handle missing values
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Scale numeric features
        numeric_cols = ['age', 'height_cm', 'weight_kg', 'bmi', 'height_z_score', 'weight_z_score']
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Extract feature array
        features = df[self.feature_columns].values.astype(np.float32)

        return features

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)


# ============================================================================
# PYTORCH DATASET CLASSES
# ============================================================================

class HGPSMultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal HGPS data.

    Loads face images and tabular features together with labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tabular_features: np.ndarray,
        image_transform: Optional[Any] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            df: DataFrame with image_path and labels
            tabular_features: Preprocessed tabular feature array
            image_transform: Optional torchvision transforms
            target_size: Image size for resizing
        """
        self.df = df.reset_index(drop=True)
        self.tabular_features = tabular_features
        self.image_transform = image_transform
        self.target_size = target_size
        self.face_preprocessor = FacePreprocessor(target_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load and preprocess image
        image_path = row['image_path']
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = self.face_preprocessor.detect_and_align(image)
        else:
            # Generate placeholder if image doesn't exist
            is_hgps = row['risk_label'] == 1
            image = generate_synthetic_face_image(is_hgps=is_hgps, random_state=idx)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.image_transform:
            image = Image.fromarray(image)
            image = self.image_transform(image)
        else:
            # Default: normalize to [0, 1] and convert to tensor
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # Get tabular features
        tabular = torch.from_numpy(self.tabular_features[idx]).float()

        # Get labels
        risk_label = torch.tensor(row['risk_label'], dtype=torch.long)
        progression_label = torch.tensor(row['progression_label'], dtype=torch.long)

        return {
            'image': image,
            'tabular': tabular,
            'risk_label': risk_label,
            'progression_label': progression_label,
            'patient_id': row['patient_id']
        }


class TabularOnlyDataset(Dataset):
    """Dataset for tabular-only models."""

    def __init__(self, features: np.ndarray, risk_labels: np.ndarray, progression_labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.risk_labels = torch.from_numpy(risk_labels).long()
        self.progression_labels = torch.from_numpy(progression_labels).long()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.risk_labels[idx], self.progression_labels[idx]


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def create_data_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_col: str = 'risk_label'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test splits with stratification.

    Args:
        df: Full dataset
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        stratify_col: Column to stratify on

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col]
    )

    # Second split: separate validation from training
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        random_state=random_state,
        stratify=train_val_df[stratify_col]
    )

    logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tabular_preprocessor: TabularPreprocessor,
    batch_size: int = 16,
    num_workers: int = 0,
    image_transform: Optional[Any] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.

    Args:
        train_df, val_df, test_df: Split DataFrames
        tabular_preprocessor: Fitted TabularPreprocessor
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_transform: Optional image transforms

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Preprocess tabular features
    train_features = tabular_preprocessor.fit_transform(train_df)
    val_features = tabular_preprocessor.transform(val_df)
    test_features = tabular_preprocessor.transform(test_df)

    # Create datasets
    train_dataset = HGPSMultimodalDataset(train_df, train_features, image_transform)
    val_dataset = HGPSMultimodalDataset(val_df, val_features, image_transform)
    test_dataset = HGPSMultimodalDataset(test_df, test_features, image_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_qml_features(
    df: pd.DataFrame,
    tabular_preprocessor: TabularPreprocessor,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract subset of features suitable for QML (6-8 features).

    Args:
        df: Input DataFrame
        tabular_preprocessor: Fitted preprocessor
        feature_names: Specific features to extract

    Returns:
        Tuple of (features, risk_labels, progression_labels)
    """
    if feature_names is None:
        # Default QML features - most discriminative for HGPS
        feature_names = [
            'age', 'height_z_score', 'weight_z_score',
            'small_jaw', 'prominent_eyes', 'thin_skin'
        ]

    # Extract and normalize features
    features = df[feature_names].values.astype(np.float32)

    # Normalize to [0, 1] range for quantum encoding
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    risk_labels = df['risk_label'].values
    progression_labels = df['progression_label'].values

    return features, risk_labels, progression_labels


# ============================================================================
# MAIN DATA PREPARATION FUNCTION
# ============================================================================

def prepare_full_dataset(
    data_dir: str = "data",
    n_hgps: int = 50,
    n_controls: int = 450,
    download_kaggle: bool = False,
    generate_images: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete data preparation pipeline.

    Args:
        data_dir: Base data directory
        n_hgps: Number of HGPS samples
        n_controls: Number of control samples
        download_kaggle: Whether to download Kaggle datasets
        generate_images: Whether to generate synthetic images
        random_state: Random seed

    Returns:
        Dictionary with all prepared data components
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Download Kaggle data if requested
    if download_kaggle:
        download_kaggle_datasets(str(data_path / "raw"))

    # Generate synthetic tabular data
    logger.info("Generating synthetic tabular data...")
    df = generate_hgps_tabular_data(n_hgps, n_controls, random_state)

    # Generate synthetic images if requested
    if generate_images:
        logger.info("Generating synthetic face images...")
        df = generate_synthetic_image_dataset(
            df,
            output_dir=str(data_path / "synthetic" / "images")
        )

    # Save tabular data
    tabular_path = data_path / "synthetic" / "tabular_data.csv"
    tabular_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tabular_path, index=False)
    logger.info(f"Saved tabular data to {tabular_path}")

    # Create splits
    train_df, val_df, test_df = create_data_splits(df, random_state=random_state)

    # Initialize preprocessor
    tabular_preprocessor = TabularPreprocessor()

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tabular_preprocessor
    )

    # Prepare QML features
    train_qml, train_risk, train_prog = get_qml_features(train_df, tabular_preprocessor)
    val_qml, val_risk, val_prog = get_qml_features(val_df, tabular_preprocessor)
    test_qml, test_risk, test_prog = get_qml_features(test_df, tabular_preprocessor)

    return {
        'full_df': df,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'tabular_preprocessor': tabular_preprocessor,
        'qml_data': {
            'train': (train_qml, train_risk, train_prog),
            'val': (val_qml, val_risk, val_prog),
            'test': (test_qml, test_risk, test_prog),
        }
    }


if __name__ == "__main__":
    # Test data generation
    print("Testing data generation pipeline...")

    # Generate small test dataset
    data = prepare_full_dataset(
        data_dir="data",
        n_hgps=20,
        n_controls=80,
        generate_images=True,
        download_kaggle=False
    )

    print(f"\nDataset summary:")
    print(f"Total samples: {len(data['full_df'])}")
    print(f"HGPS samples: {(data['full_df']['risk_label'] == 1).sum()}")
    print(f"Control samples: {(data['full_df']['risk_label'] == 0).sum()}")
    print(f"\nTrain batches: {len(data['train_loader'])}")
    print(f"Val batches: {len(data['val_loader'])}")
    print(f"Test batches: {len(data['test_loader'])}")
    print(f"\nQML feature shape: {data['qml_data']['train'][0].shape}")
