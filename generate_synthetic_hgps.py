"""
Generate synthetic HGPS-like facial images for training.

This creates simulated HGPS facial characteristics by applying transformations
to control images. These are for DEMO/LOCAL USE ONLY and should NOT be
uploaded to public repositories or used for clinical purposes.

HGPS Facial Characteristics Simulated:
- Prominent eyes (enlarged eye region)
- Small/narrow jaw (face narrowing)
- Aged skin appearance (texture changes)
- Hair thinning simulation
- Facial feature prominence
"""

import cv2
import numpy as np
from pathlib import Path
import random
from typing import Tuple, Optional
import shutil

# Paths
CONTROL_IMAGES_DIR = Path("data/images/real_faces")
HGPS_IMAGES_DIR = Path("data/images/synthetic_hgps")
HGPS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def apply_aged_skin_effect(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Apply aged/thin skin appearance effect."""
    # Increase contrast for aged skin look
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE for aged texture
    clahe = cv2.createCLAHE(clipLimit=2.0 + intensity * 2, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Slightly desaturate for aged appearance
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    # Blend with slight yellow tint (aged skin)
    tint = np.full_like(result, [255, 245, 220], dtype=np.uint8)
    result = cv2.addWeighted(result, 0.9, tint, 0.1 * intensity, 0)

    return result


def apply_face_narrowing(image: np.ndarray, factor: float = 0.15) -> np.ndarray:
    """Narrow the lower face to simulate small jaw (micrognathia)."""
    h, w = image.shape[:2]

    # Create transformation for narrowing lower face
    src_pts = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    # Narrow bottom more than top
    offset = int(w * factor * 0.5)
    dst_pts = np.float32([
        [0, 0],
        [w, 0],
        [offset, h],
        [w - offset, h]
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    return result


def apply_eye_prominence(image: np.ndarray, scale: float = 1.1) -> np.ndarray:
    """Enhance eye region to simulate prominent eyes."""
    h, w = image.shape[:2]

    # Eye region (upper third of face)
    eye_y_start = int(h * 0.2)
    eye_y_end = int(h * 0.45)

    # Extract eye region
    eye_region = image[eye_y_start:eye_y_end, :].copy()

    # Slightly enlarge eye region
    new_h = int(eye_region.shape[0] * scale)
    enlarged = cv2.resize(eye_region, (w, new_h))

    # Blend back
    result = image.copy()
    blend_start = max(0, eye_y_start - (new_h - eye_region.shape[0]) // 2)
    blend_end = min(h, blend_start + new_h)
    actual_h = blend_end - blend_start

    result[blend_start:blend_end, :] = cv2.resize(enlarged, (w, actual_h))

    return result


def apply_hair_thinning(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Simulate hair thinning/alopecia effect on forehead region."""
    h, w = image.shape[:2]

    # Top region (hair area)
    hair_region_end = int(h * 0.25)

    # Lighten hair region to simulate thinning
    result = image.copy()
    hair_region = result[:hair_region_end, :].astype(np.float32)

    # Create gradient mask (stronger at top)
    gradient = np.linspace(intensity, 0, hair_region_end)[:, np.newaxis, np.newaxis]
    gradient = np.tile(gradient, (1, w, 3))

    # Lighten towards skin tone
    skin_tone = np.array([220, 190, 170], dtype=np.float32)
    hair_region = hair_region * (1 - gradient * 0.5) + skin_tone * gradient * 0.5

    result[:hair_region_end, :] = np.clip(hair_region, 0, 255).astype(np.uint8)

    return result


def add_vein_visibility(image: np.ndarray, intensity: float = 0.2) -> np.ndarray:
    """Add subtle vein visibility effect for thin skin appearance."""
    h, w = image.shape[:2]

    # Create subtle blue vein-like patterns
    noise = np.random.randn(h // 4, w // 4) * 30
    noise = cv2.resize(noise.astype(np.float32), (w, h))
    noise = cv2.GaussianBlur(noise, (15, 15), 0)

    # Apply only to blue channel (veins)
    result = image.copy().astype(np.float32)
    result[:, :, 2] = result[:, :, 2] + noise * intensity  # Blue channel

    return np.clip(result, 0, 255).astype(np.uint8)


def generate_hgps_image(
    control_image: np.ndarray,
    severity: str = "moderate"
) -> np.ndarray:
    """
    Generate synthetic HGPS-like image from control image.

    Args:
        control_image: Input control face image
        severity: "mild", "moderate", or "severe"

    Returns:
        Transformed image with HGPS-like features
    """
    severity_params = {
        "mild": {"skin": 0.2, "narrow": 0.08, "eye": 1.05, "hair": 0.15, "vein": 0.1},
        "moderate": {"skin": 0.35, "narrow": 0.12, "eye": 1.08, "hair": 0.3, "vein": 0.2},
        "severe": {"skin": 0.5, "narrow": 0.18, "eye": 1.12, "hair": 0.5, "vein": 0.3}
    }

    params = severity_params.get(severity, severity_params["moderate"])

    result = control_image.copy()

    # Apply transformations
    result = apply_face_narrowing(result, params["narrow"])
    result = apply_eye_prominence(result, params["eye"])
    result = apply_aged_skin_effect(result, params["skin"])
    result = apply_hair_thinning(result, params["hair"])
    result = add_vein_visibility(result, params["vein"])

    # Final smoothing
    result = cv2.GaussianBlur(result, (3, 3), 0)

    return result


def main():
    print("=" * 60)
    print("GENERATING SYNTHETIC HGPS IMAGES")
    print("FOR LOCAL DEMO USE ONLY - DO NOT UPLOAD PUBLICLY")
    print("=" * 60)

    # Get control images
    control_images = sorted(list(CONTROL_IMAGES_DIR.glob("*.jpg")))
    print(f"Found {len(control_images)} control images")

    # Select subset for HGPS transformation (younger children)
    young_children = [img for img in control_images if "_age1" in img.stem or "_age2" in img.stem or "_age3" in img.stem]

    if len(young_children) < 10:
        young_children = control_images[:15]  # Use first 15 if not enough young ones

    print(f"Selected {len(young_children)} images for HGPS transformation")

    # Generate HGPS images with varying severity
    severities = ["mild", "moderate", "severe"]
    generated_count = 0

    for i, img_path in enumerate(young_children[:15]):
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Determine severity (distribute across dataset)
        severity = severities[i % 3]

        # Generate HGPS version
        hgps_img = generate_hgps_image(img, severity)

        # Save
        output_name = f"hgps_{severity}_{i:03d}.jpg"
        output_path = HGPS_IMAGES_DIR / output_name

        hgps_img_bgr = cv2.cvtColor(hgps_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), hgps_img_bgr)

        generated_count += 1
        print(f"  Generated: {output_name} ({severity})")

    print(f"\nGenerated {generated_count} synthetic HGPS images")
    print(f"Saved to: {HGPS_IMAGES_DIR}")

    # Create a .gitignore for the synthetic HGPS folder
    gitignore_path = HGPS_IMAGES_DIR / ".gitignore"
    with open(gitignore_path, "w") as f:
        f.write("# Synthetic HGPS images - LOCAL USE ONLY\n")
        f.write("# Do not upload to public repositories\n")
        f.write("*.jpg\n")
        f.write("*.png\n")

    print(f"\nCreated .gitignore to prevent accidental upload")
    print("\n" + "=" * 60)
    print("IMPORTANT: These images are for LOCAL DEMO ONLY")
    print("Do NOT upload to GitHub or any public repository")
    print("=" * 60)


if __name__ == "__main__":
    main()
