"""Classify a single Indian currency note image."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import cv2
import joblib
import mahotas
import numpy as np


LABELS = {0: "10", 1: "20", 2: "50", 3: "100", 4: "200", 5: "500"}
IMAGE_SIZE = 320
DEFAULT_MODEL = Path("model/rfclassifier_6class.sav")
DEFAULT_BOVW = Path("model/bovw_codebook_6class.pickle")
DEFAULT_SCALER = Path("model/feature_scaler_6class.joblib")


def fd_hu_moments(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()


def fd_haralick(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)


def fd_histogram(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(histogram, histogram)
    return histogram.flatten()


def feature_extract(image: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(dictionary.shape[0], dtype=np.float32)

    matches = cv2.BFMatcher(cv2.NORM_L2).match(descriptors, dictionary)
    histogram = np.bincount(
        [match.trainIdx for match in matches], minlength=dictionary.shape[0]
    )
    return histogram.astype(np.float32) / max(len(descriptors), 1)


def extract_features(image: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    """Create the feature vector expected by the trained classifier."""
    if image is None or image.size == 0:
        raise ValueError("The input image is empty.")

    height, width = image.shape[:2]
    resize_ratio = IMAGE_SIZE / max(width, height)
    target_size = (max(1, int(resize_ratio * width)), max(1, int(resize_ratio * height)))
    resized = cv2.resize(image, target_size)
    return np.hstack(
        [
            fd_hu_moments(resized),
            fd_haralick(resized),
            fd_histogram(resized),
            feature_extract(resized, dictionary),
        ]
    )


def load_assets(
    model_path: Path = DEFAULT_MODEL,
    bovw_path: Path = DEFAULT_BOVW,
    scaler_path: Path = DEFAULT_SCALER,
) -> tuple[Any, np.ndarray, Any]:
    """Load the classifier and the visual vocabulary, with clear errors."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not bovw_path.is_file():
        raise FileNotFoundError(f"BoVW file does not exist: {bovw_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"Feature scaler does not exist: {scaler_path}. "
            "Rebuild features with bovw.py before predicting."
        )

    model = joblib.load(model_path)
    with bovw_path.open("rb") as file:
        dictionary = pickle.load(file)
    scaler = joblib.load(scaler_path)
    return model, dictionary, scaler


def classify_image(
    image: np.ndarray, model: Any, dictionary: np.ndarray, scaler: Any
) -> tuple[str, float]:
    """Return the predicted denomination and its model probability."""
    features = scaler.transform(extract_features(image, dictionary).reshape(1, -1))
    probabilities = model.predict_proba(features)[0]
    class_index = int(np.argmax(probabilities))
    model_class = int(model.classes_[class_index])
    return LABELS[model_class], float(probabilities[class_index])


def classify_path(
    image_path: Path,
    model_path: Path = DEFAULT_MODEL,
    bovw_path: Path = DEFAULT_BOVW,
    scaler_path: Path = DEFAULT_SCALER,
) -> tuple[str, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")
    model, dictionary, scaler = load_assets(model_path, bovw_path, scaler_path)
    return classify_image(image, model, dictionary, scaler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one currency note image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--bovw", type=Path, default=DEFAULT_BOVW)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    args = parser.parse_args()

    denomination, confidence = classify_path(
        args.image, args.model, args.bovw, args.scaler
    )
    print(f"\nPredicted class: {denomination}")
    print(f"Confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()
