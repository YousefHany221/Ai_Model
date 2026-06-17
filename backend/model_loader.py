"""
NBIS Model Loader
=================
Rebuilds the MobileNetV2 + triplet-head architecture in-process and loads
weights only. Avoids keras.models.load_model() entirely, so the file is
immune to Keras/TF version drift between Colab and local.

Features:
    1. Version-safe model loading (rebuild + load_weights only)
    2. Confidence tiers: HIGH / MEDIUM / LOW match grading
    3. NOT_A_FINGERPRINT detection via simple image-quality heuristics
       (rejects blank, uniform, low-contrast, or off-aspect images before
        FAISS search runs)
    4. Optional NBIS_THRESHOLD_OVERRIDE env var to tighten threshold
       at deploy time without retraining
"""
from __future__ import annotations

import io
import json
import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

# Pillow 10+ compatibility
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS


# ─── Confidence tiers (cosine similarity → human label) ────────────────────
# Tiers are RELATIVE to the operating threshold:
#   HIGH    = "you can trust this match"
#   MEDIUM  = "looks right but worth a second look"
#   LOW     = "above threshold, but barely — possible false accept"
#   NONE    = below threshold → NO_MATCH
HIGH_CONFIDENCE_BOOST   = 0.05   # threshold + 0.05 → HIGH
MEDIUM_CONFIDENCE_BOOST = 0.02   # threshold + 0.02 → MEDIUM


def _grade_score(score: float, threshold: float) -> str:
    if score >= threshold + HIGH_CONFIDENCE_BOOST:
        return "HIGH"
    if score >= threshold + MEDIUM_CONFIDENCE_BOOST:
        return "MEDIUM"
    if score >= threshold:
        return "LOW"
    return "NONE"


# ─── Input validation: is this even a fingerprint? ─────────────────────────
# Cheap heuristics — no separate ML model needed. Catches:
#   • blank / uniform images       (std too low)
#   • extreme aspect ratios        (whole-page photos, panoramas)
#   • too small to be a real scan  (likely thumbnails or icons)
#   • near-monochromatic           (single-tone images)

MIN_DIMENSION     = 80      # pixels — anything smaller isn't a real fingerprint scan
MAX_ASPECT_RATIO  = 2.5     # h/w or w/h — rejects panoramas, narrow strips
MIN_PIXEL_STD     = 10.0    # on 0-255 scale — rejects blank / uniform images
MIN_DYNAMIC_RANGE = 60      # max - min on 0-255 scale — rejects low-contrast


class InvalidInputError(ValueError):
    """Raised when the uploaded image fails sanity checks."""
    def __init__(self, reason: str, detail: dict[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


def _validate_image(img: Image.Image) -> dict[str, Any]:
    """
    Run cheap sanity checks on the uploaded image. Raises InvalidInputError
    if it doesn't look like a plausible fingerprint scan.

    Returns a dict with the measured stats (useful for telemetry / debugging).
    """
    w, h = img.size

    # 1. Size check
    if min(w, h) < MIN_DIMENSION:
        raise InvalidInputError(
            f"Image too small ({w}×{h}). Minimum dimension: {MIN_DIMENSION}px.",
            {"width": w, "height": h},
        )

    # 2. Aspect ratio check
    ratio = max(w, h) / min(w, h)
    if ratio > MAX_ASPECT_RATIO:
        raise InvalidInputError(
            f"Image aspect ratio {ratio:.2f} is not fingerprint-like "
            f"(expected ≤ {MAX_ASPECT_RATIO}).",
            {"width": w, "height": h, "aspect_ratio": round(ratio, 2)},
        )

    # 3. Pixel statistics on grayscale version
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    std  = float(gray.std())
    rng  = float(gray.max() - gray.min())

    if std < MIN_PIXEL_STD:
        raise InvalidInputError(
            f"Image is nearly uniform (std={std:.1f}). "
            f"This doesn't look like a fingerprint scan.",
            {"std": round(std, 2)},
        )

    if rng < MIN_DYNAMIC_RANGE:
        raise InvalidInputError(
            f"Image has very low contrast (range={rng:.0f}). "
            f"Fingerprint ridges should produce strong contrast.",
            {"dynamic_range": round(rng, 2)},
        )

    return {
        "width"        : w,
        "height"       : h,
        "aspect_ratio" : round(ratio, 2),
        "pixel_std"    : round(std, 2),
        "dynamic_range": round(rng, 2),
    }


# ─── Custom layer ──────────────────────────────────────────────────────────
class L2Normalize(keras.layers.Layer):
    """L2-normalize along axis=1. Keeps embeddings on the unit hypersphere."""

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super().get_config()


class NBISSystem:
    """In-memory NBIS identification system. Loaded once at API startup."""

    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.model: keras.Model | None = None
        self.index: faiss.Index | None = None
        self.mapping: list[dict] = []
        self.threshold: float = 0.0
        self.prep_config: dict = {}
        self.img_size: tuple[int, int] = (224, 224)
        self.ready: bool = False

    # ─────────────────────────────────────────────────────────────────────
    def _build_model(self, img_size: tuple[int, int], embedding_dim: int) -> Model:
        """Rebuild the exact architecture from the notebook."""
        base = MobileNetV2(
            input_shape=(*img_size, 3),
            include_top=False,
            weights=None,           # weights come from the checkpoint below
        )
        inputs  = keras.Input(shape=(*img_size, 3), name="fingerprint_input")
        x       = base(inputs, training=False)
        x       = layers.GlobalAveragePooling2D()(x)
        x       = layers.Dense(256, activation="relu")(x)
        x       = layers.BatchNormalization()(x)
        x       = layers.Dropout(0.3)(x)
        x       = layers.Dense(embedding_dim)(x)
        outputs = L2Normalize(name="l2_embedding")(x)
        return Model(inputs, outputs, name="embedding_network")

    # ─────────────────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load all artifacts from disk. Called once at startup."""
        d = self.artifacts_dir
        if not d.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {d}")

        # 1. Preprocessing config
        with open(d / "nbis_preprocessing_config.json") as f:
            self.prep_config = json.load(f)
        self.img_size = tuple(self.prep_config["img_size"])

        # 2. Model — rebuild architecture + load weights only (version-safe)
        self.model = self._build_model(
            self.img_size, int(self.prep_config["embedding_dim"])
        )
        self.model.load_weights(d / "nbis_embedding_model.keras")

        # Warm up the TF graph so the first real request is fast
        dummy = np.zeros((1, *self.img_size, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)

        # 3. FAISS index
        self.index = faiss.read_index(str(d / "nbis_faiss.index"))

        # 4. Mapping
        with open(d / "nbis_index_mapping.pkl", "rb") as f:
            self.mapping = pickle.load(f)

        # 5. Threshold (with optional environment-variable override)
        with open(d / "nbis_threshold.json") as f:
            self.threshold = float(json.load(f)["eer_threshold"])

        # NBIS_THRESHOLD_OVERRIDE lets you tighten the threshold without
        # retraining or editing the artifact file. Useful for production
        # deployment when the EER threshold is too permissive.
        override = os.environ.get("NBIS_THRESHOLD_OVERRIDE")
        if override:
            try:
                new_t = float(override)
                if 0.0 <= new_t <= 1.0:
                    print(f"[NBIS] Threshold override applied: "
                          f"{self.threshold:.4f} → {new_t:.4f}")
                    self.threshold = new_t
            except ValueError:
                pass

        self.ready = True

    # ─────────────────────────────────────────────────────────────────────
    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Bytes → (float32 (H, W, 3) in [0, 1], validation_stats).
        Raises InvalidInputError if the image fails sanity checks.
        """
        img = Image.open(io.BytesIO(image_bytes))

        # Validate ON THE ORIGINAL — before resize destroys the evidence.
        validation_stats = _validate_image(img)

        # Now do the actual preprocessing
        img = img.convert("L")
        img = img.resize(self.img_size, _LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.stack([arr, arr, arr], axis=-1)
        return arr, validation_stats

    # ─────────────────────────────────────────────────────────────────────
    def identify(self, image_bytes: bytes, top_k: int = 5) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("NBIS system not loaded. Call load() first.")

        # Validate + preprocess. If validation fails, return an explicit
        # NOT_A_FINGERPRINT result instead of running the model.
        try:
            arr, val_stats = self.preprocess(image_bytes)
        except InvalidInputError as e:
            return {
                "status"     : "NOT_A_FINGERPRINT",
                "confidence" : 0.0,
                "similarity" : 0.0,
                "threshold"  : self.threshold,
                "subject_id" : None,
                "identity_id": None,
                "message"    : f"Input rejected: {e.reason}",
                "validation" : e.detail,
                "top_k"      : [],
            }

        batch = np.expand_dims(arr, 0).astype(np.float32)

        # Embed
        embedding = self.model.predict(batch, verbose=0)
        emb = embedding.astype(np.float32).copy()
        faiss.normalize_L2(emb)

        # FAISS search
        k = max(1, min(top_k, self.index.ntotal))
        scores, indices = self.index.search(emb, k)
        top_score = float(scores[0][0])
        top_idx   = int(indices[0][0])
        confidence_tier = _grade_score(top_score, self.threshold)

        top_k_results = [
            {
                "rank"       : r + 1,
                "subject_id" : self.mapping[int(indices[0][r])]["subject_id"],
                "finger"     : self.mapping[int(indices[0][r])].get("finger"),
                "hand"       : self.mapping[int(indices[0][r])].get("hand"),
                "identity_id": self.mapping[int(indices[0][r])].get("identity_id"),
                "score"      : float(scores[0][r]),
            }
            for r in range(k)
        ]

        if top_score >= self.threshold:
            m = self.mapping[top_idx]
            return {
                "status"         : "MATCH",
                "confidence_tier": confidence_tier,        # HIGH / MEDIUM / LOW
                "confidence"     : round(top_score * 100, 2),
                "similarity"     : top_score,
                "threshold"      : self.threshold,
                "subject_id"     : m["subject_id"],
                "hand"           : m.get("hand"),
                "finger"         : m.get("finger"),
                "identity_id"    : m.get("identity_id"),
                "parent_id"      : m.get("parent_id"),
                "parent_name"    : m.get("full_name"),
                "parent_phone"   : m.get("phone"),
                "parent_email"   : m.get("email"),
                "city"           : m.get("city"),
                "validation"     : val_stats,
                "top_k"          : top_k_results,
            }
        else:
            return {
                "status"         : "NO_MATCH",
                "confidence_tier": "NONE",
                "confidence"     : round(top_score * 100, 2),
                "similarity"     : top_score,
                "threshold"      : self.threshold,
                "subject_id"     : None,
                "identity_id"    : None,
                "message"        : "Similarity below threshold — identity could not be verified.",
                "validation"     : val_stats,
                "top_k"          : top_k_results,
            }

    # ─────────────────────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        return {
            "ready"        : self.ready,
            "model_loaded" : self.model is not None,
            "index_ready"  : self.index is not None and self.index.ntotal > 0,
            "index_size"   : int(self.index.ntotal) if self.index else 0,
            "embedding_dim": int(self.prep_config.get("embedding_dim", 0)),
            "img_size"     : list(self.img_size),
            "threshold"    : self.threshold,
            "artifacts_dir": str(self.artifacts_dir),
            "tiers"        : {
                "HIGH_at"  : round(self.threshold + HIGH_CONFIDENCE_BOOST, 4),
                "MEDIUM_at": round(self.threshold + MEDIUM_CONFIDENCE_BOOST, 4),
                "LOW_at"   : round(self.threshold, 4),
            },
        }