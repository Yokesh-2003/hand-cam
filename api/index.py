from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


@dataclass
class MLP:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @staticmethod
    def init(in_dim: int, hidden: int, seed: int = 7) -> "MLP":
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0.0, 0.06, size=(in_dim, hidden)).astype(np.float32)
        b1 = np.zeros((hidden,), dtype=np.float32)
        w2 = rng.normal(0.0, 0.06, size=(hidden, 1)).astype(np.float32)
        b2 = np.zeros((1,), dtype=np.float32)
        return MLP(w1=w1, b1=b1, w2=w2, b2=b2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.w1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.w2 + self.b2
        return _sigmoid(z2)

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        # forward
        z1 = x @ self.w1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.w2 + self.b2
        p = _sigmoid(z2)

        # BCE loss
        eps = 1e-7
        pp = np.clip(p, eps, 1 - eps)
        loss = float(np.mean(-(y * np.log(pp) + (1 - y) * np.log(1 - pp))))

        # backward
        dz2 = (p - y) / x.shape[0]
        dw2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.w2.T
        dz1 = dh1 * (z1 > 0).astype(np.float32)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        return loss

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def _normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    wrist = lm[0:1, :]
    centered = lm - wrist
    d = np.sqrt(np.sum(centered[:, :2] ** 2, axis=1))
    scale = float(np.max(d)) if float(np.max(d)) > 1e-6 else 1.0
    centered /= scale
    return centered.astype(np.float32)


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def _is_pointing_heuristic(lm: np.ndarray) -> bool:
    wrist = lm[0]
    index_tip, index_pip = lm[8], lm[6]
    middle_tip, middle_pip = lm[12], lm[10]
    ring_tip, ring_pip = lm[16], lm[14]
    pinky_tip, pinky_pip = lm[20], lm[18]

    index_extended = _dist2(index_tip, wrist) > _dist2(index_pip, wrist) + 0.02
    middle_extended = _dist2(middle_tip, wrist) > _dist2(middle_pip, wrist) + 0.02
    ring_extended = _dist2(ring_tip, wrist) > _dist2(ring_pip, wrist) + 0.02
    pinky_extended = _dist2(pinky_tip, wrist) > _dist2(pinky_pip, wrist) + 0.02
    return bool(index_extended and (not middle_extended) and (not ring_extended) and (not pinky_extended))


def _make_synthetic_hand(rng: np.random.Generator, pointing: bool) -> np.ndarray:
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = np.array([0.5, 0.85, 0.0], dtype=np.float32)

    lm[1] = np.array([0.43, 0.80, 0.0], dtype=np.float32)
    lm[2] = np.array([0.40, 0.74, 0.0], dtype=np.float32)
    lm[3] = np.array([0.38, 0.68, 0.0], dtype=np.float32)
    lm[4] = np.array([0.36, 0.63, 0.0], dtype=np.float32)

    lm[5] = np.array([0.47, 0.73, 0.0], dtype=np.float32)
    lm[6] = np.array([0.47, 0.62, 0.0], dtype=np.float32)
    lm[7] = np.array([0.47, 0.52, 0.0], dtype=np.float32)
    lm[8] = np.array([0.47, 0.44 if pointing else 0.58, 0.0], dtype=np.float32)

    lm[9] = np.array([0.52, 0.72, 0.0], dtype=np.float32)
    lm[10] = np.array([0.52, 0.62, 0.0], dtype=np.float32)
    lm[11] = np.array([0.52, 0.54, 0.0], dtype=np.float32)
    lm[12] = np.array([0.52, 0.62 if pointing else 0.46, 0.0], dtype=np.float32)

    lm[13] = np.array([0.57, 0.73, 0.0], dtype=np.float32)
    lm[14] = np.array([0.57, 0.64, 0.0], dtype=np.float32)
    lm[15] = np.array([0.57, 0.58, 0.0], dtype=np.float32)
    lm[16] = np.array([0.57, 0.64 if pointing else 0.50, 0.0], dtype=np.float32)

    lm[17] = np.array([0.62, 0.76, 0.0], dtype=np.float32)
    lm[18] = np.array([0.62, 0.68, 0.0], dtype=np.float32)
    lm[19] = np.array([0.62, 0.62, 0.0], dtype=np.float32)
    lm[20] = np.array([0.62, 0.69 if pointing else 0.55, 0.0], dtype=np.float32)

    dx, dy = rng.normal(0.0, 0.03, size=(2,))
    s = float(rng.normal(1.0, 0.08))
    ang = float(rng.normal(0.0, 0.10))
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], dtype=np.float32)

    xy = (lm[:, :2] - lm[0:1, :2]) * s
    xy = (xy @ rot.T) + lm[0:1, :2]
    xy += np.array([dx, dy], dtype=np.float32)
    lm[:, :2] = xy

    lm[:, :2] += rng.normal(0.0, 0.01, size=(21, 2)).astype(np.float32)
    lm = np.clip(lm, 0.0, 1.0)
    return lm


def _build_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for _ in range(n):
        y = bool(rng.random() < 0.5)
        lm = _make_synthetic_hand(rng, pointing=y)
        label = _is_pointing_heuristic(lm)
        xs.append(_normalize_landmarks(lm).reshape(-1))
        ys.append(1.0 if label else 0.0)
    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.float32).reshape(-1, 1)
    return x, y


def _train_model() -> MLP:
    seed = int(os.environ.get("HAND_MODEL_SEED", "7"))
    model = MLP.init(63, 48, seed=seed)
    x_train, y_train = _build_dataset(2400, seed + 1)
    rng = np.random.default_rng(seed + 2)
    lr = 0.08
    batch = 96
    steps = 260
    for _ in range(steps):
        idx = rng.integers(0, x_train.shape[0], size=(batch,))
        model.train_step(x_train[idx], y_train[idx], lr=lr)
    return model


_t0 = time.time()
MODEL = _train_model()
MODEL_INIT_MS = int((time.time() - _t0) * 1000)


class PredictBody(BaseModel):
    landmarks: list[list[float]] = Field(..., min_length=21, max_length=21)


app = FastAPI(title="Hand Pointing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model_init_ms": MODEL_INIT_MS}


@app.post("/api/pointing")
def pointing(body: PredictBody) -> dict[str, Any]:
    pts = []
    for p in body.landmarks:
        x = float(p[0])
        y = float(p[1])
        z = float(p[2]) if len(p) >= 3 else 0.0
        pts.append([x, y, z])
    lm = np.array(pts, dtype=np.float32)
    x = _normalize_landmarks(lm).reshape(1, -1)
    p = float(MODEL.predict_proba(x)[0, 0])
    p = max(0.0, min(1.0, p))
    return {"p_pointing": p}

