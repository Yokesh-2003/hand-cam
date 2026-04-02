from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _bce_loss(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


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

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.w1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.w2 + self.b2
        p = _sigmoid(z2)
        cache = {"x": x, "z1": z1, "h1": h1, "p": p}
        return p, cache

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        p, cache = self.forward(x)
        loss = _bce_loss(p, y)

        # dL/dz2 for sigmoid + BCE simplifies to (p - y)
        dz2 = (p - y) / x.shape[0]
        dw2 = cache["h1"].T @ dz2
        db2 = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.w2.T
        dz1 = dh1 * (cache["z1"] > 0).astype(np.float32)
        dw1 = cache["x"].T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        return loss

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p, _ = self.forward(x)
        return p


def _normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """
    lm: (21, 3) in mediapipe normalized coordinates.
    Normalize by:
      - translate so wrist is origin
      - scale by max distance to wrist
    """
    wrist = lm[0:1, :]
    centered = lm - wrist
    d = np.sqrt(np.sum(centered[:, :2] ** 2, axis=1))
    scale = float(np.max(d)) if float(np.max(d)) > 1e-6 else 1.0
    centered /= scale
    return centered.astype(np.float32)


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def _is_pointing_heuristic(lm: np.ndarray) -> bool:
    """
    Same spirit as frontend:
      - index extended: tip farther from wrist than pip
      - middle/ring/pinky not extended
    lm is (21, 3) normalized [0..1] coordinates (not centered).
    """
    wrist = lm[0]

    index_tip = lm[8]
    index_pip = lm[6]

    middle_tip = lm[12]
    middle_pip = lm[10]
    ring_tip = lm[16]
    ring_pip = lm[14]
    pinky_tip = lm[20]
    pinky_pip = lm[18]

    index_extended = _dist2(index_tip, wrist) > _dist2(index_pip, wrist) + 0.02
    middle_extended = _dist2(middle_tip, wrist) > _dist2(middle_pip, wrist) + 0.02
    ring_extended = _dist2(ring_tip, wrist) > _dist2(ring_pip, wrist) + 0.02
    pinky_extended = _dist2(pinky_tip, wrist) > _dist2(pinky_pip, wrist) + 0.02

    return bool(index_extended and (not middle_extended) and (not ring_extended) and (not pinky_extended))


def _make_synthetic_hand(rng: np.random.Generator, pointing: bool) -> np.ndarray:
    """
    Create a rough "hand-like" landmark set (21,3) in [0..1],
    then perturb. Labels come from the 'pointing' parameter.
    """
    # Base pose: wrist at center-bottom; fingers above.
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = np.array([0.5, 0.85, 0.0], dtype=np.float32)  # wrist

    # Thumb (1-4) (not used for label)
    lm[1] = np.array([0.43, 0.80, 0.0], dtype=np.float32)
    lm[2] = np.array([0.40, 0.74, 0.0], dtype=np.float32)
    lm[3] = np.array([0.38, 0.68, 0.0], dtype=np.float32)
    lm[4] = np.array([0.36, 0.63, 0.0], dtype=np.float32)

    # Index (5-8)
    lm[5] = np.array([0.47, 0.73, 0.0], dtype=np.float32)
    lm[6] = np.array([0.47, 0.62, 0.0], dtype=np.float32)
    lm[7] = np.array([0.47, 0.52, 0.0], dtype=np.float32)
    lm[8] = np.array([0.47, 0.44 if pointing else 0.58, 0.0], dtype=np.float32)

    # Middle (9-12)
    lm[9] = np.array([0.52, 0.72, 0.0], dtype=np.float32)
    lm[10] = np.array([0.52, 0.62, 0.0], dtype=np.float32)
    lm[11] = np.array([0.52, 0.54, 0.0], dtype=np.float32)
    lm[12] = np.array([0.52, 0.62 if pointing else 0.46, 0.0], dtype=np.float32)

    # Ring (13-16)
    lm[13] = np.array([0.57, 0.73, 0.0], dtype=np.float32)
    lm[14] = np.array([0.57, 0.64, 0.0], dtype=np.float32)
    lm[15] = np.array([0.57, 0.58, 0.0], dtype=np.float32)
    lm[16] = np.array([0.57, 0.64 if pointing else 0.50, 0.0], dtype=np.float32)

    # Pinky (17-20)
    lm[17] = np.array([0.62, 0.76, 0.0], dtype=np.float32)
    lm[18] = np.array([0.62, 0.68, 0.0], dtype=np.float32)
    lm[19] = np.array([0.62, 0.62, 0.0], dtype=np.float32)
    lm[20] = np.array([0.62, 0.69 if pointing else 0.55, 0.0], dtype=np.float32)

    # Random global hand translation + scale-ish + rotation-ish (in 2D)
    dx, dy = rng.normal(0.0, 0.03, size=(2,))
    s = float(rng.normal(1.0, 0.08))
    ang = float(rng.normal(0.0, 0.10))
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], dtype=np.float32)

    xy = (lm[:, :2] - lm[0:1, :2]) * s
    xy = (xy @ rot.T) + lm[0:1, :2]
    xy += np.array([dx, dy], dtype=np.float32)
    lm[:, :2] = xy

    # Per-point noise
    lm[:, :2] += rng.normal(0.0, 0.01, size=(21, 2)).astype(np.float32)
    lm = np.clip(lm, 0.0, 1.0)
    return lm


def _build_dataset(n: int, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[float] = []

    for _ in range(n):
        y = bool(rng.random() < 0.5)
        lm = _make_synthetic_hand(rng, pointing=y)
        # keep label from heuristic to align with frontend-ish rule (even if synthetic flag drifts)
        y2 = _is_pointing_heuristic(lm)
        feat = _normalize_landmarks(lm).reshape(-1)
        xs.append(feat)
        ys.append(1.0 if y2 else 0.0)

    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.float32).reshape(-1, 1)
    return x, y


def _train_model() -> MLP:
    seed = int(os.environ.get("HAND_MODEL_SEED", "7"))
    model = MLP.init(in_dim=63, hidden=48, seed=seed)

    x_train, y_train = _build_dataset(3200, seed=seed + 1)
    x_val, y_val = _build_dataset(700, seed=seed + 2)

    rng = np.random.default_rng(seed + 3)
    lr = 0.08
    batch = 96
    steps = 520

    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, x_train.shape[0], size=(batch,))
        loss = model.train_step(x_train[idx], y_train[idx], lr=lr)
        if (step + 1) in {1, 40, 120, 240, steps}:
            p = model.predict_proba(x_val)
            val_loss = _bce_loss(p, y_val)
            # keep prints minimal (useful when running in terminal)
            print(f"[train] step={step+1:>4} loss={loss:.4f} val={val_loss:.4f}")

    print(f"[train] done in {time.time() - t0:.2f}s")
    return model


app = FastAPI(title="Hand Pointing Model")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = _train_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/_/backend/health")
def health_prefixed() -> dict[str, Any]:
    return {"ok": True}


def _extract_landmarks(payload: dict[str, Any]) -> np.ndarray | None:
    arr = payload.get("landmarks")
    if not isinstance(arr, list) or len(arr) != 21:
        return None
    pts: list[list[float]] = []
    for p in arr:
        if not isinstance(p, list) or len(p) < 2:
            return None
        x = float(p[0])
        y = float(p[1])
        z = float(p[2]) if len(p) >= 3 else 0.0
        pts.append([x, y, z])
    lm = np.array(pts, dtype=np.float32)
    if lm.shape != (21, 3):
        return None
    return lm


def _predict_pointing_prob(lm: np.ndarray) -> float:
    x = _normalize_landmarks(lm).reshape(1, -1)
    p = float(MODEL.predict_proba(x)[0, 0])
    return max(0.0, min(1.0, p))


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await _ws_loop(websocket)


@app.websocket("/_/backend/ws")
async def ws_prefixed(websocket: WebSocket) -> None:
    await _ws_loop(websocket)


async def _ws_loop(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "bad_json"}))
                continue

            lm = _extract_landmarks(payload if isinstance(payload, dict) else {})
            if lm is None:
                await websocket.send_text(json.dumps({"error": "bad_payload"}))
                continue

            p = _predict_pointing_prob(lm)
            await websocket.send_text(json.dumps({"p_pointing": p}))
    except WebSocketDisconnect:
        return

