"""FastAPI inference service for fall prediction.

Run from the project root (9-channel, default):
    python -m uvicorn service.main:app --reload

Run serving the 6-channel phone-deployable model instead:
    MODEL_CHANNELS=6 python -m uvicorn service.main:app --reload --host 0.0.0.0
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.schemas import ExplainResponse, HealthResponse, NUM_CHANNELS, PredictResponse, WindowRequest
from src.data import load_norm_stats
from src.model import FallDetectorCNN, get_device
from src.saliency import grad_x_input_saliency

_suffix = "_6ch" if NUM_CHANNELS == 6 else ""
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", f"best_fall_detector_cnn{_suffix}.pt")
NORM_STATS_PATH = os.environ.get("NORM_STATS_PATH", f"norm_stats{_suffix}.npz")
FALL_THRESHOLD = 0.90

state: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = get_device()
    model = FallDetectorCNN(in_channels=NUM_CHANNELS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()

    mean, std = load_norm_stats(NORM_STATS_PATH)

    state["device"] = device
    state["model"] = model
    state["mean"] = mean
    state["std"] = std
    yield
    state.clear()


app = FastAPI(title="Fall Prediction Service", lifespan=lifespan)

# Permissive by design: this service is meant to be reached from a phone on the
# same local network (see phone_client/), a different origin than the service
# itself, with no login/session to protect. Not appropriate if ever exposed
# beyond a trusted local network.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _to_normalized_tensor(window: List[List[float]]) -> torch.Tensor:
    x = np.array(window, dtype=np.float32)  # (200, NUM_CHANNELS)
    mean, std = state["mean"], state["std"]
    x = (x - mean.squeeze(0)) / std.squeeze(0)
    return torch.from_numpy(x).unsqueeze(0).permute(0, 2, 1).float().to(state["device"])  # (1, NUM_CHANNELS, 200)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(request: WindowRequest) -> PredictResponse:
    x = _to_normalized_tensor(request.window)
    model = state["model"]

    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0, 1].item()

    return PredictResponse(prob_fall=prob, is_fall=prob >= FALL_THRESHOLD)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: WindowRequest) -> ExplainResponse:
    x = _to_normalized_tensor(request.window)
    model = state["model"]

    saliency = grad_x_input_saliency(model, x)  # (NUM_CHANNELS, 200)

    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0, 1].item()

    return ExplainResponse(prob_fall=prob, saliency=saliency.tolist())
