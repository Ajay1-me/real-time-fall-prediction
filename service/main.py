"""FastAPI inference service for fall prediction.

Run from the project root:
    python -m uvicorn service.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List

import numpy as np
import torch
from fastapi import FastAPI

from service.schemas import ExplainResponse, HealthResponse, PredictResponse, WindowRequest
from src.data import load_norm_stats
from src.model import FallDetectorCNN, get_device
from src.saliency import grad_x_input_saliency

CHECKPOINT_PATH = "best_fall_detector_cnn.pt"
NORM_STATS_PATH = "norm_stats.npz"
FALL_THRESHOLD = 0.7

state: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = get_device()
    model = FallDetectorCNN().to(device)
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


def _to_normalized_tensor(window: List[List[float]]) -> torch.Tensor:
    x = np.array(window, dtype=np.float32)  # (200, 9)
    mean, std = state["mean"], state["std"]
    x = (x - mean.squeeze(0)) / std.squeeze(0)
    return torch.from_numpy(x).unsqueeze(0).permute(0, 2, 1).float().to(state["device"])  # (1, 9, 200)


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

    saliency = grad_x_input_saliency(model, x)  # (9, 200)

    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0, 1].item()

    return ExplainResponse(prob_fall=prob, saliency=saliency.tolist())
