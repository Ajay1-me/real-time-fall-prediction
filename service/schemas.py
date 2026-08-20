"""Request/response schemas for the fall prediction service."""

import os
from typing import List

from pydantic import BaseModel, Field, field_validator

WINDOW_SIZE = 200
# MODEL_CHANNELS also controls which checkpoint/norm-stats files service/main.py
# loads at startup, so this env var is the single source of truth for both.
NUM_CHANNELS = int(os.environ.get("MODEL_CHANNELS", "9"))


class WindowRequest(BaseModel):
    window: List[List[float]] = Field(
        ..., description=f"Sensor window of shape ({WINDOW_SIZE}, {NUM_CHANNELS})"
    )

    @field_validator("window")
    @classmethod
    def check_shape(cls, window: List[List[float]]) -> List[List[float]]:
        if len(window) != WINDOW_SIZE:
            raise ValueError(f"window must have {WINDOW_SIZE} timesteps, got {len(window)}")
        if any(len(row) != NUM_CHANNELS for row in window):
            raise ValueError(f"each timestep must have {NUM_CHANNELS} channels")
        return window


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    prob_fall: float
    is_fall: bool


class ExplainResponse(BaseModel):
    prob_fall: float
    saliency: List[List[float]]  # shape (NUM_CHANNELS, WINDOW_SIZE), Grad x Input
