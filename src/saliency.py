"""Grad x Input saliency for model interpretability."""

import numpy as np
import torch
import torch.nn as nn


def grad_x_input_saliency(model: nn.Module, window_tensor: torch.Tensor) -> np.ndarray:
    """window_tensor: (1, C, T) normalized tensor, already on the model's device."""
    model.eval()
    window_tensor = window_tensor.clone().detach().requires_grad_(True)
    logits = model(window_tensor)
    prob = torch.softmax(logits, dim=1)[0, 1]
    prob.backward()

    sal = (window_tensor.grad * window_tensor).abs().squeeze(0).detach().cpu().numpy()  # (C,T)
    return sal
