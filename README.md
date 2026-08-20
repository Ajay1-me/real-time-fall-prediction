# real-time-fall-prediction
Real-time system for predicting and preventing resident falls using sensor data and AI

[ProjectUpdate](https://docs.google.com/document/d/1FxcNHMSfU63HuoRoS0GrPlL0ndzNWmxdpObpQB0qcfU/edit?usp=sharing)

## Architecture

The system is trained and served as a straight pipeline: raw sensor recordings go in, a live HTTP prediction service comes out.

```
SisFall dataset (raw IMU recordings, 9 channels, 200 Hz)
        |
1-second sliding windows (200 samples, 50% overlap)
        |
subject-wise train/val/test split (no subject appears in more than one split)
        |
1D CNN (FallDetectorCNN) -- trained in src/train.py, run via main.py train
        |
best_fall_detector_cnn.pt + norm_stats.npz (trained weights + normalization stats)
        |
FastAPI service (service/main.py) -- /health, /predict, /explain
        |
Docker container (python:3.11-slim) -- single container, port 8000
```

The model is a compact 1D CNN (three convolution blocks, ~37K parameters) rather than a large architecture, because the actual constraint on this system is not accuracy headroom but inference speed on low-power hardware -- see the latency numbers below. A TCN variant was also trained and evaluated as a comparison, but the CNN is what's deployed in the service.

## Model Comparison

Four models were trained and evaluated on the same held-out, subject-wise test split (results from `comparison.py`):

| Model               | F1    | Precision | Recall | ROC-AUC |
|---------------------|-------|-----------|--------|---------|
| CNN                 | 0.726 | 0.630     | 0.856  | 0.916   |
| TCN                 | 0.686 | 0.580     | 0.841  | 0.894   |
| Random Forest       | 0.702 | 0.706     | 0.699  | 0.877   |
| Logistic Regression | 0.658 | 0.628     | 0.692  | 0.822   |

The CNN was selected because it has the best F1 and ROC-AUC, and the highest recall of the four (0.856) -- for a fall detector, missing a real fall is worse than a false alarm, so recall matters more than precision here. The TCN's dilated convolutions, which give it a wider temporal receptive field, did not improve on the CNN -- a sign that the fall signature in a 1-second window is dominated by a short, local impact spike rather than longer-range temporal structure.

## Inference Latency

Measured with `benchmark_latency.py`, running the CNN on CPU over 500 real test-set windows, one window at a time (matching how a live stream is processed):

- Average latency: 0.351 ms
- p95 latency: 0.697 ms
- Required budget: 250 ms (the streaming simulation produces a new window every 50 samples at 200 Hz, i.e. every 0.25s)

Even the p95 latency is about 358x faster than the required budget. Inference speed is not a bottleneck for this system, on CPU, with no GPU required.

## Real-Time Detection Performance

Measured with `demo_stream.py`, which streams a real SisFall recording through the deployed FastAPI service one window at a time over HTTP -- the same code path a real client would use. The recording is a subject the model never saw during training (from the held-out test split), so this reflects generalization, not memorization.

- **Fall recording**: first alert fired 1.00s into the recording; the impact (peak acceleration) occurred at 9.11s. That's an **8.11 second lead time** -- the system flags the fall well before impact, not after it.
- **ADL (normal activity) recording**: 100 seconds of everyday movement produced 23 alerts, a rate of **13.80 false alarms per minute**.

The lead time is a strong result. The false-alarm rate is the honest weak point of the current model -- it is high enough that this model, as-is, would not be usable in a real deployment without further tuning (e.g. raising the decision threshold, adding post-processing, or collecting more ADL data for that motion pattern). This is a known, quantified limitation, not a hidden one -- and it's part of why picking the right operating threshold (see the threshold sweep in the notebook) matters as much as picking the right model architecture.

## Running with Docker

Build the image (from the project root, where `best_fall_detector_cnn.pt` and `norm_stats.npz` already exist):

```
docker build -t fall-prediction-service .
```

Run the container, mapping the service to `localhost:8000`:

```
docker run -p 8000:8000 fall-prediction-service
```

Verify it's up:

```
curl http://localhost:8000/health
```
