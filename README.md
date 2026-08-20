# real-time-fall-prediction
Real-time system for predicting and preventing resident falls using sensor data and AI

[ProjectUpdate](https://docs.google.com/document/d/1FxcNHMSfU63HuoRoS0GrPlL0ndzNWmxdpObpQB0qcfU/edit?usp=sharing)

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
