# real-time-fall-prediction
Real-time system for predicting and preventing resident falls using sensor data and AI


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

Five models were trained and evaluated on the same held-out, subject-wise test split (results from `comparison.py`):

| Model                   | F1    | Precision | Recall | ROC-AUC |
|-------------------------|-------|-----------|--------|---------|
| CNN (9-channel)         | 0.726 | 0.630     | 0.856  | 0.916   |
| CNN (6-channel, phone)  | 0.731 | 0.670     | 0.804  | 0.908   |
| TCN                     | 0.686 | 0.580     | 0.841  | 0.894   |
| Random Forest           | 0.702 | 0.706     | 0.699  | 0.877   |
| Logistic Regression     | 0.658 | 0.628     | 0.692  | 0.822   |

The 9-channel CNN was selected as the reference model because it has the best F1 and ROC-AUC, and the highest recall of the five (0.856) -- for a fall detector, missing a real fall is worse than a false alarm, so recall matters more than precision here. The TCN's dilated convolutions, which give it a wider temporal receptive field, did not improve on the CNN -- a sign that the fall signature in a 1-second window is dominated by a short, local impact spike rather than longer-range temporal structure.

### 6-channel (phone-deployable) variant

SisFall's sensor rig has two accelerometers plus one gyroscope (9 channels). A phone only exposes one of each (6 channels) -- no second accelerometer. Before assuming phone deployment is viable, the second accelerometer (`acc2_x/y/z`) was dropped and a separate CNN was trained from scratch on the remaining 6 channels, using the same subject-wise split, class weighting, and training loop as the 9-channel model, with its own normalization stats recomputed on the 6-channel data (not sliced from the 9-channel ones). It's saved as a distinct checkpoint (`best_fall_detector_cnn_6ch.pt`), so the 9-channel model stays the untouched reference model.

The overall F1 barely moves (0.731 vs 0.726) -- but that hides the actual tradeoff: precision goes up (0.630 -> 0.670) while recall goes down (0.856 -> 0.804). In other words, the 6-channel model misses more real falls in exchange for fewer false alarms. Given recall is the metric that matters most here, that is a real cost, not a wash.

As a sanity check on that result, the Random Forest's feature importances (fit on the 9-channel data) were inspected for where the dropped `acc2_*` features rank among all 36 stats features (mean/std/min/max x 9 channels):

- `acc2_y_mean` ranks **2nd of 36** (importance 0.081)
- `acc2_y_min` ranks **4th of 36** (importance 0.056)
- the remaining ten `acc2_*` features rank in the middle-to-bottom third

That's consistent with the direction of the CNN result: a couple of the dropped features are genuinely high-value (explaining why recall measurably drops), but most of what's lost is lower-importance redundant signal (explaining why the drop is moderate rather than severe). The two results corroborate each other rather than contradicting.

**Conclusion**: phone deployment (6 channels) is viable but not free -- expect a measurable recall drop, not just a rounding error. Whether that tradeoff is acceptable depends on the product requirement: a wearable with two accelerometers should stay on the 9-channel model; a phone-only product should budget for roughly a 5-point recall hit and consider compensating with a lower decision threshold.

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

## Phone Streaming Demo

`phone_client/index.html` is a single static page that streams a phone's live accelerometer + gyroscope readings to the running service and shows a live fall-probability readout. It uses the 6-channel model from Step 8, since a phone only exposes one accelerometer and one gyroscope -- not SisFall's two accelerometers.

Both the page and the API must be served over **HTTPS**, even for a purely local demo: modern browsers (iOS Safari and Android Chrome both) only expose `DeviceMotionEvent` in a secure context, and an HTTPS page cannot `fetch()` a plain HTTP API (mixed-content blocking). This was discovered by actually testing on a phone, not anticipated in advance -- plain HTTP does not work here.

**1. Find the laptop's local IP address** (needed for the certificate and both URLs below):

```
# macOS
ipconfig getifaddr en0
```

**2. Generate a local self-signed certificate**, bound to that IP, once (it's gitignored, machine-specific, and not something to commit):

```
mkdir -p .local_certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout .local_certs/key.pem -out .local_certs/cert.pem -days 365 \
  -subj "/CN=<laptop-ip>" -addext "subjectAltName=IP:<laptop-ip>"
```

**3. Start the service in 6-channel mode over HTTPS**, with `best_fall_detector_cnn_6ch.pt` already trained:

```
MODEL_CHANNELS=6 python -m uvicorn service.main:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile .local_certs/key.pem --ssl-certfile .local_certs/cert.pem
```

`--host 0.0.0.0` is required (not the default `127.0.0.1`) so the service accepts connections from another device on the network, not just the laptop itself.

**4. Serve the phone client over HTTPS**, from a second terminal:

```
node phone_client/serve_https.js
```

(A plain Python `http.server` cannot do this without extra work, and a from-scratch attempt at wrapping it with `ssl` hit an unresolved TLS handshake interop issue in this environment -- Node's `https` module is used here instead because it worked reliably.)

**5. On the phone** (same WiFi network as the laptop): open `https://<laptop-ip>:8080` in the browser. The browser will warn that the certificate isn't trusted (it's self-signed) -- accept/proceed anyway, this is expected for a local dev certificate. Do the same for `https://<laptop-ip>:8000` directly in the browser once, so the phone trusts that origin too (the page's `fetch` calls to it will otherwise be silently blocked). Then set the page's "Service URL" field to `https://<laptop-ip>:8000`, tap **Start Streaming**, and grant the motion-sensor permission prompt if one appears (iOS 13+ requires this; most Android browsers skip it). The page buffers about a second of sensor data, then starts sending live predictions once per second.

### Known limitations of this demo

This demo is a real streaming test against a real deployed model -- but it is explicitly **not** a like-for-like reproduction of the benchmark numbers in Step 8's comparison table, for reasons worth stating plainly rather than glossing over:

- **Sensor placement differs.** SisFall's rig is belt-worn at the waist; a phone is typically held, pocketed, or otherwise positioned differently. The model was trained on one specific placement and has no guarantee of transferring to another.
- **Sampling rate is lower and variable.** SisFall was recorded at a fixed 200Hz. Browser `devicemotion` events fire at whatever rate the device and browser choose to deliver -- commonly far below 200Hz, and it can vary while streaming. The page resamples/interpolates whatever it receives up to the model's expected 200-sample window, but interpolated data is not the same as data that was actually sampled at 200Hz.
- **Units and axis conventions required guesswork.** `DeviceMotionEvent` reports acceleration in m/s² (converted to g here) and rotation rate using the W3C's alpha/beta/gamma axis convention, which does not necessarily align with SisFall's physical gyroscope axis orientation. The mapping used here is a reasonable best-effort, not a verified match.
- **Accuracy should be expected to differ from the Step 8 benchmark numbers**, in either direction, for all of the reasons above. This page demonstrates that the deployed pipeline works end-to-end on live sensor data -- it is not a substitute for the held-out test-set evaluation.

Calling this out explicitly is deliberate: the gap between a benchmark run on curated lab data and a live demo on real, uncontrolled hardware is a real and expected part of shipping a model, not a flaw to hide.
