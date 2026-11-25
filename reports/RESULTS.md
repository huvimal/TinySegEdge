# TinySegEdge — measured local run

One-layer trained segmentation on authored procedural rectangles. No real-world defect or medical image accuracy claim.

## Results

```json
{
  "test_pixel_iou": 1.0,
  "onnx_max_logit_difference": 1.9073486328125e-06,
  "onnx_bytes": 212,
  "checks": [
    "Actual ONNX Runtime logits match trained PyTorch convolution",
    "Separate seed for held-out shapes"
  ],
  "scope": "One-layer trained segmentation on authored procedural rectangles. No real-world defect or medical image accuracy claim."
}
```

Local experiment assertions passed. This does not establish production readiness or performance on real customer data. Source fingerprints and dependency versions are in results.json.
