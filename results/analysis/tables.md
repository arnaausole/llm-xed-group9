# Results tables

Made by analysis/analyze.py from the files in results/.

## 1. Test results

| System | Lang | micro-F1 | Jaccard | macro-F1 | micro-P | micro-R | Hamming | Exact match | Malformed |
|---|---|---|---|---|---|---|---|---|---|
| Qwen zero-shot | en | 0.432 | 0.361 | 0.421 | 0.469 | 0.401 | 0.187 | 0.213 | 7.9% |
| Qwen zero-shot | es | 0.334 | 0.268 | 0.310 | 0.399 | 0.287 | 0.203 | 0.161 | 17.7% |
| Qwen five-shot | en | 0.444 | 0.377 | 0.444 | 0.423 | 0.467 | 0.208 | 0.179 | 2.0% |
| Qwen five-shot | es | 0.391 | 0.317 | 0.389 | 0.378 | 0.404 | 0.223 | 0.138 | 1.0% |
| Qwen QLoRA EN | en | 0.487 | 0.432 | 0.453 | 0.463 | 0.515 | 0.192 | 0.266 | 0.0% |
| Qwen QLoRA EN | es | 0.401 | 0.348 | 0.362 | 0.386 | 0.416 | 0.221 | 0.205 | 0.0% |
| Qwen QLoRA ES | es | 0.420 | 0.369 | 0.390 | 0.398 | 0.444 | 0.218 | 0.226 | 0.0% |
| Qwen QLoRA EN+ES | en | 0.478 | 0.416 | 0.447 | 0.434 | 0.531 | 0.206 | 0.252 | 0.0% |
| Qwen QLoRA EN+ES | es | 0.423 | 0.359 | 0.386 | 0.395 | 0.456 | 0.220 | 0.209 | 0.0% |
| Llama zero-shot | en | 0.466 | 0.371 | 0.446 | 0.399 | 0.560 | 0.228 | 0.112 | 0.2% |
| Llama zero-shot | es | 0.422 | 0.335 | 0.402 | 0.362 | 0.506 | 0.246 | 0.112 | 0.2% |
| Llama five-shot | en | 0.490 | 0.399 | 0.481 | 0.422 | 0.585 | 0.216 | 0.146 | 0.2% |
| Llama five-shot | es | 0.437 | 0.347 | 0.421 | 0.375 | 0.523 | 0.239 | 0.110 | 0.0% |

## 2. Dev results

| System | Lang | micro-F1 | Jaccard | macro-F1 | micro-P | micro-R | Hamming | Exact match | Malformed |
|---|---|---|---|---|---|---|---|---|---|
| Qwen zero-shot | en | 0.443 | 0.376 | 0.428 | 0.486 | 0.407 | 0.185 | 0.230 | 8.2% |
| Qwen zero-shot | es | 0.359 | 0.289 | 0.334 | 0.433 | 0.306 | 0.198 | 0.183 | 17.9% |
| Qwen five-shot | en | 0.475 | 0.397 | 0.470 | 0.449 | 0.503 | 0.202 | 0.189 | 0.4% |
| Qwen five-shot | es | 0.409 | 0.343 | 0.403 | 0.397 | 0.421 | 0.220 | 0.169 | 0.5% |
| Qwen QLoRA EN | en | 0.490 | 0.419 | 0.462 | 0.466 | 0.517 | 0.195 | 0.233 | 0.0% |
| Qwen QLoRA EN | es | 0.434 | 0.371 | 0.392 | 0.418 | 0.450 | 0.213 | 0.216 | 0.0% |
| Qwen QLoRA ES | es | 0.448 | 0.391 | 0.419 | 0.432 | 0.465 | 0.208 | 0.235 | 0.0% |
| Qwen QLoRA EN+ES | en | 0.481 | 0.412 | 0.456 | 0.447 | 0.521 | 0.203 | 0.231 | 0.0% |
| Qwen QLoRA EN+ES | es | 0.425 | 0.355 | 0.390 | 0.400 | 0.454 | 0.222 | 0.200 | 0.0% |
| Llama zero-shot | en | 0.517 | 0.411 | 0.509 | 0.442 | 0.624 | 0.211 | 0.140 | 1.2% |
| Llama zero-shot | es | 0.447 | 0.349 | 0.437 | 0.383 | 0.535 | 0.240 | 0.112 | 0.2% |
| Llama five-shot | en | 0.507 | 0.409 | 0.495 | 0.438 | 0.602 | 0.212 | 0.149 | 0.0% |
| Llama five-shot | es | 0.453 | 0.356 | 0.439 | 0.392 | 0.536 | 0.235 | 0.112 | 0.0% |

## 3. Test results with 95% bootstrap confidence intervals (1000 samples, seed 42)

| System | Lang | micro-F1 [95% CI] | Jaccard [95% CI] |
|---|---|---|---|
| Qwen zero-shot | en | 0.432 [0.399, 0.465] | 0.361 [0.327, 0.397] |
| Qwen zero-shot | es | 0.334 [0.302, 0.367] | 0.268 [0.234, 0.298] |
| Qwen five-shot | en | 0.444 [0.412, 0.477] | 0.377 [0.345, 0.410] |
| Qwen five-shot | es | 0.391 [0.359, 0.422] | 0.317 [0.287, 0.349] |
| Qwen QLoRA EN | en | 0.487 [0.455, 0.518] | 0.432 [0.398, 0.468] |
| Qwen QLoRA EN | es | 0.401 [0.367, 0.433] | 0.348 [0.313, 0.384] |
| Qwen QLoRA ES | es | 0.420 [0.387, 0.453] | 0.369 [0.335, 0.405] |
| Qwen QLoRA EN+ES | en | 0.478 [0.448, 0.510] | 0.416 [0.381, 0.450] |
| Qwen QLoRA EN+ES | es | 0.423 [0.390, 0.456] | 0.359 [0.327, 0.394] |
| Llama zero-shot | en | 0.466 [0.438, 0.494] | 0.371 [0.344, 0.398] |
| Llama zero-shot | es | 0.422 [0.391, 0.453] | 0.335 [0.306, 0.365] |
| Llama five-shot | en | 0.490 [0.460, 0.520] | 0.399 [0.370, 0.429] |
| Llama five-shot | es | 0.437 [0.407, 0.467] | 0.347 [0.319, 0.376] |

## 4. Paired comparisons on test (A − B)

| Lang | Comparison | Metric | Difference | 95% CI | CI excludes 0 |
|---|---|---|---|---|---|
| en | Qwen five-shot − Qwen zero-shot | micro-F1 | +0.012 | [-0.017, +0.040] | no |
| en | Qwen five-shot − Qwen zero-shot | Jaccard | +0.016 | [-0.011, +0.041] | no |
| en | Qwen QLoRA EN − Qwen five-shot | micro-F1 | +0.043 | [+0.006, +0.076] | yes |
| en | Qwen QLoRA EN − Qwen five-shot | Jaccard | +0.054 | [+0.019, +0.088] | yes |
| en | Qwen QLoRA EN+ES − Qwen QLoRA EN | micro-F1 | -0.009 | [-0.036, +0.015] | no |
| en | Qwen QLoRA EN+ES − Qwen QLoRA EN | Jaccard | -0.016 | [-0.043, +0.009] | no |
| en | Llama five-shot − Qwen QLoRA EN | micro-F1 | +0.003 | [-0.030, +0.033] | no |
| en | Llama five-shot − Qwen QLoRA EN | Jaccard | -0.032 | [-0.067, -0.002] | yes |
| es | Qwen five-shot − Qwen zero-shot | micro-F1 | +0.057 | [+0.026, +0.089] | yes |
| es | Qwen five-shot − Qwen zero-shot | Jaccard | +0.049 | [+0.019, +0.080] | yes |
| es | Qwen QLoRA EN − Qwen five-shot | micro-F1 | +0.010 | [-0.026, +0.046] | no |
| es | Qwen QLoRA EN − Qwen five-shot | Jaccard | +0.031 | [-0.003, +0.067] | no |
| es | Qwen QLoRA ES − Qwen QLoRA EN | micro-F1 | +0.019 | [-0.006, +0.043] | no |
| es | Qwen QLoRA ES − Qwen QLoRA EN | Jaccard | +0.021 | [-0.005, +0.046] | no |
| es | Qwen QLoRA EN+ES − Qwen QLoRA ES | micro-F1 | +0.004 | [-0.024, +0.031] | no |
| es | Qwen QLoRA EN+ES − Qwen QLoRA ES | Jaccard | -0.010 | [-0.041, +0.022] | no |
| es | Llama five-shot − Qwen QLoRA ES | micro-F1 | +0.017 | [-0.013, +0.048] | no |
| es | Llama five-shot − Qwen QLoRA ES | Jaccard | -0.022 | [-0.055, +0.009] | no |

## 5. Language gap on test (same lines in English and Spanish)

| System | micro-F1 EN − ES | 95% CI |
|---|---|---|
| Qwen zero-shot | +0.098 | [+0.065, +0.137] |
| Qwen five-shot | +0.053 | [+0.022, +0.085] |
| Qwen QLoRA EN | +0.087 | [+0.056, +0.119] |
| Qwen QLoRA EN+ES | +0.054 | [+0.026, +0.086] |
| Llama zero-shot | +0.044 | [+0.015, +0.075] |
| Llama five-shot | +0.053 | [+0.025, +0.083] |

## 6. Number of predicted labels, precision and recall (test)

| System | Lang | Labels/line (pred) | Labels/line (gold) | micro-P | micro-R |
|---|---|---|---|---|---|
| Qwen zero-shot | en | 1.21 | 1.42 | 0.469 | 0.401 |
| Qwen zero-shot | es | 1.02 | 1.42 | 0.399 | 0.287 |
| Qwen five-shot | en | 1.57 | 1.42 | 0.423 | 0.467 |
| Qwen five-shot | es | 1.51 | 1.42 | 0.378 | 0.404 |
| Qwen QLoRA EN | en | 1.58 | 1.42 | 0.463 | 0.515 |
| Qwen QLoRA EN | es | 1.53 | 1.42 | 0.386 | 0.416 |
| Qwen QLoRA ES | es | 1.58 | 1.42 | 0.398 | 0.444 |
| Qwen QLoRA EN+ES | en | 1.74 | 1.42 | 0.434 | 0.531 |
| Qwen QLoRA EN+ES | es | 1.64 | 1.42 | 0.395 | 0.456 |
| Llama zero-shot | en | 1.99 | 1.42 | 0.399 | 0.560 |
| Llama zero-shot | es | 1.99 | 1.42 | 0.362 | 0.506 |
| Llama five-shot | en | 1.97 | 1.42 | 0.422 | 0.585 |
| Llama five-shot | es | 1.98 | 1.42 | 0.375 | 0.523 |

## 7a. F1 per emotion, English (test)

| System | anger | anticipation | disgust | fear | joy | sadness | surprise | trust |
|---|---|---|---|---|---|---|---|---|
| Qwen zero-shot | 0.58 | 0.35 | 0.38 | 0.34 | 0.52 | 0.42 | 0.42 | 0.35 |
| Qwen five-shot | 0.54 | 0.42 | 0.38 | 0.41 | 0.46 | 0.43 | 0.37 | 0.54 |
| Qwen QLoRA EN | 0.64 | 0.47 | 0.45 | 0.43 | 0.59 | 0.47 | 0.18 | 0.40 |
| Qwen QLoRA EN+ES | 0.64 | 0.46 | 0.39 | 0.41 | 0.59 | 0.39 | 0.26 | 0.44 |
| Llama zero-shot | 0.60 | 0.48 | 0.31 | 0.37 | 0.56 | 0.47 | 0.33 | 0.44 |
| Llama five-shot | 0.57 | 0.50 | 0.41 | 0.43 | 0.57 | 0.47 | 0.43 | 0.46 |
| *support (gold lines)* | 134 | 110 | 62 | 73 | 93 | 77 | 81 | 91 |

## 7b. F1 per emotion, Spanish (test)

| System | anger | anticipation | disgust | fear | joy | sadness | surprise | trust |
|---|---|---|---|---|---|---|---|---|
| Qwen zero-shot | 0.42 | 0.22 | 0.19 | 0.32 | 0.46 | 0.44 | 0.30 | 0.13 |
| Qwen five-shot | 0.46 | 0.32 | 0.36 | 0.33 | 0.43 | 0.40 | 0.38 | 0.44 |
| Qwen QLoRA EN | 0.54 | 0.40 | 0.41 | 0.30 | 0.52 | 0.33 | 0.17 | 0.22 |
| Qwen QLoRA ES | 0.53 | 0.45 | 0.33 | 0.33 | 0.54 | 0.39 | 0.30 | 0.27 |
| Qwen QLoRA EN+ES | 0.58 | 0.42 | 0.27 | 0.39 | 0.49 | 0.30 | 0.26 | 0.37 |
| Llama zero-shot | 0.54 | 0.41 | 0.23 | 0.35 | 0.55 | 0.42 | 0.36 | 0.35 |
| Llama five-shot | 0.56 | 0.47 | 0.31 | 0.33 | 0.49 | 0.42 | 0.36 | 0.43 |
| *support (gold lines)* | 134 | 110 | 62 | 73 | 93 | 77 | 81 | 91 |

## 8. Prompt sensitivity, micro-F1 on dev

| System | Lang | v1 (main) | v2 | v3 | Range |
|---|---|---|---|---|---|
| Qwen zero-shot | en | 0.443 | 0.436 | 0.472 | 0.036 |
| Qwen zero-shot | es | 0.359 | 0.389 | 0.376 | 0.030 |
| Qwen five-shot | en | 0.475 | 0.473 | 0.482 | 0.009 |
| Qwen five-shot | es | 0.409 | 0.411 | 0.425 | 0.016 |
| Llama zero-shot | en | 0.517 | 0.492 | 0.517 | 0.026 |
| Llama zero-shot | es | 0.447 | 0.404 | 0.422 | 0.043 |
| Llama five-shot | en | 0.507 | 0.501 | 0.515 | 0.014 |
| Llama five-shot | es | 0.453 | 0.439 | 0.438 | 0.015 |

## 9a. Malformed outputs (test)

| System | Lang | Malformed | Empty list [] | Invalid |
|---|---|---|---|---|
| Qwen zero-shot | en | 7.9% | 7.1% | 0.8% |
| Qwen zero-shot | es | 17.7% | 17.7% | 0.0% |
| Qwen five-shot | en | 2.0% | 0.8% | 1.2% |
| Qwen five-shot | es | 1.0% | 0.8% | 0.2% |
| Qwen QLoRA EN | en | 0.0% | 0.0% | 0.0% |
| Qwen QLoRA EN | es | 0.0% | 0.0% | 0.0% |
| Qwen QLoRA ES | es | 0.0% | 0.0% | 0.0% |
| Qwen QLoRA EN+ES | en | 0.0% | 0.0% | 0.0% |
| Qwen QLoRA EN+ES | es | 0.0% | 0.0% | 0.0% |
| Llama zero-shot | en | 0.2% | 0.0% | 0.2% |
| Llama zero-shot | es | 0.2% | 0.0% | 0.2% |
| Llama five-shot | en | 0.2% | 0.0% | 0.2% |
| Llama five-shot | es | 0.0% | 0.0% | 0.0% |

## 9b. Malformed outputs (dev)

| System | Lang | Malformed | Empty list [] | Invalid |
|---|---|---|---|---|
| Qwen zero-shot | en | 8.2% | 7.9% | 0.3% |
| Qwen zero-shot | es | 17.9% | 17.2% | 0.7% |
| Qwen five-shot | en | 0.4% | 0.3% | 0.1% |
| Qwen five-shot | es | 0.5% | 0.3% | 0.2% |
| Qwen QLoRA EN | en | 0.0% | 0.0% | 0.0% |
| Qwen QLoRA EN | es | 0.0% | 0.0% | 0.0% |
| Qwen QLoRA ES | es | 0.0% | 0.0% | 0.0% |
| Qwen QLoRA EN+ES | en | 0.0% | 0.0% | 0.0% |
| Qwen QLoRA EN+ES | es | 0.0% | 0.0% | 0.0% |
| Llama zero-shot | en | 1.2% | 0.0% | 1.2% |
| Llama zero-shot | es | 0.2% | 0.0% | 0.2% |
| Llama five-shot | en | 0.0% | 0.0% | 0.0% |
| Llama five-shot | es | 0.0% | 0.0% | 0.0% |

## 10. QLoRA training (dev loss per epoch)

| System | Train examples | Dev loss ep. 1 | ep. 2 | ep. 3 | Best epoch |
|---|---|---|---|---|---|
| Qwen QLoRA EN | 3443 | 0.424 | 0.450 | 0.613 | 1 |
| Qwen QLoRA ES | 3443 | 0.452 | 0.479 | 0.585 | 1 |
| Qwen QLoRA EN+ES | 6886 | 0.467 | 0.567 | 0.767 | 1 |

## 11. Computational cost

| System | Inference ms/line (test) | Training min | Training peak GPU GB | Trainable params |
|---|---|---|---|---|
| Qwen zero-shot | 45 | – | – | – |
| Qwen five-shot | 62 | – | – | – |
| Qwen QLoRA EN | 93 | 22 | 7.2 | 43.6M (0.53%) |
| Qwen QLoRA ES | 90 | 22 | 7.2 | 43.6M (0.53%) |
| Qwen QLoRA EN+ES | 96 | 43 | 7.2 | 43.6M (0.53%) |
| Llama zero-shot | 45 | – | – | – |
| Llama five-shot | 68 | – | – | – |

GPU: NVIDIA A100-PCIE-40GB.
