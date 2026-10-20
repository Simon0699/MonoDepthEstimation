# Self-Supervised Monocular Depth on CPU: Project Plan

Planning notes from 2026-10-05. No code yet.

## 1. Goal and constraints

| Item | Decision |
|---|---|
| **Task** | Monocular depth estimation, trained self-supervised (Monodepth2-style photometric loss) |
| **Training** | H200 GPU; training time is not a constraint |
| **Inference** | Intel Core Ultra 7 155H, CPU only |
| **Input size** | ~224×224 (or the same pixel count at the camera's aspect ratio) |
| **Runtime** | torch AOTInductor + OpenCV in C++ |
| **Encoder** | Fully fine-tuned (not frozen) |
| **Pose** | Large foundation model, probably VGGT, used only during training |
| **Data** | Own calibrated camera and/or a public dataset (handled separately) |

## 2. Architecture at a glance

| Part | Choice | Approx. cost @224² |
|---|---|---|
| Encoder | MobileNetV4-Conv-Medium, ImageNet weights (timm), fully fine-tuned | ~0.8 GMAC |
| Baseline encoder | ResNet18 (Monodepth2's own) | ~1.8 GMAC |
| Depth head | Slim Monodepth2-style U-Net, output at 1/2 res + bilinear ×2 | ~0.3–0.4 GMAC |
| **Whole model** | MobileNetV4 + slim head | **~1.2 GMAC**: est. 5–10 ms fp32 on the P-cores, about half in int8 (measure) |
| Precision | fp32, then int8 via quantization-aware training | – |
| Pose (training only) | VGGT, precomputed offline | none at inference |
| Runtime | AOTInductor compiled on the laptop; fallback OpenVINO | – |

## 3. Key facts that drive the design

1. **The head can cost as much as the encoder.** Monodepth2's stock decoder at 224² is ~1.45 GMAC: about 80% of its ResNet18 encoder and ~6–7× an entire MobileNetV3-Large. High-resolution layers are also memory-bound, so real latency is worse than the FLOP count suggests.
2. **Freezing doesn't make inference cheaper**; it only affects training. With full fine-tuning, ImageNet-pretrained CNNs work well (Monodepth2 does exactly this), so the encoder can be picked purely for speed on the target CPU.
3. **The 155H wants fp32 or int8, on its P-cores only.** int8 via AVX-VNNI is the main speed lever.

## 4. Target hardware: Intel Core Ultra 7 155H

- **Cores:** 6 performance cores (12 threads), 8 efficiency cores, 2 low-power efficiency cores.
- **Instruction set:** AVX2 + AVX-VNNI. No AVX-512, no AMX, no native bf16.
  - Use fp32 or int8 only. int8 is realistically ~2–3× faster on ordinary convolutions.
  - bf16 and fp16 won't help.
- **Run only on the P-cores.** The efficiency cores are much slower at vector math, so every parallel step ends up waiting on them.
  - Pin the process to the P-cores (usually logical CPUs 0–11, but verify).
  - Try both 6 and 12 threads.
- **Windows 11 background throttling:** a minimized or background app can be moved to the efficiency cores (EcoQoS). Opt out with `SetProcessInformation` (ProcessPowerThrottling), or keep the app in the foreground.
- **Benchmark like it's a laptop:** plugged in, "Best performance" power mode, sustained for several minutes. Turbo boost runs out and latency rises.
- **Latency budget:** ~1–1.5 GMAC for the whole model ≈ 5–10 ms fp32 on the P-cores, about half in int8 (estimates; measure). A ~4.5 GMAC encoder would be roughly 3–4× slower.
- The chip also has an NPU and an Arc integrated GPU. They're not part of the plan (CPU only), but OpenVINO can target them if that changes.

## 5. Encoder

**Default: MobileNetV4-Conv-Medium** (ImageNet weights from timm)
- ~0.8 GMAC at 224, ~80% ImageNet top-1.
- Uses only ReLU activations, so it converts to int8 cleanly.
- Outputs features at 1/2, 1/4, 1/8, 1/16 and 1/32 resolution, so it plugs straight into a U-Net decoder with no extra RGB branch.

**Baseline: ResNet18**
- Monodepth2's own encoder: 1.8 GMAC of plain 3×3 convolutions, which run very efficiently on this CPU.
- Use it to check the training code reproduces Monodepth2's KITTI result (~0.115 AbsRel), and as the score MobileNetV4 has to beat.

**Not recommended now: DINOv3 ConvNeXt-Tiny or ViT-S/16** (~4.5 GMAC)
- Only revisit if int8 gets one under the latency budget and you want maximum quality.

**Fine-tuning tips**
- Encoder learning rate ~0.1× the head's.
- Keep BatchNorm in training mode; it gets merged into the convolutions at export.
- Finish with a few epochs of quantization-aware training (QAT). It's cheap on the H200 and the cleanest route to int8.

## 6. Depth head

A slimmed-down Monodepth2 U-Net decoder:
- **Widths:** ~16 / 32 / 64 / 96 channels at 1/2, 1/4, 1/8, 1/16 resolution.
- **Each level:** nearest ×2 upsample, concatenate the matching encoder features, then 1–2 × (3×3 conv + BatchNorm + ReLU).
- **Activation and padding:** ReLU instead of Monodepth2's ELU (quantizes better); zero padding instead of reflection.
- **Outputs:**
  - Sigmoid disparity at 1/8, 1/4 and 1/2 resolution for the multi-scale training loss.
  - Export only the 1/2-resolution output, and do the final bilinear ×2 upsample inside the exported model.
  - Keep that last conv + sigmoid in fp32 when quantizing, so the depth doesn't show banding.
- **Cost:** ~0.3–0.4 GMAC vs ~1.45 for the stock Monodepth2 decoder.

Design rules:
- Conv cost ∝ H × W × C²: keep channels thin at high resolution and spend capacity at low resolution, where it's cheap.
- Don't decode to full resolution. A full-res stage adds little at 224² and is expensive.
- At these small widths, plain 3×3 convs are fine and quantize best. Depthwise convs are slower on this CPU than their FLOP count suggests and gain little from int8. Only switch to them if profiling shows the head is the bottleneck.

## 7. Pose: VGGT (training only)

- **Run it once, offline.** Feed it undistorted, full-resolution frames (it works at ~518 px wide), over longer windows rather than one triplet at a time. Store a pose per frame.
- With poses fixed, only the depth network trains. That's simpler and more stable than Monodepth2's joint depth + pose training.
- **Scale caveat:** VGGT's outputs are in a normalized scale per run (roughly "average point distance from the first camera = 1").
  - The depth network will learn that relative scale, not metres, and it can shift between windows.
  - For metric depth, rescale the translations with a metric source, or use MapAnything, which takes calibrated intrinsics as input and outputs metric scale.
- **Sanity check:** VGGT predicts its own field of view. Compare it with your calibration.
- **Free bonus:** VGGT also outputs per-frame depth + confidence. Adding it as an extra loss that ignores scale is probably the biggest single quality boost (at the cost of not being purely self-supervised).

## 8. Input resolution

- **224×224 is a good fit for this chip.** Cost scales with pixel count: 160×160 ≈ 0.5×, 320×240 ≈ 1.5×.
- **Don't go much lower.** Self-supervision learns from texture and motion between frames. At low resolution, thin objects (poles, chair legs) disappear and distant depth gets noisy. Rough floor: ~160 px on the short side.
- **Match the camera's aspect ratio** at about the same pixel count instead of cropping or squashing to a square: 256×192 for 4:3, 288×160 for 16:9. Keep both dimensions divisible by 32.
- **Every crop or resize changes the intrinsics K.** A wrong K silently gives mediocre depth.
- **Worth trying:** compute the photometric loss on higher-resolution frames by upsampling the predicted disparity (Monodepth2 already upsamples to input resolution). It gives finer supervision at no inference cost.

## 9. Training objective

Keep Monodepth2's core pieces:
- Per-pixel minimum reprojection loss over the source frames.
- Auto-masking of pixels that don't change between frames (static camera, objects moving with the camera).
- Photometric error = SSIM + L1 (α = 0.85).
- Edge-aware smoothness on mean-normalized disparity (weight ~1e-3).
- Multi-scale outputs, each upsampled to input resolution before computing the loss.

Also:
- Set min/max depth for your domain (KITTI's 0.1–100 m is wrong indoors).
- Anything used only in training (pose model, warping, losses, any teacher model) can be as heavy as you like. None of it ships.

## 10. Data checklist (for reference; handled separately)

- Video with real camera translation (pure rotation gives no depth signal) and mostly static scenes.
- Calibrated intrinsics. Undistort frames if the lens needs it, with the same remap in training and at inference.
- Auto-exposure and white balance locked while recording.
- Frame spacing chosen so neighbouring frames have visible parallax. Drop clips where the camera barely moves.
- Indoor or handheld footage is much harder than driving footage.
- If possible, a small ground-truth set (depth sensor or phone LiDAR) for evaluating on your own domain.

## 11. Deployment: AOTInductor + OpenCV (C++)

**Export**
- Export only encoder + head, producing the final disparity/depth. Static input shape, batch size 1. The pose model, warping and extra-scale outputs stay in Python.
- Put preprocessing (uint8 → float, BGR → RGB, normalization) and disparity → depth conversion inside the exported model. Inductor fuses them, and C++ can pass the `cv::Mat` buffer in without a copy via `torch::from_blob`.
- Enable Inductor's `freezing` option (merges BatchNorm, pre-packs weights for oneDNN). It's one of the biggest CPU wins; confirm it takes effect in AOT mode on your PyTorch version.

**Build**
- Compile on the 155H laptop, **not** the H200 machine. The compiled code is tied to the OS and CPU instruction set. The server's CPU almost certainly has AVX-512, so code built there would likely crash on the 155H.
- On Windows, AOTInductor needs the MSVC toolchain and is less mature than on Linux. **First milestone:** a dummy model compiled with AOTInductor and called from the C++/OpenCV app.
- Fallback if it fights you: OpenVINO (Intel's runtime; handles int8 well on this chip; can also target the NPU/iGPU).

**Run**
- Don't let OpenCV's threads and PyTorch's OpenMP pool oversubscribe the cores.
- Pin to the P-cores, flush denormals, and do a few warm-up runs.
- Measure p50/p99 latency end to end (capture to depth), sustained for several minutes.

**int8**
- PT2E quantization with the X86 Inductor quantizer (now part of torchao) works with Inductor/AOTInductor.
- Use QAT at the end of training, and keep the final disparity conv + sigmoid in fp32.

**Parity**
- Use OpenCV in the training data pipeline too (same resize interpolation, same BGR/RGB order, same undistortion).
- Diff C++ and Python outputs on the same frames.

**Sanity benchmark**
- Do one ONNX export and run it through ONNX Runtime and OpenVINO. For small batch-1 CPU models they sometimes beat AOTInductor.

## 12. Order of work

1. Pin down the open questions below.
2. **Deployment spike on the 155H:** get a dummy model through AOTInductor into the C++/OpenCV app. Then benchmark MobileNetV4-Conv-Medium and ResNet18 (+ slim head) at the exact input size, in fp32 and int8.
3. **Validate the training code:** reproduce Monodepth2 (ResNet18 with its own pose network) on KITTI to ~0.115 AbsRel.
4. Switch to VGGT poses + MobileNetV4 + slim head, and train on your own data at the deployment resolution.
5. **Ablations:** MobileNetV4 vs ResNet18, head width, output stride, VGGT-depth extra loss on/off, higher-res photometric loss on/off.
6. **Optimize and verify:** QAT to int8, threads and core pinning, C++ vs Python parity, end-to-end p50/p99 latency.

## 13. Open questions

- Latency / FPS target?
- Indoor or outdoor, and how will the camera move?
- Metric or relative depth? (Decides how VGGT's scale is handled.)
- Which camera: native resolution and aspect ratio?

## 14. Considered and dropped: frozen encoder

The first plan froze the encoder. That would have required DINO-family features (DINOv3 ConvNeXt-Tiny or ViT-S/16, ~4.5 GMAC), because ImageNet-classification CNNs make weak frozen depth features. A ViT would also have needed an extra trainable RGB branch to recover edges from its coarse 14×14 token grid at 224. The idea was dropped in favour of full fine-tuning, which allows a much cheaper encoder and costs nothing extra at inference.

## 15. Reference numbers

| Model / part | Approx. cost @224² |
|---|---|
| MobileNetV3-Large | 0.22 GMAC |
| MobileNetV4-Conv-Medium | ~0.8 GMAC |
| ResNet18 | 1.8 GMAC |
| ConvNeXt-Tiny (incl. DINOv3 ConvNeXt-Tiny) | 4.5 GMAC |
| ViT-S/16 (e.g. DINOv3) | ~4.6 GMAC |
| ViT-S/14 (DINOv2, Depth Anything V2-Small) | ~6 GMAC |
| Monodepth2 stock decoder | ~1.45 GMAC |
| Depth Anything V2 DPT head (ViT-S config) | ~2 GMAC |
| Slim head (this plan) | ~0.3–0.4 GMAC |

Monodepth2 on KITTI (monocular, 640×192, ResNet18): AbsRel ≈ 0.115.

## 16. References

- **Monodepth2:** Godard et al., "Digging Into Self-Supervised Monocular Depth Estimation", ICCV 2019. [github.com/nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2)
- **VGGT:** Wang et al., "VGGT: Visual Geometry Grounded Transformer", CVPR 2025. [github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- **MapAnything:** Keetha et al., 2025 (feed-forward metric 3D reconstruction; accepts intrinsics as input).
- **MobileNetV4:** Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem", ECCV 2024. Weights in timm: [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- **Lite-Mono:** Zhang et al., CVPR 2023 (efficient self-supervised depth; reference for the small-model regime).
- **FastDepth:** Wofk et al., ICRA 2019 (efficient depth decoder at 224×224).
- **DINOv3:** Meta AI, 2025.
- **Depth Anything V2:** Yang et al., NeurIPS 2024.
