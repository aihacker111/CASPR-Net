# CASPR-Net

**CASPR** is the single name used by this repository. It means
**Curvature-Aligned Sparse Product Reparameterization**.

CASPR-Net targets the dominant cost of a mobile backbone: dense pointwise
channel mixing. A block uses

```text
efficient separable stem
-> multi-branch DWConv 3x3
-> grouped 1x1 factor (8 channels/group)
-> activation
-> perfect channel shuffle
-> grouped 1x1 factor
-> residual add
-> sparse grouped stage transitions
```

The two grouped factors replace the dense `C -> 2C -> C` MLP. For group size
`b`, their channel-mixing cost is `2 b C` MACs per spatial position instead of
`4 C^2`. The perfect shuffle changes the grouping basis between the factors;
stacking blocks therefore grows cross-group connectivity without adding extra
deploy operators.

The stem is a dense `3x3` stride-2 convolution followed by depthwise stride-2
reduction and a pointwise projection. Stage transitions use depthwise reduction
and grouped pointwise projection. After the final stage, global pooling is
performed first and a learned linear projection expands the compact
representation. Moving this projection after pooling preserves deploy capacity
while avoiding a large `1x1` operation at every `7x7` spatial location.

During training, every depthwise operator and sparse channel factor has several
parallel Conv-BN branches. Branches with identical support fuse exactly:

```text
W_deploy = sum_r fold_bn(W_r)
b_deploy = sum_r fold_bn(b_r)
```

Consequently conversion is a function-preserving algebraic operation, not
pruning or post-training approximation. The activation stays between the two
sparse factors and is never claimed to fold into a linear convolution.

Both spatial kernels and sparse channel factors use online curvature statistics
to precondition gradients. For an existing sparse edge `(o, i)`, the diagonal
channel metric approximates local curvature by

```text
H[o, i] ~= E[g_o^2] E[x_i^2].
```

The preconditioner changes optimization only; it never changes the sparse
deploy topology.

## Correctness test

```bash
python test_caspr.py
```

This checks spatial fusion, sparse channel fusion, cross-group connectivity,
metric backward stability, and full-network train/deploy equivalence.

## Synthetic train, equivalence and latency

```bash
python train_synthetic.py \
  --epochs 3 --device cpu \
  --latency-batch-size 1 --latency-warmup 50 --latency-iters 300
```

## ImageNet-1K

```bash
bash train_caspr_imagenet.sh /absolute/path/to/imagenet casprnet_n
```

For one selected GPU:

```bash
GPU=0 BATCH_SIZE=256 OUTPUT_DIR=checkpoints/casprnet_n \
  bash train_caspr_imagenet.sh /absolute/path/to/imagenet casprnet_n
```

The ImageNet script enables hard distillation by default using a pretrained
RegNetY-16GF teacher. The teacher checkpoint is downloaded on first use. It can
also be supplied as a local file or disabled explicitly:

```bash
TEACHER_PATH=/absolute/path/to/regnety_160-a5fe301d.pth \
  bash train_caspr_imagenet.sh /absolute/path/to/imagenet casprnet_n

DISTILLATION_TYPE=none \
  bash train_caspr_imagenet.sh /absolute/path/to/imagenet casprnet_n
```

For gradient accumulation on a memory-limited GPU, `BATCH_SIZE` is the
micro-batch resident on the GPU and `ACCUM_STEPS` is the number of
micro-batches per optimizer update. For example, this gives an effective batch
of `64 * 4 = 256` on one GPU:

```bash
GPU=0 BATCH_SIZE=64 ACCUM_STEPS=4 OUTPUT_DIR=checkpoints/casprnet_n \
  bash train_caspr_imagenet.sh /absolute/path/to/imagenet casprnet_n
```

On distributed training, the effective global batch is
`BATCH_SIZE * ACCUM_STEPS * number_of_GPUs`. BatchNorm still observes only the
per-GPU micro-batch, not the accumulated effective batch.

Registered variants are `casprnet_n`, `casprnet_t`, and `casprnet_s`.
Use `--disable-caspr-metric` for the identity-metric ablation and
`--caspr-metric-interval N` to control metric sampling cost.

## Parameters, MACs and latency

```bash
python count_params.py --model casprnet_n --preset imagenet
python count_params.py --model casprnet_n --preset imagenet --deploy
python benchmark.py --model casprnet_n --device cuda --batch-size 1 --deploy
python benchmark_pair.py --model casprnet_n --device cpu --threads 1
```

At input `1x3x224x224`, the current profiles are:

| Variant | Graph | Parameters | MACs |
|---|---|---:|---:|
| `casprnet_n` | train | 1.688M | 0.0492G |
| `casprnet_n` | deploy | **1.596M** | **0.0292G** |
| `casprnet_t` | deploy | 1.971M | 0.0414G |
| `casprnet_s` | deploy | 2.499M | 0.0585G |

These are architecture counts, not accuracy claims. ImageNet and COCO results
must be measured after training. Sparse grouped kernels are backend-sensitive,
so always report real device latency in addition to MACs. On the local CPU
Paired seven-round benchmarks use batch 1, 224x224 inputs, one CPU thread, 50
warm-up forwards, alternating graph order, and at least 1.5 seconds per trial:

| Variant | Unfused median | Deploy median | Speedup | Deploy throughput |
|---|---:|---:|---:|---:|
| `casprnet_n` | 23.9226 ms | 11.8704 ms | 2.015x | 84.24 image/s |
| `casprnet_t` | 60.5072 ms | 38.9185 ms | 1.555x | 25.69 image/s |
| `casprnet_s` | 91.0940 ms | 49.4525 ms | 1.842x | 20.22 image/s |

These are engineering measurements, not accuracy-matched SOTA claims.
Multi-thread scaling remains backend-sensitive because grouped 1x1 kernels and
channel shuffle are not fused by eager PyTorch.

Full `casprnet_n` latency by activation uses paired seven-round measurements,
alternating unfused and deploy graphs with at least one second per trial:

| Activation | Unfused median | Deploy median | Fusion speedup | Deploy equivalence |
|---|---:|---:|---:|---:|
| ReLU | 25.9335 ms | 12.0784 ms | 2.147x | pass |
| ReLU6 | 25.7770 ms | 11.8447 ms | 2.176x | pass |
| GELU | 29.4021 ms | 14.2784 ms | 2.059x | pass |
| SiLU | 31.7922 ms | 16.4916 ms | 1.928x | pass |
| Swish | 31.1976 ms | 16.5837 ms | 1.881x | pass |
| Mish | 44.0652 ms | 31.3016 ms | 1.408x | pass |
| Hardswish | 27.2659 ms | 12.4048 ms | 2.198x | pass |

All activation tests also pass finite-gradient backward checks. These latency
results do not compare accuracy because each activation requires a separately
trained checkpoint for a valid ImageNet comparison.

## ONNX export

```bash
python export_onnx.py --model casprnet_n \
  --checkpoint checkpoints/casprnet_n/checkpoint_best.pth \
  --output casprnet_n.onnx --deploy --verify
```

Checkpoints from earlier architectures are intentionally incompatible because
the dense channel MLP and nonlinear shear are no longer part of CASPR-Net.
