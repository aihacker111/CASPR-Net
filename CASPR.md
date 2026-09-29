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
bash train_caspr_imagenet.sh casprnet_n
```

For one selected GPU:

```bash
GPU=0 BATCH_SIZE=64 ACCUM_STEPS=4 OUTPUT_DIR=checkpoints/casprnet_n \
  bash train_caspr_imagenet.sh casprnet_n
```

The ImageNet script enables hard distillation by default using a pretrained
RegNetY-16GF teacher. The teacher checkpoint is downloaded on first use. It can
also be supplied as a local file or disabled explicitly:

```bash
TEACHER_PATH=/absolute/path/to/regnety_160-a5fe301d.pth \
  bash train_caspr_imagenet.sh casprnet_n

DISTILLATION_TYPE=none \
  bash train_caspr_imagenet.sh casprnet_n
```

For gradient accumulation on a memory-limited GPU, `BATCH_SIZE` is the
micro-batch resident on the GPU and `ACCUM_STEPS` is the number of
micro-batches per optimizer update. For example, this gives an effective batch
of `64 * 4 = 256` on one GPU:

```bash
GPU=0 BATCH_SIZE=64 ACCUM_STEPS=4 OUTPUT_DIR=checkpoints/casprnet_n \
  bash train_caspr_imagenet.sh casprnet_n
```

On distributed training, the effective global batch is
`BATCH_SIZE * ACCUM_STEPS * number_of_GPUs`. BatchNorm still observes only the
per-GPU micro-batch, not the accumulated effective batch.

Registered variants are `casprnet_n`, `casprnet_t`, and `casprnet_s`.
Use `--disable-caspr-metric` for the identity-metric ablation and
`--caspr-metric-interval N` to control metric sampling cost.

The channel group width is selected per stage rather than fixed globally. For
two grouped 1x1 factors separated by a perfect shuffle, a group width `b` can
cover at most `b^2` original channels. `coverage_group_sizes` selects the
smallest aligned divisor satisfying `b^2 >= C` for stage width `C`:

| Variant | Widths | Group widths |
|---|---|---|
| `casprnet_n` | `(48, 96, 192, 320)` | `(8, 16, 16, 32)` |
| `casprnet_t` | `(64, 128, 256, 384)` | `(8, 16, 16, 24)` |
| `casprnet_s` | `(64, 128, 320, 512)` | `(8, 16, 32, 32)` |

## Parameters, MACs and latency

```bash
python count_params.py --model casprnet_n --preset imagenet
python count_params.py --model casprnet_n --preset imagenet --deploy
python benchmark.py --model casprnet_n --device cuda --batch-size 1 --deploy
python benchmark_pair.py --model casprnet_n --device cpu --threads 1
```

At input `1x3x224x224`, the current single-head profiles
(`distillation=False`) are:

| Variant | Graph | Parameters | MACs |
|---|---|---:|---:|
| `casprnet_n` | train | 1.792M | 0.0643G |
| `casprnet_n` | deploy | **1.649M** | **0.0367G** |
| `casprnet_t` | train | 2.235M | 0.0930G |
| `casprnet_t` | deploy | 2.032M | 0.0522G |
| `casprnet_s` | train | 3.204M | 0.1847G |
| `casprnet_s` | deploy | 2.732M | 0.0970G |

These are architecture counts, not accuracy claims. ImageNet and COCO results
must be measured after training. Sparse grouped kernels are backend-sensitive,
so always report real device latency in addition to MACs. On the local CPU,
paired seven-round benchmarks use batch 1, 224x224 inputs, one CPU thread, 50
warm-up forwards, alternating graph order, and at least 1.5 seconds per trial:

| Variant | Unfused median | Deploy median | Speedup | Deploy throughput |
|---|---:|---:|---:|---:|
| `casprnet_n` | 24.0990 ms | 10.2105 ms | 2.360x | 97.94 image/s |
| `casprnet_t` | 48.3293 ms | 27.3374 ms | 1.768x | 36.58 image/s |
| `casprnet_s` | 70.1313 ms | 35.8095 ms | 1.959x | 27.93 image/s |

These are engineering measurements, not accuracy-matched SOTA claims.
Multi-thread scaling remains backend-sensitive because grouped 1x1 kernels and
channel shuffle are not fused by eager PyTorch.

All supported activations pass deploy-equivalence and finite-gradient backward
tests. Activation-specific latency must be remeasured after each architecture
change and activation accuracy requires a separately trained ImageNet
checkpoint, so stale latency numbers from the fixed-eight-group model are not
carried forward here.

## ONNX export

```bash
python export_onnx.py --model casprnet_n \
  --checkpoint checkpoints/casprnet_n/checkpoint_best.pth \
  --output casprnet_n.onnx --deploy --verify
```

Checkpoints from earlier architectures are intentionally incompatible because
the dense channel MLP and nonlinear shear are no longer part of CASPR-Net.
