# ER-Net: Entanglement Renormalization Network

ER-Net is the default backbone of this repository. CASPR-Net is still
registered (`--model casprnet_*`) so it can be trained as a baseline.

## Idea in one sentence

Every backbone downsamples like a **tree tensor network**: it merges 2x2
blocks independently, so correlations that sit *on the block boundaries* are
discarded at every scale. Natural images have correlations at every scale
(they behave like critical systems), and the representation physics uses for
critical systems is **MERA**. MERA adds one operation that no backbone has: a
local unitary **disentangler** across block boundaries, applied *before* each
coarse-graining. ER-Net uses this disentangle-then-coarse-grain step at every
change of resolution.

## Where the quantum theory is (and is not)

ER-Net is **quantum-inspired**: no qubits, no simulated quantum circuits, no
quantum hardware. Quantum many-body theory fixes the *structure* and the
*initialization* of the weights; every computation is classical and the deploy
graph is plain convolutions. In papers, describe it as quantum-inspired /
entanglement-renormalization based, never as quantum computing or speedup.

| Quantum concept | Role in ER-Net | Code | Computed as |
|---|---|---|---|
| MERA (Vidal): disentanglers + isometries at every scale | every resolution change is one MERA layer | `RenormalizationStep` | disentangler then isometry |
| Disentangler: local unitary removing short-range entanglement | rotates each 2x2 block on the shifted grid, across tree-block boundaries | `Disentangler`, `block_padding` | depthwise 2x2 conv + depth-to-space |
| Unitary gates, SO(4) = SU(2) x SU(2) / Z2 | gate `x -> q_L x q_R` with unit quaternions: exactly orthogonal, 8 numbers per channel | `quaternion_left/right`, `Disentangler.unitary()` | 4x4 matrix products |
| Isometry: keeps chi relevant directions of a block | 2x2 -> 1 coarse-graining, `W W^T = I` at init | `Isometry`, `isometry_defect()` | 2x2/s2 conv |
| MERA <-> wavelet correspondence (Evenbly & White 2016) | each renormalization step starts as an exact Daubechies-4 step | `D4_*`, `HAAR_*` angles, `reset_parameters()` | weight initialization |
| Entanglement entropy, Schmidt truncation (DMRG / MERA bond dimension) | evidence that U removes entanglement; stage widths from chi(eps) | `tools/ernet_bond_dims.py` | eigenvalues of block covariances |

## Architecture

```text
input 224x224x3
stem     : 3x3/s2 conv-BN-act (C_stem)  ->  [ U(shifted grid) -> W(2x2->1) ]   56x56, C1
stage 1  : N1 x ERBlock(C1)                                                    56x56
renorm   : [ U(shifted) -> W ]                                                 28x28, C2
stage 2  : N2 x ERBlock(C2)
renorm   : [ U(shifted) -> W ]                                                 14x14, C3
stage 3  : N3 x ERBlock(C3)
renorm   : [ U(shifted) -> W ]                                                 7x7,   C4
stage 4  : N4 x ERBlock(C4)
head     : BN -> GAP -> Linear(head_dim) -> LN -> act -> FC (+ distillation FC)

ERBlock(x):  x <- x + ls * W_o[(W_v x) * act(DW3x3(W_g x))]     # gated contraction
             (--block-disentangle adds x <- U_parity(x) first: brick-wall circuit)
```

Exactly one disentangler layer per scale, as in MERA. `forward_intermediates`
returns the four stage outputs (strides 4/8/16/32) for FPN / detection necks.

### Primitives (`model/ernet/layers.py`)

| Module | What it is | Train-time parametrization | Deploy graph |
|---|---|---|---|
| `Disentangler` | per-channel rotation of each 2x2 block on the shifted grid (no channel grouping) | block = quaternion x, gate `x -> q_L x q_R` with unit quaternions (SO(4) = SU(2)xSU(2)/Z2): exactly orthogonal, 8 numbers per channel, no matrix exponential | depthwise 2x2/s2 conv (x4) + DepthToSpace |
| `Isometry` | 2x2 -> 1 coarse-graining | 2x2/s2 conv + BN, wavelet-band init | conv (BN folded) |
| `RenormalizationStep` | one MERA layer `W . U(shifted)` | as above | polyphase: depthwise 4x4/s2 (x4) + 1x1 conv for the 2 highest-resolution steps, factorized `U` + `W` for the others (exact, no parameter growth) |
| `GatedContraction` | multiplicative channel mixer | one fused value/gate conv + split | two convs, no Split (exact) |

The disentangler only mixes space; channel mixing is done by the dense 1x1
convs that follow (a Trotter-like alternation of space and channel gates).
It does not depend on channel ordering and has no grouping hyper-parameter.

### Wavelet initialization (MERA <-> wavelets)

Evenbly & White (PRL 116, 140403, 2016) showed that a binary MERA with
2-site rotations reproduces orthogonal wavelets. We verified numerically that
a shifted rotation at +pi/6 followed by an aligned rotation at -pi/12 gives
exactly the Daubechies-4 low/high-pass filters, and a single aligned rotation
at -pi/4 (no disentangler, a tree) gives Haar. In 2D the separable rotation
`R(t) x R(t)` of a block is `L(cos t, sin t, 0, 0) R(cos t, 0, sin t, 0)` in
quaternion form, so ER-Net initializes:

* every renormalization step as **one exact 2D D4 analysis step per channel**
  (low-low band of each channel first, then detail bands; verified in
  `test_ernet.py`),
* the tree ablation as an exact Haar step, so the ablation is equally
  "wavelet-initialized",
* optional in-block disentanglers near identity.

`--no-wavelet-init` gives random rotations and orthogonal W (ablation).

### Exact deploy collapse

`model.reparameterize()` (or `--deploy`) is function-preserving (max logit
error ~1e-6): collapse disentangler circuits, fold Conv-BN and layer scales,
split every gated value/gate conv, and put the stem and first-transition
renormalization steps into their **polyphase** form: `U` evaluated on the
aligned blocks as a depthwise 4x4/s2 conv (x4 channel multiplier), then `W`
as a 1x1 conv. This avoids DepthToSpace/Slice and their layout re-orders,
and keeps the weight count at `64 C_in + 4 C_in C_out` instead of the
`16 C_in C_out` of a dense 4x4 conv.

**The deploy model never has more parameters than the training model**
(checked for every variant and ablation in `test_ernet.py`, except the
`--block-disentangle` ablation: each in-block rotation is trained with 8
quaternion numbers but stored as a 4x4 matrix, with no BN next to it to fold).

| `ernet_n` deploy form of the renormalization steps | Params (train 2.161M) | ORT median |
|---|---:|---:|
| dense 4x4 conv, 2 steps (previous default, wrong: grows the model) | 2.171M | 5.70 ms |
| factorized `U` + DepthToSpace + `W`, all steps | 2.154M | 7.52 ms |
| **polyphase, 2 highest-resolution steps (default)** | **2.155M** | 6.35 ms |
| polyphase, all 4 steps | 2.162M | 6.20 ms |

`reparameterize(fuse_renormalization=True/False/n)` selects how many leading
steps use the polyphase form.

## Speed study (why the defaults are what they are)

ONNX Runtime CPU, 1 thread, batch 1, 224x224, Intel Xeon 2.8 GHz (container),
`ernet_n`:

| Change | ORT latency |
|---|---:|
| brick-wall U in every block, Split in graph (first version) | ~11.0 ms |
| + Split removed on deploy | 9.4 ms |
| + one disentangler per scale (MERA), polyphase deploy form (**default**) | **6.4 ms** |
| tree ablation (no disentangler) | 5.0 ms |

Profiling showed the brick-wall cost was not MACs but depthwise-multiplier
convs, DepthToSpace and the layout re-orders they force (about 5.6 ms of
12.8 ms). With one disentangler per scale in polyphase form, the MERA
default costs ~25% over the tree in ORT while keeping the parameter count.

## Variants (input 224, `distillation=False`)

| Variant | Widths | Depths | Train params | Deploy params | Deploy MACs | ORT ms |
|---|---|---|---:|---:|---:|---:|
| `ernet_n` | (24, 48, 96, 192) | (2, 2, 6, 2) | 2.161M | 2.155M | 0.154G | 6.4 |
| `ernet_t` | (32, 64, 128, 256) | (2, 2, 8, 2) | 3.464M | 3.454M | 0.296G | 8.7 |
| `ernet_s` | (48, 96, 192, 384) | (2, 3, 10, 2) | 6.752M | 6.734M | 0.773G | 18.5 |
| `casprnet_n` (baseline) | | | 1.792M | 1.649M | 0.037G | 4.0 |

`ernet_t` runs faster than the first `ernet_n` (brick-wall, ~11 ms) did: the
latency freed by the speed study can be reinvested in width/depth. These are
engineering numbers, not accuracy claims; re-measure on the target device.

## Commands

```bash
python test_ernet.py                                   # correctness suite
python train_synthetic.py --epochs 3 --device cpu      # train + exact collapse check
python count_params.py --model ernet_n [--deploy]
python benchmark_pair.py --model ernet_n --threads 1
python export_onnx.py --model ernet_n --output ernet_n.onnx --deploy --verify

bash train_ernet_imagenet.sh ernet_n                   # ImageNet-1K, distillation on
GPU=0 BATCH_SIZE=64 ACCUM_STEPS=4 bash train_ernet_imagenet.sh ernet_t
python main.py --eval --deploy --model ernet_n --resume <ckpt> --data-path <imagenet>
```

`main.py` options: `--activation`, `--circuit-depth`, `--local-kernel`
(0 disables the depthwise prior) and the ablations `--no-disentangle` (tree),
`--block-disentangle` (brick-wall in every block), `--no-wavelet-init` and
`--renormalization conv4x4` (unstructured 4x4/s2 conv).

## Bond dimensions (stage widths) from the truncation rule

```bash
python tools/ernet_bond_dims.py --model ernet_n --checkpoint <proxy ckpt> \
    --data-path <imagenet>/val --max-images 2000 --eps 1e-2 1e-3
```

Reports, per renormalization step, the block entropy before (`S_tree`) and
after (`S_mera`) the disentangler, `chi(eps)`, and suggested aligned widths.
`S_mera < S_tree` on real data is direct evidence for the core claim.

## Ablations that carry the paper

1. `--no-disentangle` vs default, same recipe (core claim; expect the largest
   gain on COCO AP_small).
2. `--renormalization conv4x4`: the same 4x4/s2 receptive field as `W . U`
   but as an unstructured conv (16 C_in C_out weights); isolates the value of
   the unitary/isometry factorization.
3. `--no-wavelet-init`: value of the D4 initialization.
4. `--block-disentangle` at equal latency (e.g. brick-wall `ernet_n` vs
   default `ernet_t`).
5. Truncation-rule widths vs doubling widths at equal MACs.

Checkpoints from CASPR-Net, or from earlier ER-Net versions (grouped
disentangler), are not compatible.
