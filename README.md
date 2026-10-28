# ER-Net: Entanglement Renormalization Network

Edge vision backbone for ImageNet classification and COCO detection, designed
from the entanglement renormalization ansatz (MERA) of quantum many-body
physics. **Quantum-inspired**: the theory fixes the structure and the
initialization of the weights; all computation is classical and the deployed
model is plain convolutions.

Core idea: standard backbones downsample like a tree (independent 2x2
merges), discarding correlations on block boundaries. ER-Net applies a local
unitary **disentangler** across block boundaries *before* every
coarse-graining, at every scale.

Full design, measurements and ablations: **[ERNET.md](ERNET.md)**.
Previous backbone, kept as a baseline: [CASPR.md](CASPR.md).

## What is new

* Disentangle-then-coarse-grain at every resolution change (stem + 3 stage
  transitions), exactly one disentangler layer per scale as in MERA.
* Per-channel quaternion disentangler `x -> q_L x q_R`: exactly orthogonal,
  8 numbers per channel, no channel-group hyper-parameter.
* Daubechies-4 wavelet initialization from the MERA <-> wavelet
  correspondence (the tree ablation starts as Haar).
* Exact deploy collapse that never grows the model (polyphase form, gate
  split, BN / layer-scale folding); standard-op ONNX graph.
* `tools/ernet_bond_dims.py`: block entanglement entropy before/after the
  disentangler and truncation-rule stage widths.

No accuracy results yet: ImageNet / COCO numbers must come from training.

## Models (224x224, deploy graph, ONNX Runtime CPU 1 thread)

| Model | Params | MACs | Latency |
|---|---:|---:|---:|
| `ernet_n` | 2.16M | 0.154G | 6.4 ms |
| `ernet_t` | 3.45M | 0.296G | 8.7 ms |
| `ernet_s` | 6.73M | 0.773G | 18.5 ms |

## Quick start

```bash
pip install -r requirements.txt
python test_ernet.py                               # correctness suite
python train_synthetic.py --epochs 3 --device cpu  # tiny train + exact deploy check
python count_params.py --model ernet_n --deploy
python export_onnx.py --model ernet_n --output ernet_n.onnx --deploy --verify

# ImageNet-1K (RegNetY-16GF hard distillation by default)
DATA_PATH=/path/to/imagenet bash train_ernet_imagenet.sh ernet_n   # expects train/<wnid>/, val/<wnid>/
python main.py --eval --deploy --model ernet_n --resume <ckpt> --data-path <imagenet>
```

### Kaggle ("ImageNet Object Localization Challenge" data)

Its validation images are in one flat folder with detection-style labels
(`LOC_val_solution.csv`). Build classification folders once (symlinks, no
extra disk), then train:

```bash
python tools/prepare_kaggle_imagenet.py \
    --kaggle-root /kaggle/input/imagenet-object-localization-challenge \
    --out /kaggle/working/imagenet
DATA_PATH=/kaggle/working/imagenet NUM_WORKERS=4 BATCH_SIZE=128 ACCUM_STEPS=2 \
    bash train_ernet_imagenet.sh ernet_n
```

The script checks 1000 classes, 50,000 val images (50 per class) and that
train/val class indices match. Teacher distillation downloads RegNetY-16GF,
so enable internet or pass `DISTILLATION_TYPE=none`.

Ablations (`main.py`): `--no-disentangle` (tree), `--renormalization conv4x4`,
`--no-wavelet-init`, `--block-disentangle`, `--circuit-depth k`.

## Layout

```text
model/ernet/      ER-Net (layers.py: Disentangler, Isometry, RenormalizationStep,
                  GatedContraction, ERBlock; network.py: ERNet; variants.py)
model/casprnet/   CASPR-Net baseline
main.py, engine.py, train_ernet_imagenet.sh    ImageNet training / evaluation
test_ernet.py, train_synthetic.py              correctness checks
count_params.py, benchmark*.py, export_onnx.py profiling and export
tools/ernet_bond_dims.py                       entanglement / bond-dimension analysis
```
