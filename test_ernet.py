"""Numerical tests for ER-Net (Entanglement Renormalization Network).

Run:  python test_ernet.py
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from model.ernet import (SUPPORTED_ACTIVATIONS, Disentangler, ERNet,
                         Isometry, RenormalizationStep, ernet_n, ernet_s,
                         ernet_t)


def _randomize_batchnorm(model: nn.Module) -> None:
    """Non-trivial BN statistics so that folding is actually exercised."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 2.0)
            module.weight.data.uniform_(0.5, 1.5)
            module.bias.data.uniform_(-0.2, 0.2)


def _debug_model(**kwargs) -> ERNet:
    config = dict(widths=(16, 32, 64, 96), depths=(1, 2, 2, 1), mlp_ratios=2,
                  block_disentangle=True,
                  head_dim=128, stem_channels=8, num_classes=8,
                  drop_path_rate=0.0, layer_scale_init=0.5)
    config.update(kwargs)
    return ERNet(**config)


def test_disentangler_is_exactly_unitary() -> None:
    torch.manual_seed(0)
    gate = Disentangler(32, parity=1, circuit_depth=3, init="random")
    unitary = gate.unitary()
    eye = torch.eye(4).expand_as(unitary)
    torch.testing.assert_close(unitary @ unitary.transpose(-1, -2), eye,
                               atol=1e-5, rtol=0)
    # Norm preservation on a map whose block grid needs no padding.
    x = torch.randn(2, 32, 8, 8)
    y = Disentangler(32, parity=0)(x)
    torch.testing.assert_close(y.norm(), x.norm(), atol=1e-4, rtol=1e-5)
    # Proper rotations only (SO(4)), one gate per channel.
    assert unitary.shape == (32, 4, 4)
    assert torch.allclose(torch.linalg.det(unitary), torch.ones(32), atol=1e-5)


def test_disentangler_acts_across_block_boundaries() -> None:
    """The shifted gate must couple sites that a tree step keeps apart."""
    torch.manual_seed(1)
    gate = Disentangler(8, parity=1, init="random").eval()
    x = torch.zeros(1, 8, 8, 8, requires_grad=True)
    output = gate(x)
    # Output site (2, 2) belongs to shifted block rows/cols {1, 2}.
    output[0, :, 2, 2].sum().backward()
    support = x.grad[0].abs().sum(0).ne(0)
    # The gate never mixes channels: channel 0 only depends on channel 0.
    x.grad = None
    gate(x)[0, 0].sum().backward()
    assert x.grad[0, 1:].abs().sum() == 0
    expected = torch.zeros(8, 8, dtype=torch.bool)
    expected[1:3, 1:3] = True
    assert torch.equal(support, expected), support
    # Rows 1 and 2 lie in different parity-0 (tree) blocks.
    assert support[1, 1] and support[2, 2]


def test_disentangler_deploy_equivalence_all_sizes() -> None:
    torch.manual_seed(2)
    for parity in (0, 1):
        gate = Disentangler(16, parity=parity, circuit_depth=4, init="random").eval()
        deployed = deepcopy(gate).reparameterize()
        assert deployed.quaternions is None
        for height, width in ((8, 8), (7, 9), (13, 6), (1, 1)):
            x = torch.randn(2, 16, height, width)
            with torch.no_grad():
                torch.testing.assert_close(deployed(x), gate(x),
                                           atol=2e-5, rtol=2e-5)


def test_renormalization_step_polyphase_form() -> None:
    torch.manual_seed(3)
    step = RenormalizationStep(16, 24).eval()
    _randomize_batchnorm(step)
    reference = deepcopy(step)
    train_params = sum(p.numel() for p in step.parameters())
    step.disentangler.reparameterize()
    step.isometry.projection.reparameterize()
    step.fuse()
    assert step.phase_conv.kernel_size == (4, 4) and step.phase_conv.stride == (2, 2)
    assert step.phase_conv.groups == 16 and step.phase_conv.out_channels == 64
    assert step.mix_conv.kernel_size == (1, 1)
    # Never the dense 16 * C_in * C_out collapse.
    deploy_params = sum(p.numel() for p in step.parameters())
    assert deploy_params < 16 * 16 * 24
    assert deploy_params - train_params <= 56 * 16
    for height, width in ((16, 16), (15, 15), (14, 9), (3, 3)):
        x = torch.randn(2, 16, height, width)
        with torch.no_grad():
            expected = reference(x)
            actual = step(x)
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_wavelet_initialisation_is_daubechies4() -> None:
    """At init a renormalization step is one separable 2D D4 analysis step."""
    import math
    step = RenormalizationStep(2, 8).eval()
    step.disentangler.init_noise = 0.0
    step.disentangler.reset_parameters()
    norm = step.isometry.projection[1]
    norm.running_mean.zero_()
    norm.running_var.fill_(1.0 - norm.eps)
    step.disentangler.reparameterize()
    step.isometry.projection.reparameterize()
    s3 = math.sqrt(3.0)
    h = torch.tensor([1 + s3, 3 + s3, 3 - s3, 1 - s3]) / (4 * math.sqrt(2.0))
    g = h.flip(0) * torch.tensor([1.0, -1.0, 1.0, -1.0])
    weight, _ = step.effective_kernel()
    torch.testing.assert_close(weight[0, 0], torch.outer(h, h), atol=1e-6, rtol=0)
    torch.testing.assert_close(weight[6, 0].abs(), torch.outer(g, g).abs(),
                               atol=1e-6, rtol=0)
    assert weight[0, 1].abs().max() == 0          # channel-separable at init
    tree = RenormalizationStep(2, 8, disentangle=False)
    torch.testing.assert_close(tree.isometry.projection.conv.weight[0, 0],
                               torch.full((2, 2), 0.5))   # Haar


def test_isometry_initialisation() -> None:
    torch.manual_seed(4)
    iso = Isometry(16, 48)
    with torch.no_grad():
        assert float(iso.isometry_defect()) < 1e-5


def test_all_activations_deploy_equivalence() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 3, 64, 64)
    for activation in SUPPORTED_ACTIVATIONS:
        model = _debug_model(activation=activation).eval()
        _randomize_batchnorm(model)
        deployed = deepcopy(model).reparameterize()
        with torch.no_grad():
            expected = model(x)
            actual = deployed(x)
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)


def test_feature_maps_and_odd_resolutions() -> None:
    torch.manual_seed(6)
    model = _debug_model().eval()
    _randomize_batchnorm(model)
    deployed = model.deploy_copy()
    for size in (224, 97, 64, 33):
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            features = model.forward_intermediates(x)
            features_deploy = deployed.forward_intermediates(x)
        strides = [round(size / f.shape[-1]) for f in features]
        assert [f.shape[1] for f in features] == list(model.widths)
        if size == 224:
            assert strides == [4, 8, 16, 32], strides
        for a, b in zip(features, features_deploy):
            torch.testing.assert_close(b, a, rtol=5e-5, atol=5e-5)


def test_ablations_build_and_fold() -> None:
    torch.manual_seed(7)
    x = torch.randn(1, 3, 64, 64)
    for kwargs in (dict(disentangle=False), dict(block_disentangle=True),
                   dict(local_kernel=0), dict(circuit_depth=3),
                   dict(wavelet_init=False),
                   dict(renormalization="conv4x4")):
        model = _debug_model(**kwargs).eval()
        result = model.compare_reparameterization(x, atol=5e-5, rtol=5e-5)
        assert result["passed"], (kwargs, result)
    from model.ernet import GatedContraction
    deployed = _debug_model().eval().reparameterize()
    assert all(m.is_split for m in deployed.modules()
               if isinstance(m, GatedContraction))
    tree = _debug_model(disentangle=False)
    assert not any(isinstance(m, Disentangler) for m in tree.modules())


def test_backward_and_no_weight_decay() -> None:
    torch.manual_seed(8)
    model = _debug_model(distillation=True).train()
    x = torch.randn(4, 3, 32, 32)
    logits, dist_logits = model(x)
    (logits.square().mean() + dist_logits.square().mean()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    skip = model.no_weight_decay()
    assert any(name.endswith("quaternions") for name in skip)
    model.eval()
    assert model(x).shape == (4, 8)


def test_deploy_never_adds_parameters() -> None:
    """Reparameterization must not grow the model (default and ablations).

    Only ``block_disentangle=True`` is exempt: each in-block rotation is
    trained with 8 quaternion numbers but stored as a 4x4 matrix (16), and
    there is no BatchNorm next to it to fold away.
    """
    for factory in (ernet_n, ernet_t, ernet_s):
        for kwargs in (dict(), dict(disentangle=False),
                       dict(renormalization="conv4x4"), dict(wavelet_init=False)):
            model = factory(num_classes=1000, **kwargs)
            train = sum(p.numel() for p in model.parameters())
            deploy = sum(p.numel() for p in model.eval().reparameterize().parameters())
            assert deploy <= train, (factory.__name__, kwargs, train, deploy)


def test_headless_backbone_and_distilled_loading() -> None:
    """Distillation uses a headless ERNet; its weights load into a full model."""
    import os
    import tempfile
    from model.ernet.lightly import ERNetDistillWrapper, load_distilled_weights
    torch.manual_seed(12)
    backbone = ernet_n(headless=True).eval()
    assert backbone.headless and backbone.num_classes == 0
    assert not any(k.startswith(("head.", "head_projection."))
                   for k in backbone.state_dict())
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert backbone(x).shape == (2, backbone.widths[-1])
    assert backbone.compare_reparameterization(x, atol=1e-4, rtol=1e-4)["passed"]
    # The distillation wrapper strips the head of a full model as well.
    wrapper = ERNetDistillWrapper(ernet_n(), feature_levels=(3, 4))
    assert wrapper.get_model().headless
    with torch.no_grad():
        features = wrapper.forward_features(x)["features"]
    assert features.shape[1] == wrapper.feature_dim() == 96 + 192
    # Backbone weights -> full ERNet: everything but the head is copied.
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "distilled.pth")
        torch.save({"model": backbone.state_dict()}, path)
        full = load_distilled_weights(ernet_n(num_classes=10), path)
    for key, value in backbone.state_dict().items():
        assert torch.equal(full.state_dict()[key], value), key
    assert full(x).shape == (2, 10)


def test_variants() -> None:
    torch.manual_seed(9)
    x = torch.randn(1, 3, 64, 64)
    for factory in (ernet_n, ernet_t, ernet_s):
        model = factory(num_classes=10).eval()
        result = model.compare_reparameterization(x, atol=1e-4, rtol=1e-4)
        assert result["passed"], (factory.__name__, result)


def test_onnx_export_matches() -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed, ONNX test skipped)")
        return
    import io

    torch.manual_seed(10)
    model = _debug_model().eval()
    _randomize_batchnorm(model)
    model.reparameterize()
    x = torch.randn(1, 3, 64, 64)
    buffer = io.BytesIO()
    torch.onnx.export(model, x, buffer, opset_version=17,
                      input_names=["images"], output_names=["logits"],
                      dynamo=False)
    session = ort.InferenceSession(buffer.getvalue(),
                                   providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": x.numpy()})[0]
    with torch.no_grad():
        expected = model(x).numpy()
    assert abs(actual - expected).max() < 1e-4
    ops = {node.op_type for node in __import__("onnx").load_from_string(
        buffer.getvalue()).graph.node}
    print(f"  ONNX ops: {sorted(ops)}")


if __name__ == "__main__":
    tests = [
        ("disentangler is exactly unitary", test_disentangler_is_exactly_unitary),
        ("disentangler couples sites across tree-block boundaries",
         test_disentangler_acts_across_block_boundaries),
        ("disentangler deploy collapse, all sizes",
         test_disentangler_deploy_equivalence_all_sizes),
        ("renormalization step polyphase deploy form",
         test_renormalization_step_polyphase_form),
        ("isometry initialisation", test_isometry_initialisation),
        ("wavelet init = Daubechies-4 (MERA) / Haar (tree)",
         test_wavelet_initialisation_is_daubechies4),
        ("all activations preserve deploy equivalence",
         test_all_activations_deploy_equivalence),
        ("feature pyramid strides and odd resolutions",
         test_feature_maps_and_odd_resolutions),
        ("ablation variants build and fold", test_ablations_build_and_fold),
        ("backward, distillation head, no-weight-decay set",
         test_backward_and_no_weight_decay),
        ("ernet_n / ernet_t / ernet_s deploy equivalence", test_variants),
        ("headless backbone + distilled weight loading",
         test_headless_backbone_and_distilled_loading),
        ("deploy never adds parameters", test_deploy_never_adds_parameters),
        ("ONNX export matches PyTorch", test_onnx_export_matches),
    ]
    for name, test in tests:
        test()
        print(f"[PASS] {name}")
