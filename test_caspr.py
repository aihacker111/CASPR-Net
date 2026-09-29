"""Numerical tests for CASPR-Net sparse-product reparameterization."""
from __future__ import annotations

from copy import deepcopy

import torch

from model.casprnet import (
    SUPPORTED_ACTIVATIONS,
    CasprNet,
    CasprDenseStep,
    CasprSparseLevel,
    CasprSparseProduct,
    CasprSpatialGroup,
    casprnet_n,
    casprnet_s,
    casprnet_t,
    coverage_group_sizes,
)


def test_all_supported_activations() -> None:
    torch.manual_seed(5)
    x = torch.randn(1, 3, 32, 32)
    for activation in SUPPORTED_ACTIVATIONS:
        model = CasprNet(
            widths=(16, 32, 64, 96),
            depths=(1, 1, 1, 1),
            group_sizes=8,
            sparse_levels=1,
            spatial_branches=2,
            channel_branches=2,
            activation=activation,
            head_dim=128,
            stem_channels=8,
            num_classes=8,
            drop_path_rate=0.0,
            metric_enabled=False,
        ).eval()
        deployed = deepcopy(model).reparameterize()
        with torch.no_grad():
            expected = model(x)
            actual = deployed(x)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        assert actual.argmax(1).eq(expected.argmax(1)).all()


def test_spatial_group_exact_folding() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 8, 17, 19)
    group = CasprSpatialGroup(8, num_branches=3, metric_enabled=False).eval()
    deployed = deepcopy(group).eval().reparameterize()
    with torch.no_grad():
        expected = group(x)
        actual = deployed(x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert deployed.deploy_conv is not None
    assert deployed.deploy_conv.kernel_size == (3, 3)
    assert len(deployed.branches) == 0


def test_sparse_level_exact_folding() -> None:
    torch.manual_seed(9)
    x = torch.randn(2, 16, 11, 13)
    level = CasprSparseLevel(
        16, group_size=4, num_branches=3, metric_enabled=False).eval()
    deployed = deepcopy(level).eval().reparameterize()
    with torch.no_grad():
        expected = level(x)
        actual = deployed(x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert deployed.deploy_conv is not None
    assert deployed.deploy_conv.groups == 4
    assert len(deployed.branches) == 0


def test_sparse_product_global_dependency() -> None:
    """Every output must depend on multiple original channel groups."""
    product = CasprSparseProduct(
        16, group_size=4, levels=2, num_branches=1, metric_enabled=False).eval()
    for level in product.levels:
        branch = level.branches[0]
        branch.conv.weight.data.fill_(1.0)
        branch.norm.weight.data.fill_(1.0)
        branch.norm.bias.data.zero_()
        branch.norm.running_mean.zero_()
        branch.norm.running_var.fill_(1.0 - branch.norm.eps)
    x = torch.eye(16).reshape(16, 16, 1, 1)
    with torch.no_grad():
        jacobian_pattern = product(x).squeeze(-1).squeeze(-1).ne(0)
    assert int(jacobian_pattern.sum(dim=0).min()) >= 4
    assert int(jacobian_pattern.sum(dim=1).min()) >= 4


def test_variant_group_sizes_have_two_factor_coverage() -> None:
    expected = {
        casprnet_n: ((48, 96, 192, 320), (8, 16, 16, 32)),
        casprnet_t: ((64, 128, 256, 384), (8, 16, 16, 24)),
        casprnet_s: ((64, 128, 320, 512), (8, 16, 32, 32)),
    }
    for factory, (widths, group_sizes) in expected.items():
        assert coverage_group_sizes(widths) == group_sizes
        model = factory(num_classes=8, metric_enabled=False)
        actual = tuple(
            stage[0].channel_mixer.pre.group_size for stage in model.stages
        )
        assert actual == group_sizes
        assert all(width % size == 0 for width, size in zip(widths, actual))
        assert all(size * size >= width for width, size in zip(widths, actual))


def test_metrics_backward() -> None:
    torch.manual_seed(13)
    spatial = CasprDenseStep(
        8, kernel_size=3, metric_enabled=True, metric_update_interval=1).train()
    sparse = CasprSparseLevel(
        8, group_size=4, num_branches=1,
        metric_enabled=True, metric_update_interval=1).train()
    x = torch.randn(2, 8, 12, 12, requires_grad=True)
    sparse(spatial(torch.nn.functional.pad(x, (1, 1, 1, 1)))).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert spatial.weight.grad is not None and torch.isfinite(spatial.weight.grad).all()
    sparse_step = sparse.branches[0]
    assert sparse_step.weight.grad is not None and torch.isfinite(sparse_step.weight.grad).all()
    assert int(spatial.metric.num_updates) == 1
    assert int(sparse_step.metric.num_updates) == 1


def test_model_forward_and_folding() -> None:
    torch.manual_seed(11)
    model = casprnet_n(num_classes=1000, metric_enabled=False).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(x)
    deployed = deepcopy(model).eval().reparameterize()
    with torch.no_grad():
        actual = deployed(x)
    assert actual.shape == (1, 1000)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)


if __name__ == "__main__":
    test_all_supported_activations()
    print("[PASS] all supported activations preserve deploy equivalence")
    test_spatial_group_exact_folding()
    print("[PASS] exact spatial branch folding")
    test_sparse_level_exact_folding()
    print("[PASS] exact sparse channel-level folding")
    test_sparse_product_global_dependency()
    print("[PASS] inter-group channel connectivity")
    test_variant_group_sizes_have_two_factor_coverage()
    print("[PASS] stage-adaptive two-factor channel coverage")
    test_metrics_backward()
    print("[PASS] curvature-aligned metric backward")
    test_model_forward_and_folding()
    print("[PASS] full CASPR-Net forward and folding")
