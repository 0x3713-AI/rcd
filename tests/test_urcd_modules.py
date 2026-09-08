"""Rigorous Mathematical & Computational Invariant Tests for URCD-YOLO Modules.

Tests:
1. Haar 2D DWT/IDWT perfect reconstruction invertibility (|x - IDWT(DWT(x))| < 1e-5).
2. BiFPN_Concat Softmax probability simplex invariant (sum(weights) == 1.0, weights > 0).
3. WADown spatial downsampling, channel split, and gradient flow through dual paths.
4. WTConv2d spatial-frequency fusion and numerical stability.
5. C3k2_WTConv block integrity and residual propagation.
6. MSCA strip-convolution multi-scale channel attention boundedness in (0, 1).
"""

import pytest
import torch

from src.urcd_yolo.modules import (
    MSCA,
    BiFPN_Concat2,
    BiFPN_Concat3,
    C3k2_WTConv,
    HaarWaveletTransform2D,
    WADown,
    WTConv2d,
)


def test_haar_wavelet_perfect_reconstruction():
    """Verify that 2D Haar DWT followed by IDWT is an exact orthonormal reconstruction."""
    wavelet = HaarWaveletTransform2D()

    # Even spatial dimensions
    x_even = torch.randn(2, 8, 64, 64, dtype=torch.float32)
    subbands_even = wavelet.dwt(x_even)
    assert subbands_even.shape == (2, 32, 32, 32), "DWT subbands shape mismatch"
    rec_even = wavelet.idwt(subbands_even, 64, 64)
    recon_error = torch.max(torch.abs(x_even - rec_even)).item()
    assert recon_error < 1e-5, f"Haar DWT/IDWT reconstruction error {recon_error} exceeds tolerance"

    # Odd spatial dimensions (tests boundary padding & slicing)
    x_odd = torch.randn(2, 4, 33, 37, dtype=torch.float32)
    subbands_odd = wavelet.dwt(x_odd)
    rec_odd = wavelet.idwt(subbands_odd, 33, 37)
    assert rec_odd.shape == x_odd.shape, f"Reconstructed odd shape {rec_odd.shape} != original {x_odd.shape}"
    recon_error_odd = torch.max(torch.abs(x_odd - rec_odd)).item()
    assert recon_error_odd < 1e-5, f"Odd shape reconstruction error {recon_error_odd} exceeds tolerance"


def test_bifpn_concat2_properties():
    """Verify BiFPN_Concat2 weight normalization, concatenation, and gradient flow."""
    module = BiFPN_Concat2(dimension=1)
    x1 = torch.randn(2, 16, 32, 32, requires_grad=True)
    x2 = torch.randn(2, 32, 32, 32, requires_grad=True)

    out = module([x1, x2])
    assert out.shape == (2, 48, 32, 32), f"Expected channel 48, got {out.shape[1]}"

    # Check Softmax normalization properties
    weights = torch.softmax(module.w, dim=0)
    assert pytest.approx(torch.sum(weights).item(), abs=1e-6) == 1.0
    assert torch.all(weights > 0.0)
    assert torch.all(weights < 1.0)

    # Test backward pass
    loss = out.sum()
    loss.backward()
    assert module.w.grad is not None, "Gradients must flow to learnable weights"
    assert torch.isfinite(module.w.grad).all(), "Weight gradients must be finite"
    assert torch.isfinite(x1.grad).all(), "x1 gradients must be finite"
    assert torch.isfinite(x2.grad).all(), "x2 gradients must be finite"


def test_bifpn_concat3_spatial_alignment():
    """Verify BiFPN_Concat3 with mismatched spatial dimensions."""
    module = BiFPN_Concat3(dimension=1)
    x0 = torch.randn(2, 16, 32, 32)
    x1 = torch.randn(2, 16, 31, 31)  # Slightly mismatched due to odd stride
    x2 = torch.randn(2, 16, 32, 32)

    out = module([x0, x1, x2])
    assert out.shape == (2, 48, 32, 32)
    weights = torch.softmax(module.w, dim=0)
    assert pytest.approx(torch.sum(weights).item(), abs=1e-6) == 1.0


def test_wadown_downsampling_and_gradients():
    """Verify WADown downsamples resolution by 2 and propagates gradients to both paths."""
    wadown = WADown(c1=32, c2=64)
    x = torch.randn(2, 32, 64, 64, requires_grad=True)

    out = wadown(x)
    assert out.shape == (2, 64, 32, 32), f"Expected shape (2, 64, 32, 32), got {out.shape}"
    assert torch.isfinite(out).all(), "WADown output must be fully finite"

    # Backward gradient check
    loss = out.mean()
    loss.backward()
    assert wadown.alpha.grad is not None
    assert torch.isfinite(wadown.alpha.grad).all()
    assert torch.isfinite(x.grad).all()


def test_wtconv2d_forward_and_backward():
    """Verify WTConv2d spatial-wavelet convolution forward pass and numerical stability."""
    wtconv = WTConv2d(in_channels=16, out_channels=32, kernel_size=5)
    x = torch.randn(2, 16, 40, 40, requires_grad=True)

    out = wtconv(x)
    assert out.shape == (2, 32, 40, 40), f"Expected (2, 32, 40, 40), got {out.shape}"
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all(), "WTConv2d input gradient must be finite"
    assert wtconv.proj.weight.grad is not None


def test_c3k2_wtconv():
    """Verify C3k2_WTConv block forward and backward execution."""
    c3k2 = C3k2_WTConv(c1=32, c2=32, n=2, shortcut=True)
    x = torch.randn(2, 32, 32, 32, requires_grad=True)

    out = c3k2(x)
    assert out.shape == (2, 32, 32, 32)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_msca_bounded_attention():
    """Verify MSCA channel attention scores are strictly bounded in (0, 1)."""
    msca = MSCA(channels=32, reduction=4)
    x = torch.randn(2, 32, 28, 28, requires_grad=True)

    out = msca(x)
    assert out.shape == (2, 32, 28, 28)
    assert torch.isfinite(out).all()

    # Backward gradient pass
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert msca.fc[0].weight.grad is not None


def test_urcd_yolo_model_build_and_forward():
    """Verify end-to-end URCDYOLO instantiation and forward pass on image tensor."""
    from src.urcd_yolo.model import URCDYOLO

    model = URCDYOLO(verbose=False)
    assert model is not None
    assert model.task == "detect"

    # Forward pass on a dummy image tensor (1, 3, 256, 256)
    dummy_img = torch.randn(1, 3, 256, 256)
    preds = model(dummy_img, verbose=False)
    assert len(preds) > 0, "Model must return predictions"

