import torch

from models.afem import AFEM


def test_afem() -> None:
    afem = AFEM(in_channels=512, num_groups=32)
    dummy = torch.randn(4, 512, requires_grad=True)

    out = afem(dummy)
    assert out.shape == (4, 512), f"AFEM output shape mismatch: {out.shape}"

    assert isinstance(afem.norm, torch.nn.BatchNorm1d), "AFEM must use BatchNorm1d"
    assert afem.group_scale.shape == (32,), f"Group scale shape mismatch: {afem.group_scale.shape}"
    assert afem.group_linear.weight.shape == (32, 16, 16)

    loss = out.sum()
    loss.backward()
    assert dummy.grad is not None, "AFEM detached gradient"
    assert not torch.isnan(dummy.grad).any(), "AFEM gradient has NaN"
    print("OK: AFEM")


if __name__ == "__main__":
    test_afem()
