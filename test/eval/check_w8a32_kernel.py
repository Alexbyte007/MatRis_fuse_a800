import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant.layers import FakeQuantLinear, PackedW8A32Linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check W8A32 packed Linear against fake quant Linear.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--in-features", type=int, default=128)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--batch", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    linear = torch.nn.Linear(args.in_features, args.out_features, bias=True).to(args.device)
    fake = FakeQuantLinear(linear, "check.fake").to(args.device)
    packed_kernel = PackedW8A32Linear(linear, "check.kernel", use_cuda_kernel=True).to(args.device)
    packed_fallback = PackedW8A32Linear(linear, "check.fallback", use_cuda_kernel=False).to(args.device)

    x = torch.randn(args.batch, args.in_features, device=args.device, requires_grad=True)
    y_fake = fake(x)
    y_kernel = packed_kernel(x)
    y_fallback = packed_fallback(x)

    grad_probe = torch.randn_like(y_fake)
    grad_fake = torch.autograd.grad(y_fake, x, grad_probe, retain_graph=True, create_graph=True)[0]
    grad_kernel = torch.autograd.grad(y_kernel, x, grad_probe, retain_graph=True, create_graph=True)[0]
    grad_fallback = torch.autograd.grad(y_fallback, x, grad_probe, retain_graph=True, create_graph=True)[0]

    metrics = {
        "fake_vs_kernel_forward_max_abs": (y_fake - y_kernel).abs().max().item(),
        "fake_vs_fallback_forward_max_abs": (y_fake - y_fallback).abs().max().item(),
        "fake_vs_kernel_grad_max_abs": (grad_fake - grad_kernel).abs().max().item(),
        "fake_vs_fallback_grad_max_abs": (grad_fake - grad_fallback).abs().max().item(),
    }
    for name, value in metrics.items():
        print(f"{name}={value:.8e}")
        if value > args.atol:
            raise SystemExit(f"{name} exceeded atol={args.atol}")


if __name__ == "__main__":
    main()
