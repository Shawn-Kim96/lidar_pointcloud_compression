import argparse
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.autoencoder import RangeCompressionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage1 backbone capacity and runtime")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--num_stages", type=int, default=4)
    parser.add_argument("--blocks_per_stage", type=int, default=1)
    parser.add_argument("--norm", type=str, default="batch", choices=["batch", "group", "none"])
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "silu", "gelu", "leaky_relu"],
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--measure_iters", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class MacCounter:
    macs: int = 0

    def add(self, value: int) -> None:
        self.macs += int(value)


def _conv2d_macs(module: nn.Conv2d, out: torch.Tensor) -> int:
    b, cout, hout, wout = out.shape
    cin = module.in_channels
    kh, kw = module.kernel_size
    groups = module.groups
    mac_per_out = (cin // groups) * kh * kw
    return b * cout * hout * wout * mac_per_out


def _conv_transpose2d_macs(module: nn.ConvTranspose2d, out: torch.Tensor) -> int:
    b, cout, hout, wout = out.shape
    cin = module.in_channels
    kh, kw = module.kernel_size
    groups = module.groups
    mac_per_out = (cin // groups) * kh * kw
    return b * cout * hout * wout * mac_per_out


def _linear_macs(module: nn.Linear, out: torch.Tensor) -> int:
    if out.dim() == 1:
        batch_items = 1
        out_features = out.shape[0]
    elif out.dim() == 2:
        batch_items, out_features = out.shape
    else:
        batch_items = out.numel() // out.shape[-1]
        out_features = out.shape[-1]
    return batch_items * out_features * module.in_features


def estimate_macs(model: nn.Module, sample: torch.Tensor) -> int:
    counter = MacCounter()
    handles = []

    def hook(module: nn.Module, _inputs, output):
        if isinstance(output, (tuple, list)):
            if not output:
                return
            out = output[0]
        else:
            out = output
        if not torch.is_tensor(out):
            return

        if isinstance(module, nn.Conv2d):
            counter.add(_conv2d_macs(module, out))
        elif isinstance(module, nn.ConvTranspose2d):
            counter.add(_conv_transpose2d_macs(module, out))
        elif isinstance(module, nn.Linear):
            counter.add(_linear_macs(module, out))

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        model(sample, noise_std=0.0, quantize=True)

    for h in handles:
        h.remove()
    return counter.macs


def measure_inference_ms(model: nn.Module, sample: torch.Tensor, warmup: int, iters: int, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(sample, noise_std=0.0, quantize=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            times_ms = []
            for _ in range(max(1, iters)):
                start_event.record()
                model(sample, noise_std=0.0, quantize=True)
                end_event.record()
                torch.cuda.synchronize(device)
                times_ms.append(float(start_event.elapsed_time(end_event)))
            return sum(times_ms) / len(times_ms)

        times_ms = []
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            model(sample, noise_std=0.0, quantize=True)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
        return sum(times_ms) / len(times_ms)


def format_int(num: int) -> str:
    return f"{num:,}"


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    model = RangeCompressionModel(
        quant_bits=args.quant_bits,
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        blocks_per_stage=args.blocks_per_stage,
        norm=args.norm,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    sample = torch.randn(args.batch_size, args.in_channels, args.height, args.width, device=device)
    with torch.no_grad():
        latent = model.encoder(sample)
        recon, latent_codes = model(sample, noise_std=0.0, quantize=True)

    total_params, total_trainable = count_params(model)
    enc_params, enc_trainable = count_params(model.encoder)
    dec_params, dec_trainable = count_params(model.decoder)
    quant_params, quant_trainable = count_params(model.quantizer)
    macs = estimate_macs(model, sample)
    avg_ms = measure_inference_ms(
        model=model,
        sample=sample,
        warmup=args.warmup_iters,
        iters=args.measure_iters,
        device=device,
    )

    model_info: Dict[str, str] = {
        "device": str(device),
        "input_shape": str(list(sample.shape)),
        "latent_shape": str(list(latent.shape)),
        "recon_shape": str(list(recon.shape)),
        "latent_codes_shape": str(list(latent_codes.shape)),
    }
    param_info: Dict[str, Tuple[int, int]] = {
        "encoder": (enc_params, enc_trainable),
        "decoder": (dec_params, dec_trainable),
        "quantizer": (quant_params, quant_trainable),
        "total": (total_params, total_trainable),
    }

    print("=== Stage1 Backbone Audit ===")
    for key, value in model_info.items():
        print(f"{key}: {value}")
    print("")
    print("Parameters (total / trainable):")
    for key, (total, trainable) in param_info.items():
        print(f"  {key:8s}: {format_int(total)} / {format_int(trainable)}")
    print("")
    print(f"Approx MACs per forward: {format_int(macs)}")
    print(f"Avg inference latency: {avg_ms:.3f} ms per batch (batch_size={args.batch_size})")


if __name__ == "__main__":
    main()
