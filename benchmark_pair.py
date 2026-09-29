"""Paired end-to-end latency benchmark for CASPR train and deploy graphs."""
from __future__ import annotations

import argparse
import copy
import statistics

import torch
from torch.utils.benchmark import Timer

from model.casprnet import casprnet_n, casprnet_s, casprnet_t


FACTORIES = {
    "casprnet_n": casprnet_n,
    "casprnet_t": casprnet_t,
    "casprnet_s": casprnet_s,
}


class Runner:
    def __init__(self, model: torch.nn.Module, inputs: torch.Tensor):
        self.model = model
        self.inputs = inputs

    @torch.inference_mode()
    def __call__(self):
        return self.model(self.inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def cuda_round(runner: Runner, iterations: int, device: torch.device) -> float:
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        samples.append((start, end))
    synchronize(device)
    return statistics.median(start.elapsed_time(end) for start, end in samples)


def main() -> None:
    parser = argparse.ArgumentParser("Paired full-model CASPR benchmark")
    parser.add_argument("--model", choices=sorted(FACTORIES), default="casprnet_n")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-run-time", type=float, default=1.5)
    parser.add_argument("--cuda-iters", type=int, default=300)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    torch.manual_seed(123)
    unfused = FACTORIES[args.model](metric_enabled=False).to(device).eval()
    deploy = copy.deepcopy(unfused).reparameterize().to(device)
    inputs = torch.randn(
        args.batch_size, 3, args.resolution, args.resolution, device=device)
    runners = {
        "unfused": Runner(unfused, inputs),
        "deploy": Runner(deploy, inputs),
    }
    for runner in runners.values():
        for _ in range(args.warmup):
            runner()
    synchronize(device)

    samples = {"unfused": [], "deploy": []}
    for repeat in range(args.repeats):
        order = ("unfused", "deploy") if repeat % 2 == 0 else ("deploy", "unfused")
        for name in order:
            if device.type == "cuda":
                value = cuda_round(runners[name], args.cuda_iters, device)
            else:
                measurement = Timer(
                    stmt="runner()",
                    globals={"runner": runners[name]},
                    num_threads=args.threads,
                ).blocked_autorange(min_run_time=args.min_run_time)
                value = measurement.median * 1000.0
            samples[name].append(value)

    before = statistics.median(samples["unfused"])
    after = statistics.median(samples["deploy"])
    print(f"model={args.model} device={device} batch={args.batch_size}")
    print("unfused_trials_ms", [round(value, 4) for value in samples["unfused"]])
    print("deploy_trials_ms", [round(value, 4) for value in samples["deploy"]])
    print(f"unfused_median_ms={before:.4f}")
    print(f"deploy_median_ms={after:.4f}")
    print(f"speedup={before / after:.4f}x")
    print(f"latency_reduction={100.0 * (1.0 - after / before):.2f}%")


if __name__ == "__main__":
    main()
