import itertools
import sys
from perflab.bench import run_benchmark
from perflab.utils import timestamp_str


def get_sweep_configs(preset, quick=False):
    configs = []

    compile_options = ["off"]
    if sys.version_info >= (3, 8):
        compile_options.append("on")

    if preset == "cpu_vision":
        models = ["resnet18", "mobilenet_v3_small"]
        batch_sizes = [1, 4, 16] if quick else [1, 2, 4, 8, 16]
        threads = [2, 8] if quick else [1, 2, 4, 8]
        channels_last_options = ["off", "on"]

        for model, bs, t, comp, cl in itertools.product(
            models, batch_sizes, threads, compile_options, channels_last_options
        ):
            configs.append({
                "model_name": model,
                "device": "cpu",
                "batch_size": bs,
                "threads": t,
                "compile_mode": comp,
                "channels_last": cl == "on",
                "quantize": False,
            })

    elif preset == "cpu_text":
        batch_sizes = [1, 8, 32] if quick else [1, 2, 4, 8, 16, 32]
        threads = [2, 8] if quick else [1, 2, 4, 8]
        quantize_options = ["off", "on"]

        for bs, t, comp, q in itertools.product(
            batch_sizes, threads, compile_options, quantize_options
        ):
            configs.append({
                "model_name": "tiny_transformer",
                "device": "cpu",
                "batch_size": bs,
                "threads": t,
                "compile_mode": comp,
                "channels_last": False,
                "quantize": q == "on",
            })

    return configs


def run_sweep(preset, out_path=None, quick=False, iters=200, warmup=20):
    if out_path is None:
        out_path = f"results/sweeps/{timestamp_str()}_{preset}.jsonl"

    configs = get_sweep_configs(preset, quick=quick)
    total = len(configs)

    for i, config in enumerate(configs, 1):
        print(f"[{i}/{total}] Running: {config}")
        run_benchmark(
            iters=iters,
            warmup=warmup,
            out_path=out_path,
            **config,
        )

    print(f"Sweep complete. Results saved to {out_path}")
    return out_path
