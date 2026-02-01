import argparse
import os
from perflab.bench import run_benchmark
from perflab.sweep import run_sweep
from perflab.report import generate_report


def cmd_bench(args):
    run_benchmark(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        iters=args.iters,
        warmup=args.warmup,
        compile_mode=args.compile,
        threads=args.threads,
        interop_threads=args.interop_threads,
        channels_last=args.channels_last == "on",
        quantize=args.quantize == "on",
        out_path=args.out,
    )
    print(f"Benchmark complete. Results appended to {args.out}")


def cmd_sweep(args):
    run_sweep(
        preset=args.preset,
        out_path=args.out,
        quick=args.quick,
    )


def cmd_report(args):
    generate_report(
        input_path=args.input,
        out_path=args.out,
        constraint=args.constraint,
        latency_pct=args.latency_pct,
        latency_budget_ms=args.latency_budget_ms,
    )


def main():
    parser = argparse.ArgumentParser(prog="perflab", description="Inference benchmarking toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench_parser = subparsers.add_parser("bench", help="Run a single benchmark")
    bench_parser.add_argument("--model", required=True, choices=["resnet18", "mobilenet_v3_small", "tiny_transformer"])
    bench_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    bench_parser.add_argument("--batch-size", type=int, default=1)
    bench_parser.add_argument("--iters", type=int, default=200)
    bench_parser.add_argument("--warmup", type=int, default=20)
    bench_parser.add_argument("--compile", default="auto", choices=["on", "off", "auto"])
    bench_parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() // 2)))
    bench_parser.add_argument("--interop-threads", type=int, default=1)
    bench_parser.add_argument("--channels-last", default="auto", choices=["on", "off", "auto"])
    bench_parser.add_argument("--quantize", default="off", choices=["on", "off"])
    bench_parser.add_argument("--out", default="results/runs.jsonl")
    bench_parser.set_defaults(func=cmd_bench)

    sweep_parser = subparsers.add_parser("sweep", help="Run a parameter sweep")
    sweep_parser.add_argument("--preset", required=True, choices=["cpu_vision", "cpu_text"])
    sweep_parser.add_argument("--out", default=None)
    sweep_parser.add_argument("--quick", action="store_true", help="Reduce sweep size for fast testing")
    sweep_parser.set_defaults(func=cmd_sweep)

    report_parser = subparsers.add_parser("report", help="Generate a report from benchmark results")
    report_parser.add_argument("--input", required=True)
    report_parser.add_argument("--out", default="reports/latest.md")
    report_parser.add_argument("--constraint", default="balanced", choices=["latency", "throughput", "balanced"])
    report_parser.add_argument("--latency-pct", default="p95", choices=["p95", "p99"])
    report_parser.add_argument("--latency-budget-ms", type=float, default=50.0)
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
