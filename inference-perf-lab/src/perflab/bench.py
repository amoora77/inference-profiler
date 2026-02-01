import sys
import torch
from perflab.env import get_env_info
from perflab.metrics import compute_percentiles, compute_throughput, compute_samples_per_sec, get_peak_rss_mb
from perflab.models import get_model, is_vision_model, is_text_model
from perflab.preprocess import preprocess_vision_batch, create_text_input
from perflab.timing import Timer
from perflab.utils import append_jsonl


def run_benchmark(
    model_name,
    device="cpu",
    batch_size=1,
    iters=200,
    warmup=20,
    compile_mode="auto",
    threads=None,
    interop_threads=1,
    channels_last=False,
    quantize=False,
    out_path="results/runs.jsonl",
):
    if threads is not None:
        try:
            torch.set_num_threads(threads)
        except RuntimeError:
            pass
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = get_model(model_name, device, quantize=quantize, channels_last=channels_last)

    compile_enabled = False
    if compile_mode == "on" or (compile_mode == "auto" and sys.version_info >= (3, 8)):
        try:
            model = torch.compile(model)
            compile_enabled = True
        except Exception:
            pass

    is_vision = is_vision_model(model_name)
    is_text = is_text_model(model_name)

    warmup_timer = Timer()
    for _ in range(warmup):
        if is_vision:
            warmup_timer.start_segment("preprocess")
            inputs = preprocess_vision_batch(batch_size, device, channels_last)
            warmup_timer.end_segment()
            warmup_timer.start_segment("forward")
            with torch.no_grad():
                _ = model(inputs)
            warmup_timer.end_segment()
            warmup_timer.start_segment("postprocess")
            warmup_timer.end_segment()
        elif is_text:
            warmup_timer.start_segment("forward")
            inputs = create_text_input(batch_size, device=device)
            with torch.no_grad():
                _ = model(inputs)
            warmup_timer.end_segment()

    measure_timer = Timer()
    end_to_end_times = []

    for _ in range(iters):
        import time
        iter_start = time.perf_counter()

        if is_vision:
            measure_timer.start_segment("preprocess")
            inputs = preprocess_vision_batch(batch_size, device, channels_last)
            measure_timer.end_segment()

            measure_timer.start_segment("forward")
            with torch.no_grad():
                _ = model(inputs)
            measure_timer.end_segment()

            measure_timer.start_segment("postprocess")
            measure_timer.end_segment()

        elif is_text:
            measure_timer.start_segment("forward")
            inputs = create_text_input(batch_size, device=device)
            with torch.no_grad():
                _ = model(inputs)
            measure_timer.end_segment()

        iter_end = time.perf_counter()
        end_to_end_times.append((iter_end - iter_start) * 1000)

    warmup_totals = warmup_timer.get_totals()
    measure_means = measure_timer.get_means()
    measure_totals = measure_timer.get_totals()

    percentiles = compute_percentiles(end_to_end_times)
    total_requests = iters * batch_size
    total_time_ms = sum(end_to_end_times)
    throughput_rps = compute_throughput(total_requests, total_time_ms)
    samples_per_sec = compute_samples_per_sec(total_requests, total_time_ms)

    env_info = get_env_info()
    peak_rss = get_peak_rss_mb()

    result = {
        "model": model_name,
        "device": device,
        "batch_size": batch_size,
        "iters": iters,
        "warmup": warmup,
        "compile": compile_enabled,
        "threads": threads,
        "interop_threads": interop_threads,
        "channels_last": channels_last,
        "quantize": quantize,
        "warmup_ms_total": warmup_totals.get("forward", 0.0),
        "measured_ms_total": measure_totals.get("forward", 0.0),
        "forward_ms_per_batch": measure_means.get("forward", 0.0),
        "end_to_end_ms_per_batch": sum(end_to_end_times) / len(end_to_end_times),
        "latency_p50": percentiles["p50"],
        "latency_p90": percentiles["p90"],
        "latency_p95": percentiles["p95"],
        "latency_p99": percentiles["p99"],
        "throughput_rps": throughput_rps,
        "effective_samples_per_sec": samples_per_sec,
        "peak_rss_mb": peak_rss,
        "env": env_info,
    }

    if is_vision:
        result["preprocess_ms_per_batch"] = measure_means.get("preprocess", 0.0)
        result["postprocess_ms_per_batch"] = measure_means.get("postprocess", 0.0)

    append_jsonl(out_path, result)
    return result
