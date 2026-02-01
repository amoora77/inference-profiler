# inference-perf-lab

Benchmark PyTorch models and figure out the best serving config. Measures p50/p90/p95/p99 latency instead of just averages, because tail latency is what actually matters in production.

Sweeps batch sizes, torch.compile, thread counts, and other knobs, then tells you which config to use based on your latency budget.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For tests and optional dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

Run a single benchmark:
```bash
perflab bench --model resnet18 --batch-size 1
```

Run a quick parameter sweep:
```bash
perflab sweep --preset cpu_vision --quick
```

Generate a report from sweep results:
```bash
perflab report --input results/sweeps/<file>.jsonl
```

## Usage

### perflab bench

Benchmark a single config.

```bash
perflab bench \
  --model resnet18 \
  --device cpu \
  --batch-size 4 \
  --iters 200 \
  --warmup 20 \
  --compile auto \
  --threads 4 \
  --out results/runs.jsonl
```

**Models**: `resnet18`, `mobilenet_v3_small`, `tiny_transformer`

**Key options**:
- `--compile`: Enable torch.compile (auto/on/off)
- `--threads`: Number of intra-op threads
- `--channels-last`: Use channels_last memory format (vision only)
- `--quantize`: Apply dynamic quantization (text only)

### perflab sweep

Run a bunch of configs and dump results to JSONL.

```bash
perflab sweep --preset cpu_vision
perflab sweep --preset cpu_text --quick
```

**Presets**:
- `cpu_vision`: resnet18 and mobilenet_v3_small with various batch sizes, threads, compile, channels_last
- `cpu_text`: tiny_transformer with various batch sizes, threads, compile, quantization

Add `--quick` for a smaller sweep (faster iteration during dev).

### perflab report

Turn JSONL results into a markdown report with plots and recommendations.

```bash
perflab report \
  --input results/sweeps/20260130_143022_cpu_vision.jsonl \
  --out reports/latest.md \
  --constraint balanced \
  --latency-pct p95 \
  --latency-budget-ms 50
```

**Constraints**:
- `latency`: best throughput within your latency budget
- `throughput`: highest throughput, latency be damned
- `balanced`: best throughput/p95 ratio

## How it works

**Warmup**: First 20 iterations warm up JIT/caching, then we measure 200 iterations with consistent shapes.

**Percentiles**: p95 means 95% of requests finish within that time. Much more useful than averages when 5% of your users are getting screwed by tail latency.

**Batch size tradeoffs**: Batch 1 = lowest latency. Batch 16 = way higher throughput but each request waits longer. Pick based on whether you care more about latency or throughput.

**Preprocessing**: Vision benchmarks time preprocessing separately so you can see the real cost of resize/normalize, not just model inference.

## Output

JSONL format (one record per line):

```json
{
  "model": "resnet18",
  "batch_size": 4,
  "compile": true,
  "threads": 4,
  "latency_p50": 12.3,
  "latency_p90": 15.1,
  "latency_p95": 16.8,
  "latency_p99": 19.2,
  "throughput_rps": 245.3,
  "forward_ms_per_batch": 14.2,
  "env": {...}
}
```

## Running Tests

```bash
pytest tests/
```

## What gets benchmarked

**Models**:
- Vision: resnet18, mobilenet_v3_small (torchvision pretrained)
- Text: tiny transformer (256-dim, 4 layers, 4 heads)

**Optimizations**:
- torch.compile (PyTorch 2.0+)
- Thread counts (intra-op and inter-op)
- channels_last memory format (vision)
- Dynamic int8 quantization (text)

## License

MIT
