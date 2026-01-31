def filter_valid_runs(runs):
    valid = []
    for run in runs:
        if run.get("latency_p95") and run.get("throughput_rps"):
            valid.append(run)
    return valid


def compute_balanced_score(run):
    p95 = run.get("latency_p95", 1.0)
    if p95 <= 0:
        p95 = 1.0
    throughput = run.get("throughput_rps", 0.0)
    return throughput / p95


def get_best_for_latency_budget(runs, latency_pct="p95", budget_ms=50.0):
    candidates = [r for r in runs if r.get(f"latency_{latency_pct}", float("inf")) <= budget_ms]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("throughput_rps", 0.0))


def get_best_for_max_throughput(runs):
    if not runs:
        return None
    return max(runs, key=lambda r: r.get("throughput_rps", 0.0))


def get_best_balanced(runs):
    if not runs:
        return None
    return max(runs, key=compute_balanced_score)


def format_recommendation(run, reason):
    if run is None:
        return f"No configuration found ({reason})"

    model = run.get("model", "unknown")
    bs = run.get("batch_size", "?")
    compile_flag = "compile" if run.get("compile") else "no-compile"
    threads = run.get("threads", "?")
    p95 = run.get("latency_p95", 0.0)
    throughput = run.get("throughput_rps", 0.0)

    extra = []
    if run.get("channels_last"):
        extra.append("channels_last")
    if run.get("quantize"):
        extra.append("quantize")

    extra_str = f", {', '.join(extra)}" if extra else ""

    return (
        f"{model}: batch={bs}, {compile_flag}, threads={threads}{extra_str} "
        f"→ p95={p95:.1f}ms, throughput={throughput:.1f} req/s"
    )


def generate_recommendations(runs, constraint="balanced", latency_pct="p95", latency_budget_ms=50.0):
    runs = filter_valid_runs(runs)
    if not runs:
        return {"recommendations": [], "details": {}}

    models = set(r["model"] for r in runs)
    recommendations = []
    details = {}

    for model in sorted(models):
        model_runs = [r for r in runs if r["model"] == model]

        if constraint == "latency":
            best = get_best_for_latency_budget(model_runs, latency_pct, latency_budget_ms)
            rec = format_recommendation(best, f"within {latency_budget_ms}ms {latency_pct}")
        elif constraint == "throughput":
            best = get_best_for_max_throughput(model_runs)
            rec = format_recommendation(best, "max throughput")
        else:
            best = get_best_balanced(model_runs)
            rec = format_recommendation(best, "balanced")

        recommendations.append(rec)
        details[model] = best

    return {"recommendations": recommendations, "details": details}
