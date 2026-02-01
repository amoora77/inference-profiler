import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from perflab.recommend import generate_recommendations, compute_balanced_score
from perflab.utils import mkdirp, read_jsonl


def generate_plots(runs, plots_dir):
    mkdirp(plots_dir)
    models = set(r["model"] for r in runs)

    for model in models:
        model_runs = [r for r in runs if r["model"] == model]

        fig, ax = plt.subplots(figsize=(8, 6))
        x = [r["latency_p95"] for r in model_runs]
        y = [r["throughput_rps"] for r in model_runs]
        ax.scatter(x, y, alpha=0.6)
        ax.set_xlabel("p95 Latency (ms)")
        ax.set_ylabel("Throughput (req/s)")
        ax.set_title(f"{model}: Throughput vs p95 Latency")
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(plots_dir, f"{model}_throughput_vs_p95.png")
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        compile_groups = {}
        for r in model_runs:
            key = "compile" if r.get("compile") else "no-compile"
            if key not in compile_groups:
                compile_groups[key] = []
            compile_groups[key].append(r)

        fig, ax = plt.subplots(figsize=(8, 6))
        for label, group in compile_groups.items():
            batch_sizes = sorted(set(r["batch_size"] for r in group))
            p95_by_bs = {}
            for bs in batch_sizes:
                bs_runs = [r for r in group if r["batch_size"] == bs]
                if bs_runs:
                    p95_by_bs[bs] = min(r["latency_p95"] for r in bs_runs)
            if p95_by_bs:
                bs_sorted = sorted(p95_by_bs.keys())
                p95_values = [p95_by_bs[bs] for bs in bs_sorted]
                ax.plot(bs_sorted, p95_values, marker="o", label=label)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("p95 Latency (ms)")
        ax.set_title(f"{model}: Batch Size vs p95 Latency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(plots_dir, f"{model}_batch_vs_p95.png")
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        vision_runs = [r for r in model_runs if "preprocess_ms_per_batch" in r]
        if vision_runs:
            sample = vision_runs[0]
            labels = ["Preprocess", "Forward", "Postprocess"]
            values = [
                sample.get("preprocess_ms_per_batch", 0),
                sample.get("forward_ms_per_batch", 0),
                sample.get("postprocess_ms_per_batch", 0),
            ]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, values)
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"{model}: Pipeline Breakdown (sample)")
            ax.grid(True, alpha=0.3, axis="y")
            plot_path = os.path.join(plots_dir, f"{model}_pipeline.png")
            fig.savefig(plot_path, dpi=100, bbox_inches="tight")
            plt.close(fig)


def generate_markdown_report(runs, out_path, constraint, latency_pct, latency_budget_ms, plots_dir):
    recs = generate_recommendations(runs, constraint, latency_pct, latency_budget_ms)
    models = sorted(set(r["model"] for r in runs))

    with open(out_path, "w") as f:
        f.write("# Inference Benchmarking Report\n\n")

        if runs:
            env = runs[0].get("env", {})
            f.write("## Environment\n\n")
            f.write(f"- Python: {env.get('python_version', 'N/A')}\n")
            f.write(f"- PyTorch: {env.get('torch_version', 'N/A')}\n")
            f.write(f"- Platform: {env.get('platform', 'N/A')}\n")
            f.write(f"- CPU count: {env.get('cpu_count', 'N/A')}\n\n")

        f.write("## Recommendations\n\n")
        f.write(f"Constraint: **{constraint}**\n\n")
        for rec in recs["recommendations"]:
            f.write(f"- {rec}\n")
        f.write("\n")

        f.write("## Top Configurations by Balanced Score\n\n")
        for model in models:
            model_runs = [r for r in runs if r["model"] == model]
            scored = [(r, compute_balanced_score(r)) for r in model_runs]
            scored.sort(key=lambda x: x[1], reverse=True)
            top5 = scored[:5]

            f.write(f"### {model}\n\n")
            f.write("| Batch | Compile | Threads | p95 (ms) | Throughput (req/s) | Score |\n")
            f.write("|-------|---------|---------|----------|-------------------|-------|\n")
            for run, score in top5:
                bs = run["batch_size"]
                comp = "yes" if run.get("compile") else "no"
                threads = run.get("threads", "N/A")
                p95 = run["latency_p95"]
                throughput = run["throughput_rps"]
                f.write(f"| {bs} | {comp} | {threads} | {p95:.1f} | {throughput:.1f} | {score:.2f} |\n")
            f.write("\n")

        f.write("## Plots\n\n")
        for model in models:
            f.write(f"### {model}\n\n")
            f.write(f"![Throughput vs p95](plots/{model}_throughput_vs_p95.png)\n\n")
            f.write(f"![Batch Size vs p95](plots/{model}_batch_vs_p95.png)\n\n")
            pipeline_plot = os.path.join(plots_dir, f"{model}_pipeline.png")
            if os.path.exists(pipeline_plot):
                f.write(f"![Pipeline Breakdown](plots/{model}_pipeline.png)\n\n")


def generate_report(input_path, out_path="reports/latest.md", constraint="balanced", latency_pct="p95", latency_budget_ms=50.0):
    runs = read_jsonl(input_path)
    if not runs:
        print(f"No runs found in {input_path}")
        return

    reports_dir = os.path.dirname(out_path)
    plots_dir = os.path.join(reports_dir, "plots")
    mkdirp(plots_dir)

    generate_plots(runs, plots_dir)
    generate_markdown_report(runs, out_path, constraint, latency_pct, latency_budget_ms, plots_dir)

    print(f"Report generated: {out_path}")
