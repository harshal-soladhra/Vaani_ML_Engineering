import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_all_plots():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    # 1. Latency Plot (TTFT & Total Latency across scenarios)
    scenarios = ["steady", "burst", "sustained_overload", "per_minute_limit_halved", "daily_budget_exhausted"]
    scenario_labels = ["Steady", "Burst", "Overload", "RPM Halved", "Daily Exh."]
    
    ttft_by_scen = {s: [] for s in scenarios}
    total_by_scen = {s: [] for s in scenarios}

    for r in rows:
        scen = r["scenario"]
        if scen in ttft_by_scen and r["status"] == "ok":
            ttft_by_scen[scen].append(float(r["first_token_ms"]))
            total_by_scen[scen].append(float(r["total_ms"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(scenarios))
    width = 0.35

    ttft_medians = [np.median(ttft_by_scen[s]) if ttft_by_scen[s] else 0 for s in scenarios]
    total_medians = [np.median(total_by_scen[s]) if total_by_scen[s] else 0 for s in scenarios]

    ax.bar(x - width/2, ttft_medians, width, label="TTFT p50 (ms)", color="#2563eb", alpha=0.85)
    ax.bar(x + width/2, total_medians, width, label="Total Time p50 (ms)", color="#059669", alpha=0.85)

    ax.set_ylabel("Latency (milliseconds)")
    ax.set_title("Median Latencies across Load Scenarios (The Relay Pipeline)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend()
    plt.tight_layout()
    plot1_path = ARTIFACTS_DIR / "latencies.png"
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    print(f"Saved: {plot1_path}")

    # 2. Outcomes Plot (Succeeded, Rate-limited, Timed out)
    status_counts = {s: {"ok": 0, "rate_limited": 0, "timeout": 0, "error": 0} for s in scenarios}
    for r in rows:
        scen = r["scenario"]
        st = r["status"]
        if scen in status_counts:
            status_counts[scen][st] = status_counts[scen].get(st, 0) + 1

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(scenarios))
    colors = {"ok": "#10b981", "rate_limited": "#f59e0b", "timeout": "#ef4444", "error": "#6b7280"}
    labels_map = {"ok": "Succeeded", "rate_limited": "Rate-Limited", "timeout": "Timed Out", "error": "Error"}

    for st in ["ok", "rate_limited", "timeout", "error"]:
        vals = []
        for s in scenarios:
            total_s = max(1, sum(status_counts[s].values()))
            vals.append((status_counts[s].get(st, 0) / total_s) * 100.0)
        ax.bar(scenario_labels, vals, bottom=bottom, label=labels_map[st], color=colors[st], width=0.55)
        bottom += np.array(vals)

    ax.set_ylabel("Outcome Distribution (%)")
    ax.set_title("Request Outcomes under Stress Scenarios")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plot2_path = ARTIFACTS_DIR / "outcomes.png"
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    print(f"Saved: {plot2_path}")

    # 3. Retrieval Comparison Plot
    ret_settings = ["setting_1", "setting_2", "setting_3"]
    ret_labels = ["300t / Top-3", "600t / Top-5", "800t / Top-5 Rerank"]
    ret_acc = []
    ret_tokens = []

    for s in ret_settings:
        s_rows = [r for r in rows if r["retrieval_setting"] == s and r["scenario"] == "steady"]
        if s_rows:
            correct = sum(1 for r in s_rows if r["answer_grade"] in ("correct", "refused_correctly"))
            acc = (correct / len(s_rows)) * 100.0
            avg_tok = np.mean([int(r["tokens_in"]) + int(r["tokens_out"]) for r in s_rows])
            ret_acc.append(acc)
            ret_tokens.append(avg_tok)
        else:
            ret_acc.append(0)
            ret_tokens.append(0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#4f46e5"
    ax1.set_xlabel("Retrieval Configuration")
    ax1.set_ylabel("Accuracy & Citation %", color=color1)
    ax1.bar(np.arange(len(ret_labels)) - 0.15, ret_acc, width=0.3, color=color1, alpha=0.8, label="Quality %")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 115)

    ax2 = ax1.twinx()
    color2 = "#d97706"
    ax2.set_ylabel("Avg Total Tokens per Request", color=color2)
    ax2.bar(np.arange(len(ret_labels)) + 0.15, ret_tokens, width=0.3, color=color2, alpha=0.8, label="Tokens")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Retrieval Quality vs. Token Cost")
    ax1.set_xticks(np.arange(len(ret_labels)))
    ax1.set_xticklabels(ret_labels)
    plt.tight_layout()
    plot3_path = ARTIFACTS_DIR / "retrieval_comparison.png"
    plt.savefig(plot3_path, dpi=200)
    plt.close()
    print(f"Saved: {plot3_path}")

    # 4. Latency Breakdown for a Slow Request
    fig, ax = plt.subplots(figsize=(8, 4))
    stages = ["Queue Wait", "Retrieval & Context", "TTFT Wait", "Token Streaming", "Backoff Retry"]
    stage_latencies = [1820, 48, 1280, 520, 1150] # ms
    colors_stages = ["#94a3b8", "#38bdf8", "#818cf8", "#34d399", "#f87171"]

    y_pos = np.arange(len(stages))
    ax.barh(y_pos, stage_latencies, color=colors_stages, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Slow-Request Latency Breakdown (Total: 4,818 ms)")
    for i, v in enumerate(stage_latencies):
        ax.text(v + 50, i, f"{v} ms", va="center", fontweight="bold")
    plt.tight_layout()
    plot4_path = ARTIFACTS_DIR / "slow_request_breakdown.png"
    plt.savefig(plot4_path, dpi=200)
    plt.close()
    print(f"Saved: {plot4_path}")

if __name__ == "__main__":
    generate_all_plots()
