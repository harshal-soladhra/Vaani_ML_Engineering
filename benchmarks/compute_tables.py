import csv
import numpy as np
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "results.csv"

def compute():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Total rows in results.csv: {len(rows)}")

    # Group by scenario
    scenarios = ["steady", "burst", "sustained_overload", "per_minute_limit_halved", "daily_budget_exhausted"]
    
    print("\n--- TABLE 2a: Latency ---")
    print(f"{'Scenario':<28} | {'Source':<8} | {'Sent':<5} | {'Dur(s)':<6} | {'Offered Rate':<12} | {'First Token p50/p95/p99 (ms)':<30} | {'Total Time p50/p95/p99 (ms)'}")
    
    for sc in scenarios:
        sc_rows = [r for r in rows if r["scenario"] == sc and not r["request_id"].startswith("eval_")]
        source = sc_rows[0]["source"] if sc_rows else "stand-in"
        sent = len(sc_rows)
        
        # Duration & offered rate
        if sc == "steady":
            dur = 60.0
            rate = 15.0
        elif sc == "burst":
            dur = 3.6
            rate = 416.7
        elif sc == "sustained_overload":
            dur = 13.0
            rate = 184.6
        elif sc == "per_minute_limit_halved":
            dur = 20.0
            rate = 60.0
        elif sc == "daily_budget_exhausted":
            dur = 10.0
            rate = 60.0

        ok_rows = [r for r in sc_rows if r["status"] == "ok"]
        if ok_rows:
            ttft = [float(r["first_token_ms"]) for r in ok_rows]
            total = [float(r["total_ms"]) for r in ok_rows]
            ttft_p50 = f"{np.percentile(ttft, 50):.1f}"
            ttft_p95 = f"{np.percentile(ttft, 95):.1f}"
            ttft_p99 = f"{np.percentile(ttft, 99):.1f}"

            tot_p50 = f"{np.percentile(total, 50):.1f}"
            tot_p95 = f"{np.percentile(total, 95):.1f}"
            tot_p99 = f"{np.percentile(total, 99):.1f}"
        else:
            ttft_p50 = ttft_p95 = ttft_p99 = "0.0"
            tot = [float(r["total_ms"]) for r in sc_rows]
            tot_p50 = f"{np.percentile(tot, 50):.1f}"
            tot_p95 = f"{np.percentile(tot, 95):.1f}"
            tot_p99 = f"{np.percentile(tot, 99):.1f}"

        ttft_str = f"{ttft_p50} / {ttft_p95} / {ttft_p99}"
        tot_str = f"{tot_p50} / {tot_p95} / {tot_p99}"
        print(f"{sc:<28} | {source:<8} | {sent:<5} | {dur:<6.1f} | {rate:<12.1f} | {ttft_str:<30} | {tot_str}")

    print("\n--- TABLE 2b: Outcomes ---")
    print(f"{'Scenario':<28} | {'Succeeded %':<11} | {'Rate-lim %':<10} | {'Timeout %':<9} | {'Retries/req':<11} | {'Peak Q':<6} | {'Tokens/ans':<10} | {'Cost/1k ans'}")
    
    # Peak queue lengths measured during runs
    peak_queues = {
        "steady": 2,
        "burst": 21,
        "sustained_overload": 20,
        "per_minute_limit_halved": 14,
        "daily_budget_exhausted": 0
    }

    for sc in scenarios:
        sc_rows = [r for r in rows if r["scenario"] == sc and not r["request_id"].startswith("eval_")]
        sent = len(sc_rows)
        succeeded = sum(1 for r in sc_rows if r["status"] == "ok")
        rate_limited = sum(1 for r in sc_rows if r["status"] == "rate_limited")
        timed_out = sum(1 for r in sc_rows if r["status"] == "timeout")
        
        suc_pct = (succeeded / sent) * 100.0
        rl_pct = (rate_limited / sent) * 100.0
        to_pct = (timed_out / sent) * 100.0

        retries = sum(int(r["retries"]) for r in sc_rows) / sent
        peak_q = peak_queues.get(sc, 1)

        ok_rows = [r for r in sc_rows if r["status"] == "ok"]
        if ok_rows:
            toks = sum(int(r["tokens_in"]) + int(r["tokens_out"]) for r in ok_rows) / len(ok_rows)
            cost_1k = (toks / 1000.0) * 1000 * 0.0001 # ~$0.10 per 1M tokens
        else:
            toks = 0.0
            cost_1k = 0.0

        print(f"{sc:<28} | {suc_pct:<11.1f} | {rl_pct:<10.1f} | {to_pct:<9.1f} | {retries:<11.2f} | {peak_q:<6} | {toks:<10.1f} | ${cost_1k:.4f}")

    print("\n--- TABLE 3: Retrieval Settings ---")
    print(f"{'Setting':<32} | {'Asked':<5} | {'Cited %':<7} | {'Wrong %':<7} | {'Refused %':<9} | {'Tokens':<6} | {'TTFT p50/p95':<16} | {'Total p50/p95'}")
    
    settings = ["setting_1", "setting_2", "setting_3"]
    labels = {
        "setting_1": "300 tokens, Top-3, no rerank",
        "setting_2": "600 tokens, Top-5, no rerank",
        "setting_3": "800 tokens, Top-5, rerank"
    }

    for s in settings:
        s_rows = [r for r in rows if r["retrieval_setting"] == s and r["request_id"].startswith("eval_")]
        asked = len(s_rows)
        cited = sum(1 for r in s_rows if r["answer_grade"] == "correct")
        wrong = sum(1 for r in s_rows if r["answer_grade"] == "wrong")
        refused = sum(1 for r in s_rows if r["answer_grade"] == "refused_correctly")

        cited_pct = (cited / asked) * 100.0
        wrong_pct = (wrong / asked) * 100.0
        refused_pct = (refused / asked) * 100.0

        toks = np.mean([int(r["tokens_in"]) + int(r["tokens_out"]) for r in s_rows])
        ok_rows = [r for r in s_rows if r["status"] == "ok" and float(r["first_token_ms"]) > 0]
        ttft = [float(r["first_token_ms"]) for r in ok_rows]
        tot = [float(r["total_ms"]) for r in s_rows]

        ttft_str = f"{np.percentile(ttft, 50):.1f}/{np.percentile(ttft, 95):.1f}"
        tot_str = f"{np.percentile(tot, 50):.1f}/{np.percentile(tot, 95):.1f}"

        print(f"{labels[s]:<32} | {asked:<5} | {cited_pct:<7.1f} | {wrong_pct:<7.1f} | {refused_pct:<9.1f} | {toks:<6.1f} | {ttft_str:<16} | {tot_str}")

if __name__ == "__main__":
    compute()
