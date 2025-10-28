import json
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _coerce_status(value: str) -> str:
    if value is None:
        return "PASS"
    key = str(value).strip().upper()
    if "FAIL" in key or "ERROR" in key:
        return "FAIL"
    return "PASS"


def parse_quality_log(log_path: str = "information_quality.log") -> pd.DataFrame:
    """Parse JSONL logs into DataFrame with event as Category and check as SubCategory."""
    rows = []
    p = Path(log_path)
    if not p.exists():
        return pd.DataFrame(columns=["Category", "SubCategory", "Result", "Status", "Raw"])

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}$", line)
                if not match:
                    continue
                try:
                    obj = json.loads(match.group(0))
                except json.JSONDecodeError:
                    print(f"Skipped invalid JSON at line {line_no}")
                    continue

            event = obj.get("event") or obj.get("Category") or "UNCATEGORIZED"
            if event == "RUN":
                continue  # skip RUN events entirely
            check = obj.get("check") or obj.get("Check") or "(missing-check-field)"

            result = obj.get("result") or obj.get("Result") or obj.get("message") or ""
            status = _coerce_status(result)

            rows.append({
                "Line": line_no,
                "Category": event,
                "SubCategory": check,
                "Result": result,
                "Status": status,
                "Raw": obj,
            })

    df = pd.DataFrame(rows)
    return df


def summarize_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if df.empty:
        return pd.DataFrame(columns=["Category", "SubCategory", "OverallStatus", "Pass", "Fail"]), "No checks found in log file."

    by_cat = (
        df.groupby(["Category", "SubCategory"])["Status"]
        .apply(lambda s: pd.Series({
            "Pass": (s == "PASS").sum(),
            "Fail": (s == "FAIL").sum(),
        }))
        .unstack()
        .fillna(0)
        .astype(int)
    )

    by_cat["OverallStatus"] = by_cat.apply(lambda r: "FAIL" if r["Fail"] > 0 else "PASS", axis=1)
    summary = by_cat.reset_index()

    if (summary["OverallStatus"] == "FAIL").any():
        conclusion = "❌ ONE OR MORE CHECKS FAILED"
    else:
        conclusion = "✅ ALL CHECKS PASSED"

    return summary[["Category", "SubCategory", "OverallStatus", "Pass", "Fail"]], conclusion



def visualize_quality_results(log_path: str = "information_quality.log", save_plot: bool = False):
    df = parse_quality_log(log_path)
    if df.empty:
        print("No checks found in log file.")
        return df, pd.DataFrame(), "No checks found in log file."

    summary_df, conclusion = summarize_results(df)
    categories = summary_df["Category"].unique()
    num_categories = len(categories)

    fig, axes = plt.subplots(
        nrows=1, ncols=num_categories,
        figsize=(3* num_categories, 6),
        constrained_layout=True
    )

    if num_categories == 1:
        axes = [axes]

    for ax, category in zip(axes, categories):
        cat_df = summary_df[summary_df["Category"] == category]

        ax.bar(
            cat_df["SubCategory"], cat_df["Pass"],
            label="PASS", color="green"
        )
        ax.bar(
            cat_df["SubCategory"], cat_df["Fail"],
            bottom=cat_df["Pass"],
            label="FAIL", color="red"
        )

        ax.set_title(f"{category}", fontsize=10, fontweight='bold')
        ax.set_xlabel("SubCategory", fontsize=8)
        ax.set_ylabel("Number of Checks", fontsize=8)
        ax.legend(handles=[
            Patch(color='green', label='PASS'),
            Patch(color='red', label='FAIL')
        ])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Information Quality Summary by Category", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_plot:
        plt.savefig("information_quality_summary.png", dpi=300)
        print("Plot saved as information_quality_summary.png")

    plt.show()
    # return df, summary_df, conclusion



if __name__ == "__main__":
    visualize_quality_results()