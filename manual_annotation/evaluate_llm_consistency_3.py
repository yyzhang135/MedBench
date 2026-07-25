import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)


def main():

    human_file = "200_MedBench_Expert_Annotation_Task_Formatted_Final.jsonl"
    llm_file = "Gemini_Result.jsonl"

    cm_png = "Confusion_Matrix.png"
    cm_pdf = "Confusion_Matrix.pdf"
    cm_svg = "Confusion_Matrix.svg"

    metrics_png = "Evaluation_Metrics.png"
    metrics_pdf = "Evaluation_Metrics.pdf"
    metrics_svg = "Evaluation_Metrics.svg"

    human_data = []

    try:
        with open(human_file, "r", encoding="utf-8") as f:

            for idx, line in enumerate(f):

                if not line.strip():
                    continue

                record = json.loads(line.strip())

                record["inner_idx"] = idx

                human_data.append(record)

    except Exception as e:

        print(f"读取人类标注文件失败: {e}")
        return

    llm_dict = {}

    try:
        with open(llm_file, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                record = json.loads(line.strip())

                llm_dict[
                    int(record["inner_idx"])
                ] = int(record["label"])

    except Exception as e:

        print(f"读取LLM标注文件失败: {e}")
        return

    y_true = []
    y_pred = []

    for item in human_data:

        idx = item["inner_idx"]

        if idx in llm_dict:

            y_true.append(int(item["label"]))
            y_pred.append(int(llm_dict[idx]))

    if len(y_true) == 0:

        print("无有效数据")
        return

    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n========================")
    print("Evaluation Results")
    print("========================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Kappa    : {kappa:.4f}")
    print("========================\n")

    sns.set_theme(style="whitegrid")

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,

        "axes.titlesize": 18,
        "axes.labelsize": 14,

        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        "legend.fontsize": 12,

        "figure.dpi": 200,
        "savefig.dpi": 600
    })

    fig_cm, ax_cm = plt.subplots(
        figsize=(7, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",

        linewidths=1.2,
        linecolor="white",

        square=True,

        annot_kws={
            "size": 22,
            "weight": "bold"
        },

        cbar_kws={
            "shrink": 0.85
        },

        ax=ax_cm
    )

    ax_cm.set_title(
        "Confusion Matrix",
        fontweight="bold",
        pad=15
    )

    ax_cm.set_xlabel(
        "LLM Label",
        fontweight="bold"
    )

    ax_cm.set_ylabel(
        "Manual Label",
        fontweight="bold"
    )

    ax_cm.set_xticklabels(
        ["Irrelevant", "Relevant"]
    )

    ax_cm.set_yticklabels(
        ["Irrelevant", "Relevant"],
        rotation=0
    )

    fig_cm.tight_layout()

    fig_cm.savefig(
        cm_png,
        dpi=600,
        bbox_inches="tight"
    )

    fig_cm.savefig(
        cm_pdf,
        bbox_inches="tight"
    )

    fig_cm.savefig(
        cm_svg,
        bbox_inches="tight"
    )

    plt.close(fig_cm)

    metrics_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Kappa"
    ]

    metrics_values = [
        acc,
        prec,
        rec,
        f1,
        kappa
    ]

    colors = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B2",
        "#937860"
    ]

    fig_metric, ax_metric = plt.subplots(
        figsize=(8, 6)
    )

    bars = ax_metric.bar(
        metrics_names,
        metrics_values,
        color=colors,
        width=0.65
    )

    ax_metric.set_ylim(0, 1.05)

    ax_metric.set_ylabel(
        "Score",
        fontweight="bold"
    )

    ax_metric.set_title(
        "Evaluation Metrics",
        fontweight="bold",
        pad=15
    )

    ax_metric.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    for bar in bars:

        value = bar.get_height()

        ax_metric.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,

            f"{value:.3f}",

            ha="center",
            va="bottom",

            fontsize=12,
            fontweight="bold"
        )

    fig_metric.tight_layout()

    fig_metric.savefig(
        metrics_png,
        dpi=600,
        bbox_inches="tight"
    )

    fig_metric.savefig(
        metrics_pdf,
        bbox_inches="tight"
    )

    fig_metric.savefig(
        metrics_svg,
        bbox_inches="tight"
    )

    plt.close(fig_metric)

    print("Confusion Matrix Saved:")
    print(f"  PNG : {cm_png}")
    print(f"  PDF : {cm_pdf}")
    print(f"  SVG : {cm_svg}")

    print("\nEvaluation Metrics Saved:")
    print(f"  PNG : {metrics_png}")
    print(f"  PDF : {metrics_pdf}")
    print(f"  SVG : {metrics_svg}")

    print("\n 所有图表已独立生成完成")


if __name__ == "__main__":
    main()
