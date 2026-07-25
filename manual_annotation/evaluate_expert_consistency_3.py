import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score
)


def main():

    print("Starting Inter-Annotator Agreement Analysis...\n")


    file_fq = "../data/200_MedBench_Expert_Annotation_Task_Formatted-doctor1.xlsx"
    file_dyz = "../data/200_MedBench_Expert_Annotation_Task_Formatted-doctor2.xlsx"
    file_xsm = "../data/200_MedBench_Expert_Annotation_Task_Formatted-doctor3.xlsx"

    QUERY_COL = "Query (问题)"
    ID_COL = "Doc ID (编号)"
    LABEL_COL = "Label (0-不相关, 1-相关)"


    try:
        df_fq = pd.read_excel(file_fq)
        df_dyz = pd.read_excel(file_dyz)
        df_xsm = pd.read_excel(file_xsm)

    except Exception as e:
        print(f"Failed to read Excel files: {e}")
        return

    for df in [df_fq, df_dyz, df_xsm]:
        df.columns = df.columns.str.strip()


    df_fq = df_fq[
        [QUERY_COL, ID_COL, LABEL_COL]
    ].rename(columns={
        LABEL_COL: "label_fq"
    })

    df_dyz = df_dyz[
        [QUERY_COL, ID_COL, LABEL_COL]
    ].rename(columns={
        LABEL_COL: "label_dyz"
    })

    df_xsm = df_xsm[
        [QUERY_COL, ID_COL, LABEL_COL]
    ].rename(columns={
        LABEL_COL: "label_xsm"
    })


    merged_experts = pd.merge(
        df_fq,
        df_dyz,
        on=[QUERY_COL, ID_COL],
        how="inner"
    )

    df_eval = pd.merge(
        merged_experts,
        df_xsm,
        on=[QUERY_COL, ID_COL],
        how="inner"
    )

    print(f"Successfully aligned {len(df_eval)} samples.")

    df_eval["label_final"] = (
        df_eval[
            ["label_fq", "label_dyz", "label_xsm"]
        ]
        .mode(axis=1)[0]
        .astype(int)
    )

    experts = [
        "label_fq",
        "label_dyz",
        "label_xsm"
    ]

    expert_names = [
        "Doctor 1",
        "Doctor 2",
        "Doctor 3"
    ]

    kappa_matrix = np.zeros((3, 3))

    for i in range(3):

        for j in range(3):

            if i == j:

                kappa_matrix[i, j] = 1.0

            else:

                kappa_matrix[i, j] = cohen_kappa_score(
                    df_eval[experts[i]],
                    df_eval[experts[j]]
                )

    acc_vs_final = []
    kappa_vs_final = []

    for exp in experts:

        acc_vs_final.append(
            accuracy_score(
                df_eval["label_final"],
                df_eval[exp]
            )
        )

        kappa_vs_final.append(
            cohen_kappa_score(
                df_eval["label_final"],
                df_eval[exp]
            )
        )

    print("\n===================================")
    print("Inter-Annotator Reliability")
    print("===================================")

    print(
        f"Doctor1 vs Doctor2 Kappa: "
        f"{kappa_matrix[0,1]:.4f}"
    )

    print(
        f"Doctor1 vs Doctor3 Kappa: "
        f"{kappa_matrix[0,2]:.4f}"
    )

    print(
        f"Doctor2 vs Doctor3 Kappa: "
        f"{kappa_matrix[1,2]:.4f}"
    )

    print("\nDoctor vs Majority Vote")

    for name, acc, kap in zip(
            expert_names,
            acc_vs_final,
            kappa_vs_final):

        print(
            f"{name}: "
            f"Accuracy={acc:.4f}, "
            f"Kappa={kap:.4f}"
        )

    print("===================================\n")

    sns.set_theme(style="white")

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

    fig1, ax1 = plt.subplots(
        figsize=(6.5, 5.5),
        constrained_layout=True
    )

    sns.heatmap(
        kappa_matrix,

        annot=True,
        fmt=".3f",

        cmap="Blues",

        square=True,

        linewidths=1.2,
        linecolor="white",

        vmin=0.5,
        vmax=1.0,

        xticklabels=expert_names,
        yticklabels=expert_names,

        annot_kws={
            "fontsize": 16,
            "fontweight": "bold"
        },

        cbar_kws={
            "label": "Cohen's Kappa",
            "shrink": 0.85
        },

        ax=ax1
    )

    ax1.set_title(
        "Inter-Annotator Cohen's Kappa Matrix",
        fontweight="bold",
        pad=12
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("")

    fig1.savefig(
        "Figure1_Kappa_Matrix.png",
        dpi=600,
        bbox_inches="tight"
    )

    fig1.savefig(
        "Figure1_Kappa_Matrix.pdf",
        bbox_inches="tight"
    )

    fig1.savefig(
        "Figure1_Kappa_Matrix.svg",
        bbox_inches="tight"
    )

    plt.close(fig1)

    fig2, ax2 = plt.subplots(
        figsize=(7, 5),
        constrained_layout=True
    )

    x = np.arange(len(expert_names))
    width = 0.36

    bars1 = ax2.bar(
        x - width / 2,
        [a * 100 for a in acc_vs_final],

        width,

        label="Accuracy (%)",

        color="#4C72B0"
    )

    bars2 = ax2.bar(
        x + width / 2,
        [k * 100 for k in kappa_vs_final],

        width,

        label="Kappa ×100",

        color="#55A868"
    )

    ax2.set_title(
        "Doctor Agreement with Majority Vote",
        fontweight="bold",
        pad=12
    )

    ax2.set_ylabel(
        "Score (%)",
        fontweight="bold"
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(expert_names)

    ax2.set_ylim(0, 110)

    ax2.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    ax2.legend(
        frameon=False
    )

    for bars in [bars1, bars2]:

        for bar in bars:

            value = bar.get_height()

            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,

                f"{value:.1f}",

                ha="center",
                va="bottom",

                fontsize=11,
                fontweight="bold"
            )

    fig2.savefig(
        "Figure2_Expert_vs_Majority.png",
        dpi=600,
        bbox_inches="tight"
    )

    fig2.savefig(
        "Figure2_Expert_vs_Majority.pdf",
        bbox_inches="tight"
    )

    fig2.savefig(
        "Figure2_Expert_vs_Majority.svg",
        bbox_inches="tight"
    )

    plt.close(fig2)

    print("===================================")
    print("Figures Generated Successfully")
    print("===================================")
    print("Figure1_Kappa_Matrix.(png/pdf/svg)")
    print("Figure2_Expert_vs_Majority.(png/pdf/svg)")
    print("===================================")


if __name__ == "__main__":
    main()

