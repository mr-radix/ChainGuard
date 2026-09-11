"""
Phase 3: Exploratory Data Analysis (EDA) & Data Profiling
Decomposed into small, modular sections for clear execution and readability.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.features import bitcoinheist_features


def step_1_compute_eda_features(bh_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3.1: Transform raw dataset columns into behavioral features for EDA.
    """
    print("  [3.1] Computing preliminary behavioral features for EDA...")
    bh_feat = bitcoinheist_features(bh_df)
    return bh_feat


def step_2_plot_correlation_heatmap(bh_feat: pd.DataFrame, save_path: str):
    """
    Step 3.2: Render and save feature correlation heatmap.
    """
    print("  [3.2] Generating post-cleaning feature correlation heatmap...")
    corr_cols = [
        "income", "neighbors", "count", "looped", "length",
        "income_per_neighbor", "loop_ratio", "is_ransomware"
    ]
    available_cols = [c for c in corr_cols if c in bh_feat.columns]
    corr_matrix = bh_feat[available_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)
    ax.set_title("Post-Cleaning Feature Correlation Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3.2] Heatmap saved to: {save_path}")


def step_3_plot_income_distribution(bh_feat: pd.DataFrame, save_path: str):
    """
    Step 3.3: Render and save income-per-neighbor distribution boxplot by class.
    """
    print("  [3.3] Generating Income per Neighbor boxplot...")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=bh_feat,
        x="is_ransomware",
        y="income_per_neighbor",
        palette=["#2563EB", "#DC2626"],
        ax=ax
    )
    ax.set_yscale("log")
    ax.set_xticklabels(["White (Legit)", "Ransomware"])
    ax.set_title("Income per Neighbor (Log Scale) by Address Class", fontsize=11, fontweight="bold")
    ax.set_xlabel("Address Class")
    ax.set_ylabel("Income / Neighbors")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3.3] Boxplot saved to: {save_path}")


def run_phase_03(bh_df: pd.DataFrame, viz_dir: str = "visualizations"):
    """
    Execute Phase 3 in small, sequential steps.
    """
    print("\n--- Phase 3: Exploratory Data Analysis (EDA) & Data Profiling ---")
    os.makedirs(viz_dir, exist_ok=True)
    bh_feat = step_1_compute_eda_features(bh_df)
    
    corr_path = os.path.join(viz_dir, "post_cleaning_correlations.png")
    step_2_plot_correlation_heatmap(bh_feat, corr_path)
    
    boxplot_path = os.path.join(viz_dir, "income_per_neighbor_boxplot.png")
    step_3_plot_income_distribution(bh_feat, boxplot_path)

    print("✅ Phase 3 Complete: EDA plots rendered and exported.\n")
    return bh_feat


if __name__ == "__main__":
    from src.data_loading import load_bitcoinheist
    sample = load_bitcoinheist(sample_size=1000)
    run_phase_03(sample)
