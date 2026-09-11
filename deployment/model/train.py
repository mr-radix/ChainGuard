import os
import argparse
import sys
from typing import Optional
from src.data_loading import load_bitcoinheist, load_elliptic
from src.models.ransomware_model import RansomwareClassifier
from src.models.elliptic_gnn import EllipticGNNPipeline


def train_ransomware_task(data_dir: str, checkpoint_dir: str, sample_size: Optional[int] = None):
    print("\n========================================================")
    print(" TASK 1: Ransomware Tabular Classification (BitcoinHeist)")
    print("========================================================")
    df = load_bitcoinheist(path_or_dir=data_dir, sample_size=sample_size)

    model = RansomwareClassifier()
    metrics = model.fit(df)

    print("\n--- Ransomware Model Evaluation Results ---")
    print(f"Sample Count : {metrics['sample_count']:,}")
    print(f"Precision    : {metrics['precision']:.4f}")
    print(f"Recall       : {metrics['recall']:.4f}")
    print(f"F1 Score     : {metrics['f1']:.4f}")
    print(f"PR-AUC       : {metrics['pr_auc']:.4f}")

    model.save(checkpoint_dir)
    return metrics


def train_elliptic_task(
    data_dir: str,
    checkpoint_dir: str,
    sample_size: Optional[int] = None,
    epochs: int = 50
):
    print("\n========================================================")
    print(" TASK 2: Illicit Tx GNN Node Classification (Elliptic)")
    print("========================================================")
    features_df, classes_df, edgelist_df = load_elliptic(data_dir=data_dir, sample_size=sample_size)

    pipeline = EllipticGNNPipeline()
    data = pipeline.prepare_data(features_df, classes_df, edgelist_df)

    print(f"[EllipticGNN] Temporal Train (steps <= 34): {data['train_mask'].sum().item():,} nodes")
    print(f"[EllipticGNN] Temporal Val   (steps 35..39): {data['val_mask'].sum().item():,} nodes")
    print(f"[EllipticGNN] Temporal Test  (steps >= 40): {data['test_mask'].sum().item():,} nodes")

    test_metrics = pipeline.train(data, epochs=epochs, verbose=True)

    print("\n--- Elliptic GNN Temporal Test Set Results ---")
    print(f"Precision    : {test_metrics['precision']:.4f}")
    print(f"Recall       : {test_metrics['recall']:.4f}")
    print(f"F1 Score     : {test_metrics['f1']:.4f}")
    print(f"PR-AUC       : {test_metrics['pr_auc']:.4f}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "elliptic_gnn.pt")
    pipeline.save(checkpoint_path)
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="ChainGuard Model Training & Evaluation CLI")
    parser.add_argument(
        "--task",
        choices=["ransomware", "elliptic", "all"],
        default="all",
        help="Task to train/evaluate (default: all)"
    )
    parser.add_argument(
        "--data-dir",
        default="Dataset",
        help="Path to dataset root directory (default: Dataset)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Path to save trained checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row sample size for fast training/testing"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs for GNN training (default: 30)"
    )

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.task in ["ransomware", "all"]:
        train_ransomware_task(args.data_dir, args.checkpoint_dir, args.sample)

    if args.task in ["elliptic", "all"]:
        train_elliptic_task(args.data_dir, args.checkpoint_dir, args.sample, epochs=args.epochs)

    print("\n[ChainGuard] All requested training tasks completed successfully!")


if __name__ == "__main__":
    main()
