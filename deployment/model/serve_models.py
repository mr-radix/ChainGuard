import os
import argparse
import sys
import json
from src.data_loading import load_bitcoinheist, load_elliptic
from src.models.ransomware_model import RansomwareModelSuite
from src.models.elliptic_gnn import EllipticGNNPipeline
from src.model_serving import run_serving_server, InferenceEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, "exported_models")



def export_small_to_big_models(data_dir: str = "Dataset", sample_size: int = 500):
    print("\n=================================================================")
    print(" 🛠️  EXPORTING SMALL-TO-BIG MODEL SPECTRUM & TORCHSCRIPT ARTIFACTS ")
    print("=================================================================")

    os.makedirs(EXPORT_DIR, exist_ok=True)
    sample_size = min(sample_size, 500)

    # 1. Tabular Spectrum
    df_bh = load_bitcoinheist(path_or_dir=data_dir, sample_size=sample_size)
    suite = RansomwareModelSuite()
    benchmarks_tab = suite.train_and_benchmark(df_bh)

    print("\n--- Tabular Small-to-Big Spectrum Benchmarks ---")
    print(benchmarks_tab.to_string(index=False))

    tab_exports = suite.export_models(EXPORT_DIR)

    # 2. GNN Spectrum
    features_df, classes_df, edgelist_df = load_elliptic(data_dir=data_dir, sample_size=sample_size)
    gnn_pipeline = EllipticGNNPipeline(hidden_dim=16, lr=0.01)
    data = gnn_pipeline.prepare_data(features_df, classes_df, edgelist_df)

    gnn_benchmarks = gnn_pipeline.benchmark_models(data, epochs=2)
    print("\n--- Graph GNN Small-to-Big Spectrum Benchmarks ---")
    print(gnn_benchmarks.to_string(index=False))

    torchscript_path = gnn_pipeline.export_torchscript(EXPORT_DIR, in_features=data["num_features"])

    # 3. Model Registry Metadata
    registry_meta = {
        "tabular_benchmarks": benchmarks_tab.to_dict(orient="records"),
        "gnn_benchmarks": gnn_benchmarks.to_dict(orient="records"),
        "export_paths": {
            "tabular": tab_exports,
            "torchscript_gnn": torchscript_path
        }
    }

    meta_file = os.path.join(EXPORT_DIR, "model_registry.json")
    with open(meta_file, "w") as f:
        json.dump(registry_meta, f, indent=2)

    print(f"\n[serve_models] Model Registry metadata saved to {meta_file}")
    print("[serve_models] Export complete! Production models ready for serving.")


def main():
    parser = argparse.ArgumentParser(description="ChainGuard Model Export & Hosting CLI")
    parser.add_argument("--export-only", action="store_true", help="Export trained models and exit")
    parser.add_argument("--serve", action="store_true", help="Launch Model Serving REST Server")
    parser.add_argument("--data-dir", default="Dataset", help="Path to Dataset root directory")
    parser.add_argument("--sample", type=int, default=15000, help="Row sample size for training")
    parser.add_argument("--port", type=int, default=8080, help="HTTP REST Server port (default: 8080)")

    args = parser.parse_args()

    env_port = os.environ.get("PORT")
    if env_port:
        try:
            args.port = int(env_port)
        except ValueError:
            pass

    if not args.export_only and not args.serve:
        args.serve = True

    if args.export_only or not os.path.exists(os.path.join(EXPORT_DIR, "model_registry.json")):
        export_small_to_big_models(data_dir=args.data_dir, sample_size=args.sample)

    if args.serve:
        httpd = run_serving_server(port=args.port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[ModelServing] Server stopped.")


if __name__ == "__main__":
    main()
