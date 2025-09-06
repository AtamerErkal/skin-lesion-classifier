import argparse
import torch
import mlflow.pytorch

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Export a model from MLflow to a standalone .pth file.")
parser.add_argument("--run-id", type=str, required=True, help="MLflow Run ID of the model to export.")
parser.add_argument("--output-path", type=str, default="best_model.pth", help="Path to save the exported model.")
args = parser.parse_args()

print(f"Loading model from MLflow run: {args.run_id}")
model_uri = f"runs:/{args.run_id}/best_model"
model = mlflow.pytorch.load_model(model_uri)

print(f"Saving model to {args.output_path}")
torch.save(model, args.output_path)

print("Model exported successfully!")