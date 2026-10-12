
import json

from src.config import METRICS_PATH


def show_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "Metrics not found. Run: python -m src.train"
        )

    results = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    print("Best model:", results["best_model"])
    print("Test rows:", results["test_rows"])

    for model_name, metrics in results["metrics"].items():
        print(f"\n{model_name}")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    show_metrics()
