"""Command line entry point for training the risk-control model."""

from __future__ import annotations

from pathlib import Path

from . import risk_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / risk_model.DATASET_FILENAME
    artifacts_dir = project_root / "model_artifacts"

    artifacts = risk_model.train_risk_model(dataset_path)
    risk_model.save_artifacts(artifacts, artifacts_dir)

    print("模型训练完成，关键指标如下：")
    for metric, value in artifacts.metrics.items():
        print(f"  {metric}: {value:.2f}")

    print("\n风险分层样本分布：")
    print(artifacts.risk_summary)


if __name__ == "__main__":
    main()

