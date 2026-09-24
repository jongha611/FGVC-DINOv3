"""
metrics_utils.py

- Confusion Matrix -
1. visualize_test_confusion_matrix

- Metrics Markdown -
2. save_run_test_metadata

"""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


################### #
# Confusion matrix
################### #
def visualize_test_confusion_matrix(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    dataset_name: str,
    model_name: str,
    results_dir: Path,
    val_threshold_at_90: float | None = None,
) -> None:
    """추론 결과 confusion matrix 구현 및 저장 함수"""
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            
            # 기본은 argmax
            preds = logits.argmax(dim=1).cpu().numpy()
            
            # 임계값이 존재하면 Danger(인덱스 1) 확률에 대해 오버라이딩
            if val_threshold_at_90 is not None and logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                danger_probs = probs[:, 1]
                preds[danger_probs >= val_threshold_at_90] = 1

            all_predictions.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(y_true=all_labels, y_pred=all_predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="Actual",
        title=f"Confusion Matrix (Test: {dataset_name} / {model_name})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 셀에 수치 표시
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    # test 하위 디렉토리 생성 보장
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"confusion_matrix_{dataset_name}.png"
    fig.savefig(save_path)

    print(f"Test Confusion matrix saved to: {save_path}")
    plt.close(fig)


def save_run_test_metadata(
    hyperparam_path: Path,
    model_name: str,
    dataset_name: str,
    test_metrics: dict[str, float],
    results_dir: Path,
) -> None:
    """추론 결과 metrics 구현 및 저장 함수"""
    # 목적지 및 디렉토리 생성 보장
    results_dir.mkdir(parents=True, exist_ok=True)

    # hyperparams.yaml 복사
    destination_yaml_path = results_dir / "hyperparams.yaml"
    shutil.copy(src=hyperparam_path, dst=destination_yaml_path)

    print(f"Test Hyperparameters saved to: {destination_yaml_path}")

    # 결과 지표 마크다운 생성 및 저장
    destination_md_path = results_dir / f"metrics_{dataset_name}.md"

    # 작성될 마크다운 내용
    md_content = f"""# Test Evaluation Metrics Summary ({model_name} on {dataset_name})

| Metric | Value |
| :--- | :--- |
| Test Accuracy | {test_metrics['test_acc']:.2f}% |
| Danger Precision (w/ Val Threshold) | {test_metrics['precision']:.4f} |
| Danger Recall (w/ Val Threshold) | {test_metrics['recall']:.4f} |
| Danger F1 Score | {test_metrics['f1']:.4f} |
| MCC | {test_metrics['mcc']:.4f} |
| Danger PR-AUC | {test_metrics['pr_auc']:.4f} |
| Danger F-beta (0.5) | {test_metrics['fbeta']:.4f} |
| **Applied Val Threshold @ 0.90 Precision** | **{test_metrics['val_threshold_at_90']:.4f}** |
"""
    # 저장
    with open(file=destination_md_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Test Metrics report saved to: {destination_md_path}")
