"""
train_metrics_utils.py

- Learning Curve & Metrics with Epochs -
1. visualize_val_results

- Confusion matrix -
2. visualize_val_confusion_matrix

- Metrics Markdown -
3. save_val_run_metadata

"""

from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# results directory


####################################### #
# Learning Curve & Metrics with Epochs
####################################### #
def visualize_val_results(
    history: dict[str, list[float]],
    dataset_name: str,
    model_name: str,
    results_dir: Path,
) -> None:
    """추론 결과에 대한 그래프 시각화 함수"""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 15))
    plt.suptitle(f"Linear-Head DINOv3 Training Results ({dataset_name})", fontsize=16)

    # 1. Loss (Train & Val)
    plt.subplot(3, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="x")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Accuracy (Train & Val)
    plt.subplot(3, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    plt.plot(epochs, history["val_acc"], color="green", label="Val Acc", marker="s")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # 3. MCC (Matthews Correlation Coefficient)
    plt.subplot(3, 2, 3)
    plt.plot(epochs, history["val_mcc"], color="purple", label="MCC", marker="d")
    plt.title("Matthews Correlation Coefficient")
    plt.xlabel("Epochs")
    plt.ylabel("MCC")
    plt.legend()
    plt.grid(True)

    # 4. Precision, Recall & F1
    plt.subplot(3, 2, 4)
    plt.plot(epochs, history["val_precision"], label="Danger Precision", marker="^")
    plt.plot(epochs, history["val_recall"], label="Danger Recall", marker="v")
    plt.plot(epochs, history["val_f1"], color="red", label="Danger F1", marker="d")
    plt.title("Danger Precision, Recall & F1 (Val)")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # 5. PR-AUC (Average Precision)
    plt.subplot(3, 2, 5)
    plt.plot(epochs, history["val_pr_auc"], color="orange", label="Danger PR-AUC", marker="s")
    plt.title("Danger PR-AUC")
    plt.xlabel("Epochs")
    plt.ylabel("PR-AUC")
    plt.legend()
    plt.grid(True)

    # 6. F-beta (beta=0.5)
    plt.subplot(3, 2, 6)
    plt.plot(
        epochs,
        history["val_fbeta"],
        color="teal",
        label="Danger F\u03b2 (\u03b2=0.5)",
        marker="p",
    )
    plt.title("Danger F-beta Score (\u03b2=0.5)")
    plt.xlabel("Epochs")
    plt.ylabel("F-beta")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 결과 저장 폴더 생성 및 저장
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"learning_curve_{dataset_name}.png"
    plt.savefig(save_path)

    print(f"\nLearning curves saved to: {save_path}")

    plt.close()


################### #
# Confusion matrix
################### #
def visualize_val_confusion_matrix(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    dataset_name: str,
    model_name: str,
    results_dir: Path,
    val_threshold_at_90: float | None = None,
) -> None:
    """validation 과정에 따른 confusion matrix 구현 및 저장 함수"""
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for images, labels in val_loader:
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
        title=f"Confusion Matrix ({dataset_name} / {model_name})",
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

    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"confusion_matrix_{dataset_name}.png"
    fig.savefig(save_path)

    print(f"Confusion matrix saved to: {save_path}")

    plt.close(fig)


################### #
# Metrics Markdown
################### #
def save_val_run_metadata(
    hyperparam_path: Path,
    model_name: str,
    dataset_name: str,
    best_metrics: dict[str, Any],
    results_dir: Path,
) -> None:
    """추론 결과 metrics 저장 함수"""
    results_dir.mkdir(parents=True, exist_ok=True)

    # hyperparams.yaml 복사
    destination_yaml_path = results_dir / f"hyperparams.yaml"
    shutil.copy(src=hyperparam_path, dst=destination_yaml_path)

    print(f"Hyperparameters saved to: {destination_yaml_path}")

    # 결과 지표 마크다운 생성 및 저장
    destination_md_path = results_dir / f"metrics_{dataset_name}.md"

    # 작성될 마크다운 내용
    md_content = f"""# Training Metrics Summary ({model_name} on {dataset_name})

## Best Epoch: {best_metrics['epoch']}

| Metric | Value |
| :--- | :--- |
| Train Loss | {best_metrics['train_loss']:.4f} |
| Train Accuracy | {best_metrics['train_acc']:.2f}% |
| Val Loss | {best_metrics['val_loss']:.4f} |
| Val Accuracy | {best_metrics['val_acc']:.2f}% |
| Danger Precision | {best_metrics['precision']:.4f} |
| Danger Recall | {best_metrics['recall']:.4f} |
| Danger F1 Score | {best_metrics['f1']:.4f} |
| MCC | {best_metrics['mcc']:.4f} |
| Danger PR-AUC | {best_metrics['pr_auc']:.4f} |
| Danger F-beta (0.5) | {best_metrics['fbeta']:.4f} |
| **Danger Recall @ 0.90 Precision** | **{best_metrics.get('val_recall_at_90', 0):.4f}** |
| **Threshold @ 0.90 Precision** | **{best_metrics.get('val_threshold_at_90', 1):.4f}** |
"""
    # 저장
    with open(file=destination_md_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Metrics report saved to: {destination_md_path}")
