import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    precision_recall_curve,
)
import numpy as np


########### #
# 검증 함수
########### #
def validate_model(
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    model.eval()

    # initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    avg_loss = 0.0
    val_acc = 0.0
    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    # validation loop
    with torch.inference_mode():
        progress_bar = tqdm(iterable=val_loader, desc="val")
        for index, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            # loss
            running_loss += loss.item()
            avg_loss = running_loss / (index + 1)

            # accuracy
            predictions = logits.argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)
            val_acc = 100.0 * correct / total if total > 0 else 0.0

            # postfix
            progress_bar.set_postfix(
                {"avg_loss": f"{avg_loss:.4f}", "val_acc": f"{val_acc:.2f}%"}
            )

            # 메트릭 계산을 위해 예측값, 실제값, 확률값 저장
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

        print(
            f"epoch {epoch+1}/{num_epochs}, val loss: {avg_loss:.4f}, val acc: {val_acc:.2f}%"
        )

        # precision, recall, f1_score
        danger_precision = precision_score(
            y_true=all_labels,
            y_pred=all_predictions,
            average=None,
            zero_division=0,
        )[1]
        danger_recall = recall_score(
            y_true=all_labels,
            y_pred=all_predictions,
            average=None,
            zero_division=0,
        )[1]
        danger_f1 = f1_score(
            y_true=all_labels,
            y_pred=all_predictions,
            average=None,
            zero_division=0,
        )[1]

        mcc = matthews_corrcoef(y_true=all_labels, y_pred=all_predictions)

        # PR-AUC (one-vs-rest)
        all_probs_np = np.array(all_probs)
        num_classes = all_probs_np.shape[1]
        all_labels_onehot = np.eye(num_classes)[np.array(all_labels)]
        danger_pr_auc = average_precision_score(
            y_true=all_labels_onehot,
            y_score=all_probs_np,
            average=None,
        )[1]

        # F-beta (beta=0.5, precision 중시)
        danger_fbeta = fbeta_score(
            y_true=all_labels,
            y_pred=all_predictions,
            beta=0.5,
            average=None,
            zero_division=0,
        )[1]

        precisions, recalls, thresholds = precision_recall_curve(
            y_true=all_labels_onehot[:, 1], y_score=all_probs_np[:, 1]
        )

        target_precision = 0.90
        idx = np.where(precisions >= target_precision)[0]
        if len(idx) > 0:
            best_idx = idx[0]
            val_threshold_at_90 = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
            val_recall_at_90 = recalls[best_idx]
        else:
            val_threshold_at_90 = 1.0
            val_recall_at_90 = 0.0

        print(
            f"Danger Precision: {danger_precision:.4f}, Danger Recall: {danger_recall:.4f}, Danger F1: {danger_f1:.4f}, "
            f"MCC: {mcc:.4f}, Danger PR-AUC: {danger_pr_auc:.4f}, Danger F-beta: {danger_fbeta:.4f}\n"
            f"Val Threshold @ 0.90 Precision: {val_threshold_at_90:.4f}, Val Recall: {val_recall_at_90:.4f}"
        )

    return avg_loss, val_acc, danger_precision, danger_recall, danger_f1, mcc, danger_pr_auc, danger_fbeta, val_threshold_at_90, val_recall_at_90
