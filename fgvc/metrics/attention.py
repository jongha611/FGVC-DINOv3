"""
metrics_utils_for_samples.py

- Attention Heatmap -
1. patch_attention_module
2. generate_attention_heatmap
3. save_visualization_plot

- PR Curve -
4. collect_predictions
5. calculate_pr_curves
6. plot_pr_curve
7. print_ap_summary
8. generate_and_save_pr_curve

- Score Card -
9. generate_markdown_report

"""

from pathlib import Path
import types
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


######################### #
# Self-Attention Heatmap
######################### #
# 어텐션 맵 캡처용 전역 변수
captured_attn_weights: torch.Tensor | None = None
captured_attn_gradients: torch.Tensor | None = None


def patch_attention_module(model: nn.Module) -> None:
    """마지막 블럭의 어텐션 몽키 패치 함수"""
    # lora로 래핑되어있는 경우 (로라)model.base_model의 .model 어트리뷰트를 가져와야 합니다.
    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    # 마지막 블록의 어텐션 객체 생성
    last_attn = base_vit.blocks[-1].attn

    def custom_compute_attention(
        self: Any, qkv: torch.Tensor, attn_bias: Any = None, rope: Any = None
    ) -> torch.Tensor:
        """DINOv3의 SelfAttention클래스에 존재하는 compute_attention 메서드를 동일하게 구현"""
        global captured_attn_weights, captured_attn_gradients
        assert attn_bias is None

        # 입력 텐서의 형상 확인
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        # Q,K,V 분리
        qkv_reshaped = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv_reshaped, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        # Attention weights 직접 계산 [B, heads, N, N]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Grad-CAM을 위한 Gradient Hook 등록
        def save_attn_gradients(grad):
            global captured_attn_gradients
            captured_attn_gradients = grad.detach().cpu()

        if attn_weights.requires_grad:
            attn_weights.register_hook(save_attn_gradients)

        # CPU 메모리로 전송하여 캡처 저장
        captured_attn_weights = attn_weights.detach().cpu()

        # 실제 어텐션 출력을 얻기 위해 수동 계산 결과 사용 (그래디언트 흐름 유지)
        x = torch.matmul(attn_weights, v)
        x = x.transpose(1, 2)

        return x.reshape([B, N, C])

    last_attn.compute_attention = types.MethodType(custom_compute_attention, last_attn)


def generate_attention_heatmap(
    model: nn.Module,
    img_tensor: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> np.ndarray:
    """마지막 블록의 어텐션을 트래킹한 결과물로 히트맵을 그리는 함수"""
    global captured_attn_weights

    attn = captured_attn_weights[0]  # [heads, N, N]
    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    n_storage_tokens = base_vit.n_storage_tokens

    cls_attn = attn[:, 0, n_storage_tokens + 1 :]
    cls_attn_mean = cls_attn.mean(dim=0)

    grid_size = image_size // patch_size
    cls_attn_grid = cls_attn_mean.reshape(1, 1, grid_size, grid_size)

    attn_resized = F.interpolate(
        cls_attn_grid,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    heatmap = attn_resized.squeeze().numpy()
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap_normalized = (heatmap - h_min) / (h_max - h_min + 1e-8)

    return heatmap_normalized


def generate_gradcam_heatmap(
    model: nn.Module,
    img_tensor: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> np.ndarray:
    """Chefer et al.의 방식(Attention * Gradient)을 차용하여 Grad-CAM 히트맵을 그리는 함수"""
    global captured_attn_weights, captured_attn_gradients

    if captured_attn_gradients is None:
        raise ValueError("No gradients captured. Make sure to call .backward() before generating Grad-CAM.")

    attn = captured_attn_weights[0]  # [heads, N, N]
    grad = captured_attn_gradients[0]  # [heads, N, N]

    # 요소별 곱셈 후 양수 값만 취함 (ReLU)
    cam = F.relu(attn * grad)
    # Head 차원에 대해 평균
    cam = cam.mean(dim=0)

    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    n_storage_tokens = base_vit.n_storage_tokens

    # CLS 토큰이 패치 토큰들에 대해 가지는 중요도 추출
    cls_cam = cam[0, n_storage_tokens + 1 :]

    grid_size = image_size // patch_size
    cls_cam_grid = cls_cam.reshape(1, 1, grid_size, grid_size)

    cam_resized = F.interpolate(
        cls_cam_grid,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    heatmap = cam_resized.squeeze().numpy()
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap_normalized = (heatmap - h_min) / (h_max - h_min + 1e-8)

    return heatmap_normalized


def save_visualization_plot(
    orig_img: Image.Image,
    heatmap: np.ndarray,
    gradcam_heatmap: np.ndarray | None,
    true_label_name: str,
    pred_label_name: str,
    confidence: float,
    save_path: Path,
) -> None:
    """그려진 어텐션 및 Grad-CAM 히트맵을 플롯하는 함수"""
    h, w = heatmap.shape
    orig_img_resized = orig_img.resize((w, h), resample=Image.Resampling.BILINEAR)
    orig_np = np.array(orig_img_resized) / 255.0

    num_cols = 4 if gradcam_heatmap is not None else 3
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(5 * num_cols, 5))

    # Panel 1: Original Image
    axes[0].imshow(orig_np)
    axes[0].set_title(
        f"Original (True: {true_label_name})", fontsize=12, fontweight="bold"
    )
    axes[0].axis("off")

    # Panel 2: Attention Overlay
    axes[1].imshow(orig_np)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title(
        f"Attn Overlay (Pred: {pred_label_name} | {confidence * 100.0:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    # Panel 3: Pure Attention Heatmap
    im1 = axes[2].imshow(heatmap, cmap="jet")
    axes[2].set_title("Self-Attention Map", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Grad-CAM Overlay (if available)
    if gradcam_heatmap is not None:
        axes[3].imshow(orig_np)
        axes[3].imshow(gradcam_heatmap, cmap="jet", alpha=0.45)
        axes[3].set_title("Grad-CAM (Decision Basis)", fontsize=12, fontweight="bold")
        axes[3].axis("off")

    plt.tight_layout()

    # 저장 경로
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


########### #
# PR-Curve
########### #
def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """샘플 추론 결과 수집 함수"""
    model.eval()
    all_probs = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in data_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    return y_true, y_probs


def calculate_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int,
) -> dict[str, any]:
    """PR커브 데이터 생성 함수"""
    results = {}

    # 1. 원핫 인코딩
    y_true_onehot = np.eye(num_classes)[y_true]

    # 2. 클래스별 PR 커브 및 AP 계산
    for class_idx in range(num_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, class_idx],
            y_probs[:, class_idx],
        )
        ap = average_precision_score(
            y_true=y_true_onehot[:, class_idx],
            y_score=y_probs[:, class_idx],
        )
        results[str(class_idx)] = {
            "precision": precision,
            "recall": recall,
            "ap": ap,
        }

    # 3. Micro-average PR 커브 및 AP 계산
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true_onehot.ravel(),
        y_probs.ravel(),
    )
    ap_micro = average_precision_score(
        y_true=y_true_onehot,
        y_score=y_probs,
        average="micro",
    )
    results["micro"] = {
        "precision": precision_micro,
        "recall": recall_micro,
        "ap": ap_micro,
    }

    return results


def plot_pr_curve(
    pr_data: dict[str, any],
    class_names: list[str],
    save_path: Path,
) -> None:
    """PR 커브 그래프 생성 및 저장 함수"""
    plt.figure(figsize=(9, 8))

    # Premium Harmonious Palette (Modern vibrant colors)
    colors = ["#4361EE", "#4CC9F0", "#F72585", "#7209B7"]

    # Micro-average plot first (dashed line)
    micro = pr_data["micro"]
    plt.plot(
        micro["recall"],
        micro["precision"],
        color=colors[3],
        linestyle="--",
        linewidth=2.5,
        label=f"Micro-average (AP = {micro['ap']:.4f})",
    )

    # Class-wise plots
    for class_idx, class_name in enumerate(class_names):
        class_str = str(class_idx)
        if class_str not in pr_data:
            continue
        data = pr_data[class_str]
        color = colors[class_idx % len(colors)]
        plt.plot(
            data["recall"],
            data["precision"],
            color=color,
            linewidth=2.0,
            label=f"Class '{class_name}' (AP = {data['ap']:.4f})",
        )

    plt.title(
        "Precision-Recall Trade-off Curves",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Recall (Sensitivity/Reactivity)", fontsize=11, labelpad=10)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, labelpad=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    # Premium styling with subtle grid and legend formatting
    plt.grid(visible=True, linestyle=":", alpha=0.6, color="#CCCCCC")
    plt.legend(
        loc="lower left",
        frameon=True,
        facecolor="#F8F9FA",
        edgecolor="#E9ECEF",
        fontsize=10,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_ap_summary(pr_data: dict[str, any], class_names: list[str]) -> None:
    """AP 요약 테이블 출력 함수"""
    header = f"{'Class / Target':<25} | {'Average Precision (AP)':<25}"
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    # Class-wise AP
    for class_idx, class_name in enumerate(class_names):
        class_str = str(class_idx)
        if class_str in pr_data:
            ap_val = pr_data[class_str]["ap"]

            print(f"{class_name:<25} | {ap_val:.4f}")

    print(separator)

    # Micro AP
    micro_ap = pr_data["micro"]["ap"]

    print(f"{'Micro-average':<25} | {micro_ap:.4f}")
    print(separator + "\n")


def generate_and_save_pr_curve(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    save_path: Path,
) -> None:
    """PR-Curve를 생성하고 저장하는 함수"""

    print("[*] Compiling model predictions on full test dataset for PR curve...")

    y_true, y_probs = collect_predictions(model, test_loader, device)

    num_classes = len(class_names)
    pr_data = calculate_pr_curves(y_true, y_probs, num_classes)

    plot_pr_curve(pr_data, class_names, save_path)

    print(f"[*] Precision-Recall curve plotted and saved to: {save_path.resolve()}")

    print_ap_summary(pr_data, class_names)


############# #
# Score Card
############# #
def generate_markdown_report(
    model_name: str,
    dataset_name: str,
    checkpoint_path: Path,
    class_names: list[str],
    eval_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """추론 결과 리스트를 기반으로 Markdown 평가 보고서를 작성하여 저장하는 함수"""
    correct_count = sum(1 for res in eval_results if res["match"])
    total_count = len(eval_results)
    accuracy = (correct_count / total_count) * 100.0 if total_count > 0 else 0.0

    # 1. 헤더 및 메타데이터 작성
    md_lines = [
        "# DINOv3 Inference Evaluation Report",
        "",
        "## 📋 Run Metadata",
        f"- **Model Backbone**: `{model_name}`",
        f"- **Dataset Split**: `{dataset_name}`",
        f"- **Checkpoint Path**: `{checkpoint_path}`",
        f"- **Total Samples Evaluated**: `{total_count}`",
        f"- **Overall Accuracy**: `{accuracy:.2f}%` ({correct_count}/{total_count})",
        "",
        "## 🔍 Detailed Inference Results",
        "",
        "| Index | File Name | True Class | Predicted Class | Probabilities | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :---: |",
    ]

    # 2. 결과 테이블 행 작성
    for idx, res in enumerate(eval_results):
        # 확률 문자열 결합 (예: cut: 98.2% | danger: 1.5%)
        prob_strs = [
            f"{class_names[c_idx]}: {prob * 100.0:.2f}%"
            for c_idx, prob in enumerate(res["probs"])
        ]
        prob_summary = " | ".join(prob_strs)
        status_icon = "✅" if res["match"] else "❌"

        md_lines.append(
            f"| {idx+1} | `{res['filename']}` | {res['true_class']} | {res['pred_class']} | {prob_summary} | {status_icon} |"
        )

    md_content = "\n".join(md_lines) + "\n"

    # 3. 마크다운 파일 저장
    report_path = output_dir / "test_report.md"
    with open(file=report_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[*] Inference report successfully saved to: {report_path.resolve()}")
