from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    precision_recall_curve,
)

from fgvc.data import SquarePad
from fgvc.metrics import attention
from fgvc.metrics.attention import (
    patch_attention_module,
    generate_attention_heatmap,
    save_visualization_plot,
    generate_and_save_pr_curve,
    generate_markdown_report,
)

import numpy as np
import shutil
import matplotlib.pyplot as plt
from PIL import Image


############ #
# 테스트 함수
############ #
def test_model(
    checkpoint_path: Path,
    threshold_path: Path,
    device: torch.device,
    model: nn.Module,
    test_loader: DataLoader,
) -> tuple[float, float, float, float, float, float, float, float]:
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    # initialize metrics
    correct = 0
    total = 0
    test_acc = 0.0
    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    # test loop
    with torch.inference_mode():
        progress_bar = tqdm(iterable=test_loader, desc="test")
        for index, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            # accuracy
            predictions = logits.argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)
            test_acc = 100.0 * correct / total if total > 0 else 0.0

            # postfix
            progress_bar.set_postfix({"test_acc": f"{test_acc:.2f}%"})

            # 메트릭 계산을 위해 예측값, 실제값, 확률값 저장
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

        all_probs_np = np.array(all_probs)
        num_classes = all_probs_np.shape[1]
        all_labels_onehot = np.eye(num_classes)[np.array(all_labels)]
        all_predictions_np = np.array(all_predictions)

        # val 단계에서 기록한 precision@0.90 threshold를 읽어옵니다 (핸드오프 계약)
        threshold_file = threshold_path
        if threshold_file.exists():
            with open(threshold_file, "r", encoding="utf-8") as f:
                val_threshold_at_90 = float(f.read().strip())
        else:
            print(f"Warning: Threshold file not found at {threshold_file}, using default argmax.")
            val_threshold_at_90 = None

        # Override predictions based on val_threshold
        if val_threshold_at_90 is not None:
            danger_probs = all_probs_np[:, 1]
            all_predictions_np[danger_probs >= val_threshold_at_90] = 1

        # Recompute test accuracy after override
        test_acc = 100.0 * np.sum(all_predictions_np == np.array(all_labels)) / len(all_labels) if len(all_labels) > 0 else 0.0

        # sklearn 지표 연산
        danger_precision = precision_score(
            y_true=all_labels, y_pred=all_predictions_np, average=None, zero_division=0
        )[1]
        danger_recall = recall_score(
            y_true=all_labels, y_pred=all_predictions_np, average=None, zero_division=0
        )[1]
        danger_f1 = f1_score(
            y_true=all_labels, y_pred=all_predictions_np, average=None, zero_division=0
        )[1]
        mcc = matthews_corrcoef(y_true=all_labels, y_pred=all_predictions_np)

        danger_pr_auc = average_precision_score(
            y_true=all_labels_onehot,
            y_score=all_probs_np,
            average=None,
        )[1]
        danger_fbeta = fbeta_score(
            y_true=all_labels,
            y_pred=all_predictions_np,
            beta=0.5,
            average=None,
            zero_division=0,
        )[1]

        print(f"\n[Test Result] Acc: {test_acc:.2f}%")
        print(
            f"Danger Precision: {danger_precision:.4f}, Danger Recall: {danger_recall:.4f}, Danger F1: {danger_f1:.4f}, MCC: {mcc:.4f}, Danger PR-AUC: {danger_pr_auc:.4f}, Danger F-beta: {danger_fbeta:.4f}\n"
            f"[Applied Val Threshold: {val_threshold_at_90 if val_threshold_at_90 else 'None'}]\n"
        )

    return (
        test_acc,
        danger_precision,
        danger_recall,
        danger_f1,
        mcc,
        danger_pr_auc,
        danger_fbeta,
        val_threshold_at_90 if val_threshold_at_90 is not None else 0.0,
    )


def test_model_with_samples(
    model_name: str,
    test_loader: DataLoader,
    dataset_name: str,
    image_size: int,
    checkpoint_path: Path,
    device: torch.device,
    model: nn.Module,
    patch_size: int,
    results_dir: Path,
    visualize_samples: bool = True,
):
    # Attention 몽키 패치 적용
    patch_attention_module(model)

    # 결과 디렉토리 설정 및 생성
    results_dir.mkdir(parents=True, exist_ok=True)

    # 결과 디렉토리 초기화
    for p in results_dir.glob("attention_sample_*.png"):
        p.unlink()

    # 샘플 테스트를 위한 데이터셋 로드
    test_dataset = test_loader.dataset
    class_names = test_dataset.classes

    sample_indices = random.sample(
        range(len(test_dataset)), k=min(10, len(test_dataset))
    )

    print(f"[*] Selected indices for evaluation: {sample_indices}")

    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    patch_size = base_vit.patch_size

    eval_results = []
    for index, sample_index in enumerate(sample_indices):
        img_tensor, true_label = test_dataset[sample_index]
        img_path, _ = test_dataset.samples[sample_index]

        img_input = img_tensor.unsqueeze(dim=0).to(device)

        with torch.inference_mode():
            logits = model(img_input)
            probs = torch.softmax(input=logits, dim=1).squeeze().cpu()

        pred_label = int(probs.argmax())
        confidence = float(probs[pred_label])

        true_name = class_names[true_label]
        pred_name = class_names[pred_label]

        match_status = true_label == pred_label

        print(
            f"[{index+1}/10] Evaluated {Path(img_path).name} | "
            f"True: {true_name} | Pred: {pred_name} ({confidence * 100.0:.2f}%)"
        )
        print(
            f"[{index+1}/10] Visualizing {Path(img_path).name} | "
            f"True: {true_name} | Pred: {pred_name} ({confidence * 100.0:.2f}%)"
        )

        eval_results.append(
            {
                "filename": Path(img_path).name,
                "true_class": true_name,
                "pred_class": pred_name,
                "probs": probs.tolist(),
                "match": match_status,
            }
        )

        ######################################
        # 히트맵 렌더링 및 저장
        if visualize_samples:
            heatmap = generate_attention_heatmap(
                model=model,
                img_tensor=img_input,
                image_size=image_size,
                patch_size=patch_size,
            )

            # 다음 루프 반복을 위해 전역 캐시 초기화
            attention.captured_attn_weights = None

            raw_img = Image.open(img_path).convert("RGB")
            orig_img = SquarePad()(raw_img)

            save_path = (
                results_dir
                / "attention_heatmap"
                / f"attention_sample_{index+1}_true_{true_name}_pred_{pred_name}.png"
            )
            save_visualization_plot(
                orig_img=orig_img,
                heatmap=heatmap,
                # test는 inference_mode로 추론하므로 gradient가 없어 Grad-CAM 생성 불가
                gradcam_heatmap=None,
                true_label_name=true_name,
                pred_label_name=pred_name,
                confidence=confidence,
                save_path=save_path,
            )

    # 마크다운 리포트 생성 함수 호출
    generate_markdown_report(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        eval_results=eval_results,
        output_dir=results_dir,
    )

    # 전체 테스트 데이터셋에 대한 PR 커브 생성 및 플롯 저장
    pr_save_path = results_dir / "pr_curve.png"
    generate_and_save_pr_curve(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_path=pr_save_path,
    )

    print(
        f"[*] Done! Attention overlays, PR curve, and Report saved to: {results_dir.resolve()}"
    )
