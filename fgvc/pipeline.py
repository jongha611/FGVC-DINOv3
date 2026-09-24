"""학습 / 테스트 / 샘플 시각화 파이프라인.

기존 3개 폴더의 `train_test_elements.py`를 하나로 합친 것으로, 폴더별로 갈라져 있던
체크포인트 선정 기준은 `SELECTION_METRIC` 축으로 파라미터화했습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fgvc.config import ExperimentConfig
from fgvc.data import get_data_loader
from fgvc.engine.test import test_model, test_model_with_samples
from fgvc.engine.train import train_model
from fgvc.engine.validate import validate_model
from fgvc.losses import build_criterion
from fgvc.metrics.attention import generate_and_save_pr_curve
from fgvc.metrics.test_metrics import (
    save_run_test_metadata,
    visualize_test_confusion_matrix,
)
from fgvc.metrics.train_metrics import (
    save_val_run_metadata,
    visualize_val_confusion_matrix,
    visualize_val_results,
)
from fgvc.models import build_model, infer_patch_size

# 선정 지표별 초기값. 값이 클수록 좋은 지표들이므로 하한에서 시작합니다.
_METRIC_FLOOR: dict[str, float] = {"pr_auc": 0.0, "mcc": -1.0}


@dataclass
class Runtime:
    """한 번의 실행에서 공유되는 런타임 객체 묶음."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    device: torch.device
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: Any

    @property
    def patch_size(self) -> int:
        return infer_patch_size(self.model)


########### #
# 초기화
########### #
def build_runtime(config: ExperimentConfig) -> Runtime:
    """데이터로더·모델·손실·옵티마이저·스케줄러를 한 번에 준비합니다."""
    train_loader, val_loader, test_loader = get_data_loader(
        dataset_dir=config.dataset_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    criterion = build_criterion(config=config, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.num_epochs
    )

    return Runtime(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )


########################### #
# train & validate process
########################### #
def run_train_val(
    config: ExperimentConfig,
    runtime: Runtime,
    hyperparam_path: Path,
    save_artifacts: bool = True,
) -> float:
    """학습·검증 루프를 돌고 `SELECTION_METRIC` 기준 best 점수를 반환합니다.

    `save_artifacts=False`면 체크포인트와 그래프를 남기지 않습니다 (HPO 탐색용).
    """
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_mcc": [],
        "val_pr_auc": [],
        "val_fbeta": [],
    }
    best_score = _METRIC_FLOOR[config.selection_metric]
    patience_counter = 0
    best_metrics: dict[str, Any] = {}

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_model(
            device=runtime.device,
            train_loader=runtime.train_loader,
            model=runtime.model,
            criterion=runtime.criterion,
            optimizer=runtime.optimizer,
            epoch=epoch,
            num_epochs=config.num_epochs,
            accumulation_steps=config.accumulation_steps,
        )

        (
            val_loss,
            val_acc,
            danger_precision,
            danger_recall,
            danger_f1,
            mcc,
            danger_pr_auc,
            danger_fbeta,
            val_threshold_at_90,
            val_recall_at_90,
        ) = validate_model(
            val_loader=runtime.val_loader,
            model=runtime.model,
            device=runtime.device,
            criterion=runtime.criterion,
            epoch=epoch,
            num_epochs=config.num_epochs,
        )

        print(
            f"Epoch {epoch+1}/{config.num_epochs} - "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} - "
            f"Val PR-AUC: {danger_pr_auc:.4f} Val MCC: {mcc:.4f}\n"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(danger_precision)
        history["val_recall"].append(danger_recall)
        history["val_f1"].append(danger_f1)
        history["val_mcc"].append(mcc)
        history["val_pr_auc"].append(danger_pr_auc)
        history["val_fbeta"].append(danger_fbeta)

        runtime.lr_scheduler.step()

        # 체크포인트 선정 지표는 실험 설정이 정합니다 (HPO 목적 지표와 항상 동일)
        current_score = mcc if config.selection_metric == "mcc" else danger_pr_auc

        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "precision": danger_precision,
                "recall": danger_recall,
                "f1": danger_f1,
                "mcc": mcc,
                "pr_auc": danger_pr_auc,
                "fbeta": danger_fbeta,
                "val_threshold_at_90": val_threshold_at_90,
                "val_recall_at_90": val_recall_at_90,
            }

            if save_artifacts:
                config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(runtime.model.state_dict(), config.checkpoint_path)
                # test 단계로 넘길 threshold 핸드오프 파일
                config.threshold_path.write_text(
                    str(val_threshold_at_90), encoding="utf-8"
                )
        else:
            patience_counter += 1
            print(
                f"early stopping patience: "
                f"{patience_counter}/{config.early_stopping_patience}\n"
            )
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    if not save_artifacts:
        return best_score

    results_dir = config.results_dir("val")
    results_dir.mkdir(parents=True, exist_ok=True)

    visualize_val_results(
        history=history,
        dataset_name=config.dataset_name,
        model_name=config.model_name,
        results_dir=results_dir,
    )

    runtime.model.load_state_dict(
        torch.load(config.checkpoint_path, weights_only=True)
    )
    visualize_val_confusion_matrix(
        model=runtime.model,
        val_loader=runtime.val_loader,
        device=runtime.device,
        class_names=runtime.val_loader.dataset.classes,
        dataset_name=config.dataset_name,
        model_name=config.model_name,
        results_dir=results_dir,
        val_threshold_at_90=best_metrics.get("val_threshold_at_90"),
    )

    generate_and_save_pr_curve(
        model=runtime.model,
        test_loader=runtime.val_loader,
        device=runtime.device,
        class_names=runtime.val_loader.dataset.classes,
        save_path=results_dir / f"pr_curve_{config.dataset_name}.png",
    )

    if best_metrics:
        save_val_run_metadata(
            hyperparam_path=hyperparam_path,
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            best_metrics=best_metrics,
            results_dir=results_dir,
        )

    return best_score


########################### #
# test process for testset
########################### #
def run_testset(
    config: ExperimentConfig,
    runtime: Runtime,
    hyperparam_path: Path,
) -> None:
    """테스트셋 전체에 대해 평가하고 지표·혼동행렬을 저장합니다."""
    (
        test_acc,
        danger_precision,
        danger_recall,
        danger_f1,
        mcc,
        danger_pr_auc,
        danger_fbeta,
        val_threshold_at_90,
    ) = test_model(
        checkpoint_path=config.checkpoint_path,
        threshold_path=config.threshold_path,
        device=runtime.device,
        model=runtime.model,
        test_loader=runtime.test_loader,
    )

    results_dir = config.results_dir("test")
    results_dir.mkdir(parents=True, exist_ok=True)

    visualize_test_confusion_matrix(
        model=runtime.model,
        test_loader=runtime.test_loader,
        device=runtime.device,
        class_names=runtime.test_loader.dataset.classes,
        dataset_name=config.dataset_name,
        model_name=config.model_name,
        results_dir=results_dir,
        val_threshold_at_90=val_threshold_at_90,
    )

    save_run_test_metadata(
        hyperparam_path=hyperparam_path,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        test_metrics={
            "test_acc": test_acc,
            "precision": danger_precision,
            "recall": danger_recall,
            "f1": danger_f1,
            "mcc": mcc,
            "pr_auc": danger_pr_auc,
            "fbeta": danger_fbeta,
            "val_threshold_at_90": val_threshold_at_90,
        },
        results_dir=results_dir,
    )


########################### #
# test process for samples
########################### #
def run_samples(config: ExperimentConfig, runtime: Runtime) -> None:
    """샘플 10건에 대해 Attention 히트맵과 마크다운 리포트를 생성합니다."""
    results_dir = config.results_dir("test") / "with_samples"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_model_with_samples(
        model_name=config.model_name,
        test_loader=runtime.test_loader,
        dataset_name=config.dataset_name,
        image_size=config.image_size,
        checkpoint_path=config.checkpoint_path,
        device=runtime.device,
        model=runtime.model,
        patch_size=runtime.patch_size,
        results_dir=results_dir,
        visualize_samples=True,
    )


def run_full_pipeline(config: ExperimentConfig, hyperparam_path: Path) -> float:
    """학습 → 테스트 → 샘플 시각화를 순서대로 수행합니다."""
    runtime = build_runtime(config)
    best_score = run_train_val(
        config=config, runtime=runtime, hyperparam_path=hyperparam_path
    )
    run_testset(config=config, runtime=runtime, hyperparam_path=hyperparam_path)
    run_samples(config=config, runtime=runtime)
    return best_score
