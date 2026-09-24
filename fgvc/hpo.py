"""Optuna 하이퍼파라미터 최적화.

목적 지표는 `SELECTION_METRIC`을 그대로 따르므로, HPO가 찾은 하이퍼파라미터와
본 학습의 체크포인트 선정 기준이 항상 일치합니다.
탐색 공간은 실험 축에 따라 자동으로 좁혀집니다 (예: `frozen` 백본에서는 LoRA 파라미터 제외).
"""

from __future__ import annotations

from pathlib import Path

import optuna
import torch

from fgvc.config import ExperimentConfig
from fgvc.pipeline import build_runtime, run_train_val

N_TRIALS = 30
TPE_SEED = 2026


def _suggest_overrides(trial: optuna.Trial, config: ExperimentConfig) -> dict:
    """실험 축에 맞는 탐색 공간만 제안합니다."""
    overrides: dict = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True),
    }

    # MLP 헤드일 때만 은닉 차원과 헤드 드롭아웃이 의미를 가집니다.
    if config.head == "mlp":
        hidden_dim1 = trial.suggest_categorical("hidden_dim1", [256, 512, 1024])
        hidden_dim2 = trial.suggest_categorical(
            "hidden_dim2", [d for d in [128, 256, 512] if d <= hidden_dim1]
        )
        overrides["hidden_dim1"] = hidden_dim1
        overrides["hidden_dim2"] = hidden_dim2
        overrides["head_dropout"] = trial.suggest_categorical(
            "head_dropout", [0.1, 0.3, 0.5]
        )

    # 백본을 동결하면 LoRA 파라미터는 탐색할 의미가 없습니다.
    if config.backbone_tuning == "lora":
        lora_rank = trial.suggest_categorical("lora_rank", [4, 8, 16])
        overrides["lora_rank"] = lora_rank
        overrides["lora_alpha"] = lora_rank * 2
        overrides["lora_dropout"] = trial.suggest_categorical(
            "lora_dropout", [0.0, 0.05, 0.1, 0.2]
        )

    # label smoothing은 CrossEntropy에만 적용됩니다.
    if config.loss == "cross_entropy":
        overrides["label_smoothing"] = trial.suggest_categorical(
            "label_smoothing", [0.0, 0.05, 0.1, 0.15]
        )

    return overrides


def _objective(
    trial: optuna.Trial, config: ExperimentConfig, hyperparam_path: Path
) -> float:
    overrides = _suggest_overrides(trial=trial, config=config)
    trial_config = config.with_overrides(**overrides)

    print("\n" + "=" * 50)
    print(f"[Optuna Trial #{trial.number}] {trial_config.describe()}")
    for key, value in sorted(overrides.items()):
        formatted = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"  {key:<16}: {formatted}")
    print("=" * 50 + "\n")

    runtime = build_runtime(trial_config)
    try:
        # 탐색 중에는 체크포인트·그래프를 남기지 않습니다.
        return run_train_val(
            config=trial_config,
            runtime=runtime,
            hyperparam_path=hyperparam_path,
            save_artifacts=False,
        )
    finally:
        # VRAM 해제
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_hpo(
    config: ExperimentConfig,
    hyperparam_path: Path,
    n_trials: int = N_TRIALS,
) -> optuna.Study:
    """HPO Study를 생성하고 탐색을 실행합니다."""
    study = optuna.create_study(
        study_name=f"dinov3_{config.experiment}_hpo",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED),
    )

    study.optimize(
        lambda trial: _objective(
            trial=trial, config=config, hyperparam_path=hyperparam_path
        ),
        n_trials=n_trials,
    )

    metric_label = config.selection_metric.upper().replace("_", "-")
    print("\n" + "=" * 50)
    print("★ OPTIMIZATION COMPLETED ★")
    print(f"Best Trial Score (Val {metric_label}): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        formatted = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"  {key}: {formatted}")
    print("=" * 50)

    return study
