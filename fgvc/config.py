"""실험 설정의 단일 원천.

`hyperparams.yaml`을 읽어 한 번의 실행을 완전히 규정하는 `ExperimentConfig`로 변환합니다.
결과물·체크포인트 경로 조립도 이 모듈에서만 수행하며, 다른 모듈은 경로 문자열을
직접 만들지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

BackboneTuning = Literal["lora", "frozen"]
HeadType = Literal["mlp", "linear"]
LossType = Literal["cross_entropy", "focal"]
SelectionMetric = Literal["pr_auc", "mcc"]

DEFAULT_RESULTS_ROOT = "results"


# 설정 객체
@dataclass(frozen=True)
class ExperimentConfig:
    """한 번의 학습&평가 실행에 관한 모든 설정"""

    # 식별자
    experiment: str
    model_name: str
    dataset_name: str

    # 입력 경로
    model_path: Path
    dataset_dir: Path

    # 데이터
    batch_size: int
    image_size: int

    # 실험 요인
    backbone_tuning: BackboneTuning
    head: HeadType
    loss: LossType
    selection_metric: SelectionMetric

    # 모델 하이퍼파라미터
    num_classes: int
    hidden_dim1: int
    hidden_dim2: int
    head_dropout: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]

    # 손실 함수 하이퍼파라미터
    label_smoothing: float
    focal_alpha: list[float]
    focal_gamma: float

    # 학습 하이퍼파라미터
    num_epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    accumulation_steps: int

    # 출력 경로 루트
    results_root: Path
    checkpoint_dir: Path

    # ---------- 파생 경로 ---------- #
    @property
    def run_slug(self) -> str:
        """실행을 유일하게 식별하는 슬러그. 산출물 파일명의 공통 접미사입니다."""
        return f"{self.experiment}_{self.dataset_name}_{self.model_name}"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"best_model_{self.run_slug}.pth"

    @property
    def threshold_path(self) -> Path:
        """val에서 구한 precision@0.90 threshold를 test로 넘기는 핸드오프 파일."""
        return self.checkpoint_dir / f"best_val_threshold_90_{self.run_slug}.txt"

    def results_dir(self, split: str) -> Path:
        """`results/{experiment}/{model}/{dataset}/{split}` 경로를 돌려줍니다."""
        return (
            self.results_root
            / self.experiment
            / self.model_name
            / self.dataset_name
            / split
        )

    def with_overrides(self, **kwargs: Any) -> "ExperimentConfig":
        """HPO 등에서 일부 하이퍼파라미터만 바꾼 사본을 만듭니다."""
        unknown = set(kwargs) - {f for f in self.__dataclass_fields__}
        if unknown:
            raise ValueError(f"알 수 없는 설정 필드: {sorted(unknown)}")
        return replace(self, **kwargs)

    def describe(self) -> str:
        """사람이 읽을 한 줄 요약 (sweep --dry-run 출력용)."""
        return (
            f"{self.experiment:<12} {self.model_name:<16} {self.dataset_name:<24} "
            f"[{self.backbone_tuning}/{self.head}/{self.loss}/{self.selection_metric}]"
        )


########### #
# YAML 로드
########### #
def load_hyperparams(file_path: Path) -> dict[str, Any]:
    """하이퍼파라미터 파일을 로드합니다."""
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_experiments(hyperparams: dict[str, Any]) -> list[str]:
    return sorted(hyperparams.get("experiment", {}))


def list_models(hyperparams: dict[str, Any]) -> list[str]:
    """`PATH` 키를 가진 항목만 실제 선택 가능한 모델입니다."""
    return [
        k
        for k, v in hyperparams.get("model", {}).items()
        if isinstance(v, dict) and "PATH" in v
    ]


def list_datasets(hyperparams: dict[str, Any]) -> list[str]:
    """`DATASET_DIR` 키를 가진 항목만 실제 선택 가능한 데이터셋입니다."""
    return [
        k
        for k, v in hyperparams.get("data", {}).items()
        if isinstance(v, dict) and "DATASET_DIR" in v
    ]


def _pick(options: list[str], key: str, kind: str) -> str:
    if key not in options:
        raise ValueError(
            f"알 수 없는 {kind}: {key!r}\n  선택 가능: {', '.join(options)}"
        )
    return key


def _validate_literal(value: str, literal: Any, field_name: str) -> str:
    allowed = get_args(literal)
    if value not in allowed:
        raise ValueError(
            f"{field_name}의 값이 잘못됐습니다: {value!r}\n  선택 가능: {', '.join(allowed)}"
        )
    return value


########### #
# 설정 조립
########### #
def build_config(
    hyperparams: dict[str, Any],
    experiment: str,
    model_key: str,
    dataset_key: str,
) -> ExperimentConfig:
    """YAML + 선택된 키 3개로부터 검증된 `ExperimentConfig`를 만듭니다."""
    experiment = _pick(list_experiments(hyperparams), experiment, "experiment")
    model_key = _pick(list_models(hyperparams), model_key, "model")
    dataset_key = _pick(list_datasets(hyperparams), dataset_key, "dataset")

    exp = hyperparams["experiment"][experiment]
    model_cfg = hyperparams["model"][model_key]
    data_cfg = hyperparams["data"][dataset_key]
    model_block = hyperparams["model"]
    train_block = hyperparams["train"]
    output_block = hyperparams.get("output", {})

    return ExperimentConfig(
        experiment=experiment,
        model_name=model_cfg["NAME"],
        dataset_name=data_cfg["DATASET_NAME"],
        model_path=Path(model_cfg["PATH"]),
        dataset_dir=Path(data_cfg["DATASET_DIR"]),
        batch_size=hyperparams["data"]["BATCH_SIZE"],
        image_size=hyperparams["data"]["IMAGE_SIZE"],
        backbone_tuning=_validate_literal(
            exp["BACKBONE_TUNING"], BackboneTuning, "BACKBONE_TUNING"
        ),
        head=_validate_literal(exp["HEAD"], HeadType, "HEAD"),
        loss=_validate_literal(exp["LOSS"], LossType, "LOSS"),
        selection_metric=_validate_literal(
            exp["SELECTION_METRIC"], SelectionMetric, "SELECTION_METRIC"
        ),
        num_classes=model_block["NUM_CLASSES"],
        hidden_dim1=model_block["HIDDEN_DIM1"],
        hidden_dim2=model_block["HIDDEN_DIM2"],
        head_dropout=model_block.get("HEAD_DROPOUT", 0.5),
        lora_rank=model_block["LORA_RANK"],
        lora_alpha=model_block["LORA_ALPHA"],
        lora_dropout=model_block.get("LORA_DROPOUT", 0.5),
        target_modules=list(model_block["TARGET_MODULES"]),
        label_smoothing=float(model_block.get("LABEL_SMOOTHING", 0.0)),
        focal_alpha=list(train_block.get("FOCAL_ALPHA", [0.3, 0.4, 0.3])),
        focal_gamma=float(train_block.get("FOCAL_GAMMA", 2.0)),
        num_epochs=train_block["NUM_EPOCHS"],
        learning_rate=float(train_block["LEARNING_RATE"]),
        weight_decay=float(train_block["WEIGHT_DECAY"]),
        early_stopping_patience=train_block["EARLY_STOPPING_PATIENCE"],
        accumulation_steps=train_block.get("ACCUMULATION_STEPS", 1),
        results_root=Path(output_block.get("RESULTS_ROOT", DEFAULT_RESULTS_ROOT)),
        checkpoint_dir=Path(hyperparams["test"]["linear_lora"]["CHECKPOINT_DIR"]),
    )
