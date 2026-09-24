"""DINOv3 백본 로드와 분류 헤드·파인튜닝 전략 조립.

`BACKBONE_TUNING`(lora/frozen)과 `HEAD`(mlp/linear) 두 축의 조합을 여기서 처리합니다.
기존 3개 폴더의 `get_model()` 구현을 하나로 합친 것입니다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from fgvc.config import ExperimentConfig


########### #
# 분류 헤드
########### #
def build_head(config: ExperimentConfig, embed_dim: int) -> nn.Module:
    """`HEAD` 축에 따라 분류 헤드를 만듭니다."""
    if config.head == "linear":
        # Linear Probing의 정석: 1-layer 선형 분류기
        return nn.Sequential(nn.Linear(embed_dim, config.num_classes))

    if config.head == "mlp":
        return nn.Sequential(
            nn.Linear(embed_dim, config.hidden_dim1),
            nn.LayerNorm(config.hidden_dim1),
            nn.GELU(),
            nn.Dropout(p=config.head_dropout),
            nn.Linear(config.hidden_dim1, config.hidden_dim2),
            nn.LayerNorm(config.hidden_dim2),
            nn.GELU(),
            nn.Dropout(p=config.head_dropout),
            nn.Linear(config.hidden_dim2, config.num_classes),
        )

    raise ValueError(f"지원하지 않는 HEAD: {config.head!r}")


########### #
# 모델 조립
########### #
def build_model(config: ExperimentConfig) -> nn.Module:
    """DINOv3 백본을 로드하고 헤드 교체 + 파인튜닝 전략을 적용합니다."""
    model = torch.hub.load(
        repo_or_dir="dinov3",
        model=config.model_name,
        source="local",
        weights=str(config.model_path),
    )

    # 백본의 임베딩 차원 추출
    embed_dim = getattr(model, "embed_dim")
    model.head = build_head(config=config, embed_dim=embed_dim)

    if config.backbone_tuning == "frozen":
        # Linear Probing: 백본 전체 동결 후 헤드만 학습
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        return model

    if config.backbone_tuning == "lora":
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            modules_to_save=["head"],
        )
        return get_peft_model(model=model, peft_config=lora_config)

    raise ValueError(f"지원하지 않는 BACKBONE_TUNING: {config.backbone_tuning!r}")


def infer_patch_size(model: nn.Module, default: int = 14) -> int:
    """PEFT 래퍼를 벗겨 백본의 patch_size를 얻습니다."""
    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    return getattr(base_vit, "patch_size", default)
