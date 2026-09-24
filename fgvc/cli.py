"""CLI 진입점.

인자가 주어지면 비대화식으로 실행하고, 없으면 기존처럼 프롬프트로 물어봅니다.
`python -m fgvc.cli`로 실행하면 CWD가 `sys.path[0]`이 되므로 별도의
`PYTHONPATH` 설정이나 `sys.path` 조작이 필요 없습니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fgvc.config import (
    build_config,
    list_datasets,
    list_experiments,
    list_models,
    load_hyperparams,
)

DEFAULT_HYPERPARAMS = "./hyperparams.yaml"


def _resolve(value: str | None, options: list[str], kind: str) -> str:
    """CLI 인자를 우선하고, 없으면 프롬프트로 폴백합니다."""
    if value:
        return value.strip()
    return input(f"Enter {kind} ({', '.join(options)}): ").lower().strip()


def _resolve_many(value: str | None, options: list[str], kind: str) -> list[str]:
    """쉼표로 구분된 다중 선택. `all`이면 전체를 씁니다."""
    raw = value if value else input(f"Enter {kind} (comma-separated, or 'all'): ")
    raw = raw.strip().lower()
    if raw == "all":
        return list(options)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fgvc",
        description="DINOv3 + LoRA 기반 철스크랩 미세 분류 학습/평가 파이프라인",
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=DEFAULT_HYPERPARAMS,
        help="하이퍼파라미터 YAML 경로",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_selection_args(sub: argparse.ArgumentParser, many: bool = False) -> None:
        suffix = "s (comma-separated, or 'all')" if many else ""
        sub.add_argument("--experiment" + ("s" if many else ""), type=str, default=None,
                         help=f"실험 정의 키{suffix}")
        sub.add_argument("--model" + ("s" if many else ""), type=str, default=None,
                         help=f"모델 키{suffix}")
        sub.add_argument("--dataset" + ("s" if many else ""), type=str, default=None,
                         help=f"데이터셋 키{suffix}")

    train_parser = subparsers.add_parser("train", help="학습 + 테스트 + 샘플 시각화")
    add_selection_args(train_parser)

    hpo_parser = subparsers.add_parser("hpo", help="Optuna 하이퍼파라미터 최적화")
    add_selection_args(hpo_parser)
    hpo_parser.add_argument("--n-trials", type=int, default=None, help="탐색 trial 수")

    sweep_parser = subparsers.add_parser("sweep", help="실험 행렬 일괄 실행")
    add_selection_args(sweep_parser, many=True)
    sweep_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실행하지 않고 조합·산출물 경로만 출력합니다",
    )

    present_parser = subparsers.add_parser(
        "present", help="발표용 Attention/Grad-CAM 히트맵 생성"
    )
    add_selection_args(present_parser)

    # PCA는 백본만 보므로 실험 설정(experiment 축)이 필요 없습니다.
    pca_parser = subparsers.add_parser("pca", help="백본 패치 토큰 PCA 시각화")
    pca_parser.add_argument("--model", type=str, default=None, help="모델 키")
    pca_parser.add_argument("--dataset", type=str, default=None, help="데이터셋 키")

    return parser


def _resolve_single(args: argparse.Namespace, hyperparams: dict[str, Any]):
    experiment = _resolve(args.experiment, list_experiments(hyperparams), "experiment")
    model_key = _resolve(args.model, list_models(hyperparams), "model name")
    dataset_key = _resolve(args.dataset, list_datasets(hyperparams), "dataset name")
    return build_config(
        hyperparams=hyperparams,
        experiment=experiment,
        model_key=model_key,
        dataset_key=dataset_key,
    )


def main(argv: list[str] | None = None) -> None:
    try:
        _dispatch(_build_parser().parse_args(argv))
    except (ValueError, FileNotFoundError) as exc:
        # 설정 오류는 트레이스백 없이 메시지만 보여줍니다 (자동화 로그 가독성).
        raise SystemExit(f"\n[설정 오류] {exc}\n")


def _dispatch(args: argparse.Namespace) -> None:
    hyperparam_path = Path(args.hyperparams)
    hyperparams = load_hyperparams(file_path=hyperparam_path)

    if args.command == "train":
        from fgvc.pipeline import run_full_pipeline

        config = _resolve_single(args, hyperparams)
        run_full_pipeline(config=config, hyperparam_path=hyperparam_path)

    elif args.command == "hpo":
        from fgvc.hpo import N_TRIALS, run_hpo

        config = _resolve_single(args, hyperparams)
        run_hpo(
            config=config,
            hyperparam_path=hyperparam_path,
            n_trials=args.n_trials or N_TRIALS,
        )

    elif args.command == "sweep":
        from fgvc.sweep import build_matrix, print_plan, run_sweep

        configs = build_matrix(
            hyperparams=hyperparams,
            experiments=_resolve_many(
                args.experiments, list_experiments(hyperparams), "experiments"
            ),
            models=_resolve_many(args.models, list_models(hyperparams), "models"),
            datasets=_resolve_many(
                args.datasets, list_datasets(hyperparams), "datasets"
            ),
        )
        print_plan(configs)
        if not args.dry_run:
            run_sweep(configs=configs, hyperparam_path=hyperparam_path)

    elif args.command == "present":
        from fgvc.scripts.presentation import main as present_main

        present_main(config=_resolve_single(args, hyperparams))

    elif args.command == "pca":
        from fgvc.scripts.pca import main as pca_main

        pca_main(
            hyperparams=hyperparams,
            model_key=_resolve(args.model, list_models(hyperparams), "model name"),
            dataset_key=_resolve(
                args.dataset, list_datasets(hyperparams), "dataset name"
            ),
        )


if __name__ == "__main__":
    main()
