"""실험 행렬 러너 — 자동화 파이프라인의 진입점.

experiment × model × dataset 조합을 순회하며 전체 파이프라인을 실행합니다.
한 조합이 실패해도 나머지를 계속 진행하고, 마지막에 요약을 출력합니다.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fgvc.config import ExperimentConfig, build_config


@dataclass
class SweepResult:
    config: ExperimentConfig
    status: str  # "ok" | "failed"
    score: float | None = None
    error: str | None = None


def build_matrix(
    hyperparams: dict[str, Any],
    experiments: list[str],
    models: list[str],
    datasets: list[str],
) -> list[ExperimentConfig]:
    """조합별 설정을 미리 조립합니다. 잘못된 키는 여기서 즉시 드러납니다."""
    return [
        build_config(
            hyperparams=hyperparams,
            experiment=experiment,
            model_key=model_key,
            dataset_key=dataset_key,
        )
        for experiment in experiments
        for model_key in models
        for dataset_key in datasets
    ]


def print_plan(configs: list[ExperimentConfig]) -> None:
    """실행 계획과 산출물 경로를 출력합니다 (--dry-run)."""
    print(f"\n실행 계획: {len(configs)}개 조합\n" + "=" * 100)
    for index, config in enumerate(configs, start=1):
        print(f"[{index:>3}/{len(configs)}] {config.describe()}")
        print(f"        checkpoint : {config.checkpoint_path}")
        print(f"        threshold  : {config.threshold_path}")
        print(f"        results    : {config.results_dir('val')} | {config.results_dir('test')}")
    print("=" * 100)

    # 경로 충돌 검사 — 폴더 복제 시절의 덮어쓰기 사고를 막는 안전장치
    checkpoints = [c.checkpoint_path for c in configs]
    duplicates = {p for p in checkpoints if checkpoints.count(p) > 1}
    if duplicates:
        raise RuntimeError(
            "체크포인트 경로가 충돌합니다 (서로 덮어씁니다):\n  "
            + "\n  ".join(str(p) for p in sorted(duplicates))
        )
    print("경로 충돌 없음 ✓\n")


def run_sweep(
    configs: list[ExperimentConfig],
    hyperparam_path: Path,
) -> list[SweepResult]:
    """조합을 순차 실행합니다. 개별 실패는 기록만 하고 계속 진행합니다."""
    from fgvc.pipeline import run_full_pipeline

    results: list[SweepResult] = []

    for index, config in enumerate(configs, start=1):
        print("\n" + "#" * 100)
        print(f" [{index}/{len(configs)}] {config.describe()}")
        print("#" * 100 + "\n")

        try:
            score = run_full_pipeline(
                config=config, hyperparam_path=hyperparam_path
            )
            results.append(SweepResult(config=config, status="ok", score=score))
        except Exception as exc:  # 한 조합의 실패가 전체 행렬을 중단시키지 않도록
            traceback.print_exc()
            results.append(
                SweepResult(config=config, status="failed", error=f"{type(exc).__name__}: {exc}")
            )

    print_summary(results)
    return results


def print_summary(results: list[SweepResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status == "failed"]

    print("\n" + "=" * 100)
    print(f" SWEEP 요약 — 성공 {len(ok)} / 실패 {len(failed)} (전체 {len(results)})")
    print("=" * 100)

    for result in results:
        if result.status == "ok":
            metric = result.config.selection_metric
            print(f"  ✓ {result.config.describe()}  best {metric}={result.score:.4f}")
        else:
            print(f"  ✗ {result.config.describe()}  {result.error}")
    print("=" * 100)
