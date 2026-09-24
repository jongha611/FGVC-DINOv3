"""발표용 Attention + Grad-CAM 히트맵 일괄 생성.

학습된 체크포인트를 불러와 `data/Iron-Scraps/samples_for_presentation/` 아래의
이미지 전부에 대해 히트맵을 렌더링합니다.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from fgvc.config import ExperimentConfig
from fgvc.data import SquarePad
from fgvc.metrics import attention
from fgvc.metrics.attention import (
    generate_attention_heatmap,
    generate_gradcam_heatmap,
    patch_attention_module,
    save_visualization_plot,
)
from fgvc.models import infer_patch_size
from fgvc.pipeline import build_runtime

SAMPLES_DIR = Path("./data/Iron-Scraps/samples_for_presentation")


def main(config: ExperimentConfig) -> None:
    runtime = build_runtime(config)
    model, device = runtime.model, runtime.device

    # Grad-CAM을 위해 학습된 가중치가 반드시 필요합니다.
    model.load_state_dict(
        torch.load(config.checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    # 어텐션 몽키 패치 적용
    patch_attention_module(model)
    patch_size = infer_patch_size(model)

    results_dir = (
        config.results_root
        / "presentation_heatmaps"
        / f"{config.experiment}_{config.model_name}_{config.dataset_name}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # 검증용 트랜스폼 (test 데이터셋 처리와 동일)
    val_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(size=(config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class_names = runtime.test_loader.dataset.classes
    print(f"[*] Loaded model '{config.model_name}' trained on '{config.dataset_name}'.")
    print(f"[*] Processing images in {SAMPLES_DIR}...")

    image_files = sorted(
        list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png"))
    )
    if not image_files:
        print(f"[!] {SAMPLES_DIR} 에 이미지가 없습니다.")
        return

    for index, img_path in enumerate(image_files):
        raw_img = Image.open(img_path).convert("RGB")
        img_input = val_transform(raw_img).unsqueeze(0).to(device)
        img_input.requires_grad_(True)

        # Grad-CAM을 위해 inference_mode를 쓰지 않고 gradient를 계산합니다.
        model.zero_grad()
        logits = model(img_input)
        probs = torch.softmax(logits, dim=1).squeeze().cpu()

        pred_label = int(probs.argmax())
        confidence = float(probs[pred_label])
        pred_name = class_names[pred_label]

        # 분류기가 결정한 클래스에 대해 역전파
        logits[0, pred_label].backward(retain_graph=True)

        heatmap = generate_attention_heatmap(
            model=model,
            img_tensor=img_input,
            image_size=config.image_size,
            patch_size=patch_size,
        )
        gradcam_heatmap = generate_gradcam_heatmap(
            model=model,
            img_tensor=img_input,
            image_size=config.image_size,
            patch_size=patch_size,
        )

        # 다음 루프를 위해 캐시된 어텐션 및 그래디언트 초기화
        attention.captured_attn_weights = None
        attention.captured_attn_gradients = None

        save_path = results_dir / f"attention_gradcam_{img_path.stem}.png"
        save_visualization_plot(
            orig_img=SquarePad()(raw_img),
            heatmap=heatmap,
            gradcam_heatmap=gradcam_heatmap,
            # 폴더 내 임의의 이미지이므로 정답 라벨은 알 수 없습니다.
            true_label_name="Unknown",
            pred_label_name=pred_name,
            confidence=confidence,
            save_path=save_path,
        )
        print(
            f"[{index+1}/{len(image_files)}] Saved: {save_path.name} "
            f"(Pred: {pred_name} | {confidence*100:.1f}%)"
        )

    print(f"[*] Finished processing all {len(image_files)} images.")
    print(f"[*] Heatmaps are saved in: {results_dir.resolve()}")
