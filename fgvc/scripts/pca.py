"""백본 패치 토큰의 PCA → RGB 시각화.

분류기 없이 DINOv3 백본 자체가 뽑는 특징의 semantic consistency를 확인하는 용도라
학습된 체크포인트나 실험 설정(`experiment` 축)이 필요 없습니다.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.datasets import ImageFolder

NUM_SAMPLES = 5


def main(hyperparams: dict[str, Any], model_key: str, dataset_key: str) -> None:
    model_config = hyperparams["model"][model_key]
    dataset_config = hyperparams["data"][dataset_key]

    model_name = model_config["NAME"]
    model_path = Path(model_config["PATH"])
    dataset_dir = Path(dataset_config["DATASET_DIR"])
    dataset_name = dataset_config["DATASET_NAME"]
    image_size = hyperparams["data"]["IMAGE_SIZE"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[*] Loading DINOv3 backbone: {model_name}...")
    model = torch.hub.load(
        repo_or_dir="dinov3",
        model=model_name,
        source="local",
        weights=str(model_path),
    ).to(device)
    model.eval()

    patch_size = getattr(model, "patch_size", 14)
    if "16" in model_name:
        patch_size = 16

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = ImageFolder(root=str(dataset_dir / "test"), transform=transform)
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)

    images = []
    original_images = []
    for idx in indices:
        img_tensor, _ = dataset[idx]
        images.append(img_tensor)
        # Plotting을 위해 원본 이미지 보존
        img_path = dataset.samples[idx][0]
        original_images.append(
            Image.open(img_path).convert("RGB").resize((image_size, image_size))
        )

    x = torch.stack(images).to(device)

    print(f"[*] Extracting features for {NUM_SAMPLES} images...")
    with torch.no_grad():
        features = model.forward_features(x)

    # forward_features의 반환 타입에 맞게 패치 토큰 추출
    if isinstance(features, dict) and "x_norm_patchtokens" in features:
        patch_tokens = features["x_norm_patchtokens"]
    elif isinstance(features, tuple):
        patch_tokens = features[0]
    else:
        patch_tokens = features

    h_patch = image_size // patch_size
    w_patch = image_size // patch_size

    # CLS 토큰이 포함되어 있다면 제외
    if patch_tokens.shape[1] == h_patch * w_patch + 1:
        patch_tokens = patch_tokens[:, 1:, :]

    batch, _, dim = patch_tokens.shape
    features_flat = patch_tokens.reshape(-1, dim).cpu().numpy()

    print(f"[*] Applying PCA (reducing from {dim} to 3 dimensions)...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_flat)

    # RGB 시각화를 위해 [0, 1] 범위로 Min-Max 스케일링
    for i in range(3):
        comp = pca_features[:, i]
        pca_features[:, i] = (comp - comp.min()) / (comp.max() - comp.min())

    pca_features = pca_features.reshape(batch, h_patch, w_patch, 3)

    results_dir = Path("results_pca_visualization") / f"{dataset_name}_{model_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(4 * NUM_SAMPLES, 8))
    for i in range(NUM_SAMPLES):
        axes[0, i].imshow(original_images[i])
        axes[0, i].axis("off")
        if i == NUM_SAMPLES // 2:
            axes[0, i].set_title("Original Images", fontsize=20, pad=20)

        # 부드러운 컬러맵을 위해 패치 그리드를 원본 해상도로 보간
        pca_img = torch.tensor(pca_features[i]).permute(2, 0, 1).unsqueeze(0)
        pca_img_up = F.interpolate(
            pca_img, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        axes[1, i].imshow(pca_img_up.squeeze(0).permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
        if i == NUM_SAMPLES // 2:
            axes[1, i].set_title(
                "PCA Feature Map (Semantic Consistency)", fontsize=20, pad=20
            )

    plt.tight_layout()
    save_path = results_dir / "pca_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[*] PCA visualization saved to:\n -> {save_path}")
