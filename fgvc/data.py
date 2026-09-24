from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader

import PIL


####################### #
# 심플한 커스텀 패딩 클래스
####################### #
class SquarePad:
    """정사각형이 아닌 이미지를 정사각형으로 패딩 처리하는 클래스입니다."""

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        w, h = image.size
        max_wh = max(w, h)

        pad_left = int((max_wh - w) / 2)
        pad_top = int((max_wh - h) / 2)
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)

        return F.pad(img=image, padding=padding)


############ #
# 데이터 로더
############ #
def get_data_loader(
    dataset_dir: Path, batch_size: int = 32, image_size: int = 224
) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 파이토치의 이미지폴더는 경로 폴더 하위 폴더를 바로 클래스명으로 인식합니다.
    # 따라서, 경로를 정확하게 클래스 폴더 상위폴더로 잡아줘야 합니다.
    # 이미지폴더로 생성한 객체는 데이터셋 타입으로 데이터로더에 즉시 인자로 넣을 수 있습니다.
    train_dataset = ImageFolder(root=dataset_dir / "train", transform=train_transform)
    val_dataset = ImageFolder(root=dataset_dir / "val", transform=val_transform)
    test_dataset = ImageFolder(root=dataset_dir / "test", transform=val_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
