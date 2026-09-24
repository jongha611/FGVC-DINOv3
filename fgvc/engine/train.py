import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


########## #
# 학습 함수
########## #
def train_model(
    device: torch.device,
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int = 1,
    accumulation_steps: int = 1,
) -> tuple[float, float]:
    model.train()

    # initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    avg_loss = 0.0
    train_acc = 0.0

    optimizer.zero_grad()

    # learning loop
    progress_bar = tqdm(
        iterable=train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
    )
    for index, batch in enumerate(progress_bar):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, labels)

        # gradient accumulation을 위해 스케일링 전의 loss 값을 보존합니다.
        loss_item = loss.item()

        # 스텝수만큼 손실을 나누고 역전파합니다.
        (loss / accumulation_steps).backward()

        # accumulation_steps만큼 스텝을 쌓거나, 마지막 스텝인 경우 optimizer를 1회 업데이트합니다.
        if (index + 1) % accumulation_steps == 0 or (index + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        # loss
        running_loss += loss_item
        avg_loss = running_loss / (index + 1)

        # accuracy
        predictions = logits.argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # postfix
        progress_bar.set_postfix(
            {"avg_loss": f"{avg_loss:.4f}", "train_acc": f"{train_acc:.2f}%"}
        )

    print(
        f"epoch {epoch+1}/{num_epochs}, train loss: {avg_loss:.4f}, train acc: {train_acc:.2f}%"
    )

    return avg_loss, train_acc
