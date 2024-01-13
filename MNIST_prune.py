import os
import sys
import torch
import torch.optim

from tqdm import tqdm

from common.path import CKPT_DIR
from common.cli import get_parser
from common.net import MNIST_Net
from common.dataloader import get_mnist_dataloader
from common.procedure import train_step, eval_step, DEVICE, USE_GPU

from common.pruning import ChannelPruner, gradient_based_importance, get_structure_info


best_ckpt: str = os.path.join(CKPT_DIR, "best_mnist.pth")
best_loss: float = float("inf")
best_acc: float = -0.0


def save_best_ckpt(model, epoch, loss, acc):
    global best_loss, best_acc
    if loss < best_loss or acc > best_acc:
        best_loss = loss
        best_acc = acc
        torch.save(model.state_dict(), best_ckpt)
        print("info: saved best ckpt:", best_ckpt)


def prepare_gradients(model, dataset, criterion, max_batches=100):
    model.train()

    total = min(max_batches, len(dataset))
    progress_bar = tqdm(dataset, desc=f"Prepare Gradients", total=total)
    for i, (images, labels) in enumerate(dataset):
        if i >= total: break

        if USE_GPU:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

        # forward
        logits = model(images)
        loss = criterion(logits, labels)

        # accumulate gradients
        loss.backward()
        progress_bar.update()


def main(args):
    # Get dataset
    ds_train, ds_valid = get_mnist_dataloader(args)

    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    # Create model
    model = MNIST_Net(args.factor).to(DEVICE)
    model.load_state_dict(torch.load("./ckpt/best_mnist.pth"))

    # evaluate before pruning
    loss, acc = eval_step(model, ds_valid, criterion, args)
    print(f"Before Pruning: {loss=:.4f}, {acc=:.2f}")

    # channel pruning
    prepare_gradients(model, ds_valid, criterion)
    pruner = ChannelPruner(gradient_based_importance, args.pruning_ratio)
    pruner.prune(model, get_structure_info())

    # Evaluate model
    loss, acc = eval_step(model, ds_valid, criterion, args)
    print(f"After Pruning: {loss=:.4f}, {acc=:.2f}")


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
