import os
import sys
import torch
import torch.optim

from common.path import CKPT_DIR
from common.cli import get_parser
from common.net import MNIST_Net
from common.dataloader import get_mnist_dataloader
from common.procedure import train_step, eval_step, DEVICE


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


def main(args):
    # Get dataset
    ds_train, ds_valid = get_mnist_dataloader(args)

    # Create model
    model = MNIST_Net(args.factor).to(DEVICE)
    print(model)

    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train loop
    for epoch in range(1, args.num_epochs + 1):
        train_step(model, ds_train, criterion, optimizer, args, epoch)
        if epoch >= args.eval_epoch and epoch % args.eval_freq == 0:
            loss, acc = eval_step(model, ds_valid, criterion, args, epoch)
            save_best_ckpt(model, epoch, loss, acc)



if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
