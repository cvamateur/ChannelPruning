import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")


def train_step(model, dataset, criterion, optimizer, args, epoch=0):
    model.train()

    total_loss: float = 0.0
    total_correct: int = 0
    acc: float = 0.0
    progress_bar = tqdm(dataset, desc=f"Training  : {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        if USE_GPU:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

        # forward
        logits = model(images)
        loss = criterion(logits, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calc loss
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # metric
        preds = logits.argmax(dim=1)
        total_correct += torch.eq(preds, labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {loss:.4f}, acc: {acc:.1f}%".format(loss=avg_loss, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if (i + 1) % args.log_freq == 0:
            progress_bar.update(args.log_freq)

    progress_bar.update(len(dataset) - progress_bar.n)
    return total_loss, acc


@torch.no_grad()
def eval_step(model, dataset, criterion, args, epoch=0):
    model.eval()

    total_loss: float = 0.0
    total_correct: int = 0
    acc: float = 0.0
    progress_bar = tqdm(dataset, desc=f"Validation: {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        if USE_GPU:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

        # forward
        logits = model(images)
        loss = criterion(logits, labels)

        # calc loss
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # metric
        preds = logits.argmax(dim=1)
        total_correct += torch.eq(preds, labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {loss:.4f}, acc: {acc:.1f}%".format(loss=avg_loss, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if (i + 1) % args.log_freq == 0:
            progress_bar.update(args.log_freq)

    progress_bar.update(len(dataset) - progress_bar.n)
    return total_loss, acc


def train_step_v2(model, dataset, criterion, optimizer, args, epoch=0, loss_weight=0.1):
    model.train()
    criterion_ce, criterion_ct = criterion

    total_loss: float = 0.0
    total_correct: int = 0
    acc: float = 0.0
    progress_bar = tqdm(dataset, desc=f"Training  : {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        if USE_GPU:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

        # forward
        feats = model.forward_features(images)
        logits = model.forward_logits(feats)
        loss_ce = criterion_ce(logits, labels)
        loss_ct = criterion_ct(feats, labels)
        loss = loss_ce + loss_weight * loss_ct

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calc loss
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # metric
        preds = logits.argmax(dim=1)
        total_correct += torch.eq(preds, labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {loss:.4f} | loss_ce: {loss_ce:.4f} | loss_ct: {loss_ct:.4f} | acc: {acc:.1f}%".format(
            loss=avg_loss, loss_ce=loss_ce, loss_ct=loss_ct, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if (i + 1) % args.log_freq == 0:
            progress_bar.update(args.log_freq)

    progress_bar.update(len(dataset) - progress_bar.n)
    return total_loss, acc


def train_step_v3(model, dataset, criterion, optimizer, args, epoch=0, loss_weight=0.1):
    model.train()
    criterion_ce, criterion_ct, criterion_rn = criterion

    total_loss: float = 0.0
    total_correct: int = 0
    acc: float = 0.0
    progress_bar = tqdm(dataset, desc=f"Training  : {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        if USE_GPU:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

        # forward
        feats = model.forward_features(images)
        logits = model.forward_logits(feats)
        loss_ce = criterion_ce(logits, labels)
        loss_ct = criterion_ct(feats, labels)
        loss_rn = criterion_rn(feats)
        loss = loss_ce + loss_weight * loss_ct + loss_rn

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calc loss
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # metric
        preds = logits.argmax(dim=1)
        total_correct += torch.eq(preds, labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {loss:.4f} | " \
                   "loss_ce: {loss_ce:.4f} | " \
                   "loss_ct: {loss_ct:.4f} | " \
                   "loss_rn: {loss_rn:.4f} | " \
                   "acc: {acc:.1f}%".format(
            loss=avg_loss, loss_ce=loss_ce, loss_ct=loss_ct, loss_rn=loss_rn, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if (i + 1) % args.log_freq == 0:
            progress_bar.update(args.log_freq)

    progress_bar.update(len(dataset) - progress_bar.n)
    return total_loss, acc
