import argparse
import os
import torch
import torch.cuda
import torch.optim as optim
import utils
from dataset import get_dataloaders
from modules import IUNet2D, DiceLoss, dice_coefficient
from tqdm import tqdm
import numpy as np


class EarlyStop:
    def __init__(self, patience=10, min_change=0.001, mode='max'):
        self.patience = patience
        self.min_change = min_change
        self.mode = mode
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, metric):
        if self.best is None:
            return False

        if self.mode == 'max':
            if metric > self.best + self.min_change:
                self.best = metric
                self.counter = 0
            else:
                self.counter += 1
        else:
            if metric < self.best - self.min_change:
                self.best = metric
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop


def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimiser.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    average_loss = total_loss / num_batches
    return average_loss


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_dice = []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc='Validation', leave=False)

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)

            total_loss += loss.item()
            num_batches += 1

            dice = dice_coefficient(outputs, masks)
            all_dice.append(dice.cpu().numpy())

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    average_loss = total_loss / num_batches
    average_dice = np.mean(all_dice, axis=0)

    return average_loss, average_dice


def parse_cmd_args():
    args = argparse.ArgumentParser(description="Train 2D Improved UNet")
    args.add_argument("--directory", type=str, default="./data")
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--num_workers", type=int, default=4)
    args.add_argument("--target_label", type=int, default=1)
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--deep_supervision", action='store_true')
    args.add_argument("--dice_weight", type=float, default=0.5)
    args.add_argument("--smooth", type=float, default=1.0)
    args.add_argument("--num_classes", type=int, default=6)
    args.add_argument("--output", type=str, default="./output")
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--weight_decay", type=float, default=1e-5)
    args.add_argument("--patience", type=int, default=10)

    return args.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_cmd_args()

    train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=args.directory,
            batch_size=args.batch_size,
            num_workers=args.num_workers if torch.cuda.is_available() else 0,
            normalize=True)

    model = IUNet2D(in_channels=1, out_channels=args.num_classes, base_channel=32, depth=4, deep_supervision=args.deep_supervision).to(device)

    early_stop = EarlyStop(patience=args.patience, mode='max')

    criterion = DiceLoss(args.smooth)

    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.5, patience=5)

    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    best_dice = 0.0

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)

        val_loss, val_dice = validate(model, val_loader, criterion, device)

        with torch.no_grad():
            model.eval()
            train_dice_batch = []
            for i, (images, masks) in enumerate(train_loader):
                if i >= 10:
                    break
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dice = dice_coefficient(outputs, masks)
                train_dice_batch.append(dice.cpu().numpy())
            train_dice = np.mean(train_dice_batch, axis=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice[args.target_label])
        val_dices.append(val_dice[args.target_label])

        scheduler.step(val_dice[args.target_label])

        if val_dice[args.target_label] > best_dice:
            best_dice = val_dice[args.target_label]
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            print("Saved new best model")
            print(f"Dice: {best_dice:.4f}")

        if early_stop(val_dice[args.target_label]):
            print(f"\nConvergence detected after {epoch+1} epochs")
            break

    utils.plot_curves(train_losses, val_losses, train_dices, val_dices, os.path.join(args.output, 'training_curves.png'))


if __name__ == "__main__":
    main()
