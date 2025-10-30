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
    """
    Detect when a training or validation metrics stop improving and stop the process early

    Args:
        patience: The number of iterations to wait for a significant change
        min_change: The degree of difference in the metric to consider a change significant
        mode: Whether the process is aiming for maximising or minimising the metric
    """
    def __init__(self, patience=10, min_change=0.001, mode='max'):
        self.patience = patience
        self.min_change = min_change
        self.mode = mode
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric

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


def save_checkpoint(model, optimiser, epoch, best_dice, train_losses, val_losses, train_dices, val_dices, filepath):
    """
    Saves the current state in a checkpoint

    Args:
        model: Model to save
        optimiser: Optimiser to save
        epoch: The training epoch just completed
        best_dice: The best dice coefficient achieved so far
        train_losses: The training losses for each epoch
        val_losses: The validation losses for each epoch
        train_dices: The training dice coefficients for each epoch
        val_dices: The validation dice coefficients for each epoch
        filepath: The path where the checkpoint will be saved to
    """
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "best_dice": best_dice,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_dices": train_dices,
        "val_dices": val_dices
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimiser, filepath):
    """
    Loads a saved checkpoint

    Args:
        model: The model to load into
        optimiser: The optimiser to load into
        filepath: The path to the checkpoint
    """
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimiser.load_state_dict(checkpoint["optimiser_state"])
    epoch = checkpoint["epoch"]
    best_dice = checkpoint["best_dice"]
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    train_dices = checkpoint["train_dices"]
    val_dices = checkpoint["val_dices"]
    print(f"Checkpoint loaded from {filepath}")
    return epoch, best_dice, train_losses, val_losses, train_dices, val_dices


def train_one_epoch(model, loader, optimiser, criterion, device):
    """
    Trains a single epoch

    Args:
        model: The model to train
        loader: Training data loader
        optimiser: Optimiser
        criterion: Loss function
        device: Device to train on
    """
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


def validate(model, loader, criterion, device, num_classes):
    """
    Validates a single epoch

    Args:
        model: The model to validate
        loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        num_classes: The number of classes in the dataset
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    dice_sum = np.zeros(num_classes, dtype=np.float32)
    dice_count = np.zeros(num_classes, dtype=np.float32)

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

            # Aggregate dice coefficients, accounting for differing numbers of classes
            for i in range(min(len(dice), num_classes)):
                dice_sum[i] += dice[i].item()
                dice_count[i] += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    average_loss = total_loss / num_batches
    average_dice = np.zeros(num_classes, dtype=np.float32)

    # Average the dice coefficients, accounting for differing numbers of classes
    for i in range(num_classes):
        if dice_count[i] > 0:
            average_dice[i] = dice_sum[i] / dice_count[i]

    return average_loss, average_dice


def parse_cmd_args():
    """
    Parses command line arguments
    """
    args = argparse.ArgumentParser(description="Train 2D Improved UNet")
    args.add_argument("--directory", type=str, default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--num_workers", type=int, default=4)
    args.add_argument("--target_label", type=int, default=1)
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--dice_weight", type=float, default=0.5)
    args.add_argument("--smooth", type=float, default=1.0)
    args.add_argument("--num_classes", type=int, default=6)
    args.add_argument("--output", type=str, default="./output")
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--weight_decay", type=float, default=1e-5)
    args.add_argument("--patience", type=int, default=10)
    args.add_argument("--resume", action="store_true")
    args.add_argument("--save_frequency", type=int, default=10)
    args.add_argument("--resume_checkpoint", type=str, default=None)

    return args.parse_args()


def main():
    """
    Main training function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_cmd_args()
    train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=args.directory,
            batch_size=args.batch_size,
            num_workers=args.num_workers if torch.cuda.is_available() else 0,
            normalize=True)

    model = IUNet2D(in_channels=1, out_channels=args.num_classes, base_channel=32, depth=4).to(device)

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
    # Load from a checkpoint if in resume mode
    if args.resume:
        start_epoch, best_dice, train_losses, val_losses, train_dices, val_dices = load_checkpoint(model, optimiser, args.resume_checkpoint)
        print(best_dice)
        start_epoch += 1

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)

        val_loss, val_dice = validate(model, val_loader, criterion, device, args.num_classes)

        with torch.no_grad():
            model.eval()
            train_dice_sum = np.zeros(args.num_classes, dtype=np.float32)
            train_dice_count = np.zeros(args.num_classes, dtype=np.float32)

            for i, (images, masks) in enumerate(train_loader):
                if i >= 10:
                    break
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dice = dice_coefficient(outputs, masks)

                for j in range(min(len(dice), args.num_classes)):
                    train_dice_sum[j] += dice[j].item()
                    train_dice_count[j] += 1

            train_dice = np.zeros(args.num_classes, dtype=np.float32)
            # Calculate the average dice coefficient, accounting for potential differing number of classes
            for j in range(args.num_classes):
                if train_dice_count[j] > 0:
                    train_dice[j] = train_dice_sum[j] / train_dice_count[j]

        # Add losses and dice coefficients for the completed epoch
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice[args.target_label])
        val_dices.append(val_dice[args.target_label])

        scheduler.step(val_dice[args.target_label])

        # Update best dice score
        if val_dice[args.target_label] > best_dice:
            best_dice = val_dice[args.target_label]
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            print("Saved new best model")
            print(f"Dice: {best_dice:.4f}")

        if early_stop(val_dice[args.target_label]):
            print(f"\nConvergence detected after {epoch+1} epochs")
            break

        if (epoch + 1) % args.save_frequency == 0:
            save_checkpoint(model, optimiser, epoch, best_dice, train_losses, val_losses, train_dices, val_dices, os.path.join(args.output, f"checkpoint_epoch_{epoch+1}.pth"))

    # Produce a plot of the losses and dice coefficients across all epochs
    utils.plot_curves(train_losses, val_losses, train_dices, val_dices, os.path.join(args.output, 'images/training_curves.png'))


if __name__ == "__main__":
    main()
