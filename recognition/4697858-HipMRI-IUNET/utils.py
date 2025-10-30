import matplotlib.pyplot as plt


def plot_curves(train_losses, val_losses, train_dice, val_dice, save_path):
    """
    Produces graphs of loss across epochs and dice coefficients over epochs

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_dice: List of training dice coefficients
        val_dice: List of validation dice coefficients
        save_path: Path to save the graph image to
    """
    epochs = range(1, len(train_losses) + 1)

    figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the training and validation losses on a single graph
    axis1.plot(epochs, train_losses, 'b-', label="Training Loss")
    axis1.plot(epochs, val_losses, 'r-', label="Validation Loss")
    axis1.set_xlabel("Epoch")
    axis1.set_ylabel("Loss")
    axis1.set_title("Training and Validation Loss")
    axis1.legend()
    axis1.grid(True, alpha=0.5)

    # Plot the training and validation dice coefficients on a single graph
    axis2.plot(epochs, train_dice, 'b-', label="Training Dice")
    axis2.plot(epochs, val_dice, 'r-', label="Validation Dice")
    axis2.set_xlabel("Epoch")
    axis2.set_ylabel("Dice Coefficient")
    axis2.set_title("Dice Coefficients for Prostate Class")
    axis2.legend()
    axis2.grid(True, alpha=0.5)
    axis2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training Curves saved to {save_path}")
    plt.close()
