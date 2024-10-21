
import matplotlib.pyplot as plt


def plot_train_results_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train vs Validation Loss')
    plt.show()
    plt.savefig("train_vs_val_loss.png")


def plot_train_results_ssim(ssim_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(ssim_scores, label='SSIM Score')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')
    plt.show()
    plt.savefig("ssim_over_epochs.png")