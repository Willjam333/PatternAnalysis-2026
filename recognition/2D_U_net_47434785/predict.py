"""Evaluate the trained baseline U-Net and visualise HipMRI test predictions.
"""


import math
import matplotlib
matplotlib.use("Agg")  # Save figures without opening a window on Rangpur.
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from dataset import HipMRIDataset
from modules import UNet2D, ImprovedUNet2D


# Change DATA_ROOT if running on Windows instead of Rangpur.
DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
MODELS = {
    "baseline": UNet2D,
    "improved": ImprovedUNet2D
}
BATCH_SIZE = 2
BASE_CHANNELS = 64       # Must match the model that created best_model.pt.
NUM_CLASSES = 6
PROSTATE_CLASS = 5       # Verify this label against official dataset metadata.
MAX_VISUALISATIONS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_dice_score(intersections, totals):
    """Calculate Dice for each class using counts from the whole test set.

    Args:
        intersections: Number of correctly predicted pixels for each class.
        totals: Predicted pixel count + true pixel count for each class.

    Returns:
        List of six Dice scores. NaN means a class was absent from both masks.
    """
    scores = []
    for c in range(NUM_CLASSES):
        if totals[c] == 0:
            scores.append(float("nan"))
        else:
            scores.append((2 * intersections[c] / totals[c]).item())
    return scores


def save_comparison(image, ground_truth, prediction, filename):
    """Save an MRI image next to ground-truth and prediction overlays.

    Args:
        image: 2D NumPy array containing MRI intensities.
        ground_truth: 2D NumPy array of true class labels (0-5).
        prediction: 2D NumPy array of predicted class labels (0-5).
        filename: Output image path.

    Returns:
        None. Saves a PNG image.
    """
    colors = ListedColormap(["black", "red", "lime", "blue", "yellow", "magenta"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, title in zip(axes, ["MRI", "Ground truth", "U-Net prediction"]):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    # Hide class 0 so the MRI remains visible beneath the coloured regions.
    axes[1].imshow(np.ma.masked_equal(ground_truth, 0), cmap=colors,
                   vmin=0, vmax=NUM_CLASSES - 1, alpha=0.5, interpolation="nearest")
    axes[2].imshow(np.ma.masked_equal(prediction, 0), cmap=colors,
                   vmin=0, vmax=NUM_CLASSES - 1, alpha=0.5, interpolation="nearest")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def evaluate_model(model, test_loader, save_dir):
    """Run inference over every test slice, calculate Dice and save examples.

    Args:
        model: Trained UNet2D in evaluation mode.
        test_loader: DataLoader returning HipMRIDataset dictionary batches.

    Returns:
        List of six Dice scores for the full test set.
    """
    intersections = torch.zeros(NUM_CLASSES)
    totals = torch.zeros(NUM_CLASSES)
    saved_images = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"]  # Keep true class labels on CPU.

            outputs = model(images)                  # [B, 6, 256, 144]
            predictions = outputs.argmax(dim=1).cpu()  # [B, 256, 144]

            for i in range(len(predictions)):
                # Remove artificial side padding added by dataset.py.
                width = int(batch["original_width"][i])
                left = (predictions.shape[-1] - width) // 2
                right = left + width

                truth = masks[i, :, left:right]
                predicted = predictions[i, :, left:right]

                # Count intersections and regions for Dice across all test images.
                for c in range(NUM_CLASSES):
                    predicted_c = predicted == c
                    truth_c = truth == c
                    intersections[c] += (predicted_c & truth_c).sum()
                    totals[c] += predicted_c.sum() + truth_c.sum()

                # Save two initial images, then favour images containing prostate.
                if saved_images < MAX_VISUALISATIONS and (
                    saved_images < 2 or (truth == PROSTATE_CLASS).any()
                ):
                    image = batch["image"][i, 0, :, left:right].numpy()
                    filename = batch["filename"][i].replace(".nii.gz", ".png")
                    save_comparison(image, truth.numpy(), predicted.numpy(),
                                    save_dir / filename)
                    saved_images += 1

    print(f"Saved {saved_images} comparison images to {save_dir}")
    return compute_dice_score(intersections, totals)


def main():
    """Evaluate both trained U-Nets on the same HipMRI test dataset."""

    print("Prediction device:", DEVICE)

    # Load the test dataset once for both models.
    test_dataset = HipMRIDataset(DATA_ROOT, split="test", augment=False)

    if len(test_dataset) == 0:
        raise FileNotFoundError(f"No MRI test images found in {test_dataset.image_dir}")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Test slices:", len(test_dataset))

    # Evaluate each architecture independently.
    for model_name, model_class in MODELS.items():

        model_path = Path("runs") / model_name / "best_model.pt"
        save_dir = Path("runs") / model_name / "predictions"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nEvaluating {model_name}")

        # Load the trained weights for this architecture.
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved model not found: {model_path}")

        model = model_class(
            in_channels=1,
            num_classes=NUM_CLASSES,
            base_channels=BASE_CHANNELS
        ).to(DEVICE)

        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True)
        )
        model.eval()

        # Evaluate all test images and save visualisations.
        scores = evaluate_model(model, test_loader, save_dir)

        # Create a results report.
        lines = [f"Model: {model_name}", f"Test slices: {len(test_dataset)}"]

        for c, score in enumerate(scores):
            label = "N/A (absent)" if math.isnan(score) else f"{score:.4f}"
            lines.append(f"Class {c} Dice: {label}")

        foreground = [score for score in scores[1:] if not math.isnan(score)]

        if foreground:
            lines.append(
                f"Mean foreground Dice: {sum(foreground) / len(foreground):.4f}"
            )

        lines.append(f"Prostate Dice: {scores[PROSTATE_CLASS]:.4f}")

        report = "\n".join(lines)
        print("\n" + report)

        (save_dir / "dice_results.txt").write_text(report, encoding="utf-8")

        # Release the current model before loading the next one.
        del model

    print("\nFinished evaluating both models.")

if __name__ == "__main__":
    main()
