from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms


def compare_transforms(
    image_path,
    transform1=None,
    transform2=None,
    num_samples=5,
    img_size=224,
    output_file_path="transforms.png"):
    """
    Visualize the effect of two different transforms applied to the same image.

    Args:
        image_path (str): Path to the input image.
        transform1 (callable): First transform pipeline.
        transform2 (callable): Second transform pipeline.
        num_samples (int): Number of samples to visualize for each transform.
    """
    image = Image.open(image_path) #.convert("RGB")

    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
    fig.suptitle("Transform Comparison", fontsize=16)

    if transform1 is None:
        transform1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.ToTensor()
        ])
    if transform2 is None:
        transform2 = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.ToTensor()
        ])

    for i in range(num_samples):
        img1 = transform1(image)
        img2 = transform2(image)

        # Convert tensor to numpy for display
        img1_np = img1.permute(1, 2, 0).numpy()
        img2_np = img2.permute(1, 2, 0).numpy()

        axes[0, i].imshow(img1_np)
        axes[0, i].set_title(f"Transform 1 - #{i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(img2_np)
        axes[1, i].set_title(f"Transform 2 - #{i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_file_path)






def visualize_masked_image(image_tensor, patch_mask, patch_size=16, title=None):
    """
    Args:
        image_tensor: Tensor (3, H, W), assumed to be in [0, 1] or [0, 255]
        patch_mask: Bool tensor (N_patches,) — True = masked
        patch_size: int, e.g., 16
        title: Optional string for title
    """
    image = image_tensor.clone()
    if image.max() > 1:
        image = image / 255.0  # normalize if needed
    image = to_pil_image(image)

    H, W = image_tensor.shape[1:]
    H_p, W_p = H // patch_size, W // patch_size
    patch_mask = patch_mask.view(H_p, W_p)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw rectangles over masked patches
    for i in range(H_p):
        for j in range(W_p):
            if patch_mask[i, j]:
                rect = patches.Rectangle(
                    (j * patch_size, i * patch_size),
                    patch_size, patch_size,
                    linewidth=1,
                    edgecolor='red',
                    facecolor='red',
                    alpha=0.5
                )
                ax.add_patch(rect)

    if title:
        plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig("test.png")
