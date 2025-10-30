import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as Functional
import random


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert mask to one-hot encoded channels
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res


def load_nifti_slice(filepath: str, normalize: bool = True, dtype=np.float32) -> np.ndarray:
    """
    Load a single NIfTI slice

    Args:
        filepath: Path to .nii.gz file
        normalize: Whether to apply normalisation
        dtype: Output datatype
    """
    nifti_image = nib.load(filepath)
    image = nifti_image.get_fdata(caching='unchanged')  # Read from disk only

    # Handle extra dimensions (sometimes HipMRI data has extra dims)
    if len(image.shape) == 3:
        image = image[:, :, 0]

    image = image.astype(dtype)

    # Z-score normalization
    if normalize:
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        else:
            image = image - mean

    return image


def load_data_batch(file_paths: list, normalize: bool = True, dtype=np.float32) -> np.ndarray:
    """
    Load multiple NIfTI slices into a batch array.

    Args:
        file_paths: List of paths to NIfTI files
        normalize: Whether to normalize images
        dtype: Output data type

    Returns:
        Array of shape (N, H, W) where N is number of files
    """
    # Get dimensions from first file
    first_image = load_nifti_slice(file_paths[0], normalize=False, dtype=dtype)
    rows, cols = first_image.shape
    num_images = len(file_paths)

    # Pre-allocate array
    images = np.zeros((num_images, rows, cols), dtype=dtype)

    # Load all images
    for i, filepath in enumerate(tqdm(file_paths, desc="Loading images")):
        images[i] = load_nifti_slice(filepath, normalize=normalize, dtype=dtype)

    return images


class HipMRIDataset(Dataset):
    """
    Loads data for Hip MRI Prostate Dataset

    Args:
        image_paths: List of paths to NIfTI images
        mask_paths: List of paths to NIfTI masks
        transform: Transform node
        normalize: Whether to apply normalisation
        target_size: Expected size of image
    """
    def __init__(self,
                 image_paths: list,
                 mask_paths: list,
                 transform='val',
                 normalize: bool = True,
                 target_size: tuple = (256, 256)):
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"

        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform
        self.normalize = normalize
        self.target_size = target_size

        print(f"Dataset created with {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def apply_transforms(self, image: np.ndarray, mask: np.ndarray) -> tuple:
        # Convert to torch tensors first
        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)

        # Resize images and masks to meet expected size
        if self.target_size is not None:
            image = Functional.resize(image, list(self.target_size), interpolation=Functional.InterpolationMode.BILINEAR)
            mask = Functional.resize(mask.unsqueeze(0), list(self.target_size), interpolation=Functional.InterpolationMode.NEAREST).squeeze(0)

        if self.transform == 'train':
            # Random horizontal flip
            if random.random() > 0.5:
                image = Functional.hflip(image)
                mask = Functional.hflip(mask.unsqueeze(0)).squeeze(0)

            # Random vertical flip
            if random.random() > 0.5:
                image = Functional.vflip(image)
                mask = Functional.vflip(mask.unsqueeze(0)).squeeze(0)

            # Random rotation (-15 to +15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = Functional.rotate(image, angle, fill=0)
                mask = Functional.rotate(mask.unsqueeze(0), angle, fill=0, interpolation=Functional.InterpolationMode.NEAREST).squeeze(0)

            # Random brightness and contrast adjustment (image only)
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                image = Functional.adjust_brightness(image, brightness_factor)

            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                image = Functional.adjust_contrast(image, contrast_factor)

            # Add Gaussian noise to image
            if random.random() > 0.3:
                noise = torch.randn_like(image) * 0.1
                image = image + noise

        return image, mask

    def __getitem__(self, idx):
        """
        Returns a single 2D slice with mask.

        Returns:
            image: Tensor of shape (1, H, W) - single channel image
            mask: Tensor of shape (H, W) - segmentation mask with class indices
        """
        # Load image and mask
        image = load_nifti_slice(self.image_paths[idx], normalize=self.normalize)
        mask = load_nifti_slice(self.mask_paths[idx], normalize=False)

        # Ensure mask is integer type
        mask = mask.astype(np.int64)

        # Apply transforms
        image, mask = self.apply_transforms(image, mask)

        return image, mask


def get_file_paths_from_directory(data_dir: str, image_folder: str, mask_folder: str) -> tuple:
    """
    Loads all file paths from the data directory

    Args:
        data_dir: Base directory for data
        image_folder: Folder where NIfTI images are stored
        mask_folder: Folder where NIfTI masks are stored
    """
    image_dir = os.path.join(data_dir, image_folder)
    mask_dir = os.path.join(data_dir, mask_folder)

    # Get all .nii and .nii.gz files
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii*")))

    print(f"Found {len(image_paths)} images in {image_folder}")
    print(f"Found {len(mask_paths)} masks in {mask_folder}")

    # Verify counts match
    if len(image_paths) != len(mask_paths):
        print(f"WARNING: Number of images ({len(image_paths)}) != masks ({len(mask_paths)})")

    return image_paths, mask_paths


def get_dataloaders(data_dir: str,
                    batch_size: int = 16,
                    num_workers: int = 4,
                    normalize: bool = True) -> tuple:
    """
    Create dataloaders for training, validation, and testing

    Args:
        data_dir: Root folder for data
        batch_size: Batch size for training
        num_workers: Number of parallel workers
        normalize: Whether to apply normalisation
    """

    # Get file paths for each split
    train_images, train_masks = get_file_paths_from_directory(
        data_dir, "keras_slices_train", "keras_slices_seg_train"
    )
    val_images, val_masks = get_file_paths_from_directory(
        data_dir, "keras_slices_validate", "keras_slices_seg_validate"
    )
    test_images, test_masks = get_file_paths_from_directory(
        data_dir, "keras_slices_test", "keras_slices_seg_test"
    )

    # Create datasets with appropriate transforms
    train_dataset = HipMRIDataset(
        train_images, train_masks,
        transform='train',  # Enable augmentation
        normalize=normalize
    )
    val_dataset = HipMRIDataset(
        val_images, val_masks,
        transform='val',  # No augmentation
        normalize=normalize
    )
    test_dataset = HipMRIDataset(
        test_images, test_masks,
        transform='val',  # No augmentation
        normalize=normalize
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for stability
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_directory = "./data"

    print("Verifying dataset structure...")

    expected_dirs = [
        "keras_slices_train",
        "keras_slices_seg_train",
        "keras_slices_validate",
        "keras_slices_seg_validate",
        "keras_slices_test",
        "keras_slices_seg_test"
    ]

    for dir_name in expected_dirs:
        dir_path = os.path.join(data_directory, dir_name)
        exists = os.path.exists(dir_path)
        status = "OK" if exists else "ERR"
        print(f"{dir_path}: {status}")

    try:
        train_images, train_masks = get_file_paths_from_directory(
            data_directory, "keras_slices_train", "keras_slices_seg_train"
        )

        if len(train_images) == 0:
            print("ERROR: No training images found!")

        print(f"\nLoading 3 sample images...")
        for i in range(3):
            img = load_nifti_slice(train_images[i], normalize=True)
            mask = load_nifti_slice(train_masks[i], normalize=False)

            unique_labels = np.unique(mask)
            print(f"\nSample {i + 1}:")
            print(f"Image shape: {img.shape}, dtype: {img.dtype}")
            print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
            print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"Unique labels: {unique_labels}")
            print(f"Prostate pixels: {np.sum(mask == 1)}")

        print("Dataset verification completed.")
    except Exception as e:
        print(f"\nDataset verification ERROR: {e}")

    print("\nCreating DataLoaders...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_directory,
            batch_size=8,
            num_workers=2,
            normalize=True
        )

        print("\nTesting batch loading...")

        images, masks = next(iter(train_loader))
        print(f"\nBatch loaded")
        print(f"Images shape: {images.shape} (batch, channels, height, width)")
        print(f"Masks shape: {masks.shape} (batch, height, width)")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Unique mask values: {torch.unique(masks).tolist()}")
        print(f"Image dtype: {images.dtype}")
        print(f"Mask dtype: {masks.dtype}")

        for label in torch.unique(masks):
            count = (masks == label).sum().item()
            percentage = 100.0 * count / masks.numel()
            print(f"Class {label}: {count} pixels ({percentage:.2f}%)")

        print("Dataset module tests complete.")

        print("\nSUMMARY")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Total train samples: {len(train_loader.dataset)}")
        print(f"Total val samples: {len(val_loader.dataset)}")
        print(f"Total test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"\nDataLoader ERROR: {e}")
