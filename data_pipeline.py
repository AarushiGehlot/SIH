import os
from glob import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class SatelliteSRDataset(Dataset):
    """
    Dataset for satellite image super-resolution.

    Takes high-resolution satellite images and automatically
    creates low-resolution inputs for super-resolution training.
    """

    def __init__(
        self,
        folder_path,
        hr_size=256,
        scale_factor=4
    ):
        self.folder_path = folder_path
        self.hr_size = hr_size
        self.scale_factor = scale_factor
        self.lr_size = hr_size // scale_factor

        # Find satellite images
        extensions = ["*.jpg", "*.jpeg","*.jfif", "*.png", "*.tif", "*.tiff"]

        self.image_paths = []

        for extension in extensions:
            self.image_paths.extend(
                glob(
                    os.path.join(
                        folder_path,
                        "**",
                        extension
                    ),
                    recursive=True
                )
            )

        self.image_paths.sort()

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No satellite images found in {folder_path}"
            )

        print(f"Found {len(self.image_paths)} satellite images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # -----------------------------
        # 1. Load high-resolution image
        # -----------------------------
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        # -----------------------------
        # 2. Create HR image
        # -----------------------------
        hr_image = image.resize(
            (self.hr_size, self.hr_size),
            Image.Resampling.BICUBIC
        )

        # -----------------------------
        # 3. Create LR image
        # -----------------------------
        lr_image = hr_image.resize(
            (self.lr_size, self.lr_size),
            Image.Resampling.BICUBIC
        )

        # -----------------------------
        # 4. Convert to NumPy
        # -----------------------------
        hr_array = np.array(hr_image, dtype=np.float32)
        lr_array = np.array(lr_image, dtype=np.float32)

        # -----------------------------
        # 5. Normalize 0-255 → 0-1
        # -----------------------------
        hr_array = hr_array / 255.0
        lr_array = lr_array / 255.0

        # -----------------------------
        # 6. Convert HWC → CHW
        # PyTorch expects:
        # Channels × Height × Width
        # -----------------------------
        hr_array = np.transpose(
            hr_array,
            (2, 0, 1)
        )

        lr_array = np.transpose(
            lr_array,
            (2, 0, 1)
        )

        # -----------------------------
        # 7. Convert NumPy → Tensor
        # -----------------------------
        hr_tensor = torch.tensor(
            hr_array,
            dtype=torch.float32
        )

        lr_tensor = torch.tensor(
            lr_array,
            dtype=torch.float32
        )

        return lr_tensor, hr_tensor


def create_dataloader(
    dataset_path,
    batch_size=16,
    hr_size=256,
    scale_factor=4,
    shuffle=True
):
    """
    Creates the satellite super-resolution DataLoader.
    """

    dataset = SatelliteSRDataset(
        folder_path=dataset_path,
        hr_size=hr_size,
        scale_factor=scale_factor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    return dataloader


# ---------------------------------------
# Test the pipeline
# ---------------------------------------

if __name__ == "__main__":

    DATASET_PATH = "dataset"

    train_loader = create_dataloader(
        dataset_path=DATASET_PATH,
        batch_size=16,
        hr_size=256,
        scale_factor=4
    )

    print("\nData pipeline ready!")

    # Test one batch
    lr_batch, hr_batch = next(iter(train_loader))

    print("LR batch shape :", lr_batch.shape)
    print("HR batch shape :", hr_batch.shape)

    print("LR value range:", lr_batch.min().item(),
          "to", lr_batch.max().item())

    print("HR value range:", hr_batch.min().item(),
          "to", hr_batch.max().item())