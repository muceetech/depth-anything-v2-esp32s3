import sys
import torch

from dataset_96 import NYU96Dataset


BASE_DIR = r"D:\research\student48\data\processed\nyuv2"


dataset = NYU96Dataset(
    base_dir=BASE_DIR,
    split="train",
    use_da_teacher=True,
    use_litemono_teacher=False
)


print("\nDataset size:", len(dataset))


sample = dataset[0]


print("\nSample:")
print("RGB        :", sample["rgb"].shape)
print("Depth      :", sample["depth"].shape)
print("Valid mask :", sample["valid_mask"].shape)
print("DA teacher :", sample["da_teacher"].shape)


print("\nRGB range:")
print(
    sample["rgb"].min().item(),
    sample["rgb"].max().item()
)


print("\nDepth range:")
print(
    sample["depth"].min().item(),
    sample["depth"].max().item()
)


print("\nDA teacher range:")
print(
    sample["da_teacher"].min().item(),
    sample["da_teacher"].max().item()
)
