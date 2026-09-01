import os
import h5py


MAT_PATH = r"D:\research\student48\data\nyuv2\nyu_depth_v2_labeled.mat"


print("=" * 70)
print("NYU Depth V2 .MAT inspection")
print("=" * 70)

print("File:")
print(MAT_PATH)

if not os.path.isfile(MAT_PATH):
    raise FileNotFoundError(
        f"MAT file not found:\n{MAT_PATH}"
    )


print("\nOpening MATLAB v7.3 file...")

with h5py.File(MAT_PATH, "r") as f:

    print("\nTop-level variables/datasets:")

    def show_item(name, obj):

        if isinstance(obj, h5py.Dataset):

            print(
                f"{name:<40} "
                f"shape={obj.shape} "
                f"dtype={obj.dtype}"
            )

        elif isinstance(obj, h5py.Group):

            print(
                f"{name:<40} "
                f"[GROUP]"
            )

    f.visititems(show_item)


print("\nInspection complete.")
