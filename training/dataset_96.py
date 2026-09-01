import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class NYU96Dataset(Dataset):
    """
    NYU Depth V2 dataset loader for 96x96 Depth Anything V2
    student training.

    Dataset structure:

    nyuv2/
    ├── rgb/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    │
    ├── depth/
    │   ├── 0000.npy
    │   ├── 0001.npy
    │   └── ...
    │
    ├── teacher_depth/
    │   ├── 0000.npy
    │   ├── 0001.npy
    │   └── ...
    │
    ├── litemono8m_teacher/
    │   └── ...
    │
    └── splits/
        ├── train_indices.npy
        ├── val_indices.npy
        └── test_indices.npy


    Original dataset resolution:
        256 x 192

    Student resolution:
        96 x 96

    Preprocessing:
        256 x 192
             ↓
        center crop
        192 x 192
             ↓
        resize
         96 x 96
    """

    def __init__(
        self,
        base_dir,
        split="train",
        use_da_teacher=True,
        use_litemono_teacher=False,
    ):

        self.base_dir = base_dir

        # ----------------------------------------------------
        # DIRECTORIES
        # ----------------------------------------------------

        self.rgb_dir = os.path.join(
            base_dir,
            "rgb"
        )

        self.depth_dir = os.path.join(
            base_dir,
            "depth"
        )

        self.da_teacher_dir = os.path.join(
            base_dir,
            "teacher_depth"
        )

        self.litemono_teacher_dir = os.path.join(
            base_dir,
            "litemono8m_teacher"
        )

        self.split_dir = os.path.join(
            base_dir,
            "splits"
        )

        # ----------------------------------------------------
        # SETTINGS
        # ----------------------------------------------------

        self.target_size = 96

        self.crop_size = 192

        self.use_da_teacher = use_da_teacher

        self.use_litemono_teacher = (
            use_litemono_teacher
        )

        # ----------------------------------------------------
        # CHECK DIRECTORIES
        # ----------------------------------------------------

        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(
                f"RGB directory not found:\n{self.rgb_dir}"
            )

        if not os.path.isdir(self.depth_dir):
            raise FileNotFoundError(
                f"Depth directory not found:\n{self.depth_dir}"
            )

        # ----------------------------------------------------
        # SPLIT FILE
        # ----------------------------------------------------

        split_file = os.path.join(
            self.split_dir,
            f"{split}_indices.npy"
        )

        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"Split file not found:\n{split_file}"
            )

        self.indices = np.load(
            split_file
        )

        print(
            f"{split}: {len(self.indices)} samples"
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.indices)

    # ========================================================
    # CENTER CROP
    # ========================================================

    def center_crop_array(
        self,
        array,
        crop_height,
        crop_width
    ):
        """
        Center crop a NumPy array.

        Supports:
            H x W
        """

        h, w = array.shape[:2]

        if h < crop_height or w < crop_width:
            raise ValueError(
                f"Array too small for crop: "
                f"{array.shape}"
            )

        y1 = (h - crop_height) // 2
        x1 = (w - crop_width) // 2

        y2 = y1 + crop_height
        x2 = x1 + crop_width

        return array[
            y1:y2,
            x1:x2
        ]

    # ========================================================
    # RESIZE DEPTH
    # ========================================================

    def resize_depth(
        self,
        depth
    ):
        """
        Crop and resize float32 depth using OpenCV.

        NEAREST is deliberately used for ground-truth
        depth to avoid inventing depth values across
        discontinuities.
        """

        depth = self.center_crop_array(
            depth,
            self.crop_size,
            self.crop_size
        )

        depth = cv2.resize(
            depth,
            (
                self.target_size,
                self.target_size
            ),
            interpolation=cv2.INTER_NEAREST
        )

        return depth.astype(
            np.float32
        )

    # ========================================================
    # RESIZE TEACHER
    # ========================================================

    def resize_teacher(
        self,
        teacher
    ):
        """
        Crop and resize teacher depth.

        Teacher predictions are continuous relative-depth
        values, therefore bilinear interpolation is used.
        """

        teacher = self.center_crop_array(
            teacher,
            self.crop_size,
            self.crop_size
        )

        teacher = cv2.resize(
            teacher,
            (
                self.target_size,
                self.target_size
            ),
            interpolation=cv2.INTER_LINEAR
        )

        return teacher.astype(
            np.float32
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        position
    ):

        # ----------------------------------------------------
        # IMAGE INDEX
        # ----------------------------------------------------

        idx = int(
            self.indices[position]
        )

        name = f"{idx:04d}"

        # ----------------------------------------------------
        # FILE PATHS
        # ----------------------------------------------------

        rgb_path = os.path.join(
            self.rgb_dir,
            name + ".png"
        )

        depth_path = os.path.join(
            self.depth_dir,
            name + ".npy"
        )

        # ----------------------------------------------------
        # CHECK FILES
        # ----------------------------------------------------

        if not os.path.isfile(rgb_path):

            raise FileNotFoundError(
                f"RGB image not found:\n{rgb_path}"
            )

        if not os.path.isfile(depth_path):

            raise FileNotFoundError(
                f"Depth file not found:\n{depth_path}"
            )

        # ====================================================
        # RGB
        # ====================================================

        rgb = Image.open(
            rgb_path
        ).convert("RGB")

        rgb = np.array(
            rgb,
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # RGB CENTER CROP
        # ----------------------------------------------------

        rgb = self.center_crop_array(
            rgb,
            self.crop_size,
            self.crop_size
        )

        # ----------------------------------------------------
        # RGB RESIZE
        # ----------------------------------------------------

        rgb = cv2.resize(
            rgb,
            (
                self.target_size,
                self.target_size
            ),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # RGB → FLOAT32 TENSOR
        # ----------------------------------------------------

        rgb = rgb.astype(
            np.float32
        ) / 255.0

        rgb = torch.from_numpy(
            rgb
        )

        # HWC → CHW

        rgb = rgb.permute(
            2,
            0,
            1
        ).contiguous()

        # ====================================================
        # GROUND-TRUTH DEPTH
        # ====================================================

        depth = np.load(
            depth_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # VERIFY DEPTH
        # ----------------------------------------------------

        if depth.ndim != 2:

            raise ValueError(
                f"Expected 2D depth map, "
                f"got {depth.shape} "
                f"for {depth_path}"
            )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        depth = self.resize_depth(
            depth
        )

        # ----------------------------------------------------
        # VALID DEPTH MASK
        # ----------------------------------------------------

        valid_mask = (
            np.isfinite(depth)
            &
            (depth > 0)
        )

        # ----------------------------------------------------
        # REMOVE NaN / INF
        # ----------------------------------------------------

        depth = np.nan_to_num(
            depth,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # TENSOR
        # ----------------------------------------------------

        depth = torch.from_numpy(
            depth
        ).unsqueeze(0)

        valid_mask = torch.from_numpy(
            valid_mask.astype(np.float32)
        ).unsqueeze(0)

        # ====================================================
        # SAMPLE
        # ====================================================

        sample = {
            "rgb": rgb,
            "depth": depth,
            "valid_mask": valid_mask,
            "index": idx
        }

        # ====================================================
        # DEPTH ANYTHING V2 TEACHER
        # ====================================================

        if self.use_da_teacher:

            teacher_path = os.path.join(
                self.da_teacher_dir,
                name + ".npy"
            )

            if not os.path.isfile(
                teacher_path
            ):

                raise FileNotFoundError(
                    f"DA-V2 teacher not found:\n"
                    f"{teacher_path}"
                )

            teacher = np.load(
                teacher_path
            ).astype(
                np.float32
            )

            if teacher.ndim != 2:

                raise ValueError(
                    f"Expected 2D DA-V2 teacher, "
                    f"got {teacher.shape}"
                )

            # ------------------------------------------------
            # RESIZE WITH OPENCV
            # ------------------------------------------------

            teacher = self.resize_teacher(
                teacher
            )

            # ------------------------------------------------
            # CLEAN VALUES
            # ------------------------------------------------

            teacher = np.nan_to_num(
                teacher,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # ------------------------------------------------
            # TENSOR
            # ------------------------------------------------

            teacher = torch.from_numpy(
                teacher
            ).unsqueeze(0)

            sample[
                "da_teacher"
            ] = teacher

        # ====================================================
        # LITE-MONO-8M TEACHER
        # ====================================================

        if self.use_litemono_teacher:

            teacher_path = os.path.join(
                self.litemono_teacher_dir,
                name + ".npy"
            )

            if not os.path.isfile(
                teacher_path
            ):

                raise FileNotFoundError(
                    f"Lite-Mono teacher not found:\n"
                    f"{teacher_path}"
                )

            teacher = np.load(
                teacher_path
            ).astype(
                np.float32
            )

            if teacher.ndim != 2:

                raise ValueError(
                    f"Expected 2D Lite-Mono teacher, "
                    f"got {teacher.shape}"
                )

            # ------------------------------------------------
            # RESIZE
            # ------------------------------------------------

            teacher = self.resize_teacher(
                teacher
            )

            # ------------------------------------------------
            # CLEAN
            # ------------------------------------------------

            teacher = np.nan_to_num(
                teacher,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # ------------------------------------------------
            # TENSOR
            # ------------------------------------------------

            teacher = torch.from_numpy(
                teacher
            ).unsqueeze(0)

            sample[
                "litemono_teacher"
            ] = teacher

        # ====================================================
        # RETURN
        # ====================================================

        return sample
    