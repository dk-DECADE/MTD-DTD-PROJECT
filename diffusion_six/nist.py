"""Read the digit classes from the NIST SD19 by_class PNG archive."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from diffusion_six.core import SIZE


# NIST reserves hsf_4 for testing. Keep hsf_7 separate for model selection.
# train_XX is an aggregate of several of these groups, so never read it here.
PARTITIONS = {
    "train": ("hsf_0", "hsf_1", "hsf_2", "hsf_3", "hsf_6"),
    "validation": ("hsf_7",),
    "test": ("hsf_4",),
}


def read_digit(path: Path) -> np.ndarray | None:
    with Image.open(path) as source:
        if source.size != (128, 128):
            raise ValueError(f"Expected a 128 x 128 NIST image: {path}")
        digit = ImageOps.invert(source.convert("L"))
    bounds = digit.getbbox()
    if bounds is None:
        return None
    digit = ImageOps.contain(digit.crop(bounds), (SIZE - 8, SIZE - 8), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (SIZE, SIZE))
    canvas.paste(digit, ((SIZE - digit.width) // 2, (SIZE - digit.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255


def nist_splits(root: Path, seed: int, train_per_digit: int, eval_per_digit: int) -> dict:
    rng = np.random.default_rng(seed)
    splits = {}
    for name, partitions in PARTITIONS.items():
        images, ids, labels = [], [], []
        limit = train_per_digit if name == "train" else eval_per_digit
        for label in range(10):
            class_dir = root / str(30 + label)
            paths = []
            for partition in partitions:
                folder = class_dir / partition
                members = sorted(folder.glob("*.png"))
                if not members:
                    raise ValueError(f"No PNGs in {folder}. Pass the by_class directory and keep the hsf folders.")
                paths.extend(members)
            count = 0
            for index in rng.permutation(len(paths)):
                path = paths[index]
                image = read_digit(path)
                if image is None:
                    continue
                images.append(image)
                ids.append(path.relative_to(root).as_posix())
                labels.append(label)
                count += 1
                if count == limit:
                    break
            if count < limit:
                raise ValueError(f"{name}, digit {label}: only {count} usable images; requested {limit}.")
        splits[name] = {"images": np.stack(images), "ids": ids, "labels": np.array(labels)}
    return splits
