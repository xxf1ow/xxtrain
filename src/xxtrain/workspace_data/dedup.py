from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

SIMILARITY_DISTANCE = 2


def image_sha256(path: Path) -> str:
    """Return the SHA-256 digest of an image file's bytes."""
    return sha256(path.read_bytes()).hexdigest()


def perceptual_hash(path: Path) -> int:
    """Return the fixed-size grayscale DCT perceptual hash for a decodable image."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Unsupported image file: {path.name}')
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    values = cv2.dct(resized)[:8, :8].reshape(-1)
    median = np.median(values[1:])
    return sum(int(value > median) << index for index, value in enumerate(values))


def hamming_distance(left: int, right: int) -> int:
    """Return the number of differing bits between two perceptual hashes."""
    return (left ^ right).bit_count()
