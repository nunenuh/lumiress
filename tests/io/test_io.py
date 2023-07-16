import unittest
from pathlib import Path

import cv2
import numpy as np

from lumiress import io


class TestLoadImage(unittest.TestCase):
    def test_load_image_valid_file(self):
        # Create a temporary image file for testing
        test_image_path = "test_image.jpg"
        cv2.imwrite(test_image_path, np.zeros((10, 10, 3), dtype=np.uint8))

        # Test loading the image
        loaded_image = io.load_image(test_image_path)

        # Assert that the loaded image is not None
        self.assertIsNotNone(loaded_image)

        # Assert that the loaded image has the correct shape
        self.assertEqual(loaded_image.shape, (10, 10, 3))

        # Delete the temporary image file
        Path(test_image_path).unlink()

    def test_load_image_invalid_file(self):
        # Test loading an invalid image file
        with self.assertRaises(ValueError):
            io.load_image("invalid_image.txt")

    def test_load_image_invalid_extension(self):
        # Test loading an image with an invalid extension
        with self.assertRaises(ValueError):
            io.load_image("image.pdf")
