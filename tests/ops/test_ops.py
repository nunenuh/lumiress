import numpy as np
import torch

from lumiress import ops
import torch.nn.functional as F

def test_to_torch_cpu():
    # Test case 1: image with shape (3, 256, 256) on CPU
    device = "cpu"
    image = np.random.randint(0, 255, size=(3, 256, 256))
    expected_output = torch.from_numpy(image).float() / 255.
    expected_output = expected_output.permute(2, 0, 1).unsqueeze(0).to("cpu")
    assert torch.allclose(ops.to_torch(image), expected_output)
    
    # Test case 3: image with shape (3, 512, 512) on CPU
    image = np.random.randint(0, 255, size=(3, 512, 512))
    expected_output = torch.from_numpy(image).float() / 255.
    expected_output = expected_output.permute(2, 0, 1).unsqueeze(0).to("cpu")
    assert torch.allclose(ops.to_torch(image), expected_output)
    

def test_to_torch_cuda():
    if torch.cuda.is_available():
        # Test case 2: image with shape (3, 128, 128) on GPU
        image = np.random.randint(0, 255, size=(3, 128, 128))
        expected_output = torch.from_numpy(image).float() / 255.
        expected_output = expected_output.permute(2, 0, 1).unsqueeze(0).to("gpu")
        assert torch.allclose(ops.to_torch(image, device="gpu"), expected_output)



        # Test case 4: image with shape (1, 64, 64) on GPU
        image = np.random.randint(0, 255, size=(1, 64, 64))
        expected_output = torch.from_numpy(image).float() / 255.
        expected_output = expected_output.permute(2, 0, 1).unsqueeze(0).to("gpu")
        assert torch.allclose(ops.to_torch(image, device="gpu"), expected_output)
        
        
def test_calc_dim():
    # Testing with value = 10 and multi = 4
    assert ops.calc_dim(10, 4) == 12

    # Testing with value = 15 and multi = 4
    assert ops.calc_dim(15, 4) == 16

    # Testing with value = 20 and multi = 4
    assert ops.calc_dim(20, 4) == 24

    # Testing with value = 25 and multi = 4
    assert ops.calc_dim(25, 4) == 28
    
    
def test_calc_new_size():
    # Test case 1: Testing with height = 10, width = 20, and default multiplier = 4
    assert ops.calc_new_size(10, 20) == (12, 24)

    # Test case 2: Testing with height = 15, width = 30, and multiplier = 2
    assert ops.calc_new_size(15, 30, 2) == (16, 32)

    # Test case 3: Testing with height = 5, width = 25, and multiplier = 3
    assert ops.calc_new_size(5, 25, 3) == (6, 27)
    
    
def test_calc_pad_val_not_divisible():
    assert ops.calc_pad_val(10, 7, 4) == 3
    
def test_calc_pad_val_divisible():
    assert ops.calc_pad_val(10, 8, 4) == 0
    
def test_calc_pad_val_equal_height():
    assert ops.calc_pad_val(10, 10, 4) == 0
    
def test_calc_pad_val_less_height():
    assert ops.calc_pad_val(10, 12, 4) == 0


def test_calc_pad_val_no_multi():
    assert ops.calc_pad_val(10, 8) == 0
    
    
    
def test_calc_pad_size():
    # Testing when nsize is smaller than size
    assert ops.calc_pad_size((2, 2), (4, 4)) == (0, 0)

    # Testing when nsize is larger than size
    assert ops.calc_pad_size((6, 6), (4, 4)) == (0, 0)

    # Testing when nsize is equal to size
    assert ops.calc_pad_size((4, 4), (4, 4)) == (0, 0)

    # Testing with custom multi value
    assert ops.calc_pad_size((6, 6), (4, 4), 3) == (2, 2)
    
    assert ops.calc_pad_size((26, 26), (14, 14), 4) == (12, 12)
    


# def test_pad():
#     # Test case 1: Image size is already a multiple of img_multiple_of
#     image = torch.randn(1, 3, 8, 8)  # 8x8 image
#     img_multiple_of = 4
#     expected_output = image
#     assert torch.allclose(ops.pad(image, img_multiple_of), expected_output)

#     # Test case 2: Image size is not a multiple of img_multiple_of
#     image = torch.randn(1, 3, 7, 9)  # 7x9 image
#     img_multiple_of = 4
#     expected_output = F.pad(image, (0, 3, 0, 3), 'reflect')  # Pad 3 pixels on both sides
#     assert torch.allclose(ops.pad(image, img_multiple_of), expected_output)

#     # Test case 3: Image size is smaller than img_multiple_of
#     image = torch.randn(1, 3, 3, 3)  # 3x3 image
#     img_multiple_of = 4
#     expected_output = F.pad(image, (0, 1, 0, 1), 'reflect')  # Pad 1 pixel on both sides
#     assert torch.allclose(ops.pad(image, img_multiple_of), expected_output)

#     # Test case 4: Image size is larger than img_multiple_of
#     image = torch.randn(1, 3, 10, 12)  # 10x12 image
#     img_multiple_of = 4
#     expected_output = F.pad(image, (0, 0, 0, 0), 'reflect')  # No padding needed
#     assert torch.allclose(ops.pad(image, img_multiple_of), expected_output)

def test_unpad():
    # Test case 1
    images = torch.randn(10, 3, 32, 32)
    height = 28
    width = 28
    unpadded_images = ops.unpad(images, height, width)
    assert unpadded_images.shape == (10, 3, 28, 28)

    # Test case 2
    images = torch.randn(10, 3, 32, 32)
    height = 32
    width = 32
    unpadded_images = ops.unpad(images, height, width)
    assert unpadded_images.shape == (10, 3, 32, 32)

    # Test case 3
    images = torch.randn(10, 3, 32, 32)
    height = 36
    width = 36
    unpadded_images = ops.unpad(images, height, width)
    assert unpadded_images.shape == (10, 3, 32, 32)

    # Test case 4
    images = torch.empty(0, 3, 32, 32)
    height = 28
    width = 28
    unpadded_images = ops.unpad(images, height, width)
    assert unpadded_images.shape == (0, 3, 28, 28)

    # Test case 5
    images = torch.randn(10, 1, 32, 32)
    height = 28
    width = 28
    unpadded_images = ops.unpad(images, height, width)
    assert unpadded_images.shape == (10, 1, 28, 28)

    # Test case 6
    images = torch.randn(10, 1, 32, 32)
    height = 30
    width = 30
    unpadded_images = ops.unpad(images, height, width)
    assert unpadded_images.shape == (10, 1, 30, 30)

def test_normalize_image_0_to_255():
    image = np.array([[0, 50, 100], [150, 200, 255]])
    expected_output = np.array([[0, 50, 100], [150, 200, 255]])
    normalized_image = ops.normalize(image)
    assert np.array_equal(normalized_image, expected_output)

# def test_normalize_image_minus_100_to_100():
#     image = np.array([[-100, -50, 0], [50, 100, 100]])
#     expected_output = np.array([[0, 127, 191], [223, 255, 255]])
#     normalized_image = ops.normalize(image)
#     assert np.array_equal(normalized_image, expected_output)

def test_normalize_image_all_zeros():
    image = np.zeros((2, 3))
    expected_output = np.zeros((2, 3))
    normalized_image = ops.normalize(image)
    assert np.array_equal(normalized_image, expected_output)

# def test_normalize_image_all_255():
#     image = np.full((2, 3), 255)
#     expected_output = np.full((2, 3), 255)
#     normalized_image = ops.normalize(image)
#     assert np.array_equal(normalized_image, expected_output)


