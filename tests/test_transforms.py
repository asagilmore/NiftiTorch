import torch
from niftitorch.transforms import RandomRotation90


def test_shape():
    img = torch.rand(1, 224, 224)
    mask = torch.rand(1, 224, 224)
    transform = RandomRotation90()
    img_rot, mask_rot = transform(img, mask)
    assert img_rot.shape == img.shape
    assert mask_rot.shape == mask.shape
    assert img_rot.shape == mask_rot.shape


def test_equality():
    for _ in range(10):
        img = torch.rand(1, 224, 224)
        mask = img.clone()
        transform = RandomRotation90()
        img_rot, mask_rot = transform(img, mask)
        assert torch.allclose(img_rot, mask_rot, atol=1e-6)
