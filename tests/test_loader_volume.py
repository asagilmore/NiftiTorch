import os

import pytest
import torch
import torchvision.transforms.v2 as v2

from niftitorch import NiftiDataset3d  # noqa: E402


current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def test_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])


@pytest.fixture(scope='module')
def test_dataset(request, test_transform):
    return NiftiDataset3d(os.path.join(current_dir, 'dataset', 'image'),
                          os.path.join(current_dir, 'dataset', 'mask'),
                          test_transform)


@pytest.fixture(scope='module')
def test_dataset_shape(request, test_transform):
    return NiftiDataset3d(os.path.join(current_dir, 'dataset', 'image'),
                          os.path.join(current_dir, 'dataset', 'mask'),
                          test_transform, volume_shape=(64, 64, 64))


@pytest.fixture(scope='module')
def matched_test_dataset(request, test_transform):
    return NiftiDataset3d(os.path.join(current_dir, 'dataset', 'image'),
                          os.path.join(current_dir, 'dataset', 'image'),
                          test_transform)


@pytest.fixture(scope='module')
def matched_test_dataset_shape(test_transform):
    return NiftiDataset3d(os.path.join(current_dir, 'dataset', 'image'),
                          os.path.join(current_dir, 'dataset', 'image'),
                          test_transform, volume_shape=(64, 64, 64))


def test_shape(test_dataset_shape):
    for img, mask in test_dataset_shape:
        assert img.shape == mask.shape
        assert img.shape == (64, 64, 64)


def test_dataset_len(test_dataset, test_dataset_shape):
    assert len(test_dataset_shape) == 4
    assert len(test_dataset) == 4


def test_first_last(test_dataset, test_dataset_shape):
    datasets = [test_dataset, test_dataset_shape]
    for dataset in datasets:
        assert dataset[0]
        assert dataset[len(dataset) - 1]

        with pytest.raises(IndexError):
            assert not dataset[len(dataset)]


def test_matched(matched_test_dataset, matched_test_dataset_shape):
    for i in range(len(matched_test_dataset)):
        img, mask = matched_test_dataset[i]
        assert torch.allclose(img, mask, atol=1e-6)
