import os
import random

import pytest
import torch
import torchvision.transforms.v2 as v2

from niftitorch import NiftiDataset  # noqa: E402


current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def test_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])


@pytest.fixture(scope='module')
def test_dataset(test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'mask'),
                        test_transform)


@pytest.fixture(scope='module')
def test_dataset_3d(test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'mask'),
                        test_transform, slice_width=5)


@pytest.fixture(scope='module')
def test_dataset_3d_width(test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'mask'),
                        test_transform, slice_width=5, width_labels=True)


@pytest.fixture(scope='module')
def matched_test_dataset(test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'image'),
                        test_transform)


@pytest.fixture(scope='module')
def matched_test_dataset_3d(test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'image'),
                        test_transform, slice_width=5)


@pytest.fixture(scope='module')
def matched_test_dataset_3d_width(test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'image'),
                        test_transform, slice_width=5, width_labels=True)


@pytest.fixture(scope='module')
def test_dataset_largest(test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'mask'),
                        test_transform, scan_size='largest')


@pytest.fixture(scope='module')
def test_dataset_smallest(test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'mask'),
                        test_transform, scan_size='smallest')


def test_dataset_len(test_dataset):
    assert len(test_dataset) == (92 + 200)


def test_dataset_3d_len(test_dataset_3d, test_dataset_3d_width):
    assert len(test_dataset_3d) == (92 + 200 - 12)
    assert len(test_dataset_3d_width) == (92 + 200 - 12)


def test_first_last(test_dataset, test_dataset_3d, test_dataset_3d_width):
    datasets = [test_dataset, test_dataset_3d, test_dataset_3d_width]
    for dataset in datasets:
        assert dataset[0]
        assert dataset[len(dataset) - 1]

        with pytest.raises(IndexError):
            assert not dataset[len(dataset)]


def test_matched(matched_test_dataset):
    indexs = random.sample(range(len(matched_test_dataset)), 5)
    for i in indexs:
        img, mask = matched_test_dataset[i]
        assert torch.allclose(img, mask, atol=1e-6)


def test_matched_3d(matched_test_dataset_3d):
    indexs = random.sample(range(len(matched_test_dataset_3d)), 5)
    for i in indexs:
        img, mask = matched_test_dataset_3d[i]
        middle_index = img.shape[0] // 2
        img_middle = img[middle_index:middle_index + 1, :, :]
        assert torch.allclose(img_middle, mask, atol=1e-6)


def test_matched_3d_width(matched_test_dataset_3d_width):
    indexs = random.sample(range(len(matched_test_dataset_3d_width)), 5)
    for i in indexs:
        img, mask = matched_test_dataset_3d_width[i]
        assert torch.allclose(img, mask, atol=1e-6)


def test_shapes(test_dataset, test_dataset_largest, test_dataset_smallest):
    indexs = [0, 10, 20, 110, 142, 204, 290]

    for i in indexs:
        image, mask = test_dataset[i]
        assert image.shape == mask.shape
        assert image.shape == (1, 512, 512)

        image, mask = test_dataset_largest[i]
        assert image.shape == mask.shape
        assert image.shape == (1, 1024, 1024)

        image, mask = test_dataset_smallest[i]
        assert image.shape == mask.shape
        assert image.shape == (1, 512, 512)
