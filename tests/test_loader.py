import os

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


@pytest.fixture(scope='module',
                params=[False, True],
                ids=["normal", "mmap"])
def test_dataset(request, test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'mask'),
                        test_transform, mmap=request.param)


@pytest.fixture(scope='module',
                params=[False, True],
                ids=["normal", "mmap"])
def test_dataset_3d(request, test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'mask'),
                        test_transform, slice_width=5,
                        mmap=request.param)


@pytest.fixture(scope='module',
                params=[False, True],
                ids=["normal", "mmap"])
def test_dataset_3d_width(request, test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'mask'),
                        test_transform, slice_width=5, width_labels=True,
                        mmap=request.param)


@pytest.fixture(scope='module',
                params=[False, True],
                ids=["normal", "mmap"])
def matched_test_dataset(request, test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'image'),
                        test_transform, mmap=request.param)


@pytest.fixture(scope='module',
                params=[False, True],
                ids=["normal", "mmap"])
def matched_test_dataset_3d(request, test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'image'),
                        test_transform, slice_width=5,
                        mmap=request.param)


@pytest.fixture(scope='module',
                params=[False, True],
                ids=["normal", "mmap"])
def matched_test_dataset_3d_width(request, test_transform):
    return NiftiDataset(os.path.join(current_dir, 'dataset', 'image'),
                        os.path.join(current_dir, 'dataset', 'image'),
                        test_transform, slice_width=5, width_labels=True,
                        mmap=request.param)


def test_dataset_len(test_dataset):
    assert len(test_dataset) == 400


def test_dataset_3d_len(test_dataset_3d, test_dataset_3d_width):
    assert len(test_dataset_3d) == 384
    assert len(test_dataset_3d_width) == 384


def test_first_last(test_dataset, test_dataset_3d, test_dataset_3d_width):
    datasets = [test_dataset, test_dataset_3d, test_dataset_3d_width]
    for dataset in datasets:
        assert dataset[0]
        assert dataset[len(dataset) - 1]

        with pytest.raises(IndexError):
            assert not dataset[len(dataset)]


def test_matched(matched_test_dataset):
    for i in range(len(matched_test_dataset)):
        img, mask = matched_test_dataset[i]
        assert torch.allclose(img, mask, atol=1e-6)


def test_matched_3d(matched_test_dataset_3d):
    for i in range(len(matched_test_dataset_3d)):
        img, mask = matched_test_dataset_3d[i]
        middle_index = img.shape[0] // 2
        img_middle = img[middle_index:middle_index + 1, :, :]
        assert torch.allclose(img_middle, mask, atol=1e-6)


def test_matched_3d_width(matched_test_dataset_3d_width):
    for i in range(len(matched_test_dataset_3d_width)):
        img, mask = matched_test_dataset_3d_width[i]
        assert torch.allclose(img, mask, atol=1e-6)
