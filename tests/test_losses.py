import pytest
import torch
from niftitorch.losses import PerceptualLoss, CombinedLoss, HistogramLoss  # noqa: E402
import numpy as np


@pytest.fixture
def setup_perceptual_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual_loss = PerceptualLoss().to(device)
    return perceptual_loss, device


@pytest.fixture
def setup_combined_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual_loss = PerceptualLoss().to(device)
    MSE_loss = torch.nn.MSELoss()
    loss = CombinedLoss(criterion_1=MSE_loss, criterion_2=perceptual_loss)
    loss.to(device)
    return loss, device


@pytest.fixture
def setup_histogram_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = HistogramLoss()
    loss.to(device)
    return loss, device


def test_histogram_loss(setup_histogram_loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hist_loss = HistogramLoss(min_val=0, max_val=1).to(device)
    input_tensor = torch.rand(1, 1, 224, 224, device=device)
    target_tensor = input_tensor.clone()
    target_tensor = target_tensor.flatten()
    shuffled_indices = np.random.permutation(target_tensor.size(0))
    target_tensor = target_tensor[shuffled_indices]
    target_tensor = target_tensor.reshape(input_tensor.size())
    loss = hist_loss(input_tensor, target_tensor)
    assert_bool = loss.item() == pytest.approx(0, abs=1e-5)
    assert assert_bool, "Loss should be close to zero"


def test_equality(setup_combined_loss, setup_perceptual_loss,
                  setup_histogram_loss):
    to_test = [setup_combined_loss, setup_perceptual_loss,
               setup_histogram_loss]
    for loss_funct, device in to_test:
        input_tensor = torch.rand(1, 1, 224, 224, device=device)
        target_tensor = input_tensor.clone()

        loss = loss_funct(input_tensor, target_tensor)
        assert_bool = loss.item() == pytest.approx(0, abs=1e-6)
        assert assert_bool, "Loss should be close to " + \
                            "0 for identical inputs and targets."


def test_difference(setup_combined_loss, setup_perceptual_loss,
                    setup_histogram_loss):
    to_test = [setup_combined_loss, setup_perceptual_loss,
               setup_histogram_loss]
    for loss_funct, device in to_test:
        input_tensor = torch.rand(1, 1, 224, 224, device=device)
        target_tensor = torch.zeros(1, 1, 224, 224, device=device)

        loss = loss_funct(input_tensor, target_tensor)

        assert loss.item() > 0.1, 'Loss should be high for significantly ' + \
                                  'different inputs and targets.'
