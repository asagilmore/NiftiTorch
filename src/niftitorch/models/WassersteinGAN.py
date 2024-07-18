import torch.nn as nn
import torch
from .UNet import UNet


class Critic(nn.Module):
    def __init__(self, in_channels, ini_num_filters=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ini_num_filters, kernel_size=4,
                      stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(ini_num_filters, ini_num_filters*2, kernel_size=4,
                      stride=2, padding=1),
            nn.InstanceNorm2d(ini_num_filters*2, affine=True),
            nn.GELU(),
            nn.Conv2d(ini_num_filters*2, ini_num_filters*4, kernel_size=4,
                      stride=2, padding=1),
            nn.InstanceNorm2d(ini_num_filters*4, affine=True),
            nn.GELU(),
            nn.Conv2d(ini_num_filters*4, ini_num_filters*8, kernel_size=4,
                      stride=2, padding=1),
            nn.InstanceNorm2d(ini_num_filters*8, affine=True),
            nn.GELU(),
            nn.Conv2d(ini_num_filters*8, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x).view(-1)


class GradientPenalty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, critic, real_images, fake_images):
        batch_size, c, h, w = real_images.size()
        epsilon = torch.rand(batch_size, 1, 1, 1,
                             requires_grad=True).to(real_images.device)
        epsilon = epsilon.expand_as(real_images)

        interpolated_images = epsilon * real_images
        interpolated_images += (1 - epsilon) * fake_images

        mixed_scores = critic(interpolated_images)
        grad_outputs = torch.ones_like(mixed_scores, requires_grad=False)

        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(gradient.size(0), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1) ** 2)
        return penalty


class WassersteinGAN(nn.Module):
    '''
    Wasserstein GAN with Gradient Penalty.
    See: Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein
    generative adversarial networks."
    arXiv:1701.07875 [stat.ML]

    Parameters
    ----------
    critic : nn.Module, optional
        Network to use as critic, Default is Critic
    generator : nn.Module, optional
        Network to use as generator, Default is UNet
    critic_penalty : nn.Module, optional
        Penalty function for the critic, Default is GradientPenalty
    lambda_gp : float, optional
        Weight of the penalty function, Default is 10
    second_criterior : nn.Module, optional
        Optional, second loss function for the generator, Default is None
    second_criterion_lambda : float, optional
        Weight of the second loss function, Default is 1
    identity_loss : nn.Module, optional
        Optional, loss function to be used for identity loss, Default is None
    identity_lambda : float, optional
        Weight of the identity loss, Default is 1
    '''
    def __init__(self, critic=None, generator=None, critic_penalty=None,
                 lambda_gp=10, second_criterior=None,
                 second_criterion_lambda=1, identity_loss=None,
                 identity_lambda=1):
        super().__init__()
        if critic is None:
            critic = Critic(2)
        if generator is None:
            generator = UNet(1, 1)
        if critic_penalty is None:
            critic_penalty = GradientPenalty()

        self.critic = critic
        self.generator = generator
        self.critic_penalty = critic_penalty
        self.lambda_gp = lambda_gp
        self.second_criterior = second_criterior
        self.second_criterion_lambda = second_criterion_lambda
        self.identity_loss = identity_loss
        self.identity_lambda = identity_lambda

    def forward(self, x):
        return self.generator(x)

    def _train_critic(self, data_loader1, data_loader2, critic_optimizer,
                      device):
        self.critic.train()
        self.generator.eval()

        running_critic_loss = 0
        num_samples = 0
        for (images1, mask1), (images2, mask2) in zip(data_loader1,
                                                      data_loader2):
            # skip if batch sizes are different
            if images1.size(0) != images2.size(0):
                continue
            critic_optimizer.zero_grad()

            images1, mask1 = images1.to(device), mask1.to(device)
            images2, mask2 = images2.to(device), mask2.to(device)

            fake_masks = self.generator(images1)

            fake_concat = torch.cat([images1, fake_masks], dim=1)
            real_concat = torch.cat([images2, mask2], dim=1)

            real_validity = self.critic(real_concat)
            fake_validity = self.critic(fake_concat)

            critic_gp = self.critic_penalty(self.critic, real_concat,
                                            fake_concat)

            critic_loss = -torch.mean(real_validity)
            critic_loss += torch.mean(fake_validity)
            critic_loss += self.lambda_gp * critic_gp

            critic_loss.backward()
            critic_optimizer.step()

            running_critic_loss += critic_loss.item() * images1.size(0)
            num_samples += images1.size(0)

        final_loss = running_critic_loss / num_samples
        return final_loss

    def _train_generator(self, data_loader, generator_optimizer, device):
        self.critic.eval()
        self.generator.train()

        running_generator_loss = 0
        running_critic_loss = 0
        running_identity_loss = 0
        running_second_loss = 0
        num_samples = 0
        for images, masks in data_loader:
            generator_optimizer.zero_grad()

            images, masks = images.to(device), masks.to(device)

            outputs = self.generator(images)

            output_concat = torch.cat([images, outputs], dim=1)
            validity = self.critic(output_concat)

            critic_loss = -torch.mean(validity)

            running_critic_loss += critic_loss.item() * images.size(0)

            generator_loss = critic_loss

            if self.second_criterior is not None:
                second_loss = self.second_criterior(outputs, masks)
                second_loss *= self.second_criterion_lambda
                running_second_loss += second_loss * images.size(0)
                generator_loss += second_loss

            if self.identity_loss is not None:
                identity_img = self.generator(masks)
                identity_loss = self.identity_loss(identity_img, masks)
                identity_loss *= self.identity_lambda
                running_identity_loss += identity_loss * images.size(0)
                generator_loss += identity_loss

            generator_loss.backward()
            generator_optimizer.step()

            running_generator_loss += generator_loss.item() * images.size(0)
            num_samples += images.size(0)

        total_loss = running_generator_loss / num_samples
        critic_loss = running_critic_loss / num_samples
        second_loss = running_second_loss / num_samples
        identity_loss = running_identity_loss / num_samples
        return {'total_loss': total_loss, 'critic_loss': critic_loss,
                'second_loss': second_loss, 'identity_loss': identity_loss}

    def _valid_generator(self, data_loader, device):
        self.critic.eval()
        self.generator.eval()

        running_generator_loss = 0
        running_critic_loss = 0
        running_identity_loss = 0
        running_second_loss = 0
        num_samples = 0
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = self.generator(images)

            output_concat = torch.cat([images, outputs], dim=1)
            validity = self.critic(output_concat)
            critic_loss = -torch.mean(validity)
            running_critic_loss += critic_loss.item() * images.size(0)

            generator_loss = critic_loss

            if self.second_criterior is not None:
                second_loss = self.second_criterior(outputs, masks)
                second_loss *= self.second_criterion_lambda
                running_second_loss += second_loss * images.size(0)
                generator_loss += second_loss

            if self.identity_loss is not None:
                identity_img = self.generator(masks)
                identity_loss = self.identity_loss(identity_img, masks)
                identity_loss *= self.identity_lambda
                running_identity_loss += identity_loss * images.size(0)
                generator_loss += identity_loss

            running_generator_loss += generator_loss.item() * images.size(0)
            num_samples += images.size(0)

        total_loss = running_generator_loss / num_samples
        critic_loss = running_critic_loss / num_samples
        second_loss = running_second_loss / num_samples
        identity_loss = running_identity_loss / num_samples
        return {'total_loss': total_loss, 'critic_loss': critic_loss,
                'second_loss': second_loss, 'identity_loss': identity_loss}

    def _valid_critic(self, data_loader, device):
        self.critic.eval()
        self.generator.eval()

        running_critic_loss = 0
        num_samples = 0
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)

            fake_masks = self.generator(images)

            fake_concat = torch.cat([images, fake_masks], dim=1)
            real_concat = torch.cat([images, masks], dim=1)

            real_validity = self.critic(real_concat)
            fake_validity = self.critic(fake_concat)

            critic_loss = -torch.mean(real_validity)
            critic_loss += torch.mean(fake_validity)

            running_critic_loss += critic_loss.item() * images.size(0)
            num_samples += images.size(0)

        final_loss = running_critic_loss / num_samples
        return final_loss

    def valid_self(self, data_loader, device=None):
        '''
        Function to validate the Model

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader for the validation set
        device : torch.device, optional
            Device to use, Default uses cuda if available, else cpu

        Returns
        -------
        dict
            Dictionary with critic_loss and generator_loss as keys
        '''
        if device is None:
            device = torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.critic.eval()
        self.generator.eval()



    def train_self(self, data_loader1, data_loader2, critic_optimizer,
                   generator_optimizer, device=None, critic_iters=5):
        '''
        Function to train one epoch of the Model

        this function needs two identical data loaders with shuffle set to
        True to train the critic.

        Parameters
        ----------
        data_loader1 : torch.utils.data.DataLoader
            DataLoader for the first set of images
        data_loader2 : torch.utils.data.DataLoader
            DataLoader for the second set of images
        critic_optimizer : torch.optim.Optimizer
            Optimizer for the critic
        generator_optimizer : torch.optim.Optimizer
            Optimizer for the generator
        device : torch.device, optional
            Device to use, Default uses cuda if available, else cpu
        critic_iters : int, optional
            Number of iterations to train the critic relative to the generator
            Default is 5

        Returns
        -------
        dict
            Dictionary with critic_loss and generator_loss as keys
        '''
        if device is None:
            device = torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        running_critic_loss = 0
        for i in range(critic_iters):
            loss = self._train_critic(data_loader1, data_loader2,
                                      critic_optimizer, device)
            running_critic_loss += loss

        critic_loss = running_critic_loss / critic_iters

        # train generator
        loss_dict = self._train_generator(data_loader1, generator_optimizer,
                                          device)

        return {'critic_loss': critic_loss,
                'generator_loss': loss_dict}
