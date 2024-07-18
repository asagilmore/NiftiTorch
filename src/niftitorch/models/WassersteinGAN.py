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
    Wasserstein GAN with Gradient Penalty

    Args:
    - critic (nn.Module): the critic network, Default is Critic
    - generator (nn.Module): the generator network, Default is UNet
    - critic_penalty (nn.Module): the penalty function for the critic,
        Default is GradientPenalty
    - lambda_gp (float): the weight of the gradient penalty, Default is 10
    - second_criterior (nn.Module): the second loss function for the generator,
        Default is None, in which case the generator will only use the critic
        loss
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

            output = self.generator(images)

            output_concat = torch.cat([images, output], dim=1)
            validity = self.critic(output_concat)

            critic_loss = -torch.mean(validity)

            running_critic_loss += critic_loss.item() * images.size(0)

            generator_loss = critic_loss

            if self.second_criterior is not None:
                second_loss = self.second_criterior(output, masks)
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

    def train_self(self, data_loader1, data_loader2, critic_optimizer,
                   generator_optimizer, device=None, critic_iters=5,
                   lambda_gp=10):
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