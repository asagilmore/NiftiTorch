import torch
import torch.nn as nn


class conv_block(nn.Module):
    """ Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch
    normalization and a relu activation.
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class encoder_block(nn.Module):
    """ Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half
    after every block.
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class decoder_block(nn.Module):
    """ Decoder block:
    The decoder block begins with a transpose convolution, followed by a
    concatenation with the skip connection from the encoder block. Next comes
    the conv_block. Here the number filters decreases by half and the height
    and width doubles.
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2,
                                     padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    '''
    UNet model. See: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox.
    "U-net: Convolutional networks for biomedical image segmentation."
    arXiv:1505.04597 [cs.CV]

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ini_numb_filters : int, optional
        Number of filters in the first layer. Default is 16.

    '''
    def __init__(self, in_channels, out_channels, ini_numb_filters=16):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(in_channels, ini_numb_filters)
        self.e2 = encoder_block(ini_numb_filters, ini_numb_filters*2)
        self.e3 = encoder_block(ini_numb_filters*2, ini_numb_filters*4)
        self.e4 = encoder_block(ini_numb_filters*4, ini_numb_filters*8)

        """ Bottleneck """
        self.b = conv_block(ini_numb_filters*8, ini_numb_filters*16)

        """ Decoder """
        self.d1 = decoder_block(ini_numb_filters*16, ini_numb_filters*8)
        self.d2 = decoder_block(ini_numb_filters*8, ini_numb_filters*4)
        self.d3 = decoder_block(ini_numb_filters*4, ini_numb_filters*2)
        self.d4 = decoder_block(ini_numb_filters*2, ini_numb_filters)

        """ Classifier """
        self.outputs = nn.Conv2d(ini_numb_filters, out_channels, kernel_size=1,
                                 padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        outputs = torch.sigmoid(outputs)

        return outputs

    def train_self(self, data_loader, optimizer, criterion, device=None):
        '''
        Function to train one epoch of the model

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Data loader for the model.
        optimizer : torch.optim.Optimizer
            Optimizer for the model.
        criterion : torch.nn.Module
            Loss function for the model.
        device : str, optional
            Device to train, Default uses cuda if available, else cpu

        Returns
        -------
        float
            average loss for the epoch
        '''
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.train()

        running_loss = 0.0
        num_samples = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

        return running_loss / num_samples

    def valid_self(self, data_loader, criterion, device=None):
        '''
        Function to validate the model

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Data loader for the model.
        criterion : torch.nn.Module
            Loss function for the model.
        device : str, optional
            Device to train, Default uses cuda if available, else cpu

        Returns
        -------
        float
            average loss for the epoch
        '''
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.eval()

        running_loss = 0.0
        num_samples = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = self(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

        return running_loss / num_samples
