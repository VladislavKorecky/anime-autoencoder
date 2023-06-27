from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, LeakyReLU, ConvTranspose2d, Sigmoid, Tanh, \
    UpsamplingNearest2d


class Autoencoder(Module):
    def __init__(self) -> None:
        super().__init__()

        """self.encoder = Sequential(
            Conv2d(3, 12, 21),  # 64x64 -> 44x44
            LeakyReLU(),
            MaxPool2d(2),  # 44x44 -> 22x22

            Conv2d(12, 24, 11),  # 22x22 -> 12x12
            LeakyReLU(),

            Conv2d(24, 48, 12),  # 12x12 -> 1x1
            Tanh()
        )

        self.decoder = Sequential(
            ConvTranspose2d(48, 24, 12),  # 1x1 -> 12x12
            LeakyReLU(),

            ConvTranspose2d(24, 12, 11),  # 12x12 -> 22x22
            LeakyReLU(),

            UpsamplingNearest2d(scale_factor=2),  # 22x22 -> 44x44
            ConvTranspose2d(12, 3, 21),  # 44x44 -> 64x64
            Sigmoid()
        )"""

        self.encoder = Sequential(
            Conv2d(3, 4, 10),  # 64x64 -> 55x55
            LeakyReLU(),

            Conv2d(4, 5, 10),  # 55x55 -> 46x46
            LeakyReLU(),

            Conv2d(5, 8, 10),  # 46x46 -> 37x37
            LeakyReLU(),

            Conv2d(8, 15, 10),  # 37x37 -> 28x28
            LeakyReLU(),

            Conv2d(15, 34, 10),  # 28x28 -> 19x19
            LeakyReLU(),

            Conv2d(34, 122, 10),  # 19x19 -> 10x10
            LeakyReLU(),

            Conv2d(122, 50, 10),  # 10x10 -> 1x1
            Tanh()
        )

        self.decoder = Sequential(
            ConvTranspose2d(50, 122, 10),  # 1x1 -> 10x10
            LeakyReLU(),

            ConvTranspose2d(122, 34, 10),  # 10x10 -> 19x19
            LeakyReLU(),

            ConvTranspose2d(34, 15, 10),  # 19x19 -> 28x28
            LeakyReLU(),

            ConvTranspose2d(15, 8, 10),  # 28x28 -> 37x37
            LeakyReLU(),

            ConvTranspose2d(8, 5, 10),  # 37x37 -> 46x46
            LeakyReLU(),

            ConvTranspose2d(5, 4, 10),  # 46x46 -> 55x55
            LeakyReLU(),

            ConvTranspose2d(4, 3, 10),  # 55x55 -> 64x64
            Sigmoid()
        )

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode an image into a latent space.

        Args:
            x (Tensor): Image to encode.

        Returns:
            Tensor: Encoded latent space.
        """

        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        """
        Decode a latent space into an image.

        Args:
            x (Tensor): Latent space.

        Returns:
            Tensor: Reconstructed image.
        """

        return self.decoder(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode and then decode the input image.

        Args:
            x (Tensor): Image to encode.

        Returns:
            Tensor: Reconstructed input image.
        """

        return self.decode(self.encode(x))
