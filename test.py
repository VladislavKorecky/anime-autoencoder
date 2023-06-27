from random import randrange

import torch as t
from matplotlib.pyplot import imshow, show
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

from network import Autoencoder


# SETUP
dataset = ImageFolder("dataset", transform=Compose([
    Resize(64),
    CenterCrop(64),
    ToTensor()
]))

autoencoder = Autoencoder()
autoencoder.load_state_dict(t.load("model.pth"))  # loads the models parameters from the save file


# TESTING
while True:
    # get a random image from the dataset
    index = randrange(0, len(dataset))
    img = dataset[index][0]

    # try to encode and decode the image
    output = autoencoder.forward(img)

    # change the shape of the tensors from 3x64x64 to 64x64x3
    # this is so that the images can be displayed with matplotlib
    img = img.permute(1, 2, 0)
    output = output.permute(1, 2, 0)

    # show the original image
    imshow(img.numpy())
    show()

    # show the reconstructed image
    imshow(output.detach().numpy())
    show()
