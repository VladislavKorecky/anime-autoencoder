import torch as t
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor, CenterCrop

from network import Autoencoder


# CONFIG
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.0001


# SETUP
device = t.device("cuda" if t.cuda.is_available() else "cpu")

dataset = ImageFolder("dataset", transform=Compose([
    Resize(64),
    CenterCrop(64),
    ToTensor()
]))
data_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

autoencoder = Autoencoder()
optimizer = Adam(autoencoder.parameters(), LEARNING_RATE)


# TRAINING
for epoch in range(EPOCHS):
    for i, (img_batch, _) in enumerate(data_loader):
        # put the batch on the GPU
        img_batch = img_batch.to(device)

        # encode and decode the image batch
        output = autoencoder.forward(img_batch)

        # compare the output with the input to get better reconstructions
        loss = binary_cross_entropy(output, img_batch)

        # backpropagate and update the autoencoder
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss.item())

        # save the autoencoder parameters
        t.save(autoencoder.state_dict(), "model.pth")
