# %% Import and stuff
import torch
from torch import nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import random
import torchvision.utils as vutils
from  torch.utils import data
from mpl_toolkits.axes_grid1 import ImageGrid


NUM_EPOCHS = 150
LR = 1e-3
LATENT_DIM = 100
IMG_SIZE = 28
CHANNELS = 1
B1 = 0.5
B2 = 0.999


STATE_DICT = "state_dict"
MODEL_OPTIMIZER = "model_optimizer"

LOSSES = "losses"



SHUFFLE = True
PIN_MEMORY = True
NUM_WORKERS = 0
BATCH_SIZE = 750

specific_latent = torch.tensor([[0.7628, 0.1779, 0.3978, 0.3606, 0.6387,
         0.3044, 0.8340, 0.3884, 0.9313, 0.5635, 0.1994, 0.6934, 0.5326,
         0.3676, 0.5342, 0.9480, 0.4120, 0.5845, 0.4035, 0.5298, 0.0177,
         0.5605, 0.6453, 0.9576, 0.7153, 0.1923, 0.8122, 0.0937, 0.5744,
         0.5951, 0.8890, 0.4838, 0.5707, 0.6760, 0.3738, 0.2796, 0.1549,
         0.8220, 0.2800, 0.4051, 0.2553, 0.1831, 0.0046, 0.9021, 0.0264,
         0.2327, 0.8261, 0.0534, 0.1582, 0.4087, 0.9047, 0.1409, 0.6864,
         0.1439, 0.3432, 0.1072, 0.5907, 0.6756, 0.6942, 0.6814, 0.3368,
         0.4138, 0.8030, 0.7024, 0.3309, 0.7288, 0.2193, 0.1954, 0.9948,
         0.1201, 0.9483, 0.7407, 0.4849, 0.6500, 0.8649, 0.7405, 0.4725,
         0.5373, 0.6541, 0.5444, 0.7425, 0.8940, 0.3580, 0.3905, 0.8924,
         0.2995, 0.3726, 0.5399, 0.3057, 0.3380, 0.8313, 0.1137, 0.0120,
         0.7714, 0.2561, 0.2569, 0.2994, 0.7648, 0.2413, 0.6101
        ]])


img_shape = (CHANNELS, IMG_SIZE, IMG_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:{}'.format(device))
# %%helper functions


def plot():
    f, axarr = plt.subplots(2)

    for i, item in enumerate(image):
    # Reshape the array for plotting
        item = item.reshape(-1, 28, 28)
        axarr[0].imshow(item[0].cpu())

    for i, item in enumerate(reconstructed):
        item = item.reshape(-1, 28, 28).cpu()
        item = item.detach().numpy()
        axarr[1].imshow(item[0])
        
        
        
def showExample():
    for image, _ in example_loader:
        f, axarr = plt.subplots(2)
        image = image.reshape(-1,28*28).to(device)

        model.to(device)

        recon = model(add_noise(image,0.2))

        image = image.reshape(-1, 28, 28)
        axarr[0].imshow(image[0].cpu())


        recon = recon.reshape(-1, 28, 28).to('cpu')
        axarr[1].imshow(recon[0].detach().numpy())

        break

def add_noise(inputs,variance):
    noise = torch.randn_like(inputs)
    return inputs + variance*noise


def save_checkpoint(state, filename):
    print("=> Saving chekpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    model.load_state_dict(checkpoint[STATE_DICT])
    optimizer.load_state_dict(checkpoint[MODEL_OPTIMIZER])



# %%     
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])


train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = data.DataLoader(
                                train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=SHUFFLE,
                                num_workers=NUM_WORKERS,
                                pin_memory=PIN_MEMORY
                                )

test_loader = data.DataLoader(
                                test_dataset,
                                batch_size=32,
                                shuffle=True,
                                num_workers=0
                                )

example_loader = data.DataLoader(
                                train_dataset,
                                batch_size=1,
                                shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                )

# %%Model
input_size = 28*28  #784
hidden_size = 128
code_size = 32


class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Encoder 
        self.encoder = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,code_size)
        )
        
        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.model = nn.Sequential(
        
        )
        
        
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
        
# %% Loss func 
model = autoencoder()

 
# Validation using MSE Loss function
loss_function = nn.MSELoss()
# loss_function = nn.CrossEntropyLoss()

#Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(B1 ,B2))

if torch.cuda.is_available():
    model.cuda()
    loss_function.cuda()


# %%     

losses = []

if torch.cuda.is_available():
    model.to(device)
    loss_function.to(device)

model.train()
for epoch in range(NUM_EPOCHS):
    for image, _ in train_loader:
        image = image.reshape(-1,28*28).to(device)
        noised_image = add_noise(image,0.2)
        #set gradients to zero
        
        reconstructed = model(noised_image)
        loss = loss_function(reconstructed , noised_image)
        
        optimizer.zero_grad()
        loss.backward() # Preforms Backpropagation and calculates  gradients 
        optimizer.step() # Updates Weights based on the gradients computed above
        losses.append(loss.item())
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, NUM_EPOCHS, loss.item()))

# %%
plt.figure(figsize=(10,5))
plt.title("Loss During Training")
plt.plot(losses,label="G")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# %% Save Model
checkpoint = {STATE_DICT : model.state_dict(), 
              MODEL_OPTIMIZER : optimizer.state_dict()}
save_checkpoint(checkpoint, "denoising-autoencoder-.pth.tar")

# %%  Load Model
load_checkpoint(torch.load("cond_gan_pytorch10.pth.tar",map_location=(device)))




