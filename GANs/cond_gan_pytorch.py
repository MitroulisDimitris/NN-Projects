"""
Created on Tue Jan 17 13:24:23 2023

@author: DIMITRIS

Description: Conditional GAN Network, for eaducational purposed

"""


# %% Import and stuff
import torch
from torch import nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from mpl_toolkits.axes_grid1 import ImageGrid
import datetime
from  torch.utils import data


NUM_EPOCHS = 150
LR = 0.0002
LATENT_DIM = 100
IMG_SIZE = 28
CHANNELS = 1
B1 = 0.5
B2 = 0.999


GEN_STATE_DICT = "gen_state_dict"
DISC_STATE_DICT = "disc_state_dict"
GEN_OPTIMIZER = "gen_optimizer"
DISC_OPTIMIZER = "disc_optimizer"
G_LOSSES = "g_losses"
D_LOSSES = "d_losses"


SHUFFLE = True
PIN_MEMORY = True
NUM_WORKERS = 0
BATCH_SIZE = 128

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

# %% helper funcitons


def save_checkpoint(state, filename):
    print("=> Saving chekpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    generator.load_state_dict(checkpoint[GEN_STATE_DICT])
    optimizer_G.load_state_dict(checkpoint[GEN_OPTIMIZER])
    discriminator.load_state_dict(checkpoint[DISC_STATE_DICT])
    optimizer_D.load_state_dict(checkpoint[DISC_OPTIMIZER])
    G_losses = checkpoint[G_LOSSES]
    D_losses = checkpoint[D_LOSSES]


# takes input tensor and return a tensor of same
# size but every element has different value
def build_fake_labels(old_list):

    new_list = []
    for i, x in enumerate(old_list):

        if (i % 10) != x:
            new_list.append(i % 10)
        else:
            new_list.append((x.item()+1) % 10)

    return torch.tensor(new_list, dtype=torch.int64).to(device)


def add_noise(inputs, variance):
    noise = torch.randn_like(inputs)
    return inputs + variance*noise


def gen_image(caption=-1, randomLatent=True):

    generator.to('cpu')
    discriminator.to('cpu')

    with torch.no_grad():
        f, axarr = plt.subplots(1)

        if randomLatent:
            latent = torch.rand_like(torch.Tensor(1, 100))
        else:
            latent = specific_latent

        if caption == -1:
            caption = random.randint(0, 9)

        caption = torch.tensor(caption, dtype=torch.int64)
        fake_image = generator(latent, caption)

        axarr.imshow(fake_image[0][0])
        print("Supposed to be %d" % caption.item())

        
def discriminate_image(caption=-1,genOrReal=0):#random.randint(0, 1)):
    generator.to('cpu')
    discriminator.to('cpu')
    
    with torch.no_grad():
        for  i, (imgs, labels) in enumerate(example_loader):
            f, axarr = plt.subplots(1)
            
            fake_labels = build_fake_labels(labels.to(device))
            labels = labels.to('cpu')
            z = Variable(Tensor(np.random.normal(0, 1, (1,LATENT_DIM)))).cpu()
            if caption == -1:
                caption = random.randint(0, 9)
            caption = torch.tensor(caption, dtype=torch.int64)
            
            
            #feed discriminator fake image, expect "0" output
            if genOrReal == 0:
                fake_image = generator(z,caption)
                axarr.imshow(fake_image[0].reshape(-1, 28, 28)[0])
                pred = discriminator(fake_image,caption).detach()
                print("Discriminator Prediction: {},Should be: {}, label = {}".format(pred,"0",caption))
            #feed discriminator real image, expect "1" output
            else:
                fake_image = generator(z,labels[0])
                axarr.imshow(imgs[0].reshape(-1, 28, 28)[0])
                pred = discriminator(imgs.detach(),labels[0].detach()).detach()
                print("Discriminator Prediction: {},Should be: {}, label= {}".format(pred,"1",labels[0]+1))
            break

# %%train data

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
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=False
                                )

test_loader = data.DataLoader(
                                test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                pin_memory=False
                                )

example_loader = data.DataLoader(
                                train_dataset,
                                batch_size=1,
                                shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                )


# %% Detective: fake or no fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

        self.emb = nn.Embedding(10, 50)
        self.emb_fc = nn.Linear(50, 784)

        self.nconv1 = nn.Conv2d(2, 64, kernel_size=5)
        self.nconv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.nfc1 = nn.Linear(1152, 164)
        self.nfc2 = nn.Linear(164, 1)

    # oldWay flag to select between 2 train methods, not sure which is best yet
    def forward(self, x, c, oldWay=False):

        c = self.emb(c)
        c = self.emb_fc(c)
        c = c.view(-1, 1, 28, 28)
        x = torch.cat((c, x), 1)  # concat image[1,28,28] with text [1,28,28]

        x = F.leaky_relu(self.nconv1(x))
        x = F.leaky_relu(self.nconv2(x))
        x = self.pool(x)
        x = self.pool2(x)
        x = x.view(-1, 1152)
        x = F.leaky_relu(self.nfc1(x))
        x = F.dropout(x, training=self.training)
        x = self.nfc2(x)

        x = torch.sigmoid(x)
        return x


# %% Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(LATENT_DIM, 7*7*63)  # [n,100]->[n,3087]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)  # [n, 64, 16, 16] [32,..,..]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)  # [n, 32, , ]->[n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 16, 34, 34]-> [n, 1, 28, 28]

        self.emb = nn.Embedding(10, 50) 
        self.label_lin = nn.Linear(50, 49)
        self.conv_x_c = nn.ConvTranspose2d(65, 64, 4, stride=2)  # upsample [65,7,7] -> [64,14,14]
        self.tanh = nn.Tanh()

    def forward(self, x, c):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)  # (n,100) -> (n,3187)
        x = F.leaky_relu(x)
        x = x.view(-1, 63, 7, 7)  # (n,3187) -> (63,7,7)

        # Encode label
        c = self.emb(c)  # (n,) -> (n,50)
        c = self.label_lin(c)  # (n,50) -> (n,49)
        c = c.view(-1, 1, 7, 7)  # (n,49) -> (n,1,7,7)
        x = torch.cat((c, x), 1) # concat image[63,7,7] with text [1,7,7]

        x = self.ct1(x)  # [n, 64, 16, 16] [32,34,34]
        x = F.leaky_relu(x)

        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.leaky_relu(x)

        # Convolution to 28x28 (1 feature map)
        x = self.tanh(self.conv(x))
        return x

# %% Loss fucntion, optimizers
loss_func = nn.BCELoss()

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)


optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR,betas=(B1 ,B2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR,betas=(B1 ,B2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor   

if torch.cuda.is_available():
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    loss_func = loss_func.to(device)

#%% Train both models


img_list = []
G_losses = []
D_losses = []
real_losses = []
fake_losses = []

iters = 0

total_time = 0
epoch_times = []

 
# Adversarial ground truths
valid = torch.ones((BATCH_SIZE, 1)).to(device)
fake = torch.zeros((BATCH_SIZE, 1)).to(device)

if __name__ == '__main__':
    for epoch in range(NUM_EPOCHS):
        st= datetime.datetime.now()
        for i, (imgs, labels) in enumerate(train_loader):
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0],LATENT_DIM))))
    
            # transform to tensor [batch_size,1,28,28]
            real_imgs = Variable(imgs.type(Tensor)).to(device)
            labels = labels.to(device)
            
            # -----------------
            # Train discriminator
            # -----------------
            optimizer_D.zero_grad()
           
            #Forward pass through Discriminator
            real_pred = discriminator(real_imgs,labels)
            real_loss = loss_func(real_pred, valid)

    
            # Generate a batch of images
            gen_imgs = generator(z,labels)
            
            # Calculate D's loss on the all-fake batch
            fake_pred = discriminator(gen_imgs.detach(),labels.detach())
            fake_loss = loss_func(fake_pred, fake)

            
            real_loss.backward()
            fake_loss.backward()
            d_loss = (fake_loss + real_loss)/2
            optimizer_D.step()
    
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            #Loss measures generator's ability to fool the discriminator
            fake_pred = discriminator(gen_imgs,labels)

            g_loss = loss_func(fake_pred, valid)
            g_loss.backward()
            optimizer_G.step()

            
            g_loss = g_loss.mean().item()
            real_loss = real_loss.mean().item()
            fake_loss = fake_loss.mean().item()
            d_loss = d_loss.mean().item()
         
            end = datetime.datetime.now()

            end = (end - st).total_seconds() * 1000
            # Output training stats
            if i % 50 == 0:
                print('%.4f [%d/%d] Loss_D: %.4f\tLoss_G: %.4f tD(x): %.4f tD(G(z)): %.4f'
                      % (end/1000 ,epoch, NUM_EPOCHS, d_loss, g_loss, real_loss, fake_loss))
    
            # Save Losses for plotting later
            G_losses.append(g_loss)
            D_losses.append(d_loss)
            
            real_losses.append(real_loss)
            fake_losses.append(fake_loss)
            
    
            
    
            iters += 1
        epoch_times.append(end/1000)
        



    

# %%Plot Losses

plt.figure(figsize=(11,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses[:], label="G_losses")
plt.plot(D_losses[:], label="D_losses")


plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
    

#plt.plot(sampled_list[:], label="sampled")


#plt.plot(real_losses[:], label="real_losses")
#plt.plot(fake_losses[:], label="fake_losses")





# %% Save Model
checkpoint = {GEN_STATE_DICT : generator.state_dict(), 
              GEN_OPTIMIZER : optimizer_G.state_dict(),
              DISC_STATE_DICT : discriminator.state_dict(),
              DISC_OPTIMIZER : optimizer_D.state_dict()}
save_checkpoint(checkpoint, "cond_gan_pytorch10.pth.tar")

# %%  Load Model
load_checkpoint(torch.load("checkpoint_files\cond_gan_pytorch11.pth.tar",map_location=(device)))




