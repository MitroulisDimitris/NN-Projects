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
import matplotlib.colors as mcolors
import os
import gc


NUM_EPOCHS = 60
LR = 0.0008
LATENT_SPACE_SIZE = 20
IMG_SIZE = 28
CHANNELS = 1
B1 = 0.5
B2 = 0.999
RANDOM_SEED = 123


STATE_DICT = "state_dict"
MODEL_OPTIMIZER = "model_optimizer"
LOSSES = "losses"
RECON_LOSS = "recon_loss"
KL_DIV = "kl_div"

SHUFFLE = True
PIN_MEMORY = True
NUM_WORKERS = 0
BATCH_SIZE = 2000

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

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)

def plot():
    for i,(image, _) in example_loader:
        f, axarr = plt.subplots(2)
    
        # Reshape the array for plotting
        axarr[0].imshow(image[0].to(device))
    

        result = model.decoder(torch.tensor([-0.0,0.03]).to(device))
        result = result.squeeze(0)
        result = result.squeeze(0)
        axarr[1].imshow(result[0].to('cpu').numpy())
                  
def add_noise(inputs,variance):
    noise = torch.randn_like(inputs)
    return inputs + variance*noise


def save_checkpoint(state, filename):
    print("=> Saving chekpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    model.load_state_dict(checkpoint[STATE_DICT])
    optimizer.load_state_dict(checkpoint[MODEL_OPTIMIZER])
    losses = checkpoint[LOSSES]
    recon_losses = checkpoint[RECON_LOSS]
    kl_losses = checkpoint[KL_DIV]
    return losses, kl_losses, recon_losses
    
    
def get_numbered_images():  
    numberred_images = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        for i in range(len(labels[:])):
            if batch_idx == labels[i].item():
                numberred_images.append(images[i])
                break          
    
    return numberred_images
          

def plot_generated_images(c, figsize=(20, 2.5), n_images=10):
    model.to(device)
    c = torch.tensor(c, dtype=torch.int64).to(device)

    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    
    for batch_idx, (images, _) in enumerate(train_loader):
        
        images = images.to(device)
        with torch.no_grad():
           encoded, z_mean, z_log_var, decoded_images = model(images,c)[:n_images]

        orig_images = images[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))
            ax[i].imshow(curr_img.view((28, 28)))
            
            
def plot_numberred_images(numbered_images,figsize=(20, 2.5)):

    with torch.no_grad(): 
        fig, axes = plt.subplots(nrows=2, ncols=10, 
                                 sharex=True, sharey=True, figsize=(20, 2.5))

        for i in range(10):
            latent = torch.rand_like(torch.Tensor(20)).to(device)
            c = torch.tensor(i, dtype=torch.int64).to(device) 
            
            gen_img = model.decoder(latent,c).detach().to(torch.device('cpu'))
            axes[0][i].imshow(numbered_images[i].view((28, 28))) 
            axes[1][i].imshow(gen_img.view((28, 28)))
        
        plt.savefig('varying_latent_space/latent_space'+str(LATENT_SPACE_SIZE)+'.png')
        plt.figure().clear()
            
def plot_image(c):
    

    with torch.no_grad():
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        
        c = torch.tensor(c, dtype=torch.int64).to(device)
        latent = torch.rand_like(torch.Tensor(20)).to(device)
    
        decoded = model.decoder(latent,c).detach().to(torch.device('cpu'))
        axes.imshow(decoded.view((28, 28)))   
        
#plot a grid of r,c images,reccomended with 10,10
def plot_many_images(r=10,c=10):
    fig, axes = plt.subplots(nrows=r, ncols=c,figsize=(20, 20), sharex=True, sharey=True)
    
    for i in range(r):
        for y in range(c):
            caption = torch.tensor(y, dtype=torch.int64).to(device)
            latent = torch.rand_like(torch.Tensor(LATENT_SPACE_SIZE)).to(device)
    
            decoded = model.decoder(latent,caption).detach().to(torch.device('cpu'))
            axes[i][y].imshow(decoded.view((28, 28)))   
    
    plt.savefig('varying_latent_space/latent_size'+str(LATENT_SPACE_SIZE)+'.png')
    plt.figure().clear()


def plot_latent_space_with_labels(iteration, num_classes=10):
    d = {i:[] for i in range(num_classes)}

    with torch.no_grad():
        for i, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)
            
            embedding = model.encoding_fn(features)

            for i in range(num_classes):
                if i in targets:
                    mask = targets == i
                    d[i].append(embedding[mask].to('cpu').numpy())

    colors = list(mcolors.TABLEAU_COLORS.items())
    for i in range(num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(
            d[i][:, 0], d[i][:, 1],
            #color=colors[i][1],
            #label=f'{i}',
            alpha=0.5)

    #plt.legend()
    #plt.savefig('latent_space/iteration'+str(iteration)+'.png')
    plt.figure().clear()


def plot_losses():
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(losses[:], label="L")
    plt.plot(kl_losses[:], label="KL")
    plt.plot(recon_losses[:], label="Recon")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    
    plt.savefig('varying_plot_losses/latent_size'+str(LATENT_SPACE_SIZE)+'.png')
    plt.figure().clear()
    
def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()
    


# %%Train Data

set_deterministic
set_all_seeds(RANDOM_SEED)

transform = transforms.Compose([
    transforms.ToTensor()])


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
                                pin_memory=PIN_MEMORY,
                                drop_last=False
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

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :28, :28]

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
           
        self.z_mean = torch.nn.Linear(3136, LATENT_SPACE_SIZE) # 2 dim for visualization purposes
        self.z_log_var = torch.nn.Linear(3136, LATENT_SPACE_SIZE)
        
        self.d_emb = nn.Embedding(10, 50)
        self.e_emb_fc = nn.Linear(50, 784)
        self.d_emb_fc = nn.Linear(50, 49)
       
        #########
        # Encoder
        #########
        self.e_conv1 = nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1)
        self.e_conv2 = nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.e_conv3 = nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.e_conv4 = nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1)
        self.flatten = nn.Flatten()
        

        #########
        # Decoder
        #########  
        self.d_lin = torch.nn.Linear(LATENT_SPACE_SIZE, 3087)
        self.d_conv1 = nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1)
        self.d_conv2 = nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1)              
        self.d_conv3 = nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0)              
        self.d_conv4 = nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0)
       
        
    def encoder(self, x, c):
        #c = self.e_emb(c)
        #c = self.e_emb_fc(c)
        #c = c.view(-1, 1, 28, 28)
        #x = torch.cat((x, c), 1)

        x = F.leaky_relu(self.e_conv1(x))
        x = F.leaky_relu(self.e_conv2(x))
        x = F.leaky_relu(self.e_conv3(x))
        x = self.e_conv4(x)
        x = self.flatten(x)
        return x
    
    
    def decoder(self, x, c):
        c = self.d_emb(c)
        c = self.d_emb_fc(c)
        c = c.view(-1, 1, 7, 7)
        
        x = self.d_lin(x)
        x = x.view(-1, 63, 7, 7)
        
        x = torch.cat((c, x), 1)
        
        x = x.view(-1,64,7,7)
        x = F.leaky_relu(self.d_conv1(x))
        x = F.leaky_relu(self.d_conv2(x))
        x = F.leaky_relu(self.d_conv3(x))
        x = F.leaky_relu(self.d_conv4(x))
        x = x[:, :, :28, :28]
        x = torch.sigmoid(x)
        return x
    
    def encoding_fn(self, x, c):
        x = self.encoder(x, c)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x, c):
        x = self.encoder(x, c)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var) # sample μ,σ to create distribution
        decoded = self.decoder(encoded,c)
        return encoded, z_mean, z_log_var, decoded
        
# %% Loss func 

model = VAE()
 
# Validation using MSE Loss function
loss_function = nn.MSELoss(reduction='none')

#Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(B1 ,B2))
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

if torch.cuda.is_available():
    model.cuda()
    loss_function.cuda()


# %%Train Model

if 'numbered_images' not in locals():    
   #numbered_images = get_numbered_images()
   pass


latent_size_spaces = [2,5,20,50,200,500,2000]

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)
    loss_function.to(device)
    model.train()
    torch.backends.cudnn.benchmark = True
    
for late_size in latent_size_spaces:
    LATENT_SPACE_SIZE = late_size
    model = VAE()
         
    # Validation using MSE Loss function
    loss_function = nn.MSELoss(reduction='none')

    #Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(B1 ,B2))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    logging_interval = 10
    losses = []
    kl_losses = []
    recon_losses = []
    recon_loss = 0
    kl_div = 0
    iter = 0
    alpha = 1

    if torch.cuda.is_available():
        model.cuda()
        loss_function.cuda()
        
    for epoch in range(4):
        st = time.time()
        for batch, (imgs, labels) in enumerate(train_loader):
    
            imgs = imgs.to(device,memory_format=torch.channels_last)
            labels = labels.to(device)
    
            # set gradients to zero
            optimizer.zero_grad()
    
            encoded, z_mean, z_log_var, decoded = model(imgs,labels)
    
            # Total Loss = Recon_loss + KLDivergence
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1) # sum over latent dimension
    
            kl_div = kl_div.mean()
    
            recon_loss = loss_function(decoded, imgs)
            recon_loss = recon_loss.view(BATCH_SIZE, -1).sum(axis=1) # sum over pixels
            recon_loss = recon_loss.mean() # average over batch dimension
            
            loss = alpha*recon_loss + kl_div
            loss.backward()
            
            
            optimizer.step()  # Updates Weights
            
            loss_item = loss.item()
            recon_loss_item = recon_loss.item()
            kl_div_item = kl_div.item()
            
            #scheduler.step()
            
            
            if iter % logging_interval == 0:
                print('[%d/%d][%d/%d]\t, LOSS:%.4f (recon_loss : %.4f, kl_loss = %.6f'
                      %(epoch, NUM_EPOCHS, batch, len(train_loader), loss_item, recon_loss_item, kl_div_item))
                #plot_numberred_images(iter*BATCH_SIZE,numbered_images)
            
           
            losses.append(loss_item)
            kl_losses.append(kl_div_item)
            recon_losses.append(recon_loss_item)
            
            iter +=1
    
    plot_many_images()     
    plot_losses()
    print('----------------------RESTARTING--------------------')
    print(time.time()-st)
     
            
# %%


plt.figure(figsize=(10, 5))
plt.title("Loss During Training")
plt.plot(losses[:600], label="L")
plt.plot(kl_losses[:600], label="KL")
plt.plot(recon_losses[:600], label="Recon")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# %% Save Model
checkpoint = {STATE_DICT : model.state_dict(),
              MODEL_OPTIMIZER : optimizer.state_dict(),
              LOSSES: losses,
              RECON_LOSS:recon_losses,
              KL_DIV:kl_losses}
save_checkpoint(checkpoint, "cVAE4.pth.tar")

# %%  Load Model
losses,kl_losses,recon_losses = load_checkpoint(torch.load("cVAE4.pth.tar",map_location=(device)))
            

# %%
# Keep latent noise the same to get same results
latent = torch.rand_like(torch.Tensor(20)).to(device)


# %% print reconstructed image vs generated image
if 'numbered_images' not in locals():
    numbered_images = get_numbered_images()


c = 4
with torch.no_grad():
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    #for batch_idx, (images, labels) in enumerate(example_loader):
    
    c = torch.tensor(c, dtype=torch.int64).to(device)
    c1 = torch.tensor(4, dtype=torch.int64).to(device)
    latent = torch.rand_like(torch.Tensor(20)).to(device)

    
    decoded = model.decoder(latent,c).detach().to(torch.device('cpu'))
    
    enc = model.encoding_fn(numbered_images[c.item()][None, :].to(device),c1)
    enc = model.decoder(enc,c)
    enc = enc.detach().to(torch.device('cpu'))
    
    #image from dataset
    axes[0].imshow(numbered_images[c.item()][None, :].view((28, 28)))   
    
    #reconstructed
    axes[1].imshow(enc.view((28, 28))) 
    
    #image from decoder, caption
    axes[2].imshow(decoded.view((28, 28)))   
        
#%%
num_classes=10
iteration = 1

d = {i:[] for i in range(num_classes)}

with torch.no_grad():
    for i, (features, targets) in enumerate(train_loader):

        features = features.to(device)
        targets = targets.to(device)
        
        embedding = model.encoding_fn(features)

        for i in range(num_classes):
            if i in targets:
                mask = targets == i
                d[i].append(embedding[mask].to('cpu').numpy())

colors = list(mcolors.TABLEAU_COLORS.items())
for i in range(num_classes):
    d[i] = np.concatenate(d[i])
    plt.scatter(
        d[i][:, 0], d[i][:, 1],
        #color=colors[i][1],
        #label=f'{i}',
        alpha=0.5)

#plt.legend()
#plt.savefig('latent_space/iteration'+str(iteration)+'.png')
plt.figure().clear





