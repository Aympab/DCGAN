import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils
import os
from tqdm import tqdm


class MNIST_Encoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, encoding_dim),
             nn.ReLU()]
        )

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MNIST_Decoder(nn.Module):
    def __init__(self, encoding_dim, input_size):
        super().__init__()
        self.layers = nn.ModuleList(
                [nn.Linear(encoding_dim, input_size)]
        )
        
    def forward(self,z):
        for layer in self.layers:
            z = layer(z)
        return z

class MNIST_Autoencoder(nn.Module):
    def __init__(self, input_size = 784, encoding_dim = 32):
        super().__init__()
        self.encoder = MNIST_Encoder(input_size, encoding_dim)
        self.decoder = MNIST_Decoder(encoding_dim, input_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten to (nm, 1) vector
        x = self.encoder(x) # here we get the latent z
        x = self.decoder(x) # here we get the reconsturcted input
        x = torch.sigmoid(x)
        x = x.reshape(x.size(0), 1, 28, 28) # reshape this flatten vector to the original image size    
        return x
    

    

    
class Encoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, encoding_dim),
            nn.ReLU()])

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, encoding_dim, input_size):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [nn.Linear(encoding_dim, 4096),
            nn.ReLU(), 
            nn.Linear(4096, input_size)])
    
    def forward(self,z):
        for layer in self.layers:
            z = layer(z)
        return z
    
class Autoencoder(nn.Module):
    def __init__(self, input_size = 224*224, encoding_dim = 1024):
        super().__init__()
        self.encoder = Encoder(input_size, encoding_dim)
        self.decoder = Decoder(encoding_dim, input_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten to (nm, 1) vector
        x = self.encoder(x) # here we get the latent z
        x = self.decoder(x) # here we get the reconsturcted input
        x = torch.sigmoid(x)
        x = x.reshape(x.size(0), 1, 224, 224) # reshape this flatten vector to the original image size    
        return x
    

class C_Encoder_28(nn.Module):
    def __init__(self, fc2_input_dim, encoded_space_dim, channels = 1):
        super().__init__()
        augmentation = int(fc2_input_dim / 784)
        
        ### Convolutional section
        self.enc_conv1 = nn.Conv2d(channels, 8, 3, stride=2, padding=1)
        self.enc_relu1 = nn.ReLU(True)
        self.enc_conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.enc_batchn2 = nn.BatchNorm2d(16)
        self.enc_relu2 = nn.ReLU(True)
        self.enc_conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.enc_relu3 = nn.ReLU(True)
        """
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )"""
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32 * augmentation, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.enc_conv1(x)
        x = self.enc_relu1(x)
        x = self.enc_conv2(x)
        x = self.enc_batchn2(x)
        x = self.enc_relu2(x)
        x = self.enc_conv3(x)
        x = self.enc_relu3(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
    
class C_Decoder_28(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim, channels = 1):
        super().__init__()
        augmentation = int(fc2_input_dim / 784)
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32 * augmentation),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, int(3 * np.sqrt(augmentation)), int(3 * np.sqrt(augmentation))))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, channels, 3, stride=2, 
            padding=1, output_padding=1)
        )
        

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
class C_Autoencoder_28(nn.Module):
    def __init__(self, input_size = 28*28, encoding_dim = 64, channels = 1):
        super().__init__()
        self.input_size = input_size
        self.encoder = C_Encoder_28(input_size, encoding_dim, channels = channels)
        self.decoder = C_Decoder_28(encoding_dim, input_size, channels = channels)
    
    def forward(self, x):
        x = self.encoder(x) # here we get the latent z
        x = self.decoder(x) # here we get the reconsturcted input
        x = x.reshape(x.size(0), 1, int(np.sqrt(self.input_size)), 
                      int(np.sqrt(self.input_size))) # reshape this flatten vector to the original image size    
        return x



class C_Encoder_224(nn.Module):
    def __init__(self, fc2_input_dim, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.enc_conv1 = nn.Conv2d(1, 32, (7,7), stride=3, padding=3) # 32 * 75 * 75
        self.relu1 = nn.ReLU()
        self.enc_conv2 = nn.Conv2d(32, 64, (5, 5), stride=3, padding=2) # 64 * 25 * 25
        self.relu2 = nn.ReLU()
        self.enc_conv3 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=1) # 128 * 13 * 13
        self.relu3 = nn.ReLU()
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(128 * 13 * 13, encoded_space_dim),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.enc_conv1(x)
        #print(f"After conv1 {x.shape}")
        x = self.relu1(x)
        x = self.enc_conv2(x)
        #print(f"After conv2 {x.shape}")
        x = self.relu2(x)
        x = self.enc_conv3(x)
        #print(f"After conv3 {x.shape}")
        x = self.relu3(x)
        x = self.flatten(x)
        #print(f"After flatten {x.shape}")
        x = self.encoder_lin(x)
        #print(f"After encoder_linear {x.shape}")
        return x


class C_Decoder_224(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128 * 13 * 13),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 13, 13))
        self.dec_convt1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.dec_convt2 = nn.ConvTranspose2d(64, 32, 5, stride=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dec_convt3 = nn.ConvTranspose2d(32, 1, 7, stride=3, padding=3, output_padding=1)
        

    def forward(self, x):
        #print("DECODER")
        #print(f"Start {x.shape}")
        x = self.decoder_lin(x)
        #print(f"After decoder_lin {x.shape}")
        x = self.unflatten(x)
        #print(f"After unflatten {x.shape}")
        x = self.dec_convt1(x)
        x = self.relu1(x)
        #print(f"After transposed conv 1 {x.shape}")
        x = self.dec_convt2(x)
        x = self.relu2(x)
        #print(f"After transposed conv 2 {x.shape}")
        x = self.dec_convt3(x)
        #print(f"After transposed conv 3 {x.shape}")
        x = torch.sigmoid(x)
        return x


class C_Autoencoder_224(nn.Module):
    def __init__(self, input_size = 224*224, encoding_dim = 1024):
        super().__init__()
        self.input_size = input_size
        self.encoder = C_Encoder_224(input_size, encoding_dim)
        self.decoder = C_Decoder_224(encoding_dim, input_size)
    
    def forward(self, x):
        x = self.encoder(x) # here we get the latent z
        x = self.decoder(x) # here we get the reconsturcted input
        x = x.reshape(x.size(0), 1, 224, 224) # reshape this flatten vector to the original image size    
        return x


    
def train(model, optimizer, trainloader = None, valloader = None, num_epochs = 1):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # name dataloaders for phrase
    phases = ['train']
    dataloaders = {'train':trainloader}
    if valloader:
        phases.append('valid')
        dataloaders['valid'] = valloader
        
    model.to(device)
    #criterion = F.binary_cross_entropy(autoencoder(x), target)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss, running_correct, count = 0.0, 0, 0
            for batch_idx, x in enumerate(tqdm(dataloaders[phase])):
            #for batch_idx, (x, y) in enumerate(dataloaders[phase]):
                #print(f"Batch {batch_idx}")
                #x,y = x.to(device), y.to(device)
                if isinstance(x, list):
                    x = x[0]
                x = x.to(device)

                # zero param gradients
                optimizer.zero_grad()

                # forward: track history if training phase
                with torch.set_grad_enabled(phase=='train'): # pytorch >= 0.4
                    outputs = model(x)
                    loss    = criterion(outputs, x)
                    #preds,_ = torch.max(outputs,1) # for accuracy metric
                    # backward & optimize if training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # stats
                running_loss += loss.item() * x.size(0)
                count += len(x)
            
            epoch_loss = running_loss / count
            print(f'{phase} loss {epoch_loss:.6f}')
        print()
            