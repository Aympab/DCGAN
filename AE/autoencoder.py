import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from gan_architecture import ResConvBlock, ResUpConvBlock



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






debug=False
class C_Encoder_224(nn.Module):
    def __init__(self, fc2_input_dim, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.enc_conv1 = nn.Conv2d(3, 64, (7,7), stride=1, padding=3, bias=False) # 64 * 224 * 224
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.enc_conv2 = nn.Conv2d(64, 64, (7,7), stride=2, padding=3, bias=False) # 64 * 112 * 112
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.enc_conv3 = nn.Conv2d(64, 64, (5, 5), stride=2, padding=2, bias=False) # 64 * 56 * 56
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.enc_conv4 = nn.Conv2d(64, 128, (5, 5), stride=1, padding=2, bias=False) # 128 * 56 * 56
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.enc_conv5 = nn.Conv2d(128, 128, (3, 3), stride=2, padding=1, bias=False) # 128 * 28 * 28
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.enc_conv6 = nn.Conv2d(128, 256, (3, 3), stride=2, padding=1, bias=False) # 256 * 14 * 14
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.enc_conv7 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1, bias=False) # 256 * 14 * 14
        self.batchnorm7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()
        self.enc_conv8 = nn.Conv2d(256, 512, (3, 3), stride=2, padding=1, bias=False) # 512 * 7 * 7
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        #self.block7 = ResConvBlock(256, 512, stride=2)
        self.enc_conv9 = nn.Conv2d(512, 1024, (3, 3), stride=2, padding=1, bias=False) # 1024 * 4 * 4
        self.batchnorm9 = nn.BatchNorm2d(1024)
        self.relu9 = nn.ReLU()
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(1024 * 4 * 4, encoded_space_dim),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.enc_conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        if debug: print(f"After conv1 {x.shape}")
        x = self.enc_conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        if debug: print(f"After conv2 {x.shape}")
        x = self.enc_conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        if debug: print(f"After conv3 {x.shape}")
        x = self.enc_conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        if debug: print(f"After conv4 {x.shape}")
        x = self.enc_conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        if debug: print(f"After conv5 {x.shape}")
        x = self.enc_conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)
        if debug: print(f"After conv6 {x.shape}")
        x = self.enc_conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)
        if debug: print(f"After conv7 {x.shape}")
        x = self.enc_conv8(x)
        x = self.batchnorm8(x)
        x = self.relu8(x)
        if debug: print(f"After conv8 {x.shape}")
        #x = self.block7(x)
        #if debug: print(f"After block7 {x.shape}")
        x = self.enc_conv9(x)
        x = self.batchnorm9(x)
        x = self.relu9(x)
        if debug: print(f"After conv9 {x.shape}")

        x = self.flatten(x)
        if debug: print(f"After flatten {x.shape}")
        x = self.encoder_lin(x)
        if debug: print(f"After encoder_linear {x.shape}")
        return x


class C_Decoder_224(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1024 * 4 * 4),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 4, 4))
        self.dec_convt1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=1, output_padding=1)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()
        self.dec_convt2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.dec_convt3 = nn.ConvTranspose2d(256, 256, 2, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.dec_convt4 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        #self.upblock3 = ResUpConvBlock(256, 128, stride=2)
        self.dec_convt5 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.dec_convt6 = nn.ConvTranspose2d(128, 64, 2, stride=1)
        self.batchnorm6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()
        self.dec_convt7 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=1)
        self.batchnorm7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU()
        self.dec_convt8 = nn.ConvTranspose2d(64, 64, 2, stride=1)
        self.batchnorm8 = nn.BatchNorm2d(64)
        self.relu8 = nn.ReLU()
        self.dec_convt9 = nn.ConvTranspose2d(64, 3, 2, stride=2, padding=1)
        

    def forward(self, x):
        if debug: print("DECODER")
        if debug: print(f"Start {x.shape}")
        x = self.decoder_lin(x)
        if debug: print(f"After decoder_lin {x.shape}")
        x = self.unflatten(x)
        if debug: print(f"After unflatten {x.shape}")
        x = self.dec_convt1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        if debug: print(f"After transposed conv 1 {x.shape}")
        x = self.dec_convt2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        if debug: print(f"After transposed conv 2 {x.shape}")
        x = self.dec_convt3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        if debug: print(f"After transposed conv 3 {x.shape}")
        x = self.dec_convt4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        #x = self.upblock3(x)
        if debug: print(f"After transposed conv 4 {x.shape}")
        x = self.dec_convt5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        if debug: print(f"After transposed conv 5 {x.shape}")
        x = self.dec_convt6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)
        if debug: print(f"After transposed conv 6 {x.shape}")
        x = self.dec_convt7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)
        if debug: print(f"After transposed conv 7 {x.shape}")
        x = self.dec_convt8(x)
        x = self.batchnorm8(x)
        x = self.relu8(x)
        if debug: print(f"After transposed conv 8 {x.shape}")
        x = self.dec_convt9(x)
        if debug: print(f"After transposed conv 9 {x.shape}")
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
        x = x.reshape(x.size(0), 3, 224, 224) # reshape this flatten vector to the original image size    
        return x



class Res_Encoder_224(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.enc_conv1 = nn.Conv2d(3, 64, (7,7), stride=1, padding=3, bias=False) # 64 * 224 * 224
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.block2 = ResConvBlock(64, 64, stride=2) # 64 * 112 * 112
        self.block3 = ResConvBlock(64, 128, stride=2) # 128 * 56 * 56
        self.block4 = ResConvBlock(128, 128, stride=2) # 128 * 28 * 28
        self.block5 = ResConvBlock(128, 256, stride=2) # 256 * 14 * 14
        self.block6 = ResConvBlock(256, 512, stride=2) # 512 * 7 * 7
        self.block7 = ResConvBlock(512, 1024, stride=2) # 1024 * 4 * 4

        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(1024 * 4 * 4, encoded_space_dim),
            nn.ReLU(True)
        )

        
    def forward(self, x):
        x = self.enc_conv1(x)
        if debug: print(f"After enc conv 1 {x.shape}")
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.block2(x)
        if debug: print(f"After block2 {x.shape}")
        x = self.block3(x)
        if debug: print(f"After block3 {x.shape}")
        x = self.block4(x)
        if debug: print(f"After block4 {x.shape}")
        x = self.block5(x)
        if debug: print(f"After block5 {x.shape}")
        x = self.block6(x)
        if debug: print(f"After block6 {x.shape}")
        x = self.block7(x)
        if debug: print(f"After block7 {x.shape}")
        x = self.flatten(x)
        if debug: print(f"After flatten {x.shape}")
        x = self.encoder_lin(x)
        if debug: print(f"After encoder_linear {x.shape}")
        return x


class Res_Decoder_224(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1024 * 4 * 4),
            nn.ReLU(True)
        )
        
        ### Convolutional section
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 4, 4)) # 
        self.dec_convt1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=1, output_padding=1)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()
        self.block2 = ResUpConvBlock(512, 256, stride=2) # 254 * 14 * 14
        self.block3 = ResUpConvBlock(256, 128, stride=2) # 128 * 28 * 28
        self.block4 = ResUpConvBlock(128, 128, stride=2) # 128 * 56 * 56
        self.block5 = ResUpConvBlock(128, 64, stride=2) # 64 * 112 * 112
        self.block6 = ResUpConvBlock(64, 64, stride=2) # 64 * 224 * 224
        self.conv7 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=1)
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        if debug: print(f"After unflatten {x.shape}")
        x = self.dec_convt1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        if debug: print(f"After dec_convt1 {x.shape}")
        x = self.block2(x)
        if debug: print(f"After block2 {x.shape}")
        x = self.block3(x)
        if debug: print(f"After block3 {x.shape}")
        x = self.block4(x)
        if debug: print(f"After block4 {x.shape}")
        x = self.block5(x)
        if debug: print(f"After block5 {x.shape}")
        x = self.block6(x)
        if debug: print(f"After block6 {x.shape}")
        x = self.conv7(x)
        if debug: print(f"After dec_convt 7 {x.shape}")
        x = torch.sigmoid(x)
        return x


class Res_Autoencoder_224(nn.Module):
    def __init__(self, input_size = 224*224, encoding_dim = 1024):
        super().__init__()
        self.input_size = input_size
        self.encoder = Res_Encoder_224(encoding_dim)
        self.decoder = Res_Decoder_224(encoding_dim)

    def forward(self, x):
        x = self.encoder(x) # here we get the latent z
        x = self.decoder(x) # here we get the reconsturcted input
        x = x.reshape(-1, 3, 224, 224) # reshape this flatten vector to the original image size    
        return x


    
def train(model, optimizer, trainloader = None, valloader = None, num_epochs = 1, savefile=None):
    
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
    
    train_losses=[]
    val_losses=[]
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        show_example = True
        
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
                    elif (show_example):
                        show_example=False
                        fig,axes = plt.subplots(1,2); plt.set_cmap(['gray','viridis'][0]);
                        axes[0].imshow(x[0].cpu().numpy().transpose((1, 2, 0)));
                        axes[1].imshow(outputs[0].detach().cpu().numpy().transpose((1, 2, 0)))
                        plt.show()
                        
                
                # stats
                running_loss += loss.item() * x.size(0)
                count += len(x)
            
            epoch_loss = running_loss / count
            print(f'{phase} loss {epoch_loss:.6f}')
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
        print()

        if (savefile is not None) and (epoch % 10 == 0) and (epoch > 0):
            torch.save(model.state_dict(), f"saved_models/{savefile}.sav")
            pd.DataFrame(data=np.array([train_losses, val_losses]).T, 
                columns = ["train", "val"]).to_csv(f"saved_models/{savefile}.csv", index=False)
    
    return train_losses, val_losses