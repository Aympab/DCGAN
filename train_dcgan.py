from cProfile import label
import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from dataload import *
from gan_architecture import *
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import get_n_params, accuracy, write_params
import argparse 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data set parameters
ds = "data/berry"
run_test = False
bs = 256
crop_size=128

train_loader, test_loader = define_loaders(bs_train=bs, bs_test=bs, 
                            rescale=256,
                            crop=crop_size,
                            test_set=run_test,
                            dataset=ds)


#Architecture information, only to be printed in params file
archi_info = "upsamble type : nearest"

#Optimizer parameters
lrG = 0.00001
lrD = 0.00005
beta1 = 0.5
weight_decay = 0.0001

#Input of generator
z_dim = 512

#Training parameters
savefile = 'res-gan'
n_epoch = 5000
save_frequency = 100
k = 2 #Facteur d'apprentissage discriminateur
n_generated_save = 4 #number of images to output at each save_frequency epochs

"""if --midsave args is passed is activated, save the
evolution models every n_midsave epochs"""
n_midsave = 20

#Labels for discriminator for fake and real images
label_reals = 0.9 
label_fakes = 0.05
labels = torch.full((bs, 1), label_reals, dtype=torch.float, device=device)


# G = DCGenerator(z_dim).to(device)
# D = DCDiscriminator().to(device)
G = Generator(z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss() 

G_optimizer = optim.Adam(G.parameters(), lr = lrG, betas=(beta1, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr = lrD, betas=(beta1, 0.999))


def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.view(-1, 3, crop_size , crop_size).to(device)
    size = len(x_real)

    D_output = D(x_real)

    if size != bs: #to avoid filling labels array that is size bs
        y_real = torch.ones(size, 1).to(device)
        y_real.fill_(label_reals)
        D_real_acc = accuracy(D_output, y_real)
        D_real_loss = criterion(D_output, y_real)
    else:
        labels.fill_(label_reals)
        D_real_acc = accuracy(D_output, labels)
        D_real_loss = criterion(D_output, labels)

    D_real_loss.backward()

    # train discriminator on fake
    z = torch.randn(size, z_dim).to(device) 
    with torch.no_grad():
        x_fake = G(z)
    labels.fill_(label_fakes)

    D_output = D(x_fake)
    if size != bs:
        y_fake = torch.zeros(size, 1).to(device)
        y_fake.fill_(label_fakes)
        D_fake_loss = criterion(D_output, y_fake)
        D_fake_acc = accuracy(D_output, y_fake)
    else:
        D_fake_loss = criterion(D_output, labels)
        D_fake_acc = accuracy(D_output, labels)

    D_fake_loss.backward()
    # gradient backprop & optimize ONLY D's parameters
    full_loss = D_real_loss + D_fake_loss
    #full_loss.backward()

    #regularization
    if(weight_decay > 0) :
        apply_weight_decay(*D.modules(), weight_decay_factor=weight_decay, wo_bn=True)
    
    D_optimizer.step()

    return full_loss.data.item(), ((D_real_acc + D_fake_acc) / 2)

def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(bs, z_dim).to(device)
    labels.fill_(label_reals)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, labels)
    G_acc = accuracy(D_output, labels)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()

    if(weight_decay > 0) :
        apply_weight_decay(*G.modules(), weight_decay_factor=weight_decay, wo_bn=True)
    
    G_optimizer.step()
    
    return G_loss.data.item(), G_acc


if __name__ == "__main__":
    write_params(savefile, archi_info, lrG, lrD, beta1, weight_decay, z_dim,
                n_epoch, save_frequency, k, label_fakes, label_reals,
                ds, run_test, bs, crop_size)

    parser = argparse.ArgumentParser(description='Process options')
    parser.add_argument('--resume', action="store_true",
                        help='Wether or not to continue training')
    parser.add_argument('--midsave', action="store_true",
                        help='Save model evolution (every 20 epochs by default)')

    args = parser.parse_args()

    D_losses, G_losses = [0], [0]
    D_accs, G_accs = [0], [0]
    D_test_accs = [0] 

    if args.resume: #loading pretrained model
        G.load_state_dict(torch.load(f"saved_models/{savefile}_generator.sav"))
        D.load_state_dict(torch.load(f"saved_models/{savefile}_discriminator.sav"))
        print("Resuming training from saved states")
        losses = pd.read_csv(f"saved_models/{savefile}_losses.csv")
        acc = pd.read_csv(f"saved_models/{savefile}_accs.csv")

        if(run_test):
            testacc = pd.read_csv(f"saved_models/{savefile}_testaccs.csv")
            D_test_accs = testacc["discriminator"].values.tolist()
        
        D_losses = losses["discriminator"].values.tolist()
        G_losses = losses["generator"].values.tolist()
        D_accs = acc["discriminator"].values.tolist()
        G_accs = acc["generator"].values.tolist()

    print(f'Launching for {n_epoch} epochs...\nSave frequency = {save_frequency}')
    print(f"Number of parameters : D : {get_n_params(D)}, G : {get_n_params(G)}")

    for epoch in range(1, n_epoch+1):

        for batch_idx, (x) in enumerate(tqdm(train_loader)):
            D_current_loss, D_current_acc = D_train(x)

            if batch_idx % k == 0:
                G_current_loss, G_current_acc = G_train(x)
            else:
                G_current_loss = G_losses[-1]
                G_current_acc = G_accs[-1]


            if ((batch_idx % 100 == 0) & (batch_idx > 0)) | (batch_idx == len(train_loader)-1):
                D_losses.append(D_current_loss)
                G_losses.append(G_current_loss)
                D_accs.append(D_current_acc)
                G_accs.append(G_current_acc)


        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                (epoch), n_epoch, np.mean(D_losses[-10:]), np.mean(G_losses[-10:])))

        with torch.no_grad():
            test_z = torch.randn(n_generated_save, z_dim).to(device)
            generated = G(test_z)

            if(epoch % save_frequency == 0):
                save_image(generated.view(generated.size(0), 3, crop_size, crop_size), './generated_batchs4200/generated_batch' + str(epoch) + '.png')
            
            if(run_test):
                D_test_acc = 0
                batches_accs = []
                for tbatch_idx, (x) in enumerate(tqdm(test_loader)):
                    x = x.view(-1, 3, crop_size , crop_size).to(device)
                    size = len(x)

                    D_output = D(x)

                    if size != bs:
                        y_real = torch.ones(size, 1).to(device)
                        y_real.fill_(label_reals)
                        batches_accs.append(accuracy(D_output, y_real))
                    else:
                        labels.fill_(label_reals)
                        batches_accs.append(accuracy(D_output, labels))

                D_test_accs.append(np.mean(batches_accs))
                print('[%d/%d]: test_acc: %.3f' % (
                    (epoch), n_epoch, np.mean(batches_accs)))
        
        if (savefile is not None) and (epoch % save_frequency == 0) and (epoch > 0):
            torch.save(G.state_dict(), f"saved_models/{savefile}_generator.sav")
            torch.save(D.state_dict(), f"saved_models/{savefile}_discriminator.sav")
            pd.DataFrame(data=np.array([D_losses, G_losses]).T, 
                columns = ["discriminator", "generator"]).to_csv(f"saved_models/{savefile}_losses.csv", index=False)
            pd.DataFrame(data=np.array([D_accs, G_accs]).T, 
                columns = ["discriminator", "generator"]).to_csv(f"saved_models/{savefile}_accs.csv", index=False)
            if(run_test):
                pd.DataFrame(data=np.array(D_test_accs).T, 
                    columns = ["discriminator"]).to_csv(f"saved_models/{savefile}_testaccs.csv", index=False)

        if (savefile is not None) and (epoch % n_midsave == 0) and (epoch > 0) and (args.midsave):
            torch.save(G.state_dict(), f"saved_models/{savefile}_generator_epoch{epoch}.sav")
            torch.save(D.state_dict(), f"saved_models/{savefile}_discriminator_epoch{epoch}.sav")


