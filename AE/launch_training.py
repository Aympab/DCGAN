import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
import torch.nn.functional as F
import os

from dataload import *
from autoencoder import *


def try_model(model, bs_tr, bs_te, resc, crop, lr, n_epochs, opt, wd,
             file_sav):
    #train_loader, test_loader = define_landscapes_loaders(bs_tr, bs_te, rgb=True,
    #                                                      rescale=resc, crop=crop)
    train_loader, test_loader = define_lhq_loaders(bs_tr, bs_te, rgb=True,
                                                          rescale=resc, crop=crop)
    
    opti = opt(model.parameters(), lr = lr, weight_decay=wd) # 
    model.apply(init_weights)
    print(f"Began training : epochs = {n_epochs}, lr = {lr}")
    t_losses, v_losses = train(model, opti, trainloader=train_loader, valloader=test_loader, num_epochs=n_epochs, savefile=file_sav)
    
    torch.save(model.state_dict(), f"saved_models/{file_sav}.sav")
    pd.DataFrame(data=np.array([t_losses, v_losses]).T, columns = ["train", "val"]).to_csv(f"saved_models/{file_sav}.csv", index=False)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Launching on {device}")
    
    
    autoencoder_1 = C_Autoencoder_224(224*224, 2048)
    try_model(autoencoder_1, 8, 8, 256, 224, 0.00005, 30, torch.optim.Adam,
              1e-6, "model_c-autoenc224_lhq_adam_n30")
    #autoencoder_2 = C_Autoencoder_224(224*224, 2048)
    #try_model(autoencoder_2, 4, 4, 256, 224, 0.0001, 40, torch.optim.Adadelta,
    #          1e-5, "model_c-autoenc224_adadelta_n150")
    
