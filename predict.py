import torch
from utils import *
import numpy as np
import config as cf
emb = cf.emb
fre = cf.fre
from pinnmodel import *
from torch.optim import Adam

def pinn_eval(model,nx,nz):
    x = 2.0 * (np.arange(nx).reshape(nx,1).repeat(nz,axis=0).reshape(nx*nz,1)*0.025 - 0.0) / (nx*0.025 - 0.0) - 1.0
    y = 2.0 * (np.arange(nz).reshape(nz,1).repeat(nx,axis=1).T.reshape(nx*nz,1)*0.025 - 0.0) / (nz*0.025 - 0.0) - 1.0
    sx = x.copy() * 0.0 
    embedding_fn, input_cha = get_embedder(emb,0)
    x_input = embedding_fn(torch.cat(((torch.Tensor(x)),(torch.Tensor(y)),torch.Tensor(sx)),1))
    du_real_eval, du_imag_eval = model(x_input.cuda())
    return du_real_eval.detach().cpu().numpy().reshape(nx,nz), du_imag_eval.detach().cpu().numpy().reshape(nx,nz)
   
def plot(d):
    fig = plt.figure(figsize=(5,5))
    plt.imshow(d.T,extent=(0,2.5,2.5,0.0),cmap='jet')
    plt.colorbar()
    return fig

def pinn_predict_loader(model, testloader):
    model.eval()
    embedding_fn, input_cha = get_embedder(emb,0)
    du_real_pred, du_imag_pred, du_real_star, du_imag_star = [],[],[],[]
    ############################################
    with torch.no_grad():
        for batch_idx, x_input_all in enumerate(testloader):
            x_input = embedding_fn(x_input_all[:,0:3])
            du_real_pred_temp, du_imag_pred_temp = model(x_input.cuda())
            du_real_pred.append(du_real_pred_temp.cpu().numpy())
            du_imag_pred.append(du_imag_pred_temp.cpu().numpy())
            du_real_star.append((x_input_all[:,3:4]).numpy())
            du_imag_star.append((x_input_all[:,4:5]).numpy())
    return np.array(du_real_pred).reshape([-1,1]), np.array(du_imag_pred).reshape([-1,1]), np.array(du_real_star).reshape([-1,1]), np.array(du_imag_star).reshape([-1,1])

def pinn_predict(model, testset):
    model.eval()
    #positional encoding
    embedding_fn, input_cha = get_embedder(emb,0)
    ############################################
    x_input = torch.Tensor(testset.data[:,0:3])
    x_input = embedding_fn(x_input)
    with torch.no_grad():
        du_real_pred, du_imag_pred = model(x_input.cuda())

    return du_real_pred, du_imag_pred, testset.data[:,3:4], testset.data[:,4:5]
