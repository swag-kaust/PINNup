import torch
from torch.autograd import grad
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config as cf
nx = cf.nx
nz = cf.nz

def adjust_learning_rate(optimizer,initial_lr,epoch,adjust_interval):
    lr = initial_lr*(0.5**(epoch//adjust_interval)) + initial_lr*0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def  gradient(y,x,grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad_ = grad(y,[x],grad_outputs=grad_outputs,create_graph=True)[0]
    return grad_

def divergence(y,x):
    div = 0
    for i in range(y.shape[-1]):
        div += grad(y[...,i],x,torch.ones_like(y[...,i]),create_graph=True)[0][...,i:i+1]
    return div

def laplace(y,x):
    grad_ = gradient(y,x)
    return divergence(grad_,x)

def outloss(epoch,num_epochs,batch_idx,batch_len,loss,elapsed):
    sys.stdout.write('\r')
    sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tElapsed: %.4f\tLoss: %.4f'
                                        %(epoch, num_epochs, batch_idx+1, batch_len, elapsed, loss.item()))
    sys.stdout.flush()

def plot_results(du_real_pred,du_imag_pred,du_real_star, du_imag_star, epoch,source_number,freq):
    # Error
    error_du_real = np.linalg.norm(du_real_star-du_real_pred,2)/np.linalg.norm(du_real_star,2)
    error_du_imag = np.linalg.norm(du_imag_star-du_imag_pred,2)/np.linalg.norm(du_imag_star,2)

    print('Error u_real: %e, Error u_imag: %e' % (error_du_real,error_du_imag))
    scipy.io.savemat('du_real_pred_atan-{}.mat'.format(cf.fre),{'du_real_pred':du_real_pred})
    scipy.io.savemat('du_imag_pred_atan-{}.mat'.format(cf.fre),{'du_imag_pred':du_imag_pred})

    scipy.io.savemat('du_real_star-{}.mat'.format(cf.fre),{'du_real_star':du_real_star})
    scipy.io.savemat('du_imag_star-{}.mat'.format(cf.fre),{'du_imag_star':du_imag_star})

    ## plot the real parts of the scattered wavefield for i th source
    #source_number = 8 ## 1-9
    a = (source_number-1)*nx*nz
    b = (source_number)*nx*nz
    du_real_star_is = du_imag_star[a:b:1]
    du_real_pred_is = du_imag_pred[a:b:1]
    du_real_star_is2D = np.reshape(np.array(du_real_star_is), (nx, nz))
    du_real_pred_is2D = np.reshape(np.array(du_real_pred_is), (nx, nz))
    du_real_dif2D = du_real_star_is2D - du_real_pred_is2D

    error_du_imag = np.linalg.norm(du_real_star_is-du_real_pred_is,2)/np.linalg.norm(du_real_star_is,2)

    print('Error for shot 4 u_imag: %e' % (error_du_imag))

    plt.figure(figsize=(20,60))
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    im = ax.imshow(du_real_star_is2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_star_is2D.T, extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Depth (km)', fontsize=14)
    plt.title('Numerical solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 2)
    ax = plt.gca()
    im = ax.imshow(du_real_pred_is2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_pred_is2D.T,extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('PINN solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 3)
    ax = plt.gca()
    im = ax.imshow(du_real_dif2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz, 0], aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('Difference')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude')
    #plt.show()
    plt.savefig(cf.model_name+'Epoch'+str(epoch)+'-'+str(freq)+'result-imag.png')

    du_real_star_is = du_real_star[a:b:1]
    du_real_pred_is = du_real_pred[a:b:1]
    du_real_star_is2D = np.reshape(np.array(du_real_star_is), (nx, nz))
    du_real_pred_is2D = np.reshape(np.array(du_real_pred_is), (nx, nz))
    du_real_dif2D = du_real_star_is2D - du_real_pred_is2D

    error_du_imag = np.linalg.norm(du_real_star_is-du_real_pred_is,2)/np.linalg.norm(du_real_star_is,2)

    print('Error for shot 4 u_real: %e' % (error_du_imag))

    plt.figure(figsize=(20,60))
    plt.subplot(3, 1, 1)
    ax = plt.gca()
    im = ax.imshow(du_real_star_is2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_star_is2D.T, extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.ylabel('Depth (km)', fontsize=14)
    plt.title('Numerical solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 2)
    ax = plt.gca()
    im = ax.imshow(du_real_pred_is2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz,0],aspect=1, cmap="jet")
    #im = ax.imshow(du_real_pred_is2D.T,extent=[0, 2.5, 2.5,0],aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('PINN solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    plt.subplot(3, 1, 3)
    ax = plt.gca()
    im = ax.imshow(du_real_dif2D.T, vmin=cf.vmin,vmax=cf.vmax,extent=[0, cf.axisx, cf.axisz, 0], aspect=1, cmap="jet")
    plt.xlabel('Distance (km)', fontsize=14)
    plt.title('Difference')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.25)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude')
    #plt.show()
    plt.savefig(cf.model_name+'Epoch'+str(epoch)+'-'+str(freq)+'result-real.png')

# Positional encoding
class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):

    if i == -1:
        return torch.nn.identity, cf.embed_dim

    embed_kwargs = {
        'include_input': True,
        'input_dims': cf.embed_dim,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def _l2_normalize(d):
    #d_reshaped = d.view(d.shape[0],-1,*(1 for _ in range(d.dim() - 2)))
    #d /= torch.norm(d_reshaped,dim=1,keepdim=True) + 1e-8
    d /= torch.norm(d,keepdim=True) + 1e-8
    return d.cuda()

def neural_split(model_new, pre_model_, split_num):
    model = model_new.state_dict()
    pre_model = pre_model_.state_dict()
    weight_len = len(model)
    layer_idx = 0
    random_factor = 0.0
    for model_key, pre_model_key in zip(model.keys(), pre_model.keys()):
        if layer_idx<2:
            # first layer should be not divide within the values
            if layer_idx == 0:
                #print(torch.norm(pre_model[pre_model_key]))
                model[model_key] = pre_model[pre_model_key].repeat(split_num,1) + random_factor*_l2_normalize(torch.rand(model[model_key].shape).sub(0.5))
            else:
                model[model_key] = pre_model[pre_model_key].repeat(split_num) + random_factor*_l2_normalize(torch.rand(model[model_key].shape).sub(0.5))
            layer_idx = layer_idx + 1
        elif layer_idx > (weight_len-3):
            if layer_idx == (weight_len-2):
                model[model_key] = pre_model[pre_model_key].repeat(1,split_num) * 1.0/split_num + random_factor*_l2_normalize(torch.rand(model[model_key].shape).sub(0.5))
            else:
                model[model_key] = pre_model[pre_model_key] + random_factor*_l2_normalize(torch.rand(model[model_key].shape).sub(0.5))
            layer_idx = layer_idx + 1
        else:
            if layer_idx % 2 == 0:
                model[model_key] = pre_model[pre_model_key].repeat(split_num, split_num) * 1.0/split_num+ random_factor*_l2_normalize(torch.rand(model[model_key].shape).sub(0.5))
            else:
                model[model_key] = pre_model[pre_model_key].repeat(split_num)+ random_factor*_l2_normalize(torch.rand(model[model_key].shape).sub(0.5))
            layer_idx = layer_idx + 1
    model_new.load_state_dict(model)
    print('model splited by the pretrained model with {} times'.format(split_num))
