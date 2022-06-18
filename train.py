import time
import torch
from utils import *
from torch.optim import Adam, LBFGS, RMSprop, SGD, lr_scheduler
from TransferData2Dataset import data_prefetcher, DataProcessing
import config as cf
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from predict import plot, pinn_eval

PI = cf.PI
ub = torch.Tensor(cf.ub).cuda()
lb = torch.Tensor(cf.lb).cuda()
fre = cf.fre
emb = cf.emb
batch_len = cf.batch_len
model_name = cf.model_name

def pinn_train(model,train_loader,lr,use_lbfgs,epochs,misfit):
    writer = SummaryWriter(comment='PINNup-Adam')
    start_time = time.time()
    model.train()
    optimizer = Adam(model.parameters(),lr=lr,weight_decay=5e-4)
    if use_lbfgs:
        # copy settings from Raissi et al. (2019) and here
        optimizer = LBFGS(model.parameters(),max_iter=50000,max_eval=50000,history_size=50,line_search_fn='strong_wolfe')

    ###positional encoding
    embedding_fn, input_cha = get_embedder(emb,0)
    ##############################################
    scheduler = lr_scheduler.StepLR(optimizer,cf.damping_step,gamma=0.6,last_epoch=-1)

    ### preload
    for epoch in range(epochs):
        input_data = torch.Tensor(train_loader.data[:,:])
        randperm = np.random.permutation(len(input_data))
        batch_size = int(len(input_data)/batch_len)
        for batch_idx in range(batch_len):
            start_,end_ = batch_idx*batch_size,(batch_idx+1)*batch_size
            randperm_idx = randperm[start_:end_]
            x_train, y_train, sx_train, u0_real_train, u0_imag_train, m_train, m0_train = input_data[randperm_idx,0:1].cuda(),input_data[randperm_idx,1:2].cuda(),input_data[randperm_idx,2:3].cuda(),input_data[randperm_idx,3:4].cuda(),input_data[randperm_idx,4:5].cuda(),input_data[randperm_idx,5:6].cuda(),input_data[randperm_idx,6:7].cuda()
            if use_lbfgs:
                def closure():
                    x = x_train.clone().detach().requires_grad_(True)
                    y = y_train.clone().detach().requires_grad_(True)
                    sx = sx_train.clone().detach().requires_grad_(True)
                    f = fre
                    omega = 2.0 * PI * f
                    x_input = torch.cat((x,y,sx),1)
                    x_input[:,0:3] = 2.0 *(x_input[:,0:3]-lb)/(ub - lb) - 1.0
                    x_input = embedding_fn(x_input[:,0:3]) # positional encoding
                    optimizer.zero_grad()
                    du_real_pred, du_imag_pred = model(x_input)

                    du_real_xx = laplace(du_real_pred,x)
                    du_imag_xx = laplace(du_imag_pred,x)
                    du_real_yy = laplace(du_real_pred,y)
                    du_imag_yy = laplace(du_imag_pred,y)

                    f_real_pred = omega*omega*m_train*du_real_pred + du_real_xx + du_real_yy + omega*omega*(m_train-m0_train)*u0_real_train
                    f_imag_pred = omega*omega*m_train*du_imag_pred + du_imag_xx + du_imag_yy + omega*omega*(m_train-m0_train)*u0_imag_train
                    loss_value = torch.sum(torch.pow(f_imag_pred,2)) + torch.sum(torch.pow(f_imag_pred,2))
                    #######
                    loss_value.backward()
                    misfit.append(loss_value.item())
                    elapsed = 0
                    outloss(epoch+1,epochs,batch_idx,batch_len,loss_value, elapsed)
                    return loss_value
                optimizer.step(closure)
            else:
                x = x_train.clone().detach().requires_grad_(True)
                y = y_train.clone().detach().requires_grad_(True)
                sx = sx_train.clone().detach().requires_grad_(True)
                f = fre
                #f = input_data[randperm_idx,7:8].cuda() 
                omega = 2.0 * PI * f
                x_input = torch.cat((x,y,sx),1)
                x_input[:,0:3] = 2.0 *(x_input[:,0:3]-lb)/(ub - lb) - 1.0
                x_input = embedding_fn(x_input[:,0:3]) # positional encoding
                optimizer.zero_grad()
                du_real_pred, du_imag_pred = model(x_input)

                du_real_xx = laplace(du_real_pred,x)
                du_imag_xx = laplace(du_imag_pred,x)
                du_real_yy = laplace(du_real_pred,y)
                du_imag_yy = laplace(du_imag_pred,y)

                f_real_pred = omega*omega*m_train*du_real_pred + du_real_xx + du_real_yy + omega*omega*(m_train-m0_train)*u0_real_train
                f_imag_pred = omega*omega*m_train*du_imag_pred + du_imag_xx + du_imag_yy + omega*omega*(m_train-m0_train)*u0_imag_train
                  
                loss_value = (torch.sum(torch.pow(f_real_pred,2)) + torch.sum(torch.pow(f_imag_pred,2))) 
                #######
                loss_value.backward()
                optimizer.step()
                misfit.append(loss_value.item())
                elapsed = time.time() - start_time
                start_time = time.time()
                outloss(epoch+1,epochs,batch_idx,batch_len,loss_value, elapsed)
                writer.add_scalar('Loss/total',loss_value,epoch)
        if (epoch % 500 == 0):
            du_real_eval, du_imag_eval = pinn_eval(model,cf.nx,cf.nz)
            writer.add_figure('Wavefield/real',plot(du_real_eval),epoch)
            writer.add_figure('Wavefield/imag',plot(du_imag_eval),epoch)
            
        if ((epoch+1)%cf.saving_step==0):
            print('saving')
            state = {
                'net':model.state_dict(),
                'lb':lb.cpu().numpy(),
                'ub':ub.cpu().numpy(),
            }
            torch.save(state,'pinnmodel-{}-epoch{}-{}.pth'.format(model_name,epoch+1,fre))
        scheduler.step()