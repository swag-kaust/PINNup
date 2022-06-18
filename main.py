import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from utils import *
from pinnmodel import *
from TransferData2Dataset import DataProcessing
from train import pinn_train
from predict import pinn_predict_loader, pinn_predict
import config as cf
from thop import profile

np.random.seed(1234)
torch.manual_seed(1234)

if __name__ == '__main__':
    # Parameters setting
    model_name = cf.model_name
    fre = cf.fre # Hz
    PI = cf.PI
    niter = cf.niter

    layers = cf.layers

    check = cf.check
    split = cf.split
    split_only = cf.split_only
    check_point = cf.check_point
    if(cf.train_flag):
        print('Experiments: {}'.format(model_name))
        print('Training for {}Hz'.format(fre))
        print(layers)
        misfit = []
        model_train = PhysicsInformedNN(layers)

        #input_temp = torch.randn(4000,28)
        #flops, params = profile(model_train,inputs=(input_temp,))
        #print(params)
        #exit()
        model_train.apply(weight_init)
        if check_point:
            print('Keep training from checkpoint: {}'.format(check_point))
            if split_only:
                layers_o = list((np.array(layers) / split).astype(np.int))
                layers_o[0] = layers[0]
                layers_o[-1] = layers[-1]
                model_train_pre = PhysicsInformedNN(layers_o)
                if cf.multi_card:
                    model_train_pre.load_state_dict({k.replace('module.',''):v for k,v in torch.load('pinnmodel-{}-{}-{}.pth'.format(model_name,check_point,cf.pre_train_fre))['net'].items()})
                else:
                    model_train_pre.load_state_dict(torch.load('pinnmodel-{}-{}-{}.pth'.format(model_name,check_point,cf.pre_train_fre))['net'])
                neural_split(model_train.cuda(),model_train_pre.cuda(),split)
            else:
                if cf.multi_card:
                    model_train.load_state_dict({k.replace('module.',''):v for k,v in torch.load('pinnmodel-{}-{}-{}.pth'.format(model_name,check_point,cf.pre_train_fre))['net'].items()})
                else:
                    model_train.load_state_dict(torch.load('pinnmodel-{}-{}-{}.pth'.format(model_name,check_point,cf.pre_train_fre))['net'])

        model_train.cuda()
        if cf.multi_card:
            model_train = torch.nn.DataParallel(model_train, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        #load data
        lb = cf.lb
        ub = cf.ub
        trainset = DataProcessing('./','data/{}_{}Hz_train_data.mat'.format(model_name,fre),'train',ub,lb)
        lr = cf.lr
        if cf.adam:
            pinn_train(model=model_train,train_loader=trainset,lr=lr,use_lbfgs=False,epochs=niter,misfit=misfit)
        if check:
            state = {
                'net':model_train.state_dict(),
                'lb':lb,
                'ub':ub,
            }
            torch.save(state,'pinnmodel-{}-{}-{}.pth'.format(model_name, niter+check_point,fre))
        if cf.lbfgs:
            pinn_train(model=model_train,train_loader=trainset,lr=lr,use_lbfgs=True,epochs=1,misfit=misfit)
        #save model
        state = {
            'net':model_train.state_dict(),
            'lb':lb,
            'ub':ub,
        }
        torch.save(state,'pinnmodel-{}-{}-{}.pth'.format(model_name,niter+check_point,fre))
        plt.figure(figsize=(5,5))
        plt.plot(misfit,'-')
        plt.title('LOSS')
        plt.yscale('log')
        plt.savefig('LOSS-{}-{}.png'.format(model_name,fre),dpi=300)
        np.savetxt('./misfit-{}-{}.txt'.format(model_name,fre), np.array(misfit),fmt='%f',delimiter=' ')

    # Prediction
    torch.cuda.empty_cache()
    lb, ub = cf.lb, cf.ub
    model_pred = PhysicsInformedNN(layers)
    if cf.multi_card:
        model_pred.load_state_dict({k.replace('module.',''):v for k,v in torch.load('pinnmodel-{}-{}-{}.pth'.format(model_name,niter+check_point,fre))['net'].items()})
    else:
        model_pred.load_state_dict(torch.load('pinnmodel-{}-{}-{}.pth'.format(model_name,niter+check_point,fre))['net'])
    model_pred.cuda()
    if cf.multi_card:
        model_pred = torch.nn.DataParallel(model_pred, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    testset = DataProcessing('./','data/{}_{}Hz_testdata.mat'.format(model_name,fre),'test',ub,lb,freq_star=fre)
    testloader = torch.utils.data.DataLoader(testset,batch_size=cf.nx,shuffle=False,num_workers=0)
    #du_real_pred, du_imag_pred, du_real_star, du_imag_star = pinn_predict_loader(model_pred,testloader)
    du_real_pred, du_imag_pred, du_real_star, du_imag_star = pinn_predict(model_pred,testset)
    plot_results(du_real_pred,du_imag_pred,du_real_star,du_imag_star,niter,1,fre)