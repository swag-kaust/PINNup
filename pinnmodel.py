import torch
import torch.nn as nn
import torch.nn.functional as F

class Basicblock(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(Basicblock, self).__init__()
        self.layer1 = nn.Linear(in_planes,out_planes)

    def forward(self, x):
        out = torch.sin(self.layer1(x))
        return out

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight,gain=1)
        nn.init.constant_(m.bias, 0)
    if classname.find('Conv')!=-1:
        init.xavier_normal_(m.weight,gain=1)
        init.constant_(m.bias,0)

class PhysicsInformedNN(nn.Module):
    def __init__(self, layers):
        super(PhysicsInformedNN, self).__init__()
        self.layers = layers
        self.in_planes = self.layers[0]
        self.layer1 = self._make_layer(Basicblock,self.layers[1:len(layers)-1])
        self.linear = nn.Linear(layers[-2],layers[-1])

    def _make_layer(self, block, layers,):
        layers_net = []
        for layer in layers:
            layers_net.append(block(self.in_planes,layer))
            self.in_planes = layer
        return nn.Sequential(*layers_net)

    def forward(self,x):
        out = self.layer1(x)
        out = self.linear(out)
        return out[:,0:1], out[:,1:2]

class PhysicsInformedNNv(nn.Module):
    def __init__(self, layers):
        super(PhysicsInformedNNv, self).__init__()
        self.layers = layers
        self.in_planes = self.layers[0]
        self.layer1 = self._make_layer(Basicblock,self.layers[1:len(layers)-1])
        self.linear = nn.Linear(layers[-2],layers[-1])

    def _make_layer(self, block, layers,):
        layers_net = []
        for layer in layers:
            layers_net.append(block(self.in_planes,layer))
            self.in_planes = layer
        return nn.Sequential(*layers_net)

    def forward(self,x):
        out = self.layer1(x)
        out = (torch.sin(self.linear(out))+1.0)/2.0*(1.0/2.5**2-1.0/6.5**2)+1.0/6.5**2
        return out