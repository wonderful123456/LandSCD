import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax, Conv1d
from torchvision.models import resnet
import torch
from torchvision import models
from torch import nn

from functools import partial


nonlinearity = partial(F.relu, inplace=True)

def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)

def leakrelu_feature_map(x):
    return F.leaky_relu(x)

def elu_feature_map(x):
    return F.elu(x-2)+1

def elus_feature_map(x, param = 10):
    x1 = F.relu(x)
    x2 = x - x1
    x2 = torch.exp(param*x2) - 1
    return param*x1 + x2 + 1

def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)  # inplace=True
    )


class PAM_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.leakrelu_feature_map = elus_feature_map
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
        
#         self.last_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.leakrelu_feature_map(Q).permute(-3, -1, -2)
        K = self.leakrelu_feature_map(K)

        KV = torch.einsum("bmn, bcn->bmc", K, V)

        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        # weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
#         weight_value = self.last_conv(weight_value)
        # weight_value = self.nl(weight_value)
        return (x + self.gamma * weight_value).contiguous()



class LmsPAM_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-10):
        super(LmsPAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.feature_map = elus_feature_map
        self.eps = eps
        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv1 = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv2 = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv3 = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)    
        self.last_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x, y, z):
        # Apply the feature map to the queries and keys, y ,z 是x的上下文特征图
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv1(x).view(batch_size, -1, width * height)
        y_K = self.key_conv2(y).view(batch_size, -1, width * height)
        z_K = self.key_conv3(z).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)  
        Q = self.feature_map(Q).permute(-3, -1, -2)
        K = self.feature_map(K)
        y_K = self.feature_map(y_K)
        z_K = self.feature_map(z_K)
        KV = torch.einsum("bmn, bcn->bmc", K, V)
        y_KV = torch.einsum("bmn, bcn->bmc", y_K, V)
        z_KV = torch.einsum("bmn, bcn->bmc", z_K, V)
        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)
        y_norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(y_K, dim=-1) + self.eps)
        z_norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(z_K, dim=-1) + self.eps)
        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value1 = torch.einsum("bnm, bmc, bn->bcn", Q, y_KV, y_norm)
        weight_value2 = torch.einsum("bnm, bmc, bn->bcn", Q, z_KV, z_norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        weight_value1 = weight_value1.view(batch_size, chnnels, height, width)
        weight_value2 = weight_value2.view(batch_size, chnnels, height, width)
        weight_value = weight_value + weight_value1 + weight_value2
        attention = self.last_conv(weight_value)
        return (x + self.gamma * attention).contiguous()
    


class LmsPAM_Context_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-10):
        super(LmsPAM_Context_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.feature_map = elus_feature_map
        self.eps = eps
        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv1 = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv2 = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
        self.last_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x, y):
        # Apply the feature map to the queries and keys, y ,z 是x的上下文特征图
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv1(x).view(batch_size, -1, width * height)
        y_K = self.key_conv2(y).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Q = self.feature_map(Q).permute(-3, -1, -2)
        K = self.feature_map(K)
        y_K = self.feature_map(y_K)
        KV = torch.einsum("bmn, bcn->bmc", K, V)
        y_KV = torch.einsum("bmn, bcn->bmc", y_K, V)
        norm =  1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1)+ self.eps)
        y_norm =  1 / torch.einsum("bnc, bc->bn", Q, torch.sum(y_K, dim=-1)+ self.eps)
#         norm = 1 / torch.where(norm == 0. , torch.full_like(norm, self.eps), norm)
#         y_norm = 1 / torch.where(y_norm == 0. , torch.full_like(y_norm, self.eps), y_norm)
        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value1 = torch.einsum("bnm, bmc, bn->bcn", Q, y_KV, y_norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        weight_value1 = weight_value1.view(batch_size, chnnels, height, width)
        weight_value = weight_value + weight_value1
        attention = self.last_conv(weight_value)
        return (x + self.gamma * attention).contiguous()
    
class MsPAM_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(MsPAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv1 = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv2 = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv3 = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
        self.last_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x, y, z):
        # Apply the feature map to the queries and keys, y ,z 是x的上下文特征图
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv1(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        
        y_K = self.key_conv2(y).view(batch_size, -1, width * height)
        z_K = self.key_conv3(z).view(batch_size, -1, width * height)

        QK = torch.einsum("bcn, bcm->bnm", Q, K)
        Qy_K = torch.einsum("bcn, bcm->bnm", Q, y_K)
        Qz_K = torch.einsum("bcn, bcm->bnm", Q, z_K)
        # QK = QK + Qy_K + Qz_K
        attention1 = self.softmax(QK)
        attention2 = self.softmax(Qy_K)
        attention3 = self.softmax(Qz_K)
        
        # attention = attention1 + attention2 + attention3
        attention1 = torch.einsum("bmn, bcm->bcn", attention1, V).view(batch_size, -1, width, height)
        attention2 = torch.einsum("bmn, bcm->bcn", attention2, V).view(batch_size, -1, width, height)
        attention3 = torch.einsum("bmn, bcm->bcn", attention3, V).view(batch_size, -1, width, height)
        
        attention = (attention1 + attention2 + attention3)
        # attention = torch.cat((attention1, attention2, attention3), dim=1)
        # attention = self.last_conv(attention)
        # attention = attention1 + attention2 + attention3
        attention = self.last_conv(attention)
        # attention = self.nl(attention)
        return (x + self.gamma * attention).contiguous()


class CAM_Module(Module):
    def __init__(self, in_places):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
#         self.last_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
#         self.nl = nn.GroupNorm(4, in_places)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key) #矩阵乘法
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

#         out = self.last_conv(out)
        # out = self.nl(out)
        out = self.gamma * out + x
        return out

class MsCAM_Module(Module):
    def __init__(self, in_places):
        super(MsCAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.last_conv1 = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
        self.nl = nn.GroupNorm(4, in_places)

    def forward(self, x, y , z):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        proj_key1 = y.view(batch_size, chnnels, -1).permute(0, 2, 1)
        proj_key2 = z.view(batch_size, chnnels, -1).permute(0, 2, 1)
        
        energy = torch.bmm(proj_query, proj_key) #矩阵乘法
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        
        energy1 = torch.bmm(proj_query, proj_key1) #矩阵乘法
        energy_new1 = torch.max(energy1, -1, keepdim=True)[0].expand_as(energy1) - energy1
        
        energy2 = torch.bmm(proj_query, proj_key2) #矩阵乘法
        energy_new2 = torch.max(energy2, -1, keepdim=True)[0].expand_as(energy2) - energy2
        attention1 = self.softmax(energy_new)
        attention2 = self.softmax(energy_new1)
        attention3 = self.softmax(energy_new2)
        
        # energy_new = energy_new + energy_new1 + energy_new2
        # attention1 = self.softmax(energy_new)
        # attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)
        # attention = attention1 + attention2 + attention3
        out1 = torch.bmm(attention1, proj_value).view(batch_size, chnnels, height, width)
        out2 = torch.bmm(attention2, proj_value).view(batch_size, chnnels, height, width)
        out3 = torch.bmm(attention3, proj_value).view(batch_size, chnnels, height, width)
        # out = torch.cat((out1, out2, out3), dim=1)
        # out = self.last_conv1(out)
        
        out = (out1+ out2+ out3) 
        out = self.last_conv1(out)
        # out = self.nl(out)
        out = self.gamma * out + x
        return out

class LmsCAM_Module(Module):
    def __init__(self, in_places):
        super(LmsCAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.last_conv1 = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x, y , z):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        proj_key1 = y.view(batch_size, chnnels, -1).permute(0, 2, 1)
        proj_key2 = z.view(batch_size, chnnels, -1).permute(0, 2, 1)
        
        energy = torch.bmm(proj_query, proj_key) #矩阵乘法
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        
        energy1 = torch.bmm(proj_query, proj_key1) #矩阵乘法
        energy_new1 = torch.max(energy1, -1, keepdim=True)[0].expand_as(energy1) - energy1
        
        energy2 = torch.bmm(proj_query, proj_key2) #矩阵乘法
        energy_new2 = torch.max(energy2, -1, keepdim=True)[0].expand_as(energy2) - energy2
        attention1 = self.softmax(energy_new)
        attention2 = self.softmax(energy_new1)
        attention3 = self.softmax(energy_new2)
        
        # energy_new = energy_new + energy_new1 + energy_new2
        # attention1 = self.softmax(energy_new)
        # attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)
        # attention = attention1 + attention2 + attention3
        out1 = torch.bmm(attention1, proj_value).view(batch_size, chnnels, height, width)
        out2 = torch.bmm(attention2, proj_value).view(batch_size, chnnels, height, width)
        out3 = torch.bmm(attention3, proj_value).view(batch_size, chnnels, height, width)
        # out = torch.cat((out1, out2, out3), dim=1)
        # out = self.last_conv1(out)
        
        out = (out1+ out2+ out3) 
        out = self.last_conv1(out)
        out = self.gamma * out + x
        return out
    

    
class LmsCAM_Context_Module(Module):
    def __init__(self, in_places):
        super(LmsCAM_Context_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.last_conv1 = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
        self.nl = nn.GroupNorm(4, in_places)

    def forward(self, x, y):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        proj_key1 = y.view(batch_size, chnnels, -1).permute(0, 2, 1)
        
        
        energy = torch.bmm(proj_query, proj_key) #矩阵乘法
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        
        energy1 = torch.bmm(proj_query, proj_key1) #矩阵乘法
        energy_new1 = torch.max(energy1, -1, keepdim=True)[0].expand_as(energy1) - energy1
        
        
        attention1 = self.softmax(energy_new)
        attention2 = self.softmax(energy_new1)
        
        # energy_new = energy_new + energy_new1 + energy_new2
        # attention1 = self.softmax(energy_new)
        # attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)
        # attention = attention1 + attention2 + attention3
        out1 = torch.bmm(attention1, proj_value).view(batch_size, chnnels, height, width)
        out2 = torch.bmm(attention2, proj_value).view(batch_size, chnnels, height, width)
        
        # out = torch.cat((out1, out2, out3), dim=1)
        # out = self.last_conv1(out)  
        out = (out1+ out2) 
        out = self.last_conv1(out)
        # out = self.nl(out)
        out = self.gamma * out + x
        return out
    
class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module(in_ch)

    def forward(self, x):
        return self.PAM(x) + self.CAM(x)
 

class LmsPAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(LmsPAM_CAM_Layer, self).__init__()
        self.PAM = LmsPAM_Module(in_ch)
        self.CAM = LmsCAM_Module(in_ch)
        
    def forward(self, x, y , z):
        return self.PAM(x, y , z) + self.CAM(x, y, z)
    
        
class MsPAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(MsPAM_CAM_Layer, self).__init__()
        self.PAM = MsPAM_Module(in_ch)
        self.CAM = MsCAM_Module(in_ch)
#         self.conv = self.conv1 = nn.Conv2d(in_ch, 256, 1, bias=False)

    def forward(self, x, y , z):
        return self.PAM(x, y , z) + self.CAM(x, y, z)


def buildAttention(in_ch):
    return PAM_CAM_Layer(in_ch=in_ch)

def buildMsAttention(in_ch):
    return MsPAM_CAM_Layer(in_ch=in_ch)

def buildLmsAttention(in_ch):
    return LmsPAM_CAM_Layer(in_ch=in_ch)

if __name__ == '__main__':
    in_batch, inchannel, in_h, in_w = 4, 512, 32, 32
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    y = torch.randn(in_batch, inchannel, in_h, in_w)
    z = torch.randn(in_batch, inchannel, in_h, in_w)
    
    label = torch.randn(in_batch, inchannel, in_h, in_w)
    model = PAM_CAM_Layer(512)
    loss = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(),lr = 0.001, momentum=0.9)
    for epoch in range(100):
        opt.zero_grad()
        res = model(x,y,z)
        # res = fn(g(res))
        ls = loss(res,label)
        ls.backward()
        opt.step()
        print(loss.item())
