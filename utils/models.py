import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values
        

class WarpLayer(nn.Module):
    def __init__(self, B=4):
        super(WarpLayer, self).__init__()
        self.B = B
        self.parms = nn.Parameter(torch.randn(self.B, 2), requires_grad=True)
         
    def forward(self, x, y, t, b, W, H):  
        
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)          
        
        for _ in range(20):
            optim.zero_grad()
            
            for i in range(self.B):
                tmp_x = x[b == i] - t[b == i] * self.parms[i][0] 
                tmp_y = y[b == i] - t[b == i] * self.parms[i][1]       

                if i == 0:
                    loss = tmp_x.var() + tmp_y.var()
                else:
                    loss += tmp_x.var() + tmp_y.var()

            loss.backward()
            optim.step()
        
        for i in range(self.B):
            x[b == i] -= t[b == i] * self.parms[i][0]
            y[b == i] -= t[b == i] * self.parms[i][1]
        
        zero = torch.zeros_like(x)
        x = torch.where(x<0,zero,x)
        x = torch.where(x>=239,zero,x)
        y = torch.where(y<0,zero,y)
        y = torch.where(y>=179,zero,y)

        return x.detach(), y.detach()


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item())
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        p = (p+1)/2  # maps polarity to 0, 1

        # warp (x,y)
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()  # normalizing timestamps
            length = len(t[events[:,-1] == bi])

            for i_bin in range(C):
                start, end = int(i_bin / C * length), int((i_bin+1) / C * length)
                x_b, y_b, p_b, b_b, t_b = x[events[:,-1] == bi][start:end], y[events[:,-1] == bi][start:end], p[events[:,-1] == bi][start:end], b[events[:,-1] == bi][start:end], t[events[:,-1] == bi][start:end]

                with torch.enable_grad():
                    device = torch.device("cuda:4")
                    warp_layer = WarpLayer(B=1).to(device)
                    x_b, y_b = warp_layer.forward(x_b.clone().detach(), y_b.clone().detach(), t_b.clone().detach(), b_b.clone().detach(), W, H)
                    
                idx = x_b \
                    + W * y_b \
                    + W * H * i_bin \
                    + W * H * C * p_b \
                    + W * H * C * 2 * b_b

                values = t_b * self.value_layer.forward(t_b)
                vox.put_(idx.long(), values, accumulate=True)

        # draw in voxel grid
        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        self.classifier = resnet34(pretrained=pretrained)

        self.crop_dimension = crop_dimension

        # replace fc layer and first convolutional layer
        input_channels = 2*voxel_dimension[0]
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=output_resolution)

        return x

    def forward(self, x):
        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier.forward(vox_cropped)
        return pred, vox