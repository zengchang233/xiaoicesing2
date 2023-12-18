import torch
import torch.nn as nn
import numpy as np
import logging

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

    def forward(self, input):
        return input * torch.sigmoid(input)

class DiscriminatorFactory(nn.Module):
    def __init__(self,
                 time_length,
                 freq_length,
                 conv_channel,
                 ):
        super(DiscriminatorFactory, self).__init__()

        layers = 10
        conv_channels = conv_channel
        kernel_size = 3
        conv_in_channels = 60
        use_weight_norm = True

        self.conv_layers = torch.nn.ModuleList()
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = 1
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                nn.Conv1d(conv_in_channels, conv_channels,
                       kernel_size=kernel_size, padding=padding,
                       dilation=dilation, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.BatchNorm1d(conv_channels)
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        last_conv_layer = nn.Conv1d(
            conv_in_channels, 1,
            kernel_size=kernel_size, padding=padding, bias=True)
        self.conv_layers += [last_conv_layer]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"weight norm is applied to {m}.")
        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)

    def forward(self, x):
        """
        Args:
            x: (B, C, T), by default, C = 40.

        Returns:
            tensor: (B, 1, T)
        """
        feature_list = []
        i = 1
        for f in self.conv_layers:
            x = f(x)
            if i % 2 == 1:
                feature_list.append(x)
            i += 1
        return [x, feature_list]


class MultiWindowDiscriminator(nn.Module):
    """docstring for MultiWindowDiscriminator"""
    def __init__(self,
                 time_lengths,
                 freq_lengths,
                 conv_channels,
        ):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths

        self.conv_layers = nn.ModuleList()
        self.patch_layers = nn.ModuleList()
        for time_length, freq_length, conv_channel in zip(time_lengths, freq_lengths, conv_channels):
            conv_layer = [
                DiscriminatorFactory(np.abs(time_length),  freq_length, conv_channel),
            ] # 1d 
            self.conv_layers += conv_layer
            patch_layer = [PatchGAN()]  
            self.patch_layers += patch_layer


    def clip(self, x, x_len, win_length, y=None, random_N=None):
        '''Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, T, C).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, win_length, C).

        '''
        x_batch = []
        y_batch = []
        T_end = win_length
        if T_end > 0:
            cursor = 1
        else:
            cursor = -1
        min_a = min(x_len)
        if np.abs(T_end) + random_N > min_a:
            T_end = min_a - random_N - 1
            T_end = T_end * cursor
        #print(x_len, random_N, win_length, T_end)
        for i in range(x.size(0)):
            if T_end < 0:
                x_batch += [x[i, x_len[i].cpu() + T_end - random_N: x_len[i].cpu() - random_N, :].unsqueeze(0)]
            else:
                x_batch += [x[i, random_N : T_end + random_N, :].unsqueeze(0)]
            if y != None:
                if T_end < 0:
                    y_batch += [y[i, x_len[i].cpu() + T_end - random_N: x_len[i].cpu() - random_N, :].unsqueeze(0)]
                else:
                    y_batch += [y[i, random_N : T_end+ random_N, :].unsqueeze(0)]

        x_batch = torch.cat(x_batch, 0)
        if y != None:
            y_batch = torch.cat(y_batch, 0)
        if y != None:
            return x_batch, y_batch
        else:
            return x_batch
       
    def forward(self, x, x_len, y=None, random_N=None):
        '''
        Args:
            x (tensor): input mel, (B, T, C).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        '''
        validity_x = list()
        validity_y = list()
        #validity = 0.0
        for i in range(len(self.conv_layers)):
            if y != None:
                if self.win_lengths[i] != 1:
                    x_clip,y_clip = self.clip(x, x_len, self.win_lengths[i], y, random_N[i])  # (B, win_length, C)
                else:
                    #print(x.shape, y.shape)
                    #x_clip, y_clip = x[:,:1300,:], y[:,:1300,:]
                    x_clip, y_clip = x, y
                y_clip = y_clip.transpose(2,1)

            else:
                if self.win_lengths[i] != 1:
                    x_clip = self.clip(x, x_len, self.win_lengths[i], y, random_N[i])  # (B, win_length, C)
                else:
                    #print(x.shape)
                    #x_clip = x[:,:1300, :]
                    x_clip = x

            x_clip = x_clip.transpose(2, 1)                    # (B, C, win_length)
            x_clip_r = self.conv_layers[i](x_clip) # 1d 
            validity_x += [x_clip_r]
            x_clip_r = self.patch_layers[i](x_clip) # 2d
            validity_x += [x_clip_r]
            if y!= None:
                y_clip_r = self.conv_layers[i](y_clip)
                validity_y += [y_clip_r]
                y_clip_r = self.patch_layers[i](y_clip)
                validity_y += [y_clip_r]

            #validity += x_clip
        if y == None:
            return validity_x
        else:
            return validity_x, validity_y

class PatchGAN(nn.Module): # 
    def __init__(self):
        super(PatchGAN, self).__init__()

        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                  out_channels=32,
                                                  kernel_size=(3, 3),
                                                  stride=(1, 1),
                                                  padding=(1, 1)),
                                        GLU())

        # DownSample Layer
        self.downSample1 = self.downSample(in_channels=32,
                                           out_channels=64,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=1)

        self.downSample2 = self.downSample(in_channels=64,
                                           out_channels=128,
                                           kernel_size=(3, 3),
                                           stride=[2, 2],
                                           padding=1)
         # Conv Layer
        self.outputConvLayer_2 = nn.Sequential(nn.Conv2d(in_channels=128,
                                                        out_channels=1,
                                                        kernel_size=(1, 3),
                                                        stride=[1, 1],
                                                        padding=[0, 1])
                                              )

        self.downSample3 = self.downSample(in_channels=128,
                                           out_channels=256,
                                           kernel_size=[3, 3],
                                           stride=[2, 2],
                                           padding=1)
         # Conv Layer
        self.outputConvLayer_3 = nn.Sequential(nn.Conv2d(in_channels=256,
                                                       out_channels=1,
                                                       kernel_size=(1, 3),
                                                       stride=[1, 1],
                                                       padding=[0, 1]))

        self.downSample4 = self.downSample(in_channels=256,
                                           out_channels=512,
                                           kernel_size=[3, 3],
                                           stride=[2, 2],
                                           padding=1)
         # Conv Layer
        self.outputConvLayer_4 = nn.Sequential(nn.Conv2d(in_channels=512,
                                                       out_channels=1,
                                                       kernel_size=(1, 3),
                                                       stride=[1, 1],
                                                       padding=[0, 1]))
        self.downSample5 = self.downSample(in_channels=512,
                                           out_channels=1024,
                                           kernel_size=[3, 3],
                                           stride=[2, 2],
                                           padding=1)


        # Conv Layer
        self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=1024,
                                                       out_channels=1,
                                                       kernel_size=(1, 3),
                                                       stride=[1, 1],
                                                       padding=[0, 1]))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                                  nn.InstanceNorm2d(num_features=out_channels,
                                                    affine=True),
                                  GLU())
        return convLayer

    def forward(self, input):
        # input has shape [batch_size, num_features, time]
        # discriminator requires shape [batchSize, 1, num_features, time]
        input = input.unsqueeze(1)
        #print("input : {}".format(input.shape))
        feature_list = []
        conv_layer_1 = self.convLayer1(input)
        feature_list.append(conv_layer_1)
        #print("conv_layer_1: {}".format(conv_layer_1.shape))

        downsample1 = self.downSample1(conv_layer_1)
        feature_list.append(downsample1)
        #output_1 = torch.sigmoid(self.outputConvLayer_1(downsample1))
        #print("downsample1 {} output_1 {}".format(downsample1.shape, output_1.shape))
        downsample2 = self.downSample2(downsample1)
        feature_list.append(downsample2)
        output_2 = torch.sigmoid(self.outputConvLayer_2(downsample2))
        #print("downsample2 {} output_2 {}".format(downsample2.shape, output_2.shape))
        downsample3 = self.downSample3(downsample2)
        feature_list.append(downsample3)
        output_3 = torch.sigmoid(self.outputConvLayer_3(downsample3))
        #print("downsample3 {} output_3 {}".format(downsample3.shape, output_3.shape))
        downsample4 = self.downSample4(downsample3)
        feature_list.append(downsample4)
        output_4 = torch.sigmoid(self.outputConvLayer_4(downsample4))
        #print("downsample4 {} output_4 {}".format(downsample4.shape, output_4.shape))
        downsample5 = self.downSample5(downsample4)
        feature_list.append(downsample5)
        #print("downsample5 {}".format(downsample5.shape))

        output = torch.sigmoid(self.outputConvLayer(downsample5))
        #print("output {} ".format(output.shape))
        output = output.view(output.shape[0], output.shape[1], -1)
        output_4 = output_4.view(output.shape[0], output.shape[1], -1)
        output_3 = output_3.view(output.shape[0], output.shape[1], -1)
        output_2 = output_2.view(output.shape[0], output.shape[1], -1)
        #output_1 = output_1.view(output.shape[0], output.shape[1], -1)
        output = torch.cat((output,output_4,output_3, output_2), axis=2)
        #output = output.view(output.shape[0], output.shape[1], -1)
        return [output, feature_list]

class MultibandFrequencyDiscriminator(nn.Module):
    def __init__(self,
                 time_lengths=[200, 400, 600, 800, 1],
                 freq_lengths=[ 60,  60,  60, 60, 60],
                 multi_channels=[[87, 87, 87, 87, 87 ], [87, 87, 87, 87, 87], [87,87, 87, 87, 87]]
                 ):
        super(MultibandFrequencyDiscriminator, self).__init__()

        self.time_lengths = time_lengths
        self.multi_win_discriminator_low = MultiWindowDiscriminator(time_lengths, freq_lengths, multi_channels[0])
        self.multi_win_discriminator_middle = MultiWindowDiscriminator(time_lengths, freq_lengths, multi_channels[1])
        self.multi_win_discriminator_high = MultiWindowDiscriminator(time_lengths, freq_lengths, multi_channels[2])

    def forward(self, x, x_len, y=None, random_N=[]):
        '''
        Args:
            x (tensor): input mel, (B, T, C).
            x_length (tensor): len of per mel. (B,).

        Returns:
            list : [(B), (B,), (B,)].
        '''
        if len(random_N) == 0:
            len_min = min(x_len.cpu())
            time_max = max(self.time_lengths)
            start = 0
            end = len_min - time_max
            if end <= 0:
                end = int(len_min / 2)
            random_N = np.random.randint(start, end , len(self.time_lengths))

        #print(x_len)
        base_mel = x[:,:,:120]
        xa = base_mel[:,:,:60]
        xb = base_mel[:,:,30:90]
        xc = base_mel[:,:,60:120]
        if y != None:
            y_mel = y[:, :, :120]
            ya = y_mel[:,:,:60]
            yb = y_mel[:,:, 30:90]
            yc = y_mel[:,:,60:120]
        else:
            ya = yb = yc = None


        x_list = [
            self.multi_win_discriminator_low(xa, x_len, ya, random_N),
            self.multi_win_discriminator_middle(xb, x_len,yb, random_N),
            self.multi_win_discriminator_high(xc, x_len, yc, random_N),
        ]
        return x_list, random_N

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = MultibandFrequencyDiscriminator()

    def forward(self, x, x_len, y=None, random_N=[]):
        return self.discriminator(x, x_len, y, random_N)


if __name__ == "__main__":
    inputs = torch.randn(4, 1200, 120).cuda()
    tgt = torch.randn(4, 1200, 120).cuda()
    inputs_len = torch.tensor([1200]).cuda()
    net = Discriminator().cuda()
    print(net)
    outputs, random_n = net(inputs, inputs_len, tgt)
    #  import pdb; pdb.set_trace()
    for output in outputs:
        for a in output[0]:
            for aa in a:
                for aaa in aa:
                    print(aaa.shape)
        for b in output[1]:
            for bb in b:
                for bbb in bb:
                    print(bbb.shape)
        #  print(output[0].shape)
        #  print(output[1].shape)
