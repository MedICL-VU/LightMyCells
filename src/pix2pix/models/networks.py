import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision.models as models
from segmentation_models_pytorch import UnetPlusPlus


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())  # Conv1_2
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())  # Conv2_2
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval()) # Conv3_3
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())# Conv4_3
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).cuda()
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)).cuda()
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)).cuda()
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        for block in self.blocks:
            input = block(input)
            target = block(target)
            loss += F.l1_loss(input, target)
        return loss


def sobelLayer(input):
    pad = nn.ReflectionPad2d(1)
    kernel = create2DSobelFilter()
    act = nn.Tanh()
    padded = pad(input)
    fake_sobel = F.conv2d(padded, kernel, padding=0, groups=1) / 4
    n, c, h, w = fake_sobel.size()
    fake = torch.norm(fake_sobel, p=2, dim=1, keepdim=True) / c * 3
    fake_out = act(fake) * 2 - 1
    return fake_out


def create2DSobelFilter():
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]])
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]])
    sobelFilter = torch.zeros((2, 1, 3, 3))
    sobelFilter[0, 0, :, :] = sobel_x
    sobelFilter[1, 0, :, :] = sobel_y
    return sobelFilter.type(torch.FloatTensor).cuda()


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'res_dyn_multi_decoder':
        net = ResDynMultiDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=12)
    elif netG == 'res_multi_decoder':
        net = ResMultiDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=12)
    elif netG == 'unet_plusplus':
        net = UnetPlusPlus(
            encoder_name='resnext101_32x16d', encoder_depth=5, encoder_weights='ssl', 
            decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, 
            in_channels=input_nc, classes=output_nc, activation='tanh', aux_params=None)
    elif netG == 'dyn_unet_plusplus':
        net = UnetPlusPlus(
            encoder_name='resnext101_32x16d', encoder_depth=5, encoder_weights='ssl', 
            decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None, 
            in_channels=input_nc, classes=output_nc, activation='tanh', aux_params=None)
        net = DynamicNetwork(net, code_length=3)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


class DynamicNetwork(nn.Module):
    def __init__(self, network, code_length):
        super(DynamicNetwork, self).__init__()
        self.encoder = DynamicEncoder(network.encoder, code_length=3)
        self.decoder = network.decoder
        self.segmentation_head = network.segmentation_head

    def forward(self, x, code):        
        features = self.encoder(x, code)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks
        

class DynamicEncoder(nn.Module):
    def __init__(self, encoder, code_length):
        super(DynamicEncoder, self).__init__()
        self.encoder = encoder
        self.encoder.conv1 = DynamicScalingConv3d(layer=self.encoder.conv1, code_length=3)

    def forward(self, x, code):
        stages = self.encoder.get_stages()
        features = []
        for i in range(self.encoder._depth + 1):
            if i == 1:
                x = stages[i][0](x, code)
                x = stages[i][1](x)
                x = stages[i][2](x)
            else:
                x = stages[i](x)
            features.append(x)
        return features


class DynamicLayer(nn.Module):
    def __init__(self):
        super(DynamicLayer, self).__init__()

    def forward(self, x, code):
        pass


class DynamicScalingConv3d(DynamicLayer):
    """This is a weight-only version"""
    def __init__(self, layer, code_length):
        super(DynamicLayer, self).__init__()
        layer.bias = torch.nn.Parameter(torch.zeros(layer.out_channels))
        self.layer = layer
        self.controller = nn.Conv2d(
            in_channels=code_length, 
            out_channels=self.layer.in_channels*self.layer.out_channels + self.layer.out_channels, 
            kernel_size=1, stride=1, padding=0, bias=True)

    def generate_dyn_param(self, code):
        N = code.shape[0]
        params = self.controller(code[..., None, None].float())  # B x in*out+out x 1 x 1
        params.squeeze_(-1).squeeze_(-1)  # B x in*out+out
        weight_nums, bias_nums = [self.layer.in_channels*self.layer.out_channels], [self.layer.out_channels]
        params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))
        sw, sb = params_splits[0], params_splits[1]  # N x in*out, N x out
        sw = sw.reshape(N, self.layer.out_channels, self.layer.in_channels, 1, 1)
        sb = sb.reshape(N, self.layer.out_channels)  
        return sw, sb

    def forward(self, x, code):
        N, _, H, W = x.size()
        sw, sb = self.generate_dyn_param(code)  # sw: N x out x in x 1 x 1; sb: N x out
        weight = self.layer.weight.clone()[None, ...]  # weight: 1 x out x in x k1 x k2
        bias = self.layer.bias.clone()  # bias: out
        k1, k2 = weight.shape[-2], weight.shape[-1]
        weight = weight * sw
        bias = bias * sb 
        x = x.reshape(1, -1, H, W)
        weight = weight.reshape(N * self.layer.out_channels, -1, k1, k2)
        bias = bias.reshape(-1)
        output = F.conv2d(
            input=x, weight=weight, bias=bias, stride=self.layer.stride, 
            padding=self.layer.padding, dilation=self.layer.dilation, groups=N)
        output = output.reshape(N, self.layer.out_channels, output.shape[-2], output.shape[-1])
        return output


# class DynamicScalingConv3d(DynamicLayer):
#     def __init__(self, layer, code_length):
#         super(DynamicLayer, self).__init__()
#         self.layer = layer
#         self.layer.bias = True
#         self.controller = nn.Conv2d(
#             in_channels=code_length, 
#             out_channels=self.layer.in_channels*self.layer.out_channels + self.layer.out_channels, 
#             kernel_size=1, stride=1, padding=0, bias=True)
#         if torch.cuda.is_available():
#             self.controller = self.controller.cuda()  

#     def generate_dyn_param(self, code):
#         N = code.shape[0]
#         params = self.controller(code[..., None, None].float())  # B x in*out+out x 1 x 1
#         params.squeeze_(-1).squeeze_(-1)  # B x in*out+out
#         weight_nums, bias_nums = [self.layer.in_channels*self.layer.out_channels], [self.layer.out_channels]
#         params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))
#         sw, sb = params_splits[0], params_splits[1]  # N x in*out, N x out
#         sw = sw.reshape(N, self.layer.out_channels, self.layer.in_channels, 1, 1)
#         sb = sb.reshape(N, self.layer.out_channels)  
#         return sw, sb

#     def forward(self, x, code):
#         N, _, H, W = x.size()
#         sw, sb = self.generate_dyn_param(code)  # sw: N x out x in x 1 x 1; sb: N x out
#         weight = self.layer.weight.clone()[None, ...]  # weight: 1 x out x in x k1 x k2
#         bias = self.layer.bias.clone()  # bias: out
#         k1, k2 = weight.shape[-2], weight.shape[-1]
#         weight = weight * sw
#         bias = bias * sb 
#         x = x.reshape(1, -1, H, W)
#         weight = weight.reshape(N * self.layer.out_channels, -1, k1, k2)
#         bias = bias.reshape(-1)
#         output = F.conv2d(
#             input=x, weight=weight, bias=bias, stride=self.layer.stride, 
#             padding=self.layer.padding, dilation=self.layer.dilation, groups=N)
#         output = output.reshape(N, self.layer.out_channels, output.shape[-2], output.shape[-1])
#         return output


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResMultiDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResMultiDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)]

        # add downsampling layers
        n_downsampling = 2
        for i in range(n_downsampling):  
            mult = 2 ** i
            encoder += [
                nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf*mult*2),
                nn.ReLU(True)]

        mult = 2 ** n_downsampling

        # residual blocks --> bottleneck
        for i in range(n_blocks):       
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.encoder = nn.Sequential(*encoder)
        # output channels: 'Mitochondria', 'Nucleus', 'Tubulin', 'Actin'
        self.decoder1 = Decoder(ngf, n_downsampling, output_nc, norm_layer, use_bias)
        self.decoder2 = Decoder(ngf, n_downsampling, output_nc, norm_layer, use_bias)
        self.decoder3 = Decoder(ngf, n_downsampling, output_nc, norm_layer, use_bias)
        self.decoder4 = Decoder(ngf, n_downsampling, output_nc, norm_layer, use_bias)

    def forward(self, x, is_train=True):
        x = self.encoder(x)
        mit = self.decoder1(x)  
        nuc = self.decoder2(x)  
        tub = self.decoder3(x)  
        act = self.decoder4(x)  
        if is_train:
            return mit, nuc, tub, act
        else:
            return torch.cat((mit, nuc, tub, act), 1)


class ResDynMultiDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResDynMultiDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            # norm_layer(ngf),
            DynamicIN(ngf, code_length=3),
            nn.ReLU(True)]

        # add downsampling layers
        n_downsampling = 2
        for i in range(n_downsampling):  
            mult = 2 ** i
            encoder += [
                nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                # norm_layer(ngf*mult*2),
                DynamicIN(int(ngf*mult*2), code_length=3),
                nn.ReLU(True)]

            # anti-alising implementation
            # encoder += [
            #     nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            #     # norm_layer(ngf*mult*2),
            #     DynamicIN(int(ngf*mult*2), code_length=3),
            #     nn.ReLU(True),
            #     Downsample(ngf*mult*2)]

        mult = 2 ** n_downsampling

        # residual blocks --> bottleneck
        for i in range(n_blocks):       
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.encoder = nn.Sequential(*encoder)
        # output channels: 'Mitochondria', 'Nucleus', 'Tubulin', 'Actin'
        self.decoder1 = Decoder(ngf, n_downsampling, output_nc, norm_layer, use_bias)
        self.decoder2 = Decoder(ngf, n_downsampling, output_nc, norm_layer, use_bias)
        self.decoder3 = Decoder(ngf, n_downsampling, output_nc, norm_layer, use_bias)
        self.decoder4 = Decoder(ngf, n_downsampling, output_nc, norm_layer, use_bias)

    def forward(self, x, code, is_train=True):
        for layer in self.encoder:
            if isinstance(layer, DynamicLayer):
                x = layer(x, code)
            else:
                x = layer(x)

        mit = self.decoder1(x)  
        nuc = self.decoder2(x)  
        tub = self.decoder3(x)  
        act = self.decoder4(x)  
        if is_train:
            return mit, nuc, tub, act
        else:
            return torch.cat((mit, nuc, tub, act), 1)


class Decoder(nn.Module):
    def __init__(self, ngf, n_downsampling, output_nc, norm_layer, use_bias):
        super(Decoder, self).__init__()
        decoder = []
        for i in range(n_downsampling): 
            mult = 2 ** (n_downsampling - i)
            decoder += [
                nn.ConvTranspose2d(
                    ngf*mult, int(ngf * mult / 2),
                    kernel_size=3, stride=2,
                    padding=1, output_padding=1,
                    bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]

            # anti-alising implementation
            # decoder += [
            #     Upsample(ngf * mult),
            #     nn.Conv2d(ngf * mult, int(ngf * mult / 2),
            #             kernel_size=3, stride=1, padding=1, bias=use_bias),
            #     norm_layer(int(ngf * mult / 2)),
            #     nn.ReLU(True)]

        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, feat):
        return self.decoder(feat)


class DynamicIN(DynamicLayer):
    def __init__(self, num_features, code_length):
        super(DynamicLayer, self).__init__()
        self.num_features = num_features
        self.controller = nn.Conv2d(in_channels=code_length, out_channels=num_features*2, 
            kernel_size=1, stride=1, padding=0, bias=True)  

    def generate_dyn_param(self, code):
        params = self.controller(code[..., None, None].float())  
        params.squeeze_(-1).squeeze_(-1)
        return params[:, :self.num_features], params[:, self.num_features:]

    def forward(self, x, code):
        weight, bias = self.generate_dyn_param(code)
        outputs = []
        for i in range(x.shape[0]):
            outputs.append(F.instance_norm(input=x[i:i+1, ...], weight=weight[i], bias=bias[i]))
        output = torch.cat(outputs, dim=0)
        return output


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
