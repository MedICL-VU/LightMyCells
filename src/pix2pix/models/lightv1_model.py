import torch
from .base_model import BaseModel
from . import networks


class Lightv1Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='res_multi_decoder', dataset_mode='light')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_C', 'real_C', 'fake_D', 'real_D', 'fake_E', 'real_E']
        self.output_names = ['Mitochondria', 'Nucleus', 'Tubulin', 'Actin']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  
            self.model_names = ['G']
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # network architectures
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # loss functions and optimizers
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.input_code = input['input_code'].to(self.device)
        self.output_code = input['output_code'].to(self.device)
        self.real_A = input['input'].to(self.device)
        self.real_B = input['Mitochondria'].to(self.device) 
        self.real_C = input['Nucleus'].to(self.device) 
        self.real_D = input['Tubulin'].to(self.device) 
        self.real_E = input['Actin'].to(self.device) 
        self.real_tgt = [self.real_B, self.real_C, self.real_D, self.real_E]
        
    def forward(self):
        self.fake_B, self.fake_C, self.fake_D, self.fake_E = self.netG(self.real_A, self.input_code)  
        self.fake_tgt = [self.fake_B, self.fake_C, self.fake_D, self.fake_E]

    def backward_D(self): 
        self.loss_D_fake, self.loss_D_real = 0, 0
        for i in range(len(self.output_names)):
            idx = torch.arange(self.real_A.size()[0]).cuda()[self.output_code[:, i]==1]
            # at least one sample in the batch has ground truth
            if len(idx) > 0:  
                cur_tgt = self.real_tgt[i][idx, ...]
                cur_pred = self.fake_tgt[i][idx, ...]
                cur_input = self.real_A[idx, ...]  
                # fake
                fake = torch.cat((cur_input, cur_pred), 1)
                pred_fake = self.netD(fake.detach())
                self.loss_D_fake += self.criterionGAN(pred_fake, False)
                # real
                real = torch.cat((cur_input, cur_tgt), 1)
                pred_real = self.netD(real)
                self.loss_D_real += self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        self.loss_G_GAN, self.loss_G_L1 = 0, 0
        for i in range(len(self.output_names)):
            idx = torch.arange(self.real_A.size()[0]).cuda()[self.output_code[:, i]==1]
            if len(idx) > 0:  
                cur_tgt = self.real_tgt[i][idx, ...]
                cur_pred = self.fake_tgt[i][idx, ...]
                cur_input = self.real_A[idx, ...]
                fake = torch.cat((cur_input, cur_pred), 1)
                pred_fake = self.netD(fake)
                self.loss_G_GAN += self.criterionGAN(pred_fake, True)
                self.loss_G_L1 += self.criterionL1(cur_pred, cur_tgt) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
