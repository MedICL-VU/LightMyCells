import torch
from .base_model import BaseModel
from . import networks


class LightV5Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='res_multi_decoder', dataset_mode='light')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_percep', type=float, default=1.0, help='weight for perceptual loss')
            parser.add_argument('--lambda_edge', type=float, default=10.0, help='weight for edge loss (L1)')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'G_percep', 'G_edge'] #, 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_C', 'real_C', 'fake_D', 'real_D', 'fake_E', 'real_E']
        self.output_names = ['Mitochondria', 'Nucleus', 'Tubulin', 'Actin']

        if self.isTrain:
            self.model_names = ['G', 'D1', 'D2', 'D3', 'D4']
        else:  
            self.model_names = ['G']
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # network architectures
            self.netD1 = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD3 = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD4 = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDs = [self.netD1, self.netD2, self.netD3, self.netD4]

            # loss functions 
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionPerceptual = networks.VGGPerceptualLoss()

            # Optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D3 = torch.optim.Adam(self.netD3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D4 = torch.optim.Adam(self.netD4.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Ds = [self.optimizer_D1, self.optimizer_D2, self.optimizer_D3, self.optimizer_D4]
            self.optimizers = [self.optimizer_G] + self.optimizer_Ds

    def set_input(self, input):
        # self.input_code = input['input_code'].to(self.device)
        self.output_code = input['output_code'].to(self.device)
        self.real_A = input['input'].to(self.device)
        self.real_B = input['Mitochondria'].to(self.device) 
        self.real_C = input['Nucleus'].to(self.device) 
        self.real_D = input['Tubulin'].to(self.device) 
        self.real_E = input['Actin'].to(self.device) 
        self.real_tgt = [self.real_B, self.real_C, self.real_D, self.real_E]

        # weighting masks
        self.real_B_mask = input['Mitochondria_mask'].to(self.device) 
        self.real_C_mask = input['Nucleus_mask'].to(self.device) 
        self.real_D_mask = input['Tubulin_mask'].to(self.device) 
        self.real_E_mask = input['Actin_mask'].to(self.device) 
        self.real_tgt_mask = [self.real_B_mask, self.real_C_mask, self.real_D_mask, self.real_E_mask]
        
    def forward(self):
        self.fake_B, self.fake_C, self.fake_D, self.fake_E = self.netG(self.real_A)  
        self.fake_tgt = [self.fake_B, self.fake_C, self.fake_D, self.fake_E]

    def update_D(self): 
        for i in range(len(self.output_names)):
            idx = torch.arange(self.real_A.size()[0]).cuda()[self.output_code[:, i]==1]
            # at least one sample in the batch has ground truth
            if len(idx) > 0:  
                cur_tgt = self.real_tgt[i][idx, ...]
                cur_pred = self.fake_tgt[i][idx, ...]
                cur_input = self.real_A[idx, ...]
                self.set_requires_grad(self.netDs[i], True)
                self.optimizer_Ds[i].zero_grad()     
                
                # fake
                fake = torch.cat((cur_input, cur_pred), 1)
                pred_fake = self.netDs[i](fake.detach())
                loss_D_fake = self.criterionGAN(pred_fake, False)

                # real
                real = torch.cat((cur_input, cur_tgt), 1)
                pred_real = self.netDs[i](real)
                loss_D_real = self.criterionGAN(pred_real, True)

                loss_D = (loss_D_fake + loss_D_real) * 0.5
                loss_D.backward()
                self.optimizer_Ds[i].step()  

    def update_G(self):
        self.optimizer_G.zero_grad()    
        self.loss_G_GAN, self.loss_G_L1, self.loss_G_percep, self.loss_G_edge = 0, 0, 0, 0
        for i in range(len(self.output_names)):
            idx = torch.arange(self.real_A.size()[0]).cuda()[self.output_code[:, i]==1]
            if len(idx) > 0:  
                cur_tgt = self.real_tgt[i][idx, ...]
                cur_mask = self.real_tgt_mask[i][idx, ...]
                cur_pred = self.fake_tgt[i][idx, ...]
                cur_input = self.real_A[idx, ...]
                self.set_requires_grad(self.netDs[i], False) 
                fake = torch.cat((cur_input, cur_pred), 1)
                pred_fake = self.netDs[i](fake)
                
                if i < 2:  # mitochondria & nucleus
                    cur_pred_edge = networks.sobelLayer(cur_pred)
                    cur_tgt_edge = networks.sobelLayer(cur_tgt).detach()
                    self.loss_G_edge += self.criterionL1(cur_pred_edge, cur_tgt_edge) * self.opt.lambda_edge
                        
                self.loss_G_GAN += self.criterionGAN(pred_fake, True)
                self.loss_G_L1 += (self.criterionL1(cur_pred, cur_tgt) * cur_mask).mean() * self.opt.lambda_L1
                self.loss_G_percep += self.criterionPerceptual(cur_pred, cur_tgt) * self.opt.lambda_percep
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_percep + self.loss_G_edge
        self.loss_G.backward()
        self.optimizer_G.step()   

    def optimize_parameters(self):
        self.forward()                   
        self.update_D()
        self.update_G()
           
