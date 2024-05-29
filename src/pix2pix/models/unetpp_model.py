import torch
from .base_model import BaseModel
from . import networks


class UNETPPModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(netG='unet_plusplus', input_nc=1, output_nc=4)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')
            parser.add_argument('--lambda_percep', type=float, default=1.0, help='weight for perceptual loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_L1', 'G_percep']
        self.visual_names = ['real_A'] #, 'fake_B', 'real_B', 'fake_C', 'real_C', 'fake_D', 'real_D', 'fake_E', 'real_E']
        self.output_names = ['Mitochondria', 'Nucleus', 'Tubulin', 'Actin']

        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        # breakpoint()

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss(reduction='none')
            self.criterionPerceptual = networks.VGGPerceptualLoss()
            self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=opt.lr, weight_decay=0.0001)
            self.optimizers = [self.optimizer_G]

    def set_input(self, input):
        # self.input_code = input['input_code'].to(self.device)
        self.output_code = input['output_code'].to(self.device)
        self.real_A = input['input'].to(self.device)
        self.gt = torch.cat([
            input['Mitochondria'].to(self.device), 
            input['Nucleus'].to(self.device),
            input['Tubulin'].to(self.device),
            input['Actin'].to(self.device)], 1)

        # weighting masks
        self.weighting = torch.cat([
            input['Mitochondria_mask'].to(self.device), 
            input['Nucleus_mask'].to(self.device), 
            input['Tubulin_mask'].to(self.device), 
            input['Actin_mask'].to(self.device)], 1)
        
    def forward(self):
        self.pred = self.netG(self.real_A)  

    def update_G(self):
        self.loss_G_L1, self.loss_G_percep = 0, 0
        self.optimizer_G.zero_grad()    

        # reshape
        self.output_code = self.output_code.view(-1)
        self.pred = self.pred.view(-1, self.pred.shape[-2], self.pred.shape[-1])
        self.gt = self.gt.view(-1, self.gt.shape[-2], self.gt.shape[-1])
        self.weighting = self.weighting.view(-1, self.weighting.shape[-2], self.weighting.shape[-1])

        # find the indices for target adaptive loss
        idx = torch.arange(self.output_code.shape[0]).cuda()[self.output_code==1]
        self.loss_G_L1 = (self.criterionL1(self.pred[idx, ...], self.gt[idx, ...]) * self.weighting[idx, ...]).mean() * self.opt.lambda_L1
        self.loss_G_percep = self.criterionPerceptual(self.pred[idx, ...].unsqueeze(1), self.gt[idx, ...].unsqueeze(1)) * self.opt.lambda_percep        
        self.loss_G = self.loss_G_L1  + self.loss_G_percep 
        self.loss_G.backward()
        self.optimizer_G.step()   

    def optimize_parameters(self):
        self.forward()                   
        self.update_G()
           
