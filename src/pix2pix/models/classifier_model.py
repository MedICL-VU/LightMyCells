import torch
from .base_model import BaseModel
from monai.networks.nets import DenseNet121


class ClassifierModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['CE']
        self.visual_names = ['image']
        self.model_names = ['G']

        self.netG = DenseNet121(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            init_features=64,
            growth_rate=32,
            block_config=(6, 12, 24, 16)).to(self.device)

        self.netG = torch.nn.DataParallel(self.netG, [0])

        if self.isTrain:
            self.criterion = torch.nn.CrossEntropyLoss()    
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G]  # learning rate

    def set_input(self, input):
        self.image = input['input'].to(self.device)
        self.label = input['label'].to(self.device)
        
    def forward(self):
        self.logit = self.netG(self.image)
        self.prob = self.logit.softmax(dim=1)
        
        if self.epoch % 3 == 0:
            print(self.prob, self.label)
        
        self.optimizer_G.zero_grad()  
        self.loss_CE = self.criterion(self.logit, self.label.squeeze(1))
        self.loss_CE.backward()
        self.optimizer_G.step()   

    def optimize_parameters(self):
        self.forward()                   
    
    def get_epoch(self, epoch_num):
        self.epoch = epoch_num