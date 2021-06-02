import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        #self.features = self._make_layers(cfg[vgg_name])
        cfg1 = [64, 64, 'M']
        cfg2 = [128, 128, 'M']
        cfg3 = [256, 256, 256, 'M']
        cfg4 = [512, 512, 512, 'M']
        cfg5 = [512, 512, 512, 'M']
        self.f1 = self._make_layers(cfg1, 3)
        self.f2 = self._make_layers(cfg2, 64)
        self.f3 = self._make_layers(cfg3, 128)
        self.f4 = self._make_layers(cfg4, 256)
        self.f5 = self._make_layers(cfg5, 512)
        self.layer = nn.AvgPool2d(kernel_size=1, stride=1)
        #self.classifier = nn.Linear(512, 10)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        #out = self.features(x)
        out1 = self.f1(x)
        out2 = self.f2(out1)
        out3 = self.f3(out2)
        out4 = self.f4(out3)
        out45 = self.f5(out4)
        out5 = self.layer(out45)
        out = out5.view(out5.size(0), -1)
        out = self.classifier(out)
        return [out2, out3, out4, out5, out]
    
    def forward_mod(self,x,y,refs):
        x_ref,y_ref = refs
        x = self.f1(x)
        x_ref = self.f1(x_ref)
        conv_layers = [self.f2,self.f3,self.f4,self.f5,self.layer]
        loss = 0.0
        for i,conv_layer in enumerate(conv_layers):
            x = conv_layer(x)
            x_ref = conv_layer(x_ref)
            mask = torch.where(y[:,None]==y_ref[None,:],torch.ones(1).to(y),torch.ones(1).to(y).neg())
            # 5d tensor
            loss += ((x.unsqueeze(1)-x_ref).pow(2).mean(dim=(2,3,4)).sqrt()*mask).mean()
#             if i % 2 == 1:
#                 x = F.max_pool2d(x,2)
#                 x_ref = F.max_pool2d(x_ref,2)
        x = x.view(x.size(0), -1)
        x    = self.classifier(x)
        return x,loss
    

    def _make_layers(self, cfg, in_channels):
        layers = []
#         in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x        
        return nn.Sequential(*layers)