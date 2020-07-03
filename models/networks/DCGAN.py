# Copyright shawnzhang31. All Rights Reserved
import torch
import torch.nn  as nn

def weights_init(m:nn.Module):
    """
    调用DCGAN模型的时候的权重初始化
    @params:
        m       - Required : DCGAN的Generator或者Discriminator
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu:int = 0, nz:int = 100, ngf:int = 64, nc:int = 3):
        """ 生成器
        @params:
            ngpu    - Optional: 训练使用的GPU的数量, 默认值为0， 表示CPU模式 (int)
            nz      - Optional: 输入的向量的长度，默认值100 (int)
            ngf     - Optional: Generator输出的特征图的size，默认为64 (int)
            nc      - Optional: 训练图像的通道数量，默认值3-彩色图像 (int)
        """
        
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入向量Z
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size (ngf*8)x4x4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size (ngf*4)x8x8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size (ngf*2)x16x16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh() #

            # state size (nc) * 64 * 64
        )

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu:int = 0, nc:int = 3, ndf:int = 64):
        """ 生成器
        @params:
            ngpu    - Optional: 训练使用的GPU的数量, 默认值为0， 表示CPU模式 (int)
            nc      - Optional: 训练图像的通道数量，默认值3-彩色图像 (int)
            ndf     - Optional: Discriminator的特征图的size，默认为64 (int)
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入 nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# ngpu = 0
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# netG = Generator(ngpu=ngpu).to(device)

# if (device.type == 'cuda') and (ngpu > 0):
#     netG = nn.DataParallel(netG, list(range(ngpu)))

# print(netG)

# netD = Discriminator(ngpu=ngpu).to(device)

# if (device.type == 'cuda') and (ngpu > 0):
#     netD = nn.DataParallel(netD, list(range(ngpu)))

# print(netD)