import torch
from torch import Tensor
import torch.nn as nn


# from utils.nn import threeD_to_2D_tensor


class ConvDense(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvDense, self).__init__()

        self.dense = nn.Sequential(
            # creates relationships between various layers of the connection ->  use of the features, reduces the gradient disappearance problem
            nn.BatchNorm3d(ch_in),
            nn.ReLU(),
            nn.Conv3d(ch_in, ch_in, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=True),
            # (depth/time, height, width),
            nn.BatchNorm3d(ch_in),
            nn.ReLU(),
            nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            # (depth/time, height, width)
        )

    def forward(self, x):
        return self.dense(x)


class ConvTransition(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvTransition, self).__init__()

        self.trans = nn.Sequential(  # improves the model compactness further
            nn.BatchNorm3d(ch_in),
            nn.ReLU(),
            nn.Conv3d(ch_in, ch_out, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=True),
            # (depth/time, height, width)
            nn.AvgPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        )

    def forward(self, x):
        return self.trans(x)


class ConvUp(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvUp, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2)),
            nn.Conv3d(ch_in, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=(0, 0, 0))  # shrinks time factor so can work with 2D conv later
        )

    def forward(self, x):
        x = self.up(x)
        return x


class LipVideoTo2DEmbedding(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(LipVideoTo2DEmbedding, self).__init__()

        # input x's shape: B, C, T, H, W = B, 3, 64, 160, 160 = x.size()
        self.bottleneck = nn.Sequential(
            # reduces the input feature volumes -> reduces the model parameters, effectively suppresses overfitting, and saves computational power.
            # 3D CNN
            nn.Conv3d(ch_in, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            # (depth/time, height, width); out size: B, 32, 64, 80, 80
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0)),  # out size: B, 32, 64, 39, 39

            nn.Conv3d(32, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            # (depth/time, height, width); out size: B, 32, 64, 20, 20
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),  # out size: B, 32, 64, 10, 10

            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),# out size: B, 32, 64, 8, 8

        )

        # 3D Densely Connected CNN
        self.dense6 = nn.Sequential(*[ConvDense(32, 32) for _ in range(6)])  # out size: B, 32, 64, 8, 8
        self.trans1 = ConvTransition(32, 32)  # out size: B, 32, 64, 9, 9;

        self.dense12 = nn.Sequential(*[ConvDense(64, 64) for _ in range(12)])  # out size: B, 64, 64, 8, 8
        self.trans2 = ConvTransition(96, 96)  # out size: B, 96, 64, 9, 9;

        self.dense24 = nn.Sequential(*[ConvDense(128, 128) for _ in range(24)])  # out size: B, 128, 64, 8, 8
        self.trans3 = ConvTransition(160, 160)  # out size: B, 128, 64, 9, 9;

        self.dense16 = nn.Sequential(*[ConvDense(192, 192) for _ in range(16)])  # out size: B, 192, 64, 8, 8

        # MLFF: have 3 separate CNNs and input them separately then concatenate them together
        # Different convolutional layers can extract features at various levels of abstraction while preventing the creation of strongly correlated activations -> solves overtraining and overfitting
        self.multi_layer_feat_fusion1 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), bias=True),
            nn.BatchNorm3d(192),
            nn.ReLU(),
        )

        self.multi_layer_feat_fusion2 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), bias=True),
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.multi_layer_feat_fusion3 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), bias=True),
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.Dropout3d(p=0.3)
        )

        # upsampling
        self.conv = nn.Sequential(
            nn.Conv3d(192 * 3, 192 * 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        )
        self.conv_up1 = ConvUp(192 * 3, 512)
        self.conv_up2 = ConvUp(512, 128)
        self.conv_up3 = ConvUp(128, 64)
        self.conv_end = nn.Sequential(
            nn.Conv3d(64, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        )

        # output's shape = B, C, H, W =  B, C, 64, 64

    def forward(self, x):
        x = self.bottleneck(x)

        #print("x shape", x.shape)

        dense1 = self.dense6(x)  # out shape: 69, 32, 64,8, 8
        trans1 = self.trans1(dense1)
        cat1 = torch.cat((dense1, trans1), dim=1)  # shape: 69, 64, 64, 8, 8

        x = self.dense12(cat1)
        x = torch.cat((dense1, x), dim=1)  # shape: 69, 32+64=96, 64, 8, 8
        trans2 = self.trans2(x)
        cat2 = torch.cat((dense1, trans2), dim=1)  # shape: 69, 32+96=128, 64, 8, 8

        x = self.dense24(cat2)  # shape: 69, 128, 64, 8, 8
        x = torch.cat((dense1, x), dim=1)  # shape: 69, 32+128=160, 64, 8, 8
        trans3 = self.trans3(x)
        cat3 = torch.cat((dense1, trans3), dim=1)  # shape: 69, 32+160=192, 64, 8, 8
        #print("cat.shape:", cat3.shape)

        out = self.dense16(cat3)

        module1 = self.multi_layer_feat_fusion1(out)
        module2 = self.multi_layer_feat_fusion2(out)
        module3 = self.multi_layer_feat_fusion3(out)

        cat4 = torch.cat((module1, module2, module3), dim=1)  # shape: 69, 192*3=576, 64, 8, 8
        #print("merged:", cat4.shape)

        # upsampling to match the image size 64 x 64 and lower channels
        x = self.conv(cat4)
        x = self.conv_up1(x)
        x = self.conv_up2(x)
        x = self.conv_up3(x)
        #print(x.shape)
        x = self.conv_end(x)  # out shape: 69, ch_out=1, 1, 64, 64


        n_batch, n_channels, s_time, sx, sy = x.shape
        #return torch.squeeze(x, 1)
        #x = x.transpose(1, 2)
        return x.reshape(n_batch, n_channels * s_time, sx, sy)  # out shape: torch.Size([B * 64, ch_out=1, 64, 64])


if __name__ == "__main__":
    # testing
    model = LipVideoTo2DEmbedding(3, 1)
    input = torch.randn(2, 3, 64, 160, 160)
    output = model(input)
    print("output shape:", output.shape)
