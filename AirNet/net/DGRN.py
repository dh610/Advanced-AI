import torch.nn as nn
import torch
from .deform_conv import DCN_layer
#import clip

#clip_model, preprocess = clip.load("ViT-B/32", device='cuda')

# 동적으로 텍스트 임베딩 차원 가져오기
#text_embed_dim = clip_model.text_projection.shape[1]


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DGM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, embedder):
        super(DGM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.dcn = DCN_layer(self.channels_in, self.channels_out, kernel_size,
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sft = SFT_layer(self.channels_in, self.channels_out, embedder=embedder)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter, de_id):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        dcn_out = self.dcn(x, inter)
        sft_out = self.sft(x, inter, de_id)
        out = dcn_out + sft_out
        out = x + out

        return out

# Projection Head 정의
class TextProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextProjectionHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        ).float()
    
    def forward(self, x):
        return self.proj(x.float())

class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out, embedder):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

        self.embedder = embedder

        self.text_proj_head = TextProjectionHead(self.embedder.out_dim, channels_out)
        self.text_gamma = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        ).float()
        self.text_beta = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        ).float()

    def forward(self, x, inter, de_id):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        img_gamma = self.conv_gamma(inter)
        img_beta = self.conv_beta(inter)

        with torch.no_grad():
            text_embed = self.embedder(de_id, 'text_idx_encoder')
        
        text_proj = self.text_proj_head(text_embed).float()

        text_gamma = self.text_gamma(text_proj.unsqueeze(-1).unsqueeze(-1))  # Reshape to match (B, C, H, W)
        text_beta = self.text_beta(text_proj.unsqueeze(-1).unsqueeze(-1))  # Reshape to match (B, C, H, W)

        '''
        print ("Shape of x", x.shape)
        print ("Shape of img_gamma", img_gamma.shape)
        print ("Shape of text_gamma", text_gamma.shape)
        print ("Shape of img_beta", img_beta.shape)
        print ("Shape of text_beta", text_beta.shape)
        import sys
        sys.exit(0)
        '''

        # concat으로 text 결합 실험
        return x * (img_gamma+text_gamma) + (img_beta+text_beta)


class DGB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, embedder):
        super(DGB, self).__init__()

        # self.da_conv1 = DGM(n_feat, n_feat, kernel_size)
        # self.da_conv2 = DGM(n_feat, n_feat, kernel_size)
        self.dgm1 = DGM(n_feat, n_feat, kernel_size, embedder=embedder)
        self.dgm2 = DGM(n_feat, n_feat, kernel_size, embedder=embedder)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter, de_id):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''

        out = self.relu(self.dgm1(x, inter, de_id))
        out = self.relu(self.conv1(out))
        out = self.relu(self.dgm2(out, inter, de_id))
        out = self.conv2(out) + x

        return out


class DGG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_blocks, embedder):
        super(DGG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DGB(conv, n_feat, kernel_size, embedder=embedder) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, inter, de_id):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''
        res = x
        for i in range(self.n_blocks):
            res = self.body[i](res, inter, de_id)
        res = self.body[-1](res)
        res = res + x

        return res


class DGRN(nn.Module):
    def __init__(self, opt, embedder, conv=default_conv):
        super(DGRN, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # body
        modules_body = [
            DGG(default_conv, n_feats, kernel_size, n_blocks, embedder=embedder) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, inter, de_id):
        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i](res, inter, de_id)
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        return x
