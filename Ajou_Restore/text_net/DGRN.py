import torch.nn as nn
import torch
import torch.nn.functional as F
from text_net.deform_conv import DCN_layer
import clip

clip_model, preprocess = clip.load("ViT-B/32", device='cuda')

# 동적으로 텍스트 임베딩 차원 가져오기
text_embed_dim = clip_model.text_projection.shape[1]


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DGM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(DGM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.dcn = DCN_layer(self.channels_in, self.channels_out, kernel_size,
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sft = SFT_layer(self.channels_in, self.channels_out)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter, text_prompt):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        dcn_out = self.dcn(x, inter)
        sft_out = self.sft(x, inter, text_prompt)
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

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img_param, text_param):
        if text_param.size(2) != img_param.size(2) or text_param.size(3) != img_param.size(3):
            text_param = F.interpolate(
                text_param, size=(img_param.size(2), img_param.size(3)), mode='nearest'
            )

        combined = torch.cat([img_param, text_param], dim=1)
        
        attn = self.attention(combined)
        return img_param * attn + text_param * (1 - attn)

class FiLM(nn.Module):
    def __init__(self, text_embed_dim, feature_dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(text_embed_dim, feature_dim)
        self.beta = nn.Linear(text_embed_dim, feature_dim)

    def forward(self, x, text_embed):
        gamma = self.gamma(text_embed).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta(text_embed).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * gamma + beta

class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out, num_heads=8):
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

        self.text_proj_head = TextProjectionHead(text_embed_dim, channels_out)
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

        self.film = FiLM(text_embed_dim, channels_out)
        self.attention_fusion = AttentionFusion(channels_out)

    def forward(self, x, inter, text_prompt):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        img_gamma = self.conv_gamma(inter)
        img_beta = self.conv_beta(inter)

        text_tokens = clip.tokenize(text_prompt).to(x.device)  # Tokenize the text prompts (Batch size)
        with torch.no_grad():
            text_embed = clip_model.encode_text(text_tokens)
        
        text_proj = self.text_proj_head(text_embed).float()

        text_gamma = self.text_gamma(text_proj.unsqueeze(-1).unsqueeze(-1))  # Reshape to match (B, C, H, W)
        text_beta = self.text_beta(text_proj.unsqueeze(-1).unsqueeze(-1))  # Reshape to match (B, C, H, W)

        film_gamma = self.film(x, text_gamma)
        film_beta = self.film(x, text_beta)

        fusion_gamma = self.attention_fusion(img_gamma, film_gamma)  # (B, C, H, W)
        fusion_beta = self.attention_fusion(img_beta, film_beta)  # (B, C, H, W)

        out = x * fusion_gamma + fusion_beta
        # Add Residual Connection
        return x + out


class DGB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size):
        super(DGB, self).__init__()

        # self.da_conv1 = DGM(n_feat, n_feat, kernel_size)
        # self.da_conv2 = DGM(n_feat, n_feat, kernel_size)
        self.dgm1 = DGM(n_feat, n_feat, kernel_size)
        self.dgm2 = DGM(n_feat, n_feat, kernel_size)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter, text_prompt):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''

        out = self.relu(self.dgm1(x, inter, text_prompt))
        out = self.relu(self.conv1(out))
        out = self.relu(self.dgm2(out, inter, text_prompt))
        out = self.conv2(out) + x

        return out


class DGG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_blocks):
        super(DGG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DGB(conv, n_feat, kernel_size) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x, inter, text_prompt):
        '''
        :param x: feature map: B * C * H * W
        :param inter: degradation representation: B * C * H * W
        '''
        res = x
        for i in range(self.n_blocks):
            res = self.body[i](res, inter, text_prompt)
        res = self.body[-1](res)
        res = res + x

        return res


class DGRN(nn.Module):
    def __init__(self, opt, conv=default_conv):
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
            DGG(default_conv, n_feats, kernel_size, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, inter, text_prompt):
        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i](res, inter, text_prompt)
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        return x
