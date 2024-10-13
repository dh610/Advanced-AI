from torch import nn
from net.moco import MoCo

class CrossAttentionBlock(nn.Module):
    def __init__(self, img_dim, text_dim, out_dim):
        super(CrossAttentionBlock, self).__init__()
        self.img_linear = nn.Linear(img_dim, out_dim)
        self.text_linear = nn.Linear(text_dim, out_dim)
        self.attention = nn.MultiheadAttention(out_dim, num_heads=8)

    def forward(self, img_feat, text_feat):
        img_feat_proj = self.img_linear(img_feat)  # Image feature change
        text_feat_proj = self.text_linear(text_feat)  # Text embedding change

        # Cross-Attention
        attn_output, attn_weights = self.attention(img_feat_proj.unsqueeze(1), text_feat_proj.unsqueeze(1), text_feat_proj.unsqueeze(1))
        return attn_output.squeeze(1), attn_weights

class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()

        self.E_pre = ResBlock(in_feat=3, out_feat=64, stride=1)
        self.E = nn.Sequential(
            ResBlock(in_feat=64, out_feat=128, stride=2),
            ResBlock(in_feat=128, out_feat=256, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        inter = self.E_pre(x)
        fea = self.E(inter).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out, inter


class CBDE(nn.Module):
    def __init__(self, opt, embedder_out_dim):
        super(CBDE, self).__init__()

        dim = 256

        self.cross_attention = CrossAttentionBlock(img_dim=dim, text_dim=embedder_out_dim, out_dim=dim)
        # Encoder
        self.E = MoCo(base_encoder=ResEncoder, dim=dim, K=opt.batch_size * dim)

    def forward(self, x_query, x_key, text_embedding):

        x_query_attn, _ = self.cross_attention(x_query, text_embedding)
        x_key_attn, _ = self.cross_attention(x_key, text_embedding)

        combined_features_query = x_query + x_query_attn
        combined_features_key = x_key + x_key_attn

        if self.training:
            # degradation-aware represenetion learning
            fea, logits, labels, inter = self.E(combined_features_query, combined_features_key)

            return fea, logits, labels, inter, combined_features_query
        else:
            # degradation-aware represenetion learning
            fea, inter = self.E(combined_features_query, combined_features_query)
            return fea, inter, combined_features_query
