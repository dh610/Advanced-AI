from torch import nn
from net.moco import MoCo

class CrossAttentionBlock(nn.Module):
    def __init__(self, img_dim, text_dim, out_dim=256):
        super(CrossAttentionBlock, self).__init__()
        self.img_linear = nn.Linear(img_dim, out_dim)  # img_dim -> 256
        self.text_linear = nn.Linear(text_dim, out_dim)  # text_dim -> 256
        self.attention = nn.MultiheadAttention(out_dim, num_heads=8)

        # 원래 이미지 크기로 복원하는 Linear 레이어
        self.output_linear = nn.Linear(out_dim, 3 * 128 * 128)

    def forward(self, img_feat, text_feat):
        batch_size = img_feat.size(0)

        # Flatten 이미지 특징 (batch_size, img_dim)
        img_feat_flat = img_feat.view(batch_size, -1)
        print(f"Flattened img_feat shape: {img_feat_flat.shape}")

        # 이미지와 텍스트 임베딩 생성 (batch_size, out_dim)
        img_feat_proj = self.img_linear(img_feat_flat)
        text_feat_proj = self.text_linear(text_feat)

        print(f"Projected img_feat shape: {img_feat_proj.shape}")
        print(f"Projected text_feat shape: {text_feat_proj.shape}")

        # Multihead Attention 수행 (batch_size, 1, out_dim)
        attn_output, attn_weights = self.attention(
            img_feat_proj.unsqueeze(1),  # (batch_size, 1, 256)
            text_feat_proj.unsqueeze(1),  # (batch_size, 1, 256)
            text_feat_proj.unsqueeze(1)   # (batch_size, 1, 256)
        )
        print(f"Attention output shape: {attn_output.shape}")

        # Linear 변환으로 이미지 크기로 복원 (batch_size, 3 * 128 * 128)
        restored_output = self.output_linear(attn_output)  # (batch_size, 1, 49152)
        restored_output = restored_output.view(batch_size, 3, 128, 128)

        print(f"Restored output shape: {restored_output.shape}")

        return restored_output, attn_weights

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

        self.cross_attention = CrossAttentionBlock(img_dim=3*128*128, text_dim=embedder_out_dim)
        # Encoder
        self.E = MoCo(base_encoder=ResEncoder, dim=dim, K=opt.batch_size * dim)

    def forward(self, x_query, x_key, text_embedding):

        x_query_attn, _ = self.cross_attention(x_query, text_embedding)
        x_key_attn, _ = self.cross_attention(x_key, text_embedding)
        print(f"x_query shape: {x_query_attn.shape}")
        print(f"x_key shape: {x_key_attn.shape}")

        combined_features_query = x_query + x_query_attn
        combined_features_key = x_key + x_key_attn
        print(f"x_query shape: {combined_features_query.shape}")
        print(f"x_key shape: {combined_features_key.shape}")

        if self.training:
            # degradation-aware represenetion learning
            fea, logits, labels, inter = self.E(x_query, x_key)
            print(f"x_query shape: {x_query.shape}")
            print(f"x_query shape: {x_key.shape}")
            import sys
            sys.exit(0)

            return fea, logits, labels, inter, x_query
        else:
            # degradation-aware represenetion learning
            fea, inter = self.E(x_query, x_query)
            return fea, inter, x_query
