from torch import nn

from net.encoder import CBDE
from net.DGRN import DGRN

class AirNet(nn.Module):
    def __init__(self, opt, embedder_out_dim):
        super(AirNet, self).__init__()

        # Restorer
        self.R = DGRN(opt)

        # Encoder
        self.E = CBDE(opt)


    def forward(self, x_query, x_key, text_embedding):

        if self.training:
            fea, logits, labels, inter, x_query= self.E(x_query, x_key, text_embedding)

            restored = self.R(x_query, inter)

            return restored, logits, labels
        else:
            fea, inter, x_query= self.E(x_query, x_query, text_embedding)

            restored = self.R(x_query, inter)

            return restored
