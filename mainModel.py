import torch
import torch.nn as nn
from models.basemodels.class_classifier import Class_classifier
from models.basemodels.domain_classifier import Domain_classifier
from models.basemodels.feature import Feature
from utils import embed
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


def estimate_JSD_MI(joint, marginal, mean=False):
    joint = (torch.log(torch.tensor(2.0)) - F.softplus(-joint))
    marginal = (F.softplus(-marginal) + marginal - torch.log(torch.tensor(2.0)))

    out = joint - marginal
    if mean:
        out = out.mean()
    return out


class MINE(nn.Module):
    def __init__(self, nfeatr, nfeati):
        super(MINE, self).__init__()
        self.fc1_x = nn.Linear(nfeatr, int(nfeatr/8)) #downsample
        self.bn1_x = nn.BatchNorm1d(int(nfeatr/8))

        self.fc1_y = nn.Linear(nfeati, int(nfeati/8))
        self.bn1_y = nn.BatchNorm1d(int(nfeati/8))

        self.fc2 = nn.Linear(int(nfeati/8) + int(nfeatr/8), int(nfeati/8) + int(nfeatr/8))
        self.bn2 = nn.BatchNorm1d(int(nfeati/8) + int(nfeatr/8))

        self.fc3 = nn.Linear(int(nfeati/8) + int(nfeatr/8), 1)

    def forward(self, x, y, lambd=1):

        # GRL
        # print("x.shape", x.shape)
        x = GradReverse.grad_reverse(x, lambd)
        y = GradReverse.grad_reverse(y, lambd)
        # print(y.shape)

        x = F.dropout(self.bn1_x(self.fc1_x(x)))
        y = F.dropout(self.bn1_y(self.fc1_y(y)))

        h = F.elu(torch.cat((x, y), dim=-1))
        # print()
        h = F.elu(self.bn2(self.fc2(h)))
        h = self.fc3(h)

        return h
    
class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       # nn.Linear(hidden_size // 2, hidden_size // 2),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                      # nn.Linear(hidden_size // 2, hidden_size // 2),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class Global_disc_EEGNet(nn.Module):
    def __init__(self, nfeatl, nfeatg, num_ch):
        super(Global_disc_EEGNet, self).__init__()

        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels=nfeatl, out_channels=16, kernel_size=(3), stride=2, bias=False),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=(1), stride=1, bias=False),
            nn.BatchNorm1d(1),
            # nn.AvgPool2d(kernel_size=(3), stride=(2)),
            # nn.Dropout(0.5)
        )
        self.dense1 = nn.Linear(int(131), 1)
        self.drop1 = nn.Dropout()

    def forward(self, localf, globalf):
        # print("localf.shape", localf.shape) #128 160 32
        # localf = localf.permute(0, 2, 1)

        localff = self.local_conv(localf)
        localff = localff.view(localf.shape[0], -1)
        # print("localff", localff.shape)  ##79
        # print("globalf.shape", globalf.shape) # 128
        concat = torch.cat((localff, globalf), dim=-1)
        # print("concat.shape", concat.shape)
        out = self.drop1(self.dense1(concat))
        return out

class Local_disc_EEGNet(nn.Module):  ## from the low dimonsion to the high dimonsion
    def __init__(self, nfeatl, nfeatg, nfeatl2, num_ch):
        super(Local_disc_EEGNet, self).__init__()
        self.num_ch = num_ch
        self.nfeatl = nfeatl
        self.nfeatl2 = nfeatl2
        self.nfeatg = nfeatg

        self.drop1 = nn.Dropout()
        self.conv = nn.Conv1d(64, 1, kernel_size=1)  # why plus
        self.dense1 = nn.Linear(int(136), 1)


    def forward(self, localf, globalf):
        # Concat-and-convolve architecture
        # print("globalf.shape", globalf.shape) #64
        # print("localf.shape", localf.shape)  # 64 *8
        num = localf.shape[1]
        globalff = globalf.unsqueeze(1)  # B 64 1
        globalff = globalff.repeat(1,  num, 1)  # B 64 8
        # print("globalff.shape", globalff.shape)
        # globalff  = globalff.reshape(128*2,128 ,64)
        concat = torch.cat((localf, globalff), dim=2) # (B 64, 8*2)
        # print("concat.shape", concat.shape)
        out = self.drop1(self.conv(concat))
        # print("out.shape", out.shape)
        out = out.view(out.shape[0],-1)
        out = self.dense1(out)
        return out



class InformerFeature(nn.Module):
    def __init__(self, c_in, d_model=512, n_heads=8, e_layers=1,  factor=5,  d_ff=512, feature_dropout=0.0, embed_drop=0.1, output_attention=False):
        super(InformerFeature, self).__init__()
        self.feature = Feature(factor=factor, d_model=d_model, n_heads=n_heads,
                               e_layers=e_layers, d_ff=d_ff, dropout=feature_dropout,
                               attn='full', activation='gelu',
                               output_attention=output_attention, distil=True)

        self.en_embeding = embed.DeapEmbedding(c_in, d_model, embed_drop)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_data, enc_self_mask=None):
        input_data = self.en_embeding(input_data)
        # print("input_data.shape", input_data.shape)
        feature, atten = self.feature(input_data, enc_self_mask)

        return feature.permute(0, 2, 1), atten

class Conv(nn.Module):
    def __init__(self,  linesize=32*79*15, outsize=64):
        super(Conv, self).__init__()
        # self.c0 = nn.Conv2d(1, 64, kernel_size=(64,1), stride=1, padding =0)
        # self.c1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding = 0)
        # self.c2 = nn.Conv2d(16, 16, kernel_size=1, stride=1, )

        self.l1 = nn.Linear(linesize, outsize)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, feature):
        h = feature
        # feature = self.dropout(feature)
        # h = F.relu(self.c0(feature))
        # h = F.relu(self.c1(h))
        # h = F.relu(self.c2(h))
        # print("h.shape", h.shape)
        h = self.dropout(self.l1(h.view(feature.shape[0], -1)))
        # h = self.l1(h.view(feature.shape[0], -1))
        return h

# class Conv(nn.Module):
#     def __init__(self,  linesize=32*79*15, outsize=64):
#         super(Conv, self).__init__()
#         self.c0 = nn.Conv1d(1, 8, kernel_size=(64,1), stride=1, padding =0)
#         self.c1 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding = 0)
#         self.c2 = nn.Conv1d(64, 8, kernel_size=3, padding =1, stride=1)
#         self.dropout = nn.Dropout(p=dropout)
#
#         self.l1 = nn.Linear(linesize, outsize)
#
#     def forward(self, feature):
#         feature = self.dropout(feature)
#         h = F.relu(self.c2(feature))
#         # h = F.relu(self.c1(h))
#         # h = F.relu(self.c2(h))
#         # print("h.shape", h.shape)
#         return self.l1(h.view(feature.shape[0], -1))


class Decomposer(nn.Module):
    def __init__(self, nfeat):
        super(Decomposer, self).__init__()
        self.nfeat = nfeat
        self.embed_layer2 = nn.Sequential(nn.Conv1d(nfeat, nfeat*2, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(nfeat*2), nn.ELU(), nn.Dropout())

    def forward(self, x):
        embedded = self.embed_layer2(x)
        rele, irre = torch.split(embedded, [int(self.nfeat), int(self.nfeat)], dim=1)

        return rele, irre


class MainModel(nn.Module):
    def __init__(self, c_in =160, d_model=64, n_heads=2, e_layers=3,  factor=5,  d_ff=512, feature_dropout=0.0, embed_drop=0.1, output_attention=False,
                 convoutsize=64, convlinesize=64*15*9,
                 out_l=2):
        super().__init__()
        self.informerFeature = InformerFeature(c_in, d_model, n_heads, e_layers, factor, d_ff, feature_dropout, embed_drop, output_attention)

        self.featureConv = Conv(linesize=convlinesize, outsize=convoutsize)
        self.d = Decomposer(nfeat=d_model)

        in_l = convoutsize
        self.class_classifier = Class_classifier(in_l=in_l)
        self.domain_classifier = Domain_classifier(in_l=in_l, out_l=out_l)

    def forward(self, input_data, enc_self_mask=None):
        feature1, atten = self.informerFeature(input_data)
        # print("the shape of feature", feature.shape)
        re, ir = self.d(feature1)
        # print("ir", ir.shape)  ## 64*8
        # feature = re
        # feature = feature.unsqueeze(1)  ##
        # print("the shape of feature", feature.shape)  ## 256 1 64 8
        feature = self.featureConv(re)

        class_ouput = self.class_classifier(feature)
        domain_output = self.domain_classifier(ir)

        return re, ir, feature, class_ouput, atten, domain_output

if __name__ == '__main__':
    torch.cuda.empty_cache()
    src = torch.rand(155, 128, 160).to("cuda:0")

    model = MainModel(d_model=160, n_heads=2, e_layers=3, convlinesize=64*7*39, convoutsize=64).to("cuda:0")

    feature, class_ouput, domain_output, atten = model(src)
    print(feature.shape)
    from models.deepInfoMaxLoss import DeepInfoMaxLoss
    deepInfoMaxLoss = DeepInfoMaxLoss(1, 128, 128).to("cuda:0")
    print(deepInfoMaxLoss(feature, feature))
