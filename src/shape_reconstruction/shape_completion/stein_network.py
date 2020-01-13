import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SteinEncoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 use_skip_connections=False,
                 drop_rate=0.3,
                 use_batch_norm=True):
        super(SteinEncoder, self).__init__()
        self.use_skip_connections = use_skip_connections
        c_e = [1, 64, 128, 256]
        if use_batch_norm:
            e_convs = [
                nn.Sequential(
                    nn.Conv3d(c_e[0], c_e[1], kernel_size=4, stride=2),
                    nn.LeakyReLU(), nn.BatchNorm3d(c_e[1]))
            ]
        else:
            e_convs = [
                nn.Sequential(
                    nn.Conv3d(c_e[0], c_e[1], kernel_size=4, stride=2),
                    nn.LeakyReLU())
            ]

        self.dropout = nn.Dropout3d(drop_rate)

        for p in range(2, len(c_e)):
            if use_batch_norm:
                e_convs.append(
                    nn.Sequential(
                        nn.Conv3d(c_e[p - 1], c_e[p], kernel_size=4, stride=2),
                        nn.LeakyReLU(), self.dropout, nn.BatchNorm3d(c_e[p])))
            else:
                e_convs.append(
                    nn.Sequential(
                        nn.Conv3d(c_e[p - 1], c_e[p], kernel_size=4, stride=2),
                        nn.LeakyReLU(), self.dropout))

        self.fc = nn.Linear(c_e[-1] * 3 * 3 * 3, latent_dim)
        self.e_convs = nn.ModuleList(e_convs)
        self.relu = F.relu

    def forward(self, x, num_particles=5):
        z = []
        temp = x
        for particle in range(num_particles):
            x = self.e_convs[0](temp)
            for i in range(1, len(self.e_convs)):
                x = self.e_convs[i](x)
            x = x.view(x.size(0), -1)
            z.append(self.fc(x))
        z = tc.stack(z).transpose(0, 1)
        return z

    def activate_dropout_layers(self):
        self.dropout.train()


class SteinDecoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 use_skip_connections=False,
                 use_batch_norm=True):
        super(SteinDecoder, self).__init__()
        c_e = [1, 64, 128, 256]
        d_s = [2, 2, 2]
        d_p = [0, 1, 0]
        d_convs = []
        if use_skip_connections:
            d_convs = [
                nn.Sequential(
                    nn.ConvTranspose3d(2 * c_e[1],
                                       c_e[0],
                                       kernel_size=4,
                                       stride=d_s[0],
                                       output_padding=d_p[0]), nn.Sigmoid())
            ]
        else:
            d_convs = [
                nn.Sequential(
                    nn.ConvTranspose3d(c_e[1],
                                       c_e[0],
                                       kernel_size=4,
                                       stride=d_s[0],
                                       output_padding=d_p[0]), nn.Sigmoid())
            ]

        for p in range(2, len(c_e)):
            if use_skip_connections:
                d_convs.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(2 * c_e[p],
                                           c_e[p - 1],
                                           kernel_size=4,
                                           stride=d_s[p - 1],
                                           output_padding=d_p[p - 1]),
                        nn.ReLU()))
            else:
                if use_batch_norm:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(c_e[p],
                                               c_e[p - 1],
                                               kernel_size=4,
                                               stride=d_s[p - 1],
                                               output_padding=d_p[p - 1]),
                            nn.ReLU(), nn.BatchNorm3d(c_e[p - 1])))
                else:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(c_e[p],
                                               c_e[p - 1],
                                               kernel_size=4,
                                               stride=d_s[p - 1],
                                               output_padding=d_p[p - 1]),
                            nn.ReLU()))

        self.fc = nn.Linear(latent_dim, c_e[-1] * 3 * 3 * 3)

        self.shape_after_fc = [c_e[-1], 3, 3, 3]
        d_convs.reverse()
        self.d_convs = nn.ModuleList(d_convs)


# TODO: Parallelize this method. Currently, as z is or the form [batch_size,
# particles, latent_dim], I have chosen to loop over each particle and
# calculate the reconstruction for this. The reson for this is that the
# particles should not be considered for batch normalization (I
# think) and thus I loop over each particle. According to the current batch
# normalization in pytorch
# https://pytorch.org/docs/stable/nn.html?highlight=batchnorm#torch.nn.BatchNorm1d
# one cannot freeze a dimension which is needed when doing it in parallel.

    def forward(self, z):
        num_particles = z.shape[1]
        z = z.transpose(0, 1)
        y = []
        for i in range(num_particles):
            x = self.fc(z[i])
            x = x.view(x.size(0), self.shape_after_fc[0],
                       self.shape_after_fc[1], self.shape_after_fc[2],
                       self.shape_after_fc[3])
            for j in range(len(self.d_convs)):
                x = self.d_convs[j](x)
            y.append(x)
        y = tc.stack(y).transpose(0, 1)
        return y


class SteinVariationalEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, num_particles=5, beta=1, reg=1):
        super(SteinVariationalEncoderDecoder, self).__init__()
        self.device = "cpu"
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.regularization = reg
        self.num_train_samples = num_particles
        self.optimizer_enc = ""
        self.optimizer_dec = ""
        self.setup_optimizers()

    def set_device(self, device):
        self.device = device

    def set_mode(self, train):
        if train:
            self.train()
        else:
            self.eval()

    def _forward(self, x, num_samples):
        z = self.encoder(x, num_samples)
        return z

    def svgd_kernel(self, theta, h=-1):
        XY = tc.mm(theta, theta.t())
        x2 = tc.sum(tc.pow(theta, 2), 1).unsqueeze(-1)
        x2e = x2.repeat(1, theta.shape[0])
        H = x2e + x2e.t() - 2 * XY

        V = H.flatten()
        h = tc.pow(tc.median(V), 2) / tc.log(
            tc.tensor(theta.shape[0], dtype=tc.float).to(self.device) + 1.0)
        Kxy = tc.exp(-H / h / 2)
        dKxy = -tc.mm(Kxy, theta)
        sumkxy = tc.sum(Kxy, 1).unsqueeze(-1)
        dKxy = -(1 / h) * tc.add(dKxy, tc.mul(theta, sumkxy))
        return (Kxy, dKxy)

    def svgd_kernel_tensor(self, theta, h=-1):
        XY = tc.matmul(theta, theta.transpose(1, 2))
        x2 = tc.sum(tc.pow(theta, 2), 2).unsqueeze(-1)
        x2e = x2.repeat(1, 1, theta.shape[1])

        H = x2e + x2e.transpose(1, 2) - 2 * XY
        V = H.flatten(2)
        h = tc.pow(tc.median(V), 2) / tc.log(
            tc.tensor(theta.shape[0], dtype=tc.float).to(self.device) + 1.0)
        h = 1
        Kxy = tc.exp(-H / h / 2)
        dKxy = -tc.matmul(Kxy, theta)
        sumkxy = tc.sum(Kxy, 1).unsqueeze(-1)
        dKxy = -(1 / h) * tc.add(dKxy, tc.mul(theta, sumkxy))
        return (Kxy, dKxy)

    def reconstruct(self, x, train, num_samples):
        self.set_mode(train)
        self.encoder.activate_dropout_layers()
        z = self._forward(x, num_samples)
        x_recon = self.decoder(z)
        return x_recon, z

    def loss(self, x, y=[], train=True, debug=False):
        x_recon, z = self.reconstruct(x, train, self.num_train_samples)
        num_particles = z.shape[1]
        if type(y) == list:
            y = x.repeat(1, num_particles, 1, 1)
        else:
            y = y.unsqueeze(1)
            y = y.repeat(1, num_particles, 1, 1, 1, 1)
            if debug:
                assert (x.data.cpu().numpy().all() >= 0.
                        and x.data.cpu().numpy().all() <= 1.)
                assert (x_recon.data.cpu().numpy().all() >= 0.
                        and x_recon.data.cpu().numpy().all() <= 1.
                        and not np.isnan(x_recon.data.cpu().numpy()).all())

        L = self.cross_entropy_loss(x_recon, y)
        if x.shape[0] == 1:  # The batch size is 1
            Kxy, dKxy = self.svgd_kernel(z.squeeze(0))
        else:
            Kxy, dKxy = self.svgd_kernel_tensor(z.squeeze())

        grad_score = tc.autograd.grad(L, z, retain_graph=True)[0].squeeze(0)
        svgd = tc.matmul(Kxy, grad_score) + self.regularization * dKxy
        return L, svgd, x_recon, z

    def test(self, x, y=[], num_samples=5):
        x_recon, z = self.reconstruct(x, False, num_samples)
        num_particles = z.shape[1]
        if type(y) == list:
            loss = 0
        else:
            y = y.unsqueeze(1)
            y = y.repeat(1, num_particles, 1, 1, 1, 1)
            loss = self.cross_entropy_loss(x_recon, y)

        return loss.item(), x_recon.squeeze().cpu().detach().numpy()

    def train_network(self, input, target):
        logpxz, svgd, _, z = self.loss(input, target)
        self.optimizer_enc.zero_grad()
        tc.autograd.backward(z.squeeze(), grad_tensors=svgd, retain_graph=True)
        self.optimizer_enc.step()
        self.optimizer_dec.zero_grad()
        logpxz.backward()
        self.optimizer_dec.step()
        return logpxz

    def setup_optimizers(self, lr=0.001):
        self.optimizer_enc = tc.optim.Adam(self.encoder.parameters(), lr=lr)
        self.optimizer_dec = tc.optim.Adam(self.decoder.parameters(), lr=lr)

    def cross_entropy_loss(self, x, y):
        cse_eps = .000001
        x = tc.clamp(x, cse_eps, 1.0 - cse_eps)
        return -tc.sum(y * tc.log(x) + (1 - y) * tc.log(1 - x))
