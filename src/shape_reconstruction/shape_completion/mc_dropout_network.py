import torch as tc
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self,
                 latent_space=2000,
                 merge_mode='concat',
                 drop_rate=0.5,
                 dall=True,
                 use_batch_norm=True):
        super(EncoderDecoder, self).__init__()
        self.merge_mode = merge_mode
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

        d_s = [2, 2, 2]
        d_p = [0, 1, 0]

        if 'concat' in merge_mode:
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
        # TODO: Fix batch norm
        for p in range(2, len(c_e)):
            if dall:
                e_convs.append(
                    nn.Sequential(
                        nn.Conv3d(c_e[p - 1], c_e[p], kernel_size=4, stride=2),
                        nn.LeakyReLU(), nn.Dropout3d(drop_rate)))
            else:
                e_convs.append(
                    nn.Sequential(
                        nn.Conv3d(c_e[p - 1], c_e[p], kernel_size=4, stride=2),
                        nn.LeakyReLU()))

            if 'concat' in merge_mode:
                if dall:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(2 * c_e[p],
                                               c_e[p - 1],
                                               kernel_size=4,
                                               stride=d_s[p - 1],
                                               output_padding=d_p[p - 1]),
                            nn.ReLU(), nn.Dropout3d(drop_rate)))
                else:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(2 * c_e[p],
                                               c_e[p - 1],
                                               kernel_size=4,
                                               stride=d_s[p - 1],
                                               output_padding=d_p[p - 1]),
                            nn.ReLU()))
            else:
                if dall:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(c_e[p],
                                               c_e[p - 1],
                                               kernel_size=4,
                                               stride=d_s[p - 1],
                                               output_padding=d_p[p - 1]),
                            nn.ReLU(), nn.Dropout3d(drop_rate)))
                else:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(c_e[p],
                                               c_e[p - 1],
                                               kernel_size=4,
                                               stride=d_s[p - 1],
                                               output_padding=d_p[p - 1]),
                            nn.ReLU()))
        d_convs.reverse()
        self.e_convs = nn.ModuleList(e_convs)
        self.d_convs = nn.ModuleList(d_convs)
        # the latent space consist of 2048 parameters
        fc = [nn.Linear(c_e[-1] * 3 * 3 * 3, latent_space)]  # c_e[-1]*3*3*3)]
        fc.append(nn.Dropout3d(drop_rate))
        fc.append(nn.ReLU())
        fc.append(nn.Linear(latent_space, c_e[-1] * 3 * 3 * 3))
        fc.append(nn.Dropout3d(drop_rate))
        fc.append(nn.ReLU())
        self.fc = nn.ModuleList(fc)

    def is_dropout(self, layer):
        return type(layer) == nn.Dropout

    def encoder(self, x):
        for i in range(len(self.e_convs)):
            x = self.e_convs[i](x)
        return x

    def latent(self, x):
        preshape = x.shape
        x = x.view(x.size(0), -1)
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        return x.view(preshape)

    def decoder(self, x):
        for i in range(len(self.d_convs)):
            x = self.d_convs[i](x)
        return x

    def forward(self, x):
        encoder_outs = []
        for i in range(len(self.e_convs)):
            x = self.e_convs[i](x)
            encoder_outs.append(x)
        x = self.latent(x)
        for i in range(len(self.d_convs)):
            if 'concat' in self.merge_mode:
                skip_con = encoder_outs[-(i + 1)]
                x = tc.cat((x, skip_con), 1)
                x = self.d_convs[i](x)
            else:
                x = self.d_convs[i](x)
        return x


# network is a string pointing to either the endoer or decoder. layers is a list of numbers where each number correpsonds to the dropout layers to activate.

    def activate_dropout_layers(self):
        for i in self.modules():
            if self.is_dropout(i):
                i.train()

    def sample_and_mean(self, x, num_samples):
        Y = []
        for i in range(num_samples):
            Y.append(self._forward(x))
        Y = tc.stack(Y)
        return Y, Y.mean(dim=0)


class MCDropoutEncoderDecoder(nn.Module):
    def __init__(self, encoder_decoder):
        super(MCDropoutEncoderDecoder, self).__init__()
        self.device = "cpu"
        self.encoder_decoder = encoder_decoder
        self.optimizer = ""
        self.setup_optimizer()

    def set_mode(self, train):
        if train:
            self.train()
        else:
            self.eval()

    def set_device(self, device):
        self.device = device

    def _forward(self, x):
        y = self.encoder_decoder.forward(x)
        return y

    def train_network(self, input, target):
        self.optimizer.zero_grad()

        output = self._forward(input)
        loss = self.cross_entropy_loss(output, target)

        running_loss = loss
        loss.backward()
        self.optimizer.step()
        return running_loss

    def setup_optimizer(self, lr=0.001):
        self.optimizer = tc.optim.Adam(self.encoder_decoder.parameters(),
                                       lr=lr)

    def sample(self, x, num_samples):
        Y = []
        for i in range(num_samples):
            Y.append(self._forward(x))
        Y = tc.stack(Y)
        tc.cuda.empty_cache()
        return Y

    def test(self, x, y=[], num_samples=10):
        samples = self.sample(x, num_samples)
        return 0, samples.squeeze().cpu().detach().numpy()

    def cross_entropy_loss(self, x, y):
        cse_eps = .000001
        x = tc.clamp(x, cse_eps, 1.0 - cse_eps)
        return -tc.sum(y * tc.log(x) + (1 - y) * tc.log(1 - x))
