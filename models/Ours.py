import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepKoopman(nn.Module):

    def __init__(self, state_dim, control_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.eye_(self.K.weight)
        self.B = nn.Linear(control_dim, latent_dim, bias=False) if control_dim > 0 else None

    def forward(self, x, u=None):
        B, L, _ = x.shape
        z = self.encoder(x)

        z_next_pred = z[:, :-1, :] + self.K(z[:, :-1, :])
        if self.B is not None and u is not None:
            z_next_pred = z_next_pred + self.B(u[:, :-1, :])

        x_next_pred = self.decoder(z_next_pred)
        x_rec = self.decoder(z)

        pred_x = torch.cat([x[:, :1, :], x_next_pred], dim=1)
        pred_z = torch.cat([z[:, :1, :], z_next_pred], dim=1)

        return pred_x, x_rec, z, pred_z


class MRARec(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mem_dim: int,
        num_prototypes_per_scale: int = 32,
        commitment_cost: float = 0.25,
        scales: list = None,
    ):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4, 8]
        self.scales = scales
        self.num_scales = len(scales)
        self.mem_dim = mem_dim
        self.commitment_cost = commitment_cost

        self.convs = nn.ModuleList()
        for r in scales:
            self.convs.append(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3,
                          padding=r, dilation=r)
            )

        self.scale_projections = nn.ModuleList()
        for _ in scales:
            self.scale_projections.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, mem_dim),
                )
            )

        self.codebooks = nn.ParameterList()
        for _ in scales:
            cb = nn.Parameter(torch.empty(num_prototypes_per_scale, mem_dim))
            nn.init.xavier_uniform_(cb)
            self.codebooks.append(cb)

        self.scale_embeddings = nn.Parameter(
            torch.empty(self.num_scales, mem_dim)
        )
        nn.init.xavier_uniform_(self.scale_embeddings)

        self.W_query = nn.Linear(mem_dim, mem_dim, bias=False)
        self.W_key = nn.Linear(mem_dim, mem_dim, bias=False)
        self.W_value = nn.Linear(mem_dim, mem_dim, bias=False)

        decoder_input_dim = mem_dim * self.num_scales
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        B, L, _ = x.shape

        x_t = x.transpose(1, 2)

        H_raw = []
        for i, conv in enumerate(self.convs):
            h = F.gelu(conv(x_t))
            h = h.transpose(1, 2)
            z = self.scale_projections[i](h)
            H_raw.append(z)

        H_q = []
        vq_loss = 0.0

        for i, z in enumerate(H_raw):
            codebook = self.codebooks[i]

            z_flat = z.reshape(-1, self.mem_dim)
            z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)
            p_sq = (codebook ** 2).sum(dim=1)
            dist = z_sq + p_sq.unsqueeze(0) - 2 * z_flat @ codebook.t()

            idx = torch.argmin(dist, dim=1)
            z_q_flat = codebook[idx]
            z_q = z_q_flat.reshape(B, L, self.mem_dim)

            vq_loss = vq_loss + F.mse_loss(z, z_q.detach()) \
                    + self.commitment_cost * F.mse_loss(z.detach(), z_q)

            z_q_st = z + (z_q - z).detach()
            H_q.append(z_q_st)

        all_keys, all_values = [], []
        for j, cb in enumerate(self.codebooks):
            s_j = self.scale_embeddings[j]
            all_keys.append(self.W_key(cb + s_j.unsqueeze(0)))
            all_values.append(self.W_value(cb))

        K = torch.cat(all_keys, dim=0)
        V = torch.cat(all_values, dim=0)

        H_enhanced = []
        for i, Hr in enumerate(H_q):
            s_i = self.scale_embeddings[i]

            Q_r = self.W_query(Hr + s_i.view(1, 1, -1))

            scale = self.mem_dim ** 0.5
            attn = F.softmax((Q_r @ K.t()) / scale, dim=-1)
            R_r = attn @ V

            H_enhanced.append(Hr + R_r)

        H_fused = torch.cat(H_enhanced, dim=-1)
        x_rec = self.decoder(H_fused)

        return x_rec, vq_loss


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


class WaveletVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, scales=None):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4, 8]
        self.scales = scales
        self.vae = VAE(input_dim * len(scales), hidden_dim, latent_dim)

    def morlet_cwt(self, x):
        x = x.transpose(1, 2)
        C = x.size(1)

        omega0 = 5.0
        c = 0.75112554
        base_win = 7

        scale_mags = []
        for s in self.scales:
            window_size = base_win * s
            if window_size % 2 == 0:
                window_size += 1

            t = torch.linspace(-(window_size // 2), window_size // 2,
                               window_size, device=x.device, dtype=x.dtype)
            t_s = t / s
            norm = 1.0 / (s ** 0.5)
            real = norm * c * torch.exp(-0.5 * t_s ** 2) * torch.cos(omega0 * t_s)
            imag = norm * c * torch.exp(-0.5 * t_s ** 2) * torch.sin(omega0 * t_s)

            w_real = real.view(1, 1, window_size).repeat(C, 1, 1)
            w_imag = imag.view(1, 1, window_size).repeat(C, 1, 1)

            pad = window_size // 2
            out_r = F.conv1d(x, w_real, stride=1, padding=pad, groups=C)
            out_i = F.conv1d(x, w_imag, stride=1, padding=pad, groups=C)

            mag = torch.sqrt(out_r ** 2 + out_i ** 2 + 1e-8)
            scale_mags.append(mag)

        out = torch.stack(scale_mags, dim=-1)
        out = out.permute(0, 2, 1, 3)
        B, L, C_, S = out.shape
        return out.reshape(B, L, C_ * S)

    def forward(self, x):
        x_dwt = self.morlet_cwt(x)
        recon_dwt, mu, logvar = self.vae(x_dwt)
        return x_dwt, recon_dwt, mu, logvar


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.task_name = getattr(configs, "task_name", "anomaly_detection")
        self.seq_len = int(getattr(configs, "seq_len", 100))
        self.enc_in = int(getattr(configs, "enc_in", 1))
        self.c_out = int(getattr(configs, "c_out", self.enc_in))

        self.d_model = int(getattr(configs, "d_model", 128))
        self.latent_dim = int(getattr(configs, "latent_dim", 64))

        self.is_rflymad = (getattr(configs, "data", "RflyMAD") == "RflyMAD")
        self.is_alfa = (getattr(configs, "data", "") == "ALFA")
        if self.is_rflymad:
            self.control_dim = 8
        elif self.is_alfa:
            self.control_dim = 5
        else:
            self.control_dim = 0
        self.state_dim = self.enc_in - self.control_dim

        self.koopman = DeepKoopman(
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            hidden_dim=self.d_model,
            latent_dim=self.latent_dim,
        )

        self.mra_rec = MRARec(
            input_dim=self.state_dim,
            hidden_dim=self.d_model,
            mem_dim=128,
            num_prototypes_per_scale=32,
            commitment_cost=0.25,
            scales=[1, 2, 4, 8],
        )

        self.wavelet_vae = WaveletVAE(
            input_dim=self.enc_in,
            hidden_dim=self.d_model,
            latent_dim=64,
        )

        self.output_projection = (
            nn.Identity() if self.c_out == self.enc_in
            else nn.Linear(self.enc_in, self.c_out)
        )

        self.latest_M = None
        self.latest_P = None
        self.latest_z = None
        self.latest_pred_z = None
        self.latest_state_rec = None
        self.latest_koopman_pred = None
        self.latest_vq_loss = None
        self.latest_dwt_P = None
        self.latest_recon_dwt_P = None
        self.latest_mu_P = None
        self.latest_logvar_P = None

    def reconstruct(self, x_enc, x_mark_enc=None):
        if self.control_dim > 0:
            u = x_enc[:, :, :self.control_dim]
            state = x_enc[:, :, self.control_dim:]
        else:
            u = None
            state = x_enc

        pred_state, state_rec, z, pred_z = self.koopman(state, u)

        pred_state_rec, vq_loss = self.mra_rec(pred_state)

        if self.control_dim > 0:
            m_main = torch.cat([u, pred_state_rec], dim=-1)
            koopman_main = torch.cat([u, pred_state], dim=-1)
        else:
            m_main = pred_state_rec
            koopman_main = pred_state

        out_mra = self.output_projection(m_main)
        out_koopman = self.output_projection(koopman_main)

        out = out_mra - out_koopman + x_enc

        P = x_enc - out_koopman

        dwt_P, recon_dwt_P, mu_P, logvar_P = self.wavelet_vae(P)

        error_time = torch.mean(torch.abs(out_mra - out_koopman), dim=-1)

        error_freq = torch.mean(torch.abs(dwt_P - recon_dwt_P), dim=-1)
        if error_freq.shape[1] > error_time.shape[1]:
            error_freq = error_freq[:, :error_time.shape[1]]
        elif error_freq.shape[1] < error_time.shape[1]:
            pad_len = error_time.shape[1] - error_freq.shape[1]
            error_freq = F.pad(error_freq, (0, pad_len), mode='replicate')

        self.latest_M = m_main
        self.latest_P = P
        self.latest_z = z
        self.latest_pred_z = pred_z
        self.latest_state_rec = state_rec
        self.latest_koopman_pred = pred_state
        self.latest_vq_loss = vq_loss
        self.latest_dwt_P = dwt_P
        self.latest_recon_dwt_P = recon_dwt_P
        self.latest_mu_P = mu_P
        self.latest_logvar_P = logvar_P

        return out, error_time, error_freq

    def anomaly_detection(self, x_enc, x_mark_enc=None):
        return self.reconstruct(x_enc, x_mark_enc)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc, x_mark_enc)
        return self.reconstruct(x_enc, x_mark_enc)
