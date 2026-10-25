"""Causal GRU and LFADS-inspired stochastic latent-dynamics encoders."""
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, kind='dynamics', hidden=256):
        super().__init__()
        self.kind = kind
        self.adapter = nn.Linear(256, 256)
        nn.init.eye_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        self.readin = nn.Linear(256, 128)
        self.inference = nn.GRU(128, hidden, batch_first=True)
        self.input_posterior = nn.Linear(hidden, 64)
        self.generator = nn.GRU(32, hidden, batch_first=True)
        self.factors = nn.Linear(hidden, 64)
        self.head = nn.Linear(64, 41)
        self.count_readout = nn.Linear(64, 128)
        self.power_readout = nn.Linear(64, 128)
        self.next_readout = nn.Linear(64, 256)
        if kind == 'gru':
            # Two causal GRUs; control parameters remain genuinely used.
            self.generator = nn.GRU(hidden, hidden, batch_first=True)
            del self.input_posterior
        elif kind == 'supervised':
            # Same architecture as dynamics, without generative objectives.
            pass
        elif kind not in ('dynamics', 'predictive'):
            raise ValueError(kind)

    def forward(self, x, sample=False):
        inferred, _ = self.inference(torch.tanh(self.readin(self.adapter(x))))
        kl = x.new_zeros(x.shape[:2])
        if self.kind == 'gru':
            generated, _ = self.generator(inferred)
        else:
            mean, logvar = self.input_posterior(inferred).chunk(2, -1)
            logvar = logvar.clamp(-8, 4)
            inputs = mean + torch.exp(.5*logvar)*torch.randn_like(mean) if sample else mean
            generated, _ = self.generator(inputs)
            kl = .5*(mean.square()+logvar.exp()-1-logvar).mean(-1)
        factors = self.factors(generated)
        # CTC emits every 80 ms, using bins ending at 3,7,... only.
        logits = self.head(factors[:, 3::4])
        return {'logits': logits, 'factors': factors, 'kl': kl,
                'count_lograte': self.count_readout(factors).clamp(-8, 8),
                'power_mean': self.power_readout(factors),
                'next_mean': self.next_readout(factors)}


def objective(model, batch, auxiliary=.1, step=0):
    x, raw, targets, lengths, target_lengths = batch
    out = model(x, sample=model.training and model.kind in ('dynamics', 'predictive'))
    logp = F.log_softmax(out['logits'], -1).transpose(0, 1)
    # CPU CTC supports autograd from MPS; never silently discard impossible paths.
    ctc = F.ctc_loss(logp.cpu(), targets.cpu(), (lengths//4).cpu(),
                     target_lengths.cpu(), blank=40, zero_infinity=False)
    loss = ctc
    reconstruction = x.new_zeros(())
    prediction = x.new_zeros(())
    if model.kind in ('dynamics', 'predictive'):
        mask = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        count = F.poisson_nll_loss(out['count_lograte'], raw[:, :, :128],
                                  log_input=True, full=False, reduction='none').mean(-1)
        # Fixed variance Gaussian on log-power; channel standardization matches features.
        power = (out['power_mean']-torch.log1p(raw[:, :, 128:])).square().mean(-1)
        reconstruction = ((count+power)[mask]).mean()
        kl = out['kl'][mask].mean()
        warmup = min(1., step/1000.)
        loss = loss + auxiliary*(reconstruction+.01*warmup*kl).cpu()
        if model.kind == 'predictive':
            valid = mask[:, 1:]
            prediction = ((out['next_mean'][:, :-1]-x[:, 1:]).square().mean(-1))[valid].mean()
            loss = loss + auxiliary*prediction.cpu()
    return loss, {'ctc': float(ctc.detach()), 'reconstruction': float(reconstruction.detach()),
                  'prediction': float(prediction.detach())}
