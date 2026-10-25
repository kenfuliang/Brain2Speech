"""Prospective full-recipe port controls, not paper-performance evidence."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
from train_bit_full import effective_config, collapse, forward, upstream


def test_effective_source_config_repeatable():
    first=effective_config(800); second=effective_config(800)
    assert first == second
    assert first.training.num_epochs == 800
    assert first.seed == 42 and first.training.train_batch_size == 32
    assert first.model.encoder.embedder.masker.active is False
    assert first.model.encoder.embedder.max_channels == 256
    assert first.model.encoder.embedder.max_F == 1250


def test_blank_separator_source_deviation():
    assert collapse([1,0,1]) == [1,1]
    assert collapse([1,0,1],source=True) == [1]
    assert collapse([0,1,1,0,2,2]) == [1,2]


def test_bfloat_cpu_ctc_backward_and_scheduler_state():
    torch.set_num_threads(2)
    cfg=effective_config(800)
    net=upstream.NDT(cfg.model,method_name='ctc',features='all',vocab_size=41,
                     blank_id=0,zero_infinity=True).float()
    rows=[{'id':'x','phonemes':[1,2,3]}]
    loss,_,lengths=forward(net,rows,{'x':np.ones((50,256),np.float32)},'cpu')
    loss.backward()
    assert torch.isfinite(loss) and lengths.tolist()==[10]
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    optimizer=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-5,eps=1e-8)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=800*275,
                                                  max_lr=5e-4,pct_start=0.,div_factor=25)
    optimizer.step();scheduler.step()
    state=scheduler.state_dict()
    restored=torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=800*275,
                                                max_lr=5e-4,pct_start=0.,div_factor=25)
    restored.load_state_dict(state)
    assert restored.state_dict()==state
