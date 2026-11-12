"""Memory-saving patches for long sequences give the same results as the original implementations."""

import copy

import torch

from layoutlmv3_pretraining.long_docs import use_sdpa_attention


def tiny_layoutlmv3():
    from transformers import LayoutLMv3Config, LayoutLMv3Model

    torch.manual_seed(0)
    cfg = LayoutLMv3Config(num_hidden_layers=2, hidden_size=96, coordinate_size=16, shape_size=16, num_attention_heads=4,
                           intermediate_size=128, visual_embed=False, hidden_dropout_prob=0.0,
                           attention_probs_dropout_prob=0.0)
    return LayoutLMv3Model(cfg)


def test_sdpa_attention_matches_eager_outputs_and_gradients():
    eager = tiny_layoutlmv3()
    sdpa = use_sdpa_attention(copy.deepcopy(eager))
    ids = torch.randint(5, 100, (2, 24))
    bbox = torch.randint(0, 500, (2, 24, 4))
    bbox[..., 2:] += bbox[..., :2]
    mask = torch.ones(2, 24, dtype=torch.long)
    mask[1, 15:] = 0                                            # padding in the second example
    outs = []
    for model in (eager, sdpa):
        model.train()                                           # dropout is 0, so train mode is deterministic
        out = model(input_ids=ids, bbox=bbox, attention_mask=mask).last_hidden_state
        (out * mask[..., None]).sum().backward()
        outs.append((out.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (o1, g1), (o2, g2) = outs
    valid = mask.bool()
    assert torch.allclose(o1[valid], o2[valid], atol=1e-5, rtol=1e-4)
    assert g1.keys() == g2.keys()
    for name in g1:
        assert torch.allclose(g1[name], g2[name], atol=1e-5, rtol=1e-4), name


def test_checkpoint_encoder_layers_keeps_gradients():
    from layoutlmv3_pretraining.long_docs import checkpoint_encoder_layers

    plain = tiny_layoutlmv3()
    ckpt = checkpoint_encoder_layers(copy.deepcopy(plain))
    ids = torch.randint(5, 100, (2, 16))
    bbox = torch.randint(0, 500, (2, 16, 4))
    bbox[..., 2:] += bbox[..., :2]
    grads = []
    for model in (plain, ckpt):
        model.train()
        model(input_ids=ids, bbox=bbox).last_hidden_state.sum().backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        assert torch.allclose(grads[0][name], grads[1][name], atol=1e-6), name
