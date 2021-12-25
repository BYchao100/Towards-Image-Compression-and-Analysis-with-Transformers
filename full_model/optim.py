from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


def configure_optimizers(args, model):

    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    if args.opt =="adam":
        optimizer = optim.Adam(
            (params_dict[n] for n in sorted(parameters)),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.opt_eps,
        )
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.aux_lr,
            weight_decay=args.weight_decay,
            eps=args.opt_eps,
        )

    return optimizer, aux_optimizer
