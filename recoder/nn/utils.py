import torch


def activation(x, act):
  if act == 'none':
    return x
  func = getattr(torch, act)
  return func(x)
