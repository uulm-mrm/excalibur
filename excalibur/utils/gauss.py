from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class GaussConfig:
    mean: np.ndarray
    cov: np.ndarray
    dim: int
    height: float = 1.0
    normalize: bool = False

    def __init__(self, mean, cov, height=1.0, normalize=False):
        self.dim = len(mean)
        if not isinstance(cov, np.ndarray):
            cov = cov * np.eye(self.dim)
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.height = height
        self.normalize = normalize


def gauss_2d(x: np.ndarray, cfg: GaussConfig):
    # scaling
    scaling = cfg.height
    if cfg.normalize:
        scaling /= (np.sqrt((2 * np.pi) ** cfg.dim * np.linalg.det(cfg.cov)))

    # calculate
    cov_inv = np.linalg.inv(cfg.cov)
    return scaling * np.exp(-0.5 * np.einsum('ij,ji->i', (x - cfg.mean) @ cov_inv, (x - cfg.mean).T))


def multi_gauss_2d(x: np.ndarray, cfgs: List[GaussConfig]):
    y_stack = np.stack([gauss_2d(x, cfg) for cfg in cfgs])
    return np.sum(y_stack, axis=0)


def gauss_grad_2d(x: np.ndarray, cfg: GaussConfig):
    # scaling
    scaling = cfg.height
    if cfg.normalize:
        scaling /= (np.sqrt((2 * np.pi) ** cfg.dim * np.linalg.det(cfg.cov)))

    # calculate
    cov_inv = np.linalg.inv(cfg.cov)
    inner_grad = 2 * cov_inv @ (x - cfg.mean).T
    return scaling * inner_grad * np.exp(-0.5 * np.einsum('ij,ji->i', (x - cfg.mean) @ cov_inv, (x - cfg.mean).T))


def multi_gauss_grad_2d(x: np.ndarray, cfgs: List[GaussConfig]):
    y_stack = np.stack([gauss_grad_2d(x, cfg) for cfg in cfgs])
    return np.sum(y_stack, axis=0)
