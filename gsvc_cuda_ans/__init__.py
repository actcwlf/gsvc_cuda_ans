import torch
import numpy as np
from . import _C


class ANSCoder:
    def __init__(self, min_symbol, max_symbol, precision=16):
        self.coder = _C.CudaAnsCoder(min_symbol, max_symbol, precision)
        if precision <= 16:
            self.dtype = np.uint16
        elif precision <= 32:
            self.dtype = np.uint32
        else:
            raise ValueError('precision exceeds 32')

    def encode(self, file: str, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):

        r_msg = torch.flip(x.view(-1), dims=(0,)).cpu().numpy().astype(dtype=np.int32)
        r_mu = torch.flip(mu.view(-1), dims=(0,)).cpu().numpy().astype(dtype=np.float32)
        r_sigma = torch.flip(sigma.view(-1), dims=(0,)).cpu().numpy().astype(dtype=np.float32)

        compressed: np.ndarray = self.coder.encode_reverse(
            r_msg, r_mu, r_sigma

        )

        compressed = compressed.astype(self.dtype)

        compressed.tofile(file)
        return compressed

    def decode(self, file: str, mu: torch.Tensor, sigma: torch.Tensor):
        compressed = np.fromfile(file, dtype=self.dtype)
        mu = mu.view(-1).cpu().numpy().astype(np.float32)
        sigma = sigma.view(-1).cpu().numpy().astype(np.float32)

        decoded_x = self.coder.decode(compressed.astype(np.uint32), mu, sigma)

        return torch.Tensor(decoded_x)