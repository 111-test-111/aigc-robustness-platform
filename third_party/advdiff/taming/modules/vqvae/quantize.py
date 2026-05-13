from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class VectorQuantizer2(nn.Module):
    """Small inference-compatible VectorQuantizer2 used by LDM VQModel.

    This mirrors the API that AdvDiff's vendored LDM code needs for the
    ImageNet class-conditional checkpoint: ``forward``, ``embed_code``, and
    ``get_codebook_entry``. Training-only remapping behavior is intentionally
    unsupported because the platform only runs AdvDiff inference.
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float,
        remap=None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ) -> None:
        super().__init__()
        if remap is not None:
            raise ValueError("Offline AdvDiff inference does not support remapped codebooks.")
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.sane_index_shape = sane_index_shape
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        if rescale_logits or return_logits:
            raise ValueError("VectorQuantizer2 logits are not available in inference mode.")

        z_bhwc = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z_bhwc.view(-1, self.e_dim)

        distances = (
            torch.sum(z_flattened.pow(2), dim=1, keepdim=True)
            + torch.sum(self.embedding.weight.pow(2), dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))
        )

        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z_bhwc.shape)

        if self.legacy:
            loss = torch.mean((z_q.detach() - z_bhwc).pow(2)) + self.beta * torch.mean(
                (z_q - z_bhwc.detach()).pow(2)
            )
        else:
            loss = self.beta * torch.mean((z_q.detach() - z_bhwc).pow(2)) + torch.mean(
                (z_q - z_bhwc.detach()).pow(2)
            )

        z_q = z_bhwc + (z_q - z_bhwc).detach()
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.view(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (None, None, min_encoding_indices)

    def embed_code(self, embed_id):
        return self.embedding(embed_id)

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            if len(shape) == 4:
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
