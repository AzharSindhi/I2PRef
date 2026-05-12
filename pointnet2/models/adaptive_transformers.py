import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Attention Modules
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        h = self.heads
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=h)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context):
        """x: (B, N, D) queries; context: (B, M, D) keys/values from image tokens."""
        h = self.heads
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.to_k(context), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.to_v(context), 'b n (h d) -> b h n d', h=h)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


# =============================================================================
# Refinement Block
# =============================================================================

class RefinementBlock(nn.Module):
    """Single refinement block: self-attention, cross-attention with image tokens, FFN, offset prediction."""

    def __init__(self, dim, heads=8, dim_head=32, dropout=0.0, ff_mult=4):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.self_attn = SelfAttention(dim, heads, dim_head, dropout)
        self.cross_attn = CrossAttention(dim, dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, ff_mult, dropout)
        self.offset_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3)
        )

    def forward(self, x, image_tokens):
        """
        x: (B, N, D) - point features
        image_tokens: (B, M, D) - image feature tokens
        Returns: x (B, N, D), offsets (B, N, 3)
        """
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), image_tokens)
        x = x + self.ff(self.norm3(x))
        return x, self.offset_head(x)


# =============================================================================
# Point Refinement Transformer
# =============================================================================

class AdaptivePointRefinementTransformer(nn.Module):
    """Iterative point cloud refinement using self- and cross-attention with image tokens."""

    def __init__(
        self,
        num_points=2048,
        input_dim=3,
        embed_dim=256,
        depth=6,
        heads=8,
        dim_head=32,
        dropout=0.0,
        ff_mult=4,
        scale_factor=1.0,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        self.point_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        self.positional_embedding = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        self.blocks = nn.ModuleList([
            RefinementBlock(dim=embed_dim, heads=heads, dim_head=dim_head,
                            dropout=dropout, ff_mult=ff_mult)
            for _ in range(depth)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, points, image_features):
        """
        Args:
            points: (B, N, 3) - initial point coordinates (sparse prediction from Image2Point)
            image_features: (B, M, D) - image feature tokens
        Returns:
            pts: (B, N, 3) - refined points
        """
        x = self.point_embedding(points) + self.positional_embedding(points)
        offsets = None
        for block in self.blocks:
            x, offsets = block(x, image_features)
        return points + offsets * self.scale_factor

