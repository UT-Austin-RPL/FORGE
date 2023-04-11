import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from einops import rearrange, repeat
from typing import Optional


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, temporal_size, cls_token=False):
    def get_emb(sin_inp):
        '''Gets a base embedding for one dimension with sin and cos intertwined'''
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    channels = int(np.ceil(embed_dim / 6) * 2)
    if channels % 2:
        channels += 1
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    pos_h = torch.arange(grid_size).float()
    pos_w = torch.arange(grid_size).float()
    pos_t = torch.arange(temporal_size).float()
    sin_inp_h = torch.einsum("i,j->ij", pos_h, inv_freq)
    sin_inp_w = torch.einsum("i,j->ij", pos_w, inv_freq)
    sin_inp_t = torch.einsum("i,j->ij", pos_t, inv_freq)
    emb_h = get_emb(sin_inp_h)
    emb_w = get_emb(sin_inp_w).unsqueeze(1)
    emb_t = get_emb(sin_inp_t).unsqueeze(1).unsqueeze(1)

    emb = torch.zeros(temporal_size, grid_size, grid_size, channels * 3).float()
    emb[:,:,:,:channels] = emb_t
    emb[:,:,:,channels: 2*channels] = emb_w
    emb[:,:,:,2*channels:] = emb_h
    emb = emb.reshape(-1,channels * 3)

    if cls_token:
        emb = torch.cat([torch.zeros(1, channels * 3).float(), emb], dim=0)
    return emb[:,:embed_dim]


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = (
            torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc
    


class Block(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, return_attn=False):
        super().__init__()

        self.channels = dim

        self.encode_query = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        self.attn = Attention(dim, num_heads=num_heads)
        
        self.encode_value = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.norm = norm_layer(dim)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor)
    
    def get_attn(self, query, key, query_embed=None, key_embed=None):
        b, c, n = query.shape

        q = self.with_pos_embed(query, query_embed)
        k = self.with_pos_embed(key, key_embed)

        q = self.norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.norm(k.permute(0, 2, 1)).permute(0, 2, 1)

        q = self.encode_query(q).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)
        k = self.encode_key(k).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)
        return self.attn.get_attn(query=q, key=k)   # [b,n,n]
    
    def forward(self, query, key, query_embed=None, key_embed=None):
        b, c, n = query.shape

        q = self.with_pos_embed(query, query_embed)
        k = self.with_pos_embed(key, key_embed)

        q = self.norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.norm(k.permute(0, 2, 1)).permute(0, 2, 1)

        v = self.encode_value(key).view(b, self.channels, -1)
        v = v.permute(0, 2, 1)

        q = self.encode_query(q).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)

        k = self.encode_key(k).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)

        query = query.view(b, self.channels, -1).permute(0, 2, 1)
        query = query + self.attn(query=q, key=k, value=v)

        query = query + self.mlp(self.norm2(query))
        query = query.permute(0, 2, 1).contiguous().view(b, self.channels, -1)

        return query


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        #self.scale = 10 #head_dim ** -0.5

    def get_attn(self, query, key):
        B, N, C = query.shape
        attn = torch.matmul(query, key.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        #__import__('pdb').set_trace()
        return attn
    
    def forward(self, query, key, value):
        B, N, C = query.shape
        query = query.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(query, key.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, value).transpose(1, 2).reshape(B, N, C)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.
        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of channels query and key input channels are projected to,
            for computing the attention matrix. Defaults to number `num_q_input_channels`
        :param num_v_channels: Number of channels value input channels are projected to.
            Defaults to `num_qk_channels`.
        :param num_output_channels: Number of output channels attention result channels are projected to.
            Defaults to `num_q_input_channels`
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, pad_mask=None, attn_mask=None):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param attn_mask: Boolean attention mask. Not needed/supported yet.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """
        if attn_mask is not None:
            raise NotImplementedError("attention masks not supported yet")

        q = self.q_proj(x_q)
        k = self.k_proj(x_k)
        v = self.v_proj(x_v)

        #print(q.shape, k.shape, v.shape)
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        attn = torch.einsum("b i c, b j c -> b i j", q, k) * self.dp_scale

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, "b j -> (b h) () j", h=self.num_heads)
            attn_max_neg = -torch.finfo(attn.dtype).max
            attn.masked_fill_(pad_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        o = torch.einsum("b i j, b j c -> b i c", attn, v)
        o = rearrange(o, "(b h) n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)
    

class MLP_attention(nn.Module):
    def __init__(self, num_channels: int, widening_factor: int):
        super().__init__()
        self.mlp = nn.Sequential(*[
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels),
            ])

    def forward(self, x):
        return self.mlp(x)
    

class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        mlp_ratio: int = 1,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head cross-attention (see `MultiHeadAttention` for details)."""
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.k_norm = nn.LayerNorm(num_kv_input_channels)
        self.v_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )
        self.mlp = MLP_attention(num_channels=num_q_input_channels, widening_factor=mlp_ratio)

    def forward(self, x_q, x_k, x_v, pad_mask=None, attn_mask=None, residual=False):
        x_q = self.q_norm(x_q)
        x_k = self.k_norm(x_k)
        x_v = self.v_norm(x_v)
        if torch.is_tensor(residual):
            x_q = residual + self.attention(x_q, x_k, x_v, pad_mask=pad_mask, attn_mask=attn_mask)
        elif residual == True:
            x_q = x_q + self.attention(x_q, x_k, x_v, pad_mask=pad_mask, attn_mask=attn_mask)
        else:
            x_q = self.attention(x_q, x_k, x_v, pad_mask=pad_mask, attn_mask=attn_mask)
        x_q = self.mlp(x_q)
        return x_q


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        mlp_ratio: int = 1,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head self-attention (see `MultiHeadAttention` and for details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )
        self.mlp = MLP_attention(num_channels=num_channels, widening_factor=mlp_ratio)

    def forward(self, x, pad_mask=None, attn_mask=None):
        """Multi-head attention of input `x` to itself after applying layer normalization to the input."""
        x = self.norm(x)
        x = x + self.attention(x, x, x, pad_mask=pad_mask, attn_mask=attn_mask)
        x = self.mlp(x)
        return x


class SelfAttentionBlocks(nn.Module):
    def __init__(self, depth, num_heads, num_channels, mlp_ratio):
        super().__init__()
        self.attention = nn.Sequential(*[
            SelfAttention(num_heads, num_channels, mlp_ratio)
            for i in range(depth)])

    def forward(self, x):
        return self.attention(x)