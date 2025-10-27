import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate
from einops import rearrange

class AttnControl:
    def __init__(self):
        self.active = False

    def enable(self):
        self.active = True
    
    def disable(self):
        self.active = False

    def set_window_sizes(self, window_sizes):
        self.window_sizes = window_sizes

    def initialize(self, height_scale, width_scale):
        self.height_scale = height_scale
        self.width_scale = width_scale
        self.kv_views = {}
        for window_size in self.window_sizes:
            self.kv_views[window_size] = self.get_kv_view(window_size)

    def get_kv_view(self, window_size):
        h = int(self.height_scale * window_size)
        w = int(self.width_scale * window_size)
        p1 = int(window_size)
        p2 = window_size // 2
        p4 = window_size // 4
        r_indices = torch.arange(0, h, p2)
        c_indices = torch.arange(0, w, p2)
        r_win = torch.clamp(r_indices-p4, 0, h - p1)
        c_win = torch.clamp(c_indices-p4, 0, w - p1)

        # Collect patches using the sliding window approach.
        indices_list = []
        for r in r_win:
            for c in c_win:
               rs = torch.arange(r, r + p1)
               cs = torch.arange(c, c + p1)
               gh, gw = torch.meshgrid(rs, cs, indexing='ij')
               indices = gh * w + gw 
               indices = indices.flatten()
               indices_list.append(indices)
        indices_list = torch.stack(indices_list, dim=0)

        return indices_list
    
    def patchify_kv(self, kv, window_size):
        kv_views = self.kv_views[window_size].to(kv.device)
        kv = kv[:, :, kv_views, :]
        B, H, num_views, L, C = kv.shape
        kv = kv.permute(0, 2, 1, 3, 4).reshape(B*num_views, H, L, C)
        return kv
    
    def patchify_q(self, q, window_size):
        h = int(self.height_scale * window_size)
        w = int(self.width_scale * window_size)
        p2 = int(window_size) // 2
        return  rearrange(q, 'B H (nh p nw q) C -> (B nh nw) H (p q) C', nh=h//p2, nw=w//p2, p=p2, q=p2)
    
    def unpatchify(self, x, window_size):
        h = int(self.height_scale * window_size)
        w = int(self.width_scale * window_size)
        p2 = int(window_size) // 2
        return rearrange(x, '(B nh nw) H (p q) C -> B H (nh p nw q) C', nh=h//p2, nw=w//p2, p=p2, q=p2)

class AttnProcessor2_0_local:
    def __init__(self, controller:AttnControl, window_size):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0_local requires PyTorch 2.0. Please upgrade PyTorch to 2.0.")
        self.window_size = window_size
        self.controller = controller

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        ############################################## NPA ##############################################
        if self.controller.active == True:
            query = self.controller.patchify_q(query, self.window_size)
            key = self.controller.patchify_kv(key, self.window_size)
            value = self.controller.patchify_kv(value, self.window_size)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = self.controller.unpatchify(hidden_states, self.window_size)
        elif self.controller.active == False:
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            raise ValueError
        ##################################################################################################

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def register_attention_control(pipe):
    controller = AttnControl()
    attn_procs = {}
    window_size_lst = []
    for name in pipe.unet.attn_processors.keys():
        if name.startswith("down_blocks.1"):
            window_size = 64
        elif name.startswith("down_blocks.2"):
            window_size = 32
        elif name.startswith("mid_block"):
            window_size = 32
        elif name.startswith("up_blocks.0"):
            window_size = 32
        elif name.startswith("up_blocks.1"):
            window_size = 64
        else:
            raise ValueError("unexpected attention name")
        
        if window_size not in window_size_lst:
            window_size_lst.append(window_size)

        if name.endswith("attn1.processor"):
            attn_procs[name] = AttnProcessor2_0_local(controller, window_size)
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]

    pipe.unet.set_attn_processor(attn_procs)
    controller.set_window_sizes(window_size_lst)
    return controller