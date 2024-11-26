import abc
from typing import Tuple, List, Dict
import torch
import queue

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.backends.utils import compute_slot_mapping
from vllm import _custom_ops as ops

from lmcache.compactor.base_local_compactor import BaseLocalCompactor
from lmcache.compactor.utils import CompactorOutput
from lmcache.logging import init_logger


class SinkCompactor(BaseLocalCompactor):
    """
    SteamingLLM-like compactor
    Always retain the first 4 tokens (attention sinks)
    """
    def __init__(self, compactor_metadata):
        super().__init__(compactor_metadata)
        
        self.min_window_size = 300
        self.max_window_size = 512
        self.num_sink = 4
        
        
    
    def decide_compact(
        self,
        seq_len) -> bool:
        if seq_len >= self.max_window_size:
            return True
        return False
    
    def update_imp_scores(
        self,
        seq_id,
        idx,
        chunked_attetnion_weights):
        """
        No `imp_scores` for AttentionSink
        Do nothing
        """
        pass
    
    def adjust_positional_encoding(
        self,
        old_positions,
        new_positions,
        old_keys: torch.Tensor,
        src_slot_mapping_layer,
    ):
        """
        reverse and recover the positional encoding
        """
        
        num_tok = len(src_slot_mapping_layer)
        reshaped_keys = old_keys[src_slot_mapping_layer].reshape(num_tok, -1)
        dumb_q = torch.zeros(reshaped_keys.shape,
                             device=old_keys.device,
                             dtype=old_keys.dtype)
        
        dumb_q, no_pos_keys = self.reverse_rotary_emb(
            torch.tensor(old_positions).to(device=old_keys.device,
                             dtype=torch.long),
            dumb_q,
            old_keys)
        
        dumb_q, new_keys = self.rotary_emb(
            torch.tensor(new_positions).to(device=old_keys.device,
                             dtype=torch.long),
            dumb_q,
            no_pos_keys)
        
        # should return old_keys as rotary_emb is inplace operation
        return old_keys
    
    
    def compute_indices(self, seq_id, seq_len):
        """
        
        """
        num_last = self.min_window_size - self.num_sink
        
        sink_indices = [i for i in range(self.num_sink)]
        last_indices = [i for i in range(seq_len - num_last,
                                         seq_len)]
        compacted_indices = [sink_indices + last_indices \
            for i in range(self.num_layers)]
        
        return compacted_indices