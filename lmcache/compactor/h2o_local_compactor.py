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


logger = init_logger(__name__)

class H2OCompactor(BaseLocalCompactor):
    """
    H2O compactor
    """
    
    def __init__(self, compactor_metadata):
        super().__init__(compactor_metadata)
        
        self.min_window_size = 900
        self.max_window_size = 1500
        self.evict_threshold = 1000
    
    def decide_compact(
        self,
        seq_len) -> bool:
        return seq_len >= self.max_window_size     
    
    def update_imp_scores(
        self,
        seq_id,
        idx,
        prefill_chunked_attention_weights,
        decode_chunked_attention_weights,
        attn_meta,
        is_prefill=False):
        """
        Simply add the attention_weight to the existing imp_scores
        """
        if is_prefill:
            start = attn_meta.context_lens_tensor[idx]
            seq_len = attn_meta.seq_lens[idx]
            end = start + seq_len
            num_prefills = attn_meta.num_prefills

            # self.prefill_logits_buffer_queue.put(chunked_attetnion_weights)
            for layer_idx in range(self.num_layers):
                attn_weight = prefill_chunked_attention_weights[layer_idx][idx]
                torch.set_printoptions(profile='full')
                # print('attn_weight[0]', attn_weight[0])
                # print('attn_weight.shape', attn_weight.shape)
                # seq_len = attn_weight.shape[1]
                # accu_attn_weight = torch.sum(attn_weight, dim=0)
                # print('accu_attn_weight', accu_attn_weight)

                # print('accu_attn_weight.shape', accu_attn_weight.shape)

                # print('seq_len', seq_len)
                # print('shapes1 ', self.imp_scores[seq_id].shape)
                # print('shapes', self.imp_scores[seq_id][layer_idx,:].shape)
                # check_sum = attn_weight.sum(dim=-1)
                # print('attn_weight', attn_weight)
                # print('check_sum', check_sum)
                # assert torch.allclose(check_sum, torch.ones_like(check_sum))
                # print('attn_weight', attn_weight)
                # print('attn_weight shapes', attn_weight.shape)
                attn_weight = attn_weight[:, :num_prefills, :seq_len].sum(dim=1)
                # print('attn_weight sum shapes', attn_weight.shape)
                # # print('attn_weight', attn_weight)
                # # check that attn_weight are all ones 
                # print('imp_scores', self.imp_scores[seq_id][layer_idx,:, start:end].shape)
                print(f"seq_len: {seq_len}")
                self.imp_scores[seq_id][layer_idx,:, :seq_len] += \
                    attn_weight
        
        else:
            for layer_idx in range(self.num_layers):
                attn_weight = decode_chunked_attention_weights[layer_idx][idx]
                seq_len = attn_weight.shape[1]
                # print('decode shapes')
                # print('shapes1 ', self.imp_scores[seq_id].shape)
                # print('shapes', self.imp_scores[seq_id][layer_idx,:].shape)
                # print('attn_weight shapes', attn_weight.shape)
                print(f"seq_len: {seq_len}")
                self.imp_scores[seq_id][layer_idx,:, :seq_len] += \
                    attn_weight
        # print('imp_scores', self.imp_scores[seq_id][2,5,:50])
        
    def adjust_positional_encoding(
        self,
        old_positions,
        new_positions,
        old_keys: torch.Tensor,
    ):
        """
        Not clearly mentioned in the paper. But seems to have better quality
        with `adjusting_positional_encoding`.
        """
        new_keys = self.reverse_rotary_emb(
            old_positions,
            new_positions,
            old_keys,
            is_reverse=False,
            is_fuse=True,
        )
        return new_keys
    
    def compute_indices(
        self,
        seq_id,
        seq_len,
    ):
        """
        compute indices for schedulers
        compact imp_scores
        """
        compacted_indices = []
        imp_score = self.imp_scores[seq_id]
        for layer_idx in range(self.num_layers):
            # sum of all heads
            sum_scores_layer = torch.sum(imp_score[layer_idx], dim=0)
            # print(f"shape of sum_scores_layer: {sum_scores_layer.shape}")
            imp_indices_layer = torch.topk(
                sum_scores_layer, k=self.min_window_size).indices
            imp_indices_layer = torch.sort(imp_indices_layer).values
            # print(f"shape of imp_indices_layer: {imp_indices_layer.shape}")
            # TODO: please get rid of this `tolist`
            imp_indices_layer = imp_indices_layer.tolist()
            compacted_indices.append(imp_indices_layer)

            # compact imp_scores
            imp_score[layer_idx,: , :self.min_window_size] = \
                imp_score[layer_idx, :, imp_indices_layer]
            imp_score[layer_idx,: , self.min_window_size:] = 0
        #import pdb
        #pdb.set_trace()
        return compacted_indices