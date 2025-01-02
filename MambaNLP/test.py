import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch
import transformers
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from lm_eval.models.huggingface import HFLM
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import matplotlib.pyplot as plt
def normalize_attn_mat(attn_mat):
    return (attn_mat - torch.min(attn_mat)) / (torch.max(attn_mat) - torch.min(attn_mat))

class Mamba2EvalWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    def __init__(self, pretrained="state-spaces/mamba2-1.3b", max_length=2048, batch_size=None, device="cuda",dtype=torch.float16):
        LM.__init__(self)
        print("initializing mamba2")
        self._model = MambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

# class MambaEvalWrapper(HFLM):
#     AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
#     def __init__(self, pretrained="state-spaces/mamba-2.8b", max_length=2048, batch_size=None, device="cuda",dtype=torch.float16):
#         LM.__init__(self)
#         print("initializing mamba original")
#         self._model = MambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)
#         self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#         self.vocab_size = self.tokenizer.vocab_size
#         self._batch_size = int(batch_size) if batch_size is not None else 64
#         self._max_length = max_length
#         self._device = torch.device(device)

if __name__ == "__main__":
    #print("test for -e mode")
    #model  = MambaEvalWrapper()
    model2 = Mamba2EvalWrapper()
    print("ok")
