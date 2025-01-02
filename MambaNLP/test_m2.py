import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["CUDA_LAUNCH_BLOCKING"]='1'

import torch
from mamba_ssm.modules.mamba2 import Mamba2

if __name__ == "__main__":
    block = Mamba2(d_model=2560).to("cuda")
    # each mamba block is of input & output feature dimension 2560
    B, S, D = 1, 32, 2560  # Example: Batch=1, Sequence Length=32, Features=2560
    input_tensor = torch.randn(B, S, D).to("cuda")
    print("inferencing...")
    out = block(input_tensor)
    print(f"output dim: {out.shape}, ok")