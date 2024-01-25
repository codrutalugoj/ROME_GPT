# %%
import torch
from torch.utils.data import Dataset
from transformer_lens import HookedTransformer

# %%
def run_causal_tracing(model,
                       inputs,
                       state_to_patch)
    """
    params:
     state_to_patch: list of hidden states (corrupted) to be replaced with its equivalent clean state

    This function runs the causal tracing algorithm on the list of states to be patched.
    """

    # Embed inputs
    embeddings = 