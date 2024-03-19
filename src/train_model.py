# %%
import torch as t
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from causal_trace import run_causal_tracing

# %%
#gpt: HookedTransformer = HookedTransformer.from_pretrained(model_name="gpt2-xl", device="cuda:0")
gpt: HookedTransformer = HookedTransformer.from_pretrained(model_name="gpt2-small", device="cpu")

# %%
prompt = "The Space Needle is in downtown"

# tokenize the prompt
clean_tokens = gpt.to_tokens(prompt, prepend_bos=True)
embeddings = gpt.embed(clean_tokens) 
clean_logits, clean_cache = gpt.run_with_cache(clean_tokens)

# Get the best predictions 
clean_preds = clean_logits.argmax(dim=-1).squeeze()

print(f"Prompt: \"{prompt} ___\"")
print("Model completion:", gpt.to_str_tokens(clean_preds)[-1])


# %%
### Corrupt the token that names the subject s

# In the paper this is done by adding Gaussian noise from (0, 3 * stddev),
# where stddev comes from a sample of prompts. 
# Note: Neel Nanda e.g. just swaps around the 2 subject tokens.
# (https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Activation_Patching_in_TL_Demo.ipynb)
corrupted_prompt = "The Space Station is in downtown"
corrupted_tokens = gpt.to_tokens(corrupted_prompt, prepend_bos=True)
corrupted_logits = gpt(corrupted_tokens, )
corrupted_preds = corrupted_logits.argmax(dim=-1).squeeze()

print(f"Corrupted prompt: \"{corrupted_prompt} ___\"")
print("Model completion:", gpt.to_str_tokens(corrupted_preds)[-1]) # model's output is different 


# %%
print(embeddings)
print(embeddings.shape)
print(embeddings.std())

# %%
# "post": output after MLP block
clean_logits, corrupted_logits = run_causal_tracing(model=gpt, 
                                clean_tokens=clean_tokens, 
                                subject_dim=sub
                                state_to_patch='blocks.0.attn.hook_pattern')

