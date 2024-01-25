# %%
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# %%
#gpt: HookedTransformer = HookedTransformer.from_pretrained(model_name="gpt2-xl", device="cuda:0")
gpt: HookedTransformer = HookedTransformer.from_pretrained(model_name="gpt2-small", device="cpu")

# %%
prompt = "The Space Needle is in downtown "
# tokenize the prompt
clean_tokens = gpt.to_tokens(prompt, prepend_bos=True)
embeddings = gpt.embed(clean_tokens) # 

print(embeddings)
print(embeddings.shape)
print(embeddings.std())

### Corrupt the token that names the subject s
# This is done by adding Gaussian noise from (0, 3 * stddev),
# where stddev comes from a sample of prompts. 
# Note: Neel Nanda 
# (https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Activation_Patching_in_TL_Demo.ipynb)
# just swaps around the 2 subject tokens. 


