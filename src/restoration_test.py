# %%
import tqdm.auto as tqdm
from transformer_lens import HookedTransformer
from causal_trace import causal_tracing
from metrics import indirect_effect
from plots import plot_fig1
import torch as t
import numpy as np
import matplotlib.pyplot as plt


# %%

m_name = "gpt2-small"
#m_name = "gpt2-xl"
gpt: HookedTransformer = HookedTransformer.from_pretrained(model_name=m_name, device="cpu")

example_1 = {
            "known_id": 2,
            "subject": "Audible.com",
            "attribute": "Amazon",
            "template": "{} is owned by",
            "prediction": " Amazon.com, Inc. or its affiliates.",
            "prompt": "Audible.com is owned by",
            "corrupt_prompt": "Spotify is owned by",
            "relation_id": "P127"
        }
example_2 = {"subject": "Mary",
                "prompt": "Mary had a little lamb, something",
                "corrupt_prompt": "Nina had a little lamb, something"
        }

example_3 = {"subject": "Rome",
                "prompt": "Rome is the capital of",
                "corrupt_prompt": "Paris is the capital of"
        }
example_rome = {"subject": "The Space Needle",
                "prompt": "The Space Needle is in downtown",
                "corrupt_prompt": "The Square Beetle is in downtown",
                "attribute": "Seattle"}

example = example_rome

clean_tokens = gpt.to_tokens(example["prompt"])
corrupted_tokens = gpt.to_tokens(example["corrupt_prompt"])

num_tokens = corrupted_tokens.shape[-1]
subject_dim = gpt.to_tokens(example["subject"]).shape[-1]

clean_logits, clean_cache = gpt.run_with_cache(clean_tokens) 
corrupted_logits, _ = gpt.run_with_cache(corrupted_tokens)

obj_token_id =  gpt.to_single_token(example["attribute"])

prompt = gpt.to_str_tokens(example_rome["prompt"])[1:]


# %%
layer_types = ["resid", "mlp", "attention"]
#layer_types = ["attention"]

# dim 0: resid, mlp, attention
causal_tracing_out = t.zeros((3, num_tokens, gpt.cfg.n_layers))  

# %%
for i in tqdm.tqdm(range(len(layer_types))):

    for layer in tqdm.tqdm(range(gpt.cfg.n_layers)):
        for position in range(num_tokens):
            #print(f"Restoring layer {layer}, token {position+1}/{num_tokens}")
            # TODO: MLP + attention at last layer
            corrupted_with_restoration_logits = causal_tracing(model=gpt, 
                                                            clean_cache=clean_cache,
                                                            corrupted_tokens=corrupted_tokens,
                                                            layer_to_restore=layer,
                                                            token_to_restore=position,
                                                            type_to_patch=layer_types[i])
            causal_tracing_out[i, position, layer] = (indirect_effect(corrupt_logits=corrupted_logits,
                                                corrupt_with_restoration_logits=corrupted_with_restoration_logits,
                                                obj_token_idx=obj_token_id)).item()


# %%
#print(resid_stream_out.shape, len(prompt))
for i in range(len(layer_types)):
    plot_fig1(causal_tracing_out[i].numpy(), prompt)
    plt.show()
    