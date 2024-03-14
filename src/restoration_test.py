# %%
import tqdm.auto as tqdm
from transformer_lens import HookedTransformer
from causal_trace import causal_tracing
import torch as t


# %%
if __name__ == "__main__":
    gpt: HookedTransformer = HookedTransformer.from_pretrained(model_name="gpt2-small", device="cpu")

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

    example = example_3

    clean_tokens = gpt.to_tokens(example["prompt"])
    corrupted_tokens = gpt.to_tokens(example["corrupt_prompt"])

    num_tokens = corrupted_tokens.shape[-1]
    subject_dim = gpt.to_tokens(example["subject"]).shape[-1]

    clean_logits, clean_cache = gpt.run_with_cache(clean_tokens) 
    corrupted_logits, _ = gpt.run_with_cache(corrupted_tokens)

    resid_stream_out = t.zeros((gpt.cfg.n_layers, num_tokens))
    
    for layer in tqdm.tqdm(range(gpt.cfg.n_layers)):
        for position in range(num_tokens):
            print(f"Restoring layer {layer}, token {position+1}/{num_tokens}")
            # TODO: how to check for how well the restoration went?
            # TODO: store results for plotting 
            corrupted_w_restoration_logits = causal_tracing(model=gpt, 
                                                            clean_cache=clean_cache,
                                                            corrupted_tokens=corrupted_tokens,
                                                            layer_to_restore=layer,
                                                            token_to_restore=position,
                                                            type_to_patch="resid")
    