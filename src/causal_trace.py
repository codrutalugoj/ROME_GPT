# %%
import torch as t
from torch.utils.data import Dataset
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

# %%
def causal_tracing(model: HookedTransformer,
                   clean_cache,
                   corrupted_tokens: t.Tensor,
                   layer_to_restore,
                   token_to_restore,
                   type_to_patch):  # type of layer to patch ("resid", "mlp" or "attention")

    #print("Causal tracing...")

    ### Corrupted-with-restoration run
    # i.e. run w/ corrupted input while restoring/replacing 
    # the activations for 1 token + layer combination with the clean activations
    if type_to_patch == "resid":
        state_to_patch = f"blocks.{layer_to_restore}.hook_resid_pre"
    elif type_to_patch == "mlp":
        state_to_patch = f"blocks.{layer_to_restore}.hook_mlp_out"
    elif type_to_patch == "attention":
        state_to_patch = f"blocks.{layer_to_restore}.attn.hook_k"

    corrupted_with_restoration_logits = model.run_with_hooks(corrupted_tokens,
                                                             fwd_hooks=[(state_to_patch, make_restoration_hook(clean_cache, 
                                                                                                               token_to_restore))])


    return corrupted_with_restoration_logits



# %%
def make_activation_hook(subject_mask):
    def corrupt_activations_hook_func(activation_value,  #"batch head_idx seqQ seqK"
                                      hook: HookPoint):
        activation_value[:, 1:3, :, :] = 0 # TODO: make this as in the paper
        print(f"{subject_dim=}", f"{activation_value[:, :, :, subject_mask].shape=}")
        return activation_value
    
    return corrupt_activations_hook_func

def make_resid_hook(subject_mask):
    def corrupt_activations_hook_func(resid_value,  #"batch head_idx seqQ seqK"
                                      hook: HookPoint):
        print(f"{resid_value.shape=}")
        resid_value[:, 1:2, :] = 0 # TODO: make this as in the paper i.e add Gaussian noise

        return resid_value
    return corrupt_activations_hook_func

def make_restoration_hook(clean_cache, token):
    # activation_name: mlp, attention

    def corrupt_with_restoration_hook_func(corrupted_activation_value,
                                           hook: HookPoint):
        # corrupted_activation_value: [1, 6, 768]
        # TODO: why the 6 in the 2nd dim?
        # clean_cache[name]: [1, 7, 12, 64]
        corrupted_activation_value[:, token] = clean_cache[hook.name][:, token]
        return corrupted_activation_value
    return corrupt_with_restoration_hook_func

 
# %%
def total_effect(clean_logits,
                 corrupted_logits,
                 pos,
                 token_id):
    return clean_logits.softmax(dim=-1)[pos, token_id] - corrupted_logits.softmax(dim=-1)[pos, token_id]



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
    corrupted_tokens = gpt.to_tokens(example["corrupt_prompt"])

    subject_dim = gpt.to_tokens(example["subject"]).shape[-1]
 
    # TODO: which states to start patching first? E.g. MLP layers 
    # MLP layer 1 vs MLP mid layer

    #state_to_patch='blocks.11.attn.hook_k'
    state_to_patch='blocks.0.hook_resid_pre'
    #state_to_patch = 'blocks.0.hook_mlp_out'
    clean_logits, corrupted_logits, corrupted_w_restoration_logits = causal_tracing(model=gpt, 
                                                                                        clean_tokens=gpt.to_tokens(example["prompt"]),
                                                                                        corrupted_tokens=corrupted_tokens,
                                                                                        subject_dim=subject_dim,
                                                                                        state_to_patch=state_to_patch,
                                                                                        type_to_patch="resid")
    
    clean_preds = clean_logits.argmax(dim=-1).squeeze()
    corrupted_preds = corrupted_logits.argmax(dim=-1).squeeze()
    corrupted_w_rest_preds = corrupted_w_restoration_logits.argmax(dim=-1).squeeze()

    to_preds_last_token = t.topk(corrupted_logits[:, -1, :].squeeze(), k=10).indices

    # What preds does the model give to these other tokens?
    str_to_evaluate = [" Italy", "Italy", " France", " Germany", " the"]
    tokens_to_evaluate = gpt.to_tokens(str_to_evaluate, prepend_bos=False)


    print("Prompt tokenized", gpt.to_str_tokens(example["prompt"]))
    print("Model completion clean:", gpt.to_str_tokens(clean_preds)) 
    print("Model completion corrupted:", gpt.to_str_tokens(corrupted_preds)) 
    print("Model completion corrupted w/ restoration:", gpt.to_str_tokens(corrupted_w_rest_preds)) 


    print("Top predictions last token (corrupted):", gpt.to_str_tokens(to_preds_last_token))
    print(f"Clean logits for {str_to_evaluate}:\n", clean_logits[:, -1, tokens_to_evaluate])
    print(f"Corrupted logits for {str_to_evaluate}:\n", corrupted_logits[:, -1, tokens_to_evaluate])

    # Get the pos for the object (o) token from the prompt
    # That's just the len(prompt) + 1

    #print("Total effect \" Italy\"", total_effect(clean_logits[0], corrupted_logits[0], pos=gpt.to_tokens(example["prompt"]).shape[-1], token_id=gpt.to_single_token(" Italy")))
