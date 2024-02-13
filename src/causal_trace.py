# %%
import torch as t
from torch.utils.data import Dataset
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

# %%
def run_causal_tracing(model: HookedTransformer,
                       clean_tokens: t.Tensor,
                       subject_dim: int,
                       state_to_patch):
    """
    params:
     state_to_patch: list of hidden states (corrupted) to be replaced with its equivalent clean state

    This function runs the causal tracing algorithm on the list of states to be patched.

    We obtain the corrupted run following the steps:
        1. embed the clean prompt and obtain [h_1, h_2, ..., h_T]
        2. set certain activations to h_i = h_i + eps, eps ~ N(0, nu), where nu = 3 * std of emdeddings
        3. forward pass through net
        4. get corrupted activations
    """

    print("Causal tracing...")


    ### Clean run
    clean_logits, clean_cache = model.run_with_cache(clean_tokens) 
    print()
    print(clean_cache, type(clean_cache))
    print(clean_cache.keys)

    ### Corrupted run

    corrupted_logits = model.run_with_hooks(clean_tokens, 
                                            fwd_hooks=[(state_to_patch, make_resid_hook(subject_dim))])
                                            # fwd_hooks=[(state_to_patch, lambda x: x)])

    
    ### Corrupted-with-restoration run
    # i.e. run w/ corrupted input while restoring/replacing 
    # the activations for 1 token + layer combination with the clean activations

    return clean_logits, corrupted_logits



# %%
def make_activation_hook(subject_mask):
    def corrupt_activations_hook_func(activation_value,  #"batch head_idx seqQ seqK"
                                      hook: HookPoint):
        print(activation_value[:, :, 0, 0])
        activation_value[:, 1:3, :, :] = 0 # TODO: make this as in the paper
        print(activation_value[:, : , 0, 0])

        print(f"{subject_dim=}", f"{activation_value[:, :, :, subject_mask].shape=}")
        print(f"{activation_value.shape=}")
        return activation_value
    return corrupt_activations_hook_func

def make_resid_hook(subject_mask):
    def corrupt_activations_hook_func(resid_value,  #"batch head_idx seqQ seqK"
                                      hook: HookPoint):
        print(f"{resid_value.shape=}")
        resid_value[:, 1:2, :] = 0 # TODO: make this as in the paper

        return resid_value
    return corrupt_activations_hook_func


# %%
def total_effect(clean_logits,
                 corrupted_logits,
                 pos,
                 token_id):
    return clean_logits.softmax(dim=-1)[pos, token_id] - corrupted_logits.softmax(dim=-1)[pos, token_id]



# %%
if __name__ == "__main__":
    gpt: HookedTransformer = HookedTransformer.from_pretrained(model_name="gpt2-small", device="cpu")
    example = {
                "known_id": 2,
                "subject": "Audible.com",
                "attribute": "Amazon",
                "template": "{} is owned by",
                "prediction": " Amazon.com, Inc. or its affiliates.",
                "prompt": "Audible.com is owned by",
                "relation_id": "P127"
            }
    example_2 = {"subject": "Mary",
                 "prompt": "Mary had a little lamb, something"
            }
    
    example_3 = {"subject": "Rome",
                 "prompt": "Rome is the capital of"
            }

    example = example_3

    print(gpt.to_tokens(example["prompt"]), gpt.to_tokens(example["subject"]))
    subject_dim = gpt.to_tokens(example["subject"]).shape[-1]
    print("subject dum", subject_dim)
 
    clean_logits, corrupted_logits = run_causal_tracing(model=gpt, 
                                               clean_tokens=gpt.to_tokens(example["prompt"]),
                                               subject_dim=subject_dim,
                                               # state_to_patch='blocks.11.attn.hook_k')
                                               state_to_patch='blocks.0.hook_resid_pre')
    
    corrupted_preds = corrupted_logits.argmax(dim=-1).squeeze()
    clean_preds = clean_logits.argmax(dim=-1).squeeze()

    to_preds_last_token = t.topk(corrupted_logits[:, -1, :].squeeze(), k=10).indices

    # What preds does the model give to these other tokens?
    str_to_evaluate = [" Italy", "Italy", " France", " Germany", " the"]
    tokens_to_evaluate = gpt.to_tokens(str_to_evaluate, prepend_bos=False)


    #print(f"Corrupted prompt: \"{corrupted_prompt} ___\"")
    print("Prompt tokenized", gpt.to_str_tokens(example["prompt"]))
    print("Model completion clean:", gpt.to_str_tokens(clean_preds)) 
    print("Model completion corrupted:", gpt.to_str_tokens(corrupted_preds)) 
    print("Top predictions last token", gpt.to_str_tokens(to_preds_last_token))
    print(f"Clean predictions for {str_to_evaluate}:\n", clean_logits[:, -1, tokens_to_evaluate])
    print(f"Corrupted predictions for {str_to_evaluate}:\n", corrupted_logits[:, -1, tokens_to_evaluate])

    # Get the pos for the object (o) token from the prompt
    # That's just the len(prompt) + 1

    print("Total effect \" Italy\"", total_effect(clean_logits[0], corrupted_logits[0], pos=gpt.to_tokens(example["prompt"]).shape[-1], token_id=gpt.to_single_token(" Italy")))
