# %%
def indirect_effect(corrupt_logits,
                    corrupt_with_restoration_logits,
                    obj_token_idx  # object token id in dictionary
                    ):
    # = prob of obj (object) in corrupt_restoration - prob of obj under corrupt runs
    # TODO: what range is reasonable for IE?
    return corrupt_with_restoration_logits.softmax(-1)[0, -1, obj_token_idx] - corrupt_logits.softmax(-1)[0, -1, obj_token_idx]
