import numpy as np
import torch
import torch.nn.functional as F

def scores_stacking(scores):
    scores_sizes = [len(_score) for _score in scores]
    if len(np.unique(scores_sizes)) > 1:
        max_action_space_size = max(scores_sizes)
        stacked_scores = torch.stack([
            F.pad(
                _score,
                (0, max_action_space_size - len(_score)),
                "constant", -torch.inf)
            for _score in scores])
    else:
        stacked_scores = torch.stack(scores)

    return stacked_scores

def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask tokens with cumulative probability > p
    cutoff = cumulative_probs > p
    if torch.any(cutoff):
        cutoff_idx = torch.where(cutoff)[1][0] + 1
    else:
        cutoff_idx = logits.size(1)

    top_logits = sorted_logits[:, :cutoff_idx]
    top_indices = sorted_indices[:, :cutoff_idx]
    top_probs = torch.softmax(top_logits, dim=-1)

    sampled_idx = torch.multinomial(top_probs, num_samples=1).squeeze(-1)
    return top_indices.gather(1, sampled_idx.unsqueeze(-1)).squeeze(-1)