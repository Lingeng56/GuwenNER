import torch

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def load_tags(tag_path):
    ix_to_tag = []
    with open(tag_path) as f:
        for line in f:
            tag = line.strip()
            ix_to_tag.append(tag)
    tag_to_ix = {tag: ix for ix, tag in enumerate(ix_to_tag)}
    return tag_to_ix, ix_to_tag
