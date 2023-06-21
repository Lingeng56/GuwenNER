import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from utils import *
from sklearn.metrics import f1_score


class CRFDecoder(nn.Module):

    def __init__(self, tag_to_ix, hidden_dim):
        super(CRFDecoder, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.hidden_dim = hidden_dim
        self.tagset_size = len(self.tag_to_ix)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))



    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    
    def decode(self, batch_feats):
        tag_seqs = []
        scores = []
        for feats in batch_feats:
            score, tag_seq = self._viterbi_decode(feats)
            tag_seqs.append(torch.tensor(tag_seq))
            scores.append(score)
        tag_seqs = torch.hstack(tag_seqs).to(self.device)
        scores = torch.hstack(scores)
        return scores, tag_seqs


    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    
    def neg_log_likelihood(self, inputs, tags):
        feats = self.hidden2tag(inputs)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
        

    @property
    def device(self):
        return next(self.parameters()).device


    def forward(self, inputs):
        outputs = self.hidden2tag(inputs)
        score, tag_seq = self.decode(outputs)
        return score, tag_seq
    


class BERTLSTMEncoder(nn.Module):

    def __init__(self, pretrained_model, embedding_dim, hidden_dim):
        super(BERTLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeds = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True)



    def forward(self, inputs):
        outputs = self.word_embeds(**inputs).last_hidden_state
        outputs, _ = self.lstm(outputs)
        return outputs






class NERModel(pl.LightningModule):

    def __init__(self,
                 pretrained_model,
                 tag_to_ix,
                 embedding_dim,
                 hidden_dim,
                 lr):
        super(NERModel, self).__init__()
        self.encoder = BERTLSTMEncoder(pretrained_model, embedding_dim, hidden_dim)
        self.decoder = CRFDecoder(tag_to_ix, hidden_dim)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        inputs, targets = batch['inputs'], batch['targets']
        hidden = self.encoder(inputs)
        loss = self.decoder.neg_log_likelihood(hidden, targets).mean()
        self.log('training_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        inputs, targets, sentence_lens, target_lens = batch['inputs'], batch['targets'], batch['sentence_lens'], batch['target_lens']
        hidden = self.encoder(inputs)
        loss = self.decoder.neg_log_likelihood(hidden, targets).mean()
        self.log('valid_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        score, tag_seq = self.decoder(hidden)
        tag_seq = tag_seq.reshape(len(sentence_lens), -1)
        for idx, sentence_len in enumerate(sentence_lens):
            tag_seq[idx, sentence_len:] = -100000
        tag_seq = tag_seq.reshape(-1)
        return {
            'pred': tag_seq,
            'true': targets
        }


    def validation_epoch_end(self, outputs):
        targets = torch.hstack([out['true'] for out in outputs]).reshape(-1).tolist()
        predictions = torch.hstack([out['pred'] for out in outputs]).reshape(-1).tolist()
        predictions = [pred for pred in predictions if pred != -100000]
        score = f1_score(targets, predictions, average='micro')
        self.log('dev_f1_score', score, sync_dist=True, prog_bar=True, logger=True, on_epoch=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets, sentence_lens = batch['inputs'], batch['targets'], batch['sentence_lens']
        hidden = self.encoder(inputs)
        score, tag_seq = self.decoder(hidden)
        tag_seq = tag_seq.reshape(len(sentence_lens), -1)[:, 1:]
        for idx, sentence_len in enumerate(sentence_lens):
            tag_seq[idx, sentence_len:] = -10000
        return tag_seq

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        return optimizer









