import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import *


class CRFDecoder(nn.Module):

    def __init__(self, tag_to_ix, hidden_dim):
        super(CRFDecoder, self).__init__()

        self.tagset_size = len(tag_to_ix)
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



    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
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


    def forward(self, inputs):
        outputs = self.hidden2tag(inputs)
        score, tag_seq = self._viterbi_decode(outputs)
        return score, tag_seq



class BERTLSTMEncoder(nn.Module):

    def __init__(self, pretrained_model, embedding_dim, hidden_dim):
        super(BERTLSTMEncoder, self).__init__()
        self.word_embeds = AutoModelForMaskedLM.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))



    def forward(self, inputs):
        outputs = self.word_embeds(inputs)
        outputs, self.hidden = self.lstm(outputs, self.hidden)
        outputs = outputs.view(len(inputs), self.hidden_dim)
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
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        inputs, targets = batch['inputs'], batch['targets']
        score, tag_seq = self.decoder(self.encoder(**inputs))
        loss = self.criterion(score, tag_seq)
        self.log('training_loss', loss)


    def validation_step(self, batch, batch_idx):
        inputs, targets = batch['inputs'], batch['targets']
        score, tag_seq = self.decoder(self.encoder(**inputs))
        loss = self.criterion(score, tag_seq)
        self.log('valid_loss', loss)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.lr)
        return optimizer









