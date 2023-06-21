import torch
from torch.utils.data import Dataset


# dataset path: https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset

class CustomDataset(Dataset):

    def __init__(self, data_path, tag_to_ix):
        self.data_path = data_path
        self.tag_to_ix = tag_to_ix
        self.__prepare_data()

    def __prepare_data(self):
        self.sentences = []
        self.tags = []
        with open(self.data_path) as f:
            text = f.read()
            sentence_units = text.split('\n\n')
            for unit in sentence_units:
                if unit == '':
                    continue
                unit = unit.split('\n')
                sentence = ''
                tags = []
                for item in unit:
                    if item == '':
                        continue
                    word, tag = item.split(' ')
                    sentence += word
                    tags.append(self.tag_to_ix[tag])
                self.sentences.append(sentence)
                self.tags.append(tags)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag_seq = self.tags[idx]
        sentence_len = len(sentence)
        target_len = len(tag_seq)
        return sentence, tag_seq, sentence_len, target_len

    def __len__(self):
        return len(self.sentences)


def collate_fn(batch, tokenizer, max_len):
    batch_input_ids = []
    batch_targets = []
    batch_sentence_length = []
    batch_target_length = []
    for sentence, tag_seq, sentence_len, target_len in batch:
        batch_input_ids.append(sentence)
        batch_targets.append(torch.LongTensor(tag_seq))
        batch_sentence_length.append(sentence_len)
        batch_target_length.append(target_len)
    batch_inputs = tokenizer(batch_input_ids, padding='max_length', truncation=True, return_tensors="pt", max_length=max_len)
    batch_targets = torch.hstack(batch_targets)
    batch_data = {
        'inputs': batch_inputs,
        'targets': batch_targets,
        'sentence_lens': batch_sentence_length,
        'target_lens': batch_target_length
    }
    return batch_data
