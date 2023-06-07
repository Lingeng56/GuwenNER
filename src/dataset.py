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
                unit = unit.split('\n')
                sentence = ''
                tags = []
                for item in unit:
                    word, tag = item.split(' ')
                    sentence += word
                    tags.append(self.tag_to_ix[tag])
                self.sentences.append(sentence)
                self.tags.append(tags)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag_seq = self.tags[idx]
        return sentence, tag_seq

    def __len__(self):
        return len(self.sentences)


def collate_fn(batch, tokenizer):
    batch_input_ids = []
    batch_targets = []
    for sentence, tag_seq in batch:
        batch_input_ids.append(sentence)
        batch_targets.append(torch.LongTensor(tag_seq))
    batch_inputs = tokenizer(batch_input_ids, padding='max_length', truncation=True, return_tensors="pt")
    batch_targets = torch.hstack(batch_targets)
    batch_data = {
        'input_ids': batch_inputs,
        'targets': batch_targets
    }
    return batch_data
