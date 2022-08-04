from random import shuffle
import torch
from tokenizer import *

class AmazonReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, lines, labels, word_vectors, max_sent_length, device):
        self.lines = lines
        self.labels = labels
        self.word_vectors = word_vectors
        self.max_sent_length = max_sent_length
        self.device = device

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        label = self.labels[index]
        if(label == "True"):
          label =  1
        else:
          label = 0
        indices = None
        if type(line) == str:
            indices = self.line2indices(line)
        else:   #line is tokenized
            indices = self.tokens2indices(line)
        #input_ids, _text, text_length = self.tokens2indices(tokens)
        numberOfIndices = len(indices)

        return {
            "input_ids": torch.tensor(indices).to(self.device),
            "label": torch.tensor(label).to(self.device),
            "length": numberOfIndices,
        }

    def tokens2indices(self, tokens):
        sos_id = self.word_vectors.key_to_index["<SOS>"]
        eos_id = self.word_vectors.key_to_index["<EOS>"]
        pad_id = self.word_vectors.key_to_index["<PAD>"]
        unk_id = self.word_vectors.key_to_index["<UNK>"]

        resultList = [pad_id] * self.max_sent_length
        resultList[0] = (sos_id)

        i = 1
        for token in tokens:
            if token in self.word_vectors.key_to_index:
                # for token in embedding
                index = self.word_vectors.key_to_index[token]
                resultList[i] = index
            else:
                # for unknown tokens
                print("token Not found")
                resultList[i] = unk_id

            i += 1
            if(i>=self.max_sent_length):
                break

        resultList[len(resultList) - 1] = eos_id

        return resultList

    def line2indices(self, line):
        tokens = tokenizeLine(line)
        return self.tokens2indices(tokens)


def make_dataloader(lines, labels, word_vectors, max_sent_length, batch_size, device):
    ds = AmazonReviewsDataset(lines, labels, word_vectors, max_sent_length, device)
    return torch.utils.data.DataLoader(ds, batch_size, shuffle=True)