import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import product

def read_fasta(filename, data_dir='./data'):
    seqs = []
    with open(os.path.join(data_dir, filename)) as f:
        for line in f:
            if line[0] == '>':
                seq = ''
            elif line == '\n':
                seqs.append(seq)
            else:
                seq += line.strip()
    return seqs

def read_txt(filename, data_dir='./data'):
    lines = []
    with open(os.path.join(data_dir, filename)) as f:
        lines = [line.rstrip() for line in f]
        lines = [line for line in lines if line]
    return lines

class Tokenizer:
    def __init__(self, tokens=['<unk>', '<pad>', '<eos>'], max_len=50):
        self.tokens = tokens
        self.max_len = max_len
        self.itos = tokens + list('ACDEFGHIKLMNPQRSTVWY')
        self.stoi = dict(zip(self.itos, range(len(self.itos))))
        self.vocab_size = len(self.itos)
    
    def tokenize(self, seq):
        """
        Truncates and tokenizes the sequence.
        """
        tokens = [c for c in seq]
        tokens = tokens[:self.max_len]
        if '<eos>' in self.tokens:
            tokens.append('<eos>')
        return tokens
    
    def encode(self, seq):
        """
        Converts a sequence to a 1D integer tensor.
        """
        tokens = self.tokenize(seq)
        return torch.tensor([self.stoi.get(token, self.stoi['<unk>']) for token in tokens])
    
    def decode(self, tensor):
        """
        Converts a 1D integer tensor to a list of tokens.
        """
        return [self.itos[idx] for idx in tensor]

class Tokenizer2(Tokenizer):
    def __init__(self, tokens=['<unk>', '<pad>', '<eos>'], max_len=50, word_len=2):
        super().__init__(tokens, max_len)
        self.word_len = word_len
        self.itos = tokens + list(map(''.join, product('ACDEFGHIKLMNPQRSTVWY', repeat=word_len)))
        self.stoi = dict(zip(self.itos, range(len(self.itos))))
        self.vocab_size = len(self.itos)
    
    def tokenize(self, seq):
        # tokens = [seq[i:i+self.word_len] for i in range(len(seq) - self.word_len + 1)]
        # tokens = tokens[:self.max_len]
        tokens = [seq[i:i+self.word_len] for i in range(0, len(seq)//self.word_len*self.word_len, self.word_len)]
        tokens = tokens[:self.max_len // self.word_len]
        if '<eos>' in self.tokens:
            tokens.append('<eos>')
        return tokens

class BaselineDataset(Dataset):
    """
    A class of epitope B + epitope T sequences with binary labels.
    Inputs: a file of aa sequences and a file of lables.
    """
    def __init__(self, seq_file, label_file, tokenizer=Tokenizer(), data_dir='./data'):
        self.tokenizer = tokenizer
        self.data = torch.load(os.path.join(data_dir, seq_file))
        self.labels = torch.load(os.path.join(data_dir, label_file))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        label = self.labels[idx]
        return self.tokenizer.encode(seq), label
    
    def collate_fn(self, batch):
        """
        Pads tensors to max length of a batch.
        Returns: tensor of shape (batch_size, max_len), tensor of size batch_size.
        """
        tokens, labels = list(zip(*batch))
        tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.stoi['<pad>'])
        labels = torch.tensor(labels)
        return tokens, labels

class AntigenDataset(BaselineDataset):
    """
    A class of antigen sequences with binary labels.
    Input: a file, one sequence per line.
    """
    def __init__(self, file, tokenizer=Tokenizer(), data_dir='./data'):
        self.tokenizer = tokenizer
        with open(os.path.join(data_dir, file)) as f:
            lines = f.read().splitlines()
        self.data = lines
        self.labels = [1]*(len(lines)//2) + [0]*(len(lines)//2)

class EpitopeDataset(BaselineDataset):
    """
    A class of epitope sequences with binary labels.
    Input: a pair of posivite file and negative file.
    """
    def __init__(self, positive_file, negative_file, tokenizer=Tokenizer(), data_dir='./data'):
        self.tokenizer = tokenizer
        self.raw_data, self.labels = [], []
        if positive_file.endswith('csv'):
            with open(os.path.join(data_dir, positive_file)) as csvfile:
                next(csvfile)
                for line in csvfile:
                    self.raw_data.append(line.strip().split(',')[1])
                    self.labels.append(1)
            with open(os.path.join(data_dir, negative_file)) as csvfile:
                next(csvfile)
                for line in csvfile:
                    self.raw_data.append(line.strip().split(',')[1])
                    self.labels.append(0)
        else:
            with open(os.path.join(data_dir, positive_file)) as f:
                lines = f.read().splitlines()
                self.raw_data += lines
                self.labels += [1]*len(lines)
            with open(os.path.join(data_dir, negative_file)) as f:
                lines = f.read().splitlines()
                self.raw_data += lines
                self.labels += [0]*len(lines)
        self.data = [seq[:self.tokenizer.max_len] for seq in self.raw_data]

class EpitopeRawDataset(EpitopeDataset):
    def __getitem__(self, idx):
        seq = self.data[idx]
        label = self.labels[idx]
        return self.tokenizer.encode(seq), label, seq
    
    def collate_fn(self, batch):
        """
        Pads tensors to max length of a batch.
        Returns: tensor of shape (batch_size, max_len), tensor of size batch_size, list of raw sequences.
        """
        tokens, labels, seqs = list(zip(*batch))
        tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.stoi['<pad>'])
        labels = torch.tensor(labels)
        return tokens, labels, seqs
