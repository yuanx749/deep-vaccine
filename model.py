import math
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, tokenizer, emb_size=32, kernel_size=5, hidden_size=32, num_layers=2, dropout=0.0, bidirectional=True, pooling=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.embedding = nn.Embedding(tokenizer.vocab_size, emb_size)
        self.cnn = nn.Conv1d(emb_size, emb_size, kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.bn1 = nn.BatchNorm1d(emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_size * (bidirectional+1), 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size * (bidirectional+1), 2)

    def forward(self, inputs):
        """
        Inputs: tensor of shape (batch_size, seq_len).
        Outputs: tensor of shape (batch_size, 2).
        """
        embeddings = self.embedding(inputs)
        embeddings = self.dropout(embeddings)

        embeddings = embeddings.transpose(1, 2).contiguous()
        embeddings = self.cnn(embeddings)
        embeddings = embeddings.transpose(1, 2).contiguous()
        
        outputs, _ = self.lstm(embeddings)
        # outputs, _ = self.gru(embeddings)
        if self.pooling:
            outputs = torch.mean(outputs, 1) # average all the tokens
        else:
            # outputs = outputs[:, -1, :] # take the last token
            eos_indices = torch.nonzero(inputs == self.tokenizer.stoi['<eos>'], as_tuple=True)
            outputs = outputs[eos_indices] # take the end token
        outputs = self.fc1(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, seq_len, emb_size).
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, tokenizer, emb_size=32, num_head=2, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.emb_size = emb_size
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(emb_size, num_head, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(tokenizer.vocab_size, emb_size)
        self.decoder = nn.Linear(emb_size, 2)

    def forward(self, inputs):
        """
        Inputs: tensor of shape (batch_size, seq_len).
        Outputs: tensor of shape (batch_size, 2).
        """
        inputs = self.encoder(inputs) * math.sqrt(self.emb_size)
        inputs = self.pos_encoder(inputs)
        inputs = inputs.transpose(0, 1)
        outputs = self.transformer_encoder(inputs)
        outputs = outputs.transpose(0, 1)
        outputs = torch.mean(outputs, 1)
        outputs = self.decoder(outputs)
        return outputs
