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
