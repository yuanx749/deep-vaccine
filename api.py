import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class _SequenceDataset(Dataset):
    def __init__(self, seqs, tokenizer):
        self.data = seqs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return self.tokenizer.encode(seq)

    def collate_fn(self, batch):
        return pad_sequence(
            batch, batch_first=True, padding_value=self.tokenizer.stoi["<pad>"]
        )


class Predictor:
    """A predictor that predicts the probabilities of vaccine subunit candidates.

    Attributes:
        model: A trained model or a string representing the path of the model.
    """

    def __init__(self, model):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(model, str):
            self.model = torch.load(model, map_location=self._device)
        else:
            self.model = model.to(self._device)

    def predict_proba(self, seqs):
        """Predicts the probabilities of inputs as vaccine subunit candidates.

        Args:
            seqs: A string list of aa sequence.

        Returns:
            A float array representing the probabilities.
        """
        data = _SequenceDataset(seqs, self.model.tokenizer)
        data_loader = DataLoader(data, batch_size=1, collate_fn=data.collate_fn)

        torch.set_grad_enabled(False)
        self.model.eval()
        probs = []
        for inputs in data_loader:
            inputs = inputs.to(self._device)
            outputs = self.model(inputs)
            probs.append(torch.sigmoid(outputs[:, 1]))  # label 1 is positive
        probs = torch.cat(probs).cpu().numpy()

        return probs

    def predict(self, seqs, threshold=0.5):
        """Predicts the labels of inputs.

        Args:
            seqs: A string list of aa sequence.
            threshold: A float representing the prediction threshold.

        Returns:
            A boolean array indicating the vaccine subunit candidates.
        """
        return self.predict_proba(seqs) > threshold
