from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, data, seq_length):
        """
        data: Tensor (n_samples_total, n_features)
        seq_length: taille de la fenêtre d'entrée/sortie
        """
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        # On peut aller jusqu'à len(data) - 2*seq_length
        return len(self.data) - 2 * self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length : idx + 2 * self.seq_length]
        return x, y