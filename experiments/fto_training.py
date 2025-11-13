from torch.utils.data import DataLoader
from training.sequence_dataset import SequenceDataset


data_train = ...  # Load or generate your training data tensor here
seq_length = 10  # Define your sequence length here

train_loader = DataLoader(
    SequenceDataset(data_train, seq_length),
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)