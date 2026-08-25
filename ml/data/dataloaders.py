from torch.utils.data import DataLoader

from ml.data import config
from ml.data.dataset import DeepVisionDataset

def get_dataloader(split: str, batch_size: int = config.BATCH_SIZE, num_workers: int = config.NUM_WORKERS, pin_memory: bool = True) -> DataLoader:
    """
    Create a PyTorch DataLoader for the specified split.
    Training data is shuffled; validation and testing data are not.
    """
    dataset = DeepVisionDataset(split=split)
    shuffle = (split == "train")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def get_all_dataloaders():
    return {
        "train": get_dataloader("train"),
        "val": get_dataloader("val"),
        "test": get_dataloader("test")
    }
