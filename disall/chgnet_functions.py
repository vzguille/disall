
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from chgnet.data.dataset import collate_graphs


from pymatgen.io import ase as pgase

ase_to_pmg = pgase.AseAtomsAdaptor.get_structure
pmg_to_ase = pgase.AseAtomsAdaptor.get_atoms

def get_train_val_test_loader(
    dataset,
    *,
    batch_size: int = 64,
    # train_ratio: float = 0.8,
    # val_ratio: float = 0.1,
    train_key: list[str] | None = None,
    val_key: list[str] | None = None,
    test_key: list[str] | None = None,
    return_test: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Randomly partition a dataset into train, val, test loaders.

    Args:
        dataset (Dataset): The dataset to partition.
        batch_size (int): The batch size for the data loaders
            Default = 64
        train_ratio (float): The ratio of the dataset to use for training
            Default = 0.8
        val_ratio (float): The ratio of the dataset to use for validation
            Default: 0.1
        return_test (bool): Whether to return a test data loader
            Default = True
        num_workers (int): The number of worker processes for loading the data
            see torch Dataloader documentation for more info
            Default = 0
        pin_memory (bool): Whether to pin the memory of the data loaders
            Default: True

    Returns:
        train_loader, val_loader and optionally test_loader
    """
    total_size = len(dataset)
    indices = list(range(total_size))
    # random.shuffle(indices)
    # train_size = int(train_ratio * total_size)
    # val_size = int(val_ratio * total_size)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(indices=train_key),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(
            indices=val_key
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            sampler=SubsetRandomSampler(indices=test_key),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
    return train_loader, val_loader