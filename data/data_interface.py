"""
This is the connection between data models and main script.
Class DInterface load data model and prepare the training,
prediction dataloaders.
"""

import inspect
import importlib
from typing import List, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DInterface(pl.LightningDataModule):
    """
    Data module

    data_dir: root directory of your dataset.
    train_batch_size: the batch size to use during training.
    val_batch_size: the batch size to use during validation.
    patch_size: the size of the crop to take from the original images.
    num_workers: the number of parallel workers to create to load data
    items (see PyTorch's Dataloader documentation for more details).
    
    pin_memory: whether prepared items should be loaded into pinned memory
    or not. This can impro, joints_num: intve performance on GPUs.
    """

    def __init__(
        self,
        data_name: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataname = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs

        self.load_data_module()

    def prepare_data(self):
        self.prepared = self.data_module(**self.kwargs)

    def setup(self, stage: str="fit") -> None:
        # Assign train/val datasets for use in dataloaders
        # Input data shape: [Batch x Coord x Joints x Frames]
        if stage == "fit":
            self.trainset = self.prepared.train()

        # Assign test dataset for use in dataloader(s)
        if stage == "predict":
            self.predictset = self.prepared.predict()

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.predictset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def load_data_module(self):
        name = self.dataname
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """
        Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)


if __name__ == "__main__":
    pass