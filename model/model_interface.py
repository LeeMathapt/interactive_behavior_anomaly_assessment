"""
This is the connection between models and main script.
Class MInterface load model and control the training
process.
The name of model script need to be snake case with "ae"
or "vae" keyword and the name of main model class need
to be camel case. For example, File name "Vanilla_vae.py"
with its main model class name "VanillaVAE".
"""

import time
from datetime import timedelta
import importlib
from typing import Any, List
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

import pytorch_lightning as pl


class MInterface(pl.LightningModule):
    """
    On each training epoch, the reconstruction loss and time consume
    will be recorded.
    During prediction phase, reconstruction loss of each data will be
    recorded and its distribution will be print out. Check the function
    "on_predict_end" if need any change.
    """
    
    def __init__(self, model_params, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.params = kargs
        self.load_model(model_params)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(inputs, **kwargs)
    
    def on_train_epoch_start(self) -> None:
        self.start = time.time()

    def training_step(self, batch: List, batch_idx) -> torch.Tensor:
        data, _, _, = batch
        seq, pred = data
        out = self(seq, pred=pred)
        loss = self.model.loss_function(*out, reduction="mean")
        
        # Record loss value use for further analysis
        self.log(
            "loss", loss["loss"],
            sync_dist=True,
        )
        self.log(
            "reconstruction_loss", loss["reconstruction_loss"],
            sync_dist=True,
        )
        self.loss = loss["reconstruction_loss"]
        return loss["loss"]
    
    def on_train_epoch_end(self) -> None:
        end = time.time()
        elapsed = end-self.start
        print(f"Epoch {self.current_epoch+1:>3}: {timedelta(seconds=int(elapsed))}"\
              f" | reconstruction loss: {self.loss:.2f}")
        
    def on_predict_start(self) -> None:
        self.csv_loss = []
        self.latent = []

    def predict_step(self, batch: List, batch_idx) -> torch.Tensor:
        seq, pred = batch
        out = self(seq, pred=pred)
        loss = self.model.predict_loss(*out)

        self.csv_loss.append(loss["loss"])
        self.latent.append(out[-1].cpu())
        return 0

    def on_predict_end(self):
        loss = np.concatenate(self.csv_loss)
        loss_plot = pd.DataFrame(loss)
        
        # Collect the distribution of the embedding layer.
        # latent_data = np.concatenate(self.latent)
        # latent = pd.DataFrame(latent_data)

        # save the boxplot
        print("Saving ....")
        loss_plot.plot(kind="box")
        id = self.params["protagonist"]
        data = Path(self.params["data"]).parents[0]
        category = Path(self.params["data_path"]).stem
        save_path = Path(self.logger.log_dir)/category
        if save_path.exists() is False:
            save_path.mkdir()
            
        plt.savefig(save_path/f"reconstruction_error-{data}-{id}.png")

        # Save the loss as a csv file
        loss_plot.to_csv(save_path/f"reconstruction_error-{data}-{id}.csv")
        
        # Save the latent as a csv file
        # latent.to_csv(Path(self.logger.log_dir)/f"latent_embedding-{data}-{id}.csv")

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["LR"],
            weight_decay=self.params["weight_decay"],
        )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params["LR_2"] is not None:
                optimizer2 = torch.optim.Adam(
                    getattr(self.model, self.params["submodel"]).parameters(),
                    lr=self.params["LR_2"],
                )
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params["scheduler_gamma"] is not None:
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optims[0], gamma=self.params["scheduler_gamma"]
                )
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params["scheduler_gamma_2"] is not None:
                        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
                            optims[1], gamma=self.params["scheduler_gamma_2"]
                        )
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def load_model(self, model_params):
        name = model_params["model_name"]
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join(
            [i.capitalize() if not (i=="vae" or i=="ae") else i.upper() for i in name.split('_')]
            )
        Model = getattr(importlib.import_module(
            '.'+name, package=__package__), camel_name)

        self.model = Model(**model_params, **self.params)

if __name__ == "__main__":
    paras1 = {"A":1, "B":2, "C":3, "D":4}
    paras2 = {"E":5, "F":6, "G":7, "H":8}
    m = MInterface(model_params=paras1, **paras2)
