"""
This is the main entrance of the whole project.

Most of the code should not be changed, please directly
add all the input arguments of your model's constructor
and the dataset file's constructor. The MInterface and 
DInterface can be seen as transparent to all your args.    

Example input:
python --config config/ae.yaml -phase train

Check tensor board:
tensorboard --logdir model/model_files/VanillaVAE/version_1/
"""
import os
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.utilities import CombinedLoader

from model import MInterface
from data import DInterface

def main(args):
    """
    Main function.
    Input parameters:
    --config:  yaml file that contain parameters of model.
    --phase: phase to perform -- train, resume, predict and prepare.
    --f: file that list data need to trasform. Only used for prepare.
    """
    
    with open(Path(args.filename), "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    pl.seed_everything(config["exp_params"]["manual_seed"], True)
    
    tb_logger = TensorBoardLogger(**config["logging_params"])
    check = ModelCheckpoint(
        save_top_k=2,
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        monitor="loss",
        save_last=True,
        save_on_train_epoch_end=True
    )

    trainer = Trainer(
        logger=tb_logger,
        callbacks=[LearningRateMonitor(),check],
        enable_progress_bar=False,
        **config["trainer_params"],
    )

    if args.phase == "train":
        data_dir = Path(config["data_params"]["data_path"])
        
        datasets = dict()
        print("Preparing datasets ...")
        for d in tqdm(list(data_dir.iterdir())):
            dataset = d.stem.split("-")[0]
            id = d.stem.split("-")[1][2:]
            thres = d.stem.split("-")[2][:-2]
            config["data_params"].update({"data": Path(dataset)/Path("dumb"),
                                          "protagonist": int(id),
                                          "zone_radius": thres,
                                          "data_path": d.parent}
                                          )
            
            data_module = DInterface(num_workers=len(config["trainer_params"]),
                                    **config["data_params"])
            data_module.prepare_data()
            data_module.setup("fit")
            datasets[f"{d}_{id}"] = data_module.train_dataloader()

        combined_loader = CombinedLoader(datasets, mode="sequential")
        model = MInterface(model_params=config["model_params"],
                        **config["data_params"],
                        **config["exp_params"])
            
        print(f"================= Training ====================")
        trainer.fit(model, combined_loader)

    elif args.phase == "resume":
        data_module = DInterface(num_workers=len(config["trainer_params"]),
                                 **config["data_params"])
        data_module.prepare_data()
        data_module.setup("fit")
        checkpoint_path = tb_logger.log_dir+"/checkpoints/last.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model = MInterface(model_params=config["model_params"],
                           **config["data_params"],
                           **config["exp_params"])
        model.load_state_dict(checkpoint['state_dict'])

        print(f"================= Training {config['model_params']['model_name']} ====================")
        trainer.fit(model, datamodule=data_module)

    elif args.phase == "predict":
        print(f"model {tb_logger.name}")
        checkpoint_path = tb_logger.log_dir+"/checkpoints/last.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model = MInterface(model_params=config["model_params"],
                        **config["data_params"],
                        **config["exp_params"])
        model.load_state_dict(checkpoint['state_dict'])
        data_dir = Path(config["data_params"]["data_path"])
        for d in data_dir.iterdir():
            dataset = d.stem.split("-")[0]
            id = d.stem.split("-")[1][2:]
            thres = d.stem.split("-")[2][:-2]
            config["data_params"].update({"data": Path(dataset)/Path("dumb"),
                                          "protagonist": int(id),
                                          "zone_radius": thres,
                                          "data_path": d.parent}
                                          )
            
            data_module = DInterface(num_workers=len(config["trainer_params"]),
                                    **config["data_params"])
            data_module.prepare_data()
            data_module.setup("predict")

            model.params.update({"data": Path(dataset)/Path("dumb"),
                                "protagonist": int(id),
                                "zone_radius": thres,
                                "data_path": d.parent}
                                )

            print(f"================= Predicting {d.stem} ====================")
            trainer.predict(model, datamodule=data_module)
        

    elif args.phase == "prepare":
        """
        Default file "prepare.txt" contain information file and id to prepare
        """
        with open(Path(args.file), "r") as f:
            file = f.readlines()

        for i in file:
            i.strip("\n")
            dataset, id = i.split(",")

            config["data_params"].update({"data": Path(dataset)/Path("trajectory"),
                                          "protagonist": int(id)}
                                          )
            data_module = DInterface(num_workers=len(config["trainer_params"]),
                                **config["data_params"])
            print(f"Preparing {dataset} {id}")
            data_module.prepare_data()
        
    else:
        raise ValueError("No such phase parameter.")

if __name__ == '__main__':
    parser = ArgumentParser(description="Generic runner for time series anomaly detection")
    # Basic Training Control
    parser.add_argument("--config",
                        "-c",
                        dest="filename",
                        default="config/ae.yaml")
    parser.add_argument("--phase", "-p", dest="phase", type=str, default="train")
    parser.add_argument("--f", dest="file", type=str)

    args = parser.parse_args()

    main(args)
