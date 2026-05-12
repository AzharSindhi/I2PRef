import argparse
import json
import os
import copy
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
from lightning_model import I2PRefModule
from util import set_seed
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict
from dataset import ViPCDataModule


def load_model(ckpt_path, dataset_config):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = I2PRefModule(dataset_config=dataset_config)
    if set(ckpt.keys()) == {"state_dict"}:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model = I2PRefModule.load_from_checkpoint(
            ckpt_path, strict=False, dataset_config=dataset_config,
        )
    return model


def run(args, config, trainset_config, trainer_args, train_config):
    trainset_config['num_workers'] = max(1, os.cpu_count() // 4)
    datamodule = ViPCDataModule(trainset_config)

    output_directory = os.path.join(args.root_directory, args.experiment_name, args.run_name)
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output directory: {output_directory}")

    logger = WandbLogger(
        project=args.experiment_name,
        name=args.run_name,
        save_dir=os.path.join(args.root_directory, "../"),
        resume="allow",
    )
    logger.log_hyperparams(config)
    logger.log_hyperparams(vars(args))

    if args.test:
        assert args.ckpt_path, "--ckpt_path is required for testing"
        assert os.path.exists(args.ckpt_path), f"Checkpoint not found: {args.ckpt_path}"
        model = load_model(args.ckpt_path, trainset_config)
        trainer = pl.Trainer(logger=logger, **trainer_args)
        trainer.test(model, datamodule=datamodule)
    else:
        if args.ckpt_path:
            assert os.path.exists(args.ckpt_path), f"Checkpoint not found: {args.ckpt_path}"
            model = load_model(args.ckpt_path, trainset_config)
        else:
            model = I2PRefModule(dataset_config=trainset_config)

        checkpoints_dir = os.path.join(output_directory, 'checkpoints')
        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                dirpath=checkpoints_dir,
                filename='best',
                save_top_k=1,
                monitor='val_loss',
                mode='min',
                save_last=True,
            ),
        ]

        trainer = pl.Trainer(
            logger=logger,
            callbacks=callbacks,
            check_val_every_n_epoch=train_config.get('check_val_every_n_epoch', 10),
            default_root_dir=output_directory,
            **trainer_args,
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False,
                        help="Run in test/evaluation mode. Requires --ckpt_path.")
    parser.add_argument('--ckpt_path', type=str, default="",
                        help="Checkpoint path. Required for --test; optional for training (resumes fine-tuning).")
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--root_directory', type=str, default="exp_vipc_output")
    parser.add_argument('--experiment_name', type=str, default="i2pref")
    parser.add_argument('--category', type=str, default="all")
    args = parser.parse_args()

    with open("./exp_configs/ViPC.json") as f:
        config = json.loads(f.read())
    config = restore_string_to_list_in_a_dict(config)

    train_config = config["train_config"]
    trainer_args = config.get('trainer_args', {})
    trainset_config = config['vipc_dataset_config']

    trainset_config['batch_size'] = args.batch_size
    trainset_config['eval_batch_size'] = args.eval_batch_size
    trainset_config['category'] = args.category
    trainer_args['max_epochs'] = args.n_epochs
    trainset_config["data_dir"] = os.path.expanduser(trainset_config["data_dir"])

    if args.run_name == "":
        args.run_name = "vipc_test" if args.test else "vipc_train"
        if args.ckpt_path and not args.test:
            args.run_name += "_finetune"

    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))

    run(args, config, trainset_config, trainer_args, train_config)
