import torch
from dataloader.ViPCdataloader import ViPCDataLoader
from pytorch_lightning import LightningDataModule
from types import SimpleNamespace

class ViPCDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = SimpleNamespace(**args)
        self.args = args
    
    def setup(self, stage: str):
        self.train_loader = self._make_loader(is_train=True)
        self.val_loader = self._make_loader(is_train=False)
        self.test_loader = self._make_loader(is_train=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def _make_loader(self, is_train: bool):
        batch_size = self.args.batch_size if is_train else self.args.eval_batch_size
        dataset = ViPCDataLoader(
            self.args.data_dir,
            'train' if is_train else 'test',
            view_align=getattr(self.args, 'view_align', False),
            category=self.args.category,
            mini=getattr(self.args, 'mini', False),
            image_size=getattr(self.args, 'image_size', 224),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )


