import pytorch_lightning as pl
import torch
from metrics import calc_cd, chamfer_sqrt
from util import FPS
from models.image2point import Image2Point
from models.adaptive_transformers import AdaptivePointRefinementTransformer

class I2PRefModule(pl.LightningModule):
    def __init__(self, dataset_config):
        super().__init__()
        self.num_points = 2048
        self.model = AdaptivePointRefinementTransformer(
            num_points=self.num_points,
            embed_dim=256,
            depth=5,
            scale_factor=0.01,
        )

        self.image2pcl = Image2Point(img_ch=3, embed_dim=256)

        self.learning_rate = 1e-4
        self.categories = dataset_config.get('test_categories', ["all"])


    def forward(self, condition, image_tensor=None):
        sparse_input, image_features = self.image2pcl(image_tensor, condition)
        rec_out = self.model(sparse_input, image_features)
        return rec_out, sparse_input

    def training_step(self, batch, batch_idx):
        X, condition, image_tensor = self.split_data(batch)
        rec_out, sparse_out = self.forward(condition, image_tensor=image_tensor)

        loss_dense = chamfer_sqrt(rec_out, X)
        loss_sparse = chamfer_sqrt(sparse_out, X)
        alpha = self.sparse_weight[self.global_step]
        loss = loss_dense + alpha * loss_sparse

        bs = X.shape[0]
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
        self.log('train_loss_sparse', loss_sparse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
        self.log('train_loss_dense', loss_dense, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        X, condition, image_tensor = self.split_data(batch)
        rec_out, sparse_out = self.forward(condition, image_tensor=image_tensor)

        loss_dense = chamfer_sqrt(rec_out, X)
        loss_sparse = chamfer_sqrt(sparse_out, X)
        alpha = self.sparse_weight[self.global_step]
        loss = loss_dense + alpha * loss_sparse

        with torch.no_grad():
            if rec_out.shape[1] != X.shape[1]:
                rec_out = FPS(rec_out.transpose(1, 2).contiguous(), X.shape[1]).transpose(1, 2).contiguous()
            cd_l1, cd_l2 = calc_cd(rec_out, X)

        bs = X.shape[0]
        self.log('val_cd_l2', cd_l2, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
        self.log('val_loss_sparse', loss_sparse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
        self.log('val_loss_dense', loss_dense, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        X, condition, image_tensor = self.split_data(batch)
        rec_out, _ = self.forward(condition, image_tensor=image_tensor)

        with torch.no_grad():
            if rec_out.shape[1] != X.shape[1]:
                rec_out = FPS(rec_out.transpose(1, 2).contiguous(), X.shape[1]).transpose(1, 2).contiguous()
            cd_l1, cd_l2, f1_score = calc_cd(rec_out, X, calc_f1=True)
            f1_score = f1_score.mean()

        bs = X.shape[0]
        metric_suffix = self.categories[dataloader_idx]

        self.log(f'test_cd_l2_{metric_suffix}', cd_l2, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)
        self.log(f'test_fscore_{metric_suffix}', f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=bs)

        return 0
    

    def configure_optimizers(self):
        self.sparse_weight = torch.linspace(0.7, 0.1, self.trainer.estimated_stepping_batches + 1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler_cosine]

    def split_data(self, data):
        if isinstance(data, list):
            data = data[0]
        
        complete, partial, image_tensor = data['complete'], data['partial'], data['image_tensor']
        if complete.shape[1] != self.num_points:
            complete = FPS(complete.transpose(1, 2).contiguous(), self.num_points).transpose(1, 2).contiguous()
        if partial.shape[1] != self.num_points:
            partial = FPS(partial.transpose(1, 2).contiguous(), self.num_points).transpose(1, 2).contiguous()
        return complete, partial, image_tensor
