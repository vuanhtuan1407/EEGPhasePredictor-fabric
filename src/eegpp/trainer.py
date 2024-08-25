import time

import torch
import wandb
from lightning.fabric import Fabric
from pytorch_model_summary import summary
from torch import nn
from torch.optim import Adam
from torchmetrics import F1Score, AUROC, AveragePrecision
from tqdm import tqdm
from wandb.integration.lightning.fabric import WandbLogger

from out import OUT_DIR
from src.eegpp import params
from src.eegpp.dataloader import EEGKFoldDataLoader
from src.eegpp.utils.callback_utils import model_checkpoint
from src.eegpp.utils.general_utils import generate_normal_vector
# from src.eegpp.models.baseline.cnn_model import CNN1DModel
from src.eegpp.utils.model_utils import get_model

torch.set_float32_matmul_precision('medium')
wandb.require('core')


class EEGKFoldTrainer:
    def __init__(
            self,
            model: nn.Module,
            lr=1e-3,
            batch_size=8,
            weight_decay=0,
            n_splits=5,
            n_epochs=10,
            n_workers=0,
            accelerator='auto',
            devices='auto',
            callbacks=None,
            # save_last=False

    ):
        self.logger = WandbLogger(
            project='EEGPhasePredictor-fabric',
            name=f'{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))}',
            log_model='all',
        )
        self.logger.experiment.config['model_type'] = params.MODEL_TYPE
        self.logger.experiment.config['batch_size'] = params.BATCH_SIZE
        self.logger.experiment.config['w_out'] = params.W_OUT

        if callbacks is None:
            callbacks = [model_checkpoint]

        self.fabric = Fabric(
            accelerator=accelerator,
            devices=devices,
            strategy='auto',
            callbacks=callbacks,
            loggers=self.logger,
        )
        self.fabric.launch()
        self.device = self.fabric.device

        self.models = [model for _ in range(n_splits)]
        self.optimizers = [Adam(model.parameters(), lr=lr, weight_decay=weight_decay) for model in self.models]
        self.dataloaders = EEGKFoldDataLoader(n_splits=n_splits, batch_size=batch_size, n_workers=n_workers)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.n_epochs = n_epochs
        self.n_splits = n_splits
        # self.save_last = save_last

        self.f1score = F1Score(task='multiclass', num_classes=params.NUM_CLASSES, average='macro').to(self.device)
        self.auroc = AUROC(task='multiclass', num_classes=params.NUM_CLASSES).to(self.device)  # default is macro
        self.auprc = AveragePrecision(task='multiclass', num_classes=params.NUM_CLASSES).to(self.device)

        self.best_val_loss = 1e6
        self.early_stopping = 3

        # print trainer summary
        self.trainer_summary()

    @staticmethod
    def preprocess(x):
        return x

    @staticmethod
    def postprocess(x):
        pass

    def base_step(self, model, batch_idx, batch):
        x, y = batch
        x = self.preprocess(x)
        pred = model(x)
        loss = self.compute_loss(pred, y)
        return x, y, pred, loss

    def compute_loss(self, pred, true, window_size=5):
        w_loss = generate_normal_vector(window_size)
        loss = 0
        for i in range(window_size):
            loss_i = self.loss_fn(pred[:, :, i], true[:, :, i])
            loss += w_loss[i] * loss_i
        return loss

    def fit(self):
        wandb.define_metric('train/epoch')
        wandb.define_metric('train/*', step_metric='train/epoch')
        wandb.define_metric('val/*', step_metric='train/epoch')
        for epoch in range(self.n_epochs):
            self.fabric.print(f"Epoch {epoch}/{self.n_epochs}")
            epoch_loss = 0
            # epoch_f1score = 0
            epoch_auroc = 0
            epoch_auprc = 0
            for k in range(self.n_splits):
                self.fabric.print(f"Working on fold {k}")
                model, optimizer = self.models[k], self.optimizers[k]
                self.dataloaders.set_fold(k)
                train_dataloader, val_dataloader = self.dataloaders.train_dataloader(), self.dataloaders.val_dataloader()
                model, optimizer = self.fabric.setup(model, optimizer)
                train_dataloader, val_dataloader = self.fabric.setup_dataloaders(train_dataloader, val_dataloader)

                # TRAINING LOOP
                model.train()
                wandb.watch(model)
                train_loss = 0
                train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")
                for batch_idx, batch in train_bar:
                    optimizer.zero_grad()
                    _, _, _, loss = self.base_step(model, batch_idx, batch)
                    train_loss += loss.item()
                    self.fabric.backward(loss)
                    optimizer.step()
                    train_bar.set_postfix({"step_loss": loss.item()})

                train_loss = train_loss / len(train_dataloader)
                train_bar.set_postfix({"epoch_loss": train_loss})
                self.fabric.log_dict({
                    "train/epoch": epoch,
                    f'train/loss_fold_{k}': train_loss
                })

                # VALIDATION LOOP
                model.eval()
                val_loss = 0
                val_lb = []
                val_pred = []
                with torch.no_grad():
                    val_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validation")
                    for batch_idx, batch in val_bar:
                        _, lb, pred, loss = self.base_step(model, batch_idx, batch)
                        val_loss += loss.item()
                        val_pred.append(pred[:, params.POS_IDX, :])  # only take the main label
                        val_lb.append(lb[:, params.POS_IDX, :])
                        val_bar.set_postfix({"step_loss": loss.item()})
                        # self.f1score.update(pred, lb)

                    val_loss = val_loss / len(val_dataloader)
                    val_bar.set_postfix({"epoch_loss": val_loss})
                    self.fabric.log_dict({
                        "train/epoch": epoch,
                        f"val/loss_fold_{k}": val_loss
                    })

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        state_dict = {
                            "epoch": epoch,
                            "model": model,
                            "optimizer": optimizer,
                        }
                        self.fabric.save(f'{OUT_DIR}/checkpoints/best.pkl', state_dict)
                    else:
                        self.early_stopping -= 1

                    epoch_loss += val_loss

                    val_pred = torch.concat(val_pred, dim=0)
                    val_lbs = torch.concat(val_lb, dim=0)
                    val_lbs = torch.argmax(val_lbs, dim=-1)
                    epoch_auroc += self.auprc(val_pred, val_lbs).item()
                    epoch_auprc += self.auprc(val_pred, val_lbs).item()
                    # epoch_f1score += self.f1score.compute()

            self.fabric.log_dict({
                "train/epoch": epoch,
                "val/mean_loss": epoch_loss / self.n_splits,
                "val/mean_auroc": epoch_auroc / self.n_splits,
                "val/mean_auprc": epoch_auprc / self.n_splits,
            })

            self.fabric.print(
                f"Epoch {epoch}/{self.n_epochs}\n"
                f"Mean Validation Loss {epoch_loss / self.n_splits}\n"
                f"Mean Validation AUROC {epoch_auroc / self.n_splits}\n"
                f"Mean Validation AUPRC {epoch_auprc / self.n_splits}"
            )

            if self.early_stopping <= 0:
                self.fabric.print("Early Stopping because validation loss did not improve!")
                break

        wandb.finish(quiet=True)

    def test(self):
        pass

    def trainer_summary(self):
        print("Using device: ", self.fabric.device)
        input_shape = (params.BATCH_SIZE, 3, (params.W_OUT * params.MAX_SEQ_SIZE))
        inp = torch.ones(input_shape)
        summary(self.models[0], inp, batch_size=params.BATCH_SIZE, show_input=True, print_summary=True)


if __name__ == '__main__':
    model = get_model(params.MODEL_TYPE)
    trainer = EEGKFoldTrainer(model=model, lr=params.LEARNING_RATE, n_splits=params.N_SPLITS,
                              n_epochs=params.NUM_EPOCHS, accelerator=params.ACCELERATOR, devices=params.DEVICES)
    trainer.fit()
    # trainer.load_model()
