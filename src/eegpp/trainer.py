import torch
from lightning.fabric import Fabric
from torch import nn
from torch.optim import Adam
from torchmetrics import F1Score, AUROC, AveragePrecision
from tqdm import tqdm

from src.eegpp import params
from src.eegpp.dataloader import EEGDataLoader
from src.eegpp.utils.general_utils import generate_normal_vector
# from src.eegpp.models.baseline.cnn_model import CNN1DModel
from src.eegpp.utils.model_utils import get_model

torch.set_float32_matmul_precision('medium')


class EEGKFoldTrainer:
    def __init__(
            self,
            model: nn.Module,
            lr=1e-3,
            weight_decay=0,
            n_splits=5,
            n_epochs=10,
            accelerator='auto',
            devices='auto',
            strategy='auto',
            callbacks=None,
            loggers=None,

    ):
        self.fabric = Fabric(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.fabric.launch()
        self.device = self.fabric.device

        self.models = [model for _ in range(n_splits)]
        self.optimizers = [Adam(model.parameters(), lr=lr, weight_decay=weight_decay) for model in self.models]
        self.dataloaders = EEGDataLoader(n_splits=n_splits)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.n_epochs = n_epochs
        self.n_splits = n_splits

        self.f1score = F1Score(task='multiclass', num_classes=params.NUM_CLASSES, average='macro')
        self.auroc = AUROC(task='multiclass', num_classes=params.NUM_CLASSES, average='macro')
        self.average_precision = AveragePrecision(task='multiclass', num_classes=params.NUM_CLASSES, average='macro')

        # print trainer summary
        # self.trainer_summary()

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
        for epoch in range(self.n_epochs):
            self.fabric.print(f"Epoch {epoch}/{self.n_epochs}")
            epoch_loss = 0
            epoch_f1score = 0
            epoch_auroc = 0
            epoch_average_precision = 0
            for k in range(self.n_splits):
                self.fabric.print(f"Working on fold {k}")
                model, optimizer = self.models[k], self.optimizers[k]
                self.dataloaders.set_fold(k)
                train_dataloader, val_dataloader = self.dataloaders.train_dataloader(), self.dataloaders.val_dataloader()
                model, optimizer = self.fabric.setup(model, optimizer)
                train_dataloader, val_dataloader = self.fabric.setup_dataloaders(train_dataloader, val_dataloader)

                # TRAINING LOOP
                model.train()
                train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")
                for batch_idx, batch in train_bar:
                    optimizer.zero_grad()
                    _, _, _, loss = self.base_step(model, batch_idx, batch)
                    self.fabric.backward(loss)
                    optimizer.step()
                    train_bar.set_postfix({"step_loss": loss.item()})

                # VALIDATION LOOP
                model.eval()
                val_loss = 0
                val_lbs = []
                val_pred = []
                with torch.no_grad():
                    val_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validation")
                    for batch_idx, batch in val_bar:
                        _, lb, pred, loss = self.base_step(model, batch_idx, batch)
                        val_loss += loss.item()
                        lb = torch.argmax(pred, dim=-1)
                        val_pred.append(pred)
                        val_lbs.append(lb)
                        val_bar.set_postfix({"step_loss": loss.item()})
                        # self.f1score.update(pred, lb)

                    val_loss = val_loss / len(val_dataloader)
                    val_bar.set_postfix({"epoch_loss": val_loss})
                    epoch_loss += val_loss

                    val_pred = torch.concat(val_pred, dim=0)
                    val_lbs = torch.concat(val_lbs, dim=0)
                    val_lbs = torch.argmax(val_lbs, dim=-1)
                    print(val_pred.shape, val_lbs.shape)
                    epoch_auroc += self.f1score(val_pred, val_lbs).item()
                    epoch_average_precision += self.average_precision(val_pred, val_lbs).item()
                    # epoch_f1score += self.f1score.compute()

            self.fabric.print(
                f"Epoch {epoch}/{self.n_epochs}\n"
                f"Mean Validation Loss {epoch_loss / self.n_splits}\n"
                f"Mean Validation AUROC {epoch_auroc / self.n_splits}\n"
                f"Mean Validation Average Precision {epoch_average_precision / self.n_splits}"
            )

    def test(self):
        pass

    def trainer_summary(self):
        # input_shape = [params.BATCH_SIZE, 3, params.MAX_SEQ_SIZE]
        print(self.fabric.device)


if __name__ == '__main__':
    model = get_model(params.MODEL_TYPE)
    trainer = EEGKFoldTrainer(
        model=model,
        lr=params.LEARNING_RATE,
        n_epochs=params.NUM_EPOCHS,
        n_splits=params.N_SPLITS,
        accelerator=params.ACCELERATOR,
        devices=params.DEVICES,
    )
    trainer.fit()
