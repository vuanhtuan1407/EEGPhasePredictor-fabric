from pathlib import Path

import torch
from lightning.fabric import Fabric
from pytorch_model_summary import summary
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import roc_auc_score as auroc
from torch.optim import Adam
# from torchmetrics import F1Score, AUROC, AveragePrecision
from tqdm import tqdm

from out import OUT_DIR
from src.eegpp import params
from src.eegpp.dataloader import EEGKFoldDataLoader
from src.eegpp.logger import MyLogger
from src.eegpp.utils.general_utils import generate_normal_vector
# from src.eegpp.models.baseline.cnn_model import CNN1DModel
from src.eegpp.utils.model_utils import get_model


class EEGKFoldTrainer:
    def __init__(
            self,
            model_type: str,
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
        self.logger = MyLogger()

        self.fabric = Fabric(
            accelerator=accelerator,
            devices=devices,
            strategy='auto',
            callbacks=callbacks
        )
        self.fabric.launch()
        self.device = self.fabric.device

        self.softmax = torch.nn.Softmax(dim=-1)
        self.model_type = model_type
        self.models = [get_model(model_type) for _ in range(n_splits)]
        self.optimizers = [Adam(model.parameters(), lr=lr, weight_decay=weight_decay) for model in self.models]
        self.dataloaders = EEGKFoldDataLoader(n_splits=n_splits, batch_size=batch_size, n_workers=n_workers)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.n_epochs = n_epochs
        self.n_splits = n_splits
        # self.save_last = save_last

        # self.f1score = F1Score(task='multiclass', num_classes=params.NUM_CLASSES, average='macro').to(self.device)
        # self.auroc = AUROC(task='multiclass', num_classes=params.NUM_CLASSES).to(self.device)  # default is macro
        # self.auprc = AveragePrecision(task='multiclass', num_classes=params.NUM_CLASSES).to(self.device)
        # self.auroc_binary = AUROC(task='multiclass', num_classes=2).to(self.device)
        # self.auprc_binary = AveragePrecision(task='multiclass', num_classes=2).to(self.device)

        self.best_val_loss = 1e6
        self.best_criteria = 1e6
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
        x, lb, lb_binary = batch
        x = self.preprocess(x)
        pred, pred_binary = model(x)
        loss = self.compute_loss(pred, lb, pred_binary, lb_binary)
        return x, lb, pred, lb_binary, pred_binary, loss

    def compute_loss(self, pred, true, pred_binary, true_binary, window_size=params.W_OUT, weight_star=1):
        w_loss = generate_normal_vector(window_size)
        loss = 0
        loss_binary = 0
        for i in range(window_size):
            loss_i = self.loss_fn(pred[:, i, :], true[:, i, :])
            loss_binary_i = self.loss_fn(pred_binary[:, i, :], true_binary[:, i, :])
            loss += w_loss[i] * loss_i
            loss_binary += w_loss[i] * loss_binary_i

        return loss + weight_star * loss_binary

    def fit(self):
        for epoch in range(self.n_epochs):
            self.fabric.print(f"Epoch {epoch + 1}/{self.n_epochs}")
            epoch_loss = 0
            # epoch_f1score = 0
            epoch_auroc = 0
            epoch_auprc = 0
            epoch_auroc_binary = 0
            epoch_auprc_binary = 0
            for k in range(self.n_splits):
                self.fabric.print(f"Working on fold {k + 1}/{self.n_splits}")
                self.logger.update_flag(flag='fit', epoch=epoch, fold=k)
                model, optimizer = self.models[k], self.optimizers[k]
                self.dataloaders.set_fold(k)
                train_dataloader, val_dataloader = self.dataloaders.train_dataloader(), self.dataloaders.val_dataloader()
                model, optimizer = self.fabric.setup(model, optimizer)
                train_dataloader, val_dataloader = self.fabric.setup_dataloaders(train_dataloader, val_dataloader)

                # TRAINING LOOP
                model.train()
                train_loss = 0
                train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")
                for batch_idx, batch in train_bar:
                    optimizer.zero_grad()
                    _, _, _, _, _, loss = self.base_step(model, batch_idx, batch)
                    train_loss += loss.item()
                    self.fabric.backward(loss)
                    optimizer.step()
                    train_bar.set_postfix({"step_loss": loss.item()})

                train_loss = train_loss / len(train_dataloader)
                # train_bar.set_postfix({"epoch_loss": train_loss})
                self.logger.log_dict({'train/fold_loss': train_loss})

                # VALIDATION LOOP
                model.eval()
                val_loss = 0
                val_lb = []
                val_pred = []
                val_lb_binary = []
                val_pred_binary = []
                with torch.no_grad():
                    val_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validation")
                    for batch_idx, batch in val_bar:
                        _, lb, pred, lb_binary, pred_binary, _ = self.base_step(model, batch_idx, batch)
                        # ONLY VALIDATE ON THE MAIN SEGMENT
                        loss = self.loss_fn(pred[:, params.POS_IDX, :], lb[:, params.POS_IDX, :])
                        val_loss += loss.item()
                        val_pred.append(pred[:, params.POS_IDX, :-1])  # ignore the last class
                        val_lb.append(lb[:, params.POS_IDX, :-1])
                        val_pred_binary.append(pred_binary[:, params.POS_IDX, :])
                        val_lb_binary.append(lb_binary[:, params.POS_IDX, :])
                        val_bar.set_postfix({"step_loss": loss.item()})
                        # self.f1score.update(pred, lb)

                    val_loss = val_loss / len(val_dataloader)
                    # val_bar.set_postfix({"epoch_loss": val_loss})
                    self.logger.log_dict({f"val/fold_loss": val_loss})

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        state_dict = {
                            "epoch": epoch,
                            "model": model,
                            "optimizer": optimizer,
                        }
                        self.fabric.save(f'{OUT_DIR}/checkpoints/{self.model_type}_best.pkl', state_dict)

                    epoch_loss += val_loss

                    val_pred = self.softmax(torch.concat(val_pred, dim=0)).detach().cpu().numpy()
                    val_lb = torch.concat(val_lb, dim=0)
                    val_lb = torch.argmax(val_lb, dim=-1).detach().cpu().numpy()
                    epoch_auroc += auroc(val_lb, val_pred, multi_class='ovr')
                    epoch_auprc += auprc(val_lb, val_pred)

                    val_pred_binary = self.softmax(torch.concat(val_pred_binary, dim=0)).detach().cpu().numpy()
                    val_lb_binary = torch.concat(val_lb_binary, dim=0).detach().cpu().numpy()
                    # val_lb_binary = torch.argmax(val_lb_binary, dim=-1)
                    epoch_auroc_binary += auroc(val_lb_binary, val_pred_binary)
                    epoch_auprc_binary += auprc(val_lb_binary, val_pred_binary)
                    # epoch_f1score += self.f1score.compute()

            mean_val_loss = epoch_loss / self.n_splits
            mean_auroc = epoch_auroc / self.n_splits
            mean_auprc = epoch_auprc / self.n_splits
            mean_auroc_binary = epoch_auroc_binary / self.n_splits
            mean_auprc_binary = epoch_auprc_binary / self.n_splits

            metric = 2 * mean_auroc * mean_auprc / (mean_auroc + mean_auprc + 1e-10)
            metric_binary = 2 * mean_auroc_binary * mean_auprc_binary / (mean_auroc_binary + mean_auprc_binary + 1e-10)

            self.logger.update_flag(flag='val_metrics', epoch=epoch, fold=None)
            self.logger.log_dict({
                "val/mean_val_loss": mean_val_loss,
                "val/mean_auroc": mean_auroc,
                "val/mean_auprc": mean_auprc,
                "val/mean_auroc_binary": mean_auroc_binary,
                "val/mean_auprc_binary": mean_auprc_binary,
                "val/metric": metric,
                "val/metric_binary": metric_binary,
            })

            self.fabric.print(
                f"Epoch {epoch + 1}/{self.n_epochs}\n"
                f"Mean Validation Loss {mean_val_loss:.4f}\n"
                f"Metric {metric:.4f}\n"
                f"Metric Binary {metric_binary:.4f}"
            )

            if params.CRITERIA == 'metric':
                criteria = metric
            elif params.CRITERIA == 'metric_binary':
                criteria = metric_binary
            else:
                criteria = mean_val_loss

            if criteria < self.best_criteria:
                self.best_criteria = criteria
            else:
                self.early_stopping -= 1

            if self.early_stopping <= 0:
                self.fabric.print("Early Stopping because criteria did not improve!\n")
                break

            self.logger.save_to_csv()  # save log every epoch

    def test(self):
        model = get_model(self.model_type)
        state = self.fabric.load(str(Path(OUT_DIR, 'checkpoints', f'{self.model_type}_best.pkl')))
        model.load_state_dict(state['model'])
        model = self.fabric.setup_module(model, move_to_device=True)
        test_dataloader = self.dataloaders.test_dataloader()
        test_dataloader = self.fabric.setup_dataloaders(test_dataloader)
        self.logger.update_flag(flag='test_metrics', epoch=None, fold=None)

        model.eval()
        test_loss = 0
        test_pred = []
        test_lb = []
        test_pred_binary = []
        test_lb_binary = []

        with torch.no_grad():
            test_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing")
            for batch_idx, batch in test_bar:
                _, lb, pred, lb_binary, pred_binary, _ = self.base_step(model, batch_idx, batch)
                # ONLY TEST ON THE MAIN SEGMENT
                loss = self.loss_fn(pred[:, params.POS_IDX, :], lb[:, params.POS_IDX, :])
                test_loss += loss.item()
                test_pred.append(pred[:, params.POS_IDX, :-1])  # ignore the last class
                test_lb.append(lb[:, params.POS_IDX, :-1])
                test_pred_binary.append(pred_binary[:, params.POS_IDX, :])
                test_lb_binary.append(lb_binary[:, params.POS_IDX, :])
                test_bar.set_postfix({"step_loss": loss.item()})

            test_loss = test_loss / len(test_dataloader)
            self.logger.log_dict({f"test/loss": test_loss})

            test_pred = self.softmax(torch.concat(test_pred, dim=0)).detach().cpu().numpy()
            test_lb = torch.concat(test_lb, dim=0)
            test_lb = torch.argmax(test_lb, dim=-1).detach().cpu().numpy()
            test_auroc = auroc(test_lb, test_pred, multi_class='ovr')
            test_auprc = auprc(test_lb, test_pred)

            test_pred_binary = self.softmax(torch.concat(test_pred_binary, dim=0)).detach().cpu().numpy()
            test_lb_binary = torch.concat(test_lb_binary, dim=0).detach().cpu().numpy()
            test_auroc_binary = auroc(test_lb_binary, test_pred_binary)
            test_auprc_binary = auprc(test_lb_binary, test_pred_binary)

            metric = 2 * test_auroc * test_auprc / (test_auroc + test_auprc + 1e-10)
            metric_binary = 2 * test_auroc_binary * test_auprc_binary / (test_auroc_binary + test_auprc_binary + 1e-10)

            self.logger.log_dict({
                "test/auroc": test_auroc,
                "test/auprc": test_auprc,
                "test/auroc_binary": test_auroc_binary,
                "test/auprc_binary": test_auprc_binary,
                "test/metric": metric,
                "test/metric_binary": metric_binary,
            })

            self.fabric.print(
                f"Test Loss {test_loss:.4f}\n"
                f"Test Metric {metric:.4f}\n"
                f"Test Metric Binary {metric_binary:.4f}"
            )

            self.logger.save_to_csv()

    def trainer_summary(self):
        self.fabric.print("Using device: ", self.fabric.device)
        input_shape = (params.BATCH_SIZE, 3, (params.W_OUT * params.MAX_SEQ_SIZE))
        inp = torch.ones(input_shape)
        model_summary = summary(get_model(self.model_type), inp, batch_size=params.BATCH_SIZE, show_input=True,
                                show_hierarchical=True, show_parent_layers=True)
        self.logger.log_model_summary(model_summary)


if __name__ == '__main__':
    trainer = EEGKFoldTrainer(model_type=params.MODEL_TYPE, lr=params.LEARNING_RATE, n_splits=params.N_SPLITS,
                              n_epochs=params.NUM_EPOCHS, accelerator=params.ACCELERATOR, devices=params.DEVICES)
    # trainer.fit()
    trainer.test()
    # trainer.load_model()
