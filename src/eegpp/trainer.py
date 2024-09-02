from pathlib import Path

import numpy as np
import torch
from lightning.fabric import Fabric, seed_everything
from pytorch_model_summary import summary
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import roc_auc_score as auroc
from torch.optim import Adam
from tqdm import tqdm

from out import OUT_DIR
from src.eegpp import params
from src.eegpp.dataloader import EEGKFoldDataLoader
from src.eegpp.logger import MyLogger
from src.eegpp.utils.general_utils import generate_normal_vector
from src.eegpp.utils.model_utils import get_model

torch.set_float32_matmul_precision('medium')


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
            early_stopping=None,
            export_torchscript=False,
            # save_last=False

    ):
        self.logger = MyLogger()

        self.fabric = Fabric(
            accelerator=accelerator,
            devices=devices,
            strategy='auto',
            callbacks=callbacks
        )
        seed_everything(params.RD_SEED)
        self.fabric.launch()
        self.device = self.fabric.device

        self.softmax = torch.nn.Softmax(dim=-1)
        self.model_type = model_type
        self.models = [get_model(model_type) for _ in range(n_splits)]
        self.optimizers = [Adam(model.parameters(), lr=lr, weight_decay=weight_decay) for model in self.models]
        self.dataloaders = EEGKFoldDataLoader(n_splits=n_splits, batch_size=batch_size, n_workers=n_workers)
        self.loss_fn_train = torch.nn.CrossEntropyLoss(ignore_index=-1)
        w_binary = torch.tensor([0.1, 1], dtype=torch.float32, device=self.device)
        self.loss_fn_train_binary = torch.nn.CrossEntropyLoss(weight=w_binary)
        self.loss_fn_val = torch.nn.CrossEntropyLoss()
        self.n_epochs = n_epochs
        self.n_splits = n_splits
        self.export_torchscript = export_torchscript
        self.batch_size = batch_size

        self.early_stopping = early_stopping

    def base_step(self, model, batch_idx, batch):
        x, lb, lb_binary = batch
        pred, pred_binary = model(x)
        loss = self.compute_training_loss(pred, lb, pred_binary, lb_binary, weight_star=1)
        return x, lb, pred, lb_binary, pred_binary, loss

    def compute_training_loss(self, pred, true, pred_binary, true_binary, w_out=params.W_OUT, weight_star=1):
        w_loss = generate_normal_vector(w_out)
        loss = 0
        loss_binary = 0
        for i in range(w_out):
            loss_i = self.loss_fn_train(pred[:, i, :], true[:, i, :])
            loss_binary_i = self.loss_fn_train_binary(pred_binary[:, i, :], true_binary[:, i, :])
            loss += w_loss[i] * loss_i
            loss_binary += w_loss[i] * loss_binary_i

        return loss + weight_star * loss_binary

    def fit(self):
        # print trainer summary at training beginning
        self.fabric.print("TRAINING STAGE")
        self.fabric.print("Using device: ", self.fabric.device)
        if self.early_stopping is not None:
            self.fabric.print(f"Apply early stopping: {self.early_stopping}.")
        else:
            self.fabric.print("Not apply early stopping.")
        self.trainer_summary()

        best_criteria = 1e6
        best_fold = 0
        best_epoch = 0
        for k in range(self.n_splits):
            self.fabric.print(f"Working on fold {k + 1}/{self.n_splits}")
            model, optimizer = self.models[k], self.optimizers[k]
            self.dataloaders.set_fold(k)
            train_dataloader, val_dataloader = self.dataloaders.train_dataloader(), self.dataloaders.val_dataloader()
            model, optimizer = self.fabric.setup(model, optimizer)
            train_dataloader, val_dataloader = self.fabric.setup_dataloaders(train_dataloader, val_dataloader)

            early_stopping = self.early_stopping
            for epoch in range(self.n_epochs):
                self.fabric.print(f"Epoch {epoch + 1}/{self.n_epochs}")
                self.logger.update_flag(flag='fit', epoch=epoch, fold=k)

                # TRAINING LOOP
                model.train()
                train_loss = 0
                train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")
                for batch_idx, batch in train_bar:
                    optimizer.zero_grad()
                    _, _, _, _, _, loss = self.base_step(model, batch_idx, batch)
                    self.fabric.backward(loss, model=model)
                    optimizer.step()
                    train_loss += loss.item()
                    train_bar.set_postfix({"step_loss": loss.item()})

                train_loss = train_loss / len(train_dataloader)
                self.logger.log_dict({'train/epoch_loss': train_loss})

                # VALIDATION LOOP
                model.eval()
                val_lb = []
                val_pred = []
                val_lb_binary = []
                val_pred_binary = []
                with torch.no_grad():
                    val_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validation")
                    for batch_idx, batch in val_bar:
                        _, lb, pred, lb_binary, pred_binary, _ = self.base_step(model, batch_idx, batch)
                        # ONLY VALIDATE ON THE MAIN SEGMENT
                        loss = self.loss_fn_val(pred[:, params.POS_IDX, :-1], lb[:, params.POS_IDX, :-1])
                        val_pred.append(pred[:, params.POS_IDX, :-1])  # ignore the last class
                        val_lb.append(lb[:, params.POS_IDX, :-1])
                        val_pred_binary.append(pred_binary[:, params.POS_IDX, :])
                        val_lb_binary.append(lb_binary[:, params.POS_IDX, :])
                        val_bar.set_postfix({"step_loss": loss.item()})

                    val_pred = torch.concat(val_pred, dim=0)
                    val_lb = torch.concat(val_lb, dim=0)
                    val_pred_binary = torch.concat(val_pred_binary, dim=0)
                    val_lb_binary = torch.concat(val_lb_binary, dim=0)

                    # calculate val loss and val loss binary
                    val_loss = self.loss_fn_val(val_pred, val_lb).item()
                    val_loss_binary = self.loss_fn_val(val_pred_binary, val_lb_binary).item()

                    # calculate val AUROC and AUPRC
                    val_pred = self.softmax(val_pred).detach().cpu().numpy()
                    val_lb = torch.argmax(val_lb, dim=-1).detach().cpu().numpy()
                    np.savetxt(str(Path(OUT_DIR, 'logs', f'{self.model_type}_best_prediction.txt')), val_pred,
                               fmt='%.4f')
                    np.savetxt(str(Path(OUT_DIR, 'logs', f'{self.model_type}_lb_prediction.txt')), val_lb, fmt='%d')
                    val_auroc = auroc(val_lb, val_pred, multi_class='ovr')
                    val_auprc = auprc(val_lb, val_pred)

                    val_pred_binary = self.softmax(val_pred_binary).detach().cpu().numpy()
                    val_lb_binary = val_lb_binary.detach().cpu().numpy()
                    np.savetxt(str(Path(OUT_DIR, 'logs', f'{self.model_type}_best_prediction_binary.txt')),
                               val_pred_binary, fmt='%.4f')
                    np.savetxt(str(Path(OUT_DIR, 'logs', f'{self.model_type}_lb_prediction_binary.txt')),
                               np.argmax(val_lb_binary, axis=-1), fmt='%d')
                    val_auroc_binary = auroc(val_lb_binary, val_pred_binary)
                    val_auprc_binary = auprc(val_lb_binary, val_pred_binary)

                    # calculate val F1X = 2 * auroc * auprc / (auroc + auprc + 1e-10)
                    f1x = 2 * val_auroc * val_auprc / (val_auroc + val_auprc + 1e-10)
                    f1x_binary = 2 * val_auroc_binary * val_auprc_binary / (val_auroc_binary + val_auprc_binary + 1e-10)

                    self.logger.log_dict({
                        "val/epoch_loss": val_loss,
                        "val/epoch_loss_binary": val_loss_binary,
                        "val/epoch_auroc": val_auroc,
                        "val/epoch_auprc": val_auprc,
                        "val/epoch_auroc_binary": val_auroc_binary,
                        "val/epoch_auprc_binary": val_auprc_binary,
                        "val/epoch_f1x": f1x,
                        "val/epoch_f1x_binary": f1x_binary,
                    })

                    self.logger.save_to_csv()

                    # self.fabric.print(
                    #     f"Fold {k + 1} Epoch {epoch + 1}\n"
                    #     f"Validation loss {val_loss:.4f}\n"
                    #     f"Validation loss_binary {val_loss_binary:.4f}\n"
                    #     f"Validation AUROC {val_auroc:.4f}\n"
                    #     f"Validation AUPRC {val_auprc:.4f}\n"
                    #     f"Validation AUROC Binary {val_auroc_binary:.4f}\n"
                    #     f"Validation AUPRC Binary {val_auprc_binary:.4f}\n"
                    #     f"Validation F1X {f1x:.4f}\n"
                    #     f"Validation F1X Binary {f1x_binary:.4f}\n"
                    # )

                    # define criteria
                    if params.CRITERIA == 'f1x':
                        criteria = -f1x
                    elif params.CRITERIA == 'f1x_binary':
                        criteria = -f1x_binary
                    elif params.CRITERIA == 'val_loss_binary':
                        criteria = val_loss_binary
                    else:
                        criteria = val_loss

                    # update criteria and early stopping (optional)
                    if criteria < best_criteria:
                        best_criteria = criteria
                        best_fold = k
                        best_epoch = epoch
                        state_dict = {
                            'epoch': epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }
                        self.fabric.save(f'{OUT_DIR}/checkpoints/{self.model_type}_best.pkl', state_dict)
                    else:
                        if self.early_stopping is not None:
                            early_stopping -= 1

                    self.fabric.print(
                        f"Current: Fold {k + 1} - Epoch {epoch + 1} - Criteria {params.CRITERIA} = {abs(criteria):.4f}\n"
                        f"Best: Fold {best_fold + 1} - Epoch {best_epoch + 1} - Criteria {params.CRITERIA} = {abs(best_criteria):.4f}\n"
                    )

                    if early_stopping is not None and early_stopping <= 0:
                        self.fabric.print("Early Stopping because criteria did not improve!\n")
                        break

        if self.export_torchscript:
            self.export_to_torchscript()

    def test(self):
        self.fabric.print("TESTING STAGE")
        self.fabric.print("Using device: ", self.fabric.device)
        model = get_model(self.model_type)
        state = torch.load(str(Path(OUT_DIR, 'checkpoints', f'{self.model_type}_best.pkl')), weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        model = self.fabric.setup_module(model, move_to_device=True)
        test_dataloader = self.dataloaders.test_dataloader()
        test_dataloader = self.fabric.setup_dataloaders(test_dataloader)
        self.logger.update_flag(flag='test', epoch=None, fold=None)

        model.eval()
        test_pred = []
        test_lb = []
        test_pred_binary = []
        test_lb_binary = []
        with torch.no_grad():
            test_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing")
            for batch_idx, batch in test_bar:
                _, lb, pred, lb_binary, pred_binary, _ = self.base_step(model, batch_idx, batch)
                # ONLY TEST ON THE MAIN SEGMENT
                loss = self.loss_fn_val(pred[:, params.POS_IDX, :-1], lb[:, params.POS_IDX, :-1])
                test_pred.append(pred[:, params.POS_IDX, :-1])  # ignore the last class
                test_lb.append(lb[:, params.POS_IDX, :-1])
                test_pred_binary.append(pred_binary[:, params.POS_IDX, :])
                test_lb_binary.append(lb_binary[:, params.POS_IDX, :])
                test_bar.set_postfix({"step_loss": loss.item()})

            test_pred = torch.concat(test_pred, dim=0)
            test_lb = torch.concat(test_lb, dim=0)
            test_pred_binary = torch.concat(test_pred_binary, dim=0)
            test_lb_binary = torch.concat(test_lb_binary, dim=0)

            test_loss = self.loss_fn_val(test_pred, test_lb).item()
            test_loss_binary = self.loss_fn_val(test_pred_binary, test_lb_binary).item()
            self.logger.log_dict({
                "test/loss": test_loss,
                "test/loss_binary": test_loss_binary,
            })

            test_pred = self.softmax(test_pred).detach().cpu().numpy()
            test_lb = torch.argmax(test_lb, dim=-1).detach().cpu().numpy()
            np.savetxt(str(Path(OUT_DIR, 'logs', f'{self.model_type}_best_prediction.txt')), test_pred, fmt='%.4f')
            np.savetxt(str(Path(OUT_DIR, 'logs', f'{self.model_type}_lb_prediction.txt')), test_lb, fmt='%d')
            test_auroc = auroc(test_lb, test_pred, multi_class='ovr')
            test_auprc = auprc(test_lb, test_pred)

            test_pred_binary = self.softmax(test_pred_binary).detach().cpu().numpy()
            test_lb_binary = test_lb_binary.detach().cpu().numpy()
            np.savetxt(str(Path(OUT_DIR, 'logs', f'{self.model_type}_best_prediction_binary.txt')), test_pred_binary,
                       fmt='%.4f')
            np.savetxt(str(Path(OUT_DIR, 'logs', f'{self.model_type}_lb_prediction_binary.txt')),
                       np.argmax(test_lb_binary, axis=-1), fmt='%d')
            test_auroc_binary = auroc(test_lb_binary, test_pred_binary)
            test_auprc_binary = auprc(test_lb_binary, test_pred_binary)

            f1x = 2 * test_auroc * test_auprc / (test_auroc + test_auprc + 1e-10)
            f1x_binary = 2 * test_auroc_binary * test_auprc_binary / (test_auroc_binary + test_auprc_binary + 1e-10)

            self.logger.log_dict({
                "test/auroc": test_auroc,
                "test/auprc": test_auprc,
                "test/auroc_binary": test_auroc_binary,
                "test/auprc_binary": test_auprc_binary,
                "test/f1x": f1x,
                "test/f1x_binary": f1x_binary,
            })

            self.fabric.print(
                f"Test Loss {test_loss:.4f}\n"
                f"Test AUROC {test_auroc:.4f}\n"
                f"Test AUPRC {test_auprc:.4f}\n"
                f"Test F1X {f1x:.4f}\n"
                f"Test F1X Binary {f1x_binary:.4f}"
            )

            self.logger.save_to_csv()

    def trainer_summary(self):
        input_shape = (self.batch_size, 3, (params.W_OUT * params.MAX_SEQ_SIZE))
        inp = torch.ones(input_shape)
        model_summary = summary(get_model(self.model_type), inp, batch_size=self.batch_size, show_input=True,
                                show_hierarchical=True, show_parent_layers=True)
        self.logger.log_model_summary(model_summary)

    def export_to_torchscript(self):
        self.fabric.print("Exporting TorchScript model...")
        model = get_model(self.model_type)
        state = torch.load(str(Path(OUT_DIR, 'checkpoints', f'{self.model_type}_best.pkl')), weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        try:
            model_scripted = torch.jit.script(model)
            model_scripted.save(f'{OUT_DIR}/checkpoints/{self.model_type}_best_scripted.pt')
            self.fabric.print("Exporting TorchScript successfully!")
        except Exception as e:
            self.logger.log_error(error=e)
            self.fabric.print("Cannot convert model to TorchScript")


if __name__ == '__main__':
    trainer = EEGKFoldTrainer(model_type=params.MODEL_TYPE, lr=params.LEARNING_RATE, n_splits=params.N_SPLITS,
                              n_epochs=params.NUM_EPOCHS, accelerator=params.ACCELERATOR, devices=params.DEVICES)
    # trainer.fit()
    trainer.test()
