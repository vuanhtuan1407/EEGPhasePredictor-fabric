def backup_fit():
    """
        for epoch in range(self.n_epochs):
            self.fabric.print(f"Epoch {epoch + 1}/{self.n_epochs}")
            epoch_val_loss = 0
            epoch_val_loss_binary = 0
            epoch_val_auroc = 0
            epoch_val_auprc = 0
            epoch_val_auroc_binary = 0
            epoch_val_auprc_binary = 0
            best_val_fold_loss = 1e6
            best_fold_state_dict = None  # store best kfold training model on each epoch
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
                self.logger.log_dict({'train/fold_loss': train_loss})

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
                        loss = self.loss_fn(pred[:, params.POS_IDX, :], lb[:, params.POS_IDX, :])
                        val_pred.append(pred[:, params.POS_IDX, :-1])  # ignore the last class
                        val_lb.append(lb[:, params.POS_IDX, :-1])
                        val_pred_binary.append(pred_binary[:, params.POS_IDX, :])
                        val_lb_binary.append(lb_binary[:, params.POS_IDX, :])
                        val_bar.set_postfix({"step_loss": loss.item()})

                    val_pred = torch.concat(val_pred, dim=0)
                    val_lb = torch.concat(val_lb, dim=0)
                    val_pred_binary = torch.concat(val_pred_binary, dim=0)
                    val_lb_binary = torch.concat(val_lb_binary, dim=0)

                    val_loss = self.loss_fn(val_pred, val_lb).item()
                    val_loss_binary = self.loss_fn(val_pred_binary, val_lb_binary).item()
                    self.logger.log_dict({
                        "val/fold_loss": val_loss,
                        "val/fold_loss_binary": val_loss_binary,
                    })

                    # update epoch val loss and epoch val loss binary
                    epoch_val_loss += val_loss
                    epoch_val_loss_binary += val_loss_binary

                    # update val metrics
                    val_pred = self.softmax(val_pred).detach().cpu().numpy()
                    val_lb = torch.argmax(val_lb, dim=-1).detach().cpu().numpy()
                    epoch_val_auroc += auroc(val_lb, val_pred, multi_class='ovr')
                    epoch_val_auprc += auprc(val_lb, val_pred)

                    val_pred_binary = self.softmax(val_pred_binary).detach().cpu().numpy()
                    val_lb_binary = val_lb_binary.detach().cpu().numpy()
                    epoch_val_auroc_binary += auroc(val_lb_binary, val_pred_binary)
                    epoch_val_auprc_binary += auprc(val_lb_binary, val_pred_binary)

                    if val_loss < best_val_fold_loss:
                        best_val_fold_loss = val_loss
                        best_fold_state_dict = {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }

            mean_val_loss = epoch_val_loss / self.n_splits
            mean_val_loss_binary = epoch_val_loss_binary / self.n_splits
            mean_auroc = epoch_val_auroc / self.n_splits
            mean_auprc = epoch_val_auprc / self.n_splits
            mean_auroc_binary = epoch_val_auroc_binary / self.n_splits
            mean_auprc_binary = epoch_val_auprc_binary / self.n_splits

            # F1X = 2 * auroc * auprc / (auroc + auprc + 1e-10)
            f1x = 2 * mean_auroc * mean_auprc / (mean_auroc + mean_auprc + 1e-10)
            f1x_binary = 2 * mean_auroc_binary * mean_auprc_binary / (mean_auroc_binary + mean_auprc_binary + 1e-10)

            self.logger.update_flag(flag='val_metrics', epoch=epoch, fold=None)
            self.logger.log_dict({
                "val/mean_val_loss": mean_val_loss,
                "val/mean_val_loss_binary": mean_val_loss_binary,
                "val/mean_auroc": mean_auroc,
                "val/mean_auprc": mean_auprc,
                "val/mean_auroc_binary": mean_auroc_binary,
                "val/mean_auprc_binary": mean_auprc_binary,
                "val/f1x": f1x,
                "val/f1x_binary": f1x_binary,
            })

            self.fabric.print(
                f"Val Metrics on Epoch {epoch + 1}/{self.n_epochs}\n"
                f"Mean Validation Loss {mean_val_loss:.4f}\n"
                f"Mean Validation Loss Binary {mean_val_loss_binary:.4f}\n"
                f"Mean AUROC {mean_auroc:.4f}\n"
                f"Mean AUPRC {mean_auprc:.4f}\n"
                f"F1X {f1x:.4f}\n"
                f"F1X Binary {f1x_binary:.4f}"
            )

            self.logger.save_to_csv()  # save log every epoch

            if params.CRITERIA == "f1x":
                criteria = - f1x
            elif params.CRITERIA == "f1x_binary":
                criteria = - f1x_binary
            elif params.CRITERIA == "mean_val_loss_binary":
                criteria = mean_val_loss_binary
            else:
                criteria = mean_val_loss

            if criteria < self.best_criteria:
                self.best_criteria = criteria
                # Using fabric.save
                self.fabric.save(f'{OUT_DIR}/checkpoints/{self.model_type}_best.pkl', best_fold_state_dict)

            else:
                if self.early_stopping is not None:
                    self.early_stopping -= 1
                    if self.early_stopping <= 0:
                        self.fabric.print("Early Stopping because criteria did not improve!\n")
                        break

        if self.export_torchscript:
            self.export_to_torchscript()
    """
