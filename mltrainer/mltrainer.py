import torch
from sklearn.metrics import accuracy_score
import json
import time
from logging import Logger


class MLTrainer:
    def __init__(self, model, optimizer, loss_fn, device='cpu', scheduler=None, mixed_precision=False, log_path='training_log.txt'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'epoch_times': []
        }
        self.best_val_loss = float('inf')
        self.logger = Logger(log_path)

    def train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item(), outputs, targets

    def validate_step(self, batch):
        self.model.eval()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.scaler is not None):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

        return loss.item(), outputs, targets

    def train(self, train_loader, val_loader=None, epochs=10, log_interval=10, early_stopping_patience=None, callbacks=[]):
        early_stop_counter = 0
        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss, train_acc = 0.0, 0.0
            for i, batch in enumerate(train_loader):
                loss, outputs, targets = self.train_step(batch)
                train_loss += loss
                preds = torch.argmax(outputs, dim=1)
                train_acc += accuracy_score(targets.cpu(), preds.cpu())

                if (i + 1) % log_interval == 0:
                    self.logger.log(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss:.4f}')

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.logger.log(f'Epoch {epoch + 1} Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.logger.log(f'Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

                # Early stopping
                if early_stopping_patience:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        early_stop_counter = 0
                        self.save_checkpoint(epoch)
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= early_stopping_patience:
                            self.logger.log('Early stopping triggered.')
                            break

            if self.scheduler:
                self.scheduler.step()

            # Track epoch duration
            epoch_duration = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_duration)
            self.logger.log(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds')

            # Execute callbacks at end of epoch
            for callback in callbacks:
                callback(epoch, self.history)

        total_time = time.time() - start_time
        self.logger.log(f'Training completed in {total_time / 60:.2f} minutes')

    def evaluate(self, data_loader):
        val_loss, val_acc = 0.0, 0.0
        for batch in data_loader:
            loss, outputs, targets = self.validate_step(batch)
            val_loss += loss
            preds = torch.argmax(outputs, dim=1)
            val_acc += accuracy_score(targets.cpu(), preds.cpu())

        val_loss /= len(data_loader)
        val_acc /= len(data_loader)
        return val_loss, val_acc

    def save_model(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)
        self.logger.log(f'Model saved to {path}')

    def load_model(self, path='model.pth'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.logger.log(f'Model loaded from {path}')

    def save_checkpoint(self, epoch, path='checkpoint.pth'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }
        torch.save(checkpoint, path)
        self.logger.log(f'Checkpoint saved at epoch {epoch} to {path}')

    def load_checkpoint(self, path='checkpoint.pth'):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.logger.log(f"Checkpoint loaded from {path}, starting from epoch {checkpoint['epoch']}")

    def save_history(self, path='history.json'):
        with open(path, 'w') as f:
            json.dump(self.history, f)
        self.logger.log(f'History saved to {path}')

    def load_history(self, path='history.json'):
        with open(path, 'r') as f:
            self.history = json.load(f)
        self.logger.log(f'History loaded from {path}')

    def plot_history(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.log('Matplotlib not installed. Please install it to plot history.')
            return

        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train Accuracy')
        plt.plot(epochs, self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    def save_hyperparameters(self, hyperparams, path='hyperparameters.json'):
        with open(path, 'w') as f:
            json.dump(hyperparams, f)
        self.logger.log(f'Hyperparameters saved to {path}')

    def load_hyperparameters(self, path='hyperparameters.json'):
        with open(path, 'r') as f:
            hyperparams = json.load(f)
        self.logger.log(f'Hyperparameters loaded from {path}')
        return hyperparams
