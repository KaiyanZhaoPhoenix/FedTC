"""FedTC messages on top of the ORIGINAL clientFedAA SGD implementation."""
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from clients.clientFedAA import clientFedAA


class clientFedTC(clientFedAA):
    # Intentionally inherit train() unchanged: optimizer, clipping, epochs, decay.
    def bounded_loss(self, model, loss_cap, validation=True):
        self._prepare_training_partition()
        indices = self.validation_indices if validation else self.train_indices
        if not indices:
            raise ValueError("FedTC feedback requires a nonempty local validation split.")
        data = [self._full_train_data[i] for i in indices]
        loader = DataLoader(data, self.batch_size, shuffle=False, drop_last=False,
                            generator=torch.Generator().manual_seed(0))
        previous_mode = model.training
        total, count = 0., 0
        model.eval()
        try:
            with torch.no_grad():
                for x, y in loader:
                    if isinstance(x, (tuple, list)):
                        x = [value.to(self.device) for value in x]
                    else:
                        x = x.to(self.device)
                    losses = F.cross_entropy(model(x), y.to(self.device), reduction='none').clamp(max=loss_cap)
                    if not torch.isfinite(losses).all():
                        raise FloatingPointError(f"Client {self.id} returned nonfinite evaluation loss.")
                    total += float(losses.sum())
                    count += len(y)
        finally:
            model.train(previous_mode)
        return total / count

    def report_improvement(self, start_model, end_model, loss_cap):
        return self.bounded_loss(start_model, loss_cap) - self.bounded_loss(end_model, loss_cap)
