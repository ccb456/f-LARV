import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_funt = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        # ### BCEloss
        loss = self.loss_funt(logits, labels)
        return loss

    def get_label(self, logits):
        # Change to binary
        probs = torch.sigmoid(logits)
        output = (probs > 0.5).float()
        return output



class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):
        bce_loss = self.bce(logits, labels)
        pt = torch.exp(-bce_loss)                  # pt = probability of true class
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        if self.alpha is not None:
            alpha_factor = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            focal_loss = alpha_factor * focal_loss

        return focal_loss.mean()                   # 或 .sum() / valid elements

    def get_label(self, logits):
        # Change to binary
        probs = torch.sigmoid(logits)
        output = (probs > 0.5).float()
        return output