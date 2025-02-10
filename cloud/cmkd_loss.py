import torch
import torch.nn as nn
import torch.nn.functional as F

class CMKDLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, labels=None):
        soft_targets = F.softmax(teacher_outputs / self.T, dim=1)
        student_log_probs = F.log_softmax(student_outputs / self.T, dim=1)
        distillation_loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (self.T ** 2)
        
        if labels is not None:
            student_loss = self.criterion(student_outputs, labels)
            return self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        return distillation_loss