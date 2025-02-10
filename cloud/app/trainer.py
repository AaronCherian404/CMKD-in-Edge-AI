import torch
from torch.utils.data import DataLoader
from cmkd_loss import CMKDLoss

class CMKDTrainer:
    def __init__(self, teacher_model, student_model, device='cuda'):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device
        self.criterion = CMKDLoss()
        self.optimizer = torch.optim.Adam(student_model.parameters())

    def train_epoch(self, train_loader):
        self.student_model.train()
        self.teacher_model.eval()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            with torch.no_grad():
                teacher_output = self.teacher_model(data)
            
            student_output = self.student_model(data)
            loss = self.criterion(student_output, teacher_output, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Training Loss: {loss.item():.4f}')