from models import TeacherModel
from evaluation.metrics import PerformanceMetrics
from evaluation.comparison import ModelComparison
from evaluation.visualization import ResultsVisualizer
from edge.app.inference import StudentModel  # Updated import path
import torch
from trainer import CMKDTrainer
from data_handler import DataHandler

def main():
    # Initialize models
    teacher_model = TeacherModel()
    student_model = StudentModel()
    
    # Initialize data handler
    data_handler = DataHandler('data/nyud')
    train_loader, val_loader = data_handler.get_nyud_dataset()
    
    # Initialize trainer
    trainer = CMKDTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        trainer.train_epoch(train_loader)
        
        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'student_model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
            }, f'checkpoints/student_model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()