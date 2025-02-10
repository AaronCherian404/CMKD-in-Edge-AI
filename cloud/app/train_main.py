import torch
from models import TeacherModel
from inference import StudentModel
from trainer import CMKDTrainer
from data_handler import DataHandler
from evaluation.metrics import PerformanceMetrics
from evaluation.comparison import ModelComparison
from evaluation.visualization import ResultsVisualizer

class BaselineModel(StudentModel):
    """
    Identical architecture to StudentModel but trained without knowledge distillation
    """
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    
    def train_epoch(self, train_loader, device):
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

def main():
    # Initialize models
    teacher_model = TeacherModel()
    student_model = StudentModel()
    baseline_model = BaselineModel()
    
    # Initialize data handler
    data_handler = DataHandler('data/nyud')
    train_loader, val_loader, test_loader = data_handler.get_nyud_dataset()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize trainer for CMKD model
    trainer = CMKDTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        device=device
    )
    
    # Training history
    cmkd_history = {'accuracy': [], 'loss': []}
    baseline_history = {'accuracy': [], 'loss': []}
    
    # Initialize metrics calculator
    metrics_calculator = PerformanceMetrics(device)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train CMKD model
        cmkd_loss = trainer.train_epoch(train_loader)
        cmkd_metrics = metrics_calculator.evaluate_model(student_model, val_loader)
        cmkd_history['accuracy'].append(cmkd_metrics['accuracy'])
        cmkd_history['loss'].append(cmkd_loss)
        
        # Train baseline model
        baseline_model.train_epoch(train_loader, device)
        baseline_metrics = metrics_calculator.evaluate_model(baseline_model, val_loader)
        baseline_history['accuracy'].append(baseline_metrics['accuracy'])
        
        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'student_model_state_dict': student_model.state_dict(),
                'baseline_model_state_dict': baseline_model.state_dict(),
                'cmkd_history': cmkd_history,
                'baseline_history': baseline_history
            }, f'checkpoints/models_epoch_{epoch+1}.pth')
    
    # Final evaluation
    model_comparison = ModelComparison(
        student_model, baseline_model, metrics_calculator
    )
    comparison_results = model_comparison.compare_models(test_loader)
    
    # Visualize results
    visualizer = ResultsVisualizer()
    visualizer.plot_metrics_comparison(comparison_results)
    visualizer.plot_training_progress(cmkd_history, baseline_history)
    
    # Print final results
    print("\nFinal Results:")
    print("\nCMKD Model Metrics:")
    for metric, value in comparison_results['cmkd_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nBaseline Model Metrics:")
    for metric, value in comparison_results['baseline_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nImprovements:")
    for metric, improvement in comparison_results['improvements'].items():
        print(f"{metric}: {improvement:+.2f}%")

if __name__ == '__main__':
    main()