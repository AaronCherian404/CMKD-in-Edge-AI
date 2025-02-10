import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ResultsVisualizer:
    def __init__(self, save_dir='results/'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_metrics_comparison(self, comparison_results):
        metrics = list(comparison_results['improvements'].keys())
        improvements = list(comparison_results['improvements'].values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, improvements)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('CMKD Improvements over Baseline (%)')
        plt.xticks(rotation=45)
        
        # Color code the bars based on improvement/degradation
        for bar, improvement in zip(bars, improvements):
            if improvement >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/improvements.png')
        plt.close()
    
    def plot_training_progress(self, cmkd_history, baseline_history):
        plt.figure(figsize=(12, 6))
        plt.plot(cmkd_history['accuracy'], label='CMKD')
        plt.plot(baseline_history['accuracy'], label='Baseline')
        plt.title('Training Accuracy Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{self.save_dir}/training_progress.png')
        plt.close()