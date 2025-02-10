# evaluation/metrics.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import time
import psutil
import GPUtil

class PerformanceMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics_history = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'latency': [], 'memory_usage': [],
            'inference_time': [], 'energy_consumption': []
        }
    
    def calculate_accuracy(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        return accuracy_score(targets.cpu(), predicted.cpu())
    
    def calculate_precision_recall_f1(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets.cpu(), predicted.cpu(), average='weighted'
        )
        return precision, recall, f1
    
    def measure_latency(self, model, input_data):
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_data)
        end_time = time.time()
        return end_time - start_time
    
    def measure_memory_usage(self):
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            gpu_memory = gpu.memoryUsed  # MB
            return memory_usage, gpu_memory
        return memory_usage, 0
    
    def measure_energy_consumption(self, start_power, end_power):
        # Simplified energy calculation - in practice, you'd need hardware monitoring
        return end_power - start_power
    
    def evaluate_model(self, model, data_loader):
        model.eval()
        all_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'latency': 0.0,
            'cpu_memory': 0.0,
            'gpu_memory': 0.0,
            'inference_time': 0.0
        }
        
        num_batches = len(data_loader)
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Measure inference time and latency
                start_time = time.time()
                outputs = model(data)
                batch_inference_time = time.time() - start_time
                
                # Calculate accuracy metrics
                batch_accuracy = self.calculate_accuracy(outputs, targets)
                precision, recall, f1 = self.calculate_precision_recall_f1(outputs, targets)
                
                # Measure resource usage
                cpu_mem, gpu_mem = self.measure_memory_usage()
                
                # Update metrics
                all_metrics['accuracy'] += batch_accuracy
                all_metrics['precision'] += precision
                all_metrics['recall'] += recall
                all_metrics['f1'] += f1
                all_metrics['inference_time'] += batch_inference_time
                all_metrics['cpu_memory'] = max(all_metrics['cpu_memory'], cpu_mem)
                all_metrics['gpu_memory'] = max(all_metrics['gpu_memory'], gpu_mem)
        
        # Average the metrics
        for key in ['accuracy', 'precision', 'recall', 'f1', 'inference_time']:
            all_metrics[key] /= num_batches
            
        return all_metrics