class ModelComparison:
    def __init__(self, cmkd_model, baseline_model, metrics_calculator):
        self.cmkd_model = cmkd_model
        self.baseline_model = baseline_model
        self.metrics_calculator = metrics_calculator
        self.comparison_results = {}
    
    def compare_models(self, test_loader):
        # Evaluate CMKD model
        cmkd_metrics = self.metrics_calculator.evaluate_model(
            self.cmkd_model, test_loader
        )
        
        # Evaluate baseline model
        baseline_metrics = self.metrics_calculator.evaluate_model(
            self.baseline_model, test_loader
        )
        
        # Calculate improvements
        improvements = {}
        for metric in cmkd_metrics.keys():
            if metric in ['cpu_memory', 'gpu_memory', 'inference_time']:
                # For these metrics, lower is better
                improvement = ((baseline_metrics[metric] - cmkd_metrics[metric]) 
                             / baseline_metrics[metric] * 100)
            else:
                # For accuracy metrics, higher is better
                improvement = ((cmkd_metrics[metric] - baseline_metrics[metric]) 
                             / baseline_metrics[metric] * 100)
            improvements[metric] = improvement
        
        self.comparison_results = {
            'cmkd_metrics': cmkd_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': improvements
        }
        
        return self.comparison_results
