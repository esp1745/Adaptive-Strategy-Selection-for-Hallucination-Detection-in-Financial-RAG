# Create a simple logging system
# src/utils/logger.py

import json
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    """Track experiments and results"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_experiment = None
    
    def start_experiment(self, name, config):
        """Start logging a new experiment"""
        self.current_experiment = {
            'name': name,
            'start_time': datetime.now().isoformat(),
            'config': config,
            'results': []
        }
    
    def log_result(self, metric_name, value):
        """Log a metric"""
        if self.current_experiment:
            self.current_experiment['results'].append({
                'metric': metric_name,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
    
    def save_experiment(self):
        """Save experiment to disk"""
        if self.current_experiment:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.current_experiment['name']}_{timestamp}.json"
            
            with open(self.log_dir / filename, 'w') as f:
                json.dump(self.current_experiment, f, indent=2)
            
            print(f"Experiment saved: {filename}")

# Quick test
if __name__ == "__main__":
    logger = ExperimentLogger()
    logger.start_experiment(
        name='rag_baseline',
        config={'model': 'all-MiniLM-L6-v2', 'k': 3}
    )
    logger.log_result('retrieval_accuracy', 0.85)
    logger.save_experiment()