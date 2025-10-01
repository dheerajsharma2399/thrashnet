"""
Complete Pipeline Runner
Runs the entire pipeline from data preparation to simulation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class PipelineRunner:
    """Orchestrates the complete ML pipeline"""
    
    def __init__(self, skip_training=False, skip_export=False):
        self.skip_training = skip_training
        self.skip_export = skip_export
        self.project_root = Path.cwd()
        
    def run_command(self, command, description):
        """Run a shell command and handle errors"""
        print(f"\n{'='*60}")
        print(f"Step: {description}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(command)}\n")
        
        try:
            result = subprocess.run(
                command,
                check=True,
                cwd=self.project_root,
                text=True
            )
            print(f"\n✓ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {description} failed with error code {e.returncode}")
            return False
        except Exception as e:
            print(f"\n✗ {description} failed: {e}")
            return False
    
    def check_dataset(self):
        """Check if dataset is prepared"""
        train_dir = Path('data/materials/train')
        if not train_dir.exists() or not any(train_dir.iterdir()):
            print("\n⚠ Dataset not found or empty")
            print("Please prepare the dataset first:")
            print("  python src/prepare_data.py")
            return False
        
        # Count classes
        classes = [d for d in train_dir.iterdir() if d.is_dir()]
        if len(classes) < 5:
            print(f"\n⚠ Only {len(classes)} classes found. Minimum 5 required.")
            return False
        
        print(f"\n✓ Dataset found with {len(classes)} classes")
        return True
    
    def check_model(self):
        """Check if model is trained"""
        model_path = Path('models/best_model.pth')
        if not model_path.exists():
            print("\n⚠ Trained model not found")
            return False
        
        print("\n✓ Trained model found")
        return True
    
    def check_exported_model(self):
        """Check if model is exported"""
        onnx_path = Path('models/model.onnx')
        script_path = Path('models/model_scripted.pt')
        
        if not onnx_path.exists() and not script_path.exists():
            print("\n⚠ Exported model not found")
            return False
        
        print("\n✓ Exported model found")
        return True
    
    def prepare_data(self):
        """Run data preparation"""
        if not self.check_dataset():
            response = input("\nRun data preparation? (y/n): ").strip().lower()
            if response == 'y':
                return self.run_command(
                    ['python', 'src/prepare_data.py'],
                    "Data Preparation"
                )
            else:
                print("Skipping data preparation...")
                return False
        return True
    
    def train_model(self):
        """Run model training"""
        if self.skip_training:
            print("\n⊘ Skipping training (--skip-training flag)")
            return self.check_model()
        
        if not self.check_dataset():
            print("\n✗ Cannot train: Dataset not prepared")
            return False
        
        return self.run_command(
            ['python', 'src/train.py'],
            "Model Training"
        )
    
    def export_model(self):
        """Export model to ONNX and TorchScript"""
        if self.skip_export:
            print("\n⊘ Skipping export (--skip-export flag)")
            return self.check_exported_model()
        
        if not self.check_model():
            print("\n✗ Cannot export: Model not trained")
            return False
        
        return self.run_command(
            ['python', 'src/export_model.py'],
            "Model Export"
        )
    
    def run_simulation(self, source_dir='data/test_images', interval=1.0):
        """Run conveyor simulation"""
        if not self.check_exported_model():
            print("\n✗ Cannot run simulation: Model not exported")
            return False
        
        # Check if test images exist
        test_path = Path(source_dir)
        if not test_path.exists():
            print(f"\n⚠ Test images directory not found: {source_dir}")
            print("Creating sample directory...")
            test_path.mkdir(parents=True, exist_ok=True)
            print(f"Please add test images to: {test_path}")
            return False
        
        return self.run_command(
            ['python', 'src/conveyor_sim.py', '--source', source_dir, '--interval', str(interval)],
            "Conveyor Simulation"
        )
    
    def run_pipeline(self, run_simulation=True, simulation_source='data/test_images'):
        """Run the complete pipeline"""
        print("="*60)
        print("MATERIAL CLASSIFICATION PIPELINE")
        print("="*60)
        print(f"Project Root: {self.project_root}")
        print(f"Skip Training: {self.skip_training}")
        print(f"Skip Export: {self.skip_export}")
        print(f"Run Simulation: {run_simulation}")
        print("="*60)
        
        steps = [
            ("Dataset Check", self.prepare_data),
            ("Model Training", self.train_model),
            ("Model Export", self.export_model),
        ]
        
        if run_simulation:
            steps.append(("Conveyor Simulation", lambda: self.run_simulation(simulation_source)))
        
        # Execute pipeline
        for step_name, step_func in steps:
            success = step_func()
            if not success:
                print(f"\n{'='*60}")
                print(f"Pipeline stopped at: {step_name}")
                print(f"{'='*60}")
                return False
        
        # Pipeline completed
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        # Print summary
        print("\nGenerated Files:")
        if Path('models/best_model.pth').exists():
            print("  ✓ models/best_model.pth")
        if Path('models/model.onnx').exists():
            print("  ✓ models/model.onnx")
        if Path('models/model_scripted.pt').exists():
            print("  ✓ models/model_scripted.pt")
        
        results_dir = Path('results')
        if results_dir.exists():
            csv_files = list(results_dir.glob('conveyor_results_*.csv'))
            if csv_files:
                print(f"  ✓ {len(csv_files)} simulation result file(s)")
        
        print("\nNext Steps:")
        print("  1. Review results in: results/")
        print("  2. Check metrics: results/metrics.json")
        print("  3. View confusion matrix: results/confusion_matrix.png")
        print("  4. Read performance report: performance_report.md")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Run the complete ML pipeline')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training if model already exists')
    parser.add_argument('--skip-export', action='store_true',
                       help='Skip export if model already exported')
    parser.add_argument('--no-simulation', action='store_true',
                       help='Skip conveyor simulation')
    parser.add_argument('--simulation-source', type=str, default='data/test_images',
                       help='Source directory for simulation images')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only prepare dataset and exit')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train model and exit')
    
    args = parser.parse_args()
    
    # Create pipeline runner
    runner = PipelineRunner(
        skip_training=args.skip_training,
        skip_export=args.skip_export
    )
    
    # Handle specific modes
    if args.prepare_only:
        runner.prepare_data()
        return
    
    if args.train_only:
        if runner.prepare_data():
            runner.train_model()
        return
    
    # Run full pipeline
    success = runner.run_pipeline(
        run_simulation=not args.no_simulation,
        simulation_source=args.simulation_source
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()