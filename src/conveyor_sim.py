"""
Conveyor Belt Simulation - the real time part, mimics the belt moving
Simulates classifying materials as they come, with logging and overrides
Took a while to get the timing right, but its decent now
"""

import os
import time
import csv
import json
from datetime import datetime
from pathlib import Path
import argparse
from inference import MaterialClassifier

class ConveyorSimulator:
    """Simulates a conveyor belt with material classification - the simulation class"""
    
    def __init__(self, model_path, image_source, interval=1.0, confidence_threshold=0.7,
                 enable_override=False, active_learning=False):
        """
        Initialize conveyor simulator - sets up everything
        
        Args:
            model_path: model file path
            image_source: folder with images or video
            interval: seconds between frames
            confidence_threshold: min conf, below is low
            enable_override: allow manual fixes
            active_learning: queue bad predictions for retrain
        """
        self.model_path = model_path
        self.image_source = image_source
        self.interval = interval
        self.confidence_threshold = confidence_threshold
        self.enable_override = enable_override
        self.active_learning = active_learning
        
        # Init classifier
        print('Initializing classifier...')
        self.classifier = MaterialClassifier(model_path, confidence_threshold=confidence_threshold)
        
        # Output dirs
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_csv = self.results_dir / f'conveyor_results_{timestamp}.csv'
        self.override_log = self.results_dir / f'override_log_{timestamp}.json'
        self.retraining_queue = self.results_dir / f'retraining_queue_{timestamp}.json'
        
        # Logs lists
        self.overrides = []
        self.retraining_samples = []
        
        # Get images
        self.images = self._get_image_list()
        print(f'Found {len(self.images)} images to process - hope enough for test')
        
        # Stats dict
        self.stats = {
            'total_processed': 0,
            'low_confidence_count': 0,
            'override_count': 0,
            'class_distribution': {}
        }
    
    def _get_image_list(self):
        """Get list of images from source - supports folder or single file"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = []
        
        source_path = Path(self.image_source)
        
        if source_path.is_dir():
            for ext in image_extensions:
                images.extend(source_path.glob(f'*{ext}'))
                images.extend(source_path.glob(f'*{ext.upper()}'))
        elif source_path.is_file():
            # Single image or video, but video not implemented yet
            if source_path.suffix.lower() in image_extensions:
                images = [source_path]
            else:
                print(f'Unsupported file type: {source_path.suffix} - only images for now')
        
        return sorted(images)
    
    def _initialize_csv(self):
        """Initialize CSV file with headers - for logging results"""
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame_id', 'image_path', 'predicted_class',
                'confidence', 'low_confidence_flag', 'manual_override',
                'corrected_class', 'processing_time_ms'
            ])
    
    def _log_result(self, frame_id, result, override_class=None, processing_time=0):
        """Log result to CSV - append the row"""
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                frame_id,
                result['image_path'],
                result['predicted_class'],
                f"{result['confidence']:.4f}",
                result['low_confidence_flag'],
                override_class is not None,
                override_class if override_class else '',
                f"{processing_time:.2f}"
            ])
    
    def _handle_manual_override(self, frame_id, result):
        """Handle manual override for misclassification - ask user if wrong"""
        if not self.enable_override:
            return None
        
        print('\n--- Manual Override ---')
        print(f"Predicted: {result['predicted_class']} (Confidence: {result['confidence']:.4f})")
        print('Is this correct? (y/n/skip): ', end='')
        
        try:
            response = input().strip().lower()
            
            if response == 'n':
                print('Available classes:', ', '.join(self.classifier.class_names))
                print('Enter correct class: ', end='')
                correct_class = input().strip()
                
                if correct_class in self.classifier.class_names:
                    # Log the override
                    override_entry = {
                        'frame_id': frame_id,
                        'image_path': str(result['image_path']),
                        'predicted_class': result['predicted_class'],
                        'correct_class': correct_class,
                        'confidence': result['confidence'],
                        'timestamp': datetime.now().isoformat()
                    }
                    self.overrides.append(override_entry)
                    self.stats['override_count'] += 1
                    
                    # Add to queue if active learning
                    if self.active_learning:
                        self.retraining_samples.append(override_entry)
                    
                    return correct_class
                else:
                    print(f'Invalid class: {correct_class} - try again')
            
        except (KeyboardInterrupt, EOFError):
            print('\nSkipping override... ctrl c i guess')
        
        return None
    
    def _display_result(self, frame_id, result, override_class=None, processing_time=0):
        """Display result in console - pretty print the prediction"""
        print(f'\n{"="*60}')
        print(f'Frame #{frame_id} | {datetime.now().strftime("%H:%M:%S")}')
        print(f'{"="*60}')
        print(f'Image: {Path(result["image_path"]).name}')
        print(f'Predicted Class: {result["predicted_class"]}')
        print(f'Confidence: {result["confidence"]:.4f}')
        print(f'Processing Time: {processing_time:.2f}ms')
        
        if result['low_confidence_flag']:
            print('⚠️  LOW CONFIDENCE WARNING! - check this one')
        
        if override_class:
            print(f'✓ Manual Override: {override_class} - user fixed it')
        
        # Top 3 with bars, looks cool
        sorted_probs = sorted(result['all_probabilities'].items(),
                            key=lambda x: x[1], reverse=True)[:3]
        print('\nTop 3 Predictions:')
        for i, (cls, prob) in enumerate(sorted_probs, 1):
            bar = '█' * int(prob * 20)
            print(f'  {i}. {cls:15s} {prob:.4f} {bar}')
    
    def _save_logs(self):
        """Save override and retraining logs - json for easy read"""
        if self.overrides:
            with open(self.override_log, 'w') as f:
                json.dump(self.overrides, f, indent=4)
            print(f'\nOverride log saved: {self.override_log}')
        
        if self.retraining_samples:
            with open(self.retraining_queue, 'w') as f:
                json.dump(self.retraining_samples, f, indent=4)
            print(f'Retraining queue saved: {self.retraining_queue} - use for next train')
    
    def _print_statistics(self):
        """Print final statistics - summary at end"""
        print(f'\n{"="*60}')
        print('SIMULATION STATISTICS')
        print(f'{"="*60}')
        print(f'Total Frames Processed: {self.stats["total_processed"]}')
        print(f'Low Confidence Count: {self.stats["low_confidence_count"]}')
        print(f'Manual Overrides: {self.stats["override_count"]}')
        print(f'\nClass Distribution:')
        for cls, count in sorted(self.stats['class_distribution'].items()):
            print(f'  {cls:15s}: {count}')
        print(f'\nResults saved to: {self.output_csv} - check it out')
    
    def run(self):
        """Run the conveyor simulation - main loop"""
        print(f'\n{"="*60}')
        print('CONVEYOR BELT SIMULATION STARTED - lets see how it does')
        print(f'{"="*60}')
        print(f'Model: {self.model_path}')
        print(f'Source: {self.image_source}')
        print(f'Interval: {self.interval}s')
        print(f'Confidence Threshold: {self.confidence_threshold}')
        print(f'Manual Override: {self.enable_override}')
        print(f'Active Learning: {self.active_learning}')
        print(f'{"="*60}\n')
        
        # Init CSV
        self._initialize_csv()
        
        try:
            for frame_id, image_path in enumerate(self.images, 1):
                # Timing for conveyor
                start_time = time.time()
                
                # Classify the image
                result = self.classifier.predict(str(image_path))
                
                processing_time = (time.time() - start_time) * 1000
                
                # Update stats
                self.stats['total_processed'] += 1
                if result['low_confidence_flag']:
                    self.stats['low_confidence_count'] += 1
                
                pred_class = result['predicted_class']
                self.stats['class_distribution'][pred_class] = \
                    self.stats['class_distribution'].get(pred_class, 0) + 1
                
                # Override if low or enabled
                override_class = None
                if result['low_confidence_flag'] or self.enable_override:
                    override_class = self._handle_manual_override(frame_id, result)
                
                # Show result
                self._display_result(frame_id, result, override_class, processing_time)
                
                # Log it
                self._log_result(frame_id, result, override_class, processing_time)
                
                # Wait interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            print('\n\nSimulation interrupted by user - okay stopping')
        
        finally:
            # Save logs
            self._save_logs()
            
            # Print stats
            self._print_statistics()

def main():
    parser = argparse.ArgumentParser(description='Conveyor Belt Material Classification Simulator - the sim part')
    parser.add_argument('--model', type=str, default='models/model.onnx',
                       help='Path to model file (ONNX or TorchScript)')
    parser.add_argument('--source', type=str, default='data/test_images',
                       help='Path to image folder or video')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Time interval between frames (seconds)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold')
    parser.add_argument('--override', action='store_true',
                       help='Enable manual override for misclassifications')
    parser.add_argument('--active-learning', action='store_true',
                       help='Enable active learning queue')
    
    args = parser.parse_args()
    
    # Check model
    if not os.path.exists(args.model):
        # Try other model
        if 'onnx' in args.model and os.path.exists('models/model_scripted.pt'):
            args.model = 'models/model_scripted.pt'
            print(f'ONNX not found, using TorchScript instead')
        elif os.path.exists('models/model.onnx'):
            args.model = 'models/model.onnx'
        else:
            print(f'Model not found: {args.model}')
            print('Train and export first with train.py and export_model.py')
            return
    
    # Check source
    if not os.path.exists(args.source):
        print(f'Source not found: {args.source}')
        print('Need images in folder or single file')
        return
    
    # Make simulator
    simulator = ConveyorSimulator(
        model_path=args.model,
        image_source=args.source,
        interval=args.interval,
        confidence_threshold=args.threshold,
        enable_override=args.override,
        active_learning=args.active_learning
    )
    
    # Run it
    simulator.run()

if __name__ == '__main__':
    main()