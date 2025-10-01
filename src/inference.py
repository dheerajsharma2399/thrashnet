"""
Lightweight Inference Script - for testing single images quickly
Uses ONNX Runtime mostly, falls back to TorchScript if no ONNX
Added the confidence thing cuz sometimes model is not sure
"""

import numpy as np
from PIL import Image
import json
import os

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    import torch
    print("ONNX Runtime not available, falling back to PyTorch")

class MaterialClassifier:
    """Lightweight classifier for material classification - does the prediction part"""
    
    def __init__(self, model_path, class_names_path=None, confidence_threshold=0.7):
        """
        Initialize classifier - loads the model and stuff
        
        Args:
            model_path: path to ONNX or TorchScript model file
            class_names_path: optional json with class names
            confidence_threshold: min confidence, below this is low
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Load class names, try json first then checkpoint
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                data = json.load(f)
                self.class_names = data['class_names']
        else:
            checkpoint_path = 'models/best_model.pth'
            if os.path.exists(checkpoint_path):
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.class_names = checkpoint['class_names']
            else:
                raise ValueError("No class names found. Need json or checkpoint file.")
        
        # Load model, ONNX is preferred for speed
        if model_path.endswith('.onnx') and ONNX_AVAILABLE:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.backend = 'onnx'
        elif model_path.endswith('.pt'):
            self.model = torch.jit.load(model_path)
            self.model.eval()
            self.backend = 'torchscript'
        else:
            raise ValueError("Bad model format. Use .onnx or .pt only.")
        
        print(f'Classifier initialized with {self.backend} backend')
        print(f'Classes: {self.class_names}')
        print(f'Confidence threshold: {self.confidence_threshold} - anything below is suspicious')
    
    def preprocess(self, image_path):
        """
        Preprocess image for inference - resize and normalize
        
        Args:
            image_path: path to the image
            
        Returns:
            ready tensor for model
        """
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # model expects 224
        
        # To numpy and normalize to 0-1
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # ImageNet norm, important for pretrained models
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # CHW for the model
        img_array = img_array.transpose(2, 0, 1)
        
        # Batch dim
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path):
        """
        Predict class for a single image - the main predict function
        
        Args:
            image_path: path to image
            
        Returns:
            dict with class, conf, probs etc
        """
        # Preprocess first
        input_data = self.preprocess(image_path)
        
        # Run inference
        if self.backend == 'onnx':
            outputs = self.session.run(None, {self.input_name: input_data.astype(np.float32)})
            logits = outputs[0][0]
        else:  # torchscript
            import torch
            input_tensor = torch.from_numpy(input_data)
            with torch.no_grad():
                outputs = self.model(input_tensor)
            logits = outputs[0].numpy()
        
        # Softmax to get probs
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        
        # Best guess
        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])
        predicted_class = self.class_names[pred_idx]
        
        # Low conf check
        low_confidence = confidence < self.confidence_threshold
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': {self.class_names[i]: float(probabilities[i])
                                 for i in range(len(self.class_names))},
            'low_confidence_flag': low_confidence,
            'image_path': image_path
        }
        
        return result
    
    def batch_predict(self, image_paths):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                print(f'Error processing {img_path}: {e}')
                results.append({
                    'predicted_class': 'ERROR',
                    'confidence': 0.0,
                    'low_confidence_flag': True,
                    'image_path': img_path,
                    'error': str(e)
                })
        return results

# Test inference - run with an image path
if __name__ == '__main__':
    import sys
    
    # Pick model, ONNX first cuz faster
    if ONNX_AVAILABLE and os.path.exists('models/model.onnx'):
        model_path = 'models/model.onnx'
    elif os.path.exists('models/model_scripted.pt'):
        model_path = 'models/model_scripted.pt'
    else:
        print('No model found. Train and export first please.')
        sys.exit(1)
    
    # Init classifier
    classifier = MaterialClassifier(model_path, confidence_threshold=0.7)
    
    # Test on image from arg
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            result = classifier.predict(image_path)
            print('\n=== Prediction Result ===')
            print(f"Image: {result['image_path']}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Low Confidence: {result['low_confidence_flag']}")
            print(f"\nAll Probabilities:")
            for cls, prob in result['all_probabilities'].items():
                print(f"  {cls}: {prob:.4f}")
        else:
            print(f'Image not found: {image_path} - check the path')
    else:
        print('Usage: python inference.py <image_path> - give me an image to test!')