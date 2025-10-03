"""
Simple GUI App for Testing Material Classification Model
Drag and drop an image to get prediction, confidence, and score
Uses Tkinter for GUI, ONNX for inference
Made this to demo the model easily without terminal
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import json
import os
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    import torch
    print("ONNX not available, using TorchScript fallback")

class MaterialGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Material Classification Tester - Drag Drop Image")
        self.root.geometry("600x500")
        
        # Load model and classes
        self.class_names = self.load_classes()
        self.classifier = self.load_model()
        
        # GUI elements
        self.label = tk.Label(root, text="Drag and drop an image here or click Load Image", font=("Arial", 14))
        self.label.pack(pady=20)
        
        self.result_text = tk.Text(root, height=10, width=70)
        self.result_text.pack(pady=10)
        
        # Button for file dialog
        load_btn = tk.Button(root, text="Load Image", command=self.load_image, font=("Arial", 12))
        load_btn.pack(pady=10)
        
        # Status
        self.status = tk.Label(root, text="Ready - Use Load Image button", font=("Arial", 10))
        self.status.pack(pady=5)
    
    def load_classes(self):
        """Load class names from json or checkpoint"""
        class_path = Path('data/materials/class_names.json')
        if class_path.exists():
            with open(class_path, 'r') as f:
                data = json.load(f)
                return data['class_names']
        else:
            # Fallback to checkpoint
            checkpoint_path = 'models/best_model.pth'
            if os.path.exists(checkpoint_path):
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                return checkpoint['class_names']
        return ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # default
    
    def load_model(self):
        """Load the model for inference"""
        if ONNX_AVAILABLE and os.path.exists('models/model.onnx'):
            session = ort.InferenceSession('models/model.onnx')
            input_name = session.get_inputs()[0].name
            return {'session': session, 'input_name': input_name, 'backend': 'onnx'}
        elif os.path.exists('models/model_scripted.pt'):
            model = torch.jit.load('models/model_scripted.pt')
            model.eval()
            return {'model': model, 'backend': 'torchscript'}
        else:
            messagebox.showerror("Error", "No model found. Train and export first.")
            self.root.quit()
    
    def preprocess(self, image_path):
        """Preprocess image like in inference.py"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path):
        """Predict using the loaded model"""
        input_data = self.preprocess(image_path)
        
        if self.classifier['backend'] == 'onnx':
            outputs = self.classifier['session'].run(None, {self.classifier['input_name']: input_data.astype(np.float32)})
            logits = outputs[0][0]
        else:
            import torch
            input_tensor = torch.from_numpy(input_data)
            with torch.no_grad():
                outputs = self.classifier['model'](input_tensor)
            logits = outputs[0].numpy()
        
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])
        predicted_class = self.class_names[pred_idx]
        
        return predicted_class, confidence, probabilities
    
    def load_image(self):
        """Load image via file dialog"""
        image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if image_path:
            self.test_image(image_path)
    
    def test_image(self, image_path):
        """Test the image and display result"""
        try:
            predicted_class, confidence, probabilities = self.predict(image_path)
            
            # Display image
            image = Image.open(image_path)
            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            
            self.label.configure(image=photo, text=f"Image: {Path(image_path).name}")
            self.label.image = photo  # keep reference
            
            # Update result text
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Predicted Class: {predicted_class}\n")
            self.result_text.insert(tk.END, f"Confidence: {confidence:.4f}\n")
            self.result_text.insert(tk.END, f"Low Confidence: {confidence < 0.7}\n\n")
            self.result_text.insert(tk.END, "All Probabilities:\n")
            for cls, prob in sorted(zip(self.class_names, probabilities), key=lambda x: x[1], reverse=True):
                self.result_text.insert(tk.END, f"  {cls}: {prob:.4f}\n")
            
            self.status.configure(text="Prediction complete!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {e}")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

if __name__ == '__main__':
    root = tk.Tk()
    app = MaterialGUI(root)
    app.run()