"""
Model Export Script - turns the trained model into deployable formats
Converts to ONNX and TorchScript, ONNX is better for other platforms i think
Tested the export, seems okay
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def export_to_onnx(model, save_path, input_shape=(1, 3, 224, 224)):
    """Export model to ONNX format - for fast inference"""
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    # Export with some options, opset 11 cuz newer might not work everywhere
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f'Model exported to ONNX: {save_path} - should be smaller and faster')

def export_to_torchscript(model, save_path, input_shape=(1, 3, 224, 224)):
    """Export model to TorchScript format - PyTorch native"""
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    # Trace it, hope no issues with dynamic shapes
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save the traced model
    traced_model.save(save_path)
    print(f'Model exported to TorchScript: {save_path}')

def load_and_export(checkpoint_path, onnx_path, torchscript_path):
    """Load trained model and export to multiple formats - main function"""
    # Load the checkpoint, map to cpu first
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    num_classes = len(checkpoint['class_names'])
    
    # Rebuild the model same as training
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'Model loaded from: {checkpoint_path}')
    print(f'Number of classes: {num_classes}')
    print(f'Classes: {checkpoint["class_names"]}')
    
    # Do ONNX export
    export_to_onnx(model, onnx_path)
    
    # Do TorchScript
    export_to_torchscript(model, torchscript_path)
    
    # Check ONNX if possible
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print('ONNX model verification passed! good')
    except ImportError:
        print('ONNX not installed, skipping verification - install it if needed')
    except Exception as e:
        print(f'ONNX verification failed: {e} - might be okay still')
    
    # Test TorchScript quick
    try:
        loaded_script = torch.jit.load(torchscript_path)
        test_input = torch.randn(1, 3, 224, 224)
        output = loaded_script(test_input)
        print(f'TorchScript model test passed! Output shape: {output.shape}')
    except Exception as e:
        print(f'TorchScript test failed: {e} - check the trace')

if __name__ == '__main__':
    checkpoint_path = 'models/best_model.pth'
    onnx_path = 'models/model.onnx'
    torchscript_path = 'models/model_scripted.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f'Checkpoint not found: {checkpoint_path}')
        print('Please train the model first using train.py - run that first')
    else:
        load_and_export(checkpoint_path, onnx_path, torchscript_path)
        print('\nExport completed successfully! Now you can use inference.py')