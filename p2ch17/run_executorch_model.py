import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch


def load_executorch_model(model_path):
    return _load_for_executorch(model_path)


def run_inference(module, input_data):
    outputs = module.forward([input_data])
    return outputs[0]


def main():
    model_path = "image_classifier.pte"
    
    # Load the ExecuTorch model
    try:
        executorch_module = load_executorch_model(model_path)
    except FileNotFoundError:
        print(f"\n❌ Error: Model file '{model_path}' not found!")
        return
    
    # Create sample input (simulating camera input or sensor data)
    input_data = torch.randn(1, 3, 32, 32)
    print(f"[Device] Input shape: {input_data.shape}")
    
    # Run inference
    output = run_inference(executorch_module, input_data)
    
    # Process results
    print(f"\n[Device] Inference complete!")
    print(f"[Device] Output shape: {output.shape}")
    print(f"[Device] Output (logits): {output[0][:5].tolist()}")
    
    # Get predicted class
    predicted_class = torch.argmax(output[0]).item()
    confidence = torch.softmax(output[0], dim=0)[predicted_class].item()
    
    print(f"\n[Device] Prediction Results:")
    print(f"  Predicted Class: {predicted_class}")
    print(f"  Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
