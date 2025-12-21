import torch
import torch.nn as nn
from torch.export import export
from executorch.exir import to_edge


class SimpleImageClassifier(nn.Module):
    """A simple CNN classifier for demonstration."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def export_to_executorch(model, example_input, output_path):
    """Export a PyTorch model to ExecuTorch format."""
    exported_program = export(model, example_input)
    edge_program = to_edge(exported_program)
    executorch_program = edge_program.to_executorch()
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)
    return executorch_program


def main():
    model = SimpleImageClassifier(num_classes=10)
    model.eval()
    
    # Create example input (batch_size=1, channels=3, height=32, width=32)
    example_input = (torch.randn(1, 3, 32, 32),)
    
    # Export to ExecuTorch
    output_file = "image_classifier.pte"
    executorch_program = export_to_executorch(model, example_input, output_file)
    
    print(f"\n✓ Model successfully exported!")
    print(f"  File: {output_file}")
    print(f"  Size: {len(executorch_program.buffer) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
