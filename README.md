# Cotton Disease Classification using ConvNeXt-GCN

A deep learning model that combines ConvNeXt and Graph Convolutional Networks (GCN) for accurate cotton disease classification. This hybrid architecture leverages both traditional convolutional features and graph-based structural information to improve classification accuracy.

## Features

- Hybrid architecture combining ConvNeXt and GCN
- Superpixel-based graph construction
- Data augmentation for improved robustness
- Support for multiple disease classes
- Real-time prediction capabilities

## Requirements

```
torch
torch-geometric
torchvision
tensorflow
scikit-image
numpy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cotton-disease-classification.git
cd cotton-disease-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:
```
Cotton Disease/
├── train/
│   ├── class1/
│   ├── class2/
│   ├── class3/
│   └── class4/
└── val/
    ├── class1/
    ├── class2/
    ├── class3/
    └── class4/
```

## Usage

1. Prepare your data:
```python
train_datagenerator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagenerator = ImageDataGenerator(rescale=1./255)
```

2. Load and train the model:
```python
# Initialize model
model = ConvNeXtGCNModel(num_classes=4, hidden_dim=128, max_segments=100)
model = model.to(device)

# Set up optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Train the model
train_model(model, train_data, optimizer, criterion, device)
```

3. Make predictions:
```python
prediction = predict(model, image)
```

## Model Architecture

The model combines two powerful architectures:

1. **ConvNeXt Backbone**
   - Pre-trained feature extractor
   - Deep hierarchical feature learning

2. **Graph Convolutional Network**
   - Superpixel-based graph construction
   - Structure-aware feature processing
   - Spatial relationship modeling

## Training Process

The training pipeline includes:
- Image preprocessing and augmentation
- Superpixel generation
- Graph construction
- Combined feature extraction
- Classification with cross-entropy loss



## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Acknowledgments

- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
