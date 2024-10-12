from utils.methods import *
from torchvision import transforms
from torch.utils.data import random_split, TensorDataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def expand_channels(x):
    """
    Expand the number of channels from 1 to 3
    """
    return x.repeat(1, 3, 1, 1)


def load_data_resnet(batch_size: int, file_path: str = 'data/mnist.pkl.gz') -> Tuple[torch.utils.data.DataLoader, 
                                                                                      torch.utils.data.DataLoader, 
                                                                                      torch.utils.data.DataLoader]:
    """
    Load the data from the file path and return the train, validation and test loaders
    """
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open(file_path), map_location=torch.device(device), weights_only=True)
    
    data_train = data_train.view(-1, 1, 28, 28)
    data_test = data_test.view(-1, 1, 28, 28)

    transform = transforms.Compose([
        transforms.Lambda(expand_channels)
    ])

    data_train = transform(data_train)
    data_test = transform(data_test)
    
    train_dataset = TensorDataset(data_train, label_train)
    train_dataset, validation_dataset = random_split(train_dataset, [0.9, 0.1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(data_test, label_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, validation_loader, test_loader

def get_resnet18_model(num_classes: int = 10):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
  
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)
