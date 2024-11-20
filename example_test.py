"""  
a Python script to train ResNet-18 using PyTorch Lightning. The dataset includes 5 categories.  
Report the classification accuracy and confusion matrix with torch-metrics.  
  
Use 5-fold stratified sampling.  
Report the final average classification accuracies at the end of the program.  
"""  
    
import numpy as np  
import pytorch_lightning as pl  
from pytorch_lightning.loggers import TensorBoardLogger  
import torch  
from torch.nn import functional as F  
from torch.utils.data import DataLoader, TensorDataset  
from torchvision import models, transforms  
import torchmetrics  
from sklearn.model_selection import StratifiedKFold  
import seaborn  
import matplotlib.pyplot as plt  
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import os
  
  

MAX_EPOCH = 100  
fold_id = 0  

  
class Classifier(pl.LightningModule):  
    def __init__(self, num_classes: int, model_type: str = 'resnet18'):  
        super().__init__()  
        self.model_type = model_type  
        if model_type == 'resnet18':  
            self.model = models.resnet18(pretrained=True)  
            self.model.fc = torch.nn.Sequential(  
                    torch.nn.Linear(self.model.fc.in_features, 128),  
                    torch.nn.ReLU(),  
                    torch.nn.Linear(128, 64),  
                    torch.nn.ReLU(),  
                    torch.nn.Linear(64, num_classes)  
                    )  
        elif model_type == 'mlp':  
            self.model = torch.nn.Sequential(  
                torch.nn.Linear(784, 128),  
                torch.nn.ReLU(),  
                torch.nn.Linear(128, 64),  
                torch.nn.ReLU(),  
                torch.nn.Linear(64, num_classes)  
            )  
        else:  
            raise ValueError(f'Invalid model_type: {model_type}')  
        self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes)  
        self.conf_mat = torchmetrics.classification.MulticlassConfusionMatrix(num_classes, normalize='true')  
  
    def forward(self, x):
        if self.model_type == 'resnet18':
            if x.dim() == 3:  # If the input is (batch_size, 28, 28)
                x = x.unsqueeze(1)  # Add a channel dimension to make it (batch_size, 1, 28, 28)
            x = x.repeat(1, 3, 1, 1)  # Convert to 3-channel image
            # x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert to 3-channel image
        else:
            x = x.view(x.size(0), -1)  # Flatten for MLP
        return self.model(x.float())
  
    def training_step(self, batch, batch_idx):  
        x, y = batch  
        y_hat = self(x)  
        loss = F.cross_entropy(y_hat, y.long())  
        self.log('train_loss', loss, )  
        return loss  
  
    def validation_step(self, batch, batch_idx):  
        x, y = batch  
        y_hat = self(x)  
        self.log('val_accuracy', self.accuracy, on_epoch=True, prog_bar=True)  
        self.log('val_loss', F.cross_entropy(y_hat, y.long()), on_step=True, prog_bar=True)  
        self.conf_mat.update(y_hat, y)  
        self.accuracy.update(y_hat, y)  
  
    def on_validation_end(self):  
        # conf_matrix = self.conf_mat.compute()  
        # print(conf_matrix)  
        # plt.figure()  
        # seaborn.heatmap(conf_matrix.cpu(), annot=True)  
        # plt.savefig(f'conf_mat_{fold_id}.png')  
        accuracy_computed = self.accuracy.compute()  
        print(f'Fold Accuracy={accuracy_computed}')  
  
    def configure_optimizers(self):  
        return torch.optim.Adam(self.parameters(), lr=0.00001)  


transform = ToTensor()
train_data= MNIST(os.getcwd(), train=True, download=True,transform=transform)
val_data=  MNIST(os.getcwd(), train=False, download=True,transform=transform)

 

# Create DataLoaders  
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)  
val_loader = DataLoader(val_data, batch_size=256)  

# Model  
model = Classifier(num_classes=10)  

# Training  
tensorboard_logger = TensorBoardLogger(save_dir='.', version=fold_id)  
trainer = pl.Trainer(max_epochs=MAX_EPOCH, devices=1, accelerator='cuda', logger=tensorboard_logger)  

trainer.fit(model, train_loader, val_loader)  


