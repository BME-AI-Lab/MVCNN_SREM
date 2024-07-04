import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn as nn
import timm
import timm.data
import timm.loss
import timm.optim
import timm.utils
import torchmetrics
import wandb
#from personal_model.residual_attention_network import ResidualAttentionModel_56 as ResidualAttentionModel
import personal_model.attention56 as ResidualAttentionModel
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class MultiViewImageDataset(Dataset):
    def __init__(self, root_dir, num_views, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            num_views (int): Number of views per instance.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.transform = transform
        self.labels = pd.read_csv(os.path.join(root_dir, 'labels.csv'))
        self.instances = self.labels['folder'].values


    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance_name = self.instances[idx]
        instance_path = os.path.join(self.root_dir, instance_name)
        
        # Load each view for the current instance
        views_num = [1, 2, 3, 4, 5, 6, 7, 8] # Placeholder for the views
        
        views = []
        for i in views_num:
            view_path = os.path.join(instance_path, f'radar_map_{i}.npy')  # Adjust naming convention as needed
            try:
                view = np.load(view_path)
            except:
                print(view_path)
                #pass
            view = np.stack([view, view, view], axis=-1)  # Convert to 3-channel image
            if self.transform:
                try:
                    view = self.transform(view)
                except:
                    print(view_path)
                    #pass
            views.append(view)
        
        views = torch.stack(views)
        
        
        # Assuming labels are stored in a way that they can be loaded here
        # For simplicity, this example does not handle loading labels
        label = self.labels.loc[self.labels['folder'] == instance_name, 'postures'].values[0]  # Placeholder for actual label loading mechanism
        
        return views, label

class SingleViewNet(nn.Module):
    def __init__(self):
        super(SingleViewNet, self).__init__()
        # Load a  model

        self.densenet = timm.create_model(
        "densenet121", pretrained=False, num_classes=4)

        self.densenet = nn.Sequential(*(list(self.densenet.children())[:-1]))


    def forward(self, x):
        # Forward pass for a single view through ResNet
        x = self.densenet(x)

        # Flatten the output
        x = torch.flatten(x, 1)
        return x

class MultiViewNet(nn.Module):
    def __init__(self, num_views, num_classes):
        super(MultiViewNet, self).__init__()
        # Create a module list to hold multiple single-view ResNet networks
        self.view_networks = nn.ModuleList([SingleViewNet() for _ in range(num_views)])
        
        num_features = 1024

        # Define the layers for combining views
        self.fc_combined = nn.Linear(num_features * num_views, 1024)
        self.fc_final = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, views):
        # Process each view with its respective network
        views = [views[:,i] for i in range(views.shape[1])]
        view_features = [view_net(view) for view, view_net in zip(views, self.view_networks)]
        # Concatenate the features from all views
        combined = torch.cat(view_features, dim=1)
        # Forward pass through the combined layers
        combined = self.dropout(combined)
        combined = nn.functional.relu(self.fc_combined(combined))
        combined = self.dropout(combined)
        output = self.fc_final(combined)
        return output
    
transform = Compose([
    ToTensor(),
    Resize((224, 224)),  # Resize images to fit the model input size
])

if __name__ == '__main__':
    train_dataset = MultiViewImageDataset(root_dir='radar_train', num_views=8, transform=transform)
    #train_dataset = SingleViewImageDataset(root_dir='F:\\radar_data2\\mmpretrain\data\\radar_train_combinednpy_augmented', transform=transform)

    test_dataset = MultiViewImageDataset(root_dir='radar_test_8npy', num_views=8, transform=transform)
    #test_dataset = SingleViewImageDataset(root_dir='F:\\radar_data2\\mmpretrain\\data\\radar_test_combinednpy_augmented', transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False,num_workers=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiViewNet(num_views=8, num_classes=9).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    # Define the number of epochs
    num_epochs = 50

    best_accuracy = 0.0
    best_epoch = 0

    rank=0

    wandb.init(project="multiview-radar-project")
    wandb.config.update(optimizer, allow_val_change=True)
    wandb.watch(model)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Proc{rank}]Number of parameters:', params)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_dataloader):
            # Move images and labels to the device (e.g. GPU) if available
            images = images.to(torch.float32)
            images = images.to(device)
            labels = torch.stack(list(labels), dim=0)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            # Update the training loss
            train_loss += loss.item() * images.size(0)
            
            # Compute the predicted labels
            _, predicted = torch.max(outputs.data, 1)
            
            # Update the number of correct predictions
            train_correct += (predicted == labels).sum().item()
            
            # Update the total number of images
            train_total += labels.size(0)
        
        # Compute the average training loss and accuracy
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        
        # Print the training loss and accuracy for each epoch
        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_dataloader):
                # Move images and labels to the device (e.g. GPU) if available
                images = images.to(torch.float32)
                images = images.to(device)
                labels = torch.stack(list(labels), dim=0)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute the loss
                loss = criterion(outputs, labels)
                
                # Update the test loss
                test_loss += loss.item() * images.size(0)
                
                # Compute the predicted labels
                _, predicted = torch.max(outputs.data, 1)
                
                # Update the number of correct predictions
                test_correct += (predicted == labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
                
                # Update the total number of images
                test_total += labels.size(0)
        
        # Compute the average test loss and accuracy
        test_loss = test_loss / test_total
        test_accuracy = test_correct / test_total


        wandb.log({"train loss": train_loss, "epoch": epoch+1})
        wandb.log({"train acc": train_accuracy, "epoch": epoch+1})
        wandb.log({"val loss": test_loss, "epoch": epoch+1})
        wandb.log({"val acc": test_accuracy, "epoch": epoch+1})

        print("[Epoch: %i][Train Loss: %f][Train Acc: %f][Val Loss: %f][Val Acc: %f]" %(epoch+1, train_loss, train_accuracy, test_loss, test_accuracy))

        # Save the model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")

        # Save the best model based on the test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            true_labels = np.array(true_labels)
            predicted_labels = np.array(predicted_labels)
            model_name = "dense"
            true_label_filename = f"labels/labels_{model_name}_true_labels.npy"
            predicted_label_filename = f"labels/labels_{model_name}_predicted_labels.npy"
            np.save(true_label_filename, true_labels)
            np.save(predicted_label_filename, predicted_labels)
            torch.save(model.state_dict(), f"checkpoints/best_model_{model_name}.pt")
        
        # Print the test loss and accuracy for each epoch
        #print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    print(f"Best model achieved an accuracy of {best_accuracy:.4f} at epoch {best_epoch+1}")
    wandb.finish()