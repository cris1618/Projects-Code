import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import Adam
from torch import nn
from torcheval.metrics import MulticlassAUPRC
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes

# Dataset class
class PlanktonDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform

        # Group the annotations by image
        self.image_annotations = self.annotations.groupby('file')

    def __len__(self):
        return len(self.image_annotations)

    def __getitem__(self, idx):
        img_name = list(self.image_annotations.groups.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Get all annotations for this image
        annotations = self.image_annotations.get_group(img_name)

        boxes = []
        labels = []
        for _, row in annotations.iterrows():
            box_str = row['box'].replace(' ', ',').replace(',,', ',').strip('[]')
            box = [int(b) for b in box_str.split(',') if b.isdigit()]
            boxes.append(box)
            labels.append(int(row['label']) + 1)

        if self.transform:
            image = self.transform(image)

        # Convert to torch tensors
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        return image, target

# Manually define FastRCNNPredictor
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_images_with_boxes(images, targets, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
    for i in range(num_images):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        ax = axes[i]
        ax.imshow(image)

        for box, label in zip(targets[i]['boxes'], targets[i]['labels']):
            box = box.cpu().numpy()
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, str(label.item()), color='yellow', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        ax.axis('off')
    plt.show()

def visualize_predictions(model, dataloader, device, num_images=5, confidence_threshold=0.5):
    model.eval()
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
    transform = transforms.ToPILImage()

    for idx, (images, targets) in enumerate(dataloader):
        if idx >= num_images:
            break

        image = images[0].to(device)
        with torch.no_grad():
            prediction = model([image])

        image = image.cpu()
        image = transform(image)
        
        ax = axes[idx]
        ax.imshow(image)
        
        for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
            if score >= confidence_threshold:
                box = box.cpu().numpy()
                x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
                rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y, str(label.item()), color='yellow', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    #Model Checkpoint 
    save_dir = 'model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    folder_path = "C:/Users/Utente/OneDrive/Desktop/Plankton Research/CROPPED-20240527T151546Z-001/CROPPED"
    csv_file_path = "C:/Users/Utente/OneDrive/Desktop/Plankton Research/solace_crop.csv"

    # Load the annotations
    annotations = pd.read_csv(csv_file_path)
    train_img_from_id = [46, 33, 49, 7, 3, 2, 1, 58, 44, 6, 54, 59, 39, 4, 10, 32, 16, 14, 11, 19, 64, 36, 17, 63, 23, 66, 43, 5, 65, 42, 26, 0, 69, 21, 27, 56, 12, 50, 37, 55, 61, 52, 48, 24, 8, 62, 47, 45, 53, 20, 67, 30, 40, 25, 31, 15, 18, 57, 41, 34, 68]
    test_img_from_id = [70, 51, 22, 9, 71, 60, 35]

    # Filter the DataFrame
    train_annotations = annotations[annotations['img_from_id'].isin(train_img_from_id)]
    test_annotations = annotations[annotations['img_from_id'].isin(test_img_from_id)]

    # Sample a fraction of the dataset (0.3%)
    train_sample = train_annotations.sample(frac=0.02, random_state=42)
    test_sample = test_annotations.sample(frac=0.02, random_state=42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Try to normalize images
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #It greatly affects image's pixel, DON'T USE IT!
    ])

    # Create the datasets
    train_dataset = PlanktonDataset(annotations=train_annotations, img_dir=folder_path, transform=transform)
    test_dataset = PlanktonDataset(annotations=test_sample, img_dir=folder_path, transform=transform)

    #View a sample of an image
    sample = train_dataset[37]
    img_int = torch.tensor(sample[0] * 255).byte().clone().detach()
    plt.imshow(draw_bounding_boxes(img_int, sample[1]['boxes'], [str(label.item()) for label in sample[1]['labels']], width=4).permute(1,2,0))
    plt.show()

    # Create the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, #Decreased for testing purpose on my laptop
        shuffle=True,
        num_workers=0, #Decreased for testing purpose on my laptop
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1, #Decreased for testing purpose on my laptop
        shuffle=False,
        num_workers=0, #Decreased for testing purpose on my laptop
        collate_fn=collate_fn
    )

    # VISUALIZE IMAGES INPUTTED IN THE MODEL BEFORE TRAINING
    images, targets = next(iter(train_loader))
    #visualize_images_with_boxes(images, targets)

    # Load the pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights='COCO_V1')
    #model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 19  
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to GPU if possible
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Load model weights if available
    checkpoint_path = 'model_checkpoints/fasterrcnn_checkpoint_epoch_1.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training loop with progress tracking
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            # Use of List Comprehensions to create list of images and targets so that you don't need to use
            # the .squeeze(0) function the batch dimension is mantained and so now the model can handle batch sizes greater than 1
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images, targets)
            loss_dict = outputs
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{len(train_loader)}, Loss: {losses.item()}")
        
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'fasterrcnn_checkpoint_epoch_{epoch + 1}.pth'))
        print(f"Saved checkpoint for epoch {epoch + 1}")

    print("Training Completed!")

    # VISUALIZE PREDICTIONS OF THE MODEL
    visualize_predictions(model, test_loader, device, confidence_threshold=0.5)


    """def get_valid_predictions(outputs, num_classes, confidence_threshold=0.01):
        valid_preds_scores = []
        valid_preds_labels = []

        for output in outputs:
            scores = output['scores']
            labels = output['labels']
            keep = scores >= confidence_threshold
            if keep.sum() > 0:
                valid_preds_scores.append(F.one_hot(labels[keep], num_classes).float() * scores[keep].unsqueeze(-1))
                valid_preds_labels.append(labels[keep])

        return valid_preds_scores, valid_preds_labels

    # Evaluation
    num_classes = 19  
    metric_ap = MulticlassAUPRC(num_classes=num_classes, average=None, device=device)

    model.eval()
    confidence_threshold = 0 # Set a lower threshold for evaluation
    with torch.no_grad():
        all_preds_scores = []
        all_preds_labels = []
        all_true_labels = []

        for images, targets in test_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            # Collect valid predictions and their corresponding true labels
            valid_preds_scores, valid_preds_labels = get_valid_predictions(outputs, num_classes, confidence_threshold)

            # Debug prints
            print(f"Valid predictions: {valid_preds_scores}")
            print(f"Valid prediction labels: {valid_preds_labels}")

            for target in targets:
                all_true_labels.extend(target['labels'])

            # Flatten the lists of tensors
            if valid_preds_scores and valid_preds_labels:
                all_preds_scores.extend(valid_preds_scores)
                all_preds_labels.extend(valid_preds_labels)

        # Check if we have valid predictions
        if all_preds_scores and all_preds_labels:
            preds_scores = torch.cat(all_preds_scores).to(device)
            preds_labels = torch.cat(all_preds_labels).to(device)
            true_labels = torch.tensor(all_true_labels).to(device)

            # Debug prints
            print(f"Predicted scores shape: {preds_scores.shape}")
            print(f"Predicted labels shape: {preds_labels.shape}")
            print(f"True labels shape: {true_labels.shape}")

            # Ensure matching shapes for predictions and labels
            if preds_scores.shape[0] == preds_labels.shape[0]:
                metric_ap.update(preds_scores, preds_labels)
            else:
                print(f"Shape mismatch: preds_scores shape {preds_scores.shape}, preds_labels shape {preds_labels.shape}, true_labels shape {true_labels.shape}")

        else:
            print("No valid predictions to compute Average Precision (AP).")

    # Compute and print AP per class
    if len(metric_ap.inputs) > 0:  # Check if there are any inputs before computing
        ap_per_class = metric_ap.compute()
        print(f"Average Precision (AP) per class: {ap_per_class}")

        # Print average precision for each class
        for i, ap in enumerate(ap_per_class):
            print(f"Average Precision for class {i + 1}: {ap.item()}")
    else:
        print("No valid predictions to compute Average Precision (AP).")

        # Concatenate scores and labels
        preds_scores = torch.cat([output['scores'].to(device) for output in outputs])
        true_labels = torch.cat([target['labels'].to(device) for target in targets])

        #preds_scores = [output['scores'].to(device) for output in outputs]
        #true_labels = [target['labels'].to(device) for target in targets]

        #if len(preds_scores) > 0 and len(true_labels) > 0:
        metric_ap.update(preds_scores, true_labels)
        #metric_map.update(preds_scores, true_labels)
        
        ap_per_class = metric_ap.compute()
        print(f"Average Precision (AP) per class: {ap_per_class}")
        #else:
        #print("No valid predictions or labels found for evaluation.")"""

    
    # MEAN AVERAGE PRECISION PER CLASS
    # mAP = metric_map.compute()
    # print(f"The mean Average Precision (mAP) of the model is {mAP}")

     # Print average precision for each class
    #for i, ap in enumerate(mAP['map_per_class']):
       # print(f"Average Precision for class {i + 1}: {ap.item()}")"""