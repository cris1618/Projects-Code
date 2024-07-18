import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
import torch.optim as optim
import matplotlib.pyplot as plt
from torcheval.metrics import MulticlassAUPRC
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes

# Define the folder path containing the images
folder_path = "C:/Users/Utente/OneDrive/Desktop/Plankton Research/CROPPED-20240527T151546Z-001/CROPPED"
csv_file_path = "C:/Users/Utente/OneDrive/Desktop/Plankton Research/solace_crop.csv"

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
            labels.append(int(row['label']) + 1)  # Ensure labels start from 1

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
        
        for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
            if score >= confidence_threshold:
                box = box.cpu().numpy()
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    annotations = pd.read_csv(csv_file_path)
    train_img_from_id = [46, 33, 49, 7, 3, 2, 1, 58, 44, 6, 54, 59, 39, 4, 10, 32, 16, 14, 11, 19, 64, 36, 17, 63, 23, 66, 43, 5, 65, 42, 26, 0, 69, 21, 27, 56, 12, 50, 37, 55, 61, 52, 48, 24, 8, 62, 47, 45, 53, 20, 67, 30, 40, 25, 31, 15, 18, 57, 41, 34, 68]
    test_img_from_id = [70, 51, 22, 9, 71, 60, 35]

    # Filter the DataFrame
    train_annotations = annotations[annotations['img_from_id'].isin(train_img_from_id)]
    test_annotations = annotations[annotations['img_from_id'].isin(test_img_from_id)]

    # Sample a fraction of the dataset (3%)
    train_sample = train_annotations.sample(frac=0.1, random_state=42)
    test_sample = test_annotations.sample(frac=0.1, random_state=42)

    transform = transforms.Compose([
        transforms.ToTensor(),      
    ])

    # Create the datasets
    train_dataset = PlanktonDataset(annotations=train_sample, img_dir=folder_path, transform=transform)
    test_dataset = PlanktonDataset(annotations=test_annotations, img_dir=folder_path, transform=transform)

    #View a sample of an image
    sample = train_dataset[12]
    img_int = torch.tensor(sample[0] * 255).byte().clone().detach()
    plt.imshow(draw_bounding_boxes(img_int, sample[1]['boxes'], [str(label.item()) for label in sample[1]['labels']], width=4).permute(1,2,0))
    plt.show()

    """# Create the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, #Reduce for testing purpose
        shuffle=True,
        num_workers=0, #Reduce for testing purpose
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=2,
         collate_fn=collate_fn
    )

    # Fetch a batch to check
    images, targets = next(iter(train_loader))
    #visualize_images_with_boxes(images, targets)
    #print(image.shape)
    #print(target)

    # Define a simple backbone
    backbone = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    backbone.out_channels = 32

    # Define the anchor generator and ROI pooler
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),), 
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2
    )

    # Define the model
    num_classes = 19  
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    # Move model to GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with progress tracking
    num_epochs = 1  
    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0

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
        torch.save(model.state_dict(), f'fasterrcnn_checkpoint_epoch_{epoch + 1}.pth')

    print("Training Completed!")

    visualize_predictions(model, test_loader, device, confidence_threshold=0.5)
    # Evaluation
    metric_ap = MulticlassAUPRC(num_classes=num_classes, average=None, device=device)
    # metric_map = MeanAveragePrecision()

    model.eval()
    with torch.no_grad():

        for images, targets in test_loader:
            # Same thing that i did in the training loop
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            preds_scores = torch.cat([output['scores'].to(device) for output in outputs])
            true_labels = torch.cat([target['labels'].to(device) for target in targets])

            metric_ap.update(preds_scores, true_labels)
            #metric_map.update(preds_scores, true_labels)

    ap_per_class = metric_ap.compute()
    print(f"Average Precision (AP) per class: {ap_per_class}")"""

    # MEAN AVERAGE PRECISION PER CLASS
    # mAP = metric_map.compute()
    # print(f"The mean Average Precision (mAP) of the model is {mAP}")

     # Print average precision for each class
    #for i, ap in enumerate(mAP['map_per_class']):
       # print(f"Average Precision for class {i + 1}: {ap.item()}")"""