import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
import torch
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN

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

# Function to visualize the predictions
def visualize_predictions(model, dataloader, device, num_images=5, confidence_threshold=0.5):
    model.eval()
    fig, ax = plt.subplots(1, num_images, figsize=(20, 5))
    transform = transforms.ToPILImage()

    for i, (image, target) in enumerate(dataloader):
        if i >= num_images:
            break

        image = image.squeeze(0).to(device)
        with torch.no_grad():
            prediction = model([image])
        
        image = image.cpu()
        image = transform(image)

        ax[i].imshow(image)
        for box in prediction[0]['boxes']:
            box = box.cpu()
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax[i].add_patch(rect)
        ax[i].axis('off')
    plt.show()

if __name__ == '__main__':
    # Define the folder path containing the images
    folder_path = "C:/Users/Utente/OneDrive/Desktop/Plankton Research/CROPPED-20240527T151546Z-001/CROPPED"
    csv_file_path = "C:/Users/Utente/OneDrive/Desktop/Plankton Research/solace_crop.csv"

    # Load the annotations
    annotations = pd.read_csv(csv_file_path)
    train_img_from_id = [46, 33, 49, 7, 3, 2, 1, 58, 44, 6, 54, 59, 39, 4, 10, 32, 16, 14, 11, 19, 64, 36, 17, 63, 23, 66, 43, 5, 65, 42, 26, 0, 69, 21, 27, 56, 12, 50, 37, 55, 61, 52, 48, 24, 8, 62, 47, 45, 53, 20, 67, 30, 40, 25, 31, 15, 18, 57, 41, 34, 68]
    test_img_from_id = [70, 51, 22, 9, 71, 60, 35]

    # Filter the DataFrame
    test_annotations = annotations[annotations['img_from_id'].isin(test_img_from_id)]

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create the dataset
    test_dataset = PlanktonDataset(annotations=test_annotations, img_dir=folder_path, transform=transform)

    # Create the DataLoader for the test set
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # Define the model and load the weights
    num_classes = 19  # Including the background class
    backbone = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    backbone.out_channels = 32
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load('model_chekpoints/fasterrcnn_checkpoint_epoch_1.pth', map_location=device))
    model.to(device)

    # Visualize the predictions
    visualize_predictions(model, test_loader, device, confidence_threshold=0.5)


