import torch
import torch.optim as optim
from tests.test_python import image
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from ultralytics import YOLO
import os
from PIL import Image

model = YOLO('yolo11n-seg.pt')

data_path = '../data/task_019_tripteroides_atripes_dataset_2024_08_13_08_49_21_coco/images/'
annotation_path = '../data/task_019_tripteroides_atripes_dataset_2024_08_13_08_49_21_coco/annotations//instances_Train.json'

batch_size = 16
learning_rate = 0.001
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            xmax, ymax = xmin + width, ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])

        if self.transform:
            img = self.transform(img)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        return img, target

    def __len__(self):
        return len(self.ids)

train_dataset = COCODataset(root=data_path, annFile=annotation_path, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch}/{num_epochs}] - Loss: {epoch_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), 'yolo11n_seg.pt')
print("Training Completed and model saved")
