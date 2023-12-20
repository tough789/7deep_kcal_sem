import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import timm
import lightning as L
import wandb
import random
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import json
import os
import collections
import string

images_path = "/home/pedro/Downloads/Data_and_Labels-20231211T105756Z-001/Data_and_Labels/Data"
labels_file = "/home/pedro/Downloads/Data_and_Labels-20231211T105756Z-001/Data_and_Labels/Labels.json"
MODEL_NAME = "resnet50"
PRETRAINED = True
BATCH_SIZE = 8
EPOCHS = 20
K_FOLDS = 2
RESOLUTION = 256
WANDB_PROJECT = 'deep-kcal'
WANDB_ENTITY = 'petrdvoracek'
WANDB_GROUP = ''.join(random.choices(string.ascii_uppercase, k=10))

my_transforms = A.Compose([
    A.RandomRotate90(p=0.5),
    #A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.Resize(height=256, width=256),
    ToTensorV2(),
])

class ProductDataset(Dataset):
    """asdasd"""
    def get_labels(self):
        labels_array = [self.extract_label_digits(filename, self.labels.get(filename, -1)) for filename in self.image_paths]
        return labels_array

    def extract_label_digits(self, filename, label):
        try:
            if label != -1:
                label_int = int(label)
                # Digits are reversed for easier work
                label_array = [int(digit) for digit in str(label_int)][::-1]
                return label_array
            else:
                print(f"Invalid label for image {filename}: {label}")
                return []
        except ValueError:
            print(f"Invalid label for image {filename}: {label}")
            return []

    def __init__(self, image_paths, labels_file, transform=None):
        self.image_paths = image_paths
        self.labels = self.load_labels(labels_file)
        self.transform = transform

    def load_labels(self, labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
      image_filename = self.image_paths[idx]

      # Skip directories
      if os.path.isdir(image_filename):
          print(f"Skipping directory: {image_filename}")
          return None, None

      image_filepath = os.path.join(images_path, image_filename)

      try:
          image = cv2.imread(image_filepath)
          if image is None:
              raise Exception(f"Error reading image at index {idx}, path: {image_filepath}")

          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          # Get label from the JSON file
          label = self.labels.get(image_filename, -1)

          # Convert label to a tensor (one-hot encoding for each digit class)
          label_tensor = torch.zeros(30)  # 30 classes - 10 units, 10 tens, 10 hundreds
          if label != -1:
              digits = [int(digit) for digit in str(label)]
              if len(digits) == 1:
                label_tensor[digits[0]] = 1
              elif len(digits) == 2:
                label_tensor[digits[1] + 10] = 1  # Tens
                label_tensor[digits[0]] = 1  # Units
              elif len(digits) == 3:
                label_tensor[digits[2] + 20] = 1  # Hundreds
                label_tensor[digits[1] + 10] = 1  # Tens
                label_tensor[digits[0]] = 1  # Units

          # Resize or crop the image to a fixed size
          image = self.transform(image=image)["image"]

          # Convert the image and label tensor to float
          image = image.float()
          label_tensor = label_tensor.float()

          label_tensor = label_tensor.reshape(3, 10)
          #print(label_tensor)
          return image, label_tensor

      except Exception as e:
          print(f"Error processing image at index {idx}: {str(e)}")
          return None, None

# Initialize Dataset
image_filenames = os.listdir(images_path)

dataset = ProductDataset(image_filenames, labels_file, my_transforms)
print(f"Dataset instance: {dataset[0][1]}")
print(f"Dataset image paths: {dataset.image_paths}")

print(f"The dataset contains {len(dataset)} samples.")

# Visualizing a random image and priting some debugging info to see if nothing's broken
random_index = random.sample(range(len(dataset)), min(1, len(dataset)))
image, label = dataset[random_index[0]]
print(f"Random Image: Shape={image.shape}, Label={label}")
image_np = np.transpose(image.cpu().numpy(), (1, 2, 0))
plt.imshow(image_np)
plt.title(f"Label: {label}")
plt.show()

# Labels histograms function
def plot_label_hist(labels, name):
    units = []
    tens = []
    hundreds = []
    all_digits = []
    digit_counts = {1: 0, 2: 0, 3: 0}  # Count of labels with 1, 2, and 3 digits

    for label in labels:
      if label != -1:
        if len(label) >= 1:
            units.append(label[0])  # units place
        if len(label) >= 2:
            tens.append(label[1])   # tens place
        if len(label) >= 3:
            hundreds.append(label[2])  # hundreds place

            # Accumulate all digits
            all_digits.extend(label)

    # INDIVIDUAL DIGITS
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.hist(units, bins=np.arange(11)-0.5, edgecolor='black')
    plt.xticks(range(10))
    plt.xlabel("Units")
    plt.ylabel("Element count")

    plt.subplot(132)
    plt.hist(tens, bins=np.arange(11)-0.5, edgecolor='black')
    plt.xticks(range(10))
    plt.xlabel("Tens")
    plt.ylabel("Element count")

    plt.subplot(133)
    plt.hist(hundreds, bins=np.arange(11)-0.5, edgecolor='black')
    plt.xticks(range(10))
    plt.xlabel("Hundreds")
    plt.ylabel("Element count")

    plt.suptitle(name)
    plt.show()

    # ALL DIGITS COMBINED
    plt.figure(figsize=(10, 5))

    plt.hist(all_digits, bins=np.arange(11)-0.5, edgecolor='black')
    plt.xticks(range(10))
    plt.xlabel("Digits")
    plt.ylabel("Element count")
    print(len(all_digits))

    plt.title(f"Combined")
    plt.show()

    # DIGIT COUNTS
    for label in dataset_labels:
      if label != -1:
        num_digits = len(label)
        if num_digits in digit_counts:
            digit_counts[num_digits] += 1

    plt.figure(figsize=(8, 5))

    plt.bar(digit_counts.keys(), digit_counts.values(), edgecolor='black')
    plt.xlabel("Number of Digits")
    plt.ylabel("Label count")
    plt.title(f"Label Count")
    plt.show()

    print(len(hundreds))
    print(len(tens))
    print(len(units))

# Printing the histograms
dataset_labels = dataset.get_labels()
print(dataset_labels)
print(len(dataset_labels))
plot_label_hist(dataset_labels, "Label Histogram")

# Neural Network Training Class
class Trainee(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.storage = collections.defaultdict(list)

    def forward(self, x):
        return self.model(x)

    def _step(self, x, y, batch_idx, stage):
        prediction = self.model(x)
        prediction = prediction.reshape(-1, 3, 10)
        print(y.shape)

        loss = self.criterion(prediction, y)

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=True)

        if batch_idx == 0 and stage == "val":  # log images only for the validation set
            self.logger.log_image(f"{stage}_images", [im for im in x[:8]])

        self.storage[f"{stage}_preds"].append(prediction)
        self.storage[f"{stage}_targets"].append(y)

        digit_accuracies = []
        for idx, (pred_digit, label_digit) in enumerate(
            zip(prediction.swapaxes(0, 1), y.swapaxes(0, 1)) # swapaxes B x D x C -> D x B x C
        ):
            good = (pred_digit.argmax(-1) == label_digit.argmax(-1)).sum().item()
            all_ = len(label_digit)
            acc = good / all_
            self.log(f"{stage}_acc_{idx}", acc)
            digit_accuracies.append(acc)
        self.log(f"{stage}_acc_mean", sum(digit_accuracies) / len(digit_accuracies))
        good_digits = 0
        for pred_digits, label_digits in zip(prediction, y):
            pred = pred_digits.argmax(-1) # D x C (onehot) -> D (int)
            lbl = label_digits.argmax(-1)
            if all(pred == lbl):
                good_digits += 1
        self.log(f"{stage}_acc_final", good_digits / len(y))

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._step(x, y, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self._step(x, y, batch_idx, "val")

    def on_validation_epoch_end(self):
        preds = torch.cat(self.storage["val_preds"]).detach().cpu()
        targets = torch.cat(self.storage["val_targets"]).detach().cpu()

        del self.storage["val_preds"], self.storage["val_targets"]

    def configure_optimizers(self):
        step_fractions = [0.8, 0.9, 0.95]

        optimizer = torch.optim.Adam(self.parameters(), 1e-3)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(EPOCHS * x) for x in step_fractions],
        )
        return [optimizer], [lr_scheduler]

# Training
wandb.finish()
wandb_logger = L.pytorch.loggers.WandbLogger(
  project=WANDB_PROJECT,
  entity=WANDB_ENTITY,
  name=MODEL_NAME,
  mode='online',
  group=WANDB_GROUP,
  log_model=True,
)

model = timm.create_model(
  MODEL_NAME,
  pretrained=PRETRAINED,
  num_classes=30,
  )

trainer = L.Trainer(logger=wandb_logger, max_epochs=EPOCHS, callbacks=[L.pytorch.callbacks.ModelCheckpoint(monitor="val_accuracy", mode="max")])
trainee = Trainee(model)
trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  
    persistent_workers=False,
    timeout=0  
)

trainer.fit(model=trainee, train_dataloaders=trainloader)