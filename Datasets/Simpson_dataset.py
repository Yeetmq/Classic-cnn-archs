import os

import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
import torch
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class SimpsonDataset(torch.utils.data.Dataset):
  def __init__(self, path_to_dataset: str, mode, transforms, device) -> None:
    super().__init__()
    self.path_to_dataset = path_to_dataset
    self.mode = mode
    self.transforms = transforms
    self.class_dict = self.get_class_dict()
    self.paths_to_img_label_list = self.get_img_list()
    self.RESCALE_SIZE = 224
    self.device = device

  def __len__(self):
    return len(self.paths_to_img_label_list)


  def __getitem__(self, idx):

    if isinstance(idx, tuple):
      idx = idx[1]
    path_to_image, label = self.paths_to_img_label_list[idx]

    image = Image.open(path_to_image).convert("RGB")
    image = tv_tensors.Image(image, device=self.device)

    transforms_dict = {'image':image, 'label':label}
    transformed = self.transforms(transforms_dict)

    return transformed['image'], transformed['label']

  def get_class_dict(self) -> dict:
    path_to_train_data = os.path.join(self.path_to_dataset, 'train', 'simpsons_dataset')

    class_dict = {name: idx for idx, name in enumerate(os.listdir(path_to_train_data))}
    return class_dict

  def get_img_list(self):
    paths_to_img_label_list = []

    if self.mode == "TRAIN":
      path_to_train_data = os.path.join(self.path_to_dataset, 'train', 'simpsons_dataset')

      for folder_name in os.listdir(path_to_train_data):
        path_to_images = os.path.join(path_to_train_data, folder_name)
        if os.path.isdir(path_to_images):
          class_idx = self.class_dict[folder_name]

          for img_name in os.listdir(path_to_images):

            if img_name.endswith("jpg"):
              path_to_image = os.path.join(path_to_images, img_name)
              paths_to_img_label_list.append((path_to_image, class_idx))
    return paths_to_img_label_list


if __name__ == '__main__':
    device = torch.device('cuda:0')

    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    full_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])


    full_dataset = SimpsonDataset(
        path_to_dataset=r'D:\ethd\ml\Classic_archs\data',
        mode='TRAIN',
        transforms=train_transforms,
        device=device)

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    valid_dataset.dataset.transform = val_transforms

    dataset_size = len(train_dataset)

    idx = np.random.randint(dataset_size)

    img, label = train_dataset[idx]

    image = img.cpu()

    print('Index = {}; Label = {}'.format(idx, label))

    image_np = image.permute(1, 2, 0).numpy()

    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

  