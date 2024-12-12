
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import dense_transforms
import torch

DATASET_PATH = 'road_data'


class RoadDataset(Dataset):
	def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
		from PIL import Image
		from glob import glob
		from os import path
		self.data = []
		for f in glob(path.join(dataset_path, '*.png')):
			i = Image.open(f)
			i.load()
			#f.split('/')[1][0] is a 0, 1, or 2 denoting sharp left, approximately straight, and sharp right respectively
			self.data.append((i, int(f.split('/')[1][0]) ))

		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		#old code
		#data = self.data[idx]
		#data = self.transform(*data)
		#return data

		img, label = self.data[idx]
		
		if self.transform:

			img = self.transform(img)  # Apply transformation only to the image
		else:
			from torchvision.transforms import ToTensor
			img = ToTensor()(img)  # Default conversion to tensor
		
		label = torch.tensor(label, dtype=torch.long)  # Convert label to a tensor
		return img[0], label  # Return as (image, label)

		"""
		img_tensor = self.transform(self.data[idx][0])
		tensor_label = torch.tensor(self.data[idx][1], dtype=torch.long)
		print(tensor_label)
		print(img_tensor)
		return img_tensor, tensor_label #img, label
		"""

#change batchsize back to 128
def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=32):
	dataset = RoadDataset(dataset_path, transform=transform)
	return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

