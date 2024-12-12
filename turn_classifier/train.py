from road_classifier import RoadClassifier, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from utils import load_data
import dense_transforms
import inspect

def train_road_classifier(args):
	from os import path
	# Set threading before any computation
	torch.set_num_threads(4)  # Adjust based on your hardware
	torch.set_num_interop_threads(2)

	# Device setup
	device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
	print("Using device:", device)

	model = RoadClassifier().to(device) # move the model to the training device

	if args.continue_training:
		model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'road_classifier.th')))

	loss = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

	transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

	train_data = load_data('road_data', transform=transform, num_workers=0)


	global_step = 0
	for epoch in range(args.num_epoch):
		model.train()
		losses = []
		for img, label in train_data:
			img, label = img.to(device), label.to(device) #move the img and label to the training device
	
			pred = model(img)
			loss_val = loss(pred, label)

			optimizer.zero_grad()
			loss_val.backward()

			global_step += 1
			losses.append(loss_val.detach().cpu().numpy())

		avg_loss = np.mean(losses)
		print(f'epoch {epoch+1}/{args.num_epoch} - loss: {avg_loss:.4f}')

	save_model(model)
	#print(args)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=30)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train_road_classifier(args)


