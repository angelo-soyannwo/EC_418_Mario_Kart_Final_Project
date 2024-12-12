from gray_scale_road_classifier import GrayScaleRoadClassifier, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data
import dense_transforms
import inspect
from torchvision.transforms import Compose, Grayscale, ColorJitter, RandomHorizontalFlip, ToTensor

def train_gray_scale_road_classifier(args):
    from os import path
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print("Using device:", device)

    model = GrayScaleRoadClassifier().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'gray_scale_road_classifier.th')))

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_data('road_data', transform=transform, num_workers=0)

    global_step = 0
    epoch_losses = []  # To store the loss at each epoch

    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            pred = model(img)
            loss_val = loss(pred, label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1
            losses.append(loss_val.detach().cpu().numpy())

        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)
        print(f'epoch {epoch+1}/{args.num_epoch} - loss: {avg_loss:.4f}')

    save_model(model)

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epoch + 1), epoch_losses, marker='o', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig("training_loss.png")  # Save the plot as an image
    plt.show()  # Display the plot
    print(epoch_losses)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=30)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([Grayscale(num_output_channels=1), ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train_gray_scale_road_classifier(args)
