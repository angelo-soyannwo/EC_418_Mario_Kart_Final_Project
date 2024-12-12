import torch
import torch.nn.functional as F
import road_classifier as rc
import gray_scale_road_classifier as grc

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):

      super().__init__()

      layers = []
      layers.append(torch.nn.Conv2d(3,16,5,2,2))
      layers.append(torch.nn.BatchNorm2d(16))
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.Conv2d(16,1,1,1))

      layers_version_2 = [
            # First convolutional layer
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # Input: (B, 3, 96, 128)
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            # Second convolutional layer
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Reduces spatial dimensions further
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            # Third convolutional layer
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Extracts deeper features
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            # Final output layer
            torch.nn.Conv2d(64, 1, kernel_size=1, stride=1),  # Single-channel heatmap output
        ]


      layers_version_3 = [
          # First convolutional block
        torch.nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # Keep stride=1 to preserve spatial dimensions
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsamples by 2x

        # Second convolutional block
        torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsamples by 2x

        # Third convolutional block
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsamples by 2x

        # Fourth convolutional block
        torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsamples by 2x

        # Final output layer
        torch.nn.Conv2d(128, 1, kernel_size=1, stride=1),  # Single-channel output
      ]


      #self._conv = torch.nn.Sequential(*layers)
      #self._conv = torch.nn.Sequential(*layers_version_2)
      self._conv = torch.nn.Sequential(*layers_version_3)


    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        #print(img.shape)
        #print(x.shape)
        return spatial_argmax(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        #road_classifier = rc.load_model().eval()
        gray_scale_classifier = grc.load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose, gray_scale_road_classifier=gray_scale_classifier)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
