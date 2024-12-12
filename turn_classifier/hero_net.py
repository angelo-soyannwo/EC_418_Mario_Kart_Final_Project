
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class HeroNet(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=768, image_model_output=512):
        """
        HeroNet combines text and image features for a multi-modal superhero network.
        Args:
            num_classes (int): Number of output classes (binary or multi-class).
            hidden_dim (int): Dimension of text embeddings.
            image_model_output (int): Output size of the image feature extractor.
        """
        super(HeroNet, self).__init__()
        
        # Text processing with BERT
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(hidden_dim, hidden_dim // 2)

        # Image processing with CNN
        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, image_model_output, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        )

        # Fusion layer
        self.fusion_fc = nn.Linear(hidden_dim // 2 + image_model_output, 256)

        # Classification head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, text_input, attention_mask, image_input):
        """
        Forward pass for HeroNet.
        Args:
            text_input (torch.Tensor): Tokenized text input (IDs).
            attention_mask (torch.Tensor): Attention mask for text input.
            image_input (torch.Tensor): Image tensor of shape (batch_size, 3, H, W).
        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        # Process text with BERT
        text_features = self.bert(input_ids=text_input, attention_mask=attention_mask)["pooler_output"]
        text_features = self.text_fc(text_features)

        # Process image with CNN
        image_features = self.image_cnn(image_input).squeeze(-1).squeeze(-1)

        # Concatenate text and image features
        combined_features = torch.cat((text_features, image_features), dim=1)

        # Fusion and classification
        fused_features = self.fusion_fc(combined_features)
        output = self.classifier(fused_features)
        return output





def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, HeroNet):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'hero_net.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = HeroNet()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'hero_net.th'), map_location='cpu'))
    return r