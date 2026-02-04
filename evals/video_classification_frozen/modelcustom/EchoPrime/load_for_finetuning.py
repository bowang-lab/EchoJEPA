import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import transformers
from echo_prime import EchoPrimeTextEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## load the echo encoder
checkpoint = torch.load("model_data/weights/echo_prime_encoder.pt",map_location=device)
echo_encoder = torchvision.models.video.mvit_v2_s()
echo_encoder.head[-1] = torch.nn.Linear(echo_encoder.head[-1].in_features, 512)
echo_encoder.load_state_dict(checkpoint)
echo_encoder.eval()
echo_encoder.to(device)
for param in echo_encoder.parameters():
    param.requires_grad = False

print(f"Echo embedding shape is {echo_encoder(torch.zeros(1,3,16,224,224).to(device)).shape}")

## load the text encoder
text_encoder=EchoPrimeTextEncoder(device=device)
text_encoder.load_state_dict(torch.load("model_data/weights/echo_prime_text_encoder.pt"))
text_encoder.eval()

# produces 512 dimensional embedding
print(f"Text embedding shape is {text_encoder('Sample text').shape}")