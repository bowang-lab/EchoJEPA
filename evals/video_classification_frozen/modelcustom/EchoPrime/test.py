from echo_prime import EchoPrime
import torch
ep = EchoPrime()
x = ep.predict_metrics(ep.encode_study(torch.zeros((50, 3, 16, 224, 224))))
print(x)