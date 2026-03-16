import torch
sd = torch.load('vit_b_16-c867db91.pth', map_location='cpu', weights_only=True)
for k in sorted(sd.keys()):
    if 'layer_0' in k:
        print(k, sd[k].shape)
