from xml.etree.ElementTree import PI
import torch
torch.manual_seed(1234)
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
import cv2

# this doesn't works

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("images/pug.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# print("type", type(text))

c = 10
mse = nn.MSELoss(reduction="sum")
# messup_target_logit = torch.tensor([[0, 0, 1000]], dtype=torch.float16).to(device)
messup_target_logit = torch.tensor([[1000, 0, 0]], dtype=torch.float16).to(device)
messup_target_probs = messup_target_logit.softmax(dim=-1)
print("messup_target_probs={}".format(messup_target_probs))

# with torch.no_grad():


logits_per_image, logits_per_text = model(image, text)
r = (torch.randn(image.size(), dtype=image.dtype, device=device) * 0.3).requires_grad_()
original_r = torch.clone(r.detach())
print("r.requires_grad={}".format(r.requires_grad))
# print("image={} {}".format(image.size(), image.dtype))
optimizer = torch.optim.Adam([r])
epochs = 100
for i in range(epochs):
    optimizer.zero_grad()
    messup_logits_per_image, _ = model( r, text)
    messup_probs_per_image = messup_logits_per_image.softmax(dim=-1)
    loss =  mse(messup_target_probs, messup_probs_per_image)#  + c * torch.mean(r*r)#* torch.linalg.norm(r)
    loss.backward()
    print("epoch={}, loss={}".format(i, loss.item()))
    optimizer.step()
probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()

def save_to(img, filename):
    # img: nchw
    image = torch.clone(img[0, :, :, :])
    unNormalize = UnNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image = unNormalize(image)
    image = np.transpose(image.cpu().numpy(), (1, 2, 0)) * 255
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# print(original_r)
# print(r)
save_to(image, "original.jpg")
# save_to(image + original_r.detach(), "mess_debug.jpg")
save_to(r.detach(), "mess.jpg")


print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
print("Mess up probs:", messup_probs_per_image.cpu().detach().numpy())