import argparse
from os import replace
import os
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp, LayerCAM
import torch
from torch.utils.data.dataloader import DataLoader
from domainbed.algorithms.algorithms import ERM, Contrast
from clip.model import CLIP
from re import split
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description="Domain generalization")
parser.add_argument("--pth_dir", type=str, help="please add train_output pathdir to here, like 'train_output/OfficeHome/...'")
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()

output = args.data_dir.replace('/data', './data')
os.makedirs(output, exist_ok=True)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# pth = "train_output/OfficeHome/221107_12-56-58_OH_Contrast_RN50_1e-06/te_Clipart-swad.pth"
pth = args.pth_dir
model = torch.load(pth).eval()

if isinstance(model, ERM):
    net = model.featurizer
elif isinstance(model, Contrast):
    net = model.network
if isinstance(net.network, CLIP):
    model.float()
    net = net.network.visual.float()
else:
    net = net

transform = transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                ])

data = datasets.ImageFolder(root=args.data_dir, transform=transform)
class_names = data.classes
dataloader = DataLoader(data, batch_size=1)

cam_extractor = LayerCAM(model, target_layer=net.layer4)
for i, (data, label) in enumerate(dataloader):
    os.makedirs(output + class_names[label] + '/', exist_ok=True)
    data = data.to(device)
    # Preprocess your data and feed it to the model
    out = model.predict(data).float()
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)[0].float()

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(data.squeeze(0)), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.savefig(output + class_names[label] + '/' + '%d.png'%(i+1))  