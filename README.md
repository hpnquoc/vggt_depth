
<h1>VGGT Depth</h1>

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Overview

A simplified version that removes unnecessary heads while retaining the depth estimation head for depth estimation purposes.

## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub).

```bash
git clone git@github.com:hpnquoc/vggt_depth.git 
cd vggt
pip install -r requirements.txt
```

current models:

- [VGGT-1B-Depth](https://huggingface.co/hpnquoc/VGGT-1B-Depth)

Now, try the model with just a few lines of code:

```python
import torch
from vggt.models.vggt_depth import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("hpnquoc/{model_card}").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        # Predict attributes including depth maps, depth_conf and original images.
        predictions = model(images)
```

The model weights will be automatically downloaded from Hugging Face. If you encounter issues such as slow loading, you can manually download them [here](https://huggingface.co/hpnquoc/VGGT-1B-Depth/resolve/main/model.pt) and load, or:

```python
model = VGGT()
_URL = "https://huggingface.co/hpnquoc/{model_card}/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
```

## Checklist

- [X] VGGT-1B-Depth uploaded

## License

See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.
