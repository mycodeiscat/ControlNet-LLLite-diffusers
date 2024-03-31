# ControlNet-LLLite hack for diffusers
This is a quick hack for [ControlNet-LLLite](https://github.com/kohya-ss/sd-scripts/blob/sdxl/docs/train_lllite_README.md) to work with diffusers.

# Requirements
diffusers>=0.27.2 (Due to difference in forward calls and PEFT integration)

## Usage
```python
from PIL import Image
import numpy as np
from controlnet_lite import ControlNetLLLite

# Load ControlNetLllite weights 
path = 'kohya_controllllite_xl_canny.safetensors'
controlnet = ControlNetLLLite(path)


# Load Control Image
cond_image = diffusers.utils.load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")
# Convert the image to numpy array
cond_image = np.array(image)

conditioning_weight = 1

# Apply ControlNetLLLite to the pipeline
controlnet.apply(pipe=pipeline, cond=control_image, weight=conditioning_weight)
```
## Limitations
Currently there is no way to control start and end of the conditioning
As it can only be controled from the diffusers.DiffusionPipeline, and will require separate custom pipeline for Txt2Img, Img2Img etc.
TODO: figure out smart way to bypass this

Also looking into smarter way to unload models.
## Acknowledgements
Thanks to kohya-ss, the original author of ControlNetLLLite, for supporting the development.
