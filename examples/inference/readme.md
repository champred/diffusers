# Inference Examples

## Instructions for ONNX DML Execution on Windows
### Installing the dependencies

Before running the scripts, make sure to install the library's dependencies & this updated diffusers code (Python 3.8 is recommended):

```bash
git clone https://github.com/champred/diffusers.git
cd diffusers && git checkout dml && pip install -e .
pip install transformers ftfy scipy
```

## Download onnxruntime-directml 
From [this link](https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly/PyPI/ort-nightly-directml/overview/)
, download the onnxruntime nightly directml packages. You can know the python version using `python --version` command.

- If you are on Python3.7 download the file that ends with **-cp37-cp37m-win_amd64.whl
- If you are on Python3.8 download the file that ends with **-cp38-cp38m-win_amd64.whl
- and likewise

Copy the .whl file to the working directory and install using this command.
```bash
pip install ort_nightly_directml-1.13.0.dev20220830001-cp39-cp39-win_amd64.whl
```

An error message like this `ERROR: ort_nightly_directml-1.13.0.dev20220830001-cp38-cp38-win_amd64.whl is not a supported wheel on this platform.` means that there is mismatch in python version and the downloaded package supported python version.

### Optional: Create Diffusers Model from CKPT
If you have a `model.ckpt` and `config.yaml` you can save it in the Diffusers format with [this conversion script](/scripts/convert_original_stable_diffusion_to_diffusers.py).

```bash
cd scripts
mkdir converted
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path /path/to/model.ckpt --original_config_file /path/to/config.yaml --scheduler_type euler --dump_path converted
```

After the model has been exported, update line 15 in `save_onnx.py` to point to the exported path.

## Create ONNX files
This step requires a [Hugging Face token](https://huggingface.co/settings/tokens) if you are not importing a local model. All ONNX files are created in a folder named onnx in `examples/inference`.

```bash
huggingface-cli login
cd examples/inference/
python save_onnx.py 
```

## Run using ONNX files
Run the onnx model using DirectML Execution Provider. Please check the last few lines in `dml_onnx.py` to see the examples.

```bash
python dml_onnx.py 
```


### Note: Please raise any issues on this repo, I'll take a look. thanks!


----------------------------------------------------------------------------------------




## Installing the dependencies

Before running the scripts, make sure to install the library's dependencies:

```bash
pip install diffusers transformers ftfy
```

## Image-to-Image text-guided generation with Stable Diffusion

The `image_to_image.py` script implements `StableDiffusionImg2ImgPipeline`. It lets you pass a text prompt and an initial image to condition the generation of new images. This example also showcases how you can write custom diffusion pipelines using `diffusers`!

### How to use it


```python
import torch
from torch import autocast
import requests
from PIL import Image
from io import BytesIO

from image_to_image import StableDiffusionImg2ImgPipeline, preprocess

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

# let's download an initial image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))
init_image = preprocess(init_image)

prompt = "A fantasy landscape, trending on artstation"

with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5)["sample"]

images[0].save("fantasy_landscape.png")
```
You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/image_2_image_using_diffusers.ipynb)

## Tweak prompts reusing seeds and latents

You can generate your own latents to reproduce results, or tweak your prompt on a specific result you liked. [This notebook](stable-diffusion-seeds.ipynb) shows how to do it step by step. You can also run it in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb).


## In-painting using Stable Diffusion

The `inpainting.py` script implements `StableDiffusionInpaintingPipeline`. This script lets you edit specific parts of an image by providing a mask and text prompt.

### How to use it

```python
import torch
from io import BytesIO

from torch import autocast
import requests
import PIL

from inpainting import StableDiffusionInpaintingPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

device = "cuda"
pipe = StableDiffusionInpaintingPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

prompt = "a cat sitting on a bench"
with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75)["sample"]

images[0].save("cat_on_bench.png")
```

You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/in_painting_with_stable_diffusion_using_diffusers.ipynb)
