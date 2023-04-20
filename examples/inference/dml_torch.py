import os, time, torch, torch_directml
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
model="runwayml/stable-diffusion-v1-5"
# only use float32 if you have a lot of VRAM
pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, scheduler=scheduler)
device = torch_directml.device() if torch_directml.is_available() else "cpu"
pipe.to(device)
torch.manual_seed(time.time_ns())

prompt = input("Prompt: ")
# TODO not exhaustive, make sure your prompts don't have any other illegal characters
path = prompt.replace("/", " ")[:189]
os.makedirs(path, exist_ok=True)
for i in range(int(input("Number: "))):
    print(f"Generating #{i+1}")
    try:
        image = pipe(prompt, height=512, width=512, num_inference_steps=30, guidance_scale=7.5, eta=0.0).images[0]
        image.save(f"{path}/{time.time_ns()}.png")
    except RuntimeError:
        print("Failed to generate image")

