from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import random

model_id = "CompVis/stable-diffusion-v1-4"
print(f"Loading model... from{model_id}")

# pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float).to("cuda")
prompt = "A photo of an apple."
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("output.png")
