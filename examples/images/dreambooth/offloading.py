from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import random
from PIL import Image
# import cv2

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
# Previous model from hug face
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")

# ====================================================
seed = 47
seed_everywhere(47)

step = 20
off_step = 10
savepath = "Offload/seed"+str(seed)+"step"+str(step)
# ss: use the offloading or not
# tt: the offloading processing point

prompt = "A photo of dog on the desk"
image = pipe(prompt, num_inference_steps= step, tt= off_step, ss=True, guidance_scale=3.5).images[0]
image.save(savepath+"final_dog.png")

prompt = "A photo of cat on the desk"
image = pipe(prompt, num_inference_steps= step, tt= off_step, ss=False, guidance_scale=3.5).images[0]
image.save(savepath+"final_cat.png")

prompt = "A photo of mouse on the desk"
image = pipe(prompt, num_inference_steps= step, tt= off_step, ss=False, guidance_scale=3.5).images[0]
image.save(savepath+"final_cat.png")

