from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import random
from PIL import Image
# import cv2
# 1
def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
# Previous model from hug face
model_id = "CompVis/stable-diffusion-v1-4"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cpu")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")
# ====================================================
seed =1
# seed = np.random.randint(100)
# print(seed)
seed_everywhere(seed)

step = 30
off_step = 5
scale = 7.5
savepath = "Offload/seed"+str(seed)+"step"+str(step)
# ss: use the offloading or not
# tt: the offloading processing point

prompt = "A cup on the desk"
image = pipe(prompt, num_inference_steps= step, tt= off_step, ss=True, guidance_scale=scale).images[0]
image.save(savepath + prompt + "_final.png")

# prompt = "A Photo of bird on the desk"
# image = pipe(prompt, num_inference_steps= step, tt= off_step, ss=False, guidance_scale=scale).images[0]
# image.save(savepath + prompt + "_final.png")
#
# prompt = "A Photo of dog on the bed"
# image = pipe(prompt, num_inference_steps= step, tt= off_step, ss=False, guidance_scale=scale).images[0]
# image.save(savepath + prompt + "_final.png")



