from daam import trace, set_seed
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch
from torchinfo import summary

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'mps'

model = DiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
model = model.to(device)
print(model.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q.weight.shape)

prompt = 'A dog and a cat'
gen = set_seed(0)  # for reproducibility

with torch.no_grad():
     with trace(model) as tc:
         out = model(prompt, num_inference_steps=50, generator=gen)
         heat_map = tc.compute_global_heat_map()
         heat_map = heat_map.compute_word_heat_map('cat')
         heat_map.plot_overlay(out.images[0])
         plt.show()