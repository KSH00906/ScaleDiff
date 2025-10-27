from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline 
import torch

ckpt_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = CustomStableDiffusionXLPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16).to("cuda")
pipe.vae.enable_tiling()

prompt = "a woman"
negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

generator = torch.Generator(device='cuda').manual_seed(77)

images = pipe(prompt, 
            negative_prompt=negative_prompt,
            height = 1024, 
            width = 1024,
            generator=generator,
            num_inference_steps=50, 
            guidance_scale=7.5,
            restart_ratio = 0.4,
            scale_factor = 0.125,
            upsample_stage = 2,
            )

for i, image in enumerate(images):
    image.save(f"{prompt}_{i}.png")