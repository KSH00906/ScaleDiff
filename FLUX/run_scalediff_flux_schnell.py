import torch
from pipeline_scalediff_flux import FluxPipeline
from transformer_scalediff_flux import FluxTransformer2DModel

ckpt  = "black-forest-labs/FLUX.1-schnell"
transformer = FluxTransformer2DModel.from_pretrained(ckpt, torch_dtype=torch.bfloat16, subfolder='transformer')
pipe = FluxPipeline.from_pretrained(ckpt, torch_dtype=torch.bfloat16, transformer = transformer).to('cuda')
pipe.vae.enable_tiling()

prompt = "a woman"

generator = torch.Generator(device="cuda").manual_seed(77)

images = pipe(
    prompt + ", highly detailed, 4k resolution, best quality",
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=generator,
    restart_ratio = 0.5,
    scale_factor = 0.25,
    upsample_stage = 2,
    query_random_jitter = True,
    t5_to_cpu = True,
)

for i, image in enumerate(images):
    image.save(f"{prompt}_{i}.png")