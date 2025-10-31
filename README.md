[![](https://img.shields.io/badge/arXiv-2510.25818-b31b1b)](https://arxiv.org/abs/2510.25818)

# ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion

<p align="center">
  Sungho Koh •
  SeungJu Cha •
  Hyunwoo Oh •
  Kwanyoung Lee •
  Dong-Jin Kim
</p>

<p align="center">
  <b>Hanyang University</b>
</p>

<p align="center">
  <i>NeurIPS 2025</i>
</p>

<br>

![main figure](assets/figure-qualitative-results.png)


## 🔍 Overview

![method figure](assets/figure-method-overview.png)

ScaleDiff is a **training-free and model-agnostic framework** for extending pretrained diffusion models (SDXL, FLUX) to ultra-high resolutions (up to 4K) efficiently.  
It integrates **NPA**, **LFM**, and **SG** into a single *upsample–diffuse–denoise* pipeline.

| Component | Role |
|------------|------|
| **NPA** | Efficient attention mechanism removing redundant patch overlap |
| **LFM** | Latent–RGB frequency mixing for fine details |
| **SG** | Structure alignment for global consistency |


---

## ⚙️ Installation

```bash
git clone https://github.com/KSH00906/ScaleDiff.git
cd ScaleDiff
conda create -n scalediff python=3.13
conda activate scalediff
pip install -r requirements.txt
```

---

## 🚀 Usage

### SDXL

```python
from SDXL.pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline 
import torch

# Load pretrained SDXL model
ckpt_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = CustomStableDiffusionXLPipeline.from_pretrained(
    ckpt_path, 
    torch_dtype=torch.float16
).to("cuda")
pipe.vae.enable_tiling()

# Generate high-resolution image
prompt = "a woman"
negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

generator = torch.Generator(device='cuda').manual_seed(77)

images = pipe(
    prompt, 
    negative_prompt=negative_prompt,
    height=1024, 
    width=1024,
    generator=generator,
    num_inference_steps=50, 
    guidance_scale=7.5,
    restart_ratio=0.6,      # Controls restart schedule
    scale_factor=0.125,     # Upsampling scale factor
    upsample_stage=2,       # Number of upsampling stages
)

for i, image in enumerate(images):
    image.save(f"{prompt}_{i}.png")
```

Or run the example script:
```bash
cd SDXL
python run_scalediff_sdxl.py
```

### FLUX.1-dev

```python
import torch
from FLUX.pipeline_scalediff_flux import FluxPipeline
from FLUX.transformer_scalediff_flux import FluxTransformer2DModel

# Load pretrained FLUX model
ckpt = "black-forest-labs/FLUX.1-dev"
transformer = FluxTransformer2DModel.from_pretrained(
    ckpt, 
    torch_dtype=torch.bfloat16, 
    subfolder='transformer'
)
pipe = FluxPipeline.from_pretrained(
    ckpt, 
    torch_dtype=torch.bfloat16, 
    transformer=transformer
).to('cuda')
pipe.vae.enable_tiling()

# Generate high-resolution image
prompt = "a woman"

generator = torch.Generator(device="cuda").manual_seed(77)

images = pipe(
    prompt + ", highly detailed, 4k resolution, best quality",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=30,
    max_sequence_length=256,
    generator=generator,
    restart_ratio=0.6,
    scale_factor=0.25,
    upsample_stage=2,
    query_random_jitter=True,
    t5_to_cpu=True,
)

for i, image in enumerate(images):
    image.save(f"{prompt}_{i}.png")
```

Or run the example script:
```bash
cd FLUX
python run_scalediff_flux.py        # For FLUX.1-dev
python run_scalediff_flux_schnell.py # For FLUX.1-schnell (faster)
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `height`, `width` | Starting resolution | 1024 |
| `restart_ratio` | Noise addition step ratio | 0.4 (SDXL)<br>0.6 (FLUX) |
| `scale_factor` | Downsampling ratio for frequency decomposition | 0.125 (SDXL)<br>0.25 (FLUX) |
| `upsample_stage` | Number of progressive upsampling stages.<br>Output resolution: height × 2^upsample_stage | 2 |
| `query_random_jitter` | Reduce boundary artifacts with<br>minimal computation cost (FLUX only) | True |
| `t5_to_cpu` | Offload T5 encoder to CPU<br>to save VRAM (FLUX only) | True |

---

## 📝 Citation


```bibtex
@article{koh2025scalediff,
      title={ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion}, 
      author={Sungho Koh and SeungJu Cha and Hyunwoo Oh and Kwanyoung Lee and Dong-Jin Kim},
      journal={arXiv preprint arXiv:2510.25818},
      year={2025},
}
```
