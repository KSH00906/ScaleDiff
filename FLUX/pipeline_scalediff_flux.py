# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor


def refine(x_ref, x_pred, scale, scale_factor = 0.5):
    assert x_ref.shape == x_pred.shape, f"x_ref and x_pred must have the same shape, but got {x_ref.shape} and {x_pred.shape}"
    B, C, H, W = x_pred.shape
    h, w = int(H * scale_factor), int(W * scale_factor)
    x_ref_D = F.interpolate(x_ref, size = (h, w), mode='bilinear', antialias=True)
    x_ref_UD = F.interpolate(x_ref_D, size=(H, W), mode='bilinear')

    x_pred_D = F.interpolate(x_pred, size = (h, w), mode='bilinear', antialias=True)
    x_pred_UD = F.interpolate(x_pred_D, size = (H, W), mode='bilinear')

    x_guided = x_pred + scale * (x_ref_UD - x_pred_UD)
    return x_guided

class FluxPipeline(FluxPipeline):
    def get_pred_x0(
        self,
        xt: torch.FloatTensor,
        t: Union[float, torch.Tensor],
        v: torch.FloatTensor
        ) -> torch.FloatTensor:
        idx = self.scheduler.index_for_timestep(t)
        sigma = self.scheduler.sigmas[idx]
        while sigma.dim() < xt.dim():
            sigma = sigma.unsqueeze(-1)
        x0 = xt - sigma * v
        return x0.to(xt.dtype)

    def get_model_output(
        self,
        x0: torch.FloatTensor,
        xt: torch.FloatTensor,
        t: Union[float, torch.Tensor]
        ) -> torch.FloatTensor:
        idx = self.scheduler.index_for_timestep(t)
        sigma = self.scheduler.sigmas[idx]
        while sigma.dim() < xt.dim():
            sigma = sigma.unsqueeze(-1)
        v = (xt - x0) / sigma
        return v.to(xt.dtype)
    
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width):
        batch_size, num_patches, channels = latents.shape

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents
    
    def encode_image(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        B, C, H, W = latents.shape
        latent_image_ids = self._prepare_latent_image_ids(B, H // 2, W // 2, self.device, self.dtype)
        return latents, latent_image_ids
    
    def decode_latents(self, latents):
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return image
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids
    
    def get_noise_pred(self, t, latents, guidance, pooled_prompt_embeds, prompt_embeds, text_ids, latent_image_ids):
        B, C, H, W = latents.shape
        timestep = t.expand(B).to(latents.dtype)

        latents = self._pack_latents(latents, B, C, H, W)

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = self._unpack_latents(noise_pred, H, W)

        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        ###############################################################################################
        restart_ratio = 0.6,
        scale_factor = 0.25,
        upsample_stage = 2,
        query_random_jitter = True,
        t5_to_cpu = False, # vram optimization
    ):
        base_height = height or self.default_sample_size * self.vae_scale_factor
        base_width = width or self.default_sample_size * self.vae_scale_factor

        base_latent_height = 2 * (int(base_height) // (self.vae_scale_factor * 2))
        base_latent_width = 2 * (int(base_width) // (self.vae_scale_factor * 2))

        if query_random_jitter:
            self.transformer.query_random_jitter = True
        else:
            self.transformer.query_random_jitter = False

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        # Move text encoders to GPU for encoding
        if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None and t5_to_cpu:
            self.text_encoder_2.to(device)

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # Move text encoders back to CPU to save VRAM
        if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None and t5_to_cpu:
            self.text_encoder_2.to('cpu')

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            base_latent_height,
            base_latent_width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = 4096
        image_seq_len = max(min(image_seq_len, self.scheduler.config.max_image_seq_len), self.scheduler.config.base_image_seq_len)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        ###################################################### Phase Initialization ########################################################
        output_images = []
        self.transformer.NPAttn=False
        print(f'Phase 1 Denoising')
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = self.get_noise_pred(t, latents, guidance, pooled_prompt_embeds, prompt_embeds, text_ids, latent_image_ids)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        image = self.decode_latents(latents)
        output_images.append(self.image_processor.postprocess(image, output_type=output_type)[0])

        ####################################################### Phase Upscaling #####################################################
        for p in range(1, upsample_stage+1):
            restart_step = int(num_inference_steps * (1-restart_ratio))

            current_height = int(base_height * (2**p))
            current_width = int(base_width * (2**p))
            current_latent_height = int(base_latent_height * (2**p))
            current_latent_width = int(base_latent_width * (2**p))

            # Latent Frequency Mixing
            latents_LU = F.interpolate(latents, size=(current_latent_height, current_latent_width), mode='bicubic')
            image_RU = F.interpolate(image.to(device), size=(current_height, current_width), mode='bicubic')
            latents_RU, latent_image_ids = self.encode_image(image_RU)
            latents_LFM = refine(latents_LU, latents_RU, 1, scale_factor=scale_factor)

            # diffuse
            noise = torch.randn_like(latents_LFM)
            latents = self.scheduler.scale_noise(latents_LFM, timesteps[restart_step].unsqueeze(0), noise)
            self.scheduler._step_index = restart_step

            self.transformer.init_NPA(current_latent_height//2, current_latent_width//2)
            print(f'Phase {p+1} Denoising')
            with self.progress_bar(total=num_inference_steps-restart_step) as progress_bar:
                for i, t in enumerate(timesteps[restart_step:]):
                    noise_pred = self.get_noise_pred(t, latents, guidance, pooled_prompt_embeds, prompt_embeds, text_ids, latent_image_ids)
                    
                    # structure guidance
                    scale = self.scheduler.sigmas[self.scheduler.index_for_timestep(t)] 
                    pred_x0 = self.get_pred_x0(latents, t, noise_pred)
                    pred_x0 = refine(latents_LFM, pred_x0, scale, scale_factor=scale_factor)
                    noise_pred = self.get_model_output(pred_x0, latents, t)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
            
            image = self.decode_latents(latents)
            output_images.append(self.image_processor.postprocess(image, output_type=output_type)[0])

        # Offload all models
        self.maybe_free_model_hooks()

        return output_images


    