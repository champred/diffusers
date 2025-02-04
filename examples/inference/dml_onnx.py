import os, inspect, warnings, time
from typing import List, Optional, Union

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self, *,
        vae: Optional[AutoencoderKL] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        tokenizer: CLIPTokenizer,
        unet: Optional[UNet2DConditionModel] = None,
        scheduler: EulerDiscreteScheduler,
        safety_checker: Optional[StableDiffusionSafetyChecker] = None,
        feature_extractor: Optional[CLIPFeatureExtractor] = None
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        **kwargs
    ):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)
        
        onnx = False
        if "execution_provider" in kwargs:
            onnx = True
            ep = kwargs.pop("execution_provider")
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.enable_mem_pattern=False
            self.unet_sess = self.unet_sess if hasattr(self, "unet_sess") else ort.InferenceSession("onnx/unet.onnx", so, providers=[ep])
            self.post_quant_conv_sess = self.post_quant_conv_sess if hasattr(self, "post_quant_conv_sess") else ort.InferenceSession("onnx/post_quant_conv.onnx", so, providers=[ep])
            self.decoder_sess = self.decoder_sess if hasattr(self, "decoder_sess") else ort.InferenceSession("onnx/decoder.onnx", so, providers=[ep])
            self.encoder_sess = self.encoder_sess if hasattr(self, "encoder_sess") else ort.InferenceSession("onnx/encoder.onnx", so, providers=[ep])

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=False,
            return_tensors="pt",
        )
        
        if onnx: text_embeddings = self.encoder_sess.run(None, {"text_input": text_input.input_ids.numpy()})[0]
        else: text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            negative_prompt = kwargs.pop("negative_prompt") if "negative_prompt" in kwargs else None
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens, padding="max_length", max_length=max_length, truncation=False, return_tensors="pt"
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if onnx: 
                uncond_embeddings = self.encoder_sess.run(None, {"text_input": uncond_input.input_ids.numpy()})[0]
                text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])
            else: 
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, 4, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, generator=generator, device=self.device)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma
        if onnx: latents = latents.numpy() # use pytorch rand to get consistent results

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            if onnx and do_classifier_free_guidance: latent_model_input = np.concatenate([latents] * 2)
            else: latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            if onnx: latent_model_input = latent_model_input.numpy().astype('float32')

            # predict the noise residual
            if onnx:
                inp = {"latent_model_input": latent_model_input, 
                       "t": np.array([t], dtype=np.int64), 
                       "encoder_hidden_states": text_embeddings}
                noise_pred = self.unet_sess.run(None, inp)[0]
            else:
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            # perform guidance
            if do_classifier_free_guidance:
                if onnx: noise_pred_uncond, noise_pred_text = np.array_split(noise_pred, 2)
                else: noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if onnx:
                latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)["prev_sample"].numpy()
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        if onnx:
            latents = self.post_quant_conv_sess.run(None, {"latents": latents.astype("float32")})[0]
            image = self.decoder_sess.run(None, {"latents": latents})[0]
            image = np.clip((image / 2 + 0.5), 0, 1)
            image = np.transpose(image, (0, 2, 3, 1))
        else:
            image = self.vae.decode(latents)
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        # safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        # image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image,}# "nsfw_content_detected": has_nsfw_concept}

scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
pipe = StableDiffusionPipeline(tokenizer=tokenizer, scheduler=scheduler)
torch.manual_seed(time.time_ns())

prompt = input("Prompt: ")
# TODO not exhaustive, make sure your prompts don't have any other illegal characters
path = prompt.replace("/", " ")[:189]
os.makedirs(path, exist_ok=True)
for i in range(int(input("Number: "))):
    print(f"Generating #{i+1}")
    image = pipe(prompt, height=512, width=512, num_inference_steps=30, guidance_scale=7.5, eta=0.0, execution_provider="DmlExecutionProvider", negative_prompt="")["sample"][0]
    image.save(f"{path}/{time.time_ns()}.png")

# Works on AMD Windows platform
# Image width and height is set to 512x512
# If you need images of other sizes (size must be divisible by 8), make sure to save the model with that size in save_onnx.py
# For example, if you need height=512 and width=768, change create_onnx.py with height=512 and width=768 and run the prompt below with height=512 and width=768
# Default values are height=512, width=512, num_inference_steps=50, guidance_scale=7.5, eta=0.0, execution_provider="DmlExecutionProvider"

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt, height=512, width=768, num_inference_steps=50, guidance_scale=7.5, eta=0.0, execution_provider="DmlExecutionProvider")["sample"][0]
# image.save("Dml_1.png")
