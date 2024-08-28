import torch 
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(prompt: str, uncond_prompt:str, input_image=None,
             strength= 0.8, do_cfg= True, cfg_scales = 7.5, sampler_name = "ddpm",
             n_inference_steps = 50, models ={}, seed= None, device =None,
             idle_device= None, tokenizer=None):
    
    with torch.no_grad():
        if not (0<strength<1):
            raise ValueError("Strenght  must bbe 0 and 1")
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)

        else:
            to_idle: lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # convert prompt into tokes using tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length",max_lengths = 77).input_ids
            # (Batch_size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_Len) -> (Batch_size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length",max_lengths = 77).input_ids
            # (Batch_size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_Len) -> (Batch_size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)

            # (2, Seq_Len, Dim) -> (2,77, 768)
            context = torch.cat([cond_context,uncond_context])

        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length",max_lengths = 77).input_ids
            # (Batch_size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_Len) -> (Batch_size, Seq_Len, Dim)
            context = clip(tokens)

        if sampler_name=="ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)

        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latent_shape = (1,4,LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            
            input_image_tensor = input_image.resize(WIDTH,HEIGHT)
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0,255),(-1,1))
            # (Height, Width, Channel) -> (Batch_size, Height, Width)
            input_image_tensor = input_image_tensor.unsequeeze(0)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            # run image through through encoder of VAE
            latents = encoder(input_image_tensor, encoder_noise)

            to_idle(encoder)

        else:
            # if text-to-image, start with random noise N(0, I)
            latents =  torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timestemps = tqdm(sampler.timestemps)
        for i, timestemp in enumerate(timestemps):
            # (1,320)
            time_embedding = get_time_embedding(timestemp).to(device)

            # (Batch_size, 4, Latent_height, Latents_width)
            model_input = latents

            # (Batch_size, 4, Latent_height, Latents_width) -> (2*Batch_size, 4, Latent_height, Latents_width)
            if do_cfg:
                model_input = model_input.repeat(2,1,1,1)

            # model ouput is the predicted by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                cond_output, uncond_output = model_output.chunk(2)
                model_ouput = cfg_scales*(cond_output - uncond_output) + uncond_output

            # Remove the noise predicted by the unet
            latents = sampler.step(timestemp, latents, model_ouput)
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1,1),(0,255), clamp=True)

        # (Batch_size, Channel, Height, Width)-> (Batch_size,Height, Width, Channel)
        images = images.permute(0,2,3,1)
        images = images.to("cpu",torch.uint8).numpy()
        
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
