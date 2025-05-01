import argparse
import csv
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
import gc

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageTextDataset(Dataset):
    def __init__(self, csv_file):
        self.entries = []
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)

        # Simpler transformations to reduce memory usage
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        image = self.transform(image)
        prompt = entry["prompt"]
        return {"pixel_values": image, "prompt": prompt}


def train_model(dataset_csv, num_epochs=5, batch_size=1, learning_rate=1e-4):
    # Memory cleanup before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Load the training dataset with a small batch size
    dataset = CustomImageTextDataset(dataset_csv)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)  # Reduced workers and batch size

    # Load model components one by one to manage memory
    print("Loading tokenizer...")
    model_id = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    print("Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    )
    text_encoder = text_encoder.to(device)
    text_encoder.requires_grad_(False)  # Freeze text encoder

    print("Loading UNet and VAE...")
    # Load with lower precision and safety checker disabled
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # Enable gradient checkpointing for memory efficiency
    pipe.unet.enable_gradient_checkpointing()

    # Move VAE and UNET to device but keep VAE in eval mode to save memory
    pipe.vae = pipe.vae.to(device)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()

    pipe.unet = pipe.unet.to(device)

    # Apply LoRA with minimal parameters
    lora_config = LoraConfig(
        r=4,  # Reduced rank
        lora_alpha=8,
        target_modules=["to_k", "to_v"],
        lora_dropout=0.0,  # Remove dropout to save compute
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    print("Applied LoRA injection via PEFT.")

    # Use a lightweight optimizer
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # Add weight decay for stability
        eps=1e-8
    )

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Training loop with memory optimizations
    pipe.unet.train()

    # Calculate total number of training steps
    total_steps = len(dataloader) * num_epochs
    print(f"Starting training with {total_steps} total steps")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(dataloader):
            # Clear cache periodically
            if step % 5 == 0:
                torch.cuda.empty_cache()

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):  # Mixed precision
                # Process text conditioning
                text_inputs = tokenizer(
                    batch["prompt"],
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                )
                text_input_ids = text_inputs.input_ids.to(device)

                # Get text embeddings without gradient computation
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_input_ids)[0]

                # Process images with reduced precision
                images = batch["pixel_values"].to(device, dtype=torch.float16)

                # Convert images to latents
                with torch.no_grad():
                    latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215

                # Generate noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)

                # Add noise to latents according to noise schedule
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get model prediction
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute loss with original precision
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")

            # Explicitly delete tensors to free memory
            del noise_pred, loss, noisy_latents, latents, noise, images, text_input_ids, encoder_hidden_states
            torch.cuda.empty_cache()

    # Save only the LoRA weights (much smaller)
    os.makedirs("lora_finetuned", exist_ok=True)
    pipe.unet.save_pretrained("lora_finetuned")
    print("Training complete! LoRA weights saved to 'lora_finetuned'.")

    # Free memory
    del pipe, text_encoder, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


def generate_image(prompt, output_file="generated_image.png", num_inference_steps=25, guidance_scale=7.5):
    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # Load model with minimal components and half precision
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable safety checker to save memory
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # Load fine-tuned LoRA weights if available
    if os.path.exists("lora_finetuned"):
        try:
            # For newer versions of diffusers that support direct adapter loading
            if hasattr(pipe.unet, 'load_adapter'):
                pipe.unet.load_adapter("lora_finetuned")
                print("Loaded fine-tuned LoRA weights via adapter.")
            else:
                # Fall back to direct loading for compatibility
                state_dict = torch.load(os.path.join("lora_finetuned", "pytorch_model.bin"), map_location=device)
                pipe.unet.load_state_dict(state_dict, strict=False)
                print("Loaded fine-tuned LoRA weights via state dict.")
        except Exception as e:
            print(f"Fine-tuned weights not loaded, using base model. Error: {e}")

    # Generate image with reduced steps for speed and memory
    with torch.inference_mode():
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
        ).images[0]

    # Save the generated image
    image.save(output_file)
    print(f"Image saved to {output_file}.")

    # Free memory
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient fine-tuning of Stable Diffusion with LoRA")
    parser.add_argument("mode", choices=["t", "r"], help="Mode: 't' to train, 'r' to run/infer")
    parser.add_argument("input", help="For training: path to dataset CSV; for run: text prompt")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size (default: 1)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output file name for generation")

    args = parser.parse_args()

    if args.mode == "t":
        # Train mode
        if not os.path.exists(args.input):
            print(f"Dataset file {args.input} not found!")
            return
        train_model(
            dataset_csv=args.input,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    elif args.mode == "r":
        # Run mode
        generate_image(prompt=args.input, output_file=args.output)


if __name__ == "__main__":
    main()
