import os
import argparse
import torch
import types
from transformers import TrainingArguments
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.experiment.runner import TrainRunner
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
# Add this to your imports at the top
from gr00t.model.action_head.cross_attention_dit import TimestepEncoder
# Patch action head dtype property
@property
def safe_dtype(self):
    params = list(self.parameters())
    if not params:
        return torch.bfloat16
    return params[0].dtype

# Patch sample_time to use float32 for sampling
original_sample_time = FlowmatchingActionHead.sample_time

def patched_sample_time(self, batch_size, device, dtype):
    # Always use float32 for the sampling part
    alpha = self.beta_dist.concentration1.to(torch.float32)
    beta = self.beta_dist.concentration0.to(torch.float32)
    float32_beta_dist = torch.distributions.Beta(alpha, beta)
    
    # Sample using float32
    sample = float32_beta_dist.sample([batch_size]).to(device)
    
    # Calculate time in float32 using the same formula as the original
    t = (0.999 - sample) / 0.999  # Using 0.999 as noise_s (from the config)
    
    # Convert to the desired dtype before returning
    return t.to(dtype)  # Return just t, not a tuple
# Apply patches
FlowmatchingActionHead.dtype = safe_dtype
FlowmatchingActionHead.sample_time = patched_sample_time

def main(args):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data configuration
    print(f"Loading two_wheel data configuration...")
    data_config = DATA_CONFIG_MAP["two_wheel"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # Load model with explicit bf16 dtype
    print(f"Loading model from: {args.model_path}")
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        model = GR00T_N1.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            tune_llm=False,
            tune_visual=True,
            tune_projector=True,
            tune_diffusion_model=True,
            torch_dtype=torch.bfloat16
        )
    
    # Move to device
    print(f"Moving model to {device}")
    model = model.to(device)
    
    # Check action head initialization
    if not any(p.requires_grad for p in model.action_head.parameters()):
        print("Action head has no trainable parameters. Initializing...")
        try:
            # Create dummy data for initialization
            action_dim = model.config.action_dim
            dummy_backbone_outputs = torch.randn(1, 1024).to(device).to(torch.float32)  # Use float32 for init
            dummy_action_inputs = {
                "noisy_trajectory": torch.randn(1, 16, action_dim).to(device).to(torch.float32),
                "t_discretized": torch.randint(0, 1000, (1,)).float().to(device),
                "embodiment_id": torch.tensor([0]).to(device),
            }
            _ = model.action_head(
                backbone_outputs=dummy_backbone_outputs,
                action_inputs=dummy_action_inputs
            )
            print("Action head initialized successfully")
        except Exception as e:
            print(f"Failed to initialize action head: {e}")
    
    # Prepare training arguments
    output_dir = os.path.expanduser(args.output_dir)
    print(f"Setting up training - output will be saved to: {output_dir}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name="two_wheel_finetune",
        remove_unused_columns=False,
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        evaluation_strategy="no",
        save_total_limit=3,
        seed=args.seed,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )
    
    # Setup dataset
    from gr00t.data.dataset import LeRobotSingleDataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        transforms=modality_transform,
    )
    
    # Set up experiment runner
    print("Setting up experiment runner...")
    experiment = TrainRunner(
        train_dataset=dataset,
        model=model,
        training_args=training_args,
    )
    
    # Patch the compute_loss method to handle dtype conversion
    original_compute_loss = experiment.trainer.compute_loss
    
    def patched_compute_loss(self, model, inputs, return_outputs=False):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Convert inputs to bf16 where appropriate, excluding special cases
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].dtype == torch.float32:
                    if key != "embodiment_id" and "discretized" not in key.lower():
                        inputs[key] = inputs[key].to(torch.bfloat16)
            return original_compute_loss(model, inputs, return_outputs)
    
    experiment.trainer.compute_loss = types.MethodType(patched_compute_loss, experiment.trainer)
    print("Successfully patched trainer's compute_loss")

    # Patch the TimestepEncoder forward method
    original_timestep_encoder_forward = TimestepEncoder.forward

    def patched_timestep_encoder_forward(self, timestep):
        try:
            # Original implementation tries to get dtype from parameters
            return original_timestep_encoder_forward(self, timestep)
        except Exception:  # Catch any exception, not just StopIteration
            # Fixed embedding dimension based on the error message
            embed_dim = 1536  # Changed from 320 to 1536 to match expected size
            
            # Convert timestep to float32 for calculation
            timesteps = timestep.to(torch.float32)
            
            # Generate sinusoidal embeddings following standard practice
            half_dim = embed_dim // 2
            emb = torch.log(torch.tensor(10000.0)).to(device=timesteps.device) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
            emb = timesteps[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            
            if embed_dim % 2 == 1:  # zero pad if needed
                emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
            
            # Convert to bfloat16 for mixed precision training
            emb = emb.to(torch.bfloat16)
            
            return emb

    # Apply the patch
    TimestepEncoder.forward = patched_timestep_encoder_forward

    # Initialize timestep encoder components
    print("Attempting to initialize TimestepEncoder components...")
    try:
        # Find the timestep encoder in the model
        if hasattr(model.action_head, 'model') and hasattr(model.action_head.model, 'timestep_encoder'):
            encoder = model.action_head.model.timestep_encoder
            # Create a dummy time input to trigger initialization
            dummy_time = torch.zeros(1).to(device)
            with torch.no_grad():
                _ = encoder(dummy_time)
            print("Successfully initialized TimestepEncoder")
    except Exception as e:
        print(f"Failed to initialize TimestepEncoder: {e}")
    
    # Start training
    # Start training
    print("\nStarting training...")
    try:
        experiment.train()
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

# Parse args and run main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GR00T-N1 on a 2-wheel robot dataset")
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1-2B",
                        help="Path to the pretrained GR00T-N1 model.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to your 2-wheel robot dataset.")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment",
                        help="Embodiment tag for fine-tuning.")
    parser.add_argument("--video_backend", type=str, default="torchvision_av",
                        choices=["torchvision_av", "decord"],
                        help="Video backend to use.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device training batch size.")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Maximum training steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Steps at which to save checkpoints.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory for saving checkpoints.")
    
    args = parser.parse_args()
    main(args)