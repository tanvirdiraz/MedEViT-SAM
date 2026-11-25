import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for p in [str(_repo_dir), str(_repo_dir / 'data'), str(_repo_dir / 'models'), str(_repo_dir / 'configs'), str(_repo_dir / 'efficientvit')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from thop import profile, clever_format
import time
import numpy as np

def measure_efficiency(model_name="efficientvit-sam-l0", image_size=512):
    print(f"Measuring efficiency for {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    model = create_efficientvit_sam_model(name=model_name, pretrained=False).to(device)
    model.eval()

    # 2. Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.image_encoder.parameters())
    decoder_params = sum(p.numel() for p in model.mask_decoder.parameters())
    prompt_params = sum(p.numel() for p in model.prompt_encoder.parameters())
    
    print("\n--- Parameter Count ---")
    print(f"Total Params: {total_params / 1e6:.2f} M")
    print(f"Image Encoder: {encoder_params / 1e6:.2f} M")
    print(f"Mask Decoder: {decoder_params / 1e6:.2f} M")
    print(f"Prompt Encoder: {prompt_params / 1e6:.2f} M")

    # 3. FLOPs using thop
    print("\n--- FLOPs (thop) ---")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    # Just measure the image encoder as it's the bulk of the compute
    macs, params = profile(model.image_encoder, inputs=(dummy_input,), verbose=False)
    macs_fmt, params_fmt = clever_format([macs, params], "%.2f")
    print(f"Image Encoder MACs: {macs_fmt} (roughly {macs * 2 / 1e9:.2f} GFLOPs)")
    
    # 4. Inference Time
    print("\n--- Inference Time (Latency) ---")
    
    # Warmup
    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = model.image_encoder(dummy_input)
    
    # Measure Image Encoder
    print(f"Measuring Image Encoder latency (BS=1, {image_size}x{image_size})...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    
    with torch.no_grad():
        for i in range(100):
            start_events[i].record()
            _ = model.image_encoder(dummy_input)
            end_events[i].record()
            
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    encoder_time_ms = np.mean(times)
    
    # Measure Decoder
    print("Measuring Mask Decoder latency...")
    # Create dummy embeddings
    dummy_image_embedding = torch.randn(1, 256, 64, 64).to(device)
    dummy_point_coords = torch.randn(1, 5, 2).to(device) * image_size
    dummy_point_labels = torch.ones(1, 5).to(device)
    
    # We need to simulate the prompt encoding + decoding
    start_events_dec = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events_dec = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    
    with torch.no_grad():
        for i in range(100):
            start_events_dec[i].record()
            sparse, dense = model.prompt_encoder(
                points=(dummy_point_coords, dummy_point_labels),
                boxes=None, masks=None
            )
            _, _ = model.mask_decoder(
                image_embeddings=dummy_image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False
            )
            end_events_dec[i].record()
            
    torch.cuda.synchronize()
    dec_times = [s.elapsed_time(e) for s, e in zip(start_events_dec, end_events_dec)]
    decoder_time_ms = np.mean(dec_times)
    
    print(f"Encoder Time: {encoder_time_ms:.2f} ms")
    print(f"Decoder Time: {decoder_time_ms:.2f} ms")
    print(f"Total Pipeline: {encoder_time_ms + decoder_time_ms:.2f} ms")
    print(f"FPS (Total Pipeline): {1000 / (encoder_time_ms + decoder_time_ms):.2f} FPS")

if __name__ == "__main__":
    measure_efficiency()
