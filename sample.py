import argparse
import os

import torch
from torchvision.utils import save_image

from unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--weights", default="model", choices=["model", "ema"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    generator = UNet().to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)[args.weights])
    generator.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    index = 0
    with torch.no_grad():
        while index < args.num_samples:
            n = min(args.batch_size, args.num_samples - index)
            images = (generator(torch.randn(n, 3, 32, 32, device=device)).clamp(-1, 1) + 1) / 2
            for image in images:
                save_image(image, os.path.join(args.output_dir, f"{index:06d}.png"))
                index += 1


if __name__ == "__main__":
    main()
