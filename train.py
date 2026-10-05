import argparse
import copy
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image

from distributed import is_distributed
from drift import drift_loss
from encoder import ResidualEncoder
from kde_loss import encoder_kde_loss
from unet import UNet


def build_dataset(name, root):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    if name == "cifar10":
        return datasets.CIFAR10(root, train=True, download=True, transform=transform)
    if name == "cifar100":
        return datasets.CIFAR100(root, train=True, download=True, transform=transform)
    if name == "svhn":
        return datasets.SVHN(root, split="train", download=True, transform=transform)
    raise ValueError(name)


def infinite_loader(dataset, batch_size):
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True)
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for images, _ in loader:
            yield images
        epoch += 1


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./runs/cifar10")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--encoder-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--temps", default="0.05")
    parser.add_argument("--v-max", type=float, default=0.005)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = not distributed or dist.get_rank() == 0
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    temps = tuple(float(t) for t in args.temps.split(","))

    generator = UNet().to(memory_format=torch.channels_last).to(device)
    ema = EMA(generator, args.ema_decay)
    compiled = torch.compile(generator)
    model = DDP(compiled, device_ids=[local_rank]) if distributed else compiled
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    encoder = ResidualEncoder().to(device)
    encoder_model = DDP(encoder, device_ids=[local_rank]) if distributed else encoder
    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.encoder_lr, betas=(0.9, 0.999),
                                          weight_decay=0.0, fused=True)

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        generator.load_state_dict(state["model"])
        ema.shadow.load_state_dict(state["ema"])
        encoder.load_state_dict(state["encoder"])
        optimizer.load_state_dict(state["optimizer"])
        encoder_optimizer.load_state_dict(state["encoder_optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]

    dataset = build_dataset(args.dataset, args.data_root)
    encoder_data = infinite_loader(dataset, args.batch_size)
    generator_data = infinite_loader(dataset, args.batch_size)

    def noise(n):
        return torch.randn(n, 3, 32, 32, device=device).to(memory_format=torch.channels_last)

    if is_main:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

    for step in range(start_step + 1, args.steps + 1):
        encoder.train().requires_grad_(True)
        real = next(encoder_data).to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            fake = compiled(noise(real.shape[0]))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            groups = encoder_model(torch.cat([real, fake.detach()], dim=0))
        b = real.shape[0]
        encoder_loss = encoder_kde_loss([g[:b].float() for g in groups], [g[b:].float() for g in groups], temps)
        encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.encoder_max_grad_norm)
        encoder_optimizer.step()
        encoder.eval().requires_grad_(False)

        real = next(generator_data).to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pos_groups = [g.float() for g in encoder(real)]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = model(noise(real.shape[0]))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_groups = [g.float() for g in encoder(gen.float())]
        loss = drift_loss(gen_groups, pos_groups, temps, args.v_max)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update(generator)

        if is_main and step % args.log_every == 0:
            print(f"step {step} drift_loss {loss.item():.6f} encoder_loss {encoder_loss.item():.6f}", flush=True)
        if is_main and step % args.save_every == 0:
            torch.save({
                "step": step,
                "model": generator.state_dict(),
                "ema": ema.shadow.state_dict(),
                "encoder": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }, os.path.join(args.output_dir, "checkpoints", f"step{step:07d}.pt"))
            generator.eval()
            with torch.no_grad():
                samples = generator(noise(64)).clamp(-1, 1)
            generator.train()
            save_image(samples, os.path.join(args.output_dir, "samples", f"step{step:07d}.png"),
                       nrow=8, normalize=True, value_range=(-1, 1))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
