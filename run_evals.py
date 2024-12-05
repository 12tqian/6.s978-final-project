from torchvision import datasets, transforms
import torch
from pathlib import Path

import random
from tqdm import tqdm
from torch import nn

import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchvision.models import inception_v3
from scipy.stats import entropy
from torchvision.utils import make_grid

from networks_edm import SongUNet
from functools import partial
import pandas as pd

import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device == torch.device("cuda")


class EDMVel(torch.nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        ## parameters
        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.rho = 7.0
        self.sigma_data = 0.5
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.ema_rampup_ratio = 0.05
        self.ema_halflife_kimg = 500
        self.drop_prob = 0.1

    def forward(
        self,
        x_,
        sigma_,
        class_labels=None,
        augment_labels=None,
        force_fp32=False,
        use_ema=False,
        **model_kwargs,
    ):
        # sigma_[sigma_>0.988] = 0.988
        # Assume sigma_ is in [0.001, 80/81].
        # assert (sigma_ >= 0.001).all() and (sigma_ <= 80/81).all()
        # Rescale x and sigma.
        x = x_ / (1 - sigma_).view(-1, 1, 1, 1)
        sigma = sigma_ / (1 - sigma_)

        # Compute posterior mean D_x
        x = x.to(torch.float32)
        # augment_labels = torch.zeros([x.shape[0], self.augment_dim], device = x.device)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        # class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        if use_ema:
            F_x = self.ema(
                (c_in * x).to(dtype),
                c_noise.flatten(),
                class_labels=class_labels,
                **model_kwargs,
            )
        else:
            F_x = self.model(
                (c_in * x).to(dtype),
                c_noise.flatten(),
                class_labels=class_labels,
                **model_kwargs,
            )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Compute v_t
        v_t = (x_ - D_x) / sigma_.view(-1, 1, 1, 1)
        return v_t

    def update_ema(self, step, batch_size=512):
        # TODO(tcqian): should never be called on inference, batch size doesn't mater
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(
                ema_halflife_nimg, step * batch_size * self.ema_rampup_ratio
            )
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def create_model():
    unet = SongUNet(
        in_channels=1,
        out_channels=1,
        num_blocks=2,
        attn_resolutions=[0],
        model_channels=32,
        channel_mult=[1, 2, 2],
        dropout=0.13,
        img_resolution=28,
        label_dim=10,
        label_dropout=0.1,
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        augment_dim=0,
        channel_mult_noise=1,
        resample_filter=[1, 1],
    )
    return unet


def get_split_time_schedule(n_steps_1, n_steps_2, split=0.5):
    time_schedule = []
    split = 1 - split
    for i in range(n_steps_2):
        time_schedule += [split * (i / n_steps_2)]

    for i in range(n_steps_1 + 1):
        time_schedule += [split + (1 - split) * (i / n_steps_1)]

    time_schedule = list(reversed(time_schedule))
    time_schedule[0] = 1 - 1e-5
    return time_schedule


@torch.no_grad()
def v_edm_sampler(
    v_edm,
    latents,
    class_labels=None,
    num_steps=18,
    guide_w=0.0,
    use_ema=True,
    cfg=True,
    drop_prob=0.0,
    suppress_print=False,
    n_steps_1=None,
    n_steps_2=None,
):
    time_schedule = [(i / num_steps) for i in reversed(range(1, num_steps + 1))] + [0]
    time_schedule[0] = 1 - 1e-5
    if n_steps_1 is not None and n_steps_2 is not None:
        time_schedule = get_split_time_schedule(n_steps_1, n_steps_2, split=0.5)
    cnt = 0
    if not suppress_print:
        print(f"Time schedule: {time_schedule}")

    if class_labels is not None:
        class_labels = (
            nn.functional.one_hot(class_labels, num_classes=10)
            .type(torch.float32)
            .to(latents.device)
        )
        mask = torch.bernoulli(torch.ones_like(class_labels) * (1 - drop_prob))
        class_labels = class_labels * mask
        if cfg:
            class_labels = torch.cat([class_labels, torch.zeros_like(class_labels)])
    # Main sampling loop.
    x_next = latents.to(torch.float64).to(device) * time_schedule[0]

    rand_x = random.choice([10 + i for i in range(8)])
    rand_y = random.choice([10 + i for i in range(8)])

    random_trajs = [
        [x_next[i, 0, rand_x, rand_y].detach().cpu().item()]
        for i in range(latents.shape[0])
    ]
    middle = None
    for i in tqdm(range(len(time_schedule[:-1]))):
        t = (
            torch.ones((latents.shape[0] * (2 if cfg else 1),), device=device)
            * time_schedule[i]
        )
        t_next = (
            torch.ones((latents.shape[0] * (2 if cfg else 1),), device=x_next.device)
            * time_schedule[i + 1]
        )
        dt = t_next[0] - t[0]
        if cfg:
            x_hat = x_next.repeat(2, 1, 1, 1)
        else:
            x_hat = x_next

        vt = v_edm(x_hat, t, class_labels, use_ema=use_ema).to(torch.float64)
        x_next = x_hat + vt * dt

        # cfg guide
        if cfg:
            x_next_cond, x_next_uncond = x_next.chunk(2)
            x_next = (1 + guide_w) * x_next_cond - guide_w * x_next_uncond
        if i == (num_steps // 2):
            middle = x_next.clone().detach()

    return x_next, middle, class_labels


# Function to preprocess MNIST images for Inception-v3
def preprocess_for_inception(images):
    # Rescale images to 299x299 and convert to RGB
    images = torch.stack([TF.resize(img, (299, 299)) for img in images])
    images = images.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
    return images


def generate_batch(
    v_edm, batch_size: int, n_steps=18, n_steps_1=None, n_steps_2=None, clamp=False
):

    x_T = torch.randn([batch_size, 1, 28, 28], device=device)
    # random from 0 to 9 inclusive
    c = torch.randint(0, 10, (batch_size,), device=device)
    x_gen, _, c = v_edm_sampler(
        v_edm,
        x_T,
        class_labels=c,
        guide_w=4.5,
        suppress_print=True,
        num_steps=n_steps,
        n_steps_1=n_steps_1,
        n_steps_2=n_steps_2,
    )
    if clamp:
        x_gen = (x_gen / 2 + 0.5).clamp(0, 1)
    return x_gen


# Function to compute Inception Score
def compute_inception_score(
    model, gen_func, num_samples=5000, batch_size=32, splits=10, device="cuda"
):
    """
    Computes the Inception Score for a generative model.
    - generator: the generative model that generates MNIST images
    - num_samples: number of samples to generate for evaluation
    - batch_size: number of images per batch
    - splits: number of splits for the score calculation
    - device: 'cuda' or 'cpu'
    """
    # Load the Inception-v3 model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.double()
    inception.eval()

    # Generate samples
    all_preds = []
    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            # Generate images

            fake_images = gen_func(model, batch_size)
            fake_images = preprocess_for_inception(fake_images)
            # should be double
            fake_images = fake_images.to(torch.float64)

            # Compute predictions
            preds = torch.softmax(inception(fake_images), dim=1)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Compute Inception Score
    split_scores = []
    for k in range(splits):
        part = all_preds[
            k * (num_samples // splits) : (k + 1) * (num_samples // splits)
        ]
        py = np.mean(part, axis=0)
        scores = [entropy(p, py) for p in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def get_model(model_path: Path = Path("edm_model_checkpoint.pth")):
    unet = create_model().to(device)
    unet.load_state_dict(torch.load(model_path))
    v_edm = EDMVel(model=unet)
    v_edm.eval()
    return v_edm


def get_is(model_path: Path = Path("edm_model_checkpoint.pth"), gen_func=None):

    tensor_transform = transforms.Compose([transforms.ToTensor()])

    batch_size = 256
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=tensor_transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=tensor_transform
    )

    v_edm = get_model(model_path)

    is_mean, is_std = compute_inception_score(
        v_edm,
        gen_func,
        num_samples=1000,
        batch_size=1000,
        splits=5,
        device="cuda",
    )
    return is_mean, is_std


# Preprocessing for MNIST to fit Inception-v3 input
def preprocess_mnist(images):
    """
    Resize MNIST images to 299x299 and convert grayscale to RGB.
    Args:
        images (torch.Tensor): MNIST images (B, 1, 28, 28)
    Returns:
        torch.Tensor: Preprocessed images (B, 3, 299, 299)
    """
    transform = Compose(
        [
            Resize((299, 299)),
            Normalize(mean=[0.5], std=[0.5]),  # Normalize grayscale images
        ]
    )
    images = images.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
    images = torch.stack([transform(img) for img in images])  # Apply transformations
    return images


# Compute the FID score
def compute_fid(real_loader, fake_loader, device="cuda"):
    """
    Compute FID score between real and generated images.
    Args:
        real_loader (DataLoader): DataLoader for real images.
        fake_loader (DataLoader): DataLoader for generated images.
        device (str): Device to run the computations ('cuda' or 'cpu').
    Returns:
        float: FID score.
    """
    # Load Inception-v3 model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    inception.double()

    def get_activations(data_loader):
        activations = []
        with torch.no_grad():
            for batch in data_loader:
                images, _ = batch  # Labels are not needed
                images = preprocess_mnist(images).to(device)  # Apply preprocessing here
                images = images.to(torch.double)
                preds = inception(images)
                activations.append(preds.cpu().numpy())
        return np.concatenate(activations, axis=0)

    # Compute activations for real and fake images
    real_activations = get_activations(real_loader)
    fake_activations = get_activations(fake_loader)

    # Calculate mean and covariance of activations
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(
        real_activations, rowvar=False
    )
    mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(
        fake_activations, rowvar=False
    )

    # Compute FID score using the formula
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)
    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid


def get_fid(model_path: Path, gen_func, n_samples=1000):
    # Load MNIST dataset for real images
    real_dataset = MNIST(
        root="./data", train=False, transform=ToTensor(), download=True
    )
    use_len = min(n_samples, len(real_dataset))

    real_dataset = torch.utils.data.Subset(real_dataset, list(range(use_len)))

    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
    v_edm = get_model(model_path)
    v_edm.eval()
    # Generate a dataset of fake images
    fake_images = []

    fake_images = gen_func(v_edm, len(real_dataset))
    fake_images = fake_images.to(torch.float64)
    # list
    fake_images_list = []
    for i in range(len(fake_images)):
        fake_images_list.append((fake_images[i].detach().cpu(), 0))

    fake_images = fake_images_list

    fake_loader = DataLoader(fake_images, batch_size=32, shuffle=False)
    fid = compute_fid(real_loader, fake_loader, device="cuda")
    return fid


def compute_metrics(
    model_path: Path, n_steps_1, n_steps_2, n_samples=10000, clamp=False
):
    gen_func = partial(
        generate_batch, n_steps_1=n_steps_1, n_steps_2=n_steps_2, clamp=clamp
    )
    is_mean, is_std = get_is(model_path, gen_func)
    fid = get_fid(model_path, gen_func, n_samples=n_samples)
    return is_mean, is_std, fid


def main():
    model_dir = Path("./models")
    # list all model files
    model_files = list(model_dir.glob("*.pth"))
    clamp = True
    n_samples = 10000
    configs = [
        (1, 9),
        (1, 1),
        (9, 1),
        (9, 9),
    ]

    results_list = []

    for model_file in model_files:
        for config in configs:
            n_steps_1, n_steps_2 = config
            is_mean, is_std, fid = compute_metrics(
                model_file, n_steps_1, n_steps_2, clamp=clamp, n_samples=n_samples
            )
            results_list.append(
                {
                    "Model": model_file,
                    "First Half": n_steps_1,
                    "Second Half": n_steps_2,
                    "IS": is_mean,
                    "FID": fid,
                },
            )
            print(f"Model: {model_file}, Config: {config}, IS: {is_mean}, FID: {fid}")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("results.csv")


if __name__ == "__main__":
    main()
