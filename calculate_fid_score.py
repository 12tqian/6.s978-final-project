import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchvision.models import inception_v3
from scipy.stats import entropy
from torchvision.utils import make_grid

# Function to preprocess MNIST images for Inception-v3
def preprocess_for_inception(images):
    # Rescale images to 299x299 and convert to RGB
    images = torch.stack([TF.resize(img, (299, 299)) for img in images])
    images = images.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
    return images

# Function to compute Inception Score
def compute_inception_score(generator, num_samples=5000, batch_size=32, splits=10, device='cuda'):
    """
    Computes the Inception Score for a generative model.
    - generator: the generative model that generates MNIST images
    - num_samples: number of samples to generate for evaluation
    - batch_size: number of images per batch
    - splits: number of splits for the score calculation
    - device: 'cuda' or 'cpu'
    """
    # Load the Inception-v3 model
    inception = inception_v3(preatrained=True, transform_input=False).to(device)
    inception.eval()

    # Generate samples
    all_preds = []
    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            # Generate images
            noise = torch.randn(batch_size, generator.latent_dim).to(device)  # Assuming latent_dim exists
            fake_images = generator(noise)
            fake_images = preprocess_for_inception(fake_images)

            # Compute predictions
            preds = torch.softmax(inception(fake_images), dim=1)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Compute Inception Score
    split_scores = []
    for k in range(splits):
        part = all_preds[k * (num_samples // splits): (k + 1) * (num_samples // splits)]
        py = np.mean(part, axis=0)
        scores = [entropy(p, py) for p in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# Example usage:
if __name__ == "__main__":
    class SimpleGenerator(torch.nn.Module):
        # Define a simple generator for demonstration
        def __init__(self, latent_dim):
            super().__init__()
            self.latent_dim = latent_dim
            self.model = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 28 * 28),
                torch.nn.Tanh()
            )

        def forward(self, z):
            img = self.model(z)
            img = img.view(-1, 1, 28, 28)  # MNIST images are 28x28 grayscale
            return img

    # Instantiate generator and compute IS
    latent_dim = 100
    generator = SimpleGenerator(latent_dim).to('cuda')
    generator.eval()

    # Generate Inception Score
    is_mean, is_std = compute_inception_score(generator, num_samples=1000, batch_size=32, splits=5, device='cuda')
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")