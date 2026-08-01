"""
VAAL — Variational Adversarial Active Learning.

    Sinha, Ebrahimi & Darrell,
    "Variational Adversarial Active Learning", ICCV 2019.

THE IDEA IN ONE PARAGRAPH
-------------------------
Every other method here asks the classifier what it thinks. VAAL never
asks. Instead it learns to tell labelled images apart from unlabelled ones
by appearance alone, and then labels the images it is *most certain* are
unlabelled — because those are the ones that look least like anything a
human has annotated so far. Two networks do this: a variational
autoencoder compresses every image into a short latent code, and a
discriminator tries to guess from that code alone whether the image came
from the labelled set or the unlabelled one. The autoencoder is trained
adversarially to make the two indistinguishable; whatever the
discriminator can still confidently call "unlabelled" after that is
genuinely unrepresented.

WHY IT IS WORTH THE EXTRA COST
------------------------------
It is *task-agnostic*: it does not use the classifier's predictions, its
features, or its gradients. That makes it the strongest possible test of
whether our result depends on the classifier's own signals being good. It
is also the method whose failure mode is most interesting for us — it has
no access to class labels at all, so it cannot preferentially seek out
malignant cases even in principle.

WHAT IT COSTS
-------------
A VAE and a discriminator, trained from scratch, every round, on top of
the classifier. This is the most expensive baseline by a wide margin
(~4.5 GPU-hours per run against ~2.9 for the others). If compute has to be
cut, cut this one first — CoreSet, BADGE and CLUE together already cover
the standard methodological expectation.

IMPLEMENTATION NOTES — read before comparing against the paper
--------------------------------------------------------------
* Images are downsampled to 64x64 for the VAE only. The paper uses 32x32
  (CIFAR) and 128x128; our classifier sees the full 224x224 and is
  unaffected. Running the VAE at 224 would cost more than the classifier
  it is meant to support, for a representation used only to rank images.
* The VAE operates on ImageNet-normalised tensors, the same ones the
  classifier receives, so reconstruction error is measured in normalised
  space. This is a monotone rescaling per channel and does not change
  which images the discriminator finds unusual.
* Latent dimension 32, β = 1, one discriminator step per two VAE steps —
  all following the paper's reported settings.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

VAE_IMAGE_SIZE = 64
LATENT_DIM = 32
BETA = 1.0            # weight on the KL term
ADVERSARY_WEIGHT = 1.0
VAE_STEPS_PER_ADV_STEP = 2


class _Encoder(nn.Module):
    """64x64 -> 4x4 feature map -> (mu, logvar)."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),    # 64 -> 32
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 32 -> 16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    """latent -> 4x4 feature map -> 64x64 reconstruction."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class _Discriminator(nn.Module):
    """
    Latent code -> logit for "this image came from the labelled set".

    Deliberately small: it sees only the 32-dimensional code, never the
    image, which is what forces the adversarial pressure onto the VAE's
    representation rather than letting the discriminator memorise pixels.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(1)


def _reparameterise(mu, logvar):
    """Sample z ~ N(mu, sigma^2) differentiably."""
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def _vae_loss(recon, target, mu, logvar):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    # KL(N(mu, sigma) || N(0, 1)), averaged over the batch.
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + BETA * kl, recon_loss.item(), kl.item()


def _downsample(images):
    return F.interpolate(images, size=(VAE_IMAGE_SIZE, VAE_IMAGE_SIZE),
                         mode="bilinear", align_corners=False)


def _infinite(loader):
    """Cycle a loader forever — the two sets have very different sizes."""
    while True:
        for batch in loader:
            yield batch


def train_and_score(labeled_dataset, unlabeled_dataset, device,
                    epochs=5, batch_size=32, num_workers=2,
                    lr=5e-4, rng=None, verbose=True):
    """
    Train the VAE and discriminator for this round, then score every
    unlabelled image.

    Returns
    -------
    np.ndarray, shape (N_unlabeled,)
        P(labelled) for each unlabelled image, in dataset order. Low
        values mean the discriminator is confident the image is unlike
        anything labelled — those are the ones VAAL wants.
    """
    generator = torch.Generator()
    generator.manual_seed(int(rng.integers(0, 2 ** 31 - 1)) if rng is not None else 42)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                drop_last=True, generator=generator)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  drop_last=True, generator=generator)

    encoder = _Encoder().to(device)
    decoder = _Decoder().to(device)
    discriminator = _Discriminator().to(device)

    vae_params = list(encoder.parameters()) + list(decoder.parameters())
    opt_vae = torch.optim.Adam(vae_params, lr=lr)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr)

    labeled_stream = _infinite(labeled_loader)
    # One "epoch" is a pass over the unlabelled pool, which is the larger
    # of the two sets; labelled batches are drawn from a cycling stream so
    # both sides contribute equally to every step despite the imbalance.
    steps_per_epoch = max(len(unlabeled_loader), 1)

    for epoch in range(epochs):
        encoder.train(); decoder.train(); discriminator.train()
        totals = {"vae": 0.0, "disc": 0.0, "recon": 0.0, "kl": 0.0}

        unlabeled_stream = iter(unlabeled_loader)
        for _ in tqdm(range(steps_per_epoch),
                      desc=f"    VAAL epoch {epoch + 1}/{epochs}", leave=False):
            try:
                unlabeled_images, _, _ = next(unlabeled_stream)
            except StopIteration:
                break
            labeled_images, _, _ = next(labeled_stream)

            unlabeled_images = _downsample(unlabeled_images.to(device))
            labeled_images = _downsample(labeled_images.to(device))

            # --- VAE steps: reconstruct both sets, and push the
            # discriminator toward calling everything "labelled" ---
            for _ in range(VAE_STEPS_PER_ADV_STEP):
                mu_l, logvar_l = encoder(labeled_images)
                z_l = _reparameterise(mu_l, logvar_l)
                recon_l = decoder(z_l)

                mu_u, logvar_u = encoder(unlabeled_images)
                z_u = _reparameterise(mu_u, logvar_u)
                recon_u = decoder(z_u)

                loss_l, rec_l, kl_l = _vae_loss(recon_l, labeled_images, mu_l, logvar_l)
                loss_u, rec_u, kl_u = _vae_loss(recon_u, unlabeled_images, mu_u, logvar_u)

                # Adversarial term: the VAE wants BOTH sets to be called
                # labelled (target 1), so the latent space stops carrying
                # the labelled/unlabelled distinction.
                d_l = discriminator(z_l)
                d_u = discriminator(z_u)
                ones = torch.ones_like(d_l)
                adv = (F.binary_cross_entropy_with_logits(d_l, ones)
                       + F.binary_cross_entropy_with_logits(d_u, torch.ones_like(d_u)))

                loss_vae = loss_l + loss_u + ADVERSARY_WEIGHT * adv
                opt_vae.zero_grad(set_to_none=True)
                loss_vae.backward()
                opt_vae.step()

                totals["vae"] += loss_vae.item()
                totals["recon"] += rec_l + rec_u
                totals["kl"] += kl_l + kl_u

            # --- Discriminator step: separate the two sets again, on
            # latent codes detached from the VAE's graph ---
            with torch.no_grad():
                mu_l, logvar_l = encoder(labeled_images)
                z_l = _reparameterise(mu_l, logvar_l)
                mu_u, logvar_u = encoder(unlabeled_images)
                z_u = _reparameterise(mu_u, logvar_u)

            d_l = discriminator(z_l)
            d_u = discriminator(z_u)
            loss_disc = (
                F.binary_cross_entropy_with_logits(d_l, torch.ones_like(d_l))
                + F.binary_cross_entropy_with_logits(d_u, torch.zeros_like(d_u))
            )
            opt_disc.zero_grad(set_to_none=True)
            loss_disc.backward()
            opt_disc.step()
            totals["disc"] += loss_disc.item()

        if verbose:
            n = max(steps_per_epoch, 1)
            print(f"    VAAL epoch {epoch + 1}/{epochs}: "
                  f"vae={totals['vae'] / (n * VAE_STEPS_PER_ADV_STEP):.4f} "
                  f"disc={totals['disc'] / n:.4f} "
                  f"recon={totals['recon'] / (n * VAE_STEPS_PER_ADV_STEP):.4f}")

    # --- Score every unlabelled image, in dataset order ---
    encoder.eval(); discriminator.eval()
    score_loader = DataLoader(unlabeled_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    scores = []
    with torch.no_grad():
        for images, _, _ in tqdm(score_loader, desc="    VAAL scoring", leave=False):
            images = _downsample(images.to(device))
            mu, _ = encoder(images)
            # The mean of the posterior, not a sample — scoring must be
            # deterministic so the selection does not change if it is
            # recomputed.
            scores.append(torch.sigmoid(discriminator(mu)).cpu().numpy())

    return np.concatenate(scores, axis=0)


def select(k, labeled_dataset, unlabeled_dataset, device,
           epochs=5, batch_size=32, num_workers=2, rng=None, verbose=True):
    """
    Pick the k unlabelled images the discriminator is most confident are
    unlabelled.

    Returns
    -------
    np.ndarray of int — indices into the unlabelled pool.
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    n = len(unlabeled_dataset)
    k = int(min(k, n))
    if k <= 0:
        return np.array([], dtype=int)

    scores = train_and_score(
        labeled_dataset, unlabeled_dataset, device,
        epochs=epochs, batch_size=batch_size, num_workers=num_workers,
        rng=rng, verbose=verbose,
    )
    # Lowest P(labelled) first.
    chosen = np.argsort(scores)[:k]

    if verbose:
        print(f"    VAAL: selected {len(chosen)} of {n}; "
              f"P(labelled) of picks={scores[chosen].mean():.4f} "
              f"vs pool mean={scores.mean():.4f}")
    return chosen.astype(int)
