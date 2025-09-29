# model_autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, latent_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_dim)
        return x


class Decoder(nn.Module):
    """
    A simpler, more direct decoder. This is less prone to model collapse.
    It maps the latent vector directly to a 3D point cloud.
    """

    def __init__(self, latent_dim=256, num_points=2048):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_points * 3)  # Output coordinates for all points
        )

    def forward(self, x):
        """
        Forward pass of the decoder.
        Args:
            x (torch.Tensor): Latent vector, shape (B, latent_dim)
        Returns:
            torch.Tensor: Reconstructed point cloud, shape (B, 3, N)
        """
        B = x.shape[0]

        # Pass the latent vector through the MLP
        point_coords = self.mlp(x)

        # Reshape the output to be a point cloud
        # from (B, num_points * 3) -> (B, num_points, 3)
        reconstructed_points = point_coords.view(B, self.num_points, 3)

        # Transpose to match the expected output shape (B, 3, N)
        return reconstructed_points.transpose(1, 2)


class PointAutoencoder(nn.Module):
    """
    The complete autoencoder model using the new decoder.
    """

    def __init__(self, latent_dim=256, num_points=2048):
        super(PointAutoencoder, self).__init__()
        self.encoder = PointNetEncoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, num_points=num_points)
        self.num_points = num_points

    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed_points = self.decoder(latent_vector)
        return reconstructed_points