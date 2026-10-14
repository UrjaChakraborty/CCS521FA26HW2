
import torch


def support_radius(latent_dim, probability=0.95):
    # support_radius computes the radius of a hypercube in the latent space that contains a specified probability mass for a diagonal Gaussian distribution. The function takes two parameters: latent_dim, which is the dimensionality of the latent space, and probability, which is the desired probability mass to be contained within the hypercube. The default value for probability is set to 0.95 (95%).
    # The function first checks if the latent_dim is positive and if the probability is between
    if latent_dim < 1:
        raise ValueError("latent_dim must be positive")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")
# The function then calculates the coordinate probability by taking the nth root of the specified probability, where n is the latent dimension. This is done to find the probability mass along each individual dimension of the latent space. The upper cumulative distribution function (CDF) value is computed as (1 + coordinate_probability) / 2, which represents the upper bound of the CDF for a standard normal distribution.
    coordinate_probability = probability ** (1.0 / latent_dim)
    upper_cdf = (1.0 + coordinate_probability) / 2.0

    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )

    return normal.icdf(
        torch.tensor(upper_cdf, dtype=torch.float64)
    ).item()


def latent_support(mu, logvar, probability=0.95):
    """
    Return lower and upper bounds for a diagonal Gaussian latent
    distribution parameterized by mu and logvar.
    """
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape")

    c = support_radius(mu.shape[-1], probability)
    std = torch.exp(0.5 * logvar)

    lower = mu - c * std
    upper = mu + c * std

    return lower, upper


if __name__ == "__main__":
    mu = torch.zeros(2, 8)
    logvar = torch.zeros(2, 8)

    lower, upper = latent_support(mu, logvar)

    print("Target joint probability: 95%")
    print("Latent dimension:", mu.shape[-1])
    print("Support radius c:", round(support_radius(8), 4))
    print("Lower support shape:", tuple(lower.shape))
    print("Upper support shape:", tuple(upper.shape))
    print("First interval:",
          (round(lower[0, 0].item(), 4),
           round(upper[0, 0].item(), 4)))
