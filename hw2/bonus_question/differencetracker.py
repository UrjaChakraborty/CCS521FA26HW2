
import torch
import torch.nn.functional as F


class DifferenceTracker:


    def __init__(self, model, epsilon):
        self.model = model
        self.epsilon = epsilon

    def _check_inputs(self, x1, x2, v):
        if x1.shape != x2.shape or x1.shape != v.shape:
            raise ValueError("x1, x2, and v must have the same shape")

        if v.abs().max().item() > self.epsilon + 1e-7:
            raise ValueError("Perturbation exceeds the L-infinity budget")

    @torch.no_grad()
    def track_encoder(self, x1, x2, v):
        self._check_inputs(x1, x2, v)

        # Apply exactly the same perturbation to both inputs.
        a1 = x1 + v
        a2 = x2 + v

        result = {
            "input_difference": a1 - a2,
        }

        # Track differences through matching encoder layers.
        h1 = F.relu(self.model.encoder_fc1(a1))
        h2 = F.relu(self.model.encoder_fc1(a2))
        result["encoder_relu1_difference"] = h1 - h2

        h1 = F.relu(self.model.encoder_fc2(h1))
        h2 = F.relu(self.model.encoder_fc2(h2))
        result["encoder_relu2_difference"] = h1 - h2

        mu1 = self.model.encoder_mu(h1)
        mu2 = self.model.encoder_mu(h2)
        logvar1 = self.model.encoder_logvar(h1)
        logvar2 = self.model.encoder_logvar(h2)

        result["mu_difference"] = mu1 - mu2
        result["logvar_difference"] = logvar1 - logvar2

        result["mu1"] = mu1
        result["mu2"] = mu2
        result["logvar1"] = logvar1
        result["logvar2"] = logvar2

        return result

    @torch.no_grad()
    def track_decoder(self, z1, z2):
        if z1.shape != z2.shape:
            raise ValueError("z1 and z2 must have the same shape")

        result = {"latent_difference": z1 - z2}

        h1 = F.relu(self.model.decoder_fc1(z1))
        h2 = F.relu(self.model.decoder_fc1(z2))
        result["decoder_relu1_difference"] = h1 - h2

        h1 = F.relu(self.model.decoder_fc2(h1))
        h2 = F.relu(self.model.decoder_fc2(h2))
        result["decoder_relu2_difference"] = h1 - h2

        y1 = self.model.decoder_out(h1)
        y2 = self.model.decoder_out(h2)
        result["output_difference"] = y1 - y2

        return result


if __name__ == "__main__":
    from vae_model import VAE
    from supportset import latent_support

    torch.manual_seed(42)

    model = VAE()
    model.eval()

    epsilon = 0.05
    tracker = DifferenceTracker(model, epsilon)

    x1 = torch.rand(1, 784)
    x2 = torch.rand(1, 784)

    # The same bounded perturbation is used for both inputs.
    v = torch.empty_like(x1).uniform_(-epsilon, epsilon)

    encoder = tracker.track_encoder(x1, x2, v)

    # Demonstrate decoding two points from the support boxes.
    lo1, hi1 = latent_support(encoder["mu1"], encoder["logvar1"])
    lo2, hi2 = latent_support(encoder["mu2"], encoder["logvar2"])

    z1 = (lo1 + hi1) / 2.0
    z2 = (lo2 + hi2) / 2.0

    decoder = tracker.track_decoder(z1, z2)

    print("Maximum absolute input difference:",
          encoder["input_difference"].abs().max().item())
    print("Maximum absolute mean difference:",
          encoder["mu_difference"].abs().max().item())
    print("Maximum absolute decoder output difference:",
          decoder["output_difference"].abs().max().item())
    print("Shared perturbation is within budget:",
          v.abs().max().item() <= epsilon + 1e-7)
