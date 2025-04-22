import torch
import kornia
from numpy.random import choice
import math
import matplotlib.pyplot as plt
import numpy as np

############################## Helper function to add Gaussian blur to images ##############################

def add_blur_with(images, sigmas, weights):
    blurred_images = torch.zeros_like(images)
    for i in range(images.size(0)):
        image = images[i, :, :, :]

        sigma = choice(sigmas, 1, p=weights)[0]
        kernel_size = 2 * math.ceil(2.0 * sigma) + 1

        if sigma == 0:
            blurred_image = image
        else:
            blurred_image = kornia.filters.gaussian_blur2d(torch.unsqueeze(image, dim=0), kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[0, :, :, :]
        blurred_images[i] = blurred_image

    return blurred_images
    
######################################### Checkpoint Key Remapping #########################################
    
def convert_checkpoint_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.") and not k.startswith("model.fc"):
            new_key = k.replace("model.", "backbone.model.", 1)
        elif k.startswith("model.fc"):
            new_key = k.replace("model.fc", "classifier_head", 1)
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

############################  Helper functions for plotting spatial frequency  ###################################

def generate_grating(
    image_size: int,
    orientation: float,
    spatial_freq: float,
    phase: float,
    device: torch.device
) -> torch.Tensor:
    """
    Create a single-channel sinusoidal grating (1x1xHxW).
    orientation, spatial_freq, phase in degrees/cycles as described.
    """
    # Meshgrid from -1..1
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, steps=image_size),
        torch.linspace(-1, 1, steps=image_size),
        indexing='xy'
    )
    theta = np.deg2rad(orientation)
    phase_rad = np.deg2rad(phase)
    
    # Rotate coordinates
    x_rot = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
    
    # If spatial_freq = # cycles across [-1..1], wave_number = spatial_freq * π
    wave_number = spatial_freq * np.pi
    grating = torch.sin(wave_number * x_rot + phase_rad)
    
    # Reshape to (1, 1, H, W)
    grating = grating.unsqueeze(0).unsqueeze(0).to(device)
    grating = grating.repeat(1, 3, 1, 1)
    return grating

def get_feature_map_responses(model, layers_dict, input_batch):
    """
    Registers hooks on specified layers, runs a forward pass,
    returns {layer_name: [N, num_features, H_out, W_out]}.
    """
    responses = {}
    
    def hook_fn(name):
        def fn(_, __, output):
            responses[name] = output.detach()
        return fn
    
    hooks = []
    for lname, layer_module in layers_dict.items():
        h = layer_module.register_forward_hook(hook_fn(lname))
        hooks.append(h)
    
    with torch.no_grad():
        _ = model(input_batch)
    
    for h in hooks:
        h.remove()
    
    return responses

def measure_spatial_frequencies(model, device):
    """
    1) Generate a set of gratings (15 orientations x 25 spatial frequencies x 4 phases).
    2) For each grating, capture the average response in each chosen convolutional layer.
    3) Compute a tuning curve over spatial frequency (averaging across orientation & phase).
    4) Return a dictionary: {layer_name: np.array of shape [num_features] with peak spatial frequencies}.
    """
    # Define the layers to hook.
    layers_dict = {
        "features.0": model.backbone.feature_extractor[0],
        "features.3": model.backbone.feature_extractor[3],
        "features.6": model.backbone.feature_extractor[6],
        "features.8": model.backbone.feature_extractor[8],
        "features.10": model.backbone.feature_extractor[10],
    }
    
    # Parameter space.
    orientations = [i * 12 for i in range(15)]  # 0, 12, ..., 168 degrees.
    spatial_freqs = np.linspace(4.48, 112, 25)     # 25 frequencies.
    phases = [0, 90, 180, 270]
    image_size = 224  # Image resolution.
    
    # Accumulate responses in a dictionary: for each layer, a tensor of shape [ori, sf, phase, num_features].
    layer_responses = {lname: None for lname in layers_dict}
    
    # Iterate over all grating parameters.
    for i, ori in enumerate(orientations):
        for j, sf in enumerate(spatial_freqs):
            for k, ph in enumerate(phases):
                # Generate the grating stimulus.
                stim = generate_grating(image_size, ori, sf, ph, device)
                
                # Run a forward pass and capture responses.
                resp_dict = get_feature_map_responses(model, layers_dict, stim)
                
                # For each layer, average over spatial dimensions to get a response per feature.
                for lname, fmaps in resp_dict.items():
                    # fmaps: [1, num_features, H_out, W_out].
                    avg_resp = fmaps.mean(dim=[0, 2, 3])  # Shape: [num_features].
                    
                    # Allocate storage if needed.
                    if layer_responses[lname] is None:
                        num_features = avg_resp.shape[0]
                        layer_responses[lname] = torch.zeros(
                            len(orientations),
                            len(spatial_freqs),
                            len(phases),
                            num_features,
                            device=device
                        )
                    layer_responses[lname][i, j, k, :] = avg_resp
    
    # Compute the peak spatial frequency for each feature map.
    peak_sf_dict = {}
    for lname, tensor4d in layer_responses.items():
        # tensor4d shape: [ori, sf, phase, feature].
        # Average responses across orientation and phase => shape: [sf, feature].
        data_mean = tensor4d.mean(dim=(0, 2))  # Now shape: [sf, feature].
        data_mean = data_mean.cpu().numpy()     # Convert to NumPy array.
        
        num_sf, num_features = data_mean.shape
        peaks = np.zeros(num_features)
        
        for f in range(num_features):
            curve = data_mean[:, f]
            cmin, cmax = curve.min(), curve.max()
            if abs(cmax - cmin) < 1e-9:
                # If responses are nearly constant, assign the lowest frequency.
                peaks[f] = spatial_freqs[0]
            else:
                norm_curve = (curve - cmin) / (cmax - cmin)
                idx_peak = np.argmax(norm_curve)
                peaks[f] = spatial_freqs[idx_peak]
        
        peak_sf_dict[lname] = peaks
    return peak_sf_dict