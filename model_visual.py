from typing import List, Tuple

import torch

import utilities

class Model:
    def __init__(
        self, path: str, device: torch.device,
        target_image: torch.Tensor,
        visualizing_layer = 'pool2',
        visualizing_mask_mode = 'channel-0'
    ):  
        self.target_image = target_image.to(device)
        self.net = utilities.load_model(path).to(device).eval()
        self.device = device
        self.visualizing_layer = visualizing_layer

        # register Visualization loss hook
        self.visual_loss_hook = VisualLossHook(visualizing_mask_mode)

        for name, layer in self.net.named_children():
            if name == self.visualizing_layer:
                handle = layer.register_forward_hook(self.visual_loss_hook)

        # remove unnecessary layers
        i = 0
        for name, layer in self.net.named_children():
            if name == visualizing_layer:
                break
            i += 1

        self.net = self.net[:(i + 1)]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.visual_loss_hook.clear()

        return self.net(image)

    def get_loss(self) -> torch.Tensor:
        # return sum(self.gram_loss_hook.losses)
          return torch.stack(self.visual_loss_hook.losses, dim=0).sum(dim=0)


# Visualization loss hook
class VisualLossHook:
  def __init__(self, mask_mode):
    self.mask_mode = mask_mode
    self.losses: List[torch.Tensor] = []
  
  def __call__(self, layer: torch.nn.Module, layer_in: Tuple[torch.Tensor],
      layer_out: torch.Tensor):
    # compute visual loss
    mask = self.get_mask(layer_out)
    loss = - (torch.clamp(mask * layer_out, 0, 1e+5)).mean() # maximize the layer output
    self.losses.append(loss)

  def clear(self):
    self.losses = []
    
  def get_mask(self, input_tensor):
    if self.mask_mode[:7] == 'channel':
      channel_index = int(self.mask_mode.split("-")[-1])
      mask = torch.zeros_like(input_tensor).to(input_tensor.device)
      # print(mask.shape)
      mask[:, channel_index, :, :] = 1.
    elif self.mask_mode[:6] == 'neuron':
      _, c, h, w = self.mask_mode.split("-")
      c, h, w = int(c), int(h), int(w)
      mask = torch.zeros_like(input_tensor).to(input_tensor.device)
      mask[:, c, h, w] = 1.
    else:
      raise NotImplementedError
    return mask

