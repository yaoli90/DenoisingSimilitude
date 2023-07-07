import torch
import torch.nn as nn
import numpy as np

class denoising_PGD:
    def __init__(self, model, eps=16/255, alpha=2/255, steps=40, random_start=True, loss=nn.MSELoss()):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss = loss
        self.device = next(model.parameters()).device
    def attack(self, images):

        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = adv_images - self.model(adv_images)

            cost = -self.loss(outputs, images)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
