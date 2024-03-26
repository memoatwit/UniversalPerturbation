import torch
import numpy as np

from torch.nn import functional

def generate_universal_pertubation(image, text, clip_model, epsilon=1., num_iterations=1, regularization=1., clip_value=1.0):
    image = torch.unsqueeze(image, 0)

    pinit = np.random.uniform(-0.01, 0.01, size=image.shape)
    pertubation = torch.from_numpy(pinit)

    pertubation.requires_grad = True

    for it in range(num_iterations):
        prediction = clip_model(image, text)
        loss = functional.cross_entropy(torch.max(prediction).unsqueeze(0), prediction)
        loss.backward()
        if it%10 == 0:
            print(f"iteration: {it}, loss: {loss.data.numpy()}")
        signed_grad = torch.sign(pertubation.grad.data)
        pertubation.add(signed_grad * epsilon)
        pertubation = torch.clamp(pertubation, -clip_value, clip_value)

    return pertubation.squeeze(0).numpy()


