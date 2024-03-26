from PIL import Image
import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from torch import clip
import torch

from blender_render import BlenderRenderer

class ImageProcessor():
    def __init__(self, preprocesser, renderer=None):
        self.preprocesser = preprocesser
        self.renderer = renderer
    
    # load and preprocess
    def load_image(self, image_path, image_size = (224,224),  numpy_return = False):
        image = Image.open(image_path).convert('RGB').resize(image_size)
        
        image = self.preprocesser(image)

        if numpy_return:
            return np.array(image, dtype=np.float32)
        
        return image
    
    def render(self):
        if self.renderer:
            self.renderer.render()
        else:
            print("No renderer was defined, please set the renderer parameter before call")

    def mergenoise(self, texture_path, advnoise_path, results):
        result_paths = []

        # texture 
        image1_path = texture_path
        image1 = Image.open(image1_path)
        image1 = image1.convert("RGB")
        image1_resized = image1.resize((224, 224))

        # noise
        image2_path = advnoise_path
        image2 = Image.open(image2_path)
        image2 = image2.convert("RGB")
        image2 = image2.resize((224, 224))

        # Create a new image by combining the resized images
        # Iterate over different alpha values: from even 50%, down to 1%
        alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

        for alpha in alpha_values:
            combined_image = Image.blend(image1_resized, image2, alpha=alpha)
            result_path = f"{results}noisy_texture_{int(alpha * 100):02d}.jpg"
            combined_image.save(result_path)
            print(f"{alpha=} texture saved: {result_path}")
            result_paths.append(result_path)

        return result_paths

def get_mobilenetv2_classifier():
    return MobileNetV2(weights='imagenet'), preprocess_input

def get_clip_classifier(weights="RN50"):
    """
    Args:
        weight (str, optional): model weights to load. Defaults to "RN50".

    Returns:
        torch model, preprocess class
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return clip.load(weights, device, jit=False)

def get_renderer(blend_file_path, texture_path, model_name, output_folder, camera_distance, angle_number):
    return BlenderRenderer(blend_file_path, texture_path, model_name, output_folder, camera_distance, angle_number)
