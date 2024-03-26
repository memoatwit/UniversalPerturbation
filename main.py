import tensorflow as tf
import numpy as np

from tf_attack import generate_universal_perturbation
from image_processing import ImageProcessor, get_mobilenetv2_classifier

model, preprocess_input = get_mobilenetv2_classifier()

def mobilenet_v2_preprocess(image):
    image = tf.cast(image, tf.float32)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize_with_pad(image, 224, 224)
    image = preprocess_input(image)
    image = image[None, ...]
    return image

if __name__ == "__main__":
    # img_processor = ImageProcessor(mobilenet_v2_preprocess)
    # image = img_processor.load_image("test_image/desk.jpg", numpy_return=True)
    # image = tf.convert_to_tensor(image, tf.float32)
    # images = [image]*10
    image = np.random.rand(224,224,3) * 255
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    images = [image] * 10
    fgsm_noise = generate_universal_perturbation(images, model)
