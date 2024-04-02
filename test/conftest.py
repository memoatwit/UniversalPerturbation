import pytest
import numpy as np
import tensorflow as tf

from image_processing import get_mobilenetv2_classifier

@pytest.fixture(scope="module")
def test_image_tf() -> tf.Tensor:
    image = np.random.rand(224,224,3) * 255
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image[None, ...]

@pytest.fixture(scope="module")
def test_images_tf(test_image_tf) -> list[tf.Tensor]:
    return [test_image_tf] * 10
    
@pytest.fixture(scope="module")
def tf_classifier():
    model, _ = get_mobilenetv2_classifier()
    return model

@pytest.fixture(scope="module", params=[0, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.3, 0.5])
def eps(epsilons):
    return epsilons.param