import tensorflow as tf
import numpy as np

from tf_attack import generate_universal_perturbation, generate_fgsm_perturbation
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from image_processing import ImageProcessor, get_mobilenetv2_classifier

from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

model, preprocess_input = get_mobilenetv2_classifier()

def mobilenet_v2_preprocess(image):
    image = tf.cast(image, tf.float32)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize_with_pad(image, 224, 224)
    image = preprocess_input(image)
    image = image[None, ...]
    return image

def eval(model, images, orig_label_name):
    # Evaluate the model on the adversarial images
    predictions = model.predict(images)

    # Decode and print the predictions
    for pred in predictions:
        decoded_preds = decode_predictions((np.reshape(pred, (1,-1))))[0]
        print(decoded_preds)
        found1 = decoded_preds[0][1] == orig_label_name
        print(f"top1: {found1}")
        # print(f"top5: {decoded_preds[0][1] == orig_label_name}")
        found5 = False
        for d in decoded_preds:
            if d[1] == orig_label_name:
                found5 = True
                break
        print(f"top5: {found5}")               
        # orig_label_name
    return predictions, found1, found5

if __name__ == "__main__":
    img_processor = ImageProcessor(mobilenet_v2_preprocess)
    image = img_processor.load_image("desk.jpg", numpy_return=True)
    image = tf.convert_to_tensor(image, tf.float32)
    # eps = [0, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.3, 0.5]

    # image = np.random.rand(224,224,3) * 255
    # image = tf.convert_to_tensor(image, dtype=tf.float32)
    # images = [image]

    universal_noise = generate_universal_perturbation(image, model)
    fgsm_noise, _ = generate_fgsm_perturbation(image, model, input_label=532)

    eps = 0.05

    universal_noise_image = np.asarray(image) + universal_noise * eps
    fgsm_noise_image = np.asarray(image) + fgsm_noise * eps
    ch_fgsm = fast_gradient_method(model, image, eps=eps, norm = np.inf)

    universal_preds = decode_predictions(model(universal_noise_image))
    fgsm_preds = decode_predictions(model(fgsm_noise_image))
    ch_fgsm_preds = decode_predictions(model(ch_fgsm))

    print(f'Universal: {universal_preds}')
    print(f'FGSM: {fgsm_preds}')
    print(f'Cleverhans FGSM: {ch_fgsm_preds}')

    



