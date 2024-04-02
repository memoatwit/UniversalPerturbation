import numpy as np

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method as tf_fgsm
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method as tf_bim
from tf_attack import generate_fgsm_perturbation, generate_iterative_perturbation, generate_universal_perturbation
       
def test_cleverhans_tf_attacks(test_image_tf, tf_classifier):
    fgsm_noise = tf_fgsm(tf_classifier, test_image_tf, eps=1, norm = np.inf)
    bim_noise = tf_bim(tf_classifier, test_image_tf, eps=1, eps_iter=1, nb_iter=1, norm = np.inf)

    assert bim_noise is not None
    assert fgsm_noise is not None

def test_custom_tf_attacks(test_images_tf, tf_classifier):
    fgsm_noise = generate_fgsm_perturbation(test_images_tf, tf_classifier, input_label=1)
    bim_noise = generate_iterative_perturbation(test_images_tf, tf_classifier, num_iterations=1)
    universal_noise = generate_universal_perturbation(test_images_tf, tf_classifier)

    assert fgsm_noise is not None
    assert universal_noise is not None


    
