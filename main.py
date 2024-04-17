import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imagenet import get_dict
import random

from tf_attack import generate_universal_perturbation, generate_fgsm_perturbation, generate_iterative_perturbation
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from image_processing import ImageProcessor, get_mobilenetv2_classifier

from blender_render_1 import BlenderRenderer

from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

model, preprocess_input = get_mobilenetv2_classifier()

def mobilenet_v2_preprocess(image):
    image = tf.cast(image, tf.float32)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize_with_pad(image, 224, 224)
    image = preprocess_input(image)
    image = image[None, ...]
    return image

def eval(model, images, orig_label_name, target_label_name):
    # Evaluate the model on the adversarial images
    predictions = model.predict(images)
    org_indx = 0
    target_indx = 0

    target_label_name = target_label_name.replace(" ","")

    print("target label name: ")
    print(target_label_name)

    # Decode and print the predictions
    for pred in predictions:
        decoded_preds = decode_predictions((np.reshape(pred, (1,-1))), top= 1000)[0]

        for indx, d in enumerate(decoded_preds):
            if d[1] == orig_label_name:
                org_indx = indx
                
            if d[1].replace(" ","") == target_label_name:
                target_indx = indx


    return predictions, org_indx, target_indx

def saveimage(image,name):
    plt.figure()
    plt.imshow(image*0.5+0.5) # To change [-1, 1] to [0,1]
    plt.xticks([])
    plt.yticks([])
    plt.axis('off') 
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def save_eps_results(epsilon, train_data, test_data, target_label, perturbation_orig, wdp, attack_name, model, orig_label_name, orig_label_idx, resultsdf):

    target_label_name = get_dict()[target_label]
    
    print(f"Generating TRAIN adv {attack_name} images")
    for i, image in enumerate(train_data):
        if (perturbation_orig.shape[0]>1) and (perturbation_orig.shape[0]<101): #if stack of noises vs a single universal noise . NOT harded to 5 test images. 
            perturbation_origi = perturbation_orig[i]
        else: 
            perturbation_origi = perturbation_orig

            
        universal_perturbation = tf.clip_by_value(perturbation_origi, -1, 1)
        saveimage(universal_perturbation, name=f"{wdp}{orig_label_name}_{attack_name}_train_perturbation_eps{int(epsilon*1000):04d}_image{i}.jpg")

        adv_image = image + universal_perturbation
        adv_image = tf.clip_by_value(adv_image, -1, 1)
        saveimage(adv_image, name=f"{wdp}adv_{orig_label_name}_{attack_name}_img{i}_eps{int(epsilon*1000):04d}.jpg")
        
        adv_image = adv_image[None, ...]
        
        pr, org_indx, target_indx = eval(model, adv_image, orig_label_name,target_label_name)
        resdict = {'name' : f"{orig_label_name}_{attack_name}_train_img{i}_eps{int(epsilon*1000):04d}",
                'true label rank' :  org_indx,
                'target label rank' : target_indx,
                'true label pr' : pr[0][orig_label_idx],
                'target label pr' : pr[0][target_label]}
                #change above if using different target

        df1 = pd.DataFrame.from_dict(resdict,orient='index').T
        resultsdf = pd.concat([resultsdf, df1], join='outer')



    print(f"Generating TEST adv {attack_name} images")
    for i, image in enumerate(test_data):
        
        universal_perturbation = tf.clip_by_value(perturbation_origi, -1, 1)
        saveimage(universal_perturbation, name=f"{wdp}{orig_label_name}_{attack_name}_test_perturbation_eps{int(epsilon*1000):04d}_image{i}.jpg")
        

        adv_image = image + universal_perturbation
        adv_image = tf.clip_by_value(adv_image, -1, 1)
        saveimage(adv_image, name=f"{wdp}adv_{orig_label_name}_{attack_name}_test_img{i}_eps{int(epsilon*1000):04d}.jpg")
        print(f"adv_{orig_label_name}_{attack_name}_img{i}_eps{int(epsilon*1000):04d}")
        adv_image = adv_image[None, ...]
    
        pr, org_indx, target_indx = eval(model, adv_image, orig_label_name, target_label_name)
        resdict = {'name' : f"{orig_label_name}_{attack_name}_test_img{i}_eps{int(epsilon*1000):04d}",
                'true label rank' :  org_indx,
                'target label rank' : target_indx,
                'true label pr' : pr[0][orig_label_idx],
                'target label pr' : pr[0][target_label]}
        df1 = pd.DataFrame.from_dict(resdict,orient='index').T
        resultsdf = pd.concat([resultsdf, df1], join='outer')

    return resultsdf

def train_split(data, num_test_images):
    # Indices for testing and training images
    num_images = tf.shape(data)[0]
    all_indices = tf.range(num_images)
    test_indices = tf.random.shuffle(all_indices)[:num_test_images]

    # Create a boolean mask for training indices
    train_mask = tf.reduce_all(tf.math.not_equal(all_indices[:, tf.newaxis], test_indices), axis=1)

    # Apply the mask to get training indices
    train_indices = tf.boolean_mask(all_indices, train_mask)
    # Split the data into training and testing sets
    train_data = tf.gather(data, train_indices)
    test_data = tf.gather(data, test_indices)

    # Print the shapes of the resulting sets
    print("Training set shape:", train_data.shape)
    print("Testing set shape:", test_data.shape) 

    return train_data, test_data

def load_and_split_imgs(path, nrenders):
    img_paths = []
    for i in range(nrenders):
        img_paths.append(path+f'results_angle_{i}.jpg')

    images = [preprocess_input(keras_image.img_to_array(keras_image.load_img(img_path, target_size=(224, 224)))) for img_path in img_paths]
    images = tf.convert_to_tensor(np.array(images))

    num_test_images = nrenders//2
    train_data, test_data = train_split(images, num_test_images)

    return train_data, test_data

if __name__ == "__main__":
    nrenders = 10

    path = '/Users/chrisgreen/Desktop/UniversalPerturbation/results/'

    #lemon 'highpoly'

    blender_renderer = BlenderRenderer(
    blend_file_path = '/Users/chrisgreen/Desktop/UniversalPerturbation/3DRendering/render/broccoli.blend',
    texture_path = '/Users/chrisgreen/Desktop/UniversalPerturbation/3DRendering/textures/broccoli_text.png',
    model_name = 'Sketchfab_model',
    camera_distance = 1.0,
    output_folder = '/Users/chrisgreen/Desktop/UniversalPerturbation/results/',
    angle_number = nrenders,
    )

    blender_renderer.render()
    
    eps = [0.50, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01, 0.005, 0.0]

    orig_label_name = 'broccoli'
    orig_label_idx = 937

    train_data, test_data = load_and_split_imgs(path, nrenders)
    
    targeted = False
    #target_label = random.randint(0,999)
    target_label = 937
    print(f"TARGET LABEL: {target_label}")

    

    resultsdf = pd.DataFrame(columns = ['name',
                    'true label rank',
                    'target label rank',
                    'true label pr',
                    'target label pr'])
    
    num_iterations = 20

    

    for e in eps:
        fgsm_perturbation,loss = generate_fgsm_perturbation(train_data, target_model=model, input_label=target_label, epsilon=e, targeted=targeted)
        resultsdf = save_eps_results(e, train_data, test_data, target_label, 
                        fgsm_perturbation, path, 'fgsm', model, 
                        orig_label_name, orig_label_idx, resultsdf)
        
    for e in eps:
        iterative_perturbation = generate_iterative_perturbation(train_data, target_model=model, input_label=target_label, num_iterations=num_iterations,
                                                                 epsilon=e, targeted=targeted)
        resultsdf = save_eps_results(e, train_data, test_data, target_label, 
                        iterative_perturbation, path, f"iterative_attack_n{int(num_iterations):03d}", model, 
                        orig_label_name, orig_label_idx, resultsdf)
    
    for e in eps:
        universal_perturbation = generate_universal_perturbation(train_data, target_model=model, input_label=target_label, epsilon=e,
                                                                 num_iterations=num_iterations, targeted=targeted)
        resultsdf = save_eps_results(e, train_data, test_data, target_label, 
                        universal_perturbation, path, f"universal_n{int(num_iterations):03d}", model, 
                        orig_label_name, orig_label_idx, resultsdf)
        
    resultsdf.to_csv(f"{path}results_{orig_label_name}.csv")


    



