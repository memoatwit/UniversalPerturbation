import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as plt

def tf_train_split(data, num_test_images):
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


def saveimage(image,name):
    plt.figure()
    plt.imshow(image*0.5+0.5) # To change [-1, 1] to [0,1]
    plt.xticks([])
    plt.yticks([])
    plt.axis('off') 
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def printinfo(array):
    print(f"shape: {array.shape}, min: {np.min(array)}, max: {np.max(array)}")

def save_eps_results(epsilons, train_data, test_data, perturbation_orig, wdp, attack_name, model, orig_label_name, orig_label_idx, resultsdf):
    
    print(f"Generating TRAIN adv {attack_name} images")
    for i, image in enumerate(train_data):
        if (perturbation_orig.shape[0]>1) and (perturbation_orig.shape[0]<101): #if stack of noises vs a single universal noise . NOT harded to 5 test images. 
            perturbation_origi = perturbation_orig[i]
        else: 
            perturbation_origi = perturbation_orig

        for e, eps in enumerate(epsilons):
            universal_perturbation = (perturbation_origi/np.max(abs(perturbation_origi))*eps)
            universal_perturbation = tf.clip_by_value(universal_perturbation, -1, 1)
            saveimage(universal_perturbation, name=f"{wdp}{orig_label_name}_{attack_name}_train_perturbation_eps{int(eps*1000):04d}.jpg")
            printinfo(universal_perturbation)

            adv_image = image + universal_perturbation
            adv_image = tf.clip_by_value(adv_image, -1, 1)
            saveimage(adv_image, name=f"{wdp}adv_{orig_label_name}_{attack_name}_img{i}_eps{int(eps*1000):04d}.jpg")
            print(f"adv_universal_train_img{i}_eps{int(eps*1000):04d}")
            adv_image = adv_image[None, ...]
            printinfo(adv_image)
            pr, top1, top5 = eval(model, adv_image, orig_label_name)
            resdict = {'name' : f"{orig_label_name}_{attack_name}_train_img{i}_eps{int(eps*1000):04d}",
                    'top1' :  top1,
                    'top5' : top5,
                    'pr' : pr[0][orig_label_idx]}
            df1 = pd.DataFrame.from_dict(resdict,orient='index').T
            resultsdf = pd.concat([resultsdf, df1], join='outer')

    print(f"Generating TEST adv {attack_name} images")
    for i, image in enumerate(test_data):
        for e, eps in enumerate(epsilons):
            universal_perturbation = (perturbation_origi/np.max(abs(perturbation_origi))*eps)
            universal_perturbation = tf.clip_by_value(universal_perturbation, -1, 1)
            saveimage(universal_perturbation, name=f"{wdp}{orig_label_name}_{attack_name}_test_perturbation_eps{int(eps*1000):04d}.jpg")
            printinfo(universal_perturbation)

            adv_image = image + universal_perturbation
            adv_image = tf.clip_by_value(adv_image, -1, 1)
            saveimage(adv_image, name=f"{wdp}adv_{orig_label_name}_{attack_name}_test_img{i}_eps{int(eps*1000):04d}.jpg")
            print(f"adv_{orig_label_name}_{attack_name}_img{i}_eps{int(eps*1000):04d}")
            adv_image = adv_image[None, ...]
            printinfo(adv_image)
            pr, top1, top5 = eval(model, adv_image, orig_label_name)
            resdict = {'name' : f"{orig_label_name}_{attack_name}_test_img{i}_eps{int(eps*1000):04d}",
                    'top1' :  top1,
                    'top5' : top5,
                    'pr' : pr[0][orig_label_idx]}
            df1 = pd.DataFrame.from_dict(resdict,orient='index').T
            resultsdf = pd.concat([resultsdf, df1], join='outer')

    return resultsdf