import tensorflow as tf
import numpy as np


# Function to generate a FGSM perturbations
# https://www.tensorflow.org/tutorials/generative/adversarial_fgsm
def generate_fgsm_perturbation(images, target_model, input_label, clip_value=1.0):

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    perturbation = np.zeros_like(images)

    OHlabel = tf.one_hot(input_label, 1_000)
    OHlabel = tf.reshape(OHlabel, (1, 1_000))

    for idimage, image in enumerate(images):
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = target_model(tf.expand_dims(image, 0)) # Expects (None, 224, 224, 3)
            loss = loss_object(OHlabel, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient) #this is just a +/-1 matrix
        signed_grad = tf.Variable(tf.clip_by_value(signed_grad, -clip_value, clip_value))
        signed_grad = tf.stop_gradient(signed_grad)
        # print(f"image: {idimage}, loss: {loss}")
        perturbation[idimage] = (signed_grad.numpy())

    return perturbation, loss

# Functions to generate a basic iterative perturbations
# (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf

####
def _single_iterative_perturbation(image, target_model, epsilon=1., num_iterations=100, regularization=1., clip_value=1.0):
    #expects a single image to attack
    # pinit = tf.random.uniform(shape=image.shape, minval=-0.1, maxval=0.1)
    pinit = tf.zeros_like(input=image)
    perturbation = tf.Variable(pinit, dtype=tf.float32, trainable=True)

    for it in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(image)
            adv_images = image + perturbation #(224, 224, 3))
            adv_images = tf.expand_dims(adv_images, 0) # Expects (None, 224, 224, 3)
            predictions = target_model(adv_images) # predict on stack of images: (5,1000)
            loss = tf.keras.losses.categorical_crossentropy(tf.one_hot(tf.argmax(predictions, axis=1), 1_000), predictions)
            #loss is 1 - pred(target). so if pred(lemon)=0.9->loss=0.1
            # loss += regularization * tf.norm(perturbation) #keep track of loss on all stack of images 
            if it%10 == 0:
                print(f"iteration: {it}, loss: {loss}")
        
            gradients = tape.gradient(loss, image) #watch gradients on image            
            # Apply gradient clipping
            grads, _ = tf.clip_by_global_norm([gradients], clip_value)
            signed_grad = tf.sign(grads[0])
            signed_grad = tf.stop_gradient(signed_grad)
            perturbation.assign_add(epsilon * signed_grad)
            # perturbation.assign_add(epsilon * tf.sign(grads[0]))
            perturbation = tf.Variable(tf.clip_by_value(perturbation, -clip_value, clip_value))
    return perturbation.numpy()
####
def generate_iterative_perturbation(images, target_model, input_label, num_iterations=100, regularization=1., clip_value=1.0, epsilon=1.0):
    
    perturbations = np.zeros_like(images)

    for idimage, image in enumerate(images): #for each image
        print(f"*** Attacking image: {idimage}")
        perturbations[idimage] = _single_iterative_perturbation(image, target_model, epsilon, num_iterations=num_iterations, regularization=regularization, clip_value=clip_value)
    
    return perturbations



# Function to generate a universal perturbation
def generate_universal_perturbation(images, target_model, epsilon=1., num_iterations=1, regularization=1., clip_value=1.0):
    # perturbation = tf.Variable(np.zeros_like(images[0]), dtype=tf.float32, trainable=True)
    pinit = tf.random.uniform(shape=images[0].shape, minval=-0.01, maxval=0.01)
    perturbation = tf.Variable(pinit, dtype=tf.float32, trainable=True)
    #single image size: ([224, 224, 3])

    for it in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(perturbation)
            adv_images = images + perturbation
            predictions = target_model(adv_images) # predict on stack of images: (5,1000)
            loss = tf.keras.losses.categorical_crossentropy(tf.one_hot(tf.argmax(predictions, axis=1), 1_000), predictions)
            #loss is 1 - pred(target). so if pred(lemon)=0.9->loss=0.1
            # loss += regularization * tf.norm(perturbation) #keep track of loss on all stack of images 
            if it%10 == 0:
                print(f"iteration: {it}, loss: {loss}")

            print(loss.shape, perturbation.shape)
            gradients = tape.gradient(loss, perturbation) #trick is here: watch gradients on pert, not image! 
            # Apply gradient clipping
            grads, _ = tf.clip_by_global_norm([gradients], clip_value)
            signed_grad = tf.sign(grads[0])
            signed_grad = tf.stop_gradient(signed_grad)
            perturbation.assign_add(epsilon * signed_grad)
            # perturbation.assign_add(epsilon * tf.sign(grads[0]))
            perturbation = tf.Variable(tf.clip_by_value(perturbation, -clip_value, clip_value))
    return perturbation.numpy()