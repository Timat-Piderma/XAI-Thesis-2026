import tensorflow as tf
from matplotlib import pyplot as plt
import keras
import numpy as np
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Callback
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from pathlib import Path
script_folder = Path(__file__).parent

class PrintProgress(Callback):
    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            step = args[0]
        else:
            step = kwargs.get('i', kwargs.get('step', 0))
            
        if step % 20 == 0:
            print(f"Step {step}/200...")

# Loading model
model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
model.summary()

activation_maximization = ActivationMaximization(model, 
                                                 model_modifier=[ReplaceToLinear()], 
                                                 clone=True)

# 'Goldfish' = 1, 'Bear' = 294, 'Assault Rifle' = 413
score = CategoricalScore(1)

# Generate image
activations = activation_maximization(score, steps=200, callbacks=[PrintProgress()])

img = activations[0]
img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)

# Show and save image
plt.imshow(img_normalized)
plt.title("Ideal image for class 'Goldfish'")
plt.axis('off')
plt.savefig(script_folder / 'images/filter-vis-VGG16-goldfish.png')


# 'Goldfish' = 1, 'Bear' = 294, 'Assault Rifle' = 413
score = CategoricalScore(294)

# Generate image
activations = activation_maximization(score, steps=200, callbacks=[PrintProgress()])

img = activations[0]
img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)

# Show and save image
plt.imshow(img_normalized)
plt.title("Ideal image for class 'Bear'")
plt.axis('off')
plt.savefig(script_folder / 'images/filter-vis-VGG16-bear.png')


# 'Goldfish' = 1, 'Bear' = 294, 'Assault Rifle' = 413
score = CategoricalScore(413)

# Generate image
activations = activation_maximization(score, steps=200, callbacks=[PrintProgress()])

img = activations[0]
img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)

# Show and save image
plt.imshow(img_normalized)
plt.title("Ideal image for class 'Assault Rifle'")
plt.axis('off')
plt.savefig(script_folder / 'images/filter-vis-VGG16-rifle.png')