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
model = keras.models.load_model(script_folder.parent / 'models/bean_leaf_vgg16.keras')
model.summary()

activation_maximization = ActivationMaximization(model, 
                                                 model_modifier=[ReplaceToLinear()], 
                                                 clone=True)

# 'Angular Leaf Spot' = 0, 'Bean Rust' = 1, 'Healthy' = 2
score = CategoricalScore(0)

# Generate image
activations = activation_maximization(score, steps=200, callbacks=[PrintProgress()])

img = activations[0]
img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)

# Show and save image
plt.imshow(img_normalized)
plt.title("Ideal image for class 'Angular Leaf Spot'")
plt.axis('off')
plt.savefig(script_folder / 'images/filter-vis-angular.png')


# 'Angular Leaf Spot' = 0, 'Bean Rust' = 1, 'Healthy' = 2
score = CategoricalScore(1)

# Generate image
activations = activation_maximization(score, steps=200, callbacks=[PrintProgress()])

img = activations[0]
img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)

# Show and save image
plt.imshow(img_normalized)
plt.title("Ideal image for class 'Bean Rust'")
plt.axis('off')
plt.savefig(script_folder / 'images/filter-vis-rust.png')


# 'Angular Leaf Spot' = 0, 'Bean Rust' = 1, 'Healthy' = 2
score = CategoricalScore(2)

# Generate image
activations = activation_maximization(score, steps=200, callbacks=[PrintProgress()])

img = activations[0]
img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)

# Show and save image
plt.imshow(img_normalized)
plt.title("Ideal image for class 'Healthy'")
plt.axis('off')
plt.savefig(script_folder / 'images/filter-vis-healthy.png')