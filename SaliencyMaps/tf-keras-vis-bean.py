import warnings

warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from packaging.version import parse as version

from tf_keras_vis.utils import num_of_gpus

if version(tf.version.VERSION) < version('2.16.0'):
    import tensorflow.keras as keras
else:
    import keras

_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

from pathlib import Path
script_folder = Path(__file__).parent

model = keras.models.load_model(script_folder.parent / 'models/bean_leaf_vgg16.keras')
model.summary()

# Image titles
image_titles = ['Angular Leaf Spot', 'Bean Rust', 'Healty']

img_paths = [
    script_folder.parent / 'data/Bean_Leaf_Lesions_Classification/val/angular_leaf_spot/angular_leaf_spot_val.14.jpg',
    script_folder.parent / 'data/Bean_Leaf_Lesions_Classification/val/bean_rust/bean_rust_val.44.jpg',
    script_folder.parent / 'data/Bean_Leaf_Lesions_Classification/val/healthy/healthy_val.20.jpg'
]

img1 = keras.utils.load_img(img_paths[0], target_size=(224, 224))
img2 = keras.utils.load_img(img_paths[1], target_size=(224, 224))
img3 = keras.utils.load_img(img_paths[2], target_size=(224, 224))

images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparing input data for VGG16
X = keras.applications.vgg16.preprocess_input(images)

# Rendering
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].axis('off')
plt.tight_layout()
plt.show()

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

replace2linear = ReplaceToLinear()


# Instead of using the ReplaceToLinear instance above,
# you can also define the function from scratch as follows:
def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = keras.activations.linear

from tf_keras_vis.utils.scores import CategoricalScore

# 0 is angular_leaf_spot, 1 is bean_rust and 2 is healthy
score = CategoricalScore([0, 1, 2])


# Instead of using CategoricalScore object,
# you can also define the function from scratch as follows:
def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
    return (output[0][1], output[1][294], output[2][413])


# Make predictions
print("Making predictions...\n")
predictions = model.predict(X)

for i in range(len(img_paths)):
    predicted_index = np.argmax(predictions[i])
    predicted_class = image_titles[predicted_index]
    confidence = predictions[i][predicted_index] * 100
    
    # Verify prediction
    res = "✅" if predicted_class == image_titles[i] else "❌"
    
    print(f"Image {i+1} (real: {image_titles[i]})")
    print(f" -> Model prediction: {predicted_class} al {confidence:.2f}% {res}")
    
    # Print other classes probabilities
    print(f" -> Probability details: ALS: {predictions[i][0]*100:.1f}%, Rust: {predictions[i][1]*100:.1f}%, Healthy: {predictions[i][2]*100:.1f}%\n")


### Vanilla Saliency ###


from tf_keras_vis.saliency import Saliency

# from tf_keras_vis.utils import normalize

# Create Saliency object.
saliency = Saliency(model, model_modifier=replace2linear, clone=True)

# Generate saliency map
saliency_map = saliency(score, X)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(saliency_map[i], cmap='jet')
    ax[i].axis('off')
plt.tight_layout()
plt.show()


### Smooth Grad ###


# Generate saliency map with smoothing that reduce noise by adding noise
saliency_map = saliency(
    score,
    X,
    smooth_samples=20,  # The number of calculating gradients iterations.
    smooth_noise=0.20)  # noise spread level.

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(saliency_map[i], cmap='jet')
    ax[i].axis('off')
plt.tight_layout()
plt.savefig(script_folder / 'images/bean_smoothgrad.png')
plt.show()



### GradCAM ###


from matplotlib import cm

from tf_keras_vis.gradcam import Gradcam

# Create Gradcam object
gradcam = Gradcam(model, model_modifier=replace2linear, clone=True)

# Generate heatmap with GradCAM
cam = gradcam(score, X)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)  # overlay
    ax[i].axis('off')
plt.tight_layout()
plt.show()



### GradCAM++ ###


from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# Create GradCAM++ object
gradcam = GradcamPlusPlus(model, model_modifier=replace2linear, clone=True)

# Generate heatmap with GradCAM++
cam = gradcam(score, X, penultimate_layer=-1)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.savefig(script_folder / 'images/bean_gradcam_plus_plus.png')
plt.show()



### ScoreCAM ###


from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils import num_of_gpus

# Create ScoreCAM object
scorecam = Scorecam(model, model_modifier=replace2linear)

# Generate heatmap with ScoreCAM
cam = scorecam(score, X, penultimate_layer=-1)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()



### Faster ScoreCAM ### 


from tf_keras_vis.scorecam import Scorecam

# Create ScoreCAM object
scorecam = Scorecam(model, model_modifier=replace2linear)

# Generate heatmap with Faster-ScoreCAM
cam = scorecam(score, X, penultimate_layer=-1, max_N=10)

# Render
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()