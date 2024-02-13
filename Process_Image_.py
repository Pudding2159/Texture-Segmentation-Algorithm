import matplotlib as plt
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.color import label2rgb, rgba2rgb
from skimage.io import imread

def slic_zero_segmentation(image_path, n_segments, compactness):
    # Load the image
    image = imread(image_path)

    # Check if the image is RGBA and convert to RGB if necessary
    if image.ndim == 3 and image.shape[-1] == 4:
        image = rgba2rgb(image)

    # Determine the channel axis based on the number of dimensions
    channel_axis = -1 if image.ndim == 3 else None

    # Apply SLIC Zero segmentation
    segments = slic(image, n_segments=n_segments, compactness=compactness, channel_axis=None)

    # Create a color image to visualize the segments
    segmented_image = label2rgb(segments, image, kind='avg')

    # Show and save the segmented image
    plt.figure(figsize=(12, 6))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.savefig('./Prepared_Image.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    return segmented_image