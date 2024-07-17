from skimage import data
from matplotlib import pyplot as plt
from matplotlib import patches
# Import Cascade classifier
from skimage.feature import Cascade
def display_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
image = plt.imread(./images/'person.jpg')
display_image(image)
# Load the trained file
tfile = data.lbp_frontal_face_cascade_filename()
# Initialize detector with trained file.
detector = Cascade(tfile)
# Apply detector on the image
detect = detector.detect_multi_scale(img=image, scale_factor=1.2,
         step_ratio=1,min_size=(50, 50),max_size=(500, 500))
def display_detect_face(result, detect, title="Face image"):
plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis('off')
for patch in detect:
    img_desc.add_patch(patches.Ellipse((patch['c'] + (patch['width']//2), patch['r'] + (patch['height']//2)), patch['width'], patch['height'],fill=False,color='r', linewidth=2)
)
plt.show() 
# Display image with detected face highlighted
display_detected_face(image, detect)
    
