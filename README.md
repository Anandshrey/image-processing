# Face Detection
This simple program detects faces in an image using the Cascade classifier in skimage.feature module. It uses a trained file **lbp_frontal_face_cascade_filename** that can detect frontal faces. 
The detector model is instantiated with this trained file. The detect_multi_scale function of the model takes the image as input along with parameters like scale_factor, step_ratio, min and max sizes of the box to highlight detected faces
