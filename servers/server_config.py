# This config is for mtcnn (detection) + facenet 512 (recognition)
# detection
tolerance = 0.85
increase_ratio = 5

# recognition
img_folder = 'imgs_test'
distance_metric = 'cosine'
detector_backend = 'mtcnn'
recognition_threshold = 0.2

# draw config
box_thickness = 1
