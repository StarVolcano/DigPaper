import numpy as np

def get_center(box):
    x0, y0, x1, y1 = box
    return [(x0 + x1) / 2, (y0 + y1) / 2]

def euclidean_dist(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

def find_closest_text(target_box, text_boxes):
    target_center = get_center(target_box)
    distances = [euclidean_dist(target_center, get_center(tb)) for tb in text_boxes]
    return np.argmin(distances)

def crop_box(image, box):
    return image.crop(box)
