import cv2
import os

class ClevrImageLoader():
    def __init__(self, images_dir):
        self.images_dir = images_dir

    def get(self,index):
        padded_index = str(index).rjust(6,'0')
        img_filename = os.path.join(self.images_dir, 'val', 'CLEVR_val_{}.png'.format(padded_index))
        image = cv2.imread(img_filename)
        return image / 255.
