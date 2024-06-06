import cv2
import keras
import os
import numpy as np

ISDEBUG = True
BATCH_SIZE = 10
DATA_PATH = "/Users/flam/Desktop/Kaushik/custom model/dataset"
SIZE = (256, 256)


class FrootiDataset:
    def __init__(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.test_image_path = os.path.join(DATA_PATH, "test", "images")
        self.test_label_path = os.path.join(DATA_PATH, "test", "labels")
        self.train_image_path = os.path.join(DATA_PATH, "train", "images")
        self.train_label_path = os.path.join(DATA_PATH, "train", "labels")
        self.valid_image_path = os.path.join(DATA_PATH, "valid", "images")
        self.valid_label_path = os.path.join(DATA_PATH, "valid", "labels")
        self.size = SIZE


    def image_viewer(self):
        for image_name, label_name in zip(os.listdir(self.test_image_path),os.listdir(self.test_label_path)):

            image = keras.utils.load_img((os.path.join(self.test_image_path, image_name)))
            image_array = keras.utils.img_to_array(image)
            image_array = np.array(image_array, np.uint8)
            label_file_path = os.path.join(self.test_label_path, label_name)

            with open(label_file_path, 'r') as file:
                text = file.read().split()
                                
                if(ISDEBUG):
                    print(f"text out from label file:{type(text)}")

            if(ISDEBUG): 
                print(image_array)
            
            cv2.imshow('Image', image_array)
            key = cv2.waitKey(0)
            if key == ord('q'):  
                cv2.destroyAllWindows



dataset = FrootiDataset()
dataset.image_viewer()



    


