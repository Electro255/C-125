import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

x, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(x, y , random_state = 9, train_size = 7500, test_size = 2500)
x_train_scaled = x_train / 255.0
x_test_scaled = x_test / 255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinominal').fit(x_train_scaled, y_train)

def getprediction(image):
    image_open = Image.open(image)
    image_bw = image_open.convert('L')
    image_bw_resize = image_bw.resize((30, 30), Image.ANTIALIAS)
    pixel_filter = 20
    min_filter = np.percentile(image_bw_resize, pixel_filter)
    image_bw_resize_inverted_scale = np.clip(image_bw_resize-min_filter, 0, 255)
    max_filter = np.max(image_bw_resize)
    image_bw_resize_inverted_scale = np.asarray(image_bw_resize_inverted_scale)/ max_filter
    test_sample = np.array(image_bw_resize_inverted_scale).reshape(1, 784)
    test_predict = clf.predict(test_sample)
    
    return test_predict[0]