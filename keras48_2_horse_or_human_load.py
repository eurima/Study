import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = 'D:\\_data\\image\\predict_sodam\\sodam.jpg'
model_path = 'D:\\Study\\_save_weight\\keras48_2_horse_or_human.hdf5'


def load_my_image(img_path, show=True):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        # plt.append('off')
        plt.show()

    return img_tensor


if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    classes0 = pred[0][0] * 100
    classes1 = pred[0][1] * 100
    if classes0 > classes1:
        print(f"당신은 {round(classes0, 2)} % 확률로 말 입니다")
    else:
        print(f"당신은 {round(classes1, 2)} % 확률로 사람 입니다")


'''
당신은 77.45 % 확률로 사람 입니다
'''



