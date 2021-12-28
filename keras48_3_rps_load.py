import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# pic_path = 'D:\\_data\\image\\rps\\rps\\paper\\paper01-000.png'
# pic_path = 'D:\\_data\\image\\rps\\rps\\rock\\rock01-004.png'
# pic_path = 'D:\\_data\\image\\rps\\rps\\scissors\\scissors01-001.png'

pic_path = 'D:\\_data\\image\\predict_sodam\\sodamS.jpg'
model_path = 'D:\\Study\\_save_weight\\keras48_3_rps_IDG.hdf5'


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
    classes2 = pred[0][2] * 100
    print(classes0,classes1,classes2)
    # print(max(classes0,classes1,classes2))

    if max(classes0,classes1,classes2) == classes0:
        print(f"{round(classes0, 2)} % 확률로 보 입니다")
    elif max(classes0,classes1,classes2) == classes1:
        print(f"{round(classes1, 2)} % 확률로 바위 입니다")
    else:
        print(f"{round(classes2, 2)} % 확률로 가위 입니다")

'''
35.22 % 확률로 가위 입니다


37.56 % 확률로 가위 입니다
'''



