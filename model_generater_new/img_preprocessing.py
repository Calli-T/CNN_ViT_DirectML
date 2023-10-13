import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from keras.utils import to_categorical
from keras.utils import load_img

from env_var import specie_name

np.random.seed(42)
img_size = (448, 448, 3)  # (331, 331, 3)
root = os.getcwd()

# 파일 개수 반환
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


# 품종 반환
def get_breeds(path):
    if not os.path.exists(path):
        return 0
    return os.listdir(path)

train_dir = f"D:\pics\{specie_name}\\train"
test_dir = f"D:\pics\{specie_name}\\test"
# print(train_dir)
breeds = get_breeds(train_dir)
data_size = get_num_files(train_dir)
n_classes = len(breeds)
class_to_num = dict(zip(breeds, range(n_classes)))


def set_specie(specie_name):
    global specie
    global train_dir
    global test_dir
    global breeds
    global data_size
    global n_classes
    global class_to_num

    # 종, path for train, path for predict
    # 품종, 사진, 품종 가지수, 품종 <-> 번호 딕셔너리
    specie = specie_name
    '''
    train_dir = os.path.join(root, 'pics', specie, 'train')
    test_dir = os.path.join(root, 'pics', specie, 'test')
    '''
    train_dir = "D:\pics\dog\\train"
    test_dir = "D:\pics\dog\\test"
    # print(train_dir)
    breeds = get_breeds(train_dir)
    data_size = get_num_files(train_dir)
    n_classes = len(breeds)
    class_to_num = dict(zip(breeds, range(n_classes)))


def images_to_array(data_dir, img_size=(224, 224, 3)):
    # data_size = 10 * n_classes
    X = np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)
    y = np.zeros([data_size, 1], dtype=np.uint8)
    # print(y)
    # print(X)
    file_counter = 0

    for breed in breeds:
        # print(os.path.join(data_dir, breed))
        os.chdir(os.path.join(data_dir, breed))
        breed_path = os.getcwd()
        n_same_breed = get_num_files(breed_path)
        file_list = os.listdir()
        breed_number = class_to_num[breed]
        # print(breed + ": " + str(breed_number))
        # print(sorted(file_list))
        # print(n_same_breed)
        # print(os.getcwd())
        for i in tqdm(range(n_same_breed)):
            img_path = os.path.join(breed_path, file_list[i])
            try:
                img_pixels = load_img(img_path, target_size=img_size)
            except:
                continue
            # 테스트용 코드, 종별로 N장으로 제한
            # if i == 10:
            #    break

            X[file_counter] = img_pixels
            y[file_counter] = breed_number
            file_counter += 1

    # 원 핫 인코더
    y = to_categorical(y)

    # 번호 섞기?
    ind = np.random.permutation(data_size)
    X = X[ind]
    y = y[ind]

    print('Ouptut Data Size: ', X.shape)
    print('Ouptut Label Size: ', y.shape)
    return X, y


def images_to_array2(data_dir, img_size=(224, 224, 3)):
    os.chdir(data_dir)
    images_names = os.listdir(data_dir)
    test_size = len(images_names)
    # print(images_names)

    X = np.zeros([test_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    for i in tqdm(range(test_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name)
        img_pixels = tf.keras.preprocessing.image.load_img(img_dir, target_size=img_size)
        X[i] = img_pixels

    print('Ouptut Data Size: ', X.shape)
    return X


def get_test_image_name():
    return os.listdir(test_dir)


def get_train_to_array(specie_name='dog'):
    # train 이미지 배열로 변환, numpy 파일 없을 때만 저장
    if os.path.isfile(os.path.join(root, "np_save", f'{specie_name}_train_array_save.npy')) and os.path.isfile(
            os.path.join(root, "np_save", f'{specie_name}_train_breed.npy')):
        X = np.load(os.path.join(root, "np_save", f'{specie_name}_train_array_save.npy'))
        y = np.load(os.path.join(root, "np_save", f'{specie_name}_train_breed.npy'))
        return X, y
    else:
        print("이미지 압축 파일 없음, 생성중")
        X, y = images_to_array(train_dir, img_size)
        np.save(os.path.join(root, "np_save", f'{specie_name}_train_array_save.npy'), X)
        np.save(os.path.join(root, "np_save", f'{specie_name}_train_breed.npy'), y)
        return X, y


def get_test_to_array(specie_name='dog'):
    # test 이미지 배열로 변환, numpy 파일 없을 때만 저장
    if os.path.isfile(os.path.join(root, "np_save", f'{specie_name}_test_array_save.npy')):
        X = np.load(os.path.join(root, "np_save", f'{specie_name}_test_array_save.npy'))
        return X
    else:
        X = images_to_array2(test_dir, img_size)
        np.save(os.path.join(root, "np_save", f'{specie_name}_test_array_save.npy'), X)
        return X


# 파일 없다면 아래 내용 실행하여 자체 생성 가능, 없으면 각 모델에서 자동으로 생성
get_train_to_array(specie_name)
get_test_to_array(specie_name)

# test 이미 배열로 변환
# print(X)
# print(y)
