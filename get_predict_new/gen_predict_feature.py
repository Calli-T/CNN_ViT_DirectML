import numpy as np

from keras.layers import GlobalAveragePooling2D, Lambda, Input
import tensorflow as tf

from stack_VIT import *
from env_var import specie_name

from tqdm import tqdm

root = os.getcwd()
img_size = (448, 448, 3)
dog_model = keras.models.load_model(os.path.join(root, '../get_predict_new/models', "dog_model.h5"))
cat_model = keras.models.load_model(os.path.join(root, '../get_predict_new/models', "cat_model.h5"))

dog_breeds = os.listdir("D:\pics\dog\\train")
cat_breeds = os.listdir("D:\pics\cat\\train")


def test_images_to_array(data_dir):
    os.chdir(data_dir)
    images_names = os.listdir(data_dir)
    test_size = len(images_names)
    # print(images_names)

    X = np.zeros([test_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    # for i in tqdm(range(test_size)):
    for i in tqdm(range(test_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name)
        img_pixels = tf.keras.preprocessing.image.load_img(img_dir, target_size=img_size)
        X[i] = img_pixels

    # print('Ouptut Data Size: ', X.shape)
    return X


def get_features(model_name, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)
    feature_maps = feature_extractor.predict(data, batch_size=64, verbose=1)

    # print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


def get_features_from_model(model, ds, count=1):
    features = []

    for i in range(count):
        predictions = model.predict(ds)
        features.append(predictions)

    return np.concatenate(features)


def gen_vit_keras_test_feature(pics_array, specie_name='dog'):
    model_features = get_feature_model()
    # img_arrays = test_images_to_array('pics')

    features_test = get_features_from_model(model_features, pics_array, count=1)

    return features_test


def predict(features, model):
    return model.predict(features, batch_size=128)


test_images_features = gen_vit_keras_test_feature(test_images_to_array(os.path.join(root, "../get_predict_new/pics")), specie_name)
y_pred = predict(test_images_features, dog_model if specie_name == 'dog' else cat_model)

list = os.listdir(os.path.join(root, '../get_predict_new/pics'))

for i in range(len(y_pred)):
    # print(y_pred[i])
    if (specie_name == 'dog'):
        print(list[i] + ": " + dog_breeds[np.argmax(y_pred[i])])
    else:
        print(list[i] + ": " + cat_breeds[np.argmax(y_pred[i])])
'''
할일
1. 특징만들기
2. 모델끌어오기
3. 모델로 예측한값 전송
'''

'''
특징만들기
1. get_feature필요함
2. 
'''
