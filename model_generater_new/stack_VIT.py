import os

from keras.applications import InceptionResNetV2, EfficientNetB3
from vit_keras import vit
import keras

from keras import Model, layers
from keras.layers import Concatenate, RandomZoom

from env_var import specie_name

train_path = f"D:\pics\{specie_name}\\train"
test_path = f"D:\pics\{specie_name}\\test"
breeds = os.listdir(train_path)
breeds_count = len(breeds)
IMAGE_SIZE = (448, 448, 3)

models_search = {
    'InceptionResNetV2': [InceptionResNetV2, keras.applications.inception_resnet_v2.preprocess_input],
    'EfficientNetB3': [EfficientNetB3, keras.applications.efficientnet.preprocess_input],
    'vit_l32': [vit.vit_l32, vit.preprocess_inputs],
}
models_for_stacking = ['vit_l32', 'EfficientNetB3', 'InceptionResNetV2']


# 모델 쌓는 함수들?
def get_vit_model_feat(app_class, shape, inputs, prep_input_fx):
    base_model = app_class(
        image_size=shape[:2],  # 이미지 가로세로
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=breeds_count
    )
    x = prep_input_fx(inputs)  # 전처리
    outputs = base_model(x)  # 한걸 모델에 넣어 output
    return outputs


def get_keras_model_feat(app_class, shape, inputs, prep_input_fx):
    base_model = app_class(
        include_top=False,
        weights='imagenet',  # ImageNet으로 학습한 가중치를 이용하는 모델들과 관련? https://keras.io/ko/applications/
        input_shape=shape,
    )
    x = prep_input_fx(inputs)
    x = base_model(x)
    outputs = keras.layers.GlobalAveragePooling2D()(x)
    return outputs


def build_feat_model(models_names, shape, aug_layer=None):
    all_outputs = []
    inputs = keras.Input(shape=shape)
    if aug_layer != None:
        aug_inputs = aug_layer(inputs)
    else:
        aug_inputs = inputs
    for model_type in models_names:
        model_class = models_search[model_type][0]  # 모델?
        model_prep_input = models_search[model_type][1]  # 모델별 이미지 전처리기?
        if model_type.startswith('vit'):
            model_outputs = get_vit_model_feat(model_class, shape, aug_inputs, model_prep_input)
        else:
            model_outputs = get_keras_model_feat(model_class, shape, aug_inputs, model_prep_input)
        all_outputs.append(model_outputs)
    concat_outputs = Concatenate()(all_outputs)
    model = Model(inputs, concat_outputs)
    return model


# 이미지에 노이즈를 주어 증강, augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    RandomZoom(0.1),
], name='data_augmentation')


# 특징 뽑는 모델 (4096차원)
def get_feature_model():
    return build_feat_model(models_for_stacking, IMAGE_SIZE, data_augmentation)
