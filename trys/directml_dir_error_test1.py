import tensorflow as tf
import os

from keras.applications import InceptionResNetV2, EfficientNetB3
from vit_keras import vit
import keras

from keras import Model, layers
from keras.layers import Concatenate, RandomZoom, Input, Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.losses import categorical_crossentropy

import numpy as np

import matplotlib.pyplot as plt

# 아래는 GPU 가용 확인용 코드
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

IMAGE_SIZE = (448, 448)
BATCH_SIZE = 32
train_path = "D:\pics\dog\\train"
test_path = "D:\pics\dog\\test"
breeds = os.listdir(train_path)
breeds_count = len(breeds)
index_to_breed = {}
for i, j in zip(range(0, breeds_count), breeds):
    index_to_breed[i] = j
print(index_to_breed)

# print(breeds_count)
# os.listdir(PATH)

# 1 학습용 사진 가져오기

train_ds_raw, val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    train_path,
    seed=42,
    shuffle=False,  # shuffle=False,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.1,
    subset="both",  # train과 validation을 모두 뱉어라
    crop_to_aspect_ratio=False,  # 사진 비율 옵션
)

'''
label 보기
for images, labels in train_ds_raw.map(lambda x, y: (x, y)):
    print(labels)
'''

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    test_path,
    shuffle=False,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    labels=None,
    crop_to_aspect_ratio=False,  # 사진 비율 옵션
)

# print(test_ds_raw)

train_ds = train_ds_raw  # .batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)  # prefetch는 미리 가져와서 시간단축, autotune은 말그대로 자동 튜닝/소프트웨어 파이프라인
val_ds = val_ds_raw  # .batch(BATCH_SIZE)
test_ds = test_ds_raw  # .batch(BATCH_SIZE)

# dataset = dataset.apply(tf.data.experimental.unbatch()) # deprecated
# dataset = dataset.unbatch() 이건 진짜 왜 뭐가 문제인지 모르겠음

'''
def plot_ds_images(ds):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        lbl=labels.numpy()
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(index_to_breed[lbl[i].argmax()])
            plt.axis("off")

plot_ds_images(train_ds)
plt.show()
'''

'''
다음 할일
품종 엄선 
사진 폴더들train test도 따로 분류 해놔야함
'''

# 2 모델 세팅

# 가져올 모델들과 쌓을 이름들
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
model_features = build_feat_model(models_for_stacking, IMAGE_SIZE + (3,), data_augmentation)
# model_features.summary()
'''
tf.keras.utils.plot_model(
    model_features,
    to_file='model.png',
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=False,
)
'''


# 4 특징벡터 뽑기

def get_features_from_model(model, ds, count=1):
    features = []
    ds_has_y = isinstance(ds.element_spec, tuple)

    if ds_has_y:
        labels = []
        y = np.concatenate([y for x, y in ds], axis=0)

    for i in range(count):
        predictions = model.predict(ds)
        features.append(predictions)
        if ds_has_y:
            labels.append(y)

    if ds_has_y:
        return np.concatenate(features), np.concatenate(labels)
    else:
        return np.concatenate(features)


features_train, y_train = get_features_from_model(model_features, train_ds, count=2)  # 왜 2번?
features_val, y_val = get_features_from_model(model_features, val_ds)
print(features_train.shape, y_train.shape, features_val.shape, y_val.shape)


# 5 classification model

# drop out용 class?
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)


# callback, 최적화용?
early_stopping = EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=10,
    verbose=1,
    restore_best_weights=True)
reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    patience=3, verbose=1, factor=0.3
)

dnn_model = keras.models.Sequential([
    Input(features_train.shape[1:]),
    MCDropout(0.7),
    Dense(breeds_count, activation='softmax')
])
dnn_model.compile(
    optimizer='adam',#tf.keras.optimizers.Adam(learning_rate=0.001),  # keras.optimizers.Adam(0.001),
    loss=categorical_crossentropy,
    metrics=["categorical_accuracy"]
)

dnn_model.fit(
    features_train,
    y_train,
    validation_data=(features_val, y_val), # 설마 셔플 때문에 꼬였나?
    batch_size=128,
    epochs=50,
    callbacks=[early_stopping, reduce_lr_on_plateau]
)  # 학습
dnn_model.save("model.h5")  # 모델 저장

# ? 예측을 여러번 해서 평균을 때린다고?
y_val_probas = np.stack([dnn_model.predict(features_val) for sample in range(50)])
val_predictions = y_val_probas.mean(axis=0)

'''
(50, 305, 12)
(305, 12)
print(y_val_probas.shape)
print(val_predictions.shape)
'''


# 6 모델 평가
def get_loss_and_acc(y_true, y_pred):
    cat_ce = tf.keras.losses.CategoricalCrossentropy()
    loss_value = cat_ce(y_true, y_pred).numpy()
    cat_acc = tf.keras.metrics.CategoricalAccuracy()
    cat_acc.update_state(y_true, y_pred)
    acc_value = cat_acc.result().numpy()
    return [loss_value, acc_value]


val_loss, val_accuracy = get_loss_and_acc(y_val, val_predictions)
print(f"Validation loss: {val_loss}", f"\nValidation accuracy: {val_accuracy}")

'''
print(y_val)
print(np.argmax(val_predictions))
'''
# gc.collect()

