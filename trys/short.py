import warnings

warnings.simplefilter(action='ignore', category=(RuntimeWarning, FutureWarning, UserWarning))

import numpy as np
import pandas as pd
import gc
import tensorflow as tf
import keras
from keras import layers, Model
from keras.layers import Dense, Input, RandomZoom, Concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.losses import categorical_crossentropy
from keras.applications import InceptionResNetV2, EfficientNetB3
from sklearn.preprocessing import LabelEncoder

from vit_keras import vit

# --------------------path--------------------
np.random.seed(42)
tf.random.set_seed(42)

IMAGE_SIZE = (448, 448)
BATCH_SIZE = 32

competition = "dog-breed-identification"
labels_path = f"{competition}/labels.csv/"
submission_path = f"{competition}/sample_submission.csv/"

labels_df = pd.read_csv('dog-breed-identification/labels.csv')
submission_df = pd.read_csv('dog-breed-identification/sample_submission.csv')

train_folder = f"{competition}/train/"
test_folder = f"{competition}/test/"
print(train_folder, test_folder)

# ------------ dataset preprocessing ---------------

# 이미지에 노이즈를 주어 증강, augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    RandomZoom(0.1),
], name='data_augmentation')

# 종 번호를 데이터 프레임에서 인코?딩
labels_df['label'] = LabelEncoder().fit_transform(labels_df.breed)
# print(labels_df.head())

# 종 번호를 쭉 가져감, 한 열 통째로
labels_list = labels_df['label'].to_numpy().tolist()


'''
할일 1
https://ecogis.net/314의 예시를 사용해
단일 폴더에 csv폴더의 라벨로 붙여서 나눠진걸 따오는 코드를 개조해
여러 폴더로 가져온 이미지들을 자동으로 라벨붙이는 작업을 한다.
'''
def create_raw_ds(train_folder, labels_list, split_size=0.1, image_size=(224, 224)):
    train_ds_raw, val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_folder,
        #         seed=SEED,
        labels=labels_list, # 10222개의 파일에 대한 csv 파일속의 label, 종번호
        label_mode='categorical',
        validation_split=0.1,
        shuffle=False,
        subset="both",
        image_size=image_size,
        batch_size=None,
        crop_to_aspect_ratio=False
    )
    return train_ds_raw, val_ds_raw


train_ds_raw, val_ds_raw = create_raw_ds(train_folder, labels_list, image_size=IMAGE_SIZE)

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    test_folder,
    #     seed=SEED,
    labels=None,
    shuffle=False,
    image_size=IMAGE_SIZE,
    batch_size=None,
    crop_to_aspect_ratio=False
)

train_ds = train_ds_raw.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE) # prefetch는 미리 가져와서 시간단축, autotune은 말그대로 자동 튜닝/소프트웨어 파이프라인
val_ds = val_ds_raw.batch(BATCH_SIZE)
test_ds = test_ds_raw.batch(BATCH_SIZE)

# feature extract --------------------------------
models_search = {
    'InceptionResNetV2': [InceptionResNetV2, keras.applications.inception_resnet_v2.preprocess_input],
    'EfficientNetB3': [EfficientNetB3, keras.applications.efficientnet.preprocess_input],
    'vit_l32': [vit.vit_l32, vit.preprocess_inputs],
}
models_for_stacking = ['vit_l32', 'EfficientNetB3', 'InceptionResNetV2']


# ---------------------------------------

def get_vit_model_feat(app_class, shape, inputs, prep_input_fx):
    base_model = app_class(
        image_size=shape[:2],
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=120
    )
    x = prep_input_fx(inputs)
    outputs = base_model(x)
    return outputs


def get_keras_model_feat(app_class, shape, inputs, prep_input_fx):
    base_model = app_class(
        include_top=False,
        weights='imagenet',
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
        model_class = models_search[model_type][0]
        model_prep_input = models_search[model_type][1]
        if model_type.startswith('vit'):
            model_outputs = get_vit_model_feat(model_class, shape, aug_inputs, model_prep_input)
        else:
            model_outputs = get_keras_model_feat(model_class, shape, aug_inputs, model_prep_input)
        all_outputs.append(model_outputs)
    concat_outputs = Concatenate()(all_outputs)
    model = Model(inputs, concat_outputs)
    return model


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


# set feature model
model_features = build_feat_model(models_for_stacking, IMAGE_SIZE + (3,), data_augmentation)
model_features.summary()

# 모델에서 특징뽑기, train과 val모두
features_train, y_train = get_features_from_model(model_features, train_ds, count=2)
features_val, y_val = get_features_from_model(model_features, val_ds)
print(features_train.shape, y_train.shape, features_val.shape, y_val.shape)
#print(features_val.shape, y_val.shape)


# set classification model
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
    Dense(120, activation='softmax')
])
dnn_model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=categorical_crossentropy,
    metrics=["categorical_accuracy"]
)

# fit의 history는 왜 기록해두는가? 학습기록의 로그?
history_dnn = dnn_model.fit(
    features_train,
    y_train,
    validation_data=(features_val, y_val),
    batch_size=128,
    epochs=50,
    callbacks=[early_stopping, reduce_lr_on_plateau]
)
dnn_model.save("model.h5")

y_val_probas = np.stack([dnn_model.predict(features_val)
                         for sample in range(50)])
val_predictions = y_val_probas.mean(axis=0)

# 모델 평가
def get_loss_and_acc(y_true, y_pred):
    cat_ce = tf.keras.losses.CategoricalCrossentropy()
    loss_value = cat_ce(y_true, y_pred).numpy()
    cat_acc = tf.keras.metrics.CategoricalAccuracy()
    cat_acc.update_state(y_true, y_pred)
    acc_value = cat_acc.result().numpy()
    return [loss_value, acc_value]


val_loss, val_accuracy = get_loss_and_acc(y_val, val_predictions)
print(f"Validation loss: {val_loss}", f"\nValidation accuracy: {val_accuracy}")

gc.collect()
