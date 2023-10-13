from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, InputLayer

from feature_generater import *
from env_var import specie_name

root = os.getcwd()


def save_vit_model(vit_feature, specie_name):
    EarlyStop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    my_callback = [EarlyStop_callback]

    os.chdir(root)
    dnn = []

    breed = np.load(os.path.join(root, "np_save", f'{specie_name}_train_breed.npy'))
    n_classes = 12 if specie_name == 'dog' else 7  # 얘도 폴더 개수 셀까?

    # Prepare DNN model
    if not os.path.isfile(os.path.join(root, f"{specie_name}_model.h5")):
        dnn = keras.models.Sequential([
            InputLayer(4096, ),
            Dropout(0.7),
            Dense(n_classes, activation='softmax')
        ])

        dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        h = dnn.fit(vit_feature, breed,
                    batch_size=128,
                    epochs=60,
                    validation_split=0.1,
                    callbacks=my_callback)
        dnn.save(f"{specie_name}_model.h5")
    else:
        dnn = keras.models.load_model(f"{specie_name}_model.h5")


save_vit_model(gen_vit_keras_feature(specie_name)[0], specie_name)
