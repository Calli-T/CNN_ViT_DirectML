from img_preprocessing import *

from stack_VIT import *

root = os.getcwd()


def get_features_from_model(model, ds, count=1):
    features = []

    for i in range(count):
        predictions = model.predict(ds)
        features.append(predictions)

    return np.concatenate(features)


def gen_vit_keras_feature(specie_name='dog'):
    model_features = get_feature_model()
    img_arrays, breed = get_train_to_array(specie_name)

    print(os.path.join(root, "np_save", f'{specie_name}_vit_train.npy'))
    if not os.path.isfile(os.path.join(root, "np_save", f'{specie_name}_vit_train.npy')):
        features_train = get_features_from_model(model_features, img_arrays, count=1)
        np.save(os.path.join(root, "np_save", f'{specie_name}_vit_train.npy'), features_train)

        return features_train, breed
    else:
        features_train = np.load(os.path.join(root, "np_save", f'{specie_name}_vit_train.npy'))
        return features_train, breed

# features_train, y_train = gen_vit_keras_feature(specie_name)
