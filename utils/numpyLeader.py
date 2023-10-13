import numpy as np
import os

root = os.getcwd()
data = np.load(os.path.join(root, "../model_generater_new/np_save", "dog_train_array_save.npy"))
for line in data:
    print(line)
