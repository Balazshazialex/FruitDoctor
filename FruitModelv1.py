import os
from keras import Sequential
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers
import keras.callbacks

EPOCHS = 2000
BATCH_SIZE = 64




class FruitModelV1:
    def __init__(self):
        self.files = os.listdir('Fruits\\train')
        self.path = 'Fruits\\train\\'
        self.checkpoint_path = "training/cp.ckpt"
        self._image_array = []
        self._label_array = []
        self._X_train, self._X_test, self._Y_train, self._Y_test = [], [], [], []
        self.label_to_text = {0: "Orange", 1: "Pear", 2: "Banana", 3: "Sick-Orange", 4: "Sick-Pear", 5: "Sick-Banana"}
        self.read_images()
        self.model = Sequential()

    def read_images(self):
        for i in range(len(self.files)):
            files_list = os.listdir(self.path + self.files[i])
            for k in tqdm(range(len(files_list))):
                img = cv2.imread(self.path + self.files[i] + "\\" + files_list[k])
                img=cv2.resize(img, (48,48), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # create the arrays
                self._image_array.append(img)
                self._label_array.append(i)
        self._image_array = np.array(self._image_array)/255
        self._label_array = np.array(self._label_array)
        self.split_into_train_test()

    def split_into_train_test(self):
        self._X_train, self._X_test, self._Y_train, self._Y_test = train_test_split(self._image_array, self._label_array, test_size=0.2)

    def train(self):
        pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights="imagenet")
        pretrained_model.trainable = True
        # add layers and pretrained model
        self.model.add(pretrained_model)
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(2, activation='relu'))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="mean_squared_error", metrics=["mae"])

        # save checkpoint

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1,mode="min")

        reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.9, monitor="val_mae", mode="auto", cooldown=0, patience=5, verbose=1, min_lr=1e-6)
        history = self.model.fit(self._X_train, self._Y_train, validation_data=(self._X_test, self._Y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[model_checkpoint, reduce_lr])

    def create_model(self):
        pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights="imagenet")
        pretrained_model.trainable = True
        # add layers and pretrained model
        self.model.add(pretrained_model)
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(2, activation='relu'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="mean_squared_error", metrics=["mae"])

    def load_model(self):
        self.create_model()
        self.model.load_weights(self.checkpoint_path)

    def predict(self):
        np_array = []
        img = cv2.imread("pear1.jpg")
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np_array.append(img)
        np_array=np.array(np_array)/255
        yhat = self.model.predict(np_array[:1])[0][1]
        #print(yhat)
        print(FruitModelV1.prediction_to_text(yhat))

    @staticmethod
    def prediction_to_text(x):
        if x<0.5:
            return "healthy banana"
        elif 0.5<=x<1.5:
            return "healthy orange"
        elif 1.5<=x<2.5:
            return "healthy pear"
        elif 2.5<=x<3.5:
            return "sickl banana"
        elif 3.5<=x<4.5:
            return "sick orange"
        elif 4.5<=x:
            return "sick pear"
        return "Undefined"
f = FruitModelV1()
#f.train()
f.load_model()
f.predict()