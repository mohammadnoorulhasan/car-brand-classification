from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from glob import glob


class TrainModel:

    def __init__(self, trainDatasetPath, validationDatasetPath = None, datasetFormat = "folder",
                        imageSize = (224,224), useDatagenerator = True):

        self._imageSize = imageSize
        if datasetFormat == "folder":
            # We'll rescale while using Image Data Generator
            self._trainDatasetPath = trainDatasetPath
            self.classes = len(glob(self._trainDatasetPath))
            if useDatagenerator:
                self._trainDatagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)


            else:
                self._trainDatagen = ImageDataGenerator(rescale = 1./255)

            self._trainDataset = self.loadFolderDataset(self._trainDatasetPath,self._trainDatagen,imageSize)

            if validationDatasetPath is not None:
                self._validationDatasetPath = validationDatasetPath
                self._validationDatagen = ImageDataGenerator(rescale= 1.0/255)
                self._validationDataset = self.loadFolderDataset(self._validationDatasetPath,
                                                            self._validationDatagen,imageSize)

                
            else:
                self._validationDatagen = None

        elif datasetFormat == "csv":
            pass

        else:
            print("wrong dataset format passed\n Please select \"folder\" or \csv\"")
        


    def loadFolderDataset(self,datasetPath, datagen, image_size, batchSize = 32, classMode = "categorical"):
        dataset = datagen.flow_from_directory( datasetPath,
                                                 target_size = image_size,
                                                 batch_size = batchSize,
                                                 class_mode = classMode)
        return dataset

    def train(self, model, useDatagenerator = True, epochs = 50, 
                        filepath = "model/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"):
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                                    verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        if useDatagenerator:
            if self._validationDatagen is not None:
                self.__result = model.fit(self._trainDataset,
                                                    validation_data=self._validationDataset,
                                                    epochs=epochs,
                                                    steps_per_epoch=len(self._trainDataset),
                                                    validation_steps=len(self._validationDataset),
                                                    callbacks = callbacks_list)
            else:
                self.__result = model.fit_generator(self._trainDataset,
                                                    epochs=epochs,
                                                    steps_per_epoch=len(self._trainDataset))
        else:
            model.fit(self._trainDataset, epochs=epochs, steps_per_epoch=len(self._trainDataset))
    
    def load_csv(self, datasetPath):
        pass


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
 

class TrainCNNWithBaseLineModel(TrainModel):


    def __init__(self, trainDatasetPath, validationDatasetPath, datasetFormat = "folder", 
                                    imageSize = (224,224), useDatagenerator = True):
        self.useDatagenerator = useDatagenerator
        super().__init__(trainDatasetPath, validationDatasetPath=validationDatasetPath, \
                                datasetFormat=datasetFormat, imageSize=imageSize, \
                                useDatagenerator=useDatagenerator)
        self.model = self.getModel()
        
    
    def getModel(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape = list(self._imageSize) +[3]))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(3, activation='softmax'))
        # compile model
        opt = SGD(lr=0.01, momentum=0.9)    
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def evaluate(self):
        if self.useDatagenerator == True:
            return self.model.evaluate(self._validationDataset, verbose = 0)


if __name__ == "__main__":
    trainDatasetPath = "Dataset/Train"
    testDatasetPath = "Dataset/Test"
    obj = TrainCNNWithBaseLineModel(trainDatasetPath, testDatasetPath)
    obj.train(obj.model, epochs=10)
    loss, acc = obj.evaluate()
    print(acc)