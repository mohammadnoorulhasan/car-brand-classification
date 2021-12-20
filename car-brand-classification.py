import sys
sys.path.append("F:\\Machine Learning\\utils")
from CNNModels.TrainCNNWithResnet import TrainCNNWithResnet
from tensorflow.keras.preprocessing import image
import numpy as np
trainingdatasetPath = "Dataset\\Train"
validationdatasetPath = "Dataset\\Test"

# trainingdatasetPath = "F:\\Machine Learning\\face-mask-detector\\Dataset\\Train"  
# validationdatasetPath = "F:\\Machine Learning\\face-mask-detector\\Dataset\\Test"

obj = TrainCNNWithResnet(trainingdatasetPath, validationdatasetPath, useTrainDatasetForValidation = False)
obj.train(epochs=20)
loss, acc = obj.evaluate()
print(acc)


obj = TrainCNNWithResnet(loadModel = True)# trainDatasetPath, testDatasetPath)
path = "F:\Machine Learning\car-brand-classification\Dataset\Test\\audi\\23.jpg"
img = image.load_img(path, target_size=(224,224))
img = image.img_to_array(img)
# img = img/255.0
img = np.expand_dims(img, axis=0)

print(obj.predict(img, normalize = True))
