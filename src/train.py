import glob
import model
import data_process
from keras.callbacks import CSVLogger
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.4
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
    
    if (lrate < 4e-5):
        lrate = 4e-5
      
    print('Changing learning rate to {}'.format(lrate))
    return lrate


csv_logger = CSVLogger("check_point/model-attRNN.csv", append=True)

numClass = 15

#Load data
trainData, valData, testData = data_process.load_data()

trainGen = SpeechGen(trainData)
valGen = SpeechGen(valData)

model = model.AttRNNSpeechModel(numClass, samplingrate = 8000, inputLength = None)

model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])

model.summary()

lrate = LearningRateScheduler(step_decay)

earlystopper = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=10,
                             verbose=1, restore_best_weights=True)

checkpointer = ModelCheckpoint('model-attRNN.hdf5', monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True)

results = model.fit(trainGen, validation_data=valGen, epochs=60, verbose=1,
                    callbacks=[earlystopper, checkpointer, lrate, csv_logger])

    


        
