import numpy as np
from tensorflow import keras

class Write_to_log(keras.callbacks.Callback):

    def __init__(self, filename='log.txt'):
        super(Write_to_log, self).__init__()
        self.batches = None
        self.a = None
        self.mean = None
        self.old = None
        self.square = None
        self.T = None
        self.filename = filename
    


    def on_train_begin(self, logs=None):
        #keys = list(logs.keys())
        #print("Starting training; got log keys: {}".format(keys))
        self.mean=self.model.get_weights()
        self.old=self.mean    
        self.square=np.square(self.mean)
        fout=open(self.filename, "w")
        fout.close()
        
        
    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        #print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
        self.batches = 0
        self.mean[:] = [x * 0 for x in self.mean]
        self.square[:] = [x * 0 for x in self.square] 

    
    def on_train_batch_end(self, batch, logs=None):
        #keys = list(logs.keys())
        #print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
        
        self.batches += 1
        self.a=self.model.get_weights()
        self.mean=np.add(self.mean, self.a)
        self.square=np.add(self.square, np.square(np.subtract(self.a, self.old)))
        self.old=self.a

    def on_test_begin(self, logs=None):
        
        self.mean[:] = [(x / self.batches) for x in self.mean]
        self.square[:] = [(x / self.batches) for x in self.square]
        #self.model.set_weights(self.mean)
        

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        #print("Stop testing; got log keys: {}".format(keys))
        
        #print("Test results with averaging:", logs)
        #self.model.set_weights(self.a)


    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        #print("Stop training; got log keys: {}".format(keys))
        
        self.T = 0
        number = 0
        for i in range(len(self.square)):
            self.T += np.sum(self.square[i])
            number += self.square[i].size
            
        self.T = self.T/number
        
        result = [epoch, self.T]
        for x in keys:
            result.append(logs.get(x))
        fout=open(self.filename, "a")
        if epoch==0:
            header=["epoch", "temperature"]
            header.append(keys)
            print(*header, sep=' ', file=fout)
        print(*result, sep=' ', file=fout)
        fout.close()

    
"""
    def on_train_end(self, logs=None):
        #self.model.set_weights(self.mean)
        
    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


    


    

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
"""
