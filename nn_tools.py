'''
Created on 23.03.2021

@author: Lars
'''
import numpy as np
from sklearn.preprocessing import StandardScaler
import os.path as osp
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

class WeightLoggerCallback(Callback):
    def __init__ (self, model, logname):
        self.__model = model
        self.__logname = logname
        self.__fp = open(self.__logname, "w")
        
        
        
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        weights = self.__model.get_weights()
        
        d = [
            weight_layer.tolist()
            for weight_layer in weights
        ]
        d = json.dumps(d)+"\n"
        self.__fp.write(d)
            
    def __del__ (self):
        self.__fp.close()
        
    def close (self):
        self.__fp.close()

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

class InOutScalers ():
    def __init__ (self, input_scaler, output_scaler):
        self.__input_scaler = input_scaler
        self.__output_scaler = output_scaler
        
    def input_scaler (self):
        return self.__input_scaler
    
    def output_scaler (self):
        return self.__output_scaler

class WindowTools ():
    @classmethod
    def create_input_output_data (cls, values, input_window, output_window):
        data_count = len(values) - input_window - output_window + 1
        
        input_data = np.empty((data_count, input_window, values.shape[1]), dtype=np.float)
        output_data = np.empty((data_count, output_window, values.shape[1]), dtype=np.float)
    
        for i in range(input_window, len(values) - output_window + 1):
            input_start = i - input_window
            input_end = i
            
            output_start = i
            output_end = i + output_window
            
            c_in = values[input_start : input_end]
            c_out = values[output_start : output_end]
            
            ni = i - input_window
            input_data[ni] = c_in
            output_data[ni] = c_out
        
        print(input_data.shape, output_data.shape)
        
        return input_data, output_data
    
    @classmethod
    def create_scalers (cls, input_data, output_data):
        input_shape = input_data.shape
        output_shape = output_data.shape
        
        c_in = np.reshape(input_data, (input_shape[0] * input_shape[1], input_shape[2]))
        c_out = np.reshape(output_data, (output_shape[0] * output_shape[1], output_shape[2]))
        
        input_scaler = StandardScaler().fit(c_in)
        output_scaler = StandardScaler().fit(c_out)
        
        return InOutScalers(input_scaler, output_scaler)
        
    @classmethod
    def scale_data (cls, input_data, output_data, scalers):
        if isinstance(scalers, InOutScalers):
            input_scaler = scalers.input_scaler()
            output_scaler = scalers.output_scaler()
            
            in_shape = input_data.shape
            out_shape = output_data.shape
            
            input_data = np.reshape(input_data, (in_shape[0] * in_shape[1], in_shape[2]))
            output_data = np.reshape(output_data, (out_shape[0] * out_shape[1], out_shape[2]))
            
            input_data = input_scaler.transform(input_data)
            output_data = output_scaler.transform(output_data)
            
            input_data = np.reshape(input_data, in_shape)
            output_data = np.reshape(output_data, out_shape)
            
            return input_data, output_data
        else:
            raise TypeError("The given scalers object is not of type InOutScalers:\n"+str(type(scalers)))
            
class KerasManager ():
    MODEL_SAVE_BASE = "MODEL_"
    HISTORY_FILENAME = "history.json"
    
    def __init__ (self, manager_dir):
        manager_dir = osp.abspath(manager_dir)
        os.makedirs(manager_dir, exist_ok=True)
        
        self.__manager_dir = manager_dir
        
    def manager_dir (self):
        return self.__manager_dir
    
    def save_model (self, model_name, model_epoch, model, history=None):
        model_dir = osp.join(self.__manager_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        savedir = osp.join(model_dir, KerasManager.MODEL_SAVE_BASE+str(model_epoch))
        model.save(savedir)
        
        if history is not None:
            history_savepath = osp.join(savedir, KerasManager.HISTORY_FILENAME)
            
            with open(history_savepath, "w") as f:
                json.dump(history.history, f, indent=3)
        
    def has_model (self, model_name):
        model_dir = osp.join(self.__manager_dir, model_name)
        
        if osp.isdir(model_dir):
            subdirs = [
                    osp.join(model_dir, x)
                    for x in os.listdir(model_dir)
                    if osp.isdir(osp.join(model_dir, x)) and x.startswith(KerasManager.MODEL_SAVE_BASE)
                ]
            epochs = [
                    int(osp.basename(x).replace(KerasManager.MODEL_SAVE_BASE, ""))
                    for x in subdirs
                ]
            
            return len(epochs) != 0
        else:
            return False
        
    def load_model (self, model_name):
        model_dir = osp.join(self.__manager_dir, model_name)
        
        if osp.isdir(model_dir):
            subdirs = [
                    osp.join(model_dir, x)
                    for x in os.listdir(model_dir)
                    if osp.isdir(osp.join(model_dir, x)) and x.startswith(KerasManager.MODEL_SAVE_BASE)
                ]
            epochs = [
                    int(osp.basename(x).replace(KerasManager.MODEL_SAVE_BASE, ""))
                    for x in subdirs
                ]
            
            if len(epochs) == 0:
                return None, None
            else:
                highest_index = np.argmax(epochs)
                epoch = epochs[highest_index]
                path = subdirs[highest_index]
                model = load_model(path)
                
                return epoch, model
        else:
            return None, None
        
    def train (self, epoch, model, input_data, output_data, **kwargs):
        if output_data is not None:
            history = model.fit(input_data, output_data, initial_epoch=epoch, **kwargs)
        else:
            history = model.fit_generator(input_data, initial_epoch=epoch, **kwargs)
            
        new_epoch = epoch + len(history.history["loss"])
        
        return new_epoch, history
        
    def train_save (self, epoch, model_name, model, input_data, output_data, **kwargs):
        new_epoch, history = self.train(epoch, model, input_data, output_data, **kwargs)
        self.save_model(model_name, new_epoch, model, history)
        
        return new_epoch, history
        
    def has_file_path (self, file_path):
        full_path = osp.join(self.__manager_dir, file_path)
        return osp.isfile(full_path)
    
    def has_directory_path (self, dir_path):
        full_path = osp.join(self.__manager_dir, dir_path)
        return osp.isdir(full_path)
        
    def get_file_path (self, file_path):
        full_path = osp.join(self.__manager_dir, file_path)
        
        if osp.isfile(full_path):
            return full_path
        else:
            return None
        
    def get_directory_path (self, dir_path):
        full_path = osp.join(self.__manager_dir, dir_path)
        
        if osp.isdir(full_path):
            return full_path
        else:
            return None
        
        
        
        