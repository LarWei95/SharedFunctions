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

class DataBalancer ():
    
    @classmethod
    def _create_bins (cls, y, bintype, bincount):
        minval = np.min(y)
        maxval = np.max(y)
        
        if bintype == "linear":
            bins = np.linspace(minval, maxval, bincount+1)
        else:
            minval = np.log(minval) / np.log(10)
            maxval = np.log(maxval) / np.log(10)
            
            bins = np.logspace(minval, maxval, bincount+1)
            
        return bins
    
    @classmethod
    def _associate_bins (cls, y, bins):
        associations = np.empty(len(y), dtype=np.int)
        counts = np.empty(len(bins) - 1, dtype=np.int)
        
        maxi = len(bins) - 1
        
        for i in range(maxi):
            start = bins[i]
            end = bins[i+1]
            
            if i == 0:
                mask = y < end
            elif i == maxi - 1:
                mask = y >= start
            else:
                mask = (y >= start) & (y < end)
               
            associations[mask] = i
            counts[i] = np.sum(mask)
        
        return associations, counts
            
    @classmethod
    def _find_optimal_bins (cls, y, bintype, bincount_start):
        bincount = bincount_start
        
        bins = cls._create_bins(y, bintype, bincount)
        associations, counts = cls._associate_bins(y, bins)
        
        while True:
            if np.min(counts) > 0:
                print("Best bin count: "+str(bincount))
                return associations
            else:
                bincount -= 1
                bins = cls._create_bins(y, bintype, bincount)
                associations, counts = cls._associate_bins(y, bins)
    
    @classmethod
    def _simple_balance (cls, x, y, uniques, counts, indices):
        maxcount = np.max(counts)
        diffs = maxcount - counts
        
        x_concat = [x]
        y_concat = [y]
        
        for unique_index in range(len(uniques)):
            missing_count = diffs[unique_index]
            mask = indices == unique_index
            x_repeatables = x[mask]
            y_repeatables = y[mask]
            count = len(x_repeatables)
            
            float_repeats = (missing_count / count)
            direct_repeats = np.floor(float_repeats).astype(int)
            partial_repeats = np.round((float_repeats - direct_repeats) * count).astype(int)
                       
            if float_repeats > 0:
                x_direct_repeats = np.repeat(x_repeatables, direct_repeats, axis=0)
                y_direct_repeats = np.repeat(y_repeatables, direct_repeats, axis=0)
                
                x_partial_repeats = x_repeatables[:partial_repeats]
                y_partial_repeats = y_repeatables[:partial_repeats]
                
                x_concat.append(x_direct_repeats)
                y_concat.append(y_direct_repeats)
                
                x_concat.append(x_partial_repeats)
                y_concat.append(y_partial_repeats)
            
        x_concat = np.concatenate(x_concat, axis=0)
        y_concat = np.concatenate(y_concat, axis=0)
        
        return x_concat, y_concat
    
    @classmethod
    def _get_subindices (cls, y, bincount_start, bintype):
        all_subindices = []
        
        for i in range(y.shape[1]):
            suby = y[:,i]
            
            associations = cls._find_optimal_bins(suby, bintype, bincount_start)
            
            all_subindices.append(associations)
            
        stacked_subindices = np.stack(all_subindices, axis=1)
        
        return stacked_subindices
    
    @classmethod
    def oversample_regression (cls, x, y, bincount_start, bintype="linear",
                            advanced_balance=False):
        subindices = cls._get_subindices(y, bincount_start, bintype)
        uniques, indices, counts = np.unique(subindices, 
                                             return_inverse=True, 
                                             return_counts=True,
                                             axis=0)        
        
        x, y = cls._simple_balance(x, y, uniques, counts, indices)
        
        return x, y
    
    @classmethod
    def _simple_sample_weights (cls, y, uniques, counts, indices):
        maxcount = np.max(counts)
        
        bin_weights = counts / maxcount
        bin_weights = 1 / bin_weights
        
        sample_weights = np.empty(len(y), dtype=np.float)
        
        for unique_index in range(len(bin_weights)):
            weight = bin_weights[unique_index]
            
            mask = indices == unique_index
            sample_weights[mask] = weight
            
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
            
        print(np.sum(sample_weights), len(sample_weights))
            
        return sample_weights
            
        
    
    @classmethod
    def sample_weight_regression (cls, y, bincount_start, bintype="linear",
                                  advanced_balance=False):
        subindices = cls._get_subindices(y, bincount_start, bintype)
        uniques, indices, counts = np.unique(subindices, 
                                             return_inverse=True, 
                                             return_counts=True,
                                             axis=0)   
        return cls._simple_sample_weights(y, uniques, counts, indices)

class WeightLoggerCallback(Callback):
    def __init__ (self, model, logname, logstep=1):
        self.__model = model
        self.__logname = logname
        self.__fp = open(self.__logname, "w")
        self.__logstep = logstep
        
        
        
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.__logstep == 0:
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
            print(input_data.shape, output_data.shape)
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
        
        
        
        