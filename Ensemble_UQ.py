import numpy as np
import theano.tensor as tt
from theano import function, shared

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import struct
import scipy.stats as stat

import pickle as pkl

from os import listdir
from copy import deepcopy
import imageio

def pairs(it):
    for k in range(len(it)-1):
        yield it[k], it[k+1]
        
def log_print(msg, path=None, verbose=1):
    if path is None:
        if verbose > 0:
            print('\r' + msg, end='')
    else:
        try:
            with open(path, 'a') as f:
                f.write(msg + '\n')
        except FileNotFoundError:
            with open(path, 'r') as f:
                f.write(msg + '\n')
        if verbose > 0:
            print('\r' + msg, end='')
            
class FFNClassifier:
    def __init__(self, layers=[10,10,10], nonlinearity='relu'):
        lr = tt.dscalar()
        x = tt.dmatrix()
        y_true = tt.dmatrix()
        self.weights = [shared(np.random.randn(in_dim, out_dim)*2/np.sqrt(in_dim)) for in_dim, out_dim in pairs(layers)]
        self.biases = [shared(np.zeros(out_dim)) for out_dim in layers[1:]]
        layer = x
        for w,b in zip(self.weights[:-1], self.biases[:-1]):
            layer = tt.nnet.relu(tt.dot(layer, w) + b)
        out = tt.nnet.softmax(tt.dot(layer, self.weights[-1])+self.biases[-1])
        error = tt.mean(tt.nnet.categorical_crossentropy(out, y_true))
        
        weight_grads = tt.grad(error, self.weights)
        bias_grads = tt.grad(error, self.biases)
        
        self.predict = function(inputs=[x], outputs=out)
        self.update = function(inputs=[x, y_true, lr],
                               updates=[(w, w-lr*g_w) for w, g_w in zip(self.weights, weight_grads)] + \
                                       [(b,b-lr*g_b) for b, g_b in zip(self.biases, bias_grads)])
        self.mean_error = function(inputs=[x,y_true], outputs=error)
        
    def fit(self, data, train_split='train', tot_batches=10000, 
            valid_skip=100, reduction_test_len=7, l_rate=0.005, 
            verbose=0, log_file=None):        
        tests_since_reduction = 0
        errors = []
        l_rates = [l_rate]
        for b in range(tot_batches):
            if b%valid_skip == 0:
                if train_split == 'train':
                    errors.append(self.mean_error(data.x_valid, data.y_valid))
                elif train_split == 'anomaly':
                    errors.append(self.mean_error(data.x_valid_anomaly, data.y_valid_anomaly))
                else:
                    raise NotImplementedError
                
                if tests_since_reduction >= reduction_test_len:
                    slope, intcpt, r, p, stderr = stat.linregress(np.arange(reduction_test_len), errors[-reduction_test_len:])
                    if slope > 0:
                        l_rate = 0.5*l_rate
                        log_print('Reducing learning rate to {}'.format(l_rate), path=log_file, verbose=verbose)
                        tests_since_reduction = 0
                else:
                    tests_since_reduction += 1
                    
                l_rates.append(l_rate)
            
                log_print('Batch {0}, starting error {1:1.5f}, learning rate {2:1.5f}'.format(b, errors[-1], l_rate), path=log_file, verbose=verbose)
                
            x_batch, y_batch = data.batch(train_split)
            self.update(x_batch, y_batch, l_rate)
        
        if train_split == 'train':
            final_error = self.mean_error(data.x_valid, data.y_valid)
        elif train_split == 'anomaly':
            final_error = self.mean_error(data.x_valid_anomaly, data.y_valid_anomaly)
        log_print('Final error is {0:1.5f} after {1} batches.'.format(final_error, tot_batches), path=log_file, verbose=verbose)
        return errors, l_rates
    
class MNISTData:
    def __init__(self, splits=['train', 'test', 'ood'], valid_size=600, batch_size=600, ood_size=1000):
        self.batch_size = batch_size
        self.pointer = 0
        self.aleatoric_exists = False
        
        if 'train' in splits:
            with open(r'C:\Projects\UQ\datasets\train-images-idx3-ubyte\train-images.idx3-ubyte', 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                nrows, ncols = struct.unpack(">II", file.read(8))
                x_train = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
                x_train = x_train.reshape((size,-1))
                x_train = x_train.astype(float)
                p = np.random.permutation(size)
                x_train = x_train[p,:]
                
            self.x_train = x_train[:-valid_size,:]
            self.x_valid = x_train[-valid_size:,:]
            
            with open(r'C:\Projects\UQ\datasets\train-labels-idx1-ubyte\train-labels.idx1-ubyte', 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                y_train_idx = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
                
            y_train_idx = y_train_idx[p]
            
            self.y_idx_train = y_train_idx[:-valid_size]
            self.y_idx_valid = y_train_idx[-valid_size:]
            
            size -= valid_size
            
            self.y_train = np.zeros((size, 10))
            self.y_train[np.arange(size), self.y_idx_train] = 1
            
            self.y_valid = np.zeros((valid_size, 10))
            self.y_valid[np.arange(valid_size), self.y_idx_valid] = 1
        
        if 'test' in splits:
            with open(r'C:\Projects\UQ\datasets\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte', 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                nrows, ncols = struct.unpack(">II", file.read(8))
                x_test = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
                x_test = x_test.reshape((size, -1))
                x_test = x_test.astype(float)
                
            self.x_test = x_test
            
            with open(r'C:\Projects\UQ\datasets\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte', 'rb') as file:
                magic, size = struct.unpack(">II", file.read(8))
                y_test_idx = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))

            self.y_idx_test = y_test_idx
            self.y_test = np.zeros((size, 10))
            self.y_test[np.arange(size), self.y_idx_test] = 1
        
        if 'ood' in splits:
            path = r'C:/Projects/UQ/datasets/notMNIST_large.tar/notMNIST_large/A/'
            nMNIST_A_filenames = listdir(path)
            file_idxs = np.random.randint(0, len(nMNIST_A_filenames), ood_size)
            filenames = [nMNIST_A_filenames[k] for k in file_idxs]

            ims = []
            for im_file in filenames:
                im = imageio.imread(r'C:/Projects/UQ/datasets/notMNIST_large.tar/notMNIST_large/A/'+im_file)
                ims.append(np.array(im))
            
            ims = np.array(ims)
            ims = ims.reshape((ood_size, -1))
            
            self.ood = ims.astype(float)
    
    def create_mixed_ood(self, sizes=[1000,1000]):
        self.shuffle('test')
        self.shuffle('ood')
        self.x_mixed = np.concatenate([self.x_test[:sizes[0],:], self.ood[:sizes[1],:]], axis=0)
        self.y_mixed = np.concatenate([np.zeros(sizes[0]), np.ones(sizes[1])], axis=0)
        p = np.random.permutation(sum(sizes))
        self.x_mixed = self.x_mixed[p,:]
        self.y_mixed = self.y_mixed[p]
        
    def remove_label(self, label=0):
        label_train_idxs = self.y_idx_train != label
        label_valid_idxs = self.y_idx_valid != label
        label_test_idxs = self.y_idx_test != label
        
        self.x_train_anomaly = self.x_train[label_train_idxs]
        self.y_idx_train_anomaly = self.y_idx_train[label_train_idxs]
        self.y_train_anomaly = self.y_train[label_train_idxs][:,[k for k in range(10) if k != label]]
        
        self.x_valid_anomaly = self.x_valid[label_valid_idxs]
        self.y_idx_valid_anomaly = self.y_idx_valid[label_valid_idxs]
        self.y_valid_anomaly = self.y_valid[label_valid_idxs][:,[k for k in range(10) if k != label]]
        
        self.x_test_anomaly = self.x_test
        self.y_test_anomaly = np.ones(len(self.y_idx_test))
        self.y_test_anomaly[label_test_idxs] = 0
        
        
    def create_aleatoric_labels(self, randomization=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], randomize_test=False):
        if randomize_test:
            raise NotImplementedError
        
        self.y_idx_aleatoric = deepcopy(self.y_idx_train)
        for k in range(10):
            size = (self.y_idx_train == k).sum()
            keep = np.random.rand(size) > randomization[k]
            self.y_idx_aleatoric[self.y_idx_train == k] = np.where(keep, 
                                                                   self.y_idx_aleatoric[self.y_idx_train == k],
                                                                   np.random.randint(0,10,size))
        self.y_aleatoric = np.zeros((len(self.y_idx_aleatoric), 10))
        self.y_aleatoric[np.arange(len(self.y_idx_aleatoric)), self.y_idx_aleatoric] = 1
        self.aleatoric_exists = True 
    
    def batch(self, split='train'):
        if (split == 'train')|(split == 'aleatoric'):
            size = self.x_train.shape[0]
        elif split == 'anomaly':
            size = self.x_train_anomaly.shape[0]
        else:
            raise NotImplementedError
        
        if self.pointer + self.batch_size > size:
            self.shuffle(split)
            self.pointer = 0
        self.pointer += self.batch_size
        
        if (split == 'train')|(split == 'aleatoric'):
            x_batch = self.x_train[self.pointer-self.batch_size:self.pointer,:]
        elif split == 'anomaly':
            x_batch = self.x_train_anomaly[self.pointer-self.batch_size:self.pointer,:]
        
        if split == 'train':
            y_batch = self.y_train[self.pointer-self.batch_size:self.pointer, :]
        elif split == 'aleatoric':
            if not self.aleatoric_exists: raise NotImplementedError
            y_batch = self.y_aleatoric[self.pointer-self.batch_size:self.pointer, :]
        elif split == 'anomaly':
            y_batch = self.y_train_anomaly[self.pointer-self.batch_size:self.pointer, :]
        else:
            raise NotImplementedError
            
        return x_batch, y_batch    
    
    def shuffle(self, split='train'):
        if split == 'train':
            nrecords = len(self.y_idx_train)
        elif split == 'test':
            nrecords = len(self.y_idx_test)
        elif split == 'ood':
            nrecords = self.ood.shape[0]
        elif split == 'anomaly':
            nrecords = len(self.y_idx_train_anomaly)
        else:
            raise NotImplementedError
        
        p = np.random.permutation(nrecords)
        
        if split == 'train':
            self.x_train = self.x_train[p,:]
            self.y_idx_train = self.y_idx_train[p]
            self.y_train = self.y_train[p,:]
            
            if self.aleatoric_exists:
                self.y_idx_aleatoric = self.y_idx_aleatoric[p]
                self.y_aleatoric = self.y_aleatoric[p,:]
        
        elif split == 'anomaly':
            self.x_train_anomaly = self.x_train_anomaly[p,:]
            self.y_idx_train_anomaly = self.y_idx_train_anomaly[p]
            self.y_train_anomaly = self.y_train_anomaly[p,:]
        
        elif split == 'test':
            self.x_test = self.x_test[p,:]
            self.y_idx_test = self.y_idx_test[p]
            self.y_test = self.y_test[p,:]
            
        elif split == 'ood':
            self.ood = self.ood[p,:]
            

class Ensemble:
    def __init__(self, models):
        self.models = models
        
    def multi_predict(self, x_data):
        preds = np.array([m.predict(x_data) for m in self.models])
        return preds
        
    def predict(self, x_data):
        preds = self.multi_predict(x_data)
        mean_pred = preds.mean(axis=0)
        return mean_pred
    
    def entropies(self, x_data=None, preds=None):
        if preds is None:
            preds = self.multi_predict(x_data)
        mean_pred = preds.mean(axis=0)
        return (-np.where(mean_pred != 0, np.log(mean_pred), 0.)*mean_pred).sum(axis=1)
    
    def divergence(self, x_data=None, preds=None):
        def kl(p,q):
            unsummed = np.where(p!=0, p*np.log(p/q), 0)
            div = unsummed.sum(axis=1)
            return div
        if preds is None:
            preds = self.multi_predict(x_data)
        mean_preds = preds.mean(axis=0)
        divs = np.array([kl(preds[k,:,:], mean_preds) for k in range(len(self.models))])
        m_div = divs.mean(axis=0)
        return m_div

    
def gains_auc(y_true, y_pred):
    order = y_pred.argsort()[::-1]
    gains = (y_true[order].cumsum())/(y_true.sum())
    auc = gains.mean()
    return gains, auc