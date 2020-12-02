#%%
import os
import pandas as pd
import numpy as np

from skmultilearn.dataset import load_dataset

import tensorflow as tf
import tensorflow.keras as K
#%%
class Feature_Encoder(K.layers.Layer):
    def __init__(self, dim_model):
        super(Feature_Encoder, self).__init__()
        self.n_layer = len(dim_model)
        self.dim_model = dim_model
        self.encoder = [K.layers.Dense(self.dim_model[i], activation=tf.keras.layers.LeakyReLU(alpha=0.3)) for i in range(self.n_layer)]
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'encoder': self.encoder
        })
        return config
        
    def call(self, x):
        for i in range(self.n_layer):
            x = self.encoder[i](x)
        return x
#%%
class Label_Encoder(K.layers.Layer):
    def __init__(self, dim_model):
        super(Label_Encoder, self).__init__()
        self.n_layer = len(dim_model)
        self.dim_model = dim_model
        self.encoder = [K.layers.Dense(self.dim_model[i], activation=tf.keras.layers.LeakyReLU(alpha=0.3)) for i in range(self.n_layer)]
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'encoder': self.encoder
        })
        return config
        
    def call(self, x):
        for i in range(self.n_layer):
            x = self.encoder[i](x)
        return x
#%%
class Label_Decoder(K.layers.Layer):
    def __init__(self, dim_model):
        super(Label_Decoder, self).__init__()
        self.n_layer = len(dim_model)
        self.dim_model = dim_model
        self.decoder = [K.layers.Dense(self.dim_model[i], activation=tf.keras.layers.LeakyReLU(alpha=0.3)) for i in range(self.n_layer)]
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'decoder': self.decoder
        })
        return config
        
    def call(self, x):
        for i in range(self.n_layer):
            x = self.decoder[i](x)
        return x
#%%
class C2AE(K.layers.Layer):
    def __init__(self, feature_encoder_dim, label_encoder_dim, label_decoder_dim):
        super(C2AE, self).__init__()
        self.feature_encoder = Feature_Encoder(feature_encoder_dim)
        self.label_encoder = Label_Encoder(label_encoder_dim)
        self.label_decoder = Label_Decoder(label_decoder_dim)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'feature_encoder': self.feature_encoder,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder
        })
        
        return config
    
    def call(self, input):
        feature, label = input
        label = tf.constant(label, dtype=tf.float32)
        emb_feature = self.feature_encoder(feature)
        emb_label = self.label_encoder(label)
        recon_label =  self.label_decoder(emb_label)    
        #self.add_loss(tf.math.reduce_sum(tf.map_fn(fn=lambda t: tf.math.divide(tf.math.reduce_sum(tf.exp(tf.reshape(t[0][(t[0] * t[1]) == 0], (np.sum((t[0] * t[1]) == 0), 1)) - tf.reshape(t[0][(t[0] * t[1]) != 0], (1, np.sum((t[0] * t[1]) != 0))))) , tf.math.reduce_sum(tf.cast(t[1] == 0, tf.float32)) * tf.math.reduce_sum(tf.cast(t[1] != 0, tf.float32))), elems= (recon_label, label), fn_output_signature=tf.float32)))
        
        return emb_feature, emb_label, recon_label
#%%
class CCA_Loss(K.models.Model):
    def __init__(self, regularization_factor=0.1, name="cca loss"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, emb_feature, emb_label):
        num_instance = emb_feature.shape[0]
        c1 = emb_feature - emb_label
        c2 = tf.linalg.matmul(emb_feature, emb_feature, transpose_b=True) - tf.eye(num_instance)
        c3 = tf.linalg.matmul(emb_label, emb_label, transpose_b=True) - tf.eye(num_instance)
        loss = tf.linalg.trace(tf.matmul(c1, c1, transpose_a=True)) + self.regularization_factor * tf.linalg.trace(tf.matmul(c2, c2, transpose_a=True) +
                                                                                      tf.matmul(c3, c3, transpose_a=True))
        return loss
#%%
X, y, feature_names, label_names = load_dataset('tmc2007_500', 'train')
#%%
train_X = np.array(X.todense(), dtype=np.float32)[0:1000]
train_Y = np.array(y.todense(), dtype=np.float32)[0:1000]
#%%
c2ae = C2AE([512, 512], [512], [22])
cl = CustomLoss()
max_iter = 200
# optimizer = K.optimizers.Adam(learning_rate=0.001)
optimizer = K.optimizers.RMSprop(learning_rate=0.001)
#%%
tmc2007 = tf.data.Dataset.from_tensor_slices((np.array(X.todense(), dtype=np.float32),np.array(y.todense(), dtype=np.float32)))
#%%
BATCH_SIZE = 500
tmc_data = tmc2007.shuffle(buffer_size=15000)
tmc_data_batch = tmc_data.batch(BATCH_SIZE)
#%%
EPOCHS = 100
for i in range(EPOCHS):
    epoch_cca_loss = 0
    epoch_acc_loss = 0
    for feature, label in iter(tmc_data_batch):
        with tf.GradientTape() as tape:    
            a,b,c = c2ae([feature, label])
            accuracy_loss = tf.math.reduce_sum(tf.map_fn(fn=lambda t: tf.math.divide(tf.math.reduce_sum(tf.exp(tf.reshape(t[0][(t[0] * t[1]) == 0], (np.sum((t[0] * t[1]) == 0), 1)) - tf.reshape(t[0][(t[0] * t[1]) != 0], (1, np.sum((t[0] * t[1]) != 0))))) , tf.math.reduce_sum(tf.cast(t[1] == 0, tf.float32)) * tf.math.reduce_sum(tf.cast(t[1] != 0, tf.float32))), elems= (c, label), fn_output_signature=tf.float32))
            cca_loss = cl(a, b)
            loss = cca_loss + 1 * accuracy_loss 
            epoch_cca_loss = epoch_cca_loss + cca_loss
            epoch_acc_loss = epoch_acc_loss + accuracy_loss
            total_loss = epoch_cca_loss + epoch_acc_loss
        grad = tape.gradient(loss, c2ae.weights)
        optimizer.apply_gradients(zip(grad, c2ae.weights))
    print("iteration {:03d}: Total Loss {:.04f}, CCA Loss {:.04f}, Accuracy Loss {:.04f}".format(i+1, total_loss.numpy(), epoch_cca_loss.numpy(), epoch_acc_loss.numpy()))