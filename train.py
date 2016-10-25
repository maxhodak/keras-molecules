from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from autoencoder.model import MoleculeVAE
from autoencoder.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

NUM_EPOCHS = 1
BATCH_SIZE = 600
LATENT_DIM = 292

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    return parser.parse_args()

def train(network, data_train, data_test, epochs, batch_size, callbacks=[], shuffle = True):
    network.fit(data_train, data_train,
                shuffle = shuffle,
                nb_epoch = epochs,
                batch_size = batch_size,
                callbacks = callbacks,
                validation_data = (data_test, data_test))

def main():
    args = get_arguments()
    data_train, data_test, charset = load_dataset(args.data)
    model = MoleculeVAE()
    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim)
    else:
        model.create(charset, latent_rep_size = args.latent_dim)
    
    checkpointer = ModelCheckpoint(filepath = args.model,
                                   verbose = 1,
                                   save_best_only = True)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    train(model.autoencoder,
          data_train,
          data_test,
          args.epochs,
          args.batch_size,
          callbacks = [checkpointer, reduce_lr])

if __name__ == '__main__':
    main()
