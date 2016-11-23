from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import pandas as pd

from molecules.model import MoleculeVAE
from molecules.vectorizer import SmilesDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

NUM_EPOCHS = 1
EPOCH_SIZE = 500000
BATCH_SIZE = 600
LATENT_DIM = 292
MAX_LEN = 120

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing structures.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--epoch_size', type=int, metavar='N', default=EPOCH_SIZE,
                        help='Number of samples to process per epoch during training.')
    return parser.parse_args()

def main():
    args = get_arguments()

    data = pd.read_hdf(args.data, 'table')
    structures = data['structure']

    # import gzip
    # filepath = args.data
    # structures = [line.split()[0].strip() for line in gzip.open(filepath) if line]

    # can also use CanonicalSmilesDataGenerator
    datobj = SmilesDataGenerator(structures, MAX_LEN)
    test_divisor = int((1 - datobj.test_split) / (datobj.test_split))
    train_gen = datobj.train_generator(args.batch_size)
    test_gen = datobj.test_generator(args.batch_size)

    # reformulate generators to not use weights
    train_gen = ((tens, tens) for (tens, _, weights) in train_gen)
    test_gen = ((tens, tens) for (tens, _, weights) in test_gen)

    model = MoleculeVAE()
    if os.path.isfile(args.model):
        model.load(datobj.chars, args.model, latent_rep_size = args.latent_dim)
    else:
        model.create(datobj.chars, latent_rep_size = args.latent_dim)

    checkpointer = ModelCheckpoint(filepath = args.model,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    model.autoencoder.fit_generator(
        train_gen,
        args.epoch_size,
        nb_epoch = args.epochs,
        callbacks = [checkpointer, reduce_lr],
        validation_data = test_gen,
        nb_val_samples = args.epoch_size / test_divisor,
        pickle_safe = True
    )

if __name__ == '__main__':
    main()
