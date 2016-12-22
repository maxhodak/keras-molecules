from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import pandas as pd

from molecules.vectorizer import SmilesDataGenerator

NUM_EPOCHS = 1
EPOCH_SIZE = 500000
BATCH_SIZE = 500
LATENT_DIM = 292
MAX_LEN = 120
TEST_SPLIT = 0.20
RANDOM_SEED = 1337

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
    parser.add_argument('--test_split', type=float, metavar='N', default=TEST_SPLIT,
                        help='Fraction of dataset to use as test data, rest is training data.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
                        help='Seed to use to start randomizer for shuffling.')
    return parser.parse_args()

def main():
    args = get_arguments()
    np.random.seed(args.random_seed)
    
    from molecules.model import MoleculeVAE
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    
    data = pd.read_hdf(args.data, 'table')
    structures = data['structure']

    # import gzip
    # filepath = args.data
    # structures = [line.split()[0].strip() for line in gzip.open(filepath) if line]

    # can also use CanonicalSmilesDataGenerator
    datobj = SmilesDataGenerator(structures, MAX_LEN,
                                 test_split=args.test_split,
                                 random_seed=args.random_seed)
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
