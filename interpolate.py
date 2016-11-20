from __future__ import print_function

import h5py
import numpy
import os
import argparse
import sample
from molecules.model import MoleculeVAE
from molecules.utils import decode_smiles_from_indexes
from molecules.utils import one_hot_array, one_hot_index

SOURCE = 'Cc1ccnc(c1)NC(=O)Cc2cccc3c2cccc3'
DEST = 'c1cc(cc(c1)Cl)NNC(=O)c2cc(cnc2)Br'
LATENT_DIM = 292
STEPS = 100
WIDTH = 120

def get_arguments():
    parser = argparse.ArgumentParser(description='Interpolate from source to dest in steps')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    parser.add_argument('--source', type=str, default=SOURCE,
                        help='Source SMILES string for interpolation')
    parser.add_argument('--dest', type=str, default=DEST,
                        help='Source SMILES string for interpolation')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--width', type=int, default=WIDTH,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--steps', type=int, default=STEPS,
                        help='Number of steps to take while interpolating between source and dest')
    return parser.parse_args()

def interpolate(source, dest, steps, charset, model, latent_dim, width):
    source_just = source.ljust(width)
    dest_just = dest.ljust(width)
    one_hot_encoded_fn = lambda row: map(lambda x: one_hot_array(x, len(charset)),
                                                one_hot_index(row, charset))
    source_encoded = numpy.array(map(one_hot_encoded_fn, source_just))
    source_x_latent = model.encoder.predict(source_encoded.reshape(1, width, len(charset)))
    dest_encoded = numpy.array(map(one_hot_encoded_fn, dest_just))
    dest_x_latent = model.encoder.predict(dest_encoded.reshape(1, width, len(charset)))

    step = (dest_x_latent - source_x_latent)/float(steps)
    results = []
    for i in range(steps):
        item = source_x_latent + (step  * i)
        sampled = model.decoder.predict(item.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset)
        results.append( (i, item, sampled) )

    return results

def main():
    args = get_arguments()

    if os.path.isfile(args.data):
        h5f = h5py.File(args.data, 'r')
        charset = list(h5f['charset'][:])
        h5f.close()
    else:
        raise ValueError("Data file %s doesn't exist" % args.data)

    model = MoleculeVAE()
    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    results = interpolate(args.source, args.dest, args.steps, charset, model, args.latent_dim, args.width)
    for result in results:
        print(result[0], result[2])

if __name__ == '__main__':
    main()
