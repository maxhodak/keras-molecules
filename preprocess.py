import argparse
import pandas
import h5py
import numpy as np
from autoencoder.utils import one_hot_array, one_hot_index

from sklearn.cross_validation import train_test_split

MAX_NUM_ROWS = 500000
SMILES_COL_NAME = 'structure'
PROPERTY_COL_NAME = None #'standard_value'

def get_arguments():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('infile', type=str, help='Input file name')
    parser.add_argument('outfile', type=str, help='Output file name')
    parser.add_argument('--length', type=int, metavar='N', default = MAX_NUM_ROWS,
                        help='Maximum number of rows to include (randomly sampled).')
    parser.add_argument('--smiles_column', type=str, default = SMILES_COL_NAME,
                        help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument('--property_column', type=str, default = PROPERTY_COL_NAME,
                        help="Name of the column that contains the property values to predict. Default: %s" % \
                        PROPERTY_COL_NAME)
    return parser.parse_args()

def main():
    args = get_arguments()
    data = pandas.read_hdf(args.infile, 'table')
    keys = data[args.smiles_column].map(len) < 121

    if args.length <= len(keys):
        data = data[keys].sample(n = args.length)
    else:
        data = data[keys]

    structures = data[args.smiles_column].map(lambda x: list(x.ljust(120)))

    if args.property_column:
        properties = data[args.property_column][keys]

    del data

    train_idx, test_idx = map(np.array,
                              train_test_split(structures.index, test_size = 0.20))

    charset = list(reduce(lambda x, y: set(y) | x, structures, set()))

    one_hot_encoded = np.array(
        map(lambda row:
            map(lambda x: one_hot_array(x, len(charset)),
                one_hot_index(row, charset)),
            structures))

    h5f = h5py.File(args.outfile, 'w')
    h5f.create_dataset('charset', data = charset)
    h5f.create_dataset('data_train', data = one_hot_encoded[train_idx])
    h5f.create_dataset('data_test', data = one_hot_encoded[test_idx])

    if args.property_column:
        h5f.create_dataset('property_train', data = properties[train_idx])
        h5f.create_dataset('property_test', data = properties[test_idx])
    h5f.close()

if __name__ == '__main__':
    main()
