import os
import argparse
import urllib
import import_smiles

DEFAULT_URI = 'http://zinc.docking.org/db/bysubset/13/13_prop.xls'
DEFAULT_OUTPUT_FILE = 'zinc_druglikeclean_smiles.h5'

def get_arguments():
    parser = argparse.ArgumentParser(description='Download ZINC drug-like-clean SMILES entries and convert them to input for preprocessing')
    parser.add_argument('--uri', type=str, default=DEFAULT_URI,
                        help = 'URI to download ZINC SMILES entries from')
    parser.add_argument('--outfile', type=str, help='Output file name', default = DEFAULT_OUTPUT_FILE)
    return parser.parse_args()

def main():
    args = get_arguments()
    fname = os.path.basename(args.uri)

    urllib.urlretrieve(args.uri, fname)

    d = import_smiles.read_smiles(fname, column=10)
    import_smiles.create_h5(d, args.outfile)

if __name__ == '__main__':
    main()
