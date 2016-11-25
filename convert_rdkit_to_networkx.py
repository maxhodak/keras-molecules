from __future__ import print_function

import networkx as nx
import argparse
import multiprocessing
from rdkit import Chem

NUM_PROCESSES = 8

def get_arguments():
    parser = argparse.ArgumentParser(description='Convert an rdkit Mol to nx graph, preserving chemical attributes')
    parser.add_argument('smiles', type=str, help='The input file containing SMILES strings representing an input molecules.')
    parser.add_argument('nx_pickle', type=str, help='The output file containing sequence of pickled nx graphs')
    parser.add_argument('--num_processes', type=int, default=NUM_PROCESSES, help='The number of concurrent processes to use when converting.')
    return parser.parse_args()
    
def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol

def do_all(smiles, validate=False):
    mol = Chem.MolFromSmiles(smiles.strip())
    can_smi = Chem.MolToSmiles(mol)
    G = mol_to_nx(mol)
    if validate:
        mol = nx_to_mol(G)
        new_smi = Chem.MolToSmiles(mol)
        assert new_smi == smiles
    return G

def main():
    args = get_arguments()
    i = open(args.smiles)
    p = multiprocessing.Pool(args.num_processes)
    results = p.map(do_all, i.xreadlines())
    o = open(args.nx_pickle, 'w')
    for result in results:
        nx.write_gpickle(result, o)
    o.close()

if __name__ == '__main__':
    main()
