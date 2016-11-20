import pyparsing as pp

class AST:
  """Abstract Syntax Tree for SMILES string parsed form."""
  class SMILES:
    def __init__(self, smiles):
      self.smiles = smiles
    def __getitem__(self, item):
      if item == 0:
        return self.smiles
      else:
        raise IndexError
    def __repr__(self):
      return ''.join(map(str, self.smiles))

  class Chain:
    def __init__(self, chain):
      self.chain = chain
    def __getitem__(self, item):
      if item == 0:
        return self.chain
      else:
        raise IndexError
    def __repr__(self):
      return ''.join(map(str, self.chain))

  class Branch:
    def __init__(self, branch):
      self.branch = branch
    def __getitem__(self, item):
      if item == 0:
        return self.branch
      else:
        raise IndexError
    def __repr__(self):
      return '(' + ''.join(map(str, self.branch)) + ')'

  class Atom:
    def __init__(self, atom):
      self.atom = atom
    def __getitem__(self, item):
      if item == 0:
        return self.atom
      else:
        raise IndexError
    def __repr__(self):
      return str(self.atom)

  class Bond:
    def __init__(self, bond):
      self.bond = bond
    def __getitem__(self, item):
      if item == 0:
        return self.bond
      else:
        raise IndexError
    def __repr__(self):
      return str(self.bond)

  class OrganicSymbol:
    def __init__(self, organic_symbol):
      self.organic_symbol = organic_symbol
    def __getitem__(self, item):
      if item == 0:
        return self.organic_symbol
      else:
        raise IndexError
    def __repr__(self):
      return ''.join(map(str, self.organic_symbol))

  class AromaticSymbol:
    def __init__(self, aromatic_symbol):
      self.aromatic_symbol = aromatic_symbol
    def __getitem__(self, item):
      if item == 0:
        return self.atomatic_symbol
      else:
        raise IndexError
    def __repr__(self):
      return str(self.aromatic_symbol)

  class AtomSpec:
    def __init__(self, atom_spec):
      self.atom_spec = atom_spec
    def __getitem__(self, item):
      if item == 0:
        return self.atom_spec
      else:
        raise IndexError
    def __repr__(self):
      return '[' + ''.join(map(str,self.atom_spec)) + ']'

  class ElementSymbol:
    def __init__(self, element_symbol):
      self.element_symbol = element_symbol
    def __getitem__(self, item):
      if item == 0:
        return self.element_symbol
      else:
        raise IndexError
    def __repr__(self):
      return ''.join(map(str,self.element_symbol))

  class RingClosure:
    def __init__(self, ring_closure):
      self.ring_closure = ring_closure
    def __getitem__(self, item):
      if item == 0:
        return self.ring_closure
      else:
        raise IndexError
    def __repr__(self):
      return str(self.ring_closure)

  class ChiralClass:
    def __init__(self, chiral_class):
      self.chiral_class = chiral_class
    def __getitem__(self, item):
      if item == 0:
        return self.chiral_class
      else:
        raise IndexError
    def __repr__(self):
      return ''.join(map(str,self.chiral_class))

  class HCount:
    def __init__(self, hcount):
      self.hcount = hcount
    def __getitem__(self, item):
      if item == 0:
        return self.hcount
      else:
        raise IndexError
    def __repr__(self):
      return ''.join(map(str,self.hcount))

  class Charge:
    def __init__(self, charge):
      self.charge = charge
    def __getitem__(self, item):
      if item == 0:
        return self.charge
      else:
        raise IndexError
    def __repr__(self):
      return ''.join(map(str,self.charge))

  class Class:
    def __init__(self, class_):
      self.class_ = class_
    def __getitem__(self, item):
      if item == 0:
        return self.class_
      else:
        raise IndexError
    def __repr__(self):
      return str(self.class_)

  class Isotope:
    def __init__(self, isotope):
      self.items = isotope
    def __getitem__(self, item):
      if item == 0:
        return self.isotope
      else:
        raise IndexError
    def __repr__(self):
      return str(self.isotope)

# Wrapper functions from pyparsing parse actions to AST
def smiles_fn(s,l,t):
  return AST.SMILES(t[0])
def chain_fn(s,l,t):
  return AST.Chain(t[0])
def branch_fn(s,l,t):
  return AST.Branch(t[0])
def atom_fn(s,l,t):
  return AST.Atom(t[0])
def bond_fn(s,l,t):
  return AST.Bond(t[0])
def organic_symbol_fn(s,l,t):
  return AST.OrganicSymbol(t[0])
def aromatic_symbol_fn(s,l,t):
  return AST.AromaticSymbol(t[0])
def atom_spec_fn(s,l,t):
  return AST.AtomSpec(t[0])
def element_symbol_fn(s,l,t):
  return AST.ElementSymbol(t[0])
def ring_closure_fn(s,l,t):
  return AST.RingClosure(t[0])
def chiral_class_fn(s,l,t):
  return AST.ChiralClass(t[0])
def hcount_fn(s,l,t):
  return AST.HCount(t[0])
def charge_fn(s,l,t):
  return AST.Charge(t[0])
def class_fn(s,l,t):
  return AST.Class(t[0])
def isotope_fn(s,l,t):
  return AST.Isotope(t[0])

# Forward references for parsed elements
Atom = pp.Forward()
Chain = pp.Forward()
Branch = pp.Forward()
RingClosure = pp.Forward()
Bond = pp.Forward()
OrganicSymbol = pp.Forward()
AromaticSymbol = pp.Forward()
AtomSpec = pp.Forward()
WILDCARD = pp.Forward()
Isotope = pp.Forward()
ElementSymbol = pp.Forward()
ChiralClass = pp.Forward()
HCount = pp.Forward()
Charge = pp.Forward()
Class = pp.Forward()


# Parses a full SMILES string like C(C)C
SMILES = pp.Group(Atom + pp.ZeroOrMore(pp.Or([Chain, Branch])))
SMILES.setParseAction(smiles_fn)

# Parses a chain extension to a SMILES string like C or 1
Chain <<= pp.Group(pp.Optional(Bond) + pp.Or([Atom, RingClosure]))
Chain.setParseAction(chain_fn)

# Parses a branch off a SMILES string like (C)
Branch <<= pp.Literal('(').suppress() + pp.Group(pp.Optional(Bond) + pp.OneOrMore(SMILES)) + pp.Literal(')').suppress()
Branch.setParseAction(branch_fn)

# Parses an atom like C
Atom <<= pp.Or([OrganicSymbol, AromaticSymbol, AtomSpec, WILDCARD])
Atom.setParseAction(atom_fn)

# Parses a bond like =
Bond <<= pp.Or(map(pp.Literal, ['-', '=', '#', '$', ':', '/', '\\', '.']))
Bond.setParseAction(bond_fn)

# Parses an organic symbol like Br or C
OrganicSymbol <<= (pp.Group(pp.Literal('B') + pp.Optional(pp.Literal('r'))) | \
                   pp.Group(pp.Literal('C') + pp.Optional(pp.Literal('l'))) | \
                   pp.Literal('C') | \
                   pp.Literal('N') | \
                   pp.Literal('O') | \
                   pp.Literal('P') | \
                   pp.Literal('S') | \
                   pp.Literal('F') | \
                   pp.Literal('I'))
OrganicSymbol.setParseAction(organic_symbol_fn)

# Parses an aromatic symbol like c or n
AromaticSymbol <<= pp.Or(map(pp.Literal, ['b', 'c', 'n', 'o', 'p', 's']))
AromaticSymbol.setParseAction(aromatic_symbol_fn)

# Parses an atom specification like [Br+]
AtomSpec <<= pp.Literal('[').suppress() + \
  pp.Group( \
    pp.Optional(Isotope) + \
    pp.Or([pp.Literal('se'), pp.Literal('as'), AromaticSymbol, ElementSymbol, WILDCARD]) + \
    pp.Optional(ChiralClass) + \
    pp.Optional(HCount) + \
    pp.Optional(Charge) + \
    pp.Optional(Class)) + \
  pp.Literal(']').suppress()
AtomSpec.setParseAction(atom_spec_fn)

WILDCARD <<= '*'

# Parses an element like As or Te
ElementSymbol <<= pp.Group(pp.Word(pp.srange('[A-Z_]'), exact=1) + pp.Optional(pp.Word(pp.srange('[a-z]'), exact=1)))
ElementSymbol.setParseAction(element_symbol_fn)

# Parses a ring closure like 1
RingClosure <<= pp.Or([pp.Group(pp.Literal('%') + pp.Word(pp.srange('[1-9]'), exact=1) + pp.Word(pp.srange('[0-9]'), exact=1)), pp.Word(pp.srange('[0-9]'), exact=1)])
RingClosure.setParseAction(ring_closure_fn)

# Parses a chiral class like @@
ChiralClass <<= pp.Optional(
    pp.Group(
        pp.Literal('@') + pp.Optional(pp.Or([
            pp.Literal('@'),
            pp.Literal('TH') + pp.Word(pp.srange('[1-2]'), exact=1),
            pp.Literal('AL') + pp.Word(pp.srange('[1-2]'), exact=1),
            pp.Literal('SP') + pp.Word(pp.srange('[1-3]'), exact=1),
            pp.Literal('TB') + pp.Or([ pp.Literal('1') + pp.Optional(pp.Word(pp.srange('[0-9]'), exact=1)),
                                       pp.Literal('2') + pp.Optional(pp.Literal('0')),
                                       pp.Word(pp.srange('[3-9]'), exact=1)]),
            pp.Literal('OH') + pp.Or([ pp.Literal('1') + pp.Optional(pp.Word(pp.srange('[0-9]'), exact=1)),
                                       pp.Literal('2') + pp.Optional(pp.Word(pp.srange('[0-9]'), exact=1)),
                                       pp.Literal('3') + pp.Optional(pp.Literal('0')),
                                       pp.Word(pp.srange('[4-9]'), exact=1)])]))))
ChiralClass.setParseAction(chiral_class_fn)

# Parses an HCount like H1
HCount <<= pp.Group(pp.Literal('H') + pp.Optional(pp.Word(pp.srange('[0-9]'), exact=1)))
HCount.setParseAction(hcount_fn)

# Parses a charge like - or +
Charge <<= pp.Group(pp.Or([
    pp.Literal('-') + pp.Optional(pp.Or([pp.Literal('-'), pp.Literal('0'), pp.Literal('1') + pp.Optional(pp.Word(pp.srange('[0-5]'), exact=1)), pp.Word(pp.srange('[2-9]'), exact=1)])),
    pp.Literal('+') + pp.Optional(pp.Or([pp.Literal('+'), pp.Literal('0'), pp.Literal('1') + pp.Optional(pp.Word(pp.srange('[0-5]'), exact=1)), pp.Word(pp.srange('[2-9]'), exact=1)]))]))
Charge.setParseAction(charge_fn)

# Parses a class like :0
Class <<= pp.Group(pp.Literal(':') + pp.Word(pp.srange('[0-9]'), exact=1))
Class.setParseAction(class_fn)

# Parses an isotope like 105
Isotope <<= pp.Group(pp.Word(pp.srange('[1-9]'), exact=1) + pp.Optional(pp.Word(pp.srange('[0-9]'), exact=1)) + pp.Optional(pp.Word(pp.srange('[0-9]'), exact=1)))
Isotope.setParseAction(isotope_fn)
