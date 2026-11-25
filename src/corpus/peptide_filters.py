"""Standardise a molecule and decide what kind of peptide it is.

classify(smiles) -> dict with
    std        standardised canonical isomeric SMILES (None if unparseable)
    residues   number of alpha-amino-acid units in the backbone
    kind       'canonical' | 'noncanonical' | 'not_peptide'
    cyclic     backbone closes on itself (head-to-tail) or contains a macrocycle

Definitions (fixed here so every source is judged the same way):

  peptide      >= MIN_RES consecutive alpha-amino-acid units N-C(alpha)-C(=O),
               and <= MAX_RES of them (our corpus window)
  canonical    a LINEAR chain of the 20 standard L-amino acids with free termini:
               cutting every backbone amide bond leaves only standard residues
  noncanonical any other peptide: D-residues, N-methylation, non-standard side
               chains, cyclisation (backbone or disulfide), terminal caps, ...

Standardisation: largest fragment, neutralised, canonical isomeric SMILES -- so
the same peptide written as zwitterion in one source and neutral in another
dedupes to one entry.
"""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

MIN_RES, MAX_RES = 2, 50
STANDARD = "ACDEFGHIKLMNPQRSTVWY"

# one backbone unit: amide-or-amine N, alpha carbon, carbonyl carbon
_UNIT = Chem.MolFromSmarts("[NX3;!$(N=*)][CX4][CX3](=O)")
# backbone peptide bond between two units: C(alpha)-C(=O)-N-C(alpha)
_PEPBOND = Chem.MolFromSmarts("[CX4][CX3](=O)!@,@[NX3][CX4]")
_largest = rdMolStandardize.LargestFragmentChooser()
_uncharge = rdMolStandardize.Uncharger()


def standardise(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, None
    try:
        m = _largest.choose(m)
        m = _uncharge.uncharge(m)
        Chem.SanitizeMol(m)
    except Exception:
        return None, None
    return Chem.MolToSmiles(m), m


def _cut_amides(m):
    """Cut every backbone peptide bond; return residue fragments (dummies removed)
    and the number of bonds cut."""
    bonds = set()
    for a_ca, c, o, n, b_ca in m.GetSubstructMatches(_PEPBOND):
        bonds.add(m.GetBondBetweenAtoms(c, n).GetIdx())
    if not bonds:
        return [], 0
    frag = Chem.FragmentOnBonds(m, sorted(bonds), addDummies=True)
    out = []
    for f in Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False):
        rw = Chem.RWMol(f)
        for at in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0), reverse=True):
            rw.RemoveAtom(at)
        try:
            Chem.SanitizeMol(rw)
            out.append(Chem.MolToSmiles(rw))
        except Exception:
            out.append("?")
    return out, len(bonds)


@lru_cache(maxsize=1)
def _reference_fragments():
    """Fragments of every standard L-residue in every position (N-term, middle,
    C-term), produced by the same cutting as above -- so comparison is exact."""
    ref = set()
    for aa in STANDARD:
        for seq in (aa * 3, "G" + aa + "G"):
            m = Chem.MolFromSequence(seq)
            frags, _ = _cut_amides(m)
            ref.update(frags)
    return frozenset(ref)


def classify(smiles):
    std, m = standardise(smiles)
    if m is None:
        return {"std": None, "residues": 0, "kind": "not_peptide", "cyclic": False}
    # Residue count first: it is one substructure match, while _cut_amides below
    # fragments the molecule and sanitises every piece (~7x the total cost, and
    # worst on exactly the long peptides the window rejects). Pure reordering --
    # anything outside the window was going to be "not_peptide" either way.
    residues = len(m.GetSubstructMatches(_UNIT))
    if residues < MIN_RES or residues > MAX_RES:
        return {"std": std, "residues": residues, "kind": "not_peptide", "cyclic": False}

    frags, ncut = _cut_amides(m)
    linear = ncut > 0 and len(frags) == ncut + 1
    macro = any(len(r) >= 12 for r in m.GetRingInfo().AtomRings())
    cyclic = (ncut > 0 and not linear) or macro
    if ncut == 0:
        kind = "not_peptide"
    elif linear and not macro and all(f in _reference_fragments() for f in frags):
        kind = "canonical"
    else:
        kind = "noncanonical"
    return {"std": std, "residues": residues, "kind": kind, "cyclic": cyclic}


def _frag_smiles(f, flat=False):
    """One residue fragment (dummies removed) as SMILES, the same way _cut_amides writes it (flat: without stereo)."""
    rw = Chem.RWMol(f)
    for at in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0), reverse=True):
        rw.RemoveAtom(at)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw, isomericSmiles=not flat)
    except Exception:
        return "?"


def _ordered_residues(m, flat=False):
    """Cut every backbone peptide bond and return the residue fragments as SMILES in chain order, N-terminus first.
    None when the molecule is not one unbranched linear chain (no peptide bond, a ring through the backbone,
    a branch, or several chains)."""
    bonds = {}
    for a_ca, c, o, n, b_ca in m.GetSubstructMatches(_PEPBOND):
        bonds[m.GetBondBetweenAtoms(c, n).GetIdx()] = (c, n)
    if not bonds:
        return None
    frag = Chem.FragmentOnBonds(m, sorted(bonds), addDummies=True)
    groups = Chem.GetMolFrags(frag)                      # atom indices; the original atoms keep their indices
    mols = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)   # same order as `groups`
    n0 = m.GetNumAtoms()
    owner = {a: i for i, g in enumerate(groups) for a in g if a < n0}
    nxt, prv = {}, {}
    for c, n in bonds.values():
        fc, fn = owner[c], owner[n]
        if fc == fn or fc in nxt or fn in prv:
            return None
        nxt[fc], prv[fn] = fn, fc
    starts = [i for i in range(len(groups)) if i not in prv]
    if len(starts) != 1:
        return None
    order, i = [], starts[0]
    while i is not None:
        order.append(i)
        i = nxt.get(i)
    if len(order) != len(groups):
        return None
    return [_frag_smiles(mols[i], flat) for i in order]


@lru_cache(maxsize=1)
def _letters():
    """fragment SMILES -> one-letter code, for the 20 standard L-residues at the N-terminus, inside a chain and at the
    C-terminus (read off RDKit's own sequence builder, so the writing of the fragments is identical)."""
    table = {}
    for aa in STANDARD:
        for seq, pos in ((aa + "GG", 0), ("G" + aa + "G", 1), ("GG" + aa, 2)):
            frags = _ordered_residues(Chem.MolFromSequence(seq))
            assert frags is not None and len(frags) == 3, "reference chain %s did not split in three" % seq
            f = frags[pos]
            assert table.get(f, aa) == aa, "fragment %s is both %s and %s" % (f, table[f], aa)
            table[f] = aa
    return table


# an N-terminal acetyl (on the N of a residue: N-C(alpha)-C(=O)) and a C-terminal primary amide (the carbonyl sits on a carbon that
# carries an N: not an Asn / Gln side chain, not an epsilon-acetyl-lysine)
_ACETYL = Chem.MolFromSmarts("[CH3][CX3](=O)[NX3][CX4][CX3](=O)")
_AMIDE = Chem.MolFromSmarts("[NX3;H2][CX3](=O)[CX4][NX3]")


def _strip_caps(m):
    """The molecule without an N-terminal acetyl and a C-terminal primary amide (exactly one of each at most), or None if
    RDKit cannot sanitise the result. A protein language model sees residues, not termini."""
    rw = Chem.RWMol(m)
    hit = rw.GetSubstructMatches(_ACETYL)
    if len(hit) == 1:
        for a in sorted(hit[0][:3], reverse=True):               # the methyl, the carbonyl carbon, its oxygen
            rw.RemoveAtom(a)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    hit = rw.GetSubstructMatches(_AMIDE)
    if len(hit) == 1:
        n = rw.GetAtomWithIdx(hit[0][0])                         # amide N -> acid O
        n.SetAtomicNum(8)
        n.SetNumExplicitHs(0)
        n.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return rw.GetMol()


@lru_cache(maxsize=1)
def _letters_flat():
    """The same table without stereochemistry (D and L of a residue share one key): for molecules whose SMILES carries no
    (or unreliable) stereo flags."""
    out = {}
    for k, aa in _letters().items():
        fk = Chem.MolToSmiles(Chem.MolFromSmiles(k), isomericSmiles=False)
        assert out.get(fk, aa) == aa, "flat fragment %s is both %s and %s" % (fk, out[fk], aa)
        out[fk] = aa
    return out


def sequence(smiles, max_res=MAX_RES, caps=False, stereo=True):
    """One-letter sequence of a CANONICAL peptide (linear, only the 20 standard L-residues, free termini: the same
    definition as classify()'s kind == 'canonical', so it is exactly what a protein language model such as ESM can
    read), else None. Residues come out N- to C-terminus.
    max_res: longest chain accepted (the corpus window is MAX_RES = 50; ESM itself reads ~1000).
    caps=True: ignore an N-terminal acetyl and a C-terminal primary amide (still linear, standard L-residues only).
    stereo=False: ignore stereochemistry altogether (D-residues and molecules written without stereo flags are read as
    the same residue; ESM cannot tell them apart anyway). The default insists on L."""
    std, m = standardise(smiles)
    if m is None:
        return None
    if caps:
        m = _strip_caps(m)
        if m is None:
            return None
    if any(len(r) >= 12 for r in m.GetRingInfo().AtomRings()):
        return None
    frags = _ordered_residues(m, flat=not stereo)
    if frags is None or not (MIN_RES <= len(frags) <= max_res):
        return None
    table = _letters() if stereo else _letters_flat()
    out = [table.get(f) for f in frags]
    return None if None in out else "".join(out)


if __name__ == "__main__":
    # self-test: sequences built by RDKit are canonical; edits make them not
    lin = Chem.MolToSmiles(Chem.MolFromSequence("AFGKL"))
    tests = {"linear L-pentapeptide": lin,
             "same, cyclised head-to-tail": None,
             "single amino acid": Chem.MolToSmiles(Chem.MolFromSequence("A")),
             "benzene": "c1ccccc1"}
    m = Chem.RWMol(Chem.MolFromSequence("AFGKL"))
    # head-to-tail: bond N-terminal N to C-terminal carbonyl C, drop the OH
    nterm = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N" and a.GetTotalNumHs() == 2][0]
    cterm_o = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "O" and a.GetTotalNumHs() == 1][-1]
    cterm_c = m.GetAtomWithIdx(cterm_o).GetNeighbors()[0].GetIdx()
    m.RemoveAtom(cterm_o)
    m.AddBond(nterm, cterm_c if cterm_c < cterm_o else cterm_c - 1, Chem.BondType.SINGLE)
    Chem.SanitizeMol(m)
    tests["same, cyclised head-to-tail"] = Chem.MolToSmiles(m)
    d = Chem.MolFromSmiles(lin)                       # invert the Ala alpha carbon
    ala = d.GetSubstructMatch(Chem.MolFromSmarts("[CH3][CX4H]([NX3])C(=O)"))[1]
    d.GetAtomWithIdx(ala).InvertChirality()
    tests["D-Ala variant"] = Chem.MolToSmiles(d)
    for name, smi in tests.items():
        print("%-30s %s" % (name, classify(smi)))
