"""Tokenizers for the SMILES-only student. One interface, three families:

    released   PeptideCLM-2's own tokenizer (405 tokens), as shipped              ("A")
    fixed      the same 405 ids, with the digit defect fixed in the pre-tokenizer ("B")
    atombpe    learned on the pool: atoms are the base units, BPE merges them     ("C")

Every tokenizer gives ids WITHOUT special tokens from encode()/encode_batch(); `with_special()` adds
[CLS] ... [SEP] and truncates. All three use the ids of the released tokenizer for the special tokens
([PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4), so a model can swap tokenizers without touching them.

Why "C" is built the way it is:
  * Base units are ATOMS, not characters: a bracket atom such as [C@@H] or [nH], Br, Cl and %nn ring labels
    are each ONE base token, so a merge can never cut an atom in half (SmilesPE / Atom-Pair-Encoding idea).
    Implementation: every multi-character atom seen at least `min_atom_count` times is replaced by one
    private-use Unicode character, and the Hugging Face BPE trainer then runs on those strings, so a merge of
    two base characters is a merge of two atoms.
  * Full coverage: any other text (a rare bracket atom, an odd character) falls back to its own printable-ASCII
    characters, which are always in the alphabet. There is no [UNK] for printable ASCII input.
  * Reversible: decode(encode(s)) == s.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CKPT_DIR, PCLM_DIR  # noqa: E402

PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
TOK_DIR = f"{CKPT_DIR}/student_tokenizer"       # one folder per learned tokenizer: atombpe_v2048, ...

# a multi-character atom: bracket atom, Br, Cl, two-digit ring label. Single characters stay as they are.
MULTI = re.compile(r"\[[^\]]+\]|Br|Cl|%[0-9]{2}")
PRIVATE0 = 0xE000                               # Unicode private-use area: 6,400 code points
ASCII = [chr(c) for c in range(32, 127)]

# the released pre-tokenizer isolates a list of fixed substrings and leaves everything else as one piece. Two or more
# consecutive characters that are not in the list (ring digits "12", "/1", "-1") then reach WordPiece as ONE word that
# is not in the vocabulary and, with no "##" tokens, become [UNK]. The fix: match every single-character vocabulary
# token on its own (digits, %nn, / \ - :). The ids and the vocabulary are unchanged.
FIX = r"|%[0-9]{2}|[0-9]|/|\\|-|:"


def _tokenizers():
    from tokenizers import Tokenizer
    return Tokenizer


class Tok:
    """Common interface. Subclasses set self.tk (a tokenizers.Tokenizer), self.name, and _prep()."""
    name = "?"
    family = "?"

    def _prep(self, s):
        return s

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()

    pad_id, unk_id, cls_id, sep_id, mask_id = PAD, UNK, CLS, SEP, MASK

    def encode(self, smiles):
        return self.tk.encode(self._prep(smiles)).ids

    def encode_batch(self, smiles):
        return [e.ids for e in self.tk.encode_batch([self._prep(s) for s in smiles])]

    def with_special(self, ids, max_len=None):
        """[CLS] ids [SEP], the ids cut so the whole thing is at most max_len long (the [SEP] is kept)."""
        if max_len is not None:
            ids = ids[: max_len - 2]
        return [CLS] + list(ids) + [SEP]

    def tokens(self, smiles):
        """The tokens of one SMILES as readable text (what the model sees, before ids)."""
        return [self._show(t) for t in self.tk.encode(self._prep(smiles)).tokens]

    def _show(self, t):
        return t

    def decode(self, ids):
        out = []
        for i in ids:
            if i < len(SPECIALS):
                continue
            out.append(self._show(self.tk.id_to_token(int(i))))
        return "".join(out)

    def vocab(self):
        """Readable token strings in id order."""
        return [self._show(self.tk.id_to_token(i)) for i in range(self.vocab_size)]


class Released(Tok):
    """A (fixed=False) or B (fixed=True). Reads the shipped tokenizer.json; B patches the pre-tokenizer regex in memory."""
    family = "released"

    def __init__(self, fixed=False, path=None):
        path = path or os.path.join(PCLM_DIR, "tokenizer.json")
        t = json.load(open(path, encoding="utf-8"))
        t["post_processor"] = None              # [CLS]/[SEP] are added by with_special(), the same for every family
        t["truncation"] = None                  # measure true lengths
        t["padding"] = None
        if fixed:
            pat = t["pre_tokenizer"]["pattern"]["Regex"]
            assert pat.startswith("(") and pat.endswith(")"), "unexpected pre-tokenizer pattern"
            t["pre_tokenizer"]["pattern"]["Regex"] = pat[:-1] + FIX + ")"
        self.tk = _tokenizers().from_str(json.dumps(t))
        self.fixed = fixed
        self.name = "B_fixed" if fixed else "A_released"
        for i, s in enumerate(SPECIALS):
            assert self.tk.token_to_id(s) == i, "special token ids differ from the project convention"


class AtomBPE(Tok):
    family = "atombpe"

    def __init__(self, tk, atoms, meta=None, name=None):
        self.tk, self.atoms, self.meta = tk, list(atoms), meta or {}
        self.amap = {a: chr(PRIVATE0 + i) for i, a in enumerate(self.atoms)}
        self.inv = {c: a for a, c in self.amap.items()}
        self.name = name or "C_atombpe_v%d" % self.vocab_size
        self._sub = lambda m: self.amap.get(m.group(0), m.group(0))

    def _prep(self, s):
        return MULTI.sub(self._sub, s)

    def _show(self, t):
        return "".join(self.inv.get(c, c) for c in t)

    def fallback_atoms(self, smiles):
        """Multi-character atoms in `smiles` that are NOT base tokens (they are spelled out character by character)."""
        return [a for a in MULTI.findall(smiles) if a not in self.amap]

    # ---------------------------------------------------------------- training
    @classmethod
    def train(cls, smiles, vocab_size, min_atom_count=20, min_frequency=100, max_token_length=None, name=None):
        """min_frequency: a merge needs at least this many occurrences in `smiles` (keeps a few near-duplicate molecules from
        becoming one long token). max_token_length: longest token in BASE UNITS (atoms), None = no cap."""
        from collections import Counter
        from tokenizers import Tokenizer, models, trainers
        smiles = list(smiles)
        cnt = Counter(a for s in smiles for a in MULTI.findall(s))
        atoms = [a for a, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_atom_count]
        assert len(atoms) < 6000, "too many atoms for the private-use area"
        amap = {a: chr(PRIVATE0 + i) for i, a in enumerate(atoms)}
        sub = lambda m: amap.get(m.group(0), m.group(0))       # noqa: E731
        tk = Tokenizer(models.BPE(unk_token="[UNK]"))           # no normalizer, no pre-tokenizer: the SMILES is one word
        kw = {"max_token_length": max_token_length} if max_token_length else {}
        trn = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIALS,
                                  initial_alphabet=ASCII + list(amap.values()), show_progress=False, **kw)
        tk.train_from_iterator((MULTI.sub(sub, s) for s in smiles), trn, length=len(smiles))
        for i, s in enumerate(SPECIALS):
            assert tk.token_to_id(s) == i, "special token ids moved"
        meta = {"vocab_size_requested": vocab_size, "vocab_size": tk.get_vocab_size(), "n_train": len(smiles),
                "min_atom_count": min_atom_count, "min_frequency": min_frequency, "max_token_length": max_token_length,
                "n_atoms": len(atoms),
                "atoms_seen": len(cnt), "atoms_below_min_count": len(cnt) - len(atoms)}
        return cls(tk, atoms, meta, name)

    # ---------------------------------------------------------------- save / load
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.tk.save(os.path.join(folder, "tokenizer.json"))
        with open(os.path.join(folder, "atoms.json"), "w", encoding="utf-8") as fh:
            json.dump(self.atoms, fh, ensure_ascii=False)
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=1)
        with open(os.path.join(folder, "vocab.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.vocab()) + "\n")

    @classmethod
    def load(cls, folder):
        tk = _tokenizers().from_file(os.path.join(folder, "tokenizer.json"))
        atoms = json.load(open(os.path.join(folder, "atoms.json"), encoding="utf-8"))
        mp = os.path.join(folder, "meta.json")
        meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
        return cls(tk, atoms, meta, name="C_" + os.path.basename(os.path.normpath(folder)))


def load_tokenizer(spec):
    """'released' | 'fixed' | a learned tokenizer: a folder, or its name under TOK_DIR (e.g. 'atombpe_v2048')."""
    if spec in ("released", "A"):
        return Released(fixed=False)
    if spec in ("fixed", "B"):
        return Released(fixed=True)
    folder = spec if os.path.isdir(spec) else os.path.join(TOK_DIR, spec)
    return AtomBPE.load(folder)
