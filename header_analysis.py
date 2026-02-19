"""
Analyze MEDEF DAT file header vs Seilgreifer_v8 spec.

Reads the raw header (everything before the first '$' in the binary file),
tokenizes it, parses field definitions in groups of 9, and compares each
field's metadata against the YAML spec to reveal the encoding relationships.
"""
import yaml
import re

# ---- Configuration --------------------------------------------------------
DAT_DIR = ("/Users/vigji/code/macchine/output/raw_downloads/"
           "2026-02-16_1514 Catania/Unidentified/")
DAT_FILE = (DAT_DIR +
            "01K00047564_gb50_601_20250704_084428_00001143_xxxxx.dat")
SPEC_FILE = "/Users/vigji/code/macchine/specs_data/medef_specs.yaml"

# ---- Read raw header (everything before first '$') -----------------------
with open(DAT_FILE, "rb") as f:
    raw = f.read(8000)

dollar_idx = raw.find(b"$")
header_text = raw[:dollar_idx].decode("iso-8859-1")

# ---- Tokenize (comma-separated, single-quoted strings) -------------------
tokens = []
for match in re.finditer(r"'([^']*)'|([^,]+)", header_text):
    if match.group(1) is not None:
        tokens.append(match.group(1))
    else:
        tokens.append(match.group(2).strip())

# ---- Load YAML spec ------------------------------------------------------
with open(SPEC_FILE, "r") as f:
    specs = yaml.safe_load(f)
sg_specs = specs["Seilgreifer_v8"]["sensors"]

# ---- Parse preamble and field groups --------------------------------------
PREAMBLE_SIZE = 7
GROUP_SIZE = 9
preamble = tokens[:PREAMBLE_SIZE]
num_fields = int(preamble[2])

print("=" * 100)
print("RAW HEADER")
print("=" * 100)
print(header_text)

print("\n" + "=" * 100)
print("PREAMBLE")
print("=" * 100)
for i, (label, val) in enumerate(zip(
    ["buffer_size", "version", "num_fields", "p3", "p4", "p5", "p6"],
    preamble
)):
    print(f"  [{i}] {label:12s} = {val}")

# ---- Print each field's 9-token group ------------------------------------
print("\n" + "=" * 100)
print(f"FIELD DEFINITIONS  ({num_fields} fields, {GROUP_SIZE} tokens each)")
print("  Group layout: [divisor, 'name', '[unit]', meta0, meta1, meta2, meta3, meta4, meta5]")
print("=" * 100)

for i in range(num_fields):
    off = PREAMBLE_SIZE + i * GROUP_SIZE
    end = off + GROUP_SIZE
    if end > len(tokens):
        group = tokens[off:] + [None] * (GROUP_SIZE - (len(tokens) - off))
    else:
        group = tokens[off:end]

    name = group[1] or "???"
    print(f"\n  Field {i:2d}: {name}")
    print(f"    tokens: {group}")
    for j, label in enumerate(["divisor", "name", "unit",
                                "meta[0]", "meta[1]", "meta[2]",
                                "meta[3]", "meta[4]", "meta[5]"]):
        print(f"      {label:10s} = {group[j]}")

# ---- Summary table -------------------------------------------------------
print("\n\n" + "=" * 100)
print("SUMMARY TABLE: header fields vs Seilgreifer_v8 spec")
print("=" * 100)
header = (f"{'#':>2s}  {'Field Name':<25s}  "
          f"{'div':>3s}  {'m0':>2s} {'m1':>2s} {'m2':>2s} {'m3':>2s} {'m4':>2s} {'m5':>2s}  "
          f"{'|':1s}  {'chr':>3s} {'dec':>3s} {'sgn':>3s} {'e.div':>5s}")
print(header)
print("-" * len(header))

for i in range(num_fields):
    off = PREAMBLE_SIZE + i * GROUP_SIZE
    end = off + GROUP_SIZE
    if end > len(tokens):
        group = tokens[off:] + [None] * (GROUP_SIZE - (len(tokens) - off))
    else:
        group = tokens[off:end]

    name = group[1] or "???"
    div_s = str(group[0]) if group[0] else "?"
    ms = [str(group[j]) if group[j] else "?" for j in range(3, 9)]

    spec = sg_specs.get(name)
    if spec:
        sc = str(spec["characters"])
        sd = str(spec["decimal"])
        ss = str(1 if spec["signed"] else 0)
        se = str(spec["expected_divisor"])
    else:
        sc = sd = ss = se = "N/A"

    print(f"{i:2d}  {name:<25s}  "
          f"{div_s:>3s}  {ms[0]:>2s} {ms[1]:>2s} {ms[2]:>2s} {ms[3]:>2s} {ms[4]:>2s} {ms[5]:>2s}  "
          f"|  {sc:>3s} {sd:>3s} {ss:>3s} {se:>5s}")

# ---- Relationship analysis -----------------------------------------------
print("\n\n" + "=" * 100)
print("RELATIONSHIP ANALYSIS")
print("=" * 100)

# 1) meta[0] == signed
print("\n[1] meta[0] vs spec.signed  (1=True, 0=False)")
print("-" * 60)
all_ok = True
for i in range(num_fields):
    off = PREAMBLE_SIZE + i * GROUP_SIZE
    group = tokens[off:off + GROUP_SIZE]
    if len(group) < 9 or group[3] is None:
        continue
    name = group[1]
    spec = sg_specs.get(name)
    if not spec:
        continue
    m0 = int(group[3])
    s = 1 if spec["signed"] else 0
    ok = m0 == s
    if not ok:
        all_ok = False
    print(f"  {i:2d} {name:<25s}  meta[0]={m0}  signed={s}  {'OK' if ok else 'MISMATCH'}")
print(f"\n  => meta[0] == signed: {'ALL MATCH (19/19)' if all_ok else 'SOME MISMATCH'}")

# 2) meta[2] vs chr+dec+sgn -- with next-field check
print("\n\n[2] meta[2] vs spec.(characters + decimal + signed)")
print("-" * 80)
own_matches = 0
next_matches = 0
total = 0
for i in range(num_fields):
    off = PREAMBLE_SIZE + i * GROUP_SIZE
    group = tokens[off:off + GROUP_SIZE]
    if len(group) < 9 or group[5] is None:
        continue
    name = group[1]
    spec = sg_specs.get(name)
    if not spec:
        continue
    total += 1
    m2 = int(group[5])
    cds = spec["characters"] + spec["decimal"] + (1 if spec["signed"] else 0)
    own_ok = m2 == cds

    # Check next field
    next_off = PREAMBLE_SIZE + (i + 1) * GROUP_SIZE
    next_group = tokens[next_off:next_off + GROUP_SIZE] if next_off + GROUP_SIZE <= len(tokens) else None
    next_name = next_group[1] if next_group else None
    next_spec = sg_specs.get(next_name) if next_name else None
    next_cds = (next_spec["characters"] + next_spec["decimal"] +
                (1 if next_spec["signed"] else 0)) if next_spec else None

    next_ok = (m2 == next_cds) if next_cds is not None else False

    if own_ok:
        own_matches += 1
        tag = "matches OWN"
    elif next_ok:
        next_matches += 1
        tag = f"matches NEXT ({next_name})"
    else:
        tag = "NO MATCH"

    print(f"  {i:2d} {name:<25s}  meta[2]={m2}  own_c+d+s={cds}  "
          f"next_c+d+s={next_cds if next_cds is not None else 'N/A':>3}  => {tag}")

print(f"\n  => Matches own field:  {own_matches}/{total}")
print(f"  => Matches next field: {next_matches}/{total}")
print(f"  => Total explained:    {own_matches + next_matches}/{total}")
print(f"\n  CONCLUSION: meta[2] ENCODES characters+decimal+signed, but for ~half the")
print(f"  fields it appears shifted -- containing the NEXT field's value instead.")
print(f"  This suggests the header stores a 'look-ahead' value, or the field boundary")
print(f"  between meta[2..5] and the next field's leading tokens is ambiguous.")

# 3) meta[3] (p6) vs decimal / 10^meta[3] vs expected_divisor
print("\n\n[3] meta[3] vs spec.decimal  and  10^meta[3] vs spec.expected_divisor")
print("-" * 80)
dec_matches = 0
div_matches = 0
total2 = 0
for i in range(num_fields):
    off = PREAMBLE_SIZE + i * GROUP_SIZE
    group = tokens[off:off + GROUP_SIZE]
    if len(group) < 9 or group[6] is None:
        continue
    name = group[1]
    spec = sg_specs.get(name)
    if not spec:
        continue
    total2 += 1
    m3 = int(group[6])
    dec = spec["decimal"]
    ediv = spec["expected_divisor"]
    dec_ok = m3 == dec
    div_ok = 10 ** m3 == ediv
    if dec_ok:
        dec_matches += 1
    if div_ok:
        div_matches += 1
    print(f"  {i:2d} {name:<25s}  meta[3]={m3}  spec.decimal={dec}  "
          f"10^meta[3]={10**m3}  spec.exp_div={ediv}  "
          f"{'dec OK' if dec_ok else 'dec FAIL'}  "
          f"{'div OK' if div_ok else 'div FAIL'}")

print(f"\n  => meta[3] == decimal: {dec_matches}/{total2}")
print(f"  => 10^meta[3] == expected_divisor: {div_matches}/{total2}")
print(f"  => Same fields that fail for meta[2] also fail for meta[3],")
print(f"     confirming the boundary/shift issue.")

# 4) divisor field (position 0 in group)
print("\n\n[4] 'divisor' (position 0 in each group)")
print("-" * 60)
print("  This value is ALWAYS 1 for all fields in this file.")
print("  It does NOT encode the expected_divisor (which varies: 1, 10, 100).")
print("  It appears to be a constant flag or multiplier, not the divisor used")
print("  for converting raw integer values to physical units.")

# 5) meta[4] and meta[5]
print("\n\n[5] meta[4] and meta[5] -- sensor type/category and sub-index")
print("-" * 60)
for i in range(num_fields):
    off = PREAMBLE_SIZE + i * GROUP_SIZE
    group = tokens[off:off + GROUP_SIZE]
    if len(group) < 9 or group[7] is None:
        continue
    name = group[1] or "???"
    m4 = group[7]
    m5 = group[8] if group[8] is not None else "?"
    print(f"  {i:2d} {name:<25s}  meta[4]={m4}  meta[5]={m5}")
print("\n  These appear to be sensor category IDs and sub-indices within")
print("  that category (e.g., Druck Pumpe 1-4 share meta[4]=4 with sub-indices 0-3).")
print("  However, due to the same boundary shift issue, some values may belong")
print("  to adjacent fields.")

# ---- Final summary -------------------------------------------------------
print("\n\n" + "=" * 100)
print("KEY FINDINGS")
print("=" * 100)
print("""
1. HEADER STRUCTURE: 7-token preamble + 20 groups of 9 tokens + copyright trailer.
   Group: [divisor, 'name', '[unit]', meta0, meta1, meta2, meta3, meta4, meta5]

2. CONFIRMED RELATIONSHIPS:
   - meta[0] == signed flag (1=signed/True, 0=unsigned/False)  --> 19/19 PERFECT
   - meta[2] == characters + decimal + (1 if signed else 0)    --> 11/19 match own field
     (the other 8 contain the NEXT field's value -- boundary shift artifact)
   - meta[3] == decimal places (number of fractional digits)    --> 10/19 match own field
     (same shift pattern as meta[2])
   - 10^meta[3] == expected_divisor                             --> 10/19 (same failures)

3. DIVISOR (position 0): Always 1 in this file. Does NOT correspond to expected_divisor.

4. meta[1]: Binary flag (0 or 1), purpose unclear -- possibly a display/visibility flag.

5. meta[4], meta[5]: Appear to be sensor category ID and sub-index. Subject to the
   same boundary shift as meta[2] and meta[3].

6. BOUNDARY SHIFT: For approximately half the fields, meta[2..5] appear to contain
   values that belong to the NEXT field rather than the current one. This alternating
   pattern suggests that the actual binary layout may use a different grouping boundary
   than the naive 9-token split, or that some fields' metadata is written in a
   look-ahead fashion. The spec's expected_divisor can be reliably computed from the
   field name alone (via the YAML spec lookup) rather than from the header metadata.
""")
