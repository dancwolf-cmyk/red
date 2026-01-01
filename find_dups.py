from pathlib import Path
text = Path('lw.md').read_text(encoding='utf-8')
paras = []
current = []
for line in text.splitlines():
    current.append(line)
    if not line.strip():
        paras.append('\n'.join(current).strip('\n'))
        current = []
if current:
    paras.append('\n'.join(current).strip('\n'))
counts = {}
for p in paras:
    if not p:
        continue
    counts[p] = counts.get(p, 0) + 1
print(len([c for c in counts.values() if c>1]))
for p,c in counts.items():
    if c>1:
        print(c, repr(p[:60]))
