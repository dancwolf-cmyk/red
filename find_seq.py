from pathlib import Path
lines = Path('lw.md').read_text(encoding='utf-8').splitlines()
min_len = 20
seqs = {}
for i in range(len(lines) - min_len + 1):
    seq = '\n'.join(lines[i:i+min_len])
    seqs.setdefault(seq, []).append(i+1)
for seq, positions in seqs.items():
    if len(positions) > 1:
        print('Found repeated block at lines', positions)
        print('Sample starts with:', lines[positions[0]-1])
        print('---')
        break
