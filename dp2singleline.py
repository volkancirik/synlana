#!/usr/bin/env/python
import sys
usage = '''Converts dependency parses to single lines in a weird formate.

$> python dp2singleline.py <dp file>

$> find data/sample_transcriptions -name "*.dp" | xargs -I '{}' python dp2singleline.py '{}'
'''
if len(sys.argv) != 2:
  print(usage)
  quit(1)

inpf = sys.argv[1]
outf = inpf.replace('.dp', '.dp_single')

s = []
sentences = []
for line in open(inpf):
  l = line.strip().split('\t')
  if len(l) < 2:
    sentences.append(s)
    s = []
    continue

  s.append('___'.join([l[1], l[6], l[7]]))
f = open(outf, 'w')
for s in sentences:
  f.write('|||'.join(s))
  f.write('\n')
f.close()
print(outf, 'Done!')
