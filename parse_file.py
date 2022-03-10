from supar import Parser
import sys
import nltk
usage = '''Generates parse trees, tokenized forms, and meta information for transcripts.

$> python parse_file.py <transcript file> <# of skipped tokens>

$> find data/sample_transcriptions -name "*.txt" | xargs -I '{}' python parse_file.py '{}' 2 
'''


def main():
  if len(sys.argv) != 3:
    print(usage)
    quit(1)

  inp = sys.argv[1]
  tskip = int(sys.argv[2])

  print('Reading {}'.format(inp))
  try:
    f = open(inp)
  except:
    print('Cannot read from file {}'.format(inp))
    quit(1)

  all_lines = [l for l in f]
  meta = ['\t'.join(l.strip().split()[:tskip]) for l in all_lines]
  lines = [' '.join(l.strip().split()[tskip:]) for l in all_lines]

  dependency_parser = Parser.load('crf2o-dep-en')
  constituency_parser = Parser.load('crf-con-bert-en')

  tokenized = []
  metadata = []
  for ii, (l, m) in enumerate(zip(lines, meta)):
    t = nltk.word_tokenize(l)
    if len(t) > 1:
      tokenized += [t]
      metadata += [m]

  print('{} sentences will be parsed.'.format(len(tokenized)))
  dependency_parsed = dependency_parser.predict(
      tokenized, verbose=False).sentences
  constituency_parsed = constituency_parser.predict(
      tokenized, verbose=False).sentences

  assert (len(dependency_parsed) == len(tokenized)), '# of tokenized {} != {} # of parsed'.format(
      len(dependency_parsed), len(tokenized))
  out = '.'.join(inp.split('.')[:-1])

  mout = open(out + '.meta', 'w')
  tout = open(out + '.tokenized', 'w')
  dpout = open(out + '.dp', 'w')
  cpout = open(out + '.parsed', 'w')

  for m in metadata:
    mout.write(m+'\n')
  mout.close()

  for t in tokenized:
    tout.write(' '.join(t)+'\n')
  tout.close()

  for p in dependency_parsed:
    dpout.write(str(p))
    dpout.write('\n')
  dpout.close()

  for p in constituency_parsed:
    cpout.write(str(p))
    cpout.write('\n')
  cpout.close()


if __name__ == '__main__':
  main()
