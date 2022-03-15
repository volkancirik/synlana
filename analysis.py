import sys
import numpy as np
import os
import subprocess

from syntactic_complexity import calc_nodes, calc_yngve, calc_frazier
from nltk.tree import Tree
import io
usage = '''Runs the analysis.

$> python analysis.py <timestamp csv> <transcription folder> <parse type> <OPTIONAL debug_file>
$> python analysis.py data/sample_timestamp.csv data/sample_transcriptions parsed
$> python analysis.py data/sample_timestamp.csv data/sample_transcriptions disdecparse debug.txt
'''


def calc_mdd(dparse):
  '''Given a dependency parse tree calculate mean dependency distance.
  '''

  # tokens are separated by |||
  tokens = dparse.split('|||')

  n = max(len(tokens) - 1, 1)
  distances = 0.0
  for ii, token in enumerate(tokens):
    # parsing information separated by ____
    l = token.split('___')

    # word = l[0]
    head = int(l[1])
    # dept = l[2]
    distances += abs(ii - head)
  return distances / n


def load_vectors(fname, vocab,
                 filter_words=True):
  '''Returns a {} for word embeddings as np.arrays

  fname: path to word embeddings
  vocab: dict or set for vocabulary
  filter_words: bool. If true then only words in vocab will be returned
  '''
  fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
  n, d = map(int, fin.readline().split())
  data = {}
  hits = 0.0
  for line in fin:
    tokens = line.rstrip().split(' ')
    if tokens[0] not in vocab and filter_words:
      continue
    data[tokens[0]] = np.array([float(v) for v in tokens[1:]]).reshape(1, 300)
    hits += 1
  print('vocabulary coverage:', hits, len(vocab), hits/len(vocab))
  return data


def open_file(fname):
  try:
    f = open(fname)
    return f
  except:
    print('Cannot open file {}'.format(fname))
    quit(1)


def study2patient(timestamp_file, olid, day):
  '''Returns a {} study -> patient

  timestamp_file: path to csv file
  olid: not used
  day: not used
  '''

  f = open_file(timestamp_file)
  lines = [line.strip().split(',') for line in f][1:]

  s2pt = {l[0]+'_'+l[1]: l[2] for l in lines}

  return s2pt


def find_missing_files(root_path):
  '''Finds missing processed interviews for given root_path.

  For each interview;
      .parsed : constituency parse
      .dp_single: single line dependency parses
      .disdec: constituency parses with disfluency
      .tokenized: tokenized lines
      .meta: meta data for each line
  '''

  # find all txt files
  result = subprocess.run(
      ['find', root_path, '-name', '*.txt'], stdout=subprocess.PIPE)
  flist = result.stdout.decode('utf-8').split('\n')

  missing = []
  for f in flist:
    pfile = f.replace('.txt', '.parsed')
    dpfile = f.replace('.txt', '.dp_single')
    disdecpfile = f.replace('.txt', '.disdecparse')
    tfile = f.replace('.txt', '.tokenized')
    mfile = f.replace('.txt', '.meta')

    # check if all extensions exist
    if os.path.isfile(pfile) and os.path.isfile(tfile) and os.path.isfile(mfile) and os.path.isfile(dpfile) and os.path.isfile(disdecpfile):
      continue
    missing += [f]

  # report missing files
  if len(missing) > 1:
    print('Out of {} studies {} are missing files.'.format(
        len(flist), len(missing)))
    print('\n'.join(missing))
    return 0
  else:
    return 1


def check_files(root_path):
  '''Checks and returns data for interviews for given root_path.

  For each interview;
      .parsed : constituency parse
      .dp_single: single line dependency parses
      .disdec: constituency parses with disfluency
      .tokenized: tokenized lines
      .meta: meta data for each line
  '''

  result = subprocess.run(
      ['find', root_path, '-name', '*.txt'], stdout=subprocess.PIPE)

  flist = result.stdout.decode('utf-8').split('\n')

  unmatched = []
  for f in flist:
    if not len(f):
      continue
    pfile = f.replace('.txt', '.parsed')
    dpfile = f.replace('.txt', '.dp_single')
    tfile = f.replace('.txt', '.tokenized')
    mfile = f.replace('.txt', '.meta')
    disdecpfile = f.replace('.txt', '.disdecparse')

    # read all extensions
    plines = [line.strip() for line in open_file(pfile)]
    dplines = [line.strip() for line in open_file(dpfile)]
    disdecplines = [line.strip() for line in open_file(disdecpfile)]
    tlines = [line.strip() for line in open_file(tfile)]
    mlines = [line.strip() for line in open_file(mfile)]

    # they should all be same length
    if len(plines) == len(tlines) and len(plines) == len(mlines) and len(plines) == len(disdecplines) and len(plines) == len(dplines):
      continue
    unmatched += ['plines {} tlines {} mlines {} disdecplines {} dplines {}'.format(
        len(plines), len(tlines), len(mlines), len(disdecplines), len(dplines))+'\t'+f]

  # report results
  if len(unmatched) > 0:
    print('Out of {} studies {} are not matching.'.format(
        len(flist), len(unmatched)))
    print('\n'.join(unmatched))
    return 0
  else:
    print('Everything is fine with parsed files.')
    return 1


def get_parse_filename(olid, study_day, transcript, root_path,
                       parse_suffix='parsed'):
  '''Returns file path for a parse tree output file.
  '''

  day = '0'*(4-len(study_day))+str(study_day)
  suffix = 'ab2avi-{}-day{}-onsite_transcript.{}'.format(
      olid, day, parse_suffix)
  fname = os.path.join(root_path, olid, 'transcripts', suffix)

  return fname


def get_parse_trees(parse_file, transcript, vocab,
                    parse_suffix='parsed'):
  '''Return all lines for parsed outputs.
  '''
  plines = [line.strip() for line in open_file(parse_file)]

  mfile = parse_file.replace('.{}'.format(parse_suffix), '.meta')
  mlines = [line.strip() for line in open_file(mfile)]

  dpfile = parse_file.replace('.{}'.format(parse_suffix), '.dp_single')
  dplines = [line.strip() for line in open_file(dpfile)]

  txtfile = parse_file.replace('.{}'.format(parse_suffix), '.tokenized')
  txtlines = [line.strip().lower().split() for line in open_file(txtfile)]

  trees = []
  txts = []
  dparses = []
  sample = ''

  # doubl-checking everything is okay with files
  assert(len(plines) == len(mlines))
  assert(len(plines) == len(dplines))
  assert(len(plines) == len(txtlines))

  other_utterances = []
  other_utterance = []
  for ii, (p, dp, m, txt) in enumerate(zip(plines, dplines, mlines, txtlines)):

    # read the target transcript line
    if m.split('\t')[0].split(':')[0] == transcript:
      flag = 0
      try:
        # convert to a Tree object
        t = Tree.fromstring(p)
      except:
        flag = 1
        pass
      if flag:
        # if there is a problem keep a sample issue
        if sample == '':
          sample = '[{}]'.format(ii)+p
        continue
      # add to list if all good
      trees.append(t)
      txts.append(txt)
      dparses.append(dp)
      other_utterances.append(other_utterance)
      other_utterance = []
      for w in txt:
        vocab.add(w)
    else:
      other_utterance += txt

  return trees, dparses, sample, txts, other_utterances, vocab


def study_syntactic_scores(timestamp_file, root_path,
                           parse_suffix='parsed',
                           debug_file=''):
  '''Calculate scores for all interviews.
  By changing parse_suffix one can use different constituency parses. 
  Options are parsed and disdecparse for now.
  '''
  print('Using Parse Suffix:', parse_suffix)

  # check if everything is okay with the setup
  if not(find_missing_files(root_path) and check_files(root_path)):
    print('Fix parse outputs first')
    quit(1)

  # read all interview timestamp data
  f = open_file(timestamp_file)
  lines = [line.strip().split(',') for line in f][1:]
  f.close()

  if debug_file:
    debug_f = open(debug_file, 'w')

  problems = []
  results = []

  all_trees = []
  all_dparses = []
  all_sample = []
  all_txts = []
  all_lines = []
  all_other_utts = []
  # build a vocabulary
  vocab = set()

  # for each interview
  for l in lines:
    # get parsing output filename
    fname = get_parse_filename(l[0], l[1], l[2], root_path,
                               parse_suffix=parse_suffix)
    # read all extensions
    trees, dparses, sample, txts, other_utts, vocab = get_parse_trees(fname, l[2], vocab,
                                                                      parse_suffix=parse_suffix)

    # if there are issues keep a list of problems
    if not len(trees):
      problems.append(
          '\t\t'.join(l[:3]+[fname.split('/')[-1]]+['{}'.format(len(trees))]+[sample]))
      n_tabs = int(len(fname.split('/')[-1])/8)
      continue

    # accumulate all info
    all_trees.append(trees)
    all_dparses.append(dparses)
    all_sample.append(sample)
    all_txts.append(txts)
    all_lines.append(l)
    all_other_utts.append(other_utts)

  # report problems if any
  if len(problems):
    print('Out of {} studies {} have 0 parse trees. Please take a look'.format(
        len(lines), len(problems)))

    print('\t\t'.join(['olid', 'day', 'target',
                       'file', '\t'*n_tabs+'# of parsed', 'sample']))
    print('\n'.join(problems))

  # read word vectors
  wordvec_path = '/usr0/home/vcirik/wordvec.glove'  # TODO: add to argparse
  print('loading vocabulary from', wordvec_path)
  wvec = load_vectors(wordvec_path, vocab)

  n = len(all_trees)
  total = 0.0
  oov = 0.0

  # for each interviews
  for ii in range(n):

    # retrieve info
    trees = all_trees[ii]
    dparses = all_dparses[ii]
    sample = all_sample[ii]
    txts = all_txts[ii]
    other_utts = all_other_utts[ii]
    lines = all_lines[ii]
    # min and max w measures how big the space words cover in embedding space
    # bigger space more diverse set of topics
    min_w = np.ones((1, 300))*float('inf')
    max_w = np.ones((1, 300))*float('-inf')

    sent_vecs = []
    sent_vecs_debug = []
    other_utt_vecs = []
    other_utt_vecs_debug = []
    # for each sentence
    for txt in txts:
      sent_vec = np.zeros((1, 300))
      # for each token in a sentence
      for w in txt:
        total += 1.0
        if w in wvec:
          vec = wvec[w]
        else:
          # keep track of out of vocabulary words
          vec = np.zeros((1, 300))
          oov += 1.0

        sent_vec += vec

        # calculate min and max for each dimension
        for d in range(300):
          min_w[0, d] = min(min_w[0, d], vec[0, d])
          max_w[0, d] = max(max_w[0, d], vec[0, d])
      # average sentence vector for circularity and temporal circularity
      sent_vec /= len(txt)
      sent_vecs.append(sent_vec)
      sent_vecs_debug.append(txt)

    for utts in other_utts:
      # for each token in other utterance
      sent_vec = np.zeros((1, 300))
      for w in utts:
        total += 1.0
        if w in wvec:
          vec = wvec[w]
        else:
          # keep track of out of vocabulary words
          vec = np.zeros((1, 300))
        sent_vec += vec
      if len(utts):
        sent_vec /= len(utts)
      other_utt_vecs.append(sent_vec)
      other_utt_vecs_debug.append(utts)

    # measure the coverage of each dimension
    size_w = np.zeros((1, 300))
    for d in range(300):
      size_w[0, d] = max_w[0, d] - min_w[0, d]

    # calculate mean dependency distance
    scores_mdd = []
    for dp in dparses:
      mdd = calc_mdd(dp)
      # TODO add more dep-parsing metrics
      scores_mdd.append(mdd)

    # calculate constituency parsing metrics
    scores_yngve = []
    scores_nodes = []
    scores_frazier = []
    for t in trees:
      yngve = calc_yngve(t, 0)
      nodes = calc_nodes(t)
      frazier = calc_frazier(t, 0, "")
      scores_yngve.append(yngve)
      scores_nodes.append(nodes)
      scores_frazier.append(frazier)

    # calculate pairwise distances for participants own sentences
    n = len(sent_vecs)
    pairwise_distances = []
    pairwise_distances_debug = []
    for ii in range(n):
      for jj in range(ii+1, n):
        diff = sent_vecs[ii] - sent_vecs[jj]
        distance = np.linalg.norm(diff)
        pairwise_distances.append(distance)
        pairwise_distances_debug.append((ii, jj))

    # calculate the distance between successive sentences
    temporal_distances = []
    shuffled_distances = []
    temporal_distances_debug = []
    shuffled_distances_debug = []

    for ii in range(n-1):
      diff = sent_vecs[ii] - sent_vecs[ii+1]
      distance = np.linalg.norm(diff)
      temporal_distances.append(distance)
      temporal_distances_debug.append((ii, ii+1))

      jj = np.random.randint(n-1)
      diff = sent_vecs[ii] - sent_vecs[jj]
      distance = np.linalg.norm(diff)
      shuffled_distances.append(distance)
      shuffled_distances_debug.append((ii, jj))

    # calculate the distance between responses
    diadic_distances = []
    diadic_distances_debug = []
    for ii in range(n):
      diff = sent_vecs[ii] - other_utt_vecs[ii]
      distance = np.linalg.norm(diff)
      diadic_distances.append(distance)
      diadic_distances_debug.append(ii)

    # keep a dictionary of scores for each metrics
    scores = {'yngve': scores_yngve,
              'nodes': scores_nodes,
              'frazier': scores_frazier,
              'mdd': scores_mdd
              }

    # calculate functions of each metrics
    calculations = ['min', 'max', 'mean', 'std']
    functions = [np.min, np.max, np.mean, np.std]
    for fn, calc in zip(functions, calculations):

      scores['pairwise_sent_dist_'+calc] = fn(pairwise_distances)
      scores['temporal_sent_dist_'+calc] = fn(temporal_distances)
      scores['diadic_sent_dist_'+calc] = fn(diadic_distances)
      scores['shuffled_sent_dist_'+calc] = fn(shuffled_distances)

    # calculate diversity for all dimension
    diversity_all = []
    for d in range(300):
      diversity_all += [size_w[0, d]]

    calculations = ['mean', 'std']
    functions = [np.mean, np.std]
    for fn, calc in zip(functions, calculations):
      scores['diversity_all_'+calc] = fn(diversity_all)

    # for each dimension calculate the diversity size
    # for d in range(300):
    #   scores['diversity_d{}'.format(d)] = size_w[0, d]

    # percentage of out-of-vocabulary word usage
    scores['oov'] = oov/total

    # DEBUGGING
    if debug_file:
      functions = [np.argmin, np.argmax]
      calculations = ['argmin', 'argmax']
      tuples = [('pairwise_distances', pairwise_distances, pairwise_distances_debug),
                ('temporal_distances', temporal_distances, temporal_distances_debug),
                ('diadic_distances', diadic_distances, diadic_distances_debug),
                ('shuffled_distances', shuffled_distances, shuffled_distances_debug),
                ('yngve', scores_yngve, txts),
                ('nodes', scores_nodes, txts),
                ('frazier', scores_frazier, txts),
                ('mdd', scores_mdd, txts)
                ]
      for t in tuples:
        n, s, d = t
        for fn, calc in zip(functions, calculations):
          idx = fn(s)
          debug = d[idx]
          if type(debug) == int:
            if 'diadic' in n:
              debug = '[Patient]:' + ' '.join(txts[debug]) + \
                  ' [Interviewer]:' + ' '.join(other_utt_vecs_debug[debug])
            else:
              raise NotImplementedError()
          elif type(debug) == tuple:
            debug = '[Sent_{}]:'.format(debug[0]) + ' '.join(txts[debug[0]]) + \
                    ' [Sent_{}]:'.format(debug[1]) + \
                ' '.join(txts[debug[1]])
          elif type(debug) == list:
            debug = ' '.join(debug)
          else:
            raise NotImplementedError()
          debug_line = '{} {} {} {}'.format(n, calc, s[idx], debug)
          debug_f.write(debug_line + '\n')

    # add meta data and scores
    results.append((lines[:3], scores))
  print('Returning results for {} studies'.format(len(results)))
  if debug_file:
    debug_f.close()
  return results


def print_sss(results):
  '''Print results in CSV format.
  '''

  # preparing CSV header
  header = ['olid', 'study_day', 'transcript_target']
  metrics = list(results[0][1].keys())
  calculations = ['mean', 'std']
  functions = [np.mean, np.std]
  for m in metrics:
    if m in set(['yngve', 'nodes', 'frazier', 'mdd']):
      for calc in calculations:
        header.append(m+'_'+calc)
    else:
      header.append(m)

  print(','.join(header))
  # print out each line
  for r in results:
    l = r[0]
    for m in metrics:
      if m in set(['yngve', 'nodes', 'frazier', 'mdd']):
        for calc, fn in zip(calculations, functions):
          l += ['{:2.2f}'.format(fn(r[1][m]))]
      else:
        l += ['{:2.2f}'.format(r[1][m])]
    print(','.join(l))


if __name__ == '__main__':

  # if no input given use "parsed" as the extension
  # alternative is disdecparse
  if not(4 <= len(sys.argv) <= 5):
    print(usage)
    quit(0)
  parse_suffix = 'parsed'
  parse_suffix = sys.argv[3]
  if len(sys.argv) == 5:
    debug_file = sys.argv[4]
  else:
    debug_file = ''
  results = study_syntactic_scores(sys.argv[1], sys.argv[2],
                                   parse_suffix=parse_suffix,
                                   debug_file=debug_file)
  print_sss(results)
