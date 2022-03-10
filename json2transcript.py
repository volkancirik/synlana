import json
import sys
import os
usage = '''Converts annotations in json format to transcription format where each interview session is under a different folder.

$> python json2transcript.py <json file> <output folder>

$> python json2transcript.py data/sample.json data/sample_transcriptions
'''


def main():
  if len(sys.argv) != 3:
    print(usage)
    quit(1)

  inp = sys.argv[1]
  out = sys.argv[2]

  cleaned_data = json.load(open(inp))

  for study in cleaned_data:
    fname = study['filename']
    studyname = fname.split('-')[1]

    path = '{}/{}/transcripts'.format(out, studyname)
    file_path = '{}/{}.txt'.format(path, fname)

    os.makedirs(path,
                exist_ok=True)
    with open(file_path, 'w') as f:
      for turn in study['turns']:
        f.write('\n')
        f.write(
            '\t'.join([turn['speaker'], '{}'.format(turn['timestamp']), turn['text']]))


if __name__ == '__main__':
  main()
