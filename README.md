# Syntactic Language Analysis for Dialog

This repository contains scripts to run a simple syntactic analysis for dialog.

### Installation

Run `install.sh` to create the conda environment `synlana`. This script install necessary packages. Once install activate the environment:

    `conda activate synlana`

If `conda activate synlana` does not work please try `source activate synlana`.

### Preparing Data

A sample data is under `data/sample.json`. Please use this format otherwise scripts do not work.
To prepare data and run the parsing algorithms use the following steps:

    python json2transcript.py data/sample.json data/sample_transcriptions
    find data/sample_transcriptions -name "*.txt" | xargs -I '{}' python parse_file.py '{}' 2
    find data/sample_transcriptions -name "*.dp" | xargs -I '{}' python dp2singleline.py '{}'
    bash run_disdetparse.sh data/sample_transcriptions/
    cd data; tar xvfz wordvec.glove.tar.gz;

If all goes well the output of `tree data` should look like this:

    data
    ├── sample.json
    └── sample_transcriptions
        └── ANNIEHALL
            └── transcripts
                ├── ab2avi-ANNIEHALL-day0001-onsite_transcript.disdetparse
                ├── ab2avi-ANNIEHALL-day0001-onsite_transcript.dp
                ├── ab2avi-ANNIEHALL-day0001-onsite_transcript.dp_single
                ├── ab2avi-ANNIEHALL-day0001-onsite_transcript.meta
                ├── ab2avi-ANNIEHALL-day0001-onsite_transcript.parsed
                ├── ab2avi-ANNIEHALL-day0001-onsite_transcript.tokenized
                └── ab2avi-ANNIEHALL-day0001-onsite_transcript.txt

* `.disdetparse` is the output of [this parser](https://github.com/pariajm/joint-disfluency-detector-and-parser) where disfluencies are detected.
* `.dp` is the output of [the state-of-the-art dependency parsing method](https://aclanthology.org/2020.acl-main.302/)
* `.meta` is the metadata for the interview
* `.parsed` is the output of [the state-of-the-art constituency parsing method](https://www.ijcai.org/Proceedings/2020/560/)
* `.tokenized` is the tokenized output
* `.txt` is the raw txt data for the dialog

Of course change the path to files accordingly for generating these files for your data.

### Running the Analysis

For each dialog data, ideally we use timestamp data where each important act (e.g. question) is annotated. However, for the sample data we do use this metadata. Run this command for generating the syntactic analysis for the dialog data:

     python analysis.py data/sample_timestamp.csv data/sample_transcriptions parsed

This will generate a csv file with following headers:

* olid: dialog id
* study_day: the day of the dialog you can safely ignore this.
transcript_target: target participant name
* yngve_{mean,std}: yngve score mean and std
* nodes_{mean,std}: nodes score mean and std
* frazier_{mean,std}: frazier score mean and std
* mdd_{mean,std}: mean dependency distance and std
* pairwise_sent_dist_{min,max,mean,std}: pairwise sentence similarity with min,max,mean,std
* temporal_sent_dist_{min,max,mean,std}: temporal (i.e. successive) sentence similarity with min,max,mean,std
* diadic_sent_dist_{min,max,mean,std}: diadic (i.e. response to other participant) sentence similarity with min,max,mean,std
* diversity_all_{mean,std}: word embedding space coverage for sentences.
* diversity_d[0...299]: word embedding space coverage for individual dimensions for sentences.
* oov: out of vocabulary rate
