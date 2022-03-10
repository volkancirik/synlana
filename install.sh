#!/usr/bin/env/bash
set -e
conda create -n synlana python=3.8.3 --yes
source `which python | sed 's/bin\/python/etc\/profile.d\/conda.sh/g'`
conda activate synlana
pip install -U supar==1.0.0
pip install -U transformers==3.2.0 torch==1.6.0
pip install -U Cython==0.29.21
pip install -U pytorch-pretrained-bert==0.6.2
git clone https://github.com/pariajm/joint-disfluency-detector-and-parser.git; cd joint-disfluency-detector-and-parser; cd best_models; wget https://github.com/pariajm/joint-disfluency-detector-and-parser/releases/download/naacl2019/swbd_fisher_bert_Edev.0.9078.pt; cd ..; mkdir model && cd model; wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt; wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz; tar -xf bert-base-uncased.tar.gz
cd data;  wget https://www.dropbox.com/s/4ce8ez7y2vrvdfi/wordvec.glove.tar.gz; tar xvfz wordvec.glove.tar.gz; rm wordvec.glove.tar.gz
