#/usr/bin/bash

transcriptions=$1
find ${transcriptions} -name "*.tokenized" | xargs -I '{}' realpath '{}' > TMP.list
list=TMP.list
root='joint-disfluency-detector-and-parser'
while read inp; do
    cd ${root};
    out=`echo $inp| sed 's/tokenized/disdetparse/'`
    echo $inp
    echo $out
    python3 src/main.py parse --input-path $inp --output-path $out --model-path-base best_models/swbd_fisher_bert_Edev.0.9078.pt --eval-batch-size 1 >> best_models/out.log
done < $list
cd ..; rm $list
