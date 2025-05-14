#!/bin/bash
lyric=lyric/lyric.txt
dict_path=data_short/para/
data_dir=test_data
gen_at_dir=gen_at
dst=l2m_midi

stage=5
# data preparation
if [ $stage -le 0  ]
then
    echo "------------------------"
    echo "Step0: Data preparation ... "
    echo "------------------------"
    rm -r ${data_dir}
    mkdir ${data_dir}
    mkdir ${data_dir}/mono
    mkdir ${data_dir}/para
    mkdir ${data_dir}/processed

    python songmass_lyric_convert.py ${lyric} lines ${data_dir}/mono
fi

if [ $stage -le 1 ]
then
    echo "------------------------"
    echo "Step1: Copy data ... "
    echo "------------------------"
    cp ${dict_path}/dict.lyric.txt ${data_dir}/mono
    cp ${dict_path}/dict.melody.txt ${data_dir}/mono
    cp ${data_dir}/mono/valid.lyric ${data_dir}/mono/train.lyric
    cp ${data_dir}/mono/valid.melody ${data_dir}/mono/train.melody
    cp ${data_dir}/mono/* ${data_dir}/para
    cp ${data_dir}/mono/* ${data_dir}/processed
fi

# preprocessing data
if [ $stage -le 2 ]
then
    echo "------------------------"
    echo "Step2: Preprocessing data ... "
    echo "------------------------"
    sh l2m_preprocess.sh ${data_dir}
fi

# infer
if [ $stage -le 3 ]
then
    echo "------------------------"
    echo "Step3: Infer"
    echo "------------------------"
    rm -r result
    mkdir result
 
    sh l2m_pr_align_score.sh > result/l2m_test
fi

# gen_align
if [ $stage -le 4 ]
then
    echo "------------------------"
    echo "Step4: Generate Align File"
    echo "------------------------"
    cd ${gen_at_dir}
    
    sh run_all.sh ../result/l2m_test pretrain_align ../${data_dir} 
    cd ../
fi

# gen_midi
if [ $stage -le 5 ]
then
    echo "------------------------"
    echo "Step5: Generate MIDI"
    echo "------------------------"

    # get midi without lyrics
    python generate_melody_songmass.py gen_at/l2m_merge

    rm -r ${dst}
    mkdir ${dst}
    mkdir ${dst}/midi
    python convert_midi_songmass_en.py gen_at/l2m_merge ${dst}/midi
fi
