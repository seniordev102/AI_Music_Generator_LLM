# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# !/usr/bin/env python
# -*- coding: utf-8 -*-


import jieba
import json
import numpy as np
import os
import re
import random
import pypinyin
import shutil
from pypinyin import lazy_pinyin, Style
from tqdm.auto import tqdm
from utils import special_tokens, get_sentence_pinyin_finals
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass
class LyricData:
    """Data class for storing lyric information"""
    content: str
    finals: str  
    sentences: str
    pos: str
    beats: Optional[str] = None
    reverse_content: Optional[str] = None
    reverse_finals: Optional[str] = None
    reverse_pos: Optional[str] = None
    reverse_beats: Optional[str] = None
    valid: bool = True
    num_lines: int = 0


def get_shuffled_samples(a: List, b: List, c: List, d: List, e: List) -> Tuple[List, Optional[List], Optional[List], Optional[List], Optional[List]]:
    length = len(a)
    flag = [1, 1, 1, 1]
    if b == []:
        b = np.zeros(length)
        flag[0] = 0
    if c == []:
        c = np.zeros(length)
        flag[1] = 0
    if d == []:
        d = np.zeros(length)
        flag[2] = 0
    if e == []:
        e = np.zeros(length)
        flag[3] = 0
    samples = list(zip(a, b, c, d, e))
    random.shuffle(samples)
    a, b, c, d, e = zip(*samples)
    if flag[0] == 0:
        b = None
    if flag[1] == 0:
        c = None
    if flag[2] == 0:
        d = None
    if flag[3] == 0:
        e = None
    return a, b, c, d, e


def remove_prefix(text: str, prefix: str) -> str:
    while text.startswith(prefix):
        text = text[len(prefix):]
    return text


def remove_suffix(text: str, suffix: str) -> str:
    while text.endswith(suffix):
        text = text[:-len(suffix)]
    return text


def segment_text(lines: List[str]) -> List[str]:
    # jieba.enable_paddle()
    # l = ' '.join(jieba.lcut(lines[0], use_paddle=True))
    # print(l)
    all_len = len(lines)
    k = 0
    for i in range(all_len):
        try:
            line = ' '.join(jieba.lcut(lines[i]))
            lines[i] = line
        except Exception:
            k += 1
            print(line)
    print(f'{k}/{all_len}')
    return lines


def build_files_separate(num_pieces: int,
                         stride: int,
                         min_length: int,
                         lines: Optional[List[str]] = None,
                         finals: Optional[List[str]] = None,
                         sentences: Optional[List[str]] = None,
                         pos: Optional[List[str]] = None,
                         beats: Optional[List[str]] = None,
                         tokenized_data_path: Optional[str] = None,
                         finalized_data_path: Optional[str] = None,
                         sentenced_data_path: Optional[str] = None,
                         posed_data_path: Optional[str] = None,
                         beated_data_path: Optional[str] = None,
                         full_tokenizer: Optional[object] = None,
                         full_finalizer: Optional[object] = None,
                         full_sentencer: Optional[object] = None,
                         full_poser: Optional[object] = None,
                         full_beater: Optional[object] = None,
                         enable_final: bool = False,
                         enable_sentence: bool = False,
                         enable_pos: bool = False,
                         enable_beat: bool = False,
                         segment: bool = False) -> None:
    print('Start tokenizing..')
    assert len(lines) == len(finals) == len(sentences)
    if segment:
        lines = segment_text(lines)
    path = tokenized_data_path.rsplit('/', 1)[0]
    if not os.path.exists(path):
        os.mkdir(path)
    print(f'#lines: {len(lines)}')
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    if enable_final:
        print(f'#finals: {len(finals)}')
        if not os.path.exists(finalized_data_path):
            os.mkdir(finalized_data_path)
    if enable_sentence:
        print(f'#sentences: {len(sentences)}')
        if not os.path.exists(sentenced_data_path):
            os.mkdir(sentenced_data_path)
    if enable_pos:
        print(f'#pos: {len(pos)}')
        if not os.path.exists(posed_data_path):
            os.mkdir(posed_data_path)
    if enable_beat:
        print(f'#beats: {len(beats)}')
        if not os.path.exists(beated_data_path):
            os.mkdir(beated_data_path)

    all_len = len(lines)

    for k in range(num_pieces):
        max_length = stride - 2
        print(max_length)

        for i in range(len(lines)):
            line = lines[i]
            if len(line) > min_length:
                line = full_tokenizer.tokenize(line)
                line = full_tokenizer.convert_tokens_to_ids(line)
                line_length = len(line)
                skip = full_tokenizer.convert_tokens_to_ids('[SKIP]')
                skips = [skip] * max_length
                if line_length >= max_length:
                    line = line[0:max_length]
                else:
                    skips[0:line_length] = line[0:line_length]
                    line = skips

                if enable_final:
                    final = finals[i]
                    final = full_finalizer.tokenize(final)
                    final = full_finalizer.convert_tokens_to_ids(final)
                    skip = full_finalizer.convert_tokens_to_ids('[SKIP]')
                    skips = [skip] * max_length
                    if line_length >= max_length:
                        final = final[0:max_length]
                    else:
                        skips[0:line_length] = final[0:line_length]
                        final = skips
                    assert len(final) == len(line)

                if enable_sentence:
                    sentence = sentences[i]
                    sentence = full_sentencer.tokenize(sentence)
                    sentence = full_sentencer.convert_tokens_to_ids(sentence)
                    skip = full_sentencer.convert_tokens_to_ids('[SKIP]')
                    skips = [skip] * max_length
                    if line_length >= max_length:
                        sentence = sentence[0:max_length]
                    else:
                        skips[0:line_length] = sentence[0:line_length]
                        sentence = skips
                    assert len(sentence) == len(line)

                if enable_pos:
                    p = pos[i]
                    p = full_poser.tokenize(p)
                    p = full_poser.convert_tokens_to_ids(p)
                    skip = full_poser.convert_tokens_to_ids('[SKIP]')
                    skips = [skip] * max_length
                    if line_length >= max_length:
                        p = p[0:max_length]
                    else:
                        skips[0:line_length] = p[0:line_length]
                        p = skips
                    assert len(p) == len(line)

                if enable_beat:
                    beat = beats[i]
                    beat = full_beater.tokenize(beat)
                    beat = full_beater.convert_tokens_to_ids(beat)
                    skip = full_beater.convert_tokens_to_ids('[SKIP]')
                    skips = [skip] * max_length
                    if line_length >= max_length:
                        beat = beat[0:max_length]
                    else:
                        skips[0:line_length] = beat[0:line_length]
                        beat = skips
                    assert len(beat) == len(line)

                lines[i] = line
                if enable_final:
                    finals[i] = final
                if enable_sentence:
                    sentences[i] = sentence
                if enable_pos:
                    pos[i] = p
                if enable_beat:
                    beats[i] = beat

        full_line, full_final, full_sentence, full_pos, full_beat = [], [], [], [], []
        for i in range(len(lines)):
            mask = full_tokenizer.convert_tokens_to_ids('[MASK]')
            clss = full_tokenizer.convert_tokens_to_ids('[CLS]')
            full_line.append(mask)  # start of the document
            full_line.extend(lines[i])
            full_line.append(clss)  # end of the document

            if enable_final:
                mask = full_finalizer.convert_tokens_to_ids('[MASK]')
                clss = full_finalizer.convert_tokens_to_ids('[CLS]')
                full_final.append(mask)  # start of the document
                full_final.extend(finals[i])
                full_final.append(clss)  # end of the document

            if enable_sentence:
                mask = full_sentencer.convert_tokens_to_ids('[MASK]')
                clss = full_sentencer.convert_tokens_to_ids('[CLS]')
                full_sentence.append(mask)  # start of the document
                full_sentence.extend(sentences[i])
                full_sentence.append(clss)  # end of the document

            if enable_pos:
                mask = full_poser.convert_tokens_to_ids('[MASK]')
                clss = full_poser.convert_tokens_to_ids('[CLS]')
                full_pos.append(mask)  # start of the document
                full_pos.extend(pos[i])
                full_pos.append(clss)  # end of the document

            if enable_beat:
                mask = full_beater.convert_tokens_to_ids('[MASK]')
                clss = full_beater.convert_tokens_to_ids('[CLS]')
                full_beat.append(mask)  # start of the document
                full_beat.extend(beats[i])
                full_beat.append(clss)  # end of the document

        if enable_final:
            assert len(full_line) == len(full_final), f'line: {len(full_line)}, final: {len(full_final)}'
        if enable_sentence:
            assert len(full_line) == len(full_sentence), f'line: {len(full_line)}, sentence: {len(full_sentence)}'
        if enable_pos:
            assert len(full_line) == len(full_pos), f'line: {len(full_line)}, pos: {len(full_pos)}'
        if enable_beat:
            assert len(full_line) == len(full_beat), f'line: {len(full_line)}, beat: {len(full_beat)}'

        with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
            for idx in full_line:
                f.write(str(idx) + ' ')

        if enable_final:
            with open(os.path.join(finalized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_final:
                    f.write(str(idx) + ' ')

        if enable_sentence:
            with open(os.path.join(sentenced_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_sentence:
                    f.write(str(idx) + ' ')

        if enable_pos:
            with open(os.path.join(posed_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_pos:
                    f.write(str(idx) + ' ')

        if enable_beat:
            with open(os.path.join(beated_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_beat:
                    f.write(str(idx) + ' ')
    print('finish')


def build_files(num_pieces: int,
                min_length: int,
                lines: Optional[List[str]] = None,
                finals: Optional[List[str]] = None,
                sentences: Optional[List[str]] = None,
                pos: Optional[List[str]] = None,
                beats: Optional[List[str]] = None,
                tokenized_data_path: Optional[str] = None,
                finalized_data_path: Optional[str] = None,
                sentenced_data_path: Optional[str] = None,
                posed_data_path: Optional[str] = None,
                beated_data_path: Optional[str] = None,
                full_tokenizer: Optional[object] = None,
                full_finalizer: Optional[object] = None,
                full_sentencer: Optional[object] = None,
                full_poser: Optional[object] = None,
                full_beater: Optional[object] = None,
                enable_final: bool = False,
                enable_sentence: bool = False,
                enable_pos: bool = False,
                enable_beat: bool = False,
                segment: bool = False) -> None:
    print('Start tokenizing..')
    assert len(lines) == len(finals) == len(sentences)
    if segment:
        lines = segment_text(lines)
    path = tokenized_data_path.rsplit('/', 1)[0]
    if not os.path.exists(path):
        os.mkdir(path)
    print(f'#lines: {len(lines)}')
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    if enable_final:
        print(f'#finals: {len(finals)}')
        if not os.path.exists(finalized_data_path):
            os.mkdir(finalized_data_path)
    if enable_sentence:
        print(f'#sentences: {len(sentences)}')
        if not os.path.exists(sentenced_data_path):
            os.mkdir(sentenced_data_path)
    if enable_pos:
        print(f'#pos: {len(pos)}')
        if not os.path.exists(posed_data_path):
            os.mkdir(posed_data_path)
    if enable_beat:
        print(f'#beats: {len(beats)}')
        if not os.path.exists(beated_data_path):
            os.mkdir(beated_data_path)

    all_len = len(lines)
    for k in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
        if k == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (k + 1):])  # put the last documents to the last piece

        if enable_final:
            subfinals = finals[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subfinals.extend(finals[all_len // num_pieces * (k + 1):])  # put the last documents to the last piece

        if enable_sentence:
            subsentences = sentences[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subsentences.extend(sentences[all_len // num_pieces * (k + 1):])  # put the last documents to the last piece

        if enable_pos:
            subpos = pos[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subpos.extend(pos[all_len // num_pieces * (k + 1):])  # put the last documents to the last piece

        if enable_beat:
            subbeats = beats[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subbeats.extend(beats[all_len // num_pieces * (k + 1):])  # put the last documents to the last piece

        for i in range(len(sublines)):
            line = sublines[i]
            if len(line) > min_length:
                line = full_tokenizer.tokenize(line)
                line = full_tokenizer.convert_tokens_to_ids(line)

                if enable_final:
                    final = subfinals[i]
                    final = full_finalizer.tokenize(final)
                    final = full_finalizer.convert_tokens_to_ids(final)
                    assert len(final) == len(line)

                if enable_sentence:
                    sentence = subsentences[i]
                    sentence = full_sentencer.tokenize(sentence)
                    sentence = full_sentencer.convert_tokens_to_ids(sentence)
                    assert len(sentence) == len(line)

                if enable_pos:
                    p = subpos[i]
                    p = full_poser.tokenize(p)
                    p = full_poser.convert_tokens_to_ids(p)
                    assert len(p) == len(line)

                if enable_beat:
                    beat = subbeats[i]
                    beat = full_beater.tokenize(beat)
                    beat = full_beater.convert_tokens_to_ids(beat)
                    assert len(beat) == len(line)

                sublines[i] = line
                if enable_final:
                    subfinals[i] = final
                if enable_sentence:
                    subsentences[i] = sentence
                if enable_pos:
                    subpos[i] = p
                if enable_beat:
                    subbeats[i] = beat

        full_line, full_final, full_sentence, full_pos, full_beat = [], [], [], [], []
        for i in range(len(sublines)):
            mask = full_tokenizer.convert_tokens_to_ids('[MASK]')
            clss = full_tokenizer.convert_tokens_to_ids('[CLS]')
            full_line.append(mask)  # start of the document
            full_line.extend(sublines[i])
            full_line.append(clss)  # end of the document

            if enable_final:
                mask = full_finalizer.convert_tokens_to_ids('[MASK]')
                clss = full_finalizer.convert_tokens_to_ids('[CLS]')
                full_final.append(mask)  # start of the document
                full_final.extend(subfinals[i])
                full_final.append(clss)  # end of the document

            if enable_sentence:
                mask = full_sentencer.convert_tokens_to_ids('[MASK]')
                clss = full_sentencer.convert_tokens_to_ids('[CLS]')
                full_sentence.append(mask)  # start of the document
                full_sentence.extend(subsentences[i])
                full_sentence.append(clss)  # end of the document

            if enable_pos:
                mask = full_poser.convert_tokens_to_ids('[MASK]')
                clss = full_poser.convert_tokens_to_ids('[CLS]')
                full_pos.append(mask)  # start of the document
                full_pos.extend(subpos[i])
                full_pos.append(clss)  # end of the document

            if enable_beat:
                mask = full_beater.convert_tokens_to_ids('[MASK]')
                clss = full_beater.convert_tokens_to_ids('[CLS]')
                full_beat.append(mask)  # start of the document
                full_beat.extend(subbeats[i])
                full_beat.append(clss)  # end of the document

        if enable_final:
            assert len(full_line) == len(full_final), f'line: {len(full_line)}, final: {len(full_final)}'
        if enable_sentence:
            assert len(full_line) == len(full_sentence), f'line: {len(full_line)}, sentence: {len(full_sentence)}'
        if enable_pos:
            assert len(full_line) == len(full_pos), f'line: {len(full_line)}, pos: {len(full_pos)}'
        if enable_beat:
            assert len(full_line) == len(full_beat), f'line: {len(full_line)}, beat: {len(full_beat)}'

        with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
            for idx in full_line:
                f.write(str(idx) + ' ')

        if enable_final:
            with open(os.path.join(finalized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_final:
                    f.write(str(idx) + ' ')

        if enable_sentence:
            with open(os.path.join(sentenced_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_sentence:
                    f.write(str(idx) + ' ')

        if enable_pos:
            with open(os.path.join(posed_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_pos:
                    f.write(str(idx) + ' ')

        if enable_beat:
            with open(os.path.join(beated_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_beat:
                    f.write(str(idx) + ' ')
    print('finish')


""" Processing Lyrics Data """


def process_lyric(ins_path: str = 'data/lyrics/RAP_DATASET_LYRIC/', out_path: str = 'data/lyrics/RAP_DATASET_LYRIC_valid/', invalid_songs: set = set([])) -> set:
    """
    preprocssing lyrics: remove non-lyric symbols, remove empty lines.
    homepath = '/ssddata/lxueaa/controllable-text-generation/data'
    lyric_base = f'{homepath}/lyrics/RAP_DATASET_LYRIC'
    :return: list of invalid song path
    """
    i = 0  # total num
    j = 0  # number of empty songs

    # enumerate singers
    for rap_name in os.listdir(ins_path):
        rap_path = os.path.join(ins_path, rap_name)

        if os.path.isdir(rap_path):
            # enumerate album dirs
            for s_name in os.listdir(rap_path):
                s_path = os.path.join(rap_path, s_name)

                if os.path.isdir(s_path):

                    # enumerate songs
                    for song_name in os.listdir(s_path):
                        i += 1
                        lyric_path = os.path.join(s_path, song_name, f'{song_name}_content.txt')
                        if os.path.exists(lyric_path):
                            finals_path = os.path.join(s_path, song_name, f'{song_name}_finals.txt')
                            with open(finals_path, 'w') as of:
                                with open(lyric_path) as f:
                                    for line in f:
                                        r = line.index(']')
                                        time = line[:r+1]
                                        content = line[r:]
                                        finals = get_sentence_pinyin_finals(content)
                                        finals = ' '.join(finals).rstrip(' \r\n')

                                        of.write(f'{time + finals}\n')
                        else:
                            j += 1
                            invalid_songs.add(lyric_path)
    print(f'End. Total songs:  {i}, invalid songs: {j}, left songs: {i - j}')

    return invalid_songs


def read_lyrics(root_path: str, reverse: bool = False) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    out_path = os.path.join(root_path, 'train')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # check whether preprocessed cache exists or not
    lines = []
    finals = []
    sentences = []
    pos = []
    beats = []

    reverse_str = '_reverse' if reverse else ''
    out_content_path = f'{out_path}/content{reverse_str}.json'
    out_finals_path = f'{out_path}/finals{reverse_str}.json'
    out_sentences_path = f'{out_path}/sentences{reverse_str}.json'
    out_pos_path = f'{out_path}/pos{reverse_str}.json'
    out_beats_path = f'{out_path}/beats{reverse_str}.json'

    # read cached data
    if os.path.exists(out_content_path) and os.path.exists(out_sentences_path) and \
       os.path.exists(out_finals_path) and os.path.exists(out_pos_path) and \
       os.path.exists(out_beats_path):
        # load cached data
        with open(out_content_path, encoding='utf8') as ins:
            for line in ins:
                lines.append(line)
        with open(out_finals_path, encoding='utf8') as ins:
            for line in ins:
                finals.append(line)
        with open(out_sentences_path, encoding='utf8') as ins:
            for line in ins:
                sentences.append(line)
        with open(out_pos_path, encoding='utf8') as ins:
            for line in ins:
                pos.append(line)
        with open(out_beats_path, encoding='utf8') as ins:
            for line in ins:
                beats.append(line)
        return lines, finals, sentences, pos, beats

    # If not exists, to preprocess data
    # process new data
    print('Start to read processed lyrics from dataset....')
    ins_path = os.path.join(root_path, 'lyrics.json')
    with open(ins_path, encoding='utf8') as ins:
        # enumerate each line in the file
        # each line is an article
        i = j = 0
        for line in ins:
            song = eval(json.loads(line))
            # print(type(song))
            if song['valid']:
                if not reverse:
                    lines.append(song['lyric'])
                    finals.append(song['vowel'])
                    pos.append(song['pos'])
                    beats.append(song['beat'])
                else:
                    lines.append(song['lyric-reverse'])
                    finals.append(song['vowel-reverse'])
                    pos.append(song['pos-reverse'])
                    beats.append(song['beat-reverse'])
                sentences.append(song['sentence'])
                i += 1
            else:
                # print(l)
                j += 1
        print(f'valid: {i}, invalid: {j}')

    with open(out_content_path, mode='w', encoding='utf8') as f:
        for line in lines:
            f.write(f'{line}\n')
    with open(out_finals_path, mode='w', encoding='utf8') as f:
        for final in finals:
            f.write(f'{final}\n')
    with open(out_sentences_path, mode='w', encoding='utf8') as f:
        for sentence in sentences:
            f.write(f'{sentence}\n')
    with open(out_pos_path, mode='w', encoding='utf8') as f:
        for p in pos:
            f.write(f'{p}\n')
    with open(out_beats_path, mode='w', encoding='utf8') as f:
        for beat in beats:
            f.write(f'{beat}\n')
    return lines, finals, sentences, pos, beats


def get_beat_token(cnt: int, line: str) -> Tuple[int, str]:
    lines = line.split()
    beat = ['0'] * len(lines)
    for idx, item in enumerate(lines):
        if item == '[BEAT]':
            cnt += 1
            beat[idx] = str(cnt)
    beat = ' '.join(beat) + ' '
    return cnt, beat


def get_inner_pos(line: str) -> str:
    lines = line.split()
    pos = ['0'] * len(lines)
    cnt = 0
    for idx, item in enumerate(lines):
        if item in special_tokens:
            pos[idx] = item
        else:
            pos[idx] = str(cnt)
            cnt += 1
    pos = ' '.join(pos) + ' '
    return pos


def parse_lyric(l_content_path: str, l_finals_path: str, with_beat: bool = False, beat_mode: int = 0) -> Tuple[str, str, str, str, str, str, str, str, str, bool, int]:
    lyric = ''
    lyric_reverse = ''
    sentence = ''
    with open(l_content_path) as f:
        num_line = 0
        valid = False
        for line in f:
            # line format: [00:12.338]rap god rap gpd
            if ']' in line:
                j = line.index(']')
                line = line[j + 1:]
                if beat_mode == 1 and num_line == 0:
                    tempo = line[:3]
                    line = line[3:]

            # ignore begin lines
            if ':' in line or '：' in line:
                continue

            if with_beat:
                line = line.strip(' \r\n').lstrip(' ')
                if beat_mode == 1:
                    line_reverse = '[BEAT]'.join(line[::-1].split(']TAEB['))
                    if num_line == 0:
                        line = tempo + line
                        line_reverse = tempo + line_reverse
                elif beat_mode == 2:
                    line_reverse = line[::-1]
                    line_reverse = '[S]'.join(line_reverse.split(']S['))
                    line_reverse = '[M]'.join(line_reverse.split(']M['))
                    line_reverse = '[F]'.join(line_reverse.split(']F['))
                else:
                    line_reverse = '[BEAT]'.join(line[::-1].split(']TAEB['))

            else:
                line = line.strip(' \r\n')
                line_reverse = line[::-1]
            line = re.sub('\s+', '[PAD]', line)
            line_reverse = re.sub('\s+', '[PAD]', line_reverse)
            assert len(line) == len(line_reverse)

            if len(line) == 0:  # end of block
                if len(lyric) > 0:  # not start of the file
                    continue
            else:
                line_reverse += '[SEP]'
                line += '[SEP]'
                nSEP = len(re.findall('\[SEP\]', line))
                nPAD = len(re.findall('\[PAD\]', line))

                if with_beat:
                    nBEAT = len(re.findall('\[BEAT\]', line))
                    if beat_mode != 0:
                        nSMF = len(re.findall('\[S\]', line)) + \
                               len(re.findall('\[M\]', line)) + \
                               len(re.findall('\[F\]', line))
                    else:
                        nSMF = 0
                    
                    nids = len(line) - 4 * (nSEP + nPAD) - 5 * nBEAT - 2 * nSMF
                else:
                    nids = len(line) - 4 * (nSEP + nPAD)
                ids = [str(num_line) for k in range(nids)]
                    
                sentence += ' '.join(ids) + ' '
                num_line += 1
                
            lyric += line
            lyric_reverse += line_reverse
    
    final = final_reverse = ''
    innerpos = innerpos_reverse = ''
    beat = beat_reverse = ''
    cnt = rcnt = 0
    with open(l_finals_path) as f:
        num_line = 0
        for line in f:
            # line format: [00:12.338]rap god rap god
            if ']' in line:
                i = line.index(']')
                line = line[i + 1:]
                if beat_mode == 1 and num_line == 0:
                    tempo = line[:4]
                    line = line[4:]

            # ignore begin lines
            if ':' in line or '：' in line:
                continue
            
            if with_beat:
                line = remove_prefix(line.strip(' \r\n'), '[SEP] ')
                line = remove_prefix(line, '[PAD] ')
                line = remove_suffix(line, ' [PAD]')
                line = remove_suffix(line, '[PAD]')
                line = remove_suffix(line, ' [SEP]')
                line = re.sub('(\[SEP\])', '[PAD]', line)
                line = re.sub('(\[PAD\]\s)+', '[PAD] ', line)
                if line == '[PAD]':
                    continue
                line = ' '.join(line.split())
            else:    
                line = line.strip(' \r\n')
            line_reverse = ' '.join(line.split()[::-1])
            if beat_mode == 1 and num_line == 0:
                line = tempo + ' ' + line
                line_reverse = tempo + ' ' + line_reverse
            
            if len(line) == 0:  # end of block
                if len(final) > 0:  # not start of the file
                    continue
            else:
                line_reverse += ' [SEP] '
                line += ' [SEP] '
                num_line += 1
            if with_beat:
                cnt, lbeat = get_beat_token(cnt, line)
                rcnt, lbeat_reverse = get_beat_token(rcnt, line_reverse)
            lpos = get_inner_pos(line)
            lpos_reverse = get_inner_pos(line_reverse)
                
            final += line
            final_reverse += line_reverse
            if with_beat:
                beat += lbeat
                beat_reverse += lbeat_reverse
            innerpos += lpos
            innerpos_reverse += lpos_reverse

    lyric, final, sentence, innerpos = lyric.strip(' \n'), final.strip(' \n'), sentence.strip(' \n'), innerpos.strip(' \n')
    lyric_reverse, final_reverse, innerpos_reverse = lyric_reverse.strip(' \n'), final_reverse.strip(' \n'), innerpos_reverse.strip(' \n')
    if with_beat:
        beat, beat_reverse = beat.strip(' \n'), beat_reverse.strip(' \n')

    len_lyric = len(lyric) - \
        4 * (len(re.findall('\[SEP\]', lyric)) + len(re.findall('\[PAD\]', lyric))) - \
        5 * len(re.findall('\[BEAT\]', lyric)) - \
        2 * (len(re.findall('\[S\]', lyric)) + len(re.findall('\[M\]', lyric)) + len(re.findall('\[F\]', lyric)))
    len_final = len(final.split())
    len_sentence = len(sentence.split())
    try:
        assert len_lyric == len_final == len_sentence
    except Exception:
        print(len_lyric, len_final, len_sentence)
        print(lyric)
        print(final)
        print(l_content_path)
        return
    
    if num_line > 4:
        valid = True

    return lyric, lyric_reverse, final, final_reverse, sentence, innerpos, innerpos_reverse, beat, beat_reverse, valid, num_line


def prepare_lyrics(ins_path: str, out_path: str, with_beat: bool = False, beat_mode: int = 0) -> None:
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    out_path = os.path.join(out_path, 'lyrics.json')
    if os.path.exists(out_path):
        while True:
            ins = input('Found cached files...Continue to overwrite? (Y/N)\n')
            if ins == 'Y':
                print('Start to reprocess raw data...')
                break
            elif ins == 'N':
                print('Use cached files.')
                return
            else:
                print('Invalid inputs.')

    with open(out_path, 'w', encoding='utf8') as outs:
        l_info = {}  # lyric info
        # enumerate singers
        i = 0  # total num
        j = 0  # number of empty songs
        max_num_lines = 0
        for s_path in os.listdir(ins_path):
            l_info['singer'] = s_path
            s_path = os.path.join(ins_path, s_path)

            if os.path.isdir(s_path):
                # enumerate album
                for a_path in os.listdir(s_path):
                    l_info['album'] = a_path

                    a_path = os.path.join(s_path, a_path)

                    if os.path.isdir(a_path):
                        # enumerate songs
                        for l_path in os.listdir(a_path):
                            l_file_name = l_path
                            l_path = os.path.join(a_path, l_path)
                                                  
                            if os.path.isdir(l_path):
                                # enumerate lyric
                                for l_song in os.listdir(l_path):
                                    l_info['song'] = l_file_name  # remove '_content.txt' extension
                                    
                                    if with_beat:
                                        if beat_mode == 0:
                                            if l_song != 'mapped_final_with_beat.txt':
                                                continue
                                            l_content_path = os.path.join(l_path, 'lyric_with_beat.txt')
                                            l_finals_path = os.path.join(l_path, 'mapped_final_with_beat.txt')
                                        elif beat_mode == 1:
                                            if l_song != 'mapped_final_with_beat_global.txt':
                                                continue
                                            l_content_path = os.path.join(l_path, 'lyric_with_beat_global.txt')
                                            l_finals_path = os.path.join(l_path, 'mapped_final_with_beat_global.txt')
                                        elif beat_mode == 2:
                                            if l_song != 'mapped_final_with_beat_local.txt':
                                                continue
                                            l_content_path = os.path.join(l_path, 'lyric_with_beat_local.txt')
                                            l_finals_path = os.path.join(l_path, 'mapped_final_with_beat_local.txt')
                                    else:
                                        # if l_song[-5] != 't':
                                        if l_song != 'mapped_final_with_beat.txt':
                                            continue
                                        # l_content_path = os.path.join(l_path, l_file_name+'_content.txt')
                                        # l_finals_path = os.path.join(l_path, l_file_name+'_mapped_finals.txt')
                                        l_content_path = os.path.join(l_path, 'lyric_with_beat.txt')
                                        l_finals_path = os.path.join(l_path, 'mapped_final_with_beat.txt')
                                    if os.path.isfile(l_content_path):
                                        l_info['lyric'], l_info['lyric-reverse'], l_info['vowel'], \
                                        l_info['vowel-reverse'], l_info['sentence'], l_info['pos'], \
                                        l_info['pos-reverse'], l_info['beat'], l_info['beat-reverse'], \
                                        l_info['valid'], num_lines = parse_lyric(l_content_path, l_finals_path, with_beat, beat_mode)
#                                         print(l_info)
                                        if max_num_lines < num_lines:
                                            max_num_lines = num_lines

                                    l_info_str = str(l_info)
                                    outs.write(f'{json.dumps(l_info_str, ensure_ascii=False)}\n')
                                    if not l_info['valid']:
                                        j += 1

                                    i += 1
                                    if i % 1000 == 0:
                                        print(f'Processed songs:{i}', end='\r', flush=True)
  
    print(f'End. Total songs:  {i}, invalid songs: {j}, left songs: {i - j}, max line in song: {max_num_lines}.')

    
if __name__ == '__main__':

    prepare_lyrics(ins_path='data/lyrics/lyrics_with_finals_large', 
                   out_path='data/lyrics/lyrics/lyrics', 
                   with_beat=False, 
                   beat_mode=0)
    read_lyrics(path='data/lyrics/lyrics', 
                out_content_path='data/lyrics/lyrics/train/content',
                out_finals_path='data/lyrics/lyrics/train/finals',
                out_sentences_path='data/lyrics/lyrics/train/sentences',
                out_pos_path='data/lyrics/lyrics/train/pos',
                out_beats_path='data/lyrics/lyrics/train/beats',
                reverse=True, 
                with_beat=False,
                beat_mode=0)

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from tqdm.auto import tqdm
from config.data_config import DataConfig

class FileHandler:
    """Handles file operations with better error handling and validation"""
    
    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Ensure directory exists, create if not"""
        path.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def validate_path(path: Path, must_exist: bool = True) -> bool:
        """Validate that a path exists if required"""
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
        return True
        
    @staticmethod
    def safe_read_json(path: Path) -> Dict:
        """Safely read and parse JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {str(e)}")
        except Exception as e:
            raise IOError(f"Error reading {path}: {str(e)}")
            
    @staticmethod
    def safe_write_json(data: Dict, path: Path) -> None:
        """Safely write data to JSON file"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise IOError(f"Error writing to {path}: {str(e)}")

class DataValidator:
    """Validates data integrity and format"""
    
    @staticmethod
    def validate_lyric_format(lyric: Dict) -> bool:
        """Validate lyric data format"""
        required_fields = ['content', 'finals', 'sentences', 'pos']
        
        for field in required_fields:
            if field not in lyric:
                raise ValueError(f"Missing required field: {field}")
                
        if not isinstance(lyric['content'], str):
            raise ValueError("Content must be string")
            
        return True
        
    @staticmethod
    def validate_line_counts(lines: List[str], finals: List[str], sentences: List[str]) -> bool:
        """Validate that all data lists have matching lengths"""
        if not (len(lines) == len(finals) == len(sentences)):
            raise ValueError("Mismatched lengths in data lists")
        return True
        
    @staticmethod
    def validate_beat_format(beat_data: str, beat_mode: int) -> bool:
        """Validate beat data format"""
        if beat_mode not in [0, 1, 2]:
            raise ValueError(f"Invalid beat mode: {beat_mode}")
            
        # Add specific validation for each beat mode
        if beat_mode == 1:
            if not beat_data.startswith('['):
                raise ValueError("Invalid beat format for mode 1")
        
        return True

# Update the DataProcessor to use these new classes
class DataProcessor:
    """Handles all data processing operations for lyrics and related data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self.file_handler = FileHandler()
        self.validator = DataValidator()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def process_dataset(self) -> None:
        """Main entry point for processing the entire dataset"""
        try:
            self.logger.info("Starting dataset processing...")
            
            # Create output directories if they don't exist
            self.config.output_path.mkdir(parents=True, exist_ok=True)
            
            # Process lyrics first
            self._process_lyrics()
            
            # Then process the files into training format
            self._process_training_files()
            
            self.logger.info("Dataset processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            raise
            
    def _process_lyrics(self) -> None:
        """Process raw lyrics with improved error handling"""
        output_file = self.config.output_path / 'lyrics.json'
        
        try:
            self.file_handler.ensure_directory(output_file.parent)
            
            if output_file.exists() and not self._should_overwrite():
                self.logger.info("Using existing processed lyrics")
                return
                
            stats = self._process_lyric_files(output_file)
            self._log_processing_stats(stats)
            
        except Exception as e:
            self.logger.error(f"Fatal error processing lyrics: {str(e)}")
            raise
            
    def _process_lyric_files(self, output_file: Path) -> Dict[str, int]:
        """Process individual lyric files with validation"""
        stats = {'total': 0, 'invalid': 0, 'max_lines': 0}
        
        try:
            with open(output_file, 'w', encoding='utf8') as f:
                for lyric in self._iterate_lyrics():
                    try:
                        self.validator.validate_lyric_format(lyric.__dict__)
                        if self.config.with_beat:
                            self.validator.validate_beat_format(lyric.beats, self.config.beat_mode)
                            
                        stats['total'] += 1
                        if not lyric.valid:
                            stats['invalid'] += 1
                        stats['max_lines'] = max(stats['max_lines'], lyric.num_lines)
                        
                        if stats['total'] % 1000 == 0:
                            self.logger.info(f"Processed {stats['total']} songs")
                            
                        json.dump(self._to_dict(lyric), f, ensure_ascii=False)
                        f.write('\n')
                        
                    except ValueError as e:
                        self.logger.warning(f"Invalid lyric format: {str(e)}")
                        stats['invalid'] += 1
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error writing processed lyrics: {str(e)}")
            raise
            
        return stats
        
    def _process_training_files(self) -> None:
        """Process lyrics into training format"""
        try:
            lines, finals, sentences, pos, beats = self._read_lyrics()
            
            self._build_files(
                lines=lines,
                finals=finals, 
                sentences=sentences,
                pos=pos,
                beats=beats
            )
            
        except Exception as e:
            self.logger.error(f"Error processing training files: {str(e)}")
            raise
            
    def _read_lyrics(self) -> Tuple[List[str], ...]:
        """Read processed lyrics from json files"""
        self.logger.info("Reading processed lyrics...")
        
        lines, finals, sentences, pos, beats = [], [], [], [], []
        
        try:
            with open(self.config.output_path / 'lyrics.json', encoding='utf8') as f:
                for line in f:
                    song = json.loads(line)
                    if song['valid']:
                        lines.append(song['lyric'])
                        finals.append(song['vowel'])
                        sentences.append(song['sentence'])
                        pos.append(song['pos'])
                        if self.config.with_beat:
                            beats.append(song['beat'])
                            
        except Exception as e:
            self.logger.error(f"Error reading lyrics: {str(e)}")
            raise
            
        return lines, finals, sentences, pos, beats
        
    def _should_overwrite(self) -> bool:
        """Check if existing files should be overwritten"""
        while True:
            response = input('Found cached files. Continue to overwrite? (Y/N)\n')
            if response.upper() == 'Y':
                return True
            elif response.upper() == 'N':
                return False
    
    @staticmethod
    def _to_dict(lyric: LyricData) -> Dict[str, Any]:
        """Convert LyricData to dictionary format"""
        return {
            'lyric': lyric.content,
            'vowel': lyric.finals,
            'sentence': lyric.sentences,
            'pos': lyric.pos,
            'beat': lyric.beats,
            'lyric-reverse': lyric.reverse_content,
            'vowel-reverse': lyric.reverse_finals,
            'pos-reverse': lyric.reverse_pos,
            'beat-reverse': lyric.reverse_beats,
            'valid': lyric.valid,
            'num_lines': lyric.num_lines
        }

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class DataConfig:
    """Configuration for data processing"""
    input_path: Path
    output_path: Path
    beat_mode: int = 1
    with_beat: bool = True
    max_length: int = 512
    min_length: int = 4
    overwrite: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)
        
        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {self.input_path}")
            
        if self.beat_mode not in [0, 1, 2]:
            raise ValueError(f"Invalid beat mode: {self.beat_mode}")
            
        if self.max_length < self.min_length:
            raise ValueError("max_length must be greater than min_length")

def create_default_config(
    input_path: str,
    output_path: str,
    **kwargs
) -> DataConfig:
    """Create a configuration with default values"""
    return DataConfig(
        input_path=Path(input_path),
        output_path=Path(output_path),
        **kwargs
    )

# Update main function to use the new config
def main():
    """Main entry point with configuration management"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to input data")
    parser.add_argument("--output_path", required=True, help="Path for processed output")
    parser.add_argument("--beat_mode", type=int, default=1, help="Beat processing mode (0,1,2)")
    parser.add_argument("--with_beat", action="store_true", help="Include beat processing")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--min_length", type=int, default=4, help="Minimum sequence length")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    try:
        config = create_default_config(**vars(args))
        processor = DataProcessor(config)
        processor.process()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

import logging
import sys
from typing import List, Dict, Any
import json

class DataProcessor:
    """Handles data processing operations with proper error handling"""
    def __init__(self, config: DataConfig):
        self.config = config
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def process(self) -> None:
        """Main processing pipeline"""
        try:
            self._setup_output_directory()
            data = self._load_input_data()
            processed_data = self._process_data(data)
            self._save_processed_data(processed_data)
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise

    def _setup_output_directory(self) -> None:
        """Ensure output directory exists"""
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        
    def _load_input_data(self) -> List[Dict[str, Any]]:
        """Load and validate input data"""
        try:
            with open(self.config.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Input data must be a JSON array")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input file: {str(e)}")
            
    def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the data according to configuration"""
        processed = []
        for item in data:
            try:
                if len(item.get('tokens', [])) < self.config.min_length:
                    self.logger.debug(f"Skipping item: too short ({len(item.get('tokens', []))})")
                    continue
                    
                processed_item = self._process_item(item)
                if processed_item:
                    processed.append(processed_item)
                    
            except Exception as e:
                self.logger.warning(f"Failed to process item: {str(e)}")
                continue
                
        return processed
        
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process individual data item with beat handling"""
        if not self._validate_item(item):
            return None
            
        processed = item.copy()
        if self.config.with_beat:
            processed = self._add_beat_info(processed)
            
        # Truncate if needed
        if len(processed['tokens']) > self.config.max_length:
            processed['tokens'] = processed['tokens'][:self.config.max_length]
            
        return processed
        
    def _validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate individual data item"""
        required_fields = ['tokens']
        return all(field in item for field in required_fields)
        
    def _add_beat_info(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Add beat information based on beat_mode"""
        if self.config.beat_mode == 0:
            return item
        # Implementation for beat modes 1 and 2
        # ...existing beat processing code...
        return item
        
    def _save_processed_data(self, data: List[Dict[str, Any]]) -> None:
        """Save processed data with error handling"""
        output_file = self.config.output_path / "processed_data.json"
        if output_file.exists() and not self.config.overwrite:
            raise FileExistsError(f"Output file already exists: {output_file}")
            
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Successfully saved processed data to {output_file}")
        except IOError as e:
            raise IOError(f"Failed to save processed data: {str(e)}")

class TrainingDataPreparator:
    """Handles preparation of training data with proper error handling and validation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def prepare_training_files(self, 
                             lines: List[str],
                             finals: List[str],
                             sentences: List[str],
                             pos: Optional[List[str]] = None,
                             beats: Optional[List[str]] = None) -> None:
        """Prepare training files from processed data"""
        try:
            self._validate_inputs(lines, finals, sentences)
            
            if self.config.segment:
                lines = segment_text(lines)
                
            self._create_output_directories()
            self._process_training_data(lines, finals, sentences, pos, beats)
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training files: {str(e)}")
            raise
            
    def _validate_inputs(self,
                        lines: List[str],
                        finals: List[str],
                        sentences: List[str]) -> None:
        """Validate input data consistency"""
        if not lines or not finals or not sentences:
            raise ValueError("Empty input data")
            
        if not (len(lines) == len(finals) == len(sentences)):
            raise ValueError("Mismatched lengths in input data")
            
    def _create_output_directories(self) -> None:
        """Create necessary output directories"""
        paths = [
            self.config.output_path / 'tokenized',
            self.config.output_path / 'finalized',
            self.config.output_path / 'sentenced',
            self.config.output_path / 'posed',
            self.config.output_path / 'beated'
        ]
        
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
            
    def _process_training_data(self,
                             lines: List[str],
                             finals: List[str],
                             sentences: List[str],
                             pos: Optional[List[str]],
                             beats: Optional[List[str]]) -> None:
        """Process and save training data"""
        for i in tqdm(range(self.config.num_pieces), desc="Processing pieces"):
            try:
                piece_data = self._prepare_piece(
                    lines[i:i+self.config.max_length],
                    finals[i:i+self.config.max_length],
                    sentences[i:i+self.config.max_length],
                    pos[i:i+self.config.max_length] if pos else None,
                    beats[i:i+self.config.max_length] if beats else None
                )
                
                self._save_piece(piece_data, i)
                
            except Exception as e:
                self.logger.warning(f"Failed to process piece {i}: {str(e)}")
                continue
                
    def _prepare_piece(self,
                      lines: List[str],
                      finals: List[str],
                      sentences: List[str],
                      pos: Optional[List[str]],
                      beats: Optional[List[str]]) -> Dict[str, List]:
        """Prepare a single training piece"""
        piece_data = {
            'lines': self._tokenize_and_pad(lines),
            'finals': self._tokenize_and_pad(finals),
            'sentences': self._tokenize_and_pad(sentences)
        }
        
        if pos is not None:
            piece_data['pos'] = self._tokenize_and_pad(pos)
        if beats is not None:
            piece_data['beats'] = self._tokenize_and_pad(beats)
            
        return piece_data
        
    def _tokenize_and_pad(self, data: List[str]) -> List[int]:
        """Tokenize and pad sequence to max_length"""
        tokenized = []
        for item in data:
            if len(item) < self.config.min_length:
                continue
                
            tokens = self._tokenize(item)
            if len(tokens) > self.config.max_length:
                tokens = tokens[:self.config.max_length]
            else:
                tokens.extend([self.config.pad_token] * (self.config.max_length - len(tokens)))
                
            tokenized.extend(tokens)
            
        return tokenized
        
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text using appropriate tokenizer"""
        # Implementation depends on tokenizer type
        pass
        
    def _save_piece(self, piece_data: Dict[str, List], piece_idx: int) -> None:
        """Save processed piece data"""
        for data_type, data in piece_data.items():
            output_file = self.config.output_path / data_type / f'train_{piece_idx}.txt'
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(' '.join(map(str, data)))
            except Exception as e:
                self.logger.error(f"Failed to save {data_type} piece {piece_idx}: {str(e)}")
                raise

# Update the main processing logic to use the new class
def process_training_data(config: DataConfig) -> None:
    """Main entry point for training data preparation"""
    try:
        preparator = TrainingDataPreparator(config)
        
        # Load and preprocess data
        data = load_data(config.input_path)
        lines, finals, sentences = preprocess_data(data)
        
        # Prepare training files
        preparator.prepare_training_files(lines, finals, sentences)
        
    except Exception as e:
        logging.error(f"Training data preparation failed: {str(e)}")
        raise

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import jieba
import pypinyin
from pypinyin import lazy_pinyin, Style

from dataclasses import dataclass
from segment_search import segment_text

@dataclass
class DataConfig:
    """Configuration for training data preparation"""
    input_path: Path
    output_path: Path
    max_length: int
    min_length: int
    num_pieces: int
    pad_token: int
    segment: bool = False

def load_data(input_path: Path) -> List[str]:
    """Load raw data from input path"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Failed to load data from {input_path}: {str(e)}")
        raise

def preprocess_data(data: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Preprocess raw data into required formats"""
    try:
        lines = []
        finals = []
        sentences = []
        
        for line in data:
            # Basic text processing
            processed_line = ' '.join(jieba.cut(line))
            lines.append(processed_line)
            
            # Get pinyin finals
            pinyin_list = lazy_pinyin(line, style=Style.FINALS)
            finals.append(' '.join(pinyin_list))
            
            # Simple sentence segmentation
            sentences.append(line.replace('。', ' 。 ').replace('，', ' ， '))
            
        return lines, finals, sentences
        
    except Exception as e:
        logging.error(f"Failed to preprocess data: {str(e)}")
        raise

class DataPreparator:
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def prepare(self) -> None:
        """Main method to prepare training data"""
        try:
            # Load and preprocess data
            raw_data = load_data(self.config.input_path)
            lines, finals, sentences = preprocess_data(raw_data)

            # Filter by length
            filtered_data = self._filter_by_length(lines, finals, sentences)
            
            # Segment if required
            if self.config.segment:
                filtered_data = self._segment_data(filtered_data)

            # Save processed data
            self._save_data(filtered_data)
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {str(e)}")
            raise

    def _filter_by_length(self, lines: List[str], finals: List[str], 
                         sentences: List[str]) -> List[Tuple[str, str, str]]:
        """Filter data by configured length constraints"""
        filtered = []
        for line, final, sent in zip(lines, finals, sentences):
            if self.config.min_length <= len(line) <= self.config.max_length:
                filtered.append((line, final, sent))
        return filtered[:self.config.num_pieces]

    def _segment_data(self, data: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Apply text segmentation if configured"""
        segmented = []
        for line, final, sent in data:
            segments = segment_text(sent)
            for seg in segments:
                if self.config.min_length <= len(seg) <= self.config.max_length:
                    # Process segment to get corresponding finals
                    seg_finals = ' '.join(lazy_pinyin(seg, style=Style.FINALS))
                    seg_sent = seg.replace('。', ' 。 ').replace('，', ' ， ')
                    segmented.append((seg, seg_finals, seg_sent))
        return segmented

    def _save_data(self, data: List[Tuple[str, str, str]]) -> None:
        """Save processed data to output files"""
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        
        lines, finals, sentences = zip(*data)
        
        files = {
            'lines.txt': lines,
            'finals.txt': finals,
            'sentences.txt': sentences
        }
        
        for filename, content in files.items():
            output_file = self.config.output_path / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in content:
                    f.write(f"{line}\n")

def main():
    parser = argparse.ArgumentParser(description='Prepare training data for LLM')
    parser.add_argument('--input-path', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--min-length', type=int, default=10)
    parser.add_argument('--num-pieces', type=int, default=1000)
    parser.add_argument('--pad-token', type=int, default=0)
    parser.add_argument('--segment', action='store_true')
    
    args = parser.parse_args()
    config = DataConfig(**vars(args))
    
    logging.basicConfig(level=logging.INFO)
    
    preparator = DataPreparator(config)
    preparator.prepare()

if __name__ == '__main__':
    main()