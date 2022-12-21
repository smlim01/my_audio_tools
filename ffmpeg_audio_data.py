import argparse
import os
from functools import partial
from multiprocessing import Pool

import ffmpeg
from tqdm import tqdm


def ffmpeg_audio_data(data_dict):
    os.makedirs(
        os.path.dirname(data_dict['output_path']), 
        exist_ok=True)
    
    # Do ffmpeg 
    (
        ffmpeg
        .input(
            data_dict['input_path']
        )
        .output(
            data_dict['output_path'],
            acodec=data_dict['acodec'],
            ac=data_dict['ac'],
            ar=data_dict['sr'],
            af="aresample=resampler=soxr"
        )
        .global_args('-loglevel', 'quiet')
        .overwrite_output()
        .run()
    )

def search_input_audio(input_dir, input_ext='.wav'):
    input_audio_list = []
    for (_path, _dir, files) in os.walk(input_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == input_ext:
                input_audio_list.append(os.path.join(_path, filename))
    return input_audio_list

def main(args):
    input_audio_list = search_input_audio(args.input_dir, input_ext=args.input_ext)
    print(f"Num of input audio: {len(input_audio_list)}")

    # Prepare data_dict for multi processing
    data_dict_list = []
    for idx, input_audio in enumerate(input_audio_list):
        data_dict = dict()
        data_dict['input_path'] = input_audio
        data_dict['output_path'] = os.path.join(args.output_dir, 
            os.path.splitext(
                input_audio[len(args.input_dir):].strip(os.path.sep)
            )[0] + args.ext
        )
        data_dict['ac'] = args.ac
        data_dict['acodec'] = args.acodec
        data_dict['sr'] = args.sr
        data_dict_list.append(data_dict)

    # Multi processing
    with Pool(args.worker) as p:
        results = list(tqdm(p.imap(ffmpeg_audio_data, data_dict_list)
            , total=len(data_dict_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFmpeg audio data')
    parser.add_argument('input_dir', type=str, 
                        help='Input audio data directory')
    parser.add_argument('output_dir', type=str, 
                        help='Output audio data directory')
    parser.add_argument('--input-ext', type=str, default=".wav", 
                        help='Input audio extension')
    parser.add_argument('--ext', type=str, default=".wav", 
                        help='Output audio extension')
    parser.add_argument('--ac', type=int, default=1, 
                        help='Output audio num of channel')
    parser.add_argument('--acodec', type=str, default="pcm_s16le", 
                        help='Output audio acodec')
    parser.add_argument('--sr', type=int, 
                        help='Output audio sampling rate')
    parser.add_argument('--worker', type=int, default=8,
                        help='Number of workers')
    args = parser.parse_args()
    
    main(args)
