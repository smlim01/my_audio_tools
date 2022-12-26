import argparse
import os
import sys
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def librosa_audio_dir(data_dict):
    # Load
    y_source, sr_source = librosa.load(
        data_dict['input_path'], 
        sr=None,
        mono=data_dict['mono'],
        res_type='soxr_hq',
    )
    y_out, sr_out = deepcopy(y_source), sr_source

    # Resampling
    if data_dict['resample'] == True and sr_source != data_dict['sr']:
        y_out = librosa.resample(y_out, orig_sr=sr_source, target_sr=data_dict['sr'], res_type='soxr_hq')
        sr_out = data_dict['sr']
    
    # Trim
    len_before_trim = y_out.shape[-1]
    if data_dict['trim'] == True:
        y_out, index_trimmed = librosa.effects.trim(
            y_out, 
            top_db=data_dict['top_db'], 
            frame_length=data_dict['frame_length'],
            hop_length=data_dict['hop_length'],
        )
    len_after_trim = y_out.shape[-1]


    # Make output directory
    os.makedirs(os.path.dirname(data_dict['output_path']), exist_ok=True)
    sf.write(
        data_dict['output_path'], 
        y_out, 
        sr_out, 
        subtype=data_dict['subtype'], 
        format=data_dict['format']
    )
    
    return len_before_trim, len_after_trim

def search_input_audio(input_dir, input_ext='.wav'):
    input_audio_list = []
    for (_path, _dir, files) in os.walk(input_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == input_ext:
                input_audio_list.append(os.path.join(_path, filename))
    return input_audio_list

def main(args):
    # Validate resampling and trimming
    if args.resample == True:
        assert args.sr != None, "Please give sampling rate(--sr SAMPLE_RATE) to resample."
    print("-"*20)
    print(f"Input: {args.input_ext}")
    print("-"*20)
    print(f"Resapmle: {args.resample}")
    print(f" > Resampling rate: {args.sr}")
    print("-"*20)
    print(f"Trim: {args.trim}")
    print(f" > Trim top_db: {args.top_db}")
    print(f" > Trim frame_length: {args.frame_length}")
    print(f" > Trim hop_length: {args.hop_length}")
    print("-"*20)
    print(f"Output: {args.ext}")
    print(f" > output format: {args.format}")
    print(f" > output subtype: {args.subtype}")
    print("-"*20)

    input_audio_list = search_input_audio(args.input_dir, input_ext=args.input_ext)
    print(f"Num of input audio: {len(input_audio_list)}")

    # Prepare data_dict for multi-processing
    data_dict_list = []
    for idx, input_audio in enumerate(input_audio_list):
        data_dict = deepcopy(vars(args))
        data_dict['input_path'] = input_audio
        data_dict['output_path'] = os.path.join(args.output_dir, 
            os.path.splitext(
                input_audio[len(args.input_dir):].strip(os.path.sep)
            )[0] + args.ext
        )
        data_dict_list.append(data_dict)
    
    # Do multi-processing
    with Pool(args.worker) as p:
        results = list(tqdm(
            p.imap(librosa_audio_dir, data_dict_list),
            total=len(data_dict_list)
        ))
    results_npy = np.asarray(results)
    results_sum = np.sum(results_npy, axis=0)
    if args.trim == True:
        print(f"Total trimmed audio: {(1 - results_sum[1]/results_sum[0])*100:.1f}%")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Resamplig and trim audio data directory using Librosa',
        epilog=f"Available formats = {str(list(sf.available_formats().keys()))}",
    )
    parser.add_argument('input_dir', type=str, 
                        help='Input audio data directory')
    parser.add_argument('output_dir', type=str, 
                        help='Output audio data directory')
    parser.add_argument('--input_ext', type=str, default=".wav", 
                        help='Input audio extension')
    parser.add_argument('--ext', type=str, default=".wav", 
                        help='Output audio extension')
    parser.add_argument('--mono', type=str2bool, default=True,
                        help='Load input audio to mono channel.')
    parser.add_argument('--format', type=str, default="WAV", 
                        help='Output audio format (soundfile)')
    parser.add_argument('--subtype', type=str, default=None, 
                        help='Output audio subtype (soundfile)')
    parser.add_argument('--resample', type=str2bool, default=False,
                        help='Resample audio using librosa.resample()')
    parser.add_argument('--sr', type=int, default=None,
                        help='Output audio sampling rate')
    parser.add_argument('--trim', type=str2bool, default=False,
                        help='Trim audio using librosa.effects.trim()')
    parser.add_argument('--top_db', type=float, default=60,
                        help='Top_db to trim audio')
    parser.add_argument('--frame_length', type=int, default=2048,
                        help='frame_length to trim audio (after resampling)')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='hop_length to trim audio (after resampling)')
    parser.add_argument('--worker', type=int, default=8,
                        help='Number of workers')
    args = parser.parse_args()

    main(args)
