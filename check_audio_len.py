import argparse
import os
import sys
from multiprocessing import Pool
from collections import defaultdict

import numpy as np
from pymediainfo import MediaInfo
from tqdm import tqdm

audio_ext_list = ['.aac', '.wav', '.flac', '.m4a', 'mp4', 'mp3', '.ogg']

def get_audio_list(input_dir):
    audio_list = []
    for (_path, _dir, files) in os.walk(input_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext.lower() in audio_ext_list:
                audioname = os.path.join(_path, filename)
                audio_list.append(audioname)
    return audio_list

def get_mediainfo_from_file(audiofile):
    media_info = MediaInfo.parse(audiofile)
    for track in media_info.tracks:
        if track.track_type == "Audio":
            dur = track.to_data()['duration']/1000
            sr = track.to_data()['sampling_rate']
            return dict({
                "dur": dur,
                "sr": sr
            })
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check length of audio files from input directory')
    parser.add_argument('input_dir', type=str, help='Input audio data directory')
    parser.add_argument('--worker', type=int, help='Number of workers', default=8)
    args = parser.parse_args()

    input_dir = args.input_dir

    audio_list = get_audio_list(input_dir)
    with Pool(args.worker) as p:
        dict_results = list(tqdm(p.imap(get_mediainfo_from_file, audio_list), total=len(audio_list)))
        
    none_list = [x for x in dict_results if x == None]
    print(f"Total input audiofiles: {len(audio_list)}")
    print(f"None results: {len(none_list)}")
        
    results_dur = [x['dur'] for x in dict_results if x != None]
    results_sr = [x['sr'] for x in dict_results if x != None]
    
    print(f"Total dur: {sum(results_dur)/60:.1f}min")
    print(f"Total dur: {sum(results_dur)/3600:.1f}hours")
    print(f"Max dur: {np.max(results_dur):.2f}sec")
    print(f"Min dur: {np.min(results_dur):.2f}sec")
    print(f"Mean dur: {np.mean(results_dur):.2f}sec")
    sr_dict = defaultdict(int)
    for sr in results_sr:
        sr_dict[sr] += 1
    print(f"Sampling rates: {dict(sr_dict)}")
