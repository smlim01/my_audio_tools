import os
import sys
from multiprocessing import Pool
from collections import defaultdict

import numpy as np
from pymediainfo import MediaInfo
from tqdm import tqdm


def get_audio_list(input_dir):
    audio_list = []
    for (_path, _dir, files) in os.walk(input_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.wav' or ext == '.flac':
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
    input_dir = sys.argv[1]
    process_num = int(sys.argv[2])

    audio_list = get_audio_list(input_dir)
    with Pool(process_num) as p:
        dict_results = list(tqdm(p.imap(get_mediainfo_from_file, audio_list), total=len(audio_list)))
    results = [x['dur'] for x in dict_results]
    sr_results = [x['sr'] for x in dict_results]
    print(f"Total audiofiles: {len(audio_list)}")
    none_list = [x for x in results if x == None]
    dur_list = [x for x in results if x != None]
    print(f"None: {len(none_list)}")
    print(f"Not None: {len(dur_list)}")
    print(f"Total dur: {sum(dur_list)/60:.1f}min")
    print(f"Total dur: {sum(dur_list)/3600:.1f}hours")
    print(f"Max dur: {np.max(dur_list):.2f}sec")
    print(f"Min dur: {np.min(dur_list):.2f}sec")
    print(f"Mean dur: {np.mean(dur_list):.2f}sec")
    sr_dict = defaultdict(int)
    for sr in sr_results:
        sr_dict[sr] += 1
    print(f"Sampling rates: {dict(sr_dict)}")