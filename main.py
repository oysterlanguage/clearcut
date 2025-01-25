import logging
import os
import torch
import onnxruntime as ort
from diskcache import Cache
import librosa
import time
from models import separate_fast
from models.respiro import BreathDetector
from models.uroman_align import align, quick_align, get_valleys, find_threshold_crossing
from intervaltree import IntervalTree
from textgrid import TextGrid, IntervalTier
import statistics
import numpy as np
import soundfile as sf
import yaml
import argparse
from dataclasses import replace
import random
import re
from difflib import SequenceMatcher
from number_normalizer import replace_numbers_in_text, extract_and_update_sequence
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

cache = Cache("cache")
# Create a custom logger
logger = logging.getLogger("custom_logger")
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("app.log")

# Set levels for handlers
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

# Create formatter and add to handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def time_logger(func):
    """
    Decorator to log the execution time of a function.

    Args:
        func (callable): The function whose execution time is to be logged.

    Returns:
        callable: The wrapper function that logs the execution time of the original function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(
            f"Function {func.__name__} took {end_time - start_time} seconds to execute"
        )
        return result

    return wrapper


@time_logger
def asr(audio):
    # resample to 16k
    if audio["sample_rate"] != 16000:
        temp_audio = librosa.resample(
            audio["waveform"], orig_sr=audio["sample_rate"], target_sr=16000
        )
    else:
        temp_audio = audio["waveform"]
    results = whisperx_transcribe(temp_audio)
    return results


def byob_transcribe(audio, transcripts, language="en"):
    segments = quick_align(audio, transcripts, language)
    # resample to 16k
    if audio["sample_rate"] != 16000:
        temp_audio = librosa.resample(
            audio["waveform"], orig_sr=audio["sample_rate"], target_sr=16000
        )
    else:
        temp_audio = audio["waveform"]
    whisperx_asr_model.load_asr_model()
    results = whisperx_asr_model.just_align(temp_audio, segments)
    whisperx_asr_model.unload_asr_model()
    return results


def whisperx_transcribe(audio):
    whisperx_asr_model.load_asr_model()
    results = whisperx_asr_model.transcribe_and_align(audio)
    whisperx_asr_model.unload_asr_model()
    return results


@time_logger
def get_breath_intervals(audio, valleys) -> IntervalTree:
    breath_model.load_model()
    # resample to 16k
    if audio["sample_rate"] != 16000:
        temp_audio = librosa.resample(
            audio["waveform"], orig_sr=audio["sample_rate"], target_sr=16000
        )
    else:
        temp_audio = audio["waveform"]
    results = breath_model.get_breath_timestamps(temp_audio, valleys)
    breath_model.unload_model()
    return results


@time_logger
def get_alignment(audio, segments, valleys, breaths=None):
    alignment, ranges = align(audio, segments, valleys=valleys, breaths=breaths)
    return list(zip(segments, alignment)), sorted(ranges)


def create_textgrid(dictionary, output_path):
    # Initialize a TextGrid object
    tg = TextGrid()

    for key, values in dictionary.items():
        if values:
            # Create an IntervalTier for words
            words_tier = IntervalTier(name=key)
            for entry in values:
                start = float(entry["start"])
                end = float(entry["end"])
                word = entry["word"]
                try:
                    words_tier.add(start, end, word)
                except Exception as e:
                    logger.error(f"Error adding entry to TextGrid tier: {entry}")
                    logger.error(e)
            tg.append(words_tier)

    # Write the TextGrid to a file
    tg.write(output_path)
    logger.info(f"TextGrid saved to {output_path}")


@time_logger
def source_separation(predictor, audio) -> (dict, dict):
    """
    Separate the audio into vocals and non-vocals using the given predictor.

    Args:
        predictor: The separation model predictor.
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.

    Returns:
        dict: A dictionary containing the separated vocals and updated audio waveform.
    """

    mix, rate = None, None

    if isinstance(audio, str):
        mix, rate = librosa.load(audio, mono=False, sr=44100)
    else:
        # resample to 44100
        rate = audio["sample_rate"]
        mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)

    vocals, no_vocals = predictor.predict(mix)

    # convert vocals back to previous sample rate
    logger.debug(f"vocals shape before resample: {vocals.shape}")
    vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    logger.debug(f"vocals shape after resample: {vocals.shape}")
    audio["waveform"] = vocals[:, 0]  # vocals is stereo, only use one channel

    logger.debug(f"no_vocals shape before resample: {no_vocals.shape}")
    no_vocals = librosa.resample(no_vocals.T, orig_sr=44100, target_sr=rate).T
    logger.debug(f"no_vocals shape after resample: {no_vocals.shape}")
    no_vocals_audio = {"waveform": no_vocals[:, 0], "sample_rate": rate}

    return audio, no_vocals_audio


def dbfs(waveform):
    """Compute the approximate dBFS of a waveform."""
    # RMS of the waveform
    rms = np.sqrt(np.mean(waveform**2)) if np.mean(waveform**2) > 0 else 1e-10
    return 20 * np.log10(rms + 1e-10)


@time_logger
def standardization(audio, sample_rate=None, start_s=0, duration_s=None):
    """
    Preprocess the audio file using librosa instead of pydub.
    Operations include setting sample rate, converting to mono, trimming,
    normalizing volume, and scaling amplitude.

    Args:
        audio (str or np.ndarray):
            - If str: Path to the audio file to load.
            - If np.ndarray: A raw waveform array.
        sample_rate (int): The target sample rate for the audio.
        start_s (float): The start time in seconds to trim the audio.
        duration_s (float): The duration in seconds to trim the audio.

    Returns:
        dict: A dictionary containing:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, shape (num_samples,)
                              dtype is np.float32
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor a NumPy ndarray.
    """
    global audio_count
    name = "audio"

    # Load audio
    # If audio is a file path
    if isinstance(audio, str):
        name = os.path.basename(audio)
        name = audio
        # librosa.load automatically converts to float32 and can handle trimming
        # offset=start_s sets the start position, duration=duration_s sets the length
        waveform, sr = librosa.load(
            audio, mono=True, offset=start_s, duration=duration_s, sr=sample_rate
        )
        sample_rate = sr
    # If audio is a NumPy array (raw waveform)
    elif isinstance(audio, np.ndarray):
        # Here we assume `audio` is a waveform at the correct rate, or we might need to resample if needed.
        # If you need to ensure a specific sample rate, you can resample here:
        # waveform = librosa.resample(audio, original_sr, sample_rate)
        # But since we don't have original_sr here, we assume it's correct.
        waveform = audio.astype(np.float32)
        sr = sample_rate
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError(
            "Invalid audio type. Must be either a file path (str) or a NumPy array."
        )

    logger.debug("Entering the preprocessing of audio (using librosa)")

    # waveform is now a float32 array in [-1.0, 1.0], single channel, at the specified sample_rate.

    # Now, we want to replicate the volume normalization logic:
    # Original code aimed for a target of -20 dBFS and then clipped the gain between -3 dB and +3 dB.

    # Compute current dBFS
    current_dBFS = dbfs(waveform)
    target_dBFS = -20.0
    gain = target_dBFS - current_dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Clamp the gain between -3 and 3 dB
    gain_clamped = np.clip(gain, -3, 3)

    # Apply gain (in linear scale)
    linear_gain = 10.0 ** (gain_clamped / 20.0)
    waveform = waveform * linear_gain

    # Final normalization (to ensure the max amplitude is 1.0)
    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
        waveform /= max_amplitude

    # Ensure float32 type
    waveform = waveform.astype(np.float32)

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "path": name,
        "sample_rate": sample_rate,
        "audio_length": len(waveform) / sample_rate,
    }

def merge_texts_preserving_newlines(text1, text2):
    tokens1 = re.findall(r'\n|\S+', text1)
    tokens2 = re.findall(r'\n|\S+', text2)

    sm = SequenceMatcher(None, tokens1, tokens2)
    output_tokens = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ['equal', 'delete']:
            k = j1
            for pos in range(i1, i2):
                if tokens1[pos] == '\n':
                    output_tokens.append('\n')
                else:
                    if k < j2:
                        output_tokens.append(tokens2[k])
                        k += 1
                    else:
                        output_tokens.append(tokens1[pos])

        elif tag == 'replace':
            k = j1
            for pos in range(i1, i2):
                if tokens1[pos] != '\n':
                    if k < j2:
                        output_tokens.append(tokens2[k])
                        k += 1
                    else:
                        output_tokens.append(tokens1[pos])
            while k < j2:
                if tokens2[k] != '\n':
                    output_tokens.append(tokens2[k])
                    k += 1
                else:
                    break
            index = 1
            while tokens1[i2-index] == '\n':
                output_tokens.append('\n')
                index += 1

    result = []
    for token in output_tokens:
        if token == '\n':
            result.append(token)
        else:
            if result and not result[-1].endswith('\n'):
                result.append(' ' + token)
            else:
                result.append(token)

    return ''.join(result)

@time_logger
def process(config, audio_path):
    debug = config.get("general",{}).get("debug", False)

    logger.info("Step 0: Load audio")
    # y, sr = librosa.load(audio_path, sr=None)
    # audio = {"waveform": y, "sample_rate": sr, "path": audio_path}

    logger.info("Step 0: Standardize audio")
    sample_rate = config.get("general", {}).get("sample_rate", 22000)
    audio = standardization(audio_path,sample_rate=sample_rate)

    no_vocal = None
    if config.get("separate_fast", {}).get("enabled", False):

        separate_predictor1.load_model()
        logger.info("Step 0: Separate audio using separate_fast")
        audio, no_vocal = source_separation(separate_predictor1, audio)

        # write no_vocal to file
        # sf.write(audio_path.replace("audio", "no_vocal"), no_vocal["waveform"], no_vocal["sample_rate"])
        separate_predictor1.unload_model()
    original_text = ""
    asrx_result = None
    asr_files = [f"{audio_path}.fixed.txt", f"{".".join(audio_path.split(".")[:-1])}.fixed.txt", f"{audio_path}.txt", f"{".".join(audio_path.split(".")[:-1])}.txt"]
    for asr_file in asr_files:
        if os.path.exists(asr_file):
            logger.info(f"Step 1: Using provided text instead of ASR {asr_file}")
            with open(asr_file, "r", encoding="utf-8") as f:
                segments = f.readlines()
            original_text = "".join(segments)
            if config.get("normalize_byo_transcript",{}).get("aggressive_segment", False):
                original_text = normalize_text(original_text)
                original_text = aggressive_split(original_text)
                segments = original_text.split("\n")
            if "fixed" in asr_file:
                original_text = ""
            if debug:
                asrx_result = cache.get(f"byob_transcribe:{asr_file}", None)
            if asrx_result is None:
                asrx_result = byob_transcribe(audio, segments)
            if debug:
                cache.set(f"byob_transcribe:{audio_path}", asrx_result)
            break
    if asrx_result is None:
        logger.info("Step 1: whisperx ASR")
        if debug:
            asrx_result = cache.get(f"asrx_result:{audio_path}", None)
            if asrx_result is None:
                asrx_result = asr(audio)
                cache.set(f"asrx_result:{audio_path}", asrx_result)
        else:
            asrx_result = asr(audio)
        # logger.info(f"ASR Result: {asrx_result}")

    if config.get("normalize_byo_transcript", {}).get("enabled", False) and original_text != "":
        logger.info("step 5: Fixing missing word start and end times")
        fix_missing_word_times(asrx_result, audio)

        text = " ".join([word.get("word") for word in asrx_result.get("word_segments")])
        # text = normalize_text(text)

        fixed_text = merge_texts_preserving_newlines(original_text, text)

        with open(f"{audio_path}.fixed.txt", "w", encoding="utf-8") as f:
            logger.debug(f"Found differences between original and fixed text saved to {audio_path}.fixed.txt")
            f.write(fixed_text)
        return False

    segments = asrx_result["segments"]
    # joined_segments = []
    # for i, segment in enumerate(segments):
    #     if i == 0:
    #         segment["text"] = segment["text"].strip()
    #         joined_segments.append(segment)
    #     else:
    #         if segments[i - 1]["text"][-1] not in ["?", ".", "!"]:
    #             joined_segments[-1]["text"] += " " + segment["text"].strip()
    #             joined_segments[-1]["end"] = segment["end"]
    #             joined_segments[-1]["words"].append(segment["words"])
    #         else:
    #             segment["text"] = segment["text"].strip()
    #             joined_segments.append(segment)
    logger.info("Step 2: Calculate db levels")
    raw_valleys, valleys = get_valleys(audio)

    logger.info("Step 3: Get breath intervals")
    breaths = get_breath_intervals(audio, valleys)
    # logger.info(breaths)

    logger.info("Step 4: Finegrained alignment")
    alignment, ranges = get_alignment(audio, segments, valleys, breaths)


    threshold = config.get("find_threshold_crossing", {}).get("threshold", -60)
    for i in range(len(alignment)):
        start = alignment[i][1].get("start")
        end = alignment[i][1].get("end")
        start_crossing = find_threshold_crossing(
            valleys=raw_valleys,
            start=start,
            end=end,
            threshold=threshold,
            direction="right",
        )
        end_crossing = find_threshold_crossing(
            valleys=raw_valleys,
            start=start,
            end=end,
            threshold=threshold,
            direction="left",
        )
        if start_crossing is None:
            alignment[i][1]["padding_start"] = start
        else:
            alignment[i][1]["padding_start"] = start_crossing[0]
        if end_crossing is None:
            alignment[i][1]["padding_end"] = end
        else:
            alignment[i][1]["padding_end"] = end_crossing[0]


    # padding = .15
    # output_path = audio.get("path")
    # output_path = output_path.replace("data/audio", "data/processed")
    # output_path += "_segments"
    # segment_with_minima(audio, alignment, output_path, padding)

    start_padding = config.get("segment_with_threshold", {}).get("padding", 0.15)
    symmetrical = config.get("segment_with_threshold", {}).get("symmetrical", False)
    segments_count = len(alignment)
    for i in range(len(alignment)):
        try:
            start = alignment[i][1].get("padding_start")
            end = alignment[i][1].get("padding_end")

            left_padding = start_padding
            if symmetrical == True or i == segments_count - 1:
                right_padding = start_padding
            else:
                right_padding = max(start_padding, (alignment[i + 1][1].get("padding_start") - end - start_padding))
            alignment[i][1]["left_padding"] = left_padding
            alignment[i][1]["right_padding"] = right_padding
        except Exception as e:
            logger.error(f"Error processing segment {alignment[i]}")
            logger.error(e)
            alignment[i][1]["left_padding"] = start_padding
            alignment[i][1]["right_padding"] = start_padding

    output_path = config.get("general", {}).get("output_path", "data/segments")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # build interval tree from words
    logger.info("Step 6: Create word intervals")
    word_intervals = IntervalTree()
    for index in range(len(asrx_result.get("word_segments"))):
        if "start" not in asrx_result.get("word_segments")[index]:
            if index == 0:
                asrx_result.get("word_segments")[index]["start"] = 0
            else:
                asrx_result.get("word_segments")[index]["start"] = asrx_result.get(
                    "word_segments"
                )[index - 1]["end"]
        if "end" not in asrx_result.get("word_segments")[index]:
            if index == len(asrx_result.get("word_segments")) - 1:
                asrx_result.get("word_segments")[index]["end"] = audio.get("audio_length")
            else:
                asrx_result.get("word_segments")[index]["end"] = asrx_result.get(
                    "word_segments"
                )[index + 1]["start"]
        if (
            asrx_result.get("word_segments")[index]["start"]
            >= asrx_result.get("word_segments")[index]["end"]
        ):
            logger.error(
                f"Word {asrx_result.get('word_segments')[index]} has start time after end time"
            )
            continue
        word_intervals.addi(
            asrx_result.get("word_segments")[index]["start"],
            asrx_result.get("word_segments")[index]["end"],
            data=asrx_result.get("word_segments")[index].get("word"),
        )
    mixed = breaths | word_intervals

    train_path = config.get("general", {}).get("train_path", "data/train")
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    audio_file_name = os.path.basename(audio_path)
    train_file = os.path.join(train_path, f"{audio_file_name}.train.txt")

    if config.get("combine_segments", {}).get("enabled", False):
        max_len = config.get("combine_segments", {}).get("max_len", 30)
        max_segments = config.get("combine_segments", {}).get("max_segments", 2)
        segment_with_threshold_expand(audio, alignment, output_path, mixed, train_file, max_len, max_segments)
    else:
        segment_with_threshold(
            audio, alignment, output_path, mixed
        )
        with open(train_file, "w", encoding="utf-8") as f:
            for segment in alignment:
                line = segment[1].get("line")
                f.write(line)

    if config.get("separate_fast", {}).get("enabled", False):
        for segment in alignment:
            # calculate peak and average db level of same time in the no_vocal audio
            start = segment[1].get("start")
            end = segment[1].get("end")
            start_sample = int(start * no_vocal["sample_rate"])
            end_sample = int(end * no_vocal["sample_rate"])
            clip = no_vocal["waveform"][start_sample:end_sample]
            db = dbfs(clip)
            average_db = np.mean(clip)
            max_db = np.max(clip)
            print(
                f"Segment {segment[0]['text']} has average db level of {average_db} and max db level of {max_db}, and dbfs of {db}"
            )
            pass


    breath_peaks = [breath.data[1] for breath in breaths if breath.data]
    if len(breath_peaks) > 1:
        mean = statistics.mean(breath_peaks)
        std_dev = statistics.stdev(breath_peaks)
    else:
        mean = 0
        std_dev = 1

    possible_breath_misses = [
        breath.data
        for breath in sorted(breaths)
        if breath.data and breath.data[1] < (mean - (std_dev * 1.5))
    ]
    print(
        f"These breaths are unusually quiet and could affect alignment: {possible_breath_misses}"
    )

    edge_db = [segment[1]["lowest"]["start"]["db"] for segment in alignment] + [
        segment[1]["lowest"]["end"]["db"] for segment in alignment
    ]
    if len(edge_db) > 1:
        mean = statistics.mean(edge_db)
        std_dev = statistics.stdev(edge_db)
    else:
        mean = 0
        std_dev = 1

    # we may want to join on loud splits
    possible_edge_misses = [
        segment
        for segment in alignment
        if segment[1]["lowest"]["start"]["db"] > (mean + (std_dev * 3))
        or segment[1]["lowest"]["end"]["db"] > (mean + (std_dev * 3))
    ]
    print(
        f"These segments end unusually loud and could cause clipping: {possible_edge_misses}"
    )


    if config.get("create_textgrid", {}).get("enabled", False):
        logger.info("Step 5: Create TextGrid")
        create_textgrid(
            {
                "words": asrx_result.get("word_segments"),
                "breaths": [
                    {"start": breath.begin, "end": breath.end, "word": "breath"}
                    for breath in breaths
                ],
                "whisperx": [
                    {
                        "start": segment[0]["start"],
                        "end": segment[0]["end"],
                        "word": segment[0]["text"],
                    }
                    for segment in alignment
                ],
                "uroman": [
                    {
                        "start": segment[1]["start"],
                        "end": segment[1]["end"],
                        "word": segment[1]["text"],
                    }
                    for segment in alignment
                ],
                "split_ranges": [
                    {"start": x.begin, "end": x.end, "word": "range"} for x in ranges
                ],
            },
            f"{audio_path}.TextGrid",
        )
    return True


def get_text_with_breath(mixed, start, end):
    words = sorted(mixed[start:end])
    text = []
    for word in words:
        if type(word.data) == str:
            text.append(word.data)
        else:
            text.append("<breath>")
    # remove breath tag if its the first or last word in text
    if len(text) and text[0] == "<breath>":
        text.pop(0)
    if len(text) and text[-1] == "<breath>":
        text.pop()
    return text


def normalize_text(text):
    ABBREVIATIONS = {
        "Mr. ": "Mister ",
        "Mrs. ": "Misses ",
        "No. ": "Number ",
        "Dr. ": "Doctor ",
        "Ms. ": "Miss ",
        "Ave. ": "Avenue ",
        "Blvd. ": "Boulevard ",
        "Ln. ": "Lane ",
        "Rd. ": "Road ",
        "ft. ": "feet ",
        "hr. ": "hour ",
        "min. ": "minute ",
        "sq. ": "square ",
        "St. ": "street ",
        "Asst. ": "assistant ",
        "Corp. ": "corporation "
    }
    for key, value in ABBREVIATIONS.items():
        pattern = fr'(?:^|\b){re.escape(str(key))}(?:\b|$)'
        text = re.sub(pattern, value, text)
    return text

def aggressive_split(text):
    text = text.replace(". ", ".\n")
    text = text.replace("? ", "?\n")
    text = text.replace("! ", "!\n")
    text = text.replace(", ", ",\n")
    text = text.replace(": ", ":\n")
    text = text.replace(' – ', " - ")
    text = text.replace(' - ', " -\n")
    return text


def fix_missing_word_times(asrx_result, audio):
    """
    Find consecutive words that lack start/end times, then
    evenly distribute times between the nearest known timestamps.

    Args:
        asrx_result (dict): The ASR result dict with a 'word_segments' list.
        total_audio_duration (float): The total duration of the audio in seconds.
    """
    total_audio_duration = audio.get("audio_length")
    word_segments = asrx_result.get("word_segments", [])
    n_words = len(word_segments)
    if n_words == 0:
        return  # nothing to fix

    # 1. Collect consecutive runs of missing times.
    #    A "run" is defined as consecutive words that are missing either "start" or "end".
    runs = []
    i = 0
    while i < n_words:
        # Check if current word is missing times
        current_missing = ("start" not in word_segments[i]) or ("end" not in word_segments[i])
        if current_missing:
            start_run = i
            # move i until we find a word that has both "start" and "end"
            while i < n_words and (
                ("start" not in word_segments[i]) or ("end" not in word_segments[i])
            ):
                i += 1
            end_run = i - 1
            runs.append((start_run, end_run))
        else:
            i += 1
    replacements = []
    whisperx_asr_model.load_asr_model()
    # 2. For each run, find the boundary times and distribute
    for (start_run, end_run) in runs:
        replacement = {"indices":list(range(start_run,end_run+1)), "replacements":[]}
        new_word_segments = []

        # The run might be only 1 word or multiple consecutive words.
        run_length = end_run - start_run + 1

        # 2a. Find the boundary on the left
        #     If there's a previous word that has a valid 'end', use that.
        #     Otherwise, fallback to 0.0
        if start_run > 0 and "end" in word_segments[start_run - 1]:
            left_time = word_segments[start_run - 1]["end"]
        else:
            left_time = 0.0

        # 2b. Find the boundary on the right
        #     If there's a next word that has a valid 'start', use that.
        #     Otherwise, fallback to total_audio_duration
        if end_run < n_words - 1 and "start" in word_segments[end_run + 1]:
            right_time = word_segments[end_run + 1]["start"]
        else:
            right_time = total_audio_duration

        # 2c. Edge case: If left_time is already >= right_time, we can't do a normal distribution
        if left_time >= right_time:
            # fallback: just set them all to left_time or do some minimal logic
            increment = 0.1  # or however you want to handle this pathological case
            for k in range(run_length):
                idx = start_run + k
                # at least assign something so they don't remain unset
                start_t = left_time + (k * increment)
                word_segments[idx]["start"] = start_t
                word_segments[idx]["end"] = start_t + increment
            continue

        prompt = " ".join([word.get("word") for word in word_segments[start_run:end_run + 1]])
        if re.findall(r"\d+", prompt):
            for attempt in range(5):
                # look through each index less than start_run from greatest to least for the first index that ends in a punctuation
                left_index = start_run
                buffer = 0
                for i in reversed(range(0, start_run-1)):
                    left_index = i+1
                    if word_segments[i].get("word")[-1] in [".", "?", "!"] and word_segments[i + 1].get("start", None):
                        buffer += 1
                        if buffer > attempt:
                            break
                right_index = end_run
                buffer = 0
                for i in range(end_run+1, n_words):
                    right_index = i
                    if word_segments[i].get("word")[-1] in [".", "?", "!"] and word_segments[i].get("end", None):
                        buffer += 1
                        if buffer > attempt:
                            break


                extra = 0.1
                clip_left_time = max(0,(word_segments[left_index].get("start", 0) - extra))
                clip_right_time = min(audio.get("audio_length"),(word_segments[right_index].get("end", audio.get("audio_length")) + extra))
                # create a clip from the start_time and end_time and pass it to asr
                clip = audio.get("waveform")[int(clip_left_time * audio.get("sample_rate")):int(clip_right_time * audio.get("sample_rate"))]
                # resample to 16k
                if audio["sample_rate"] != 16000:
                    clip = librosa.resample(
                        clip, orig_sr=audio["sample_rate"], target_sr=16000
                    )


                #save clip to file name it based on the index
                sf.write(f"clip_{start_run}.wav", clip, 16000)
                # def shuffle_digits_in_string(input_string):
                #     # Function to shuffle digits in a number
                #     def shuffle_digits(match):
                #         digits = list(match.group())
                #         random.shuffle(digits)
                #         return ''.join(digits)
                #
                #     # Find all numbers in the string and shuffle their digits
                #     result = re.sub(r'\d+', shuffle_digits, input_string)
                #     return result

                # if attempt == 0:
                #     whisperx_asr_model.model.options = replace(whisperx_asr_model.model.options, initial_prompt=f'{context}')
                # # whisperx_asr_model.model.options = replace(whisperx_asr_model.model.options, initial_prompt=f'{shuffle_digits_in_string(prompt)}')
                # elif end_run - start_run > 0:
                #     whisperx_asr_model.model.options = replace(whisperx_asr_model.model.options, initial_prompt=f'{shuffle_digits_in_string(prompt)}')
                # else:
                #     whisperx_asr_model.model.options = replace(whisperx_asr_model.model.options, initial_prompt=f'{whisperx_asr_model.model.options.initial_prompt} {" ".join(random.sample(["digit", "number", "year", "fraction"], 4))}')
                numbers = whisperx_asr_model.transcribe_and_align(clip)
                # whisperx_asr_model.model.options = whisperx_asr_model.options
                normalized, conversion = replace_numbers_in_text(" ".join([part.get("word") for part in numbers.get("word_segments", [])]))
                if normalized != prompt:
                    test_prompt = prompt
                    # use regex to extract all numbers from the prompt and replace them with the string from the conversion dictionary
                    for key, value in conversion.items():
                        # only replace if the value matches the entire number in the prompt
                        # test_prompt = re.sub(f'\\b{str(value)}\\b', key, test_prompt)
                        pattern = fr'(?:^|\b){re.escape(str(value))}(?:\b|$)'
                        test_prompt = re.sub(pattern, key, test_prompt)
                    if re.findall(r"\d+", test_prompt):
                        continue
                    # try to align the normalized text to the original text
                    for word in numbers.get("word_segments"):
                        if word.get("start", None) and word.get("end", None):
                            word["word"] = word["word"].strip(".,!?:'\"")
                    fixed_words = extract_and_update_sequence(numbers.get("word_segments"),test_prompt)
                    if " ".join([word.get("word") for word in fixed_words]) != test_prompt:
                        continue
                    else:
                        numbers["word_segments"] = fixed_words

                for word in numbers.get("word_segments"):
                    if word.get("start", None) and word.get("end", None):
                        word["start"] = max(0,word["start"] + left_time - extra)
                        word["end"] = min(audio.get('audio_length'), word["end"] + left_time - extra)
                        new_word_segments.append(word)
                replacement["replacements"] = new_word_segments
                replacements.append(replacement)
                break
        # 3. backup option: evenly distribute the times between the boundaries for non-matcher
        interval = right_time - left_time
        segment_length = interval / run_length
        for k in range(run_length):
            idx = start_run + k
            # The start = left boundary + k*segment_length
            # The end = left boundary + (k+1)*segment_length
            word_segments[idx]["start"] = left_time + k * segment_length
            word_segments[idx]["end"] = left_time + (k + 1) * segment_length
    whisperx_asr_model.unload_asr_model()
    for replacement in reversed(replacements):
        asrx_result["word_segments"][replacement["indices"][0]:replacement["indices"][-1]+1] = replacement["replacements"]
    pass

def segment_with_threshold(audio, alignment, output_path, breath_transcript_tree):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    filename = os.path.basename(audio.get("path"))
    for i in range(len(alignment)):
        try:
            start = alignment[i][1].get("padding_start")
            end = alignment[i][1].get("padding_end")

            left_padding = np.zeros(int(alignment[i][1].get("left_padding") * audio.get("sample_rate")))
            right_padding = np.zeros(int(alignment[i][1].get("right_padding") * audio.get("sample_rate")))

            # Calculate sample indices
            start_sample = int(start * audio.get("sample_rate"))
            end_sample = int(end * audio.get("sample_rate"))
            # Extract the audio segment
            y_segment = audio.get("waveform")[start_sample:end_sample]

            # Add padding to the segment
            y_padded = np.concatenate((left_padding, y_segment, right_padding))
            alignment[i][1]["filename"] = os.path.join(
                output_path, f"{filename}.segment.{i}.wav"
            )

            text = get_text_with_breath(breath_transcript_tree, start, end)
            line = f"{alignment[i][1].get('filename')}|{' '.join(text)}\n"
            alignment[i][1]["line"] = line

            # Save the padded audio segment as a WAV file
            sf.write(
                os.path.join(output_path, f"{filename}.segment.{i}.wav"),
                y_padded,
                audio.get("sample_rate"),
            )
        except Exception as e:
            logger.error(f"Error processing segment {alignment[i]}")
            logger.error(e)

def segment_with_threshold_expand(audio, alignment, output_path, breath_transcript_tree, train_file, max_len, max_segments):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    filename = os.path.basename(audio.get("path"))
    segment_index = 0
    with open(train_file, "w", encoding="utf-8") as f:
        for i in range(len(alignment)):
            group = 0
            try:
                start = 0
                end = .1
                while end - start <= max_len and i+group < len(alignment) and group < max_segments:
                    start = alignment[i][1].get("padding_start")
                    end = alignment[i+group][1].get("padding_end")

                    left_padding_duration = alignment[i][1].get("left_padding")
                    right_padding_duration = alignment[i+group][1].get("right_padding")

                    if (end - start) + left_padding_duration + right_padding_duration > max_len:
                        break

                    left_padding = np.zeros(int(left_padding_duration * audio.get("sample_rate")))
                    right_padding = np.zeros(int(right_padding_duration * audio.get("sample_rate")))

                    # Calculate sample indices
                    start_sample = int(start * audio.get("sample_rate"))
                    end_sample = int(end * audio.get("sample_rate"))
                    # Extract the audio segment
                    y_segment = audio.get("waveform")[start_sample:end_sample]

                    # Add padding to the segment
                    y_padded = np.concatenate((left_padding, y_segment, right_padding))
                    file_name = f"{filename}.segment.{segment_index}.wav"

                    text = get_text_with_breath(breath_transcript_tree, start, end)

                    # Save the padded audio segment as a WAV file
                    sf.write(
                        os.path.join(output_path, file_name),
                        y_padded,
                        audio.get("sample_rate"),
                    )
                    f.write(f"{file_name}|{' '.join(text)}\n")

                    group += 1
                    segment_index += 1
            except Exception as e:
                logger.error(f"Error processing segment {alignment[i]}")
                logger.error(e)


def segment_with_minima(audio, alignment, output_path, padding):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(len(alignment)):
        start = alignment[i][1].get("start")
        end = alignment[i][1].get("end")
        padding_start = alignment[i][1].get("padding_start")
        padding_end = alignment[i][1].get("padding_end")

        left_padding_duration = max(start - (padding_start - padding), 0)
        right_padding_duration = max((padding_end + padding) - end, 0)
        left_padding = np.zeros(int(left_padding_duration * audio.get("sample_rate")))
        right_padding = np.zeros(int(right_padding_duration * audio.get("sample_rate")))

        # Calculate sample indices
        start_sample = int(start * audio.get("sample_rate"))
        end_sample = int(end * audio.get("sample_rate"))

        # Extract the audio segment
        y_segment = audio.get("waveform")[start_sample:end_sample]

        # Add padding to the segment
        y_padded = np.concatenate((left_padding, y_segment, right_padding))

        # Save the padded audio segment as a WAV file
        sf.write(
            os.path.join(output_path, f"segment{i}.wav"),
            y_padded,
            audio.get("sample_rate"),
        )


def detect_gpu():
    """Detect if GPU is available and print related information."""

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        logger.info("ENV: CUDA_VISIBLE_DEVICES not set, use default setting")
    else:
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info(f"ENV: CUDA_VISIBLE_DEVICES = {gpu_id}")

    if not torch.cuda.is_available():
        logger.error("Torch CUDA: No GPU detected. torch.cuda.is_available() = False.")
        return False

    num_gpus = torch.cuda.device_count()
    logger.debug(f"Torch CUDA: Detected {num_gpus} GPUs.")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        logger.debug(f" * GPU {i}: {gpu_name}")

    logger.debug("Torch: CUDNN version = " + str(torch.backends.cudnn.version()))
    if not torch.backends.cudnn.is_available():
        logger.error("Torch: CUDNN is not available.")
        return False
    logger.debug("Torch: CUDNN is available.")

    ort_providers = ort.get_available_providers()
    logger.debug(f"ORT: Available providers: {ort_providers}")
    if "CUDAExecutionProvider" not in ort_providers:
        logger.warning(
            "ORT: CUDAExecutionProvider is not available. "
            "Please install a compatible version of ONNX Runtime. "
            "See https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html"
        )

    return True


def get_audio_files(folder_path):
    """Get all audio files in a folder."""
    audio_files = []
    for root, _, files in os.walk(folder_path):
        if "_processed" in root:
            continue
        for file in files:
            if ".temp" in file:
                continue
            if file.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
                audio_files.append(os.path.join(root, file))
    return audio_files

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # Optionally: parse CLI just for the config path (and maybe for overrides)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg:
        logger.error("Config file not found.")
        exit(1)


    # Load models
    logger.debug(" * Setting device")
    if cfg.get("general", {}).get("device") == "auto":
        if detect_gpu():
            logger.info("Using GPU")
            device_name = "cuda"
            device = torch.device(device_name)
        else:
            logger.info("Using CPU")
            device_name = "cpu"
            device = torch.device(device_name)
    else:
        device_name = cfg.get("general", {}).get("device")
        device = torch.device(device_name)


    logger.debug(" * Loading ASRX Model")
    from models import whisperx_asr

    whisperx_asr_params = cfg.get("whisperx_asr", {})
    asr_options = {
        "hotwords": None,
        "multilingual": True,
        "suppress_numerals": True,
        "initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."
    }
    # merge in whisperx_asr_params.get("asr_options",{}), into asr_options overwriting over the same keys
    if whisperx_asr_params.get("asr_options", {}):
        asr_options.update(whisperx_asr_params.get("asr_options", {}))
    whisperx_asr_model = whisperx_asr.WhisperXModel(
        model=whisperx_asr_params.get("model", "large-v3"),
        language=whisperx_asr_params.get("language", "en"),
        device=device_name,
        compute_type=whisperx_asr_params.get("compute_type", "float16"),
        suppress_numerals=whisperx_asr_params.get("suppress_numerals", True),
        threads=1,
        asr_options=asr_options
    )
    logger.debug("ASRX Model loaded")

    logger.debug(" * Loading Separate Model")
    separate_fast_params = cfg.get("separate_fast", {})
    if not separate_fast_params:
        logger.error("Separate model parameters not found in config.")
        exit(1)
    separate_predictor1 = separate_fast.Predictor(
        args=separate_fast_params,
        device=device_name,
    )
    logger.debug(" * Separate Model loaded")

    logger.debug(" * Loading Respiro Model")
    # breath detection

    respiro_params = cfg.get("respiro", {})
    breath_model = BreathDetector(
        model_path=respiro_params.get("model_path", "models/respiro-en.pt"),
        threshold=respiro_params.get("threshold", 0.064),
        min_length=respiro_params.get("min_length", 10),
    )

    logger.debug("Respiro Model loaded")

    input_folder_path = cfg.get("general", {}).get("input_folder", "data/audio")
    print(f"looking in {input_folder_path} for files")
    audio_paths = get_audio_files(input_folder_path)  # Get all audio files
    print(audio_paths)
    for path in audio_paths:
        completed = process(cfg, path)
        if not completed:
            process(cfg, path)
