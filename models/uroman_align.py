import intervaltree
import torchaudio.transforms as T
import torchaudio.functional as F
import unicodedata
from collections import defaultdict

import re
import os
import math
from dataclasses import dataclass
from torchaudio.models import wav2vec2_model
import uroman as ur

import torch
import numpy as np
from bisect import bisect_left, bisect_right
from functools import lru_cache
from intervaltree import IntervalTree
import logging

logger = logging.getLogger(__name__)


def find_minimum_in_range(valleys: dict, start_time, end_time):
    start_sec = int(start_time)
    end_sec = int(end_time)

    # Collect all valleys from relevant seconds
    relevant_valleys = []
    for sec in range(start_sec, end_sec + 1):
        if sec in valleys:
            relevant_valleys.extend(valleys[sec])
    # Filter valleys that fall within the exact start_time and end_time
    relevant_valleys = [v for v in relevant_valleys if start_time <= v[0] <= end_time]
    if not relevant_valleys:
        return None
    # Find the minimum value in the selected range
    # min_valley = min(valleys[start_idx:end_idx], key=lambda x: x[2])  # Min by dB value
    sorted_valleys = sorted(relevant_valleys, key=lambda x: x[1])[:3]  # Sort by time
    min_valley = min(sorted_valleys, key=lambda x: x[2])

    return min_valley


def find_maxima_in_range(valleys: dict, start_time, end_time):
    start_sec = int(start_time)
    end_sec = int(end_time)

    # Collect all valleys from relevant seconds
    relevant_valleys = []
    for sec in range(start_sec, end_sec + 1):
        if sec in valleys:
            relevant_valleys.extend(valleys[sec])
    # Filter valleys that fall within the exact start_time and end_time
    relevant_valleys = [v for v in relevant_valleys if start_time <= v[0] <= end_time]
    if not relevant_valleys:
        return None
    # Find the minimum value in the selected range
    max_valley = max(relevant_valleys, key=lambda x: x[1])

    return max_valley


def find_threshold_crossing(valleys: dict, start, end, threshold, direction="right"):
    start_sec = int(start)
    end_sec = int(end)

    # Collect all valleys from relevant seconds
    if direction == "right":
        for sec in range(start_sec, end_sec + 1):
            if sec in valleys:
                for valley in valleys[sec]:
                    if start <= valley[0] <= end:
                        if valley[2] >= threshold:
                            return valley
    elif direction == "left":
        for sec in reversed(range(start_sec, end_sec + 1)):
            if sec in valleys:
                for valley in reversed(valleys[sec]):
                    if start <= valley[0] <= end:
                        if valley[3] >= threshold:
                            return valley
    return None


@lru_cache
def find_minimum_in_range2(valleys, start_time, end_time):
    # Extract the times from the valleys
    times = [time for _, time, _ in valleys]

    # Use binary search to find the start index
    start_idx = bisect_left(times, start_time)

    # Perform a binary search for the end index starting from start_idx
    end_idx = bisect_right(times, end_time, lo=start_idx)

    # If no valleys fall in the range, return None
    if start_idx == end_idx:
        return None

    # Find the minimum value in the selected range
    # min_valley = min(valleys[start_idx:end_idx], key=lambda x: x[2])  # Min by dB value
    sorted_valleys = sorted(valleys[start_idx:end_idx], key=lambda x: x[2])[
        :3
    ]  # Sort by time

    min_db_ratio = float("inf")
    min_valley = None
    for x in sorted_valleys:
        if x[0] == 0:
            before = valleys[x[0]]
        else:
            before = valleys[x[0] - 1]
        current = valleys[x[0]]
        if x[0] + 1 == len(valleys):
            after = valleys[x[0]]
        else:
            after = valleys[x[0] + 1]
        db = (before[2] + current[2] + current[2] + after[2]) / 4
        if db < min_db_ratio:
            min_db_ratio = db
            min_valley = x

    return min_valley


def get_valley_tree(y, sr, offset=0):
    # Convert the waveform to a tensor properly
    waveform = (
        y.clone().detach()
        if torch.is_tensor(y)
        else torch.tensor(y, dtype=torch.float32)
    )

    # Set the frame length and hop length
    frame_length = int(sr * 0.001)  # 6 ms window
    hop_length = frame_length // 2  # 50% overlap between frames

    # Compute the short-time Fourier transform (STFT)
    stft = torch.stft(
        waveform,
        n_fft=frame_length,
        hop_length=hop_length,
        win_length=frame_length,
        window=torch.hann_window(frame_length).to(waveform.device),
        return_complex=True,
    )

    # Compute the magnitude spectrogram
    magnitude = torch.abs(stft)

    # Compute the RMS envelope from the magnitude spectrogram
    rms_envelope = torch.sqrt(torch.mean(magnitude**2, dim=1)).cpu().numpy()

    # Avoid division by zero by adding a small epsilon
    max_rms = np.max(rms_envelope)
    if max_rms == 0:
        return []  # No valleys can exist if the signal is completely silent
    log_envelope = 20 * np.log10(rms_envelope / max_rms + 1e-10)

    # Ensure log_envelope is 1D
    log_envelope = log_envelope.flatten()

    # Find local minima (valleys) on the log envelope
    valleys = np.where(
        np.r_[True, log_envelope[1:] < log_envelope[:-1]]
        & np.r_[log_envelope[:-1] < log_envelope[1:], True]
    )[0]

    # Convert valley indices to time
    valley_times = valleys * hop_length / sr + offset

    return tuple(zip(range(len(valley_times)), valley_times, log_envelope[valleys]))


def get_valley_dict(data, num_items=10):
    time_dict = defaultdict(list)

    # Helper function to calculate average for the next num_items
    def get_next_average(index, data, num_items):
        end = min(len(data), index + num_items + 1)
        values = data[index:end]
        return sum([val[2] for val in values]) / len(values)

    # Helper function to calculate average for the previous num_items
    def get_previous_average(index, data, num_items):
        start = max(0, index - num_items)
        values = data[start : index + 1]
        return sum([val[2] for val in values]) / len(values)

    for i in range(len(data)):
        current_value = data[i]
        next_average_value = get_next_average(i, data, num_items)
        previous_average_value = get_previous_average(i, data, num_items)
        time_dict[int(current_value[1])].append(
            (
                round(float(current_value[1]), 3),
                round(float(current_value[2]), 5),
                round(float(next_average_value), 5),
                round(float(previous_average_value), 5),
            )
        )

    return time_dict


def filter_sequential(data) -> dict:
    """
    Filters out sequentially increasing or decreasing items from a list of tuples,
    keeping only the first and last elements of each trend.

    Args:
        data (list): A list of tuples, where each tuple contains at least one numeric value.

    Returns:
        dict: A filtered list of tuples.
    """
    if not data:
        return {}
    index = 0
    result = [
        (index, data[0][1], data[0][2])
    ]  # Initialize result with the first element
    prev_value = int(
        data[0][2]
    )  # Assuming we are comparing the second item in the tuple
    prev_trend = 0  # 0: no trend, 1: increasing, -1: decreasing

    for i in range(1, len(data)):
        current_value = int(data[i][2])
        if current_value > prev_value:
            current_trend = 1  # Increasing
        elif current_value < prev_value:
            current_trend = -1  # Decreasing
        else:
            current_trend = 0  # No change

        if prev_trend == 0:
            prev_trend = current_trend
        elif current_trend != prev_trend and current_trend != 0:
            # Trend changed, add the previous element to the result
            index += 1
            result.append((index, data[i - 1][1], data[i - 1][2]))
            prev_trend = current_trend

        prev_value = current_value

    # Ensure the last element is included
    if result[-1] != data[-1]:
        index += 1
        result.append((index, data[-1][1], data[-1][2]))

    time_dict = defaultdict(list)
    time_dict[int(result[0][1])].append(
        (
            round(float(result[0][1]), 3),
            round(float(result[0][2]), 5),
            round(
                float((result[0][2] + result[0][2] + result[0][2] + result[1][2]) / 4),
                5,
            ),
        )
    )
    for i in range(len(result) - 2):
        prev_value = result[i]
        current_value = result[i + 1]
        next_value = result[i + 2]
        time_dict[int(current_value[1])].append(
            (
                round(float(current_value[1]), 3),
                round(float(current_value[2]), 5),
                round(
                    float(
                        (
                            prev_value[2]
                            + current_value[2]
                            + current_value[2]
                            + next_value[2]
                        )
                        / 4
                    ),
                    5,
                ),
            )
        )

    time_dict[int(result[-1][1])].append(
        (
            round(float(result[-1][1]), 3),
            round(float(result[-1][2]), 5),
            round(
                float(
                    (result[-2][2] + result[-1][2] + result[-1][2] + result[-1][2]) / 4
                ),
                5,
            ),
        )
    )
    return time_dict


# iso codes with specialized rules in uroman
special_isos_uroman = "ara, bel, bul, deu, ell, eng, fas, grc, ell, eng, heb, kaz, kir, lav, lit, mkd, mkd2, oss, pnt, pus, rus, srp, srp2, tur, uig, ukr, yid".split(
    ","
)
special_isos_uroman = [i.strip() for i in special_isos_uroman]


def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def get_uroman_tokens(norm_transcripts, iso=None):
    uroman = ur.Uroman()
    outtexts = []
    for line in norm_transcripts:
        if iso in special_isos_uroman:
            new_line = uroman.romanize_string(line, lcode=iso)
            new_line = " ".join(new_line.strip())
            new_line = re.sub(r"\s+", " ", new_line).strip()
            outtexts.append(new_line)
        else:
            new_line = uroman.romanize_string(line)
            new_line = " ".join(new_line.strip())
            new_line = re.sub(r"\s+", " ", new_line).strip()
            outtexts.append(new_line)

    assert len(outtexts) == len(norm_transcripts)
    uromans = []
    for ot in outtexts:
        uromans.append(normalize_uroman(ot))
    return uromans


@dataclass
class Segment:
    label: str
    start: int
    end: int

    def __repr__(self):
        return f"{self.label}: [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, idx_to_token_map):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)


def load_model_dict():
    model_path_name = "/tmp/ctc_alignment_mling_uroman_model.pt"

    logger.info("Downloading model and dictionary...")
    if os.path.exists(model_path_name):
        logger.info("Model path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt",
            model_path_name,
        )
        assert os.path.exists(model_path_name)
    state_dict = torch.load(model_path_name, map_location="cpu", weights_only=True)

    model = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.0,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.1,
        aux_num_out=31,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dict_path_name = "/tmp/ctc_alignment_mling_uroman_model.dict"
    if os.path.exists(dict_path_name):
        logger.info("Dictionary path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt",
            dict_path_name,
        )
        assert os.path.exists(dict_path_name)
    dictionary = {}
    with open(dict_path_name) as f:
        dictionary = {l.strip(): i for i, l in enumerate(f.readlines())}

    return model, dictionary


def get_spans(tokens, segments):
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    sil = "<blank>"
    for seg_idx, seg in enumerate(segments):
        if tokens_idx == len(tokens):
            assert seg_idx == len(segments) - 1
            assert seg.label == "<blank>"
            continue
        cur_token = tokens[tokens_idx].split(" ")
        ltr = cur_token[ltr_idx]
        if seg.label == "<blank>":
            continue
        assert seg.label == ltr
        if (ltr_idx) == 0:
            start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                intervals.append((seg_idx, seg_idx))
                tokens_idx += 1
        else:
            ltr_idx += 1
    spans = []
    for idx, (start, end) in enumerate(intervals):
        span = segments[start : end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == sil:
                pad_start = (
                    prev_seg.start
                    if (idx == 0)
                    else int((prev_seg.start + prev_seg.end) / 2)
                )
                span = [Segment(sil, pad_start, span[0].start)] + span
        if end + 1 < len(segments):
            next_seg = segments[end + 1]
            if next_seg.label == sil:
                pad_end = (
                    next_seg.end
                    if (idx == len(intervals) - 1)
                    else math.floor((next_seg.start + next_seg.end) / 2)
                )
                span = span + [Segment(sil, span[-1].end, pad_end)]
        spans.append(span)
    return spans


colon = ":"
comma = ","
exclamation_mark = "!"
period = re.escape(".")
question_mark = re.escape("?")
semicolon = ";"

left_curly_bracket = "{"
right_curly_bracket = "}"
quotation_mark = '"'

basic_punc = (
    period
    + question_mark
    + comma
    + colon
    + exclamation_mark
    + left_curly_bracket
    + right_curly_bracket
)

# General punc unicode block (0x2000-0x206F)
zero_width_space = r"\u200B"
zero_width_nonjoiner = r"\u200C"
left_to_right_mark = r"\u200E"
right_to_left_mark = r"\u200F"
left_to_right_embedding = r"\u202A"
pop_directional_formatting = r"\u202C"

# Here are some commonly ill-typed versions of apostrophe
right_single_quotation_mark = r"\u2019"
left_single_quotation_mark = r"\u2018"

# Language specific definitions
# Spanish
inverted_exclamation_mark = r"\u00A1"
inverted_question_mark = r"\u00BF"


# Hindi
hindi_danda = "\u0964"

# Egyptian Arabic
# arabic_percent = r"\u066A"
arabic_comma = r"\u060C"
arabic_question_mark = r"\u061F"
arabic_semicolon = r"\u061B"
arabic_diacritics = r"\u064B-\u0652"


arabic_subscript_alef_and_inverted_damma = r"\u0656-\u0657"


# Chinese
full_stop = r"\u3002"
full_comma = r"\uFF0C"
full_exclamation_mark = r"\uFF01"
full_question_mark = r"\uFF1F"
full_semicolon = r"\uFF1B"
full_colon = r"\uFF1A"
full_parentheses = r"\uFF08\uFF09"
quotation_mark_horizontal = r"\u300C-\u300F"
quotation_mark_vertical = r"\uFF41-\uFF44"
title_marks = r"\u3008-\u300B"
wavy_low_line = r"\uFE4F"
ellipsis = r"\u22EF"
enumeration_comma = r"\u3001"
hyphenation_point = r"\u2027"
forward_slash = r"\uFF0F"
wavy_dash = r"\uFF5E"
box_drawings_light_horizontal = r"\u2500"
fullwidth_low_line = r"\uFF3F"
chinese_punc = (
    full_stop
    + full_comma
    + full_exclamation_mark
    + full_question_mark
    + full_semicolon
    + full_colon
    + full_parentheses
    + quotation_mark_horizontal
    + quotation_mark_vertical
    + title_marks
    + wavy_low_line
    + ellipsis
    + enumeration_comma
    + hyphenation_point
    + forward_slash
    + wavy_dash
    + box_drawings_light_horizontal
    + fullwidth_low_line
)

# Armenian
armenian_apostrophe = r"\u055A"
emphasis_mark = r"\u055B"
exclamation_mark = r"\u055C"
armenian_comma = r"\u055D"
armenian_question_mark = r"\u055E"
abbreviation_mark = r"\u055F"
armenian_full_stop = r"\u0589"
armenian_punc = (
    armenian_apostrophe
    + emphasis_mark
    + exclamation_mark
    + armenian_comma
    + armenian_question_mark
    + abbreviation_mark
    + armenian_full_stop
)

lesser_than_symbol = r"&lt;"
greater_than_symbol = r"&gt;"

lesser_than_sign = r"\u003c"
greater_than_sign = r"\u003e"

nbsp_written_form = r"&nbsp"

# Quotation marks
left_double_quotes = r"\u201c"
right_double_quotes = r"\u201d"
left_double_angle = r"\u00ab"
right_double_angle = r"\u00bb"
left_single_angle = r"\u2039"
right_single_angle = r"\u203a"
low_double_quotes = r"\u201e"
low_single_quotes = r"\u201a"
high_double_quotes = r"\u201f"
high_single_quotes = r"\u201b"

all_punct_quotes = (
    left_double_quotes
    + right_double_quotes
    + left_double_angle
    + right_double_angle
    + left_single_angle
    + right_single_angle
    + low_double_quotes
    + low_single_quotes
    + high_double_quotes
    + high_single_quotes
    + right_single_quotation_mark
    + left_single_quotation_mark
)
mapping_quotes = (
    "["
    + high_single_quotes
    + right_single_quotation_mark
    + left_single_quotation_mark
    + "]"
)


# Digits

english_digits = r"\u0030-\u0039"
bengali_digits = r"\u09e6-\u09ef"
khmer_digits = r"\u17e0-\u17e9"
devanagari_digits = r"\u0966-\u096f"
oriya_digits = r"\u0b66-\u0b6f"
extended_arabic_indic_digits = r"\u06f0-\u06f9"
kayah_li_digits = r"\ua900-\ua909"
fullwidth_digits = r"\uff10-\uff19"
malayam_digits = r"\u0d66-\u0d6f"
myanmar_digits = r"\u1040-\u1049"
roman_numeral = r"\u2170-\u2179"
nominal_digit_shapes = r"\u206f"

# Load punctuations from MMS-lab data
with open(
    f"{os.path.dirname(__file__)}/punctuations.lst", "r", encoding="utf-8"
) as punc_f:
    punc_list = punc_f.readlines()

punct_pattern = r""
for punc in punc_list:
    # the first character in the tab separated line is the punc to be removed
    punct_pattern += re.escape(punc.split("\t")[0])

shared_digits = (
    english_digits
    + bengali_digits
    + khmer_digits
    + devanagari_digits
    + oriya_digits
    + extended_arabic_indic_digits
    + kayah_li_digits
    + fullwidth_digits
    + malayam_digits
    + myanmar_digits
    + roman_numeral
    + nominal_digit_shapes
)

shared_punc_list = (
    basic_punc
    + all_punct_quotes
    + greater_than_sign
    + lesser_than_sign
    + inverted_question_mark
    + full_stop
    + semicolon
    + armenian_punc
    + inverted_exclamation_mark
    + arabic_comma
    + enumeration_comma
    + hindi_danda
    + quotation_mark
    + arabic_semicolon
    + arabic_question_mark
    + chinese_punc
    + punct_pattern
)

shared_mappping = {
    lesser_than_symbol: "",
    greater_than_symbol: "",
    nbsp_written_form: "",
    r"(\S+)" + mapping_quotes + r"(\S+)": r"\1'\2",
}

shared_deletion_list = (
    left_to_right_mark
    + zero_width_nonjoiner
    + arabic_subscript_alef_and_inverted_damma
    + zero_width_space
    + arabic_diacritics
    + pop_directional_formatting
    + right_to_left_mark
    + left_to_right_embedding
)

norm_config = {
    "*": {
        "lower_case": True,
        "punc_set": shared_punc_list,
        "del_set": shared_deletion_list,
        "mapping": shared_mappping,
        "digit_set": shared_digits,
        "unicode_norm": "NFKC",
        "rm_diacritics": False,
    }
}

# =============== Mongolian ===============#

norm_config["mon"] = norm_config["*"].copy()
# add soft hyphen to punc list to match with fleurs
norm_config["mon"]["del_set"] += r"\u00AD"

norm_config["khk"] = norm_config["mon"].copy()

# =============== Hebrew ===============#

norm_config["heb"] = norm_config["*"].copy()
# add "HEBREW POINT" symbols to match with fleurs
norm_config["heb"]["del_set"] += r"\u05B0-\u05BF\u05C0-\u05CF"

# =============== Thai ===============#

norm_config["tha"] = norm_config["*"].copy()
# add "Zero width joiner" symbols to match with fleurs
norm_config["tha"]["punc_set"] += r"\u200D"

# =============== Arabic ===============#
norm_config["ara"] = norm_config["*"].copy()
norm_config["ara"]["mapping"]["ٱ"] = "ا"
norm_config["arb"] = norm_config["ara"].copy()

# =============== Javanese ===============#
norm_config["jav"] = norm_config["*"].copy()
norm_config["jav"]["rm_diacritics"] = True


def text_normalize(
    text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False
):
    """Given a text, normalize it by changing to lower case, removing punctuations, removing words that only contain digits and removing extra spaces

    Args:
        text : The string to be normalized
        iso_code :
        remove_numbers : Boolean flag to specify if words containing only digits should be removed

    Returns:
        normalized_text : the string after all normalization

    """

    config = norm_config.get(iso_code, norm_config["*"])

    for field in [
        "lower_case",
        "punc_set",
        "del_set",
        "mapping",
        "digit_set",
        "unicode_norm",
    ]:
        if field not in config:
            config[field] = norm_config["*"][field]

    text = unicodedata.normalize(config["unicode_norm"], text)

    # Convert to lower case

    if config["lower_case"] and lower_case:
        text = text.lower()

    # brackets

    # always text inside brackets with numbers in them. Usually corresponds to "(Sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)

    # Apply mappings

    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)

    # Replace punctutations with space

    punct_pattern = r"[" + config["punc_set"]

    punct_pattern += "]"

    normalized_text = re.sub(punct_pattern, " ", text)

    # remove characters in delete list

    delete_patten = r"[" + config["del_set"] + "]"

    normalized_text = re.sub(delete_patten, "", normalized_text)

    # Remove words containing only digits
    # We check for 3 cases  a)text starts with a number b) a number is present somewhere in the middle of the text c) the text ends with a number
    # For each case we use lookaround regex pattern to see if the digit pattern in preceded and followed by whitespaces, only then we replace the numbers with space
    # The lookaround enables overlapping pattern matches to be replaced

    if remove_numbers:
        digits_pattern = "[" + config["digit_set"]

        digits_pattern += "]+"

        complete_digit_pattern = (
            r"^"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + "$"
        )

        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)

    if config["rm_diacritics"]:
        from unidecode import unidecode

        normalized_text = unidecode(normalized_text)

    # Remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text


SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_emissions(model, audio):
    # Convert NumPy array to PyTorch tensor
    waveform = torch.tensor(audio["waveform"], dtype=torch.float32).unsqueeze(
        0
    )  # Add a batch dimension (1 channel)
    # Move to the desired device
    waveform = waveform.to(DEVICE)

    audio_sf = audio["sample_rate"]

    # total_duration = sox.file_info.duration(audio_file)
    total_duration = waveform.shape[1] / audio_sf

    # Resample if the audio sample rate doesn't match the target
    if audio_sf != SAMPLING_FREQ:
        logger.info(f"Resampling from {audio_sf} Hz to {SAMPLING_FREQ} Hz")
        resampler = T.Resample(orig_freq=audio_sf, new_freq=SAMPLING_FREQ).to(
            DEVICE
        )  # Move resampler to the same device
        waveform = resampler(waveform)
        audio_sf = SAMPLING_FREQ

    assert audio_sf == SAMPLING_FREQ

    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)

            context = EMISSION_INTERVAL * 0.5
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                :,
                int(SAMPLING_FREQ * input_start_time) : int(
                    SAMPLING_FREQ * (input_end_time)
                ),
            ]

            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return emissions, stride


def get_alignments(
    audio,
    tokens,
    model,
    dictionary,
    use_star,
):
    # Generate emissions
    emissions, stride = generate_emissions(model, audio)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)

    # Force Alignment
    if tokens:
        token_indices = [
            dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary
        ]
    else:
        logger.error(f"Empty transcript!!!!! for audio file")
        token_indices = []

    blank = dictionary["<blank>"]

    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)

    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
    path, _ = F.forced_align(
        emissions.unsqueeze(0),
        targets.unsqueeze(0),
        input_lengths,
        target_lengths,
        blank=blank,
    )
    path = path.squeeze().to("cpu").tolist()

    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride


def quick_align(audio, transcripts, lang="en"):
    transcripts = [segment.strip() for segment in transcripts if segment.strip()]
    transcripts = [
        item for pair in zip(transcripts, [""] * len(transcripts)) for item in pair
    ][:-1] + [""]
    norm_transcripts = [text_normalize(line.strip(), lang) for line in transcripts]

    tokens = get_uroman_tokens(norm_transcripts, lang)
    model, dictionary = load_model_dict()
    model = model.to(DEVICE)

    dictionary["<star>"] = len(dictionary)
    tokens = ["<star>"] + tokens
    transcripts = ["<star>"] + transcripts
    norm_transcripts = ["<star>"] + norm_transcripts

    segments, stride = get_alignments(
        audio,
        tokens,
        model,
        dictionary,
        use_star=True,
    )
    # Get spans of each line in input text file
    spans = get_spans(tokens, segments)
    segment_data = list(zip(transcripts, norm_transcripts, tokens, spans))

    # Load the audio file
    # Convert NumPy array to PyTorch tensor
    waveform = torch.tensor(audio["waveform"], dtype=torch.float32).unsqueeze(
        0
    )  # Add a batch dimension (1 channel)
    # Move to the desired device
    waveform = waveform.to(DEVICE)
    # Sampling rate can be passed if needed
    sample_rate = audio["sample_rate"]
    # test = get_valley_tree(waveform, sample_rate)
    valleys = get_valley_tree(waveform, sample_rate)
    valleys = filter_sequential(valleys)
    results = []
    for i, t in enumerate(
        [segment_data[i : i + 3] for i in range(0, len(segment_data) - 2, 2)]
    ):
        start_span = t[0][3]
        end_span = t[-1][3]
        start_seg_start_idx = start_span[0].start
        start_seg_end_idx = start_span[-1].end

        end_seg_start_idx = end_span[0].start
        end_seg_end_idx = end_span[-1].end

        start_audio_start_sec = start_seg_start_idx * stride / 1000
        start_audio_end_sec = start_seg_end_idx * stride / 1000

        end_audio_start_sec = end_seg_start_idx * stride / 1000
        end_audio_end_sec = end_seg_end_idx * stride / 1000

        start_lowest = find_minimum_in_range(
            valleys, start_audio_start_sec, start_audio_end_sec
        )
        end_lowest = find_minimum_in_range(
            valleys, end_audio_start_sec, end_audio_end_sec
        )

        if end_lowest is None:
            actual_end = round(float(end_audio_end_sec), 3)
        else:
            actual_end = round(float(end_lowest[0]), 3)

        if start_lowest is None:
            actual_start = round(float(start_audio_end_sec), 3)
        else:
            actual_start = round(float(start_lowest[0]), 3)
        sample = {
            "start": actual_start,
            "end": actual_end,
            "text": t[1][0],
        }
        results.append(sample)

    return results


def align(
    audio,
    transcription_segments,
    valleys,
    lang="eng",
    breaths: intervaltree.IntervalTree = None,
):
    transcripts = [segment["text"].strip() for segment in transcription_segments]
    transcripts = [
        item for pair in zip(transcripts, [""] * len(transcripts)) for item in pair
    ][:-1] + [""]
    norm_transcripts = [text_normalize(line.strip(), lang) for line in transcripts]

    tokens = get_uroman_tokens(norm_transcripts, lang)
    model, dictionary = load_model_dict()
    model = model.to(DEVICE)

    dictionary["<star>"] = len(dictionary)
    tokens = ["<star>"] + tokens
    transcripts = ["<star>"] + transcripts
    norm_transcripts = ["<star>"] + norm_transcripts

    segments, stride = get_alignments(
        audio,
        tokens,
        model,
        dictionary,
        use_star=True,
    )
    # Get spans of each line in input text file
    spans = get_spans(tokens, segments)
    segment_data = list(zip(transcripts, norm_transcripts, tokens, spans))

    results = []
    intervaltree = IntervalTree()
    for i, t in enumerate(
        [segment_data[i : i + 3] for i in range(0, len(segment_data) - 2, 2)]
    ):
        start_span = t[0][3]
        end_span = t[-1][3]
        start_seg_start_idx = start_span[0].start
        start_seg_end_idx = start_span[-1].end

        end_seg_start_idx = end_span[0].start
        end_seg_end_idx = end_span[-1].end

        start_audio_start_sec = start_seg_start_idx * stride / 1000
        start_audio_end_sec = start_seg_end_idx * stride / 1000

        end_audio_start_sec = end_seg_start_idx * stride / 1000
        end_audio_end_sec = end_seg_end_idx * stride / 1000

        # start_lowest = find_minimum_in_range(
        #     valleys, start_audio_start_sec, start_audio_end_sec
        # )
        if 499 < start_audio_end_sec < 502:
            pass

        if start_audio_end_sec < transcription_segments[i].get("start"):
            if i == 0:
                start_lowest = find_minimum_in_range(
                    valleys,
                    start_audio_start_sec,
                    transcription_segments[i].get("start"),
                )
            else:
                start = start_audio_end_sec
                if breaths:
                    overlaps = breaths.overlap(
                        start_audio_end_sec, transcription_segments[i].get("start")
                    )
                    for overlap in overlaps:
                        peak = overlap.data
                        if peak and start_audio_end_sec < peak[
                            0
                        ] < transcription_segments[i].get("start"):
                            start = peak[0]
                        else:
                            pass

                start_lowest = find_minimum_in_range(
                    valleys, start, transcription_segments[i].get("start")
                )
        else:
            start_lowest = find_minimum_in_range(
                valleys, start_audio_start_sec, start_audio_end_sec
            )
        end_lowest = find_minimum_in_range(
            valleys, end_audio_start_sec, end_audio_end_sec
        )
        intervaltree.addi(
            start_audio_start_sec, start_audio_end_sec + 0.000001, "range"
        )
        intervaltree.addi(end_audio_start_sec, end_audio_end_sec + 0.000001, "range")

        if end_lowest is None:
            actual_end = round(float(end_audio_end_sec), 3)
            end_db = 0
        else:
            actual_end = round(float(end_lowest[0]), 3)
            end_db = round(float(end_lowest[1]), 2)

        if start_lowest is None:
            actual_start = round(float(transcription_segments[i].get("start")), 3)
            start_db = 0
        else:
            actual_start = round(float(start_lowest[0]), 3)
            start_db = round(float(start_lowest[1]), 2)
        if actual_start > actual_end:
            actual_start = round(float(start_audio_end_sec), 3)
        if actual_start > actual_end:
            actual_start = round(float(start_audio_start_sec), 3)
        if abs(start_audio_end_sec - transcription_segments[i].get("start")) < 0.05:
            print(
                "small distance",
                start_audio_end_sec - transcription_segments[i].get("start"),
                start_audio_end_sec,
                transcription_segments[i].get("start"),
                start_db,
            )
        if abs(end_audio_start_sec - end_audio_end_sec) < 0.05:
            print(
                "small distance",
                end_audio_start_sec - end_audio_end_sec,
                end_audio_start_sec,
                end_audio_end_sec,
                end_db,
            )

        lowest_data = {
            "start": {
                "start": start_audio_start_sec,
                "end": start_audio_end_sec,
                "time": actual_start,
                "db": start_db,
            },
            "end": {
                "start": end_audio_start_sec,
                "end": end_audio_end_sec,
                "time": actual_end,
                "db": end_db,
            },
        }
        sample = {
            "start": actual_start,
            "lowest": lowest_data,
            "end": actual_end,
            "duration": actual_end - actual_start,
            "text": t[1][0],
            "normalized_text": t[1][1],
            "uroman_tokens": t[1][2],
        }
        results.append(sample)

    return results, intervaltree


def get_valleys(audio: dict) -> (dict, dict):
    # Load the audio file
    # Convert NumPy array to PyTorch tensor
    waveform = torch.tensor(audio["waveform"], dtype=torch.float32).unsqueeze(
        0
    )  # Add a batch dimension (1 channel)
    # Move to the desired device
    waveform = waveform.to(DEVICE)
    # Sampling rate can be passed if needed
    sample_rate = audio["sample_rate"]
    # test = get_valley_tree(waveform, sample_rate)
    valleys = get_valley_tree(waveform, sample_rate)
    return get_valley_dict(valleys, num_items=50), filter_sequential(valleys)
