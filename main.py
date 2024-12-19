import logging
import os
import torch
import onnxruntime as ort
from diskcache import Cache
import librosa
import time
from models.respiro import BreathDetector
from models.uroman_align import align, quick_align, get_valleys, find_threshold_crossing
from intervaltree import IntervalTree
from textgrid import TextGrid, IntervalTier
import statistics
import numpy as np
import soundfile as sf

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


def create_textgrid_from_data(dictionary, output_path):
    # Initialize a TextGrid object
    tg = TextGrid()

    for key, values in dictionary.items():
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
def process(audio_path):
    logger.info("Step 0: Load audio")
    y, sr = librosa.load(audio_path, sr=None)
    audio = {"waveform": y, "sample_rate": sr, "path": audio_path}

    if os.path.exists(f"{audio_path}.txt"):
        logger.info("Step 1: Using provided test instead of ASR")
        if debug:
            with open(f"{audio_path}.txt", "r") as f:
                segments = f.readlines()
            asrx_result = cache.get(f"byob_transcribe:{audio_path}", None)
            if asrx_result is None:
                asrx_result = byob_transcribe(audio, segments)
                cache.set(f"byob_transcribe:{audio_path}", asrx_result)
        else:
            with open(f"{audio_path}.txt", "r") as f:
                segments = f.readlines()
            asrx_result = byob_transcribe(audio, segments)
    else:
        logger.info("Step 1: whisperx ASR")
        if debug:
            asrx_result = cache.get(f"asrx_result:{audio_path}", None)
            if asrx_result is None:
                asrx_result = asr(audio)
                cache.set(f"asrx_result:{audio_path}", asrx_result)
        else:
            asrx_result = asr(audio)
    # logger.info(f"ASR Result: {asrx_result}")

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

    threshold = -60
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

    padding = 0.15
    output_path = audio.get("path")
    output_path = output_path.replace("data/audio", "data/processed")
    output_path += "_segments_test"
    segment_with_threshold(audio, alignment, output_path, padding)

    logger.info("step 5: Fixing missing word start and end times")
    for index in range(len(asrx_result.get("word_segments")) - 1):
        if "end" not in asrx_result.get("word_segments")[index]:
            if index == len(asrx_result.get("word_segments")) - 1:
                asrx_result.get("word_segments")[index]["end"] = len(y) / sr
            else:
                asrx_result.get("word_segments")[index]["end"] = asrx_result.get(
                    "word_segments"
                )[index + 1]["start"]
        if "start" not in asrx_result.get("word_segments")[index]:
            if index == 0:
                asrx_result.get("word_segments")[index]["start"] = 0
            else:
                asrx_result.get("word_segments")[index]["start"] = asrx_result.get(
                    "word_segments"
                )[index - 1]["end"]

    breath_peaks = [breath.data[1] for breath in breaths if breath.data]
    mean = statistics.mean(breath_peaks)
    std_dev = statistics.stdev(breath_peaks)

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
    mean = statistics.mean(edge_db)
    std_dev = statistics.stdev(edge_db)

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

    logger.info("Step 5: Create TextGrid")
    create_textgrid_from_data(
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


def segment_with_threshold(audio, alignment, output_path, padding):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(len(alignment)):
        try:
            start = alignment[i][1].get("padding_start")
            end = alignment[i][1].get("padding_end")

            left_padding = np.zeros(int(padding * audio.get("sample_rate")))
            right_padding = np.zeros(int(padding * audio.get("sample_rate")))

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


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    debug = False
    # Load models
    if detect_gpu():
        logger.info("Using GPU")
        device_name = "cuda"
        device = torch.device(device_name)
    else:
        logger.info("Using CPU")
        device_name = "cpu"
        device = torch.device(device_name)

    logger.debug(" * Loading ASRX Model")
    from models import whisperx_asr

    whisperx_asr_model = whisperx_asr.WhisperXModel(
        "large-v3-turbo",
        language="en",
        device=device_name,
        compute_type="float16",
        suppress_numerals=True,
        threads=1,
        asr_options={
            "hotwords":None,
            "multilingual": True,
            "suppress_numerals": True,
            "initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음.",
        },
    )
    logger.debug("ASRX Model loaded")

    logger.debug(" * Loading Respiro Model")
    # breath detection
    breath_model = BreathDetector(
        model_path="models/respiro-en.pt",
        threshold=0.064,
        min_length=10,
    )
    logger.debug("Respiro Model loaded")

    input_folder_path = "data/audio"
    audio_paths = get_audio_files(input_folder_path)  # Get all audio files
    for path in audio_paths:
        process(path)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
