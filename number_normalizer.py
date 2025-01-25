import re
import string
import logging

logging.basicConfig(level=logging.WARNING)  # Adjust level as needed

# Extended mapping with higher magnitude words
word_mapping = {
    # Basic numbers
    "zero":      [(0, 1)],
    "one":       [(1, 1)],
    "two":       [(2, 1)],
    "three":     [(3, 1)],
    "four":      [(4, 1)],
    "five":      [(5, 1)],
    "six":       [(6, 1)],
    "seven":     [(7, 1)],
    "eight":     [(8, 1)],
    "nine":      [(9, 1)],
    "ten":       [(1, 10), (0, 1)],
    "eleven":    [(1, 10), (1, 1)],
    "twelve":    [(1, 10), (2, 1)],
    "thirteen":  [(1, 10), (3, 1)],
    "fourteen":  [(1, 10), (4, 1)],
    "fifteen":   [(1, 10), (5, 1)],
    "sixteen":   [(1, 10), (6, 1)],
    "seventeen": [(1, 10), (7, 1)],
    "eighteen":  [(1, 10), (8, 1)],
    "nineteen":  [(1, 10), (9, 1)],
    "twenty":    [(2, 10)],
    "thirty":    [(3, 10)],
    "forty":     [(4, 10)],
    "fifty":     [(5, 10)],
    "sixty":     [(6, 10)],
    "seventy":   [(7, 10)],
    "eighty":    [(8, 10)],
    "ninety":    [(9, 10)],
    "hundred":   [(1, 100)],
    "thousand":  [(1, 1000)],
    "million":   [(1, 10**6)],
    "billion":   [(1, 10**9)],
    "trillion":  [(1, 10**12)],
    # You can continue up to decillion, googol, etc. as needed...
    "and":       [],
    # Removed "a" from here to avoid the bug
}

# For fractional words: "half" => 0.5, "quarter" => 0.25, etc.
pre_process_mapping = {
    "half":      0.5,
    "quarter":   0.25,
    "third":     1/3,
    "fourth":    0.25,
    "fifth":     0.2,
    "sixth":     1/6,
    "seventh":   1/7,
    "eighth":    0.125,
    "ninth":     1/9,
    "tenth":     0.1
}

large_scales = {"hundred", "thousand", "million", "billion", "trillion"}
fraction_words = set(pre_process_mapping.keys())  # e.g. "half", "quarter", ...

def parse_small_number(tokens):
    """
    Parse a short sequence of tokens (e.g., 'twenty one', 'ninety nine') that
    do not contain large scale breaks (like thousand, million, etc.).

    :param tokens: List of string tokens for a small portion of the number
    :return: Integer value representing the parsed number from the tokens
    """
    total = 0
    current = 0
    last_scale = None

    # We'll do an index-based loop so we can look ahead for the "a" logic.
    i = 0
    while i < len(tokens):
        word = tokens[i]

        # --- Special-case handling for "a" ---
        # If "a" is followed by a recognized numeric scale or numeric token, treat "a" as 1.
        # Otherwise, skip it to avoid "a pig" => "1 pig".
        if word.lower() == 'a':
            # Look ahead
            next_i = i + 1
            if next_i < len(tokens):
                next_word = tokens[next_i].lower()
                # Decide if next_word is numeric or a scale:
                if next_word in word_mapping or next_word in large_scales:
                    # Treat 'a' as 1 => parse as if it were "one"
                    # We'll artificially set pairs to (1, 1) for processing
                    pairs = [(1, 1)]
                else:
                    # Skip 'a' since next word is not numeric or scale
                    i += 1
                    continue
            else:
                # 'a' is the last word, skip it
                i += 1
                continue
        else:
            # Normal numeric token
            if word.lower() not in word_mapping or not word_mapping[word.lower()]:
                i += 1
                continue
            pairs = word_mapping[word.lower()]

        # Now handle the parse logic for the pairs
        # (For example, "ten" => [(1, 10), (0, 1)], "hundred" => [(1, 100)] )

        # Special-case: teen numbers like "fourteen" => 1*10 + 4
        if len(pairs) == 2 and pairs[0][1] == 10 and pairs[1][1] == 1:
            teen_value = pairs[0][0] * 10 + pairs[1][0]
            if last_scale == 10:
                current = int(f"{current}{teen_value}")
            else:
                current += teen_value
            last_scale = 10
            i += 1
            continue

        # General handling
        for (value, scale) in pairs:
            if scale == 1:
                # e.g., "two", "one", "nine"
                if last_scale == 10:
                    # "twenty one" => 20 + 1
                    current += value
                else:
                    # "two" => current * 10 + value can help with sequences like "one nine"
                    current = current * 10 + value
            elif scale == 10:
                # e.g., "twenty", "thirty", etc.
                computed_value = value * scale
                if last_scale == 10:
                    # chain tens
                    current = int(f"{current}{computed_value}")
                else:
                    current += computed_value
                last_scale = 10
            else:
                # e.g., "hundred"
                if current == 0:
                    current = 1
                current *= scale
                last_scale = scale

        i += 1

    total += current
    return total


def parse_number_segment(segment):
    """
    Parse a continuous segment of words that represent a number (e.g., "nineteen eighty-five",
    "one thousand three hundred").

    :param segment: The string (pre-split or pre-cleaned) containing numeric words
    :return: The integer value (or float if needed) of that segment
    """
    segment = segment.lower()
    segment = re.sub(r"[^\w\s\-]", "", segment)  # remove punctuation except hyphens
    tokens = segment.replace('-', ' ').split()

    # "Groups" will break on large scales, so we handle them separately
    groups = []
    current_group = []

    for token in tokens:
        if token in large_scales and token != "hundred":
            # If we see thousand/million/trillion,
            # finalize the current group, then this becomes its own group
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([token])
        else:
            current_group.append(token)
    if current_group:
        groups.append(current_group)

    total = 0
    current_value = 0

    i = 0
    while i < len(groups):
        group = groups[i]
        # If the group is just a large scale (e.g., ["thousand"])
        if len(group) == 1 and group[0] in large_scales and group[0] != "hundred":
            scale_word = group[0]
            # Usually we multiply a preceding group by this scale
            preceding_value = 1
            if i > 0:
                preceding_value = parse_small_number(groups[i - 1])
                # Zero out the preceding group so we don't double-parse
                groups[i - 1] = []
            # e.g., "thousand" => scale=1000
            scale = word_mapping[scale_word][0][1]
            total += preceding_value * scale
            i += 1
            current_value = 0
        else:
            # parse small group
            value = parse_small_number(group)
            current_value += value
            i += 1

    final_result = total + current_value
    return final_result


# Prepare regex pattern for numeric words.
# Include "and" to handle it in the match, but skip "a" so it doesn't get turned into 1 automatically.
valid_numeric_words = sorted(
    [w for w, v in word_mapping.items() if v is not None] + ["and"],  # ignoring "a"
    key=len,
    reverse=True
)

fraction_keywords = fraction_words.union({"point", "dot"})
# If someone typed "point" or "dot" we handle decimals

pattern = r'\b(?!and\b)(?:' + '|'.join(re.escape(w) for w in valid_numeric_words) + r')(?:[\-\s]+(?:' + '|'.join(re.escape(w) for w in valid_numeric_words) + r'))*\b'

# Store global phrase->number mappings if desired
phrase_to_number = {}

def replace_numeric(match):
    """
    For each regex match of numeric words, parse and replace with the corresponding
    numeric value (as an integer or float). Also store them in phrase_to_number.
    """
    numeric_phrase = match.group(0).strip()

    # Check if there's a decimal-like word
    if "point" in numeric_phrase or "dot" in numeric_phrase:
        # e.g., "two point five" => 2.5
        if "point" in numeric_phrase:
            integer_part, fractional_part = numeric_phrase.split("point", 1)
        else:
            integer_part, fractional_part = numeric_phrase.split("dot", 1)

        integer_value = parse_number_segment(integer_part.strip())
        fractional_tokens = fractional_part.strip().split()
        fractional_str = ""
        for token in fractional_tokens:
            digit_value = parse_small_number([token])
            fractional_str += str(digit_value)
        fractional_value = float("0." + fractional_str) if fractional_str else 0.0
        number = integer_value + fractional_value

    # Check if there's a fraction word
    elif any(frac in numeric_phrase for frac in fraction_words):
        # If a fraction word is present, we separate "integer part" and "fraction part"
        # using the word "and" if present. E.g. "eight and a half" => 8 + 0.5
        parts = numeric_phrase.split("and")

        if len(parts) > 1:
            # There's an "integer part" + "fraction part"
            whole_part = parts[0].strip()
            whole_value = parse_number_segment(whole_part) if whole_part else 0
            fraction_value = 0
            # Sum up all fraction keywords in the "and ..." part
            for part in parts[1:]:
                for word in part.split():
                    if word in pre_process_mapping:
                        fraction_value += pre_process_mapping[word]
            number = whole_value + fraction_value
        else:
            # No 'and' => treat the entire phrase as purely fractional or a single piece
            # e.g., "a half" => 0.5, "half" => 0.5, "three quarters" => not defined here,
            # but "three quarter" might need more complex logic
            text_tokens = parts[0].strip().split()
            # Attempt to parse an integer portion ignoring fraction words
            fraction_value = 0
            fraction_indices = []
            for idx, tok in enumerate(text_tokens):
                if tok in pre_process_mapping:
                    fraction_value += pre_process_mapping[tok]
                    fraction_indices.append(idx)

            # Remove fraction words from the tokens, parse the remainder as the "whole part"
            # so "a half" => skip 'a', skip 'half' from parse_number_segment
            # effectively " " => 0 + 0.5 => 0.5
            remainder = [t for i, t in enumerate(text_tokens) if i not in fraction_indices and t not in ("a", "an")]
            whole_str = " ".join(remainder)
            whole_value = parse_number_segment(whole_str) if whole_str else 0

            number = whole_value + fraction_value

    else:
        # Simple integer logic
        number = parse_number_segment(numeric_phrase)

    phrase_to_number[numeric_phrase] = number
    return str(number)


def replace_numbers_in_text(text):
    """
    Find all numeric phrases in `text` that match the pattern of recognized words
    and replace them with the corresponding numeric value (integer or float).
    Also returns a dictionary of phrase->number mappings.
    """
    global phrase_to_number
    phrase_to_number = {}  # reset for each new text processing

    # Use case-insensitive substitution
    result = re.sub(pattern, replace_numeric, text, flags=re.IGNORECASE)
    return result, phrase_to_number


def normalize_text(text):
    """
    Lowercase the text and remove punctuation for normalization.
    """
    return text.lower().translate(str.maketrans('', '', string.punctuation))


def extract_and_update_sequence(data, target):
    """
    Given a list of data entries (dictionaries) each containing a 'word' field,
    match those words in sequence with `target` (a string), and update the entries
    to exactly match the target words (including punctuation, case, etc.) if found.
    """
    target_words = target.split()
    normalized_target = [normalize_text(word) for word in target_words]

    matched_entries = []
    target_index = 0

    for entry in data:
        normalized_entry_word = normalize_text(entry['word'])
        if normalized_entry_word == normalized_target[target_index]:
            entry['word'] = target_words[target_index]
            matched_entries.append(entry)
            target_index += 1
            if target_index == len(normalized_target):
                break

    if target_index != len(normalized_target):
        logging.warning("Not all parts of the target string were found in sequence.")

    return matched_entries


def main():
    """
    Demo/test usage of our numeric replacement logic.
    """
    test_phrases = [
        "one billion thirty-four million eight hundred ninety",
        "nineteen eighty-five",
        "a thousand nine hundred eighty-five",
        "a thousand nine hundred and eighty-five",
        "one thousand nine hundred eighty-five",
        "one thousand nine hundred and eighty-five",
        "nineteen hundred eighty-five",
        "nineteen hundred and eighty-five",
        "one nine eight five",
        "hello and Nineteen ninety-eight, Twenty twenty-three.",
        "one billion two hundred thirty-four million five hundred sixty-seven thousand eight hundred ninety",
        "eight dot three",
        "eight and a half",
        "eight and three quarters",  # "quarter" is 0.25 in your mapping
        "two point nine five",
        "one hundred point nine two six three",
        "a pig",                 # Test the fix: should remain "a pig", not "1 pig"
        "a thousand",           # Should become 1000
        "a half",               # Depending on logic, you may want 0.5
    ]

    for phrase in test_phrases:
        replaced, mapping = replace_numbers_in_text(phrase)
        print(f"Original: {phrase}")
        print(f"Replaced: {replaced}")
        print(f"Mapping:  {mapping}")
        print("-" * 50)


if __name__ == "__main__":
    main()