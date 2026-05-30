from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from cs336_data.common import get_shared_assets_path
import fasttext
import re
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


def extract_text(html_bytes: bytes) -> str:
    encoding_format = detect_encoding(html_bytes)
    decoded_str = html_bytes.decode(encoding_format)
    ret = extract_plain_text(decoded_str)
    return ret


def identify_language(text: str) -> tuple[str, float]:
    pred_model = fasttext.load_model(
        f"{get_shared_assets_path()}/classifiers/lid.176.bin"
    )
    text = text.replace("\n", " ")
    labels, scores = pred_model.predict(text, k=1)
    label = labels[0]
    score = scores[0]
    lang = label[len("__label__") :]
    return lang, score


def mask_email(text: str):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    masked_text = re.sub(pattern, "|||EMAIL_ADDRESS|||", text)
    count = len(re.findall(pattern, text))
    return masked_text, count


def mask_phone(text: str):
    pattern = r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    masked_text = re.sub(pattern, "|||PHONE_NUMBER|||", text)
    count = len(re.findall(pattern, text))
    return masked_text, count


def mask_ip(text: str):
    pattern = r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    masked_text = re.sub(pattern, "|||IP_ADDRESS|||", text)
    count = len(re.findall(pattern, text))
    return masked_text, count


def detect_nsfw(text: str) -> tuple[str, float]:
    model = fasttext.load_model(
        f"{get_shared_assets_path()}/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
    )
    labels, scores = model.predict(text, k=1)
    label = labels[0]
    score = scores[0]
    label = label[len("__label__") :]
    return label, score


def detect_toxic_speech(text: str) -> tuple[str, float]:
    model = fasttext.load_model(
        f"{get_shared_assets_path()}/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"
    )
    labels, scores = model.predict(text, k=1)
    label = labels[0]
    score = scores[0]
    label = label[len("__label__") :]
    return label, score


def check_quality(text: str) -> bool:
    words = nltk.word_tokenize(text)
    if not _check_words_count(words):
        return False
    if not _check_mean_word_length(words):
        return False
    if not _check_words_with_alphabet(words):
        return False
    lines = text.split("\n")
    if not _check_lines_end_with_ellipsis(lines):
        return False
    return True


def _check_lines_end_with_ellipsis(lines: list[str]) -> bool:
    cnt = 0
    for line in lines:
        if line.endswith("..."):
            cnt += 1
    return cnt / len(lines) <= 0.3


def _check_words_with_alphabet(words: list[str]) -> bool:
    cnt = 0
    for word in words:
        if any(c.isalpha() for c in word):
            cnt += 1
    return cnt / len(words) >= 0.8


def _check_words_count(words: list[str]) -> bool:
    return len(words) >= 50 and len(words) <= 100000


def _check_mean_word_length(words: list[str]) -> bool:
    length = 0
    for word in words:
        length += len(word)
    avg_length = length / len(words)
    return avg_length >= 3.0 and avg_length <= 10.0
