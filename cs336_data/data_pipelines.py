from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from cs336_data.common import get_shared_assets_path
import fasttext


def extract_text(html_bytes: bytes) -> str:
    encoding_format = detect_encoding(html_bytes)
    decoded_str = html_bytes.decode(encoding_format)
    ret = extract_plain_text(decoded_str)
    return ret


def identify_language(text: str) -> tuple[str, float]:
    pred_model = fasttext.load_model(f"{get_shared_assets_path()}/classifiers/lid.176.bin")
    label, score = pred_model.predict(text, k=1)
    lang = label[len("__label__") :]
    return lang, score
