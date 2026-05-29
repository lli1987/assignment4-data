from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

def extract_text(html_bytes: bytes):
    encoding_format = detect_encoding(html_bytes)
    decoded_str = html_bytes.decode(encoding_format)
    ret = extract_plain_text(decoded_str)
    return ret
