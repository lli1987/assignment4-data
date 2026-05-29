import gzip
from warcio.archiveiterator import ArchiveIterator
from cs336_data.data_pipelines import extract_text

# Reading WARC (HTML content)
with gzip.open("local-shared-data/CC/example.warc.gz", "rb") as f:
    for record in ArchiveIterator(f):
        if record.rec_type == "response":  # HTTP response with HTML
            payload_bytes = record.content_stream().read()
            print(extract_text(payload_bytes))
            break

print("--------------- wet file content below ---------------")
# Reading WET (already extracted text)  
with gzip.open("local-shared-data/CC/example.warc.wet.gz", "rb") as f:
    for record in ArchiveIterator(f):
        if record.rec_type == "conversion":  # Extracted text
            text = record.content_stream().read().decode("utf-8")
            print(text)
            break