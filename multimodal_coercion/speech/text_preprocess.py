import re


def clean_tamil_text(text: str) -> str:
    s = text.strip()
    s = re.sub(r"[^\w\u0B80-\u0BFF\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE)
    return s.strip()

