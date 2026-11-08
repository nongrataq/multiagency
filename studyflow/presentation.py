import re


def format_plan(text):
    text = text.strip()
    day = re.compile(r'(?i)(?<!\w)день\s+(\d+)\s*[:.)—-]?\s*')
    matches = list(day.finditer(text))
    if not matches:
        return text
    pieces = []
    if text[:matches[0].start()].strip():
        pieces.append(text[:matches[0].start()].strip())
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip(' \n;')
        pieces.append(f"### День {match.group(1)}\n\n{body}" if body else f"### День {match.group(1)}")
    return '\n\n'.join(pieces)


def display_topic(text):
    return re.sub(r'^\s*\d+[.)]\s*', '', text).strip()
