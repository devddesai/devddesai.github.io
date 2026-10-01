#!/usr/bin/env python3
"""Fill in captions for new photos in content/photos/ and strip their GPS data.

For every image that has no entry in content/photos/_index.md yet:
  - reads the capture date from EXIF
  - reads the GPS coordinates and looks up "City, Country" (OpenStreetMap)
  - appends a caption entry to _index.md
Existing entries are never changed, so manual edits are safe.

GPS data is then removed from every JPEG in the folder (losslessly: the image
itself is not re-encoded), so coordinates never get committed.

Usage:  python3 scripts/add_photos.py
Runs automatically from the pre-commit hook in .githooks/.
Uses only the Python standard library.
"""

import json
import re
import struct
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "content" / "photos"
INDEX = PHOTOS / "_index.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff"}
JPEG_EXTS = {".jpg", ".jpeg"}
USER_AGENT = "devddesai.github.io-photo-captions/1.0 (+https://github.com/devddesai/devddesai.github.io)"

TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}
TAG_EXIF_IFD, TAG_GPS_IFD = 0x8769, 0x8825
TAG_DATETIME, TAG_DATETIME_ORIGINAL = 0x0132, 0x9003
XMP_HEADERS = (b"http://ns.adobe.com/xap/1.0/\x00", b"http://ns.adobe.com/xmp/extension/\x00")


# ─── JPEG / EXIF parsing ───

def jpeg_segments(data):
    """Yield (marker, start, end) for each segment before the image data."""
    if data[:2] != b"\xff\xd8":
        return
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: image data follows
            return
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        yield marker, i, i + 2 + length
        i += 2 + length


class Tiff:
    def __init__(self, data, base):
        self.data, self.base = data, base
        self.end = ">" if data[base:base + 2] == b"MM" else "<"

    def u16(self, off):
        return struct.unpack(self.end + "H", self.data[self.base + off:self.base + off + 2])[0]

    def u32(self, off):
        return struct.unpack(self.end + "I", self.data[self.base + off:self.base + off + 4])[0]

    def entries(self, ifd):
        """Yield (entry_offset, tag, type, count, value_offset) for an IFD."""
        for n in range(self.u16(ifd)):
            e = ifd + 2 + n * 12
            tag, typ, count = self.u16(e), self.u16(e + 2), self.u32(e + 4)
            size = TYPE_SIZES.get(typ, 1) * count
            yield e, tag, typ, count, (e + 8 if size <= 4 else self.u32(e + 8)), size

    def ascii(self, off, count):
        raw = self.data[self.base + off:self.base + off + count]
        return raw.split(b"\x00")[0].decode("ascii", "ignore").strip()

    def rationals(self, off, count):
        vals = []
        for k in range(count):
            num, den = self.u32(off + 8 * k), self.u32(off + 8 * k + 4)
            vals.append(num / den if den else 0.0)
        return vals


def find_exif(data):
    for marker, start, end in jpeg_segments(data):
        if marker == 0xE1 and data[start + 4:start + 10] == b"Exif\x00\x00":
            return Tiff(data, start + 10)
    return None


def read_metadata(data):
    """Return (date "YYYY-MM-DDTHH:MM:SS" or "", (lat, lon) or None)."""
    tiff = find_exif(data)
    if not tiff:
        return "", None
    ifd0 = tiff.u32(4)
    date, exif_ifd, gps_ifd = "", None, None
    for _, tag, typ, count, voff, _ in tiff.entries(ifd0):
        if tag == TAG_DATETIME and typ == 2:
            date = tiff.ascii(voff, count)
        elif tag == TAG_EXIF_IFD:
            exif_ifd = tiff.u32(voff)
        elif tag == TAG_GPS_IFD:
            gps_ifd = tiff.u32(voff)
    if exif_ifd:
        for _, tag, typ, count, voff, _ in tiff.entries(exif_ifd):
            if tag == TAG_DATETIME_ORIGINAL and typ == 2:
                date = tiff.ascii(voff, count)

    m = re.match(r"(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})", date)
    date = f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6]}" if m else ""

    coords = None
    if gps_ifd:
        g = {}
        for _, tag, typ, count, voff, _ in tiff.entries(gps_ifd):
            if tag in (1, 3) and typ == 2:
                g[tag] = tiff.ascii(voff, count)
            elif tag in (2, 4) and typ == 5 and count == 3:
                d, mnt, s = tiff.rationals(voff, 3)
                g[tag] = d + mnt / 60 + s / 3600
        if 2 in g and 4 in g:
            lat = -g[2] if g.get(1) == "S" else g[2]
            lon = -g[4] if g.get(3) == "W" else g[4]
            if (lat, lon) != (0.0, 0.0):
                coords = (lat, lon)
    return date, coords


def strip_gps(data):
    """Remove GPS from EXIF (in place, same length) and drop XMP blocks that contain GPS.
    Returns the new bytes, or None if there was nothing to remove."""
    data = bytearray(data)
    changed = False

    tiff = find_exif(data)
    if tiff:
        for _, tag, _, _, voff, _ in tiff.entries(tiff.u32(4)):
            if tag == TAG_GPS_IFD:
                gps = tiff.u32(voff)
                n = tiff.u16(gps)
                if n:
                    for e, _, _, _, v, size in list(tiff.entries(gps)):
                        start = tiff.base + v
                        data[start:start + size] = bytes(size)
                    # Empty the IFD: zero entries, entry count and next-IFD pointer
                    start = tiff.base + gps
                    data[start:start + 2 + n * 12 + 4] = bytes(2 + n * 12 + 4)
                    changed = True

    for marker, start, end in reversed(list(jpeg_segments(data))):
        seg = data[start + 4:end]
        if marker == 0xE1 and seg.startswith(XMP_HEADERS) and b"GPS" in seg:
            del data[start:end]
            changed = True

    return bytes(data) if changed else None


# ─── Location lookup ───

def lookup_place(lat, lon):
    """Return "City, Country" (or "City, ST" in the US), or "" if the lookup fails."""
    query = urllib.parse.urlencode({
        "format": "jsonv2", "lat": f"{lat:.5f}", "lon": f"{lon:.5f}",
        "zoom": 10, "addressdetails": 1, "accept-language": "en",
    })
    req = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/reverse?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            addr = json.load(resp).get("address", {})
    except Exception as err:
        print(f"    location lookup failed ({err}); leaving it blank", file=sys.stderr)
        return ""
    finally:
        time.sleep(1)  # OpenStreetMap's usage policy: max 1 request per second

    place = next((addr[k] for k in ("city", "town", "village", "hamlet", "municipality", "county") if k in addr), "")
    if addr.get("country_code") == "us":
        region = addr.get("ISO3166-2-lvl4", "").removeprefix("US-") or addr.get("state", "")
    else:
        region = addr.get("country", "")
    return ", ".join(p for p in (place, region) if p)


# ─── _index.md ───

def existing_sources(text):
    # Ignores trailing "# comments" after the file name
    return {m.strip().strip("\"'") for m in re.findall(r"^\s*-\s*src:\s*([^#\n]+?)\s*(?:#.*)?$", text, re.M)}


def add_entries(text, entries):
    """Append resource entries to the front matter, before its closing ---."""
    m = re.match(r"(---\n.*?\n)(---\s*\n?)(.*)", text, re.S)
    if not m:
        raise SystemExit(f"Could not find front matter in {INDEX}")
    front, close, body = m.groups()
    if not re.search(r"^resources:\s*$", front, re.M):
        front += "resources:\n"
    for src, location, date in entries:
        loc = location.replace('"', '\\"')
        front += f'  - src: {src}\n    params:\n      location: "{loc}"\n      date: "{date}"\n'
    return front + close + body


def main():
    if not INDEX.exists():
        raise SystemExit(f"Missing {INDEX}")
    text = INDEX.read_text()
    known = existing_sources(text)
    # Natural sort, so 2.jpg comes before 10.jpg
    natural = lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]
    photos = sorted((p for p in PHOTOS.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural)

    new_entries = []
    for photo in photos:
        if photo.name in known:
            continue
        date, coords, location = "", None, ""
        if photo.suffix.lower() in JPEG_EXTS:
            date, coords = read_metadata(photo.read_bytes())
            if coords:
                location = lookup_place(*coords)
        new_entries.append((photo.name, location, date))
        caption = " · ".join(p for p in (location, date[:10]) if p) or "no date or location found"
        print(f"  added {photo.name}: {caption}")

    if new_entries:
        INDEX.write_text(add_entries(text, new_entries))

    stripped = 0
    for photo in photos:
        if photo.suffix.lower() in JPEG_EXTS:
            cleaned = strip_gps(photo.read_bytes())
            if cleaned is not None:
                photo.write_bytes(cleaned)
                stripped += 1
                print(f"  removed GPS from {photo.name}")

    if not new_entries and not stripped:
        print("  no new photos")


if __name__ == "__main__":
    main()
