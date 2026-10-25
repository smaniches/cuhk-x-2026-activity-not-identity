"""Frame count of an MP4 with NO decoding and NO third-party dependency.

Why this exists: our largest scoring mechanism (emotion pos+dur, +12 public rows) needs
only ONE number per clip - the frame count. If the live DATA_DIR ships Depth/Depth.mp4 but
no Skeleton/, we must still get it. ffprobe and cv2 are BOTH absent on this machine, so
relying on them would silently drop the mechanism at the worst possible moment.

An MP4 stores its sample count in the sample-table `stsz` (or `stz2`) atom inside
moov/trak/mdia/minf/stbl. For a video track that is exactly the frame count. We walk the
atom tree with the standard library only, reading headers - never the media payload.
"""
import struct

CONTAINERS = {b'moov', b'trak', b'mdia', b'minf', b'stbl', b'edts', b'udta'}


def _walk(f, end, depth=0, want=b'stsz', hdlr=None):
    """Yield (atom_type, payload_offset, payload_size) and recurse into containers."""
    while f.tell() < end - 8:
        start = f.tell()
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack('>I4s', hdr)
        if size == 1:                       # 64-bit extended size
            ext = f.read(8)
            if len(ext) < 8:
                return
            size = struct.unpack('>Q', ext)[0]
            body = start + 16
        elif size == 0:                     # extends to end of file
            size = end - start
            body = start + 8
        else:
            body = start + 8
        if size < 8 or start + size > end:
            return
        yield typ, body, start + size
        if typ in CONTAINERS:
            f.seek(body)
            for item in _walk(f, start + size, depth + 1, want, hdlr):
                yield item
        f.seek(start + size)


def frame_count(path, max_bytes=None):
    """-> int frames, or None if it cannot be determined. Never raises."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            end = f.tell()
            f.seek(0)
            counts, is_video = [], []
            cur_video = False
            for typ, body, atom_end in _walk(f, end):
                if typ == b'hdlr':
                    f.seek(body + 8)
                    cur_video = f.read(4) == b'vide'
                elif typ in (b'stsz', b'stz2'):
                    f.seek(body)
                    raw = f.read(12)
                    if len(raw) < 12:
                        continue
                    _ver_flags, _sample_size, count = struct.unpack('>III', raw)
                    counts.append(count)
                    is_video.append(cur_video)
            for c, v in zip(counts, is_video):
                if v and c > 0:
                    return c
            return max(counts) if counts else None
    except Exception:
        return None


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        print(p, frame_count(p))
