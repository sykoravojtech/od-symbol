import hashlib


def generate_short_hash(text: str, digest_size_bytes: int = 4) -> str:
    """Generate a short hash using Blake2s which can give very short hashes"""
    h = hashlib.blake2s(text.encode("utf-8"), digest_size=digest_size_bytes)
    return h.hexdigest()
