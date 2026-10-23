"""Download a video from a link (direct file, YouTube/Vimeo, or a public Google Drive file)."""
import ipaddress
import os
import socket
from urllib.parse import urlparse


def _check(url):
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("Please paste a full link starting with http:// or https://")
    try:
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ValueError("This link points to a private address and is not allowed.")
    except socket.gaierror:
        raise ValueError("Could not find this link. Check the address.")


def download(url, folder, max_mb=1500):
    _check(url)
    import yt_dlp
    opts = {"outtmpl": os.path.join(folder, "video.%(ext)s"), "format": "best[height<=720]/best",
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "max_filesize": max_mb * 1024 * 1024, "socket_timeout": 30}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise ValueError("Could not download this link. It must point to ONE video file (not a folder) "
                         f"and be public or 'anyone with the link'. Details: {str(e)[:150]}")
    files = [f for f in os.listdir(folder) if f.startswith("video.")]
    if not files:
        raise ValueError("Download produced no video file (it may be larger than the allowed size).")
    return os.path.join(folder, files[0])
