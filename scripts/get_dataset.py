#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests

BASE_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542"

# Basic, conservative retry settings
MAX_RETRIES = 5
BACKOFF_BASE = 1.5
TIMEOUT = (10, 60)  # (connect, read) seconds
CHUNK = 1024 * 1024  # 1 MiB

HEADERS = {"User-Agent": "dataset-downloader/1.0 (+https://example.com)"}
SESSION_LIST = [1,2]

def build_url_and_path(out_dir: Path, session: int, subject: int):
    
    filename = f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    url = f"{BASE_URL}/session{session}/s{subject}/{filename}"
    local_dir = out_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / filename
    return url, dest


def head_content_length(url: str):
    try:
        r = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 404:
            return None  # doesn't exist
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception:
        return None


def download_one(url: str, dest: Path):
    # Check remote size (if available)
    remote_size = head_content_length(url)
    if remote_size is None:
        # Either 404 or HEAD failed; try GET once to confirm existence
        try:
            r = requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT)
            if r.status_code == 404:
                return f"404: {url}"
            r.raise_for_status()
            # If no Content-Length, download fresh
            remote_size = int(r.headers.get("Content-Length") or 0)
            r.close()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return f"404: {url}"
            return f"HEAD/GET error: {url} ({e})"
        except Exception as e:
            # Continue; we'll attempt download with retries
            pass

    # If file already complete, skip
    if dest.exists() and remote_size and dest.stat().st_size == remote_size:
        return f"OK (cached): {dest}"

    # Resume if partial exists and server supports Range
    resume_pos = dest.stat().st_size if dest.exists() else 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = dict(HEADERS)
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"

            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 404:
                    return f"404: {url}"
                # 206 = partial, 200 = full
                if r.status_code not in (200, 206):
                    r.raise_for_status()

                # When starting a fresh full download, write to temp then move
                mode = "ab" if resume_pos > 0 and r.status_code == 206 else "wb"
                # If server ignored Range and sent 200, reset file
                if mode == "ab" and r.status_code == 200:
                    resume_pos = 0
                    mode = "wb"

                with open(dest, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)

            # Optionally, verify size when known
            if remote_size and dest.stat().st_size != remote_size:
                raise IOError(
                    f"Incomplete: {dest.stat().st_size} of {remote_size} bytes"
                )

            return f"OK: {dest}"
        except Exception as e:
            if attempt == MAX_RETRIES:
                return f"FAIL ({attempt}/{MAX_RETRIES}): {url} -> {dest} [{e}]"
            # backoff
            sleep_s = BACKOFF_BASE ** (attempt - 1)
            time.sleep(sleep_s)
            # Recompute resume position after partial write
            resume_pos = dest.stat().st_size if dest.exists() else 0


def main():
    parser = argparse.ArgumentParser(
        description="Download EEG dataset (2 sessions x 54 subjects) with resume/retries."
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("./raw_data"),
        help="Output directory (default: ./raw_data)"
    )
    parser.add_argument(
        "--min-subj", type=int, default=1, help="First subject index (default: 1)"
    )
    parser.add_argument(
        "--max-subj", type=int, default=54, help="Last subject index (default: 54)"
    )
    parser.add_argument(
        "--sessions", type=int, nargs="*", default=[1, 2],
        help="Sessions to download (default: 1 2)"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=6, help="Concurrent downloads (default: 6)"
    )
    args = parser.parse_args()

    tasks = []
    for s in sorted(set(args.sessions)):
        if s not in SESSION_LIST:
            continue
        for subj in range(args.min_subj, args.max_subj + 1):
            url, dest = build_url_and_path(args.out, s, subj)
            tasks.append((url, dest))

    print(f"Planned files: {len(tasks)}")
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(download_one, url, dest): (url, dest) for url, dest in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(res)

    # Simple summary
    ok = sum(r.startswith("OK") for r in results)
    cached = sum(r.startswith("OK (cached)") for r in results)
    not_found = sum(r.startswith("404") for r in results)
    fails = sum(r.startswith("FAIL") for r in results)
    print("\nSummary:")
    print(f"  OK new:      {ok}")
    print(f"  OK cached:   {cached}")
    print(f"  Not found:   {not_found}")
    print(f"  Failed:      {fails}")


if __name__ == "__main__":
    main()
