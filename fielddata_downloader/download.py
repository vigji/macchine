#!/usr/bin/env python3
"""
Bulk downloader for fielddata.io FILEBOX files.

Usage:
    python3 download.py --config sections.json --dest /path/to/output

Or import and use programmatically:
    from download import FielddataDownloader
    dl = FielddataDownloader(token="eyJ...", dest="/path/to/output")
    dl.download_section("my_section", project_id=4739, dir_id=10652)

See README.md for how to obtain bearer tokens.
"""

import json
import base64
import os
import sys
import time
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.fielddata.io/fileboxapi"
DEFAULT_WORKERS = 4
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class FielddataDownloader:
    def __init__(self, token, dest, workers=DEFAULT_WORKERS):
        self.token = token
        self.dest = dest
        self.workers = workers
        os.makedirs(dest, exist_ok=True)

    def _request(self, url, timeout=60):
        """Make an authenticated GET request."""
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/json, text/plain, */*")
        return urllib.request.urlopen(req, timeout=timeout)

    def list_directories(self, project_id):
        """List all directories in a project."""
        url = f"{BASE_URL}/directory/{project_id}"
        resp = self._request(url)
        return json.loads(resp.read().decode("utf-8"))

    def list_files(self, project_id, dir_id):
        """List all files in a directory."""
        url = f"{BASE_URL}/file/{project_id}/directory/{dir_id}"
        resp = self._request(url)
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("files", [])

    def download_file(self, project_id, file_id, filename, dest_dir):
        """Download, decode from base64 blob, validate, and save one file."""
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return ("skipped", filename, "already exists")

        url = f"{BASE_URL}/file/{project_id}/{file_id}/download"

        for attempt in range(MAX_RETRIES):
            try:
                resp = self._request(url)
                raw = resp.read()

                # The API returns {"blob": "<base64>", "fileFormat": "..."}
                envelope = json.loads(raw.decode("utf-8"))
                blob_b64 = envelope.get("blob")
                if not blob_b64:
                    return ("error", filename, "no blob field in response")

                decoded = base64.b64decode(blob_b64)
                if len(decoded) == 0:
                    return ("error", filename, "decoded to empty content")

                # Validate
                if filename.endswith(".json"):
                    try:
                        json.loads(decoded)
                    except json.JSONDecodeError as e:
                        return ("error", filename, f"invalid JSON: {e}")
                elif filename.endswith(".dat"):
                    try:
                        head = decoded[:200].decode("iso-8859-1")
                        if not any(c.isalnum() for c in head[:50]):
                            return ("error", filename, f"dat looks corrupt: {head[:50]!r}")
                    except Exception as e:
                        return ("error", filename, f"dat decode error: {e}")

                with open(dest_path, "wb") as f:
                    f.write(decoded)

                return ("ok", filename, len(decoded))

            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return ("error", filename, f"HTTP {e.code}: {e.reason}")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return ("error", filename, str(e))

        return ("error", filename, "max retries exceeded")

    def download_section(self, section_name, project_id, dir_id):
        """Download all files from one directory into a named subfolder."""
        dest_dir = os.path.join(self.dest, section_name)
        os.makedirs(dest_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Section: {section_name}")
        print(f"  Project: {project_id}, Directory: {dir_id}")

        try:
            files = self.list_files(project_id, dir_id)
        except Exception as e:
            print(f"  ERROR listing files: {e}")
            return 0, 0, []

        total = len(files)
        print(f"  Files found: {total}")
        if total == 0:
            return 0, 0, []

        ok_count = 0
        skip_count = 0
        errors = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {}
            for f in files:
                fut = executor.submit(
                    self.download_file,
                    project_id, f["fileId"], f["fileName"], dest_dir
                )
                futures[fut] = f["fileName"]

            done = 0
            for fut in as_completed(futures):
                done += 1
                status, fname, detail = fut.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skipped":
                    skip_count += 1
                else:
                    errors.append((fname, detail))
                    print(f"  FAILED: {fname} - {detail}")

                if done % 50 == 0 or done == total:
                    print(f"  [{done}/{total}] ok={ok_count} skip={skip_count} err={len(errors)}")

        print(f"  DONE: {ok_count} downloaded, {skip_count} skipped, {len(errors)} errors")
        return ok_count, skip_count, errors


def main():
    parser = argparse.ArgumentParser(description="Fielddata.io FILEBOX bulk downloader")
    parser.add_argument("--config", required=True, help="Path to sections.json config file")
    parser.add_argument("--dest", required=True, help="Destination root directory")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel downloads (default: 4)")
    parser.add_argument("--section", help="Download only this section name (optional)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    token = config.get("token")
    if not token:
        print("ERROR: 'token' field missing from config. See README.md for how to get it.")
        sys.exit(1)

    sections = config.get("sections", [])
    if args.section:
        sections = [s for s in sections if s["name"] == args.section]
        if not sections:
            print(f"ERROR: section '{args.section}' not found in config")
            sys.exit(1)

    dl = FielddataDownloader(token=token, dest=args.dest, workers=args.workers)

    # Verify token works
    print("Testing token...")
    try:
        s = sections[0]
        test_files = dl.list_files(s["project_id"], s["dir_id"])
        print(f"  OK - token works ({len(test_files)} files in first section)")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Token may be expired. Get a fresh one (see README.md).")
        sys.exit(1)

    grand_ok = 0
    grand_skip = 0
    all_errors = []

    for s in sections:
        ok, skip, errors = dl.download_section(s["name"], s["project_id"], s["dir_id"])
        grand_ok += ok
        grand_skip += skip
        for fname, detail in errors:
            all_errors.append((s["name"], fname, detail))

    # Final report
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(f"Total downloaded: {grand_ok}")
    print(f"Total skipped: {grand_skip}")
    print(f"Total errors: {len(all_errors)}")
    if all_errors:
        print("\nFailed files:")
        for section, fname, detail in all_errors:
            print(f"  [{section}] {fname}: {detail}")

    total_on_disk = sum(
        len(os.listdir(os.path.join(args.dest, s["name"])))
        for s in sections
        if os.path.isdir(os.path.join(args.dest, s["name"]))
    )
    print(f"\nTotal files on disk: {total_on_disk}")


if __name__ == "__main__":
    main()
