#!/usr/bin/env python3
"""
List all directories and files in a fielddata.io FILEBOX project.

Usage:
    python3 list_project.py --token "eyJ..." --project 4739
    python3 list_project.py --token "eyJ..." --project 4739 --dir 10651
"""

import json
import argparse
import urllib.request


BASE_URL = "https://www.fielddata.io/fileboxapi"


def request(url, token):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json, text/plain, */*")
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read().decode("utf-8"))


def list_directories(project_id, token):
    """List all directories in a project (flat list, use parentId to build tree)."""
    return request(f"{BASE_URL}/directory/{project_id}", token)


def list_files(project_id, dir_id, token):
    """List all files in a specific directory."""
    data = request(f"{BASE_URL}/file/{project_id}/directory/{dir_id}", token)
    return data.get("files", [])


def main():
    parser = argparse.ArgumentParser(description="List fielddata.io project structure")
    parser.add_argument("--token", required=True, help="Bearer token")
    parser.add_argument("--project", type=int, required=True, help="Project ID")
    parser.add_argument("--dir", type=int, help="Directory ID (list files in this dir)")
    args = parser.parse_args()

    if args.dir:
        files = list_files(args.project, args.dir, args.token)
        print(f"Files in project {args.project}, directory {args.dir}: {len(files)}")
        print()
        total_size = 0
        for f in files:
            size = f.get("fileSize", 0)
            total_size += size
            print(f"  {f['fileId']:>10}  {size:>10,} bytes  {f['fileName']}")
        print(f"\nTotal: {len(files)} files, {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    else:
        dirs = list_directories(args.project, args.token)
        print(f"Directories in project {args.project}: {len(dirs)}")
        print()

        # Build tree
        by_parent = {}
        for d in dirs:
            pid = d.get("parentId", 0)
            by_parent.setdefault(pid, []).append(d)

        def print_tree(parent_id, indent=0):
            for d in sorted(by_parent.get(parent_id, []), key=lambda x: x["dirName"]):
                prefix = "  " * indent
                print(f"{prefix}{d['dirId']:>6}  {d['dirName']}")
                print_tree(d["dirId"], indent + 1)

        print_tree(0)


if __name__ == "__main__":
    main()
