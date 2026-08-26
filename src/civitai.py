#!/usr/bin/env python3
"""Cliente Civitai do foto-macos, com credencial no Keychain do macOS.

O token nunca vai para argumentos, logs, configs de agentes ou Git. Todos os
clientes chamam este modulo por meio do MCP ``foto-macos``; este processo le a
credencial somente no momento da requisicao.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request

API = "https://civitai.com/api/v1"
KEYCHAIN_SERVICE = "civitai-api"
DEFAULT_DIR = os.path.expanduser(
    "~/Library/Application Support/foto-macos/civitai")


def token() -> str:
    result = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-s", KEYCHAIN_SERVICE,
         "-a", getpass.getuser(), "-w"],
        capture_output=True, text=True,
    )
    value = result.stdout.strip()
    if result.returncode or not value:
        raise RuntimeError(
            "token Civitai ausente no Keychain (service=civitai-api)"
        )
    return value


def request_json(path: str) -> dict:
    request = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {token()}",
                 "User-Agent": "foto-macos/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def parse_reference(value: str) -> tuple[int | None, int | None]:
    """Retorna (model_id, version_id) a partir de URL ou numero."""
    value = value.strip()
    if value.isdigit():
        return None, int(value)
    parsed = urllib.parse.urlparse(value)
    match = re.search(r"/models/(\d+)", parsed.path)
    params = urllib.parse.parse_qs(parsed.query)
    version = params.get("modelVersionId", [None])[0]
    return (int(match.group(1)) if match else None,
            int(version) if version and version.isdigit() else None)


def resolve(value: str) -> dict:
    model_id, version_id = parse_reference(value)
    if version_id:
        return request_json(f"/model-versions/{version_id}")
    if model_id:
        model = request_json(f"/models/{model_id}")
        versions = model.get("modelVersions") or []
        if not versions:
            raise RuntimeError("modelo sem versoes para download")
        return versions[0]
    raise ValueError("use uma URL Civitai com /models/ID ou um version ID")


def summary(version: dict) -> dict:
    model = version.get("model") or {}
    files = []
    for item in version.get("files") or []:
        hashes = item.get("hashes") or {}
        files.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "size_kb": item.get("sizeKB"),
            "type": item.get("type"),
            "primary": bool(item.get("primary")),
            "sha256": hashes.get("SHA256"),
        })
    return {
        "model": model.get("name"),
        "model_id": model.get("id"),
        "type": model.get("type"),
        "version": version.get("name"),
        "version_id": version.get("id"),
        "base_model": version.get("baseModel"),
        "trigger_words": version.get("trainedWords") or [],
        "files": files,
        "url": f"https://civitai.com/models/{model.get('id')}?modelVersionId={version.get('id')}",
    }


def choose_file(version: dict, file_id: int | None) -> dict:
    files = version.get("files") or []
    if file_id is not None:
        for item in files:
            if int(item.get("id", -1)) == file_id:
                return item
        raise RuntimeError(f"file id {file_id} nao pertence a esta versao")
    for item in files:
        if item.get("primary"):
            return item
    if not files:
        raise RuntimeError("versao sem arquivos")
    return files[0]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(reference: str, destination: str = "", file_id: int | None = None) -> str:
    version = resolve(reference)
    item = choose_file(version, file_id)
    target_dir = os.path.abspath(os.path.expanduser(destination or DEFAULT_DIR))
    if os.path.splitext(target_dir)[1]:
        target = target_dir
    else:
        target = os.path.join(target_dir, item["name"])
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".partial"
    url = item.get("downloadUrl") or (
        f"https://civitai.com/api/download/models/{version['id']}"
        f"?type=Model&format=SafeTensor")
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token()}",
                 "User-Agent": "foto-macos/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    actual = sha256(partial)
    expected = ((item.get("hashes") or {}).get("SHA256") or "").lower()
    if expected and actual != expected:
        os.unlink(partial)
        raise RuntimeError(f"SHA-256 incorreto: esperado {expected}, obtido {actual}")
    os.replace(partial, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info")
    info.add_argument("reference")
    fetch = sub.add_parser("baixar")
    fetch.add_argument("reference")
    fetch.add_argument("--destino", default="")
    fetch.add_argument("--arquivo", type=int, default=None)
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        if args.command == "status":
            token()
            print("Civitai: credencial no Keychain OK")
        elif args.command == "info":
            print(json.dumps(summary(resolve(args.reference)), ensure_ascii=False, indent=2))
        else:
            print(download(args.reference, args.destino, args.arquivo))
        return 0
    except Exception as exc:
        print(f"erro Civitai: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
