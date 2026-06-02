#!/usr/bin/env python3
"""Build a double-blind submission package from the paper artifact manifest.

Manifest-driven: the package contains exactly the files enumerated in
``results/paper_artifact_manifest.json`` (the paper PDF + Table evidence JSONs +
official template files) plus the manifest itself. Nothing else is included, so
identity-bearing repo files (README, runtime/publication reports) can never leak
in by accident.

Two hard gates before a ZIP is written:
  1. Integrity   -- every listed file's sha256 must match the manifest.
  2. Anonymity   -- no packaged file may contain an author-identifying string.
                    The PDF is scanned via its rendered text (pdftotext), not raw
                    bytes, so compressed content is checked too.

Usage:
    python experiments/build_submission_package.py
    python experiments/build_submission_package.py --out dist/submission/anon.zip
    python experiments/build_submission_package.py --extra-needle "my lab"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

DEFAULT_MANIFEST = "results/paper_artifact_manifest.json"
DEFAULT_OUT = "dist/submission/submission.zip"

# Case-insensitive substrings that must never appear in a double-blind package.
IDENTITY_NEEDLES = [
    "hua yan",
    "randallyan",
    "randall.yanh",
    "github.com",
    "gmail.com",
]


class PackagingError(Exception):
    """Raised when the package cannot be built safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entries(manifest: dict) -> list[dict]:
    """All file entries the package must ship (paper + evidence + templates)."""
    entries = [manifest["paper"], *manifest["evidence_files"], *manifest["template_files"]]
    seen, unique = set(), []
    for entry in entries:
        if entry["path"] not in seen:
            seen.add(entry["path"])
            unique.append(entry)
    return unique


def verify_integrity(repo: Path, entries: list[dict]) -> None:
    for entry in entries:
        path = repo / entry["path"]
        if not path.exists():
            raise PackagingError(f"missing artifact: {entry['path']}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise PackagingError(
                f"sha256 mismatch for {entry['path']}: "
                f"manifest {entry['sha256'][:12]} != file {actual[:12]}"
            )


def pdf_rendered_text(path: Path) -> str | None:
    """Return rendered PDF text via pdftotext, or None if it is unavailable."""
    try:
        return subprocess.check_output(
            ["pdftotext", str(path), "-"], text=True, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as exc:
        raise PackagingError(f"pdftotext failed on {path}: rc={exc.returncode}") from exc


def scan_for_needles(
    repo: Path,
    paths: list[str],
    identity_needles: list[str],
    extra_needles: list[str],
    allow_identity: bool,
) -> list[tuple[str, str, str]]:
    """Return (category, path, needle) hits. category is 'identity' or 'extra-needle'.

    'identity' = a built-in author-identifying string (a real double-blind leak).
    'extra-needle' = a caller-supplied sentinel via --extra-needle (e.g. for tests);
    not an identity leak, but still blocks unless --allow-identity.
    """
    catalog = [("identity", n) for n in identity_needles] + [
        ("extra-needle", n) for n in extra_needles
    ]
    hits: list[tuple[str, str, str]] = []
    for rel in paths:
        path = repo / rel
        if path.suffix.lower() == ".pdf":
            text = pdf_rendered_text(path)
            if text is None:
                msg = (
                    f"cannot verify PDF anonymity for {rel}: pdftotext not installed "
                    "(install poppler) or pass --allow-identity"
                )
                if allow_identity:
                    print(f"  WARN: {msg}")
                    continue
                raise PackagingError(msg)
            blob = text.lower()
        else:
            try:
                blob = path.read_text(errors="replace").lower()
            except OSError as exc:
                raise PackagingError(f"cannot read {rel} for scan: {exc}") from exc
        for category, needle in catalog:
            if needle in blob:
                hits.append((category, rel, needle))
    return hits


def build_zip(repo: Path, members: list[str], out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and atomically rename on success, so a failure mid-write
    # never corrupts an existing package. Deterministic: sorted members, fixed
    # timestamp, fixed mode -> reproducible bytes.
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for rel in sorted(members):
                info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, (repo / rel).read_bytes())
        tmp.replace(out_zip)  # atomic on the same filesystem
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--extra-needle",
        action="append",
        default=[],
        help="additional identity substring to scan for (repeatable)",
    )
    parser.add_argument(
        "--allow-identity",
        action="store_true",
        help="build even if identity strings are found (NOT for double-blind)",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    manifest_rel = args.manifest
    manifest_path = repo / manifest_rel
    if not manifest_path.exists():
        raise PackagingError(f"manifest not found: {manifest_rel}")
    manifest = json.loads(manifest_path.read_text())

    entries = manifest_entries(manifest)
    member_paths = sorted({e["path"] for e in entries} | {manifest_rel})

    print(f"manifest:   {manifest_rel}")
    print(f"provenance: head={manifest['repository']['head_revision_short']} "
          f"dirty={manifest['repository']['source_dirty']}")
    if manifest["repository"]["source_dirty"]:
        print("  WARN: manifest records a dirty tree; regenerate it from a clean commit.")
    print(f"artifacts:  {len(entries)} files + manifest")

    verify_integrity(repo, entries)
    print("integrity:  all sha256 match")

    extra = [n.lower() for n in args.extra_needle]
    hits = scan_for_needles(repo, member_paths, IDENTITY_NEEDLES, extra, args.allow_identity)
    if hits:
        for category, rel, needle in hits:
            label = "IDENTITY LEAK" if category == "identity" else "extra-needle hit"
            print(f"  {label}: {rel} contains '{needle}'")
        identity_hits = sum(1 for category, _, _ in hits if category == "identity")
        extra_hits = len(hits) - identity_hits
        if not args.allow_identity:
            raise PackagingError(
                f"refusing to package: {len(hits)} blocking match(es) "
                f"({identity_hits} identity, {extra_hits} extra-needle); "
                "use --allow-identity only for a non-blind build"
            )
    else:
        total_needles = len(IDENTITY_NEEDLES) + len(extra)
        print(f"anonymity:  clean ({len(member_paths)} files scanned for {total_needles} needles)")

    out_zip = repo / args.out
    build_zip(repo, member_paths, out_zip)
    zip_hash = sha256(out_zip)
    sidecar = out_zip.with_name(out_zip.name + ".sha256")
    sidecar.write_text(f"{zip_hash}  {out_zip.name}\n")
    print(f"\nwrote {args.out}  ({out_zip.stat().st_size} bytes)")
    print(f"zip sha256: {zip_hash}")
    sidecar_rel = sidecar.relative_to(repo)
    verify_cmd = f"(cd {sidecar.parent.relative_to(repo)} && shasum -a 256 -c {sidecar.name})"
    print(f"sidecar:    {sidecar_rel}  (verify: {verify_cmd})")
    for rel in sorted(member_paths):
        print(f"  + {rel}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackagingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
