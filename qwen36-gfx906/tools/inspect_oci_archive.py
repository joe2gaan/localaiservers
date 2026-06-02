#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def member_meta(member: tarfile.TarInfo) -> str:
    parts = [
        member.name,
        str(member.type),
        str(member.mode),
        str(member.uid),
        str(member.gid),
        member.uname or "",
        member.gname or "",
        str(member.size),
        str(member.mtime),
        member.linkname or "",
        str(member.devmajor),
        str(member.devminor),
    ]
    return "\0".join(parts)


def summarize_layer(
    outer: tarfile.TarFile,
    layer_name: str,
    file_manifest_path: Path | None = None,
) -> dict:
    ordered_meta = hashlib.sha256()
    order_hash = hashlib.sha256()
    sorted_meta = []
    sorted_file_content = []
    names = []
    counts = {
        "entries": 0,
        "files": 0,
        "dirs": 0,
        "symlinks": 0,
        "hardlinks": 0,
        "other": 0,
    }

    with outer.extractfile(layer_name) as layer_file:
        with tarfile.open(fileobj=layer_file, mode="r|*") as layer:
            for member in layer:
                counts["entries"] += 1
                if member.isfile():
                    counts["files"] += 1
                elif member.isdir():
                    counts["dirs"] += 1
                elif member.issym():
                    counts["symlinks"] += 1
                elif member.islnk():
                    counts["hardlinks"] += 1
                else:
                    counts["other"] += 1

                meta = member_meta(member)
                ordered_meta.update(meta.encode("utf-8", "surrogateescape"))
                ordered_meta.update(b"\n")
                order_hash.update(member.name.encode("utf-8", "surrogateescape"))
                order_hash.update(b"\0")
                sorted_meta.append(meta)
                names.append(member.name)

                if file_manifest_path is not None:
                    content_hash = "-"
                    if member.isfile():
                        file_hash = hashlib.sha256()
                        extracted = layer.extractfile(member)
                        if extracted is None:
                            raise RuntimeError(
                                f"regular file had no extractable content: {member.name}"
                            )
                        for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                            file_hash.update(chunk)
                        content_hash = file_hash.hexdigest()
                    sorted_file_content.append(
                        "\t".join([
                            member.name,
                            str(member.type),
                            str(member.size),
                            content_hash,
                            str(member.mode),
                            str(member.uid),
                            str(member.gid),
                            member.linkname or "",
                        ])
                    )

    sorted_meta_hash = hashlib.sha256()
    for meta in sorted(sorted_meta):
        sorted_meta_hash.update(meta.encode("utf-8", "surrogateescape"))
        sorted_meta_hash.update(b"\n")

    result = {
        "layer": layer_name,
        "counts": counts,
        "ordered_meta_sha256": ordered_meta.hexdigest(),
        "sorted_meta_sha256": sorted_meta_hash.hexdigest(),
        "order_sha256": order_hash.hexdigest(),
        "first_names": names[:25],
        "last_names": names[-25:],
    }

    if file_manifest_path is not None:
        sorted_file_content.sort()
        content_manifest_hash = hashlib.sha256()
        file_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with file_manifest_path.open("w", encoding="utf-8", newline="\n") as out:
            for line in sorted_file_content:
                out.write(line)
                out.write("\n")
                content_manifest_hash.update(line.encode("utf-8", "surrogateescape"))
                content_manifest_hash.update(b"\n")
        result["file_manifest_path"] = str(file_manifest_path)
        result["file_manifest_sha256"] = content_manifest_hash.hexdigest()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--file-manifest")
    args = parser.parse_args()

    with tarfile.open(args.archive, mode="r") as outer:
        manifest = json.load(outer.extractfile("manifest.json"))
        config_name = manifest[0]["Config"]
        config = json.load(outer.extractfile(config_name))
        docker_manifest_name = None
        for name in outer.getnames():
            if name.startswith("blobs/sha256/") and name != config_name:
                try:
                    candidate = json.load(outer.extractfile(name))
                except Exception:
                    continue
                if isinstance(candidate, dict) and candidate.get("schemaVersion") == 2:
                    docker_manifest_name = name
                    docker_manifest = candidate
                    break
        else:
            docker_manifest = None

        print(json.dumps({
            "archive": args.archive,
            "config": config_name,
            "rootfs": config.get("rootfs"),
            "manifest": manifest,
            "docker_manifest": docker_manifest,
            "docker_manifest_blob": docker_manifest_name,
            "layers": [
                summarize_layer(
                    outer,
                    layer_name,
                    Path(args.file_manifest)
                    if args.file_manifest and index == 0
                    else None,
                )
                for index, layer_name in enumerate(manifest[0]["Layers"])
            ],
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
