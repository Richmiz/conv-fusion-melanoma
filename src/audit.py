"""Prepare the lesion-level split from the official HAM10000 and BCN20000 releases."""

import argparse
import csv
import hashlib
import io
import json
import platform
import random
import shutil
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


@dataclass(frozen=True)
class ReleaseFile:
    name: str
    url: str
    size: int
    md5: str


HARVARD = "https://dataverse.harvard.edu/api/access/datafile/"
FIGSHARE = "https://ndownloader.figshare.com/files/"
METADATA = (
    ReleaseFile("HAM10000_metadata.csv", HARVARD + "4338392?format=original", 690218, "8f85fb1aa29d80a2797247e434deb79d"),
    ReleaseFile("bcn_20k_train.csv", FIGSHARE + "49291684", 945681, "5fd3522a69512dcf5b3c225dc1664ec6"),
)
ARCHIVES = {
    "HAM": (
        ReleaseFile("HAM10000_images_part_1.zip", HARVARD + "3172585", 1366522108, "4639bfa73ab251610530a97c898e6e46"),
        ReleaseFile("HAM10000_images_part_2.zip", HARVARD + "3172584", 1403566547, "da43d6cc50f6613013be07e8986b384b"),
    ),
    "BCN": (
        ReleaseFile("BCN_20k_train.zip", FIGSHARE + "45833178", 6368901290, "2293c9c654a3f76d6506a57ad17cddd1"),
    ),
}

# Abhishek et al., Scientific Data (2025), Figure 5. These pairs were missed by HAM lesion IDs.
HAM_DUPLICATES = (
    (33483, 33421), (33556, 33263), (29152, 32075), (29036, 27027),
    (24988, 23529), (27689, 28198), (30904, 24879), (23728, 27379),
    (25226, 33469), (26087, 25664), (24633, 24774), (26097, 32912),
    (28202, 29384), (30576, 31459), (29265, 29979), (27523, 29967),
    (24602, 29248), (24513, 24968),
)


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def valid_file(path, spec):
    return path.is_file() and path.stat().st_size == spec.size and digest(path, "md5") == spec.md5


_VERIFIED_ARCHIVES = {}


def download(folder, spec):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / spec.name
    if target.exists():
        signature = (target.stat().st_size, target.stat().st_mtime_ns, spec.md5)
        if spec.size > 1_000_000 and _VERIFIED_ARCHIVES.get(target) == signature:
            return target
        if valid_file(target, spec):
            if spec.size > 1_000_000:
                _VERIFIED_ARCHIVES[target] = signature
            return target
        raise RuntimeError(f"Publisher checksum failed for {target}. Move the invalid file aside and retry.")

    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= spec.size:
            break
        separator = "&" if "?" in spec.url else "?"
        # A fresh redirect avoids reusing expired signed archive URLs.
        url = spec.url + separator + "pilot=" + uuid.uuid4().hex
        print(f"Getting {spec.name} from byte {offset:,} of {spec.size:,}", flush=True)
        try:
            subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "30", "--speed-time", "60", "--speed-limit", "1",
                "--max-filesize", str(spec.size), "--continue-at", "-", "--output", str(partial), url,
            ], check=True)
        except subprocess.CalledProcessError as error:
            if attempt == 3:
                raise RuntimeError(f"Could not complete {spec.name}. Rerun to resume the partial download.") from error
            time.sleep(2 ** attempt)
    if not valid_file(partial, spec):
        raise RuntimeError(f"Published size or MD5 did not match for {partial}. A complete invalid partial must be moved aside before retrying.")
    partial.replace(target)
    if spec.size > 1_000_000:
        _VERIFIED_ARCHIVES[target] = (target.stat().st_size, target.stat().st_mtime_ns, spec.md5)
    return target


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_table(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def eligible(root):
    rows = []
    specifications = (
        ("HAM", "image_id", "dx", {"mel": 1, "nv": 0}),
        ("BCN", "bcn_filename", "diagnosis", {"MEL": 1, "NV": 0}),
    )
    for spec, (source, id_field, diagnosis_field, labels) in zip(METADATA, specifications, strict=True):
        path = download(root / "metadata", spec)
        for row in table(path):
            if row[diagnosis_field] not in labels:
                continue
            image_id = PurePosixPath(row[id_field].strip()).stem
            lesion_id = row["lesion_id"].strip()
            if not image_id or not lesion_id:
                raise ValueError(f"Missing image or lesion identifier in {spec.name}")
            rows.append({"source": source, "image_id": image_id, "lesion_id": lesion_id,
                         "label": labels[row[diagnosis_field]]})
    pairs = [(row["source"], row["image_id"]) for row in rows]
    if len(set(pairs)) != len(rows):
        raise ValueError("Duplicate image IDs in eligible metadata")
    return sorted(rows, key=lambda row: (row["source"], row["image_id"]))


def counts(rows):
    result = {}
    for source in sorted({row["source"] for row in rows}):
        result[source] = {}
        for label, name in ((0, "nevus"), (1, "melanoma")):
            selected = [row for row in rows if row["source"] == source and int(row["label"]) == label]
            result[source][name] = {"images": len(selected), "lesion_ids": len({row["lesion_id"] for row in selected})}
    return result


def image_hashes(stream, *, near_duplicates=True):
    from PIL import Image, ImageOps

    with Image.open(stream) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    sha = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB\0".encode())
    sha.update(rgb.tobytes())
    return sha.hexdigest(), perceptual_hashes(rgb) if near_duplicates else ""


def perceptual_hashes(rgb):
    import numpy as np
    from PIL import Image, ImageOps

    full = np.asarray(ImageOps.grayscale(rgb.resize((32, 32), Image.Resampling.LANCZOS)), dtype=np.float32)
    x = round(rgb.width * 0.08)
    y = round(rgb.height * 0.08)
    crop = rgb.crop((x, y, rgb.width - x, rgb.height - y))
    center = np.asarray(ImageOps.grayscale(crop.resize((32, 32), Image.Resampling.LANCZOS)), dtype=np.float32)
    # The eight low-frequency rows of an orthonormal 32-pixel DCT.
    dct = np.cos((np.pi / 32) * np.arange(8)[:, None] * (np.arange(32) + 0.5)).astype(np.float32) / 4
    dct[0] /= np.sqrt(2)
    hashes = []
    for view in (full, center):
        for rotated in (view, np.rot90(view), np.rot90(view, 2), np.rot90(view, 3)):
            for oriented in (rotated, np.fliplr(rotated)):
                frequency = (dct @ oriented @ dct.T).ravel()
                mask = frequency > np.median(frequency[1:])
                mask[0] = False
                hashes.append(np.packbits(mask).tobytes().hex())
    return ":".join(hashes)


def fingerprint(root, rows, source, output, *, near_duplicates=True):
    from PIL import UnidentifiedImageError

    remaining = {row["image_id"]: row for row in rows if row["source"] == source}
    discovered = set()
    found = []
    for spec in ARCHIVES[source]:
        path = download(root / "archives", spec)
        print(f"Fingerprinting eligible images in {spec.name}", flush=True)
        with ZipFile(path) as archive:
            for member in archive.namelist():
                name = PurePosixPath(member)
                if name.suffix.lower() not in {".jpg", ".jpeg", ".png"} or "__MACOSX" in name.parts:
                    continue
                if name.stem in discovered:
                    raise ValueError(f"Multiple archive members for {source}:{name.stem}")
                row = remaining.pop(name.stem, None)
                if row is None:
                    continue
                discovered.add(name.stem)
                result = dict(row, archive=spec.name, member=member, pixel_sha256="", phash="", error="")
                try:
                    with archive.open(member) as stream:
                        result["pixel_sha256"], result["phash"] = image_hashes(
                            io.BytesIO(stream.read()), near_duplicates=near_duplicates,
                        )
                except (OSError, UnidentifiedImageError) as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                found.append(result)
    if remaining:
        raise ValueError(f"{source} images missing from the published archive: {sorted(remaining)[:10]} ({len(remaining)})")
    columns = ["source", "image_id", "lesion_id", "label", "archive", "member", "pixel_sha256", "phash", "error"]
    write_table(output, columns, sorted(found, key=lambda row: row["image_id"]))


def image_key(row):
    return f"{row['source']}:{row['image_id']}"


def candidate_pairs(rows, focus_keys=None):
    import numpy as np

    # Scan a fixed pHash radius across both collections and within HAM10000.
    # Rotated, mirrored and centre-cropped variants catch some reframed copies.
    # This search cannot establish patient identity or rule out all alternate views.
    valid = [row for row in rows if row["phash"]]
    if not valid:
        return []
    values = np.array([[int(value, 16) for value in row["phash"].split(":")] for row in valid], dtype=np.uint64)
    right = values.T.copy()[:, None, :]
    anchors = [i for i, row in enumerate(valid) if focus_keys is None or image_key(row) in focus_keys]
    anchor_set = set(anchors)
    results = []
    for start in range(0, len(anchors), 48):
        batch = anchors[start:start + 48]
        best = np.full((len(batch), len(valid)), 64, dtype=np.uint8)
        for variant in (0, 8):
            probe = values[batch, variant][None, :, None]
            min_distance = np.bitwise_count(probe ^ right).min(axis=0)
            np.minimum(best, min_distance, out=best)
        if focus_keys is not None:
            # Preserve the original comparison direction for rows preceding an anchor.
            reverse = np.full_like(best, 64)
            anchors_by_variant = values[batch].T[:, :, None]
            for variant in (0, 8):
                distance = np.bitwise_count(anchors_by_variant ^ values[:, variant][None, None, :]).min(axis=0)
                np.minimum(reverse, distance, out=reverse)
            before_anchor = np.arange(len(valid))[None, :] < np.asarray(batch)[:, None]
            np.copyto(best, reverse, where=before_anchor)
        for i_relative, j in zip(*np.where(best <= 7), strict=True):
            i = batch[i_relative]
            a, b = valid[i], valid[j]
            if (i >= j and j in anchor_set) or (a["source"] == b["source"] and a["lesion_id"] == b["lesion_id"]):
                continue
            if a["pixel_sha256"] == b["pixel_sha256"]:
                continue  # Exact copies are joined automatically.
            left, right_index = sorted((i, j))
            results.append({"left": image_key(valid[left]), "right": image_key(valid[right_index]),
                            "distance": int(best[i_relative, j]), "decision": ""})
    return sorted(results, key=lambda row: (row["distance"], row["left"], row["right"]))


CANDIDATE_COLUMNS = ["left", "right", "distance", "decision"]


def scan(root, out, publish=None, restore=None, *, near_duplicates=True):
    from PIL import __version__ as pillow_version

    rows = eligible(root)
    release = {item.name: asdict(item) for item in (*METADATA, *ARCHIVES["HAM"], *ARCHIVES["BCN"])}
    software = {"python": platform.python_version(), "pillow": pillow_version}
    if near_duplicates:
        import numpy as np

        if not hasattr(np, "bitwise_count"):
            raise RuntimeError("The optional near-duplicate scan requires NumPy 2.x")
        software["numpy"] = np.__version__
    config = {"software": software, "near_duplicate_search": near_duplicates,
              "implementation_sha256": digest(Path(__file__)), "release_files": release}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    out = out / key
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "scan_config.json", config)
    for source in ARCHIVES:
        output = out / f"{source}_fingerprints.csv"
        receipt = output.with_suffix(".sha256")
        if restore:
            restore(output)
            restore(receipt)
        if not (output.is_file() and receipt.is_file() and receipt.read_text().strip() == digest(output)):
            archives = root / "archives"
            archives.mkdir(parents=True, exist_ok=True)
            remaining = 0
            for item in ARCHIVES[source]:
                if not (archives / item.name).is_file():
                    partial = archives / (item.name + ".part")
                    remaining += max(0, item.size - (partial.stat().st_size if partial.is_file() else 0))
            if shutil.disk_usage(archives).free < remaining:
                raise RuntimeError(f"Insufficient runtime disk space for the {source} archives")
            fingerprint(root, rows, source, output, near_duplicates=near_duplicates)
            receipt.write_text(digest(output) + "\n", encoding="ascii")
        if publish:
            publish(output)
            publish(receipt)
    fingerprints = [row for source in ARCHIVES for row in table(out / f"{source}_fingerprints.csv")]
    candidates = candidate_pairs(fingerprints) if near_duplicates else []
    write_table(out / "candidates.csv", CANDIDATE_COLUMNS, candidates)
    undecodable = [image_key(row) for row in fingerprints if row["error"]]
    summary = {"eligible_before_audit": counts(rows), "near_duplicate_search": near_duplicates,
               "near_duplicate_candidates": len(candidates) if near_duplicates else None,
               "undecodable": undecodable,
               "sha256": {name: digest(out / name) for name in ("HAM_fingerprints.csv", "BCN_fingerprints.csv", "candidates.csv", "scan_config.json")}}
    write_json(out / "scan_receipt.json", summary)
    if publish:
        for name in ("scan_config.json", "candidates.csv", "scan_receipt.json"):
            publish(out / name)
    status = {"audit_directory": str(out), "undecodable": len(undecodable)}
    if near_duplicates:
        status["near_duplicate_candidates"] = len(candidates)
    print(json.dumps(status, indent=2))
    return out


def freeze(audit_dir, review=None):
    scan_receipt = audit_dir / "scan_receipt.json"
    receipt = json.loads(scan_receipt.read_text(encoding="utf-8"))
    for name, sha in receipt["sha256"].items():
        if digest(audit_dir / name) != sha:
            raise ValueError(f"Audit file changed after scanning: {name}")
    rows = [row for source in ARCHIVES for row in table(audit_dir / f"{source}_fingerprints.csv")]
    expected = table(audit_dir / "candidates.csv")
    decisions = table(review) if review else []
    if [tuple(row[field] for field in CANDIDATE_COLUMNS[:3]) for row in expected] != [tuple(row[field] for field in CANDIDATE_COLUMNS[:3]) for row in decisions]:
        raise ValueError("Review must cover exactly the candidate rows from this scan, in the same order")
    if any(row["decision"] not in {"same", "different", "uncertain"} for row in decisions):
        raise ValueError("Enter same, different, or uncertain for each candidate")

    records = {image_key(row): row for row in rows}
    parents = {key: key for key in records}

    def group(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(a, b):
        left, right = sorted((group(a), group(b)))
        parents[right] = left

    by_lesion = {}
    by_pixel = {}
    for key, row in records.items():
        lesion = (row["source"], row["lesion_id"])
        join(key, by_lesion.setdefault(lesion, key))
        if row["pixel_sha256"]:
            join(key, by_pixel.setdefault(row["pixel_sha256"], key))
    for a, b in HAM_DUPLICATES:
        left, right = f"HAM:ISIC_{a:07d}", f"HAM:ISIC_{b:07d}"
        if left in records and right in records:
            join(left, right)
    for row in decisions:
        if row["decision"] in {"same", "uncertain"}:
            join(row["left"], row["right"])

    groups = defaultdict(list)
    for key, row in records.items():
        groups[group(key)].append(row)
    exclusions = []
    eligible_groups = defaultdict(list)
    for identity, images in sorted(groups.items()):
        sources = {row["source"] for row in images}
        labels = {row["label"] for row in images}
        retained = []
        for row in images:
            reason = ""
            if len(labels) != 1:
                reason = "diagnosis conflict within linked group"
            elif row["source"] == "BCN" and sources == {"HAM", "BCN"}:
                reason = "cross-source match or unresolved suspected match"
            elif row["error"]:
                reason = "unreadable image"
            if reason:
                exclusions.append({"image": image_key(row), "group": identity, "reason": reason})
            else:
                retained.append(row)
        if retained:
            row = retained[0]
            eligible_groups[(row["source"], row["label"])].append((identity, retained))

    manifest = []
    rng = random.Random(34107)
    for source in sorted(ARCHIVES):
        for label in ("0", "1"):
            cohort = eligible_groups[(source, label)]
            if len(cohort) < 2:
                raise ValueError(f"Insufficient eligible groups for {source} label {label}")
            rng.shuffle(cohort)
            validation_n = max(1, round(len(cohort) * 0.20))
            for i, (identity, images) in enumerate(cohort):
                split = "external" if source == "BCN" else "validation" if i < validation_n else "train"
                # Fixed hash ordering is independent of model scores and repeated views count once in evaluation.
                chosen = images if split == "train" else [min(images, key=lambda row: hashlib.sha256(image_key(row).encode()).digest())]
                for row in chosen:
                    manifest.append({key: row[key] for key in ("source", "image_id", "lesion_id", "label", "archive", "member", "pixel_sha256")} | {"group": identity, "split": split})

    review_sha = digest(review) if review else "empty"
    frozen = audit_dir / ("frozen_" + hashlib.sha256((digest(scan_receipt) + review_sha).encode()).hexdigest()[:16])
    frozen.mkdir(exist_ok=True)
    manifest_path = frozen / "manifest.csv"
    columns = ["source", "image_id", "lesion_id", "label", "archive", "member", "pixel_sha256", "group", "split"]
    write_table(manifest_path, columns, sorted(manifest, key=lambda row: (row["split"], row["image_id"])))
    write_table(frozen / "exclusions.csv", ["image", "group", "reason"], exclusions)
    write_table(frozen / "review.csv", CANDIDATE_COLUMNS, decisions)
    summary = {"manifest_sha256": digest(manifest_path), "scan_receipt_sha256": digest(scan_receipt),
               "near_duplicate_search": receipt.get("near_duplicate_search", True),
               "review_sha256": digest(frozen / "review.csv"), "exclusions_sha256": digest(frozen / "exclusions.csv"),
               "images_by_split_and_label": dict(Counter(f"{row['split']}:{row['label']}" for row in manifest)),
               "groups_by_split_and_label": dict(Counter(f"{split}:{label}" for split, label, _ in {(row["split"], row["label"], row["group"]) for row in manifest})),
               "excluded_images": len(exclusions)}
    write_json(frozen / "freeze_receipt.json", summary)
    print(json.dumps(summary, indent=2))
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("metadata", "scan", "freeze"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("audit"))
    parser.add_argument("--review", type=Path)
    parser.add_argument("--exact-only", action="store_true", help="Scan decoded image pixels for exact copies only")
    args = parser.parse_args()
    if args.stage == "metadata":
        print(json.dumps(counts(eligible(args.data)), indent=2))
    elif args.stage == "scan":
        scan(args.data, args.out, near_duplicates=not args.exact_only)
    else:
        freeze(args.out, args.review)


if __name__ == "__main__":
    main()
