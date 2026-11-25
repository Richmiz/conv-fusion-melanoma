"""Recover the pinned model release for Notebook 23 without changing run identity."""
import shutil
import subprocess
import zlib
from pathlib import Path

# Uncompressed sizes and CRC32 values from Kaggle version 1 ZIP central directory.
FILES = {
    "4c_b5ns_1.5e_640_ext_15ep_best_fold0.pth": (114271334, 0x038ca4a1),
    "4c_b5ns_1.5e_640_ext_15ep_best_fold1.pth": (114271334, 0xbb8d768c),
    "4c_b5ns_1.5e_640_ext_15ep_best_fold2.pth": (114271338, 0x62cfd7ec),
    "4c_b5ns_1.5e_640_ext_15ep_best_fold3.pth": (114271334, 0x250444c6),
    "4c_b5ns_1.5e_640_ext_15ep_best_fold4.pth": (114271338, 0x0f868752),
    "9c_b4ns_2e_896_ext_15ep_best_fold0.pth": (70912131, 0xaf3de03e),
    "9c_b4ns_2e_896_ext_15ep_best_fold1.pth": (70912131, 0xc9d620a2),
    "9c_b4ns_2e_896_ext_15ep_best_fold2.pth": (70912131, 0xabf80179),
    "9c_b4ns_2e_896_ext_15ep_best_fold3.pth": (70912131, 0x7d98ac7a),
    "9c_b4ns_2e_896_ext_15ep_best_fold4.pth": (70912135, 0xcf5a8d6f),
    "9c_b4ns_448_ext_15ep-newfold_best_fold0.pth": (70903499, 0x354d3e7c),
    "9c_b4ns_448_ext_15ep-newfold_best_fold1.pth": (70903499, 0x315ad10e),
    "9c_b4ns_448_ext_15ep-newfold_best_fold2.pth": (70903499, 0xe1ae8025),
    "9c_b4ns_448_ext_15ep-newfold_best_fold3.pth": (70903499, 0xcb2579c3),
    "9c_b4ns_448_ext_15ep-newfold_best_fold4.pth": (70903499, 0x519aae61),
    "9c_b4ns_768_640_ext_15ep_best_fold0.pth": (70912131, 0x89908664),
    "9c_b4ns_768_640_ext_15ep_best_fold1.pth": (70912135, 0xa25bb48f),
    "9c_b4ns_768_640_ext_15ep_best_fold2.pth": (70912133, 0x475e86e6),
    "9c_b4ns_768_640_ext_15ep_best_fold3.pth": (70912133, 0x7e7cb0d2),
    "9c_b4ns_768_640_ext_15ep_best_fold4.pth": (70912137, 0xf5198525),
    "9c_b4ns_768_768_ext_15ep_best_fold0.pth": (70912131, 0xd4f93896),
    "9c_b4ns_768_768_ext_15ep_best_fold1.pth": (70912135, 0xd09a28c6),
    "9c_b4ns_768_768_ext_15ep_best_fold2.pth": (70912135, 0x11533725),
    "9c_b4ns_768_768_ext_15ep_best_fold3.pth": (70912133, 0xbdaf57a2),
    "9c_b4ns_768_768_ext_15ep_best_fold4.pth": (70912139, 0x3b559a11),
    "9c_b5ns_1.5e_640_ext_15ep_best_fold0.pth": (114312314, 0x31c96eab),
    "9c_b5ns_1.5e_640_ext_15ep_best_fold1.pth": (114312314, 0x725d10a8),
    "9c_b5ns_1.5e_640_ext_15ep_best_fold2.pth": (114312316, 0x0b1c3acb),
    "9c_b5ns_1.5e_640_ext_15ep_best_fold3.pth": (114312316, 0xa62e2710),
    "9c_b5ns_1.5e_640_ext_15ep_best_fold4.pth": (114312318, 0x36f783f9),
    "9c_b5ns_448_ext_15ep-newfold_best_fold0.pth": (114312314, 0x87f77c2c),
    "9c_b5ns_448_ext_15ep-newfold_best_fold1.pth": (114312314, 0x22fa0642),
    "9c_b5ns_448_ext_15ep-newfold_best_fold2.pth": (114312314, 0x90971a95),
    "9c_b5ns_448_ext_15ep-newfold_best_fold3.pth": (114312316, 0x9b51bbf4),
    "9c_b5ns_448_ext_15ep-newfold_best_fold4.pth": (114312314, 0x3b9c743f),
    "9c_b6ns_448_ext_15ep-newfold_best_fold0.pth": (164136635, 0x9d13e471),
    "9c_b6ns_448_ext_15ep-newfold_best_fold1.pth": (164136635, 0x26ad1f14),
    "9c_b6ns_448_ext_15ep-newfold_best_fold2.pth": (164136635, 0x442bbe22),
    "9c_b6ns_448_ext_15ep-newfold_best_fold3.pth": (164136635, 0xdd2350c0),
    "9c_b6ns_448_ext_15ep-newfold_best_fold4.pth": (164136635, 0x4cbc7012),
    "9c_b6ns_576_ext_15ep_oldfold_best_fold0.pth": (164136635, 0x46f7dffb),
    "9c_b6ns_576_ext_15ep_oldfold_best_fold1.pth": (164136635, 0x6bcdd84a),
    "9c_b6ns_576_ext_15ep_oldfold_best_fold2.pth": (164136635, 0xcfe0c7ca),
    "9c_b6ns_576_ext_15ep_oldfold_best_fold3.pth": (164136635, 0x6600f70c),
    "9c_b6ns_576_ext_15ep_oldfold_best_fold4.pth": (164136635, 0x93129f0f),
    "9c_b6ns_640_ext_15ep_best_fold0.pth": (164136635, 0xc888a176),
    "9c_b6ns_640_ext_15ep_best_fold1.pth": (164136635, 0xca476c30),
    "9c_b6ns_640_ext_15ep_best_fold2.pth": (164136637, 0xf96ff7ec),
    "9c_b6ns_640_ext_15ep_best_fold3.pth": (164136635, 0x4fae66ad),
    "9c_b6ns_640_ext_15ep_best_fold4.pth": (164136635, 0x5bf01631),
    "9c_b7ns_1e_576_ext_15ep_oldfold_best_fold0.pth": (256742790, 0x50cde9c4),
    "9c_b7ns_1e_576_ext_15ep_oldfold_best_fold1.pth": (256742790, 0x5994b3d3),
    "9c_b7ns_1e_576_ext_15ep_oldfold_best_fold2.pth": (256742790, 0xf5f7a553),
    "9c_b7ns_1e_576_ext_15ep_oldfold_best_fold3.pth": (256742790, 0x724400b3),
    "9c_b7ns_1e_576_ext_15ep_oldfold_best_fold4.pth": (256742790, 0xca1e5cef),
    "9c_b7ns_1e_640_ext_15ep_best_fold0.pth": (256742790, 0x56126d97),
    "9c_b7ns_1e_640_ext_15ep_best_fold1.pth": (256742790, 0xe55ac098),
    "9c_b7ns_1e_640_ext_15ep_best_fold2.pth": (256742790, 0x9725e67d),
    "9c_b7ns_1e_640_ext_15ep_best_fold3.pth": (256742790, 0x3cd03da0),
    "9c_b7ns_1e_640_ext_15ep_best_fold4.pth": (256742790, 0xfab0e680),
    "9c_meta128_32_b5ns_384_ext_15ep_best_fold0.pth": (114332216, 0x2d5b1f65),
    "9c_meta128_32_b5ns_384_ext_15ep_best_fold1.pth": (114332216, 0xc59a2895),
    "9c_meta128_32_b5ns_384_ext_15ep_best_fold2.pth": (114332216, 0x6001911b),
    "9c_meta128_32_b5ns_384_ext_15ep_best_fold3.pth": (114332216, 0xc6b49edd),
    "9c_meta128_32_b5ns_384_ext_15ep_best_fold4.pth": (114332216, 0x0dc6c996),
    "9c_meta_1.5e-5_b7ns_384_ext_15ep_best_fold0.pth": (257053609, 0x09d4f326),
    "9c_meta_1.5e-5_b7ns_384_ext_15ep_best_fold1.pth": (257053611, 0x3459c554),
    "9c_meta_1.5e-5_b7ns_384_ext_15ep_best_fold2.pth": (257053609, 0xe804bf8d),
    "9c_meta_1.5e-5_b7ns_384_ext_15ep_best_fold3.pth": (257053609, 0xc14f1eb9),
    "9c_meta_1.5e-5_b7ns_384_ext_15ep_best_fold4.pth": (257053609, 0x899b93a5),
    "9c_meta_b3_768_512_ext_18ep_best_fold0.pth": (43623899, 0x8e3542ec),
    "9c_meta_b3_768_512_ext_18ep_best_fold1.pth": (43623905, 0xb6e3db89),
    "9c_meta_b3_768_512_ext_18ep_best_fold2.pth": (43623905, 0x2f5ebc32),
    "9c_meta_b3_768_512_ext_18ep_best_fold3.pth": (43623909, 0x4609eb0f),
    "9c_meta_b3_768_512_ext_18ep_best_fold4.pth": (43623907, 0x9705a6a9),
    "9c_meta_b4ns_640_ext_15ep_best_fold0.pth": (71222950, 0x88fafdab),
    "9c_meta_b4ns_640_ext_15ep_best_fold1.pth": (71222950, 0xa6e02225),
    "9c_meta_b4ns_640_ext_15ep_best_fold2.pth": (71222950, 0xd758426a),
    "9c_meta_b4ns_640_ext_15ep_best_fold3.pth": (71222950, 0xf9318252),
    "9c_meta_b4ns_640_ext_15ep_best_fold4.pth": (71222952, 0xea369b7e),
    "9c_nest101_2e_640_ext_15ep_best_fold0.pth": (185693403, 0xc26e0508),
    "9c_nest101_2e_640_ext_15ep_best_fold1.pth": (185693403, 0x5ab375c8),
    "9c_nest101_2e_640_ext_15ep_best_fold2.pth": (185693403, 0x8ccae077),
    "9c_nest101_2e_640_ext_15ep_best_fold3.pth": (185693403, 0x7e3b11bc),
    "9c_nest101_2e_640_ext_15ep_best_fold4.pth": (185693403, 0xedc8ca84),
    "9c_se_x101_640_ext_15ep_best_fold0.pth": (188413379, 0x7b10e7b9),
    "9c_se_x101_640_ext_15ep_best_fold1.pth": (188413379, 0x49de8557),
    "9c_se_x101_640_ext_15ep_best_fold2.pth": (188413379, 0xc8c6df31),
    "9c_se_x101_640_ext_15ep_best_fold3.pth": (188413379, 0x46fc646e),
    "9c_se_x101_640_ext_15ep_best_fold4.pth": (188413379, 0xf706bdb8),
}

def valid(path, name):
    size, checksum = FILES[name]
    if not path.is_file() or path.stat().st_size != size:
        return False
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            crc = zlib.crc32(block, crc)
    return crc == checksum


def fetch_one(root, name):
    """Kaggle serves individual files raw or inside a single-member ZIP."""
    from zipfile import ZipFile, BadZipFile
    size, crc = FILES[name]
    dest = root / name
    if valid(dest, name):
        return
    transfer = root / (name + ".download")
    staging = root / (name + ".verified-part")
    url = "https://www.kaggle.com/api/v1/datasets/download/boliu0/melanoma-winning-models/" + name
    for attempt in range(2):
        if not transfer.exists() or attempt:
            args = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "4",
                    "--retry-all-errors", "--connect-timeout", "30", "--speed-time", "120",
                    "--speed-limit", "1", "--max-filesize", str(size + 2**20), "--output", str(transfer)]
            if not attempt:
                args.extend(["--continue-at", "-"])
            address = (url + "?datasetVersionNumber=1" if not attempt else
                       "https://www.kaggle.com/api/v1/datasets/download/boliu0/melanoma-winning-models"
                       + "?filename=" + name + "&datasetVersionNumber=1")
            result = subprocess.run(args + [address])
            if result.returncode:
                if attempt:
                    raise RuntimeError(f"Kaggle rejected the individual weight {name} (curl {result.returncode}). "
                                       "Keep the session open. Completed predictions and verified weights can be reused.")
                continue
        if valid(transfer, name):
            transfer.replace(dest)
            return
        try:
            with ZipFile(transfer) as zf:
                members = zf.infolist()
                if len(members) != 1 or (members[0].filename, members[0].file_size, members[0].CRC) != (name, size, crc):
                    continue
                with zf.open(members[0]) as src, staging.open("wb") as out:
                    shutil.copyfileobj(src, out, 2**20)
            if valid(staging, name):
                staging.replace(dest)
                transfer.unlink()
                return
        except (BadZipFile, EOFError):
            pass
    raise RuntimeError(f"Individual Kaggle download did not match the pinned release: {name}")


def stage_weights(root):
    from zipfile import ZipFile
    import hashlib
    root = Path(root)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    # A finished archive can still have the .part suffix if the old process stopped.
    for archive in (root / "winning.zip", root / "winning.zip.part"):
        if archive.is_file() and archive.stat().st_size == 11563945101:
            with archive.open("rb") as stream:
                good = hashlib.file_digest(stream, "md5").hexdigest() == "12e29f0707ab529f0a1e3ea5ea913c11"
            if good:
                with ZipFile(archive) as zf:
                    for name in FILES:
                        if not valid(models / name, name):
                            zf.extract(name, models)
                break
    for index, name in enumerate(FILES, 1):
        fetch_one(models, name)
        if index % 10 == 0:
            print(f"Verified {index}/90 competition weights", flush=True)
    (root / "kaggle_v1_ready").touch()
    return models


def recover(bench):
    from inference_store import digest
    release = "ab32a2132826264355723d0f6ad0468f31cbc0458b35ee81675b42d37caddd85"
    if bench.run_id != release[:20] or digest(bench.bundle) != release:
        raise RuntimeError("Recovery applies only to the interrupted ab32a213282626435572 release")
    bench.drive.check_writer()
    refs = {}
    for key in ("protocol", "bundle", "isic_base", "ph2_base", "derm_base", "prove_base"):
        ref = bench.refs.get(key)
        if ref is None:
            suffix = {"protocol": "protocol.json", "bundle": bench.bundle.name}.get(key, key + ".csv")
            path = bench.root / suffix
            if not bench.drive.restore(f"{bench.run_id}__{suffix}", path):
                raise RuntimeError(f"Missing saved {key}. Existing runtime and files were left in place.")
            ref = bench.publish(path)
        bench.drive.verify_receipt(ref)
        refs[key] = ref
    bench.save_json("weights_transport_recovery", {
        "run_id": bench.run_id, "protocol_sha256": bench.protocol_sha, "verified": refs,
        "helper_sha256": digest(Path(__file__))})
    print("Verified four saved cohort predictions in Drive. Recovering model weights.", flush=True)
    stage_weights(bench.root / "committee")


if __name__ == "__main__" and "bench" in globals():
    recover(bench)
