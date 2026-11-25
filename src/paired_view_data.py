"""Frozen source lesions, real repeat eligibility, and single-photo inputs."""

from collections import defaultdict
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from acquisition_stress import correct_rgb, pixel_hash
from backbone_resolution_study import download_file_curl
from inference_store import RemoteZip, fetch_zip_member_bytes
from audit import perceptual_hashes


ISIC_URL = "https://isic-archive.s3.amazonaws.com/challenges/2020/ISIC_2020_Training_JPEG.zip"
ISIC_BYTES = 24707698022
DOIS = "https://isic-archive.s3.amazonaws.com/dois/"
ARCHIVES = {
    "HAM": ("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_Input.zip",
            2771732744, None, None),
    "BCN": ("https://ndownloader.figshare.com/files/45833178", 6368901290, None,
            "2293c9c654a3f76d6506a57ad17cddd1"),
    "PROVe-AI": (DOIS + "10.34970-576276/prove-ai.zip", 325980070,
                 "6619e608704826a337fcbc3ae640d32176eb46340e3bf1adb525406f9c32fd51", None),
    "DERM12345": (DOIS + "10.34970-705541/derm12345.zip", 1863258972,
                  "432bbbcf9f3f9ff3ae81cfaf740b909a8e697aa811f68298d7946178a6f94a54", None),
    "HIBA": (DOIS + "10.34970-559884/hiba-skin-lesions.zip", 545980514,
             "e5635c842be4a9e0614dbf551a2830ae93157d8ce27a34076d6a131e6718bd2f", None),
}


def read_table(path):
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def anatomy(value):
    value = value.strip().lower()
    regions = {"trunk": ("torso", "trunk", "back", "abdomen", "chest"),
               "upper limb": ("upper extremity", "upper extremities", "arm", "hand"),
               "lower limb": ("lower extremity", "lower extremities", "leg", "foot"),
               "head/neck": ("head", "neck", "face", "scalp", "ear"),
               "acral": ("acral", "palms", "soles"), "genital": ("genital",)}
    return next((k for k, words in regions.items() if any(w in value for w in words)), "unknown")


def source_lesions(split, ham, bcn):
    if split.groupby("group").split.nunique().max() != 1 or split.groupby("group").label.nunique().max() != 1:
        raise ValueError("A source lesion group crosses splits or diagnoses.")
    training = split.loc[split.split.eq("train")].sort_values(["source", "image_id"]).drop_duplicates("group").copy()
    ham = ham.set_index("image_id", verify_integrity=True)
    bcn = bcn.set_index("bcn_filename", verify_integrity=True)
    centers, sites, biopsy = [], [], []
    same_day = defaultdict(list)
    for name, row in bcn.iterrows():
        if row.capture_date:
            same_day[row.lesion_id, row.capture_date, row.diagnosis].append(name)
    candidates = {}
    for row in training.itertuples():
        if row.source == "HAM":
            meta = ham.loc[row.image_id]
            centers.append("Queensland-Rosendahl" if meta.dataset == "rosendahl" else "Vienna")
            sites.append(anatomy(meta.localization))
            biopsy.append(meta.dx_type)
        else:
            name = row.image_id + ".jpg"
            meta = bcn.loc[name]
            if (meta.lesion_id != row.lesion_id or meta.diagnosis != ("MEL" if int(row.label) else "NV")):
                raise ValueError("Barcelona training record differs from the source registry.")
            centers.append("Barcelona")
            sites.append(anatomy(meta.anatom_site_general))
            biopsy.append("histo" if int(row.label) else "unknown")
            candidates[row.image_id] = sorted(n for n in same_day[meta.lesion_id, meta.capture_date, meta.diagnosis]
                                              if n != name)
    training["center"], training["body"], training["confirmation"] = centers, sites, biopsy
    training["patient_id"] = training.get("patient_id", "")
    return training.reset_index(drop=True), candidates


class Photos:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stack, self.zips, self.index = ExitStack(), {}, None

    def open(self, source):
        if source == "ISIC2020":
            if self.index is None:
                with ZipFile(RemoteZip(ISIC_URL, ISIC_BYTES)) as z:
                    self.index = {Path(i.filename).stem: i for i in z.infolist() if i.filename.lower().endswith(".jpg")}
        elif source not in self.zips:
            url, size, sha, md5 = ARCHIVES[source]
            path = download_file_curl(self.folder / (source + ".zip"), url, size, sha, md5)
            self.zips[source] = self.stack.enter_context(ZipFile(path))

    def get(self, source, member):
        self.open(source)
        if source == "ISIC2020":
            name = Path(member).stem
            path = self.folder/"ISIC2020_jpegs"/(name+".jpg")
            data = path.read_bytes() if path.is_file() else fetch_zip_member_bytes(ISIC_URL,self.index[name],ISIC_BYTES)
        else:
            data = self.zips[source].read(member)
        try:
            with Image.open(BytesIO(data)) as im:
                return ImageOps.exif_transpose(im).convert("RGB")
        except OSError as error:
            raise UnidentifiedImageError(f"Undecodable photograph {member}") from error

    def prefetch_isic(self, members):
        self.open("ISIC2020")
        folder = self.folder/"ISIC2020_jpegs"
        folder.mkdir(exist_ok=True)
        def download(member):
            name = Path(member).stem
            path = folder/(name+".jpg")
            if name in self.index and not path.exists():
                content = fetch_zip_member_bytes(ISIC_URL,self.index[name],ISIC_BYTES)
                part = path.with_suffix(".part")
                part.write_bytes(content)
                part.replace(path)
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(download,members))

    def close(self):
        self.stack.close()


def image_views(im):
    """Original full-frame recipe and four overlapping native field crops."""
    resize = Image.Resampling.BILINEAR
    full = np.asarray(im.resize((224, 224), resize))
    brightness = full.max(axis=2) > 18
    ys, xs = np.flatnonzero(brightness.mean(1) > .2), np.flatnonzero(brightness.mean(0) > .2)
    if len(xs) > 8 and len(ys) > 8:
        x0, x1 = int(xs[0] * im.width / 224), max(1, int(np.ceil((xs[-1] + 1) * im.width / 224)))
        y0, y1 = int(ys[0] * im.height / 224), max(1, int(np.ceil((ys[-1] + 1) * im.height / 224)))
    else:
        x0, y0, x1, y1 = 0, 0, im.width, im.height
    w, h = max(1, round((x1 - x0) * .65)), max(1, round((y1 - y0) * .65))
    boxes = [(x, y, x+w, y+h) for y in (y0, y1-h) for x in (x0, x1-w)]
    crops = [correct_rgb(np.asarray(im.crop(b).resize((224, 224), resize))) for b in boxes]
    return correct_rgb(full), np.stack(crops), (im.width, im.height, w, h)


def pair_quality(first, second):
    """Diagnosis-blind proxy for shared tissue. The masks are approximate, not diagnostic outlines."""
    import cv2
    cv2.setRNGSeed(17)
    photos = [np.asarray(im.resize((320,320),Image.Resampling.BILINEAR)) for im in (first,second)]
    arrays = [cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in photos]
    drift = float(np.abs(arrays[0].astype(float) - arrays[1]).mean())
    if drift < 3:
        return False, "near-identical appearance"
    yy, xx = np.ogrid[:320, :320]
    radius = (xx-159.5)**2 + (yy-159.5)**2
    central = radius < 135**2
    # Normalize luminance for geometric matching only. Diagnostic inputs retain their original appearance.
    sift = cv2.SIFT_create(nfeatures=450,contrastThreshold=.015)
    local_contrast = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    features = []
    for a,rgb in zip(arrays,photos,strict=True):
        mask = (central & (a > 18) & (a < 247)).astype("uint8")
        if mask.mean() < .12 or np.std(a[mask.astype(bool)]) < 8:
            return False, "insufficient visible structure"
        annulus = (radius>100**2) & (mask>0)
        if annulus.sum()<500:
            return False, "insufficient surrounding skin"
        lab = cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
        distance = np.linalg.norm(lab-np.median(lab[annulus],axis=0),axis=2)
        threshold = max(9.,float(np.percentile(distance[annulus],70)))
        tissue = ((distance>threshold) & (mask>0)).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        tissue = cv2.morphologyEx(tissue,cv2.MORPH_CLOSE,kernel)
        _, regions, stats, _ = cv2.connectedComponentsWithStats(tissue)
        choices = [(int(((regions==j)&(radius<90**2)).sum()),j) for j in range(1,len(stats))
                   if stats[j,cv2.CC_STAT_AREA]>=650]
        if not choices or max(choices)[0]<350:
            return False, "no distinct central tissue region"
        lesion = (regions==max(choices)[1]).astype("uint8")
        interior = cv2.dilate(lesion,kernel) & mask
        features.append(sift.detectAndCompute(local_contrast.apply(a),interior*255))
    (ka, da), (kb, db) = features
    if da is None or db is None or min(len(da), len(db)) < 8:
        return False, "few local features"
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [a for a, b in matches if a.distance < .76 * b.distance]
    if len(good) < 8:
        return False, "few matched features"
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    transform, mask = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4)
    if transform is None or mask is None:
        return False, "unresolved geometry"
    idx = mask.ravel().astype(bool)
    if idx.sum() < 8 or idx.mean() < .5:
        return False, "inconsistent geometry"
    area = min(cv2.contourArea(cv2.convexHull(points[idx])) for points in (pa, pb))
    return (True, "eligible") if area >= 650 else (False, "limited common tissue")


def choose_peers(training, candidates, photos, forbidden, blocked_ids, original_rows):
    known = set(forbidden)
    used = set()
    records, audit = [], []
    for row in training.to_dict("records"):
        row["peer_id"], row["single_id"] = "", ""
        if not candidates.get(row["image_id"]):
            records.append(row)
            continue
        anchor = photos.get("BCN", row["member"])
        if pixel_hash(anchor) != row["pixel_sha256"]:
            raise RuntimeError(f"Training anchor pixels changed: {row['image_id']}")
        singleton = None
        for name in candidates[row["image_id"]]:
            peer = Path(name).stem
            member = str(Path(row["member"]).with_name(name)).replace("\\", "/")
            if peer in blocked_ids or peer in original_rows:
                audit.append((row["image_id"], peer, "", "reserved image ID"))
                continue
            image = photos.get("BCN", member)
            sha = pixel_hash(image)
            if sha in known or sha in used:
                good, reason = False, "reserved or duplicate pixels"
            else:
                good, reason = pair_quality(anchor, image)
            audit.append((row["image_id"], peer, sha, reason))
            if good:
                used.add(sha)
                row["peer_id"] = peer
                records.append({**row, "image_id": peer, "member": member, "pixel_sha256": sha,
                                "peer_id": "", "kind": "repeat"})
                break
            if reason in {"few matched features","inconsistent geometry","unresolved geometry","limited common tissue"} and singleton is None:
                singleton = (peer,member,sha)
        if not row["peer_id"] and singleton is not None:
            peer,member,sha = singleton
            used.add(sha)
            row["single_id"] = peer
            records.append({**row,"image_id":peer,"member":member,"pixel_sha256":sha,"peer_id":"",
                            "single_id":"","kind":"single_view"})
        records.append(row)
    frame = pd.DataFrame(records)
    frame["kind"] = frame.get("kind", pd.Series(index=frame.index, dtype=str)).fillna("anchor")
    return frame.reset_index(drop=True), pd.DataFrame(audit, columns=["anchor_id", "peer_id", "pixel_sha256", "reason"])


class RepeatIndex:
    """Find a 64-bit perceptual match within seven bits, including rotations and reflections."""
    def __init__(self, signatures):
        self.keys, self.values, self.parts = [], [], [defaultdict(list) for _ in range(8)]
        for key, signature in signatures.items():
            for value in dict.fromkeys(int(x, 16) for x in signature.split(":")):
                index = len(self.keys)
                self.keys.append(key)
                self.values.append(value)
                for i, table in enumerate(self.parts):
                    table[(value >> (8*i)) & 255].append(index)

    def matches(self, signature):
        hits = set()
        for encoded in signature.split(":")[::8]:
            query = int(encoded, 16)
            # At least one equal byte is necessary at Hamming distance <= 7.
            candidates = {j for i, table in enumerate(self.parts)
                          for j in table.get((query >> (8*i)) & 255, ())}
            hits.update(self.keys[j] for j in candidates if (query ^ self.values[j]).bit_count() <= 7)
        return hits


def screen_source_repeats(training, original, photos):
    repeats = training.loc[training.kind.ne("anchor")]
    signatures = {r.image_id: perceptual_hashes(photos.get(r.source, r.member)) for r in repeats.itertuples()}
    index = RepeatIndex(signatures)
    groups = repeats.set_index("image_id").group.to_dict()
    conflicts = []
    for r in pd.concat([original, repeats], ignore_index=True).itertuples():
        signature = signatures.get(r.image_id) or perceptual_hashes(photos.get(r.source, r.member))
        for peer in sorted(index.matches(signature)):
            if groups[peer] != r.group:
                conflicts.append((peer, r.image_id, r.group))
    excluded = {c[0] for c in conflicts}
    clean = training.loc[~training.image_id.isin(excluded)].copy()
    clean.loc[clean.peer_id.isin(excluded), "peer_id"] = ""
    clean.loc[clean.single_id.isin(excluded), "single_id"] = ""
    clean["phash"] = clean.image_id.map(signatures).fillna("")
    if set(clean.loc[clean.peer_id.ne(""), "label"].astype(int)) != {0,1}:
        raise RuntimeError("Insufficient usable, source-disjoint repeat photographs for both diagnoses.")
    return clean, pd.DataFrame(conflicts, columns=["excluded_peer", "other_image", "other_group"])
