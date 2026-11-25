"""Verified Drive storage and bounded ZIP range reads for Colab inference."""
import hashlib
import io
import struct
import tempfile
import time
import uuid
import zlib
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZIP_STORED, ZIP_DEFLATED

def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()

sha256 = digest


def read_range(url, start, end, total):
    """Read only the requested archive bytes; never accept a full ZIP response."""
    if not (0 <= start <= end < total) or end - start + 1 > 32 * 1024**2:
        raise ValueError("Invalid or oversized ZIP range.")
    for attempt in range(5):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}",
                                           "Accept-Encoding": "identity"})
            with urlopen(request, timeout=45) as response:
                if response.status != 206:
                    raise RuntimeError("The archive server ignored the range request; no full archive was downloaded.")
                expected = f"bytes {start}-{end}/{total}"
                if response.headers.get("Content-Range") != expected:
                    raise RuntimeError("The archive returned a different byte range or archive size.")
                result = response.read(end - start + 2)
            if len(result) != end - start + 1:
                raise RuntimeError("The archive range was incomplete.")
            return result
        except HTTPError as error:
            if attempt == 4 or error.code not in {408, 429, 500, 502, 503, 504}:
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            error.close()
            delay = float(retry_after) if retry_after and retry_after.isdigit() else float(2 ** (attempt + 1))
            time.sleep(min(delay, 30.0))
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


class DriveStore:
    def __init__(self, service, folder_name):
        self.service = service
        query = ("'root' in parents and trashed = false and "
                 "mimeType = 'application/vnd.google-apps.folder' and name = " + self.quote(folder_name))
        folders = self.list_files(query, "id,ownedByMe,shared,webViewLink")
        if len(folders) > 1:
            raise RuntimeError("Multiple Drive folders have the pilot name. Choose a unique folder name.")
        if folders:
            folder = folders[0]
        else:
            folder = self.service.files().create(
                body={"name": folder_name, "mimeType": "application/vnd.google-apps.folder"},
                fields="id,ownedByMe,shared,webViewLink",
            ).execute()
        if not folder.get("ownedByMe") or folder.get("shared"):
            raise RuntimeError("The pilot folder must be owned by this Google account and private.")
        self.folder_id = folder["id"]
        self.link = folder.get("webViewLink", f"https://drive.google.com/drive/folders/{self.folder_id}")

    @staticmethod
    def quote(value):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    def list_files(self, query, fields):
        files = []
        token = None
        while True:
            response = self.service.files().list(
                q=query, pageToken=token, fields=f"nextPageToken,files({fields})",
            ).execute(num_retries=3)
            files.extend(response.get("files", []))
            token = response.get("nextPageToken")
            if not token:
                return files

    def lookup(self, name):
        files = self.list_files(
            f"'{self.folder_id}' in parents and trashed = false and name = {self.quote(name)}",
            "id,name,size,md5Checksum,sha256Checksum,ownedByMe,shared",
        )
        if len(files) > 1:
            raise RuntimeError(f"Duplicate Drive artifact name: {name}")
        if files:
            self.assert_private(files[0])
        return files[0] if files else None

    def assert_private(self, entry):
        if not entry.get("ownedByMe") or entry.get("shared"):
            raise RuntimeError(f"Drive artifact is not private: {entry.get('name', entry['id'])}")

    def names_starting_with(self, prefix):
        items = self.list_files(
            f"'{self.folder_id}' in parents and trashed = false and name contains {self.quote(prefix)}", "name",
        )
        names = [item["name"] for item in items if item["name"].startswith(prefix)]
        if len(names) != len(set(names)):
            raise RuntimeError(f"Conflicting Drive artifact names for {prefix}")
        return names

    def verify_receipt(self, receipt):
        remote = self.lookup(receipt["name"])
        if not remote or remote["id"] != receipt["id"] or receipt["folder_id"] != self.folder_id:
            raise RuntimeError(f"Drive artifact missing or replaced: {receipt['name']}")
        if remote.get("sha256Checksum"):
            matches = remote["sha256Checksum"] == receipt["sha256"]
        else:
            matches = bool(receipt.get("md5")) and remote.get("md5Checksum") == receipt["md5"]
        if int(remote.get("size", -1)) != receipt["bytes"] or not matches:
            raise RuntimeError(f"Drive artifact no longer matches its receipt: {receipt['name']}")

    def quota(self, required_bytes):
        response = self.service.about().get(fields="storageQuota").execute(num_retries=3)
        quota = response.get("storageQuota")
        if quota is None or "usage" not in quota:
            raise RuntimeError("Could not verify Google Drive quota")
        free = int(quota["limit"]) - int(quota["usage"]) if "limit" in quota else None
        if free is not None and free < required_bytes:
            raise RuntimeError(f"Drive has {free:,} bytes free. The next operation requires {required_bytes:,} bytes.")
        return free

    @staticmethod
    def verify(path, remote):
        if int(remote.get("size", -1)) != path.stat().st_size:
            raise RuntimeError(f"Drive size mismatch for {path.name}")
        if remote.get("sha256Checksum"):
            verified = remote["sha256Checksum"] == digest(path)
        else:
            verified = remote.get("md5Checksum") == digest(path, "md5")
        if not verified:
            raise RuntimeError(f"Drive checksum mismatch for {path.name}")

    def publish(self, path, name):
        from googleapiclient.http import MediaFileUpload

        remote = self.lookup(name)
        if not remote:
            self.quota(path.stat().st_size)
            upload = self.service.files().create(
                body={"name": name, "parents": [self.folder_id]},
                media_body=MediaFileUpload(str(path), resumable=True),
                fields="id,name,size,md5Checksum,sha256Checksum,ownedByMe,shared",
            )
            _, remote = upload.next_chunk(num_retries=3)
            while remote is None:
                _, remote = upload.next_chunk(num_retries=3)
        self.assert_private(remote)
        self.verify(path, remote)
        return {"id": remote["id"], "name": name, "bytes": int(remote["size"]),
                "sha256": digest(path), "md5": remote.get("md5Checksum"), "folder_id": self.folder_id}

    def restore(self, name, path):
        from googleapiclient.http import MediaIoBaseDownload

        remote = self.lookup(name)
        if not remote:
            return False
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".download")
            with partial.open("wb") as stream:
                transfer = MediaIoBaseDownload(stream, self.service.files().get_media(fileId=remote["id"]))
                complete = False
                while not complete:
                    _, complete = transfer.next_chunk(num_retries=3)
            self.verify(partial, remote)
            partial.replace(path)
        self.verify(path, remote)
        return True

    def preflight(self, required_bytes=64 * 1024**2):
        free = self.quota(required_bytes)
        sample = uuid.uuid4().hex.encode()
        with tempfile.TemporaryDirectory() as folder:
            probe = Path(folder) / "probe.txt"
            probe.write_bytes(sample)
            receipt = self.publish(probe, "preflight-" + sample.decode() + ".txt")
        result = self.service.files().get_media(fileId=receipt["id"]).execute(num_retries=3)
        if io.BytesIO(result).read() != sample:
            raise RuntimeError("Drive preflight read-back did not match the uploaded bytes")
        return {"folder": self.link, "quota_free_bytes": free, "probe": receipt}


class CheckpointDriveStore(DriveStore):
    def publish(self, path, name):
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload
        remote = self.lookup(name)
        if remote is None:
            size = path.stat().st_size
            self.quota(size)
            api = self.service.files()
            file_id = api.generateIds(count=1, space="drive", fields="ids").execute(num_retries=3)["ids"][0]
            resumable = size > 5_000_000
            for attempt in range(3):
                try:
                    upload = api.create(
                        body={"id": file_id, "name": name, "parents": [self.folder_id]},
                        media_body=MediaFileUpload(str(path), resumable=resumable),
                        fields="id,name,size,md5Checksum,sha256Checksum,ownedByMe,shared",
                    )
                    if resumable:
                        while remote is None:
                            _, remote = upload.next_chunk(num_retries=2)
                    else:
                        remote = upload.execute(num_retries=2)
                    break
                except (HttpError, OSError):
                    remote = self.lookup(name)
                    if remote is not None:
                        break
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        self.assert_private(remote)
        self.verify(path, remote)
        return {"id": remote["id"], "name": name, "bytes": int(remote["size"]),
                "sha256": sha256(path), "md5": remote.get("md5Checksum"), "folder_id": self.folder_id}

class FolderReader:
    def __init__(self, service, folder_id, cache_dir):
        if not folder_id:
            raise ValueError("Configure the relevant CONV_FUSION Drive folder before starting this workflow.")
        self.service, self.folder_id = service, folder_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, meta, path):
        from googleapiclient.http import MediaIoBaseDownload
        if path.is_file():
            try:
                DriveStore.verify(path, meta)
                return path
            except RuntimeError:
                pass
        partial = path.with_suffix(path.suffix + ".download")
        with partial.open("wb") as stream:
            transfer = MediaIoBaseDownload(stream, self.service.files().get_media(fileId=meta["id"]))
            done = False
            while not done:
                _, done = transfer.next_chunk(num_retries=3)
        DriveStore.verify(partial, meta)
        partial.replace(path)
        return path

    def fetch(self, name):
        query = f"'{self.folder_id}' in parents and trashed = false and name = {DriveStore.quote(name)}"
        files = self.service.files().list(q=query, fields="files(id,name,size,md5Checksum,sha256Checksum)").execute(num_retries=3)["files"]
        if len(files) != 1:
            raise RuntimeError(f"Expected one file named {name}; found {len(files)}.")
        return self.download(files[0], self.cache_dir / name)

    def fetch_artifact(self, ref):
        meta = self.service.files().get(fileId=ref["id"], fields="id,name,size,sha256Checksum,md5Checksum").execute(num_retries=3)
        if meta["name"] != ref["name"] or int(meta["size"]) != ref["bytes"]:
            raise RuntimeError("A selected artifact has changed its identity or size.")
        path = self.download(meta, self.cache_dir / meta["name"])
        if sha256(path) != ref["sha256"]:
            raise RuntimeError(f"Saved artifact checksum mismatch: {meta['name']}")
        return path

    def verify_artifact(self, ref):
        meta = self.service.files().get(fileId=ref["id"], fields="id,name,size,sha256Checksum,md5Checksum").execute(num_retries=3)
        if meta["name"] != ref["name"] or int(meta["size"]) != ref["bytes"]:
            raise RuntimeError("A selected model is missing or changed.")
        if meta.get("sha256Checksum"):
            matches = meta["sha256Checksum"] == ref["sha256"]
        else:
            expected_md5 = ref.get("md5")
            if not expected_md5:
                path = self.fetch_artifact(ref)
                with path.open("rb") as stream:
                    expected_md5 = hashlib.file_digest(stream, "md5").hexdigest()
            matches = meta.get("md5Checksum") == expected_md5
        if not matches:
            raise RuntimeError("A selected model does not match its durable receipt.")

class RemoteZip:
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.start, self.cache = -1, b""

    def seek(self, offset, whence=0):
        self.pos = (self.pos if whence == 1 else self.size if whence == 2 else 0) + offset
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, count=-1):
        count = min(count if count >= 0 else self.size, self.size - self.pos)
        if count <= 0:
            return b""
        if not (self.start <= self.pos <= self.pos + count <= self.start + len(self.cache)):
            end = min(self.size, self.pos + max(count, 131072)) - 1
            self.cache = read_range(self.url, self.pos, end, self.size)
            self.start = self.pos
        offset = self.pos - self.start
        self.pos += count
        return self.cache[offset:offset + count]

def fetch_zip_member_bytes(url, info, archive_size):
    offset = info.header_offset
    end = min(archive_size - 1, offset + 30 + len(info.filename.encode("utf8")) + 1024 + info.compress_size)
    for attempt in range(4):
        try:
            raw = read_range(url, offset, end, archive_size)
            header = struct.unpack("<IHHHHHIIIHH", raw[:30])
            if header[0] != 0x04034B50:
                raise RuntimeError("Invalid ZIP local header.")
            start = 30 + header[-2] + header[-1]
            stop = start + info.compress_size
            if stop > len(raw):
                compressed = read_range(url, offset + start, offset + stop - 1, archive_size)
            else:
                compressed = raw[start:stop]
            if info.compress_type == ZIP_STORED:
                result = compressed
            elif info.compress_type == ZIP_DEFLATED:
                result = zlib.decompress(compressed, -zlib.MAX_WBITS)
            else:
                raise RuntimeError("Unsupported archive compression.")
            if len(result) != info.file_size or zlib.crc32(result) != info.CRC:
                raise RuntimeError("An archive member failed its length or CRC check.")
            return result
        except OSError:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
