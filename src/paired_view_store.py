"""Immutable, resumable pilot artifacts in the user's designated Drive project."""
import os

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from inference_store import CheckpointDriveStore, digest


RESEARCH_FOLDER = os.getenv("CONV_FUSION_RESEARCH_FOLDER", "")
PROJECT_FOLDER = os.getenv("CONV_FUSION_PROJECT_FOLDER", "")
FOLDER_MIME = "application/vnd.google-apps.folder"


class ProjectRunStore(CheckpointDriveStore):
    def __init__(self, service, run_id, *, title="18 Paired-view melanoma training"):
        self.service = service
        files = service.files()
        project = files.get(fileId=PROJECT_FOLDER, fields="id,name,parents,mimeType,trashed").execute(num_retries=3)
        if (project.get("trashed") or project["mimeType"] != FOLDER_MIME
                or RESEARCH_FOLDER not in project.get("parents", [])):
            raise RuntimeError("The designated melanoma project must be inside the Research parent.")
        name = f"{title} - {run_id}"
        query = (f"'{PROJECT_FOLDER}' in parents and trashed = false and "
                 f"mimeType = '{FOLDER_MIME}' and name = {self.quote(name)}")
        fields = "id,name,parents,ownedByMe,shared,inheritedPermissionsDisabled,webViewLink"
        matches = self.list_files(query, fields)
        if len(matches) > 1:
            raise RuntimeError("Several run folders share this protocol name. Resume using the original folder.")
        if matches:
            folder = matches[0]
        else:
            folder = files.create(body={"name": name, "mimeType": FOLDER_MIME, "parents": [PROJECT_FOLDER]},
                                  fields=fields).execute()
        if not folder.get("ownedByMe") or folder.get("parents") != [PROJECT_FOLDER]:
            raise RuntimeError("Sign in with the Google account that owns the run folder.")
        if folder.get("shared") and not folder.get("inheritedPermissionsDisabled"):
            # Narrow access only on the newly created empty run folder.
            if matches and self.list_files(f"'{folder['id']}' in parents and trashed = false", "id"):
                raise RuntimeError("This existing run folder has inherited sharing and contains files. Verify its access.")
            folder = files.update(fileId=folder["id"], body={"inheritedPermissionsDisabled": True},
                                  fields=fields).execute(num_retries=3)
        self.folder_id = folder["id"]
        self.link = folder.get("webViewLink", f"https://drive.google.com/drive/folders/{self.folder_id}")
        self.writer = None
        self.verify_destination()

    def claim_writer(self):
        # Running the same launcher in a fresh session supersedes the previous writer.
        self.writer = uuid.uuid4().hex
        self.service.files().update(fileId=self.folder_id,body={"appProperties":{"pairedViewWriter":self.writer}},
                                    fields="id").execute(num_retries=3)
        self.check_writer()

    def check_writer(self):
        if self.writer is not None:
            folder = self.service.files().get(fileId=self.folder_id,fields="appProperties").execute(num_retries=3)
            if folder.get("appProperties",{}).get("pairedViewWriter") != self.writer:
                raise RuntimeError("A newer Colab session is writing this run. Continue in that session.")

    def assert_private(self, entry):
        if not entry.get("ownedByMe"):
            raise RuntimeError("The signed-in account must own the stored artifact.")
        if entry.get("shared") is False:
            return
        outside = []
        token = None
        while True:
            page = self.service.permissions().list(
                fileId=entry["id"], pageToken=token,
                fields="nextPageToken,permissions(id,role,type,view,deleted)").execute(num_retries=3)
            permissions = page.get("permissions", [])
            if not permissions and not token:
                raise RuntimeError("Could not verify effective Drive access.")
            outside.extend(p for p in permissions if not p.get("deleted") and p.get("role") != "owner"
                           and p.get("view") != "metadata")
            token = page.get("nextPageToken")
            if not token:
                break
        if outside:
            raise RuntimeError("A non-owner has content access to this run. No checkpoint may be written here.")

    def verify_destination(self):
        folder = self.service.files().get(fileId=self.folder_id,
            fields="id,parents,ownedByMe,shared,trashed").execute(num_retries=3)
        if folder.get("trashed") or folder.get("parents") != [PROJECT_FOLDER]:
            raise RuntimeError("Run destination moved or was trashed.")
        self.assert_private(folder)


class RunFiles:
    def __init__(self, drive, root):
        self.drive, self.root = drive, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def restore(self, name):
        path = self.root / name
        # A crashed local writer or an old partial must never mask an intact Drive copy.
        remote = self.drive.lookup(name)
        if not remote:
            return None
        if path.exists():
            try:
                self.drive.verify(path, remote)
            except RuntimeError:
                path.unlink()
        self.drive.restore(name, path)
        return path

    def publish(self, path):
        self.drive.check_writer()
        ref = self.drive.publish(path, path.name)
        self.drive.verify_receipt(ref)
        return ref

    def json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
        return self.publish(path)

    def csv(self, name, frame):
        path = self.root / name
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        return self.publish(path)

    def read_json(self, name):
        path = self.restore(name)
        return json.loads(path.read_bytes()) if path else None

    def from_ref(self, ref):
        self.drive.verify_receipt(ref)
        path = self.restore(ref["name"])
        if path is None or digest(path) != ref["sha256"]:
            raise RuntimeError(f"Cannot restore {ref['name']}.")
        return path


class ComputeBudget:
    """Renew small durable time reservations. A lost session forfeits only its last unused slice."""
    limit = 7200
    slice_seconds = 120

    def __init__(self, files):
        self.files = files
        self.spent = 0.0
        self.active = None
        self.next_check = 0.
        names = set(files.drive.names_starting_with("compute_"))
        for name in sorted(n for n in names if n.endswith("_open.json")):
            record = files.read_json(name)
            closing = name.replace("_open.json", "_close.json")
            if closing in names:
                self.spent += files.read_json(closing)["seconds"]
            else:
                prefix = name.removesuffix("_open.json")+"_lease_"
                renewals = sorted(n for n in names if n.startswith(prefix))
                self.spent += (files.read_json(renewals[-1])["reserved"] if renewals else record["reserved"])

    @property
    def remaining(self):
        return max(0.0, self.limit-self.spent)

    def check(self):
        if not self.active:
            return
        elapsed = time.monotonic()-self.active["start"]
        if elapsed >= self.active["reserved"]:
            raise RuntimeError("The reserved GPU-stage time has elapsed. Earlier Drive checkpoints are preserved.")
        if time.monotonic()>=self.next_check:
            self.files.drive.check_writer()
            self.next_check = time.monotonic()+60
        if self.active["reserved"]-elapsed < self.slice_seconds/2 and self.active["reserved"] < self.active["ceiling"]:
            reservation = min(self.active["ceiling"],elapsed+self.slice_seconds)
            self.active["renewals"] += 1
            name = self.active["stem"]+f"_lease_{self.active['renewals']:04d}.json"
            self.files.json(name,{"reserved":reservation})
            self.active["reserved"] = reservation

    @contextmanager
    def section(self, label, ceiling):
        if self.active:
            raise RuntimeError("Nested compute reservations are unsupported.")
        maximum = min(self.remaining, float(ceiling))
        if maximum < 2:
            raise RuntimeError("The two-hour compute budget is exhausted. The run is preserved.")
        stem = "compute_" + uuid.uuid4().hex
        reserved = min(maximum,self.slice_seconds)
        self.files.json(stem+"_open.json", {"stage":label,"reserved":reserved,"maximum":maximum})
        start = time.monotonic()
        self.active = {"start":start,"reserved":reserved,"ceiling":maximum,"stem":stem,"renewals":0}
        try:
            yield
            self.check()
        finally:
            elapsed = time.monotonic()-start
            self.active = None
            self.spent += elapsed
            self.files.json(stem+"_close.json", {"stage":label,"seconds":elapsed})
