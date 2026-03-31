"""
Google Drive upload helper for the Transtrack model pipeline.

Token  : model-dev-pipeline/ssh/gdrive_token.pickle  (OAuth2, pre-authenticated)
Root   : 1045M3e02dkI7JV9u0ZHgdoLOAvSyb9Ub  — model weights
Visuals: 1T9V04epSHhTPJUyLheLA5LLIVP5aJEQ_  — result videos shown to client

Folder structure created automatically under root:
    transtrack/
      {run_name}/
        seg/best.pt
        det/best.pt

Visuals structure under visuals folder:
    {run_name}/
      segment_day.mp4
      segment_wet.mp4
      segment_night.mp4
      detect_day.mp4
      detect_wet.mp4
      detect_night.mp4
"""

import pickle
from pathlib import Path

# Absolute path to the OAuth2 token — pre-authenticated, no browser needed
_TOKEN_PATH = Path(__file__).resolve().parents[1] / "ssh" / "gdrive_token.pickle"

GDRIVE_ROOT_FOLDER_ID    = "1045M3e02dkI7JV9u0ZHgdoLOAvSyb9Ub"
GDRIVE_VISUALS_FOLDER_ID = "1T9V04epSHhTPJUyLheLA5LLIVP5aJEQ_"


def build_service():
    """
    Build and return an authenticated Google Drive v3 service object.
    Refreshes the token automatically if expired.
    """
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    if not _TOKEN_PATH.exists():
        raise FileNotFoundError(
            f"GDrive token not found: {_TOKEN_PATH}\n"
            "Generate it with: python scripts/gdrive_auth.py"
        )

    with open(_TOKEN_PATH, "rb") as f:
        creds = pickle.load(f)

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(_TOKEN_PATH, "wb") as f:
            pickle.dump(creds, f)

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_or_create_folder(service, name: str, parent_id: str) -> str:
    """Return the Drive folder ID for `name` under `parent_id`, creating it if needed."""
    query = (
        f"name='{name}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    result = service.files().list(q=query, fields="files(id)").execute()
    if result.get("files"):
        return result["files"][0]["id"]

    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def upload_file(service, local_path: str | Path, folder_id: str) -> str:
    """
    Upload a file to Drive with resumable upload and progress logging.
    Skips upload if a file with the same name already exists in the folder.

    Returns the Drive file ID.
    """
    from googleapiclient.http import MediaFileUpload

    local_path = Path(local_path)
    fname = local_path.name

    # Check for existing file
    query = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=query, fields="files(id)").execute()
    if result.get("files"):
        fid = result["files"][0]["id"]
        print(f"  GDrive: already exists — {fname} (id={fid})")
        return fid

    size_mb = local_path.stat().st_size / 1e6
    print(f"  GDrive: uploading {fname}  ({size_mb:.1f} MB)")

    # Choose mimetype based on extension
    ext = local_path.suffix.lower()
    mime = {
        ".pt":  "application/octet-stream",
        ".mp4": "video/mp4",
        ".log": "text/plain",
    }.get(ext, "application/octet-stream")

    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    meta  = {"name": fname, "parents": [folder_id]}
    req   = service.files().create(body=meta, media_body=media, fields="id")

    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"    {int(status.progress() * 100)}%", end="\r")

    fid = resp["id"]
    print(f"  GDrive: done — {fname} (id={fid})")
    return fid


def make_public(service, file_id: str) -> str:
    """
    Grant anyone-with-link read access and return the shareable URL.
    Safe to call multiple times — Drive ignores duplicate permission grants.
    """
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()
    return f"https://drive.google.com/file/d/{file_id}/view"


def upload_and_share(service, local_path: str | Path, folder_id: str) -> str:
    """Upload a file, make it public, and return the shareable link."""
    fid  = upload_file(service, local_path, folder_id)
    link = make_public(service, fid)
    return link


def get_run_weights_folder(service, run_name: str, task: str) -> str:
    """
    Get/create the Drive folder for model weights:
        Root / transtrack / {run_name} / {task}
    task: "seg" or "det"
    """
    root     = get_or_create_folder(service, "transtrack", GDRIVE_ROOT_FOLDER_ID)
    run_dir  = get_or_create_folder(service, run_name, root)
    task_dir = get_or_create_folder(service, task, run_dir)
    return task_dir


def get_run_visuals_folder(service, run_name: str) -> str:
    """
    Get/create the Drive folder for result videos:
        Visuals / {run_name}
    """
    return get_or_create_folder(service, run_name, GDRIVE_VISUALS_FOLDER_ID)
