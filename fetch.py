import os
import io
import json
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import fitz  # PyMuPDF

load_dotenv()

# ── Config ─────────────────────────────────────────────────
CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
FOLDER_ID        = os.getenv("DRIVE_FOLDER_ID")
OUTPUT_FILE      = "fetched_docs.json"
SCOPES           = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service():
    """Authenticate and return a Google Drive service client."""
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def list_pdfs(service, folder_id):
    """Recursively list all PDFs in a folder and its subfolders."""
    pdfs = []

    # Get all items in this folder
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, parents)"
    ).execute()

    for item in results.get("files", []):
        if item["mimeType"] == "application/pdf":
            pdfs.append(item)
        elif item["mimeType"] == "application/vnd.google-apps.folder":
            # Recurse into subfolders
            pdfs.extend(list_pdfs(service, item["id"]))

    return pdfs


def download_pdf(service, file_id):
    """Download a PDF file and return its bytes."""
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buffer.seek(0)
    return buffer.read()


def extract_text_from_pdf(pdf_bytes):
    """Extract plain text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    text = " ".join(text_parts)
    return " ".join(text.split())  # normalise whitespace


def fetch():
    print("=== Material Specs KB — PDF Fetcher ===\n")

    service = get_drive_service()
    print(f"Connected to Google Drive\n")

    print(f"Scanning folder: {FOLDER_ID}")
    pdfs = list_pdfs(service, FOLDER_ID)
    print(f"Found {len(pdfs)} PDFs\n")

    results = []
    skipped = 0

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf['name']}")
        try:
            pdf_bytes = download_pdf(service, pdf["id"])
            text = extract_text_from_pdf(pdf_bytes)

            if len(text.split()) < 50:
                print(f"  — skipped (too short)")
                skipped += 1
                continue

            results.append({
                "title": pdf["name"].replace(".pdf", ""),
                "file_id": pdf["id"],
                "text": text
            })
            print(f"  ✓ {len(text.split())} words extracted")

        except Exception as e:
            print(f"  [error] {e}")
            skipped += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== Done ===")
    print(f"PDFs processed: {len(results)}")
    print(f"Skipped: {skipped}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch()