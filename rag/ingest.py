"""
FinSight India — Week 3: ingest filings and SEBI regulations into a local
vector store (Chroma), tagged with a doc_type (and company, for filings) so
retrieval can filter by source type or company later.

Folder convention — everything sits directly under data/, no extra wrapper
folders needed:
    data/<Company Name>/*.pdf   -> one subfolder per company, e.g. data/TCS/*.pdf
    data/regulations/*.pdf      -> SEBI circulars, flat. Folder name is matched
                                    case-insensitively, so 'Regulations' works too.

Run:
    python ingest.py
"""

import os
import glob
from pypdf import PdfReader
# chromadb and sentence_transformers are imported lazily inside main() instead of
# here at module level — this is Part 2's territory. Importing them only when
# main() actually runs means extract_text()/chunk_text() (Part 1) can be used
# on their own without installing sentence-transformers/chromadb first.

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

CHUNK_SIZE = 800      # characters — a reasonable middle ground for prose-heavy financial text
CHUNK_OVERLAP = 100   # ~12.5% overlap — keeps a sentence split across a chunk boundary
                       # from disappearing entirely from both chunks

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, free, runs fully local — no API calls


def extract_text(filepath: str) -> str:
    """Read a .pdf or .txt file into a single text string."""
    if filepath.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window chunker. Not fully sentence-aware — good enough to
    start, and simple enough that you can see exactly what it's doing (unlike a
    framework's RecursiveCharacterTextSplitter, which does something similar but
    hides the logic). It snaps to word boundaries where it safely can, but never
    at the cost of two invariants that matter more than cosmetic word-splitting:
      1. Forward progress every iteration (or it hangs forever) — real financial
         PDFs often extract as long unbroken runs of digits with no whitespace
         (dense numeric tables), and naive space-snapping can get stuck there.
      2. No content gap between chunks — a snap that jumps past the current
         chunk's end would silently drop the text in between.
    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    if not text:
        return []

    MIN_CHUNK = max(50, chunk_size // 4)  # don't let backward-snapping shrink a chunk to near-nothing

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # walk backward from the raw cut point to the nearest space, so we
            # don't split a word in half — but only if that still leaves a
            # reasonably sized chunk. On a long space-sparse run (e.g. a
            # numeric table), the nearest space backward might be right next
            # to `start`, producing a near-empty chunk — skip snapping there
            # and accept a mid-word cut instead, which is the lesser problem.
            snap = text.rfind(" ", start, end)
            if snap - start >= MIN_CHUNK:
                end = snap
        else:
            end = len(text)

        chunks.append(text[start:end].strip())

        if end >= len(text):
            # This chunk already reached the end of the text — there's nothing
            # left to overlap into. Without this check, the code below would
            # keep computing a "next_start" short of the end and generate a
            # string of spurious near-duplicate tail fragments (e.g. chunking
            # a 10-character string could otherwise produce 5 overlapping
            # fragments of the same short string instead of just 1 chunk).
            break

        # Next chunk starts `overlap` chars before this one ends, but never
        # before start+1 (guarantees forward progress => guarantees the loop
        # terminates) and never after `end` (guarantees no content gap).
        next_start = end - overlap
        next_start = max(next_start, start + 1)
        next_start = min(next_start, end)

        if next_start < end:
            # snap forward to the next space so the overlap region doesn't
            # start mid-word — but only if that space is still within this
            # chunk (<= end). A space found beyond `end` would create exactly
            # the content gap we're guarding against, so ignore it in that case.
            snap_forward = text.find(" ", next_start, end)
            if snap_forward != -1:
                next_start = snap_forward + 1

        start = next_start
    return chunks


def ingest_folder(folder: str, doc_type: str, collection, embedder, company: str = None) -> int:
    """Chunk + embed every .pdf/.txt file in a folder, store with doc_type
    (and company, if given) metadata."""
    filepaths = glob.glob(os.path.join(folder, "*.pdf")) + glob.glob(os.path.join(folder, "*.txt"))
    total_chunks = 0

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        label = f"{company}/{filename}" if company else filename
        print(f"  Processing {label}...")
        text = extract_text(filepath)
        chunks = chunk_text(text)

        if not chunks:
            print(f"    WARNING: no text extracted from {label} — skipping. "
                  f"(Scanned/image-only PDFs need OCR, which isn't set up here yet.)")
            continue

        embeddings = embedder.encode(chunks).tolist()
        # include company in the id so the same filename in two company folders
        # (e.g. two 'fact-sheet.pdf's) doesn't collide in Chroma
        id_prefix = f"{doc_type}:{company}:{filename}" if company else f"{doc_type}:{filename}"
        ids = [f"{id_prefix}:{i}" for i in range(len(chunks))]
        metadata_base = {"doc_type": doc_type, "source_file": filename}
        if company:
            metadata_base["company"] = company
        metadatas = [dict(metadata_base, chunk_index=i) for i in range(len(chunks))]

        collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"    -> {len(chunks)} chunks")

    return total_chunks


def ingest_filings(folder: str, collection, embedder) -> int:
    """Filings are organized as data/<Company Name>/*.pdf — one subfolder per
    company, sitting directly under data/ (no separate 'filings' wrapper
    folder needed). Walk each subfolder and tag its chunks with that company
    name. The one folder name treated specially is 'regulations' (any casing)
    — that one is handled by ingest_regulations() instead, not here.
    """
    total = 0
    subfolders = [f for f in glob.glob(os.path.join(folder, "*")) if os.path.isdir(f)]

    for subfolder in subfolders:
        company = os.path.basename(subfolder)
        if company.lower() == "regulations":
            continue  # handled separately, not a company
        total += ingest_folder(subfolder, "filing", collection, embedder, company=company)

    return total


def ingest_regulations(folder: str, collection, embedder) -> int:
    """SEBI circulars live in a folder named 'regulations' (any casing —
    'Regulations' works too) directly under data/, flat, no subfolders."""
    for name in os.listdir(folder):
        candidate = os.path.join(folder, name)
        if os.path.isdir(candidate) and name.lower() == "regulations":
            return ingest_folder(candidate, "regulation", collection, embedder)
    print(f"  No 'regulations' folder found directly under {folder} — skipping.")
    return 0


def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection("finsight_docs")

    print("\nIngesting filings/concalls...")
    filings_count = ingest_filings(DATA_DIR, collection, embedder)

    print("\nIngesting SEBI regulations...")
    regs_count = ingest_regulations(DATA_DIR, collection, embedder)

    print(f"\nDone. {filings_count} filing chunks + {regs_count} regulation chunks "
          f"stored in {DB_DIR}")

    if filings_count == 0 and regs_count == 0:
        print("\nNo documents found. Add PDFs/txt files under data/<Company Name>/ "
              "and data/regulations/ first — see the README in each folder for where to get them.")


if __name__ == "__main__":
    main()  