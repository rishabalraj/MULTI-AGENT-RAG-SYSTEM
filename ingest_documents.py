#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from core.document_processor import DocumentProcessor
from core.vector_store import get_vector_store
from core.hybrid_search import HybridSearcher


def ingest_directory(source_dir, department=None, roles=None):
    logger.info(f"Ingesting directory: {source_dir}")
    logger.info(f"Department: {department}, Roles: {roles}")

    processor = DocumentProcessor()
    chunks = processor.process_directory(
        source_dir,
        department=department,
        access_roles=roles or ["all"],
    )

    if not chunks:
        logger.error("No documents found or could not extract text")
        return 0

    logger.info(f"Processed {len(chunks)} chunks. Storing in vector DB...")

    store = get_vector_store()
    store.add_chunks(chunks)

    searcher = HybridSearcher(store)
    searcher.rebuild_bm25()

    logger.success(f"Ingested {len(chunks)} chunks from {source_dir}")
    return len(chunks)


def ingest_file(file_path, department=None, roles=None):
    logger.info(f"Ingesting file: {file_path}")

    processor = DocumentProcessor()
    chunks = processor.process_file(
        file_path,
        department=department,
        access_roles=roles or ["all"],
    )

    if not chunks:
        logger.error(f"Could not extract text from {file_path}")
        return 0

    store = get_vector_store()
    store.add_chunks(chunks)

    logger.success(f"Ingested {len(chunks)} chunks from {file_path}")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Enterprise Knowledge Assistant"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", "-s", help="Directory to ingest")
    group.add_argument("--file", "-f", help="Single file to ingest")
    parser.add_argument("--department", "-d", default=None)
    parser.add_argument("--roles", "-r", default="all")

    args = parser.parse_args()
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    if args.source:
        count = ingest_directory(args.source, args.department, roles)
    else:
        count = ingest_file(args.file, args.department, roles)

    if count == 0:
        sys.exit(1)

    print(f"\n Ingestion complete! {count} chunks indexed.")
    print("You can now start the API: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()