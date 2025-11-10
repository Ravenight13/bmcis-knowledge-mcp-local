#!/usr/bin/env python3
"""Quick BMCIS ingestion - minimal version."""
import hashlib
import os
from pathlib import Path
import psycopg2
import requests

DB_URL = "postgresql://cliffclarke@localhost:5432/bmcis_knowledge_dev"
OLLAMA_URL = "http://localhost:11434/api/embed"

def ingest():
    """Minimal ingestion."""
    base = Path("/Users/cliffclarke/Library/CloudStorage/Box-Box/BMCIS VP Sales System")

    # Get files from 00_-09_
    files = []
    for i in range(10):
        for d in base.glob(f"0{i}_*"):
            if d.is_dir():
                files.extend(d.glob("**/*.md"))
                files.extend(d.glob("**/*.txt"))

    files = sorted(set(files))  # Remove duplicates
    print(f"Found {len(files)} files")

    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    inserted = 0
    for idx, fpath in enumerate(files, 1):
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            if not content or len(content) < 50:
                continue

            # Simple chunking
            chunks = [content[i:i+1500] for i in range(0, len(content), 1500)]

            for chunk_num, chunk_text in enumerate(chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text or len(chunk_text) < 20:
                    continue

                # Get embedding from Ollama
                embedding = None
                try:
                    resp = requests.post(OLLAMA_URL, json={
                        "model": "nomic-embed-text",
                        "input": [chunk_text]
                    }, timeout=30)
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and len(embeddings[0]) > 0:
                        embedding = embeddings[0]
                except Exception as e:
                    pass

                # Use None for NULL in database if embedding failed
                if not embedding:
                    embedding = None

                # Insert
                chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                cursor.execute("""
                    INSERT INTO knowledge_base (
                        chunk_text, chunk_hash, embedding,
                        source_file, source_category,
                        chunk_index, total_chunks
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_hash) DO NOTHING
                """, (
                    chunk_text, chunk_hash, embedding,
                    fpath.name, str(fpath.parent.name),
                    chunk_num, len(chunks)
                ))
                inserted += 1

                if inserted % 100 == 0:
                    conn.commit()
                    print(f"{idx}/{len(files)}: {inserted} chunks inserted")

        except Exception as e:
            print(f"Error: {fpath}: {e}")

    conn.commit()
    cursor.close()
    conn.close()

    print(f"✅ Done: {inserted} chunks inserted")

if __name__ == "__main__":
    ingest()
