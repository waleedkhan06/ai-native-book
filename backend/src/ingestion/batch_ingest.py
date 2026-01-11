#!/usr/bin/env python3
"""
Batch ingestion script for processing multiple documents at once
"""
import asyncio
import os
import sys
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from ..database.models import Base
from ..database.session import engine, get_db
from ..services.document_service import document_service
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_directory(directory_path: str):
    """
    Process all text files in a directory and ingest them
    """
    db_gen = get_db()
    db = next(db_gen)

    try:
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return

        # Process all .txt, .md, and .py files in the directory
        text_files = list(directory.glob("*.txt")) + list(directory.glob("*.md")) + list(directory.glob("*.py"))

        logger.info(f"Found {len(text_files)} files to process")

        for file_path in text_files:
            try:
                logger.info(f"Processing file: {file_path}")

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.strip():
                    logger.warning(f"Skipping empty file: {file_path}")
                    continue

                # Create document
                document = await document_service.create_document(
                    title=file_path.name,
                    content=content,
                    source_url=f"file://{file_path.absolute()}",
                    db=db
                )

                logger.info(f"Successfully processed document: {document.id} - {document.title}")

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue  # Continue with next file

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
    finally:
        db.close()


async def main():
    """
    Main function to run the batch ingestion
    """
    if len(sys.argv) < 2:
        print("Usage: python batch_ingest.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    logger.info(f"Starting batch ingestion from directory: {directory_path}")

    await process_directory(directory_path)

    logger.info("Batch ingestion completed")


if __name__ == "__main__":
    asyncio.run(main())