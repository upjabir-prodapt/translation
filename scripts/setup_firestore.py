#!/usr/bin/env python3
"""Setup Firestore database and collections for BabelDOC Translation Service.

This script initializes the Firestore database with the required collection
structure and indexes.

Usage:
    python scripts/setup_firestore.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from google.cloud import firestore

from config.constants import settings
from config.logging_config import logger


def setup_firestore():
    """Setup Firestore collection and indexes."""
    try:
        # Initialize Firestore client
        logger.info(f"Connecting to Firestore database: {settings.FIRESTORE_DATABASE}")
        client = firestore.Client(
            project=settings.GOOGLE_CLOUD_PROJECT_ID,
            database=settings.FIRESTORE_DATABASE,
        )

        # Reference to the collection
        collection_ref = client.collection(settings.FIRESTORE_COLLECTION)

        # Check if collection exists by trying to get documents
        docs = list(collection_ref.limit(1).stream())

        if docs:
            logger.info(
                f"✅ Collection '{settings.FIRESTORE_COLLECTION}' already exists with {len(list(collection_ref.stream()))} documents"
            )
        else:
            logger.info(f"Creating collection '{settings.FIRESTORE_COLLECTION}'...")

            # Create a dummy document to initialize the collection
            # Firestore collections are created implicitly when you add the first document
            dummy_doc_ref = collection_ref.document("_setup_test")
            dummy_doc_ref.set(
                {"setup": True, "message": "Collection initialization document"}
            )

            # Delete the dummy document
            dummy_doc_ref.delete()

            logger.info(
                f"✅ Collection '{settings.FIRESTORE_COLLECTION}' created successfully"
            )

        # Print configuration summary
        logger.info("\n" + "=" * 60)
        logger.info("Firestore Configuration:")
        logger.info(f"  Project ID: {settings.GOOGLE_CLOUD_PROJECT_ID}")
        logger.info(f"  Database: {settings.FIRESTORE_DATABASE}")
        logger.info(f"  Collection: {settings.FIRESTORE_COLLECTION}")
        logger.info(f"  Location: {settings.GOOGLE_CLOUD_LOCATION}")
        logger.info("=" * 60 + "\n")

        # Suggested indexes (manual setup required)
        logger.info("📋 Recommended Firestore Indexes:")
        logger.info("   1. Composite index on (status, created_at)")
        logger.info("   2. TTL policy on 'expire_at' field")
        logger.info("\n   Create indexes at:")
        logger.info(
            f"   https://console.cloud.google.com/firestore/indexes?project={settings.GOOGLE_CLOUD_PROJECT_ID}"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to setup Firestore: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting Firestore setup...")
    success = setup_firestore()

    if success:
        logger.info("✅ Firestore setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Firestore setup failed!")
        sys.exit(1)
