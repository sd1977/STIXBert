"""Retrieve STIX data from open TAXII 2.1 servers."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Open TAXII servers
TAXII_SOURCES = {
    "mitre_attack": {
        "url": "https://cti-taxii.mitre.org/taxii2",
        "description": "MITRE ATT&CK",
    },
    "cisa_ais": {
        "url": "https://taxii.cisa.gov/taxii2",
        "description": "CISA Automated Indicator Sharing",
        "requires_auth": True,
    },
}


def list_collections(taxii_url: str) -> list[dict]:
    """List available collections on a TAXII 2.1 server.

    Args:
        taxii_url: TAXII server discovery URL.

    Returns:
        List of collection metadata dicts.
    """
    from taxii2client.v21 import Server

    server = Server(taxii_url)
    collections = []
    for api_root in server.api_roots:
        for collection in api_root.collections:
            collections.append({
                "id": collection.id,
                "title": collection.title,
                "description": getattr(collection, "description", ""),
                "url": collection.url,
            })
    return collections


def fetch_collection(
    collection_url: str,
    output_dir: str = "data/raw/taxii",
    added_after: str = None,
    limit: int = None,
) -> dict:
    """Fetch all objects from a TAXII 2.1 collection.

    Args:
        collection_url: URL of the specific collection.
        output_dir: Directory to cache results.
        added_after: Only fetch objects added after this datetime (ISO format).
        limit: Maximum number of objects to fetch per page.

    Returns:
        STIX bundle dict with all fetched objects.
    """
    from taxii2client.v21 import Collection

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    collection = Collection(collection_url)
    logger.info(f"Fetching collection: {collection.title}...")

    kwargs = {}
    if added_after:
        kwargs["added_after"] = added_after
    if limit:
        kwargs["per_request"] = limit

    all_objects = []

    # Paginate through the collection
    envelope = collection.get_objects(**kwargs)
    if hasattr(envelope, "text"):
        data = json.loads(envelope.text)
    else:
        data = envelope

    objects = data.get("objects", [])
    all_objects.extend(objects)
    logger.info(f"  Page 1: {len(objects)} objects")

    # Handle pagination via 'more' and 'next'
    page = 2
    while data.get("more", False) and data.get("next"):
        kwargs["next"] = data["next"]
        envelope = collection.get_objects(**kwargs)
        data = json.loads(envelope.text) if hasattr(envelope, "text") else envelope
        objects = data.get("objects", [])
        all_objects.extend(objects)
        logger.info(f"  Page {page}: {len(objects)} objects")
        page += 1

    bundle = {
        "type": "bundle",
        "id": f"bundle--taxii-{collection.id}",
        "objects": all_objects,
    }

    dest = output_path / f"taxii_{collection.id}.json"
    with open(dest, "w") as f:
        json.dump(bundle, f, indent=2)
    logger.info(f"Saved {len(all_objects)} objects to {dest}")

    return bundle


def fetch_all_open_feeds(
    output_dir: str = "data/raw/taxii",
) -> list[dict]:
    """Fetch data from all configured open (no-auth) TAXII sources.

    Returns:
        List of STIX bundles.
    """
    bundles = []
    for name, source in TAXII_SOURCES.items():
        if source.get("requires_auth"):
            logger.info(f"Skipping {name} (requires authentication)")
            continue
        try:
            collections = list_collections(source["url"])
            for coll in collections:
                bundle = fetch_collection(coll["url"], output_dir)
                bundles.append(bundle)
        except Exception as e:
            logger.warning(f"Failed to fetch {name}: {e}")
    return bundles
