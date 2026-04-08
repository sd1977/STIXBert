"""Retrieve MITRE ATT&CK STIX 2.1 data via GitHub or TAXII."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ATTACK_STIX_GITHUB = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
ATTACK_STIX_MOBILE = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/mobile-attack/mobile-attack.json"
ATTACK_STIX_ICS = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/ics-attack/ics-attack.json"

ATTACK_TAXII_URL = "https://cti-taxii.mitre.org/taxii2"
ATTACK_TAXII_COLLECTION_ENTERPRISE = "enterprise-attack"
ATTACK_TAXII_COLLECTION_MOBILE = "mobile-attack"
ATTACK_TAXII_COLLECTION_ICS = "ics-attack"


def download_attack_bundle(
    domain: str = "enterprise",
    output_dir: str = "data/raw/mitre_attack",
    use_taxii: bool = False,
) -> dict:
    """Download MITRE ATT&CK STIX bundle.

    Args:
        domain: One of 'enterprise', 'mobile', 'ics'.
        output_dir: Directory to save the downloaded bundle.
        use_taxii: If True, use TAXII 2.1 server; otherwise download from GitHub.

    Returns:
        Parsed STIX bundle as dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_taxii:
        return _download_via_taxii(domain, output_path)
    return _download_via_github(domain, output_path)


def _download_via_github(domain: str, output_path: Path) -> dict:
    import urllib.request

    urls = {
        "enterprise": ATTACK_STIX_GITHUB,
        "mobile": ATTACK_STIX_MOBILE,
        "ics": ATTACK_STIX_ICS,
    }
    url = urls[domain]
    dest = output_path / f"{domain}-attack.json"

    if dest.exists():
        logger.info(f"Using cached bundle: {dest}")
        with open(dest) as f:
            return json.load(f)

    logger.info(f"Downloading ATT&CK {domain} from GitHub...")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Saved to {dest}")

    with open(dest) as f:
        return json.load(f)


def _download_via_taxii(domain: str, output_path: Path) -> dict:
    from taxii2client.v21 import Server, Collection

    server = Server(ATTACK_TAXII_URL)
    api_root = server.api_roots[0]

    collection_map = {
        "enterprise": ATTACK_TAXII_COLLECTION_ENTERPRISE,
        "mobile": ATTACK_TAXII_COLLECTION_MOBILE,
        "ics": ATTACK_TAXII_COLLECTION_ICS,
    }
    target_title = collection_map[domain]

    collection = None
    for c in api_root.collections:
        if c.title == target_title:
            collection = Collection(c.url)
            break

    if collection is None:
        raise ValueError(f"Collection '{target_title}' not found on TAXII server")

    logger.info(f"Fetching ATT&CK {domain} via TAXII...")
    bundle = collection.get_objects()
    bundle_dict = json.loads(bundle.text) if hasattr(bundle, "text") else bundle

    dest = output_path / f"{domain}-attack-taxii.json"
    with open(dest, "w") as f:
        json.dump(bundle_dict, f, indent=2)
    logger.info(f"Saved to {dest}")

    return bundle_dict


def parse_attack_objects(bundle: dict) -> dict:
    """Parse a STIX bundle into categorized object lists.

    Returns:
        Dict keyed by STIX type, values are lists of objects.
    """
    objects_by_type = {}
    for obj in bundle.get("objects", []):
        obj_type = obj.get("type", "unknown")
        objects_by_type.setdefault(obj_type, []).append(obj)

    logger.info("ATT&CK bundle summary:")
    for obj_type, objs in sorted(objects_by_type.items()):
        logger.info(f"  {obj_type}: {len(objs)}")

    return objects_by_type


def get_attack_data(
    domains: Optional[list] = None,
    output_dir: str = "data/raw/mitre_attack",
    use_taxii: bool = False,
) -> list[dict]:
    """Download and return ATT&CK bundles for specified domains.

    Args:
        domains: List of domains to fetch. Defaults to all three.
        output_dir: Where to cache bundles.
        use_taxii: Use TAXII server instead of GitHub.

    Returns:
        List of parsed STIX bundles.
    """
    if domains is None:
        domains = ["enterprise", "mobile", "ics"]

    bundles = []
    for domain in domains:
        bundle = download_attack_bundle(domain, output_dir, use_taxii)
        bundles.append(bundle)
        objects_by_type = parse_attack_objects(bundle)
        total = sum(len(v) for v in objects_by_type.values())
        logger.info(f"ATT&CK {domain}: {total} objects")

    return bundles
