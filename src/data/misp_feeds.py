"""Retrieve STIX data from MISP OSINT feeds."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Public MISP instances / OSINT feed endpoints
CIRCL_OSINT_URL = "https://www.circl.lu/doc/misp/feed-osint/"
DIGITALSIDE_API_URL = "https://api.github.com/repos/davidonzo/Threat-Intel/contents/stix2"
DIGITALSIDE_RAW_URL = "https://raw.githubusercontent.com/davidonzo/Threat-Intel/master/stix2/"
MAX_DIGITALSIDE_BUNDLES = 50
# NOTE: The ThreatFox POST API now requires auth. Use the public export endpoint.
THREATFOX_EXPORT_URL = "https://threatfox.abuse.ch/export/json/recent/"


def fetch_threatfox_iocs(
    output_dir: str = "data/raw/threatfox",
) -> list[dict]:
    """Fetch recent IOCs from ThreatFox public export endpoint.

    Args:
        output_dir: Directory to cache results.

    Returns:
        List of IOC dicts from ThreatFox.
    """
    import urllib.request

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching ThreatFox recent IOCs (export endpoint)...")
    req = urllib.request.Request(THREATFOX_EXPORT_URL)
    with urllib.request.urlopen(req) as resp:
        raw = json.loads(resp.read())

    # Export format: dict keyed by IOC id -> IOC record
    iocs = list(raw.values()) if isinstance(raw, dict) else raw
    flat_iocs = []
    for item in iocs:
        if isinstance(item, dict):
            flat_iocs.append(item)
        elif isinstance(item, list):
            flat_iocs.extend(item)

    logger.info(f"Retrieved {len(flat_iocs)} IOCs from ThreatFox")

    dest = output_path / "threatfox_recent.json"
    with open(dest, "w") as f:
        json.dump(flat_iocs, f, indent=2)

    return flat_iocs


def convert_misp_to_stix(misp_event: dict) -> dict:
    """Convert a MISP event to STIX 2.1 bundle using misp-stix.

    Args:
        misp_event: MISP event dict.

    Returns:
        STIX 2.1 bundle dict.
    """
    from misp_stix_converter import MISPtoSTIX21Parser

    parser = MISPtoSTIX21Parser()
    parser.parse_misp_event(misp_event)
    bundle = parser.bundle
    return json.loads(bundle.serialize())


def fetch_digitalside_stix(
    output_dir: str = "data/raw/digitalside",
    max_bundles: int = MAX_DIGITALSIDE_BUNDLES,
) -> list[dict]:
    """Fetch STIX 2.1 bundles from DigitalSide Threat-Intel.

    The stix2/ directory contains ~1000 individual hash-named bundles.
    Uses the GitHub API to list files and downloads a batch.

    Args:
        output_dir: Directory to cache results.
        max_bundles: Maximum number of bundles to download.

    Returns:
        List of STIX bundle dicts.
    """
    import urllib.request

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Listing DigitalSide stix2/ via GitHub API...")
    req = urllib.request.Request(
        DIGITALSIDE_API_URL,
        headers={"Accept": "application/vnd.github.v3+json"},
    )
    with urllib.request.urlopen(req) as resp:
        listing = json.loads(resp.read())

    json_files = [
        f["name"] for f in listing
        if isinstance(f, dict) and f.get("name", "").endswith(".json")
    ]
    logger.info(f"Found {len(json_files)} STIX bundles; downloading {max_bundles}")

    bundles = []
    for fname in json_files[:max_bundles]:
        url = f"{DIGITALSIDE_RAW_URL}{fname}"
        dest = output_path / fname
        try:
            urllib.request.urlretrieve(url, dest)
            with open(dest) as f:
                bundle = json.load(f)
            bundles.append(bundle)
        except Exception as e:
            logger.warning(f"Failed to fetch DigitalSide {fname}: {e}")

    logger.info(f"DigitalSide: loaded {len(bundles)} STIX bundles")
    return bundles


def threatfox_to_stix(iocs: list[dict]) -> dict:
    """Convert ThreatFox IOCs to a STIX 2.1 bundle.

    Creates Indicator SDOs with malware family relationships where available.

    Args:
        iocs: List of ThreatFox IOC dicts.

    Returns:
        STIX 2.1 bundle dict.
    """
    from stix2 import Bundle, Indicator, Malware, Relationship
    from datetime import datetime

    stix_objects = []
    malware_cache = {}

    for ioc in iocs:
        ioc_type = ioc.get("ioc_type", "")
        ioc_value = ioc.get("ioc", "").strip()
        malware_name = ioc.get("malware_printable", "")
        threat_type = ioc.get("threat_type", "")

        if not ioc_value:
            continue

        # Map ThreatFox type to STIX pattern
        pattern = None
        if "ip" in ioc_type.lower():
            ip = ioc_value.split(":")[0]
            pattern = f"[ipv4-addr:value = '{ip}']"
        elif "domain" in ioc_type.lower():
            pattern = f"[domain-name:value = '{ioc_value}']"
        elif "url" in ioc_type.lower():
            pattern = f"[url:value = '{ioc_value}']"
        elif "md5" in ioc_type.lower():
            pattern = f"[file:hashes.MD5 = '{ioc_value}']"
        elif "sha256" in ioc_type.lower():
            pattern = f"[file:hashes.'SHA-256' = '{ioc_value}']"

        if not pattern:
            continue

        # ThreatFox uses "YYYY-MM-DD HH:MM:SS"; stix2 needs ISO 8601
        raw_ts = ioc.get("first_seen_utc", "")
        valid_from = (
            raw_ts.replace(" ", "T") + "Z"
            if raw_ts
            else datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        indicator = Indicator(
            name=f"ThreatFox: {ioc_value}",
            pattern=pattern,
            pattern_type="stix",
            valid_from=valid_from,
            labels=[threat_type] if threat_type else ["malicious-activity"],
        )
        stix_objects.append(indicator)

        # Create Malware SDO and relationship if malware family is known
        if malware_name and malware_name not in malware_cache:
            malware = Malware(
                name=malware_name,
                is_family=True,
                malware_types=["unknown"],
            )
            malware_cache[malware_name] = malware
            stix_objects.append(malware)

        if malware_name and malware_name in malware_cache:
            rel = Relationship(
                source_ref=indicator.id,
                target_ref=malware_cache[malware_name].id,
                relationship_type="indicates",
            )
            stix_objects.append(rel)

    bundle = Bundle(objects=stix_objects)
    return json.loads(bundle.serialize())
