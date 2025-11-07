#!/usr/bin/env python3
"""
Diagnostic script to check spatial search connectivity.

This checks each step of the spatial search flow:
1. Environment variables are loaded
2. Spacy NLP can extract locations
3. Google Maps API can geocode
4. OpenSearch is accessible

Run with:
    python rag_pipeline/tests/diagnose_spatial.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

print("=" * 70)
print("SPATIAL SEARCH CONNECTIVITY DIAGNOSTIC")
print("=" * 70)

# Load environment
env_path = Path(__file__).parent.parent / ".env.local"
if env_path.exists():
    load_dotenv(env_path)
    print(f"\n✓ Loaded environment from: {env_path}")
else:
    print(f"\n⚠️  No .env.local found at: {env_path}")

# Check environment variables
print("\n" + "=" * 70)
print("STEP 1: Environment Variables")
print("=" * 70)

google_key = os.getenv("GOOGLE_MAPS_API_KEY")
opensearch_node = os.getenv("OPENSEARCH_NODE")
opensearch_index = os.getenv("OPENSEARCH_INDEX")

if google_key:
    print(f"✓ GOOGLE_MAPS_API_KEY: {google_key[:8]}...")
else:
    print("✗ GOOGLE_MAPS_API_KEY not set")

if opensearch_node:
    print(f"✓ OPENSEARCH_NODE: {opensearch_node}")
else:
    print("✗ OPENSEARCH_NODE not set")

if opensearch_index:
    print(f"✓ OPENSEARCH_INDEX: {opensearch_index}")
else:
    print("✗ OPENSEARCH_INDEX not set")

# Check Spacy
print("\n" + "=" * 70)
print("STEP 2: Spacy NLP (Location Extraction)")
print("=" * 70)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    
    test_query = "Find flood risk map layers within 10 km of Hagerstown, Washington County, MD"
    doc = nlp(test_query)
    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    
    if locations:
        print(f"✓ Spacy model loaded successfully")
        print(f"✓ Extracted locations from test query: {locations}")
    else:
        print(f"⚠️  Spacy loaded but found no locations in: '{test_query}'")
        print(f"   Entities found: {[(ent.text, ent.label_) for ent in doc.ents]}")
except Exception as e:
    print(f"✗ Spacy error: {e}")

# Check Google Maps API
print("\n" + "=" * 70)
print("STEP 3: Google Maps API (Geocoding)")
print("=" * 70)

if google_key:
    try:
        import requests
        
        test_location = "Hagerstown, Washington County, MD"
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": test_location, "key": google_key}
        
        response = requests.get(geocode_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            result = data["results"][0]
            geometry = result["geometry"]
            location = geometry["location"]
            bounds = geometry.get("bounds")
            
            print(f"✓ Google Maps API accessible")
            print(f"✓ Geocoded '{test_location}':")
            print(f"  - Lat/Lng: {location['lat']}, {location['lng']}")
            if bounds:
                print(f"  - Bounds: {bounds}")
            else:
                viewport = geometry.get("viewport")
                print(f"  - Viewport: {viewport}")
        else:
            print(f"⚠️  Google Maps API accessible but no results for: {test_location}")
            print(f"   Status: {data.get('status')}")
    except Exception as e:
        print(f"✗ Google Maps API error: {e}")
else:
    print("⊘ Skipped (no GOOGLE_MAPS_API_KEY)")

# Check OpenSearch
print("\n" + "=" * 70)
print("STEP 4: OpenSearch Connectivity")
print("=" * 70)

if opensearch_node and opensearch_index:
    try:
        from opensearchpy import OpenSearch
        
        user = os.getenv("OPENSEARCH_USERNAME", "")
        pwd = os.getenv("OPENSEARCH_PASSWORD", "")
        use_ssl = opensearch_node.lower().startswith("https")
        
        print(f"Attempting connection to: {opensearch_node}")
        print(f"Using SSL: {use_ssl}")
        print(f"Using auth: {bool(user or pwd)}")
        
        client = OpenSearch(
            hosts=[opensearch_node],
            http_auth=(user, pwd) if (user or pwd) else None,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=10,
            max_retries=1,
            retry_on_timeout=False,
        )
        
        # Try a simple ping
        if client.ping():
            print(f"✓ OpenSearch ping successful")
        else:
            print(f"⚠️  OpenSearch ping returned False")
        
        # Try to get cluster info
        info = client.info()
        print(f"✓ OpenSearch accessible")
        print(f"  - Version: {info.get('version', {}).get('number')}")
        print(f"  - Cluster: {info.get('cluster_name')}")
        
        # Try to check if index exists
        if client.indices.exists(index=opensearch_index):
            print(f"✓ Index '{opensearch_index}' exists")
            
            # Get index stats
            stats = client.indices.stats(index=opensearch_index)
            doc_count = stats.get("_all", {}).get("primaries", {}).get("docs", {}).get("count", 0)
            print(f"  - Document count: {doc_count}")
        else:
            print(f"⚠️  Index '{opensearch_index}' does not exist")
        
    except Exception as e:
        print(f"✗ OpenSearch error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⊘ Skipped (missing OPENSEARCH_NODE or OPENSEARCH_INDEX)")

# Final summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if google_key and opensearch_node:
    print("\nIf all steps passed ✓, the spatial search should work.")
    print("If OpenSearch failed ✗, check:")
    print("  1. Network connectivity to the OpenSearch server")
    print("  2. Firewall/VPN settings")
    print("  3. SSL/TLS configuration (the error shows WRONG_VERSION_NUMBER)")
    print("  4. Try connecting from the actual deployment environment")
else:
    print("\nMissing required configuration. Set in .env.local:")
    if not google_key:
        print("  - GOOGLE_MAPS_API_KEY")
    if not opensearch_node:
        print("  - OPENSEARCH_NODE")

print("\n" + "=" * 70)



