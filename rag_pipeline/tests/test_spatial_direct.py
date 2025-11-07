#!/usr/bin/env python3
"""Test Google Maps API and OpenSearch spatial fields"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.local")

print("="*80)
print("  SPATIAL SEARCH CONNECTIVITY TEST")
print("="*80)

# Test Google Maps API
print("\n[1] Google Maps Geocoding API")
print("-"*80)

google_key = os.getenv("GOOGLE_MAPS_API_KEY")
print(f"API Key: {'***' + google_key[-8:] if google_key else '[NOT SET]'}")

if google_key:
    import requests
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": "California", "key": google_key}
        
        print("\nGeocoding 'California'...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            location = data["results"][0]["geometry"]["location"]
            print(f"✅ Google Maps API working!")
            print(f"   Lat: {location['lat']}, Lng: {location['lng']}")
        else:
            print(f"⚠️  API returned status: {data.get('status')}")
    except Exception as e:
        print(f"❌ Failed: {e}")
else:
    print("⚠️  Not configured")

# Test OpenSearch spatial fields
print("\n[2] OpenSearch Spatial Metadata")
print("-"*80)

from opensearchpy import OpenSearch

node = os.getenv("OPENSEARCH_NODE")
index = os.getenv("OPENSEARCH_INDEX")
user = os.getenv("OPENSEARCH_USERNAME", "")
pwd = os.getenv("OPENSEARCH_PASSWORD", "")

print(f"Index: {index}")

try:
    client = OpenSearch(
        hosts=[node],
        http_auth=(user, pwd) if user else None,
        use_ssl=node.lower().startswith("https"),
        verify_certs=False,
        timeout=10
    )
    
    # Count docs with spatial data
    query = {
        "query": {"exists": {"field": "spatial-bounding-box-geojson"}},
        "size": 0
    }
    result = client.search(index=index, body=query)
    spatial_count = result["hits"]["total"]["value"]
    
    print(f"✅ Documents with spatial metadata: {spatial_count:,}")
    
    if spatial_count > 0:
        # Get a sample
        sample = client.search(index=index, body={
            "query": {"exists": {"field": "spatial-bounding-box-geojson"}},
            "size": 1
        })
        if sample["hits"]["hits"]:
            doc = sample["hits"]["hits"][0]["_source"]
            print(f"   Sample: {doc.get('title', 'No title')[:50]}")
    else:
        print("   ⚠️  No documents have spatial bounding boxes")
    
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "="*80)



