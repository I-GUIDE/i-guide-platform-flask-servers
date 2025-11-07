#!/usr/bin/env python3
"""Test Neo4j connectivity directly"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.local")

print("="*80)
print("  NEO4J CONNECTIVITY TEST")
print("="*80)

neo4j_uri = os.getenv("NEO4J_CONNECTION_STRING") or os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

print(f"\nConfiguration:")
print(f"  URI: {neo4j_uri or '[NOT SET]'}")
print(f"  User: {neo4j_user or '[NOT SET]'}")
print(f"  Password: {'***' if neo4j_password else '[NOT SET]'}")

if not all([neo4j_uri, neo4j_user, neo4j_password]):
    print("\n⚠️  Neo4j not fully configured")
    sys.exit(0)

try:
    print("\nImporting neo4j package...")
    from neo4j import GraphDatabase
    print("✅ neo4j package available")
except ImportError:
    print("❌ neo4j package not installed")
    print("   Install with: pip install neo4j")
    sys.exit(1)

try:
    print(f"\nConnecting to {neo4j_uri}...")
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password),
        connection_timeout=10
    )
    
    print("Running test query...")
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        value = result.single()[0]
        print(f"✅ Connection successful! Test returned: {value}")
        
        # Count nodes
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()[0]
        print(f"✅ Total nodes in database: {count:,}")
        
        # Sample query
        result = session.run("""
            MATCH (n)
            WHERE n.title IS NOT NULL
            RETURN n.title as title
            LIMIT 3
        """)
        print(f"\n✅ Sample nodes:")
        for record in result:
            print(f"   • {record['title']}")
    
    driver.close()
    print("\n✅ Neo4j is working correctly!")
    
except Exception as e:
    print(f"\n❌ Neo4j connection failed:")
    print(f"   {type(e).__name__}: {e}")
    sys.exit(1)



