#!/usr/bin/env node
//Need to specify the geoshape mapping
import { Client } from '@opensearch-project/opensearch';
import wktParser from 'wkt-parser';
import dotenv from 'dotenv';
import fs from 'fs';

dotenv.config();

// === CONFIGURATION ===
const OLD_INDEX = 'neo4j-elements';
const NEW_INDEX = 'i_guide_spatial_embedding';
const BATCH_SIZE = 1000;
const WKT_GEOSHAPE_FIELDS = ['spatial-geometry', 'spatial-bounding-box'];
const WKT_GEOPOINT_FIELDS = ['spatial-centroid'];

const client = new Client({
  node: 'https://149.165.155.195:9200' || 'http://localhost:9200',
  auth: {
    username: process.env.OPENSEARCH_USERNAME || 'admin',
    password: process.env.OPENSEARCH_PASSWORD || 'admin',
  },
  ssl: {
    rejectUnauthorized: false,
  },
});

// === IMPROVED WKT SANITIZATION ===
function sanitizeWkt(wktString) {
  return wktString
    .replace(/(\b[A-Z]+)\s*(\()/gi, '$1$2')  // Remove space between type and (
    .replace(/\s*([(),])\s*/g, '$1')         // Remove spaces around brackets/commas
    .replace(/(\d)-/g, '$1 -')               // Fix negative numbers
    .replace(/\s+/g, ' ')                    // Collapse multiple spaces
    .trim();
}


function parseEnvelope(envelopeStr) {
  const sanitized = sanitizeWkt(envelopeStr);
  
  // Extract coordinates using regex
  const match = sanitized.match(/ENVELOPE\(([^)]+)\)/i);
  if (!match) throw new Error(`Invalid ENVELOPE format: ${sanitized}`);
  
  const parts = match[1].split(',').map(p => {
    const num = parseFloat(p.trim());
    if (isNaN(num)) throw new Error(`Invalid number: ${p}`);
    return num;
  });

  if (parts.length !== 4) throw new Error(`Need 4 coordinates, got ${parts.length}`);
  
  // ENVELOPE order: minLon, maxLon, maxLat, minLat
  const [minLon, maxLon, maxLat, minLat] = parts;
  
  return {
    type: 'Polygon',
    coordinates: [[
      [minLon, minLat],
      [maxLon, minLat],
      [maxLon, maxLat],
      [minLon, maxLat],
      [minLon, minLat]
    ]]
  };
}

function parsePoint(pointStr) {
  const sanitized = pointStr
    .replace(/^POINT\s*/i, '')
    .replace(/[()]/g, '')
    .trim();
  
  const coords = sanitized.split(/\s+/).map(parseFloat);
  
  if (coords.length !== 2 || coords.some(isNaN)) {
    throw new Error(`Invalid POINT coordinates: ${pointStr}`);
  }
  
  return {
    type: 'Point',
    coordinates: coords
  };
}

function parsePolygon(polygonStr) {
  const sanitized = polygonStr
    .replace(/^POLYGON\s*/i, '')
    .replace(/(\()|(\)$)/g, '')
    .trim();

  const coordPairs = sanitized.split(/,\s*/).map(pair => {
    const coords = pair.trim().split(/\s+/).map(parseFloat);
    if (coords.length !== 2 || coords.some(isNaN)) {
      throw new Error(`Invalid polygon coordinate pair: ${pair}`);
    }
    return coords;
  });

  // Close the polygon if not already closed
  if (coordPairs.length > 0 && JSON.stringify(coordPairs[0]) !== JSON.stringify(coordPairs[coordPairs.length - 1])) {
    coordPairs.push([...coordPairs[0]]);
  }

  return {
    type: 'Polygon',
    coordinates: [coordPairs]
  };
}

function parseMultiPolygon(multiPolygonStr) {
  // Sanitize input
  const sanitized = multiPolygonStr
    .replace(/^MULTIPOLYGON\s*/i, '')
    .replace(/\)\s*,\s*\(/g, '|||') // Temporary separator
    .replace(/[()]/g, '')
    .trim();

  // Split into individual polygons
  const polygons = sanitized.split('|||').filter(p => p);
  const coordinates = [];

  for (const polyStr of polygons) {
    // Split into rings (outer and potential holes)
    const rings = polyStr.split(/\),\s*\(/).map(ring => {
      const coordPairs = ring.split(/,\s*/).map(pair => {
        const [lon, lat] = pair.trim().split(/\s+/);
        const lonNum = parseFloat(lon);
        const latNum = parseFloat(lat);
        
        if (isNaN(lonNum) || isNaN(latNum)) {
          throw new Error(`Invalid coordinate pair: ${pair}`);
        }
        return [lonNum, latNum];
      });

      // Close the ring if not closed
      if (coordPairs.length > 0 && 
          !(coordPairs[0][0] === coordPairs[coordPairs.length-1][0] &&
            coordPairs[0][1] === coordPairs[coordPairs.length-1][1])) {
        coordPairs.push([...coordPairs[0]]);
      }
      
      return coordPairs;
    });

    coordinates.push(rings);
  }

  return {
    type: 'MultiPolygon',
    coordinates: coordinates
  };
}

function convertWktToGeoJson(wktString) {
  try {
    const upperWkt = wktString.toUpperCase().trim();
    
    if (upperWkt.startsWith('ENVELOPE')) {
      return parseEnvelope(wktString);
    }
    if (upperWkt.startsWith('POINT')) {
      return parsePoint(wktString);
    }
    if (upperWkt.startsWith('POLYGON')) {
      return parsePolygon(wktString);
    }
    if (upperWkt.startsWith('MULTIPOLYGON')) {
      return parseMultiPolygon(wktString);
    }
    
    throw new Error(`Unsupported geometry type: ${wktString}`);
    
  } catch (err) {
    fs.appendFileSync('wkt_errors.log', `
      ORIGINAL: ${wktString}
      ERROR: ${err.message}
    `);
    return null;
  }
}

function transformSource(docSource) {
  const newDoc = { ...docSource };

  // Process all spatial fields
  const spatialFields = [...WKT_GEOSHAPE_FIELDS, ...WKT_GEOPOINT_FIELDS];
  
  for (const field of spatialFields) {
    const value = newDoc[field];
    if (!value) continue;

    try {
      let geoJson;
      if (field === 'spatial-bounding-box') {
        geoJson = parseEnvelope(value);
      } else {
        geoJson = convertWktToGeoJson(value);
      }

      if (geoJson) {
        newDoc[`${field}-geojson`] = geoJson;
      }
    } catch (err) {
      console.error(`Field ${field} error: ${err.message}`);
      console.error(`Problem value: ${value}`);
      console.error(`id: ${newDoc['_id']}`);
    }
  }

  return newDoc;
}

async function runReindex() {
  let totalDocs = 0;
  let scrollId = null;

  try {
    const { body: searchBody } = await client.search({
      index: OLD_INDEX,
      scroll: '5m',
      size: BATCH_SIZE,
      body: { query: { match_all: {} } }
    });

    scrollId = searchBody._scroll_id;
    let hits = searchBody.hits.hits;

    while (hits && hits.length > 0) {
      const bulkOps = [];
      for (const hit of hits) {
        const transformed = transformSource(hit._source);
        bulkOps.push({ index: { _index: NEW_INDEX, _id: hit._id } });
        bulkOps.push(transformed);
      }

      const { body: bulkResponse } = await client.bulk({ body: bulkOps });
      if (bulkResponse.errors) {
        bulkResponse.items.forEach(item => {
          if (item.index.error) {
            console.error('Bulk index error:', item.index.error);
          }
        });
      }

      totalDocs += hits.length;
      console.log(`Processed ${hits.length} documents (Total: ${totalDocs})`);

      const { body: scrollBody } = await client.scroll({
        scroll_id: scrollId,
        scroll: '5m'
      });
      hits = scrollBody.hits.hits;
      scrollId = scrollBody._scroll_id;
    }
  } finally {
    if (scrollId) {
      await client.clearScroll({ scroll_id: scrollId });
    }
  }

  console.log(`Reindexing completed. Total documents processed: ${totalDocs}`);
}

runReindex()
  .then(() => process.exit(0))
  .catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
