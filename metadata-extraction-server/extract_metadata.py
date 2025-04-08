import sys
import boto3
import rasterio
import fiona
import zipfile
import tempfile
import logging
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sanitize_tag_value(value):
    """Sanitize metadata values for Minio tags."""
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-=._:/ "
    sanitized = "".join([c if c in allowed_chars else "_" for c in str(value)])
    return sanitized[:128]

def extract_from_zip(zip_path):
    """Extract spatial files from a ZIP and return their metadata."""
    metadata = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if file.lower().endswith(('.tif', '.tiff')):
                            with rasterio.open(file_path) as src:
                                metadata[file] = {
                                    "crs": src.crs.to_string(),
                                    "bounds": list(src.bounds),
                                    "resolution": src.res
                                }
                        elif file.lower().endswith(('.shp', '.geojson')):
                            with fiona.open(file_path) as src:
                                metadata[file] = {
                                    "crs": src.crs,
                                    "schema": src.schema,
                                    "bounds": src.bounds
                                }
                    except Exception as e:
                        logging.warning(f"Failed to process {file}: {e}")
    return metadata

def extract_metadata(bucket, key):
    try:
        # Initialize S3 client for Minio
        s3 = boto3.client(
            's3',
            endpoint_url="http://i-guide-storage-dev.cis220065.projects.jetstream-cloud.org:9010",
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY_ID', ''),
            aws_secret_access_key=os.getenv('MINIO_ACCESS_KEY', ''),
            config=boto3.session.Config(signature_version='s3v4')
        )

        # Download file to temporary storage
        tmp_path = f"/tmp/{key}"
        logging.info(f"Downloading {key} from bucket {bucket}")
        s3.download_file(bucket, key, tmp_path)
        logging.info(f"Download complete: {tmp_path}")

        metadata = {}
        if key.lower().endswith('.zip'):
            logging.info("Processing ZIP file")
            metadata = extract_from_zip(tmp_path)
        elif key.lower().endswith(('.tif', '.tiff', '.shp', '.geojson')):
            logging.info("Processing spatial file")
            if key.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(tmp_path) as src:
                    metadata = {
                        "crs": src.crs.to_string(),
                        "bounds": list(src.bounds),
                        "resolution": src.res
                    }
            else:
                with fiona.open(tmp_path) as src:
                    metadata = {
                        "crs": src.crs,
                        "schema": src.schema,
                        "bounds": src.bounds
                    }
        else:
            logging.warning(f"Unsupported file format: {key}")
            return

        logging.info(f"Extracted metadata: {metadata}")

        # Sanitize and attach metadata to Minio object
        sanitized_metadata = {k: sanitize_tag_value(v) for k, v in metadata.items()}
        """s3.put_object_tagging(
            Bucket=bucket,
            Key=key,
            Tagging={'TagSet': [{'Key': k, 'Value': v} for k, v in sanitized_metadata.items()]}
        )"""
        logging.info("Metadata extracted successfully")

    except NoCredentialsError:
        logging.error("Missing Minio credentials.")
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        logging.error("Usage: python extract_metadata.py <bucket> <key>")
        sys.exit(1)
    bucket, key = sys.argv[1], sys.argv[2]
    extract_metadata(bucket, key)