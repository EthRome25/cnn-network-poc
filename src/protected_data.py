# dataprotector deserializer module
import os
import zipfile
from borsh_construct import String, I128, F64, Bool


def getRawValue(path: str) -> bytes:
    IEXEC_IN = os.getenv('IEXEC_IN')
    IEXEC_DATASET_FILENAME = os.getenv('IEXEC_DATASET_FILENAME')

    if IEXEC_DATASET_FILENAME == None:
        raise Exception('Missing protected data')

    # For direct file access (not ZIP), try to read the file directly
    direct_file_path = os.path.join(IEXEC_IN, path)
    if os.path.exists(direct_file_path):
        with open(direct_file_path, 'rb') as f:
            return f.read()
    
    # Fallback to ZIP archive approach
    dataset_file_path = os.path.join(IEXEC_IN, IEXEC_DATASET_FILENAME)
    if os.path.exists(dataset_file_path):
        with open(dataset_file_path, 'rb') as f:
            return f.read()
    
    raise Exception(f"File not found: {path}")

def getValue(path: str, schema: str):
    file_path = path.replace('.', '/')
    IEXEC_IN = os.getenv('IEXEC_IN')
    IEXEC_DATASET_FILENAME = os.getenv('IEXEC_DATASET_FILENAME')

    if IEXEC_DATASET_FILENAME == None:
        raise Exception('Missing protected data')

    dataset_file_path = os.path.join(IEXEC_IN, IEXEC_DATASET_FILENAME)

    file_bytes: bytes
    try:
        # Open the ZIP archive
        with zipfile.ZipFile(dataset_file_path, 'r') as zipf:
            # Read the file from the ZIP archive as bytes
            with zipf.open(file_path) as file:
                file_bytes = file.read()
    except:
        raise Exception(f"Failed to load path {path}")

    try:
        if schema == 'bool':
            return Bool.parse(file_bytes)
        if schema == 'f64':
            return F64.parse(file_bytes)
        if schema == 'i128':
            return I128.parse(file_bytes)
        if schema == 'string':
            return String.parse(file_bytes)
    except:
        raise Exception(f"Failed to deserialize \"{path}\" as \"{schema}\"")

    return file_bytes
