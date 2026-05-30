"""
ACN Data Downloader & Converter
Handles downloading raw ACN charging session data from Caltech's API and 
converts the raw Excel JSON dump (acndata_sessions.json.xlsx) to clean JSON/CSV format.
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import (
    DATA_RAW,
    ACN_RAW_JSON,
    ACN_RAW_CSV
)
from src.utils import get_logger, save_csv

logger = get_logger("convert_data")

# Default API settings
ACN_API_URL = "https://ev.caltech.edu/api/v1.0/sessions"
DEFAULT_SITE = "caltech"

def download_acn_data(
    api_token: str,
    start_date: str,
    end_date: str,
    site: str = DEFAULT_SITE,
    output_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Download charging session data directly from Caltech's ACN-Data API.

    Parameters:
    -----------
    api_token : str
        Caltech ACN-Data API bearer token.
    start_date : str
        Start date in format 'YYYY-MM-DD' or RFC1123 'Thu, 01 Jan 2019 00:00:00 GMT'.
    end_date : str
        End date in format 'YYYY-MM-DD' or RFC1123 'Thu, 31 Dec 2019 23:59:59 GMT'.
    site : str
        ACN site identifier ('caltech' or 'jpl').
    output_path : Path, optional
        File path to save the downloaded JSON data.

    Returns:
    --------
    List[Dict[str, Any]]
        List of session dictionaries.
    """
    logger.info(f"Initiating ACN data download for site: '{site}' between {start_date} and {end_date}...")
    
    headers = {"Authorization": f"Bearer {api_token}"}
    url = f"{ACN_API_URL}/{site}/{start_date}/{end_date}"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract items
        sessions = data.get("_items", [])
        logger.info(f"Successfully downloaded {len(sessions)} sessions from ACN API.")
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=4)
            logger.info(f"Saved raw JSON to {output_path}")
            
        return sessions
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from ACN API: {e}")
        raise e


def convert_xlsx_to_json_csv(
    xlsx_path: Path,
    json_out_path: Path = ACN_RAW_JSON,
    csv_out_path: Path = ACN_RAW_CSV
) -> pd.DataFrame:
    """
    Convert the raw acndata_sessions.json.xlsx spreadsheet containing tabular session records
    to standardized, cleaned JSON and CSV files.

    Parameters:
    -----------
    xlsx_path : Path
        Path to the raw acndata_sessions.json.xlsx Excel file.
    json_out_path : Path
        Path where the final JSON array should be saved.
    csv_out_path : Path
        Path where the final CSV file should be saved.

    Returns:
    --------
    pd.DataFrame
        DataFrame representing the converted sessions.
    """
    logger.info(f"Opening Excel file for conversion: {xlsx_path}")
    
    if not xlsx_path.exists():
        error_msg = f"Excel file does not exist at path: {xlsx_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Read the excel file normally
    df = pd.read_excel(xlsx_path)
    logger.info(f"Excel loaded. Shape: {df.shape}")

    # Drop columns that are completely null (e.g. metadata or placeholder columns)
    initial_cols = df.columns.tolist()
    df = df.dropna(how="all", axis=1)
    dropped_cols = [c for c in initial_cols if c not in df.columns]
    if dropped_cols:
        logger.info(f"Dropped completely null columns: {dropped_cols}")

    # Clean IDs or subfields if any
    if "_id" in df.columns:
        df["_id"] = df["_id"].apply(lambda x: x.get("$oid") if isinstance(x, dict) else x)

    # 1. Save as structured JSON (list of records)
    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(json_out_path, orient="records", indent=4)
    logger.info(f"Successfully wrote clean JSON (orient='records') to: {json_out_path}")

    # 2. Save as CSV
    save_csv(df, csv_out_path, index=False)
    logger.info(f"Successfully converted and saved ACN sessions to CSV at: {csv_out_path}")

    return df


if __name__ == "__main__":
    # Test/Run local Excel conversion
    default_xlsx = DATA_RAW / "acndata_sessions.json.xlsx"
    if default_xlsx.exists():
        convert_xlsx_to_json_csv(default_xlsx)
    else:
        logger.info(f"Raw Excel not found at default path '{default_xlsx}'. Conversion skipped.")
