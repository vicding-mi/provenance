import os
import argparse
import pytz
import logging
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

"""
Define all the static values
"""
who_person = "REPUBLIC project"
where_location = None
how_delta = None
why_motivation = "HTR or OCR on the scan to text"
how_init = None


class SourceType(Enum):
    SOURCE = "source_resources"
    TARGET = "target_resources"


@dataclass
class Provenance:
    who_person: str
    where_location: Optional[str]
    when_time: str
    when_timestamp: str
    how_software: str
    how_delta: Optional[str]
    why_motivation: str
    why_provenance_schema: str
    how_init: Optional[str]

    def __init__(self, who_person: str, where_location: Optional[str], when_time: str, when_timestamp: str,
                 how_software: str, how_delta: Optional[str], why_motivation: str, why_provenance_schema: str,
                 how_init: Optional[str]):
        self.who_person = who_person
        self.where_location = where_location
        self.when_time = when_time
        self.when_timestamp = when_timestamp
        self.how_software = how_software
        self.how_delta = how_delta
        self.why_motivation = why_motivation
        self.why_provenance_schema = why_provenance_schema
        self.how_init = how_init

    def generate_insert_query(self) -> str:
        return f"INSERT INTO provenance (who_person, where_location, when_time, when_timestamp, how_software, how_delta, why_motivation, why_provenance_schema, how_init) VALUES ('{self.who_person}', {self.where_location if self.where_location is not None else 'NULL'}, '{self.when_time}', '{self.when_timestamp}', '{self.how_software}', {self.how_delta if self.how_delta is not None else 'NULL'}, '{self.why_motivation}', '{self.why_provenance_schema}', {self.how_init if self.how_init is not None else 'NULL'}) RETURNING id INTO last_inserted_id;"


@dataclass
class Source:
    source_type: SourceType
    prov_id: int
    res: str
    rel: str

    def __init__(self, source_type: SourceType, prov_id: int, res: str, rel: str = "primary"):
        self.source_type = source_type
        self.prov_id = prov_id
        self.res = res
        self.rel = rel

    def generate_insert_query(self) -> str:
        return f"INSERT INTO {self.source_type.value} (prov_id, res, rel) VALUES (last_inserted_id, '{self.res}', '{self.rel}');"


def create_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Create file handler and set level to info
    if not os.path.exists('log'):
        os.makedirs('log')
    fh = logging.FileHandler('log/application.log')
    fh.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to handlers
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


logger = create_logger(__name__)


def load_csv(file_path: str, has_header: bool = True) -> pd.DataFrame:
    header = 0 if has_header else None
    result = pd.read_csv(file_path, header=header)
    return result


def parse_page_number(page_number: str) -> list:
    elements = page_number.split('-')
    if len(elements) != 2:
        raise ValueError("Invalid page number format")
    elements = elements[1].split("_")
    if len(elements) != 4:
        raise ValueError("Invalid page number format")
    str_archive_number: str = "_".join(elements[:2])
    str_inventory_numbe: str = elements[2]
    str_page_number: str = elements[3].split(".")[0]
    return [str_archive_number, str_inventory_numbe, str_page_number]


def get_args():
    parser = argparse.ArgumentParser(description='Process some CSV files.')
    parser.add_argument("--output", help="Output file name", required=False, default="output.sql")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


def adjust_time_string(time_str: str, delta: timedelta):
    try:
        # Try to parse the string as a datetime with timezone
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("Invalid time string format")

    # Adjust the datetime by the given delta
    adjusted_dt = dt + delta

    # Return the new value as a datetime object and a timestamp
    return adjusted_dt.timestamp()


def string_to_timestamp(time_str: str) -> float:
    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    return dt.timestamp()


def timestamp_to_string(timestamp: float) -> str:
    dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
    return dt.isoformat().replace("+00:00", "Z")


def date_string_to_unix_timestamp(date_str: str) -> float:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(tzinfo=pytz.UTC)  # Set the time zone to UTC
    return dt.timestamp()


def convert_unix_timestamp_to_postgres_timestamp(str_when_unix_timestamp) -> str:
    dt = datetime.fromtimestamp(float(str_when_unix_timestamp), tz=pytz.UTC)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f%z')


def init() -> tuple:
    """
    Loading the CSV files and returning the DataFrames in a tuple

    :return:
    """
    logger.info("Initializing...")
    # this file contains extracted trails info from postgreSQL
    loghi_trails_csv = "data-1733497149273.csv"
    # This file contains IIIF urls to scans relations
    iiif_csv = "allfiles.csv"
    df = load_csv(loghi_trails_csv)
    iiif_df = load_csv(iiif_csv, has_header=False)
    return df, iiif_df


def merge_dfs(df: pd.DataFrame, iiif_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merging the DataFrames based on the res_part column

    :param df:
    :param iiif_df:
    :return:
    """
    logger.info("Merging DataFrames")
    # Create a new column in iiif_df with the part of the string to match
    iiif_df['res_part'] = iiif_df[1].str.extract(r'([^/]+)\.jpg$')

    # Create a new column in df with the part of the string to match
    df['res_part'] = df['res'].str.extract(r'([^/]+)\.xml$')

    # Merge the DataFrames based on the new column
    merged_df = pd.merge(df, iiif_df, left_on='res_part', right_on='res_part', how='left')
    return merged_df


def process_row(row: pd.Series) -> tuple:
    """
    Process a single row from the DataFrame

    who_person = "REPUBLIC project"
    where_location = None
    how_delta = None
    why_motivation = "HTR or OCR on the scan to text"
    how_init = None

    :param row:
    :return:
    """
    logger.debug(f"Processing row {row.name}")

    prov: Optional[Provenance] = None
    source_sources: Optional[Source] = None
    source_targets: Optional[Source] = None

    str_archive_number, str_inventory_number, str_page_number = parse_page_number(row.res)

    if pd.notna(row[0]):
        iiif_url = row[0]
        logger.debug(f"IIIF URL: {iiif_url}")
    else:
        logger.error(f"IIIF URL not found for {row['res']}")
        return None, None, None

    if str_archive_number == "HaNA_1.01.02" and (
            3097 <= int(str_inventory_number) <= 3347 or 4542 <= int(str_inventory_number) <= 4862):
        # case A
        """
        For case A, the date is set to a static value of "2024-07-27"
        """
        str_org_when_time: str = "2024-07-27"
        str_when_unix_timestamp = date_string_to_unix_timestamp(str_org_when_time)
        str_how_software = "https://github.com/knaw-huc/loghi/releases/tag/2.1.0"
        str_why_provenance_schema = """data:application/json;charset=UTF-8,{"version": "version 2.1.0 for laypa/loghi-htr/loghi-tooling", "used_models": "baseline2", "used_htr_models": "republic-recommended-batch-8-2024-06-16", "steps_run": "baseline-detection, loghi-htr, recalculatereadingorder, detectlanguage, splitwords"}"""
    elif str_archive_number == "HaNA_1.10.94" and str_inventory_number in ("455", "456"):
        # case B
        """
        For case B, the date is set to a static value of "2024-08-10"
        """
        str_org_when_time: str = "2024-08-10"
        str_when_unix_timestamp = date_string_to_unix_timestamp(str_org_when_time)
        str_how_software = "https://github.com/knaw-huc/loghi/releases/tag/2.1.2"
        str_why_provenance_schema = """data:application/json;charset=UTF-8,{"version": "version 2.1.2 for laypa/loghi-htr/loghi-tooling", "used_models": "RUN_2024-07-22_17-50-05_republicprint-baseline / RUN_2024-07-26_17-12-52_republicprint-regions", "used_htr_models": "republic-print-2024-08-10-5epochs", "steps_run": "region-detection, baseline-detection, loghi-htr, detectlanguage, splitwords"}"""
    else:
        # case C (fallback)
        """
        For case C, the date is set to the previous day of the original date
        which is extracted from the when_time column
        """
        str_when_unix_timestamp = adjust_time_string(row.when_time, timedelta(days=-1))
        str_how_software = "https://github.com/knaw-huc/loghi"
        str_why_provenance_schema = """data:application/json;charset=UTF-8,{"version": "P2PaLa: private version / Loghi-htr: 0.x / Loghi-tooling: 0.x", "used_models": "baselines: P2PaLa-republic-baselines / regions: P2PaLa-republic-regions", "used_htr_models": "manuscripts: republic-2023-01-02-base-generic_new14-2022-12-20 prints: model-new7-republicprint-height48-cer-0.0009"}"""

    str_when_time = timestamp_to_string(str_when_unix_timestamp)
    str_when_timestamp = convert_unix_timestamp_to_postgres_timestamp(str_when_unix_timestamp)

    prov = Provenance(who_person=who_person, where_location=where_location, when_time=str_when_time,
                      when_timestamp=str_when_timestamp, how_software=str_how_software, how_delta=how_delta,
                      why_motivation=why_motivation, why_provenance_schema=str_why_provenance_schema,
                      how_init=how_init)
    # TODO: prov_id should be the last inserted id, for testing set to 1, remove the line below
    # TODO: write a function to generate transaction queries which use the last inserted id
    prov_id = 1

    source_sources = Source(SourceType.SOURCE, prov_id, iiif_url)
    source_targets = Source(SourceType.TARGET, prov_id, row.res)
    return prov, source_sources, source_targets


def generate_sql_transaction(prov: Provenance, source_resources: Source, target_resources: Source) -> str:
    return f"""
    -- {prov.why_motivation}; {prov.when_time}; {prov.who_person}; {prov.how_software}; ### {source_resources.res} -> {target_resources.res} ###
    DO $$
    DECLARE
        last_inserted_id INTEGER;
    BEGIN
        {prov.generate_insert_query()}
        {source_resources.generate_insert_query()}
        {target_resources.generate_insert_query()}
    END $$;
    """


def process_rows(rows: pd.DataFrame, output_file: str) -> None:
    logger.info(f"Processing {len(rows)} records")
    with open(output_file, 'a') as f:
        for index, row in tqdm(rows.iterrows(), total=len(rows)):
            prov, source_resources, target_resources = process_row(row)
            if prov and source_resources and target_resources:
                str_sql_query = generate_sql_transaction(prov, source_resources, target_resources)
                f.write(f"{str_sql_query}\n")


def init_output_file(output_file: str) -> None:
    with open(output_file, 'w') as f:
        f.write("")


if __name__ == '__main__':
    args = get_args()
    # load the csv files
    df, iiif_df = init()
    # merge the dataframes for performance
    merged_df = merge_dfs(df, iiif_df)

    if args.debug:
        logger.info("Debugging ON!!!")
        rows = merged_df.head(5)
    else:
        rows = merged_df

    # create sql output file
    init_output_file(args.output)

    # process the rows and add the sql queries to the output file
    process_rows(rows, args.output)

    logger.info(f"Done processing {len(rows)} records")
