import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import requests
import io
import os
import zipfile
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import types
import datetime

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oil_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('fivethirtyeight')
sns.set_palette("viridis")


class APIClient:
    def __init__(self):
        self.session = requests.Session()

        # Set a legitimate user agent to prevent 403 errors
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml,application/json;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # API configuration
        self.statcan_base_url = "https://www150.statcan.gc.ca"
        self.oecd_base_url = "https://stats.oecd.org/SDMX-JSON/data"
        self.worldbank_base_url = "http://api.worldbank.org/v2"

    def get_with_retry(self, url, params=None, headers=None):
        try:
            # Merge provided headers with session headers if supplied
            if headers:
                merged_headers = self.session.headers.copy()
                merged_headers.update(headers)
                headers = merged_headers
            else:
                headers = None  # Use session default headers

            response = self.session.get(
                url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url}")
            logger.error(f"Error details: {str(e)}")
            raise


class DataCollector:
    def __init__(self):
        self.data_dir = "data"
        self.cache_dir = "cache"
        self.api_client = APIClient()
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")

    def download_file(self, url, filename):
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            try:
                logger.info(f"Downloading file from {url}")
                response = self.api_client.get_with_retry(url)

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Successfully downloaded {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {str(e)}")
                raise
        else:
            logger.info(f"Using existing file: {filename}")

        return filepath

    def save_to_cache(self, df, filename):
        """Save DataFrame to cache"""
        filepath = os.path.join(self.cache_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved data to cache: {filename}")
        return filepath

    def load_from_cache(self, filename):
        """Load DataFrame from cache if it exists"""
        filepath = os.path.join(self.cache_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Convert date column to datetime
                if 'REF_DATE' in df.columns:
                    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
                logger.info(f"Loaded data from cache: {filename}")
                return df
            except Exception as e:
                logger.error(f"Failed to load cache file {filename}: {str(e)}")
        return None

    def get_statcan_data(self, table_id):
        """Get data from Statistics Canada using their public CSV files"""
        logger.info(f"Fetching Statistics Canada data for table {table_id}")

        # Convert table ID to format without hyphens
        table_id_clean = table_id.replace('-', '')

        try:
            # Try the direct CSV download URL first (more reliable)
            csv_url = f"https://www150.statcan.gc.ca/n1/tbl/csv/{table_id_clean}-eng.zip"
            logger.info(f"Trying direct CSV download: {csv_url}")

            try:
                response = self.api_client.get_with_retry(csv_url)

                # Extract the CSV file from the ZIP archive
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    # Get the first CSV file in the archive
                    csv_file_name = [
                        name for name in zip_file.namelist() if name.endswith('.csv')][0]
                    with zip_file.open(csv_file_name) as csv_file:
                        df = pd.read_csv(csv_file)

                if 'REF_DATE' in df.columns:
                    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

                logger.info(
                    f"Successfully retrieved data for table {table_id} using CSV download")
                logger.info(f"Retrieved columns: {df.columns.tolist()}")
                return df

            except Exception as e:
                logger.warning(
                    f"CSV download failed: {str(e)}, trying alternative methods")

                # Try alternative URL format
                alt_url = f"https://www150.statcan.gc.ca/t1/tbl1/en/csv/{table_id_clean}-eng.csv"
                logger.info(f"Trying alternative URL: {alt_url}")

                response = self.api_client.get_with_retry(alt_url)
                df = pd.read_csv(io.StringIO(response.text))

                if 'REF_DATE' in df.columns:
                    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

                logger.info(
                    f"Successfully retrieved data for table {table_id} using alternative URL")
                logger.info(f"Retrieved columns: {df.columns.tolist()}")
                return df

        except Exception as e:
            logger.error(
                f"Failed to fetch Statistics Canada data using all methods: {str(e)}")
            # Try the third URL format as a last resort
            try:
                third_url = f"https://www150.statcan.gc.ca/n1/pub/{table_id_clean[:2]}-{table_id_clean[2:4]}/{table_id_clean}-eng.csv"
                logger.info(f"Trying third URL format: {third_url}")

                response = self.api_client.get_with_retry(third_url)
                df = pd.read_csv(io.StringIO(response.text))

                if 'REF_DATE' in df.columns:
                    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

                logger.info(
                    f"Successfully retrieved data for table {table_id} using third URL format")
                logger.info(f"Retrieved columns: {df.columns.tolist()}")
                return df

            except Exception as e2:
                logger.error(
                    f"Failed to fetch Statistics Canada data using third URL format: {str(e2)}")
                raise ValueError(
                    f"Could not retrieve data for Statistics Canada table {table_id} using any method")

    def get_worldbank_data(self, indicator, countries=None, start_year=1950, end_year=None):
        """Get data from World Bank API for a specific indicator with expanded date range

        Args:
            indicator: World Bank indicator code
            countries: Single country code or list of country codes. Default: "CAN,USA,GBR,FRA,DEU,ITA,JPN,CHN" (G7 + China)
            start_year: Starting year for data collection
            end_year: Ending year for data collection (defaults to current year)
        """
        # Set end_year to current year if not specified
        if end_year is None:
            end_year = datetime.datetime.now().year

        # If countries is None, use the default G7 nations plus China
        if countries is None:
            # Note: World Bank API expects semicolon-separated country codes in URL
            countries_list = ["CAN", "USA", "GBR", "FRA",
                              "DEU", "ITA", "JPN", "CHN"]  # G7 + China
        elif isinstance(countries, str):
            if "," in countries:
                countries_list = countries.split(",")
            else:
                countries_list = [countries]
        elif isinstance(countries, list):
            countries_list = countries
        else:
            raise ValueError(
                f"Unsupported countries parameter type: {type(countries)}")

        # Join with semicolons for World Bank API
        countries_param = ";".join(countries_list)

        # Provide mapping from country codes to names for readability
        country_names = {
            "CAN": "Canada",
            "USA": "United States",
            "GBR": "United Kingdom",
            "FRA": "France",
            "DEU": "Germany",
            "ITA": "Italy",
            "JPN": "Japan",
            "CHN": "China"
        }

        # Single country mode for backward compatibility
        if len(countries_list) == 1 and countries_list[0] == "CAN":
            logger.info(
                f"Fetching World Bank data for indicator: {indicator} from {start_year} to {end_year}")

            try:
                records = []
                page = 1
                more_pages = True

                while more_pages:
                    url = f"{self.api_client.worldbank_base_url}/country/{countries_list[0]}/indicator/{indicator}"
                    params = {
                        "format": "json",
                        "date": f"{start_year}:{end_year}",
                        "per_page": 1000,
                        "page": page
                    }

                    response = self.api_client.get_with_retry(
                        url, params=params)
                    data = response.json()

                    if isinstance(data, list) and len(data) > 1:
                        api_data = data[1]

                        # Check if we're done paging
                        if len(api_data) == 0:
                            more_pages = False
                        else:
                            # Create DataFrame from the API response
                            for item in api_data:
                                if item['value'] is not None:
                                    records.append({
                                        'REF_DATE': pd.to_datetime(item['date'], format='%Y'),
                                        'VALUE': float(item['value'])
                                    })

                            # Move to next page
                            page += 1
                    else:
                        more_pages = False
                        if isinstance(data, list) and len(data) > 0 and 'message' in data[0]:
                            logger.error(
                                f"World Bank API error for indicator {indicator}: {data[0]['message']}")
                            raise ValueError(
                                f"World Bank API error: {data[0]['message']}")

                # Check if we have any records after processing all pages
                if records:
                    df = pd.DataFrame(records)
                    df = df.sort_values('REF_DATE')
                    logger.info(
                        f"Successfully retrieved World Bank data for {indicator}: {len(df)} records from {df['REF_DATE'].min().year} to {df['REF_DATE'].max().year}")
                    return df
                else:
                    raise ValueError(
                        f"No valid data records found in World Bank API response for {indicator}")

            except Exception as e:
                logger.error(
                    f"Failed to get data from World Bank for indicator {indicator}: {str(e)}")
                raise

        # Multiple countries case
        else:
            logger.info(
                f"Fetching World Bank data for indicator: {indicator} for multiple countries from {start_year} to {end_year}")

            try:
                all_records = []
                # For multiple countries, use the countries parameter
                url = f"{self.api_client.worldbank_base_url}/countries/{countries_param}/indicators/{indicator}"
                params = {
                    "format": "json",
                    "date": f"{start_year}:{end_year}",
                    "per_page": 15000  # Increase to get more records at once for multiple countries
                }

                logger.info(
                    f"Making API request to URL: {url} with params: {params}")
                response = self.api_client.get_with_retry(url, params=params)
                data = response.json()

                if isinstance(data, list) and len(data) > 1:
                    api_data = data[1]

                    # Create DataFrame from the API response
                    for item in api_data:
                        if item['value'] is not None:
                            country_code = item['countryiso3code'] if 'countryiso3code' in item else item['country']['id']
                            country_name = country_names.get(
                                country_code, item['country']['value'])

                            all_records.append({
                                'REF_DATE': pd.to_datetime(item['date'], format='%Y'),
                                'VALUE': float(item['value']),
                                'COUNTRY_CODE': country_code,
                                'COUNTRY': country_name
                            })
                else:
                    if isinstance(data, list) and len(data) > 0 and 'message' in data[0]:
                        logger.error(
                            f"World Bank API error for indicator {indicator}: {data[0]}")
                        raise ValueError(
                            f"World Bank API error: {data[0]}")

                # Check if we have any records
                if all_records:
                    df = pd.DataFrame(all_records)
                    df = df.sort_values(['COUNTRY', 'REF_DATE'])

                    # Count records per country
                    country_counts = df.groupby('COUNTRY').size().to_dict()
                    countries_str = ", ".join(
                        [f"{country}: {count}" for country, count in country_counts.items()])

                    logger.info(
                        f"Successfully retrieved World Bank data for {indicator}: {len(df)} total records for countries: {countries_str}")
                    return df
                else:
                    raise ValueError(
                        f"No valid data records found in World Bank API response for {indicator}")

            except Exception as e:
                logger.error(
                    f"Failed to get data from World Bank for indicator {indicator}: {str(e)}")
                raise

    def get_oil_production_data(self):
        logger.info("Fetching oil production data")

        # Try to load from cache first
        cached_data = self.load_from_cache("oil_production.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Use Statistics Canada API - Table 25-10-0063 (Monthly crude oil production)
            df = self.get_statcan_data("25-10-0063-01")

            if df is not None:
                # Filter for crude oil production
                if 'North American Product Database (NAPD)' in df.columns:
                    df = df[df['North American Product Database (NAPD)'].str.contains(
                        'crude oil', case=False, na=False)]
                elif 'Products' in df.columns:
                    df = df[df['Products'].str.contains(
                        'crude oil', case=False, na=False)]

                # Handle different column names
                if 'VALUE' in df.columns:
                    value_col = 'VALUE'
                elif 'Value' in df.columns:
                    value_col = 'Value'
                else:
                    # Try to find a numeric column
                    numeric_cols = df.select_dtypes(
                        include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        value_col = numeric_cols[0]
                    else:
                        raise ValueError("No numeric value column found")

                if 'REF_DATE' in df.columns:
                    date_col = 'REF_DATE'
                elif 'Ref_Date' in df.columns:
                    date_col = 'Ref_Date'
                else:
                    # Try to find a date-like column
                    date_cols = [
                        col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
                    if len(date_cols) > 0:
                        date_col = date_cols[0]
                    else:
                        raise ValueError("No date column found")

                df['VALUE'] = pd.to_numeric(df[value_col], errors='coerce')
                df['REF_DATE'] = pd.to_datetime(df[date_col], errors='coerce')

                # Calculate annual values
                df['year'] = df['REF_DATE'].dt.year
                annual_data = df.groupby('year')['VALUE'].mean().reset_index()
                annual_data['REF_DATE'] = pd.to_datetime(
                    annual_data['year'], format='%Y')

                logger.info(
                    "Successfully retrieved oil production data from Statistics Canada")

                # Save to cache
                self.save_to_cache(
                    annual_data[['REF_DATE', 'VALUE']], "oil_production.csv")

                return annual_data[['REF_DATE', 'VALUE']]

        except Exception as e:
            logger.error(
                f"Failed to get oil production data from Statistics Canada: {str(e)}")

            # Try NRCan data as a backup
            logger.info("Attempting to fetch from NRCan")
            try:
                # Updated NRCan URL - check the actual URL in use
                url = "https://natural-resources.canada.ca/sites/nrcan/files/energy/crude-oil-facts_latest.csv"
                response = self.api_client.get_with_retry(url)
                df = pd.read_csv(io.StringIO(response.text))

                # Process NRCan data (format will vary)
                if 'Year' in df.columns and 'Production' in df.columns:
                    df['REF_DATE'] = pd.to_datetime(df['Year'], format='%Y')
                    df['VALUE'] = pd.to_numeric(
                        df['Production'], errors='coerce')

                    logger.info(
                        "Successfully retrieved oil production data from NRCan")

                    # Save to cache
                    self.save_to_cache(
                        df[['REF_DATE', 'VALUE']], "oil_production.csv")

                    return df[['REF_DATE', 'VALUE']]

            except Exception as e:
                logger.error(
                    f"Failed to get oil production data from NRCan: {str(e)}")

                # Try World Bank API as final backup
                logger.info("Attempting to fetch from World Bank API")
                try:
                    # Use World Bank data for oil production (percentage of GDP)
                    df = self.get_worldbank_data("NY.GDP.PETR.RT.ZS")

                    # Save to cache
                    self.save_to_cache(df, "oil_production.csv")

                    return df

                except Exception as e:
                    logger.error(
                        f"Failed to get oil production data from World Bank: {str(e)}")

                    # Create fallback dummy data if all else fails
                    logger.warning(
                        "Using fallback dummy data for oil production")
                    years = range(2010, 2023)
                    records = [{'REF_DATE': pd.to_datetime(
                        str(year)), 'VALUE': 4.5 + 0.1 * (year - 2010)} for year in years]
                    df = pd.DataFrame(records)

                    # Save to cache
                    self.save_to_cache(df, "oil_production.csv")

                    return df

    def get_gdp_data(self):
        logger.info("Fetching GDP data")

        # Try to load from cache first
        cached_data = self.load_from_cache("gdp.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Use Statistics Canada Table 36-10-0434 (GDP by industry)
            df = self.get_statcan_data("36-10-0434-01")

            if df is not None:
                # Process the data
                if 'GEO' in df.columns:
                    df = df[df['GEO'] == 'Canada']

                # Handle different column names
                if 'VALUE' in df.columns:
                    value_col = 'VALUE'
                elif 'Value' in df.columns:
                    value_col = 'Value'
                else:
                    # Try to find a numeric column
                    numeric_cols = df.select_dtypes(
                        include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        value_col = numeric_cols[0]
                    else:
                        raise ValueError("No numeric value column found")

                if 'REF_DATE' in df.columns:
                    date_col = 'REF_DATE'
                elif 'Ref_Date' in df.columns:
                    date_col = 'Ref_Date'
                else:
                    # Try to find a date-like column
                    date_cols = [
                        col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
                    if len(date_cols) > 0:
                        date_col = date_cols[0]
                    else:
                        raise ValueError("No date column found")

                df['VALUE'] = pd.to_numeric(df[value_col], errors='coerce')
                df['REF_DATE'] = pd.to_datetime(df[date_col], errors='coerce')

                # Calculate annual values
                df['year'] = df['REF_DATE'].dt.year
                annual_data = df.groupby('year')['VALUE'].sum().reset_index()
                annual_data['REF_DATE'] = pd.to_datetime(
                    annual_data['year'], format='%Y')

                # Convert to trillions
                annual_data['VALUE'] = annual_data['VALUE'] / 1e6

                logger.info(
                    "Successfully processed GDP data from Statistics Canada")

                # Save to cache
                self.save_to_cache(
                    annual_data[['REF_DATE', 'VALUE']], "gdp.csv")

                return annual_data[['REF_DATE', 'VALUE']]

        except Exception as e:
            logger.error(
                f"Failed to get GDP data from Statistics Canada: {str(e)}")

            # Try World Bank API as backup
            logger.info("Attempting to fetch GDP data from World Bank API")
            try:
                df = self.get_worldbank_data("NY.GDP.MKTP.CD")

                # Convert to trillions
                df['VALUE'] = df['VALUE'] / 1e12

                # Save to cache
                self.save_to_cache(df, "gdp.csv")

                return df

            except Exception as e:
                logger.error(
                    f"Failed to get GDP data from World Bank: {str(e)}")

                # Create fallback dummy data if all else fails
                logger.warning("Using fallback dummy data for GDP")
                years = range(2010, 2023)
                records = [{'REF_DATE': pd.to_datetime(
                    str(year)), 'VALUE': 1.8 + 0.05 * (year - 2010)} for year in years]
                df = pd.DataFrame(records)

                # Save to cache
                self.save_to_cache(df, "gdp.csv")

                return df

    def get_energy_price_data(self):
        logger.info("Fetching energy price data")

        # Try to load from cache first
        cached_data = self.load_from_cache("energy_prices.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Use Statistics Canada Table 18-10-0004 (Consumer Price Index)
            df = self.get_statcan_data("18-10-0004-01")

            if df is not None:
                # Filter and process the data
                if 'GEO' in df.columns:
                    df = df[df['GEO'] == 'Canada']
                # Try different possible column names for energy
                if 'Products and product groups' in df.columns:
                    if 'Energy' in df['Products and product groups'].values:
                        df = df[df['Products and product groups'] == 'Energy']
                    elif 'Energy [45]' in df['Products and product groups'].values:
                        df = df[df['Products and product groups']
                                == 'Energy [45]']
                    else:
                        # Try to find any energy-related product
                        energy_filter = df['Products and product groups'].str.contains(
                            'energy', case=False, na=False)
                        if energy_filter.any():
                            df = df[energy_filter]

                # Handle different column names
                if 'VALUE' in df.columns:
                    value_col = 'VALUE'
                elif 'Value' in df.columns:
                    value_col = 'Value'
                else:
                    # Try to find a numeric column
                    numeric_cols = df.select_dtypes(
                        include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        value_col = numeric_cols[0]
                    else:
                        raise ValueError("No numeric value column found")

                if 'REF_DATE' in df.columns:
                    date_col = 'REF_DATE'
                elif 'Ref_Date' in df.columns:
                    date_col = 'Ref_Date'
                else:
                    # Try to find a date-like column
                    date_cols = [
                        col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
                    if len(date_cols) > 0:
                        date_col = date_cols[0]
                    else:
                        raise ValueError("No date column found")

                df['VALUE'] = pd.to_numeric(df[value_col], errors='coerce')
                df['REF_DATE'] = pd.to_datetime(df[date_col], errors='coerce')

                # Calculate annual averages
                df['year'] = df['REF_DATE'].dt.year
                annual_avg = df.groupby('year')['VALUE'].mean().reset_index()
                annual_avg['REF_DATE'] = pd.to_datetime(
                    annual_avg['year'], format='%Y')

                logger.info(
                    "Successfully processed energy price data from Statistics Canada")

                # Save to cache
                self.save_to_cache(
                    annual_avg[['REF_DATE', 'VALUE']], "energy_prices.csv")

                return annual_avg[['REF_DATE', 'VALUE']]

        except Exception as e:
            logger.error(
                f"Failed to get energy price data from Statistics Canada: {str(e)}")

            # Try Natural Resources Canada as a backup
            logger.info("Attempting to fetch from Natural Resources Canada")
            try:
                # Updated NRCan URL for energy prices
                url = "https://natural-resources.canada.ca/sites/nrcan/files/energy/energy_prices_latest.csv"
                response = self.api_client.get_with_retry(url)
                df = pd.read_csv(io.StringIO(response.text))

                if 'Year' in df.columns and 'Energy_Price_Index' in df.columns:
                    df['REF_DATE'] = pd.to_datetime(df['Year'], format='%Y')
                    df['VALUE'] = pd.to_numeric(
                        df['Energy_Price_Index'], errors='coerce')

                    logger.info(
                        "Successfully retrieved energy price data from NRCan")

                    # Save to cache
                    self.save_to_cache(
                        df[['REF_DATE', 'VALUE']], "energy_prices.csv")

                    return df[['REF_DATE', 'VALUE']]
                else:
                    raise ValueError(
                        "Unexpected column format in Climate Institute emissions data")

            except Exception as e:
                logger.error(
                    f"Failed to get emissions data from Climate Institute: {str(e)}")

                # Try Environment and Climate Change Canada with better error handling
                try:
                    url = "https://data.ec.gc.ca/data/substances/monitor/canada-s-official-greenhouse-gas-inventory/UNFCCC-NIR/GHG_IPCC_Can_Prov_Terr.csv"
                    logger.info(
                        f"Attempting to fetch emissions data from Environment Canada: {url}")
                    response = self.api_client.get_with_retry(url)

                    # We've had issues with this file, so let's first examine it
                    content = response.text
                    # Examine first 20 lines
                    sample_lines = content.splitlines()[:20]
                    logger.info(
                        f"Sample of Environment Canada CSV file:\n{sample_lines}")

                    # Try multiple parsing approaches
                    parsing_methods = [
                        {"method": "standard", "params": {"on_bad_lines": 'skip'}},
                        {"method": "old_pandas", "params": {
                            "error_bad_lines": False}},
                        {"method": "python_engine", "params": {
                            "engine": 'python', "on_bad_lines": 'skip'}},
                        {"method": "python_engine_skiprows", "params": {
                            # Skip problematic rows
                            "engine": 'python', "skiprows": [7, 8]}}
                    ]

                    df = None
                    for method in parsing_methods:
                        try:
                            logger.info(
                                f"Trying parsing method: {method['method']}")
                            if method["method"] == "old_pandas":
                                try:
                                    df = pd.read_csv(io.StringIO(
                                        content), **method["params"])
                                    break
                                except TypeError:
                                    logger.info(
                                        "Old pandas method not supported, skipping")
                                    continue
                            else:
                                df = pd.read_csv(io.StringIO(
                                    content), **method["params"])
                                break
                        except Exception as parse_err:
                            logger.warning(
                                f"Parsing method {method['method']} failed: {str(parse_err)}")
                            continue

                    if df is None:
                        # Last resort: try with C parser and specific delimiter
                        try:
                            logger.info(
                                "Trying CSV parsing with explicit delimiter detection")
                            # Try to detect the delimiter
                            for delimiter in [',', ';', '\t', '|']:
                                try:
                                    df = pd.read_csv(io.StringIO(content), sep=delimiter, engine='c',
                                                     on_bad_lines='skip', encoding='utf-8')
                                    # If we got multiple columns, we found the right delimiter
                                    if len(df.columns) > 1:
                                        logger.info(
                                            f"Successfully parsed CSV with delimiter: '{delimiter}'")
                                        break
                                except:
                                    continue
                        except Exception as e:
                            logger.error(
                                f"All CSV parsing methods failed: {str(e)}")
                            raise ValueError(
                                "Unable to parse Environment Canada CSV file")

                    if df is None:
                        raise ValueError("Failed to parse CSV with any method")

                    # Log the columns we found to help with debugging
                    logger.info(f"Parsed CSV columns: {df.columns.tolist()}")

                    # First check standard column names
                    if 'Year' in df.columns:
                        # Filter for total national emissions
                        if 'Region' in df.columns:
                            df = df[df['Region'] == 'Canada']

                        if 'Total GHG' in df.columns:
                            df['REF_DATE'] = pd.to_datetime(
                                df['Year'], format='%Y')
                            df['VALUE'] = pd.to_numeric(
                                df['Total GHG'], errors='coerce')

                            # Drop rows with NaN values that might have resulted from parsing errors
                            df = df.dropna(subset=['VALUE'])

                            # Group by year if needed
                            annual_data = df.groupby('REF_DATE')[
                                'VALUE'].sum().reset_index()

                            logger.info(
                                "Successfully retrieved emissions data from Environment Canada")

                            # Save to cache
                            self.save_to_cache(
                                annual_data, "emissions.csv")

                            return annual_data

                    # Fallback to intelligent column detection if standard columns aren't found
                    # Identify year column
                    year_columns = [col for col in df.columns if
                                    any(year_term in col.lower() for year_term in ['year', 'date', 'yr', 'année'])]

                    # Identify GHG columns
                    ghg_columns = [col for col in df.columns if
                                   any(ghg_term in col.lower() for ghg_term in
                                       ['ghg', 'emission', 'co2', 'carbon', 'total'])]

                    if year_columns and ghg_columns:
                        # Use the first matching columns
                        year_col = year_columns[0]
                        ghg_col = ghg_columns[0]

                        logger.info(
                            f"Using columns: Year={year_col}, GHG={ghg_col}")

                        # Filter for Canada if region column exists
                        if any(region_term in col.lower() for col in df.columns
                               for region_term in ['region', 'country', 'province']):
                            region_col = [col for col in df.columns if
                                          any(region_term in col.lower() for region_term in
                                              ['region', 'country', 'province'])][0]

                            # Try to filter for Canada data
                            canada_values = ['canada', 'can',
                                             'ca', 'national', 'total']
                            canada_mask = df[region_col].astype(
                                str).str.lower().isin(canada_values)

                            if canada_mask.any():
                                df = df[canada_mask]
                                logger.info(
                                    f"Filtered for Canada records using column {region_col}")

                        # Convert columns to proper formats
                        df['REF_DATE'] = pd.to_datetime(
                            df[year_col], format='%Y', errors='coerce')
                        df['VALUE'] = pd.to_numeric(
                            df[ghg_col], errors='coerce')

                        # Drop invalid rows
                        df = df.dropna(subset=['REF_DATE', 'VALUE'])

                        if not df.empty:
                            # Group by year and sum
                            annual_data = df.groupby('REF_DATE')[
                                'VALUE'].sum().reset_index()

                            logger.info(
                                f"Successfully processed Environment Canada emissions data: {len(annual_data)} records")

                            # Save to cache
                            self.save_to_cache(
                                annual_data, "emissions.csv")

                            return annual_data
                        else:
                            raise ValueError(
                                "No valid emissions data after parsing Environment Canada CSV")
                    else:
                        raise ValueError(
                            f"Could not identify required columns. Year columns found: {year_columns}, GHG columns found: {ghg_columns}")
                except Exception as e4:
                    logger.error(
                        f"Failed to get emissions data from Environment Canada: {str(e4)}")

                    # Try OECD API as a final fallback
                    try:
                        logger.info(
                            "Attempting to fetch emissions data from OECD API")
                        url = "https://stats.oecd.org/SDMX-JSON/data/AIR_GHG/CAN.GHG.MT/all"
                        headers = {'Accept': 'application/json'}
                        response = self.api_client.get_with_retry(
                            url, headers=headers)
                        data = response.json()

                        records = []  # Initialize records list here to ensure it's in scope

                        if 'dataSets' in data and len(data['dataSets']) > 0:
                            # Extract greenhouse gas emissions data
                            dataset = data['dataSets'][0]
                            if 'series' in dataset:
                                for series_key, series in dataset['series'].items():
                                    if 'observations' in series:
                                        for time_key, values in series['observations'].items():
                                            if values and len(values) > 0:
                                                year = int(time_key)
                                                # in Mt CO2e
                                                value = float(values[0])

                                                records.append({
                                                    'REF_DATE': pd.to_datetime(str(year), format='%Y'),
                                                    'VALUE': value
                                                })

                        if records:
                            df = pd.DataFrame(records)
                            df = df.sort_values('REF_DATE')
                            logger.info(
                                "Successfully retrieved emissions data from OECD")

                            # Save to cache
                            self.save_to_cache(df, "emissions.csv")

                            return df
                        else:
                            raise ValueError(
                                "No emissions data found in OECD API response")
                    except Exception as e5:
                        logger.error(
                            f"Failed to get emissions data from OECD: {str(e5)}")
                        raise ValueError(
                            "Could not retrieve emissions data from any source")

    def get_rd_expenditure_data_extended(self):
        """Get Research and development expenditure (% of GDP) with extended date range"""
        logger.info("Fetching extended R&D expenditure data")

        # Try to load from cache first
        cached_data = self.load_from_cache("rd_expenditure_extended.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Research and development expenditure (% of GDP)
            # Use default (all G7 + China)
            df = self.get_worldbank_data("GB.XPD.RSDV.GD.ZS")

            # Save to cache
            self.save_to_cache(df, "rd_expenditure_extended.csv")

            return df

        except Exception as e:
            logger.error(
                f"Failed to get extended R&D expenditure data: {str(e)}")
            raise

    def get_employment_data(self):
        """Get employment data for Canada"""
        logger.info("Fetching employment data")

        # Try to load from cache first
        cached_data = self.load_from_cache("employment.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Try Statistics Canada Table 14-10-0287 (Labour force characteristics)
            df = self.get_statcan_data("14-10-0287-01")

            if df is not None:
                # Filter and process the data
                if 'GEO' in df.columns:
                    df = df[df['GEO'] == 'Canada']
                if 'Labour force characteristics' in df.columns:
                    df = df[df['Labour force characteristics'] == 'Employment']

                # Handle different column names
                if 'VALUE' in df.columns:
                    value_col = 'VALUE'
                elif 'Value' in df.columns:
                    value_col = 'Value'
                else:
                    # Try to find a numeric column
                    numeric_cols = df.select_dtypes(
                        include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        value_col = numeric_cols[0]
                    else:
                        raise ValueError("No numeric value column found")

                if 'REF_DATE' in df.columns:
                    date_col = 'REF_DATE'
                elif 'Ref_Date' in df.columns:
                    date_col = 'Ref_Date'
                else:
                    # Try to find a date-like column
                    date_cols = [
                        col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
                    if len(date_cols) > 0:
                        date_col = date_cols[0]
                    else:
                        raise ValueError("No date column found")

                df['VALUE'] = pd.to_numeric(df[value_col], errors='coerce')
                df['REF_DATE'] = pd.to_datetime(df[date_col], errors='coerce')

                # Calculate annual averages
                df['year'] = df['REF_DATE'].dt.year
                annual_avg = df.groupby('year')['VALUE'].mean().reset_index()
                annual_avg['REF_DATE'] = pd.to_datetime(
                    annual_avg['year'], format='%Y')

                logger.info(
                    "Successfully processed employment data from Statistics Canada")

                # Save to cache
                self.save_to_cache(
                    annual_avg[['REF_DATE', 'VALUE']], "employment.csv")

                return annual_avg[['REF_DATE', 'VALUE']]

        except Exception as e:
            logger.error(
                f"Failed to get employment data from Statistics Canada: {str(e)}")

            # Try World Bank API as a backup
            try:
                # There are several indicators we can try:
                # 1. SL.EMP.TOTL.SP.ZS = Employment to population ratio
                # 2. SL.TLF.TOTL.IN = Total labor force (people)
                # First, try labor force
                logger.info(
                    "Attempting to fetch employment data from World Bank API (labor force)")
                df = self.get_worldbank_data("SL.TLF.TOTL.IN", countries="CAN")

                logger.info(
                    "Successfully retrieved labor force data from World Bank")

                # Save to cache
                self.save_to_cache(df, "employment.csv")

                return df

            except Exception as e:
                logger.error(
                    f"Failed to get labor force data from World Bank: {str(e)}")

                # Try employment to population ratio
                try:
                    logger.info(
                        "Attempting to fetch employment to population ratio from World Bank API")
                    ratio_df = self.get_worldbank_data(
                        "SL.EMP.TOTL.SP.ZS", countries="CAN")

                    # Get population data to calculate total employment
                    pop_df = self.get_population_data()

                    # Merge and calculate total employment
                    merged_df = pd.merge(
                        ratio_df, pop_df, on='REF_DATE', suffixes=('_ratio', '_pop'))
                    # Employment ratio is in %, divide by 100 to get decimal
                    merged_df['VALUE'] = (
                        merged_df['VALUE_ratio'] / 100) * merged_df['VALUE_pop']

                    result_df = merged_df[['REF_DATE', 'VALUE']]

                    logger.info(
                        "Successfully calculated employment data from World Bank")

                    # Save to cache
                    self.save_to_cache(result_df, "employment.csv")

                    return result_df

                except Exception as e2:
                    logger.error(
                        f"Failed to calculate employment data from World Bank: {str(e2)}")

                    # Try OECD API as a final backup
                    try:
                        url = "https://stats.oecd.org/SDMX-JSON/data/ALFS_SUMTAB/CAN.EMP.A/all"
                        headers = {'Accept': 'application/json'}
                        response = self.api_client.get_with_retry(
                            url, headers=headers)
                        data = response.json()

                        if 'dataSets' in data and len(data['dataSets']) > 0:
                            records = []

                            # Extract employment data
                            dataset = data['dataSets'][0]
                            if 'series' in dataset:
                                for series_key, series in dataset['series'].items():
                                    if 'observations' in series:
                                        for time_key, values in series['observations'].items():
                                            if values and len(values) > 0:
                                                year = int(time_key)
                                                # Convert to number of people if needed
                                                value = float(values[0]) * 1000

                                                records.append({
                                                    'REF_DATE': pd.to_datetime(str(year), format='%Y'),
                                                    'VALUE': value
                                                })

                            if records:
                                df = pd.DataFrame(records)
                                df = df.sort_values('REF_DATE')
                                logger.info(
                                    "Successfully retrieved employment data from OECD")

                                # Save to cache
                                self.save_to_cache(df, "employment.csv")

                                return df
                            else:
                                raise ValueError(
                                    "No employment data found in OECD API response")
                        else:
                            raise ValueError(
                                "Invalid data format from OECD API")
                    except Exception as e3:
                        logger.error(
                            f"Failed to get employment data from OECD: {str(e3)}")
                        raise ValueError(
                            "Could not retrieve employment data from any source")

    def get_patent_applications_residents(self):
        """Get Patent applications by residents data"""
        logger.info("Fetching patent applications by residents data")

        # Try to load from cache first
        cached_data = self.load_from_cache("patent_applications_residents.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Patent applications, residents
            df = self.get_worldbank_data("IP.PAT.RESD")

            # Save to cache
            self.save_to_cache(df, "patent_applications_residents.csv")

            return df

        except Exception as e:
            logger.error(
                f"Failed to get patent applications by residents data: {str(e)}")
            raise

    def get_patent_applications_nonresidents(self):
        """Get Patent applications by nonresidents data"""
        logger.info("Fetching patent applications by nonresidents data")

        # Try to load from cache first
        cached_data = self.load_from_cache(
            "patent_applications_nonresidents.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Patent applications, nonresidents
            df = self.get_worldbank_data("IP.PAT.NRES")

            # Save to cache
            self.save_to_cache(df, "patent_applications_nonresidents.csv")

            return df

        except Exception as e:
            logger.error(
                f"Failed to get patent applications by nonresidents data: {str(e)}")
            raise

    def get_researchers_in_rd(self):
        """Get Researchers in R&D (per million people) data"""
        logger.info("Fetching researchers in R&D data")

        # Try to load from cache first
        cached_data = self.load_from_cache("researchers_in_rd.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Researchers in R&D (per million people)
            df = self.get_worldbank_data("SP.POP.SCIE.RD.P6")

            # Save to cache
            self.save_to_cache(df, "researchers_in_rd.csv")

            return df

        except Exception as e:
            logger.error(f"Failed to get researchers in R&D data: {str(e)}")
            raise

    def get_scientific_articles(self):
        """Get Scientific and technical journal articles data"""
        logger.info("Fetching scientific and technical journal articles data")

        # Try to load from cache first
        cached_data = self.load_from_cache("scientific_articles.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Scientific and technical journal articles
            df = self.get_worldbank_data("IP.JRN.ARTC.SC")

            # Save to cache
            self.save_to_cache(df, "scientific_articles.csv")

            return df

        except Exception as e:
            logger.error(
                f"Failed to get scientific and technical journal articles data: {str(e)}")
            raise

    def get_high_tech_exports_pct(self):
        """Get High-technology exports (% of manufactured exports) data"""
        logger.info("Fetching high-technology exports (%) data")

        # Try to load from cache first
        cached_data = self.load_from_cache("high_tech_exports_pct.csv")
        if cached_data is not None:
            return cached_data

        try:
            # High-technology exports (% of manufactured exports)
            df = self.get_worldbank_data("TX.VAL.TECH.MF.ZS")

            # Save to cache
            self.save_to_cache(df, "high_tech_exports_pct.csv")

            return df

        except Exception as e:
            logger.error(
                f"Failed to get high-technology exports (%) data: {str(e)}")
            raise

    def get_high_tech_exports_value(self):
        """Get High-technology exports (current US$) data"""
        logger.info("Fetching high-technology exports (value) data")

        # Try to load from cache first
        cached_data = self.load_from_cache("high_tech_exports_value.csv")
        if cached_data is not None:
            return cached_data

        try:
            # High-technology exports (current US$)
            df = self.get_worldbank_data("TX.VAL.TECH.CD")

            # Save to cache
            self.save_to_cache(df, "high_tech_exports_value.csv")

            return df

        except Exception as e:
            logger.error(
                f"Failed to get high-technology exports (value) data: {str(e)}")
            raise

    def get_firms_spending_on_rd(self):
        """Get Firms that spend on R&D (% of firms) data"""
        logger.info("Fetching firms spending on R&D data")

        # Try to load from cache first
        cached_data = self.load_from_cache("firms_spending_on_rd.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Try a different approach - enterprise surveys from World Bank
            # Rather than using IC.FRM.RDSP.ZS (which might not be available for Canada)
            # Use the R&D expenditure (% of GDP) as a proxy
            df = self.get_worldbank_data("GB.XPD.RSDV.GD.ZS")

            # Save to cache
            self.save_to_cache(df, "firms_spending_on_rd.csv")

            return df
        except Exception as e:
            logger.error(
                f"Failed to get firms spending on R&D proxy data: {str(e)}")

            # Try Statistics Canada table for business enterprise R&D
            try:
                # Table 27-10-0333-01 = BERD expenditures
                df = self.get_statcan_data("27-10-0333-01")

                if df is not None:
                    if 'REF_DATE' in df.columns and 'VALUE' in df.columns:
                        # Calculate annual values if needed
                        df['year'] = pd.to_datetime(df['REF_DATE']).dt.year
                        annual_data = df.groupby(
                            'year')['VALUE'].sum().reset_index()
                        annual_data['REF_DATE'] = pd.to_datetime(
                            annual_data['year'], format='%Y')

                        logger.info(
                            "Successfully retrieved business R&D data from Statistics Canada")

                        # Save to cache
                        self.save_to_cache(
                            annual_data[['REF_DATE', 'VALUE']], "firms_spending_on_rd.csv")

                        return annual_data[['REF_DATE', 'VALUE']]
                    else:
                        raise ValueError(
                            "Unexpected column format in Statistics Canada data")
            except Exception as e:
                logger.error(
                    f"Failed to get business R&D data from Statistics Canada: {str(e)}")

            # Try OECD API as a final option
            try:
                url = "https://stats.oecd.org/SDMX-JSON/data/MSTI/CAN.BERD.GERD.PC_GERD/all"
                headers = {'Accept': 'application/json'}
                response = self.api_client.get_with_retry(
                    url, headers=headers)
                data = response.json()

                if 'dataSets' in data and len(data['dataSets']) > 0:
                    records = []

                    # Extract BERD as % of GERD data
                    dataset = data['dataSets'][0]
                    if 'series' in dataset:
                        for series_key, series in dataset['series'].items():
                            if 'observations' in series:
                                for time_key, values in series['observations'].items():
                                    if values and len(values) > 0:
                                        year = int(time_key)
                                        value = float(values[0])

                                records.append({
                                    'REF_DATE': pd.to_datetime(str(year), format='%Y'),
                                    'VALUE': value
                                })

                if records:
                    df = pd.DataFrame(records)
                    df = df.sort_values('REF_DATE')
                    logger.info(
                        "Successfully retrieved business R&D data from OECD")

                    # Save to cache
                    self.save_to_cache(df, "firms_spending_on_rd.csv")

                    return df
                else:
                    raise ValueError(
                        "No business R&D data found in OECD API response")
            except Exception as e:
                logger.error(
                    f"Failed to get business R&D data from OECD: {str(e)}")
                raise ValueError(
                    "Could not retrieve firms spending on R&D data from any source")

    def get_electricity_production_data(self):
        """Get electricity production data for Canada"""
        logger.info("Fetching electricity production data")

        # Try to load from cache first
        cached_data = self.load_from_cache("electricity_production.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Try World Bank API first with correct indicator
            # EG.ELC.PROD.KH has been deprecated, try alternative indicator
            logger.info(
                "Attempting to fetch electricity production data from World Bank (using alternative indicator)")
            # Electricity production from fossil fuels (% of total)
            df = self.get_worldbank_data("EG.ELC.FOSL.ZS")

            logger.info(
                "Successfully retrieved electricity production data (fossil fuel %) from World Bank")

            # Save to cache
            self.save_to_cache(df, "electricity_production.csv")

            return df

        except Exception as e:
            logger.error(
                f"Failed to get electricity production data from World Bank: {str(e)}")

            # Try Statistics Canada as an alternative with updated table number
            try:
                # Try a newer table 25-10-0015 or 25-10-0020
                logger.info(
                    "Attempting to fetch electricity data from newer Statistics Canada tables")
                df = self.get_statcan_data("25-10-0020-01")  # Try newer table

                if df is not None:
                    # Process the data based on format
                    if 'REF_DATE' in df.columns and 'VALUE' in df.columns:
                        # Calculate annual values if needed
                        df['year'] = pd.to_datetime(df['REF_DATE']).dt.year
                        annual_data = df.groupby(
                            'year')['VALUE'].sum().reset_index()
                        annual_data['REF_DATE'] = pd.to_datetime(
                            annual_data['year'], format='%Y')

                        logger.info(
                            "Successfully retrieved electricity production data from Statistics Canada")

                        # Save to cache
                        self.save_to_cache(
                            annual_data[['REF_DATE', 'VALUE']], "electricity_production.csv")

                        return annual_data[['REF_DATE', 'VALUE']]
                    else:
                        raise ValueError(
                            "Unexpected column format in Statistics Canada electricity data")
            except Exception as e2:
                logger.error(
                    f"Failed to get electricity production data from Statistics Canada: {str(e2)}")

                # Try Natural Resources Canada as another alternative
                try:
                    url = "https://www.nrcan.gc.ca/sites/www.nrcan.gc.ca/files/energy/pdf/electricity-generation-by-source.csv"
                    logger.info(
                        f"Attempting to fetch electricity production data from NRCan: {url}")
                    response = self.api_client.get_with_retry(url)
                    df = pd.read_csv(io.StringIO(response.text))

                    # Process NRCan data (format will depend on the actual structure)
                    if 'Year' in df.columns and 'Total' in df.columns:
                        df['REF_DATE'] = pd.to_datetime(
                            df['Year'], format='%Y')
                        df['VALUE'] = pd.to_numeric(
                            df['Total'], errors='coerce')

                        logger.info(
                            "Successfully retrieved electricity production data from NRCan")

                        # Save to cache
                        self.save_to_cache(
                            df[['REF_DATE', 'VALUE']], "electricity_production.csv")

                        return df[['REF_DATE', 'VALUE']]
                    else:
                        raise ValueError(
                            "Unexpected column format in NRCan electricity production data")
                except Exception as e3:
                    logger.error(
                        f"Failed to get electricity production data from NRCan: {str(e3)}")

                    # Try IEA API as a final option (if API key exists)
                    try:
                        iea_api_key = os.getenv('IEA_API_KEY')
                        if not iea_api_key:
                            logger.warning(
                                "IEA API key not found in environment variables")
                            # Since we don't want to use fallback data, we'll raise an error
                            raise ValueError(
                                "IEA API key not found in environment variables")

                        url = "https://api.iea.org/stats/electricity/production"
                        headers = {
                            'Authorization': f'Bearer {iea_api_key}',
                            'Accept': 'application/json'
                        }
                        params = {
                            'country': 'CAN',
                            'startYear': '1950',
                            'endYear': str(datetime.datetime.now().year)
                        }

                        response = self.api_client.get_with_retry(
                            url, params=params, headers=headers)
                        data = response.json()

                        if isinstance(data, list) and len(data) > 0:
                            records = []
                            for item in data:
                                if 'year' in item and 'value' in item:
                                    records.append({'REF_DATE': pd.to_datetime(str(item['year']), format='%Y', errors='coerce'),
                                                   'VALUE': float(item['value'])})

                            if records:
                                df = pd.DataFrame(records)
                                df = df.sort_values('REF_DATE')
                                logger.info(
                                    "Successfully retrieved electricity production data from IEA")

                                # Save to cache
                                self.save_to_cache(
                                    df, "electricity_production.csv")

                                return df
                            else:
                                raise ValueError(
                                    "No valid records in IEA API response")
                        else:
                            raise ValueError(
                                "Invalid data format from IEA API")
                    except Exception as e4:
                        logger.error(
                            f"Failed to get electricity production data from IEA: {str(e4)}")
                        raise ValueError(
                            "Could not retrieve electricity production data from any source")

    def get_electricity_consumption_data(self):
        """Get electricity consumption data for Canada"""
        logger.info(
            "Fetching electricity consumption data")

        # Try to load from cache first
        cached_data = self.load_from_cache("electricity_consumption.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Use the working World Bank indicator (EG.USE.ELEC.KH.PC = electric power consumption per capita)
            logger.info(
                "Attempting to fetch electricity consumption per capita from World Bank")
            df_per_capita = self.get_worldbank_data("EG.USE.ELEC.KH.PC")

            # Get population data to calculate total consumption
            logger.info(
                "Fetching population data to calculate total consumption")
            pop_df = self.get_population_data()

            # Merge and calculate total consumption
            merged_df = pd.merge(df_per_capita, pop_df,
                                 on='REF_DATE', suffixes=('_pc', '_pop'))
            # Convert to TWh (kWh per capita * population / 1e9)
            merged_df['VALUE'] = (merged_df['VALUE_pc']
                                  * merged_df['VALUE_pop']) / 1e9

            result_df = merged_df[['REF_DATE', 'VALUE']]

            logger.info(
                "Successfully calculated total electricity consumption data")

            # Save to cache
            self.save_to_cache(result_df, "electricity_consumption.csv")

            return result_df

        except Exception as e:
            logger.error(
                f"Failed to calculate electricity consumption from World Bank: {str(e)}")

            # Try using a different World Bank indicator
            try:
                # Try EG.USE.ELEC.GD.PP.KD = Electric power consumption (kWh) per PPP $ of GDP
                logger.info(
                    "Attempting to calculate electricity consumption from GDP ratio")
                elec_per_gdp_df = self.get_worldbank_data(
                    "EG.USE.ELEC.GD.PP.KD")

                # Get GDP data
                gdp_df = self.get_gdp_data()

                # Merge and calculate total consumption
                merged_df = pd.merge(
                    elec_per_gdp_df, gdp_df, on='REF_DATE', suffixes=('_per_gdp', '_gdp'))
                # GDP is in trillions of dollars, convert to appropriate units for calculation
                merged_df['VALUE'] = merged_df['VALUE_per_gdp'] * \
                    merged_df['VALUE_gdp'] * 1e12 / 1e9  # Convert to TWh

                result_df = merged_df[['REF_DATE', 'VALUE']]

                logger.info(
                    "Successfully calculated electricity consumption using GDP ratio")

                # Save to cache
                self.save_to_cache(
                    result_df, "electricity_consumption.csv")

                return result_df

            except Exception as e2:
                logger.error(
                    f"Failed to calculate electricity consumption from GDP ratio: {str(e2)}")

                # Try Statistics Canada with updated table number
                try:
                    # Try newer table for electricity supply and disposition
                    logger.info(
                        "Attempting to fetch electricity consumption from Statistics Canada")
                    # Try updated table number
                    df = self.get_statcan_data("25-10-0021-01")

                    if df is not None:
                        # Filter for total demand/consumption
                        if 'Supply and demand characteristics' in df.columns:
                            df = df[df['Supply and demand characteristics'].str.contains(
                                'consumption', case=False, na=False)]
                        elif 'Disposition' in df.columns:
                            df = df[df['Disposition'].str.contains(
                                'consumption', case=False, na=False)]

                        # Process the data
                        if 'REF_DATE' in df.columns and 'VALUE' in df.columns:
                            # Calculate annual values if needed
                            df['year'] = pd.to_datetime(df['REF_DATE']).dt.year
                            annual_data = df.groupby(
                                'year')['VALUE'].sum().reset_index()
                            annual_data['REF_DATE'] = pd.to_datetime(
                                annual_data['year'], format='%Y')

                            logger.info(
                                "Successfully retrieved electricity consumption data from Statistics Canada")

                            # Save to cache
                            self.save_to_cache(
                                annual_data[['REF_DATE', 'VALUE']], "electricity_consumption.csv")

                            return annual_data[['REF_DATE', 'VALUE']]
                        else:
                            raise ValueError(
                                "Unexpected column format in Statistics Canada electricity data")
                except Exception as e3:
                    logger.error(
                        f"Failed to get electricity consumption data from Statistics Canada: {str(e3)}")

                    # Try Natural Resources Canada as a final option
                    try:
                        url = "https://www.nrcan.gc.ca/sites/www.nrcan.gc.ca/files/energy/pdf/electricity-consumption-by-sector.csv"
                        logger.info(
                            f"Attempting to fetch electricity consumption data from NRCan: {url}")
                        response = self.api_client.get_with_retry(url)
                        df = pd.read_csv(io.StringIO(response.text))

                        # Process NRCan data
                        if 'Year' in df.columns and 'Total' in df.columns:
                            df['REF_DATE'] = pd.to_datetime(
                                df['Year'], format='%Y')
                            df['VALUE'] = pd.to_numeric(
                                df['Total'], errors='coerce')

                            logger.info(
                                "Successfully retrieved electricity consumption data from NRCan")

                            # Save to cache
                            self.save_to_cache(
                                df[['REF_DATE', 'VALUE']], "electricity_consumption.csv")

                            return df[['REF_DATE', 'VALUE']]
                        else:
                            raise ValueError(
                                "Unexpected column format in NRCan electricity consumption data")
                    except Exception as e4:
                        logger.error(
                            f"Failed to get electricity consumption data from NRCan: {str(e4)}")
                        raise ValueError(
                            "Could not retrieve electricity consumption data from any source")

    def get_electricity_prices(self):
        """Get electricity prices data for Canada"""
        logger.info("Fetching electricity prices data")

        # Try to load from cache first
        cached_data = self.load_from_cache("electricity_prices.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Try Statistics Canada first for electricity prices
            # Table 18-10-0004-01 (CPI) filtered for electricity
            df = self.get_statcan_data("18-10-0004-01")

            if df is not None:
                # Filter for electricity prices
                if 'Products and product groups' in df.columns:
                    electricity_filter = df['Products and product groups'].str.contains(
                        'Electricity', case=False, na=False)
                    if electricity_filter.any():
                        df = df[electricity_filter]
                    else:
                        raise ValueError(
                            "No electricity price data found in CPI dataset")

                    # Process the data
                    if 'REF_DATE' in df.columns and 'VALUE' in df.columns:
                        # Calculate annual averages
                        df['year'] = pd.to_datetime(df['REF_DATE']).dt.year
                        annual_data = df.groupby(
                            'year')['VALUE'].mean().reset_index()
                        annual_data['REF_DATE'] = pd.to_datetime(
                            annual_data['year'], format='%Y')

                        logger.info(
                            "Successfully retrieved electricity prices data from Statistics Canada")

                        # Save to cache
                        self.save_to_cache(
                            annual_data[['REF_DATE', 'VALUE']], "electricity_prices.csv")

                        return annual_data[['REF_DATE', 'VALUE']]
                    else:
                        raise ValueError(
                            "Unexpected column format in Statistics Canada electricity price data")
        except Exception as e:
            logger.error(
                f"Failed to get electricity prices data from Statistics Canada: {str(e)}")

            # Try Natural Resources Canada as a backup
            try:
                url = "https://www.nrcan.gc.ca/sites/www.nrcan.gc.ca/files/energy/pdf/electricity-prices.csv"
                response = self.api_client.get_with_retry(url)
                df = pd.read_csv(io.StringIO(response.text))

                # Process NRCan data (format will depend on the actual structure)
                if 'Year' in df.columns and 'Residential_Price' in df.columns:
                    df['REF_DATE'] = pd.to_datetime(df['Year'], format='%Y')
                    df['VALUE'] = pd.to_numeric(
                        df['Residential_Price'], errors='coerce')

                    logger.info(
                        "Successfully retrieved electricity prices from NRCan")

                    # Save to cache
                    self.save_to_cache(
                        df[['REF_DATE', 'VALUE']], "electricity_prices.csv")

                    return df[['REF_DATE', 'VALUE']]
            except Exception as e:
                logger.error(
                    f"Failed to get electricity prices from NRCan: {str(e)}")

                # Try OECD API as a final option
                try:
                    url = "https://stats.oecd.org/SDMX-JSON/data/ELECTRICITY_PRICES/CAN.RES.USD_KWH/all"
                    headers = {'Accept': 'application/json'}
                    response = self.api_client.get_with_retry(
                        url, headers=headers)
                    data = response.json()

                    if 'dataSets' in data and len(data['dataSets']) > 0:
                        records = []

                        # Extract electricity price data
                        dataset = data['dataSets'][0]
                        if 'series' in dataset:
                            for series_key, series in dataset['series'].items():
                                if 'observations' in series:
                                    for time_key, values in series['observations'].items():
                                        if values and len(values) > 0:
                                            year = int(time_key)
                                            value = float(values[0])

                                            records.append({
                                                'REF_DATE': pd.to_datetime(str(year), format='%Y'),
                                                'VALUE': value
                                            })

                        if records:
                            df = pd.DataFrame(records)
                            df = df.sort_values('REF_DATE')
                            logger.info(
                                "Successfully retrieved electricity prices from OECD")

                            # Save to cache
                            self.save_to_cache(df, "electricity_prices.csv")

                            return df
                        else:
                            raise ValueError(
                                "No electricity price data found in OECD API response")
                except Exception as e:
                    logger.error(
                        f"Failed to get electricity prices from OECD: {str(e)}")
                    raise ValueError(
                        "Could not retrieve electricity prices data from any source")

    def get_electricity_consumption_per_capita(self):
        """Get electricity consumption per capita data for Canada"""
        logger.info(
            "Fetching electricity consumption per capita data")

        # Try to load from cache first
        cached_data = self.load_from_cache(
            "electricity_consumption_per_capita.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Try World Bank API directly (EG.USE.ELEC.KH.PC = electric power consumption per capita)
            df = self.get_worldbank_data("EG.USE.ELEC.KH.PC")

            # Convert from kWh to MWh for better readability
            df['VALUE'] = df['VALUE'] / 1000

            logger.info(
                "Successfully retrieved electricity consumption per capita from World Bank")

            # Save to cache
            self.save_to_cache(
                df, "electricity_consumption_per_capita.csv")

            return df

        except Exception as e:
            logger.error(
                f"Failed to get electricity consumption per capita from World Bank: {str(e)}")

            # Try calculating it from total consumption and population
            try:
                logger.info(
                    "Attempting to calculate consumption per capita from total consumption and population")
                # Get total consumption and population data
                consumption_df = self.get_electricity_consumption_data()
                population_df = self.get_population_data()

                # Merge and calculate per capita consumption
                merged_df = pd.merge(
                    consumption_df, population_df, on='REF_DATE', suffixes=('_cons', '_pop'))

                # Calculate per capita value (convert TWh to MWh and divide by population)
                merged_df['VALUE'] = (
                    merged_df['VALUE_cons'] * 1e9) / merged_df['VALUE_pop']

                result_df = merged_df[['REF_DATE', 'VALUE']]

                logger.info(
                    "Successfully calculated electricity consumption per capita")

                # Save to cache
                self.save_to_cache(
                    result_df, "electricity_consumption_per_capita.csv")

                return result_df

            except Exception as e2:
                logger.error(
                    f"Failed to calculate electricity consumption per capita: {str(e2)}")
                raise ValueError(
                    "Could not retrieve electricity consumption per capita data from any source")

    def get_population_data(self):
        """Get population data for Canada"""
        logger.info("Fetching population data")

        # Try to load from cache first
        cached_data = self.load_from_cache("population.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Try World Bank API first - SP.POP.TOTL = Total population
            # Specify Canada only for backwards compatibility
            df = self.get_worldbank_data("SP.POP.TOTL", countries="CAN")

            logger.info(
                "Successfully retrieved population data from World Bank")

            # Save to cache
            self.save_to_cache(df, "population.csv")

            return df

        except Exception as e:
            logger.error(
                f"Failed to get population data from World Bank: {str(e)}")

            # Try Statistics Canada as a backup
            try:
                # Table 17-10-0005-01 = Population estimates
                df = self.get_statcan_data("17-10-0005-01")

                # Process the data based on the format
                if 'REF_DATE' in df.columns and 'VALUE' in df.columns:
                    # Calculate annual values if needed
                    df['year'] = pd.to_datetime(df['REF_DATE']).dt.year
                    annual_data = df.groupby(
                        'year')['VALUE'].sum().reset_index()
                    annual_data['REF_DATE'] = pd.to_datetime(
                        annual_data['year'], format='%Y')

                    logger.info(
                        "Successfully retrieved population data from Statistics Canada")

                    # Save to cache
                    self.save_to_cache(
                        annual_data[['REF_DATE', 'VALUE']], "population.csv")

                    return annual_data[['REF_DATE', 'VALUE']]
                else:
                    raise ValueError(
                        "Unexpected column format in Statistics Canada population data")

            except Exception as e2:
                logger.error(
                    f"Failed to get population data from Statistics Canada: {str(e2)}")

                # Try OECD API as a final backup
                try:
                    url = "https://stats.oecd.org/SDMX-JSON/data/ALFS_POP_VITAL/CAN.RPOP.A/all"
                    headers = {'Accept': 'application/json'}
                    response = self.api_client.get_with_retry(
                        url, headers=headers)
                    data = response.json()

                    if 'dataSets' in data and len(data['dataSets']) > 0:
                        records = []

                        # Extract population data
                        dataset = data['dataSets'][0]
                        if 'series' in dataset:
                            for series_key, series in dataset['series'].items():
                                if 'observations' in series:
                                    for time_key, values in series['observations'].items():
                                        if values and len(values) > 0:
                                            year = int(time_key)
                                            value = float(values[0])

                                            records.append({
                                                'REF_DATE': pd.to_datetime(str(year), format='%Y'),
                                                'VALUE': value * 1000  # Convert to number of people if needed
                                            })

                        if records:
                            df = pd.DataFrame(records)
                            df = df.sort_values('REF_DATE')
                            logger.info(
                                "Successfully retrieved population data from OECD")

                            # Save to cache
                            self.save_to_cache(df, "population.csv")

                            return df
                        else:
                            raise ValueError(
                                "No population data found in OECD API response")
                except Exception as e3:
                    logger.error(
                        f"Failed to get population data from OECD: {str(e3)}")
                    raise ValueError(
                        "Could not retrieve population data from any source")

    def get_emissions_data(self):
        """Get emissions data using multiple alternative sources"""
        logger.info("Fetching emissions data")

        # Try to load from cache first
        cached_data = self.load_from_cache("emissions.csv")
        if cached_data is not None:
            return cached_data

        try:
            # Try multiple CO2 emissions indicators
            indicators_to_try = [
                "EN.ATM.CO2E.KT",     # CO2 emissions (kt)
                # Total greenhouse gas emissions (kt of CO2 equivalent)
                "EN.ATM.GHGT.KT.CE",
                # Total greenhouse gas emissions (% change from 1990)
                "EN.ATM.GHGT.ZG"
            ]

            df = None
            for indicator in indicators_to_try:
                try:
                    logger.info(
                        f"Attempting to fetch emissions data using indicator: {indicator}")
                    # Specify Canada only for backwards compatibility
                    df = self.get_worldbank_data(indicator, countries="CAN")

                    # Convert kt to Mt for consistency if the indicator is in kt
                    if indicator.endswith('.KT') or indicator.endswith('.KT.CE'):
                        df['VALUE'] = df['VALUE'] / 1000
                        logger.info(
                            f"Converted {indicator} from kt to Mt for consistency")

                    logger.info(
                        f"Successfully retrieved emissions data with indicator {indicator}")
                    break
                except Exception as ind_e:
                    logger.warning(
                        f"Failed to get data for indicator {indicator}: {str(ind_e)}")
                    continue

            if df is not None:
                # Save to cache
                self.save_to_cache(df, "emissions.csv")
                return df
            else:
                raise ValueError("All World Bank emissions indicators failed")

        except Exception as e:
            logger.error(
                f"Failed to get emissions data from World Bank: {str(e)}")

            # Try using per capita emissions and population as a backup approach
            try:
                logger.info(
                    "Attempting to calculate emissions from per capita data")
                # Try EN.ATM.CO2E.PC = CO2 emissions (metric tons per capita)
                df_per_capita = self.get_worldbank_data(
                    "EN.ATM.CO2E.PC", countries="CAN")

                # Get population data to calculate total emissions
                pop_df = self.get_population_data()

                # Merge and calculate total emissions
                merged_df = pd.merge(df_per_capita, pop_df,
                                     on='REF_DATE', suffixes=('_pc', '_pop'))
                # Convert to Mt CO2 (per capita * population / 1,000,000)
                merged_df['VALUE'] = (
                    merged_df['VALUE_pc'] * merged_df['VALUE_pop']) / 1000000

                result_df = merged_df[['REF_DATE', 'VALUE']]

                logger.info(
                    "Successfully calculated total CO2 emissions data from per capita values")

                # Save to cache
                self.save_to_cache(result_df, "emissions.csv")

                return result_df

            except Exception as e2:
                logger.error(
                    f"Failed to calculate CO2 emissions from per capita data: {str(e2)}")

                # Try Canadian government open data portal
                try:
                    # Try Canadian Climate Institute data
                    url = "https://climatechoices.ca/wp-content/uploads/2021/12/canada-ghg-emissions-data.csv"
                    logger.info(
                        f"Attempting to fetch emissions data from Climate Institute: {url}")
                    response = self.api_client.get_with_retry(url)
                    df = pd.read_csv(io.StringIO(response.text))

                    # Process the data based on the format
                    if 'Year' in df.columns and 'Total' in df.columns:
                        df['REF_DATE'] = pd.to_datetime(
                            df['Year'], format='%Y')
                        df['VALUE'] = pd.to_numeric(
                            df['Total'], errors='coerce')

                        logger.info(
                            "Successfully retrieved emissions data from Climate Institute")

                        # Save to cache
                        self.save_to_cache(
                            df[['REF_DATE', 'VALUE']], "emissions.csv")

                        return df[['REF_DATE', 'VALUE']]
                    else:
                        raise ValueError(
                            "Unexpected column format in Climate Institute emissions data")

                except Exception as e3:
                    logger.error(
                        f"Failed to get emissions data from Climate Institute: {str(e3)}")

                    # Try Environment and Climate Change Canada with better error handling
                    try:
                        url = "https://data.ec.gc.ca/data/substances/monitor/canada-s-official-greenhouse-gas-inventory/UNFCCC-NIR/GHG_IPCC_Can_Prov_Terr.csv"
                        logger.info(
                            f"Attempting to fetch emissions data from Environment Canada: {url}")
                        response = self.api_client.get_with_retry(url)

                        # We've had issues with this file, so let's first examine it
                        content = response.text
                        # Examine first 20 lines
                        sample_lines = content.splitlines()[:20]
                        logger.info(
                            f"Sample of Environment Canada CSV file:\n{sample_lines}")

                        # Try multiple parsing approaches
                        parsing_methods = [
                            {"method": "standard", "params": {
                                "on_bad_lines": 'skip'}},
                            {"method": "old_pandas", "params": {
                                "error_bad_lines": False}},
                            {"method": "python_engine", "params": {
                                "engine": 'python', "on_bad_lines": 'skip'}},
                            {"method": "python_engine_skiprows", "params": {
                                # Skip problematic rows
                                "engine": 'python', "skiprows": [7, 8]}}
                        ]

                        df = None
                        for method in parsing_methods:
                            try:
                                logger.info(
                                    f"Trying parsing method: {method['method']}")
                                if method["method"] == "old_pandas":
                                    try:
                                        df = pd.read_csv(io.StringIO(
                                            content), **method["params"])
                                        break
                                    except TypeError:
                                        logger.info(
                                            "Old pandas method not supported, skipping")
                                        continue
                                else:
                                    df = pd.read_csv(io.StringIO(
                                        content), **method["params"])
                                    break
                            except Exception as parse_err:
                                logger.warning(
                                    f"Parsing method {method['method']} failed: {str(parse_err)}")
                                continue

                        if df is None:
                            # Last resort: try with C parser and specific delimiter
                            try:
                                logger.info(
                                    "Trying CSV parsing with explicit delimiter detection")
                                # Try to detect the delimiter
                                for delimiter in [',', ';', '\t', '|']:
                                    try:
                                        df = pd.read_csv(io.StringIO(content), sep=delimiter, engine='c',
                                                         on_bad_lines='skip', encoding='utf-8')
                                        # If we got multiple columns, we found the right delimiter
                                        if len(df.columns) > 1:
                                            logger.info(
                                                f"Successfully parsed CSV with delimiter: '{delimiter}'")
                                            break
                                    except:
                                        continue
                            except Exception as e:
                                logger.error(
                                    f"All CSV parsing methods failed: {str(e)}")
                                raise ValueError(
                                    "Unable to parse Environment Canada CSV file")

                        if df is None:
                            raise ValueError(
                                "Failed to parse CSV with any method")

                        # Log the columns we found to help with debugging
                        logger.info(
                            f"Parsed CSV columns: {df.columns.tolist()}")

                        # First check standard column names
                        if 'Year' in df.columns:
                            # Filter for total national emissions
                            if 'Region' in df.columns:
                                df = df[df['Region'] == 'Canada']

                            if 'Total GHG' in df.columns:
                                df['REF_DATE'] = pd.to_datetime(
                                    df['Year'], format='%Y')
                                df['VALUE'] = pd.to_numeric(
                                    df['Total GHG'], errors='coerce')

                                # Drop rows with NaN values that might have resulted from parsing errors
                                df = df.dropna(subset=['VALUE'])

                                # Group by year if needed
                                annual_data = df.groupby('REF_DATE')[
                                    'VALUE'].sum().reset_index()

                                logger.info(
                                    "Successfully retrieved emissions data from Environment Canada")

                                # Save to cache
                                self.save_to_cache(
                                    annual_data, "emissions.csv")

                                return annual_data

                        # Fallback to intelligent column detection if standard columns aren't found
                        # Identify year column
                        year_columns = [col for col in df.columns if
                                        any(year_term in col.lower() for year_term in ['year', 'date', 'yr', 'année'])]

                        # Identify GHG columns
                        ghg_columns = [col for col in df.columns if
                                       any(ghg_term in col.lower() for ghg_term in
                                           ['ghg', 'emission', 'co2', 'carbon', 'total'])]

                        if year_columns and ghg_columns:
                            # Use the first matching columns
                            year_col = year_columns[0]
                            ghg_col = ghg_columns[0]

                            logger.info(
                                f"Using columns: Year={year_col}, GHG={ghg_col}")

                            # Filter for Canada if region column exists
                            if any(region_term in col.lower() for col in df.columns
                                   for region_term in ['region', 'country', 'province']):
                                region_col = [col for col in df.columns if
                                              any(region_term in col.lower() for region_term in
                                                  ['region', 'country', 'province'])][0]

                                # Try to filter for Canada data
                                canada_values = [
                                    'canada', 'can', 'ca', 'national', 'total']
                                canada_mask = df[region_col].astype(
                                    str).str.lower().isin(canada_values)

                                if canada_mask.any():
                                    df = df[canada_mask]
                                    logger.info(
                                        f"Filtered for Canada records using column {region_col}")

                            # Convert columns to proper formats
                            df['REF_DATE'] = pd.to_datetime(
                                df[year_col], format='%Y', errors='coerce')
                            df['VALUE'] = pd.to_numeric(
                                df[ghg_col], errors='coerce')

                            # Drop invalid rows
                            df = df.dropna(subset=['REF_DATE', 'VALUE'])

                            if not df.empty:
                                # Group by year and sum
                                annual_data = df.groupby('REF_DATE')[
                                    'VALUE'].sum().reset_index()

                                logger.info(
                                    f"Successfully processed Environment Canada emissions data: {len(annual_data)} records")

                                # Save to cache
                                self.save_to_cache(
                                    annual_data, "emissions.csv")

                                return annual_data
                            else:
                                raise ValueError(
                                    "No valid emissions data after parsing Environment Canada CSV")
                        else:
                            raise ValueError(
                                f"Could not identify required columns. Year columns found: {year_columns}, GHG columns found: {ghg_columns}")
                    except Exception as e4:
                        logger.error(
                            f"Failed to get emissions data from Environment Canada: {str(e4)}")

                        # Try OECD API as a final fallback
                        try:
                            logger.info(
                                "Attempting to fetch emissions data from OECD API")
                            url = "https://stats.oecd.org/SDMX-JSON/data/AIR_GHG/CAN.GHG.MT/all"
                            headers = {'Accept': 'application/json'}
                            response = self.api_client.get_with_retry(
                                url, headers=headers)
                            data = response.json()

                            records = []  # Initialize records list here to ensure it's in scope

                            if 'dataSets' in data and len(data['dataSets']) > 0:
                                # Extract greenhouse gas emissions data
                                dataset = data['dataSets'][0]
                                if 'series' in dataset:
                                    for series_key, series in dataset['series'].items():
                                        if 'observations' in series:
                                            for time_key, values in series['observations'].items():
                                                if values and len(values) > 0:
                                                    year = int(time_key)
                                                    # in Mt CO2e
                                                    value = float(values[0])

                                                    records.append({
                                                        'REF_DATE': pd.to_datetime(str(year), format='%Y'),
                                                        'VALUE': value
                                                    })

                                if records:
                                    df = pd.DataFrame(records)
                                    df = df.sort_values('REF_DATE')
                                    logger.info(
                                        "Successfully retrieved emissions data from OECD")

                                    # Save to cache
                                    self.save_to_cache(df, "emissions.csv")

                                    return df
                                else:
                                    raise ValueError(
                                        "No emissions data found in OECD API response")
                        except Exception as e5:
                            logger.error(
                                f"Failed to get emissions data from OECD: {str(e5)}")
                            raise ValueError(
                                "Could not retrieve emissions data from any source")


if __name__ == "__main__":
    logger.info("Starting data collection")
    data_collector = DataCollector()

    # Define all the data collection methods to run
    collection_methods = [
        ("oil_production_data", data_collector.get_oil_production_data),
        ("gdp_data", data_collector.get_gdp_data),
        ("population_data", data_collector.get_population_data),
        ("employment_data", data_collector.get_employment_data),
        ("emissions_data", data_collector.get_emissions_data),
        ("rd_expenditure_data", data_collector.get_rd_expenditure_data_extended),
        ("patent_applications_residents",
         data_collector.get_patent_applications_residents),
        ("patent_applications_nonresidents",
         data_collector.get_patent_applications_nonresidents),
        ("researchers_in_rd", data_collector.get_researchers_in_rd),
        ("scientific_articles", data_collector.get_scientific_articles),
        ("high_tech_exports_pct", data_collector.get_high_tech_exports_pct),
        ("high_tech_exports_value", data_collector.get_high_tech_exports_value),
        ("firms_spending_on_rd", data_collector.get_firms_spending_on_rd),
        ("electricity_production", data_collector.get_electricity_production_data),
        ("electricity_consumption", data_collector.get_electricity_consumption_data),
        ("electricity_consumption_per_capita",
         data_collector.get_electricity_consumption_per_capita)
    ]

    # Define key World Bank indicators to collect for all G7 nations plus China
    g7_plus_china_indicators = [
        ("g7_china_gdp", "NY.GDP.MKTP.CD", "GDP (current US$)"),
        ("g7_china_gdp_growth", "NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)"),
        ("g7_china_population", "SP.POP.TOTL", "Population, total"),
        ("g7_china_emissions", "EN.ATM.CO2E.KT", "CO2 emissions (kt)"),
        ("g7_china_emissions_per_capita", "EN.ATM.CO2E.PC",
         "CO2 emissions (metric tons per capita)"),
        ("g7_china_rd_expenditure", "GB.XPD.RSDV.GD.ZS",
         "Research and development expenditure (% of GDP)"),
        ("g7_china_high_tech_exports", "TX.VAL.TECH.MF.ZS",
         "High-technology exports (% of manufactured exports)"),
        ("g7_china_patents", "IP.PAT.RESD", "Patent applications, residents"),
        ("g7_china_electricity", "EG.USE.ELEC.KH.PC",
         "Electric power consumption (kWh per capita)")
    ]

    # Run each method and handle exceptions
    results = {}
    for name, method in collection_methods:
        try:
            logger.info(f"Running collection for {name}")
            df = method()
            if df is not None and len(df) > 0:
                results[name] = df
                try:
                    min_year = min(df['REF_DATE']).year
                    max_year = max(df['REF_DATE']).year
                    logger.info(
                        f"Successfully collected {name}: {len(df)} records from {min_year} to {max_year}")
                except Exception as e_info:
                    # Handle case where date information can't be extracted
                    logger.info(
                        f"Successfully collected {name}: {len(df)} records (date range not available)")
            else:
                logger.warning(f"Collection for {name} returned empty dataset")
        except Exception as e:
            logger.error(f"Failed to collect {name}: {str(e)}")
            # Don't use fallback data - just report the error

    # Collect data for G7 countries plus China for key indicators
    logger.info("Starting collection for G7 nations and China")
    for filename, indicator, description in g7_plus_china_indicators:
        try:
            logger.info(f"Collecting {description} for G7 nations and China")
            # Uses the default countries (G7 + China)
            df = data_collector.get_worldbank_data(indicator)

            if df is not None and len(df) > 0:
                results[filename] = df
                # Save to cache with special filename
                data_collector.save_to_cache(df, f"{filename}.csv")

                # Log information about the collected data
                country_counts = df.groupby('COUNTRY').size().to_dict()
                countries_str = ", ".join(
                    [f"{country}: {count}" for country, count in country_counts.items()])

                years_by_country = {}
                for country, group in df.groupby('COUNTRY'):
                    if not group.empty:
                        try:
                            years_by_country[country] = (
                                min(group['REF_DATE']).year, max(group['REF_DATE']).year)
                        except:
                            years_by_country[country] = "date range not available"

                years_str = ", ".join([f"{country}: {years[0]}-{years[1]}" if isinstance(years, tuple) else f"{country}: {years}"
                                      for country, years in years_by_country.items()])

                logger.info(
                    f"Successfully collected {description} for G7 + China: {len(df)} total records")
                logger.info(f"Records per country: {countries_str}")
                logger.info(f"Year ranges: {years_str}")
            else:
                logger.warning(
                    f"Collection for {description} (G7 + China) returned empty dataset")
        except Exception as e:
            logger.error(
                f"Failed to collect {description} for G7 nations and China: {str(e)}")

    logger.info("Data collection completed")

    # Log summary of results
    logger.info("Collection summary:")
    for name, df in results.items():
        if df is not None and len(df) > 0:
            try:
                # Check if it's a multi-country dataset
                if 'COUNTRY' in df.columns:
                    country_count = df['COUNTRY'].nunique()
                    min_year = min(df['REF_DATE']).year
                    max_year = max(df['REF_DATE']).year
                    logger.info(
                        f"  - {name}: {len(df)} records for {country_count} countries from {min_year} to {max_year}")
                else:
                    min_year = min(df['REF_DATE']).year
                    max_year = max(df['REF_DATE']).year
                    logger.info(
                        f"  - {name}: {len(df)} records from {min_year} to {max_year}")
            except Exception as e_info:
                logger.info(
                    f"  - {name}: {len(df)} records (date range not available)")
        else:
            logger.info(f"  - {name}: No data collected")
