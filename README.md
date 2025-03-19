# Economic Indicators Data Collector

A robust data collection tool that fetches economic, industrial, environmental, and research indicators for Canada and G7+ countries.

## Overview

This tool automatically collects data from multiple sources including:

- World Bank API
- Statistics Canada
- OECD
- Natural Resources Canada

The collector implements fallback mechanisms when primary data sources are unavailable and caches results to avoid redundant API calls.

## Features

- Collects over 15 key economic indicators including:

  - Oil production data
  - GDP data (national and G7+ comparisons)
  - Population statistics
  - Employment figures
  - Emissions data
  - R&D expenditure metrics
  - Patent applications
  - Energy production and consumption
  - High-tech exports
  - Scientific research metrics

- Multi-source resilience with fallback mechanisms
- Local data caching
- Comprehensive logging
- G7+ international comparisons (Canada, USA, UK, France, Germany, Italy, Japan, and China)

## Setup

1. Clone this repository:

```bash
git clone <repository-url>
cd economic_indicator_datacollector
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Create a `.env` file for any API keys:

```
EIA_API_KEY=your-eia-api-key
```

## Usage

Run the data collector to fetch all indicators:

```bash
python datacollector.py
```

This will:

1. Create `cache` directories if they don't exist
2. Fetch all configured indicators
3. Save data as CSV files in the cache directory
4. Generate detailed logs in `oil_analysis.log`

## Data Sources

The collector prioritizes official national statistical agencies and international organizations:

1. **Statistics Canada** - Primary source for Canadian-specific data
2. **World Bank** - Primary source for international comparisons and indicators not available from Statistics Canada
3. **OECD** - Alternative source for economic indicators
4. **Natural Resources Canada** - Alternative source for energy and resource data

## Output

Collected data is stored in CSV format in the `cache` directory with standardized column names:

- `REF_DATE`: Datetime of the observation
- `VALUE`: The indicator value
- `COUNTRY` and `COUNTRY_CODE`: For international comparisons

## Troubleshooting

- Check `oil_analysis.log` for detailed information about any failed API calls
- Ensure you have internet connectivity
- If a specific indicator fails, the system will try alternative sources automatically

## Requirements

See `requirements.txt` for a complete list of dependencies.
