# Survey inflation-forecast data files

Drop the raw Excel downloads here to enable the SPF and Greenbook rows in the
**Faust–Wright horse race tab**.

## SPF (Survey of Professional Forecasters)

- Source: [Philly Fed SPF Mean Responses](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters)
- Download the **Mean Level Responses** file
- Rename to `spf_mean_level.xlsx` and place it in this directory
- Expected columns: `YEAR, QUARTER, CPI1..CPI6, PCE1..PCE6, PGDP1..PGDP6, CORECPI1..CORECPI6`

## Greenbook / Tealbook (Fed staff forecast)

- Source: [Philly Fed Greenbook Data Sets](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/greenbook-data-sets)
- Download the **row format** file
- Rename to `greenbook_row_format.xlsx` and place it here
- Expected columns: `GBdate` (or `GByear` + `GBmonth`), and forecast columns like `gPGDPF0..gPGDPF9`, `gPCPIF0..gPCPIF9`, `gPCCPIF0..gPCCPIF9`
- Note: Greenbook data is embargoed 5 years, so the public file ends around 2019.

## Blue Chip

Blue Chip is a subscription-only Wolters Kluwer product. This app ships a **free
surrogate** — the mean of Michigan Survey 1-yr (MICH) and Cleveland Fed 1-yr
(EXPINF1YR) from FRED. No file upload required.

If you have Blue Chip data through an institutional subscription, save it as
`blue_chip_cpi.xlsx` with columns `date, h1..h8` and the loader in
`src/data/surveys.py` will need a small extension (grep for `load_blue_chip_surrogate`).
