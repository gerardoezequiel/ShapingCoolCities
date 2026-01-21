# VoxCity Building Data

This directory contains building footprint data for VoxCity processing.

## Coordinate Reference System
- All files are in **EPSG:3857** (Web Mercator)
- This is the CRS required by VoxCity

## Files
- `{city}_buildings.gpkg` - Building footprints for each city

## Data Source and Processing

### Original Data
- **Source**: EUBUCCO (European Building Stock Characteristics in a Common and Open Database for 206 Million Buildings)
- **Download**: Country-level files downloaded from https://zenodo.org/records/7225259
- **Original CRS**: EPSG:3035 (ETRS89-extended / LAEA Europe)

### Processing Steps
1. **Raw Data Location**: Country files stored in `/Users/gerardoezequiel/Developer/ShapingCoolCities/data/0-raw/VoxCity/buildings`
2. **Spatial Processing**: Used ogr2ogr to:
   - Clip building footprints to city bounding boxes
   - Transform coordinate reference system from EPSG:3035 to EPSG:3857 (Web Mercator)
3. **Output Format**: Processed data saved as GeoPackage (.gpkg) files for each city

### Reproducibility Notes
- EUBUCCO data is openly available under Creative Commons license
- Processing workflow can be reproduced using standard GDAL/ogr2ogr tools
- Transformation preserves building geometry accuracy while ensuring VoxCity compatibility
