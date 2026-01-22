# VoxCity Building Data

This directory contains building footprint data prepared for VoxCity processing.

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
2. **Spatial Processing**: Building footprints were clipped to city bounding boxes and CRS transformed using `ogr2ogr` (GDAL).

   **Example ogr2ogr command:**
   ```bash
   ogr2ogr \
     -f GPKG output_city_buildings.gpkg \
     input_country_buildings.gpkg \
     -clipsrc city_bbox.geojson \
     -t_srs EPSG:3857
   ```

   - `output_city_buildings.gpkg`: Output file for the city's buildings
   - `input_country_buildings.gpkg`: Original EUBUCCO country file (in EPSG:3035)
   - `city_bbox.geojson`: GeoJSON file containing the bounding box (or polygon) for the city
   - `-t_srs EPSG:3857`: Reprojects the result to EPSG:3857

   **Notes:**
   - Make sure the bounding box (`city_bbox.geojson`) is in the same CRS as the input, or specify `-clipsrc` layer SRS using `-clipsrcspat_srs` if needed.
   - You must have GDAL (with ogr2ogr) installed.

3. **Output Format**: The processed building footprints are saved as GeoPackage (.gpkg) files for each city.

### Reproducibility Notes
- EUBUCCO data is openly available under a Creative Commons license.
- This processing workflow is reproducible using the example `ogr2ogr` command above and standard GDAL tools.
- The transformation preserves building geometry accuracy and ensures VoxCity compatibility.
