# POI JSON Coverage Report

## Basic Statistics
- raw_rows: 109385
- output_rows: 109385
- removed_empty_poi_id_rows: 0
- removed_invalid_latlon_rows: 0
- removed_duplicate_poi_id_rows: 0
- unique_poi_id: 109385
- empty_poi_json_count: 0
- empty_poi_text_count: 0
- raw_missing_poi_id_rows: 0
- raw_missing_name_rows: 0
- raw_missing_address_rows: 0
- raw_missing_lat_rows: 0
- raw_missing_lon_rows: 0

## Missing Fields
- name_missing_count / rate: 0 / 0.0000
- address_missing_count / rate: 0 / 0.0000
- city_missing_count / rate: 2863 / 0.0262
- category_l1_missing_count / rate: 45372 / 0.4148
- tags_missing_count / rate: 45372 / 0.4148

## JSON Field Coverage
- name_non_empty_count / rate: 109385 / 1.0000
- address_non_empty_count / rate: 109385 / 1.0000
- city_non_empty_count / rate: 106522 / 0.9738
- district_non_empty_count / rate: 0 / 0.0000
- business_area_non_empty_count / rate: 0 / 0.0000
- category_l1_non_empty_count / rate: 64013 / 0.5852
- category_l2_non_empty_count / rate: 0 / 0.0000
- brand_non_empty_count / rate: 0 / 0.0000
- tags_non_empty_count / rate: 64013 / 0.5852
- rating_non_empty_count / rate: 0 / 0.0000
- price_non_empty_count / rate: 0 / 0.0000
- open_hours_non_empty_count / rate: 0 / 0.0000
- location_non_empty_count / rate: 109385 / 1.0000

## Parse Warnings
- missing_category_l1: 45372
- missing_tags: 45372
- missing_city: 2863

## Manual Check Samples
- missing_category_l1: 80 / requested 80
- missing_city: 80 / requested 80
- complete_basic_fields: 80 / requested 80
- random: 60 / requested 60
