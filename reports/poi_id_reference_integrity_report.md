# POI ID Reference Integrity Report

## POI ID Basic Stats
- poi_sid_train.parquet: rows=109385, unique_poi_id=109385, empty_poi_id_rows=0, duplicate_poi_id_rows=0
- poi_embedding_meta.parquet: rows=109385, unique_poi_id=109385, empty_poi_id_rows=0, duplicate_poi_id_rows=0
- poi_geo_meta.parquet: rows=109385, unique_poi_id=109385, empty_poi_id_rows=0, duplicate_poi_id_rows=0
- qrels.csv: rows=9735, unique_poi_id=9657, empty_poi_id_rows=0, duplicate_poi_id_rows=78
- candidates.csv: rows=63804, unique_poi_id=62512, empty_poi_id_rows=0, duplicate_poi_id_rows=1292

## Alignment
- sid_minus_embedding_meta_count: 0
- embedding_meta_minus_sid_count: 0
- sid_minus_geo_meta_count: 0
- geo_meta_minus_sid_count: 0
- sid_embedding_row_order_same: True
- embedding_geo_row_order_same: True
- sid_embedding_row_id_same: True
- embedding_geo_row_id_same: True

## qrels/candidates References
- qrel_poi_id_missing_in_sid_count: 0
- candidate_poi_id_missing_in_sid_count: 0
- qrel_duplicate_query_poi_pair_rows: 0
- candidate_duplicate_query_poi_pair_rows: 0
- qrel_query_poi_pairs_missing_in_candidates_count: 0

## Verdict
- PASS: poi_id references are complete across SID, embedding meta, geo meta, qrels, and candidates.
