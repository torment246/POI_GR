"""Query-guided relational Semantic ID components."""

from .relations import (
    DIRECT_NUMERIC_RELATION_TYPES,
    ExtractionResult,
    NumericRelation,
    QueryRelationFeatures,
    RelationConflict,
    extract_numeric_relations,
    extract_query_relation_features,
    parse_contextual_integer,
)
from .proxy import (
    ProxyResult,
    QgrSidProxyError,
    evaluate_lexical_relation_proxy,
    profile_temporal_split,
    validate_proxy_output,
)
from .geo_proxy import (
    GeoProxyResult,
    QgrSidGeoProxyError,
    derive_geo_candidates,
    evaluate_geo_relation_proxy,
    validate_geo_proxy_output,
)
from .embedding_proxy import (
    EmbeddingProxyResult,
    QgrSidEmbeddingProxyError,
    rank_embedding_candidates,
    run_query_embedding_proxy,
    validate_embedding_proxy_output,
)
from .reliable_proxy import (
    QgrSidReliableProxyError,
    ReliableProxyResult,
    run_reliable_lexical_proxy,
    select_reliable_relation_types,
    validate_reliable_proxy_output,
)

__all__ = [
    "DIRECT_NUMERIC_RELATION_TYPES",
    "ExtractionResult",
    "EmbeddingProxyResult",
    "GeoProxyResult",
    "NumericRelation",
    "QueryRelationFeatures",
    "RelationConflict",
    "ProxyResult",
    "QgrSidProxyError",
    "QgrSidGeoProxyError",
    "QgrSidEmbeddingProxyError",
    "QgrSidReliableProxyError",
    "ReliableProxyResult",
    "derive_geo_candidates",
    "evaluate_geo_relation_proxy",
    "evaluate_lexical_relation_proxy",
    "extract_numeric_relations",
    "extract_query_relation_features",
    "parse_contextual_integer",
    "profile_temporal_split",
    "rank_embedding_candidates",
    "run_query_embedding_proxy",
    "run_reliable_lexical_proxy",
    "select_reliable_relation_types",
    "validate_embedding_proxy_output",
    "validate_proxy_output",
    "validate_geo_proxy_output",
    "validate_reliable_proxy_output",
]
