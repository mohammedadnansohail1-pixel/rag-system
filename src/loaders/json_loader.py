"""JSON Loader for RAG System - follows existing loader patterns."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.loaders.base import BaseLoader, Document
from src.loaders.factory import register_loader


@dataclass
class JSONFieldConfig:
    """Configuration for JSON to Document conversion."""
    content_fields: List[str]
    metadata_fields: List[str] = field(default_factory=list)
    content_template: Optional[str] = None
    id_field: Optional[str] = None


@register_loader("json")
class JSONLoader(BaseLoader):
    """
    Load JSON files into Documents.
    
    Supports both array of objects and object with array field.
    """
    
    def __init__(self, config: Optional[JSONFieldConfig] = None):
        self.config = config
    
    def supported_extensions(self) -> List[str]:
        return [".json"]
    
    def load(self, file_path: Path) -> List[Document]:
        """Load JSON file with auto-detected or configured field mapping."""
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get records
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = next((v for v in data.values() if isinstance(v, list)), [])
        else:
            return []
        
        # Auto-detect config if not provided
        config = self.config or self._auto_detect_config(records[:5])
        
        # Convert to Documents
        documents = []
        for i, record in enumerate(records):
            doc = self._record_to_document(record, i, str(file_path), config)
            if doc:
                documents.append(doc)
        
        return documents
    
    def load_with_config(self, file_path: Path, config: JSONFieldConfig) -> List[Document]:
        """Load with explicit config."""
        self.config = config
        return self.load(file_path)
    
    def _auto_detect_config(self, sample_records: List[Dict]) -> JSONFieldConfig:
        """Auto-detect field config from sample records."""
        if not sample_records:
            return JSONFieldConfig(content_fields=[])
        
        text_fields = []
        meta_fields = []
        id_field = None
        
        all_fields = set().union(*(r.keys() for r in sample_records))
        
        for field in all_fields:
            values = [str(r.get(field, "")) for r in sample_records if field in r]
            avg_len = sum(len(v) for v in values) / len(values) if values else 0
            
            if 'id' in field.lower():
                id_field = field
                meta_fields.append(field)
            elif avg_len > 100:
                text_fields.append(field)
            else:
                meta_fields.append(field)
        
        return JSONFieldConfig(
            content_fields=text_fields or list(all_fields)[:2],
            metadata_fields=meta_fields[:10],
            id_field=id_field
        )
    
    def _record_to_document(
        self, 
        record: Dict, 
        index: int, 
        source: str,
        config: JSONFieldConfig
    ) -> Optional[Document]:
        """Convert a JSON record to Document."""
        # Build content
        if config.content_template:
            try:
                content = config.content_template.format(
                    **{k: (v if v else "") for k, v in record.items()}
                )
            except KeyError:
                content = self._default_content(record, config)
        else:
            content = self._default_content(record, config)
        
        if not content.strip():
            return None
        
        # Build metadata
        metadata = {"source": source}
        for f in config.metadata_fields:
            if f in record:
                metadata[f] = record[f]
        
        # Add doc_id
        if config.id_field and config.id_field in record:
            metadata["doc_id"] = str(record[config.id_field])
        else:
            metadata["doc_id"] = str(index)
        
        return Document(content=content, metadata=metadata)
    
    def _default_content(self, record: Dict, config: JSONFieldConfig) -> str:
        """Build content from content_fields."""
        parts = []
        for field in config.content_fields:
            if field in record and record[field]:
                parts.append(str(record[field]))
        return "\n\n".join(parts)
