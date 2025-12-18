"""Document registry for tracking ingested documents."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """Information about an ingested document."""
    
    doc_id: str
    source_path: str
    company_name: Optional[str] = None
    filing_type: Optional[str] = None
    filing_date: Optional[str] = None
    
    chunk_count: int = 0
    summary_count: int = 0
    total_chars: int = 0
    sections: List[str] = field(default_factory=list)
    
    ingested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentInfo":
        return cls(**data)


class DocumentRegistry:
    """Registry for tracking ingested documents."""
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = Path(persist_path) if persist_path else None
        self._documents: Dict[str, DocumentInfo] = {}
        
        if self.persist_path and self.persist_path.exists():
            self._load()
        
        logger.info(f"DocumentRegistry: {len(self._documents)} documents loaded")
    
    def register(self, doc_info: DocumentInfo) -> None:
        self._documents[doc_info.doc_id] = doc_info
        self._save()
    
    def unregister(self, doc_id: str) -> bool:
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._save()
            return True
        return False
    
    def get(self, doc_id: str) -> Optional[DocumentInfo]:
        return self._documents.get(doc_id)
    
    def is_ingested(self, source_path: str) -> bool:
        return any(d.source_path == source_path for d in self._documents.values())
    
    def get_by_company(self, company_name: str) -> List[DocumentInfo]:
        return [
            d for d in self._documents.values()
            if d.company_name and company_name.lower() in d.company_name.lower()
        ]
    
    def get_by_filing_type(self, filing_type: str) -> List[DocumentInfo]:
        return [
            d for d in self._documents.values()
            if d.filing_type and d.filing_type.upper() == filing_type.upper()
        ]
    
    def get_companies(self) -> List[str]:
        companies = set()
        for d in self._documents.values():
            if d.company_name:
                companies.add(d.company_name)
        return sorted(companies)
    
    def get_filing_types(self) -> List[str]:
        types = set()
        for d in self._documents.values():
            if d.filing_type:
                types.add(d.filing_type)
        return sorted(types)
    
    @property
    def all_documents(self) -> List[DocumentInfo]:
        return list(self._documents.values())
    
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self._documents),
            "companies": self.get_companies(),
            "filing_types": self.get_filing_types(),
            "total_chunks": sum(d.chunk_count for d in self._documents.values()),
        }
    
    def _save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {doc_id: doc.to_dict() for doc_id, doc in self._documents.items()}
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        with open(self.persist_path, "r") as f:
            data = json.load(f)
        self._documents = {
            doc_id: DocumentInfo.from_dict(doc_data)
            for doc_id, doc_data in data.items()
        }
    
    def clear(self) -> None:
        self._documents = {}
        self._save()
