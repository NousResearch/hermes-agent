#!/usr/bin/env python3
"""Tool to manually ingest, parse, and verify PDF documents."""

import json
import logging
from pathlib import Path

from agent.graph_manager import GraphManager
from hermes_constants import get_hermes_dir

logger = logging.getLogger(__name__)

def ingest_research_pdf(
    pdf_path: str,
    name: str = "",
    group_id: str = "research",
) -> str:
    """Uses OpenDataLoader to parse a PDF and ingest it into the GraphManager."""
    pdf_file = Path(pdf_path).expanduser()
    if not pdf_file.exists():
        return json.dumps({"success": False, "error": f"PDF not found at {pdf_path}"})

    try:
        import opendataloader_pdf
    except ImportError:
        return json.dumps({
            "success": False, 
            "error": "opendataloader-pdf is not installed. Please install it first."
        })

    try:
        # Initialize graph manager
        db_path = get_hermes_dir("context-graph/kuzu_db", "kuzu_db")
        manager = GraphManager(db_path)
        
        # Async execution of graph manager inside sync tool wrapper
        import asyncio
        loop = asyncio.get_event_loop()
        
        import os
        metadata = {
            "source_type": "manual_pdf_ingestion"
        }
        
        result = loop.run_until_complete(
            manager.add_academic_pdf(
                pdf_path=str(pdf_file),
                name=name,
                metadata=metadata,
                group_id=group_id
            )
        )
        
        return json.dumps({
            "success": True,
            "message": f"Successfully ingested {pdf_file.name}",
            "episode_uuid": result.get("episode_uuid"),
            "entities_extracted": result.get("entities_extracted"),
            "edges_extracted": result.get("edges_extracted")
        }, indent=2)

    except Exception as e:
        logger.error("Failed to ingest PDF: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def check_pdf_requirements() -> bool:
    try:
        import opendataloader_pdf  # noqa
        return True
    except ImportError:
        return False


PDF_INGESTION_SCHEMA = {
    "name": "ingest_research_pdf",
    "description": (
        "Parses an academic or complex PDF document and ingests its fact-based "
        "content securely into the Hermes Knowledge Graph. Uses XY-Cut++ to preserve "
        "reading order and tables, then performs citation verification to prevent hallucination."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pdf_path": {
                "type": "string",
                "description": "Absolute path to the PDF file to ingest.",
            },
            "name": {
                "type": "string",
                "description": "Optional title for the document.",
            },
            "group_id": {
                "type": "string",
                "description": "Namespace inside the Graphiti DB (default: 'research')",
                "default": "research",
            },
        },
        "required": ["pdf_path"],
    },
}

from tools.registry import registry

registry.register(
    name="ingest_research_pdf",
    toolset="knowledge",
    schema=PDF_INGESTION_SCHEMA,
    handler=lambda args, **kw: ingest_research_pdf(
        pdf_path=args.get("pdf_path"),
        name=args.get("name", ""),
        group_id=args.get("group_id", "research"),
    ),
    check_fn=check_pdf_requirements,
    emoji="📄",
    description="Ingest PDFs into the academic knowledge graph using robust layout parsing",
    mutates=True,
    requires_confirmation=True,
)
