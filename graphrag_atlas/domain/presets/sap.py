from __future__ import annotations

import re

from graphrag_atlas.domain.base import DomainPack, RelationSpec

SAP_RELATIONS = [
    RelationSpec(
        "WRITES_TO_TABLE", "Writes records to an SAP transparent table", "directed"
    ),
    RelationSpec(
        "READS_FROM_TABLE", "Reads records from an SAP transparent table", "directed"
    ),
    RelationSpec("INVOKES_BAPI", "Calls a Business API function module", "directed"),
    RelationSpec(
        "EXPOSES_ODATA",
        "Exposes an OData service or RAP behavior definition",
        "directed",
    ),
    RelationSpec(
        "EXTENDS_CDS", "Extends a Core Data Services view or entity", "directed"
    ),
    RelationSpec("TRIGGERS_IDOC", "Dispatches an Intermediate Document", "directed"),
    RelationSpec(
        "POSTS_DOCUMENT",
        "Posts a financial or material accounting document",
        "directed",
    ),
    RelationSpec(
        "REPLACES", "Modern clean core replacement for obsolete artifact", "directed"
    ),
]

# Fast-path deterministic regexes for standard SAP artifacts
SAP_REGEXES = [
    re.compile(
        r"\b([A-Z][A-Z0-9_]{2,15}_[A-Z0-9_]+)\b"
    ),  # Standard DDIC / custom table
    re.compile(r"\b(Z[A-Z0-9_]{2,29}|Y[A-Z0-9_]{2,29})\b"),  # Customer namespace Z*/Y*
    re.compile(r"\b(BAPI_[A-Z0-9_]+)\b"),  # BAPI function module
    re.compile(
        r"\b(I_[A-Z0-9_]+|C_[A-Z0-9_]+|R_[A-Z0-9_]+)\b"
    ),  # CDS views (Interface/Consumption)
]

SAP_ACRONYMS = {
    "ACDOCA": "Universal Journal Entry Line Items",
    "BKPF": "Accounting Document Header",
    "BSEG": "Accounting Document Segment",
    "MARA": "General Material Data",
    "VBAK": "Sales Document Header",
    "VBAP": "Sales Document Item",
    "EKKO": "Purchasing Document Header",
    "EKPO": "Purchasing Document Item",
    "RAP": "RESTful Application Programming Model",
    "CAP": "Cloud Application Programming Model",
    "CDS": "Core Data Services",
    "BAPI": "Business Application Programming Interface",
    "IDOC": "Intermediate Document",
}

SAP_STOPLIST = {
    "SAP",
    "ERP",
    "ABAP",
    "FIORI",
    "GUI",
    "HANA",
    "DATABASE",
    "SYSTEM",
    "CLIENT",
    "TRANSACTION",
    "PROGRAM",
    "INCLUDE",
    "REPORT",
    "MODULE",
}

SAP_VOCABULARY = {
    "acdoca",
    "bapi",
    "cds",
    "odata",
    "fiori",
    "abap",
    "clean core",
    "posting",
    "gl",
    "ledger",
    "idoc",
    "badi",
    "enhancement",
    "material",
}


class SAPCleanCorePack(DomainPack):
    """Domain pack optimized for SAP ERP, Clean Core migrations, and ABAP Cloud development."""

    def __init__(self) -> None:
        super().__init__(
            name="sap_clean_core",
            ontology=SAP_RELATIONS,
            entity_regexes=SAP_REGEXES,
            stoplist=SAP_STOPLIST,
            acronyms=SAP_ACRONYMS,
            domain_vocabulary=SAP_VOCABULARY,
        )
