# PostgreSQL State Store v15 lexical-search evidence

## Implemented bounded contract

PostgreSQL 18 has a tenant-schema-local, generated `search_document` using the built-in `simple` `tsvector`, plus a tenant-schema-local GIN index. Migrations v14 (generated document) and v15 (GIN index) are transactionally advisory-locked and catalog-validated on every open. This grammar change needs no migration: it changes only query compilation over that existing derived document.

`PostgreSQLStateStore.search_messages()` accepts the following **ASCII-only** subset, using upper-case FTS5 operators:

- ASCII alphanumeric term; adjacent terms are implicit `AND`;
- quoted phrase containing one or more ASCII alphanumeric terms;
- ASCII alphanumeric prefix (`term*`);
- `AND`, `OR`, and binary `NOT` (for example `alpha NOT beta`), with FTS5-compatible `NOT`/`AND` precedence.

The parser emits only static PostgreSQL expression structure. Terms, phrases, and the parser-added prefix marker are separately bound parameters to `plainto_tsquery`, `phraseto_tsquery`, or `to_tsquery`; no raw user query is interpolated into SQL or a `tsquery` string. Visibility/source/role filters remain bound, fields and ordering remain allowlisted, and the schema is the trusted tenant schema derived from canonical `HERMES_HOME`—a caller cannot select a tenant.

CJK is intentionally separate: only literal, whitespace-separated CJK/ASCII-alphanumeric tokens are accepted, each as a parameterized canonical-row substring requirement. This route has no tokenizer, phrase, prefix, boolean, ranking, or ordering equivalence claim; it orders newest-first by timestamp/id and uses a bounded content prefix snippet.

## Explicit fail-closed rejection

`PostgreSQLSearchQueryError` is raised for unclosed/malformed syntax and for every unsupported or ambiguous FTS5 feature: parenthesized grouping (the SQLite compatibility facade strips it), leading `NOT`, phrase prefixes, `NEAR`, column selectors, punctuation/operator syntax, non-ASCII lexical tokens, and CJK mixed with grammar syntax. Rejection is explicit; it never becomes an empty query, a broad literal search, or an altered boolean query.

## Verified

The live PostgreSQL 18 integration suite exercises fresh/upgrade ledgers through v15 and catalog drift checks. Its SQLite/PG differential corpus compares normalized identity rows for literal terms, implicit `AND`, phrases, prefixes, `AND`/`OR`/binary `NOT`, CJK literals, source/role/exclusion/visibility filters, snippets, deterministic `newest`/`oldest` pagination, generated-document update/delete maintenance, `REINDEX`, and unavailable `pg_trgm`/`vector` extensions. Snippets and rank values are deliberately not compared byte-for-byte.

## Remaining gaps / non-claims

This is not complete FTS5 equivalence or a cutover. It does not claim BM25/rank parity, FTS5 tokenizer/CJK-bigram/trigram parity, parenthesized grouping, phrase-prefix/`NEAR`/column-query support, SQLite FTS corruption fail-open/rebuild behavior, full tool-body rules, contextual neighboring-message projection, query-plan guarantees, RLS/role-based tenant authorization, runtime consumer selection, SQLite import/export, dual-write, or production configuration/deployment. `pgvector` is not used as FTS.
