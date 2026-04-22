#!/usr/bin/env python3
"""Hermes-native DealSeek UGC analytics tools.

TrackerMane is the primary analytics source:
- Postgres for canonical content/account metadata
- ClickHouse for latest + historical snapshot metrics

SideShift is the secondary/fallback source for UGC-program-specific data.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import os
import uuid
from typing import Any, Iterable
from urllib.parse import urljoin

import requests

from tools.registry import registry, tool_error, tool_result

DEFAULT_SIDESHIFT_API_BASE_URL = "https://app.sideshift.app/api/v1"
CLICKHOUSE_TIMEOUT = (10, 20)
SIDESHIFT_TIMEOUT = (10, 20)
POSTGRES_STATEMENT_TIMEOUT_MS = 20_000
DEFAULT_TOP_VIDEO_LIMIT = 10
DEFAULT_TOP_ACCOUNT_LIMIT = 10
DEFAULT_RESULT_LIMIT = 15
MAX_RESULT_LIMIT = 25
MAX_HISTORY_LIMIT = 50
MAX_COMMENT_LIMIT = 25
SEARCH_CANDIDATE_LIMIT = 75
TRACKER_VIDEO_SORT_FIELDS = {
    "latest_snapshot_at",
    "published_at",
    "view_count",
    "like_count",
    "comment_count",
    "share_count",
    "save_count",
    "play_count",
    "engagement_count",
}
TRACKER_ACCOUNT_SORT_FIELDS = {
    "latest_snapshot_at",
    "followers_count",
    "following_count",
    "video_count",
    "total_views",
    "total_likes",
}


def _get_trackermane_postgres_url() -> str | None:
    return os.getenv("TRACKERMANE_POSTGRES_URL") or os.getenv("POSTGRES_URL")


def _get_trackermane_clickhouse_url() -> str | None:
    return os.getenv("TRACKERMANE_CLICKHOUSE_URL") or os.getenv("CLICKHOUSE_URL")


def _get_sideshift_api_key() -> str | None:
    return os.getenv("SIDESHIFT_API_KEY")


def _get_sideshift_base_url() -> str:
    return (os.getenv("SIDESHIFT_API_BASE_URL") or DEFAULT_SIDESHIFT_API_BASE_URL).rstrip("/")


def _check_trackermane_available() -> bool:
    return bool(_get_trackermane_postgres_url() and _get_trackermane_clickhouse_url())


def _check_sideshift_available() -> bool:
    return bool(_get_sideshift_api_key())


def _clamp(value: Any, fallback: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(1, min(parsed, maximum))


def _clamp_days(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return min(parsed, 3650)


def _normalize_sort_order(value: Any) -> str:
    return "asc" if str(value).lower() == "asc" else "desc"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, decimal.Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return value


def _get_sortable_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return 0.0
    return 0.0


def _escape_clickhouse_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _to_clickhouse_array(values: Iterable[str]) -> str:
    return ", ".join(f"'{_escape_clickhouse_string(value)}'" for value in values)


def _parse_json_each_row(raw_text: str) -> list[dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _build_playback_url(stream_uid: str | None, stream_customer_subdomain: str | None) -> str | None:
    if not stream_uid:
        return None
    if stream_customer_subdomain:
        return f"https://{stream_customer_subdomain}/{stream_uid}/iframe"
    return f"https://iframe.videodelivery.net/{stream_uid}"


def _get_pg_connection():
    postgres_url = _get_trackermane_postgres_url()
    if not postgres_url:
        raise RuntimeError(
            "TrackerMane Postgres is not configured. Set TRACKERMANE_POSTGRES_URL or POSTGRES_URL."
        )
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - protected by dependency + runtime message
        raise RuntimeError(
            "TrackerMane tools require psycopg. Install hermes-agent with psycopg support."
        ) from exc

    conn = psycopg.connect(postgres_url, autocommit=True, row_factory=dict_row)
    conn.execute(f"SET statement_timeout = '{POSTGRES_STATEMENT_TIMEOUT_MS}ms'")
    return conn


def _run_postgres_query(sql: str, params: Iterable[Any] | None = None) -> list[dict[str, Any]]:
    with _get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params or ()))
            return [_json_ready(row) for row in cur.fetchall()]


def _run_clickhouse_query(sql: str) -> list[dict[str, Any]]:
    clickhouse_url = _get_trackermane_clickhouse_url()
    if not clickhouse_url:
        raise RuntimeError(
            "TrackerMane ClickHouse is not configured. Set TRACKERMANE_CLICKHOUSE_URL or CLICKHOUSE_URL."
        )
    response = requests.post(
        clickhouse_url,
        data=sql.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        timeout=CLICKHOUSE_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_json_each_row(response.text)


def _fetch_content_metadata_by_ids(ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    rows = _run_postgres_query(
        """
        SELECT
          ci.id,
          ci.platform,
          ci.platform_content_id,
          ci.account_id,
          ci.canonical_url,
          ci.content_kind,
          ci.title,
          ci.description,
          ci.published_at,
          ci.duration_seconds,
          ci.thumbnail_url,
          ci.platform_permalink,
          ci.first_seen_at,
          ci.latest_ingested_at,
          ci.latest_comment_sync_at,
          ci.comments_supported,
          ci.comments_capture_mode,
          ci.stream_uid,
          ci.stream_customer_subdomain,
          ca.handle AS account_handle,
          ca.display_name AS account_display_name,
          ca.canonical_url AS account_canonical_url
        FROM content_items ci
        LEFT JOIN content_accounts ca ON ca.id = ci.account_id
        WHERE ci.id = ANY(%s::uuid[])
        """,
        [ids],
    )
    return {row["id"]: row for row in rows}


def _fetch_account_metadata_by_ids(ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    rows = _run_postgres_query(
        """
        SELECT
          id,
          platform,
          platform_account_id,
          handle,
          display_name,
          canonical_url,
          is_verified,
          created_at,
          updated_at,
          profile_image_url
        FROM content_accounts
        WHERE id = ANY(%s::uuid[])
        """,
        [ids],
    )
    return {row["id"]: row for row in rows}


def _search_content_metadata_candidates(
    query: str,
    platform: str | None,
    published_within_days: int | None,
) -> list[dict[str, Any]]:
    search_term = f"%{query}%"
    params: list[Any] = [search_term]
    clauses = [
        """
        (
          ci.title ILIKE %s OR
          ci.description ILIKE %s OR
          ci.platform_content_id ILIKE %s OR
          ci.canonical_url ILIKE %s OR
          ci.platform_permalink ILIKE %s OR
          ca.handle ILIKE %s OR
          ca.display_name ILIKE %s
        )
        """.strip()
    ]
    params.extend([search_term] * 6)
    if platform:
        clauses.append("ci.platform = %s")
        params.append(platform)
    if published_within_days:
        clauses.append("ci.published_at >= NOW() - (%s::text || ' days')::interval")
        params.append(published_within_days)
    params.append(SEARCH_CANDIDATE_LIMIT)
    return _run_postgres_query(
        f"""
        SELECT
          ci.id,
          ci.platform,
          ci.platform_content_id,
          ci.account_id,
          ci.canonical_url,
          ci.content_kind,
          ci.title,
          ci.description,
          ci.published_at,
          ci.duration_seconds,
          ci.thumbnail_url,
          ci.platform_permalink,
          ci.first_seen_at,
          ci.latest_ingested_at,
          ci.latest_comment_sync_at,
          ci.comments_supported,
          ci.comments_capture_mode,
          ci.stream_uid,
          ci.stream_customer_subdomain,
          ca.handle AS account_handle,
          ca.display_name AS account_display_name,
          ca.canonical_url AS account_canonical_url
        FROM content_items ci
        LEFT JOIN content_accounts ca ON ca.id = ci.account_id
        WHERE {' AND '.join(clauses)}
        ORDER BY COALESCE(ci.published_at, ci.latest_ingested_at, ci.first_seen_at) DESC, ci.id DESC
        LIMIT %s
        """,
        params,
    )


def _search_account_metadata_candidates(query: str, platform: str | None, limit: int) -> list[dict[str, Any]]:
    search_term = f"%{query}%"
    params: list[Any] = [search_term, search_term, search_term, search_term]
    clauses = [
        """
        (
          handle ILIKE %s OR
          display_name ILIKE %s OR
          platform_account_id ILIKE %s OR
          canonical_url ILIKE %s
        )
        """.strip()
    ]
    if platform:
        clauses.append("platform = %s")
        params.append(platform)
    params.append(limit)
    return _run_postgres_query(
        f"""
        SELECT
          id,
          platform,
          platform_account_id,
          handle,
          display_name,
          canonical_url,
          is_verified,
          created_at,
          updated_at,
          profile_image_url
        FROM content_accounts
        WHERE {' AND '.join(clauses)}
        ORDER BY updated_at DESC, id DESC
        LIMIT %s
        """,
        params,
    )


def _fetch_latest_video_metrics_by_ids(ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    return _run_clickhouse_query(
        f"""
        SELECT
          content_id,
          max(snapshot_at) AS latest_snapshot_at,
          argMax(platform, snapshot_at) AS platform,
          argMax(platform_content_id, snapshot_at) AS platform_content_id,
          argMax(view_count, snapshot_at) AS view_count,
          argMax(like_count, snapshot_at) AS like_count,
          argMax(comment_count, snapshot_at) AS comment_count,
          argMax(share_count, snapshot_at) AS share_count,
          argMax(save_count, snapshot_at) AS save_count,
          argMax(play_count, snapshot_at) AS play_count,
          argMax(engagement_count, snapshot_at) AS engagement_count,
          argMax(published_at, snapshot_at) AS published_at
        FROM video_stat_snapshots
        WHERE content_id IN ({_to_clickhouse_array(ids)})
        GROUP BY content_id
        FORMAT JSONEachRow
        """
    )


def _fetch_top_latest_video_metrics(
    *,
    platform: str | None,
    published_within_days: int | None,
    sort_by: str,
    sort_order: str,
    limit: int,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    if platform:
        filters.append(f"platform = '{_escape_clickhouse_string(platform)}'")
    if published_within_days:
        filters.append(f"published_at >= now() - INTERVAL {published_within_days} DAY")
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    return _run_clickhouse_query(
        f"""
        SELECT *
        FROM (
          SELECT
            content_id,
            max(snapshot_at) AS latest_snapshot_at,
            argMax(platform, snapshot_at) AS platform,
            argMax(platform_content_id, snapshot_at) AS platform_content_id,
            argMax(view_count, snapshot_at) AS view_count,
            argMax(like_count, snapshot_at) AS like_count,
            argMax(comment_count, snapshot_at) AS comment_count,
            argMax(share_count, snapshot_at) AS share_count,
            argMax(save_count, snapshot_at) AS save_count,
            argMax(play_count, snapshot_at) AS play_count,
            argMax(engagement_count, snapshot_at) AS engagement_count,
            argMax(published_at, snapshot_at) AS published_at
          FROM video_stat_snapshots
          GROUP BY content_id
        )
        {where_clause}
        ORDER BY {sort_by} {sort_order}, latest_snapshot_at DESC
        LIMIT {limit}
        FORMAT JSONEachRow
        """
    )


def _fetch_latest_account_metrics_by_ids(ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    return _run_clickhouse_query(
        f"""
        SELECT
          account_id,
          max(snapshot_at) AS latest_snapshot_at,
          argMax(platform, snapshot_at) AS platform,
          argMax(platform_account_id, snapshot_at) AS platform_account_id,
          argMax(followers_count, snapshot_at) AS followers_count,
          argMax(following_count, snapshot_at) AS following_count,
          argMax(video_count, snapshot_at) AS video_count,
          argMax(total_views, snapshot_at) AS total_views,
          argMax(total_likes, snapshot_at) AS total_likes
        FROM account_stat_snapshots
        WHERE account_id IN ({_to_clickhouse_array(ids)})
        GROUP BY account_id
        FORMAT JSONEachRow
        """
    )


def _fetch_top_latest_account_metrics(
    *, platform: str | None, sort_by: str, sort_order: str, limit: int
) -> list[dict[str, Any]]:
    where_clause = (
        f"WHERE platform = '{_escape_clickhouse_string(platform)}'" if platform else ""
    )
    return _run_clickhouse_query(
        f"""
        SELECT *
        FROM (
          SELECT
            account_id,
            max(snapshot_at) AS latest_snapshot_at,
            argMax(platform, snapshot_at) AS platform,
            argMax(platform_account_id, snapshot_at) AS platform_account_id,
            argMax(followers_count, snapshot_at) AS followers_count,
            argMax(following_count, snapshot_at) AS following_count,
            argMax(video_count, snapshot_at) AS video_count,
            argMax(total_views, snapshot_at) AS total_views,
            argMax(total_likes, snapshot_at) AS total_likes
          FROM account_stat_snapshots
          GROUP BY account_id
        )
        {where_clause}
        ORDER BY {sort_by} {sort_order}, latest_snapshot_at DESC
        LIMIT {limit}
        FORMAT JSONEachRow
        """
    )


def _map_video(metric: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "contentId": metric.get("content_id"),
        "platform": metadata.get("platform") or metric.get("platform"),
        "platformContentId": metadata.get("platform_content_id") or metric.get("platform_content_id"),
        "title": metadata.get("title"),
        "description": metadata.get("description"),
        "contentKind": metadata.get("content_kind"),
        "canonicalUrl": metadata.get("canonical_url"),
        "platformPermalink": metadata.get("platform_permalink"),
        "thumbnailUrl": metadata.get("thumbnail_url"),
        "publishedAt": metadata.get("published_at") or metric.get("published_at"),
        "durationSeconds": metadata.get("duration_seconds"),
        "accountId": metadata.get("account_id"),
        "accountHandle": metadata.get("account_handle"),
        "accountDisplayName": metadata.get("account_display_name"),
        "accountCanonicalUrl": metadata.get("account_canonical_url"),
        "latestSnapshotAt": metric.get("latest_snapshot_at"),
        "viewCount": metric.get("view_count"),
        "likeCount": metric.get("like_count"),
        "commentCount": metric.get("comment_count"),
        "shareCount": metric.get("share_count"),
        "saveCount": metric.get("save_count"),
        "playCount": metric.get("play_count"),
        "engagementCount": metric.get("engagement_count"),
        "playbackUrl": _build_playback_url(
            metadata.get("stream_uid"), metadata.get("stream_customer_subdomain")
        ),
        "latestCommentSyncAt": metadata.get("latest_comment_sync_at"),
        "latestIngestedAt": metadata.get("latest_ingested_at"),
    }


def _map_account(metric: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "accountId": metric.get("account_id"),
        "platform": metadata.get("platform") or metric.get("platform"),
        "platformAccountId": metadata.get("platform_account_id") or metric.get("platform_account_id"),
        "handle": metadata.get("handle"),
        "displayName": metadata.get("display_name"),
        "canonicalUrl": metadata.get("canonical_url"),
        "isVerified": metadata.get("is_verified"),
        "profileImageUrl": metadata.get("profile_image_url"),
        "latestSnapshotAt": metric.get("latest_snapshot_at"),
        "followersCount": metric.get("followers_count"),
        "followingCount": metric.get("following_count"),
        "videoCount": metric.get("video_count"),
        "totalViews": metric.get("total_views"),
        "totalLikes": metric.get("total_likes"),
    }


def get_trackermane_overview(options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    platform = (options.get("platform") or "").strip() or None
    published_within_days = _clamp_days(options.get("publishedWithinDays"))
    top_video_limit = _clamp(options.get("topVideoLimit"), DEFAULT_TOP_VIDEO_LIMIT, MAX_RESULT_LIMIT)
    top_account_limit = _clamp(options.get("topAccountLimit"), DEFAULT_TOP_ACCOUNT_LIMIT, MAX_RESULT_LIMIT)

    filters: list[str] = []
    if platform:
        filters.append(f"platform = '{_escape_clickhouse_string(platform)}'")
    if published_within_days:
        filters.append(f"published_at >= now() - INTERVAL {published_within_days} DAY")
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    summary_rows = _run_clickhouse_query(
        f"""
        SELECT
          count() AS video_count,
          sum(view_count) AS total_views,
          sum(like_count) AS total_likes,
          sum(comment_count) AS total_comments,
          sum(share_count) AS total_shares,
          sum(save_count) AS total_saves,
          sum(play_count) AS total_plays,
          sum(engagement_count) AS total_engagements,
          round(avg(view_count), 2) AS avg_views,
          quantileExact(0.5)(view_count) AS median_views,
          countIf(view_count >= 10000) AS videos_over_10k,
          countIf(view_count >= 100000) AS videos_over_100k
        FROM (
          SELECT
            content_id,
            argMax(view_count, snapshot_at) AS view_count,
            argMax(like_count, snapshot_at) AS like_count,
            argMax(comment_count, snapshot_at) AS comment_count,
            argMax(share_count, snapshot_at) AS share_count,
            argMax(save_count, snapshot_at) AS save_count,
            argMax(play_count, snapshot_at) AS play_count,
            argMax(engagement_count, snapshot_at) AS engagement_count,
            argMax(published_at, snapshot_at) AS published_at,
            argMax(platform, snapshot_at) AS platform
          FROM video_stat_snapshots
          GROUP BY content_id
        )
        {where_clause}
        FORMAT JSONEachRow
        """
    )
    top_video_metrics = _fetch_top_latest_video_metrics(
        platform=platform,
        published_within_days=published_within_days,
        sort_by="view_count",
        sort_order="desc",
        limit=top_video_limit,
    )
    top_account_metrics = _fetch_top_latest_account_metrics(
        platform=platform,
        sort_by="total_views",
        sort_order="desc",
        limit=top_account_limit,
    )
    video_metadata = _fetch_content_metadata_by_ids([row["content_id"] for row in top_video_metrics])
    account_metadata = _fetch_account_metadata_by_ids([row["account_id"] for row in top_account_metrics])

    return {
        "source": "TrackerMane",
        "filters": {"platform": platform, "publishedWithinDays": published_within_days},
        "notes": (
            [
                "Date filters apply to the video summary and top videos. Top accounts are filtered by platform only because account snapshots do not carry upload-date semantics."
            ]
            if published_within_days
            else []
        ),
        "summary": summary_rows[0] if summary_rows else None,
        "topVideos": [_map_video(row, video_metadata.get(row["content_id"])) for row in top_video_metrics],
        "topAccounts": [_map_account(row, account_metadata.get(row["account_id"])) for row in top_account_metrics],
    }


def search_trackermane_videos(options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    query = (options.get("query") or "").strip()
    platform = (options.get("platform") or "").strip() or None
    published_within_days = _clamp_days(options.get("publishedWithinDays"))
    sort_by = options.get("sortBy") or "view_count"
    if sort_by not in TRACKER_VIDEO_SORT_FIELDS:
        raise ValueError(f"Invalid sortBy '{sort_by}'.")
    sort_order = _normalize_sort_order(options.get("sortOrder"))
    limit = _clamp(options.get("limit"), DEFAULT_RESULT_LIMIT, MAX_RESULT_LIMIT)

    if query:
        metadata_rows = _search_content_metadata_candidates(query, platform, published_within_days)
        metadata_map = {row["id"]: row for row in metadata_rows}
        metrics = _fetch_latest_video_metrics_by_ids([row["id"] for row in metadata_rows])
        reverse = sort_order == "desc"
        metrics = sorted(
            metrics,
            key=lambda row: (
                _get_sortable_value(row.get(sort_by)),
                _get_sortable_value(row.get("latest_snapshot_at")),
            ),
            reverse=reverse,
        )[:limit]
    else:
        metrics = _fetch_top_latest_video_metrics(
            platform=platform,
            published_within_days=published_within_days,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
        )
        metadata_map = _fetch_content_metadata_by_ids([row["content_id"] for row in metrics])

    return {
        "source": "TrackerMane",
        "filters": {
            "query": query or None,
            "platform": platform,
            "publishedWithinDays": published_within_days,
            "sortBy": sort_by,
            "sortOrder": sort_order,
            "limit": limit,
        },
        "results": [_map_video(row, metadata_map.get(row["content_id"])) for row in metrics],
    }


def search_trackermane_accounts(options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = options or {}
    query = (options.get("query") or "").strip()
    platform = (options.get("platform") or "").strip() or None
    sort_by = options.get("sortBy") or "total_views"
    if sort_by not in TRACKER_ACCOUNT_SORT_FIELDS:
        raise ValueError(f"Invalid sortBy '{sort_by}'.")
    sort_order = _normalize_sort_order(options.get("sortOrder"))
    limit = _clamp(options.get("limit"), DEFAULT_RESULT_LIMIT, MAX_RESULT_LIMIT)

    if query:
        metadata_rows = _search_account_metadata_candidates(query, platform, SEARCH_CANDIDATE_LIMIT)
        metadata_map = {row["id"]: row for row in metadata_rows}
        metrics = _fetch_latest_account_metrics_by_ids([row["id"] for row in metadata_rows])
        reverse = sort_order == "desc"
        metrics = sorted(
            metrics,
            key=lambda row: (
                _get_sortable_value(row.get(sort_by)),
                _get_sortable_value(row.get("latest_snapshot_at")),
            ),
            reverse=reverse,
        )[:limit]
    else:
        metrics = _fetch_top_latest_account_metrics(
            platform=platform,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
        )
        metadata_map = _fetch_account_metadata_by_ids([row["account_id"] for row in metrics])

    return {
        "source": "TrackerMane",
        "filters": {
            "query": query or None,
            "platform": platform,
            "sortBy": sort_by,
            "sortOrder": sort_order,
            "limit": limit,
        },
        "results": [_map_account(row, metadata_map.get(row["account_id"])) for row in metrics],
    }


def get_trackermane_video_details(options: dict[str, Any]) -> dict[str, Any]:
    history_limit = _clamp(options.get("historyLimit"), 12, MAX_HISTORY_LIMIT)
    comment_limit = _clamp(options.get("commentLimit"), 10, MAX_COMMENT_LIMIT)
    content_id = (options.get("contentId") or "").strip() or None
    platform_content_id = (options.get("platformContentId") or "").strip() or None
    url = (options.get("url") or "").strip() or None

    if not any([content_id, platform_content_id, url]):
        raise ValueError("Provide at least one of contentId, platformContentId, or url.")

    rows = _run_postgres_query(
        """
        SELECT
          ci.id,
          ci.platform,
          ci.platform_content_id,
          ci.account_id,
          ci.canonical_url,
          ci.content_kind,
          ci.title,
          ci.description,
          ci.published_at,
          ci.duration_seconds,
          ci.thumbnail_url,
          ci.platform_permalink,
          ci.first_seen_at,
          ci.latest_ingested_at,
          ci.latest_comment_sync_at,
          ci.comments_supported,
          ci.comments_capture_mode,
          ci.stream_uid,
          ci.stream_customer_subdomain,
          ca.handle AS account_handle,
          ca.display_name AS account_display_name,
          ca.canonical_url AS account_canonical_url
        FROM content_items ci
        LEFT JOIN content_accounts ca ON ca.id = ci.account_id
        WHERE (%s::text IS NOT NULL AND ci.id::text = %s::text)
           OR (%s::text IS NOT NULL AND ci.platform_content_id = %s)
           OR (
             %s::text IS NOT NULL AND (
               ci.canonical_url = %s OR
               ci.platform_permalink = %s
             )
           )
        ORDER BY COALESCE(ci.published_at, ci.latest_ingested_at, ci.first_seen_at) DESC
        LIMIT 1
        """,
        [content_id, content_id, platform_content_id, platform_content_id, url, url, url],
    )
    metadata = rows[0] if rows else None
    if not metadata:
        return {"source": "TrackerMane", "video": None, "history": [], "recentComments": []}

    content_uuid = metadata["id"]
    metrics = _fetch_latest_video_metrics_by_ids([content_uuid])
    video = _map_video(metrics[0], metadata) if metrics else None
    history = _run_clickhouse_query(
        f"""
        SELECT
          snapshot_at,
          view_count,
          like_count,
          comment_count,
          share_count,
          save_count,
          play_count,
          engagement_count
        FROM video_stat_snapshots
        WHERE content_id = '{_escape_clickhouse_string(content_uuid)}'
        ORDER BY snapshot_at DESC
        LIMIT {history_limit}
        FORMAT JSONEachRow
        """
    )
    recent_comments = _run_clickhouse_query(
        f"""
        SELECT
          snapshot_at,
          platform_comment_id,
          comment_posted_at,
          author_handle,
          author_profile_url,
          comment_text,
          like_count,
          reply_count
        FROM comment_snapshots
        WHERE content_id = '{_escape_clickhouse_string(content_uuid)}'
        ORDER BY snapshot_at DESC, comment_posted_at DESC
        LIMIT {comment_limit}
        FORMAT JSONEachRow
        """
    )
    return {
        "source": "TrackerMane",
        "video": video,
        "history": list(reversed(history)),
        "recentComments": recent_comments,
    }


def _call_sideshift(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = _get_sideshift_api_key()
    if not api_key:
        raise RuntimeError("SideShift is not configured. Set SIDESHIFT_API_KEY.")
    response = requests.get(
        urljoin(f"{_get_sideshift_base_url()}/", path.lstrip("/")),
        params={k: v for k, v in (params or {}).items() if v is not None and v != ""},
        headers={"x-api-key": api_key},
        timeout=SIDESHIFT_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    return {"data": payload}


def get_sideshift_overview(program_id: str) -> dict[str, Any]:
    payload = _call_sideshift("analytics/overview", {"programId": program_id})
    return {
        "source": "SideShift",
        "programId": program_id,
        "data": payload.get("data", payload),
    }


def get_sideshift_creators(program_id: str) -> dict[str, Any]:
    payload = _call_sideshift("analytics/accounts", {"programId": program_id})
    data = payload.get("data", payload)
    return {
        "source": "SideShift",
        "programId": program_id,
        "count": len(data) if isinstance(data, list) else None,
        "data": data,
    }


def get_sideshift_posts(program_id: str, creator_id: str | None = None) -> dict[str, Any]:
    params = {"program": program_id}
    if creator_id:
        params["creator"] = creator_id
    payload = _call_sideshift("posts", params)
    data = payload.get("data", payload)
    return {
        "source": "SideShift",
        "programId": program_id,
        "creatorId": creator_id,
        "count": len(data) if isinstance(data, list) else None,
        "data": data,
    }


def get_sideshift_post(post_id: str) -> dict[str, Any]:
    payload = _call_sideshift(f"posts/{post_id}")
    return {"source": "SideShift", "postId": post_id, "data": payload.get("data", payload)}


def _handle_tracker_get_overview(args: dict[str, Any], **_: Any) -> str:
    try:
        return tool_result(get_trackermane_overview(args))
    except Exception as exc:
        return tool_error(str(exc))


def _handle_tracker_search_videos(args: dict[str, Any], **_: Any) -> str:
    try:
        return tool_result(search_trackermane_videos(args))
    except Exception as exc:
        return tool_error(str(exc))


def _handle_tracker_search_accounts(args: dict[str, Any], **_: Any) -> str:
    try:
        return tool_result(search_trackermane_accounts(args))
    except Exception as exc:
        return tool_error(str(exc))


def _handle_tracker_get_video_details(args: dict[str, Any], **_: Any) -> str:
    try:
        return tool_result(get_trackermane_video_details(args))
    except Exception as exc:
        return tool_error(str(exc))


def _handle_sideshift_get_overview(args: dict[str, Any], **_: Any) -> str:
    program_id = (args.get("programId") or "").strip()
    if not program_id:
        return tool_error("programId is required.")
    try:
        return tool_result(get_sideshift_overview(program_id))
    except Exception as exc:
        return tool_error(str(exc))


def _handle_sideshift_get_creators(args: dict[str, Any], **_: Any) -> str:
    program_id = (args.get("programId") or "").strip()
    if not program_id:
        return tool_error("programId is required.")
    try:
        return tool_result(get_sideshift_creators(program_id))
    except Exception as exc:
        return tool_error(str(exc))


def _handle_sideshift_get_posts(args: dict[str, Any], **_: Any) -> str:
    program_id = (args.get("programId") or "").strip()
    if not program_id:
        return tool_error("programId is required.")
    creator_id = (args.get("creatorId") or "").strip() or None
    try:
        return tool_result(get_sideshift_posts(program_id, creator_id))
    except Exception as exc:
        return tool_error(str(exc))


def _handle_sideshift_get_post(args: dict[str, Any], **_: Any) -> str:
    post_id = (args.get("postId") or "").strip()
    if not post_id:
        return tool_error("postId is required.")
    try:
        return tool_result(get_sideshift_post(post_id))
    except Exception as exc:
        return tool_error(str(exc))


TRACKER_GET_OVERVIEW_SCHEMA = {
    "name": "tracker_get_overview",
    "description": "TrackerMane primary overview for broad analytics: total video stats, threshold counts, top videos, and top accounts.",
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {"type": "string", "description": "Optional platform filter like tiktok, instagram, or youtube."},
            "publishedWithinDays": {"type": "integer", "description": "Optional upload recency filter in days for summary and top videos."},
            "topVideoLimit": {"type": "integer", "description": "Number of top videos to return (max 25)."},
            "topAccountLimit": {"type": "integer", "description": "Number of top accounts to return (max 25)."},
        },
    },
}

TRACKER_SEARCH_VIDEOS_SCHEMA = {
    "name": "tracker_search_videos",
    "description": "TrackerMane primary video search and ranking tool. Search by title, description, URL, platform content ID, or creator handle; or list top videos by metric.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional text query for title, description, handle, permalink, or platform content ID."},
            "platform": {"type": "string", "description": "Optional platform filter like tiktok, instagram, or youtube."},
            "publishedWithinDays": {"type": "integer", "description": "Optional upload recency filter in days."},
            "sortBy": {"type": "string", "enum": sorted(TRACKER_VIDEO_SORT_FIELDS)},
            "sortOrder": {"type": "string", "enum": ["asc", "desc"]},
            "limit": {"type": "integer", "description": "Maximum number of videos to return (max 25)."},
        },
    },
}

TRACKER_SEARCH_ACCOUNTS_SCHEMA = {
    "name": "tracker_search_accounts",
    "description": "TrackerMane primary account search and ranking tool for creator stats, follower counts, total views, and top accounts.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional text query for handle, display name, URL, or platform account ID."},
            "platform": {"type": "string", "description": "Optional platform filter like tiktok, instagram, or youtube."},
            "sortBy": {"type": "string", "enum": sorted(TRACKER_ACCOUNT_SORT_FIELDS)},
            "sortOrder": {"type": "string", "enum": ["asc", "desc"]},
            "limit": {"type": "integer", "description": "Maximum number of accounts to return (max 25)."},
        },
    },
}

TRACKER_GET_VIDEO_DETAILS_SCHEMA = {
    "name": "tracker_get_video_details",
    "description": "TrackerMane primary detailed video lookup. Returns one video with metadata, latest stats, recent history, and recent comments when available.",
    "parameters": {
        "type": "object",
        "properties": {
            "contentId": {"type": "string", "description": "TrackerMane content_items.id UUID if known."},
            "platformContentId": {"type": "string", "description": "Platform-specific content ID if known."},
            "url": {"type": "string", "description": "Canonical or platform permalink URL if known."},
            "historyLimit": {"type": "integer", "description": "Number of history points to return (max 50)."},
            "commentLimit": {"type": "integer", "description": "Number of recent comments to return (max 25)."},
        },
    },
}

SIDESHIFT_GET_OVERVIEW_SCHEMA = {
    "name": "sideshift_get_overview",
    "description": "SideShift fallback: get UGC program overview with top posts and top creators.",
    "parameters": {
        "type": "object",
        "properties": {
            "programId": {"type": "string", "description": "The SideShift UGC program ID."},
        },
        "required": ["programId"],
    },
}

SIDESHIFT_GET_CREATORS_SCHEMA = {
    "name": "sideshift_get_creators",
    "description": "SideShift fallback: get all creators in a program with aggregated stats and contractor/program info.",
    "parameters": {
        "type": "object",
        "properties": {
            "programId": {"type": "string", "description": "The SideShift UGC program ID."},
        },
        "required": ["programId"],
    },
}

SIDESHIFT_GET_POSTS_SCHEMA = {
    "name": "sideshift_get_posts",
    "description": "SideShift fallback: list posts for a program, optionally filtered to a specific creator.",
    "parameters": {
        "type": "object",
        "properties": {
            "programId": {"type": "string", "description": "The SideShift UGC program ID."},
            "creatorId": {"type": "string", "description": "Optional SideShift contractor/creator ID."},
        },
        "required": ["programId"],
    },
}

SIDESHIFT_GET_POST_SCHEMA = {
    "name": "sideshift_get_post",
    "description": "SideShift fallback: fetch one SideShift post by its post ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "postId": {"type": "string", "description": "The SideShift post ID."},
        },
        "required": ["postId"],
    },
}


registry.register(
    name="tracker_get_overview",
    toolset="trackermane",
    schema=TRACKER_GET_OVERVIEW_SCHEMA,
    handler=_handle_tracker_get_overview,
    check_fn=_check_trackermane_available,
    emoji="📈",
    max_result_size_chars=100_000,
)
registry.register(
    name="tracker_search_videos",
    toolset="trackermane",
    schema=TRACKER_SEARCH_VIDEOS_SCHEMA,
    handler=_handle_tracker_search_videos,
    check_fn=_check_trackermane_available,
    emoji="🎬",
    max_result_size_chars=100_000,
)
registry.register(
    name="tracker_search_accounts",
    toolset="trackermane",
    schema=TRACKER_SEARCH_ACCOUNTS_SCHEMA,
    handler=_handle_tracker_search_accounts,
    check_fn=_check_trackermane_available,
    emoji="👤",
    max_result_size_chars=100_000,
)
registry.register(
    name="tracker_get_video_details",
    toolset="trackermane",
    schema=TRACKER_GET_VIDEO_DETAILS_SCHEMA,
    handler=_handle_tracker_get_video_details,
    check_fn=_check_trackermane_available,
    emoji="📹",
    max_result_size_chars=100_000,
)
registry.register(
    name="sideshift_get_overview",
    toolset="sideshift",
    schema=SIDESHIFT_GET_OVERVIEW_SCHEMA,
    handler=_handle_sideshift_get_overview,
    check_fn=_check_sideshift_available,
    emoji="↔️",
    max_result_size_chars=100_000,
)
registry.register(
    name="sideshift_get_creators",
    toolset="sideshift",
    schema=SIDESHIFT_GET_CREATORS_SCHEMA,
    handler=_handle_sideshift_get_creators,
    check_fn=_check_sideshift_available,
    emoji="↔️",
    max_result_size_chars=100_000,
)
registry.register(
    name="sideshift_get_posts",
    toolset="sideshift",
    schema=SIDESHIFT_GET_POSTS_SCHEMA,
    handler=_handle_sideshift_get_posts,
    check_fn=_check_sideshift_available,
    emoji="↔️",
    max_result_size_chars=100_000,
)
registry.register(
    name="sideshift_get_post",
    toolset="sideshift",
    schema=SIDESHIFT_GET_POST_SCHEMA,
    handler=_handle_sideshift_get_post,
    check_fn=_check_sideshift_available,
    emoji="↔️",
    max_result_size_chars=100_000,
)
