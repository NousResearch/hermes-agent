-- Test-only historical Files schema. No old production code is imported/executed.
-- Source f086299b3406815115a3765fd7257998727cffee gateway/hosted_rooms.py:_SCHEMA_DDL and
-- gateway/hosted_room_attachments.py:HostedRoomAttachmentStore._initialize.

CREATE TABLE IF NOT EXISTS hosted_rooms (
            room_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            members_json TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL DEFAULT 1 CHECK (authority_epoch >= 1),
            next_seq INTEGER NOT NULL DEFAULT 1 CHECK (next_seq >= 1),
            event_bytes INTEGER NOT NULL DEFAULT 0 CHECK (event_bytes >= 0),
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            disbanded_at REAL
        );

CREATE TABLE IF NOT EXISTS hosted_room_events (
            room_id TEXT NOT NULL,
            seq INTEGER NOT NULL CHECK (seq >= 1),
            event_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            actor_json TEXT NOT NULL,
            authority_epoch INTEGER CHECK (authority_epoch IS NULL OR authority_epoch >= 1),
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (room_id, seq),
            UNIQUE (room_id, event_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id)
        );

CREATE TABLE IF NOT EXISTS hosted_room_retired_ids (
            room_id TEXT PRIMARY KEY,
            retired_at REAL NOT NULL
        );

CREATE TABLE IF NOT EXISTS hosted_room_links (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_url TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            grant TEXT NOT NULL,
            catalog_json TEXT NOT NULL,
            cancellation_scope_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            transport_security TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            updated_at REAL NOT NULL,
            PRIMARY KEY (room_id, member_id)
        );

CREATE TABLE IF NOT EXISTS hosted_room_remote_runs (
            room_id TEXT NOT NULL,
            home_install_id TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            member_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            execution_generation INTEGER NOT NULL CHECK (execution_generation >= 1),
            target_install_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (
                room_id, home_install_id, authority_gateway_id, authority_epoch,
                member_id, target_install_id, target_profile, task_id,
                execution_generation
            )
        );

CREATE TABLE IF NOT EXISTS hosted_room_revoked_grants (
            scope_key TEXT PRIMARY KEY,
            expires_at REAL NOT NULL,
            revoked_before REAL NOT NULL
        );

CREATE TABLE IF NOT EXISTS hosted_room_peer_reservations (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            expires_at REAL NOT NULL,
            revoked_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (room_id, member_id, target_profile)
        );

CREATE INDEX IF NOT EXISTS idx_hosted_room_events_cursor ON hosted_room_events(room_id, seq);

CREATE TABLE IF NOT EXISTS hosted_room_attachment_blobs (
                blob_id TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL UNIQUE,
                size INTEGER NOT NULL CHECK (size > 0),
                ref_count INTEGER NOT NULL CHECK (ref_count > 0),
                created_at REAL NOT NULL
            );

CREATE TABLE IF NOT EXISTS hosted_room_attachments (
                attachment_id TEXT PRIMARY KEY,
                upload_id TEXT NOT NULL,
                room_id TEXT NOT NULL,
                event_id TEXT,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                size INTEGER NOT NULL CHECK (size > 0),
                mime TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                blob_id TEXT NOT NULL,
                recipient_member_ids_json TEXT NOT NULL DEFAULT '[]',
                viewer_access INTEGER NOT NULL DEFAULT 0 CHECK (viewer_access IN (0, 1)),
                state TEXT NOT NULL CHECK (state IN ('uploaded', 'committed', 'disbanded')),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL,
                UNIQUE (room_id, upload_id),
                FOREIGN KEY (blob_id) REFERENCES hosted_room_attachment_blobs(blob_id)
            );

CREATE INDEX IF NOT EXISTS idx_hosted_room_attachments_room_state
               ON hosted_room_attachments(room_id, state, created_at);

CREATE INDEX IF NOT EXISTS idx_hosted_room_attachments_expiry
               ON hosted_room_attachments(expires_at);
