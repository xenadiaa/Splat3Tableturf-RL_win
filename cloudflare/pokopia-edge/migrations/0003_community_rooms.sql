CREATE TABLE IF NOT EXISTS community_rooms (
    id TEXT PRIMARY KEY NOT NULL,
    owner_hash TEXT NOT NULL,
    publisher_hash TEXT NOT NULL,
    player_name TEXT NOT NULL,
    player_search TEXT NOT NULL,
    code TEXT NOT NULL,
    room_type TEXT NOT NULL CHECK (room_type IN ('stamp', 'task', 'other')),
    description TEXT NOT NULL DEFAULT '',
    business_day TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    full_reported_at INTEGER,
    invalid_reported_at INTEGER,
    invalid_votes INTEGER NOT NULL DEFAULT 0,
    deleted_at INTEGER,
    delete_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_community_rooms_visible
ON community_rooms(created_at DESC, deleted_at, invalid_reported_at);

CREATE INDEX IF NOT EXISTS idx_community_rooms_leaderboard
ON community_rooms(business_day, player_name, deleted_at);

CREATE TABLE IF NOT EXISTS community_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id TEXT NOT NULL,
    reporter_hash TEXT NOT NULL,
    action TEXT NOT NULL CHECK (
        action IN ('full', 'full_wrong', 'invalid', 'invalid_wrong')
    ),
    created_at INTEGER NOT NULL,
    room_age_seconds INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(room_id) REFERENCES community_rooms(id)
);

CREATE INDEX IF NOT EXISTS idx_community_feedback_room_time
ON community_feedback(room_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_community_feedback_reporter_time
ON community_feedback(reporter_hash, created_at DESC);

CREATE TABLE IF NOT EXISTS community_usage_daily (
    day TEXT NOT NULL,
    actor_hash TEXT NOT NULL,
    uploads INTEGER NOT NULL DEFAULT 0,
    feedbacks INTEGER NOT NULL DEFAULT 0,
    early_invalid_feedbacks INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(day, actor_hash)
);

CREATE TABLE IF NOT EXISTS community_settings (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

INSERT OR IGNORE INTO community_settings(key, value, updated_at)
VALUES('writes_enabled', 'true', unixepoch());
