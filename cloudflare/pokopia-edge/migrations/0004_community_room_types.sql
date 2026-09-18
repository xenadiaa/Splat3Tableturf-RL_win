PRAGMA foreign_keys = OFF;

CREATE TABLE community_rooms_v4 (
    id TEXT PRIMARY KEY NOT NULL,
    owner_hash TEXT NOT NULL,
    publisher_hash TEXT NOT NULL,
    player_name TEXT NOT NULL,
    player_search TEXT NOT NULL,
    code TEXT NOT NULL,
    room_type TEXT NOT NULL CHECK (room_type IN ('stamp', 'task', 'flower', 'material_solo', 'other')),
    description TEXT NOT NULL DEFAULT '',
    business_day TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    full_reported_at INTEGER,
    invalid_reported_at INTEGER,
    invalid_votes INTEGER NOT NULL DEFAULT 0,
    deleted_at INTEGER,
    delete_reason TEXT
);

INSERT INTO community_rooms_v4
SELECT * FROM community_rooms;

DROP TABLE community_rooms;
ALTER TABLE community_rooms_v4 RENAME TO community_rooms;

CREATE INDEX idx_community_rooms_visible
ON community_rooms(created_at DESC, deleted_at, invalid_reported_at);

CREATE INDEX idx_community_rooms_leaderboard
ON community_rooms(business_day, player_name, deleted_at);

PRAGMA foreign_keys = ON;
