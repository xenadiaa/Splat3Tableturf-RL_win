CREATE TABLE IF NOT EXISTS daily_summary (
    day TEXT PRIMARY KEY NOT NULL,
    rounds INTEGER NOT NULL DEFAULT 0 CHECK (rounds >= 0),
    successful_visits INTEGER NOT NULL DEFAULT 0 CHECK (successful_visits >= 0),
    failed_visits INTEGER NOT NULL DEFAULT 0 CHECK (failed_visits >= 0),
    task_participations INTEGER NOT NULL DEFAULT 0 CHECK (task_participations >= 0),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS player_daily (
    day TEXT NOT NULL,
    name TEXT NOT NULL,
    name_search TEXT NOT NULL,
    visits INTEGER NOT NULL DEFAULT 0 CHECK (visits >= 0),
    tasks INTEGER NOT NULL DEFAULT 0 CHECK (tasks >= 0),
    returned_before_arrival INTEGER NOT NULL DEFAULT 0 CHECK (returned_before_arrival >= 0),
    room_closed_before_arrival INTEGER NOT NULL DEFAULT 0 CHECK (room_closed_before_arrival >= 0),
    PRIMARY KEY (day, name)
);

CREATE INDEX IF NOT EXISTS idx_player_daily_day
ON player_daily(day);

CREATE INDEX IF NOT EXISTS idx_player_daily_name_search_day
ON player_daily(name_search, day);
