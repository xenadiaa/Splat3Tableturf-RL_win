ALTER TABLE player_daily
ADD COLUMN network_error_before_arrival INTEGER NOT NULL DEFAULT 0
CHECK (network_error_before_arrival >= 0);
