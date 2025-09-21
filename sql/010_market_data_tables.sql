-- 1-minute bars
CREATE TABLE IF NOT EXISTS market.bars_1m (
  ts            timestamptz   NOT NULL,
  symbol        text          NOT NULL,
  open          numeric(18,8) NOT NULL,
  high          numeric(18,8) NOT NULL,
  low           numeric(18,8) NOT NULL,
  close         numeric(18,8) NOT NULL,
  volume        bigint        NOT NULL,
  vwap          numeric(18,8),
  trades_count  integer,
  PRIMARY KEY (symbol, ts)
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM timescaledb_information.hypertables
    WHERE hypertable_schema = 'market' AND hypertable_name = 'bars_1m'
  ) THEN
    PERFORM create_hypertable(
      'market.bars_1m','ts',
      chunk_time_interval => interval '1 day',
      if_not_exists => true
    );
  END IF;
END
$$;
