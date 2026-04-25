-- =====================================================================
-- DIMENSION SCHEMA (finalized) — ibor.*
-- Reference masters are non-SCD; core dims & bridges are SCD2.
-- =====================================================================

DROP SCHEMA IF EXISTS ibor CASCADE;
CREATE SCHEMA ibor;

-- ---------------------------
-- Reference (non-SCD) dims
-- ---------------------------
CREATE TABLE ibor.dim_currency (
  currency_vid   BIGSERIAL PRIMARY KEY,
  currency_code  CHAR(3) NOT NULL UNIQUE,
  currency_name  TEXT,
  minor_unit     INT,
  created_at     TIMESTAMP NOT NULL DEFAULT now(),
  updated_at     TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE ibor.dim_exchange (
  exchange_vid   BIGSERIAL PRIMARY KEY,
  exchange_code  TEXT NOT NULL UNIQUE,
  exchange_name  TEXT,
  created_at     TIMESTAMP NOT NULL DEFAULT now(),
  updated_at     TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE ibor.dim_price_source (
  price_source_vid  BIGSERIAL PRIMARY KEY,
  price_source_code TEXT NOT NULL UNIQUE,
  price_source_name TEXT,
  created_at        TIMESTAMP NOT NULL DEFAULT now(),
  updated_at        TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE ibor.dim_strategy (
  strategy_vid      BIGSERIAL PRIMARY KEY,
  strategy_code     TEXT NOT NULL UNIQUE,
  strategy_name     TEXT,
  strategy_category TEXT,
  status            TEXT DEFAULT 'ACTIVE',
  created_at        TIMESTAMP NOT NULL DEFAULT now(),
  updated_at        TIMESTAMP NOT NULL DEFAULT now()
);

-- ---------------------------
-- Core SCD2 dims
-- ---------------------------
CREATE TABLE ibor.dim_instrument (
  instrument_vid   BIGSERIAL PRIMARY KEY,
  instrument_code  TEXT NOT NULL,
  instrument_type  TEXT NOT NULL CHECK (instrument_type IN ('EQUITY','BOND','FUT','OPT','FX','INDEX','OTHER')),
  instrument_name  TEXT,
  exchange_code    TEXT REFERENCES ibor.dim_exchange(exchange_code),
  currency_code    CHAR(3) REFERENCES ibor.dim_currency(currency_code),
  status           TEXT DEFAULT 'ACTIVE',
  valid_from       DATE NOT NULL,
  valid_to         DATE NOT NULL,
  is_current       BOOLEAN NOT NULL DEFAULT TRUE,
  created_at       TIMESTAMP NOT NULL DEFAULT now(),
  updated_at       TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE (instrument_code, valid_from)
);

CREATE INDEX IF NOT EXISTS idx_instr_code ON ibor.dim_instrument (instrument_code);
CREATE INDEX IF NOT EXISTS idx_instr_pit  ON ibor.dim_instrument (valid_from, valid_to);

-- Equity subtype (rich)
CREATE TABLE ibor.dim_instrument_equity (
  instrument_vid      BIGINT PRIMARY KEY REFERENCES ibor.dim_instrument(instrument_vid),
  ticker              TEXT,
  cusip               TEXT,
  isin                TEXT,
  sedol               TEXT,
  exchange_country    TEXT,
  sector              TEXT,
  industry_group      TEXT,
  shares_outstanding  NUMERIC(20,2),
  dividend_yield      NUMERIC(10,6),
  dividend_currency   CHAR(3),
  dividend_frequency  TEXT,
  fiscal_year_end     DATE,
  listing_date        DATE,
  delisting_date      DATE,
  created_at          TIMESTAMP NOT NULL DEFAULT now(),
  updated_at          TIMESTAMP NOT NULL DEFAULT now()
);

-- Bond subtype (rich)
CREATE TABLE ibor.dim_instrument_bond (
  instrument_vid        BIGINT PRIMARY KEY REFERENCES ibor.dim_instrument(instrument_vid),
  isin                  TEXT,
  cusip                 TEXT,
  issuer_name           TEXT,
  coupon_rate           NUMERIC(10,6),
  coupon_type           TEXT,
  coupon_frequency      TEXT,
  day_count_convention  TEXT,
  maturity_date         DATE,
  issue_date            DATE,
  dated_date            DATE,
  first_coupon_date     DATE,
  last_coupon_date      DATE,
  face_value            NUMERIC(20,2),
  currency_code         CHAR(3),
  country_of_issue      TEXT,
  call_type             TEXT,
  call_price            NUMERIC(20,6),
  created_at            TIMESTAMP NOT NULL DEFAULT now(),
  updated_at            TIMESTAMP NOT NULL DEFAULT now()
);

-- Futures subtype (rich)
CREATE TABLE ibor.dim_instrument_futures (
  instrument_vid     BIGINT PRIMARY KEY REFERENCES ibor.dim_instrument(instrument_vid),
  contract_code      TEXT,
  underlying_symbol  TEXT,
  contract_size      NUMERIC(20,6),
  tick_size          NUMERIC(20,6),
  tick_value         NUMERIC(20,6),
  quote_currency     CHAR(3),
  delivery_month     TEXT,
  expiry_date        DATE,
  first_notice_date  DATE,
  last_trading_date  DATE,
  settlement_type    TEXT,  -- CASH / PHYSICAL
  exchange_code      TEXT,
  created_at         TIMESTAMP NOT NULL DEFAULT now(),
  updated_at         TIMESTAMP NOT NULL DEFAULT now()
);

-- Options subtype (rich)
CREATE TABLE ibor.dim_instrument_options (
  instrument_vid     BIGINT PRIMARY KEY REFERENCES ibor.dim_instrument(instrument_vid),
  option_symbol      TEXT,
  underlying_symbol  TEXT,
  option_type        TEXT CHECK (option_type IN ('CALL','PUT')),
  strike_price       NUMERIC(20,6),
  expiry_date        DATE,
  multiplier         NUMERIC(20,6),
  premium_currency   CHAR(3),
  exercise_style     TEXT CHECK (exercise_style IN ('AMERICAN','EUROPEAN')),
  settlement_type    TEXT CHECK (settlement_type IN ('CASH','PHYSICAL')),
  exchange_code      TEXT,
  created_at         TIMESTAMP NOT NULL DEFAULT now(),
  updated_at         TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE ibor.dim_portfolio (
  portfolio_vid     BIGSERIAL PRIMARY KEY,
  portfolio_code    TEXT NOT NULL,
  portfolio_name    TEXT,
  region            TEXT,
  entitlement_group TEXT,
  status            TEXT DEFAULT 'ACTIVE',
  valid_from        DATE NOT NULL,
  valid_to          DATE NOT NULL,
  is_current        BOOLEAN NOT NULL DEFAULT TRUE,
  created_at        TIMESTAMP NOT NULL DEFAULT now(),
  updated_at        TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE (portfolio_code, valid_from)
);
CREATE INDEX idx_portfolio_current ON ibor.dim_portfolio (portfolio_code) WHERE is_current;

CREATE TABLE ibor.dim_account (
  account_vid       BIGSERIAL PRIMARY KEY,
  account_code      TEXT NOT NULL,
  account_name      TEXT,
  region            TEXT,
  entitlement_group TEXT,
  status            TEXT DEFAULT 'ACTIVE',
  valid_from        DATE NOT NULL,
  valid_to          DATE NOT NULL,
  is_current        BOOLEAN NOT NULL DEFAULT TRUE,
  created_at        TIMESTAMP NOT NULL DEFAULT now(),
  updated_at        TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE (account_code, valid_from)
);
CREATE INDEX idx_account_current ON ibor.dim_account (account_code) WHERE is_current;

-- ---------------------------
-- SCD2 bridges (dims)
-- ---------------------------
CREATE TABLE ibor.dim_portfolio_strategy (
  portfolio_strategy_vid BIGSERIAL PRIMARY KEY,
  portfolio_vid          BIGINT NOT NULL REFERENCES ibor.dim_portfolio(portfolio_vid),
  strategy_vid           BIGINT NOT NULL REFERENCES ibor.dim_strategy(strategy_vid),
  valid_from             DATE NOT NULL,
  valid_to               DATE NOT NULL,
  is_current             BOOLEAN NOT NULL DEFAULT TRUE,
  created_at             TIMESTAMP NOT NULL DEFAULT now(),
  updated_at             TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE (portfolio_vid, valid_from)
);
CREATE INDEX idx_ps_current ON ibor.dim_portfolio_strategy (portfolio_vid) WHERE is_current;

CREATE TABLE ibor.dim_account_portfolio (
  account_portfolio_vid BIGSERIAL PRIMARY KEY,
  account_vid           BIGINT NOT NULL REFERENCES ibor.dim_account(account_vid),
  portfolio_vid         BIGINT NOT NULL REFERENCES ibor.dim_portfolio(portfolio_vid),
  valid_from            DATE NOT NULL,
  valid_to              DATE NOT NULL,
  is_current            BOOLEAN NOT NULL DEFAULT TRUE,
  created_at            TIMESTAMP NOT NULL DEFAULT now(),
  updated_at            TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE (account_vid, valid_from)
);
CREATE INDEX idx_ap_current ON ibor.dim_account_portfolio (account_vid) WHERE is_current;

-- =========================================================
-- RAG TABLES (requires pgvector image)
-- =========================================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents you ingest (files, web pages, etc.)
CREATE TABLE IF NOT EXISTS ibor.rag_documents (
  document_id   BIGSERIAL PRIMARY KEY,
  source        TEXT,               -- e.g., 'file','web','manual'
  external_id   TEXT,               -- path/URL/hash; stable id from source
  title         TEXT,
  metadata      JSONB,
  created_at    TIMESTAMP NOT NULL DEFAULT now(),
  updated_at    TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE (source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_doc_meta ON ibor.rag_documents USING GIN (metadata);

-- Embeddable chunks for retrieval
CREATE TABLE IF NOT EXISTS ibor.rag_chunks (
  chunk_id     BIGSERIAL PRIMARY KEY,
  document_id  BIGINT NOT NULL REFERENCES ibor.rag_documents(document_id) ON DELETE CASCADE,
  chunk_index  INT NOT NULL,
  content      TEXT NOT NULL,
  embedding    vector(1536),        -- adjust dim if you use a different model
  metadata     JSONB,
  created_at   TIMESTAMP NOT NULL DEFAULT now(),
  updated_at   TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE (document_id, chunk_index)
);
-- ANN index (cosine). You can tune lists based on data size.
CREATE INDEX IF NOT EXISTS idx_rag_chunk_vec
  ON ibor.rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_rag_chunk_doc ON ibor.rag_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunk_meta ON ibor.rag_chunks USING GIN (metadata);

-- =========================================================
-- FACT TABLES
-- =========================================================

-- Prices (intraday/EOD)
CREATE TABLE IF NOT EXISTS ibor.fact_price (
  instrument_vid    BIGINT NOT NULL REFERENCES ibor.dim_instrument(instrument_vid),
  price_source_vid  BIGINT NOT NULL REFERENCES ibor.dim_price_source(price_source_vid),
  price_ts          TIMESTAMPTZ NOT NULL,
  price_type        TEXT NOT NULL CHECK (price_type IN ('MID','BID','ASK','CLOSE','THEO')),
  price             NUMERIC(28,10) NOT NULL,
  currency_code     CHAR(3) NOT NULL REFERENCES ibor.dim_currency(currency_code),
  volume            NUMERIC(28,4),
  trade_count       BIGINT,
  is_eod_flag       BOOLEAN NOT NULL DEFAULT FALSE,
  created_at        TIMESTAMP NOT NULL DEFAULT now(),
  updated_at        TIMESTAMP NOT NULL DEFAULT now(),
  PRIMARY KEY (instrument_vid, price_source_vid, price_ts)
);
CREATE INDEX IF NOT EXISTS idx_price_instr_eod_ts ON ibor.fact_price (instrument_vid, is_eod_flag, price_ts DESC);

-- FX rates vs base (e.g., USD)
CREATE TABLE ibor.fact_fx_rate (
   from_currency_code CHAR(3) NOT NULL REFERENCES ibor.dim_currency(currency_code),
   to_currency_code   CHAR(3) NOT NULL REFERENCES ibor.dim_currency(currency_code),
   rate_date          DATE NOT NULL,
   rate               NUMERIC(28,10) NOT NULL,
   source_system      TEXT,
   source_ref         TEXT,
   ingest_batch_id    TEXT,
   created_at         TIMESTAMP NOT NULL DEFAULT now(),
   updated_at         TIMESTAMP NOT NULL DEFAULT now(),
   PRIMARY KEY (from_currency_code, to_currency_code, rate_date)
);
CREATE INDEX IF NOT EXISTS idx_fx_rate_pair_date ON ibor.fact_fx_rate (from_currency_code, to_currency_code, rate_date DESC);

-- Executed fills (immutable grain = execution_id)
CREATE TABLE IF NOT EXISTS ibor.fact_trade (
  execution_id       TEXT PRIMARY KEY,
  trade_code         TEXT,
  account_vid        BIGINT NOT NULL REFERENCES ibor.dim_account(account_vid),
  instrument_vid     BIGINT NOT NULL REFERENCES ibor.dim_instrument(instrument_vid),
  trade_date         DATE NOT NULL,
  quantity           NUMERIC(28,6) NOT NULL,
  price              NUMERIC(28,10) NOT NULL,
  gross_amount       NUMERIC(28,10),
  net_amount         NUMERIC(28,10),
  broker_code        TEXT,           -- keep as code (no FK) unless you decide to add dim_broker
  counterparty_code  TEXT,           -- keep as code (no FK) unless you add dim_counterparty
  created_at         TIMESTAMP NOT NULL DEFAULT now(),
  updated_at         TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_trade_instr_date      ON ibor.fact_trade (instrument_vid, trade_date);
CREATE INDEX IF NOT EXISTS idx_trade_acct_date       ON ibor.fact_trade (account_vid, trade_date);
-- Composite: covers drilldown by account + instrument + date in one scan
CREATE INDEX IF NOT EXISTS idx_trade_acct_instr_date ON ibor.fact_trade (account_vid, instrument_vid, trade_date DESC);

-- EOD position snapshots (derived)
CREATE TABLE IF NOT EXISTS ibor.fact_position_snapshot (
  portfolio_vid   BIGINT NOT NULL REFERENCES ibor.dim_portfolio(portfolio_vid),
  instrument_vid  BIGINT NOT NULL REFERENCES ibor.dim_instrument(instrument_vid),
  position_date   DATE NOT NULL,
  quantity        NUMERIC(28,6) NOT NULL,
  created_at      TIMESTAMP NOT NULL DEFAULT now(),
  updated_at      TIMESTAMP NOT NULL DEFAULT now(),
  PRIMARY KEY (portfolio_vid, instrument_vid, position_date)
);
CREATE INDEX IF NOT EXISTS idx_possnap_instr         ON ibor.fact_position_snapshot (instrument_vid, position_date);
-- Composite: covers "all positions for a portfolio as of date" — the most common query
CREATE INDEX IF NOT EXISTS idx_possnap_portfolio_date ON ibor.fact_position_snapshot (portfolio_vid, position_date DESC);

-- Cash events (coupons/dividends/fees/etc.)
CREATE TABLE IF NOT EXISTS ibor.fact_cash_event (
  portfolio_vid   BIGINT NOT NULL REFERENCES ibor.dim_portfolio(portfolio_vid),
  event_date      DATE NOT NULL,
  amount          NUMERIC(28,10) NOT NULL,
  currency_code   CHAR(3) REFERENCES ibor.dim_currency(currency_code),
  event_type      TEXT,                 -- e.g., 'COUPON','DIVIDEND','FEE'
  notes           TEXT,
  created_at      TIMESTAMP NOT NULL DEFAULT now(),
  updated_at      TIMESTAMP NOT NULL DEFAULT now(),
  PRIMARY KEY (portfolio_vid, event_date, amount)
);
CREATE INDEX IF NOT EXISTS idx_cash_event_date ON ibor.fact_cash_event (event_date);

-- Position adjustments (manual/systemic deltas)
CREATE TABLE IF NOT EXISTS ibor.fact_position_adjustment (
  position_adjustment_id BIGSERIAL PRIMARY KEY,
  portfolio_vid   BIGINT NOT NULL REFERENCES ibor.dim_portfolio(portfolio_vid),
  instrument_vid  BIGINT NOT NULL REFERENCES ibor.dim_instrument(instrument_vid),
  effective_date  DATE   NOT NULL,
  quantity_delta  NUMERIC(28,6) NOT NULL,
  reason          TEXT   NOT NULL,
  created_at      TIMESTAMP NOT NULL DEFAULT now(),
  updated_at      TIMESTAMP NOT NULL DEFAULT now(),
  CONSTRAINT uq_pos_adj_natural UNIQUE (portfolio_vid, instrument_vid, effective_date, reason)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pos_adj_natural
  ON ibor.fact_position_adjustment (portfolio_vid, instrument_vid, effective_date, quantity_delta, COALESCE(reason,''));
CREATE INDEX IF NOT EXISTS idx_pos_adj_effective ON ibor.fact_position_adjustment (effective_date);

-- Corporate actions applied (optional if you ingest them)
CREATE TABLE IF NOT EXISTS ibor.fact_corporate_action_applied (
  ca_applied_id   BIGSERIAL PRIMARY KEY,
  instrument_vid  BIGINT NOT NULL REFERENCES ibor.dim_instrument(instrument_vid),
  action_code     TEXT NOT NULL,        -- e.g., 'SPLIT','DIV','SPINOFF'
  ex_date         DATE NOT NULL,
  record_date     DATE,
  pay_date        DATE,
  factor          NUMERIC(28,10),       -- e.g., split ratio or adjustment factor
  amount          NUMERIC(28,10),
  currency_code   CHAR(3) REFERENCES ibor.dim_currency(currency_code),
  notes           TEXT,
  created_at      TIMESTAMP NOT NULL DEFAULT now(),
  updated_at      TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ca_applied_exdate ON ibor.fact_corporate_action_applied (ex_date);
