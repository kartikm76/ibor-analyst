#!/bin/bash
set -euo pipefail

# IBOR Bootstrap - Initialize PostgreSQL schema and load CSV data
# Self-contained: all SQL scripts and CSV data are baked into this image.

DB_HOST="${DB_HOST:-postgres}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-ibor}"
DB_USER="${DB_USER:-ibor}"
DB_PASSWORD="${DB_PASSWORD:-ibor}"
DATA_DIR="${DATA_DIR:-/bootstrap/data}"
INIT_DIR="${INIT_DIR:-/bootstrap/init}"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
fail() { echo -e "${RED}✗ $*${NC}"; exit 1; }
info() { echo -e "${YELLOW}==> $*${NC}"; }

info "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
ATTEMPTS=0
MAX_ATTEMPTS=30
while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
  if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; then
    ok "PostgreSQL is ready"
    break
  fi
  ATTEMPTS=$((ATTEMPTS + 1))
  sleep 2
done

if [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; then
  fail "PostgreSQL did not become ready after $((MAX_ATTEMPTS * 2)) seconds"
fi

export PGPASSWORD="$DB_PASSWORD"

info "Checking if data is already loaded..."
ROWCOUNT=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -A -c \
  "SELECT COUNT(*) FROM ibor.fact_position_snapshot" 2>/dev/null || echo "0")

if [ "${ROWCOUNT:-0}" -gt 0 ]; then
  ok "Data already loaded ($ROWCOUNT position rows) — skipping initialization"
  exit 0
fi

info "Applying SQL initialization scripts..."
for script in "$INIT_DIR"/*.sql; do
  if [ -f "$script" ]; then
    SCRIPT_NAME=$(basename "$script")
    info "Applying $SCRIPT_NAME..."
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f "$script" 2>&1 | grep -v "^NOTICE\|^SET\|^CREATE\|^INSERT" || true
    ok "$SCRIPT_NAME"
  fi
done

info "Loading CSV data into staging tables..."
declare -a CSV_FILES=(
  "stg_currency.csv:stg.currency:currency_code,currency_name,minor_unit,source_system,source_ref,ingest_batch_id"
  "stg_exchange.csv:stg.exchange:exchange_code,exchange_name,source_system,source_ref,ingest_batch_id"
  "stg_price_source.csv:stg.price_source:price_source_code,price_source_name,source_system,source_ref,ingest_batch_id"
  "stg_strategy.csv:stg.strategy:strategy_code,strategy_name,strategy_category,source_system,source_ref,ingest_batch_id"
  "stg_portfolio.csv:stg.portfolio:portfolio_code,portfolio_name,region,entitlement_group,valid_from,valid_to,status,source_system,source_ref,ingest_batch_id"
  "stg_portfolio_strategy.csv:stg.portfolio_strategy:portfolio_code,strategy_code,valid_from,valid_to,source_system,source_ref,ingest_batch_id"
  "stg_account.csv:stg.account:account_code,account_name,region,entitlement_group,valid_from,valid_to,status,source_system,source_ref,ingest_batch_id"
  "stg_account_portfolio.csv:stg.account_portfolio:account_code,portfolio_code,valid_from,valid_to,source_system,source_ref,ingest_batch_id"
  "stg_instrument.csv:stg.instrument:instrument_code,instrument_type,instrument_name,exchange_code,currency_code,status,valid_from,valid_to,source_system,source_ref,ingest_batch_id"
  "stg_instrument_equity.csv:stg.instrument_equity:instrument_code,ticker,cusip,isin,sedol,exchange_country,sector,industry_group,shares_outstanding,dividend_yield,dividend_currency,dividend_frequency,fiscal_year_end,listing_date,delisting_date,source_system,source_ref,ingest_batch_id"
  "stg_instrument_bond.csv:stg.instrument_bond:instrument_code,isin,cusip,issuer_name,coupon_rate,coupon_type,coupon_frequency,day_count_convention,maturity_date,issue_date,dated_date,first_coupon_date,last_coupon_date,face_value,currency_code,country_of_issue,call_type,call_price,source_system,source_ref,ingest_batch_id"
  "stg_instrument_futures.csv:stg.instrument_futures:instrument_code,contract_code,underlying_symbol,contract_size,tick_size,tick_value,quote_currency,delivery_month,expiry_date,first_notice_date,last_trading_date,settlement_type,exchange_code,source_system,source_ref,ingest_batch_id"
  "stg_instrument_options.csv:stg.instrument_options:instrument_code,option_symbol,underlying_symbol,option_type,strike_price,expiry_date,multiplier,premium_currency,exercise_style,settlement_type,exchange_code,source_system,source_ref,ingest_batch_id"
  "stg_price.csv:stg.price:instrument_code,price_source_code,price_ts,price_type,price,currency_code,is_eod_flag,source_system,source_ref,ingest_batch_id"
  "stg_fx_rate.csv:stg.fx_rate:from_currency_code,to_currency_code,rate_date,rate,source_system,source_ref,ingest_batch_id"
  "stg_trade_fill.csv:stg.trade_fill:execution_id,trade_code,account_code,instrument_code,trade_date,quantity,price,gross_amount,net_amount,broker_code,counterparty_code,source_system,source_ref,ingest_batch_id"
  "stg_position_snapshot.csv:stg.position_snapshot:portfolio_code,instrument_code,position_date,quantity,source_system,source_ref,ingest_batch_id"
  "stg_cash_event.csv:stg.cash_event:portfolio_code,event_date,amount,currency_code,event_type,notes,source_system,source_ref,ingest_batch_id"
  "stg_position_adjustment.csv:stg.position_adjustment:portfolio_code,instrument_code,effective_date,quantity_delta,reason,source_system,source_ref,ingest_batch_id"
  "stg_broker.csv:stg.broker:broker_code,broker_name,source_system,source_ref,ingest_batch_id"
  "stg_counterparty.csv:stg.counterparty:counterparty_code,counterparty_name,country,source_system,source_ref,ingest_batch_id"
  "stg_calendar.csv:stg.calendar:cal_date,is_business_day,market,source_system,source_ref,ingest_batch_id"
  "stg_corporate_action_applied.csv:stg.corporate_action_applied:instrument_code,action_code,ex_date,record_date,pay_date,factor,amount,currency_code,notes,source_system,source_ref,ingest_batch_id"
)

for entry in "${CSV_FILES[@]}"; do
  IFS=':' read -r csv_file table_name columns <<< "$entry"
  csv_path="$DATA_DIR/$csv_file"

  if [ ! -f "$csv_path" ]; then
    ok "Skipped $csv_file (not found)"
    continue
  fi

  psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    -c "\COPY $table_name ($columns) FROM STDIN WITH (FORMAT CSV, HEADER)" \
    < "$csv_path" 2>&1 | head -1

  ok "Loaded $csv_file → $table_name"
done

info "Promoting staging data to curated schema..."
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
  -c "SELECT ibor.run_all_loaders();" 2>&1 | head -1
ok "Data promotion complete"

info "Verifying loaded data..."
POS=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -A -c \
  "SELECT COUNT(*) FROM ibor.fact_position_snapshot")
INSTR=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -A -c \
  "SELECT COUNT(*) FROM ibor.dim_instrument")
PRICES=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -A -c \
  "SELECT COUNT(*) FROM ibor.fact_price")
TRADES=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -A -c \
  "SELECT COUNT(*) FROM ibor.fact_trade")

ok "dim_instrument:          $INSTR instruments"
ok "fact_position_snapshot:  $POS  position rows"
ok "fact_price:              $PRICES price rows"
ok "fact_trade:              $TRADES trade rows"

info "Bootstrap initialization complete!"
exit 0
