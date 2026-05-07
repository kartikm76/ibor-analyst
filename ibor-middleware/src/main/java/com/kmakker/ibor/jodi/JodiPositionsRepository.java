package com.kmakker.ibor.jodi;

import com.kmakker.ibor.dto.PositionDTO;
import org.jooq.DSLContext;
import org.jooq.Record;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

@Repository
public class JodiPositionsRepository {
    private final DSLContext dslContext;

    public JodiPositionsRepository(DSLContext dslContext) {
        this.dslContext = dslContext;
    }

    public List<PositionDTO> findPositions(LocalDate asOf, String portfolioCode, String accountCode, Integer page, Integer size) {
        int p = (page == null || page < 1) ? 1 : page;
        int s = (size == null || size <= 0) ? 50 : size;
        int offset = Math.max(0, (p - 1) * s);

        boolean filterByAccount = accountCode != null && !accountCode.isBlank();

        // Account filter CTE: only included when accountCode is provided.
        // The join path is: dim_account -> dim_account_portfolio -> dim_portfolio.
        String accountCte = filterByAccount ? """
            acct AS (
              SELECT da.account_vid
              FROM ibor.dim_account da
              WHERE da.account_code = ?          -- account param
                AND da.valid_from <= ?
                AND da.valid_to   >= ?
              LIMIT 1
            ),
            acct_ptf AS (
              SELECT dap.portfolio_vid
              FROM ibor.dim_account_portfolio dap
              JOIN acct ON dap.account_vid = acct.account_vid
              WHERE dap.valid_from <= ?
                AND dap.valid_to   >= ?
            ),
            """ : "";

        String accountJoin = filterByAccount
                ? "JOIN acct_ptf ON acct_ptf.portfolio_vid = dp.portfolio_vid"
                : "";

        final String sql = """
        WITH
        """ + accountCte + """
            p AS (
              SELECT dp.portfolio_vid, dp.portfolio_code
              FROM ibor.dim_portfolio dp
              """ + accountJoin + """
              WHERE dp.portfolio_code = ?        -- portfolioCode
                AND dp.valid_from <= ?
                AND dp.valid_to   >= ?
              ORDER BY dp.valid_from DESC
              LIMIT 1
            ),
            latest_snap AS (
              SELECT MAX(ps.position_date) AS snap_date
              FROM ibor.fact_position_snapshot ps
              JOIN p ON ps.portfolio_vid = p.portfolio_vid
              WHERE ps.position_date <= ?        -- 4
            ),
            pos AS (
              SELECT ps.instrument_vid, SUM(ps.quantity) AS qty
              FROM ibor.fact_position_snapshot ps
              JOIN p ON ps.portfolio_vid = p.portfolio_vid
              JOIN latest_snap ON ps.position_date = latest_snap.snap_date
              GROUP BY ps.instrument_vid
            ),
            adj AS (
              SELECT a.instrument_vid, SUM(a.quantity_delta) AS qty_adj
              FROM ibor.fact_position_adjustment a
              JOIN p ON a.portfolio_vid = p.portfolio_vid
              WHERE a.effective_date <= ?        -- 5
              GROUP BY a.instrument_vid
            ),
            i AS (
              SELECT di.instrument_vid, di.instrument_code, di.instrument_name, di.instrument_type, di.currency_code,
                     eq.ticker
              FROM ibor.dim_instrument di
              LEFT JOIN ibor.dim_instrument_equity eq ON eq.instrument_vid = di.instrument_vid
              WHERE di.valid_from <= ?           -- 6
                AND di.valid_to   >= ?           -- 7
            ),
            cur AS (
              SELECT COALESCE(pos.instrument_vid, adj.instrument_vid) AS instrument_vid,
                     COALESCE(pos.qty, 0)::numeric + COALESCE(adj.qty_adj, 0)::numeric AS net_qty
              FROM pos
              FULL OUTER JOIN adj ON pos.instrument_vid = adj.instrument_vid
            ),
            price_pick AS (
              SELECT DISTINCT ON (fp.instrument_vid)
                     fp.instrument_vid,
                     fp.price,
                     fp.currency_code AS price_currency,
                     dps.price_source_code,
                     fp.price_ts
              FROM ibor.fact_price fp
              JOIN ibor.dim_price_source dps ON dps.price_source_vid = fp.price_source_vid
              WHERE fp.price_ts <= (?::date + time '23:59:59')  -- 8
              ORDER BY fp.instrument_vid,
                       fp.price_ts DESC,
                       (dps.price_source_code = 'BBG') DESC
            ),
            mult AS (
              SELECT i.instrument_vid,
                     COALESCE(fut.contract_size, opt.multiplier, 1)::numeric AS contract_multiplier
              FROM i
              LEFT JOIN ibor.dim_instrument_futures fut USING (instrument_vid)
              LEFT JOIN ibor.dim_instrument_options  opt USING (instrument_vid)
            )
            SELECT
              ?                                   AS as_of,            -- 9
              latest_snap.snap_date               AS snap_date,
              (SELECT portfolio_code FROM p)      AS portfolio_id,
              i.instrument_code                   AS instrument_id,
              i.instrument_name                   AS instrument_name,
              COALESCE(i.ticker, i.instrument_code) AS ticker,
              i.instrument_type                   AS instrument_type,
              cur.net_qty                         AS net_qty,
              pp.price                            AS price,
              pp.price_source_code                AS price_source,
              (cur.net_qty * COALESCE(pp.price, 0) * m.contract_multiplier) AS mkt_value,
              NULL::numeric                       AS cost,
              NULL::numeric                       AS unrealized_pnl,
              COALESCE(pp.price_currency, i.currency_code) AS currency,
              m.contract_multiplier               AS contract_multiplier
            FROM cur
            CROSS JOIN latest_snap
            JOIN i            ON i.instrument_vid = cur.instrument_vid
            LEFT JOIN price_pick pp ON pp.instrument_vid = cur.instrument_vid
            LEFT JOIN mult       m  ON m.instrument_vid  = cur.instrument_vid
            WHERE cur.net_qty IS NOT NULL
            ORDER BY i.instrument_code
            LIMIT ? OFFSET ?;
        """;

        // Build param list dynamically
        List<Object> params = new ArrayList<>();
        if (filterByAccount) {
            params.add(accountCode); // acct.account_code
            params.add(asOf);        // acct.valid_from <=
            params.add(asOf);        // acct.valid_to >=
            params.add(asOf);        // acct_ptf.valid_from <=
            params.add(asOf);        // acct_ptf.valid_to >=
        }
        params.add(portfolioCode);   // p.portfolio_code
        params.add(asOf);            // p.valid_from <=
        params.add(asOf);            // p.valid_to >=
        params.add(asOf);            // latest_snap.position_date <=
        params.add(asOf);            // adj.effective_date <=
        params.add(asOf);            // i.valid_from <=
        params.add(asOf);            // i.valid_to >=
        params.add(asOf);            // price_pick.price_ts <=
        params.add(asOf);            // as_of literal
        params.add(s);               // LIMIT
        params.add(offset);          // OFFSET

        return dslContext
                .resultQuery(sql, params.toArray())
                .fetch(this::toDto);
    }

    private PositionDTO toDto(Record record) {
        return new PositionDTO(
                record.get("as_of", LocalDate.class),
                record.get("snap_date", LocalDate.class),
                record.get("portfolio_id", String.class),
                record.get("instrument_id", String.class),
                record.get("instrument_name", String.class),
                record.get("ticker", String.class),
                record.get("instrument_type", String.class),
                record.get("net_qty", BigDecimal.class),
                record.get("price", BigDecimal.class),
                record.get("price_source", String.class),
                record.get("mkt_value", BigDecimal.class),
                record.get("cost", BigDecimal.class),
                record.get("unrealized_pnl", BigDecimal.class),
                record.get("currency", String.class),
                record.get("contract_multiplier", BigDecimal.class)
        );
    }
}
