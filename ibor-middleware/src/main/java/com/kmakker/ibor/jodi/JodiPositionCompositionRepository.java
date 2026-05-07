package com.kmakker.ibor.jodi;

import com.kmakker.ibor.dto.PositionSummaryDTO;
import org.jooq.DSLContext;
import org.jooq.Record;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

@Repository
public class JodiPositionCompositionRepository {
    private final DSLContext dslContext;

    public JodiPositionCompositionRepository(DSLContext dslContext) {
        this.dslContext = dslContext;
    }

    public List<PositionSummaryDTO> findComposition(LocalDate asOf, String portfolioCode, Integer page, Integer size) {
        // TODO: implement
        int offset = Math.max(0, (page - 1) * size);
        String sql = """
                With p AS (
                SELECT portfolio_vid, portfolio_code
                  FRON ibor.dim_portfolio
                 WHERE portfolio_code = ?           -- 1
                   AND valid_from <= ?                -- 2
                   AND valid_to   >= ?                -- 3
                 ORDER BY valid_from DESC
                  LIMIT 1
                  ),
                  pos AS (
                    SELECT ps.instrument_vid, SUM(ps.quantity) AS qty
                    FROM ibor.fact_position_snapshot ps
                    JOIN p ON ps.portfolio_vid = p.portfolio_vid
                    WHERE ps.position_date = ?         -- 4
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
                    SELECT di.instrument_vid, di.instrument_code, di.instrument_type, di.currency_code
                    FROM ibor.dim_instrument di
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
                        (SELECT portfolio_code FROM p)      AS portfolio_code,
                        i.instrument_code                   AS instrument_code,
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
                    JOIN i            ON i.instrument_vid = cur.instrument_vid
                    LEFT JOIN price_pick pp ON pp.instrument_vid = cur.instrument_vid
                    LEFT JOIN mult       m  ON m.instrument_vid  = cur.instrument_vid
                    WHERE cur.net_qty IS NOT NULL
                    ORDER BY i.instrument_code
                    LIMIT ? OFFSET ?;
                """;

        // Bind each '?' in the same order:
        return dslContext
                .resultQuery(
                        sql,
                        portfolioCode,   // 1
                        asOf,            // 2
                        asOf,            // 3
                        asOf,            // 4
                        asOf,            // 5
                        asOf,            // 6
                        asOf,            // 7
                        asOf,            // 8
                        asOf,            // 9
                        size,            // 10
                        offset           // 11
                ).fetch(this::toDto);
    }

    private PositionSummaryDTO toDto(Record record) {
        return new PositionSummaryDTO(
                record.get("as_of", LocalDate.class),
                record.get("portfolio_id", String.class),
                record.get("instrument_id", String.class),
                record.get("instrument_type", String.class),
                record.get("net_qty", BigDecimal.class),
                record.get("price", BigDecimal.class),
                record.get("price_source", String.class),
                record.get("mkt_value", BigDecimal.class),
                record.get("currency", String.class)
        );
    }
}

