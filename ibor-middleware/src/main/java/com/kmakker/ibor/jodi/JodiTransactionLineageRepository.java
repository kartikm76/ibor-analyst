package com.kmakker.ibor.jodi;

import com.kmakker.ibor.dto.PositionDetailDTO;
import com.kmakker.ibor.dto.TransactionDTO;
import lombok.extern.slf4j.Slf4j;
import org.jooq.DSLContext;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.sql.Date;
import java.time.LocalDate;
import java.util.List;

/**
 * Repository for transaction lineage and position detail lookups using jOOQ with plain SQL.
 *
 * <p>Key implementation notes:</p>
 * <ul>
 *   <li>All queries use a single <code>args</code> CTE to strongly type and bind input parameters
 *       (<code>as_of</code>, <code>portfolio_code</code>, <code>instrument_code</code>). This keeps
 *       the SQL readable and avoids repeating placeholders.</li>
 *   <li>Temporal SCD2 joins: dimension tables (portfolio/instrument) are constrained to the
 *       provided <code>as_of</code> window (<code>valid_from &lt;= as_of &amp;&amp; valid_to &gt;= as_of</code>).</li>
 *   <li>Explicit JOINs only (no implicit comma joins) to ensure alias scope is unambiguous and
 *       avoid errors such as “invalid reference to FROM-clause entry”.</li>
 *   <li>Dates are bound as <code>java.sql.Date</code> and compared via <code>args.as_of</code> or
 *       <code>cast(? as date)</code> to prevent PostgreSQL from mis-typing null/untyped values
 *       (e.g., “null$8”).</li>
 *   <li>Unioned result sets (trades + adjustments) return the exact same columns in the same order.</li>
 *   <li>DEBUG logging can inline parameters for easier troubleshooting without executing string-built SQL.</li>
 * </ul>
 */
@Slf4j
@Repository
public class JodiTransactionLineageRepository {
    //private static final Logger log = LoggerFactory.getLogger(JodiTransactionLineageRepository.class);
    private final DSLContext dslContext;

    public JodiTransactionLineageRepository(DSLContext dslContext) {
        this.dslContext = dslContext;
    }

    /**
     * Fetch high-level position header for a (portfolioCode, instrumentCode) as-of a date.
     *
     * <p>Returns a single row with net quantity, last price, market value, currency and basic
     * instrument metadata resolved via SCD2 dimensions at the given as-of.</p>
     */
    public PositionDetailDTO fetchHeader(LocalDate asOf, String portfolioCode, String instrumentCode) {
        String sql = """
                  WITH args AS (
                    SELECT cast(? as date) AS as_of,
                           cast(? as text) AS portfolio_code,
                           cast(? as text) AS instrument_code
                  ),
                  p AS (
                    SELECT dp.portfolio_vid, dp.portfolio_code
                      FROM ibor.dim_portfolio dp, args
                     WHERE dp.portfolio_code = args.portfolio_code
                       AND dp.valid_from <= args.as_of AND dp.valid_to >= args.as_of
                     ORDER BY valid_from DESC
                     LIMIT 1
                  ),
                  i AS (
                    SELECT di.instrument_vid, di.instrument_code, di.instrument_name, di.instrument_type, di.currency_code
                      FROM ibor.dim_instrument di, args
                     WHERE di.instrument_code = args.instrument_code
                       AND di.valid_from <= args.as_of AND di.valid_to >= args.as_of
                     ORDER BY valid_from DESC
                     LIMIT 1
                  ),
                  pos AS (
                    SELECT SUM(ps.quantity) AS qty
                      FROM args
                      JOIN p ON true
                      JOIN i ON true
                      JOIN ibor.fact_position_snapshot ps
                        ON ps.portfolio_vid = p.portfolio_vid
                       AND ps.instrument_vid = i.instrument_vid
                     WHERE ps.position_date = args.as_of
                  ),
                  adj AS (
                    SELECT COALESCE(SUM(a.quantity_delta), 0)::numeric AS qty_adj
                      FROM args
                      JOIN p ON true
                      JOIN i ON true
                      JOIN ibor.fact_position_adjustment a
                        ON a.portfolio_vid = p.portfolio_vid
                       AND a.instrument_vid = i.instrument_vid
                     WHERE a.effective_date <= args.as_of
                  ),
                  price_pick AS (
                    SELECT DISTINCT ON (fp.instrument_vid)
                           fp.price,
                           fp.currency_code AS price_currency
                      FROM args
                      JOIN i ON true
                      JOIN ibor.fact_price fp ON fp.instrument_vid = i.instrument_vid
                     WHERE fp.price_ts <= (args.as_of + time '23:59:59')
                     ORDER BY fp.instrument_vid, fp.price_ts DESC
                  )
                  SELECT
                    args.as_of AS as_of,
                    (SELECT portfolio_code FROM p) AS portfolio_code,
                    (SELECT instrument_code FROM i) AS instrument_code,
                    (SELECT instrument_name FROM i) AS instrument_name,
                    (SELECT instrument_type FROM i) AS instrument_type,
                    (COALESCE((SELECT qty FROM pos),0) + (SELECT qty_adj FROM adj))::numeric AS net_qty,
                    (SELECT price FROM price_pick) AS price,
                    (COALESCE((SELECT qty FROM pos),0) + (SELECT qty_adj FROM adj))
                    * COALESCE((SELECT price FROM price_pick), 0) AS market_value,
                    COALESCE((SELECT price_currency FROM price_pick), (SELECT currency_code FROM i)) AS currency
                  FROM args
                """;

        Date d = Date.valueOf(asOf);
        Object[] params = new Object[]{ d, portfolioCode, instrumentCode };
        if (log.isDebugEnabled()) {
            log.debug("fetchHeader SQL:\n{}", inlineParameters(sql, params));
        }
        var result = dslContext.fetchOne(sql, params);

        if (result == null) {
            return null;
        }
        return new PositionDetailDTO(
                result.get("as_of", LocalDate.class),
                result.get("portfolio_code", String.class),
                result.get("instrument_code", String.class),
                result.get("instrument_name", String.class),
                result.get("instrument_type", String.class),
                result.get("net_qty", BigDecimal.class),
                result.get("price", BigDecimal.class),
                result.get("market_value", BigDecimal.class),
                result.get("currency", String.class),
                null,   // unrealizedPnl: add when you have cost basis
                "NONE",             // lottingMethod: add when you have cost basis
                List.of(),          // transactions: add when you have transaction data
                List.of()           // lots: add when you have lot data
        );
    }

    /**
     * Fetch transaction lineage (trades and position adjustments) for a
     * (portfolioCode, instrumentCode) pair up to the as-of timestamp.
     *
     * <p>The result is a time-ordered union of:
     * <ul>
     *   <li>Trades (BUY/SELL) joined through the SCD2 account→portfolio bridge at the trade date</li>
     *   <li>Adjustments (ADJUST) applied up to the as-of date</li>
     * </ul>
     * Both sources project the same column list and are ordered by timestamp then source.</p>
     */
    public List<TransactionDTO> fetchTransactions(LocalDate asOf, String portfolioCode, String instrumentCode) {
        String sql = """
                WITH args AS (
                    SELECT cast(? as date) AS as_of,
                           cast(? as text) AS portfolio_code,
                           cast(? as text) AS instrument_code
                ),
                p AS (
                      SELECT dp.portfolio_vid
                        FROM ibor.dim_portfolio dp, args
                      WHERE dp.portfolio_code = args.portfolio_code
                        AND dp.valid_from <= args.as_of AND dp.valid_to >= args.as_of
                      ORDER BY valid_from DESC
                      LIMIT 1
                ),
                i AS (
                      SELECT di.instrument_vid
                        FROM ibor.dim_instrument di, args
                       WHERE di.instrument_code = args.instrument_code
                        AND di.valid_from <= args.as_of AND di.valid_to >= args.as_of
                      ORDER BY valid_from DESC
                      LIMIT 1
                ),
                trades AS (
                    SELECT 'TRADE'::text AS source,
                           tf.trade_code AS external_id,
                           tf.trade_date::timestamp AS ts,
                           CASE WHEN tf.quantity > 0 THEN 'BUY'::text ELSE 'SELL'::text END AS action,
                           tf.quantity::numeric AS quantity,
                           tf.price::numeric AS price,
                           (tf.quantity * tf.price)::numeric AS gross_amount,
                           tf.broker_code AS broker,
                           NULL::text AS strategy,
                           NULL::text AS notes
                    FROM args
                    JOIN p ON TRUE
                    JOIN i ON TRUE
                    JOIN ibor.fact_trade tf
                      ON tf.instrument_vid = i.instrument_vid
                    JOIN ibor.dim_account_portfolio ap
                      ON ap.account_vid   = tf.account_vid
                     AND ap.portfolio_vid = p.portfolio_vid
                     AND ap.valid_from   <= tf.trade_date
                     AND ap.valid_to     >= tf.trade_date
                   WHERE tf.trade_date <= (args.as_of + time '23:59:59')
                ),
                adjustments AS (
                    SELECT 'ADJUST'::text AS source,
                           a.position_adjustment_id::text AS external_id,
                           a.effective_date::timestamp AS ts,
                           'ADJUST'::text AS action,
                           a.quantity_delta::numeric AS quantity,
                           NULL::numeric AS price,
                           a.quantity_delta::numeric AS gross_amount,
                           NULL::text AS broker,
                           NULL::text AS strategy,
                           a.reason AS notes
                    FROM args
                    JOIN p ON TRUE
                    JOIN i ON TRUE
                    JOIN ibor.fact_position_adjustment a
                      ON a.portfolio_vid = p.portfolio_vid
                     AND a.instrument_vid = i.instrument_vid
                    WHERE a.effective_date <= args.as_of
                )
                SELECT * FROM (
                    SELECT * FROM trades
                    UNION ALL
                    SELECT * FROM adjustments
                ) AS all_transactions
                ORDER BY ts ASC, source DESC;
        """;

        Date d2 = Date.valueOf(asOf);
        Object[] params2 = new Object[]{ d2, portfolioCode, instrumentCode };
        if (log.isDebugEnabled()) {
            log.debug("fetchTransactions SQL:\n{}", inlineParameters(sql, params2));
        }

        return dslContext.resultQuery(sql, params2).fetch(r -> new TransactionDTO(
                r.get("source", String.class),
                r.get("external_id", String.class),
                r.get("ts", java.time.LocalDateTime.class),
                r.get("action", String.class),
                r.get("quantity", BigDecimal.class),
                r.get("price", BigDecimal.class),
                r.get("gross_amount", BigDecimal.class),
                r.get("broker", String.class),
                r.get("strategy", String.class),
                r.get("notes", String.class)
        ));
    }

    /**
     * Debug helper: produces a best-effort SQL string with parameters inlined in place of '?'
     * for log inspection only. Do not execute the returned string.
     */
    private static String inlineParameters(String sql, Object[] params) {
        StringBuilder sb = new StringBuilder();
        int idx = 0;
        for (int i = 0; i < sql.length(); i++) {
            char c = sql.charAt(i);
            if (c == '?' && idx < params.length) {
                sb.append(formatParam(params[idx++]));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /** Formats a single bound parameter for {@link #inlineParameters(String, Object[])}. */
    private static String formatParam(Object v) {
        switch (v) {
            case null -> {
                return "NULL";
            }
            case Date d -> {
                return "DATE '" + d.toLocalDate() + "'";
            }
            case LocalDate ld -> {
                return "DATE '" + ld + "'";
            }
            case Number number -> {
                return v.toString();
            }
            default -> {
            }
        }
        String s = v.toString().replace("'", "''");
        return "'" + s + "'";
    }
}