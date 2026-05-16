import React, { useState, useEffect, useCallback, useMemo } from 'react'
import AiChat from './components/AiChat.jsx'
import { fetchPositions, fetchPositionDetail } from './api/ibor.js'
import axios from 'axios'

const TYPE_COLORS = {
  EQUITY: '#4a9eff',
  BOND:   '#2ecc71',
  FUT:    '#f39c12',
  OPT:    '#9b59b6',
  FX:     '#1abc9c',
  INDEX:  '#e74c3c',
  OTHER:  '#7f8c8d',
}

const usdFmt = (v) =>
  v != null
    ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v)
    : ''

const numFmt = (v) =>
  v != null ? new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(v) : ''

const compactAum = (v) => {
  if (v == null || isNaN(v)) return '$0'
  if (Math.abs(v) >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`
  if (Math.abs(v) >= 1_000)     return `$${(v / 1_000).toFixed(1)}K`
  return `$${Math.round(v)}`
}

function computeAssetMix(positions) {
  const totals = {}
  let grand = 0
  for (const p of positions) {
    const type = p.instrumentType || p.type || 'OTHER'
    const mv = Math.abs(p.mktValue ?? p.marketValue ?? 0)
    const mult = p.contractMultiplier ?? 1
    const val = mult > 1 ? mv / mult : mv
    totals[type] = (totals[type] || 0) + val
    grand += val
  }
  if (grand === 0) return []
  return Object.entries(totals)
    .map(([type, amount]) => ({ type, amount, pct: (amount / grand) * 100 }))
    .sort((a, b) => b.amount - a.amount)
}

const DEFAULT_AS_OF = new Date().toISOString().slice(0, 10)
const DEFAULT_PORTFOLIO = 'P-ALPHA'

function PositionCard({ position, isExpanded, onToggle, onSelectTrades, loadingTrades, hasTransactions }) {
  const qty = position.netQty ?? position.quantity ?? 0
  const direction = qty > 0 ? 'LONG' : qty < 0 ? 'SHORT' : 'FLAT'
  const directionColor = direction === 'LONG' ? 'var(--green)' : direction === 'SHORT' ? 'var(--red)' : 'var(--text-3)'

  const mktValue = position.mktValue ?? position.marketValue ?? 0
  const ticker = position.ticker || (position.instrumentId || '').replace(/^(EQ|BOND|FUT|OPT|FX|INDEX)-/, '')
  const assetName = position.instrumentName || position.instrumentId || ticker

  return (
    <div className="mobile-position-card">
      <button
        className="mobile-position-header"
        onClick={() => onToggle(position.instrumentId)}
      >
        <div className="mobile-position-header-left">
          <div className="mobile-position-name-ticker">
            <div className="mobile-position-name">{assetName}</div>
            <div className="mobile-position-ticker">{ticker}</div>
          </div>
        </div>
        <div className="mobile-position-header-right">
          <div className="mobile-position-value">{usdFmt(mktValue)}</div>
          <div className="mobile-position-direction" style={{ color: directionColor }}>
            {direction}
          </div>
          <div className={`mobile-position-caret ${isExpanded ? 'expanded' : ''}`}>▼</div>
        </div>
      </button>

      {isExpanded && (
        <div className="mobile-position-details">
          <button
            className="mobile-position-details-close"
            onClick={() => onToggle(position.instrumentId)}
            aria-label="Close details"
          >
            ✕
          </button>
          <div className="mobile-detail-row">
            <span className="mobile-detail-label">Type</span>
            <span className="mobile-detail-value">{position.instrumentType || '-'}</span>
          </div>
          <div className="mobile-detail-row">
            <span className="mobile-detail-label">Quantity</span>
            <span className="mobile-detail-value">{numFmt(qty)}</span>
          </div>
          <div className="mobile-detail-row">
            <span className="mobile-detail-label">Price</span>
            <span className="mobile-detail-value">{usdFmt(position.price ?? 0)}</span>
          </div>
          <div className="mobile-detail-row">
            <span className="mobile-detail-label">Currency</span>
            <span className="mobile-detail-value">{position.currency || 'USD'}</span>
          </div>
          {position.contractMultiplier && position.contractMultiplier !== 1 && (
            <div className="mobile-detail-row">
              <span className="mobile-detail-label">Multiplier</span>
              <span className="mobile-detail-value">{position.contractMultiplier}</span>
            </div>
          )}

          <button
            className="mobile-detail-action-btn"
            onClick={() => onSelectTrades(position.instrumentId)}
            disabled={loadingTrades}
          >
            {loadingTrades ? 'Loading trades...' : hasTransactions ? 'View trades' : 'No trades'}
          </button>
        </div>
      )}
    </div>
  )
}

function TradesModal({ isOpen, onClose, trades, instrumentId, loading }) {
  if (!isOpen) return null

  return (
    <div className="mobile-modal-overlay" onClick={onClose}>
      <div className="mobile-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="mobile-modal-handle" onClick={onClose} />
        <div className="mobile-modal-header">
          <h3>Trades — {instrumentId}</h3>
          <button className="mobile-modal-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="mobile-modal-body">
          {loading ? (
            <div className="mobile-trades-empty">Loading...</div>
          ) : trades.length === 0 ? (
            <div className="mobile-trades-empty">No trades found</div>
          ) : (
            <div className="mobile-trades-list">
              {trades.map((t, idx) => (
                <div key={idx} className="mobile-trade-item">
                  <div className="mobile-trade-date-action">
                    <span className="mobile-trade-date">{t.transactionDate || t.date}</span>
                    <span
                      className="mobile-trade-action"
                      style={{
                        color: t.action === 'BUY' ? 'var(--green)' : t.action === 'SELL' ? 'var(--red)' : 'var(--text-3)',
                      }}
                    >
                      {t.action || t.side}
                    </span>
                  </div>
                  <div className="mobile-trade-qty-amount">
                    <span className="mobile-trade-label">Qty</span>
                    <span className="mobile-trade-value">{numFmt(t.quantity ?? 0)}</span>
                  </div>
                  <div className="mobile-trade-qty-amount">
                    <span className="mobile-trade-label">Price</span>
                    <span className="mobile-trade-value">{usdFmt(t.price ?? null)}</span>
                  </div>
                  <div className="mobile-trade-qty-amount">
                    <span className="mobile-trade-label">Amount</span>
                    <span className="mobile-trade-value">{usdFmt(t.grossAmount ?? t.amount ?? null)}</span>
                  </div>
                  {t.broker && (
                    <div className="mobile-trade-broker">Broker: {t.broker}</div>
                  )}
                  {t.notes && (
                    <div className="mobile-trade-notes">{t.notes}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ChatModal({ isOpen, onClose, positions, totalAum }) {
  if (!isOpen) return null

  return (
    <div className="mobile-modal-overlay" onClick={onClose}>
      <div className="mobile-modal-content mobile-chat-modal" onClick={(e) => e.stopPropagation()}>
        <div className="mobile-modal-handle" onClick={onClose} />
        <div className="mobile-modal-header">
          <h3>Ask Me</h3>
          <button className="mobile-modal-close" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="mobile-modal-body mobile-chat-body">
          <AiChat
            onAnswer={() => {}}
            useContext={true}
            onContextChange={() => {}}
            positions={positions}
            totalAum={totalAum}
          />
        </div>
      </div>
    </div>
  )
}

export default function AppMobile({ theme, toggleTheme }) {
  const [asOf, setAsOf] = useState(DEFAULT_AS_OF)
  const [portfolioCode, setPortfolioCode] = useState(DEFAULT_PORTFOLIO)
  const [positions, setPositions] = useState([])
  const [transactions, setTransactions] = useState([])
  const [totalAum, setTotalAum] = useState(0)
  const [pnlDelta, setPnlDelta] = useState(null)
  const [snapDate, setSnapDate] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingTrades, setLoadingTrades] = useState(false)
  const [expandedPosition, setExpandedPosition] = useState(null)
  const [showTradesModal, setShowTradesModal] = useState(false)
  const [showChatModal, setShowChatModal] = useState(false)
  const [selectedInstrument, setSelectedInstrument] = useState(null)
  const [filterOpen, setFilterOpen] = useState(false)
  const [mixOpen, setMixOpen] = useState(false)

  const handleSubmit = useCallback(async () => {
    setLoading(true)
    setExpandedPosition(null)
    setSelectedInstrument(null)
    setTransactions([])
    setShowTradesModal(false)
    try {
      const posData = await fetchPositions(portfolioCode, asOf)
      const rows = Array.isArray(posData) ? posData : posData?.positions || []
      setPositions(rows)
      const aum = rows.reduce((sum, p) => sum + (p.mktValue ?? p.marketValue ?? 0), 0)
      setTotalAum(aum)
      setSnapDate(rows[0]?.snapDate || null)

      try {
        const { data: pnlData } = await axios.get('/api/pnl', { params: { portfolioCode, asOf } })
        const delta = pnlData?.delta ?? ((pnlData?.currentMarketValue ?? 0) - (pnlData?.previousMarketValue ?? 0))
        setPnlDelta(delta)
      } catch { setPnlDelta(null) }
    } catch (err) {
      console.error('Failed to load data:', err)
      setPositions([])
      setTotalAum(0)
      setSnapDate(null)
      setPnlDelta(null)
    } finally {
      setLoading(false)
    }
  }, [portfolioCode, asOf])

  useEffect(() => { handleSubmit() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSelectTrades = useCallback(async (instrumentId) => {
    setSelectedInstrument(instrumentId)
    setShowTradesModal(true)
    setLoadingTrades(true)
    try {
      const detail = await fetchPositionDetail(portfolioCode, instrumentId, asOf)
      setTransactions(detail?.transactions || [])
    } catch (err) {
      console.warn('Failed to load trades:', err)
      setTransactions([])
    } finally {
      setLoadingTrades(false)
    }
  }, [portfolioCode, asOf])

  const positionList = useMemo(() => {
    return Array.isArray(positions) ? positions : []
  }, [positions])

  const assetMix = useMemo(() => computeAssetMix(positionList), [positionList])

  const transactionList = useMemo(() => {
    return Array.isArray(transactions)
      ? transactions.map((t) => ({
          ...t,
          transactionDate: t.transactionDate || t.tradeDate || t.date || '',
          action: t.action || t.side || '',
          quantity: t.quantity ?? 0,
          price: t.price ?? null,
          grossAmount: t.grossAmount ?? t.amount ?? null,
          broker: t.broker || t.brokerCode || '',
          notes: t.notes || '',
        }))
      : []
  }, [transactions])

  return (
    <div className="mobile-app">
      {/* Header */}
      <div className="mobile-header">
        <div className="mobile-header-brand">
          <div className="filter-bar-brand-dot" />
          <span>IBOR</span>
        </div>

        <button
          className="mobile-header-filters"
          onClick={() => setFilterOpen(!filterOpen)}
          title="Date & Portfolio"
        >
          {asOf}
          <span className="mobile-filter-caret">{filterOpen ? '▲' : '▼'}</span>
        </button>

        <div className="mobile-header-actions">
          <button
            className="mobile-header-btn"
            onClick={handleSubmit}
            disabled={loading}
            title="Refresh"
          >
            {loading ? '⟳' : '↻'}
          </button>
          <button className="mobile-header-btn" onClick={toggleTheme} title="Toggle theme">
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </div>

      {/* Collapsible filter row */}
      {filterOpen && (
        <div className="mobile-filter-panel">
          <div className="mobile-filter-field">
            <label>As of</label>
            <input
              type="date"
              value={asOf}
              onChange={(e) => setAsOf(e.target.value)}
              style={{ width: '100%' }}
            />
          </div>
          <div className="mobile-filter-field">
            <label>Portfolio</label>
            <select value={portfolioCode} onChange={(e) => setPortfolioCode(e.target.value)} style={{ width: '100%' }}>
              <option value="P-ALPHA">P-ALPHA</option>
            </select>
          </div>
          <button
            className="mobile-filter-btn"
            onClick={() => { handleSubmit(); setFilterOpen(false) }}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Load'}
          </button>
        </div>
      )}

      {/* Main content — scrollable area */}
      <div className="mobile-content">
        {/* Compact summary bar */}
        <div className="mobile-summary-compact">
          <div className="mobile-summary-row">
            <div className="mobile-summary-item">
              <div className="mobile-summary-label">AUM</div>
              <div className="mobile-summary-aum">{compactAum(totalAum)}</div>
            </div>
            <div className="mobile-summary-item mobile-summary-item-right">
              <div className="mobile-summary-label">Daily P&L</div>
              <div
                className="mobile-summary-pnl"
                style={{ color: pnlDelta == null ? 'var(--text-3)' : pnlDelta >= 0 ? 'var(--green)' : 'var(--red)' }}
              >
                {pnlDelta == null ? '—' : `${pnlDelta >= 0 ? '▲' : '▼'} ${compactAum(Math.abs(pnlDelta))}`}
              </div>
            </div>
          </div>

          {/* Horizontal asset mix bar */}
          {assetMix.length > 0 && (
            <button className="mobile-mix-toggle" onClick={() => setMixOpen(!mixOpen)}>
              <div className="mobile-mix-stack">
                {assetMix.map(({ type, pct }) => (
                  <div
                    key={type}
                    className="mobile-mix-segment"
                    style={{ width: `${pct}%`, background: TYPE_COLORS[type] || TYPE_COLORS.OTHER }}
                    title={`${type} ${pct.toFixed(0)}%`}
                  />
                ))}
              </div>
              <span className="mobile-mix-caret">{mixOpen ? '▲' : '▼'}</span>
            </button>
          )}

          {mixOpen && (
            <div className="mobile-mix-detail">
              {assetMix.map(({ type, amount, pct }) => (
                <div key={type} className="mobile-mix-row">
                  <span className="mobile-mix-dot" style={{ background: TYPE_COLORS[type] || TYPE_COLORS.OTHER }} />
                  <span className="mobile-mix-name">{type}</span>
                  <span className="mobile-mix-amount">{compactAum(amount)}</span>
                  <span className="mobile-mix-pct">{pct.toFixed(0)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Positions list */}
        <div className="mobile-positions-section">
          <div className="mobile-section-header">
            <h2>Holdings</h2>
            {positionList.length > 0 && (
              <span className="mobile-section-count">{positionList.length}</span>
            )}
          </div>

          {loading ? (
            <div className="mobile-loading">Loading positions...</div>
          ) : positionList.length === 0 ? (
            <div className="mobile-empty">No positions found</div>
          ) : (
            <div className="mobile-positions-list">
              {positionList.map((pos) => (
                <PositionCard
                  key={pos.instrumentId}
                  position={pos}
                  isExpanded={expandedPosition === pos.instrumentId}
                  onToggle={(id) => setExpandedPosition((prev) => (prev === id ? null : id))}
                  onSelectTrades={handleSelectTrades}
                  loadingTrades={loadingTrades && selectedInstrument === pos.instrumentId}
                  hasTransactions={true}
                />
              ))}
            </div>
          )}
        </div>

        {/* Spacer to push button below content */}
        <div style={{ height: '80px' }} />
      </div>

      {/* Chat button - fixed at bottom */}
      <div className="mobile-chat-button-fixed">
        <button
          className="mobile-chat-button"
          onClick={() => setShowChatModal(true)}
        >
          💬 Ask Me
        </button>
      </div>

      {/* Modals */}
      <TradesModal
        isOpen={showTradesModal}
        onClose={() => setShowTradesModal(false)}
        trades={transactionList}
        instrumentId={selectedInstrument}
        loading={loadingTrades}
      />
      <ChatModal
        isOpen={showChatModal}
        onClose={() => setShowChatModal(false)}
        positions={positionList}
        totalAum={totalAum}
      />
    </div>
  )
}
