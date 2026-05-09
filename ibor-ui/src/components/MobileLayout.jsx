import React, { useState } from 'react'
import { DatePicker, Select } from 'antd'
import dayjs from 'dayjs'
import { AgGridReact } from 'ag-grid-react'
import PortfolioSnapshot from './PortfolioSnapshot.jsx'
import AiChat from './AiChat.jsx'

// ── Icons ────────────────────────────────────────────────────────────────────

function IconPortfolio({ active }) {
  const c = active ? 'var(--accent)' : 'var(--text-3)'
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
      <rect x="3" y="12" width="4" height="9" rx="1" fill={c} opacity={active ? 1 : 0.7} />
      <rect x="10" y="7" width="4" height="14" rx="1" fill={c} opacity={active ? 1 : 0.7} />
      <rect x="17" y="3" width="4" height="18" rx="1" fill={c} opacity={active ? 1 : 0.7} />
    </svg>
  )
}

function IconPositions({ active }) {
  const c = active ? 'var(--accent)' : 'var(--text-3)'
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
      <rect x="3" y="5" width="18" height="2.5" rx="1.25" fill={c} opacity={active ? 1 : 0.7} />
      <rect x="3" y="10.75" width="18" height="2.5" rx="1.25" fill={c} opacity={active ? 1 : 0.7} />
      <rect x="3" y="16.5" width="12" height="2.5" rx="1.25" fill={c} opacity={active ? 1 : 0.7} />
    </svg>
  )
}

function IconTrades({ active }) {
  const c = active ? 'var(--accent)' : 'var(--text-3)'
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
      <path d="M7 16L3 12L7 8" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity={active ? 1 : 0.7} />
      <path d="M3 12H14" stroke={c} strokeWidth="2" strokeLinecap="round" opacity={active ? 1 : 0.7} />
      <path d="M17 8L21 12L17 16" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity={active ? 1 : 0.7} />
      <path d="M21 12H10" stroke={c} strokeWidth="2" strokeLinecap="round" opacity={active ? 1 : 0.7} />
    </svg>
  )
}

function IconChat({ active }) {
  const c = active ? 'var(--accent)' : 'var(--text-3)'
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
      <path
        d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H7L3 21V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V15Z"
        stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        fill={active ? 'var(--accent-dim)' : 'none'}
        opacity={active ? 1 : 0.7}
      />
    </svg>
  )
}

// ── Column defs (mobile-optimised — fewer columns, wider touch targets) ───────

const usdFmt = (p) =>
  p.value != null
    ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(p.value)
    : ''

const numFmt = (p) =>
  p.value != null ? new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(p.value) : ''

const dirStyle = (p) => {
  if (p.value === 'LONG')  return { color: 'var(--green)', fontWeight: 600 }
  if (p.value === 'SHORT') return { color: 'var(--red)',   fontWeight: 600 }
  return {}
}

const actionStyle = (p) => {
  if (p.value === 'BUY')  return { color: 'var(--green)', fontWeight: 600 }
  if (p.value === 'SELL') return { color: 'var(--red)',   fontWeight: 600 }
  return {}
}

const MOBILE_POSITION_COLS = [
  { field: 'assetName',   headerName: 'Asset',     flex: 2,   minWidth: 110 },
  { field: 'assetType',   headerName: 'Type',      width: 70 },
  { field: 'direction',   headerName: 'Dir',       width: 58, cellStyle: dirStyle },
  { field: 'marketValue', headerName: 'Mkt Value', type: 'numericColumn', valueFormatter: usdFmt, flex: 1.4, minWidth: 100 },
]

const MOBILE_TRANSACTION_COLS = [
  { field: 'transactionDate', headerName: 'Date',   width: 92 },
  { field: 'action',          headerName: 'Action', width: 70, cellStyle: actionStyle },
  { field: 'quantity',        headerName: 'Qty',    type: 'numericColumn', valueFormatter: numFmt,  flex: 1, minWidth: 70 },
  { field: 'grossAmount',     headerName: 'Amount', type: 'numericColumn', valueFormatter: usdFmt,  flex: 1.2, minWidth: 90 },
]

const PORTFOLIO_OPTIONS = [{ value: 'P-ALPHA', label: 'P-ALPHA' }]

// ── Main component ────────────────────────────────────────────────────────────

export default function MobileLayout({
  theme, toggleTheme,
  asOf, setAsOf, portfolioCode, setPortfolioCode,
  handleSubmit, loading,
  positions, positionRows, transactionRows,
  totalAum, snapDate, pnlDelta,
  selectedInstrument, setSelectedInstrument, loadingTxns,
  gridClass, defaultColDef,
  onPositionRowSelected,
  handleAiAnswer, useContext, setUseContext,
}) {
  const [activeTab, setActiveTab] = useState('portfolio')
  const [filterOpen, setFilterOpen] = useState(false)

  const hasTrades = transactionRows.length > 0

  function handleTabChange(tab) {
    setActiveTab(tab)
  }

  function handlePositionSelect(event) {
    onPositionRowSelected(event)
    // Auto-navigate to trades tab when a position is tapped
    const selected = event.api.getSelectedRows()
    if (selected.length > 0) {
      setActiveTab('trades')
    }
  }

  return (
    <div className="mobile-root">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="mobile-header">
        <div className="mobile-brand">
          <div className="filter-bar-brand-dot" />
          <span className="mobile-brand-text">IBOR</span>
        </div>

        <div className="mobile-header-center" onClick={() => setFilterOpen(o => !o)}>
          <span className="mobile-portfolio-pill">
            {portfolioCode} · {asOf}
            <span className="mobile-filter-caret">{filterOpen ? '▲' : '▼'}</span>
          </span>
        </div>

        <div className="mobile-header-actions">
          <button
            className="mobile-icon-btn"
            onClick={handleSubmit}
            disabled={loading}
            title="Refresh"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M1 4V10H7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M23 20V14H17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M20.49 9A9 9 0 005.64 5.64L1 10M23 14L18.36 18.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <button className="mobile-icon-btn" onClick={toggleTheme} title="Toggle theme">
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </div>

      {/* ── Collapsible filter row ───────────────────────────────────────── */}
      {filterOpen && (
        <div className="mobile-filter-row">
          <div className="mobile-filter-item">
            <span className="mobile-filter-label">As of</span>
            <DatePicker
              value={asOf ? dayjs(asOf) : null}
              format="YYYY-MM-DD"
              onChange={(d) => { if (d) setAsOf(d.format('YYYY-MM-DD')) }}
              allowClear={false}
              size="small"
              style={{ width: 140 }}
            />
          </div>
          <div className="mobile-filter-item">
            <span className="mobile-filter-label">Portfolio</span>
            <Select
              value={portfolioCode}
              options={PORTFOLIO_OPTIONS}
              onChange={setPortfolioCode}
              size="small"
              style={{ width: 120 }}
            />
          </div>
          <button
            className="submit-btn"
            onClick={() => { handleSubmit(); setFilterOpen(false) }}
            disabled={loading}
          >
            {loading ? '…' : 'Load'}
          </button>
        </div>
      )}

      {/* ── Tab content ─────────────────────────────────────────────────── */}
      <div className="mobile-content">

        {/* Portfolio tab */}
        <div className={`mobile-tab-pane ${activeTab === 'portfolio' ? 'active' : ''}`}>
          <div className="mobile-scroll-area">
            <PortfolioSnapshot
              positions={positions}
              totalAum={totalAum}
              snapDate={snapDate}
              asOf={asOf}
              portfolioCode={portfolioCode}
              loading={loading}
              pnlDelta={pnlDelta}
            />
          </div>
        </div>

        {/* Positions tab */}
        <div className={`mobile-tab-pane ${activeTab === 'positions' ? 'active' : ''}`}>
          <div className="mobile-grid-header">
            <span>Positions</span>
            {positionRows.length > 0 && (
              <span className="mobile-grid-count">{positionRows.length} holdings</span>
            )}
          </div>
          <div className={`${gridClass} mobile-grid`}>
            <AgGridReact
              columnDefs={MOBILE_POSITION_COLS}
              rowData={positionRows}
              defaultColDef={defaultColDef}
              rowHeight={44}
              headerHeight={40}
              rowSelection="single"
              onSelectionChanged={handlePositionSelect}
              suppressCellFocus
            />
          </div>
          {positionRows.length > 0 && (
            <div className="mobile-grid-hint">Tap a position to view its trades</div>
          )}
        </div>

        {/* Trades tab */}
        <div className={`mobile-tab-pane ${activeTab === 'trades' ? 'active' : ''}`}>
          <div className="mobile-grid-header">
            <span>Trades</span>
            {selectedInstrument && (
              <span className="mobile-instrument-pill">
                {selectedInstrument}
                <button className="grid-title-clear" onClick={() => setSelectedInstrument(null)}>×</button>
              </span>
            )}
            {loadingTxns && <span className="mobile-hint-text">Loading…</span>}
          </div>
          <div className={`${gridClass} mobile-grid`}>
            <AgGridReact
              columnDefs={MOBILE_TRANSACTION_COLS}
              rowData={transactionRows}
              defaultColDef={defaultColDef}
              rowHeight={44}
              headerHeight={40}
              suppressCellFocus
              overlayNoRowsTemplate={
                selectedInstrument
                  ? '<span style="color:var(--text-3)">No transactions found</span>'
                  : '<span style="color:var(--text-3)">Tap a position to see its trades</span>'
              }
            />
          </div>
        </div>

        {/* Chat tab */}
        <div className={`mobile-tab-pane mobile-chat-pane ${activeTab === 'chat' ? 'active' : ''}`}>
          <AiChat
            onAnswer={handleAiAnswer}
            useContext={useContext}
            onContextChange={setUseContext}
            positions={positions}
            totalAum={totalAum}
          />
        </div>

      </div>

      {/* ── Bottom tab bar ───────────────────────────────────────────────── */}
      <nav className="mobile-tab-bar">
        {[
          { id: 'portfolio', label: 'Portfolio', Icon: IconPortfolio },
          { id: 'positions', label: 'Positions', Icon: IconPositions },
          { id: 'trades',    label: 'Trades',    Icon: IconTrades,    badge: hasTrades && selectedInstrument },
          { id: 'chat',      label: 'Chat',      Icon: IconChat },
        ].map(({ id, label, Icon, badge }) => (
          <button
            key={id}
            className={`mobile-tab-btn ${activeTab === id ? 'active' : ''}`}
            onClick={() => handleTabChange(id)}
          >
            <div className="mobile-tab-icon">
              <Icon active={activeTab === id} />
              {badge && <span className="mobile-tab-badge" />}
            </div>
            <span className="mobile-tab-label">{label}</span>
          </button>
        ))}
      </nav>

    </div>
  )
}
