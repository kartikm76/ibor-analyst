import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import FilterBar from './components/FilterBar.jsx'
import PortfolioSnapshot from './components/PortfolioSnapshot.jsx'
import AiChat from './components/AiChat.jsx'
import DisclaimerModal from './components/DisclaimerModal.jsx'
import { fetchPositions, fetchPositionDetail } from './api/ibor.js'
import { AgGridReact } from 'ag-grid-react'
import axios from 'axios'
import 'ag-grid-community/styles/ag-grid.css'
import 'ag-grid-community/styles/ag-theme-alpine.css'

const usdFmt = (p) =>
  p.value != null
    ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(p.value)
    : ''

const numFmt = (p) =>
  p.value != null ? new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(p.value) : ''

const directionCellStyle = (params) => {
  if (params.value === 'LONG') return { color: 'var(--green)', fontWeight: 600 }
  if (params.value === 'SHORT') return { color: 'var(--red)', fontWeight: 600 }
  return { color: 'var(--text-3)' }
}

const actionCellStyle = (params) => {
  if (params.value === 'BUY') return { color: 'var(--green)', fontWeight: 600 }
  if (params.value === 'SELL') return { color: 'var(--red)', fontWeight: 600 }
  if (params.value === 'ADJUST') return { color: 'var(--orange)', fontWeight: 600 }
  return {}
}

const POSITION_COLS = [
  { field: 'assetName', headerName: 'Asset Name', flex: 1, minWidth: 120 },
  { field: 'ticker', headerName: 'Ticker', flex: 0.6, minWidth: 80 },
  { field: 'assetType', headerName: 'Asset Type', flex: 0.6, minWidth: 90 },
  { field: 'direction', headerName: 'Direction', width: 100, cellStyle: directionCellStyle },
  { field: 'quantity', headerName: 'Quantity', type: 'numericColumn', valueFormatter: numFmt, flex: 0.6, minWidth: 90 },
  { field: 'price', headerName: 'Price', type: 'numericColumn', valueFormatter: usdFmt, flex: 0.6, minWidth: 90 },
  { field: 'marketValue', headerName: 'Market Value', type: 'numericColumn', valueFormatter: usdFmt, flex: 0.8, minWidth: 120 },
  { field: 'currency', headerName: 'Currency', width: 90 },
  { field: 'contractMultiplier', headerName: 'Multiplier', type: 'numericColumn', width: 95 },
]

const TRANSACTION_COLS = [
  { field: 'externalId', headerName: 'Transaction ID', flex: 1.2, minWidth: 120 },
  { field: 'source', headerName: 'Source', flex: 0.5, minWidth: 70 },
  { field: 'transactionDate', headerName: 'Date', flex: 0.7, minWidth: 90 },
  { field: 'action', headerName: 'Action', flex: 0.5, minWidth: 70, cellStyle: actionCellStyle },
  { field: 'quantity', headerName: 'Quantity', type: 'numericColumn', valueFormatter: numFmt, flex: 0.6, minWidth: 80 },
  { field: 'price', headerName: 'Price', type: 'numericColumn', valueFormatter: usdFmt, flex: 0.6, minWidth: 80 },
  { field: 'grossAmount', headerName: 'Gross Amount', type: 'numericColumn', valueFormatter: usdFmt, flex: 0.8, minWidth: 100 },
  { field: 'broker', headerName: 'Broker', flex: 0.5, minWidth: 70 },
  { field: 'notes', headerName: 'Notes', flex: 1, minWidth: 80 },
]

function normalizePosition(p) {
  const qty = p.netQty ?? p.quantity ?? 0
  return {
    instrumentId: p.instrumentId || p.instrument || '',
    assetName: p.instrumentName || p.instrumentId || p.instrument || '',
    ticker: p.ticker || (p.instrumentId || '').replace(/^(EQ|BOND|FUT|OPT|FX|INDEX)-/, ''),
    assetType: p.instrumentType || p.type || '',
    direction: qty > 0 ? 'LONG' : qty < 0 ? 'SHORT' : 'FLAT',
    quantity: qty,
    price: p.price ?? 0,
    marketValue: p.mktValue ?? p.marketValue ?? 0,
    currency: p.currency || 'USD',
    contractMultiplier: p.contractMultiplier ?? 1,
  }
}

function normalizeTransaction(t) {
  const date = t.transactionDate || t.tradeDate || t.date || ''
  return {
    externalId: t.externalId || t.tradeId || t.transactionId || t.id || '',
    source: t.source || 'TRADE',
    transactionDate: date ? date.substring(0, 10) : '',
    action: t.action || t.side || '',
    quantity: t.quantity ?? 0,
    price: t.price ?? null,
    grossAmount: t.grossAmount ?? t.amount ?? null,
    broker: t.broker || t.brokerCode || '',
    notes: t.notes || '',
  }
}

function VerticalDivider({ onDrag }) {
  const [isDragging, setIsDragging] = useState(false)

  useEffect(() => {
    if (!isDragging) return
    const handleMouseMove = (e) => onDrag(e.clientX)
    const handleMouseUp = () => setIsDragging(false)
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    document.addEventListener('mouseleave', handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.removeEventListener('mouseleave', handleMouseUp)
    }
  }, [isDragging, onDrag])

  return (
    <div
      className={`vertical-divider ${isDragging ? 'active' : ''}`}
      onMouseDown={() => setIsDragging(true)}
    >
      <div className="vertical-divider-grip" />
    </div>
  )
}

const DEFAULT_AS_OF = new Date().toISOString().slice(0, 10)
const DEFAULT_PORTFOLIO = 'P-ALPHA'

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('ibor-theme') || 'dark')
  const [asOf, setAsOf] = useState(DEFAULT_AS_OF)
  const [portfolioCode, setPortfolioCode] = useState(DEFAULT_PORTFOLIO)
  const [positions, setPositions] = useState([])
  const [transactions, setTransactions] = useState([])
  const [snapDate, setSnapDate] = useState(null)
  const [totalAum, setTotalAum] = useState(0)
  const [pnlDelta, setPnlDelta] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingTxns, setLoadingTxns] = useState(false)
  const [useContext, setUseContext] = useState(true) // Always use portfolio context by default
  const [selectedInstrument, setSelectedInstrument] = useState(null)
  const [chatWidth, setChatWidth] = useState(550)  // Wider default for better readability
  const contentRef = useRef(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('ibor-theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  const handleChatDrag = useCallback((clientX) => {
    if (!contentRef.current) return
    const rect = contentRef.current.getBoundingClientRect()
    const newWidth = rect.right - clientX
    if (newWidth >= 300 && newWidth <= 900) {  // Allow wider chat window
      setChatWidth(newWidth)
    }
  }, [])

  const handleSubmit = useCallback(async () => {
    setLoading(true)
    setSelectedInstrument(null)
    setTransactions([])
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

  useEffect(() => {
    if (!selectedInstrument) {
      setTransactions([])
      return
    }
    let cancelled = false
    async function loadTxns() {
      setLoadingTxns(true)
      try {
        const detail = await fetchPositionDetail(portfolioCode, selectedInstrument, asOf)
        if (!cancelled) {
          setTransactions(detail?.transactions || [])
        }
      } catch (err) {
        console.warn('Failed to load transactions:', err)
        if (!cancelled) setTransactions([])
      } finally {
        if (!cancelled) setLoadingTxns(false)
      }
    }
    loadTxns()
    return () => { cancelled = true }
  }, [selectedInstrument, portfolioCode, asOf])

  const positionRows = useMemo(() => {
    const raw = Array.isArray(positions) ? positions : []
    return raw.map(normalizePosition)
  }, [positions])

  const transactionRows = useMemo(() => {
    const raw = Array.isArray(transactions) ? transactions : []
    return raw.map(normalizeTransaction)
  }, [transactions])

  const gridClass = theme === 'dark' ? 'ag-theme-alpine-dark' : 'ag-theme-alpine'
  const defaultColDef = useMemo(() => ({ resizable: true, sortable: true, filter: true }), [])

  const onPositionRowSelected = useCallback((event) => {
    const selected = event.api.getSelectedRows()
    if (selected.length > 0) {
      setSelectedInstrument(selected[0].instrumentId)
    } else {
      setSelectedInstrument(null)
    }
  }, [])

  function handleAiAnswer() {
    // Chat responses are informational only — never overwrite the grid data.
  }

  return (
    <div className="app-root">
      <DisclaimerModal />
      <FilterBar
        asOf={asOf}
        portfolioCode={portfolioCode}
        onDateChange={setAsOf}
        onPortfolioChange={setPortfolioCode}
        onSubmit={handleSubmit}
        loading={loading}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <div className="content-area" ref={contentRef}>
        {/* Left sidebar — composition + P&L */}
        <div className="sidebar">
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

        {/* Center — stacked grids (70/30 split) */}
        <div className="center-content">
          <div className="grid-section grid-positions">
            <div className="grid-title">
              Positions
              {positionRows.length > 0 && <span className="grid-title-count">({positionRows.length})</span>}
            </div>
            <div className={gridClass} style={{ flex: 1, minHeight: 0 }}>
              <AgGridReact
                columnDefs={POSITION_COLS}
                rowData={positionRows}
                defaultColDef={defaultColDef}
                rowHeight={32}
                headerHeight={38}
                rowSelection="single"
                onSelectionChanged={onPositionRowSelected}
                suppressCellFocus
              />
            </div>
          </div>

          <div className="grid-section grid-transactions">
            <div className="grid-title">
              Transactions
              {selectedInstrument ? (
                <span className="grid-title-filter">
                  {selectedInstrument}
                  <button className="grid-title-clear" onClick={() => setSelectedInstrument(null)}>×</button>
                </span>
              ) : (
                <span className="grid-title-hint">Select a position above</span>
              )}
              {loadingTxns && <span className="grid-title-hint">Loading...</span>}
              {!loadingTxns && selectedInstrument && transactionRows.length > 0 && (
                <span className="grid-title-count">({transactionRows.length})</span>
              )}
            </div>
            <div className={gridClass} style={{ flex: 1, minHeight: 0 }}>
              <AgGridReact
                columnDefs={TRANSACTION_COLS}
                rowData={transactionRows}
                defaultColDef={defaultColDef}
                rowHeight={32}
                headerHeight={38}
                suppressCellFocus
                overlayNoRowsTemplate={
                  selectedInstrument
                    ? '<span style="color: var(--text-3)">No transactions found</span>'
                    : '<span style="color: var(--text-3)">Click a position to see its transactions</span>'
                }
              />
            </div>
          </div>
        </div>

        {/* Vertical drag divider */}
        <VerticalDivider onDrag={handleChatDrag} />

        {/* Right sidebar — chat */}
        <div className="chat-sidebar" style={{ width: chatWidth, minWidth: chatWidth }}>
          <AiChat
            onAnswer={handleAiAnswer}
            useContext={useContext}
            onContextChange={setUseContext}
            positions={positions}
            totalAum={totalAum}
          />
        </div>
      </div>
    </div>
  )
}
