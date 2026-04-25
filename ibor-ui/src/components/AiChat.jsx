import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'

const GREETING = "Ask me about positions, trades, P&L, and market data."
const MAX_LINES_COLLAPSED = 30

function ThinkingDots() {
  return (
    <span className="thinking-dots">
      <span /><span /><span />
    </span>
  )
}

function MessageBubble({ message }) {
  const isUser = message.role === 'user'
  const [expanded, setExpanded] = useState(false)

  let content = message.content
  let needsExpand = false

  if (!isUser && Array.isArray(content)) {
    needsExpand = content.length > MAX_LINES_COLLAPSED
    const visible = expanded ? content : content.slice(0, MAX_LINES_COLLAPSED)
    content = (
      <>
        {visible.map((line, i) => (
          <div key={i} className={`response-line level-${line.level || 0}`}>
            {line.text}
          </div>
        ))}
        {needsExpand && !expanded && (
          <button className="chat-expand-btn" onClick={() => setExpanded(true)}>
            Show more ({content.length - MAX_LINES_COLLAPSED} lines)
          </button>
        )}
        {needsExpand && expanded && (
          <button className="chat-expand-btn" onClick={() => setExpanded(false)}>
            Show less
          </button>
        )}
      </>
    )
  } else if (!isUser && typeof content === 'string') {
    const lines = content.split('\n')
    needsExpand = lines.length > MAX_LINES_COLLAPSED
    const visible = expanded ? lines : lines.slice(0, MAX_LINES_COLLAPSED)
    content = (
      <>
        {visible.map((line, i) => (
          <div key={i} style={{ marginBottom: '3px' }}>{line}</div>
        ))}
        {needsExpand && !expanded && (
          <button className="chat-expand-btn" onClick={() => setExpanded(true)}>
            Show more ({lines.length - MAX_LINES_COLLAPSED} lines)
          </button>
        )}
        {needsExpand && expanded && (
          <button className="chat-expand-btn" onClick={() => setExpanded(false)}>
            Show less
          </button>
        )}
      </>
    )
  }

  const timeLabel = message.timestamp
    ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null

  return (
    <div className={`chat-bubble-wrap ${isUser ? 'user' : 'assistant'}`}>
      <div className={`chat-bubble ${isUser ? 'user' : 'assistant'}`}>
        {message.thinking ? <ThinkingDots /> : content}
      </div>
      {timeLabel && <div className="chat-timestamp">{timeLabel}</div>}
    </div>
  )
}

export default function AiChat({ onAnswer, useContext, onContextChange, positions, totalAum }) {
  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', content: GREETING, timestamp: Date.now() }
  ])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [marketContents, setMarketContents] = useState(true)  // Toggle for market data
  const [quotaStatus, setQuotaStatus] = useState(null)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)
  const nextId = useRef(2)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    textareaRef.current?.focus()
  }, [])

  async function handleSend() {
    const question = input.trim()
    if (!question || sending) return

    // Reject garbage input — must have at least 2 real English words
    const words = question.split(/\s+/).filter(w => /^[a-zA-Z$]{2,}$/.test(w))
    if (words.length < 2) {
      const userMsgId = nextId.current++
      const t = Date.now()
      setMessages(prev => [
        ...prev,
        { id: userMsgId, role: 'user', content: question, timestamp: t },
        { id: nextId.current++, role: 'assistant', content: 'Please ask a valid question about your portfolio, positions, trades, or market data.', timestamp: t },
      ])
      setInput('')
      return
    }

    const userMsgId = nextId.current++
    const thinkingMsgId = nextId.current++

    const now = Date.now()
    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: 'user', content: question, timestamp: now },
      { id: thinkingMsgId, role: 'assistant', content: '', thinking: true },
    ])
    setInput('')
    setSending(true)

    try {
      const { data } = await axios.post(
        '/analyst/chat',
        {
          question,
          portfolio_code: 'P-ALPHA',
          market_contents: marketContents
        },
        { headers: { 'Content-Type': 'application/json' } }
      )

      // Capture quota status from response
      if (data.quota_status) {
        setQuotaStatus(data.quota_status)
      }

      let summary = data.summary || '(No response)'

      // Format: concise data-driven answer, with AI narrative trimmed
      if (!useContext) {
        summary = formatConciseResponse(summary, positions, totalAum, question)
      } else {
        summary = formatContextResponse(summary, positions, question)
      }

      setMessages(prev =>
        prev.map(m =>
          m.id === thinkingMsgId
            ? { ...m, content: summary, thinking: false, timestamp: Date.now() }
            : m
        )
      )

      if (onAnswer) onAnswer(data)
    } catch (err) {
      const errMsg = err?.response?.data?.detail || err.message || 'An error occurred.'

      // Capture quota status from error response too
      if (err?.response?.data?.quota_status) {
        setQuotaStatus(err.response.data.quota_status)
      }

      setMessages(prev =>
        prev.map(m =>
          m.id === thinkingMsgId
            ? { ...m, content: `${errMsg}`, thinking: false, timestamp: Date.now() }
            : m
        )
      )
    } finally {
      setSending(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <>
      <div className="chat-header">
        IBOR Chat
      </div>

      <div className="context-checkbox">
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input
            type="checkbox"
            id="market-toggle"
            checked={marketContents}
            onChange={(e) => setMarketContents(e.target.checked)}
          />
          <span>Market data & sentiment</span>
        </label>
      </div>

      <div className="chat-messages">
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        {quotaStatus?.quota_exceeded && (
          <div className="quota-banner">
            ❌ Daily question limit reached. Come back tomorrow at {new Date(quotaStatus.reset_time).toLocaleString()}
          </div>
        )}
        {quotaStatus && !quotaStatus.quota_exceeded && (
          <div className="quota-info">
            {quotaStatus.questions_remaining} of {quotaStatus.questions_limit} questions remaining
          </div>
        )}
        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            className="chat-input-field"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={quotaStatus?.quota_exceeded ? "Daily limit reached" : "What do you want to know?"}
            rows={2}
            disabled={sending || quotaStatus?.quota_exceeded}
          />
          <button
            className="chat-send-btn"
            onClick={handleSend}
            disabled={sending || !input.trim() || quotaStatus?.quota_exceeded}
          >
            {sending ? '…' : 'Send'}
          </button>
        </div>
      </div>
    </>
  )
}

const fmtUsd = v => new Intl.NumberFormat('en-US', {
  style: 'currency', currency: 'USD',
  minimumFractionDigits: 0, maximumFractionDigits: 0,
}).format(v)

/**
 * Concise response: data-driven, no AI narrative.
 * Shows a brief summary card — positions visible in the grid.
 */
function formatConciseResponse(summary, positions, totalAum, question) {
  try {
    const tickers = extractTickersFromQuestion(question)
    let relevantPositions = positions || []

    if (tickers.length > 0) {
      relevantPositions = relevantPositions.filter(p => {
        const inst = (p.instrumentId || p.instrument || '').toUpperCase()
        const name = (p.instrumentName || '').toUpperCase()
        return tickers.some(t => inst.includes(t) || name.includes(t))
      })
    }

    const lines = []
    const posCount = relevantPositions.length

    if (tickers.length > 0 && posCount === 0) {
      lines.push({ level: 0, text: `No holdings found for ${tickers.join(', ')}.` })
      return lines
    }

    if (tickers.length > 0) {
      lines.push({ level: 0, text: `${tickers.join(', ')} — ${posCount} holding(s):` })
    } else {
      lines.push({ level: 0, text: `Portfolio — ${posCount} positions` })
    }

    // Show up to 5 positions as compact lines
    const shown = relevantPositions.slice(0, 5)
    shown.forEach(pos => {
      const name = pos.instrumentName || pos.instrumentId || pos.instrument || '?'
      const qty = pos.netQty ?? pos.quantity ?? 0
      const mktValue = pos.mktValue ?? pos.marketValue ?? 0
      const dir = qty > 0 ? 'L' : qty < 0 ? 'S' : '—'
      lines.push({ level: 1, text: `${name}: ${dir} ${Math.abs(qty).toLocaleString()} · ${fmtUsd(mktValue)}` })
    })

    if (relevantPositions.length > 5) {
      lines.push({ level: 1, text: `… and ${relevantPositions.length - 5} more` })
    }

    // One-line total
    const total = relevantPositions.reduce((s, p) => s + (p.mktValue ?? p.marketValue ?? 0), 0)
    lines.push({ level: 0, text: '' })
    lines.push({ level: 0, text: `Total: ${fmtUsd(total)}` })

    return lines
  } catch (err) {
    console.warn('Format error:', err)
    return [{ level: 0, text: summary }]
  }
}

/**
 * Context response: data card from local positions (source of truth) +
 * filtered AI commentary that doesn't contradict the data.
 */
function formatContextResponse(summary, positions, question) {
  try {
    // Clean wrappers
    let text = summary.replace(/^```json\s*/i, '').replace(/```\s*$/, '')
    text = text.replace(/^```\s*/i, '').replace(/```\s*$/, '')
    text = text.replace(/^[\s\[\{]*"?summary"?\s*:\s*\[?\s*/i, '')
    text = text.replace(/\]?\s*[\}\]]*$/, '')
    text = text.replace(/^"+/, '').replace(/"+$/, '')
    text = text.replace(/\\"/g, '"').replace(/\\n/g, '\n')
    text = text.replace(/\*\*/g, '')

    const lines = []
    const tickers = extractTickersFromQuestion(question)

    // Always lead with verified data from local positions
    let relevant = []
    if (tickers.length > 0 && positions?.length > 0) {
      relevant = positions.filter(p => {
        const inst = (p.instrumentId || p.instrument || '').toUpperCase()
        const name = (p.instrumentName || '').toUpperCase()
        return tickers.some(t => inst.includes(t) || name.includes(t))
      })
    }

    if (relevant.length > 0) {
      lines.push({ level: 0, text: `📊 ${tickers.join(', ')} — ${relevant.length} holding(s)` })
      relevant.slice(0, 8).forEach(pos => {
        const name = pos.instrumentName || pos.instrumentId || pos.instrument || '?'
        const qty = pos.netQty ?? pos.quantity ?? 0
        const price = pos.price ?? 0
        const mktValue = pos.mktValue ?? pos.marketValue ?? 0
        const dir = qty > 0 ? '📈' : qty < 0 ? '📉' : '➖'
        lines.push({ level: 1, text: `${dir} ${name}: ${Math.abs(qty).toLocaleString()} @ $${price.toFixed(2)} = ${fmtUsd(mktValue)}` })
      })
      if (relevant.length > 8) {
        lines.push({ level: 1, text: `+ ${relevant.length - 8} more` })
      }
      const total = relevant.reduce((s, p) => s + (p.mktValue ?? p.marketValue ?? 0), 0)
      lines.push({ level: 0, text: `💰 Total: ${fmtUsd(total)}` })
      lines.push({ level: 0, text: '' })
    } else if (tickers.length > 0) {
      lines.push({ level: 0, text: `No holdings found for ${tickers.join(', ')}.` })
      lines.push({ level: 0, text: '' })
    }

    // Extract key bullet points from summary
    const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 10)

    // Filter and extract 3-5 most important market points
    const keyPatterns = [
      /down\s+\d+%|up\s+\d+%|trading\s+at|notional|concentration|dominates|pressure|opportunity|risk/i
    ]

    const keyPoints = sentences.filter((s, idx) => {
      // Get first few sentences and key ones mentioning numbers/risks
      if (idx < 2) return true
      return keyPatterns.some(p => p.test(s))
    }).slice(0, 5)

    if (keyPoints.length > 0) {
      lines.push({ level: 0, text: '🎯 Key Points:' })
      keyPoints.forEach(point => {
        lines.push({ level: 1, text: `• ${point}` })
      })
    }

    return lines.length > 0 ? lines : [{ level: 0, text: text.substring(0, 300) }]
  } catch (err) {
    console.warn('Format error:', err)
    return [{ level: 0, text: summary.substring(0, 300) }]
  }
}

function extractTickersFromQuestion(question) {
  if (!question) return []
  const matches = question.match(/(\b[A-Z]{1,6}\b)/g) || []
  const exclude = ['THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'HAVE',
    'ABOUT', 'PORTFOLIO', 'POSITION', 'IBOR', 'AUM', 'ETF', 'HOW', 'GET',
    'ALL', 'SHOW', 'WHAT', 'TELL', 'GIVE', 'LIST', 'MY', 'HAS', 'ARE', 'NOT']
  return [...new Set(matches)].filter(t => !exclude.includes(t)).slice(0, 10)
}
