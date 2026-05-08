import React, { useState, useEffect } from 'react'

const STORAGE_KEY = 'ibor_disclaimer_accepted'

export default function DisclaimerModal() {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (!localStorage.getItem(STORAGE_KEY)) {
      setVisible(true)
    }
  }, [])

  function accept() {
    localStorage.setItem(STORAGE_KEY, '1')
    setVisible(false)
  }

  if (!visible) return null

  return (
    <div className="disclaimer-overlay">
      <div className="disclaimer-modal">
        <h2 className="disclaimer-title">Demo Tool — Not Financial Advice</h2>
        <div className="disclaimer-body">
          <p>
            This application is a <strong>technology demonstration</strong> only. It is not a licensed financial
            advisory service and does not constitute investment advice, a recommendation, or a solicitation to
            buy or sell any security.
          </p>
          <p>
            The portfolio data shown is <strong>synthetic and fictitious</strong>. Any AI-generated analysis,
            commentary, or market observations are illustrative and may be inaccurate, incomplete, or out of date.
          </p>
          <p>
            <strong>Do not use this tool to make real trading or investment decisions.</strong> Always consult a
            qualified financial professional before acting on any information. Past performance of any instrument
            referenced is not indicative of future results.
          </p>
          <p className="disclaimer-fine">
            By continuing, you acknowledge that this is a demo application provided for evaluation purposes only,
            with no warranty of any kind, express or implied.
          </p>
        </div>
        <button className="disclaimer-btn" onClick={accept}>
          I understand — continue to demo
        </button>
      </div>
    </div>
  )
}
