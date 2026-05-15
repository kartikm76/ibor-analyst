import React, { useState, useEffect } from 'react'
import AppDesktop from './AppDesktop.jsx'
import AppMobile from './AppMobile.jsx'
import DisclaimerModal from './components/DisclaimerModal.jsx'
import { useIsMobile } from './hooks/useIsMobile.js'

export default function App() {
  const isMobile = useIsMobile()
  const [theme, setTheme] = useState(() => localStorage.getItem('ibor-theme') || 'dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('ibor-theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  return (
    <>
      <DisclaimerModal />
      {isMobile ? (
        <AppMobile theme={theme} toggleTheme={toggleTheme} />
      ) : (
        <AppDesktop theme={theme} toggleTheme={toggleTheme} />
      )}
    </>
  )
}
