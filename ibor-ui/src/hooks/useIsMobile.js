import { useState, useEffect } from 'react'

export function useIsMobile(breakpoint = 768) {
  const getIsMobileValue = () => {
    if (typeof window === 'undefined') return false
    return Math.min(window.innerWidth, window.visualViewport?.width || window.innerWidth) < breakpoint
  }

  const [isMobile, setIsMobile] = useState(getIsMobileValue())

  useEffect(() => {
    const initialValue = getIsMobileValue()
    setIsMobile(initialValue)

    const handleResize = () => {
      setIsMobile(getIsMobileValue())
    }

    const handleOrientationChange = () => {
      setTimeout(() => {
        setIsMobile(getIsMobileValue())
      }, 100)
    }

    window.addEventListener('resize', handleResize)
    window.addEventListener('orientationchange', handleOrientationChange)

    return () => {
      window.removeEventListener('resize', handleResize)
      window.removeEventListener('orientationchange', handleOrientationChange)
    }
  }, [breakpoint])

  return isMobile
}
