import React from 'react'

export default function HologramParticles({ text, onStatusChange, onHandDetect, style, className }) {
  const canvasRef = React.useRef(null)
  const videoRef = React.useRef(null)
  const containerRef = React.useRef(null)
  const FRICTION = 0.94
  const EASE = 0.05
  const RADIUS_ATTRACT = 300
  const RADIUS_REPEL = 200

  React.useEffect(() => {
    let animationFrameId
    let particles = []
    let mouse = { x: -10000, y: -10000 }
    let handData = null
    let handLandmarker = null

    class Particle {
      constructor(x, y, color) {
        this.baseX = x
        this.baseY = y
        this.x = Math.random() * window.innerWidth
        this.y = Math.random() * window.innerHeight
        this.z = Math.random()
        this.vx = 0
        this.vy = 0
        this.color = color
        this.size = 1.5 + this.z
        this.density = Math.random() * 30 + 1
      }
      draw(ctx) {
        ctx.fillStyle = this.color
        ctx.globalAlpha = 0.4 + this.z * 0.6
        ctx.beginPath()
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2)
        ctx.closePath()
        ctx.fill()
        ctx.globalAlpha = 1
      }
      update() {
        let interactionX = -10000
        let interactionY = -10000
        let isActive = false
        let isAttracting = false
        if (handData) {
          interactionX = handData.x
          interactionY = handData.y
          isActive = true
          isAttracting = handData.isClosed
        } else if (mouse.x > 0) {
          interactionX = mouse.x
          interactionY = mouse.y
          isActive = true
          isAttracting = false
        }
        if (isActive) {
          const dx = interactionX - this.x
          const dy = interactionY - this.y
          const distance = Math.sqrt(dx * dx + dy * dy)
          const radius = isAttracting ? RADIUS_ATTRACT : RADIUS_REPEL
          if (distance < radius) {
            const forceDirectionX = dx / distance
            const forceDirectionY = dy / distance
            const maxDistance = radius
            const force = (maxDistance - distance) / maxDistance
            const direction = isAttracting ? 1 : -1
            const power = isAttracting ? 40 : 15
            const parallax = 0.5 + this.z
            const forceX = forceDirectionX * force * this.density * parallax * direction * power
            const forceY = forceDirectionY * force * this.density * parallax * direction * power
            this.vx += forceX
            this.vy += forceY
          }
        }
        const returnForce = isAttracting ? EASE * 0.1 : EASE
        const dxBase = this.baseX - this.x
        const dyBase = this.baseY - this.y
        this.vx += dxBase * returnForce
        this.vy += dyBase * returnForce
        this.vx *= FRICTION
        this.vy *= FRICTION
        this.x += this.vx
        this.y += this.vy
      }
    }

    const init = async () => {
      onStatusChange && onStatusChange('INITIALIZING OPTICS...')
      if (!canvasRef.current || !containerRef.current) return
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const resize = () => {
        canvas.width = containerRef.current?.offsetWidth || window.innerWidth
        canvas.height = containerRef.current?.offsetHeight || window.innerHeight
        setTimeout(createParticles, 100)
      }
      const createParticles = () => {
        particles = []
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        const fontSize = Math.min(canvas.width / 5, 250)
        ctx.font = `900 ${fontSize}px "Orbitron"`
        ctx.fillStyle = 'white'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(text || 'SMILER488', canvas.width / 2, canvas.height / 3)
        const textCoordinates = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const gap = 5
        for (let y = 0; y < textCoordinates.height; y += gap) {
          for (let x = 0; x < textCoordinates.width; x += gap) {
            if (textCoordinates.data[y * 4 * textCoordinates.width + x * 4 + 3] > 128) {
              const r = Math.random()
              let color = '#e5e7eb' // Silver
              if (r > 0.7) color = '#d1d5db' // Darker silver
              else if (r < 0.2) color = '#f3f4f6' // Lighter silver
              particles.push(new Particle(x, y, color))
            }
          }
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
      window.addEventListener('resize', resize)
      resize()
      try {
        onStatusChange && onStatusChange('CONNECTING SENSORS...')
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } } })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.onloadeddata = async () => {
            const mod = await import('@mediapipe/tasks-vision')
            const vision = await mod.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm')
            handLandmarker = await mod.HandLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', delegate: 'GPU' }, runningMode: 'VIDEO', numHands: 2 })
            onStatusChange && onStatusChange('SYSTEM ONLINE - WAITING FOR INPUT')
            try { videoRef.current.muted = true } catch { }
            videoRef.current?.play()
            predictWebcam()
          }
        }
      } catch (err) {
        onStatusChange && onStatusChange('CAMERA ERROR - MOUSE MODE ONLY')
      }
      const predictWebcam = () => {
        if (!videoRef.current || !handLandmarker) return
        const now = performance.now()
        const results = handLandmarker.detectForVideo(videoRef.current, now)
        if (results.landmarks && results.landmarks.length > 0) {
          const lm = results.landmarks[0]
          const rawX = (lm[0].x + lm[9].x) / 2
          const rawY = (lm[0].y + lm[9].y) / 2
          const x = (1 - rawX) * canvas.width
          const y = rawY * canvas.height
          const thumb = lm[4]
          const index = lm[8]
          const middle = lm[12]
          const d1 = Math.hypot(thumb.x - index.x, thumb.y - index.y)
          const d2 = Math.hypot(thumb.x - middle.x, thumb.y - middle.y)
          const isClosed = d1 < 0.08 || d2 < 0.08
          handData = { x, y, isClosed }
          onHandDetect && onHandDetect(handData)
          onStatusChange && onStatusChange(isClosed ? 'GRAVITY WELL ENGAGED' : 'REPULSION FIELD ACTIVE')
        } else {
          handData = null
          onHandDetect && onHandDetect(null)
          onStatusChange && onStatusChange('SYSTEM ONLINE - WAITING FOR INPUT')
        }
        requestAnimationFrame(predictWebcam)
      }
      const animate = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        particles.forEach(p => { p.update(); p.draw(ctx) })
        ctx.fillStyle = 'rgba(45, 212, 191, 0.03)'
        ctx.fillRect(0, (performance.now() / 10) % canvas.height, canvas.width, 2)
        animationFrameId = requestAnimationFrame(animate)
      }
      animate()
      const handleMouseMove = e => { if (!handData) { mouse.x = e.clientX; mouse.y = e.clientY } }
      const handleMouseLeave = () => { mouse.x = -10000; mouse.y = -10000 }
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseout', handleMouseLeave)
      return () => {
        window.removeEventListener('resize', resize)
        window.removeEventListener('mousemove', handleMouseMove)
        window.removeEventListener('mouseout', handleMouseLeave)
        cancelAnimationFrame(animationFrameId)
        if (videoRef.current && videoRef.current.srcObject) {
          const tracks = videoRef.current.srcObject.getTracks()
          tracks.forEach(t => t.stop())
        }
      }
    }
    init()
  }, [text, onStatusChange, onHandDetect])

  return (
    <div ref={containerRef} className={className} style={{ position: 'relative', width: '100%', height: '70vh', ...style }}>
      <video
        ref={videoRef}
        playsInline
        autoPlay
        muted
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', zIndex: 0 }}
      />
      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', display: 'block', zIndex: 1 }} />
    </div>
  )
}
