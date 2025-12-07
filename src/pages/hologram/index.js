import React from 'react'
import Layout from '@theme/Layout'
import HologramParticles from '@site/src/components/HologramParticles'

export default function HologramPage() {
  const [status, setStatus] = React.useState('INITIALIZING...')
  const [hand, setHand] = React.useState(null)
  return (
    <Layout title="Hologram" description="Holographic gesture interaction">
      <div className="app-card" style={{ margin: 16 }}>
        <div className="app-header">
          <h2 className="app-title">Holographic Interaction</h2>
          <div className="app-muted">Camera permission required</div>
        </div>
        <HologramParticles text="SMILER488" onStatusChange={setStatus} onHandDetect={setHand} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
          <div className="app-muted">{status}</div>
          <div className="app-muted">X: {hand?.x?.toFixed?.(2) || '—'} Y: {hand?.y?.toFixed?.(2) || '—'}</div>
        </div>
      </div>
    </Layout>
  )
}

