import React from 'react';
import Layout from '@theme/Layout';
import RequireAuthBanner from '../../../components/RequireAuthBanner';
import CitationNotice from '../../../components/CitationNotice';

export default function BiologicalSampleAnalysisApp() {
  const spaceUrl = 'https://smiler488-image-quantifier.hf.space';
  return (
    <Layout title="Biological Sample Quantification Tool">
      <main className="container margin-vert--lg app-container">
        <div className="app-header" style={{ marginBottom: 12 }}>
          <h1 className="app-title">Biological Sample Quantification Tool</h1>
          <a href="/docs/tutorial-apps/image-quantifier-tutorial" className="button button--secondary">
            Tutorial
          </a>
        </div>
        <RequireAuthBanner />
        <p>
          Quantify leaves or seeds/grains with server-side analysis. Upload images, set reference and expected count,
          and get morphology and color metrics. The analysis runs on HuggingFace Spaces (Gradio), keeping the site static.
        </p>
        <div style={{ border: '1px solid var(--ifm-color-emphasis-300)', borderRadius: 8, overflow: 'hidden', marginTop: 12 }}>
          <div style={{ position: 'relative' }}>
          <iframe
            src={`${spaceUrl}/?__theme=light`}
            title="Image Quantifier"
            style={{ width: '100%', height: 900, border: '0' }}
            allow="camera; microphone; clipboard-read; clipboard-write; encrypted-media"
            loading="lazy"
          />
          {typeof window !== 'undefined' && !window.__APP_AUTH_OK__ && (
            <div style={{ position: 'absolute', inset: 0, background: 'var(--app-overlay-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <a href="/auth" className="button button--secondary">Login to use</a>
            </div>
          )}
          </div>
        </div>
        <div style={{ marginTop: 12, fontSize: 12, color: 'var(--ifm-color-emphasis-700)' }}>
          If the app does not load, check your network or open the Space directly: <a href={spaceUrl} target="_blank" rel="noreferrer">{spaceUrl}</a>
        </div>
        <div style={{ marginTop: 16 }}>
          <CitationNotice />
        </div>
      </main>
    </Layout>
  );
}
