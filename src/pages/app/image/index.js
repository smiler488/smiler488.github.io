import React from 'react';
import Layout from '@theme/Layout';
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
        <p>
          Quantify leaves or seeds/grains with server-side analysis. Upload images, set reference and expected count,
          and get morphology and color metrics. The analysis runs on HuggingFace Spaces (Gradio), keeping the site static.
        </p>
        <div style={{ border: '1px solid var(--ifm-color-emphasis-300)', borderRadius: 8, overflow: 'hidden', marginTop: 12 }}>
          <iframe
            src={`${spaceUrl}/?__theme=light`}
            title="Image Quantifier"
            style={{ width: '100%', height: 900, border: '0' }}
            allow="camera; microphone; clipboard-read; clipboard-write; encrypted-media"
            loading="lazy"
          />
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
