import React, { useEffect } from 'react';
import Layout from '@theme/Layout';
import './styles.css';

export default function BiologicalSampleAnalysisApp() {
  useEffect(() => {
    // Load the biological sample analysis script
    const script = document.createElement('script');
    script.src = '/js/image_app.js';
    script.async = true;
    script.onload = () => {
      console.log('Biological sample analysis script loaded successfully');
    };
    script.onerror = () => {
      console.error('Failed to load biological sample analysis script');
    };
    document.body.appendChild(script);

    return () => {
      // Cleanup
      if (document.body.contains(script)) {
        document.body.removeChild(script);
      }
    };
  }, []);

  return (
    <Layout title="Biological Sample Quantification Tool">
      <div className="container">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h1 className="title"> Biological Sample Quantification Tool</h1>
          <a 
            href="/docs/image-quantifier-tutorial" 
            style={{
              padding: "8px 16px",
              backgroundColor: "#007bff",
              color: "white",
              textDecoration: "none",
              borderRadius: "6px",
              fontSize: "14px",
              fontWeight: "500",
              transition: "all 0.2s ease"
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = "#0056b3";
              e.target.style.transform = "scale(1.05)";
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = "#007bff";
              e.target.style.transform = "scale(1)";
            }}
          >
             Tutorial
          </a>
        </div>
        <p className="description">
          Professional tool for quantifying biological samples including leaves, seeds, grains, and other specimens. 
          Supports flexible layouts and batch processing with precise morphological and color analysis.
        </p>

        {/* Status Bar */}
        <div id="statusBar" className="status-bar info">
          <div id="statusMessage" className="status-message">
            Ready for leaf analysis - Please set sample ID and upload image
          </div>
        </div>

        {/* Sample Information */}
        <div className="panel">
          <h3 className="panel-title"> Sample Information</h3>
          <div className="controls-grid">
            <div>
              <div className="input-group">
                <label htmlFor="sampleType" className="input-label">
                  Sample Type
                </label>
                <select
                  id="sampleType"
                  name="sampleType"
                  className="select-field"
                  defaultValue="leaves"
                  onChange={(e) => {
                    const expectedInput = document.getElementById('expectedSamples');
                    const speciesInput = document.getElementById('sampleSpecies');
                    const sampleIdInput = document.getElementById('sampleId');
                    
                    if (e.target.value === 'leaves') {
                      expectedInput.value = '5';
                      expectedInput.max = '50';
                      speciesInput.placeholder = 'e.g., Arabidopsis thaliana';
                      sampleIdInput.placeholder = 'e.g., LEAF-2024-001';
                    } else if (e.target.value === 'seeds') {
                      expectedInput.value = '20';
                      expectedInput.max = '500';
                      speciesInput.placeholder = 'e.g., Oryza sativa';
                      sampleIdInput.placeholder = 'e.g., SEED-2024-001';
                    } else if (e.target.value === 'grains') {
                      expectedInput.value = '50';
                      expectedInput.max = '1000';
                      speciesInput.placeholder = 'e.g., Triticum aestivum';
                      sampleIdInput.placeholder = 'e.g., GRAIN-2024-001';
                    }
                  }}
                >
                  <option value="leaves"> Leaves</option>
                  <option value="seeds"> Seeds</option>
                  <option value="grains"> Grains</option>
                  <option value="fruits"> Fruits</option>
                  <option value="other"> Other Specimens</option>
                </select>
                <small className="input-help">
                  Type of biological samples to analyze
                </small>
              </div>

              <div className="input-group">
                <label htmlFor="sampleId" className="input-label">
                  Sample ID / Batch Number
                </label>
                <input
                  type="text"
                  id="sampleId"
                  name="sampleId"
                  className="input-field"
                  placeholder="e.g., LEAF-2024-001"
                  defaultValue=""
                />
                <small className="input-help">
                  Unique identifier for this batch of samples
                </small>
              </div>

              <div className="input-group">
                <label htmlFor="expectedSamples" className="input-label">
                  Expected Number of Samples
                </label>
                <input
                  type="number"
                  id="expectedSamples"
                  name="expectedSamples"
                  className="input-field"
                  defaultValue="5"
                  min="1"
                  max="50"
                />
                <small className="input-help">
                  Approximate number of samples in the image
                </small>
              </div>
            </div>

            <div>
              <div className="input-group">
                <label htmlFor="sampleSpecies" className="input-label">
                  Species / Variety (Optional)
                </label>
                <input
                  type="text"
                  id="sampleSpecies"
                  name="sampleSpecies"
                  className="input-field"
                  placeholder="e.g., Arabidopsis thaliana"
                />
                <small className="input-help">
                  Species or variety name for documentation
                </small>
              </div>

              <div className="input-group">
                <label htmlFor="analysisDate" className="input-label">
                  Analysis Date
                </label>
                <input
                  type="date"
                  id="analysisDate"
                  name="analysisDate"
                  className="input-field"
                  defaultValue={new Date().toISOString().split('T')[0]}
                />
                <small className="input-help">
                  Date of analysis
                </small>
              </div>

              <div className="input-group">
                <label htmlFor="layoutType" className="input-label">
                  Sample Layout
                </label>
                <select
                  id="layoutType"
                  name="layoutType"
                  className="select-field"
                  defaultValue="flexible"
                >
                  <option value="flexible"> Flexible (Any arrangement)</option>
                  <option value="horizontal">↔ Horizontal (Left to right)</option>
                  <option value="grid">⊞ Grid (Rows and columns)</option>
                  <option value="scattered"> Scattered (Random distribution)</option>
                </select>
                <small className="input-help">
                  How samples are arranged in the image
                </small>
              </div>
            </div>
          </div>
        </div>

        {/* Reference Object Setup */}
        <div className="panel">
          <h3 className="panel-title"> Reference Object Configuration</h3>
          <div className="controls-grid">
            <div>
              <div className="input-group">
                <label htmlFor="referenceType" className="input-label">
                  Reference Object Type
                </label>
                <select
                  id="referenceType"
                  name="referenceType"
                  className="select-field param-input"
                  defaultValue="coin"
                  onChange={(e) => {
                    const sizeInput = document.getElementById('referenceSize');
                    const helpText = document.getElementById('referenceSizeHelp');
                    if (e.target.value === 'coin') {
                      sizeInput.placeholder = 'e.g., 25.0';
                      sizeInput.value = '25.0';
                      helpText.textContent = 'Diameter of coin in millimeters';
                    } else if (e.target.value === 'square') {
                      sizeInput.placeholder = 'e.g., 20.0';
                      sizeInput.value = '20.0';
                      helpText.textContent = 'Side length of square card in millimeters';
                    }
                  }}
                >
                  <option value="coin">Coin (Circle) - Recommended</option>
                  <option value="square">Square Card</option>
                </select>
                <small className="input-help">
                  Place in top-left corner of image
                </small>
              </div>
            </div>

            <div>
              <div className="input-group">
                <label htmlFor="referenceSize" className="input-label">
                  Reference Size (mm)
                </label>
                <input
                  type="number"
                  id="referenceSize"
                  name="referenceSize"
                  className="input-field param-input"
                  defaultValue="25.0"
                  min="0.1"
                  max="1000"
                  step="0.1"
                  placeholder="e.g., 25.0"
                />
                <small id="referenceSizeHelp" className="input-help">
                  Diameter of coin in millimeters
                </small>
              </div>
            </div>
          </div>
        </div>

        {/* Image Upload */}
        <div className="panel">
          <h3 className="panel-title"> Image Upload</h3>
          <div className="input-group">
            <label htmlFor="imageInput" className="input-label">
              Upload Leaf Image
            </label>
            <div className="file-input-container">
              <input
                type="file"
                id="imageInput"
                className="file-input"
                accept="image/*"
                capture="environment"
              />
              <label htmlFor="imageInput" className="file-input-label">
                 Choose Image or Take Photo
              </label>
            </div>
            <small className="input-help">
              Layout: Reference object (top-left) → Leaves arranged horizontally from left to right
            </small>
          </div>
        </div>

        {/* Detection Parameters */}
        <div className="panel">
          <h3 className="panel-title">🔧 Detection Parameters</h3>
          <div className="controls-grid">
            <div>
              <div className="input-group">
                <label htmlFor="minSampleArea" className="input-label">
                  Min Sample Area: <span id="minSampleAreaValue">500</span> px²
                </label>
                <input
                  type="range"
                  id="minSampleArea"
                  name="minSampleArea"
                  className="param-input"
                  defaultValue="500"
                  min="10"
                  max="5000"
                  step="10"
                  onInput={(e) => document.getElementById('minSampleAreaValue').textContent = e.target.value}
                />
                <small className="input-help">
                  Minimum area to be considered as a sample
                </small>
              </div>

              <div className="input-group">
                <label htmlFor="maxSampleArea" className="input-label">
                  Max Sample Area: <span id="maxSampleAreaValue">50000</span> px²
                </label>
                <input
                  type="range"
                  id="maxSampleArea"
                  name="maxSampleArea"
                  className="param-input"
                  defaultValue="50000"
                  min="1000"
                  max="200000"
                  step="1000"
                  onInput={(e) => document.getElementById('maxSampleAreaValue').textContent = e.target.value}
                />
                <small className="input-help">
                  Maximum area to be considered as a sample
                </small>
              </div>

              <div className="input-group">
                <label htmlFor="colorTolerance" className="input-label">
                  Color Tolerance: <span id="colorToleranceValue">40</span>
                </label>
                <input
                  type="range"
                  id="colorTolerance"
                  name="colorTolerance"
                  className="param-input"
                  defaultValue="40"
                  min="10"
                  max="100"
                  step="5"
                  onInput={(e) => document.getElementById('colorToleranceValue').textContent = e.target.value}
                />
                <small className="input-help">
                  Color similarity threshold for sample detection
                </small>
              </div>
            </div>

            <div>
              <div className="input-group">
                <label htmlFor="edgeThreshold" className="input-label">
                  Edge Sensitivity: <span id="edgeThresholdValue">100</span>
                </label>
                <input
                  type="range"
                  id="edgeThreshold"
                  name="edgeThreshold"
                  className="param-input"
                  defaultValue="100"
                  min="50"
                  max="200"
                  step="10"
                  onInput={(e) => document.getElementById('edgeThresholdValue').textContent = e.target.value}
                />
                <small className="input-help">
                  Edge detection sensitivity for sample boundaries
                </small>
              </div>

              <div className="input-group">
                <label htmlFor="separationDistance" className="input-label">
                  Min Separation: <span id="separationDistanceValue">10</span> px
                </label>
                <input
                  type="range"
                  id="separationDistance"
                  name="separationDistance"
                  className="param-input"
                  defaultValue="10"
                  min="5"
                  max="50"
                  step="5"
                  onInput={(e) => document.getElementById('separationDistanceValue').textContent = e.target.value}
                />
                <small className="input-help">
                  Minimum distance between separate samples
                </small>
              </div>

              <div className="input-group">
                <label htmlFor="shapeFilter" className="input-label">
                  Shape Filter
                </label>
                <select
                  id="shapeFilter"
                  name="shapeFilter"
                  className="select-field param-input"
                  defaultValue="adaptive"
                >
                  <option value="adaptive"> Adaptive (Auto-adjust)</option>
                  <option value="circular"> Circular (Seeds/Grains)</option>
                  <option value="elongated"> Elongated (Leaves)</option>
                  <option value="irregular"> Irregular (Any shape)</option>
                </select>
                <small className="input-help">
                  Shape characteristics for filtering
                </small>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="button-group">
          <button id="analyzeBtn" className="btn btn-primary">
             Analyze Samples
          </button>
          <button id="resetBtn" className="btn btn-ghost">
             Reset Analysis
          </button>
          <button id="downloadBtn" className="btn btn-blue">
             Download Results
          </button>
          <button id="previewBtn" className="btn btn-ghost">
             Preview Detection
          </button>
        </div>

        {/* Canvas Display */}
        <div className="canvas-grid">
          <div className="panel">
            <h3 className="panel-title"> Original Image</h3>
            <canvas 
              id="originalCanvas" 
              className="canvas-box"
              style={{cursor: 'crosshair'}}
            />
            <small className="input-help">
              Click on the reference object to mark it for scale calculation
            </small>
          </div>

          <div className="panel">
            <h3 className="panel-title"> Analysis Results</h3>
            <canvas 
              id="processedCanvas" 
              className="canvas-box"
            />
            <small className="input-help">
              Processed image with detected leaves and measurements
            </small>
          </div>
        </div>

        {/* Layout Guide */}
        <div className="panel guide-panel">
          <h3 className="panel-title"> Sample Layout Guidelines</h3>
          <div className="guide-content">
            <div className="guide-text">
              <h4> Optimal Setup for Sample Analysis:</h4>
              <ul>
                <li><strong>Background:</strong> Use white or light-colored uniform background</li>
                <li><strong>Reference Object:</strong> Place coin or square card in a clear area</li>
                <li><strong>Sample Arrangement:</strong> Flexible - any layout works (horizontal, grid, scattered)</li>
                <li><strong>Spacing:</strong> Ensure samples don't overlap or touch each other</li>
                <li><strong>Lighting:</strong> Use even, diffused lighting to avoid shadows</li>
                <li><strong>Camera Position:</strong> Keep camera parallel to the surface</li>
              </ul>
              <h4> Measured Parameters:</h4>
              <ul>
                <li><strong>Morphological:</strong> Length, Width, Area, Perimeter, Aspect Ratio, Circularity</li>
                <li><strong>Shape:</strong> Solidity, Convexity, Roundness, Compactness</li>
                <li><strong>Color:</strong> RGB values, HSV values, Color Indices</li>
                <li><strong>Position:</strong> X,Y coordinates, Spatial distribution</li>
              </ul>
              <h4> Sample Types:</h4>
              <ul>
                <li><strong>Leaves:</strong> Morphological analysis, green index, shape characteristics</li>
                <li><strong>Seeds/Grains:</strong> Size distribution, roundness, color uniformity</li>
                <li><strong>Fruits:</strong> Size, shape, color maturity indicators</li>
                <li><strong>Other:</strong> Custom analysis based on sample characteristics</li>
              </ul>
            </div>
            <div className="guide-visual">
              <div className="sample-layout-demo">
                <div className="reference-position"> Reference</div>
                <div className="sample-position sample-1"> S1</div>
                <div className="sample-position sample-2"> S2</div>
                <div className="sample-position sample-3"> S3</div>
                <div className="sample-position sample-4"> S4</div>
                <div className="sample-position sample-5"> S5</div>
                <div className="sample-position sample-6"> S6</div>
                <div className="sample-position sample-7"> S7</div>
                <div className="sample-position sample-8"> S8</div>
              </div>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="panel metrics-panel">
          <h3 className="panel-title"> Sample Analysis Results</h3>
          <div id="resultsContainer">
            <p className="info-message">
               Analysis results will appear here after processing. Each detected sample will be numbered 
              and measured for morphological and color characteristics. Data can be exported as CSV with 
              sample ID and specimen numbering. Supports batch analysis of leaves, seeds, grains, and other biological specimens.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}