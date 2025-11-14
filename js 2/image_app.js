/**
 * Professional Biological Sample Quantification Analysis Tool
 * Features: Multi-sample type analysis (leaves, seeds, grains), flexible layout detection, batch processing
 */

class BiologicalSampleAnalysisApp {
    constructor() {
        this.originalImage = null;
        this.processedCanvas = null;
        this.processedCtx = null;
        this.referenceObjects = [];
        this.detectedSamples = [];
        this.analysisResults = [];
        this.pixelsPerMM = 1;
        this.isProcessing = false;
        this.sampleInfo = {};
        this.sampleType = 'leaves'; // 'leaves' or 'seeds'
        
        // Dynamic analysis parameters based on sample type
        this.params = {
            referenceSize: 25.0, // mm
            minSampleArea: 500,
            maxSampleArea: 50000,
            colorTolerance: 40,
            edgeThreshold: 100,
            morphKernel: 3,
            minAspectRatio: 1.0,
            maxAspectRatio: 8.0,
            minSolidity: 0.5,
            minCircularity: 0.1,
            maxCircularity: 1.0
        };
        
        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.updateStatus('Ready for biological sample analysis - Please set sample type and upload image', 'info');
        this.loadOpenCV();
    }

    loadOpenCV() {
        if (typeof cv !== 'undefined') {
            this.updateStatus('OpenCV loaded successfully', 'info');
            return;
        }

        const script = document.createElement('script');
        script.src = '/js/opencv.js';
        script.onload = () => {
            cv.onRuntimeInitialized = () => {
                this.updateStatus('OpenCV initialized successfully', 'info');
            };
        };
        script.onerror = () => {
            this.updateStatus('OpenCV not available. Using basic detection algorithms.', 'warning');
        };
        document.head.appendChild(script);
    }

    setupEventListeners() {
        // Sample type selection
        const sampleTypeSelect = document.getElementById('sampleType');
        if (sampleTypeSelect) {
            sampleTypeSelect.addEventListener('change', (e) => this.changeSampleType(e.target.value));
        }

        // File input
        const fileInput = document.getElementById('imageInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleImageUpload(e));
        }

        // Parameter controls
        const paramInputs = document.querySelectorAll('.param-input');
        paramInputs.forEach(input => {
            input.addEventListener('change', (e) => this.updateParameter(e));
        });

        // Action buttons
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzeSamples());
        }

        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetAnalysis());
        }

        const downloadBtn = document.getElementById('downloadBtn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadResults());
        }

        const previewBtn = document.getElementById('previewBtn');
        if (previewBtn) {
            previewBtn.addEventListener('click', () => this.previewDetection());
        }

        // Canvas interactions
        const originalCanvas = document.getElementById('originalCanvas');
        if (originalCanvas) {
            originalCanvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        }
    }

    changeSampleType(type) {
        this.sampleType = type;
        this.adjustParametersForSampleType();
        this.updateUI();
        this.updateStatus(`Sample type changed to: ${type}`, 'info');
    }

    adjustParametersForSampleType() {
        if (this.sampleType === 'leaves') {
            this.params = {
                ...this.params,
                minSampleArea: 200,
                maxSampleArea: 50000,
                colorTolerance: 40,
                minAspectRatio: 1.0,
                maxAspectRatio: 8.0,
                minSolidity: 0.6,
                minCircularity: 0.05,
                maxCircularity: 0.95
            };
        } else if (this.sampleType === 'seeds') {
            this.params = {
                ...this.params,
                minSampleArea: 50,
                maxSampleArea: 5000,
                colorTolerance: 50,
                minAspectRatio: 1.0,
                maxAspectRatio: 3.0,
                minSolidity: 0.8,
                minCircularity: 0.3,
                maxCircularity: 1.0
            };
        }
        
        // Update UI parameter values
        this.updateParameterInputs();
    }

    updateParameterInputs() {
        const inputs = {
            'minSampleArea': this.params.minSampleArea,
            'maxSampleArea': this.params.maxSampleArea,
            'colorTolerance': this.params.colorTolerance,
            'minAspectRatio': this.params.minAspectRatio,
            'maxAspectRatio': this.params.maxAspectRatio
        };
        
        Object.entries(inputs).forEach(([name, value]) => {
            const input = document.getElementById(name);
            if (input) {
                input.value = value;
                const display = document.getElementById(name + 'Value');
                if (display) display.textContent = value;
            }
        });
    }

    updateUI() {
        // Update titles and labels based on sample type
        const title = document.querySelector('.sample-analysis-title');
        if (title) {
            title.textContent = this.sampleType === 'leaves' ? 
                '🍃 Leaf Quantification Analysis' : 
                '🌾 Seed/Grain Quantification Analysis';
        }

        // Update layout demo
        const layoutDemo = document.querySelector('.sample-layout-demo, .leaf-layout-demo');
        if (layoutDemo) {
            layoutDemo.className = this.sampleType === 'leaves' ? 'leaf-layout-demo' : 'sample-layout-demo';
        }

        // Update expected count label
        const expectedLabel = document.querySelector('label[for="expectedSamples"]');
        if (expectedLabel) {
            expectedLabel.textContent = this.sampleType === 'leaves' ? 
                'Expected Leaves:' : 
                'Expected Seeds/Grains:';
        }
    }

    updateParameter(event) {
        const { name, value } = event.target;
        if (this.params.hasOwnProperty(name)) {
            this.params[name] = parseFloat(value) || parseInt(value) || value;
        }
        
        // Update parameter display
        const display = document.getElementById(name + 'Value');
        if (display) {
            display.textContent = value;
        }
    }

    collectSampleInfo() {
        this.sampleInfo = {
            sampleId: document.getElementById('sampleId')?.value || 'UNKNOWN',
            sampleType: this.sampleType,
            expectedSamples: parseInt(document.getElementById('expectedSamples')?.value) || 5,
            species: document.getElementById('sampleSpecies')?.value || '',
            analysisDate: document.getElementById('analysisDate')?.value || new Date().toISOString().split('T')[0],
            referenceType: document.getElementById('referenceType')?.value || 'coin',
            referenceSize: parseFloat(document.getElementById('referenceSize')?.value) || 25.0,
            timestamp: new Date().toISOString()
        };
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            this.updateStatus('Please select a valid image file', 'error');
            return;
        }

        this.updateStatus('Loading image...', 'processing');
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.originalImage = img;
                this.displayOriginalImage();
                this.resetAnalysis();
                this.updateStatus('Image loaded. Click on reference object to set scale.', 'info');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    displayOriginalImage() {
        const canvas = document.getElementById('originalCanvas');
        const ctx = canvas.getContext('2d');
        
        // Calculate display size while maintaining aspect ratio
        const maxWidth = 600;
        const maxHeight = 450;
        const scale = Math.min(maxWidth / this.originalImage.width, maxHeight / this.originalImage.height);
        
        canvas.width = this.originalImage.width * scale;
        canvas.height = this.originalImage.height * scale;
        
        ctx.drawImage(this.originalImage, 0, 0, canvas.width, canvas.height);
        
        // Store scale for coordinate conversion
        this.displayScale = scale;
    }

    handleCanvasClick(event) {
        if (!this.originalImage) return;

        const canvas = event.target;
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / this.displayScale;
        const y = (event.clientY - rect.top) / this.displayScale;

        this.addReferencePoint(x, y);
        this.redrawCanvas();
    }

    addReferencePoint(x, y) {
        // Allow multiple reference points for better accuracy
        this.referenceObjects.push({ 
            x, 
            y, 
            size: this.params.referenceSize,
            type: document.getElementById('referenceType')?.value || 'coin'
        });
        this.updateStatus(`Reference object ${this.referenceObjects.length} marked at (${Math.round(x)}, ${Math.round(y)})`, 'info');
    }

    redrawCanvas() {
        const canvas = document.getElementById('originalCanvas');
        const ctx = canvas.getContext('2d');
        
        // Clear and redraw image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(this.originalImage, 0, 0, canvas.width, canvas.height);
        
        // Draw reference points
        this.referenceObjects.forEach((ref, index) => {
            const x = ref.x * this.displayScale;
            const y = ref.y * this.displayScale;
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.08)';
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 3;
            
            if (ref.type === 'coin') {
                ctx.beginPath();
                ctx.arc(x, y, 15, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            } else {
                ctx.fillRect(x - 15, y - 15, 30, 30);
                ctx.strokeRect(x - 15, y - 15, 30, 30);
            }
            
            // Label
            ctx.fillStyle = '#000';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(`R${index + 1}`, x + 18, y - 5);
        });
        
        // Draw detected samples with flexible numbering
        this.detectedSamples.forEach((sample, index) => {
            const x = sample.centerX * this.displayScale;
            const y = sample.centerY * this.displayScale;
            
            // Draw sample boundary
            const color = '#000000';
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            if (this.sampleType === 'seeds') {
                // Draw circle for seeds
                ctx.arc(x, y, Math.sqrt(sample.area / Math.PI) * this.displayScale, 0, 2 * Math.PI);
            } else {
                // Draw ellipse for leaves
                ctx.ellipse(x, y, sample.width/2 * this.displayScale, sample.height/2 * this.displayScale, 0, 0, 2 * Math.PI);
            }
            ctx.stroke();
            
            // Draw sample number
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(x, y, 12, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.stroke();
            
            ctx.fillStyle = color;
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText((index + 1).toString(), x, y + 4);
            ctx.textAlign = 'left';
        });
    }

    async analyzeSamples() {
        if (!this.originalImage) {
            this.updateStatus('Please upload an image first', 'error');
            return;
        }

        if (this.referenceObjects.length === 0) {
            this.autoDetectReferenceTopLeft();
            if (this.referenceObjects.length === 0) {
                this.updateStatus('No reference detected; proceeding with default scale', 'warning');
            } else {
                this.updateStatus('Reference detected automatically (s0)', 'info');
            }
        }

        this.isProcessing = true;
        this.updateStatus(`Analyzing ${this.sampleType}...`, 'processing');
        
        try {
            // Collect sample information
            this.collectSampleInfo();
            
            // Calculate pixels per mm from reference objects
            this.calculatePixelsPerMM();
            
            // Detect and analyze samples
            await this.detectSamples();
            
            // Sort samples by position (flexible sorting based on sample type)
            this.sortSamples();
            
            // Generate detailed results
            this.generateSampleResults();
            
            // Display processed image
            this.displayProcessedImage();
            
            this.updateStatus(`Analysis complete. Found ${this.detectedSamples.length} ${this.sampleType}.`, 'info');
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.updateStatus('Analysis failed: ' + error.message, 'error');
        } finally {
            this.isProcessing = false;
        }
    }

    autoDetectReferenceTopLeft() {
        if (!this.originalImage) return;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.originalImage.width;
        canvas.height = this.originalImage.height;
        ctx.drawImage(this.originalImage, 0, 0);

        const searchW = Math.floor(canvas.width * 0.25);
        const searchH = Math.floor(canvas.height * 0.25);
        let best = { score: 0, x: 0, y: 0, size: 0, type: null };

        // Circle search
        for (let y = 15; y < searchH - 15; y += 8) {
            for (let x = 15; x < searchW - 15; x += 8) {
                for (let r = 10; r <= 60; r += 5) {
                    const score = this.evaluateCircle(ctx.getImageData(0, 0, canvas.width, canvas.height), x, y, r);
                    if (score > best.score) {
                        best = { score, x, y, size: r * 2, type: 'coin' };
                    }
                }
            }
        }

        // Square search (fallback)
        let bestSquare = { score: 0, x: 0, y: 0, size: 0 };
        for (let y = 20; y < searchH - 20; y += 12) {
            for (let x = 20; x < searchW - 20; x += 12) {
                for (let s = 12; s <= 60; s += 4) {
                    const score = this.evaluateSquare(ctx.getImageData(0, 0, canvas.width, canvas.height), x, y, s);
                    if (score > bestSquare.score) {
                        bestSquare = { score, x, y, size: s };
                    }
                }
            }
        }

        if (!best.type || best.score < 0.25) {
            if (bestSquare.score > 0.25) {
                this.referenceObjects = [{ x: bestSquare.x, y: bestSquare.y, size: bestSquare.size, type: 'square' }];
            }
        } else {
            this.referenceObjects = [{ x: best.x, y: best.y, size: best.size, type: 'coin' }];
        }
    }

    async previewDetection() {
        if (!this.originalImage) {
            this.updateStatus('Please upload an image first', 'error');
            return;
        }
        if (this.referenceObjects.length === 0) {
            this.updateStatus('Please click on at least one reference object to set scale', 'warning');
            return;
        }
        this.updateStatus(`Previewing ${this.sampleType} detection…`, 'processing');
        try {
            this.collectSampleInfo();
            this.calculatePixelsPerMM();
            await this.detectSamples();
            this.sortSamples();
            this.displayProcessedImage();
            this.updateStatus('Preview updated. Run full analysis to compute metrics and export.', 'info');
        } catch (error) {
            console.error('Preview error:', error);
            this.updateStatus('Preview failed: ' + error.message, 'error');
        }
    }

    sortSamples() {
        if (this.sampleType === 'leaves') {
            // Sort leaves left to right
            this.detectedSamples.sort((a, b) => a.centerX - b.centerX);
        } else {
            // Sort seeds/grains by position (top-left to bottom-right)
            this.detectedSamples.sort((a, b) => {
                const aScore = a.centerY * 0.3 + a.centerX * 0.7; // Prioritize left-right, then top-bottom
                const bScore = b.centerY * 0.3 + b.centerX * 0.7;
                return aScore - bScore;
            });
        }
    }

    calculatePixelsPerMM() {
        if (this.referenceObjects.length === 0) {
            this.pixelsPerMM = this.sampleType === 'seeds' ? 5.0 : 3.0;
            return;
        }

        let totalRatio = 0;
        let validReferences = 0;

        this.referenceObjects.forEach(ref => {
            let detectedSizePixels = 0;
            
            if (ref.type === 'coin') {
                detectedSizePixels = this.detectCircularReference(ref);
            } else if (ref.type === 'square') {
                detectedSizePixels = this.detectSquareReference(ref);
            }
            
            if (detectedSizePixels > 0) {
                totalRatio += detectedSizePixels / ref.size;
                validReferences++;
            }
        });
        
        if (validReferences > 0) {
            this.pixelsPerMM = totalRatio / validReferences;
            console.log(`Average scale from ${validReferences} references: ${this.pixelsPerMM.toFixed(4)} px/mm`);
        } else {
            // Fallback estimation based on sample type
            this.pixelsPerMM = this.sampleType === 'seeds' ? 5.0 : 3.0;
            console.warn('Could not detect reference objects, using default scale');
        }
    }

    detectCircularReference(ref) {
        // Enhanced circular detection
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.originalImage.width;
        canvas.height = this.originalImage.height;
        ctx.drawImage(this.originalImage, 0, 0);
        
        const searchRadius = 60;
        let bestRadius = 0;
        let bestScore = 0;
        
        for (let r = 5; r <= searchRadius; r += 1) {
            const score = this.evaluateCircle(ctx.getImageData(0, 0, canvas.width, canvas.height), ref.x, ref.y, r);
            if (score > bestScore) {
                bestScore = score;
                bestRadius = r;
            }
        }
        
        return bestRadius * 2; // return diameter
    }

    detectSquareReference(ref) {
        // Enhanced square detection
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.originalImage.width;
        canvas.height = this.originalImage.height;
        ctx.drawImage(this.originalImage, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const searchSize = 50;
        let bestSize = 0;
        let bestScore = 0;
        
        for (let size = 10; size <= searchSize; size += 2) {
            const score = this.evaluateSquare(imageData, ref.x, ref.y, size);
            if (score > bestScore) {
                bestScore = score;
                bestSize = size;
            }
        }
        
        return bestSize;
    }

    evaluateCircle(imageData, centerX, centerY, radius) {
        const data = imageData.data;
        const width = imageData.width;
        
        let edgePoints = 0;
        let totalPoints = 0;
        
        for (let angle = 0; angle < 2 * Math.PI; angle += Math.PI / 16) {
            const x = Math.round(centerX + radius * Math.cos(angle));
            const y = Math.round(centerY + radius * Math.sin(angle));
            
            if (x >= 1 && x < width - 1 && y >= 1 && y < imageData.height - 1) {
                if (this.isEdgePixel(imageData, x, y)) edgePoints++;
                totalPoints++;
            }
        }
        
        return totalPoints > 0 ? edgePoints / totalPoints : 0;
    }

    evaluateSquare(imageData, centerX, centerY, halfSize) {
        let edgePoints = 0;
        let totalPoints = 0;
        
        // Check all four sides of the square
        const sides = [
            // Top side
            () => {
                for (let x = centerX - halfSize; x <= centerX + halfSize; x += 2) {
                    if (this.isEdgePixel(imageData, x, centerY - halfSize)) edgePoints++;
                    totalPoints++;
                }
            },
            // Bottom side
            () => {
                for (let x = centerX - halfSize; x <= centerX + halfSize; x += 2) {
                    if (this.isEdgePixel(imageData, x, centerY + halfSize)) edgePoints++;
                    totalPoints++;
                }
            },
            // Left side
            () => {
                for (let y = centerY - halfSize; y <= centerY + halfSize; y += 2) {
                    if (this.isEdgePixel(imageData, centerX - halfSize, y)) edgePoints++;
                    totalPoints++;
                }
            },
            // Right side
            () => {
                for (let y = centerY - halfSize; y <= centerY + halfSize; y += 2) {
                    if (this.isEdgePixel(imageData, centerX + halfSize, y)) edgePoints++;
                    totalPoints++;
                }
            }
        ];
        
        sides.forEach(checkSide => checkSide());
        
        return totalPoints > 0 ? edgePoints / totalPoints : 0;
    }

    isEdgePixel(imageData, x, y) {
        const data = imageData.data;
        const width = imageData.width;
        
        if (x < 1 || x >= width - 1 || y < 1 || y >= imageData.height - 1) return false;
        
        const getPixel = (px, py) => {
            const idx = (py * width + px) * 4;
            return (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        };
        
        const center = getPixel(x, y);
        const left = getPixel(x - 1, y);
        const right = getPixel(x + 1, y);
        const top = getPixel(x, y - 1);
        const bottom = getPixel(x, y + 1);
        
        const gradientX = Math.abs(right - left);
        const gradientY = Math.abs(bottom - top);
        const gradient = Math.sqrt(gradientX * gradientX + gradientY * gradientY);
        
        return gradient > this.params.edgeThreshold * 0.4;
    }

    async detectSamples() {
        return new Promise((resolve, reject) => {
            try {
                if (typeof cv !== 'undefined' && cv.Mat) {
                    setTimeout(() => {
                        try {
                            this.detectSamplesOpenCV();
                            resolve();
                        } catch (e) {
                            reject(e);
                        }
                    }, 0);
                } else {
                    this.detectSamplesBasicChunked().then(resolve).catch(reject);
                }
            } catch (error) {
                reject(error);
            }
        });
    }

    detectSamplesBasicChunked() {
        const maxSide = 1024;
        const scale = Math.min(1, maxSide / Math.max(this.originalImage.width, this.originalImage.height));
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = Math.round(this.originalImage.width * scale);
        canvas.height = Math.round(this.originalImage.height * scale);
        ctx.drawImage(this.originalImage, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const width = canvas.width;
        const height = canvas.height;
        this.detectedSamples = [];
        const expected = Math.max(1, parseInt(document.getElementById('expectedSamples')?.value) || 999);
        const mask = new Uint8Array(width * height);
        const visited = new Uint8Array(width * height);
        const data = imageData.data;
        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const idx = (y * width + x) * 4;
                if (this.isSampleColor(data[idx], data[idx + 1], data[idx + 2])) {
                    mask[y * width + x] = 1;
                }
            }
        }
        const stepSize = this.sampleType === 'seeds' ? 4 : 6;
        const positions = [];
        for (let y = 1; y < height - 1; y += stepSize) {
            for (let x = 1; x < width - 1; x += stepSize) {
                positions.push(y * width + x);
            }
        }
        let index = 0;
        const total = positions.length;
        return new Promise((resolve) => {
            const runBatch = () => {
                const start = performance.now();
                while (index < total && performance.now() - start < 30) {
                    const p = positions[index];
                    if (mask[p] && !visited[p]) {
                        const blob = this.extractComponent(imageData, p, width, height, mask, visited, scale);
                        if (blob && this.isValidSample(blob)) {
                            const sample = this.analyzeSampleBlob(blob, imageData);
                            if (sample) {
                                this.detectedSamples.push(sample);
                                if (this.detectedSamples.length >= expected + 2) break;
                            }
                        }
                    }
                    index += 1;
                }
                const percent = Math.min(100, Math.round((index / total) * 100));
                this.updateStatus(`Scanning… ${percent}%`, 'processing');
                if (index < total && this.detectedSamples.length < expected + 2) {
                    setTimeout(runBatch, 16);
                } else {
                    resolve();
                }
            };
            runBatch();
        });
    }

    extractComponent(imageData, startIndex, width, height, mask, visited, scale) {
        const data = imageData.data;
        const stack = [startIndex];
        let area = 0;
        let minX = width, maxX = 0, minY = height, maxY = 0;
        let sumX = 0, sumY = 0;
        let sumR = 0, sumG = 0, sumB = 0;
        while (stack.length) {
            const p = stack.pop();
            if (visited[p]) continue;
            visited[p] = 1;
            if (!mask[p]) continue;
            const y = Math.floor(p / width);
            const x = p % width;
            const idx = (y * width + x) * 4;
            sumR += data[idx];
            sumG += data[idx + 1];
            sumB += data[idx + 2];
            area += 1;
            sumX += x;
            sumY += y;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            const neighbors = [p - 1, p + 1, p - width, p + width];
            for (let i = 0; i < 4; i += 1) {
                const np = neighbors[i];
                if (np > 0 && np < width * height && !visited[np]) {
                    stack.push(np);
                }
            }
        }
        if (area === 0) return null;
        const centerX = sumX / area;
        const centerY = sumY / area;
        const blob = {
            pixels: null,
            area: area,
            sumX: sumX,
            sumY: sumY,
            minX: minX,
            maxX: maxX,
            minY: minY,
            maxY: maxY,
            centerX: centerX / scale,
            centerY: centerY / scale,
            width: (maxX - minX) / scale,
            height: (maxY - minY) / scale,
            meanR: sumR / area,
            meanG: sumG / area,
            meanB: sumB / area
        };
        return blob;
    }

    floodFillSample(imageData, startX, startY, width, height, visited) {
        const data = imageData.data;
        const startIndex = (startY * width + startX) * 4;
        const targetR = data[startIndex];
        const targetG = data[startIndex + 1];
        const targetB = data[startIndex + 2];
        
        // Check if this looks like a sample color
        if (!this.isSampleColor(targetR, targetG, targetB)) return null;
        
        const stack = [{x: startX, y: startY}];
        const blob = {
            pixels: [],
            area: 0,
            minX: startX, maxX: startX,
            minY: startY, maxY: startY,
            sumX: 0, sumY: 0
        };
        
        while (stack.length > 0 && blob.area < this.params.maxSampleArea) {
            const {x, y} = stack.pop();
            const key = `${x},${y}`;
            
            if (visited.has(key) || x < 0 || x >= width || y < 0 || y >= height) continue;
            
            const index = (y * width + x) * 4;
            const r = data[index];
            const g = data[index + 1];
            const b = data[index + 2];
            
            const colorDiff = Math.sqrt(
                Math.pow(r - targetR, 2) + 
                Math.pow(g - targetG, 2) + 
                Math.pow(b - targetB, 2)
            );
            
            if (colorDiff > this.params.colorTolerance) continue;
            
            visited.add(key);
            blob.pixels.push({x, y, r, g, b});
            blob.area++;
            blob.sumX += x;
            blob.sumY += y;
            
            blob.minX = Math.min(blob.minX, x);
            blob.maxX = Math.max(blob.maxX, x);
            blob.minY = Math.min(blob.minY, y);
            blob.maxY = Math.max(blob.maxY, y);
            
            // Add neighbors
            stack.push({x: x + 1, y}, {x: x - 1, y}, {x, y: y + 1}, {x, y: y - 1});
        }
        
        if (blob.area < this.params.minSampleArea) return null;
        
        blob.centerX = blob.sumX / blob.area;
        blob.centerY = blob.sumY / blob.area;
        blob.width = blob.maxX - blob.minX;
        blob.height = blob.maxY - blob.minY;
        
        return blob;
    }

    isSampleColor(r, g, b) {
        if (this.sampleType === 'leaves') {
            // HSV-based green detection
            const { h, s, v } = this.rgbToHsv(r, g, b);
            return h >= 35 && h <= 85 && s >= 20 && v >= 20;
        } else {
            // For seeds/grains, look for broad natural colors
            const brightness = (r + g + b) / 3;
            const colorfulness = Math.max(r, g, b) - Math.min(r, g, b);
            return brightness > 30 && brightness < 235 && colorfulness > 8;
        }
    }

    isValidSample(blob) {
        if (!blob || blob.area < this.params.minSampleArea || blob.area > this.params.maxSampleArea) {
            return false;
        }
        
        const aspectRatio = Math.max(blob.width, blob.height) / Math.min(blob.width, blob.height);
        if (aspectRatio < this.params.minAspectRatio || aspectRatio > this.params.maxAspectRatio) {
            return false;
        }
        
        // Calculate circularity
        const perimeter = this.estimatePerimeter(blob);
        const circularity = (4 * Math.PI * blob.area) / (perimeter * perimeter);
        if (circularity < this.params.minCircularity || circularity > this.params.maxCircularity) {
            return false;
        }
        
        return true;
    }

    analyzeSampleBlob(blob, imageData) {
        // Calculate morphological features
        const area = blob.area;
        const perimeter = this.estimatePerimeter(blob);
        const length = Math.max(blob.width, blob.height) / this.pixelsPerMM;
        const width = Math.min(blob.width, blob.height) / this.pixelsPerMM;
        const areaReal = area / (this.pixelsPerMM * this.pixelsPerMM);
        const aspectRatio = length / width;
        const circularity = (4 * Math.PI * area) / (perimeter * perimeter);
        
        // Calculate equivalent diameter for seeds
        const equivalentDiameter = 2 * Math.sqrt(area / Math.PI) / this.pixelsPerMM;
        
        // Calculate color features
        let sumR = 0, sumG = 0, sumB = 0;
        blob.pixels.forEach(pixel => {
            sumR += pixel.r;
            sumG += pixel.g;
            sumB += pixel.b;
        });
        
        const meanR = sumR / blob.pixels.length;
        const meanG = sumG / blob.pixels.length;
        const meanB = sumB / blob.pixels.length;
        
        // Convert to HSV
        const hsv = this.rgbToHsv(meanR, meanG, meanB);
        
        // Calculate color indices
        const greenIndex = (2 * meanG - meanR - meanB) / (meanR + meanG + meanB);
        const brownIndex = (meanR - meanB) / (meanR + meanG + meanB);
        
        return {
            centerX: blob.centerX,
            centerY: blob.centerY,
            area: area,
            areaReal: areaReal,
            length: length,
            width: width,
            equivalentDiameter: equivalentDiameter,
            perimeter: perimeter / this.pixelsPerMM,
            aspectRatio: aspectRatio,
            circularity: circularity,
            meanR: meanR,
            meanG: meanG,
            meanB: meanB,
            hue: hsv.h,
            saturation: hsv.s,
            value: hsv.v,
            greenIndex: greenIndex,
            brownIndex: brownIndex,
            boundingWidth: blob.width,
            boundingHeight: blob.height
        };
    }

    estimatePerimeter(blob) {
        // Enhanced perimeter estimation
        if (this.sampleType === 'seeds') {
            // For circular objects, use circle perimeter
            return 2 * Math.sqrt(Math.PI * blob.area);
        } else {
            // For elongated objects like leaves, use ellipse approximation
            const a = blob.width / 2;
            const b = blob.height / 2;
            return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
        }
    }

    rgbToHsv(r, g, b) {
        r /= 255;
        g /= 255;
        b /= 255;
        
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const diff = max - min;
        
        let h = 0;
        if (diff !== 0) {
            if (max === r) h = ((g - b) / diff) % 6;
            else if (max === g) h = (b - r) / diff + 2;
            else h = (r - g) / diff + 4;
        }
        h = Math.round(h * 60);
        if (h < 0) h += 360;
        
        const s = max === 0 ? 0 : diff / max;
        const v = max;
        
        return {
            h: h,
            s: Math.round(s * 100),
            v: Math.round(v * 100)
        };
    }

    detectSamplesOpenCV() {
        // Advanced sample detection using OpenCV
        const src = cv.imread(this.originalImage);
        const hsv = new cv.Mat();
        const mask = new cv.Mat();
        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        
        try {
            // Convert to HSV for better color detection
            cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);
            
            // Create mask based on sample type
            let lowerBound, upperBound;
            if (this.sampleType === 'leaves') {
                lowerBound = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [35, 40, 40, 0]);
                upperBound = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [85, 255, 255, 255]);
            } else {
                // For seeds/grains, use broader color range
                lowerBound = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [10, 30, 30, 0]);
                upperBound = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [170, 255, 255, 255]);
            }
            
            cv.inRange(hsv, lowerBound, upperBound, mask);
            
            // Morphological operations
            const kernelSize = this.sampleType === 'seeds' ? 3 : 5;
            const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kernelSize, kernelSize));
            cv.morphologyEx(mask, mask, cv.MORPH_OPEN, kernel);
            cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, kernel);
            
            // Find contours
            cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            
            this.detectedSamples = [];
            
            for (let i = 0; i < contours.size(); i++) {
                const contour = contours.get(i);
                const area = cv.contourArea(contour);
                
                if (area < this.params.minSampleArea || area > this.params.maxSampleArea) {
                    contour.delete();
                    continue;
                }
                
                const rect = cv.boundingRect(contour);
                const aspectRatio = Math.max(rect.width, rect.height) / Math.min(rect.width, rect.height);
                
                if (aspectRatio < this.params.minAspectRatio || aspectRatio > this.params.maxAspectRatio) {
                    contour.delete();
                    continue;
                }
                
                const moments = cv.moments(contour);
                const centerX = moments.m10 / moments.m00;
                const centerY = moments.m01 / moments.m00;
                
                const perimeter = cv.arcLength(contour, true);
                const circularity = (4 * Math.PI * area) / (perimeter * perimeter);
                
                if (circularity < this.params.minCircularity || circularity > this.params.maxCircularity) {
                    contour.delete();
                    continue;
                }
                
                // Rotated minimum area rectangle for length/width
                const rrect = cv.minAreaRect(contour);
                const rotWidth = Math.max(rrect.size.width, rrect.size.height) / this.pixelsPerMM;
                const rotHeight = Math.min(rrect.size.width, rrect.size.height) / this.pixelsPerMM;
                let rotAngle = rrect.angle;
                if (rrect.size.width < rrect.size.height) rotAngle += 90;
                rotAngle = ((rotAngle % 180) + 180) % 180;

                // Approximate polygon for contour drawing
                const approx = new cv.Mat();
                cv.approxPolyDP(contour, approx, 2, true);
                const poly = [];
                for (let j = 0; j < approx.rows; j++) {
                    const point = approx.intPtr(j);
                    poly.push({ x: point[0], y: point[1] });
                }

                // Calculate color features
                const maskSingle = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
                cv.drawContours(maskSingle, contours, i, new cv.Scalar(255), -1);
                const meanColor = cv.mean(src, maskSingle);
                
                const hsv_mean = this.rgbToHsv(meanColor[0], meanColor[1], meanColor[2]);
                const greenIndex = (2 * meanColor[1] - meanColor[0] - meanColor[2]) / (meanColor[0] + meanColor[1] + meanColor[2]);
                const brownIndex = (meanColor[0] - meanColor[2]) / (meanColor[0] + meanColor[1] + meanColor[2]);
                
                const equivalentDiameter = 2 * Math.sqrt(area / Math.PI) / this.pixelsPerMM;
                
                this.detectedSamples.push({
                    centerX: centerX,
                    centerY: centerY,
                    area: area,
                    areaReal: area / (this.pixelsPerMM * this.pixelsPerMM),
                    length: Math.max(rect.width, rect.height) / this.pixelsPerMM,
                    width: Math.min(rect.width, rect.height) / this.pixelsPerMM,
                    rotWidth: rotWidth,
                    rotHeight: rotHeight,
                    rotAngle: rotAngle,
                    equivalentDiameter: equivalentDiameter,
                    perimeter: perimeter / this.pixelsPerMM,
                    aspectRatio: aspectRatio,
                    circularity: circularity,
                    meanR: meanColor[0],
                    meanG: meanColor[1],
                    meanB: meanColor[2],
                    hue: hsv_mean.h,
                    saturation: hsv_mean.s,
                    value: hsv_mean.v,
                    greenIndex: greenIndex,
                    brownIndex: brownIndex,
                    boundingWidth: rect.width,
                    boundingHeight: rect.height,
                    polygon: poly
                });
                
                approx.delete();
                maskSingle.delete();
                contour.delete();
            }
            
            lowerBound.delete();
            upperBound.delete();
            kernel.delete();
            
        } finally {
            src.delete();
            hsv.delete();
            mask.delete();
            contours.delete();
            hierarchy.delete();
        }
    }

    generateSampleResults() {
        const prefix = this.sampleType === 'leaves' ? 'L' : 'S';
        
        this.analysisResults = this.detectedSamples.map((sample, index) => ({
            sampleId: `${this.sampleInfo.sampleId}-${prefix}${String(index + 1).padStart(2, '0')}`,
            sampleNumber: index + 1,
            batchId: this.sampleInfo.sampleId,
            sampleType: this.sampleType,
            species: this.sampleInfo.species,
            analysisDate: this.sampleInfo.analysisDate,
            
            // Position
            centerX: Math.round(sample.centerX),
            centerY: Math.round(sample.centerY),
            
            // Morphological features
            length: sample.length.toFixed(2),
            width: sample.width.toFixed(2),
            equivalentDiameter: sample.equivalentDiameter.toFixed(2),
            area: sample.areaReal.toFixed(2),
            perimeter: sample.perimeter.toFixed(2),
            aspectRatio: sample.aspectRatio.toFixed(2),
            circularity: sample.circularity.toFixed(3),
            
            // Color features
            meanR: Math.round(sample.meanR),
            meanG: Math.round(sample.meanG),
            meanB: Math.round(sample.meanB),
            hue: Math.round(sample.hue),
            saturation: Math.round(sample.saturation),
            value: Math.round(sample.value),
            greenIndex: sample.greenIndex.toFixed(3),
            brownIndex: sample.brownIndex.toFixed(3),
            
            // Technical data
            pixelsPerMM: this.pixelsPerMM.toFixed(4),
            areaPixels: Math.round(sample.area)
        }));
        
        this.displayResults();
    }

    displayResults() {
        const container = document.getElementById('resultsContainer');
        if (!container) return;
        
        if (this.analysisResults.length === 0) {
            container.innerHTML = `<p class="info-message">No ${this.sampleType} detected. Try adjusting the parameters or check image quality.</p>`;
            return;
        }
        
        // Create results table
        const tableContainer = document.createElement('div');
        tableContainer.className = 'results-table-container';
        
        const table = document.createElement('table');
        table.className = `results-table ${this.sampleType === 'leaves' ? 'leaf-results-table' : 'sample-results-table'}`;
        
        // Header
        const header = table.createTHead();
        const headerRow = header.insertRow();
        const headers = this.sampleType === 'leaves' ? 
            ['Sample ID', 'Length (mm)', 'Width (mm)', 'Area (mm²)', 'Perimeter (mm)', 
             'Aspect Ratio', 'Circularity', 'Mean R', 'Mean G', 'Mean B', 
             'Hue', 'Saturation', 'Value', 'Green Index'] :
            ['Sample ID', 'Diameter (mm)', 'Length (mm)', 'Width (mm)', 'Area (mm²)', 
             'Aspect Ratio', 'Circularity', 'Mean R', 'Mean G', 'Mean B', 
             'Hue', 'Saturation', 'Value', 'Brown Index'];
        
        headers.forEach((text, index) => {
            const th = document.createElement('th');
            th.textContent = text;
            if (index === 0) th.className = this.sampleType === 'leaves' ? 'leaf-id' : 'sample-id';
            else if (index <= 6) th.className = 'morphological';
            else th.className = 'color-data';
            headerRow.appendChild(th);
        });
        
        // Body
        const tbody = table.createTBody();
        this.analysisResults.forEach(result => {
            const row = tbody.insertRow();
            const values = this.sampleType === 'leaves' ?
                [result.sampleId, result.length, result.width, result.area, result.perimeter,
                 result.aspectRatio, result.circularity, result.meanR, result.meanG, result.meanB,
                 result.hue, result.saturation, result.value, result.greenIndex] :
                [result.sampleId, result.equivalentDiameter, result.length, result.width, result.area,
                 result.aspectRatio, result.circularity, result.meanR, result.meanG, result.meanB,
                 result.hue, result.saturation, result.value, result.brownIndex];
            
            values.forEach((value, index) => {
                const cell = row.insertCell();
                cell.textContent = value;
                if (index === 0) cell.className = this.sampleType === 'leaves' ? 'leaf-id' : 'sample-id';
                else if (index <= 6) cell.className = 'morphological';
                else cell.className = 'color-data';
            });
        });
        
        tableContainer.appendChild(table);
        
        // Summary statistics
        const summary = document.createElement('div');
        summary.className = 'summary-stats';
        const sampleTypeDisplay = this.sampleType === 'leaves' ? 'leaves' : 'seeds/grains';
        summary.innerHTML = `
            <h4>📊 Analysis Summary</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div>
                    <p><strong>Sample ID:</strong> ${this.sampleInfo.sampleId}</p>
                    <p><strong>Sample type:</strong> ${sampleTypeDisplay}</p>
                    <p><strong>Total detected:</strong> ${this.analysisResults.length}</p>
                    <p><strong>Expected count:</strong> ${this.sampleInfo.expectedSamples}</p>
                </div>
                <div>
                    <p><strong>Average length:</strong> ${this.calculateAverage('length')} mm</p>
                    <p><strong>Average width:</strong> ${this.calculateAverage('width')} mm</p>
                    <p><strong>Average area:</strong> ${this.calculateAverage('area')} mm²</p>
                    ${this.sampleType === 'seeds' ? `<p><strong>Average diameter:</strong> ${this.calculateAverage('equivalentDiameter')} mm</p>` : ''}
                </div>
                <div>
                    <p><strong>Scale factor:</strong> ${this.pixelsPerMM.toFixed(2)} px/mm</p>
                    <p><strong>Analysis date:</strong> ${this.sampleInfo.analysisDate}</p>
                    <p><strong>Species:</strong> ${this.sampleInfo.species || 'Not specified'}</p>
                    <p><strong>Reference points:</strong> ${this.referenceObjects.length}</p>
                </div>
            </div>
        `;
        
        container.innerHTML = '';
        container.appendChild(summary);
        container.appendChild(tableContainer);
    }

    calculateAverage(property) {
        if (this.analysisResults.length === 0) return '0.00';
        const sum = this.analysisResults.reduce((acc, result) => acc + parseFloat(result[property] || 0), 0);
        return (sum / this.analysisResults.length).toFixed(2);
    }

    displayProcessedImage() {
        if (!this.processedCanvas) {
            this.processedCanvas = document.getElementById('processedCanvas');
            this.processedCtx = this.processedCanvas.getContext('2d');
        }
        
        // Set canvas size
        const maxWidth = 600;
        const maxHeight = 450;
        const scale = Math.min(maxWidth / this.originalImage.width, maxHeight / this.originalImage.height);
        
        this.processedCanvas.width = this.originalImage.width * scale;
        this.processedCanvas.height = this.originalImage.height * scale;
        
        // Draw original image
        this.processedCtx.drawImage(this.originalImage, 0, 0, this.processedCanvas.width, this.processedCanvas.height);
        
        // Overlay analysis results
        this.drawSampleAnalysisOverlay(scale);
    }

    drawSampleAnalysisOverlay(scale) {
        const ctx = this.processedCtx;
        
        // Draw reference objects
        this.referenceObjects.forEach((ref, index) => {
            const x = ref.x * scale;
            const y = ref.y * scale;
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.08)';
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 3;
            
            if (ref.type === 'coin') {
                ctx.beginPath();
                ctx.arc(x, y, 15, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            } else {
                ctx.fillRect(x - 15, y - 15, 30, 30);
                ctx.strokeRect(x - 15, y - 15, 30, 30);
            }
            
            ctx.fillStyle = '#000';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(`s0`, x + 18, y - 5);
        });
        
        // Draw detected samples with detailed annotations
        const color = '#000000';
        
        this.detectedSamples.forEach((sample, index) => {
            const x = sample.centerX * scale;
            const y = sample.centerY * scale;
            const sampleNumber = index + 1; // start from s1
            
            // Draw sample boundary
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            if (sample.polygon && sample.polygon.length > 0) {
                ctx.moveTo(sample.polygon[0].x * scale, sample.polygon[0].y * scale);
                for (let i = 1; i < sample.polygon.length; i++) {
                    ctx.lineTo(sample.polygon[i].x * scale, sample.polygon[i].y * scale);
                }
                ctx.closePath();
            } else {
                // Fallback: axis-aligned bounding box
                ctx.rect((sample.centerX - sample.boundingWidth/2) * scale, (sample.centerY - sample.boundingHeight/2) * scale, sample.boundingWidth * scale, sample.boundingHeight * scale);
            }
            ctx.stroke();
            
            // Draw sample number background
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 16, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw sample number
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(sampleNumber.toString(), x, y + 5);
            
            // Draw measurements
            ctx.fillStyle = color;
            ctx.font = '10px Arial';
            ctx.textAlign = 'left';
            
            const L = (sample.rotWidth || sample.length || sample.boundingWidth / this.pixelsPerMM);
            const W = (sample.rotHeight || sample.width || sample.boundingHeight / this.pixelsPerMM);
            ctx.fillText(`L:${Number(L).toFixed(1)}mm`, x + 20, y - 12);
            ctx.fillText(`W:${Number(W).toFixed(1)}mm`, x + 20, y + 0);
            ctx.fillText(`A:${sample.areaReal.toFixed(1)}mm²`, x + 20, y + 12);
            if (typeof sample.rotAngle === 'number') {
                ctx.fillText(`θ:${Math.round(sample.rotAngle)}°`, x + 20, y + 24);
            }

            ctx.textAlign = 'center';
        });
        
        // Draw title
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'left';
        const sampleTypeDisplay = this.sampleType === 'leaves' ? 'leaves' : 'seeds/grains';
        ctx.fillText(`${this.sampleInfo.sampleId} - ${this.detectedSamples.length} ${sampleTypeDisplay} detected`, 10, 25);
    }

    resetAnalysis() {
        this.referenceObjects = [];
        this.detectedSamples = [];
        this.analysisResults = [];
        this.pixelsPerMM = 1;
        
        // Clear results
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) {
            resultsContainer.innerHTML = '<p class="info-message">📋 Analysis results will appear here after processing.</p>';
        }
        
        // Clear processed canvas
        if (this.processedCanvas) {
            this.processedCtx.clearRect(0, 0, this.processedCanvas.width, this.processedCanvas.height);
        }
        
        // Redraw original image
        if (this.originalImage) {
            this.displayOriginalImage();
        }
        
        this.updateStatus('Analysis reset. Ready for new analysis.', 'info');
    }

    downloadResults() {
        if (this.analysisResults.length === 0) {
            this.updateStatus(`No results to download. Please analyze ${this.sampleType} first.`, 'warning');
            return;
        }
        
        try {
            this.collectSampleInfo();
            
            // Prepare comprehensive data package
            const analysisData = {
                metadata: {
                    sampleId: this.sampleInfo.sampleId,
                    sampleType: this.sampleType,
                    species: this.sampleInfo.species,
                    analysisDate: this.sampleInfo.analysisDate,
                    timestamp: this.sampleInfo.timestamp,
                    totalSamples: this.analysisResults.length,
                    expectedSamples: this.sampleInfo.expectedSamples,
                    pixelsPerMM: this.pixelsPerMM,
                    referenceObjects: this.referenceObjects.map(ref => ({
                        type: ref.type,
                        size: ref.size,
                        position: { x: ref.x, y: ref.y }
                    }))
                },
                parameters: this.params,
                results: this.analysisResults,
                summary: {
                    averageLength: this.calculateAverage('length'),
                    averageWidth: this.calculateAverage('width'),
                    averageArea: this.calculateAverage('area'),
                    averageAspectRatio: this.calculateAverage('aspectRatio'),
                    averageCircularity: this.calculateAverage('circularity'),
                    ...(this.sampleType === 'seeds' && {
                        averageDiameter: this.calculateAverage('equivalentDiameter'),
                        averageBrownIndex: this.calculateAverage('brownIndex')
                    }),
                    ...(this.sampleType === 'leaves' && {
                        averageGreenIndex: this.calculateAverage('greenIndex')
                    })
                }
            };
            
            // Generate CSV content
            const csvContent = this.generateSampleCSV();
            
            // Download files
            this.downloadFile(JSON.stringify(analysisData, null, 2), 
                `${this.sampleInfo.sampleId}_${this.sampleType}_analysis.json`, 'application/json');
            
            setTimeout(() => {
                this.downloadFile(csvContent, 
                    `${this.sampleInfo.sampleId}_${this.sampleType}_data.csv`, 'text/csv');
            }, 100);
            
            // Download processed image
            if (this.processedCanvas) {
                setTimeout(() => {
                    this.processedCanvas.toBlob((blob) => {
                        this.downloadFile(blob, 
                            `${this.sampleInfo.sampleId}_${this.sampleType}_analyzed_image.png`, 'image/png');
                    });
                }, 200);
            }
            
            this.updateStatus('Results downloaded successfully', 'info');
            
        } catch (error) {
            console.error('Download error:', error);
            this.updateStatus('Failed to download results: ' + error.message, 'error');
        }
    }

    generateSampleCSV() {
        const baseHeaders = [
            'Sample_ID', 'Sample_Number', 'Batch_ID', 'Sample_Type', 'Species', 'Analysis_Date',
            'Length_mm', 'Width_mm', 'Area_mm2', 'Perimeter_mm', 'Aspect_Ratio', 'Circularity',
            'Mean_R', 'Mean_G', 'Mean_B', 'Hue', 'Saturation', 'Value',
            'Center_X', 'Center_Y', 'Pixels_Per_MM', 'Area_Pixels'
        ];
        
        const specificHeaders = this.sampleType === 'leaves' ? 
            ['Green_Index'] : 
            ['Equivalent_Diameter_mm', 'Brown_Index'];
        
        const headers = [...baseHeaders.slice(0, 18), ...specificHeaders, ...baseHeaders.slice(18)];
        const rows = [headers.join(',')];
        
        this.analysisResults.forEach(result => {
            const baseValues = [
                result.sampleId,
                result.sampleNumber,
                result.batchId,
                result.sampleType,
                `"${result.species || ''}"`,
                result.analysisDate,
                result.length,
                result.width,
                result.area,
                result.perimeter,
                result.aspectRatio,
                result.circularity,
                result.meanR,
                result.meanG,
                result.meanB,
                result.hue,
                result.saturation,
                result.value
            ];
            
            const specificValues = this.sampleType === 'leaves' ? 
                [result.greenIndex] : 
                [result.equivalentDiameter, result.brownIndex];
            
            const endValues = [
                result.centerX,
                result.centerY,
                result.pixelsPerMM,
                result.areaPixels
            ];
            
            const row = [...baseValues, ...specificValues, ...endValues];
            rows.push(row.join(','));
        });
        
        return rows.join('\n');
    }

    downloadFile(content, filename, mimeType) {
        const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    updateStatus(message, type = 'info') {
        const statusBar = document.getElementById('statusBar');
        const statusMessage = document.getElementById('statusMessage');
        
        if (!statusBar || !statusMessage) return;
        
        statusBar.className = `status-bar ${type}`;
        
        if (type === 'processing') {
            statusMessage.innerHTML = `<div class="spinner"></div>${message}`;
        } else {
            statusMessage.textContent = message;
        }
        
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
}

// Initialize the application robustly in SPA/React environments
(function initSampleApp() {
    if (window.sampleApp) return;
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            if (!window.sampleApp) window.sampleApp = new BiologicalSampleAnalysisApp();
        });
    } else {
        window.sampleApp = new BiologicalSampleAnalysisApp();
    }
})();

// Export for potential external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BiologicalSampleAnalysisApp;
}
