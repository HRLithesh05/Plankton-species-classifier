// Enhanced Plankton Classifier - Fixed Version with Proper Error Handling
class EnhancedPlanktonClassifierFixed {
    constructor() {
        // Single image properties
        this.currentImage = null;
        this.predictions = [];
        this.chart = null;

        // Batch processing properties
        this.batchFiles = [];
        this.batchResultsData = [];  // Changed name to avoid conflict

        // Species database
        this.speciesData = null;

        // Model information
        this.modelInfo = null;

        // Export data
        this.exportData = [];

        // UI state
        this.currentTab = 'single';

        this.initializeElements();
        this.setupEventListeners();
        this.loadModelInfo();
        this.initializeTheme();
        this.loadSpeciesDatabase();
    }

    initializeElements() {
        // Tab elements
        this.tabButtons = document.querySelectorAll('.tab');
        this.tabContents = document.querySelectorAll('.tab-content');

        // Single image elements
        this.themeToggle = document.getElementById('theme-toggle');
        this.htmlElement = document.getElementById('root');
        this.fileInput = document.getElementById('file-input');
        this.uploadBtn = document.getElementById('upload-image-btn');
        this.uploadArea = document.getElementById('file-upload-area');
        this.imageUrl = document.getElementById('image-url');
        this.loadUrlBtn = document.getElementById('load-url-btn');

        // States
        this.uploadState = document.getElementById('upload-state');
        this.previewState = document.getElementById('image-preview-state');
        this.loadingState = document.getElementById('loading-state');

        // Preview
        this.previewImage = document.getElementById('preview-image');
        this.removeImageBtn = document.getElementById('remove-image');
        this.analyzeBtn = document.getElementById('analyze-btn');

        // Results
        this.analysisDefault = document.getElementById('analysis-default');
        this.analysisResults = document.getElementById('analysis-results');
        this.topSpecies = document.getElementById('top-species');
        this.topConfidence = document.getElementById('top-confidence');
        this.confidenceStatus = document.getElementById('confidence-status');
        this.resultImageSmall = document.getElementById('result-image-small');
        this.chartCanvas = document.getElementById('predictions-chart');
        this.speciesInfo = document.getElementById('species-info');
        this.speciesDetails = document.getElementById('species-details');

        // Details
        this.toggleDetails = document.getElementById('toggle-details');
        this.toggleIcon = document.getElementById('toggle-icon');
        this.detailedResults = document.getElementById('detailed-results');

        // Batch elements
        this.batchFileInput = document.getElementById('batch-file-input');
        this.batchUploadBtn = document.getElementById('batch-upload-btn');
        this.batchUploadArea = document.getElementById('batch-upload-area');
        this.batchPreview = document.getElementById('batch-preview');
        this.batchImages = document.getElementById('batch-images');
        this.batchClear = document.getElementById('batch-clear');
        this.batchAnalyze = document.getElementById('batch-analyze');
        this.batchProcessing = document.getElementById('batch-processing');
        this.batchProgress = document.getElementById('batch-progress');
        this.batchStatus = document.getElementById('batch-status');
        this.batchResults = document.getElementById('batch-results');
        this.batchResultsGrid = document.getElementById('batch-results-grid');
        this.batchExportCsv = document.getElementById('batch-export-csv');
        this.batchExportExcel = document.getElementById('batch-export-excel');

        // Species elements
        this.speciesSearch = document.getElementById('species-search');
        this.speciesGrid = document.getElementById('species-grid');
        this.speciesLoading = document.getElementById('species-loading');

        // Model info
        this.modelStatus = document.getElementById('model-status');
    }

    setupEventListeners() {
        // Tab switching
        this.tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const tabId = e.currentTarget.id.replace('tab-', '');
                this.switchTab(tabId);
            });
        });

        // Theme toggle
        this.themeToggle?.addEventListener('click', () => this.toggleTheme());

        // Single image upload
        this.uploadBtn?.addEventListener('click', () => this.fileInput.click());
        this.fileInput?.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
        this.loadUrlBtn?.addEventListener('click', () => this.loadImageFromUrl());
        this.removeImageBtn?.addEventListener('click', () => this.resetToUploadState());
        this.analyzeBtn?.addEventListener('click', () => this.analyzeImage());

        // Drag and drop for single image
        this.setupDragDrop(this.uploadArea, (files) => {
            if (files.length > 0) this.handleFileSelect(files[0]);
        });

        // Results toggle
        this.toggleDetails?.addEventListener('click', () => this.toggleDetailedResults());

        // Batch processing
        this.batchUploadBtn?.addEventListener('click', () => this.batchFileInput.click());
        this.batchFileInput?.addEventListener('change', (e) => this.handleBatchFileSelect(e.target.files));
        this.batchClear?.addEventListener('click', () => this.clearBatchFiles());
        this.batchAnalyze?.addEventListener('click', () => this.processBatchImages());
        this.batchExportCsv?.addEventListener('click', () => this.exportBatchResults('csv'));
        this.batchExportExcel?.addEventListener('click', () => this.exportBatchResults('excel'));

        // Drag and drop for batch
        this.setupDragDrop(this.batchUploadArea, (files) => {
            this.handleBatchFileSelect(files);
        });

        // Species search
        this.speciesSearch?.addEventListener('input', (e) => this.searchSpecies(e.target.value));

        // URL input enter key
        this.imageUrl?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.loadImageFromUrl();
        });

        // Model info button
        const modelInfoBtn = document.getElementById('modelInfoBtn');
        modelInfoBtn?.addEventListener('click', () => this.showModelInfoModal());
    }

    switchTab(tabId) {
        // Update tab buttons
        this.tabButtons.forEach(button => {
            button.classList.remove('tab-active');
            button.classList.add('text-gray-600', 'dark:text-gray-300', 'hover:bg-gray-100', 'dark:hover:bg-gray-700');
        });

        const activeButton = document.getElementById(`tab-${tabId}`);
        if (activeButton) {
            activeButton.classList.add('tab-active');
            activeButton.classList.remove('text-gray-600', 'dark:text-gray-300', 'hover:bg-gray-100', 'dark:hover:bg-gray-700');
        }

        // Update content
        this.tabContents.forEach(content => {
            content.classList.add('hidden');
        });

        const activeContent = document.getElementById(`content-${tabId}`);
        if (activeContent) {
            activeContent.classList.remove('hidden');
        }

        this.currentTab = tabId;

        // Load tab-specific data
        if (tabId === 'species' && !this.speciesData) {
            this.loadSpeciesDatabase();
        }
    }

    setupDragDrop(element, callback) {
        if (!element) return;

        element.addEventListener('dragover', (e) => {
            e.preventDefault();
            element.classList.add('dragover');
        });

        element.addEventListener('dragleave', (e) => {
            e.preventDefault();
            element.classList.remove('dragover');
        });

        element.addEventListener('drop', (e) => {
            e.preventDefault();
            element.classList.remove('dragover');
            const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/'));
            if (files.length > 0) {
                callback(files);
            }
        });
    }

    // Single Image Functions
    handleFileSelect(file) {
        if (!file) return;

        this.currentImage = {
            type: 'file',
            data: file
        };

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.showPreviewState();
        };
        reader.readAsDataURL(file);
    }

    async loadImageFromUrl() {
        const url = this.imageUrl.value.trim();
        if (!url) return;

        this.currentImage = {
            type: 'url',
            data: url
        };

        // Show preview
        this.previewImage.src = url;
        this.showPreviewState();
    }

    showPreviewState() {
        this.uploadState.classList.add('hidden');
        this.previewState.classList.remove('hidden');
        this.analysisDefault.classList.remove('hidden');
        this.analysisResults.classList.add('hidden');
    }

    resetToUploadState() {
        this.uploadState.classList.remove('hidden');
        this.previewState.classList.add('hidden');
        this.loadingState.classList.add('hidden');
        this.analysisDefault.classList.remove('hidden');
        this.analysisResults.classList.add('hidden');
        this.currentImage = null;
        this.imageUrl.value = '';
    }

    async analyzeImage() {
        if (!this.currentImage) return;

        // Show loading state
        this.previewState.classList.add('hidden');
        this.loadingState.classList.remove('hidden');

        const progressText = document.getElementById('progress-text');
        progressText.textContent = 'Uploading and analyzing image...';

        try {
            const predictions = await this.makePrediction();

            // Store result for export
            this.exportData.push({
                type: 'single',
                filename: this.currentImage.type === 'file' ? this.currentImage.data.name : 'url_image',
                timestamp: new Date().toISOString(),
                predictions: predictions
            });

            this.showResults(predictions);

        } catch (error) {
            console.error('Analyze error:', error);
            this.showError(error.message);
        }
    }

    async makePrediction() {
        let response;

        try {
            if (this.currentImage.type === 'file') {
                const formData = new FormData();
                formData.append('file', this.currentImage.data);

                response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
            } else if (this.currentImage.type === 'url') {
                response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: this.currentImage.data })
                });
            }

            if (!response.ok) {
                let errorMessage = `Server error: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    // If can't parse JSON, use status message
                }
                throw new Error(errorMessage);
            }

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Prediction failed');
            }

            return result.predictions;

        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Network error: Unable to connect to the server. Please check if the server is running.');
            }
            throw error;
        }
    }

    showResults(predictions) {
        this.predictions = predictions;

        // Hide loading but keep the preview visible
        this.loadingState.classList.add('hidden');
        this.previewState.classList.remove('hidden');

        // Show results
        this.analysisDefault.classList.add('hidden');
        this.analysisResults.classList.remove('hidden');

        // Update result image
        this.resultImageSmall.src = this.previewImage.src;

        // Top prediction
        const topPrediction = predictions[0];
        this.topSpecies.textContent = topPrediction.species.replace(/_/g, ' ');
        this.topConfidence.textContent = `${topPrediction.confidence.toFixed(1)}%`;

        // Confidence status
        const confidence = topPrediction.confidence;
        if (confidence >= 70) {
            this.confidenceStatus.textContent = 'High Confidence';
            this.confidenceStatus.className = 'px-4 py-2 bg-green-500 bg-opacity-20 rounded-full text-sm font-semibold text-green-300';
        } else if (confidence >= 50) {
            this.confidenceStatus.textContent = 'Medium Confidence';
            this.confidenceStatus.className = 'px-4 py-2 bg-yellow-500 bg-opacity-20 rounded-full text-sm font-semibold text-yellow-300';
        } else {
            this.confidenceStatus.textContent = 'Low Confidence';
            this.confidenceStatus.className = 'px-4 py-2 bg-red-500 bg-opacity-20 rounded-full text-sm font-semibold text-red-300';
        }

        // Load species information
        this.loadSpeciesInfo(topPrediction.species);

        // Build chart
        this.buildChart(predictions);

        // Build detailed results
        this.buildDetailedResults(predictions);
    }

    async loadSpeciesInfo(speciesName) {
        try {
            const response = await fetch(`/api/species/${encodeURIComponent(speciesName)}`);
            if (response.ok) {
                const speciesData = await response.json();
                this.displaySpeciesInfo(speciesData);
            } else {
                console.log('Species info not found for:', speciesName);
                this.speciesInfo.classList.add('hidden');
            }
        } catch (error) {
            console.error('Failed to load species info:', error);
            this.speciesInfo.classList.add('hidden');
        }
    }

    displaySpeciesInfo(data) {
        if (!data) return;

        this.speciesDetails.innerHTML = `
            <div class="grid md:grid-cols-2 gap-4">
                <div>
                    <h5 class="font-semibold text-blue-700 dark:text-blue-300 mb-2">Scientific Classification</h5>
                    <p class="text-sm"><strong>Scientific Name:</strong> ${data.scientific_name || 'Unknown'}</p>
                    <p class="text-sm"><strong>Common Name:</strong> ${data.common_name || data.species_id.replace(/_/g, ' ')}</p>
                </div>
                <div>
                    <h5 class="font-semibold text-blue-700 dark:text-blue-300 mb-2">Habitat & Role</h5>
                    <p class="text-sm"><strong>Habitat:</strong> ${data.habitat || 'Marine waters'}</p>
                    <p class="text-sm"><strong>Ecological Role:</strong> ${data.ecological_role || 'Primary producer/consumer'}</p>
                </div>
            </div>
            <div class="mt-4">
                <h5 class="font-semibold text-blue-700 dark:text-blue-300 mb-2">Description</h5>
                <p class="text-sm">${data.description || `${data.common_name || data.species_id} is a plankton species found in marine environments.`}</p>
            </div>
            ${data.characteristics ? `
            <div class="mt-4">
                <h5 class="font-semibold text-blue-700 dark:text-blue-300 mb-2">Characteristics</h5>
                <p class="text-sm">${data.characteristics}</p>
            </div>
            ` : ''}
        `;

        this.speciesInfo.classList.remove('hidden');
    }

    buildChart(predictions) {
        if (this.chart) {
            this.chart.destroy();
        }

        const ctx = this.chartCanvas.getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: predictions.map(p => p.species.replace(/_/g, ' ')),
                datasets: [{
                    data: predictions.map(p => p.confidence),
                    backgroundColor: [
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(139, 92, 246, 0.8)'
                    ],
                    borderColor: [
                        'rgba(59, 130, 246, 1)',
                        'rgba(16, 185, 129, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(239, 68, 68, 1)',
                        'rgba(139, 92, 246, 1)'
                    ],
                    borderWidth: 2,
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: (value) => value + '%'
                        },
                        grid: {
                            color: 'rgba(156, 163, 175, 0.2)'
                        }
                    },
                    x: {
                        grid: { display: false }
                    }
                }
            }
        });
    }

    buildDetailedResults(predictions) {
        this.detailedResults.innerHTML = predictions.map((pred, index) => `
            <div class="flex items-center justify-between p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-600">
                <div class="flex items-center space-x-3">
                    <span class="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-semibold">
                        ${index + 1}
                    </span>
                    <span class="font-medium text-gray-900 dark:text-white">${pred.species.replace(/_/g, ' ')}</span>
                </div>
                <div class="text-right">
                    <div class="font-semibold text-sm">${pred.confidence.toFixed(2)}%</div>
                    <div class="w-24 h-2 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                        <div class="h-full bg-gradient-to-r from-blue-500 to-blue-700 rounded-full"
                             style="width: ${(pred.confidence / predictions[0].confidence) * 100}%"></div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    toggleDetailedResults() {
        const isHidden = this.detailedResults.classList.contains('hidden');

        if (isHidden) {
            this.detailedResults.classList.remove('hidden');
            this.toggleIcon.textContent = 'expand_less';
        } else {
            this.detailedResults.classList.add('hidden');
            this.toggleIcon.textContent = 'expand_more';
        }
    }

    // Batch Processing Functions
    handleBatchFileSelect(files) {
        const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/')).slice(0, 50);

        if (imageFiles.length === 0) {
            alert('No valid image files selected.');
            return;
        }

        this.batchFiles = imageFiles;
        this.displayBatchPreview();
    }

    displayBatchPreview() {
        this.batchImages.innerHTML = '';

        this.batchFiles.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const imageDiv = document.createElement('div');
                imageDiv.className = 'relative';
                imageDiv.innerHTML = `
                    <img src="${e.target.result}" alt="${file.name}" class="w-full h-24 object-cover rounded-lg border border-gray-300">
                    <button class="absolute -top-2 -right-2 w-6 h-6 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center text-xs"
                            onclick="planktonClassifier.removeBatchFile(${index}); this.parentElement.remove();">
                        ×
                    </button>
                    <div class="mt-1 text-xs text-gray-600 dark:text-gray-400 truncate">${file.name}</div>
                `;
                this.batchImages.appendChild(imageDiv);
            };
            reader.readAsDataURL(file);
        });

        this.batchPreview.classList.remove('hidden');
    }

    removeBatchFile(index) {
        this.batchFiles.splice(index, 1);
        if (this.batchFiles.length === 0) {
            this.clearBatchFiles();
        }
    }

    clearBatchFiles() {
        this.batchFiles = [];
        this.batchResultsData = [];
        this.batchPreview.classList.add('hidden');
        this.batchProcessing.classList.add('hidden');
        this.batchResults.classList.add('hidden');
        this.batchFileInput.value = '';
    }

    async processBatchImages() {
        if (this.batchFiles.length === 0) return;

        this.batchResultsData = [];
        this.batchPreview.classList.add('hidden');
        this.batchProcessing.classList.remove('hidden');

        const total = this.batchFiles.length;
        let processed = 0;

        for (let i = 0; i < this.batchFiles.length; i++) {
            const file = this.batchFiles[i];

            this.batchStatus.textContent = `Processing ${processed + 1} of ${total} images... (${file.name})`;

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                let result;
                if (response.ok) {
                    const data = await response.json();
                    result = {
                        filename: file.name,
                        success: true,
                        predictions: data.predictions,
                        timestamp: new Date().toISOString()
                    };
                } else {
                    let errorMessage;
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.error || 'Prediction failed';
                    } catch (e) {
                        errorMessage = `Server error: ${response.status}`;
                    }
                    result = {
                        filename: file.name,
                        success: false,
                        error: errorMessage,
                        timestamp: new Date().toISOString()
                    };
                }

                this.batchResultsData.push(result);

                // Store for export
                if (result.success) {
                    this.exportData.push({
                        type: 'batch',
                        filename: file.name,
                        timestamp: result.timestamp,
                        predictions: result.predictions
                    });
                }

            } catch (error) {
                this.batchResultsData.push({
                    filename: file.name,
                    success: false,
                    error: error.message,
                    timestamp: new Date().toISOString()
                });
            }

            processed++;
            const progress = (processed / total) * 100;
            this.batchProgress.style.width = `${progress}%`;
        }

        this.batchProcessing.classList.add('hidden');
        this.displayBatchResults();
    }

    displayBatchResults() {
        console.log('Displaying batch results:', this.batchResultsData.length); // Debug log

        this.batchResultsGrid.innerHTML = this.batchResultsData.map(result => {
            if (result.success) {
                const topPrediction = result.predictions[0];
                return `
                    <div class="bg-white dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600">
                        <div class="flex items-center justify-between mb-2">
                            <h5 class="font-medium text-sm truncate flex-1">${result.filename}</h5>
                            <span class="text-xs text-green-600 ml-2">✓</span>
                        </div>
                        <div class="text-lg font-semibold text-blue-600 dark:text-blue-400">${topPrediction.species.replace(/_/g, ' ')}</div>
                        <div class="text-sm text-gray-600 dark:text-gray-400">${topPrediction.confidence.toFixed(1)}% confidence</div>
                        <div class="mt-2 w-full h-1 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                            <div class="h-full bg-blue-600 rounded-full" style="width: ${topPrediction.confidence}%"></div>
                        </div>
                    </div>
                `;
            } else {
                return `
                    <div class="bg-white dark:bg-gray-700 rounded-lg p-4 border border-red-200 dark:border-red-600">
                        <div class="flex items-center justify-between mb-2">
                            <h5 class="font-medium text-sm truncate flex-1">${result.filename}</h5>
                            <span class="text-xs text-red-600 ml-2">✗</span>
                        </div>
                        <div class="text-sm text-red-600 dark:text-red-400">Error: ${result.error}</div>
                    </div>
                `;
            }
        }).join('');

        // Make sure to show the results section
        this.batchResults.classList.remove('hidden');
        console.log('Batch results should now be visible'); // Debug log
    }

    exportBatchResults(format) {
        if (this.batchResultsData.length === 0) return;

        const successfulResults = this.batchResultsData.filter(r => r.success);
        if (successfulResults.length === 0) {
            alert('No successful results to export.');
            return;
        }

        if (format === 'csv') {
            // CSV Export
            let csvContent = 'Filename,Species,Confidence\n';
            successfulResults.forEach(result => {
                const topPrediction = result.predictions[0];
                csvContent += `"${result.filename}","${topPrediction.species}",${topPrediction.confidence.toFixed(2)}\n`;
            });

            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `batch_results_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

        } else if (format === 'excel') {
            // Excel Export (XML format that Excel can read)
            let excelContent = `<?xml version="1.0"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:x="urn:schemas-microsoft-com:office:excel"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:html="http://www.w3.org/TR/REC-html40">
<DocumentProperties xmlns="urn:schemas-microsoft-com:office:office">
<Title>Plankton Classification Results</Title>
<Author>Plankton Classifier</Author>
<Created>${new Date().toISOString()}</Created>
</DocumentProperties>
<Worksheet ss:Name="Batch Results">
<Table>
<Row>
<Cell><Data ss:Type="String">Filename</Data></Cell>
<Cell><Data ss:Type="String">Species</Data></Cell>
<Cell><Data ss:Type="String">Confidence (%)</Data></Cell>
<Cell><Data ss:Type="String">Timestamp</Data></Cell>
</Row>`;

            successfulResults.forEach(result => {
                const topPrediction = result.predictions[0];
                const species = topPrediction.species.replace(/_/g, ' ');
                excelContent += `
<Row>
<Cell><Data ss:Type="String">${result.filename}</Data></Cell>
<Cell><Data ss:Type="String">${species}</Data></Cell>
<Cell><Data ss:Type="Number">${topPrediction.confidence.toFixed(2)}</Data></Cell>
<Cell><Data ss:Type="String">${new Date(result.timestamp).toLocaleString()}</Data></Cell>
</Row>`;
            });

            excelContent += `
</Table>
</Worksheet>
</Workbook>`;

            const blob = new Blob([excelContent], {
                type: 'application/vnd.ms-excel'
            });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `batch_results_${new Date().toISOString().split('T')[0]}.xls`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

        } else {
            alert('Supported formats: CSV and Excel');
        }
    }

    // Species Database Functions
    async loadSpeciesDatabase() {
        if (this.speciesData) {
            this.displaySpeciesGrid();
            return;
        }

        if (this.currentTab !== 'species') return;

        this.speciesLoading.classList.remove('hidden');
        this.speciesGrid.classList.add('hidden');

        try {
            const response = await fetch('/api/species');
            if (response.ok) {
                const data = await response.json();
                this.speciesData = {};

                // Convert array format to object format for easier handling
                if (data.species && Array.isArray(data.species)) {
                    data.species.forEach(species => {
                        this.speciesData[species.species_id] = species;
                    });
                } else {
                    this.speciesData = data;
                }

                this.displaySpeciesGrid();
            } else {
                throw new Error('Failed to load species database');
            }
        } catch (error) {
            console.error('Failed to load species data:', error);
            this.speciesGrid.innerHTML = '<div class="col-span-full text-center text-red-600 p-8">Failed to load species database. Please check server connection.</div>';
        }

        this.speciesLoading.classList.add('hidden');
        this.speciesGrid.classList.remove('hidden');
    }

    displaySpeciesGrid(filteredData) {
        const data = filteredData || this.speciesData;
        if (!data) return;

        const entries = Object.entries(data);

        // Update species count
        const speciesCount = document.getElementById('species-count');
        if (speciesCount) {
            speciesCount.textContent = `${entries.length}`;
        }

        if (entries.length === 0) {
            this.speciesGrid.innerHTML = '<div class="col-span-full text-center text-gray-500 p-8">No species found.</div>';
            return;
        }

        this.speciesGrid.innerHTML = entries.map(([key, species]) => `
            <div class="bg-white dark:bg-gray-700 rounded-lg p-6 border border-gray-200 dark:border-gray-600 hover:shadow-lg transition-shadow">
                <div class="mb-4">
                    <h4 class="text-lg font-semibold text-gray-900 dark:text-white">${species.common_name || key.replace(/_/g, ' ')}</h4>
                    <p class="text-sm italic text-gray-600 dark:text-gray-400">${species.scientific_name || 'Scientific name unknown'}</p>
                </div>
                <p class="text-sm text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">${species.description || `${species.common_name || key} is a plankton species found in marine environments.`}</p>
                <div class="space-y-2 text-xs">
                    <div><strong>Habitat:</strong> ${species.habitat || 'Marine waters'}</div>
                    <div><strong>Ecological Role:</strong> ${species.ecological_role || 'Primary producer/consumer'}</div>
                </div>
            </div>
        `).join('');
    }

    searchSpecies(query) {
        if (!this.speciesData) return;

        if (!query.trim()) {
            this.displaySpeciesGrid();
            return;
        }

        const filtered = {};
        const searchTerm = query.toLowerCase();

        Object.entries(this.speciesData).forEach(([key, species]) => {
            const searchableText = [
                key,
                species.common_name || '',
                species.scientific_name || '',
                species.description || '',
                species.habitat || '',
                species.ecological_role || ''
            ].join(' ').toLowerCase();

            if (searchableText.includes(searchTerm)) {
                filtered[key] = species;
            }
        });

        this.displaySpeciesGrid(filtered);
    }

    // Utility Functions
    showError(message) {
        this.loadingState.classList.add('hidden');
        this.previewState.classList.remove('hidden');

        // Show error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mt-4';
        errorDiv.innerHTML = `
            <div class="flex">
                <span class="material-symbols-outlined text-red-600 mr-3">error</span>
                <div>
                    <h4 class="text-red-800 dark:text-red-200 font-medium">Analysis Failed</h4>
                    <p class="text-red-700 dark:text-red-300 text-sm mt-1">${message}</p>
                </div>
            </div>
        `;

        // Remove existing error messages
        const existingErrors = document.querySelectorAll('.bg-red-50, .dark\\:bg-red-900\\/20');
        existingErrors.forEach(el => el.remove());

        // Add new error message
        this.previewState.appendChild(errorDiv);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 10000);
    }

    async loadModelInfo() {
        try {
            const response = await fetch('/api/model-info');
            if (response.ok) {
                this.modelInfo = await response.json();

                // Update all model info fields
                const modelArchitecture = document.getElementById('model-architecture');
                const modelClasses = document.getElementById('model-classes');
                const modelAccuracy = document.getElementById('model-accuracy');

                if (modelArchitecture) {
                    modelArchitecture.textContent = this.modelInfo.architecture || 'EfficientNet-B2';
                }

                if (modelClasses) {
                    modelClasses.textContent = `${this.modelInfo.classes || 67} species`;
                }

                if (modelAccuracy) {
                    const accuracy = this.modelInfo.accuracy?.validation || 89.51;
                    modelAccuracy.textContent = `${accuracy}%`;
                }

                if (this.modelInfo.status === 'ready') {
                    this.modelStatus.textContent = 'Ready';
                    this.modelStatus.className = 'font-semibold text-green-600';
                } else {
                    this.modelStatus.textContent = 'Loading...';
                    this.modelStatus.className = 'font-semibold text-yellow-600';
                }
            } else {
                this.modelStatus.textContent = 'Error';
                this.modelStatus.className = 'font-semibold text-red-600';
            }
        } catch (error) {
            console.error('Failed to load model info:', error);
            this.modelStatus.textContent = 'Offline';
            this.modelStatus.className = 'font-semibold text-red-600';
        }
    }

    showModelInfoModal() {
        if (!this.modelInfo) {
            alert('Model information is not available.');
            return;
        }

        // Create modal overlay
        const modalOverlay = document.createElement('div');
        modalOverlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modalOverlay.onclick = (e) => {
            if (e.target === modalOverlay) {
                document.body.removeChild(modalOverlay);
            }
        };

        // Create modal content
        const modalContent = document.createElement('div');
        modalContent.className = 'bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto';

        modalContent.innerHTML = `
            <div class="p-6">
                <div class="flex items-center justify-between mb-6">
                    <h3 class="text-2xl font-bold text-gray-900 dark:text-white flex items-center">
                        <span class="material-symbols-outlined mr-3 text-blue-600">biotech</span>
                        Plankton Species Classifier - Project Overview
                    </h3>
                    <button onclick="document.body.removeChild(this.closest('.fixed'))" class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                        <span class="material-symbols-outlined">close</span>
                    </button>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div class="space-y-6">
                        <!-- Project Overview -->
                        <div>
                            <h4 class="text-lg font-semibold text-blue-600 dark:text-blue-400 mb-3 flex items-center">
                                <span class="material-symbols-outlined mr-2">info</span>
                                Project Overview
                            </h4>
                            <div class="space-y-3 text-sm">
                                <p class="text-gray-600 dark:text-gray-300">
                                    AI-powered marine biology research platform for automated plankton species identification from microscopic images.
                                </p>
                                <div class="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                                    <p><strong>Purpose:</strong> Scientific research and marine ecosystem analysis</p>
                                    <p><strong>Application:</strong> Marine biology, oceanography, environmental monitoring</p>
                                    <p><strong>Users:</strong> Researchers, marine biologists, environmental scientists</p>
                                </div>
                            </div>
                        </div>

                        <!-- Model Performance -->
                        <div>
                            <h4 class="text-lg font-semibold text-green-600 dark:text-green-400 mb-3 flex items-center">
                                <span class="material-symbols-outlined mr-2">analytics</span>
                                AI Model Performance
                            </h4>
                            <div class="space-y-2 text-sm">
                                <div class="flex justify-between">
                                    <span>Model Architecture:</span>
                                    <span class="font-semibold">${this.modelInfo.architecture || 'EfficientNet-B2'}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Validation Accuracy:</span>
                                    <span class="font-semibold text-green-600">${this.modelInfo.accuracy?.validation || 89.51}%</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Species Classes:</span>
                                    <span class="font-semibold">${this.modelInfo.classes || 67} species</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Training Method:</span>
                                    <span class="font-semibold">Progressive 3-Stage</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Dataset Size:</span>
                                    <span class="font-semibold">20,644+ images</span>
                                </div>
                            </div>
                        </div>

                        <!-- Dataset Information -->
                        <div>
                            <h4 class="text-lg font-semibold text-purple-600 dark:text-purple-400 mb-3 flex items-center">
                                <span class="material-symbols-outlined mr-2">database</span>
                                Dataset & Training
                            </h4>
                            <div class="space-y-2 text-sm">
                                <div><strong>Source:</strong> WHOI Plankton 2014 Dataset (Woods Hole Oceanographic Institution)</div>
                                <div><strong>Image Quality:</strong> High-resolution microscopic images</div>
                                <div><strong>Species Coverage:</strong> 67 different plankton species</div>
                                <div><strong>Training Stages:</strong>
                                    <ul class="ml-4 mt-1 list-disc">
                                        <li>Stage 1: Foundation training (224px)</li>
                                        <li>Stage 2: Refinement training (288px)</li>
                                        <li>Stage 3: Fine-tuning (384px)</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="space-y-6">
                        <!-- Platform Features -->
                        <div>
                            <h4 class="text-lg font-semibold text-orange-600 dark:text-orange-400 mb-3 flex items-center">
                                <span class="material-symbols-outlined mr-2">features</span>
                                Platform Capabilities
                            </h4>
                            <div class="space-y-2 text-sm">
                                <div class="flex items-start">
                                    <span class="material-symbols-outlined text-green-500 mr-2 mt-0.5 text-base">check_circle</span>
                                    <div>
                                        <strong>Single Image Analysis:</strong> Upload and classify individual plankton images with confidence scores
                                    </div>
                                </div>
                                <div class="flex items-start">
                                    <span class="material-symbols-outlined text-green-500 mr-2 mt-0.5 text-base">check_circle</span>
                                    <div>
                                        <strong>Batch Processing:</strong> Analyze multiple images simultaneously for research efficiency
                                    </div>
                                </div>
                                <div class="flex items-start">
                                    <span class="material-symbols-outlined text-green-500 mr-2 mt-0.5 text-base">check_circle</span>
                                    <div>
                                        <strong>Species Database:</strong> Comprehensive information on 68 plankton species
                                    </div>
                                </div>
                                <div class="flex items-start">
                                    <span class="material-symbols-outlined text-green-500 mr-2 mt-0.5 text-base">check_circle</span>
                                    <div>
                                        <strong>Export Results:</strong> Download analysis results in CSV format for further research
                                    </div>
                                </div>
                                <div class="flex items-start">
                                    <span class="material-symbols-outlined text-green-500 mr-2 mt-0.5 text-base">check_circle</span>
                                    <div>
                                        <strong>Real-time Analysis:</strong> Instant species identification and confidence assessment
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Technical Specifications -->
                        <div>
                            <h4 class="text-lg font-semibold text-red-600 dark:text-red-400 mb-3 flex items-center">
                                <span class="material-symbols-outlined mr-2">settings</span>
                                Technical Details
                            </h4>
                            <div class="space-y-2 text-sm">
                                <div><strong>Framework:</strong> PyTorch with torchvision</div>
                                <div><strong>Backend:</strong> Flask Python web application</div>
                                <div><strong>Frontend:</strong> Modern responsive web interface</div>
                                <div><strong>Image Processing:</strong> Advanced preprocessing and augmentation</div>
                                <div><strong>Model Size:</strong> ~9M parameters (optimized for efficiency)</div>
                                <div><strong>Input Format:</strong> JPG, JPEG, PNG, BMP images</div>
                                <div><strong>Processing Speed:</strong> Near real-time classification</div>
                            </div>
                        </div>

                        <!-- Usage Guidelines -->
                        <div>
                            <h4 class="text-lg font-semibold text-cyan-600 dark:text-cyan-400 mb-3 flex items-center">
                                <span class="material-symbols-outlined mr-2">help</span>
                                Usage Guidelines
                            </h4>
                            <div class="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                                <div>• Upload clear, high-quality microscopic images</div>
                                <div>• Best results with well-focused plankton specimens</div>
                                <div>• Confidence scores >70% indicate high reliability</div>
                                <div>• Use batch processing for large-scale studies</div>
                                <div>• Consult species database for detailed information</div>
                                <div>• Export results for statistical analysis and reporting</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="mt-8 pt-6 border-t border-gray-200 dark:border-gray-600">
                    <div class="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 p-4 rounded-lg">
                        <div class="flex items-start">
                            <span class="material-symbols-outlined text-blue-600 mr-3 mt-1">science</span>
                            <div class="text-sm">
                                <h5 class="font-semibold text-gray-900 dark:text-white mb-2">Research Impact</h5>
                                <p class="text-gray-700 dark:text-gray-300">
                                    This AI-powered platform accelerates marine biological research by automating plankton species identification,
                                    enabling researchers to process large image datasets efficiently and focus on ecological analysis and
                                    environmental monitoring studies.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        modalOverlay.appendChild(modalContent);
        document.body.appendChild(modalOverlay);
    }

    initializeTheme() {
        // Set initial theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            this.htmlElement.classList.remove('dark');
        } else {
            this.htmlElement.classList.add('dark');
        }
        this.updateThemeIcon();
    }

    toggleTheme() {
        this.htmlElement.classList.toggle('dark');
        const isDark = this.htmlElement.classList.contains('dark');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        this.updateThemeIcon();
    }

    updateThemeIcon() {
        const icon = this.themeToggle.querySelector('.material-symbols-outlined');
        const isDark = this.htmlElement.classList.contains('dark');
        icon.textContent = isDark ? 'dark_mode' : 'light_mode';
    }
}

// Initialize the application
let planktonClassifier;
document.addEventListener('DOMContentLoaded', () => {
    planktonClassifier = new EnhancedPlanktonClassifierFixed();
});

// Make it globally accessible for the remove buttons
window.planktonClassifier = planktonClassifier;