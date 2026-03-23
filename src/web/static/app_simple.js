// Simplified Plankton Classifier - Core Functionality Only
class PlanktonClassifier {
    constructor() {
        this.currentImage = null;
        this.predictions = [];
        this.chart = null;

        this.initializeElements();
        this.setupEventListeners();
        this.loadModelInfo();
        this.initializeTheme();
    }

    initializeElements() {
        // Theme
        this.themeToggle = document.getElementById('theme-toggle');
        this.htmlElement = document.getElementById('root');

        // Upload
        this.fileInput = document.getElementById('file-input');
        this.uploadBtn = document.getElementById('upload-image-btn');
        this.uploadArea = document.getElementById('file-upload-area');
        this.imageUrl = document.getElementById('image-url');

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
        this.resultImage = document.getElementById('result-image');
        this.chartCanvas = document.getElementById('predictions-chart');

        // Info
        this.modelClasses = document.getElementById('model-classes');
        this.modelStatus = document.getElementById('model-status');
    }

    setupEventListeners() {
        // Theme toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        // File upload
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        document.getElementById('upload-btn').addEventListener('click', () => this.fileInput.click());
        document.getElementById('new-analysis-btn').addEventListener('click', () => this.fileInput.click());

        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // URL input
        this.imageUrl.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleUrlInput();
        });
        this.imageUrl.addEventListener('blur', () => this.handleUrlInput());

        // Actions
        this.removeImageBtn.addEventListener('click', () => this.resetToUpload());
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
    }

    // Theme Management
    initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        this.setTheme(savedTheme);
    }

    toggleTheme() {
        const currentTheme = this.htmlElement.classList.contains('dark') ? 'dark' : 'light';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    }

    setTheme(theme) {
        this.htmlElement.classList.toggle('dark', theme === 'dark');
        localStorage.setItem('theme', theme);

        // Update theme toggle icon
        const icon = this.themeToggle.querySelector('.material-symbols-outlined');
        icon.textContent = theme === 'dark' ? 'light_mode' : 'dark_mode';

        if (this.chart) {
            this.updateChartTheme();
        }
    }

    // File Handling
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.loadImageFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        this.uploadArea.style.borderColor = '#06b6d4';
        this.uploadArea.style.backgroundColor = 'rgba(6, 182, 212, 0.1)';
    }

    handleDragLeave(event) {
        event.preventDefault();
        this.uploadArea.style.borderColor = '';
        this.uploadArea.style.backgroundColor = '';
    }

    handleDrop(event) {
        event.preventDefault();
        this.uploadArea.style.borderColor = '';
        this.uploadArea.style.backgroundColor = '';

        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            this.loadImageFile(file);
        }
    }

    handleUrlInput() {
        const url = this.imageUrl.value.trim();
        if (url && this.isValidUrl(url)) {
            this.loadImageFromUrl(url);
        }
    }

    isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }

    loadImageFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImage = {
                type: 'file',
                data: file,
                src: e.target.result
            };
            this.showImagePreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    loadImageFromUrl(url) {
        this.currentImage = {
            type: 'url',
            data: url,
            src: url
        };
        this.showImagePreview(url);
    }

    showImagePreview(src) {
        this.previewImage.src = src;
        this.uploadState.classList.add('hidden');
        this.previewState.classList.remove('hidden');
        this.imageUrl.value = '';
    }

    resetToUpload() {
        this.currentImage = null;
        this.previewState.classList.add('hidden');
        this.uploadState.classList.remove('hidden');
        this.analysisResults.classList.add('hidden');
        this.analysisDefault.classList.remove('hidden');
        this.fileInput.value = '';
        this.imageUrl.value = '';
    }

    // Analysis
    async analyzeImage() {
        if (!this.currentImage) return;

        this.showLoading();

        try {
            const predictions = await this.sendPredictionRequest();
            this.showResults(predictions);
        } catch (error) {
            this.showError('Analysis failed: ' + error.message);
        }
    }

    showLoading() {
        this.previewState.classList.add('hidden');
        this.loadingState.classList.remove('hidden');
    }

    async sendPredictionRequest() {
        let response;

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
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || 'Prediction failed');
        }

        return result.predictions;
    }

    showResults(predictions) {
        this.predictions = predictions;

        // Hide loading
        this.loadingState.classList.add('hidden');

        // Show results
        this.analysisResults.classList.remove('hidden');
        this.analysisDefault.classList.add('hidden');

        // Update display
        const topPrediction = predictions[0];
        const speciesName = this.formatSpeciesName(topPrediction.species);
        const confidence = topPrediction.confidence;

        this.resultImage.src = this.currentImage.src;
        this.topSpecies.textContent = speciesName;
        this.topConfidence.textContent = confidence.toFixed(1) + '%';

        // Update confidence status
        this.updateConfidenceStatus(confidence);

        // Create chart
        this.createChart(predictions);
    }

    updateConfidenceStatus(confidence) {
        const statusElement = this.confidenceStatus;
        statusElement.className = 'inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold';

        if (confidence >= 70) {
            statusElement.classList.add('bg-green-100', 'dark:bg-green-900', 'text-green-800', 'dark:text-green-200');
            statusElement.innerHTML = '<span class="w-2 h-2 rounded-full bg-green-500"></span><span>High Confidence</span>';
        } else if (confidence >= 50) {
            statusElement.classList.add('bg-yellow-100', 'dark:bg-yellow-900', 'text-yellow-800', 'dark:text-yellow-200');
            statusElement.innerHTML = '<span class="w-2 h-2 rounded-full bg-yellow-500"></span><span>Medium Confidence</span>';
        } else {
            statusElement.classList.add('bg-red-100', 'dark:bg-red-900', 'text-red-800', 'dark:text-red-200');
            statusElement.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-500"></span><span>Low Confidence</span>';
        }
    }

    createChart(predictions) {
        const ctx = this.chartCanvas.getContext('2d');

        if (this.chart) {
            this.chart.destroy();
        }

        const isDark = this.htmlElement.classList.contains('dark');
        const textColor = isDark ? '#e5e7eb' : '#374151';
        const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

        const labels = predictions.map(p => this.formatSpeciesName(p.species));
        const confidences = predictions.map(p => p.confidence);
        const colors = ['#06b6d4', '#3b82f6', '#8b5cf6', '#a855f7', '#c084fc'];

        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: confidences,
                    backgroundColor: colors,
                    borderColor: colors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: isDark ? '#1f2937' : '#ffffff',
                        titleColor: textColor,
                        bodyColor: textColor,
                        borderColor: gridColor,
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: gridColor },
                        ticks: {
                            color: textColor,
                            callback: (value) => value + '%'
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: textColor }
                    }
                }
            }
        });
    }

    updateChartTheme() {
        if (!this.chart) return;

        const isDark = this.htmlElement.classList.contains('dark');
        const textColor = isDark ? '#e5e7eb' : '#374151';
        const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

        this.chart.options.scales.x.ticks.color = textColor;
        this.chart.options.scales.y.ticks.color = textColor;
        this.chart.options.scales.x.grid.color = gridColor;

        this.chart.update('none');
    }

    formatSpeciesName(species) {
        return species.replace(/_/g, ' ')
                     .split(' ')
                     .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                     .join(' ');
    }

    showError(message) {
        this.loadingState.classList.add('hidden');

        const errorHTML = `
            <div class="text-center space-y-4 p-8">
                <div class="w-16 h-16 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center mx-auto">
                    <span class="material-symbols-rounded text-red-600 text-2xl">error</span>
                </div>
                <h3 class="text-lg font-bold text-red-600 dark:text-red-400">Analysis Failed</h3>
                <p class="text-slate-600 dark:text-slate-400">${message}</p>
                <button onclick="planktonApp.resetToUpload()" class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                    Try Again
                </button>
            </div>
        `;

        this.uploadArea.innerHTML = errorHTML;
    }

    async loadModelInfo() {
        try {
            const response = await fetch('/api/model-info');
            const info = await response.json();

            this.modelClasses.textContent = info.classes;
            this.modelStatus.textContent = info.available ? 'Ready' : 'Unavailable';
            this.modelStatus.className = info.available
                ? 'font-bold text-green-500'
                : 'font-bold text-red-500';
        } catch (error) {
            console.error('Failed to load model info:', error);
            this.modelStatus.textContent = 'Error';
            this.modelStatus.className = 'font-bold text-red-500';
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.planktonApp = new PlanktonClassifier();
});