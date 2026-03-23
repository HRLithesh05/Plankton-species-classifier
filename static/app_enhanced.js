// Enhanced Plankton Classifier - Professional UI with Full Functionality
class EnhancedPlanktonClassifier {
    constructor() {
        this.currentImage = null;
        this.predictions = [];
        this.chart = null;
        this.fileDialogOpen = false; // Prevent multiple file dialogs

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
        this.resultImageSmall = document.getElementById('result-image-small');
        this.chartCanvas = document.getElementById('predictions-chart');

        // Details
        this.toggleDetails = document.getElementById('toggle-details');
        this.toggleIcon = document.getElementById('toggle-icon');
        this.detailedResults = document.getElementById('detailed-results');

        // Info
        this.modelClasses = document.getElementById('model-classes');
        this.modelStatus = document.getElementById('model-status');
    }

    // Utility method to open file dialog with debounce
    openFileDialog() {
        if (this.fileDialogOpen) return; // Prevent multiple dialogs

        this.fileDialogOpen = true;
        this.fileInput.click();

        // Reset flag after short delay
        setTimeout(() => {
            this.fileDialogOpen = false;
        }, 500);
    }

    setupEventListeners() {
        // Theme toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        // File upload - prevent duplicate triggers
        this.uploadBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent bubbling to uploadArea
            this.openFileDialog();
        });
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop - only trigger on area, not on button
        this.uploadArea.addEventListener('click', (e) => {
            // Only open file dialog if clicking on the area itself, not on the button
            if (e.target === this.uploadArea || (!this.uploadBtn.contains(e.target) && !e.target.closest('button'))) {
                this.openFileDialog();
            }
        });
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // URL input
        this.imageUrl.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.target.value.trim()) {
                this.handleUrlInput();
            }
        });
        this.imageUrl.addEventListener('blur', (e) => {
            if (e.target.value.trim()) {
                this.handleUrlInput();
            }
        });

        // Actions
        this.removeImageBtn.addEventListener('click', () => this.resetToUpload());
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());

        // Details toggle
        this.toggleDetails.addEventListener('click', () => this.toggleDetailedResults());
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
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        if (!this.uploadArea.contains(event.relatedTarget)) {
            this.uploadArea.classList.remove('dragover');
        }
    }

    handleDrop(event) {
        event.preventDefault();
        this.uploadArea.classList.remove('dragover');

        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            this.loadImageFile(file);
        } else {
            this.showToast('Please drop a valid image file', 'error');
        }
    }

    handleUrlInput() {
        const url = this.imageUrl.value.trim();
        if (url && this.isValidUrl(url)) {
            // Show loading state for URL processing
            this.showToast('Loading image from URL...', 'info');
            this.loadImageFromUrl(url);
        } else if (url) {
            this.showToast('Please enter a valid image URL', 'error');
        }
    }

    isValidUrl(string) {
        try {
            const url = new URL(string);
            // Check for common problematic domains
            const problematicDomains = ['shutterstock.com', 'gettyimages.com', 'adobe.com', 'istockphoto.com'];
            const hostname = url.hostname.toLowerCase();

            if (problematicDomains.some(domain => hostname.includes(domain))) {
                this.showToast('Cannot access protected stock photos. Please download and upload the image file instead.', 'error');
                return false;
            }

            return true;
        } catch (_) {
            return false;
        }
    }

    loadImageFile(file) {
        // Validate file size (max 500MB)
        if (file.size > 500 * 1024 * 1024) {
            this.showToast('File size too large. Maximum 500MB allowed.', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImage = {
                type: 'file',
                data: file,
                src: e.target.result,
                name: file.name
            };
            this.showImagePreview(e.target.result);
            // Remove success toast to reduce popup spam
        };
        reader.onerror = () => {
            this.showToast('Failed to read image file', 'error');
        };
        reader.readAsDataURL(file);
    }

    loadImageFromUrl(url) {
        this.currentImage = {
            type: 'url',
            data: url,
            src: url,
            name: 'URL Image'
        };
        this.showImagePreview(url);
        // Remove URL success toast to reduce popup spam
    }

    showImagePreview(src) {
        this.previewImage.src = src;

        // Smooth transition
        this.uploadState.style.opacity = '0';
        setTimeout(() => {
            this.uploadState.classList.add('hidden');
            this.previewState.classList.remove('hidden');
            this.previewState.style.opacity = '0';
            setTimeout(() => {
                this.previewState.style.opacity = '1';
            }, 50);
        }, 300);

        this.imageUrl.value = '';
    }

    resetToUpload() {
        this.currentImage = null;

        // Smooth transition
        this.previewState.style.opacity = '0';
        setTimeout(() => {
            this.previewState.classList.add('hidden');
            this.analysisResults.classList.add('hidden');
            this.analysisDefault.classList.remove('hidden');
            this.uploadState.classList.remove('hidden');
            this.uploadState.style.opacity = '1';
        }, 300);

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
        this.previewState.style.opacity = '0';
        setTimeout(() => {
            this.previewState.classList.add('hidden');
            this.loadingState.classList.remove('hidden');
        }, 300);
    }

    async sendPredictionRequest() {
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
                const errorData = await response.json().catch(() => ({}));
                const errorMessage = errorData.error || `Server error: ${response.status}`;
                throw new Error(errorMessage);
            }

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Prediction failed');
            }

            return result.predictions;

        } catch (error) {
            // Handle network errors vs server errors differently
            if (!response) {
                throw new Error('Network error: Unable to connect to the server. Check your internet connection.');
            } else {
                throw error; // Re-throw server errors as they have specific messages
            }
        }
    }

    showResults(predictions) {
        this.predictions = predictions;

        // Hide loading
        this.loadingState.classList.add('hidden');

        // Show results with smooth transition
        this.analysisDefault.classList.add('hidden');
        this.analysisResults.classList.remove('hidden');

        // Update display
        const topPrediction = predictions[0];
        const speciesName = this.formatSpeciesName(topPrediction.species);
        const confidence = topPrediction.confidence;

        this.resultImageSmall.src = this.currentImage.src;
        this.topSpecies.textContent = speciesName;
        this.topConfidence.textContent = confidence.toFixed(1) + '%';

        // Update confidence status
        this.updateConfidenceStatus(confidence);

        // Create enhanced chart with confidence numbers
        this.createEnhancedChart(predictions);

        // Update detailed results
        this.updateDetailedResults(predictions);

        // Show success toast with advanced processing info
        this.showToast(`Species identified: ${speciesName} (Advanced uniformity processing)`, 'success');
    }

    updateConfidenceStatus(confidence) {
        let statusText, bgClass;

        if (confidence >= 80) {
            statusText = 'Excellent Confidence';
            bgClass = 'bg-gradient-to-r from-emerald-400 to-teal-500 bg-opacity-20 text-emerald-700 dark:text-emerald-300';
        } else if (confidence >= 60) {
            statusText = 'High Confidence';
            bgClass = 'bg-gradient-to-r from-blue-400 to-cyan-500 bg-opacity-20 text-blue-700 dark:text-blue-300';
        } else if (confidence >= 40) {
            statusText = 'Medium Confidence';
            bgClass = 'bg-gradient-to-r from-amber-400 to-orange-400 bg-opacity-20 text-amber-700 dark:text-amber-300';
        } else {
            statusText = 'Low Confidence';
            bgClass = 'bg-gradient-to-r from-rose-400 to-red-500 bg-opacity-20 text-rose-700 dark:text-rose-300';
        }

        this.confidenceStatus.textContent = statusText;
        this.confidenceStatus.className = `px-4 py-2 rounded-full text-sm font-semibold ${bgClass}`;
    }

    createEnhancedChart(predictions) {
        const ctx = this.chartCanvas.getContext('2d');

        if (this.chart) {
            this.chart.destroy();
        }

        const isDark = this.htmlElement.classList.contains('dark');
        const textColor = isDark ? '#e5e7eb' : '#374151';
        const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

        const labels = predictions.map(p => this.formatSpeciesName(p.species));
        const confidences = predictions.map(p => p.confidence);

        // Marine Biology Inspired Color Scheme 🌊
        const colors = [
            'rgba(0, 191, 255, 0.85)',    // Deep Sky Blue (primary plankton)
            'rgba(32, 178, 170, 0.8)',    // Light Sea Green (secondary)
            'rgba(72, 209, 204, 0.75)',   // Medium Turquoise (tertiary)
            'rgba(0, 206, 209, 0.7)',     // Dark Turquoise (fourth)
            'rgba(95, 158, 160, 0.65)'    // Cadet Blue (fifth)
        ];

        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: confidences,
                    backgroundColor: colors,
                    borderColor: colors.map(color => color.replace('0.85', '1').replace('0.8', '1').replace('0.75', '1').replace('0.7', '1').replace('0.65', '1')),
                    borderWidth: 3,
                    borderRadius: 12,
                    borderSkipped: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                layout: {
                    padding: 20
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'rgba(248, 250, 252, 0.95)',
                        titleColor: textColor,
                        bodyColor: textColor,
                        borderColor: isDark ? 'rgba(56, 178, 172, 0.5)' : 'rgba(14, 165, 233, 0.5)',
                        borderWidth: 2,
                        cornerRadius: 12,
                        padding: 16,
                        titleFont: { size: 14, weight: 'bold' },
                        bodyFont: { size: 13 },
                        callbacks: {
                            title: () => '',
                            label: (context) => {
                                return `${context.label}: ${context.parsed.x.toFixed(1)}% confidence`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: gridColor,
                            lineWidth: 1
                        },
                        ticks: {
                            color: textColor,
                            font: { size: 12, weight: '500' },
                            callback: (value) => value + '%'
                        },
                        title: {
                            display: true,
                            text: 'Confidence Level (%)',
                            color: textColor,
                            font: { size: 13, weight: '600' }
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            color: textColor,
                            font: { size: 12, weight: '600' },
                            maxRotation: 0
                        }
                    }
                },
                // Ocean-inspired animations
                animation: {
                    duration: 1500,
                    easing: 'easeOutQuart'
                },
                // Show confidence numbers on bars
                onHover: (event, elements, chart) => {
                    event.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
                    // Trigger redraw to show/hide confidence numbers based on hover
                    chart.update('none');
                }
            },
            plugins: [{
                // Custom plugin to show confidence numbers on bars - only on hover for small bars
                afterDatasetsDraw: (chart) => {
                    const ctx = chart.ctx;
                    chart.data.datasets.forEach((dataset, i) => {
                        const meta = chart.getDatasetMeta(i);
                        meta.data.forEach((bar, index) => {
                            const data = dataset.data[index];
                            const barWidth = bar.width;
                            const barHeight = Math.abs(bar.y - bar.base);

                            // Only show numbers if bar is large enough OR on hover
                            const showNumber = barWidth > 60 || chart._active?.some(activeElement =>
                                activeElement.datasetIndex === i && activeElement.index === index
                            );

                            if (showNumber) {
                                ctx.fillStyle = '#ffffff';
                                ctx.font = 'bold 12px Inter';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';

                                const text = data.toFixed(1) + '%';
                                const x = bar.x - 30;
                                const y = bar.y;

                                // Marine-themed background with oceanic gradient
                                const textWidth = ctx.measureText(text).width;
                                const gradient = ctx.createLinearGradient(x - textWidth/2 - 4, y - 8, x + textWidth/2 + 4, y + 8);
                                gradient.addColorStop(0, 'rgba(14, 116, 144, 0.9)'); // Deep ocean
                                gradient.addColorStop(1, 'rgba(56, 178, 172, 0.9)'); // Teal
                                ctx.fillStyle = gradient;
                                ctx.fillRect(x - textWidth/2 - 6, y - 10, textWidth + 12, 20);

                                ctx.fillStyle = '#ffffff';
                                ctx.fillText(text, x, y);
                            }
                        });
                    });
                }
            }]
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
        this.chart.options.scales.x.title.color = textColor;

        this.chart.update('none');
    }

    updateDetailedResults(predictions) {
        this.detailedResults.innerHTML = '';

        predictions.forEach((pred, index) => {
            const speciesName = this.formatSpeciesName(pred.species);
            const confidence = pred.confidence;

            const resultItem = document.createElement('div');
            resultItem.className = 'flex items-center justify-between p-3 bg-white dark:bg-gray-600 rounded-lg';

            resultItem.innerHTML = `
                <div class="flex items-center space-x-3">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 via-teal-500 to-cyan-600 text-white font-bold flex items-center justify-center text-sm">
                        ${index + 1}
                    </div>
                    <div>
                        <div class="font-semibold text-gray-900 dark:text-white">${speciesName}</div>
                        <div class="text-xs text-gray-500 dark:text-gray-400">${pred.species}</div>
                    </div>
                </div>
                <div class="text-right">
                    <div class="font-bold text-lg text-gray-900 dark:text-white">${confidence.toFixed(2)}%</div>
                    <div class="text-xs text-gray-500 dark:text-gray-400">Confidence</div>
                </div>
            `;

            this.detailedResults.appendChild(resultItem);
        });
    }

    toggleDetailedResults() {
        const isHidden = this.detailedResults.classList.contains('hidden');

        if (isHidden) {
            this.detailedResults.classList.remove('hidden');
            this.toggleIcon.style.transform = 'rotate(180deg)';
        } else {
            this.detailedResults.classList.add('hidden');
            this.toggleIcon.style.transform = 'rotate(0deg)';
        }
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
            <div class="text-center space-y-6 p-12">
                <div class="w-20 h-20 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center mx-auto">
                    <span class="material-symbols-outlined text-red-600 text-3xl">error</span>
                </div>
                <div>
                    <h3 class="text-xl font-bold text-red-600 dark:text-red-400 mb-3">Analysis Failed</h3>
                    <p class="text-gray-600 dark:text-gray-400 mb-6 max-w-md mx-auto">${message}</p>
                    <button onclick="planktonApp.resetToUpload()" class="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-semibold px-6 py-3 rounded-lg transition-all duration-300 transform hover:scale-105">
                        Try Again
                    </button>
                </div>
            </div>
        `;

        this.uploadArea.innerHTML = errorHTML;
        this.showToast(message, 'error');
    }

    async loadModelInfo() {
        try {
            const response = await fetch('/api/model-info');
            const info = await response.json();

            this.modelClasses.textContent = info.classes;
            this.modelStatus.textContent = info.available ? 'Ready' : 'Unavailable';
            this.modelStatus.className = info.available
                ? 'font-semibold text-teal-600 dark:text-cyan-400'
                : 'font-semibold text-red-600 dark:text-red-400';

            if (info.available) {
                // Model ready but don't show toast to reduce popup spam
            }
        } catch (error) {
            console.error('Failed to load model info:', error);
            this.modelStatus.textContent = 'Error';
            this.modelStatus.className = 'font-semibold text-red-600 dark:text-red-400';
            this.showToast('Failed to connect to model', 'error');
        }
    }

    // Toast Notifications
    showToast(message, type = 'info') {
        // Remove existing toasts
        const existingToasts = document.querySelectorAll('.toast');
        existingToasts.forEach(toast => toast.remove());

        const toast = document.createElement('div');
        toast.className = `toast fixed top-4 left-1/2 transform -translate-x-1/2 z-50 px-6 py-3 rounded-lg shadow-lg transition-all duration-300 translate-y-[-100px] opacity-0`;

        const config = {
            success: { bg: 'bg-gradient-to-r from-teal-500 to-cyan-600', icon: 'check_circle' },
            error: { bg: 'bg-gradient-to-r from-red-500 to-orange-600', icon: 'error' },
            info: { bg: 'bg-gradient-to-r from-blue-500 to-sky-600', icon: 'info' }
        };

        const { bg, icon } = config[type] || config.info;

        // Split the bg classes and add them individually
        const bgClasses = bg.split(' ');
        bgClasses.forEach(cls => toast.classList.add(cls));

        toast.innerHTML = `
            <div class="flex items-center space-x-2 text-white">
                <span class="material-symbols-outlined text-sm">${icon}</span>
                <span class="text-sm font-medium">${message}</span>
            </div>
        `;

        document.body.appendChild(toast);

        // Animate in
        setTimeout(() => {
            toast.classList.remove('translate-y-[-100px]', 'opacity-0');
        }, 100);

        // Auto remove
        setTimeout(() => {
            toast.classList.add('translate-y-[-100px]', 'opacity-0');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.planktonApp = new EnhancedPlanktonClassifier();
});