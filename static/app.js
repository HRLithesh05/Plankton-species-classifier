// Advanced Oceanic Precision - Premium Interactive System
class AdvancedPlanktonClassifier {
    constructor() {
        this.currentImage = null;
        this.predictions = [];
        this.recentPredictions = [];
        this.chart = null;
        this.isAnalyzing = false;
        this.analysisStartTime = null;

        this.initializeElements();
        this.setupEventListeners();
        this.loadModelInfo();
        this.initializeTheme();
        this.initializeAnimations();
    }

    initializeElements() {
        // Theme elements
        this.themeToggle = document.getElementById('theme-toggle');
        this.htmlElement = document.getElementById('root');

        // Upload elements
        this.fileInput = document.getElementById('file-input');
        this.uploadArea = document.getElementById('file-upload-area');
        this.uploadBtn = document.getElementById('upload-image-btn');
        this.imageUrl = document.getElementById('image-url');
        this.analyzeBtn = document.getElementById('analyze-btn');

        // State elements
        this.uploadState = document.getElementById('upload-state');
        this.previewState = document.getElementById('image-preview-state');
        this.loadingState = document.getElementById('loading-state');

        // Preview elements
        this.previewImage = document.getElementById('preview-image');
        this.removeImageBtn = document.getElementById('remove-image');

        // Analysis elements
        this.analysisDefault = document.getElementById('analysis-default');
        this.analysisLoading = document.getElementById('analysis-loading');
        this.analysisResults = document.getElementById('analysis-results');
        this.analyzingImage = document.getElementById('analyzing-image');
        this.scanEffect = document.querySelector('.scan-line');

        // Result elements
        this.currentConfidence = document.getElementById('current-confidence');
        this.currentLatency = document.getElementById('current-latency');
        this.topSpecies = document.getElementById('top-species');
        this.topConfidence = document.getElementById('top-confidence');
        this.confidenceStatus = document.getElementById('confidence-status');
        this.resultImage = document.getElementById('result-image');
        this.viewDetailsBtn = document.getElementById('view-details');

        // Chart
        this.chartCanvas = document.getElementById('predictions-chart');

        // Info elements
        this.modelClasses = document.getElementById('model-classes');
        this.modelStatus = document.getElementById('model-status');
        this.nodeStatus = document.getElementById('node-status');
        this.recentPredictionsContainer = document.getElementById('recent-predictions');
    }

    setupEventListeners() {
        // Theme toggle with smooth animation
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        // Enhanced file upload
        this.uploadBtn.addEventListener('click', () => this.triggerFileUpload());
        document.getElementById('upload-btn').addEventListener('click', () => this.triggerFileUpload());
        document.getElementById('new-analysis-btn').addEventListener('click', () => this.triggerFileUpload());

        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Advanced drag and drop
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.uploadArea.addEventListener('dragenter', (e) => this.handleDragEnter(e));

        // URL input with validation
        this.imageUrl.addEventListener('input', () => this.validateUrlInput());
        this.imageUrl.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleUrlInput();
        });
        this.imageUrl.addEventListener('blur', () => this.handleUrlInput());

        // Image actions
        this.removeImageBtn.addEventListener('click', () => this.resetToUpload());
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());

        // Results interaction
        this.viewDetailsBtn.addEventListener('click', () => this.showDetailedResults());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));

        // Window resize handler for responsive updates
        window.addEventListener('resize', () => this.handleResize());

        // Performance monitoring
        this.setupPerformanceMonitoring();
    }

    // Advanced Theme Management
    initializeTheme() {
        const savedTheme = localStorage.getItem('oceanic-theme') || 'dark';
        this.setTheme(savedTheme, false);
    }

    toggleTheme() {
        const currentTheme = this.htmlElement.classList.contains('dark') ? 'dark' : 'light';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme, true);
    }

    setTheme(theme, animate = false) {
        // Add transition class for smooth theme change
        if (animate) {
            document.body.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            setTimeout(() => {
                document.body.style.transition = '';
            }, 300);
        }

        this.htmlElement.classList.toggle('dark', theme === 'dark');
        localStorage.setItem('oceanic-theme', theme);

        // Update status indicator
        this.updateSystemStatus('Theme switched to ' + theme + ' mode', 'success');

        // Update chart if active
        if (this.chart) {
            this.updateChartTheme();
        }

        // Animate theme toggle button
        this.animateThemeToggle(theme);
    }

    animateThemeToggle(theme) {
        const button = this.themeToggle;
        button.style.transform = 'rotate(360deg) scale(1.1)';
        setTimeout(() => {
            button.style.transform = '';
        }, 300);
    }

    // Enhanced File Handling
    triggerFileUpload() {
        this.fileInput.click();
        this.addRippleEffect(event.target);
    }

    handleFileSelect(event) {
        const files = Array.from(event.target.files);
        if (files.length > 0) {
            this.loadImageFile(files[0]);
            this.trackAnalytics('file_upload', 'user_interaction');
        }
    }

    handleDragEnter(event) {
        event.preventDefault();
        this.uploadArea.style.transform = 'scale(1.02)';
    }

    handleDragOver(event) {
        event.preventDefault();
        this.uploadArea.classList.add('dragover');

        // Visual feedback enhancement
        const rect = this.uploadArea.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        this.uploadArea.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(6, 182, 212, 0.2), transparent)`;
    }

    handleDragLeave(event) {
        event.preventDefault();
        if (!this.uploadArea.contains(event.relatedTarget)) {
            this.uploadArea.classList.remove('dragover');
            this.uploadArea.style.transform = '';
            this.uploadArea.style.background = '';
        }
    }

    handleDrop(event) {
        event.preventDefault();
        this.uploadArea.classList.remove('dragover');
        this.uploadArea.style.transform = '';
        this.uploadArea.style.background = '';

        const files = Array.from(event.dataTransfer.files);
        const imageFiles = files.filter(file => file.type.startsWith('image/'));

        if (imageFiles.length > 0) {
            this.loadImageFile(imageFiles[0]);
            this.showSuccessToast('Image loaded successfully!');
        } else {
            this.showErrorToast('Please drop a valid image file');
        }
    }

    validateUrlInput() {
        const url = this.imageUrl.value.trim();
        const inputElement = this.imageUrl;

        if (url === '') {
            inputElement.style.borderColor = '';
            return;
        }

        if (this.isValidImageUrl(url)) {
            inputElement.style.borderColor = '#10b981'; // Green
            inputElement.style.boxShadow = '0 0 0 3px rgba(16, 185, 129, 0.1)';
        } else {
            inputElement.style.borderColor = '#ef4444'; // Red
            inputElement.style.boxShadow = '0 0 0 3px rgba(239, 68, 68, 0.1)';
        }
    }

    handleUrlInput() {
        const url = this.imageUrl.value.trim();
        if (url && this.isValidImageUrl(url)) {
            this.loadImageFromUrl(url);
        }
    }

    isValidImageUrl(string) {
        try {
            const url = new URL(string);
            const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.tiff', '.bmp'];
            return imageExtensions.some(ext => url.pathname.toLowerCase().includes(ext)) ||
                   url.hostname.includes('unsplash') ||
                   url.hostname.includes('imgur') ||
                   url.hostname.includes('googleusercontent');
        } catch (_) {
            return false;
        }
    }

    loadImageFile(file) {
        // Validate file size
        if (file.size > 500 * 1024 * 1024) { // 500MB
            this.showErrorToast('File size too large. Maximum 500MB allowed.');
            return;
        }

        // Show loading state
        this.showLoadingToast('Processing image...');

        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImage = {
                type: 'file',
                data: file,
                src: e.target.result,
                name: file.name,
                size: file.size
            };
            this.showImagePreview(e.target.result);
            this.hideLoadingToast();
        };
        reader.onerror = () => {
            this.showErrorToast('Failed to read image file');
            this.hideLoadingToast();
        };
        reader.readAsDataURL(file);
    }

    async loadImageFromUrl(url) {
        this.showLoadingToast('Loading image from URL...');

        try {
            // Validate URL accessibility without CORS issues
            this.currentImage = {
                type: 'url',
                data: url,
                src: url,
                name: url.split('/').pop() || 'image',
                size: 'Unknown'
            };
            this.showImagePreview(url);
            this.hideLoadingToast();
        } catch (error) {
            this.showErrorToast('Failed to load image from URL');
            this.hideLoadingToast();
        }
    }

    showImagePreview(src) {
        // Animate transition to preview state
        this.uploadState.style.opacity = '0';
        this.uploadState.style.transform = 'translateY(-20px)';

        setTimeout(() => {
            this.uploadState.classList.add('hidden');
            this.previewState.classList.remove('hidden');
            this.previewImage.src = src;

            // Animate preview appearance
            this.previewState.style.opacity = '0';
            this.previewState.style.transform = 'translateY(20px)';

            requestAnimationFrame(() => {
                this.previewState.style.opacity = '1';
                this.previewState.style.transform = 'translateY(0)';
            });
        }, 150);

        this.imageUrl.value = '';
        this.updateSystemStatus('Image loaded successfully', 'success');
    }

    resetToUpload() {
        // Animate transition back to upload state
        this.previewState.style.opacity = '0';
        this.previewState.style.transform = 'translateY(20px)';

        setTimeout(() => {
            this.previewState.classList.add('hidden');
            this.analysisResults.classList.add('hidden');
            this.analysisDefault.classList.remove('hidden');
            this.uploadState.classList.remove('hidden');

            // Reset styles
            this.uploadState.style.opacity = '1';
            this.uploadState.style.transform = 'translateY(0)';
        }, 150);

        this.currentImage = null;
        this.fileInput.value = '';
        this.imageUrl.value = '';
        this.imageUrl.style.borderColor = '';
        this.imageUrl.style.boxShadow = '';

        this.updateSystemStatus('Ready for new analysis', 'ready');
    }

    // Enhanced Analysis Process
    async analyzeImage() {
        if (!this.currentImage || this.isAnalyzing) return;

        this.isAnalyzing = true;
        this.analysisStartTime = Date.now();
        this.startAdvancedAnalysis();

        try {
            const predictions = await this.sendPredictionRequest();
            await this.showAdvancedResults(predictions);
            this.addToHistory(predictions[0], this.currentImage.src);
            this.trackAnalytics('analysis_completed', 'success');
        } catch (error) {
            this.showError('Analysis failed: ' + error.message);
            this.trackAnalytics('analysis_failed', 'error');
        } finally {
            this.isAnalyzing = false;
        }
    }

    startAdvancedAnalysis() {
        // Hide preview state with animation
        this.previewState.style.opacity = '0';
        this.previewState.style.transform = 'scale(0.95)';

        setTimeout(() => {
            this.previewState.classList.add('hidden');
            this.analysisDefault.classList.add('hidden');
            this.loadingState.classList.remove('hidden');
            this.analysisLoading.classList.remove('hidden');

            // Set analyzing image
            this.analyzingImage.src = this.currentImage.src;

            // Activate scan effect
            this.scanEffect.classList.add('active');

            // Start progress simulation
            this.simulateAdvancedProgress();

            this.updateSystemStatus('Neural analysis in progress...', 'analyzing');
        }, 200);
    }

    simulateAdvancedProgress() {
        const updateInterval = 75; // More frequent updates
        let confidence = 0;

        const progressUpdater = () => {
            if (!this.isAnalyzing) return;

            const elapsed = (Date.now() - this.analysisStartTime) / 1000;

            // Simulate realistic confidence growth
            if (elapsed < 1) {
                confidence = Math.floor(elapsed * 25 + Math.random() * 5);
            } else if (elapsed < 2) {
                confidence = Math.floor(25 + (elapsed - 1) * 35 + Math.random() * 8);
            } else {
                confidence = Math.min(95, Math.floor(60 + (elapsed - 2) * 15 + Math.random() * 10));
            }

            this.currentConfidence.textContent = confidence + '%';
            this.currentLatency.textContent = elapsed.toFixed(1) + 's';

            if (this.isAnalyzing) {
                setTimeout(progressUpdater, updateInterval);
            }
        };

        progressUpdater();
    }

    async sendPredictionRequest() {
        // Add minimum delay for UX (users expect some processing time)
        const minDelay = 2000;
        const startTime = Date.now();

        try {
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

            // Ensure minimum delay for smooth UX
            const elapsedTime = Date.now() - startTime;
            if (elapsedTime < minDelay) {
                await new Promise(resolve => setTimeout(resolve, minDelay - elapsedTime));
            }

            return result.predictions;

        } catch (error) {
            // Ensure minimum delay even for errors
            const elapsedTime = Date.now() - startTime;
            if (elapsedTime < minDelay) {
                await new Promise(resolve => setTimeout(resolve, minDelay - elapsedTime));
            }
            throw error;
        }
    }

    async showAdvancedResults(predictions) {
        this.predictions = predictions;

        // Hide loading states with smooth animation
        this.loadingState.style.opacity = '0';
        this.analysisLoading.style.opacity = '0';
        this.scanEffect.classList.remove('active');

        setTimeout(() => {
            this.loadingState.classList.add('hidden');
            this.analysisLoading.classList.add('hidden');
            this.analysisResults.classList.remove('hidden');

            // Animate results appearance
            this.analysisResults.style.opacity = '0';
            this.analysisResults.style.transform = 'translateY(30px)';

            requestAnimationFrame(() => {
                this.analysisResults.style.opacity = '1';
                this.analysisResults.style.transform = 'translateY(0)';
            });

            this.populateResults(predictions);
        }, 300);
    }

    populateResults(predictions) {
        const topPrediction = predictions[0];
        const speciesName = this.formatSpeciesName(topPrediction.species);
        const confidence = topPrediction.confidence;

        // Update result display with animations
        this.resultImage.src = this.currentImage.src;

        // Animate species name typing effect
        this.animateTextTyping(this.topSpecies, speciesName);

        // Animate confidence counter
        this.animateCounter(this.topConfidence, 0, confidence, '%', 1000);

        // Update confidence status with appropriate styling
        this.updateConfidenceStatus(confidence);

        // Create enhanced chart
        setTimeout(() => this.createAdvancedChart(predictions), 500);

        // Update system status
        this.updateSystemStatus('Analysis completed successfully', 'success');
    }

    animateTextTyping(element, text, speed = 50) {
        element.textContent = '';
        let i = 0;

        const typeWriter = () => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, speed);
            }
        };

        typeWriter();
    }

    animateCounter(element, start, end, suffix = '', duration = 1000) {
        const startTime = Date.now();

        const updateCounter = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function
            const easedProgress = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * easedProgress;

            element.textContent = current.toFixed(1) + suffix;

            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            }
        };

        updateCounter();
    }

    updateConfidenceStatus(confidence) {
        const statusElement = this.confidenceStatus;
        const indicator = statusElement.querySelector('span:first-child');
        const text = statusElement.querySelector('span:last-child');

        // Remove existing classes
        statusElement.className = 'inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold transition-all duration-300';

        if (confidence >= 80) {
            statusElement.classList.add('bg-emerald-100', 'dark:bg-emerald-900', 'text-emerald-800', 'dark:text-emerald-200');
            indicator.className = 'w-2 h-2 rounded-full bg-emerald-500 animate-pulse';
            text.textContent = 'Excellent Confidence';
        } else if (confidence >= 70) {
            statusElement.classList.add('bg-green-100', 'dark:bg-green-900', 'text-green-800', 'dark:text-green-200');
            indicator.className = 'w-2 h-2 rounded-full bg-green-500 animate-pulse';
            text.textContent = 'High Confidence';
        } else if (confidence >= 50) {
            statusElement.classList.add('bg-yellow-100', 'dark:bg-yellow-900', 'text-yellow-800', 'dark:text-yellow-200');
            indicator.className = 'w-2 h-2 rounded-full bg-yellow-500 animate-pulse';
            text.textContent = 'Medium Confidence';
        } else {
            statusElement.classList.add('bg-red-100', 'dark:bg-red-900', 'text-red-800', 'dark:text-red-200');
            indicator.className = 'w-2 h-2 rounded-full bg-red-500 animate-pulse';
            text.textContent = 'Low Confidence';
        }

        // Animate status appearance
        statusElement.style.transform = 'scale(0.8)';
        statusElement.style.opacity = '0';

        setTimeout(() => {
            statusElement.style.transform = 'scale(1)';
            statusElement.style.opacity = '1';
        }, 200);
    }

    createAdvancedChart(predictions) {
        const ctx = this.chartCanvas.getContext('2d');

        // Destroy existing chart
        if (this.chart) {
            this.chart.destroy();
        }

        const isDark = this.htmlElement.classList.contains('dark');
        const textColor = isDark ? '#e5e7eb' : '#374151';
        const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

        // Enhanced data preparation
        const labels = predictions.map(p => this.formatSpeciesName(p.species));
        const confidences = predictions.map(p => p.confidence);

        // Premium gradient colors
        const colors = [
            'rgba(6, 182, 212, 0.8)',
            'rgba(59, 130, 246, 0.8)',
            'rgba(139, 92, 246, 0.8)',
            'rgba(168, 85, 247, 0.8)',
            'rgba(192, 132, 252, 0.8)'
        ];

        const borderColors = [
            'rgb(6, 182, 212)',
            'rgb(59, 130, 246)',
            'rgb(139, 92, 246)',
            'rgb(168, 85, 247)',
            'rgb(192, 132, 252)'
        ];

        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: confidences,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 2,
                    borderRadius: {
                        topLeft: 8,
                        topRight: 8,
                        bottomLeft: 0,
                        bottomRight: 0
                    },
                    borderSkipped: false
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
                        backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                        titleColor: textColor,
                        bodyColor: textColor,
                        borderColor: gridColor,
                        borderWidth: 1,
                        cornerRadius: 12,
                        padding: 12,
                        titleFont: { size: 14, weight: 'bold' },
                        bodyFont: { size: 13 },
                        callbacks: {
                            title: () => '',
                            label: (context) => `Confidence: ${context.parsed.x.toFixed(1)}%`
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
                            font: { size: 11, weight: '500' },
                            callback: (value) => value + '%'
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
                animation: {
                    duration: 1500,
                    easing: 'easeOutCubic',
                    delay: (context) => context.dataIndex * 100
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

    // Enhanced History Management
    addToHistory(prediction, imageSrc) {
        const historyItem = {
            id: Date.now(),
            species: prediction.species,
            confidence: prediction.confidence,
            image: imageSrc,
            timestamp: new Date(),
            analysisTime: (Date.now() - this.analysisStartTime) / 1000
        };

        this.recentPredictions.unshift(historyItem);
        if (this.recentPredictions.length > 3) {
            this.recentPredictions.pop();
        }

        this.updateHistoryDisplay();
        this.saveHistoryToStorage();
    }

    updateHistoryDisplay() {
        // Clear existing items except the placeholder
        const container = this.recentPredictionsContainer;
        const placeholder = container.querySelector('.group'); // Keep the placeholder
        container.innerHTML = '';

        // Add recent predictions with staggered animation
        this.recentPredictions.forEach((item, index) => {
            const historyElement = this.createHistoryItem(item, index);
            container.appendChild(historyElement);
        });

        // Add placeholder back if there are less than 4 items
        if (this.recentPredictions.length < 4) {
            container.appendChild(placeholder);
        }
    }

    createHistoryItem(item, index) {
        const timeAgo = this.getTimeAgo(item.timestamp);
        const speciesName = this.formatSpeciesName(item.species);

        const element = document.createElement('div');
        element.className = 'group neural-card backdrop-blur-xl p-6 rounded-2xl hover-lift cursor-pointer opacity-0 transform translate-y-4';
        element.style.animationDelay = `${index * 0.1}s`;

        element.innerHTML = `
            <div class="aspect-square rounded-xl overflow-hidden mb-4 relative border border-slate-200 dark:border-slate-700">
                <img class="w-full h-full object-cover transition-all duration-500 group-hover:scale-110 group-hover:brightness-110" src="${item.image}" alt="${speciesName}">
                <div class="absolute top-3 right-3 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md px-3 py-1 rounded-lg text-xs font-mono font-bold text-cyan-600 dark:text-cyan-400 shadow-lg">
                    ${item.confidence.toFixed(1)}%
                </div>
                <div class="absolute inset-0 bg-gradient-to-t from-black/20 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            </div>
            <div class="space-y-2">
                <h4 class="font-bold text-slate-900 dark:text-white text-sm truncate">${speciesName}</h4>
                <div class="flex items-center justify-between text-xs">
                    <span class="text-slate-500 dark:text-slate-400 font-mono uppercase tracking-wide">${timeAgo}</span>
                    <span class="text-cyan-600 dark:text-cyan-400 font-semibold">${item.analysisTime.toFixed(1)}s</span>
                </div>
            </div>
        `;

        // Animate appearance
        setTimeout(() => {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, index * 100);

        return element;
    }

    // Utility Functions
    formatSpeciesName(species) {
        return species
            .replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }

    getTimeAgo(timestamp) {
        const seconds = Math.floor((new Date() - timestamp) / 1000);

        if (seconds < 60) return `${seconds}s ago`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }

    updateSystemStatus(message, type = 'info') {
        const statusTypes = {
            ready: { color: 'text-cyan-600 dark:text-cyan-400', text: 'READY' },
            analyzing: { color: 'text-orange-500 dark:text-orange-400', text: 'ANALYZING' },
            success: { color: 'text-green-500 dark:text-green-400', text: 'COMPLETE' },
            error: { color: 'text-red-500 dark:text-red-400', text: 'ERROR' }
        };

        const config = statusTypes[type] || statusTypes.ready;

        this.nodeStatus.innerHTML = `
            <div class="flex flex-col items-end">
                <span class="text-lg font-mono">${config.text}</span>
                <span class="text-xs text-slate-500 dark:text-slate-400">${message}</span>
            </div>
            <div class="relative">
                <div class="status-dot w-3 h-3 bg-current rounded-full"></div>
            </div>
        `;
        this.nodeStatus.className = `flex items-center justify-end gap-3 font-bold transition-colors duration-300 ${config.color}`;
    }

    // Toast Notifications
    showSuccessToast(message) {
        this.showToast(message, 'success');
    }

    showErrorToast(message) {
        this.showToast(message, 'error');
    }

    showLoadingToast(message) {
        this.showToast(message, 'loading');
    }

    hideLoadingToast() {
        const loadingToast = document.querySelector('.toast.loading');
        if (loadingToast) {
            loadingToast.remove();
        }
    }

    showToast(message, type = 'info') {
        const toastContainer = this.getOrCreateToastContainer();

        const toast = document.createElement('div');
        toast.className = `toast ${type} fixed right-6 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-4 shadow-xl backdrop-blur-md transform translate-x-full transition-all duration-300 z-50`;

        const config = {
            success: { icon: 'check_circle', color: 'text-green-600' },
            error: { icon: 'error', color: 'text-red-600' },
            loading: { icon: 'hourglass_empty', color: 'text-blue-600' },
            info: { icon: 'info', color: 'text-slate-600' }
        };

        const { icon, color } = config[type] || config.info;

        toast.innerHTML = `
            <div class="flex items-center gap-3">
                <span class="material-symbols-rounded ${color} ${type === 'loading' ? 'animate-spin' : ''}">${icon}</span>
                <span class="text-sm font-medium text-slate-900 dark:text-white">${message}</span>
            </div>
        `;

        toastContainer.appendChild(toast);

        // Position toast
        const toasts = toastContainer.children;
        toast.style.top = `${Array.from(toasts).indexOf(toast) * 80 + 24}px`;

        // Animate in
        setTimeout(() => {
            toast.style.transform = 'translateX(0)';
        }, 10);

        // Auto remove (except loading toasts)
        if (type !== 'loading') {
            setTimeout(() => {
                this.removeToast(toast);
            }, 5000);
        }
    }

    removeToast(toast) {
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }

    getOrCreateToastContainer() {
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container fixed top-0 right-0 z-50';
            document.body.appendChild(container);
        }
        return container;
    }

    // Enhanced Features
    initializeAnimations() {
        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        // Observe elements for animation
        document.querySelectorAll('.animate-fade-in-up').forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(30px)';
            el.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
            observer.observe(el);
        });
    }

    addRippleEffect(element) {
        const ripple = document.createElement('span');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);

        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = (event.clientX - rect.left - size / 2) + 'px';
        ripple.style.top = (event.clientY - rect.top - size / 2) + 'px';
        ripple.className = 'ripple absolute rounded-full bg-white/30 animate-ping';

        element.style.position = 'relative';
        element.appendChild(ripple);

        setTimeout(() => {
            ripple.remove();
        }, 600);
    }

    handleKeyboardShortcuts(event) {
        // Ctrl/Cmd + U: Upload file
        if ((event.ctrlKey || event.metaKey) && event.key === 'u') {
            event.preventDefault();
            this.triggerFileUpload();
        }

        // Escape: Reset to upload
        if (event.key === 'Escape') {
            this.resetToUpload();
        }

        // Ctrl/Cmd + Enter: Analyze image (if image is loaded)
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && this.currentImage) {
            event.preventDefault();
            this.analyzeImage();
        }
    }

    handleResize() {
        // Update chart if it exists
        if (this.chart) {
            this.chart.resize();
        }
    }

    // Analytics and Performance
    setupPerformanceMonitoring() {
        // Monitor performance
        this.performanceMetrics = {
            loadTime: Date.now(),
            interactions: 0,
            analyses: 0
        };
    }

    trackAnalytics(action, category) {
        this.performanceMetrics.interactions++;

        if (category === 'success') {
            this.performanceMetrics.analyses++;
        }

        // Here you could send analytics to your preferred service
        console.log(`Analytics: ${action} - ${category}`);
    }

    // Data Persistence
    saveHistoryToStorage() {
        try {
            localStorage.setItem('oceanic-history', JSON.stringify(this.recentPredictions));
        } catch (e) {
            console.warn('Failed to save history to localStorage');
        }
    }

    loadHistoryFromStorage() {
        try {
            const stored = localStorage.getItem('oceanic-history');
            if (stored) {
                this.recentPredictions = JSON.parse(stored).map(item => ({
                    ...item,
                    timestamp: new Date(item.timestamp)
                }));
                this.updateHistoryDisplay();
            }
        } catch (e) {
            console.warn('Failed to load history from localStorage');
        }
    }

    // API Integration
    async loadModelInfo() {
        try {
            const response = await fetch('/api/model-info');
            const info = await response.json();

            this.modelClasses.textContent = info.classes;
            this.modelStatus.textContent = info.available ? 'Ready' : 'Unavailable';
            this.modelStatus.className = info.available
                ? 'font-bold text-green-500 text-lg'
                : 'font-bold text-red-500 text-lg';

            if (info.available) {
                this.updateSystemStatus('Model loaded and ready', 'ready');
            } else {
                this.updateSystemStatus('Model unavailable', 'error');
            }
        } catch (error) {
            console.error('Failed to load model info:', error);
            this.modelStatus.textContent = 'Error';
            this.modelStatus.className = 'font-bold text-red-500 text-lg';
            this.updateSystemStatus('Failed to connect to model', 'error');
        }
    }

    // Detailed Results Modal
    showDetailedResults() {
        if (!this.predictions || this.predictions.length === 0) return;

        const modal = this.createAdvancedModal();
        document.body.appendChild(modal);

        // Animate modal appearance
        setTimeout(() => {
            modal.style.opacity = '1';
            modal.querySelector('.modal-content').style.transform = 'scale(1)';
        }, 10);
    }

    createAdvancedModal() {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/60 dark:bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4 opacity-0 transition-opacity duration-300';

        const content = document.createElement('div');
        content.className = 'modal-content bg-white dark:bg-slate-900 rounded-3xl p-8 max-w-3xl w-full max-h-[90vh] overflow-y-auto transform scale-0.9 transition-transform duration-300 shadow-2xl border border-slate-200 dark:border-slate-700';

        let detailsHTML = `
            <div class="flex justify-between items-center mb-8">
                <div>
                    <h3 class="text-3xl font-black text-slate-900 dark:text-white mb-2">Analysis Report</h3>
                    <p class="text-slate-600 dark:text-slate-400">Complete neural network classification results</p>
                </div>
                <button id="close-modal" class="p-3 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-xl transition-all duration-200">
                    <span class="material-symbols-rounded text-2xl">close</span>
                </button>
            </div>

            <div class="space-y-6">
        `;

        this.predictions.forEach((pred, index) => {
            const speciesName = this.formatSpeciesName(pred.species);
            const confidence = pred.confidence;
            const barWidth = (confidence / this.predictions[0].confidence) * 100;

            detailsHTML += `
                <div class="neural-card backdrop-blur-xl p-6 rounded-2xl">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center gap-4">
                            <div class="w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 text-white font-bold flex items-center justify-center text-lg shadow-lg">
                                ${index + 1}
                            </div>
                            <div>
                                <div class="text-xl font-bold text-slate-900 dark:text-white">${speciesName}</div>
                                <div class="text-sm text-slate-500 dark:text-slate-400 font-mono">${pred.species}</div>
                            </div>
                        </div>
                        <div class="text-right">
                            <div class="text-2xl font-black text-slate-900 dark:text-white">${confidence.toFixed(2)}%</div>
                            <div class="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wide">Confidence</div>
                        </div>
                    </div>

                    <div class="relative h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div class="absolute top-0 left-0 h-full bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full transition-all duration-1000 ease-out"
                             style="width: ${barWidth}%; animation-delay: ${index * 100}ms;"></div>
                    </div>
                </div>
            `;
        });

        detailsHTML += `
            </div>

            <div class="mt-8 pt-6 border-t border-slate-200 dark:border-slate-700">
                <div class="text-center text-sm text-slate-500 dark:text-slate-400">
                    Analysis completed in ${((Date.now() - this.analysisStartTime) / 1000).toFixed(2)} seconds
                </div>
            </div>
        `;

        content.innerHTML = detailsHTML;
        modal.appendChild(content);

        // Close modal handlers
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeModal(modal);
            }
        });

        content.querySelector('#close-modal').addEventListener('click', () => {
            this.closeModal(modal);
        });

        return modal;
    }

    closeModal(modal) {
        modal.style.opacity = '0';
        modal.querySelector('.modal-content').style.transform = 'scale(0.9)';

        setTimeout(() => {
            modal.remove();
        }, 300);
    }

    showError(message) {
        this.loadingState.classList.add('hidden');
        this.analysisLoading.classList.add('hidden');
        this.scanEffect.classList.remove('active');

        const errorHTML = `
            <div class="text-center space-y-6 p-8">
                <div class="w-24 h-24 bg-gradient-to-br from-red-500 to-red-600 rounded-3xl flex items-center justify-center mx-auto shadow-2xl">
                    <span class="material-symbols-rounded text-white text-4xl">error</span>
                </div>
                <div>
                    <h3 class="text-2xl font-bold text-red-600 dark:text-red-400 mb-3">Analysis Failed</h3>
                    <p class="text-slate-600 dark:text-slate-400 mb-6 max-w-md mx-auto">${message}</p>
                    <button onclick="planktonApp.resetToUpload()" class="btn-primary text-white font-bold px-8 py-3 rounded-xl">
                        Try Again
                    </button>
                </div>
            </div>
        `;

        this.uploadArea.innerHTML = errorHTML;
        this.updateSystemStatus('Analysis failed', 'error');
    }
}

// Initialize the Advanced Application
document.addEventListener('DOMContentLoaded', () => {
    window.planktonApp = new AdvancedPlanktonClassifier();

    // Add some initial loading animations
    document.body.style.opacity = '0';
    document.body.style.transform = 'translateY(20px)';

    setTimeout(() => {
        document.body.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
        document.body.style.opacity = '1';
        document.body.style.transform = 'translateY(0)';
    }, 100);
});

// Add global error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});