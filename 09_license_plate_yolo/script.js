class LicensePlateDetector {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.videoStream = null;
        this.isCameraActive = false;
        this.currentImage = null;
        
        // YOLO model parameters (adjust based on your model)
        this.modelSize = 640;
        this.confidenceThreshold = 0.5;
        this.iouThreshold = 0.5;
        
        this.initializeElements();
        this.loadModel();
        this.setupEventListeners();
    }
    
    initializeElements() {
        this.canvas = document.getElementById('inputCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.video = document.getElementById('cameraFeed');
        this.imageInput = document.getElementById('imageInput');
        this.cameraBtn = document.getElementById('cameraBtn');
        this.stopCameraBtn = document.getElementById('stopCameraBtn');
        this.detectBtn = document.getElementById('detectBtn');
        this.status = document.getElementById('status');
        this.results = document.getElementById('results');
    }
    
    async loadModel() {
        try {
            this.updateStatus('Loading YOLO model...');
            
            // Load the ONNX model
            this.model = await ort.InferenceSession.create('./license_plate_yolo.onnx');
            this.isModelLoaded = true;
            
            this.updateStatus('Model loaded successfully. Ready to detect.');
            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Error loading model:', error);
            this.updateStatus('Error loading model. Please check console.');
        }
    }
    
    setupEventListeners() {
        this.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        this.cameraBtn.addEventListener('click', () => this.startCamera());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
        this.detectBtn.addEventListener('click', () => this.detect());
    }
    
    updateStatus(message) {
        this.status.textContent = message;
    }
    
    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const img = new Image();
            const url = URL.createObjectURL(file);
            
            img.onload = () => {
                this.currentImage = img;
                this.drawImageToCanvas(img);
                this.updateStatus('Image loaded. Click "Detect" to find license plates.');
                URL.revokeObjectURL(url);
            };
            
            img.src = url;
        } catch (error) {
            console.error('Error loading image:', error);
            this.updateStatus('Error loading image');
        }
    }
    
    async startCamera() {
        try {
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };
            
            this.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.videoStream;
            this.video.style.display = 'block';
            this.isCameraActive = true;
            
            // Wait for video to be ready
            await new Promise(resolve => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    resolve();
                };
            });
            
            this.updateStatus('Camera active. Click "Detect" to find license plates.');
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.updateStatus('Error accessing camera. Please check permissions.');
        }
    }
    
    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
            this.video.style.display = 'none';
            this.isCameraActive = false;
            this.updateStatus('Camera stopped.');
        }
    }
    
    drawImageToCanvas(img) {
        const canvas = this.canvas;
        const ctx = this.ctx;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Set canvas dimensions
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Draw image
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
    
    async detect() {
        if (!this.isModelLoaded) {
            this.updateStatus('Model not loaded yet. Please wait.');
            return;
        }
        
        try {
            this.updateStatus('Detecting license plates...');
            
            let imageData;
            
            if (this.isCameraActive && this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
                // Capture from camera
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.ctx.drawImage(this.video, 0, 0);
                imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
            } else if (this.currentImage) {
                // Use uploaded image
                imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
            } else {
                this.updateStatus('No image source available. Please upload an image or start camera.');
                return;
            }
            
            // Preprocess image
            const inputTensor = this.preprocessImage(imageData);
            
            // Run inference
            const outputs = await this.model.run({ [this.model.inputNames[0]]: inputTensor });
            
            // Process results
            const detections = this.processOutput(outputs);
            
            // Draw bounding boxes
            this.drawDetections(detections);
            
            // Display results
            this.displayResults(detections);
            
            this.updateStatus(`Detection complete. Found ${detections.length} license plate(s).`);
            
        } catch (error) {
            console.error('Error during detection:', error);
            this.updateStatus('Error during detection. Please try again.');
        }
    }
    
    preprocessImage(imageData) {
        const { width, height } = this.canvas;
        
        // Create a temporary canvas for resizing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.modelSize;
        tempCanvas.height = this.modelSize;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Draw and resize image
        tempCtx.drawImage(this.canvas, 0, 0, this.modelSize, this.modelSize);
        
        // Get image data
        const resizedData = tempCtx.getImageData(0, 0, this.modelSize, this.modelSize);
        
        // Convert to tensor (normalize to 0-1)
        const rgbData = new Float32Array(this.modelSize * this.modelSize * 3);
        
        for (let i = 0; i < resizedData.data.length; i += 4) {
            const pixelIndex = i / 4;
            rgbData[pixelIndex * 3] = resizedData.data[i] / 255.0;     // R
            rgbData[pixelIndex * 3 + 1] = resizedData.data[i + 1] / 255.0; // G
            rgbData[pixelIndex * 3 + 2] = resizedData.data[i + 2] / 255.0; // B
        }
        
        // Create tensor (assuming NCHW format)
        const tensor = new ort.Tensor('float32', rgbData, [1, 3, this.modelSize, this.modelSize]);
        
        return tensor;
    }
    
    processOutput(outputs) {
        const detections = [];
        
        // Get the output tensor (adjust based on your model output structure)
        const outputKey = Object.keys(outputs)[0];
        const outputData = outputs[outputKey].data;
        const outputDims = outputs[outputKey].dims;
        
        // Process YOLO output (this is a simplified version)
        // You'll need to adjust this based on your specific YOLO model output format
        const numDetections = outputDims[1];
        const numClasses = 1; // License plate class only
        
        for (let i = 0; i < numDetections; i++) {
            const startIdx = i * (5 + numClasses);
            
            // Get bounding box coordinates (adjust based on your model output format)
            const x = outputData[startIdx] * this.canvas.width / this.modelSize;
            const y = outputData[startIdx + 1] * this.canvas.height / this.modelSize;
            const width = outputData[startIdx + 2] * this.canvas.width / this.modelSize;
            const height = outputData[startIdx + 3] * this.canvas.height / this.modelSize;
            const confidence = outputData[startIdx + 4];
            
            if (confidence > this.confidenceThreshold) {
                detections.push({
                    x: x - width / 2,
                    y: y - height / 2,
                    width,
                    height,
                    confidence,
                    class: 'license_plate'
                });
            }
        }
        
        // Apply Non-Maximum Suppression
        return this.nonMaximumSuppression(detections);
    }
    
    nonMaximumSuppression(detections) {
        if (detections.length === 0) return [];
        
        // Sort by confidence
        detections.sort((a, b) => b.confidence - a.confidence);
        
        const suppressed = [];
        
        while (detections.length > 0) {
            const current = detections.shift();
            suppressed.push(current);
            
            // Filter out overlapping detections
            detections = detections.filter(det => {
                const iou = this.calculateIoU(current, det);
                return iou < this.iouThreshold;
            });
        }
        
        return suppressed;
    }
    
    calculateIoU(box1, box2) {
        const x1 = Math.max(box1.x, box2.x);
        const y1 = Math.max(box1.y, box2.y);
        const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
        const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
        
        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const area1 = box1.width * box1.height;
        const area2 = box2.width * box2.height;
        
        return intersection / (area1 + area2 - intersection);
    }
    
    drawDetections(detections) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear previous drawings by redrawing the original image
        if (this.currentImage) {
            ctx.drawImage(this.currentImage, 0, 0, width, height);
        } else if (this.isCameraActive) {
            ctx.drawImage(this.video, 0, 0, width, height);
        }
        
        // Draw bounding boxes
        detections.forEach((det, index) => {
            const { x, y, width: w, height: h, confidence } = det;
            
            // Draw bounding box
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);
            
            // Draw label background
            ctx.fillStyle = '#00ff00';
            const label = `License Plate ${(confidence * 100).toFixed(1)}%`;
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x, y - 25, textWidth + 10, 25);
            
            // Draw label text
            ctx.fillStyle = '#000';
            ctx.font = '16px Arial';
            ctx.fillText(label, x + 5, y - 8);
            
            // Draw index number
            ctx.fillStyle = '#ff0000';
            ctx.beginPath();
            ctx.arc(x + w - 10, y + 10, 15, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText((index + 1).toString(), x + w - 10, y + 10);
        });
    }
    
    displayResults(detections) {
        this.results.innerHTML = '';
        
        if (detections.length === 0) {
            this.results.innerHTML = '<div class="result-item">No license plates detected</div>';
            return;
        }
        
        detections.forEach((det, index) => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result-item';
            resultDiv.innerHTML = `
                <h4>License Plate #${index + 1}</h4>
                <p><strong>Confidence:</strong> ${(det.confidence * 100).toFixed(2)}%</p>
                <p><strong>Position:</strong> X=${det.x.toFixed(0)}, Y=${det.y.toFixed(0)}</p>
                <p><strong>Size:</strong> ${det.width.toFixed(0)} × ${det.height.toFixed(0)} pixels</p>
            `;
            this.results.appendChild(resultDiv);
        });
    }
}

// Initialize the application when the page loads
window.addEventListener('DOMContentLoaded', () => {
    new LicensePlateDetector();
});
