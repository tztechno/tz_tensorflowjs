// app.js

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');


let session = null;
let stream = null;
let rafId = null;


async function loadModel() {
    session = await ort.InferenceSession.create("./license_plate_yolo.onnx");
    console.log("Model loaded");
}

const targetSize = 640; // ONNX に合わせる
const scoreThreshold = 0.25;
const iouThreshold = 0.45;


// util: letterbox resize (preserve aspect ratio and pad)
function letterboxImage(imgWidth, imgHeight, target) {
const scale = Math.min(target / imgWidth, target / imgHeight);
const nw = Math.round(imgWidth * scale);
const nh = Math.round(imgHeight * scale);
const dx = Math.floor((target - nw) / 2);
const dy = Math.floor((target - nh) / 2);
return { nw, nh, dx, dy, scale };
}


function preprocessImageToTensor(imageBitmap) {
// draw to an offscreen canvas, resize with letterbox
const tmp = document.createElement('canvas');
tmp.width = targetSize;
tmp.height = targetSize;
const tctx = tmp.getContext('2d');
// fill black
tctx.fillStyle = 'black';
tctx.fillRect(0, 0, tmp.width, tmp.height);


const { nw, nh, dx, dy } = letterboxImage(imageBitmap.width, imageBitmap.height, targetSize);
tctx.drawImage(imageBitmap, 0, 0, imageBitmap.width, imageBitmap.height, dx, dy, nw, nh);


// get image data and convert to Float32Array in CHW order normalized
const imgData = tctx.getImageData(0, 0, targetSize, targetSize).data; // RGBA
// create Float32Array [1,3,H,W]
const float32 = new Float32Array(1 * 3 * targetSize * targetSize);
let offset = 0;
for (let y = 0; y < targetSize; y++) {
for (let x = 0; x < targetSize; x++) {
const idx = (y * targetSize + x) * 4;
// normalize to 0..1 and switch to RGB
float32[offset] = imgData[idx] / 255.0; // R
float32[offset + targetSize*targetSize] = imgData[idx+1] / 255.0; // G
float32[offset + 2*targetSize*targetSize] = imgData[idx+2] / 255.0; // B
offset++;
}
}
return float32;
}


function nms(boxes, scores, iouThreshold) {
// boxes: [[x1,y1,x2,y2],...]
const idxs = scores
.map((s,i) => [s,i])
.filter(x => x[0] > scoreThreshold)
.sort((a,b) => b[0]-a[0])
.map(x => x[1]);
});
