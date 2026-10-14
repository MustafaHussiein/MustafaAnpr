document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM fully loaded and parsed");

    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    let stream = null;
    let currentImage = null;

    function resizeCanvas(w, h) {
        canvas.width = w;
        canvas.height = h;
    }

    // ============================================================
    // INPUT CONTROLS
    // ============================================================

    const uploadInput = document.getElementById("uploadInput");
    if (uploadInput) {
        uploadInput.addEventListener("change", e => {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = ev => {
                const img = new Image();
                img.onload = () => {
                    resizeCanvas(img.naturalWidth, img.naturalHeight);
                    ctx.drawImage(img, 0, 0);
                    currentImage = canvas.toDataURL("image/jpeg", 0.95);
                    canvas.style.display = "block";
                    video.style.display = "none";
                };
                img.src = ev.target.result;
            };
            reader.readAsDataURL(file);
        });
    }

    document.getElementById("startCamBtn").onclick = async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720 }
            });
            video.srcObject = stream;
            video.style.display = "block";
            canvas.style.display = "none";
        } catch (err) {
            alert("Camera access denied: " + err.message);
        }
    };

    document.getElementById("captureBtn").onclick = () => {
        if (!video.srcObject) {
            alert("Start camera first!");
            return;
        }
        resizeCanvas(video.videoWidth, video.videoHeight);
        ctx.drawImage(video, 0, 0);
        currentImage = canvas.toDataURL("image/jpeg", 0.95);
        canvas.style.display = "block";
        video.style.display = "none";
    };

    document.getElementById("closeCamBtn").onclick = () => {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        video.style.display = "none";
    };

    // ============================================================
    // ANALYSIS LOGIC
    // ============================================================

    // Helper to restart camera without repeating code
async function restartCamera() {
    try {
        const newStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
        });
        video.srcObject = newStream;
        video.style.display = "block";
        canvas.style.display = "none";
        return newStream;
    } catch (err) {
        console.error("Failed to restart camera:", err);
    }
}

async function performAnalysis(endpoint, mode, buttonId) {
    if (!currentImage) {
        alert("No image to analyze!");
        return;
    }

    const wasStreamOpen = (stream !== null);
    const currentBtn = document.getElementById(buttonId);
    currentBtn.disabled = true;
    currentBtn.innerText = "⏳ Analyzing...";

    try {
        // 1. If stream is open, stop it to free resources during heavy AI task
        if (wasStreamOpen) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }

        const blob = await (await fetch(currentImage)).blob();
        const formData = new FormData();
        formData.append("image", blob, "image.jpg");

        const res = await fetch(endpoint, { method: "POST", body: formData });
        const result = await res.json();
        
        if (result.plates) displayPlates(result.plates);

        // 2. Display the annotated result briefly
        if (result.annotated_image_url) {
            const img = new Image();
            img.onload = () => {
                resizeCanvas(img.naturalWidth, img.naturalHeight);
                ctx.drawImage(img, 0, 0);
            };
            img.src = result.annotated_image_url;
        }

        // 3. WAIT and then restart the stream if it was originally open
        setTimeout(async () => {
            if (wasStreamOpen) {
                stream = await restartCamera();
            } else {
                // If it was an upload, just leave the result on screen
                currentImage = null; 
            }
        }, 3000); // 3-second delay to view the result before stream resumes

    } catch (err) {
        console.error(err);
        alert("Analysis failed.");
        if (wasStreamOpen) stream = await restartCamera();
    } finally {
        currentBtn.disabled = false;
        currentBtn.innerText = "Analyze Plates";
    }
}
    function displayPlates(plates) {
        const plateList = document.getElementById("plateList");
        
        if (plates && plates.length > 0) {
            // --- MODIFIED: Only remove the placeholder paragraph, not the old results ---
            const placeholder = plateList.querySelector("p");
            if (placeholder) {
                placeholder.remove();
            }

            plates.forEach((plate) => {
                const item = document.createElement("div");
                item.className = "plate-item";
                
                // Use the crop_url from backend history, fallback to current session image
                const displayImg = plate.crop_url ? plate.crop_url : currentImage;

                item.innerHTML = `
                    <div class="plate-history-card">
                        <img src="${displayImg}" class="plate-thumb" alt="Plate Crop">
                        <div class="plate-meta">
                            <span class="plate-number-text">${plate.text}</span>
                            <span class="conf-badge">${(plate.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div class="plate-timestamp">${new Date().toLocaleTimeString()}</div>
                    </div>
                `;
                // Add new detections to the TOP of the history
                plateList.prepend(item);
            });
        }
    }
    // ============================================================
    // LIVE CLOCK LOGIC
    // ============================================================
    function updateClock() {
        const clockEl = document.getElementById("liveClock");
        if (clockEl) {
            const now = new Date();
            // Formats to: DD/MM/YYYY, HH:MM:SS
            const dateStr = now.toLocaleDateString('en-GB');
            const timeStr = now.toLocaleTimeString('en-GB');
            clockEl.innerText = `${dateStr} ${timeStr}`;
        }
    }

    // Start the clock and update every second
    setInterval(updateClock, 1000);
    updateClock(); // Initial call so it doesn't start empty
    const analyzeBtn = document.getElementById("analyzePlatesBtn");
    if (analyzeBtn) {
        analyzeBtn.onclick = () => performAnalysis('/anpr/analyze-plates', 'plates', 'analyzePlatesBtn');
    }

    const analyzeBothBtn = document.getElementById("analyzeBothBtn");
    if (analyzeBothBtn) {
        analyzeBothBtn.onclick = () => performAnalysis('/anpr/analyze', 'both', 'analyzeBothBtn');
    }
});