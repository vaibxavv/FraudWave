const API_BASE_URL = "https://fraudwave-api.onrender.com";

let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById("recordBtn");
const statusText = document.getElementById("status");
const resultCard = document.getElementById("resultCard");
const labelText = document.getElementById("label");
const confidenceText = document.getElementById("confidence");

if (recordBtn) {
    recordBtn.addEventListener("click", async () => {
        if (!mediaRecorder || mediaRecorder.state === "inactive") {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                    await sendAudioForDetection(audioBlob);
                };

                mediaRecorder.start();
                recordBtn.innerText = "🛑 Stop Recording";
                recordBtn.classList.add("recording");
                statusText.innerText = "Listening... Speak naturally.";
            } catch (err) {
                console.error("Mic access denied:", err);
                statusText.innerText = "Microphone permission required.";
            }
        } else if (mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            recordBtn.innerText = "🎤 Start Recording";
            recordBtn.classList.remove("recording");
            statusText.innerText = "Analyzing acoustic patterns on Render cloud...";
        }
    });
}

async function sendAudioForDetection(blob) {
    const formData = new FormData();
    formData.append("file", blob, "sample.wav");

    try {
        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned status: ${response.status}`);
        }

        const data = await response.json();
        displayResult(data);
    } catch (error) {
        console.error("Detection error:", error);
        statusText.innerText = "Server warming up or connection error. Try again in 10s.";
    }
}

function displayResult(data) {
    statusText.innerText = "Analysis Complete";
    if (resultCard) {
        resultCard.style.display = "block";
    }

    const prediction = data.prediction || "Unknown";
    const confidence = data.confidence !== undefined ? (data.confidence * 100).toFixed(1) : "0";

    if (labelText) {
        labelText.innerText = `Verdict: ${prediction.toUpperCase()}`;
        labelText.style.color = prediction.toLowerCase() === "fake" ? "#ff4d4f" : "#52c41a";
    }

    if (confidenceText) {
        confidenceText.innerText = `Confidence: ${confidence}%`;
    }
}
