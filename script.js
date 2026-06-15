const imageInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultContainer = document.getElementById('resultContainer');
const diagnosisText = document.getElementById('diagnosis');
const confidenceText = document.getElementById('confidence');

let selectedFile = null;

// Handle image upload and display preview
imageInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        const reader = new FileReader();
        reader.onload = function(event) {
            previewContainer.innerHTML = `<img src="${event.target.result}" alt="X-Ray Preview">`;
        }
        reader.readAsDataURL(selectedFile);
    }
});

// Send image to Python Backend Flask API
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert("Please upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    analyzeBtn.innerText = "Analyzing Patterns...";

    try {
        // Calling our local Flask backend server running on port 5000
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        // Show result section and update values
        resultContainer.classList.remove('hidden');
        resultContainer.className = data.prediction === 'PNEUMONIA' ? 'pneumonia-detected' : 'normal-detected';
        
        diagnosisText.innerText = data.prediction;
        confidenceText.innerText = `${data.confidence.toFixed(2)}%`;

    } catch (error) {
        console.error("Error communicating with backend:", error);
        alert("Could not connect to the Python backend server.");
    } finally {
        analyzeBtn.innerText = "🚀 Run AI Diagnostics";
    }
});