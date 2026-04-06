document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');
    
    // State
    let selectedFiles = [];

    // Drag and Drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        selectedFiles = Array.from(files).filter(f => f.name.endsWith('.hea') || f.name.endsWith('.dat'));
        
        fileList.innerHTML = '';
        selectedFiles.forEach(file => {
            const chip = document.createElement('div');
            chip.className = 'file-chip';
            chip.textContent = file.name;
            fileList.appendChild(chip);
        });

        checkValidation();
    }

    function checkValidation() {
        const hasHea = selectedFiles.some(f => f.name.endsWith('.hea'));
        const hasDat = selectedFiles.some(f => f.name.endsWith('.dat'));
        const isPair = selectedFiles.length === 2 && hasHea && hasDat;
        
        // Check if names match
        let namesMatch = false;
        if (isPair) {
            const heaName = selectedFiles.find(f => f.name.endsWith('.hea')).name.replace('.hea', '');
            const datName = selectedFiles.find(f => f.name.endsWith('.dat')).name.replace('.dat', '');
            namesMatch = (heaName === datName);
        }

        submitBtn.disabled = !namesMatch;
    }

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (submitBtn.disabled) return;

        const heaFile = selectedFiles.find(f => f.name.endsWith('.hea'));
        const datFile = selectedFiles.find(f => f.name.endsWith('.dat'));

        const formData = new FormData();
        formData.append('hea_file', heaFile);
        formData.append('dat_file', datFile);

        // UI transitions
        document.getElementById('loadingOverlay').classList.remove('hidden');
        document.getElementById('resultsSection').classList.add('hidden');
        document.getElementById('resultsDivider').classList.add('hidden');
        submitBtn.disabled = true;

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                alert(data.error || 'An error occurred during processing.');
            } else {
                displayResults(data);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('A network error occurred. Ensure the server is running.');
        } finally {
            document.getElementById('loadingOverlay').classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Success message
        document.getElementById('successMessage').textContent = data.message;
        
        // Plot
        document.getElementById('signalPlot').src = 'data:image/png;base64,' + data.plot_b64;

        // Metrics
        document.getElementById('metric-LB').textContent = data.metrics.LB;
        document.getElementById('metric-MSTV').textContent = data.metrics.MSTV.toFixed(2);
        document.getElementById('metric-UCR').textContent = data.metrics.UC_rate.toFixed(1);
        
        document.getElementById('metric-AC').textContent = data.metrics.AC;
        document.getElementById('metric-DL').textContent = data.metrics.DL;
        document.getElementById('metric-DS').textContent = data.metrics.DS;
        document.getElementById('metric-DP').textContent = data.metrics.DP;

        // DL Assessment
        const dlCard = document.getElementById('dlCard');
        dlCard.className = 'result-card ' + (data.dl_result.class === 'Normal' ? 'normal' : 'abnormal');
        document.getElementById('dlValue').textContent = data.dl_result.class;
        document.getElementById('dlConf').textContent = `Model confidence: ${data.dl_result.confidence}%`;

        // FIGO Assessment
        const figoCard = document.getElementById('figoCard');
        const figoClass = data.figo_result.class.toLowerCase() === 'pathologic' ? 'abnormal' : data.figo_result.class.toLowerCase();
        figoCard.className = 'result-card ' + figoClass;
        document.getElementById('figoValue').textContent = data.figo_result.class;
        document.getElementById('figoExplanation').textContent = data.figo_result.explanation;

        // Show sections
        document.getElementById('resultsDivider').classList.remove('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');
        
        // Scroll to results
        setTimeout(() => {
            document.getElementById('resultsDivider').scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }

    // Tab Switching Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked button and target tab
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });
});

