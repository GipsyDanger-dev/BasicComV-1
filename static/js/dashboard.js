let statsInterval;

function startDetection() {
    fetch('/api/start', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('status').textContent = 'Running';
                document.getElementById('status').className = 'badge bg-success';
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                
                if (!statsInterval) {
                    statsInterval = setInterval(updateStats, 1000);
                }
            }
        });
}

function stopDetection() {
    fetch('/api/stop', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            document.getElementById('status').textContent = 'Stopped';
            document.getElementById('status').className = 'badge bg-danger';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            if (statsInterval) {
                clearInterval(statsInterval);
                statsInterval = null;
            }
        });
}

function updateStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('fps').textContent = data.fps || 0;
            document.getElementById('frameCount').textContent = data.frame_count || 0;
            
            if (data.detections && data.detections.length > 0) {
                const detectionsList = document.getElementById('detectionsList');
                detectionsList.innerHTML = '';
                
                data.detections.forEach(det => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item d-flex justify-content-between align-items-center';
                    item.innerHTML = `
                        ${det.class}
                        <span>
                            <span class="badge bg-primary rounded-pill">${det.count}</span>
                            <span class="badge bg-secondary rounded-pill">${det.confidence}</span>
                        </span>
                    `;
                    detectionsList.appendChild(item);
                });
            }
            
            if (data.zones) {
                const zoneStats = document.getElementById('zoneStats');
                zoneStats.innerHTML = '';
                
                Object.entries(data.zones).forEach(([name, count]) => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item d-flex justify-content-between align-items-center';
                    item.innerHTML = `
                        ${name}
                        <span class="badge bg-info rounded-pill">${count}</span>
                    `;
                    zoneStats.appendChild(item);
                });
            }
        });
}

document.getElementById('confidenceSlider').addEventListener('input', function() {
    document.getElementById('confidenceValue').textContent = this.value;
});

document.getElementById('iouSlider').addEventListener('input', function() {
    document.getElementById('iouValue').textContent = this.value;
});

function downloadStats() {
    window.location.href = '/api/download/stats';
}

function resetStats() {
    if (confirm('Are you sure you want to reset all statistics?')) {
        fetch('/api/reset', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                alert('Statistics reset successfully');
                updateStats();
            });
    }
}

document.getElementById('stopBtn').disabled = true;
