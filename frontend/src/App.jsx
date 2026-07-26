import { useState, useCallback } from 'react';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ConfidenceBar Component
function ConfidenceBar({ confidence }) {
  const percent = Math.round(confidence * 100);
  const color = percent >= 85 ? '#22c55e' : percent >= 70 ? '#f59e0b' : '#ef4444';
  return (
    <div className="confidence-section">
      <div className="confidence-label">
        <span>Model Confidence</span>
        <span style={{ color, fontWeight: 700 }}>{percent}%</span>
      </div>
      <div className="confidence-bar-track">
        <div className="confidence-bar-fill" style={{ width: `${percent}%`, background: `linear-gradient(90deg, ${color}88, ${color})` }} />
      </div>
    </div>
  );
}

// ResultCard Component
function ResultCard({ result }) {
  const getBadge = () => {
    if (result.disease_key === 'uncertain') return { cls: 'uncertain', icon: '⚠️', text: 'Cannot Determine' };
    if (result.is_healthy) return { cls: 'healthy', icon: '✅', text: 'Plant is Healthy' };
    return { cls: 'diseased', icon: '🔴', text: 'Disease Detected' };
  };
  const badge = getBadge();

  return (
    <div className="result-card">
      <div className={`status-badge ${badge.cls}`}>
        <span>{badge.icon}</span><span>{badge.text}</span>
      </div>
      <div className="disease-name">{result.disease_name}</div>
      <ConfidenceBar confidence={result.confidence} />
      <div className="info-grid">
        {result.description && (
          <div className="info-item">
            <div className="info-item-label">📋 Description</div>
            <div className="info-item-value">{result.description}</div>
          </div>
        )}
        {result.cause && result.cause !== 'N/A' && (
          <div className="info-item">
            <div className="info-item-label">🔬 Cause</div>
            <div className="info-item-value">{result.cause}</div>
          </div>
        )}
        {result.treatment && result.treatment !== 'No treatment needed.' && (
          <div className="info-item treatment">
            <div className="info-item-label">💊 Treatment</div>
            <div className="info-item-value">{result.treatment}</div>
          </div>
        )}
        {result.prevention && (
          <div className="info-item">
            <div className="info-item-label">🛡️ Prevention</div>
            <div className="info-item-value">{result.prevention}</div>
          </div>
        )}
      </div>
      {result.top3_predictions && result.top3_predictions.length > 0 && (
        <div className="top3-section">
          <div className="info-item-label" style={{ marginBottom: '8px' }}>🎯 Top Predictions</div>
          {result.top3_predictions.map((item, idx) => (
            <div key={idx} className="top3-item">
              <span className="top3-name">{idx + 1}. {item.display_name || item.class}</span>
              <span className="top3-conf">{item.confidence}</span>
            </div>
          ))}
        </div>
      )}
      {result.note && (
        <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '8px', fontSize: '0.8rem', color: '#f59e0b' }}>
          ℹ️ {result.note}
        </div>
      )}
    </div>
  );
}

// Main App Component
function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileSelect = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please upload an image file (JPG, PNG, etc.)');
      return;
    }
    setSelectedFile(file);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(file));
  }, []);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = () => setIsDragOver(false);
  const handleDrop = (e) => { e.preventDefault(); setIsDragOver(false); handleFileSelect(e.dataTransfer.files[0]); };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    setIsLoading(true); setError(null); setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
      if (!response.ok) { const err = await response.json(); throw new Error(err.detail || 'Prediction failed'); }
      setResult(await response.json());
    } catch (err) {
      setError(err.message.includes('Failed to fetch') ? 'Cannot connect to server. Make sure backend is running on port 8000.' : err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="logo"><span className="logo-icon">🌿</span><span>CropAI</span></div>
        <div className="nav-badge">AI Powered</div>
      </header>

      <section className="hero">
        <div className="hero-badge"><span className="dot"></span><span>ResNet50 • PlantVillage • 38 Classes</span></div>
        <h1>Detect Crop Diseases<br />Instantly with AI</h1>
        <p>Upload a photo of a diseased leaf. Our AI model analyzes it in seconds and gives you the disease name, cause, and treatment.</p>
        <div className="hero-stats">
          <div className="stat"><div className="stat-number">38</div><div className="stat-label">Disease Classes</div></div>
          <div className="stat"><div className="stat-number">54K+</div><div className="stat-label">Training Images</div></div>
          <div className="stat"><div className="stat-number">~94%</div><div className="stat-label">Accuracy</div></div>
          <div className="stat"><div className="stat-number">&lt;1s</div><div className="stat-label">Detection Time</div></div>
        </div>
      </section>

      <main className="main-content">
        <div className="app-grid">

          {/* Upload Panel */}
          <div className="card">
            <div className="card-title"><span>📤</span><span>Upload Leaf Image</span></div>
            <div className={`upload-area ${isDragOver ? 'drag-over' : ''}`}
              onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}
              onClick={() => document.getElementById('fileInput').click()}>
              <input id="fileInput" type="file" className="file-input" accept="image/*"
                onChange={(e) => handleFileSelect(e.target.files[0])} />
              {preview ? (
                <><img src={preview} alt="Leaf preview" className="image-preview" />
                <p style={{ marginTop: '12px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>Click to change image</p></>
              ) : (
                <><span className="upload-icon">🍃</span>
                <p className="upload-text">Drop your leaf photo here</p>
                <p className="upload-subtext">or click to browse files</p>
                <p className="upload-subtext" style={{ marginTop: '8px' }}>Supports: JPG, PNG, WEBP</p></>
              )}
            </div>
            <button className={`btn-analyze ${isLoading ? 'loading' : ''}`}
              onClick={analyzeImage} disabled={!selectedFile || isLoading}>
              {isLoading ? 'Analyzing...' : '🔍 Analyze Disease'}
            </button>
            {error && (
              <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', color: '#ef4444', fontSize: '0.85rem' }}>
                ❌ {error}
              </div>
            )}
            <div style={{ marginTop: '16px', fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center' }}>
              💡 Use clear, well-lit photos of individual leaves for best results
            </div>
          </div>

          {/* Results Panel */}
          <div className="card">
            <div className="card-title"><span>🧬</span><span>Analysis Result</span></div>
            {isLoading ? (
              <div className="empty-state">
                <div style={{ fontSize: '3rem', marginBottom: '16px' }}>⏳</div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '1rem' }}>AI is analyzing your leaf...</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '8px' }}>Running image through ResNet50 model</p>
              </div>
            ) : result ? (
              <ResultCard result={result} />
            ) : (
              <div className="empty-state">
                <div className="empty-state-icon">🌱</div>
                <p className="empty-state-text">Upload a leaf image and click analyze to see the AI prediction here</p>
              </div>
            )}
          </div>

        </div>

        <div className="how-it-works">
          <h2 className="section-title">How It Works</h2>
          <div className="steps-grid">
            <div className="step-card"><div className="step-number">1</div><div className="step-title">Upload Photo</div><div className="step-desc">Take a clear photo of the affected leaf and upload it to the app</div></div>
            <div className="step-card"><div className="step-number">2</div><div className="step-title">AI Analysis</div><div className="step-desc">ResNet50 CNN processes the image through 50 deep learning layers</div></div>
            <div className="step-card"><div className="step-number">3</div><div className="step-title">Get Results</div><div className="step-desc">Receive disease name, confidence score, cause and treatment instantly</div></div>
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>Built with ❤️ using PyTorch + FastAPI + React</p>
        <p style={{ marginTop: '8px' }}>Trained on <a href="https://plantvillage.psu.edu/" target="_blank" rel="noreferrer">PlantVillage Dataset</a> • 54,306 images • 38 disease classes</p>
      </footer>
    </div>
  );
}

export default App;
