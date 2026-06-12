import React, { useState } from 'react';
import axios from 'axios';
import { getStyleImage } from './data/haircuts';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);

  const selectFile = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    selectFile(e.dataTransfer.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      if (response.data.face_shape === 'Unknown') {
        setError("Couldn't detect a face. Try a clear, front-facing, well-lit photo.");
        setResult(null);
      } else {
        setResult(response.data);
      }
    } catch (err) {
      console.error('Error uploading file:', err);
      setError('Could not reach the analysis server. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header className="hero">
          <span className="hero__eyebrow">Computer Vision × GPT-4o</span>
          <h1 className="hero__title">AI Hairstyle Recommender</h1>
          <p className="hero__subtitle">
            Upload a front-facing photo. A MediaPipe vision pipeline maps your facial
            geometry, then 14 years of barbering expertise, explained by GPT-4o,
            recommends the cuts that suit your face shape.
          </p>
        </header>

        {/* Upload */}
        <section className="panel upload">
          <label
            className={`dropzone ${dragging ? 'dropzone--active' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <div className="dropzone__icon">📸</div>
            <div><strong>Drop a photo here</strong> or click to browse</div>
            <div className="dropzone__hint">JPG or PNG · clear, front-facing works best</div>
            <input type="file" accept="image/*" onChange={(e) => selectFile(e.target.files[0])} />
          </label>

          {preview && <img src={preview} alt="Your upload preview" className="preview" />}

          <button className="btn" onClick={handleUpload} disabled={loading || !file}>
            {loading ? <><span className="spinner" />Analyzing facial geometry…</> : 'Get My Haircut Recommendations'}
          </button>

          {error && <div className="error">{error}</div>}
        </section>

        {/* Results */}
        {result && (
          <section className="panel results">
            <div className="result-head">
              <div className="shape-badge"><span>Face shape</span>{result.face_shape}</div>
              {result.ratios && (
                <div className="ratios">
                  {result.ratios.cheek_to_jaw != null && (
                    <span className="ratio-chip">cheek/jaw {result.ratios.cheek_to_jaw}</span>
                  )}
                  {result.ratios.height_to_cheek != null && (
                    <span className="ratio-chip">height/cheek {result.ratios.height_to_cheek}</span>
                  )}
                </div>
              )}
            </div>

            {result.explanation ? (
              <div className="explanation">
                <span className="explanation__tag">Your barber's take</span>
                {result.explanation}
              </div>
            ) : (
              <p className="description">"{result.description}"</p>
            )}

            <h2 className="section-title">Recommended Styles</h2>
            <div className="styles-grid">
              {result.recommended_styles.map((style, index) => (
                <div className="style-card" key={index}>
                  <img
                    className="style-card__img"
                    src={getStyleImage(style)}
                    alt={style}
                    loading="lazy"
                  />
                  <div className="style-card__name">{style}</div>
                </div>
              ))}
            </div>
          </section>
        )}

        <footer className="footer">
          Built on 14 years of barbering ·{' '}
          <a href="https://github.com/henry-hai/AI-Hairstyle-Recommender" target="_blank" rel="noreferrer">
            View source on GitHub
          </a>
        </footer>
      </div>
    </div>
  );
}

export default App;