import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Connect to Python Backend
      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-10 font-sans">
      <h1 className="text-4xl font-bold mb-8 text-blue-400">AI Hairstyle Recommender</h1>
      
      {/* Upload Section */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-md text-center">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange} 
          className="mb-4 block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
        />
        
        {preview && (
          <img src={preview} alt="Preview" className="w-64 h-64 object-cover mx-auto rounded-md mb-4 border-2 border-blue-500" />
        )}

        <button 
          onClick={handleUpload} 
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-200"
        >
          {loading ? "Analyzing Geometry..." : "Get Haircut Advice"}
        </button>
      </div>

      {/* Results Section */}
      {result && (
        <div className="mt-10 bg-gray-800 p-8 rounded-lg shadow-lg w-full max-w-2xl border-l-4 border-green-500">
          <h2 className="text-3xl font-bold mb-2">Face Shape: <span className="text-green-400">{result.face_shape}</span></h2>
          <p className="text-gray-300 mb-6 italic">"{result.description}"</p>
          
          <h3 className="text-xl font-semibold mb-4">Recommended Styles:</h3>
          <div className="grid grid-cols-2 gap-4">
            {result.recommended_styles.map((style, index) => (
              <div key={index} className="bg-gray-700 p-3 rounded text-center font-medium hover:bg-gray-600 transition">
                {style}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
