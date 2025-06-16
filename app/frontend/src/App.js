import React, { useState } from 'react';
import axios from 'axios';

const API_URL = 'https://rod-pump-mvp.onrender.com/api/calculate'; // Backend API endpoint
const EXPORT_URL = 'https://rod-pump-mvp.onrender.com/api/export'; // Report download URL

function App() {
  const [spm, setSpm] = useState('');
  const [pumpDepth, setPumpDepth] = useState('');
  const [rodString, setRodString] = useState('');
  const [plungerDiameter, setPlungerDiameter] = useState('1.5');
  const [fluidSG, setFluidSG] = useState('0.85');
  const [surfaceFile, setSurfaceFile] = useState(null);
  const [downholeFile, setDownholeFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!surfaceFile || !downholeFile) {
      setError('Please upload both Surface and Downhole Dynacard Excel files.');
      return;
    }

    const formData = new FormData();
    formData.append('spm', spm);
    formData.append('pump_depth', pumpDepth);
    formData.append('rod_string', rodString);
    formData.append('plunger_diameter', plungerDiameter);
    formData.append('fluid_sg', fluidSG);
    formData.append('surface_card_file', surfaceFile);
    formData.append('downhole_card_file', downholeFile);

    try {
      const response = await axios.post(API_URL, formData);
      setResult(response.data);
      setError('');
    } catch (err) {
      setError('Error during calculation: ' + err.message);
      setResult(null);
    }
  };

  const handleDownload = () => {
    window.open(EXPORT_URL, '_blank');
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h2>Rod Pump Performance Analyzer</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>SPM:</label>
          <input type="number" value={spm} onChange={(e) => setSpm(e.target.value)} required />
        </div>
        <div>
          <label>Pump Depth (m):</label>
          <input type="number" value={pumpDepth} onChange={(e) => setPumpDepth(e.target.value)} required />
        </div>
        <div>
          <label>Rod String (e.g., 1.125x1000,0.875x1500):</label>
          <input type="text" value={rodString} onChange={(e) => setRodString(e.target.value)} required />
        </div>
        <div>
          <label>Plunger Diameter (in):</label>
          <input type="number" value={plungerDiameter} onChange={(e) => setPlungerDiameter(e.target.value)} />
        </div>
        <div>
          <label>Fluid Specific Gravity:</label>
          <input type="number" step="0.01" value={fluidSG} onChange={(e) => setFluidSG(e.target.value)} />
        </div>
        <div>
          <label>Surface Dynacard Excel:</label>
          <input type="file" onChange={(e) => setSurfaceFile(e.target.files[0])} required />
        </div>
        <div>
          <label>Downhole Dynacard Excel:</label>
          <input type="file" onChange={(e) => setDownholeFile(e.target.files[0])} required />
        </div>
        <button type="submit">Calculate</button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {result && (
        <div style={{ marginTop: '20px' }}>
          <h3>Results:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
          <button onClick={handleDownload}>Download Report (CSV, PDF, Images)</button>
        </div>
      )}
    </div>
  );
}

export default App;
