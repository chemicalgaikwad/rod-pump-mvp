#File: app/frontend/src/App.jsx
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [spm, setSpm] = useState('');
  const [pumpDepth, setPumpDepth] = useState('');
  const [rodString, setRodString] = useState('');
  const [plungerDiameter, setPlungerDiameter] = useState('1.5');
  const [fluidSG, setFluidSG] = useState('0.85');
  const [surfaceCardFile, setSurfaceCardFile] = useState(null);
  const [downholeCardFile, setDownholeCardFile] = useState(null);
  const [response, setResponse] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('spm', spm);
    formData.append('pump_depth', pumpDepth);
    formData.append('rod_string', rodString);
    formData.append('plunger_diameter', plungerDiameter);
    formData.append('fluid_sg', fluidSG);
    formData.append('surface_card_file', surfaceCardFile);
    formData.append('downhole_card_file', downholeCardFile);

    try {
      const res = await axios.post('/api/calculate', formData);
      setResponse(res.data);
    } catch (err) {
      console.error(err);
      alert('Error submitting form');
    }
  };

  return (
    <div className="App">
      <h2>Rod Pump Performance Analyzer</h2>
      <form onSubmit={handleSubmit}>
        <label>SPM:</label>
        <input type="number" step="any" value={spm} onChange={(e) => setSpm(e.target.value)} required /><br />

        <label>Pump Depth (m):</label>
        <input type="number" step="any" value={pumpDepth} onChange={(e) => setPumpDepth(e.target.value)} required /><br />

        <label>Rod String (format: diameterxlength, ...):</label>
        <input type="text" value={rodString} onChange={(e) => setRodString(e.target.value)} required /><br />

        <label>Plunger Diameter (in):</label>
        <input type="number" step="any" value={plungerDiameter} onChange={(e) => setPlungerDiameter(e.target.value)} required /><br />

        <label>Fluid Specific Gravity:</label>
        <input type="number" step="any" value={fluidSG} onChange={(e) => setFluidSG(e.target.value)} required /><br />

        <label>Surface Dynacard Excel File:</label>
        <input type="file" onChange={(e) => setSurfaceCardFile(e.target.files[0])} required /><br />

        <label>Downhole Dynacard Excel File:</label>
        <input type="file" onChange={(e) => setDownholeCardFile(e.target.files[0])} required /><br />

        <button type="submit">Analyze</button>
      </form>

      {response && (
        <div>
          <h3>Results:</h3>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;