import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [formData, setFormData] = useState({
    spm: '',
    rod_weight: '',
    pump_depth: '',
    fluid_level: '',
    rod_string: '',
    surface_card_file: null,
    downhole_card_file: null
  });
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: files ? files[0] : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = new FormData();
    Object.keys(formData).forEach(key => {
      data.append(key, formData[key]);
    });

    try {
      const res = await axios.post('https://rod-pump-mvp.onrender.com/calculate', data); // ✅ Using relative path
      setResult(res.data);
    } catch (error) {
      alert('Error during calculation: ' + error.message);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>Rod Pump Dynacard Analyzer</h2>
      <form onSubmit={handleSubmit}>
        <input type="number" name="spm" placeholder="SPM" onChange={handleChange} required /><br />
        <input type="number" name="rod_weight" placeholder="Rod Weight (lbs)" onChange={handleChange} required /><br />
        <input type="number" name="pump_depth" placeholder="Pump Depth (ft)" onChange={handleChange} required /><br />
        <input type="number" name="fluid_level" placeholder="Fluid Level (ft)" onChange={handleChange} required /><br />
        <input type="text" name="rod_string" placeholder="Rod String (e.g. 1.0x1000,0.875x2000)" onChange={handleChange} required /><br />
        <input type="file" name="surface_card_file" onChange={handleChange} required /><br />
        <input type="file" name="downhole_card_file" onChange={handleChange} required /><br />
        <button type="submit">Calculate</button>
      </form>
      {result && (
        <div>
          <h3>Results:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
          {result.dynocard_image && <img src={result.dynocard_image} alt="Dynocard" width="500" />}
          {result.report_path && <a href={result.report_path} target="_blank" rel="noopener noreferrer">Download Report PDF</a>}
        </div>
      )}
    </div>
  );
}

export default App;
