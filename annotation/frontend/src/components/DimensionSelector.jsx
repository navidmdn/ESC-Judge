import React from 'react';

function DimensionSelector({ dimensions, onSelectDimension, selectedDimension }) {
  return (
    <div style={{ marginBottom: '1rem' }}>
      <h2>Select Dimension</h2>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {dimensions.map((dim) => (
          <li key={dim.name} style={{ margin: '0.5rem 0' }}>
            <button
              onClick={() => onSelectDimension(dim.name)}
              style={{
                backgroundColor: dim.name === selectedDimension ? '#007BFF' : '#f0f0f0',
                color: dim.name === selectedDimension ? '#fff' : '#000',
                padding: '0.5rem 1rem',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              {dim.name}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default DimensionSelector;
