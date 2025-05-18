import React, { useState } from 'react';

function RatingForm({ dimension, subcategory, onSubmit }) {
  const [winner, setWinner] = useState('');
  const [comments, setComments] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      dimension,
      subcategory,
      winner,
      comments
    });
    // Reset
    setWinner('');
    setComments('');
  };

  return (
    <form onSubmit={handleSubmit} style={{ margin: '1rem 0' }}>
      <h4>{subcategory}</h4>
      <div style={{ marginBottom: '0.5rem' }}>
        <label>
          <input
            type="radio"
            name={`${subcategory}-winner`}
            value="model_a"
            checked={winner === 'model_a'}
            onChange={(e) => setWinner(e.target.value)}
          />
          Model A
        </label>
        {'  '}
        <label>
          <input
            type="radio"
            name={`${subcategory}-winner`}
            value="model_b"
            checked={winner === 'model_b'}
            onChange={(e) => setWinner(e.target.value)}
          />
          Model B
        </label>
        {'  '}
        <label>
          <input
            type="radio"
            name={`${subcategory}-winner`}
            value="tie"
            checked={winner === 'tie'}
            onChange={(e) => setWinner(e.target.value)}
          />
          Tie
        </label>
      </div>
      <div style={{ marginBottom: '0.5rem' }}>
        <textarea
          rows="3"
          placeholder="Comments (optional)"
          style={{ width: '100%' }}
          value={comments}
          onChange={(e) => setComments(e.target.value)}
        />
      </div>
      <button
        type="submit"
        style={{
          backgroundColor: '#28a745',
          color: '#fff',
          border: 'none',
          padding: '0.5rem 1rem',
          cursor: 'pointer'
        }}
      >
        Save
      </button>
    </form>
  );
}

export default RatingForm;
