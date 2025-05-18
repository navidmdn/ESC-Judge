import React, { useState } from 'react';
import ConversationDisplay from './ConversationDisplay';
import DimensionSelector from './DimensionSelector';
import RatingForm from './RatingForm';
import { saveAnnotation } from '../services/annotationService';

function AnnotationPanel({
  conversation,
  dimensions,
  annotatorId,
  onNextFile,
  showNextButton,
  isLastFile
}) {
  const [selectedDimension, setSelectedDimension] = useState(dimensions[0].name);
  const [ratingsDone, setRatingsDone] = useState({});

  const modelA = conversation.modelA;
  const modelB = conversation.modelB;
  const fileName = conversation.fileName;

  const handleSelectDimension = (dimName) => {
    setSelectedDimension(dimName);
  };

  const handleSubmitRating = async ({ dimension, subcategory, winner, comments }) => {
    const payload = {
      annotatorId,
      fileName,         // <-- important to store which file is being annotated
      dimension,
      subcategory,
      winner,
      comments
    };
    await saveAnnotation(payload);

    // Mark that subcategory as done
    setRatingsDone((prev) => ({
      ...prev,
      [`${dimension}-${subcategory}`]: true
    }));
  };

  const dimObj = dimensions.find((d) => d.name === selectedDimension);
  const subcategories = dimObj ? dimObj.subcategories : [];

  return (
    <div style={{ display: 'flex', gap: '1rem' }}>
      {/* Conversations side by side */}
      <div style={{ flex: '1' }}>
        <h2>Conversation A</h2>
        <ConversationDisplay conversation={modelA} />
      </div>
      <div style={{ flex: '1' }}>
        <h2>Conversation B</h2>
        <ConversationDisplay conversation={modelB} />
      </div>

      {/* Annotation controls */}
      <div style={{ flex: '1' }}>
        <DimensionSelector
          dimensions={dimensions}
          onSelectDimension={handleSelectDimension}
          selectedDimension={selectedDimension}
        />

        <div>
          <h3>Subcategories for: {selectedDimension}</h3>
          {subcategories.map((subcat) => (
            <RatingForm
              key={subcat}
              dimension={selectedDimension}
              subcategory={subcat}
              onSubmit={handleSubmitRating}
            />
          ))}
        </div>

        {/* “Next” button to proceed to next file once user is done */}
        {showNextButton && (
          <div style={{ marginTop: '1rem' }}>
            <button
              onClick={onNextFile}
              style={{
                backgroundColor: '#007bff',
                color: '#fff',
                padding: '0.5rem 1rem',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              {isLastFile ? 'Finish' : 'Next File'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default AnnotationPanel;
