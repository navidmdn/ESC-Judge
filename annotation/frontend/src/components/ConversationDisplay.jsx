import React from 'react';

function ConversationDisplay({ conversation }) {
  if (!conversation) return null;

  const getRoleColor = (role) => {
    switch (role) {
      case 'seeker:':
        return 'yellow';
      case 'supporter:':
        return 'lightgreen';
      default:
        return 'lightgray'; // Fallback color
    }
  };

  return (
    <div style={{ border: '1px solid #ccc', padding: '0.5rem' }}>
      <h3>{conversation.model_id}</h3>

      {conversation.messages.map((msg, index) => {

      const safeText = typeof msg.text === 'string'
        ? msg.text
        : JSON.stringify(msg.text);

        const highlightColor = getRoleColor(msg.role);
        return (
          <div key={index} style={{ marginBottom: '1rem', whiteSpace: 'pre-wrap'}}>
            <strong>
              <span style={{ backgroundColor: highlightColor }}>
                {msg.role}
              </span>
            </strong> {safeText}
          </div>
        );
      })}
    </div>
  );
}

export default ConversationDisplay;
