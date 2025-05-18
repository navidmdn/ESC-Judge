import React, { useEffect, useState } from 'react';
import {
  fetchConversationFiles,
  fetchConversationFile,
  fetchDimensions
} from './services/annotationService';
import AnnotationPanel from './components/AnnotationPanel';

function App() {
  const [fileList, setFileList] = useState([]);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [conversation, setConversation] = useState(null); // {modelA, modelB}
  const [dimensions, setDimensions] = useState([]);

  useEffect(() => {
    // Load the dimension data and file list on mount
    fetchDimensions().then((dims) => {
      setDimensions(dims.dimensions);
    });
    fetchConversationFiles().then((files) => {
      setFileList(files);
      setCurrentFileIndex(0);
    });
  }, []);

  // Whenever currentFileIndex changes, load that file’s conversation
  useEffect(() => {
    if (fileList.length > 0 && currentFileIndex < fileList.length) {
      const fileName = fileList[currentFileIndex];
      fetchConversationFile(fileName).then((data) => {
        setConversation({ ...data.conversation, fileName });
      });
    }
  }, [fileList, currentFileIndex]);

  const handleNextFile = () => {
    // Move to next file if any left
    if (currentFileIndex < fileList.length - 1) {
      setCurrentFileIndex(currentFileIndex + 1);
    } else {
      alert("No more files to annotate!");
    }
  };

  if (dimensions.length === 0) {
    return <div>Loading dimensions...</div>;
  }

  if (fileList.length === 0) {
    return <div>Loading conversation files...</div>;
  }

  if (!conversation) {
    return <div>Loading conversation data...</div>;
  }

  return (
    <div style={{ margin: '1rem' }}>
      <h1>Annotation Task</h1>
      <h2>Annotating file: {conversation.fileName}</h2>
      <AnnotationPanel
        conversation={conversation}
        dimensions={dimensions}
        annotatorId="demo-annotator-001"
        onNextFile={handleNextFile}
        showNextButton={true}
        isLastFile={currentFileIndex === fileList.length - 1}
      />
    </div>
  );
}

export default App;
