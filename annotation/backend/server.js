const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

// A path to your folder of conversation JSON files
const conversationsDir = path.join(__dirname, 'data', 'conversations');

// Load dimension data
const dimensionsData = require('./data/dimensions.json');

// In-memory store for annotations (for demo)
let annotationsStore = [];
let annotationsStorePath = null

/**
 * GET /api/conversation-files
 * Returns a list of all JSON files in the conversations directory
 */
app.get('/api/conversation-files', (req, res) => {
  fs.readdir(conversationsDir, (err, files) => {
    if (err) {
      return res.status(500).json({ error: 'Error reading conversation directory.' });
    }
    // Filter out non-JSON if needed
    const jsonFiles = files.filter((f) => f.endsWith('.json'));
    res.json(jsonFiles);
  });
});

/**
 * GET /api/conversation-file?file=somefile.json
 * Returns the contents of a single conversation file
 */
app.get('/api/conversation-file', (req, res) => {
  const fileName = req.query.file;
  if (!fileName) {
    return res.status(400).json({ error: 'No file parameter provided.' });
  }
  const filePath = path.join(conversationsDir, fileName);

  // Check that the file is in the folder
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'Conversation file not found.' });
  }

  try {
    const fileContents = fs.readFileSync(filePath, 'utf8');
    const conversationData = JSON.parse(fileContents);
    res.json(conversationData);
  } catch (err) {
    console.error('Error reading or parsing file:', err);
    return res.status(500).json({ error: 'Error reading or parsing conversation file.' });
  }
});

/**
 * GET /api/dimensions
 * Returns the dimension/subcategory structure
 */
app.get('/api/dimensions', (req, res) => {
  res.json(dimensionsData);
});

/**
 * POST /api/annotations
 * Accepts annotation data
 */

function generateTimestampedFilename(extension = 'txt') {
  const now = new Date();
  const pad = (n) => n.toString().padStart(2, '0');

  const date = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
  const time = `${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;

  return `data/annotations/${date}_${time}.${extension}`;
}

app.post('/api/annotations', (req, res) => {
  const { annotatorId, dimension, subcategory, winner, comments, fileName } = req.body;

  if (!annotatorId || !dimension || !subcategory || !winner || !fileName) {
    return res.status(400).json({ error: 'Missing required fields (annotatorId, dimension, subcategory, winner, fileName).' });
  }

  // Store the annotation in memory (or DB)
  annotationsStore.push({
    annotatorId,
    fileName,
    dimension,
    subcategory,
    winner,
    comments
  });

  // if annotationsStore is empty generate a new file name
  if (annotationsStorePath == null) {
     annotationsStorePath = generateTimestampedFilename('json')
  }

  // For production usage, you'd likely write to a DB or to a file:
  fs.writeFileSync(annotationsStorePath, JSON.stringify(annotationsStore, null, 2));

  res.json({ success: true, message: 'Annotation saved.' });
});

// Start server
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Annotation backend running on port ${PORT}`);
});
