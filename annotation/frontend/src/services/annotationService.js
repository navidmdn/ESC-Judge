import axios from 'axios';

const API_BASE = 'http://localhost:4000/api';

export async function fetchConversationFiles() {
  const res = await axios.get(`${API_BASE}/conversation-files`);
  return res.data; // array of file names, e.g. ["file1.json", "file2.json"]
}

export async function fetchConversationFile(fileName) {
  const res = await axios.get(`${API_BASE}/conversation-file`, {
    params: { file: fileName }
  });
  return res.data; // the object with { conversation: {} }
}

export async function fetchDimensions() {
  const res = await axios.get(`${API_BASE}/dimensions`);
  return res.data;
}

export async function saveAnnotation(payload) {
  const res = await axios.post(`${API_BASE}/annotations`, payload);
  return res.data;
}
