
const express = require('express');
const cors = require('cors'); // Импортируем cors
const app = express();
const port = 3000;

// Используем cors middleware
app.use(cors());

app.post('/api/v1/query/generate', (req, res) => {
  setTimeout(() =>{
  res.send({
    "user_id": "string",
    "chat_id": 0,
    "query": "Расскажи, как стать поставщиком",
    "response": {
      "human_handoff": true,
      "conversation_id": "string",
      "source_documents": [
        {
          "content": "string",
          "source": "string",
          "file_name": "string",
          "chunk_id": 0,
          "page": 0,
          "is_semantic_chunk": true
        }
      ],
      "used_files": [
        "string"
      ],
      "response": {
        "think": "string",
        "theme": "string",
        "answer": "string"
      }
    },
    "category": "string",
    "time": 0
  });
}, 60000)
});

app.get('/', (req, res) => {
    return res.send({'hi': 5})

})

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});

