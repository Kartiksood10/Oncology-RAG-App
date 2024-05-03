# Oncology-RAG-App

Medical RAG App built using Meditron Medical LLM and PubMedBERT Embeddings

Meditron :
Fine tuned llama2 model on medical data : https://huggingface.co/TheBloke/meditron-7B-GGUF

PubMedBERT Embeddings :
Fine tuned sentence transformer for medical data : https://huggingface.co/NeuML/pubmedbert-base-embeddings

Qdrant Vector Database :
Qdrant is a vector similarity search engine and vector database : https://hub.docker.com/r/qdrant/qdrant

Command to run Docker image on localhost:6333 :

```
docker pull qdrant/qdrant
```

## UI

![image](https://github.com/Kartiksood10/Oncology-RAG-App/assets/82945071/b0577355-171c-4cf5-ac4f-4c02df1fc919)
