# Minimal RAG Q&A App

A simple Streamlit app for Retrieval-Augmented Generation (RAG) using LangChain, ChromaDB, SQLite, and OpenAI.

## Features
- Upload `.txt` files and ask questions
- Answers are grounded in your uploaded data
- All storage is local (no server setup needed)

## Quickstart

1. **Clone & Install**
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
pip install -r requirements.txt

text

2. **Add OpenAI Key**
- Create a `.env` file:  
  `OPENAI_API_KEY=sk-...`

3. **Run Locally**
streamlit run app.py

text

## Deploy

- **Render:**  
- Push to GitHub, create a new Web Service, set build/start commands, add `OPENAI_API_KEY` as env variable.
- **Streamlit Cloud:**  
- Push to GitHub, create app at [streamlit.io/cloud](https://streamlit.io/cloud), add `OPENAI_API_KEY` in secrets.

## Author

- [LinkedIn](https://www.linkedin.com/in/jai-chaudhary-54bb86221/)
- [GitHub](https://github.com/jcb03?tab=repositories)
- [Microsoft Learn](https://learn.microsoft.com/en-us/users/jaichaudhary-6371/)

That’s it! Your app is ready to deploy and share.