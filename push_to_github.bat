@echo off
echo Adding all files to git...
git add .

echo Committing changes...
git commit -m "Fix dependency conflicts and add robust error handling

- Updated LangChain/LangGraph versions to resolve dependency conflicts
- Added FAISS to ChromaDB fallback for Windows compatibility
- Implemented robust error handling for web search results
- Fixed 'title' KeyError in search result parsing
- Added debug logging and fallback mechanisms
- Improved Streamlit app error handling and user experience"

echo Pushing to GitHub...
git push origin main

echo Done! Your changes have been pushed to GitHub.
pause
