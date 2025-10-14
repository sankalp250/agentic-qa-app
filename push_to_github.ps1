Write-Host "Adding all files to git..." -ForegroundColor Green
git add .

Write-Host "Committing changes..." -ForegroundColor Green
git commit -m "Fix dependency conflicts and add robust error handling

- Updated LangChain/LangGraph versions to resolve dependency conflicts
- Added FAISS to ChromaDB fallback for Windows compatibility  
- Implemented robust error handling for web search results
- Fixed 'title' KeyError in search result parsing
- Added debug logging and fallback mechanisms
- Improved Streamlit app error handling and user experience"

Write-Host "Pushing to GitHub..." -ForegroundColor Green
git push origin main

Write-Host "Done! Your changes have been pushed to GitHub." -ForegroundColor Green
Read-Host "Press Enter to continue"
