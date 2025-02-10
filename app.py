import streamlit as st
import pandas as pd
import requests

# Streamlit app title
st.title("Cohere Rerank Model Debugger")

# Input for API key
api_key = st.text_input("Enter your Cohere API Key", type="password")

# File uploader for queries and content
uploaded_file = st.file_uploader("Upload a CSV file with 'query' and 'content' columns", type="csv")

if uploaded_file and api_key:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Check if required columns are present
    if 'query' in data.columns and 'content' in data.columns:
        # Button to trigger rerank
        if st.button("Rerank"):
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            results = []
            for index, row in data.iterrows():
                query = row['query']
                content = row['content']
                
                # Prepare the payload for the API request
                payload = {
                    "query": query,
                    "documents": [content]
                }
                
                # Call the Rerank API
                response = requests.post("https://api.cohere.ai/rerank", headers=headers, json=payload)
                
                if response.status_code == 200:
                    score = response.json().get('results', [{}])[0].get('score', 0)
                    results.append((query, content, score))
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    break
            
            # Create a DataFrame from the results
            results_df = pd.DataFrame(results, columns=['Query', 'Content', 'Score'])
            
            # Sort results by score
            results_df = results_df.sort_values(by='Score', ascending=False)
            
            # Display the results
            st.write(results_df)
            
            # Button to download results
            st.download_button(
                label="Download results as CSV",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name='rerank_results.csv',
                mime='text/csv'
            )
    else:
        st.error("CSV file must contain 'query' and 'content' columns.") 
