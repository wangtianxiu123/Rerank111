import streamlit as st
import pandas as pd
import requests
import io

# Streamlit app title
st.title("Cohere Rerank 模型调试器")

# Input for API key
api_key = st.text_input("请输入您的 Cohere API 密钥", type="password")

# File uploader for queries and content
uploaded_file = st.file_uploader("上传一个包含 'query' 和 'content' 列的 CSV 文件", type="csv")

# Button to download template
st.download_button(
    label="下载 CSV 模板",
    data="query,content\n示例查询,内容段1;内容段2;内容段3\n",
    file_name='template.csv',
    mime='text/csv'
)

if uploaded_file and api_key:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Check if required columns are present
    if 'query' in data.columns and 'content' in data.columns:
        # Button to trigger rerank
        if st.button("提交"):
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            results = []
            for index, row in data.iterrows():
                query = row['query']
                # Split content by semicolon to create a list of documents
                contents = row['content'].split(';')
                
                # Prepare the payload for the API request
                payload = {
                    "query": query,
                    "documents": contents
                }
                
                # Call the Rerank API
                response = requests.post("https://api.cohere.ai/rerank", headers=headers, json=payload)
                
                if response.status_code == 200:
                    # Debug: Print the full response to check its content
                    st.write("API Response:", response.json())
                    
                    # Extract relevance scores
                    scores = response.json().get('results', [])
                    for score in scores:
                        content_index = score.get('index', 0)
                        relevance_score = score.get('relevance_score', 0)
                        results.append((query, contents[content_index], relevance_score))
                else:
                    st.error(f"错误: {response.status_code} - {response.text}")
                    break
            
            # Create a DataFrame from the results
            results_df = pd.DataFrame(results, columns=['查询', '内容', '匹配分数'])
            
            # Sort results by score
            results_df = results_df.sort_values(by='匹配分数', ascending=False)
            
            # Display the results
            st.write(results_df)
            
            # Button to download results
            st.download_button(
                label="下载结果为 CSV",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name='rerank_results.csv',
                mime='text/csv'
            )
    else:
        st.error("CSV 文件必须包含 'query' 和 'content' 列。") 
