import openai
import streamlit as st
import subprocess
import requests 

st.title("Tax Tajweez")

# Initialize session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Get user input
if prompt := st.chat_input("Ask me anything related to income tax..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.expander("Assistant Response", expanded=True):
        with st.spinner("I'm thinking..."):
            # Define the URL of the API endpoint
            url = "http://localhost:3000/send_to_backend"
            # Define the data you want to send in the request body
            data = {"userMsg": prompt}
            # Make the POST request
            response = requests.post(url, json=data)
            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Render the assistant response with markdown and allow HTML
                assistant_response = (response.json())['response']
                if assistant_response not in [msg.get("content") for msg in st.session_state.messages if msg.get("role") == "assistant"]:
                    st.markdown(assistant_response, unsafe_allow_html=True)
                    # Add assistant's response to session state
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})










            else:
                st.error(f"Error: {response.status_code}")






# Specify the path to the Python file you want to run
file_path = 'api.py'
# Run the Python file
subprocess.run(['python',file_path])  