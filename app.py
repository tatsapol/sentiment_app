import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Cache the sentiment analysis pipeline to load it only once
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="Cincin-nvp/wisesight_sentiment_XLM-R")

sentiment_pipeline = load_sentiment_pipeline()

def get_sentiment(text):
    if isinstance(text, str):  # Check if the input is a string
        result = sentiment_pipeline(text)
        return result[0]['label'], result[0]['score']
    else:
        return None, None  # Default values for non-string inputs

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    st.write("Upload an Excel file and select a column for sentiment analysis.")
    
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.write("Preview of the uploaded file:")
            st.dataframe(df.head())
            
            column = st.selectbox("Select a column for sentiment analysis", df.columns)
            
            if st.button("Analyze Sentiment"):
                df[['sentiment', 'confidence']] = df[column].apply(lambda x: pd.Series(get_sentiment(x)))
                
                st.write("Sentiment analysis completed! Preview of results:")
                st.dataframe(df.head())
                st.write("Sentiment Distribution:")

                # Define colors for each sentiment value
                sentiment_colors = {
                    'Positive': '#63E66E',  # Light green
                    'Negative': '#FF7700',  # Light red
                    'Neutral': '#D3D3D3'    # Gray
                }
                
                # Visualize the sentiment column as a pie chart with specific colors and a header
                sentiment_counts = df['sentiment'].value_counts()
                colors = [sentiment_colors[label] for label in sentiment_counts.index]
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
                
                output_file = "sentiment_output.xlsx"
                df.to_excel(output_file, index=False)
                
                with open(output_file, "rb") as file:
                    st.download_button(
                        label="Download Processed File",
                        data=file,
                        file_name="sentiment_output.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
