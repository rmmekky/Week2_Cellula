import streamlit as st
import pandas as pd
from PIL import Image
from imagecaption import generate_caption
from textclassifier import classify_text
import os
import io
import zipfile
import plotly.express as px

#  Clean Database (Run Once Automatically)
def clean_database():
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)

        # Normalize Classification column
        df['Classification'] = (
            df['Classification']
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "positive": "Positive",
                "negative": "Negative",
                "neutral": "Neutral"
            })
        )

        # Normalize Type column
        df['Type'] = df['Type'].astype(str).str.strip()

        df.to_csv(DB_FILE, index=False)

DB_FILE = "database.csv"
clean_database()

# function to save to CSV
def save_to_db(input_text, input_type, classification):

    # Normalize Data Before Saving
    input_text = str(input_text).strip()

    classification = (
        str(classification)
        .strip()          # remove spaces
        .capitalize()     # Positive / Negative / Neutral
    )

    input_type = str(input_type).strip()

    #  Create DB if not exists
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["Input", "Type", "Classification"])
        df.to_csv(DB_FILE, index=False)

    # Append New Row
    df = pd.read_csv(DB_FILE)
    # Fix old inconsistent values
    df['Classification'] = df['Classification'].astype(str).str.strip().str.capitalize()

    df.to_csv(DB_FILE, index=False)

    new_row = pd.DataFrame([[input_text, input_type, classification]],
                           columns=["Input", "Type", "Classification"])

    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(DB_FILE, index=False)


st.title("Text & Image Captioning App")


option = st.sidebar.selectbox("Choose Mode", ["Text Classification", "Image Captioning", "View Database"])

if option == "Text Classification":
    st.header("Text Classification")
    user_input = st.text_area("Enter your text:")

    if st.button("Classify Text"):
        if user_input.strip() != "":
            with st.spinner("Classifying text... "):
                classification = classify_text(user_input, input_type="Text")
                save_to_db(user_input, "Text", classification)
            st.success(f"Classification: {classification}")
        else:
            st.warning("Please enter some text.")

#  Image Captioning
elif option == "Image Captioning":
    st.header("Image Captioning")

    uploaded_files = st.file_uploader(
        "Drag & Drop Image(s) here or click to upload",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image {idx+1}", use_column_width=True)

            if st.button(f"Generate Caption for Image {idx+1}"):
                with st.spinner("Generating caption... "):
                    caption = generate_caption(image)
                    classification = classify_text(caption, input_type="Image Caption")  # Neutral automatically
                    save_to_db(caption, "Image Caption", classification)

                st.success(f"Caption: {caption}")
                st.info(f"Classification: {classification}")

                #  Download Image + Caption as ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                    # Save image
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format="PNG")
                    zip_file.writestr(f"image_{idx+1}.png", img_bytes.getvalue())
                    # Save caption
                    zip_file.writestr(f"caption_{idx+1}.txt", caption)

                zip_buffer.seek(0)
                st.download_button(
                    label=f"Download Image {idx+1} + Caption",
                    data=zip_buffer,
                    file_name=f"image_{idx+1}_with_caption.zip",
                    mime="application/zip"
                )

# View Database
elif option == "View Database":
    st.header("Stored Records")
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        st.dataframe(df)

        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="database.csv",
            mime="text/csv"
        )

        #  Bar Charts with Hover
        st.subheader("Classification Count")
        class_counts = df['Classification'].value_counts().reset_index()
        class_counts.columns = ["Classification", "Count"]
        fig_bar1 = px.bar(class_counts, x="Classification", y="Count",
                          text="Count", title="Classification Count")
        fig_bar1.update_traces(textposition="outside", hovertemplate="%{x}: %{y}")
        st.plotly_chart(fig_bar1)

        st.subheader("Input Type Count")
        type_counts = df['Type'].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig_bar2 = px.bar(type_counts, x="Type", y="Count",
                          text="Count", title="Input Type Count")
        fig_bar2.update_traces(textposition="outside", hovertemplate="%{x}: %{y}")
        st.plotly_chart(fig_bar2)

        #  Pie Charts with Hover
        st.subheader("Classification Distribution")
        class_counts_pie = df['Classification'].value_counts().reset_index()
        class_counts_pie.columns = ['Classification', 'Count']
        fig1 = px.pie(class_counts_pie, names='Classification', values='Count',
                      title='Classification Distribution', hole=0.3)
        fig1.update_traces(textinfo='percent+label', hovertemplate='%{label}: %{value} entries (%{percent})')
        st.plotly_chart(fig1)

        st.subheader("Input Type Distribution")
        type_counts_pie = df['Type'].value_counts().reset_index()
        type_counts_pie.columns = ['Type', 'Count']
        fig2 = px.pie(type_counts_pie, names='Type', values='Count',
                      title='Input Type Distribution', hole=0.3)
        fig2.update_traces(textinfo='percent+label', hovertemplate='%{label}: %{value} entries (%{percent})')
        st.plotly_chart(fig2)

    else:
        st.info("Database is empty.")
