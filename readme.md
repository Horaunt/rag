# Streamlit Q&A Assistant with RAG (Retrieval-Augmented Generation)

This project implements a Retrieval-Augmented Generation (RAG) powered Q&A assistant using Streamlit. The assistant uses external knowledge sources (e.g., a document database, pre-trained models, etc.) to provide answers to users' questions. The backend logic utilizes a retrieval system to fetch relevant information, which is then used to generate coherent and contextually relevant answers.

## Features
- **Document-based Q&A**: Users can ask questions, and the system will retrieve relevant documents or information from a pre-indexed database.
- **AI-Powered Responses**: The system generates answers based on the retrieved information, using a generative model such as GPT.
- **User Interface**: Built using Streamlit, allowing users to interact with the assistant through a simple web interface.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Horaunt/streamlit-qa-assistant.git
cd streamlit-qa-assistant
````

### 2. Install dependencies

Make sure you have Python 3.8+ installed. You can use `pip` or `conda` to set up your environment. First, create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Now, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

After the dependencies are installed, you can start the Streamlit app by running:

```bash
streamlit run app.py
```

The app should now be running at `http://localhost:8501` (or the URL shown in your terminal).

## Requirements

* Python 3.8+
* Streamlit
* PyTorch
* Transformers
* Other dependencies listed in `requirements.txt`

## Dependencies

Make sure to have all required Python libraries by installing them with the following command:

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`:

```txt
streamlit==1.20.0
torch==2.0.1
transformers==4.28.0
faiss-cpu==1.7.2
```

## Usage

1. **Ask a Question**: Type your question in the input box.
2. **Generate Answer**: The system will retrieve relevant information from a knowledge base or document and generate an answer.
3. **View Results**: The answer will be displayed on the interface.

## Troubleshooting

If you run into any issues related to dependencies or versions, make sure to check if you have the correct versions installed by running:

```bash
pip freeze
```

Ensure that all dependencies in `requirements.txt` are properly installed.

### Common Errors:

* **RuntimeError: no running event loop**: This can happen due to an event loop issue in the environment. Try re-running the app in a clean virtual environment.

* **Torch issues**: Make sure PyTorch is installed correctly, and the correct version is being used for your setup.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

* This project uses the [Streamlit](https://streamlit.io/) framework for the front-end UI.
* PyTorch and [Transformers](https://huggingface.co/transformers/) are used for the backend NLP model.

```

### How to Customize:

- Modify the **title** and **features** section according to your app’s functionalities.
- Add any specific dependencies or setup instructions unique to your project.
- Update the **license** section if necessary.
```
