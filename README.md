# Enterprise Data Assistant

Enterprise Data Assistant is an advanced AI-driven tool designed to assist enterprise users in managing, querying, and visualizing data from multiple databases. By combining advanced natural language processing with powerful data analysis capabilities, it empowers users across all levels of the organization to explore data.

## Table of Contents

* [Documentation](#documentation)
* [Usage](#usage)
* [Configuration](#configuration)
* [Development Environment Setup](#development-environment-setup)
* [Managing Dependencies](#managing-dependencies)
* [Troubleshooting](#troubleshooting)
* [Contact](#contact)

## Documentation

## Usage

To run the Enterprise Data Assistant:

```bash
python -m streamlit run chat_app.py
```

## Configuration

### Prerequisites

* Python 3.11.7
* pip (usually comes with Python)
* venv module (usually comes with Python)
* Certificate/SSL file (`ca-bundle.crt`) for Stork API certificate verification

To configure the certificate verification, set the following environment variables:

```bash
export NETWORK_ENV=uat
export REQUESTS_CA_BUNDLE="/yourlocalpath/ca-bundle.crt"
```

Replace `"/yourlocalpath/ca-bundle.crt"` with the actual path to your `ca-bundle.crt` file. Download the crt file from https://confluence.sgp.dbs.com:8443/dcifcnfl/display/~utsavsarkar/Issues?preview=/696523909/708889090/ca-bundle%203.crt

### Environment Variables

Create a `.env` file in the root directory and add the following environment variables:

* `SQLITE_DB_PATH`: path to your database
* `STORK_PROVIDER`: your Stork provider
* `STORK_PROVIDER_ID`: your Stork provider ID
* `STORK_MODEL_ID`: your Stork model ID
* `VERTEXAI_PROJECT_ID`: your Vertex AI project ID
* `VERTEXAI_LOCATION`: your Vertex AI location name
* `GEMINI_MODEL_NAME`: your Gemini model name

## Development Environment Setup

### Virtual Environment Setup

1. Clone the repository:
   ```bash
   git clone ssh://git@bitbucket.sgp.dbs.com:7999/ada_jbrg/test-suite.git
   cd feature/texttosqlgenerator
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   * On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   * On macOS and Linux:
     ```bash
     source .venv/bin/activate
     ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Installing ada-genai Package

To install the `ada-genai` package, use the following command:
```bash
pip3 install ada-genai --index-url https://jfrogbin.sgp.dbs.com:8443/artifactory/api/pypi/pypi-all/simple --trusted-host jfrogbin.sgp.dbs.com
```
Note: Ensure you have the necessary permissions to access this internal PyPI server.


## Managing Dependencies

* View installed packages:
  ```bash
  pip list
  ```
* Add a new package:
  ```bash
  pip install package-name
  ```
* Update `requirements.txt` after adding new packages:
  ```bash
  pip freeze > requirements.txt
  ```

## Troubleshooting

If you encounter issues with the virtual environment:

1. Ensure you're using Python 3.11.7
2. Delete the `.venv` folder and recreate it
3. Make sure all required packages are listed in `requirements.txt`

For any persistent issues, please open an issue in the project repository.




### Architecture Diagram:
![WhatsApp Image 2024-08-20 at 10 09 14](https://github.com/user-attachments/assets/1a04e124-204b-4447-bfe0-f49d52a099a3)


Utsav Sarkar - utsavsarkar0703@gmail.com
