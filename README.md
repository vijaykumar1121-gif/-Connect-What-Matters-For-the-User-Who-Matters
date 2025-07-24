# Persona-Driven Document Intelligence

## Overview
This project extracts and prioritizes relevant sections from PDFs based on a user persona and job-to-be-done.

## How to Run

### Build Docker Image
```
docker build -t doc-intel .
```

### Run the Container
```
docker run --rm -v %cd%/test_docs:/app/test_docs doc-intel
```

The output will be saved as `challenge1b_output.json` in the container (and in your project directory if you mount the whole project).

## Files
- `src/`: Source code
- `requirements.txt`: Python dependencies
- `challenge1b_output.json`: Sample output
- `approach_explanation.md`: Methodology
- `Dockerfile`: For containerized execution
- `test_docs/`: Sample PDFs 