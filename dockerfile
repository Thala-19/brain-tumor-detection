# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the 'app' directory from your local machine to the container's /app directory
COPY ./app /app

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "script.py"]