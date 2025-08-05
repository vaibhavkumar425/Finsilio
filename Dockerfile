# Official lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container to /app
WORKDIR /app

RUN pip install uv

# Copy and install our Python dependencies first
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# The commands to run when the container starts
# This command starts FastAPI server using uvicorn for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]