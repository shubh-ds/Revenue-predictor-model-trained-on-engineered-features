# Base image
FROM python:3.12-slim

# Work directory
WORKDIR /app

# Copy code to work directory
COPY flask_app/ /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]