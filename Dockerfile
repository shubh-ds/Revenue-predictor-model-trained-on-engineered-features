# Stage 1: Base Stage
FROM python:3.12 AS build

# Work directory
WORKDIR /app

# Copy code to work directory
COPY flask_app/ /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2:  Run Stage
FROM python:3.12 AS run

# Work directory
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=build /app /app

# Expose port
EXPOSE 5000

# Run command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]