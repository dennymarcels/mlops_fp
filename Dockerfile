FROM python:3.12-slim

# Copy the entire project
COPY . mlops/

# Install the package with dependencies
RUN pip install ./mlops

# Set the working directory
WORKDIR /mlops

# Expose the port gunicorn will listen on
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:8000", "app.main:app"]