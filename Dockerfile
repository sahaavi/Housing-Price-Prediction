FROM python:3.9
# Copy the application code to the container
COPY . /app
# Set the working directory
WORKDIR /app
# Install application dependencies
RUN pip install -r requirements.txt
# Start the application
CMD ["python3", "application.py"]
