FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY . /app

# Install Poetry and project dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Expose the API port
EXPOSE 8000

# Use the launch script as the entrypoint
CMD ["poetry", "run", "python", "launch"]
