#!/bin/bash

# Build the image
docker build -t lazyup-ai .

# Run the container
docker run -d \
  --name lazyup-ai-container \
  --restart unless-stopped \
  -p 1234:1234 \
  -v "$(pwd)/device_input:/app/device_input" \
  -v "$(pwd)/process_names:/app/process_names" \
  -v "$(pwd)/tokens_dictionary.json:/app/tokens_dictionary.json" \
  -v "$(pwd)/gui/tokens_dictionary.json:/app/gui/tokens_dictionary.json" \
  -v "$(pwd)/process_names/process_name_tokenizing/tokens_dictionary.json:/app/process_names/process_name_tokenizing/tokens_dictionary.json" \
  lazyup-ai
