@echo off
REM Build the image
docker build -t lazyup-ai .

REM Run the container
docker run -d ^
  --name lazyup-ai-container ^
  --restart unless-stopped ^
  -p 1234:1234 ^
  -v "%CD%\device_input:/app/device_input" ^
  -v "%CD%\process_names:/app/process_names" ^
  -v "%CD%\tokens_dictionary.json:/app/tokens_dictionary.json" ^
  -v "%CD%\gui\tokens_dictionary.json:/app/gui/tokens_dictionary.json" ^
  -v "%CD%\process_names\process_name_tokenizing\tokens_dictionary.json:/app/process_names/process_name_tokenizing/tokens_dictionary.json" ^
  lazyup-ai
