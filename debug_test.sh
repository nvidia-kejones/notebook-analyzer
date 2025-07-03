#!/bin/bash
TEMP_DIR="/tmp/debug_test_$$"
mkdir -p "$TEMP_DIR"

cat > "$TEMP_DIR/legitimate.ipynb" << 'NOTEBOOK'
{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import torch\n",
        "model = torch.nn.Linear(10, 1)\n",
        "print('Hello World')"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
NOTEBOOK

response_file="$TEMP_DIR/response"
http_code=$(curl -s -m 10 -o "$response_file" -w "%{http_code}" \
    -F "file=@$TEMP_DIR/legitimate.ipynb" \
    "http://localhost:8080/api/analyze" 2>/dev/null)

echo "HTTP Code: $http_code"
if [ -f "$response_file" ]; then
    echo "Response file exists, size: $(wc -c < "$response_file") bytes"
    echo "First 200 chars:"
    head -c 200 "$response_file"
    echo
    echo "Contains success: $(grep -c "success" "$response_file" 2>/dev/null || echo 0)"
fi

rm -rf "$TEMP_DIR"
