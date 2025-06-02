# Deepfake Live Camera for macOS M2
Developed by Ilyas AKKUS
Real-time face swap application optimized for Apple Silicon (M1/M2/M3) with OBS Studio integration.

## Features

- Real-time face swapping using a single source image
- Optimized for Apple Silicon using CoreML
- OBS Studio virtual camera integration
- Low latency for live streaming
- Simple GUI interface

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10 (IMPORTANT: Must use Python 3.10, not newer versions)
- OBS Studio
- Webcam

## Installation

### 1. Install Python 3.10

```bash
# Install Python 3.10 using Homebrew
brew install python@3.10

# Verify installation
python3.10 --version
```

### 2. Create Virtual Environment

```bash
# Create virtual environment with Python 3.10
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt
```

### 4. Download Models

Models will be automatically downloaded on first run (~300MB).

## Usage

### Basic Usage

1. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Run the application:
   ```bash
   python3.10 run.py --execution-provider coreml
   ```

3. Select source face image
4. Click "Live" to start webcam mode
5. Use OBS Studio to capture the output window

### OBS Studio Integration

1. Open OBS Studio
2. Add a new "Window Capture" source
3. Select the Deepfake Live Camera preview window
4. Alternatively, use "Display Capture" to capture the entire screen

## Tips for Best Results

- Use high-quality source images with clear face visibility
- Ensure good lighting for webcam
- Source image should have similar angle to your webcam position
- Close-up portraits work best

## Troubleshooting

### Common Issues

1. **"_tkinter not found" error**:
   ```bash
   brew reinstall python-tk@3.10
   ```

2. **Python version conflicts**:
   ```bash
   # Always use python3.10 explicitly
   python3.10 run.py --execution-provider coreml
   ```

3. **Performance issues**:
   - Make sure to use CoreML execution provider for Apple Silicon
   - Close unnecessary applications
   - Reduce preview window size if needed

## M2 Optimizations and Performance

### Performance Modes

The application supports three performance modes that can be set using the `--performance` flag:

```bash
python3.10 run.py --performance [fast|balanced|quality]
```

- **fast**: Best performance, lower quality
  - Reduced resolution (40% of original)
  - Higher frame skip rate
  - Recommended for live streaming

- **balanced**: Good balance between performance and quality
  - Medium resolution (50% of original)
  - Moderate frame skip rate
  - Good for general use

- **quality**: Highest quality, lower performance
  - Higher resolution (80% of original)
  - Lower frame skip rate
  - Best for recording

### M2-Specific Optimizations

1. **CoreML Settings**:
   - Uses Neural Engine for inference
   - Optimized model conversion for M2
   - Efficient memory management

2. **System Recommendations**:
   - Set Mac power settings to "High Performance"
   - Ensure proper cooling
   - Close resource-intensive applications
   - Keep macOS updated

3. **Performance Tips**:
   - Use "fast" mode for live streaming
   - Disable lip sync if not needed
   - Use lower resolution webcam if available
   - Monitor FPS counter in the application

### Advanced Settings

You can combine performance settings with other options:

```bash
# Fast mode with specific camera
python3.10 run.py --performance fast --camera-id 1

# Balanced mode without GUI
python3.10 run.py --performance balanced --no-gui --source path/to/image.jpg
```

## License

This project is licensed under AGPL-3.0 License. 
