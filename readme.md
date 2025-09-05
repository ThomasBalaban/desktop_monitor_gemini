# Gemini Screen Watcher

A Python application that captures a selected screen region and streams it to Google's Gemini 2.0 Flash for real-time AI analysis.

## Features

- **Interactive Screen Region Selection**: Click and drag to select any area of your screen
- **Real-time Streaming**: Continuously captures and analyzes the selected region
- **Customizable Analysis**: Write your own prompts to guide Gemini's analysis
- **Adjustable Frame Rate**: Control how frequently frames are captured (1-10 FPS)
- **Live Response Display**: See Gemini's analysis in real-time with timestamps

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
asyncio-extras==1.3.2
websockets==12.0
mss==9.0.1
Pillow==10.1.0
opencv-python==4.8.1.78
numpy==1.24.3
tkinter
```

### System Requirements

- **Windows**: No additional setup required
- **macOS**: You may need to grant screen recording permissions in System Preferences > Security & Privacy
- **Linux**: Install `python3-tk` if not already available:
  ```bash
  sudo apt-get install python3-tk  # Ubuntu/Debian
  sudo yum install tkinter         # CentOS/RHEL
  ```

## Setup Instructions

### 1. Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key (starts with `AIza...`)

### 2. Install Dependencies

```bash
# Clone or download the project files
# Navigate to the project directory

# Install required packages
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python screen_watcher.py
```

## How to Use

### 1. Configure API Key
- Paste your Gemini API key in the "Gemini API Key" field
- The key will be hidden for security

### 2. Select Screen Region
- Click "Select Region" button
- Your screen will show a red overlay
- Click and drag to select the area you want to monitor
- Release to confirm the selection

### 3. Customize Analysis
- Edit the prompt in the "Analysis Prompt" text area
- Example prompts:
  - `"Watch this screen region and describe what you see. Alert me of any significant changes."`
  - `"Monitor this dashboard for any error messages or alerts."`
  - `"Describe the activity in this application window."`
  - `"Track changes in this data visualization and summarize trends."`

### 4. Adjust Settings
- Set the capture frame rate (1-10 FPS)
- Higher FPS = more responsive but uses more API calls
- 2-3 FPS is usually sufficient for most monitoring tasks

### 5. Start Monitoring
- Click "Start Watching"
- The status will show "Connecting..." then "Streaming..."
- Gemini's responses will appear in the bottom text area with timestamps
- Click "Stop" to end the session

## Use Cases

- **Dashboard Monitoring**: Watch data dashboards for changes or alerts
- **Application Testing**: Monitor app behavior during testing
- **Live Presentations**: Get AI commentary on presentations or videos
- **Gaming**: Analyze gameplay or get tips on game status
- **Development**: Monitor build processes or log outputs
- **Video Calls**: Get meeting summaries or note important moments

## Troubleshooting

### Connection Issues
- Verify your API key is correct
- Check your internet connection
- Ensure you have sufficient API quota

### Screen Capture Issues
- **macOS**: Grant screen recording permissions
- **Linux**: Ensure X11 is running (Wayland may have issues)
- Try selecting a smaller region if capture fails

### Performance Issues
- Reduce the FPS setting
- Select a smaller screen region
- Close unnecessary applications

### API Errors
- Check the console output for detailed error messages
- Verify your API key has access to Gemini 2.0 Flash
- Check if you've exceeded rate limits

## Technical Details

### Architecture
- **GUI**: Tkinter for cross-platform interface
- **Screen Capture**: MSS (Multi-Screen Screenshots) for fast screen capture
- **Image Processing**: PIL for image manipulation and compression
- **WebSocket**: Real-time bidirectional communication with Gemini API
- **Async**: Asyncio for handling concurrent operations

### API Integration
- Uses Gemini 2.0 Flash Multimodal Live API
- Streams JPEG-compressed images at configurable intervals
- Maintains persistent WebSocket connection for low latency
- Handles both text prompts and image data in each request

### Privacy & Security
- API key is stored only in memory during runtime
- Screen captures are sent directly to Google's servers
- No local storage of captured images
- Connection uses secure WebSocket (WSS)

## Customization

You can modify the code to:
- Add audio capture alongside video
- Save interesting frames locally
- Add voice output using Gemini's TTS
- Create automated actions based on analysis
- Add multiple region monitoring
- Integrate with other APIs or services

## License

This project is provided as-is for educational and personal use. Please respect Google's API terms of service and usage limits.

# build

``` pyinstaller "Gemini Screen Watcher.spec"```