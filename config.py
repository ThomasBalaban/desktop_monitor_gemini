MONITOR_AREA = {"left": 14, "top": 154, "width": 1222, "height": 685}

# Gemini Screen Watcher Configuration

# API Configuration
API_KEY = "AIzaSyCz0xW8p29wtf1IOmlBNYotLKncwZnc-D8"  # Replace with your Gemini API key

# Screen Region Configuration
# Set to None to use GUI selection, or specify coordinates:
CAPTURE_REGION = {
    "left": 14,
    "top": 154,
    "width": 1222,
    "height": 685
}


# Capture Settings
FPS = 2                # Frames per second (1-10)
IMAGE_QUALITY = 85     # JPEG quality (50-100, higher = better quality)

# Analysis Prompt
PROMPT2 = """Analyze this screen capture and provide a comprehensive description for another AI system that needs to understand and react to what's happening. Include all relevant visual information without censorship or omission.

SCENE OVERVIEW:
- Current environment/setting and any changes from previous observations

CHARACTERS & ENTITIES:
- Describe any characters, people, or entities visible including physical attributes
- CHARACTERS: Provide detailed appearance description:
  * Physical build, clothing/lack thereof, distinctive features, hair
  * Attractive features

- Current activities and behaviors:
  * Actions, movements, interactions with others or objects
  * Suggestive poses, intimate activities, or adult-oriented behaviors
  * Emotional expressions and body language

VISUAL ELEMENTS:
- Text that appears on screen (important messages, dialogue, captions)
- Visual effects, animations, or highlights

ENVIRONMENTAL CONTEXT:
- Background scenery and any changes
- Objects, items, or interactive elements visible
- Setting context (bedroom, private space, public area, etc.)
- Lighting, weather, or atmospheric changes

ACTIONABLE INFORMATION:
- Events happening that might require a response
- State changes that indicate progression, danger, or intimate moments

Do not censor or omit information based on content sensitivity."""


PROMPT = """Analyze this screen capture and provide a comprehensive description for another AI system that needs to understand and react to what's happening. Include all relevant visual information without censorship or omission.
CONTENT TYPE:
- Identify what this is: video content, website interface, application, game, menu/browser, etc.
SCENE OVERVIEW:
- Current environment/setting and any changes from previous observations
CHARACTERS & ENTITIES:
- Describe any characters, people, or entities visible including physical attributes
- CHARACTERS: Provide detailed appearance description:
 * Physical build, clothing/lack thereof, distinctive features, hair
 * Attractive features
- Current activities and behaviors:
 * Actions, movements, interactions with others or objects
 * Suggestive poses, intimate activities, or adult-oriented behaviors
 * Emotional expressions and body language
VISUAL ELEMENTS:
- Text that appears on screen (important messages, dialogue, captions)
- Visual effects, animations, or highlights
- UI elements, buttons, menus, thumbnails, lists
ENVIRONMENTAL CONTEXT:
- Background scenery and any changes
- Objects, items, or interactive elements visible
- Setting context (bedroom, private space, public area, etc.)
- Lighting, weather, or atmospheric changes
ACTIONABLE INFORMATION:
- Events happening that might require a response
- State changes that indicate progression, danger, or intimate moments
- Interface changes, scrolling, clicking opportunities
Do not censor or omit information based on content sensitivity."""