PROMPT4 = """Describe this screen in 1-2 paragraphs. Start with content type. Describe characters with full detail: appearance, clothing/revealing outfits, poses (especially suggestive ones), expressions. Include UI/text elements. Be explicit about provocative content - e.g., "leaning forward to emphasize cleavage" not "casual pose." No euphemisms."""

PROMPT3 = """Analyze this screen capture and provide a comprehensive description in the form of one or two paragraphs. Include all relevant visual information without censorship or omission. start with what this is: video content, website interface, application, game, menu/browser, etc. Do not censor or omit information based on content sensitivity.

Stuff to look for
- 
SCENE OVERVIEW:
- Current environment/setting and any changes from previous observations
CHARACTERS & ENTITIES:
- Describe any characters, people, or entities visible including physical attributes
- CHARACTERS: Provide detailed appearance description:
 * Physical build, clothing/lack thereof, distinctive features, hair
 * Attractive features
 * if a character is doing a sexy pose, do a brief description of the pose
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


Example 1 (Music Video): "This appears to be a video clip from a streaming platform or personal content channel, centered on a young woman in a bedroom-style setting. She has long brown hair and is wearing a red bikini top paired with denim shorts. The room is softly lit with string lights and decorative pillows in the background, giving it a relaxed, intimate aesthetic. She is seated on a bed with white sheets, framed in a way that emphasizes her figure within the environment.

The subject is leaning slightly forward with her legs angled to the side, positioning her body to accentuate both her buttocks and cleavage. Her bikini top is cut low, drawing attention to her chest, while the denim shorts are snug and slightly hiked up, revealing a generous portion of her thighs and the curve of her hips. She looks directly into the camera with a subtle smile, suggesting a performative or flirtatious intent. The pose and framing are clearly designed to be visually enticing, evoking the kind of content typically intended to capture and hold viewer attention."

Example 2 (Video Game Gameplay): "The image shows a detailed character creation screen from a dark fantasy role-playing game, reminiscent of Elden Ring in tone and visual fidelity. At the center is a female character model — a pale elf-like figure with long platinum blonde hair and pointed ears — standing in a quiet forest glade rendered with cinematic lighting and ambient fog. She wears minimal leather armor designed more for aesthetic flair than protection: a cropped chestpiece that leaves her midriff exposed and low-cut bottoms that reveal the tops of her thighs. Her stance is confident and stylized, contributing to a sense of intentional visual impact.

UI elements surround the character on all sides. On the left are RPG attribute sliders — Strength (12), Dexterity (18), Intelligence (14) — and tabs labeled “Body Type,” “Clothing Style,” and “Pose Selection,” with “Seductive” currently chosen. That selection is reflected in her posture: one hip cocked outward, spine slightly arched, and one arm resting on her waist, enhancing the prominence of her chest and hips. The leather top is shaped to contour her breasts with minimal coverage, while the angle of the lighting adds definition to her figure. Buttons at the bottom — “Randomize,” “Save,” and “Continue” — suggest the player is about to lock in this visual identity before entering the game world."

"""

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