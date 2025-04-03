from agno.agent import Agent
from agno.models.google.gemini import Gemini
from agno.team.team import Team

from agno.tools import Toolkit
from agno.utils.log import logger

import os
import subprocess
from datetime import datetime
from typing import List, Optional, Any # Added Any for client flexibility

# --- PIL & IO Imports ---
from PIL import Image
from io import BytesIO

from google import genai
from google.genai import types

model_name = 'gemini-2.0-flash'

# --- Updated ImageTools Class ---
class ImageTools(Toolkit):
    # Accept the genai client instance during initialization
    def __init__(self):
        super().__init__(name="image_tools")
        self.client = genai.Client()

        self.base_folder = "images"
        # Register the new combined function
        self.register(self.generate_img_and_save)

    # Renamed and combined function
    def generate_img_and_save(self,
                              prompt: str,
                             ) -> str:
        """
        Generates an image using the Gemini API based on a prompt and saves it
        to a file with a timestamp. Also prints any accompanying text response.

        Args:
            prompt (str): The text prompt for image generation.
            filename_prefix (str): A prefix for the filename.
            model_name (str): The specific Gemini model to use for generation.
            generation_config (Optional[dict]): Configuration for the generation API call
                                                 (e.g., response modalities).
        Returns:
            str: The path to the saved image file or an error message starting with "Error:".
        """
        image_model = 'gemini-2.0-flash-exp-image-generation'
        logger.info(f"Attempting to generate and save image for prompt: '{prompt}...'")
        logger.info(f"Using model: {image_model}")
        print("given prompt: ", prompt)

        if self.client is None:
            logger.error("Cannot generate image: GenAI client is not configured.")
            return "Error: GenAI client not available."


        try:
            # --- Call Gemini API ---
            logger.info(f"Calling Gemini API (model: {model_name})...")
            response = self.client.models.generate_content(
                model=image_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            logger.info("Gemini API call successful.")
            # logger.debug(f"Full API Response: {response}") # Can be very verbose

        # Specific API errors can be caught if known, otherwise general Exception
        except Exception as api_e:
            logger.error(f"Gemini API call failed: {api_e}", exc_info=True)
            return f"Error: API call failed: {api_e}"

        # --- Process Response ---
        image_blob = None
        text_response = ""
        try:
            if not response.candidates:
                 logger.warning("API response received, but contains no candidates.")
                 return "Error: No candidates found in API response."

            # Process parts from the first candidate
            candidate = response.candidates[0]
            if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
                logger.warning("API response candidate has unexpected structure.")
                return "Error: Invalid response structure (no content/parts)."

            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text is not None:
                    logger.info(f"Found text part: '{part.text}'")
                    # Print text part here, similar to original script
                    print(f"API Text Response: {part.text}")
                    text_response += part.text + "\n" # Collect text if needed later

                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    logger.info("Found image data part (blob).")
                    # Assuming only one image blob is expected per call here
                    if image_blob is not None:
                         logger.warning("Multiple image blobs found, using the first one.")
                    else:
                         # Ensure it looks like a blob (has data and mime_type)
                         if hasattr(part.inline_data, 'data') and hasattr(part.inline_data, 'mime_type'):
                             image_blob = part.inline_data
                         else:
                             logger.warning(f"Part has inline_data but missing 'data' or 'mime_type': {part.inline_data}")

            if image_blob is None:
                logger.warning("API response processed, but no valid image blob found.")
                return "Error: No image data found in the API response."

        except Exception as proc_e:
             logger.error(f"Failed to process API response parts: {proc_e}", exc_info=True)
             return f"Error: Failed processing response: {proc_e}"

        filename_prefix= prompt[:20]
        # --- Save Image (Logic adapted from save_image_from_blob) ---
        logger.info(f"Attempting to save image with prefix '{filename_prefix}'")
        try:
            # 1. Create the base folder if it doesn't exist
            os.makedirs(self.base_folder, exist_ok=True)
            logger.debug(f"Ensured directory exists: {self.base_folder}")

            # 2. Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            extension = ".png" # Default
            if image_blob.mime_type and image_blob.mime_type.startswith("image/"):
                # Sanitize extension (e.g., image/jpeg -> .jpeg)
                ext_part = image_blob.mime_type.split("/")[-1].split(";")[0].strip().lower()
                if ext_part and ext_part.isalpha(): # Basic check
                   extension = f".{ext_part}"
                   logger.debug(f"Determined extension '{extension}' from mime type '{image_blob.mime_type}'")
                else:
                    logger.warning(f"Could not reliably determine extension from mime type '{image_blob.mime_type}', using default '.png'.")

            filename = f"{timestamp}{extension}"
            filepath = os.path.join(self.base_folder, filename)
            logger.debug(f"Generated filepath: {filepath}")

            # 3. Process and save the image
            image_data = image_blob.data
            image = Image.open(BytesIO(image_data))
            image.save(filepath)

            logger.info(f"Successfully saved image to: {filepath}")
            # Optionally display image here if the tool should also handle that
            # try:
            #    logger.info("Displaying generated image...")
            #    image.show()
            # except Exception as show_e:
            #    logger.warning(f"Could not display image automatically: {show_e}")

            return filepath # Return the path on success

        except FileNotFoundError:
             logger.error(f"Error creating/accessing directory '{self.base_folder}'. Check permissions.", exc_info=True)
             return f"Error: Could not access or create directory '{self.base_folder}'."
        except AttributeError as e:
             # This might happen if image_blob wasn't a proper blob after all
             logger.error(f"Error accessing image blob attributes (data/mime_type): {e}", exc_info=True)
             return f"Error: Invalid image blob structure received: {e}"
        except Exception as save_e:
            logger.error(f"Failed to save image file: {save_e}", exc_info=True)
            return f"Error: Failed to save image: {save_e}"




art_detail_agent = Agent(
    name="Art Detail Agent",
    role="You will be given a single topic and generate a full detailed image description, optimize for online marketplaces like Amazon Print on Demand, we want catchy images optimized for sale.",
    model=Gemini(id=model_name),
    exponential_backoff=True,
    delay_between_retries=2
)

validation_agent = Agent(
    name="Art Validation Agent",
    role="Make sure that the given prompt is detailed enough to generate a good image, make sure that the description is at least 1 line. If the description is not good enough, rewrite it",
    model=Gemini(id=model_name),
    exponential_backoff=True,
    delay_between_retries=2
)

image_making_agent = Agent(
    name="Image Creation Agent",
    role="Given the input prompt, use the tools to generate an image.",
    model=Gemini(id=model_name),
    exponential_backoff=True,
    delay_between_retries=2,
    tools=[ImageTools()], show_tool_calls=True, markdown=True
)

image_generation_team = Team(
    name="Image Generation Team",
    mode="coordinate",
    model=Gemini(id=model_name),
    members=[art_detail_agent, validation_agent, image_making_agent],
    show_tool_calls=True,
    markdown=True,
    description="You are an image generator tool that can make a description for an given image topic.",
    instructions=[
        "Identify the topic of the user's art suggestion, if the input is short, route to the Art Detail Agent to get a full description.",
        "When the prompt is ready, call the Image Creation Agent to generate an image.",
        "Make sure that the Image Creation Agent is called at the end."
    ],
    show_members_responses=True,
    # send_team_context_to_members=True,
    # exponential_backoff=True,
    # delay_between_retries=2,
)


if __name__ == "__main__":
    # Ask "How are you?" in all supported languages
    image_generation_team.print_response("Fox catching fish, cutey cartoon style in summer time")
    # image_tools = ImageTools()

    # image_tools.generate_img_and_save("A Fox cleaning the ears of a bunny. Make the image anime style, cute, focus on nature, remove all backgrounds.")
