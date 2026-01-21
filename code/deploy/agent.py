import json
import time
import base64
import re
import ast
from io import BytesIO
from PIL import Image
from typing import Dict, Tuple

from deploy.prompt import AGENT_PROMPT 

class CUAgent:
    def __init__(
        self, 
        client, 
        model, 
        max_steps=30, 
        screenshot_size=(1280, 720), 
        prompt=AGENT_PROMPT
    ):
        self.retry_click_elements = []
        self.history = []
        self.history_cut_off = 10
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.screenshot_size = screenshot_size # (width, height)
        self.prompt = prompt
        self.steps = 0
        
        print(f"Initializing CU-Agent with model: {model}")

    def encode_image(self, image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    def extract_plan_and_action(self, text):

        text = text.strip()
        

        if text.startswith("{") and text.endswith("}"):
            try:

                data = ast.literal_eval(text)
                
                if isinstance(data, dict):

                    action_type = data.get('action', 'click')
                    

                    position = data.get('position', [])
                    
                    final_action_str = ""
                    
                    if position and len(position) >= 2:

                        norm_x, norm_y = position[0], position[1]
                        abs_x = int(norm_x * self.screenshot_size[0])
                        abs_y = int(norm_y * self.screenshot_size[1])
                        

                        final_action_str = f"{action_type} ({abs_x}, {abs_y})"
                    else:

                        value = data.get('value', '')
                        if value:
                            final_action_str = f"{action_type} \"{value}\""
                        else:
                            final_action_str = action_type


                    return f"ShowUI Raw: {text}", final_action_str
            
            except Exception as e:
                print(f"Attempt to parse ShowUI dictionary failed: {e}")
        
        pattern = r"(?:Action|Actions)\s*:"
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        
        if matches:
            last_match = matches[-1]
            split_start = last_match.start()
            split_end = last_match.end()
            
            plan_str = text[:split_start].strip()
            action_str = text[split_end:].strip()
            

            plan_str = plan_str.strip('{}').strip()
            plan_str = re.sub(
                r"^(?:Plan|Thought|Thoughts|Your Plan\/Thought process|thoughts?)\s*:", 
                "", 
                plan_str, 
                flags=re.IGNORECASE
            ).strip()
            

            action_str = action_str.strip('{}').strip()
            
            return plan_str, action_str
        else:

            return text.strip(), "None"

    def get_plan_instruction(self, task_input, feedback=""):
        base_prompt = f"{self.prompt}\n\n{task_input}"
        if feedback:
            base_prompt += feedback
        return base_prompt

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, str]:

        image_bytes = obs['screenshot']
        image_file = BytesIO(image_bytes)
        try:
            view_image = Image.open(image_file)
            self.screenshot_size = view_image.size
        except Exception as e:
            print(f"Warning: Image load failed: {e}")

            if not self.screenshot_size:
                self.screenshot_size = (1280, 720)
        
        base64_image = self.encode_image(image_bytes)

        feedback = ""
        try_time = 3 
        
        while try_time > 0:
            try:
                full_prompt = self.get_plan_instruction(instruction, feedback)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.1, 
                )
                
                output_text = response.choices[0].message.content
                

                plan_text, action_text = self.extract_plan_and_action(output_text)
                

                if action_text == "None" or not action_text:
                    print(f"Invalid model response. Retrying ({try_time-1} left)...")
                    print(f"Raw Output: {output_text}") 
                    
                    feedback = f"\n\nNote: Please output the standard format containing 'Action:' or the standard JSON dictionary format."
                    try_time -= 1
                    
                    if try_time == 0:
                        return plan_text, "Error: Model failed to follow format after retries."
                    continue 
                
                return plan_text, action_text

            except Exception as e:
                print(f"Error during prediction (Retry {try_time}): {e}")
                try_time -= 1
                time.sleep(1)
                if try_time == 0:
                    return "Error", f"API Error: {str(e)}"

        return "Error", "Error: Unknown failure"