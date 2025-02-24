import json
import re
from typing import Optional, Dict, Any, Union, List, Tuple
from pydantic import BaseModel, Field, field_validator
import httpx
from variables import *
from metaprompt_router import metaprompt_router

class LLMResponse(BaseModel):
    initial_prompt_evaluation: str = Field(..., description="Evaluation of the initial prompt")
    refined_prompt: str = Field(..., description="The refined version of the prompt")
    explanation_of_refinements: Union[str, List[str]] = Field(..., description="Explanation of the refinements made")
    response_content: Optional[Union[Dict[str, Any], str]] = Field(None, description="Raw response content")

    @field_validator('response_content', mode='before')
    def validate_response_content(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {"raw_content": v}
        return v

    @field_validator('initial_prompt_evaluation', 'refined_prompt', 'explanation_of_refinements')
    @classmethod
    def clean_text_fields(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(v, str):
            return v.strip().replace('\\n', '\n').replace('\\"', '"')
        elif isinstance(v, list):
            return [item.strip().replace('\\n', '\n').replace('\\"', '"').replace('•', '-')
                   for item in v if isinstance(item, str)]
        return v

class PromptRefiner:
    def __init__(self, api_endpoint: str, api_key: Optional[str], meta_prompts: dict, metaprompt_explanations: dict):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.client = httpx.Client(timeout=120)
        self.meta_prompts = meta_prompts
        self.metaprompt_explanations = metaprompt_explanations

    def _make_api_request(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.8, max_tokens: int = 3000) -> Dict:
        """Make a request to the OpenAI-compatible API endpoint with retry logic."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        max_retries = 3
        retry_delay = 1.0
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=120  # Explicit timeout
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                last_error = e
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                continue
            except Exception as e:
                last_error = e
                break

        error_msg = f"API request failed after {max_retries} attempts. Last error: {str(last_error)}"
        print(error_msg)  # Log the error
        raise Exception(error_msg)

    def _clean_json_string(self, content: str) -> str:
        """Clean and prepare JSON string for parsing."""
        try:
            # First, normalize the content
            content = content.strip()

            # Handle common JSON formatting issues
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            content = content.replace('•', '-')  # Handle bullet points
            content = content.replace('\\"', '"')  # Handle escaped quotes
            content = re.sub(r'(?<!\\)\\n', ' ', content)  # Handle unescaped newlines
            content = re.sub(r',(\s*[}\]])', r'\1', content)  # Remove trailing commas

            # Basic cleanup
            content = content.strip()
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace

            # Simple brace balancing
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces > close_braces:
                content += '}'

            return content
        except Exception as e:
            print(f"Error cleaning JSON string: {str(e)}")
            return content

    def _parse_response(self, response_content: str) -> dict:
        """Parse the LLM response with enhanced error handling."""
        try:
            # First try to find JSON within tags
            json_match = re.search(r'<json>\s*(.*?)\s*</json>', response_content, re.DOTALL)
            if json_match:
                json_str = self._clean_json_string(json_match.group(1))
                try:
                    # Try to parse the cleaned JSON string
                    parsed_json = json.loads(json_str)
                    print(f"Initial JSON parse: {parsed_json}")

                    # Handle nested JSON strings
                    if isinstance(parsed_json, str):
                        try:
                            parsed_json = json.loads(parsed_json)
                            print(f"Nested JSON parse: {parsed_json}")
                        except json.JSONDecodeError:
                            pass  # Keep the string value if it's not valid JSON

                    prompt_analysis = f"""
                    #### Original prompt analysis
                    - {parsed_json.get("initial_prompt_evaluation", "")}
                    """
                    explanation_of_refinements = f"""
                    #### Refinement Explanation
                    - {parsed_json.get("explanation_of_refinements", "")}
                    """
                    return {
                        "initial_prompt_evaluation": prompt_analysis,
                        "refined_prompt": parsed_json.get("refined_prompt", ""),
                        "explanation_of_refinements": explanation_of_refinements,
                        "response_content": parsed_json
                    }
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {str(e)}")
                    print(f"Problematic JSON string: {json_str}")
                    # If JSON parsing fails, try to fix common issues
                    json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                    try:
                        parsed_json = json.loads(json_str)
                        print(f"Fixed JSON parse: {parsed_json}")
                        return self._parse_response(json.dumps(parsed_json))
                    except json.JSONDecodeError:
                        return self._parse_with_regex(response_content)

            return self._parse_with_regex(response_content)

        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return self._create_error_dict(str(e))

    def _parse_with_regex(self, content: str) -> dict:
        """Parse content using regex when JSON parsing fails."""
        try:
            output = {}

            refinements_match = re.search(r'"explanation_of_refinements":\s*$(.*?)$', content, re.DOTALL)
            if refinements_match:
                refinements_str = refinements_match.group(1)
                refinements = [
                    item.strip().strip('"').strip("'").replace('•', '-')
                    for item in re.findall(r'[•"]([^"•]+)[•"]', refinements_str)
                ]
                output["explanation_of_refinements"] = refinements
            else:
                pattern = r'"explanation_of_refinements":\s*"(.*?)"(?:,|\})'
                match = re.search(pattern, content, re.DOTALL)
                output["explanation_of_refinements"] = match.group(1).strip() if match else ""

            for key in ["initial_prompt_evaluation", "refined_prompt"]:
                pattern = rf'"{key}":\s*"(.*?)"(?:,|\}})'
                match = re.search(pattern, content, re.DOTALL)
                output[key] = match.group(1).strip() if match else ""

            output["response_content"] = {"raw_content": content}
            print(content)
            return output
        except Exception as e:
            print(f"Error in regex parsing: {str(e)}")
            return self._create_error_dict(str(e))

    def _create_error_dict(self, error_message: str) -> dict:
        """Create a standardized error response dictionary."""
        return {
            "initial_prompt_evaluation": f"Error parsing response: {error_message}",
            "refined_prompt": "",
            "explanation_of_refinements": "",
            "response_content": {"error": error_message}
        }

    def automatic_metaprompt(self, prompt: str) -> Tuple[str, str]:
        """Automatically select the most appropriate metaprompt."""
        try:
            router_messages = [
                {
                    "role": "system",
                    "content": "You are an AI Prompt Selection Assistant that helps choose the most appropriate metaprompt based on the user's query."
                },
                {
                    "role": "user",
                    "content": metaprompt_router.replace("[Insert initial prompt here]", prompt)
                }
            ]

            router_response = self._make_api_request(
                messages=router_messages,
                model=prompt_refiner_model,
                temperature=0.2
            )

            router_content = router_response["choices"][0]["message"]["content"].strip()
            json_match = re.search(r'<json>(.*?)</json>', router_content, re.DOTALL)

            if not json_match:
                raise ValueError("No JSON found in router response")

            # Clean and parse the JSON with multiple attempts
            json_str = self._clean_json_string(json_match.group(1))
            print(f"Cleaned JSON string: {json_str}")

            try:
                router_result = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Initial JSON parse error: {str(e)}")
                print(f"Error position: char {e.pos}")
                print(f"Problem section: {json_str[max(0, e.pos-20):min(len(json_str), e.pos+20)]}")

                # Simple retry with brace balancing
                try:
                    # Count braces and add missing closing brace if needed
                    open_braces = json_str.count('{')
                    close_braces = json_str.count('}')
                    if open_braces > close_braces:
                        json_str += '}'

                    print(f"Attempting to parse with brace balancing: {json_str}")
                    router_result = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"Parse attempt failed: {str(e2)}")
                    raise ValueError(f"Failed to parse router response: {str(e)}")

            # Safely get the recommended key with fallback
            recommended_key = (router_result.get("recommended_metaprompt", {})
                             .get("key", next(iter(self.meta_prompts))))

            # Check if the recommended key exists in available metaprompts
            if recommended_key not in self.meta_prompts:
                # Fallback to default if recommended doesn't exist
                recommended_key = next(iter(self.meta_prompts))
                router_result["recommended_metaprompt"]["name"] = "Default Template"
                router_result["recommended_metaprompt"]["description"] = "Fallback to default template as recommended template is not available"

            metaprompt_analysis = f"""
            #### Selected MetaPrompt
            - **Primary Choice**: {router_result["recommended_metaprompt"]["name"]}
            - *Description*: {router_result["recommended_metaprompt"]["description"]}
            - *Why This Choice*: {router_result["recommended_metaprompt"]["explanation"]}
            - *Similar Sample*: {router_result["recommended_metaprompt"]["similar_sample"]}
            - *Customized Sample*: {router_result["recommended_metaprompt"]["customized_sample"]}

            #### Alternative Option
            - **Secondary Choice**: {router_result["alternative_recommendation"]["name"]}
            - *Why Consider This*: {router_result["alternative_recommendation"]["explanation"]}
            """

            return metaprompt_analysis, recommended_key

        except Exception as e:
            return f"Error in automatic metaprompt: {str(e)}", ""

    def refine_prompt(self, prompt: str, meta_prompt_choice: str) -> Tuple[str, str, str, dict]:
        """Refine the given prompt using the selected meta prompt."""
        try:
            # Get the template or fall back to default
            selected_meta_prompt = self.meta_prompts.get(meta_prompt_choice)
            if not selected_meta_prompt:
                # Fallback to first available template
                meta_prompt_choice = next(iter(self.meta_prompts))
                selected_meta_prompt = self.meta_prompts[meta_prompt_choice]

            selected_meta_prompt_explanations = self.metaprompt_explanations.get(meta_prompt_choice)

            messages = [
                {
                    "role": "system",
                    "content": 'You are an expert at refining and extending prompts.'
                },
                {
                    "role": "user",
                    "content": selected_meta_prompt.replace("[Insert initial prompt here]", prompt)
                }
            ]

            response = self._make_api_request(
                messages=messages,
                model=prompt_refiner_model,
                temperature=0.8
            )

            result = self._parse_response(response["choices"][0]["message"]["content"].strip())
            llm_response = LLMResponse(**result)
            llm_response_dico = {}
            llm_response_dico['initial_prompt'] = prompt
            llm_response_dico['meta_prompt'] = meta_prompt_choice
            llm_response_dico = llm_response_dico | llm_response.dict()

            return (
                llm_response.initial_prompt_evaluation,
                llm_response.refined_prompt,
                llm_response.explanation_of_refinements,
                llm_response_dico
            )

        except Exception as e:
            return (
                f"Error: {str(e)}",
                "",
                "",
                {}
            )

    def apply_prompt(self, prompt: str, model: str) -> str:
        """Apply formatting to the prompt using the specified model."""
        try:
            if not prompt or not model:
                return "Error: Prompt and model are required"

            messages = [
                {
                    "role": "system",
                    "content": "You are a markdown formatting expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = self._make_api_request(
                messages=messages,
                model=model,
                temperature=0.8
            )

            result = response["choices"][0]["message"]["content"].strip()
            return f"""{result}"""

        except Exception as e:
            return f"Error: {str(e)}"
