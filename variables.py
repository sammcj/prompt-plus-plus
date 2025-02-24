import json
import os

# Default template if none provided
default_templates = {
    "OpenAI Meta Prompt": {
        "template": """Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

Initial Prompt: [Insert initial prompt here]

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it is simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions: Encourage reasoning steps before any conclusions are reached.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability.
- Output Format: Explicitly specify the most appropriate output format, in detail.

# Output Format

Provide the output in JSON format with the following structure:
{
  "initial_prompt_evaluation": "Evaluation with strengths and weaknesses",
  "refined_prompt": "Refined prompt text",
  "explanation_of_refinements": "Explanation of improvements made"
}""",
        "description": "A template for creating detailed system prompts that guide language models effectively",
        "examples": ["Write a story about a magical forest"],
    }
}

# Load templates from environment variable or use default
templates_json = os.getenv("PROMPT_TEMPLATES")

try:
    # Parse JSON data with error handling if env var exists
    prompt_data = json.loads(templates_json) if templates_json else default_templates
except json.JSONDecodeError:
    # Fallback to default templates if JSON is invalid
    prompt_data = default_templates

metaprompt_list = [key for key in prompt_data.keys()] if prompt_data else []
print(metaprompt_list)

# Create explanations dictionary with safe access
metaprompt_explanations = {
  key: data.get("description", "No description available")
  for key, data in prompt_data.items()
} if prompt_data else {}

# Generate markdown explanation
explanation_markdown = "".join(
    [f"- **{key}**: {value}\n" for key, value in metaprompt_explanations.items()]
)

# Define default models list - can be overridden by environment variable
default_models = [
    "dolphin3.0-r1-mistral-24b:q6_k_l",
    "fuseo1-deekseekr1-qwq-skyt1-32b-preview:q6_k",
]
models = json.loads(os.getenv("AVAILABLE_MODELS", json.dumps(default_models)))

# Extract examples only from JSON templates
examples = []
for key, data in prompt_data.items():
    template_examples = data.get("examples", [])
    if template_examples:
        examples.extend(
            [
                [example[0], key] if isinstance(example, list) else [example, key]
                for example in template_examples
            ]
        )

# Get API endpoint and optional key
api_endpoint = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if api_endpoint:
    api_endpoint = f"{api_endpoint.rstrip('/')}/v1/chat/completions"
api_key = os.getenv("LLM_API_KEY")  # Optional

# Create meta_prompts dictionary with safe access
meta_prompts = {
  key: data.get("template", "No template available")
  for key, data in prompt_data.items()
} if prompt_data else {}

prompt_refiner_model = os.getenv(
    "prompt_refiner_model", "dolphin3.0-r1-mistral-24b:q6_k_l"
)
print("prompt_refiner_model used :" + prompt_refiner_model)

echo_prompt_refiner = os.getenv('echo_prompt_refiner')
openai_metaprompt = os.getenv('openai_metaprompt')
advanced_meta_prompt = os.getenv('advanced_meta_prompt')
