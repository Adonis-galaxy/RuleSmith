"""
LLM client for InternVL model.

Provides a unified interface for calling InternVL or compatible language models.
"""

import os
import torch
from typing import Optional
import logging
import json
import re

# Fix CUDA environment before any CUDA operations
# This prevents "CUDA unknown error" from CUDA_VISIBLE_DEVICES changes
if 'CUDA_VISIBLE_DEVICES' not in os.environ and torch.cuda.is_available():
    # Set default if not set
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InternVLLM:
    """
    Wrapper for InternVL or compatible language models.
    
    This class provides a simple chat interface for text-only inference.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "OpenGVLab/InternVL3_5-2B",
        device: str = "cuda",
        use_half_precision: bool = True,
    ):
        """
        Initialize the LLM.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to run on ("cuda" or "cpu")
            use_half_precision: Whether to use fp16
        """
        self.device = device
        self.model_name = model_name_or_path
        self.use_half_precision = use_half_precision
        
        # Check CUDA availability before proceeding
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
                device = "cpu"
            else:
                logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
                logger.info(f"Current GPU: {torch.cuda.current_device()}")
        
        logger.info(f"Loading model {model_name_or_path} on {device}...")
        
        try:
            # Try to load as InternVL model (supports both vision-language and text-only)
            from transformers import AutoTokenizer, AutoModel
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading model...")
            # InternVL models should be loaded with AutoModel, not AutoModelForCausalLM
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if use_half_precision and device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            
            logger.info(f"Moving model to {device}...")
            self.model.to(device)
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model type: {type(self.model).__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            logger.warning("Falling back to dummy model for testing...")
            self.model = None
            self.tokenizer = None
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            system_prompt: System message (role/instructions)
            user_prompt: User query
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text response
        """
        if self.model is None or self.tokenizer is None:
            # Dummy response for testing
            return self._dummy_response(user_prompt)
        
        try:
            # Check if model has chat method (InternVL style)
            if hasattr(self.model, 'chat'):
                # Use InternVL's native chat interface
                logger.debug("Using InternVL native chat interface")
                
                # Combine system and user prompts
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=None,  # Text-only mode
                    question=full_prompt,
                    generation_config=dict(
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                    )
                )
                return response.strip()
            
            # Fallback: standard transformer generation
            logger.debug("Using standard transformer generation")
            
            # Format the conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            # Try to use chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback: simple concatenation
                prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            logger.warning("Falling back to dummy response")
            return self._dummy_response(user_prompt)
    
    def _dummy_response(self, user_prompt: str) -> str:
        """
        Generate a dummy response for testing without a real model.
        
        This is useful for testing the overall system logic.
        """
        # Try to parse action request
        if "action" in user_prompt.lower() or "choose" in user_prompt.lower():
            # Return a dummy action
            return '''{"action_type": "GATHER", "unit_id": "empire_unit_0"}'''
        
        # Try to parse validation request
        if "valid" in user_prompt.lower() or "legal" in user_prompt.lower():
            return "YES - This action is valid."
        
        # Default response
        return "I will PASS this turn."
    
    def extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON object from LLM response.
        
        Args:
            text: Raw LLM output
            
        Returns:
            Parsed JSON dict or None
        """
        # Try to find JSON in the text
        # Look for {...} pattern
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                return parsed
            except json.JSONDecodeError:
                continue
        
        # Try to parse the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def extract_yes_no(self, text: str) -> bool:
        """
        Extract yes/no decision from LLM response.
        
        Args:
            text: Raw LLM output
            
        Returns:
            True for yes, False for no
        """
        text_lower = text.lower()
        
        # Look for clear yes/no indicators
        if "yes" in text_lower or "valid" in text_lower or "legal" in text_lower:
            return True
        if "no" in text_lower or "invalid" in text_lower or "illegal" in text_lower:
            return False
        
        # Default to yes (permissive)
        return True


class MockLLM(InternVLLM):
    """
    Mock LLM for testing that doesn't require loading a real model.
    
    Always uses dummy responses.
    """
    
    def __init__(self, **kwargs):
        """Initialize mock LLM."""
        self.device = "cpu"
        self.model_name = "mock"
        self.model = None
        self.tokenizer = None
        logger.info("Using MockLLM for testing")
    
    def chat(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate mock response."""
        return self._dummy_response(user_prompt)


def get_llm_client(
    model_name: Optional[str] = None,
    device: str = "cuda",
    use_mock: bool = False,
) -> InternVLLM:
    """
    Factory function to get an LLM client.
    
    Args:
        model_name: Model name or path (None for default)
        device: Device to use
        use_mock: If True, use MockLLM instead of real model
        
    Returns:
        LLM client instance
    """
    if use_mock:
        return MockLLM()
    
    if model_name is None:
        model_name = "OpenGVLab/InternVL3_5-2B"
    
    return InternVLLM(model_name_or_path=model_name, device=device)

