import re
import time
import yaml
import traceback
from typing import List, Dict, Optional, Type

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError # Добавлен импорт pydantic

from .base import BaseTranslator, register_translator

class InvalidNumTranslations(Exception):
    pass
class TranslationElement(BaseModel):
    id: int
    translation: str = Field(..., description="The translated text corresponding to the id.")

class TranslationResponse(BaseModel):
    translations: List[TranslationElement] = Field(..., description="List of translation elements.")

@register_translator("LLM_API_Translator_JSON")
class LLM_API_Translator_JSON(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "apikey": {
            "value": "",
            "description": "Single API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). One key per line for readability.",
        },
        "model": {
            "type": "selector",
             "options": [
                "OAI: gpt-4o",
                "OAI: gpt-4-turbo",
                "GGL: gemini-1.5-pro-latest",
                "GGL: gemini-1.5-flash-latest",
                "GGL: gemini-pro", 
            ],
            "value": "OAI: gpt-4o",
            "description": "Select the model supporting structured output (JSON mode).",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default. Ensure compatibility with structured output.",
        },
        "chat system template": {
            "type": "editor",
            "value": "You are a translation engine. Extract translations for the provided text snippets and format the output as requested.",
            "description": "System message for the LLM, focusing on the task."
        },
        "invalid repeat count": {
            "value": 2,
            "description": "Number of retries allowed if the count of translations mismatch.",
        },
        "max requests per minute": {
            "value": 20,
            "description": "Maximum requests per minute for EACH API key.",
        },
        "delay": {
            "value": 0.3,
            "description": "Global delay in seconds between requests.",
        },
        "max tokens": {
            "value": 4096,
            "description": "Maximum tokens for the response.",
        },
        "temperature": {
            "value": 0.1, # Еще ниже для строгого JSON
            "description": "Temperature for sampling. Very low for structured output.",
        },
        "top p": {
            "value": 1,
            "description": "Top P for sampling.", # Google игнорирует, OpenAI может использовать
        },
        "retry attempts": {
            "value": 3,
            "description": "Number of retry attempts on API failure (not mismatch error).",
        },
        "retry timeout": {
            "value": 15,
            "description": "Timeout between retry attempts (seconds).",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)",
        },
        # frequency/presence penalties могут быть менее важны для JSON mode
        "frequency penalty": {"value": 0.0, "description": "Frequency penalty (OpenAI)."},
        "presence penalty": {"value": 0.0, "description": "Presence penalty (OpenAI)."},
        "low vram mode": {
            "value": False,
            "description": "Check if running locally and facing VRAM issues.",
            "type": "checkbox",
        },
    }

    def _setup_translator(self):
        self.lang_map = {
            "简体中文": "Simplified Chinese", "繁體中文": "Traditional Chinese", "日本語": "Japanese",
            "English": "English", "한국어": "Korean", "Tiếng Việt": "Vietnamese",
            "čeština": "Czech", "Français": "French", "Deutsch": "German",
            "magyar nyelv": "Hungarian", "Italiano": "Italian", "Polski": "Polish",
            "Português": "Portuguese", "limba română": "Romanian", "русский язык": "Russian",
            "Español": "Spanish", "Türk dili": "Turkish", "украї́нська мо́ва": "Ukrainian",
            "Thai": "Thai", "Arabic": "Arabic", "Malayalam": "Malayalam",
            "Tamil": "Tamil", "Hindi": "Hindi",
        }
        self.token_count = 0
        self.token_count_last = 0
        self.current_key_index = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self._initialize_client()
        self._original_params = {k: p.get('value') for k, p in self.params.items()}

    def _initialize_client(self):
        proxy = self.proxy
        http_client = None
        if proxy:
            try: http_client = httpx.Client(proxy=httpx.Proxy(proxy))
            except Exception as e:
                self.logger.error(f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy.")
                http_client = httpx.Client()
        else: http_client = httpx.Client()

        api_keys = self.multiple_keys_list
        api_key_to_use = self.apikey if not api_keys else api_keys[0]

        if not api_key_to_use:
            self.logger.warning("No API key initially available. Client not fully initialized.")
            self.client = None
        else:
            endpoint = self.endpoint
            provider = self.provider
            if not endpoint:
                if provider == "Google":
                    endpoint = "https://generativelanguage.googleapis.com/v1beta/openai" # Endpoint из примера
                    self.logger.info(f"Using default Google endpoint for structured output: {endpoint}.")
                else:
                    endpoint = "https://api.openai.com/v1"

            masked_key = api_key_to_use[:6] + "..." if len(api_key_to_use) > 6 else api_key_to_use
            try:
                self.logger.debug(f"Initializing OpenAI client shell with endpoint: {endpoint}, provider: {provider}")
                self.client = OpenAI(api_key=api_key_to_use, base_url=endpoint, http_client=http_client)
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None

    @property
    def provider(self) -> str: return self.get_param_value("provider")
    @property
    def apikey(self) -> str: return self.get_param_value("apikey")
    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys") #
        if not isinstance(keys_str, str):
            keys_str = ""
        keys_str = keys_str.strip()
        return [key.strip() for key in keys_str.split(";") if key.strip()]
    @property
    def model(self) -> str: return self.get_param_value("model")
    @property
    def override_model(self) -> Optional[str]: return self.get_param_value("override model") or None
    @property
    def endpoint(self) -> Optional[str]: return self.get_param_value("endpoint") or None
    @property
    def temperature(self) -> float: return float(self.get_param_value("temperature"))
    @property
    def top_p(self) -> float: return float(self.get_param_value("top p"))
    @property
    def max_tokens(self) -> int: return int(self.get_param_value("max tokens"))
    @property
    def retry_attempts(self) -> int: return int(self.get_param_value("retry attempts"))
    @property
    def retry_timeout(self) -> int: return int(self.get_param_value("retry timeout"))
    @property
    def proxy(self) -> str: return self.get_param_value("proxy")
    @property
    def chat_system_template(self) -> str: return self.params["chat system template"]["value"]
    @property
    def invalid_repeat_count(self) -> int: return int(self.get_param_value("invalid repeat count"))
    @property
    def frequency_penalty(self) -> float: return float(self.get_param_value("frequency penalty"))
    @property
    def presence_penalty(self) -> float: return float(self.get_param_value("presence penalty"))
    @property
    def max_rpm(self) -> int: return int(self.get_param_value("max requests per minute"))
    @property
    def global_delay(self) -> float: return float(self.get_param_value("delay"))

    def _assemble_prompts(self, queries: List[str], to_lang: str = None, max_len_approx=8000):
        if not self.client:
             self.logger.error("Client not initialized. Cannot assemble prompts.")
             return

        if to_lang is None: to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)

        prompt_instructions = f"Translate the following {from_lang} text snippets to {to_lang}. Provide the translation for each snippet corresponding to its ID.\n\nInput Snippets:\n"
        current_prompt_content = ""
        num_src = 0
        i_offset = 0

        for i, query in enumerate(queries):
            element = f"{i+1-i_offset}: {query}\n"

            # Примерная проверка длины
            if len(prompt_instructions) + len(current_prompt_content) + len(element) > max_len_approx and num_src > 0:
                full_prompt = prompt_instructions + current_prompt_content
                self.logger.debug(f"Yielding prompt with {num_src} elements (approx len: {len(full_prompt)}).")
                yield full_prompt, num_src
                current_prompt_content = element
                num_src = 1
                i_offset = i
            else:
                current_prompt_content += element
                num_src += 1

        if num_src > 0:
            full_prompt = prompt_instructions + current_prompt_content
            self.logger.debug(f"Yielding final prompt with {num_src} elements (approx len: {len(full_prompt)}).")
            yield full_prompt, num_src

    def _respect_delay(self):
        current_time = time.time()
        rpm = self.max_rpm
        delay = self.global_delay
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f} seconds.")
                    time.sleep(wait_time)
                self.request_count_minute = 0; self.minute_start_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            if self.debug_mode: self.logger.debug(f"Global delay: Waiting {sleep_time:.3f} seconds.")
            time.sleep(sleep_time)
        self.last_request_time = time.time(); self.request_count_minute += 1

    def _respect_key_limit(self, key: str):
        rpm = self.max_rpm
        if rpm <= 0: return
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60: count = 0; start_time = now; self.key_usage[key] = (count, start_time)
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(f"RPM limit ({rpm}) reached for key {key[:6]}... Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())

    def _select_api_key(self) -> Optional[str]:
        api_keys = self.multiple_keys_list; single_key = self.apikey
        if not api_keys and not single_key: self.logger.error("No API keys provided."); return None
        key_to_use = None
        if not api_keys: self._respect_key_limit(single_key); key_to_use = single_key
        else:
            num_keys = len(api_keys); start_index = self.current_key_index
            for i in range(num_keys):
                index = (start_index + i) % num_keys; key = api_keys[index]
                self._respect_key_limit(key); self.current_key_index = (index + 1) % num_keys
                key_to_use = key; break
        if key_to_use:
             now = time.time(); count, start_time = self.key_usage.get(key_to_use, (0, now))
             if now - start_time >= 60: count = 0; start_time = now
             self.key_usage[key_to_use] = (count + 1, start_time); return key_to_use
        else: self.logger.error("All API keys rate-limited or selection failed."); return None


    def _request_translation(self, prompt: str, response_format_model: Type[BaseModel]) -> Optional[TranslationResponse]:
        if not self.client: raise ConnectionError("OpenAI client is not initialized.")

        self._respect_delay()

        current_api_key = self._select_api_key()
        if not current_api_key: raise ConnectionError("No available API key found.")
        self.client.api_key = current_api_key

        model_name = self.override_model or self.model
        provider = self.provider
        if ": " in model_name: model_name = model_name.split(": ", 1)[1]

        self.logger.debug(f"Requesting structured translation with Provider: {provider}, Model: {model_name}")

        messages = [
            {"role": "system", "content": self.chat_system_template},
            {"role": "user", "content": prompt},
        ]

        api_args = {
            "model": model_name, "messages": messages,
            "temperature": self.temperature, "top_p": self.top_p,
            "max_tokens": self.max_tokens // 2,
        }
        # Penalties can conflict with JSON mode, let's check the compatibility
        # if provider == "OpenAI":
        #     api_args["frequency_penalty"] = self.frequency_penalty
        #     api_args["presence_penalty"] = self.presence_penalty

        self.logger.debug(f"API call arguments (excluding response_format): {api_args}")

        try:
            completion = self.client.beta.chat.completions.parse(
                 **api_args,
                 response_format=response_format_model # Passing the pydantic model
            )
        except Exception as e:
             self.logger.error(f"API request failed: {e}")
             if hasattr(e, 'response') and e.response:
                  self.logger.error(f"API Response Status: {e.response.status_code}")
                  try: self.logger.error(f"API Response Data: {e.response.json()}")
                  except: self.logger.error(f"API Response Text: {e.response.text}")
             raise # Reject for retry logic

        parsed_data: Optional[TranslationResponse] = None
        if completion.choices and completion.choices[0].message:
            try:
                parsed_data = completion.choices[0].message.parsed
                if not isinstance(parsed_data, response_format_model):
                     self.logger.error(f"Parsed data type mismatch: Expected {response_format_model}, got {type(parsed_data)}")
                     raise ValueError("Parsed data type mismatch")
            except (AttributeError, ValidationError, Exception) as parse_error:
                 self.logger.error(f"Failed to access or validate parsed structured response: {parse_error}")
                 self.logger.debug(f"Raw message content (if available): {getattr(completion.choices[0].message, 'content', 'N/A')}")
                 raise ValueError(f"Failed to parse/validate response: {parse_error}") from parse_error
        else: self.logger.warning("No message or choices found in the structured response completion.")


        if hasattr(completion, 'usage') and completion.usage:
            total_tokens = completion.usage.total_tokens
            self.token_count += total_tokens
            self.token_count_last = total_tokens
            self.logger.debug(f"API Usage: {total_tokens} tokens.")
        else:
            self.logger.warning("Usage data might not be available with client.beta.chat.completions.parse.")
            self.token_count_last = 0 

        return parsed_data


    def _translate(self, src_list: List[str]) -> List[str]:
        if not self.client:
            self.logger.error("Translator not initialized. Returning empty translations.")
            return [""] * len(src_list)
        if not src_list: return []

        translations = []
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)

        for prompt, num_src in self._assemble_prompts(src_list, to_lang=to_lang):
            if not prompt:
                 self.logger.warning("Empty prompt generated, skipping.")
                 translations.extend([""] * num_src)
                 continue

            retry_attempt = 0
            mismatch_retry_attempt = 0

            while True:
                try:
                    if self.debug_mode:
                        self.logger.debug(f"Requesting structured translation for {num_src} elements. API Attempt: {retry_attempt+1}, Mismatch Attempt: {mismatch_retry_attempt+1}")
                        self.logger.debug(f"Prompt (first 500 chars): {prompt[:500]}...")

                    parsed_response: Optional[TranslationResponse] = self._request_translation(prompt, TranslationResponse)

                    if parsed_response is None or not parsed_response.translations:
                        self.logger.warning("Received empty or invalid parsed response from API.")
                        raise ValueError("Empty or invalid parsed response received")

                    parsed_translations = parsed_response.translations
                    if len(parsed_translations) != num_src:
                        self.logger.error(f"Mismatch: Expected {num_src}, got {len(parsed_translations)} items in JSON.")
                        self.logger.debug(f"Parsed JSON content: {parsed_response.model_dump_json(indent=2)}")
                        raise InvalidNumTranslations(f"Expected {num_src}, got {len(parsed_translations)}")

                    translations_dict = {item.id: item.translation.strip() for item in parsed_translations}
                    ordered_translations = [translations_dict.get(i, "") for i in range(1, num_src + 1)]

                    if len(ordered_translations) != num_src or any(t is None for t in ordered_translations):
                         missing_ids = [i for i in range(1, num_src + 1) if i not in translations_dict]
                         self.logger.error(f"Mismatch after ordering: Missing translations for IDs: {missing_ids}")
                         raise InvalidNumTranslations(f"Data missing for some IDs after ordering.")

                    translations.extend(ordered_translations)
                    self.logger.info(f"Successfully translated batch of {num_src} via JSON. Tokens info: {self.token_count_last}")
                    break 

                except InvalidNumTranslations as e:
                    mismatch_retry_attempt += 1
                    message = f"Translation count/structure mismatch ({e}). Attempt {mismatch_retry_attempt}/{self.invalid_repeat_count}."
                    self.logger.warning(message)

                    if mismatch_retry_attempt >= self.invalid_repeat_count:
                        self.logger.error(f"Failed to get correct translation structure after {self.invalid_repeat_count} attempts.")
                        translations.extend(["ERROR: Structure Mismatch"] * num_src)
                        break
                    else:
                        time.sleep(self.retry_timeout / 2)

                except Exception as e:
                    retry_attempt += 1
                    message = f"API request/parsing failed: {e}. Attempt {retry_attempt}/{self.retry_attempts}."
                    self.logger.warning(message)
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")

                    if retry_attempt >= self.retry_attempts:
                        self.logger.error(f"Failed to translate batch after {self.retry_attempts} API/parsing attempts.")
                        translations.extend([f"ERROR: API/Parse Failed ({type(e).__name__})"] * num_src)
                        break
                    else:
                         self.logger.info(f"Waiting {self.retry_timeout}s before retrying...")
                         time.sleep(self.retry_timeout)

        if self.token_count_last is not None:
            self.logger.info(f"Finished translation request. Last batch token info: {self.token_count_last} (Total accumulated: {self.token_count})")
        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        self.logger.debug(f"Parameter '{param_key}' updated to: {param_content}")
        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint"]:
            self.logger.info(f"Re-initializing client due to change in '{param_key}'.")
            self._initialize_client()