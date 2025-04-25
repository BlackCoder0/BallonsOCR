import re
import time
import yaml
import traceback
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

import httpx
from openai import OpenAI

from .base import BaseTranslator, register_translator


class InvalidNumTranslations(Exception):
    pass


@register_translator("LLM_API_Translator_XML")
class LLM_API_Translator_XML(BaseTranslator):
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
                "OAI: gpt-3.5-turbo",
                "GGL: gemini-1.5-pro-latest",
                "GGL: gemini-2.0-flash-exp",
                "GGL: gemini-2.0-flash",
            ],
            "value": "OAI: gpt-4o",
            "description": "Select the model. Provider prefix indicates the provider. Leave empty for provider default.",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty to use provider default.",
        },
        "chat system template": {
            "type": "editor",
            "value": "You are a professional translation engine. Translate the text provided in the XML structure.",
            "description": "System message for the LLM."
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
            "value": 0.3,
            "description": "Temperature for sampling. Lower values make output more deterministic.",
        },
        "top p": {
            "value": 1,
            "description": "Top P for sampling. Forced to 1 for Google models.",
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
        "frequency penalty": {
            "value": 0.1,
            "description": "Frequency penalty (OpenAI).",
        },
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
            try:
                http_client = httpx.Client(proxy=httpx.Proxy(proxy))
            except Exception as e:
                self.logger.error(f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy.")
                http_client = httpx.Client()
        else:
            http_client = httpx.Client()

        api_keys = self.multiple_keys_list
        api_key_to_use = self.apikey if not api_keys else api_keys[0] # Initial key

        if not api_key_to_use:
            self.logger.warning("No API key initially available. Client not fully initialized.")
            self.client = None
            # No need to return, _select_api_key will handle selection later if keys are added
        else:
            endpoint = self.endpoint
            provider = self.provider
            if not endpoint:
                if provider == "Google":
                    endpoint = "https://generativelanguage.googleapis.com/v1beta"
                    self.logger.warning(f"Using default Google endpoint: {endpoint}. Ensure it's OpenAI SDK compatible.")
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
        keys_str = self.get_param_value("multiple_keys") # <- Исправлено
        # Добавим проверку на случай, если параметр не задан и возвращается None
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

        prompt_instructions = f"""You are a precise translation engine.
Translate the {from_lang} text snippets provided within the <src> tags into {to_lang}.
Input format is XML: <root><element><id>N</id><src>Source Text</src></element>...</root>
Output format MUST be XML, containing ONLY the <root>...</root> structure.
Inside each <element>:
1. Copy the <id> exactly from the input.
2. Provide the translation ONLY inside the <dst> tag.
3. Escape XML special characters in the translation if necessary (e.g., `&` to `&`, `<` to `<`, `>` to `>`, `"` to `"`, `'` to `'`).
4. DO NOT include the original <src> tag in the output.
5. DO NOT add any explanations, apologies, or text outside the <root>...</root> tags. Just the XML.

Example Input:
<root>
<element><id>1</id><src>こんにちは</src></element>
<element><id>2</id><src>ありがとう</src></element>
</root>

Example Output (for English):
<root>
<element><id>1</id><dst>Hello</dst></element>
<element><id>2</id><dst>Thank you</dst></element>
</root>

Now, translate the following input:
"""
        current_prompt_xml_content = ""
        num_src = 0
        i_offset = 0

        for i, query in enumerate(queries):
            escaped_query = query.replace('&', '&').replace('<', '<').replace('>', '>')
            element = f'<element><id>{i+1-i_offset}</id><src>{escaped_query}</src></element>\n'

            if len(prompt_instructions) + len("<root>\n") + len(current_prompt_xml_content) + len(element) + len("</root>") > max_len_approx and num_src > 0:
                full_prompt = prompt_instructions + "<root>\n" + current_prompt_xml_content + "</root>"
                self.logger.debug(f"Yielding prompt with {num_src} elements (approx len: {len(full_prompt)}).")
                yield full_prompt, num_src
                current_prompt_xml_content = element
                num_src = 1
                i_offset = i
            else:
                current_prompt_xml_content += element
                num_src += 1

        if num_src > 0:
            full_prompt = prompt_instructions + "<root>\n" + current_prompt_xml_content + "</root>"
            self.logger.debug(f"Yielding final prompt with {num_src} elements (approx len: {len(full_prompt)}).")
            yield full_prompt, num_src

    def _parse_xml_response(self, response: str, expected_count: int) -> List[str]:
        self.logger.debug(f"Attempting to parse XML response: {response[:500]}...")
        translations_dict = {}
        try:
            match = re.search(r'<root\s*?>(.*?)</root\s*?>', response, re.DOTALL | re.IGNORECASE)
            if not match:
                self.logger.error("Could not find <root>...</root> tags in the response.")
                raise ValueError("No <root> tag found")

            xml_content = match.group(1).strip()
            root = ET.fromstring(f"<root>{xml_content}</root>")

            count = 0
            for element in root.findall('.//element'):
                id_tag = element.find('id')
                dst_tag = element.find('dst')

                if id_tag is not None and dst_tag is not None and id_tag.text is not None:
                    try:
                        elem_id = int(id_tag.text)
                        translation = dst_tag.text if dst_tag.text is not None else ""
                        translations_dict[elem_id] = translation.strip()
                        count += 1
                    except ValueError:
                        self.logger.warning(f"Found element with non-integer ID: {id_tag.text}. Skipping.")
                    except Exception as e_inner:
                         self.logger.warning(f"Error processing element with ID {id_tag.text}: {e_inner}")
                else:
                    self.logger.warning(f"Skipping element missing id or dst tag: {ET.tostring(element, encoding='unicode')}")

            self.logger.debug(f"Successfully parsed {count} elements from XML.")

        except ET.ParseError as e:
            self.logger.error(f"XML ParseError: {e}")
            self.logger.debug(f"Invalid XML content received: {response[:1000]}...")
            raise ValueError(f"XML parsing failed: {e}") from e
        except Exception as e:
            self.logger.error(f"Error during XML parsing: {e}")
            self.logger.debug(f"Response causing error: {response[:1000]}...")
            raise ValueError(f"Generic XML parsing error: {e}") from e

        if len(translations_dict) != expected_count:
            self.logger.error(f"Mismatch: Expected {expected_count} translations, but parsed {len(translations_dict)}.")
            missing_ids = set(range(1, expected_count + 1)) - set(translations_dict.keys())
            extra_ids = set(translations_dict.keys()) - set(range(1, expected_count + 1))
            if missing_ids: self.logger.error(f"Missing translations for IDs: {sorted(list(missing_ids))}")
            if extra_ids: self.logger.error(f"Found unexpected translations for IDs: {sorted(list(extra_ids))}")
            raise InvalidNumTranslations(f"Expected {expected_count}, got {len(translations_dict)}")

        ordered_translations = [translations_dict[i] for i in range(1, expected_count + 1)]
        return ordered_translations

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
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            if self.debug_mode:
                 self.logger.debug(f"Global delay: Waiting {sleep_time:.3f} seconds.")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str):
        rpm = self.max_rpm
        if rpm <= 0: return

        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))

        if now - start_time >= 60:
            count = 0
            start_time = now
            self.key_usage[key] = (count, start_time)

        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(f"RPM limit ({rpm}) reached for key {key[:6]}... Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())


    def _select_api_key(self) -> Optional[str]:
        api_keys = self.multiple_keys_list
        single_key = self.apikey

        if not api_keys and not single_key:
            self.logger.error("No API keys provided in 'multiple_keys' or 'apikey'.")
            return None

        key_to_use = None
        if not api_keys:
            self._respect_key_limit(single_key)
            key_to_use = single_key
        else:
            num_keys = len(api_keys)
            start_index = self.current_key_index
            for i in range(num_keys):
                index = (start_index + i) % num_keys
                key = api_keys[index]
                self._respect_key_limit(key)
                self.current_key_index = (index + 1) % num_keys
                key_to_use = key
                break # Found a key after respecting limit

        if key_to_use:
             now = time.time()
             count, start_time = self.key_usage.get(key_to_use, (0, now))
             if now - start_time >= 60:
                 count = 0
                 start_time = now
             self.key_usage[key_to_use] = (count + 1, start_time)
             return key_to_use
        else:
            # Should not happen if _respect_key_limit waits, but as a fallback
            self.logger.error("All API keys seem to be rate-limited simultaneously or selection failed.")
            return None


    def _request_translation(self, prompt: str) -> str:
        if not self.client:
            raise ConnectionError("OpenAI client is not initialized.")

        self._respect_delay()

        current_api_key = self._select_api_key()
        if not current_api_key:
             raise ConnectionError("No available API key found.")
        self.client.api_key = current_api_key

        model_name = self.override_model or self.model
        provider = self.provider
        if ": " in model_name: model_name = model_name.split(": ", 1)[1]

        self.logger.debug(f"Requesting translation with Provider: {provider}, Model: {model_name}")

        messages = [
            {"role": "system", "content": self.chat_system_template},
            {"role": "user", "content": prompt},
        ]

        api_args = {
            "model": model_name, "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p if provider != "Google" else 1.0,
            "max_tokens": self.max_tokens // 2,
        }
        if provider == "OpenAI":
            api_args["frequency_penalty"] = self.frequency_penalty
            api_args["presence_penalty"] = self.presence_penalty

        self.logger.debug(f"API call arguments: {api_args}")

        try:
            response = self.client.chat.completions.create(**api_args)
        except Exception as e:
             self.logger.error(f"API request failed: {e}")
             if hasattr(e, 'response') and e.response:
                  self.logger.error(f"API Response Status: {e.response.status_code}")
                  try: self.logger.error(f"API Response Data: {e.response.json()}")
                  except: self.logger.error(f"API Response Text: {e.response.text}")
             raise

        content = ""
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ""
        else: self.logger.warning("No message content found in the response.")

        if response.usage:
            total_tokens = response.usage.total_tokens
            self.token_count += total_tokens
            self.token_count_last = total_tokens
            self.logger.debug(f"API Usage: {total_tokens} tokens.")
        else:
            self.logger.warning("Usage data not found in API response.")
            self.token_count_last = 0

        return content

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
                        self.logger.debug(f"Requesting translation for {num_src} elements. API Attempt: {retry_attempt+1}, Mismatch Attempt: {mismatch_retry_attempt+1}")
                        self.logger.debug(f"Prompt (first 500 chars): {prompt[:500]}...")

                    response_text = self._request_translation(prompt)

                    if not response_text:
                         self.logger.warning("Received empty response from API.")
                         raise ValueError("Empty response received from API")

                    new_translations = self._parse_xml_response(response_text, num_src)

                    translations.extend(new_translations)
                    self.logger.info(f"Successfully translated batch of {num_src}. Tokens used: {self.token_count_last}")
                    break

                except InvalidNumTranslations as e:
                    mismatch_retry_attempt += 1
                    message = f"Translation count mismatch ({e}). Attempt {mismatch_retry_attempt}/{self.invalid_repeat_count}."
                    self.logger.warning(message)
                    self.logger.debug(f"Mismatch details:\nResponse:\n{response_text}")

                    if mismatch_retry_attempt >= self.invalid_repeat_count:
                        self.logger.error(f"Failed to get correct translation count after {self.invalid_repeat_count} attempts. Returning empty strings.")
                        translations.extend(["ERROR: Mismatch"] * num_src)
                        break
                    else:
                        time.sleep(self.retry_timeout / 2)

                except Exception as e:
                    retry_attempt += 1
                    message = f"API request failed: {e}. Attempt {retry_attempt}/{self.retry_attempts}."
                    self.logger.warning(message)
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")

                    if retry_attempt >= self.retry_attempts:
                        self.logger.error(f"Failed to translate batch after {self.retry_attempts} API attempts. Returning empty strings.")
                        translations.extend([f"ERROR: API Failed ({type(e).__name__})"] * num_src)
                        break
                    else:
                         self.logger.info(f"Waiting {self.retry_timeout}s before retrying...")
                         time.sleep(self.retry_timeout)

        if self.token_count_last:
            self.logger.info(f"Finished translation request. Last batch used {self.token_count_last} tokens (Total accumulated: {self.token_count})")
        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        self.logger.debug(f"Parameter '{param_key}' updated to: {param_content}")
        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint"]:
            self.logger.info(f"Re-initializing client due to change in '{param_key}'.")
            self._initialize_client()