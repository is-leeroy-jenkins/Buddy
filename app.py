'''
  ******************************************************************************************
      Assembly:                Wario
      Filename:                app.py
      Author:                  Terry D. Eppler
      Created:                 01-31-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        01-20-2026
  ******************************************************************************************
  <copyright file="app.py" company="Terry D. Eppler">

	     app.py
	     Copyright ©  2026  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com

  </copyright>
  <summary>
    app.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations
from boogr import Error
import config as cfg
import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from openai import OpenAI
from PIL import Image, ImageFilter, ImageEnhance
import hashlib
import json
import base64
import fitz
import io
from pathlib import Path
import plotly.express as px
import multiprocessing
import os
import sqlite3
import time
import typing_extensions
import tiktoken
from typing import Any, Dict, List, Tuple, Optional
import tempfile
import re
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer
import zipfile
import xml.etree.ElementTree as ET
import gpt as gpt_provider
import gemini as gemini_provider
import grok as grok_provider

# ======================================================================================
# SESSION STATE PARAMETER DEFINITIONS
# ======================================================================================
if 'api_keys' not in st.session_state:
	st.session_state.api_keys = { 'GPT': None, 'Grok': None, 'Gemini': None, }

if 'openai_api_key' not in st.session_state:
	st.session_state.openai_api_key = ''

if 'gemini_api_key' not in st.session_state:
	st.session_state.gemini_api_key = ''

if 'groq_api_key' not in st.session_state:
	st.session_state.groq_api_key = ''

if 'google_api_key' not in st.session_state:
	st.session_state.google_api_key = ''

if 'xai_api_key' not in st.session_state:
	st.session_state.xai_api_key = ''

if 'google_cse_id' not in st.session_state:
	st.session_state[ 'google_cse_id' ] = ''

if 'googlemaps_api_key' not in st.session_state:
	st.session_state[ 'googlemaps_api_key' ] = ''

if 'geocoding_api_key' not in st.session_state:
	st.session_state[ 'geocoding_api_key' ] = ''
	
	
if st.session_state.openai_api_key == '':
	default = cfg.OPENAI_API_KEY
	if default:
		st.session_state.openai_api_key = default
		os.environ[ 'OPENAI_API_KEY' ] = default

if st.session_state.gemini_api_key == '':
	default = cfg.GEMINI_API_KEY
	if default:
		st.session_state.gemini_api_key = default
		os.environ[ 'GEMINI_API_KEY' ] = default

if st.session_state.groq_api_key == '':
	default = cfg.GROQ_API_KEY
	if default:
		st.session_state.groq_api_key = default
		os.environ[ 'GROQ_API_KEY' ] = default

if st.session_state.google_api_key == '':
	default = cfg.GOOGLE_API_KEY
	if default:
		st.session_state.google_api_key = default
		os.environ[ 'GOOGLE_API_KEY' ] = default

if st.session_state.xai_api_key == '':
	default = cfg.XAI_API_KEY
	if default:
		st.session_state.xai_api_key = default
		os.environ[ 'XAI_API_KEY' ] = default
		
if st.session_state.google_cse_id == '':
	default = cfg.GOOGLE_CSE_ID
	if default:
		st.session_state.google_cse_id = default
		os.environ[ 'GOOGLE_CSE_ID' ] = default

if st.session_state.googlemaps_api_key == '':
	default = cfg.GOOGLEMAPS_API_KEY
	if default:
		st.session_state.googlemaps_api_key = default
		os.environ[ 'GOOGLEMAPS_API_KEY' ] = default

if st.session_state.geocoding_api_key == '':
	default = cfg.GEOCODING_API_KEY
	if default:
		st.session_state.geocoding_api_key = default
		os.environ[ 'GEOCODING_API_KEY' ] = default
		
if 'provider' not in st.session_state or st.session_state[ 'provider' ] is None:
	st.session_state[ 'provider' ] = 'GPT'

if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
	st.session_state[ 'mode' ] = 'Chat'

if 'files' not in st.session_state:
	st.session_state.files: List[ str ] = [ ]

# ----------MODEL PARAMETERS --------------------------------

if 'chat_model' not in st.session_state:
	st.session_state.chat_model = ''

if 'text_model' not in st.session_state:
	st.session_state[ 'text_model' ] = ''

if 'image_model' not in st.session_state:
	st.session_state[ 'image_model' ] = ''

if 'audio_model' not in st.session_state:
	st.session_state[ 'audio_model' ] = ''

if 'embedding_model' not in st.session_state:
	st.session_state[ 'embedding_model' ] = ''

if 'docqna_model' not in st.session_state:
	st.session_state[ 'docqna_model' ] = ''

if 'files_model' not in st.session_state:
	st.session_state[ 'files_model' ] = ''

if 'stores_model' not in st.session_state:
	st.session_state[ 'stores_model' ] = ''

if 'tts_model' not in st.session_state:
	st.session_state[ 'tts_model' ] = ''

if 'transcription_model' not in st.session_state:
	st.session_state[ 'transcription_model' ] = ''

if 'translation_model' not in st.session_state:
	st.session_state[ 'translation_model' ] = ''

# --------SYSTEM PARAMETERS----------------------

if 'instructions' not in st.session_state:
	st.session_state[ 'instructions' ] = ''

if 'chat_system_instructions' not in st.session_state:
	st.session_state[ 'chat_system_instructions' ] = ''

if 'text_system_instructions' not in st.session_state:
	st.session_state[ 'text_system_instructions' ] = ''

if 'image_system_instructions' not in st.session_state:
	st.session_state[ 'image_system_instructions' ] = ''

if 'audio_system_instructions' not in st.session_state:
	st.session_state[ 'audio_system_instructions' ] = ''

if 'docqna_system_instructions' not in st.session_state:
	st.session_state[ 'docqna_systems_instructions' ] = ''

# --------CHAT-GENERATION PARAMETERS--------------------

if 'max_tools' not in st.session_state:
	st.session_state[ 'max_tools' ] = 0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 0

if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.0

if 'top_percent' not in st.session_state:
	st.session_state[ 'top_percent' ] = 0.0

if 'frequency_penalty' not in st.session_state:
	st.session_state[ 'frequency_penalty' ] = 0.0

if 'presense_penalty' not in st.session_state:
	st.session_state[ 'presense_penalty' ] = 0.0

if 'background' not in st.session_state:
	st.session_state[ 'background' ] = False

if 'parallel_tools' not in st.session_state:
	st.session_state[ 'parallel_tools' ] = False

if 'store' not in st.session_state:
	st.session_state[ 'store' ] = False

if 'stream' not in st.session_state:
	st.session_state[ 'stream' ] = False
	
if 'execution_mode' not in st.session_state:
	st.session_state[ 'execution_mode' ] = 'Standard'

if 'chat_history' not in st.session_state:
	st.session_state.chat_history: List[ Dict[ str, str ] ] = [ ]

if 'response_format' not in st.session_state:
	st.session_state[ 'response_format' ] = ''

if 'tool_choice' not in st.session_state:
	st.session_state[ 'tool_choice' ] = ''

if 'reasoning' not in st.session_state:
	st.session_state[ 'reasoning' ] = ''

if 'stops' not in st.session_state:
	st.session_state[ 'stops' ] = [ ]

if 'include' not in st.session_state:
	st.session_state[ 'include' ] = [ ]

if 'input' not in st.session_state:
	st.session_state[ 'input' ] = [ ]

if 'tools' not in st.session_state:
	st.session_state[ 'tools' ] = [ ]

if 'messages' not in st.session_state:
	st.session_state[ 'messages' ] = [ ]

if 'last_sources' not in st.session_state:
	st.session_state[ 'last_sources' ] = [ ]

# --------TEXT-GENERATION PARAMETERS--------------------
if 'text_number' not in st.session_state:
	st.session_state[ 'text_number' ] = 0

if 'text_max_calls' not in st.session_state:
	st.session_state[ 'text_max_calls' ] = 0

if 'text_top_k' not in st.session_state:
	st.session_state[ 'text_top_k' ] = 0

if 'text_max_searches' not in st.session_state:
	st.session_state[ 'text_max_searches' ] = 0

if 'text_max_tokens' not in st.session_state:
	st.session_state[ 'text_max_tokens' ] = 0

if 'text_temperature' not in st.session_state:
	st.session_state[ 'text_temperature' ] = 0.0

if 'text_top_percent' not in st.session_state:
	st.session_state[ 'text_top_percent' ] = 0.0

if 'text_frequency_penalty' not in st.session_state:
	st.session_state[ 'text_frequency_penalty' ] = 0.0

if 'text_presense_penalty' not in st.session_state:
	st.session_state[ 'text_presense_penalty' ] = 0.0

if 'text_parallel_tools' not in st.session_state:
	st.session_state[ 'text_parallel_tools' ] = False

if 'text_background' not in st.session_state:
	st.session_state[ 'text_background' ] = False

if 'text_store' not in st.session_state:
	st.session_state[ 'text_store' ] = False

if 'text_stream' not in st.session_state:
	st.session_state[ 'text_stream' ] = False

if 'text_response_format' not in st.session_state:
	st.session_state[ 'text_response_format' ] = ''

if 'text_tool_choice' not in st.session_state:
	st.session_state[ 'text_tool_choice' ] = ''

if 'text_resolution' not in st.session_state:
	st.session_state[ 'text_resolution' ] = ''

if 'text_media_resolution' not in st.session_state:
	st.session_state[ 'text_media_resolution' ] = ''

if 'text_reasoning' not in st.session_state:
	st.session_state[ 'text_reasoning' ] = ''

if 'text_safety_profile' not in st.session_state:
	st.session_state[ 'text_safety_profile' ] = ''

if 'text_input' not in st.session_state:
	st.session_state[ 'text_input' ] = ''

if 'text_stops' not in st.session_state:
	st.session_state[ 'text_stops' ] = [ ]

if 'text_modalities' not in st.session_state:
	st.session_state[ 'text_modalities' ] = [ ]

if 'text_include' not in st.session_state:
	st.session_state[ 'text_include' ] = [ ]

if 'text_domains' not in st.session_state:
	st.session_state[ 'text_domains' ] = [ ]

if 'text_urls' not in st.session_state:
	st.session_state[ 'text_urls' ] = [ ]

if 'text_tools' not in st.session_state:
	st.session_state[ 'text_tools' ] = [ ]

if 'text_context' not in st.session_state:
	st.session_state[ 'text_context' ] = [ ]

if 'text_content' not in st.session_state:
	st.session_state[ 'text_content' ] = [ ]

if 'text_messages' not in st.session_state:
	st.session_state[ 'text_messages' ] = [ ]
	
if 'text_gemini_history' not in st.session_state:
	st.session_state[ 'text_gemini_history' ] = [ ]
	
# --------IMAGE-GENERATION PARAMETERS--------------------

if 'image_max_tokens' not in st.session_state:
	st.session_state[ 'image_max_tokens' ] = 0

if 'image_max_calls' not in st.session_state:
	st.session_state[ 'image_max_calls' ] = 0

if 'image_max_searches' not in st.session_state:
	st.session_state[ 'image_max_searches' ] = 0

if 'image_number' not in st.session_state:
	st.session_state[ 'image_number' ] = 0

if 'image_compression' not in st.session_state:
	st.session_state[ 'image_compression' ] = 0.0

if 'image_temperature' not in st.session_state:
	st.session_state[ 'image_temperature' ] = 0.0

if 'image_top_percent' not in st.session_state:
	st.session_state[ 'image_top_percent' ] = 0.0

if 'image_frequency_penalty' not in st.session_state:
	st.session_state[ 'image_frequency_penalty' ] = 0.0

if 'image_presence_penalty' not in st.session_state:
	st.session_state[ 'image_presence_penalty' ] = 0.0

if 'image_parallel_calls' not in st.session_state:
	st.session_state[ 'image_parallel_calls' ] = False

if 'image_background' not in st.session_state:
	st.session_state[ 'image_background' ] = False

if 'image_store' not in st.session_state:
	st.session_state[ 'image_store' ] = False

if 'image_stream' not in st.session_state:
	st.session_state[ 'image_stream' ] = False

if 'image_tool_choice' not in st.session_state:
	st.session_state[ 'image_tool_choice' ] = ''

if 'image_reasoning' not in st.session_state:
	st.session_state[ 'image_reasoning' ] = ''

if 'image_mime_type' not in st.session_state:
	st.session_state[ 'image_mime_type' ] = ''

if 'image_response_format' not in st.session_state:
	st.session_state[ 'image_response_format' ] = ''

if 'image_previous_response_id' not in st.session_state:
	st.session_state[ 'image_previous_response_id' ] = ''

if 'image_input' not in st.session_state:
	st.session_state[ 'image_input' ] = [ ]

if 'image_include' not in st.session_state:
	st.session_state[ 'image_include' ] = [ ]

if 'image_tools' not in st.session_state:
	st.session_state[ 'image_tools' ]: List[ Dict[ str, Any ] ] = [ ]

if 'image_modalities' not in st.session_state:
	st.session_state[ 'image_modalities' ] = [ ]

if 'image_messages' not in st.session_state:
	st.session_state[ 'image_messages' ] = [ ]

if 'image_context' not in st.session_state:
	st.session_state[ 'image_context' ]: List[ Dict[ str, Any ] ] = [ ]

if 'image_domains' not in st.session_state:
	st.session_state[ 'image_domains' ] = [ ]

if 'image_content' not in st.session_state:
	st.session_state[ 'image_content' ] = [ ]

# ------- IMAGE-SPECIFIC PARAMETER---------------
if 'image_mode' not in st.session_state:
	st.session_state[ 'image_mode' ] = ''

if 'image_style' not in st.session_state:
	st.session_state[ 'image_style' ] = ''

if 'image_detail' not in st.session_state:
	st.session_state[ 'image_detail' ] = ''

if 'image_backcolor' not in st.session_state:
	st.session_state[ 'image_backcolor' ] = ''

if 'image_output' not in st.session_state:
	st.session_state[ 'image_output' ] = ''

if 'image_url' not in st.session_state:
	st.session_state[ 'image_url' ] = ''

if 'image_size' not in st.session_state:
	st.session_state[ 'image_size' ] = ''

if 'image_quality' not in st.session_state:
	st.session_state[ 'image_quality' ] = ''

if 'image_grounded' not in st.session_state:
	st.session_state[ 'image_grounded' ] = False

if 'image_image_search' not in st.session_state:
	st.session_state[ 'image_image_search' ] = False
	
# --------AUDIO-GENERATION PARAMETERS--------------------

if 'audio_max_tokens' not in st.session_state:
	st.session_state[ 'audio_max_tokens' ] = 0

if 'audio_temperature' not in st.session_state:
	st.session_state[ 'audio_temperature' ] = 0.0

if 'audio_top_percent' not in st.session_state:
	st.session_state[ 'audio_top_percent' ] = 0.0

if 'audio_frequency_penalty' not in st.session_state:
	st.session_state[ 'audio_frequency_penalty' ] = 0.0

if 'audio_presence_penalty' not in st.session_state:
	st.session_state[ 'audio_presence_penalty' ] = 0.0

if 'audio_background' not in st.session_state:
	st.session_state[ 'audio_background' ] = False

if 'audio_store' not in st.session_state:
	st.session_state[ 'audio_store' ] = False

if 'audio_stream' not in st.session_state:
	st.session_state[ 'audio_stream' ] = False

if 'audio_tool_choice' not in st.session_state:
	st.session_state[ 'audio_tool_choice' ] = ''

if 'audio_reasoning' not in st.session_state:
	st.session_state[ 'audio_reasoning' ] = ''

if 'audio_response_format' not in st.session_state:
	st.session_state[ 'audio_response_format' ] = ''

if 'audio_input' not in st.session_state:
	st.session_state[ 'audio_input' ] = ''

if 'audio_mime_type' not in st.session_state:
	st.session_state[ 'audio_mime_type' ] = ''

if 'audio_stops' not in st.session_state:
	st.session_state[ 'audio_stops' ] = [ ]

if 'audio_includes' not in st.session_state:
	st.session_state[ 'audio_includes' ] = [ ]

if 'audio_tools' not in st.session_state:
	st.session_state.audio_tools: List[ Dict[ str, Any ] ] = [ ]

if 'audio_context' not in st.session_state:
	st.session_state.audio_context: List[ Dict[ str, Any ] ] = [ ]

if 'audio_modalities' not in st.session_state:
	st.session_state[ 'audio_modalities' ] = [ ]

if 'audio_messages' not in st.session_state:
	st.session_state.audio_messages = [ ]

# -------AUDIO-SECIFIC PARAMETERS--------------

if 'audio_task' not in st.session_state:
	st.session_state[ 'audio_task' ] = ''

if 'audio_file' not in st.session_state:
	st.session_state[ 'audio_file' ] = ''

if 'audio_rate' not in st.session_state:
	st.session_state[ 'audio_rate' ] = int( cfg.SAMPLE_RATES[ 0 ] ) if cfg.SAMPLE_RATES else 44100

if 'audio_language' not in st.session_state:
	st.session_state[ 'audio_language' ] = ''

if 'audio_voice' not in st.session_state:
	st.session_state[ 'audio_voice' ] = ''

if 'audio_start_time' not in st.session_state:
	st.session_state[ 'audio_start_time' ] = 0.0

if 'audio_end_time' not in st.session_state:
	st.session_state[ 'audio_end_time' ] = 0.0

if 'audio_loop' not in st.session_state:
	st.session_state[ 'audio_loop' ] = False

if 'audio_autoplay' not in st.session_state:
	st.session_state[ 'audio_autoplay' ] = False

if 'audio_output' not in st.session_state:
	st.session_state[ 'audio_output' ] = ''

# ------- EMBEDDING-SPECIFIC PARAMETERS ----------------------

if 'embeddings_dimensions' not in st.session_state:
	st.session_state[ 'embeddings_dimensions' ] = 0

if 'embeddings_chunk_size' not in st.session_state:
	st.session_state[ 'embeddings_chunk_size' ] = 0

if 'embeddings_overlap_amount' not in st.session_state:
	st.session_state[ 'embeddings_overlap_amount' ] = 0

if 'embeddings_input_text' not in st.session_state:
	st.session_state[ 'embeddings_input_text' ] = ''

if 'embeddings_encoding_format' not in st.session_state:
	st.session_state[ 'embeddings_encoding_format' ] = ''

if 'embeddings_method' not in st.session_state:
	st.session_state[ 'embeddings_method' ] = ''

# --------FILES-GENERATION PARAMETERS--------------------
if 'files_max_tokens' not in st.session_state:
	st.session_state[ 'files_max_tokens' ] = 0

if 'files_temperature' not in st.session_state:
	st.session_state[ 'files_temperature' ] = 0.0

if 'files_top_percent' not in st.session_state:
	st.session_state[ 'files_top_percent' ] = 0.0

if 'files_frequency_penalty' not in st.session_state:
	st.session_state[ 'files_frequency_penalty' ] = 0.0

if 'files_presense_penalty' not in st.session_state:
	st.session_state[ 'files_presense_penalty' ] = 0.0

if 'files_background' not in st.session_state:
	st.session_state[ 'files_background' ] = False

if 'files_store' not in st.session_state:
	st.session_state[ 'files_store' ] = False

if 'files_stream' not in st.session_state:
	st.session_state[ 'files_stream' ] = False

if 'files_tool_choice' not in st.session_state:
	st.session_state[ 'files_tool_choice' ] = ''

if 'files_reasoning' not in st.session_state:
	st.session_state[ 'files_reasoning' ] = ''

if 'files_response_format' not in st.session_state:
	st.session_state[ 'files_response_format' ] = ''

if 'files_input' not in st.session_state:
	st.session_state[ 'files_input' ] = ''

if 'files_media_resolution' not in st.session_state:
	st.session_state[ 'files_media_resolution' ] = ''

if 'files_stops' not in st.session_state:
	st.session_state[ 'files_stops' ] = [ ]

if 'files_includes' not in st.session_state:
	st.session_state[ 'files_includes' ] = [ ]

if 'files_tools' not in st.session_state:
	st.session_state.files_tools: List[ Dict[ str, Any ] ] = [ ]

if 'files_context' not in st.session_state:
	st.session_state.files_context: List[ Dict[ str, Any ] ] = [ ]

# ------- FILES-SPECIFIC PARAMETERS --------------------------

if 'files_purpose' not in st.session_state:
	st.session_state[ 'files_purpose' ] = ''

if 'files_type' not in st.session_state:
	st.session_state[ 'files_type' ] = ''

if 'files_id' not in st.session_state:
	st.session_state[ 'files_id' ] = ''

if 'files_url' not in st.session_state:
	st.session_state[ 'files_url' ] = ''

if 'files_table' not in st.session_state:
	st.session_state[ 'files_table' ] = ''

if 'files_messages' not in st.session_state:
	st.session_state[ 'files_messages' ] = [ ]

# -------- VECTORSTORES-GENERATION PARAMETERS --------------------

if 'stores_temperature' not in st.session_state:
	st.session_state[ 'stores_temperature' ] = 0.0

if 'stores_top_percent' not in st.session_state:
	st.session_state[ 'stores_top_percent' ] = 0.0

if 'stores_max_tokens' not in st.session_state:
	st.session_state[ 'stores_max_tokens' ] = 0

if 'stores_frequency_penalty' not in st.session_state:
	st.session_state[ 'stores_frequency_penalty' ] = 0.0

if 'stores_presense_penalty' not in st.session_state:
	st.session_state[ 'stores_presense_penalty' ] = 0.0

if 'stores_max_calls' not in st.session_state:
	st.session_state[ 'stores_max_calls' ] = 0

if 'stores_tool_choice' not in st.session_state:
	st.session_state[ 'stores_tool_choice' ] = ''

if 'stores_response_format' not in st.session_state:
	st.session_state[ 'stores_response_format' ] = ''

if 'stores_reasoning' not in st.session_state:
	st.session_state[ 'stores_reasoning' ] = ''

if 'stores_resolution' not in st.session_state:
	st.session_state[ 'stores_resolution' ] = ''

if 'stores_media_resolution' not in st.session_state:
	st.session_state[ 'stores_media_resolution' ] = ''

if 'stores_parallel_tools' not in st.session_state:
	st.session_state[ 'stores_parallel_tools' ] = False

if 'stores_background' not in st.session_state:
	st.session_state[ 'stores_background' ] = False

if 'stores_store' not in st.session_state:
	st.session_state[ 'stores_store' ] = False

if 'stores_stream' not in st.session_state:
	st.session_state[ 'stores_stream' ] = False

if 'stores_input' not in st.session_state:
	st.session_state[ 'stores_input' ] = [ ]

if 'stores_tools' not in st.session_state:
	st.session_state[ 'stores_tools' ] = [ ]

if 'stores_messages' not in st.session_state:
	st.session_state[ 'stores_messages' ] = [ ]

if 'stores_stops' not in st.session_state:
	st.session_state[ 'stores_stops' ] = [ ]

if 'stores_include' not in st.session_state:
	st.session_state[ 'stores_include' ] = [ ]

# ------- VECTORSTORES-SPECIFIC PARAMETERS -------------------
if 'stores_id' not in st.session_state:
	st.session_state[ 'stores_id' ] = ''

# ------ DOCQNA GENERATION PARAMETERS -----------------
if 'docqna_max_tools' not in st.session_state:
	st.session_state[ 'docqna_max_tools' ] = 0

if 'docqna_max_tokens' not in st.session_state:
	st.session_state[ 'docqna_max_tokens' ] = 0

if 'docqna_max_calls' not in st.session_state:
	st.session_state[ 'docqna_max_calls' ] = 0

if 'docqna_temperature' not in st.session_state:
	st.session_state[ 'docqna_temperature' ] = 0.0

if 'docqna_top_percent' not in st.session_state:
	st.session_state[ 'docqna_top_percent' ] = 0.0

if 'docqna_frequency_penalty' not in st.session_state:
	st.session_state[ 'docqna_frequency_penalty' ] = 0.0

if 'docqna_presense_penalty' not in st.session_state:
	st.session_state[ 'docqna_presense_penalty' ] = 0.0

# --------DOCQNA PARAMETERS--------------------
if 'docqna_number' not in st.session_state:
	st.session_state[ 'docqna_number' ] = 0

if 'docqna_top_k' not in st.session_state:
	st.session_state[ 'docqna_top_k' ] = 0

if 'docqna_max_searches' not in st.session_state:
	st.session_state[ 'docqna_max_searches' ] = 0

if 'docqna_parallel_tools' not in st.session_state:
	st.session_state[ 'docqna_parallel_tools' ] = False

if 'docqna_background' not in st.session_state:
	st.session_state[ 'docqna_background' ] = False

if 'docqna_store' not in st.session_state:
	st.session_state[ 'docqna_store' ] = False

if 'docqna_stream' not in st.session_state:
	st.session_state[ 'docqna_stream' ] = False

if 'docqna_response_format' not in st.session_state:
	st.session_state[ 'docqna_response_format' ] = ''

if 'docqna_tool_choice' not in st.session_state:
	st.session_state[ 'docqna_tool_choice' ] = ''

if 'docqna_resolution' not in st.session_state:
	st.session_state[ 'docqna_resolution' ] = ''

if 'docqna_media_resolution' not in st.session_state:
	st.session_state[ 'docqna_media_resolution' ] = ''

if 'docqna_reasoning' not in st.session_state:
	st.session_state[ 'docqna_reasoning' ] = ''

if 'docqna_input' not in st.session_state:
	st.session_state[ 'docqna_input' ] = ''

if 'docqna_stops' not in st.session_state:
	st.session_state[ 'docqna_stops' ] = [ ]

if 'docqna_modalities' not in st.session_state:
	st.session_state[ 'docqna_modalities' ] = [ ]

if 'docqna_include' not in st.session_state:
	st.session_state[ 'docqna_include' ] = [ ]

if 'docqna_domains' not in st.session_state:
	st.session_state[ 'docqna_domains' ] = [ ]

if 'docqna_tools' not in st.session_state:
	st.session_state[ 'docqna_tools' ] = [ ]

if 'docqna_context' not in st.session_state:
	st.session_state[ 'docqna_context' ] = [ ]

if 'docqna_content' not in st.session_state:
	st.session_state[ 'docqna_content' ] = [ ]

# ------- DOCQA-SPECIFIC PARAMATERS  ---------------------------
if 'docqna_files' not in st.session_state:
	st.session_state[ 'docqna_files' ] = [ ]

if 'docqna_uploaded' not in st.session_state:
	st.session_state[ 'docqna_uploaded' ] = ''

if 'docqna_messages' not in st.session_state:
	st.session_state.docqna_messages = [ ]

if 'docqna_active_docs' not in st.session_state:
	st.session_state.docqna_active_docs = [ ]

if 'docqna_source' not in st.session_state:
	st.session_state.docqna_source = ''

if 'docqna_multi_mode' not in st.session_state:
	st.session_state.docqna_multi_mode = False

# ------- TOKEN PARAMATERS  ---------------------------
if 'last_answer' not in st.session_state:
	st.session_state.last_answer = ''

if 'last_sources' not in st.session_state:
	st.session_state.last_sources = [ ]

if 'last_analysis' not in st.session_state:
	st.session_state.last_analysis = {
			'tables': [ ],
			'docqna_files': [ ],
			'text': [ ],
	}

if 'last_call_usage' not in st.session_state:
	st.session_state.last_call_usage = {
			'prompt_tokens': 0,
			'completion_tokens': 0,
			'total_tokens': 0, }

if 'token_usage' not in st.session_state:
	st.session_state.token_usage = { 'prompt_tokens': 0, 'completion_tokens': 0,
	                                 'total_tokens': 0, }

# ==============================================================================
# LLM UTILITIES
# ==============================================================================

@st.cache_resource
def load_embedder( ) -> SentenceTransformer:
	"""
	
		Purpose:
		--------
		Load the sentence-transformers embedder used by the local prompt builder.
	
		Parameters:
		-----------
		None
	
		Returns:
		--------
		SentenceTransformer
			Loaded embedding model.
			
	"""
	return SentenceTransformer( 'all-MiniLM-L6-v2' )

def resolve_gemini_api_key( ) -> Optional[ str ]:
	"""
	
		Resolve Gemini API key using the following precedence:
		1) Session override (user-entered)
		2) config.py default
		3) Environment variable (optional fallback)
		
	"""
	session_key = st.session_state.get( "gemini_api_key" )
	if session_key:
		return session_key
	
	cfg_key = getattr( cfg, "GOOGLE_API_KEY", None )
	if cfg_key:
		return cfg_key
	
	return os.environ.get( "GOOGLE_API_KEY" )

def _apply_gemini_runtime_config( ) -> None:
	"""
		
		Ensure Gemini client initializes in API-key mode (not Vertex AI).
	
		This avoids: "Project/location and API key are mutually exclusive in the client initializer."
		
	"""
	key = resolve_gemini_api_key( )
	if key:
		os.environ[ "GOOGLE_API_KEY" ] = key
	
	# Ensure project/location do not get passed when using API key mode.
	# gemini.py reads these from the shared config module at runtime.
	try:
		setattr( cfg, "GOOGLE_CLOUD_PROJECT", None )
	except Exception:
		pass
	try:
		setattr( cfg, "GOOGLE_CLOUD_LOCATION", None )
	except Exception:
		pass

# ==============================================================================
# RESPONSE/CHAT UTILITIES
# ==============================================================================

def extract_response_text( response: object ) -> str:
	"""
		
		Purpose:
		--------
		Safely extract assistant text from a Responses API object.
	
		Parameters:
		-----------
		response (object): The response returned from the OpenAI client.
	
		Returns:
		--------
		str: Concatenated assistant text output. Empty string if none found.
		
	"""
	if response is None:
		return ""
	
	output = getattr( response, "output", None )
	if not output or not isinstance( output, list ):
		return ""
	
	text_chunks: list[ str ] = [ ]
	
	for item in output:
		if not hasattr( item, "type" ):
			continue
		
		if item.type == "message":
			content = getattr( item, "content", None )
			if not content or not isinstance( content, list ):
				continue
			
			for part in content:
				if getattr( part, "type", None ) == "output_text":
					text = getattr( part, "text", "" )
					if text:
						text_chunks.append( text )
	
	return "".join( text_chunks ).strip( )

def convert_xml( text: str ) -> str:
	"""
		
			Purpose:
			_________
			Convert XML-delimited prompt text into Markdown by treating XML-like
			tags as section delimiters, not as strict XML.
	
			Parameters:
			-----------
			text (str) - Prompt text containing XML-like opening and closing tags.
	
			Returns:
			---------
			Markdown-formatted text using level-2 headings (##).
	"""
	markdown_blocks: List[ str ] = [ ]
	for match in cfg.XML_BLOCK_PATTERN.finditer( text ):
		raw_tag: str = match.group( "tag" )
		body: str = match.group( "body" ).strip( )
		
		# Humanize tag name for Markdown heading
		heading: str = raw_tag.replace( "_", " " ).replace( "-", " " ).title( )
		markdown_blocks.append( f"## {heading}" )
		if body:
			markdown_blocks.append( body )
	return "\n\n".join( markdown_blocks )

def convert_markdown( text: Any ) -> str:
	"""
		Purpose:
		--------
		Convert between Markdown headings and simple XML-like heading tags.
	
		Behavior:
		---------
		Auto-detects direction:
		  - If <h1>...</h1> / <h2>...</h2> ... exist, converts to Markdown (# / ## / ###).
		  - Otherwise converts Markdown headings (# / ## / ###) to <hN>...</hN> tags.
	
		Parameters:
		-----------
		text : Any
			Source text. Non-string values return "".
	
		Returns:
		--------
		str
			Converted text.
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return ""
	
	# Normalize newlines
	src = text.replace( "\r\n", "\n" ).replace( "\r", "\n" )
	
	htag_pattern = re.compile( r"<h([1-6])>(.*?)</h\1>", flags=re.IGNORECASE | re.DOTALL )
	md_heading_pattern = re.compile( r"^(#{1,6})[ \t]+(.+?)[ \t]*$", flags=re.MULTILINE )
	
	# ------------------------------------------------------------------
	# Direction detection
	# ------------------------------------------------------------------
	contains_htags = bool( htag_pattern.search( src ) )
	
	# ------------------------------------------------------------------
	# XML-like heading tags -> Markdown headings
	# ------------------------------------------------------------------
	if contains_htags:
		def _htag_to_md( match: re.Match ) -> str:
			level = int( match.group( 1 ) )
			content = match.group( 2 ).strip( )
			
			# Preserve inner newlines safely by collapsing interior whitespace
			# while keeping content readable.
			content = re.sub( r"[ \t]+\n", "\n", content )
			content = re.sub( r"\n[ \t]+", "\n", content )
			
			return f"{'#' * level} {content}"
		
		out = htag_pattern.sub( _htag_to_md, src )
		return out.strip( )
	
	# ------------------------------------------------------------------
	# Markdown headings -> XML-like heading tags
	# ------------------------------------------------------------------
	def _md_to_htag( match: re.Match ) -> str:
		hashes = match.group( 1 )
		content = match.group( 2 ).strip( )
		level = len( hashes )
		return f"<h{level}>{content}</h{level}>"
	
	out = md_heading_pattern.sub( _md_to_htag, src )
	return out.strip( )

def encode_image_base64( path: str ) -> str:
	"""
	
		Purpose:
		_________
		
		Parametes:
		----------
		
		
		Returns:
		--------
		
		
	"""
	data = Path( path ).read_bytes( )
	return base64.b64encode( data ).decode( "utf-8" )

def normalize_text( text: str ) -> str:
	"""
		
		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace
	
		Parameters
		----------
		text: str
	
		Returns
		-------
		str
		
	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", "", text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, max_tokens: int = 400 ) -> list[ str ]:
	"""
		
		Purpose
		-------
		Segment normalized text into chunks by:
			1. Sentence boundaries
			2. Fallback to token windowing if needed
	
		Parameters
		----------
		text: str
		max_tokens: int
	
		Returns
		-------
		list[str]
		
	"""
	if not text:
		return [ ]
	
	# Sentence-based segmentation
	sentences = re.split( r"(?<=[.!?])\s+", text )
	sentences = [ s.strip( ) for s in sentences if s.strip( ) ]
	
	if len( sentences ) > 1:
		return sentences
	
	# Fallback: token window segmentation
	words = text.split( )
	chunks = [ ]
	current_chunk = [ ]
	token_count = 0
	
	for word in words:
		current_chunk.append( word )
		token_count += 1
		
		if token_count >= max_tokens:
			chunks.append( " ".join( current_chunk ) )
			current_chunk = [ ]
			token_count = 0
	
	if current_chunk:
		chunks.append( " ".join( current_chunk ) )
	
	return chunks

def cosine_sim( a: np.ndarray, b: np.ndarray ) -> float:
	denom = np.linalg.norm( a ) * np.linalg.norm( b )
	return float( np.dot( a, b ) / denom ) if denom else 0.0

def sanitize_markdown( text: str ) -> str:
	"""
	
		Purpose:
		_________
		
		
	"""
	# Remove bold markers
	text = re.sub( r"\*\*(.*?)\*\*", r"\1", text )
	# Optional: remove italics
	text = re.sub( r"\*(.*?)\*", r"\1", text )
	return text

def inject_response_css( ) -> None:
	"""
	
		Purpose:
		_________
		Set the the format via css.
		
	"""
	st.markdown(
		"""
		<style>
		/* Chat message text */
		.stChatMessage p {
			color: rgb(220, 220, 220);
			font-size: 1rem;
			line-height: 1.6;
		}

		/* Headings inside chat responses */
		.stChatMessage h1 {
			color: rgb(0, 120, 252); /* DoD Blue */
			font-size: 1.6rem;
		}

		.stChatMessage h2 {
			color: rgb(0, 120, 252);
			font-size: 1.35rem;
		}

		.stChatMessage h3 {
			color: rgb(0, 120, 252);
			font-size: 1.15rem;
		}
		
		.stChatMessage a {
			color: rgb(0, 120, 252); /* DoD Blue */
			text-decoration: underline;
		}
		
		.stChatMessage a:hover {
			color: rgb(80, 160, 255);
		}

		</style>
		""", unsafe_allow_html=True )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
	
def init_state( ) -> None:
	"""
	
		Purpose:
		_________
		Initializes all session state variables.
		
		
	"""
	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = [ ]
	
	if 'chat_messages' not in st.session_state:
		st.session_state.chat_messages = [ ]
		
	if 'execution_mode' not in st.session_state:
		st.session_state.execution_mode = 'Standard'
		
	for k in ( 'audio_system_instructions',
				'image_system_instructions',
				'docqna_system_instructions',
				'text_system_instructions' ):
		st.session_state.setdefault( k, "" )

def reset_state( ) -> None:
	"""
	
		Purpose:
		_________
		Resets the session state to default values
		
	"""
	st.session_state.chat_history = [ ]
	st.session_state.last_answer = ""
	st.session_state.last_sources = [ ]
	st.session_state.last_analysis = {
			'tables': [ ],
			'files': [ ],
			'text': [ ],
	}

def normalize( obj ):
	if obj is None or isinstance( obj, (str, int, float, bool) ):
		return obj
	
	if isinstance( obj, dict ):
		return { k: normalize( v ) for k, v in obj.items( ) }
	
	if isinstance( obj, (list, tuple, set) ):
		return [ normalize( v ) for v in obj ]
	if hasattr( obj, "model_dump" ):
		try:
			return obj.model_dump( )
		except Exception:
			return str( obj )
	return str( obj )

def extract_sources( response: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		_________
		Parses-out sources from structured response object.
		
		Parameters:
		------------
		response: Any
			Structured API response.
		
		Returns:
		---------
		List[ Dict[ str, Any ] ]
			List of normalized source dictionaries.
	
	"""
	sources: List[ Dict[ str, Any ] ] = [ ]
	
	if response is None:
		return sources
	
	output = getattr( response, 'output', None )
	if not isinstance( output, list ):
		return sources
	
	for item in output:
		if item is None:
			continue
		
		t = getattr( item, 'type', None )
		
		# ------------------------------------------------
		# Web search
		# ------------------------------------------------
		if t == 'web_search_call':
			action = getattr( item, 'action', None )
			raw = getattr( action, 'sources', None ) if action else None
			
			if not isinstance( raw, (list, tuple) ):
				continue
			
			for src in raw:
				s = normalize( src )
				if not isinstance( s, dict ):
					continue
				
				sources.append( { 'title': s.get( 'title' ), 'snippet': s.get( 'snippet' ),
						'url': s.get( 'url' ), 'files_id': None, } )
		
		# ------------------------------------------------
		# File search (vector store)
		# ------------------------------------------------
		elif t == 'file_search_call':
			raw = getattr( item, 'results', None )
			
			if not isinstance( raw, (list, tuple) ):
				continue
			
			for r in raw:
				s = normalize( r )
				if not isinstance( s, dict ):
					continue
				
				sources.append( { 'title': s.get( 'file_name' ) or s.get( 'title' ),
						'snippet': s.get( 'text' ), 'url': None, 'files_id': s.get( 'files_id' ), } )
	
	return sources

def save_temp( upload ) ->  str | None:
	"""
		Purpose:
		--------
		Save a Streamlit UploadedFile object to a temporary file on disk
		and return the filesystem path.
	
		Parameters:
		-----------
		upload : streamlit.runtime.uploaded_file_manager.UploadedFile
			Uploaded file object from st.file_uploader.
	
		Returns:
		--------
		str | None
			Path to the temporary file, or None if invalid input.
	"""
	if upload is None:
		return None
	
	try:
		_, ext = os.path.splitext( upload.name )
		ext = ext or ""
		with tempfile.NamedTemporaryFile( delete=False, suffix=ext ) as tmp:
			tmp.write( upload.getbuffer( ) )
			tmp_path = tmp.name
		
		return tmp_path
	except Exception:
		return None

def _extract_usage_from_response( resp: Any ) -> Dict[ str, int ]:
	"""
	
		Purpose:
		_________
		Extract token usage from a response object/dict.
		Returns dict with prompt_tokens, completion_tokens, total_tokens.
		Defensive: returns zeros if not present.
		
	"""
	usage = { 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, }
	if not resp:
		return usage
	
	raw = None
	try:
		raw = getattr( resp, "usage", None )
	except Exception:
		raw = None
	
	if not raw and isinstance( resp, dict ):
		raw = resp.get( "usage" )
	
	if not raw:
		return usage
	
	try:
		if isinstance( raw, dict ):
			usage[ "prompt_tokens" ] = int( raw.get( "prompt_tokens", 0 ) )
			usage[ "completion_tokens" ] = int(
				raw.get( "completion_tokens", raw.get( "output_tokens", 0 ) )
			)
			usage[ "total_tokens" ] = int(
				raw.get(
					"total_tokens",
					usage[ "prompt_tokens" ] + usage[ "completion_tokens" ],
				)
			)
		else:
			usage[ "prompt_tokens" ] = int( getattr( raw, "prompt_tokens", 0 ) )
			usage[ "completion_tokens" ] = int(
				getattr( raw, "completion_tokens", getattr( raw, "output_tokens", 0 ) ) )
			usage[ "total_tokens" ] = int(
				getattr( raw, "total_tokens",
					usage[ "prompt_tokens" ] + usage[ "completion_tokens" ], ) )
	except Exception:
		usage[ "total_tokens" ] = (usage[ "prompt_tokens" ] + usage[ "completion_tokens" ])
	
	return usage

def update_token_counters( resp: Any ) -> None:
	"""
	
		Purpose:
		_________
		Update session_state.last_call_usage and accumulate into session_state.token_usage.
		
	"""
	usage = _extract_usage_from_response( resp )
	st.session_state.last_call_usage = usage
	st.session_state.token_usage[ "prompt_tokens" ] += usage.get( "prompt_tokens", 0 )
	st.session_state.token_usage[ "completion_tokens" ] += usage.get( "completion_tokens", 0 )
	st.session_state.token_usage[ "total_tokens" ] += usage.get( "total_tokens", 0 )

def _display_value( val: Any ) -> str:
	"""
		Render a friendly display string for header values.
		None -> em dash; otherwise str(value).
	"""
	if val is None:
		return "—"
	try:
		return str( val )
	except Exception:
		return "—"

def build_intent_prefix( mode: str ) -> str:
	if mode == 'Guidance Only':
		return (
				'[ANALYST INTENT]\n'
				'Respond using authoritative policy and guidance only. '
				'Do not perform financial computation.\n\n'
		)
	if mode == 'Analysis Only':
		return (
				'[ANALYST INTENT]\n'
				'Respond using financial analysis and computation only. '
				'Minimize policy citation.\n\n'
		)
	return ''

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		return conn.execute( "SELECT role, content FROM chat_history ORDER BY id" ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

def format_results( results ):
	formatted_results = ''
	for result in results.data:
		formatted_result = f"<li> '{result.name}'"
		formatted_results += formatted_result + "</li>"
	return f"<p>{formatted_results}</p>"

def count_tokens( text: str ) -> int:
	"""
		
		Purpose
		----------
		Returns the number of tokens in a text string.
		
		Parmeters
		-----------
		string : str
		encoding_name : str
		
		Return
		------------
		int
		
	"""
	encoding = tiktoken.get_encoding( 'cl100k_base' )
	num_tokens = len( encoding.encode( text ) )
	return num_tokens

# ==============================================================================
# PROMPT ENGINEERING UTILITIES
# ==============================================================================

def fetch_prompt_names( db_path: str ) -> list[ str ]:
	"""
		Purpose:
		--------
		Retrieve template names from Prompts table.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
	
		Returns:
		--------
		list[str]
			Sorted prompt names.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Caption FROM Prompts ORDER BY PromptsId;" )
		rows = cur.fetchall( )
		conn.close( )
		return [ r[ 0 ] for r in rows if r and r[ 0 ] is not None ]
	except Exception:
		return [ ]

def fetch_prompt_text( db_path: str, name: str ) -> str | None:
	"""
		Purpose:
		--------
		Retrieve template text by name.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
		name : str
			Template name.
	
		Returns:
		--------
		str | None
			Prompt text if found.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Text FROM Prompts WHERE Caption = ?;", (name,) )
		row = cur.fetchone( )
		conn.close( )
		return str( row[ 0 ] ) if row and row[ 0 ] is not None else None
	except Exception:
		return None

def fetch_prompts_df( ) -> pd.DataFrame:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		df = pd.read_sql_query(
			"SELECT PromptsId, Caption,  Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
			conn )
	df.insert( 0, "Selected", False )
	return df

def fetch_prompt_by_id( pid: int ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
			(pid,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def fetch_prompt_by_name( name: str ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE Caption=?",
			(name,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def insert_prompt( data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO Prompts (Caption, Name, Text, Version, ID) VALUES (?, ?, ?, ?)',
			(data[ 'Caption' ], data[ 'Name' ], data[ 'Text' ], data[ 'Version' ], data[ 'ID' ]) )

def update_prompt( pid: int, data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"UPDATE Prompts SET Caption=?, Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
			(data[ "Caption" ], data[ "Name" ], data[ "Text" ], data[ "Version" ], data[ "ID" ], pid)
		)

def delete_prompt( pid: int ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM Prompts WHERE PromptsId=?", (pid,) )

def build_prompt( user_input: str ) -> str:
	"""
		Purpose:
		--------
		Build a llama.cpp-compatible prompt using the application's system instructions, optional
		retrieval context (semantic + basic RAG), and the current in-memory chat history.

		Parameters:
		-----------
		user_input : str
			The current user turn to append to the prompt.

		Returns:
		--------
		str
			A fully constructed prompt in chat template format.
	"""
	system_instructions = st.session_state.get( 'system_instructions', '' )
	use_semantic = bool( st.session_state.get( 'use_semantic', False ) )
	basic_docs = st.session_state.get( 'basic_docs', [ ] )
	messages = st.session_state.get( 'messages', [ ] )
	
	top_k_value = int( st.session_state.get( 'top_k', 0 ) )
	if top_k_value <= 0:
		top_k_value = 4
	
	prompt = f"<|system|>\n{system_instructions}\n</s>\n"
	
	if use_semantic:
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			rows = conn.execute( "SELECT chunk, vector FROM embeddings" ).fetchall( )
		
		if rows:
			q = embedder.encode( [ user_input ] )[ 0 ]
			scored = [ (c, cosine_sim( q, np.frombuffer( v ) )) for c, v in rows ]
			for c, _ in sorted( scored, key=lambda x: x[ 1 ], reverse=True )[ :top_k_value ]:
				prompt += f"<|system|>\n{c}\n</s>\n"
	
	for d in basic_docs[ :6 ]:
		prompt += f"<|system|>\n{d}\n</s>\n"
	
	if isinstance( messages, list ):
		for msg in messages:
			role = ''
			content = ''
			
			if isinstance( msg, tuple ) or isinstance( msg, list ):
				if len( msg ) == 2:
					role = str( msg[ 0 ] or '' ).strip( )
					content = str( msg[ 1 ] or '' )
			elif isinstance( msg, dict ):
				role = str( msg.get( 'role', '' ) or '' ).strip( )
				content = str( msg.get( 'content', '' ) or '' )
			
			if role:
				prompt += f"<|{role}|>\n{content}\n</s>\n"
	
	prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
	return prompt

DM_DB_PATH = os.path.join( 'stores', 'sqlite', 'Data.db' )
os.makedirs( os.path.dirname( DM_DB_PATH ), exist_ok=True )

# ==============================================================================
# DATABASE UTILITIES
# ==============================================================================

def initialize_database( ) -> None:
	Path( "stores/sqlite" ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS chat_history
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          role
                          TEXT,
                          content
                          TEXT
                      )
		              """ )
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS embeddings
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          chunk
                          TEXT,
                          vector
                          BLOB
                      )
		              """ )
		conn.execute( """
                      CREATE TABLE IF NOT EXISTS Prompts
                      (
                          PromptsId
                          INTEGER
                          NOT
                          NULL
                          UNIQUE,
                          Name
                          TEXT
                      (
                          80
                      ),
                          Text TEXT,
                          Version TEXT
                      (
                          80
                      ),
                          ID TEXT
                      (
                          80
                      ),
                          PRIMARY KEY
                      (
                          PromptsId
                          AUTOINCREMENT
                      )
                          )
		              """ )

def create_connection( ) -> sqlite3.Connection:
	return sqlite3.connect( DM_DB_PATH )

def list_tables( ) -> List[ str ]:
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int = None, offset: int=0 ) -> pd.DataFrame:
	query = f'SELECT rowid, * FROM "{table}"'
	if limit:
		query += f" LIMIT {limit} OFFSET {offset}"
	with create_connection( ) as conn:
		return pd.read_sql_query( query, conn )

def drop_table( table: str ) -> None:
	"""
		Purpose:
		--------
		Safely drop a table if it exists.
	
		Parameters:
		-----------
		table : str
			Table name.
	"""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def create_index( table: str, column: str ) -> None:
	"""
		Purpose:
		--------
		Create a safe SQLite index on a specified table column.
	
		Handles:
			- Spaces in column names
			- Special characters
			- Reserved words
			- Duplicate index names
			- Validation against actual table schema
	
		Parameters:
		-----------
		table : str
			Table name.
		column : str
			Column name to index.
	"""
	if not table or not column:
		return
	
	# ------------------------------------------------------------------
	# Validate table exists
	# ------------------------------------------------------------------
	tables = list_tables( )
	if table not in tables:
		raise ValueError( "Invalid table name." )
	
	# ------------------------------------------------------------------
	# Validate column exists
	# ------------------------------------------------------------------
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( "Invalid column name." )
	
	# ------------------------------------------------------------------
	# Sanitize index name (identifier only)
	# ------------------------------------------------------------------
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ------------------------------------------------------------------
	# Create index safely (quote identifiers)
	# ------------------------------------------------------------------
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: pd.DataFrame ) -> pd.DataFrame:
	st.subheader( 'Advanced Filters' )
	conditions = [ ]
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=',  '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: pd.DataFrame ):
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: pd.DataFrame ):
	st.subheader( 'Visualization Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	categorical_cols = df.select_dtypes( include=[ 'object' ] ).columns.tolist( )
	
	chart = st.selectbox( 'Chart Type', [ 'Histogram', 'Bar', 'Line',
			'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram' and numeric_cols:
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.histogram( df, x=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.bar( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.line( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		x = st.selectbox( 'X', numeric_cols )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.scatter( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.box( df, y=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		col = st.selectbox( 'Category Column', categorical_cols )
		fig = px.pie( df, names=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation' and len( numeric_cols ) > 1:
		corr = df[ numeric_cols ].corr( )
		fig = px.imshow( corr, text_auto=True )
		st.plotly_chart( fig, use_container_width=True )

def dm_create_table_from_df( table_name: str, df: pd.DataFrame ):
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}')
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: pd.DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	"""
		Purpose:
		--------
		Map a pandas dtype to an appropriate SQLite column type.
	
		Parameters:
		-----------
		dtype : pandas dtype
			The dtype of a pandas Series.
	
		Returns:
		--------
		str
			SQLite column type.
	"""
	dtype_str = str( dtype ).lower( )
	
	# ------------------------------------------------------------------
	# Integer Types (including nullable Int64)
	# ------------------------------------------------------------------
	if "int" in dtype_str:
		return "INTEGER"
	
	# ------------------------------------------------------------------
	# Float Types
	# ------------------------------------------------------------------
	if "float" in dtype_str:
		return "REAL"
	
	# ------------------------------------------------------------------
	# Boolean
	# ------------------------------------------------------------------
	if "bool" in dtype_str:
		return "INTEGER"
	
	# ------------------------------------------------------------------
	# Datetime
	# ------------------------------------------------------------------
	if "datetime" in dtype_str:
		return "TEXT"
	
	# ------------------------------------------------------------------
	# Categorical
	# ------------------------------------------------------------------
	if "category" in dtype_str:
		return "TEXT"
	
	# ------------------------------------------------------------------
	# Default fallback
	# ------------------------------------------------------------------
	return "TEXT"

def create_custom_table( table_name: str, columns: list ) -> None:
	"""
		Purpose:
		--------
		Create a custom SQLite table from column definitions.
	
		Parameters:
		-----------
		table_name : str
			Name of table.
	
		columns : list of dict
			[
				{
					"name": str,
					"type": str,
					"not_null": bool,
					"primary_key": bool,
					"auto_increment": bool
				}
			]
	"""
	if not table_name:
		raise ValueError( "Table name required." )
	
	# Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( "Invalid table name." )
	
	col_defs = [ ]
	
	for col in columns:
		col_name = col[ "name" ]
		col_type = col[ "type" ].upper( )
		
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		
		if col[ "primary_key" ]:
			definition += " PRIMARY KEY"
			if col[ "auto_increment" ] and col_type == "INTEGER":
				definition += " AUTOINCREMENT"
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
		
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether a SQL query is read-only and safe to execute.
	
		Allows:
			SELECT
			WITH (CTE returning SELECT)
			EXPLAIN SELECT
			PRAGMA (read-only)
	
		Blocks:
			INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH,
			DETACH, VACUUM, REPLACE, TRIGGER, and multiple statements.
			
	"""	
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ------------------------------------------------------------------
	# Block multiple statements
	# ------------------------------------------------------------------
	if ';' in q[ :-1 ]:
		return False
	
	# ------------------------------------------------------------------
	# Remove SQL comments
	# ------------------------------------------------------------------
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ------------------------------------------------------------------
	# Allowed starting keywords
	# ------------------------------------------------------------------
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ------------------------------------------------------------------
	# Block dangerous keywords anywhere
	# ------------------------------------------------------------------
	blocked_keywords = ( 'insert ', 'update ', 'delete ', 'drop ', 'alter ',
			'create ', 'attach ', 'detach ', 'vacuum ', 'replace ', 'trigger ' )
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""
	
		Purpose:
		--------
		Sanitize a string into a safe SQLite identifier.
	
		- Replaces invalid characters with underscores
		- Ensures it starts with a letter or underscore
		- Prevents empty names
		
	"""
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )

	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'

	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	with create_connection( ) as conn:
		rows = conn.execute(f'PRAGMA index_list("{table}");').fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute(
			f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};')
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]		
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )		
		row = \
		{ 
				'column': col, 'dtype': str( series.dtype ),
				'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
				'distinct_%': round( (distinct_count / total_rows) * 100, 2 ) if total_rows else 0,
		}
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ "min" ] = series.min( )
			row[ "max" ] = series.max( )
			row[ "mean" ] = series.mean( )
		else:
			row[ "min" ] = None
			row[ "max" ] = None
			row[ "mean" ] = None
		
		profile_rows.append( row )
	
	return pd.DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( "Table and column required." )
	
	with create_connection( ) as conn:
		# ------------------------------------------------------------
		# Fetch original CREATE TABLE statement
		# ------------------------------------------------------------
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		# ------------------------------------------------------------
		# Extract column definitions
		# ------------------------------------------------------------
		open_paren = create_sql.find( "(" )
		close_paren = create_sql.rfind( ")" )
		
		if open_paren == -1 or close_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		inner = create_sql[ open_paren + 1: close_paren ]
		
		column_defs = [ c.strip( ) for c in inner.split( "," ) ]
		
		# Remove target column
		new_defs = [ ]
		for col_def in column_defs:
			col_name = col_def.split( )[ 0 ].strip( '"' )
			if col_name != column:
				new_defs.append( col_def )
		
		if len( new_defs ) == len( column_defs ):
			raise ValueError( "Column not found." )
		
		# ------------------------------------------------------------
		# Build new CREATE TABLE statement
		# ------------------------------------------------------------
		temp_table = f"{table}_rebuild_temp"
		
		new_create_sql = (
				f'CREATE TABLE "{temp_table}" ('
				+ ", ".join( new_defs )
				+ ");"
		)
		
		# ------------------------------------------------------------
		# Begin transaction
		# ------------------------------------------------------------
		conn.execute( "BEGIN" )
		
		conn.execute( new_create_sql )
		
		remaining_cols = [
				c.split( )[ 0 ].strip( '"' )
				for c in new_defs
		]
		
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_table}" ({col_list}) '
			f'SELECT {col_list} FROM "{table}";'
		)
		
		# Preserve indexes
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table,)
		).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute(
			f'ALTER TABLE "{temp_table}" RENAME TO "{table}";'
		)
		
		# Recreate indexes
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

def rename_table( old_name: str, new_name: str ) -> None:
	"""
		Purpose:
		--------
		Rename an existing SQLite table. Attempts native ALTER TABLE rename first; if it fails,
		falls back to a schema-safe rebuild using the original CREATE TABLE statement and
		preserves indexes.

		Parameters:
		-----------
		old_name : str
			Existing table name.

		new_name : str
			New table name.

		Returns:
		--------
		None
	"""
	if not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(old_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(old_name,)
		).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";'
		)
		
		conn.execute( f'DROP TABLE "{old_name}";' )
		conn.execute( f'ALTER TABLE "{temp_name}" RENAME TO "{new_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'ON "{old_name}"', f'ON "{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def rename_column( table_name: str, old_name: str, new_name: str ) -> None:
	"""
		Purpose:
		--------
		Rename a column within an existing SQLite table. Attempts native ALTER TABLE rename
		first; if it fails, falls back to a schema-safe rebuild preserving column order, data,
		and indexes.

		Parameters:
		-----------
		table_name : str
			Table containing the column.

		old_name : str
			Existing column name.

		new_name : str
			New column name.

		Returns:
		--------
		None
	"""
	if not table_name or not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute(
				f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";'
			)
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table_name,)
		).fetchall( )
		
		schema = conn.execute( f'PRAGMA table_info("{table_name}");' ).fetchall( )
		cols = [ r[ 1 ] for r in schema ]
		if old_name not in cols:
			raise ValueError( "Column not found." )
		
		mapped_cols = [ (new_name if c == old_name else c) for c in cols ]
		
		temp_table = f"{table_name}__rebuild_temp"
		
		col_defs: List[ str ] = [ ]
		pk_cols = [ r for r in schema if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		for row in schema:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			out_name = new_name if col_name == old_name else col_name
			col_def = f'"{out_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			col_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( col_defs )});'
		
		old_select = ", ".join( [ f'"{c}"' for c in cols ] )
		new_insert = ", ".join( [ f'"{c}"' for c in mapped_cols ] )
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		conn.execute(
			f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";'
		)
		
		conn.execute( f'DROP TABLE "{table_name}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'"{old_name}"', f'"{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

# ======================================================================================
# PROVIDER UTILITIES
# ======================================================================================

PROVIDER_MODULES: Dict[ str, Any ] = {
		'GPT': gpt_provider,
		'Gemini': gemini_provider,
		'Grok': grok_provider,
}

def ensure_session_key( key: str, default: Any ) -> None:
	"""
	
		Purpose:
		--------
		Ensure a Streamlit session-state key exists before any widget with the same key
		is instantiated.
		
		Parameters:
		-----------
		key: str
			Session-state key to initialize.
		
		default: Any
			Default value assigned only when the key is missing.
		
		Returns:
		--------
		None
		
	"""
	if key not in st.session_state:
		st.session_state[ key ] = default

def get_provider_name( provider_name: Optional[ str ] = None ) -> str:
	"""
	
		Purpose:
		--------
		Return a safe provider name for Buddy's provider-routed application modes.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name. When omitted, the function reads
			st.session_state['provider'].
		
		Returns:
		--------
		str
			Provider name constrained to GPT, Gemini, or Grok.
		
	"""
	selected = provider_name
	
	if selected is None:
		selected = st.session_state.get( 'provider', 'GPT' )
	
	if not isinstance( selected, str ) or selected.strip( ) not in PROVIDER_MODULES:
		return 'GPT'
	
	return selected.strip( )

def get_provider_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return the provider module associated with the selected provider name.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name. When omitted, the active session provider is used.
		
		Returns:
		--------
		Any
			Imported provider module.
		
	"""
	return PROVIDER_MODULES[ get_provider_name( provider_name ) ]

def provider_supports( capability_name: str, provider_name: Optional[ str ] = None ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether a provider module exposes a named capability class.
		
		Parameters:
		-----------
		capability_name: str
			Class or capability name to test.
		
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		bool
			True when the selected provider exposes the capability; otherwise False.
		
	"""
	if not isinstance( capability_name, str ) or not capability_name.strip( ):
		return False
	
	provider_module = get_provider_module( provider_name )
	return hasattr( provider_module, capability_name.strip( ) )

def get_provider_capability( capability_name: str, provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a provider capability class from the selected provider module.
		
		Parameters:
		-----------
		capability_name: str
			Provider capability class name, such as Chat, Files, VectorStores, FileSearch,
			or CloudBuckets.
		
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Provider capability class.
		
	"""
	provider = get_provider_name( provider_name )
	provider_module = get_provider_module( provider )
	name = str( capability_name or '' ).strip( )
	
	if not name:
		raise AttributeError( 'Provider capability name cannot be empty.' )
	
	if not hasattr( provider_module, name ):
		raise AttributeError( f'{provider} provider does not expose {name}.' )
	
	return getattr( provider_module, name )

def create_provider_capability( capability_name: str, provider_name: Optional[ str ]=None ) -> Any:
	"""
	
		Purpose:
		--------
		Instantiate a provider capability class from the selected provider module.
		
		Parameters:
		-----------
		capability_name: str
			Provider capability class name.
		
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Initialized provider wrapper instance.
		
	"""
	capability = get_provider_capability( capability_name=capability_name,
		provider_name=provider_name )
	
	return capability( )

def get_chat_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a provider-specific Chat wrapper instance.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Provider Chat wrapper instance.
		
	"""
	return create_provider_capability( 'Chat', provider_name )

def get_images_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a provider-specific Images wrapper instance.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Provider Images wrapper instance.
		
	"""
	return create_provider_capability( 'Images', provider_name )

def get_embeddings_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a provider-specific Embeddings wrapper instance.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Provider Embeddings wrapper instance.
		
	"""
	return create_provider_capability( 'Embeddings', provider_name )

def get_tts_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a provider-specific Text-to-Speech wrapper instance.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Provider TTS wrapper instance.
		
	"""
	return create_provider_capability( 'TTS', provider_name )

def get_transcription_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a provider-specific Transcription wrapper instance.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Provider Transcription wrapper instance.
		
	"""
	return create_provider_capability( 'Transcription', provider_name )

def get_translation_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a provider-specific Translation wrapper instance.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Provider Translation wrapper instance.
		
	"""
	return create_provider_capability( 'Translation', provider_name )

def get_files_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a provider-specific Files wrapper instance.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Provider Files wrapper instance.
		
	"""
	return create_provider_capability( 'Files', provider_name )

def get_file_search_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a Gemini FileSearch wrapper instance.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name. This must resolve to Gemini.
		
		Returns:
		--------
		Any
			Gemini FileSearch wrapper instance.
		
	"""
	provider = get_provider_name( provider_name )
	
	if provider != 'Gemini':
		raise AttributeError( f'{provider} does not expose Gemini FileSearch.' )
	
	return create_provider_capability( 'FileSearch', provider )

def get_cloudbuckets_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return a Gemini CloudBuckets wrapper instance.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name. This must resolve to Gemini.
		
		Returns:
		--------
		Any
			Gemini CloudBuckets wrapper instance.
		
	"""
	provider = get_provider_name( provider_name )
	
	if provider != 'Gemini':
		raise AttributeError( f'{provider} does not expose Gemini CloudBuckets.' )
	
	return create_provider_capability( 'CloudBuckets', provider )

def get_cloud_buckets_module( provider_name: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Backward-compatible alias for get_cloudbuckets_module().
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Any
			Gemini CloudBuckets wrapper instance.
		
	"""
	return get_cloudbuckets_module( provider_name )

def get_gemini_vector_backend( ) -> str:
	"""
	
		Purpose:
		--------
		Return the selected Gemini backend used under Buddy's visible Vector Stores alias.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		str
			Either File Search Stores or Cloud Buckets.
		
	"""
	backend = st.session_state.get( 'stores_backend', 'File Search Stores' )
	
	if backend not in [ 'File Search Stores', 'Cloud Buckets' ]:
		return 'File Search Stores'
	
	return backend

def get_vectorstores_module( provider_name: Optional[ str ] = None,
		backend: Optional[ str ] = None ) -> Any:
	"""
	
		Purpose:
		--------
		Return the storage wrapper used by Buddy's visible Vector Stores mode.
		
		Provider routing:
			GPT     -> gpt.VectorStores()
			Grok    -> grok.VectorStores()
			Gemini  -> gemini.FileSearch() or gemini.CloudBuckets()
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		backend: Optional[str]
			Optional Gemini backend override. Valid values are File Search Stores and
			Cloud Buckets.
		
		Returns:
		--------
		Any
			Provider storage wrapper instance.
		
	"""
	provider = get_provider_name( provider_name )
	
	if provider in [ 'GPT', 'Grok' ]:
		return create_provider_capability( 'VectorStores', provider )
	
	if provider == 'Gemini':
		selected_backend = backend if backend is not None else get_gemini_vector_backend( )
		
		if selected_backend == 'Cloud Buckets':
			return get_cloudbuckets_module( provider )
		
		return get_file_search_module( provider )
	
	raise AttributeError( f'{provider} does not expose a Vector Stores backend.' )

def get_vectorstores_backend_name( provider_name: Optional[ str ] = None,
		backend: Optional[ str ] = None ) -> str:
	"""
	
		Purpose:
		--------
		Return the concrete backend label used by the visible Vector Stores mode.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		backend: Optional[str]
			Optional Gemini backend override.
		
		Returns:
		--------
		str
			Concrete backend label.
		
	"""
	provider = get_provider_name( provider_name )
	
	if provider == 'GPT':
		return 'OpenAI Vector Stores'
	
	if provider == 'Grok':
		return 'xAI Collections'
	
	if provider == 'Gemini':
		selected_backend = backend if backend is not None else get_gemini_vector_backend( )
		return f'Gemini {selected_backend}'
	
	return 'Unsupported Storage Backend'

def _provider( ) -> str:
	"""
	
		Purpose:
		--------
		Return the active provider name for legacy option helper functions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		str
			Active provider name.
		
	"""
	return get_provider_name( )

def _safe( module_name: str, attr_name: str, fallback: Any ) -> Any:
	"""
	
		Purpose:
		--------
		Safely retrieve an attribute from a provider module while preserving a fallback.
		
		Parameters:
		-----------
		module_name: str
			Provider module name: gpt, gemini, or grok.
		
		attr_name: str
			Attribute name to retrieve.
		
		fallback: Any
			Fallback value returned when the provider attribute does not exist.
		
		Returns:
		--------
		Any
			Resolved attribute value or fallback.
		
	"""
	provider_modules = {
			'gpt': gpt_provider,
			'gemini': gemini_provider,
			'grok': grok_provider,
	}
	
	provider_module = provider_modules.get( str( module_name or '' ).strip( ).lower( ) )
	
	if provider_module is None:
		return fallback
	
	return getattr( provider_module, attr_name, fallback )

# ======================================================================================
# VECTOR STORES STATE UTILITIES
# ======================================================================================

def ensure_vectorstores_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure all non-widget Vector Stores session-state keys exist before Buddy's
		Vector Stores mode instantiates controls.
		
		Important:
		----------
		This function intentionally does not initialize any st.file_uploader widget keys.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	# ------------------------------------------------------------------
	# Shared Vector Stores alias keys
	# ------------------------------------------------------------------
	ensure_session_key( 'stores_backend', 'File Search Stores' )
	ensure_session_key( 'stores_id', '' )
	ensure_session_key( 'stores_name', '' )
	ensure_session_key( 'stores_description', '' )
	ensure_session_key( 'stores_metadata', '' )
	ensure_session_key( 'stores_manual_id', '' )
	ensure_session_key( 'stores_selected_id', '' )
	ensure_session_key( 'stores_selected_label', '' )
	ensure_session_key( 'stores_table', [ ] )
	ensure_session_key( 'stores_files_table', [ ] )
	ensure_session_key( 'stores_store_metadata', { } )
	ensure_session_key( 'stores_file_metadata', { } )
	ensure_session_key( 'stores_operation_result', { } )
	ensure_session_key( 'stores_batch_result', { } )
	ensure_session_key( 'stores_upload_result', { } )
	ensure_session_key( 'stores_delete_result', { } )
	ensure_session_key( 'stores_search_result', { } )
	ensure_session_key( 'stores_survey_result', { } )
	ensure_session_key( 'stores_query', '' )
	ensure_session_key( 'stores_answer', '' )
	ensure_session_key( 'stores_last_operation', '' )
	ensure_session_key( 'stores_file_id', '' )
	ensure_session_key( 'stores_file_ids', [ ] )
	ensure_session_key( 'stores_file_ids_text', '' )
	ensure_session_key( 'stores_batch_id', '' )
	ensure_session_key( 'stores_content', '' )
	ensure_session_key( 'stores_limit', 100 )
	ensure_session_key( 'stores_order', 'desc' )
	
	# ------------------------------------------------------------------
	# Gemini File Search backend keys
	# ------------------------------------------------------------------
	ensure_session_key( 'filestore_id', '' )
	ensure_session_key( 'filestore_name', '' )
	ensure_session_key( 'filestore_selected_id', '' )
	ensure_session_key( 'filestore_selected_label', '' )
	ensure_session_key( 'filestore_metadata', { } )
	ensure_session_key( 'filestore_table', [ ] )
	ensure_session_key( 'filestore_upload_result', { } )
	ensure_session_key( 'filestore_delete_result', { } )
	ensure_session_key( 'filestore_operation_result', { } )
	
	# ------------------------------------------------------------------
	# Gemini Cloud Bucket backend keys
	# ------------------------------------------------------------------
	ensure_session_key( 'bucket_name', '' )
	ensure_session_key( 'bucket_object_name', '' )
	ensure_session_key( 'bucket_selected_id', '' )
	ensure_session_key( 'bucket_selected_label', '' )
	ensure_session_key( 'bucket_metadata', { } )
	ensure_session_key( 'bucket_table', [ ] )
	ensure_session_key( 'bucket_upload_result', { } )
	ensure_session_key( 'bucket_delete_result', { } )
	ensure_session_key( 'bucket_operation_result', { } )
	ensure_session_key( 'bucket_results', { } )
	
	# ------------------------------------------------------------------
	# Shared storage display keys
	# ------------------------------------------------------------------
	ensure_session_key( 'storage_operation_result', { } )
	ensure_session_key( 'storage_table_data', [ ] )
	ensure_session_key( 'storage_last_operation', '' )
	ensure_session_key( 'storage_selected_option', '' )
	ensure_session_key( 'storage_last_answer', '' )
	
	# ------------------------------------------------------------------
	# Defensive type normalization for non-widget keys only
	# ------------------------------------------------------------------
	if not isinstance( st.session_state.get( 'stores_table' ), list ):
		st.session_state[ 'stores_table' ] = [ ]
	
	if not isinstance( st.session_state.get( 'stores_files_table' ), list ):
		st.session_state[ 'stores_files_table' ] = [ ]
	
	if not isinstance( st.session_state.get( 'stores_store_metadata' ), dict ):
		st.session_state[ 'stores_store_metadata' ] = { }
	
	if not isinstance( st.session_state.get( 'stores_file_metadata' ), dict ):
		st.session_state[ 'stores_file_metadata' ] = { }
	
	if not isinstance( st.session_state.get( 'stores_operation_result' ), dict ):
		st.session_state[ 'stores_operation_result' ] = { }
	
	if not isinstance( st.session_state.get( 'filestore_table' ), list ):
		st.session_state[ 'filestore_table' ] = [ ]
	
	if not isinstance( st.session_state.get( 'filestore_metadata' ), dict ):
		st.session_state[ 'filestore_metadata' ] = { }
	
	if not isinstance( st.session_state.get( 'bucket_table' ), list ):
		st.session_state[ 'bucket_table' ] = [ ]
	
	if not isinstance( st.session_state.get( 'bucket_metadata' ), dict ):
		st.session_state[ 'bucket_metadata' ] = { }
	
	if not isinstance( st.session_state.get( 'storage_operation_result' ), dict ):
		st.session_state[ 'storage_operation_result' ] = { }
	
	if not isinstance( st.session_state.get( 'storage_table_data' ), list ):
		st.session_state[ 'storage_table_data' ] = [ ]
	
	if st.session_state.get( 'stores_backend' ) not in [ 'File Search Stores', 'Cloud Buckets' ]:
		st.session_state[ 'stores_backend' ] = 'File Search Stores'

def ensure_storage_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Backward-compatible alias for ensure_vectorstores_mode_state().
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_vectorstores_mode_state( )

# ======================================================================================
# VECTOR STORES STORAGE UTILITIES
# ======================================================================================

def normalize_storage_object( value: Any ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Normalize a provider storage, collection, bucket, file, or operation response into
		a serializable dictionary that can be displayed safely in Streamlit.
		
		Parameters:
		-----------
		value: Any
			Provider object, dictionary, model object, scalar, or None.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized metadata dictionary.
		
	"""
	if value is None:
		return { }
	
	if isinstance( value, dict ):
		return value
	
	if hasattr( value, 'model_dump' ):
		try:
			return value.model_dump( )
		except Exception:
			pass
	
	if hasattr( value, 'to_dict' ):
		try:
			result = value.to_dict( )
			if isinstance( result, dict ):
				return result
		except Exception:
			pass
	
	if hasattr( value, '__dict__' ):
		try:
			row: Dict[ str, Any ] = { }
			
			for key, item in vars( value ).items( ):
				if str( key ).startswith( '_' ):
					continue
				
				if item is None or isinstance( item, (str, int, float, bool) ):
					row[ key ] = item
					continue
				
				if isinstance( item, (list, tuple, set) ):
					row[ key ] = list( item )
					continue
				
				if isinstance( item, dict ):
					row[ key ] = item
					continue
				
				row[ key ] = str( item )
			
			return row
		except Exception:
			pass
	
	return { 'value': str( value ) }

def normalize_storage_rows( value: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Normalize provider list responses into display-ready metadata rows.
		
		Parameters:
		-----------
		value: Any
			Provider list response, pager, list, dictionary, or scalar object.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Normalized metadata rows.
		
	"""
	if value is None:
		return [ ]
	
	if isinstance( value, list ):
		return [ normalize_storage_object( item ) for item in value ]
	
	if isinstance( value, tuple ):
		return [ normalize_storage_object( item ) for item in value ]
	
	if isinstance( value, dict ):
		for key in [ 'data', 'items', 'files', 'buckets', 'collections', 'stores',
		             'file_search_stores', 'vector_stores', 'results' ]:
			items = value.get( key )
			if isinstance( items, list ):
				return [ normalize_storage_object( item ) for item in items ]
		
		return [ normalize_storage_object( value ) ]
	
	for attr_name in [ 'data', 'items', 'files', 'buckets', 'collections', 'stores',
	                   'file_search_stores', 'vector_stores', 'results' ]:
		try:
			items = getattr( value, attr_name, None )
			if isinstance( items, list ):
				return [ normalize_storage_object( item ) for item in items ]
		except Exception:
			continue
	
	try:
		return [ normalize_storage_object( item ) for item in value ]
	except Exception:
		return [ normalize_storage_object( value ) ]

def parse_storage_ids( value: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Parse comma-, semicolon-, newline-, or list-delimited storage IDs into a clean
		string list.
		
		Parameters:
		-----------
		value: Any
			Raw text, list, tuple, set, or None.
		
		Returns:
		--------
		List[str]
			Clean storage IDs.
		
	"""
	if value is None:
		return [ ]
	
	if isinstance( value, (list, tuple, set) ):
		return [
				str( item ).strip( )
				for item in value
				if str( item ).strip( )
		]
	
	if not isinstance( value, str ):
		return [ str( value ).strip( ) ] if str( value ).strip( ) else [ ]
	
	text = value.strip( )
	if not text:
		return [ ]
	
	parts = re.split( r'[,;\n\r\t]+', text )
	
	return [
			part.strip( )
			for part in parts
			if part.strip( )
	]

def parse_storage_json( value: Any ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Parse optional JSON metadata or configuration text into a dictionary.
		
		Parameters:
		-----------
		value: Any
			Raw JSON string, dictionary, or None.
		
		Returns:
		--------
		Dict[str, Any]
			Parsed dictionary, or an empty dictionary when omitted.
		
	"""
	if value is None:
		return { }
	
	if isinstance( value, dict ):
		return value
	
	if not isinstance( value, str ) or not value.strip( ):
		return { }
	
	try:
		result = json.loads( value.strip( ) )
		return result if isinstance( result, dict ) else { 'value': result }
	except Exception as exc:
		raise ValueError( f'Invalid JSON metadata: {exc}' ) from exc

def get_storage_identifier( row: Dict[ str, Any ] ) -> str:
	"""
	
		Purpose:
		--------
		Return the most likely identifier from a normalized storage metadata row.
		
		Parameters:
		-----------
		row: Dict[str, Any]
			Normalized storage metadata row.
		
		Returns:
		--------
		str
			Storage identifier or empty string.
		
	"""
	if not isinstance( row, dict ):
		return ''
	
	for key in [
			'id',
			'name',
			'resource_name',
			'resourceName',
			'uri',
			'file_id',
			'store_id',
			'vector_store_id',
			'collection_id',
			'bucket',
			'bucket_name',
			'bucketName',
	]:
		value = row.get( key )
		if isinstance( value, str ) and value.strip( ):
			return value.strip( )
	
	return ''

def get_storage_display_name( row: Dict[ str, Any ] ) -> str:
	"""
	
		Purpose:
		--------
		Return the most useful display name from a normalized storage metadata row.
		
		Parameters:
		-----------
		row: Dict[str, Any]
			Normalized storage metadata row.
		
		Returns:
		--------
		str
			Human-readable display name.
		
	"""
	if not isinstance( row, dict ):
		return 'resource'
	
	for key in [
			'display_name',
			'displayName',
			'filename',
			'file_name',
			'name',
			'id',
			'bucket_name',
			'bucketName',
			'collection',
			'title',
	]:
		value = row.get( key )
		if isinstance( value, str ) and value.strip( ):
			return value.strip( )
	
	return 'resource'

def build_storage_selectors( rows: List[ Dict[ str, Any ] ] ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Build user-facing selection labels from normalized storage metadata rows.
		
		Parameters:
		-----------
		rows: List[Dict[str, Any]]
			Normalized metadata rows.
		
		Returns:
		--------
		List[str]
			Selection labels in the form display name — identifier.
		
	"""
	if not isinstance( rows, list ):
		return [ ]
	
	options: List[ str ] = [ ]
	
	for row in rows:
		if not isinstance( row, dict ):
			continue
		
		identifier = get_storage_identifier( row )
		display_name = get_storage_display_name( row )
		
		if identifier:
			options.append( f'{display_name} — {identifier}' )
	
	return options

def get_storage_id_from_option( option: Optional[ str ] ) -> str:
	"""
	
		Purpose:
		--------
		Extract a storage resource identifier from a selection label.
		
		Parameters:
		-----------
		option: Optional[str]
			UI selection label.
		
		Returns:
		--------
		str
			Extracted identifier or empty string.
		
	"""
	if not isinstance( option, str ) or not option.strip( ):
		return ''
	
	text = option.strip( )
	if ' — ' in text:
		return text.rsplit( ' — ', 1 )[ 1 ].strip( )
	
	return text

def get_selected_store_id( manual_key: str='stores_manual_id',
		selected_key: str='stores_selected_id', fallback_key: str='stores_id' ) -> str:
	"""
	
		Purpose:
		--------
		Return the active storage identifier using manual input first, selected row second,
		and fallback state third.
		
		Parameters:
		-----------
		manual_key: str
			Session key for a manual identifier input.
		
		selected_key: str
			Session key for a selected resource identifier.
		
		fallback_key: str
			Session key for a fallback identifier.
		
		Returns:
		--------
		str
			Selected storage identifier or empty string.
		
	"""
	for key in [ manual_key, selected_key, fallback_key ]:
		value = st.session_state.get( key, '' )
		if isinstance( value, str ) and value.strip( ):
			return value.strip( )
	
	return ''

def get_vectorstores_selected_id( ) -> str:
	"""
	
		Purpose:
		--------
		Return the selected provider-specific storage identifier for the visible Vector
		Stores mode.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		str
			Selected vector store, collection, File Search Store, or bucket identifier.
		
	"""
	provider_name = get_provider_name( )
	
	if provider_name == 'Gemini':
		backend = get_gemini_vector_backend( )
		
		if backend == 'Cloud Buckets':
			return get_selected_store_id(
				manual_key='bucket_name',
				selected_key='bucket_selected_id',
				fallback_key='bucket_name' )
		
		return get_selected_store_id(
			manual_key='filestore_id',
			selected_key='filestore_selected_id',
			fallback_key='filestore_id' )
	
	return get_selected_store_id(
		manual_key='stores_manual_id',
		selected_key='stores_selected_id',
		fallback_key='stores_id' )

def call_storage_method( target: Any, method_names: List[ str ], *args: Any, **kwargs: Any ) -> Any:
	"""
	
		Purpose:
		--------
		Call the first compatible method exposed by a provider storage wrapper. This
		absorbs wrapper method-name differences across GPT, Grok, and Gemini.
		
		Parameters:
		-----------
		target: Any
			Provider storage wrapper instance.
		
		method_names: List[str]
			Ordered method names to attempt.
		
		*args: Any
			Positional arguments forwarded to the provider method.
		
		**kwargs: Any
			Keyword arguments forwarded to the provider method.
		
		Returns:
		--------
		Any
			Provider method result.
		
	"""
	if target is None:
		raise ValueError( 'Storage target cannot be None.' )
	
	if not isinstance( method_names, list ) or len( method_names ) == 0:
		raise ValueError( 'At least one storage method name is required.' )
	
	last_error: Optional[ Exception ] = None
	for method_name in method_names:
		if not isinstance( method_name, str ) or not method_name.strip( ):
			continue
		
		name = method_name.strip( )
		
		if not hasattr( target, name ):
			continue
		
		method = getattr( target, name )
		
		if not callable( method ):
			continue
		
		try:
			return method( *args, **kwargs )
		except TypeError as exc:
			last_error = exc
			
			if len( kwargs ) > 0 and len( args ) > 0:
				try:
					return method( *args )
				except TypeError as inner_exc:
					last_error = inner_exc
			
			if len( kwargs ) > 0:
				try:
					return method( **kwargs )
				except TypeError as inner_exc:
					last_error = inner_exc
			
			continue
		except Exception:
			raise
	
	if last_error is not None:
		raise AttributeError(
			f'No compatible storage method was callable. Last error: {last_error}' )
	
	raise AttributeError(
		f'Target does not expose any of these methods: {", ".join( method_names )}' )

def save_uploaded_storage_file( uploaded_file: Any ) -> str:
	"""
	
		Purpose:
		--------
		Save a Streamlit uploaded file to a temporary path for provider upload calls.
		
		Parameters:
		-----------
		uploaded_file: Any
			Streamlit UploadedFile object.
		
		Returns:
		--------
		str
			Temporary file path.
		
	"""
	if uploaded_file is None:
		raise ValueError( 'An uploaded file is required.' )
	
	name = getattr( uploaded_file, 'name', 'uploaded_file' )
	suffix = Path( name ).suffix
	if not suffix:
		suffix = '.bin'
	
	with tempfile.NamedTemporaryFile( delete=False, suffix=suffix ) as tmp:
		if hasattr( uploaded_file, 'getbuffer' ):
			tmp.write( uploaded_file.getbuffer( ) )
		elif hasattr( uploaded_file, 'read' ):
			tmp.write( uploaded_file.read( ) )
		else:
			raise ValueError( 'Uploaded file object does not expose getbuffer() or read().' )
		
		return tmp.name

def set_storage_rows( rows: Any, table_key: str = 'storage_table_data' ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Normalize storage rows and write them to shared and optional provider-specific
		session-state table keys.
		
		Parameters:
		-----------
		rows: Any
			Provider list response.
		
		table_key: str
			Session-state table key to update.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Normalized rows.
		
	"""
	normalized_rows = normalize_storage_rows( rows )
	st.session_state[ 'storage_table_data' ] = normalized_rows
	st.session_state[ table_key ] = normalized_rows
	
	return normalized_rows

def set_storage_result( result: Any, operation: str,
		result_key: str = 'storage_operation_result' ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Normalize a provider operation result and store it in shared and provider-specific
		session-state result keys.
		
		Parameters:
		-----------
		result: Any
			Provider operation result.
		
		operation: str
			Operation name stored for diagnostics and display.
		
		result_key: str
			Provider-specific session-state result key.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized operation result.
		
	"""
	normalized = normalize_storage_object( result )
	st.session_state[ 'storage_operation_result' ] = normalized
	st.session_state[ result_key ] = normalized
	st.session_state[ 'storage_last_operation' ] = operation
	st.session_state[ 'stores_last_operation' ] = operation
	
	return normalized

def sync_storage_selection( selected_option: Optional[ str ], provider_name: Optional[ str ] = None,
		backend: Optional[ str ]=None ) -> str:
	"""
	
		Purpose:
		--------
		Synchronize a selected storage option into the correct provider-specific session
		keys without modifying the widget-owned storage_selected_option key.
		
		Parameters:
		-----------
		selected_option: Optional[str]
			Selected resource label from the storage selector widget.
		
		provider_name: Optional[str]
			Optional explicit provider name.
		
		backend: Optional[str]
			Optional Gemini backend name.
		
		Returns:
		--------
		str
			Selected identifier.
		
	"""
	selected_id = get_storage_id_from_option( selected_option )
	
	if not selected_id:
		return ''
	
	provider = get_provider_name( provider_name )
	
	if provider == 'Gemini':
		selected_backend = backend if backend is not None else get_gemini_vector_backend( )
		
		if selected_backend == 'Cloud Buckets':
			st.session_state[ 'bucket_selected_id' ] = selected_id
			st.session_state[ 'bucket_selected_label' ] = selected_option or ''
			st.session_state[ 'bucket_name' ] = selected_id
		else:
			st.session_state[ 'filestore_selected_id' ] = selected_id
			st.session_state[ 'filestore_selected_label' ] = selected_option or ''
			st.session_state[ 'filestore_id' ] = selected_id
		
		return selected_id
	
	st.session_state[ 'stores_selected_id' ] = selected_id
	st.session_state[ 'stores_selected_label' ] = selected_option or ''
	st.session_state[ 'stores_id' ] = selected_id
	
	return selected_id

def clear_vectorstore_outputs( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Vector Stores output state while preserving provider and control values.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'stores_table' ] = [ ]
	st.session_state[ 'stores_files_table' ] = [ ]
	st.session_state[ 'stores_store_metadata' ] = { }
	st.session_state[ 'stores_file_metadata' ] = { }
	st.session_state[ 'stores_operation_result' ] = { }
	st.session_state[ 'stores_batch_result' ] = { }
	st.session_state[ 'stores_upload_result' ] = { }
	st.session_state[ 'stores_delete_result' ] = { }
	st.session_state[ 'stores_search_result' ] = { }
	st.session_state[ 'stores_survey_result' ] = { }
	st.session_state[ 'stores_answer' ] = ''
	st.session_state[ 'stores_content' ] = ''
	st.session_state[ 'stores_last_operation' ] = ''
	
	st.session_state[ 'filestore_metadata' ] = { }
	st.session_state[ 'filestore_table' ] = [ ]
	st.session_state[ 'filestore_upload_result' ] = { }
	st.session_state[ 'filestore_delete_result' ] = { }
	st.session_state[ 'filestore_operation_result' ] = { }
	
	st.session_state[ 'bucket_metadata' ] = { }
	st.session_state[ 'bucket_table' ] = [ ]
	st.session_state[ 'bucket_upload_result' ] = { }
	st.session_state[ 'bucket_delete_result' ] = { }
	st.session_state[ 'bucket_operation_result' ] = { }
	st.session_state[ 'bucket_results' ] = { }
	
	st.session_state[ 'storage_operation_result' ] = { }
	st.session_state[ 'storage_table_data' ] = [ ]
	st.session_state[ 'storage_last_operation' ] = ''
	st.session_state[ 'storage_last_answer' ] = ''

def reset_vectorstore_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Vector Stores non-uploader controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [
			'stores_backend',
			'stores_id',
			'stores_name',
			'stores_description',
			'stores_metadata',
			'stores_manual_id',
			'stores_selected_id',
			'stores_selected_label',
			'stores_query',
			'stores_file_id',
			'stores_file_ids',
			'stores_file_ids_text',
			'stores_batch_id',
			'stores_limit',
			'stores_order',
			'filestore_id',
			'filestore_name',
			'filestore_selected_id',
			'filestore_selected_label',
			'bucket_name',
			'bucket_object_name',
			'bucket_selected_id',
			'bucket_selected_label',
			'storage_selected_option',
	]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_vectorstore_all( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Vector Stores controls and output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	reset_vectorstore_controls( )
	clear_vectorstore_outputs( )
	ensure_vectorstores_mode_state( )

def require_storage_value( name: str, value: Any ) -> str:
	"""
	
		Purpose:
		--------
		Validate and return a required storage value.
		
		Parameters:
		-----------
		name: str
			User-facing value name.
		
		value: Any
			Value to validate.
		
		Returns:
		--------
		str
			Clean value string.
		
	"""
	if value is None:
		raise ValueError( f'{name} is required.' )
	
	text = str( value ).strip( )
	
	if not text:
		raise ValueError( f'{name} is required.' )
	
	return text

def get_grok_collections( vectorstores: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Return configured xAI collection rows from the Grok VectorStores wrapper.
		
		Parameters:
		-----------
		vectorstores: Any
			Grok VectorStores wrapper instance.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Configured collection rows.
		
	"""
	for method_names in [ [ 'list_collections', 'list_stores', 'list' ], [ 'survey_collections', 'survey' ], ]:
		try:
			result = call_storage_method( vectorstores, method_names )
			rows = normalize_storage_rows( result )
			if len( rows ) > 0:
				return rows
		except Exception:
			continue
	
	collections = getattr( vectorstores, 'collections', None )
	if isinstance( collections, dict ):
		return [ {
						'name': name,
						'id': collection_id,
						'type': 'xAI Collection',
				} for name, collection_id in collections.items( ) ]
	
	return [ ]

def warn_grok_unsupported_operation( operation_name: str ) -> None:
	"""
	
		Purpose:
		--------
		Display a clear warning for Grok Vector Stores operations that require remote
		collection-management capability.
		
		Parameters:
		-----------
		operation_name: str
			Operation name.
		
		Returns:
		--------
		None
		
	"""
	st.warning(
		f'Grok {operation_name} requires xAI collection-management capability. '
		f'This Buddy integration currently supports configured collection list, '
		f'retrieve, search, and survey operations.' )

def get_storage_backend_summary( provider_name: Optional[ str ] = None,
		backend: Optional[ str ] = None ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Return a backend summary for diagnostics and UI display.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		backend: Optional[str]
			Optional Gemini backend name.
		
		Returns:
		--------
		Dict[str, Any]
			Backend summary.
		
	"""
	provider = get_provider_name( provider_name )
	backend_name = get_vectorstores_backend_name( provider, backend )
	supports_create = provider == 'GPT' or (
			provider == 'Gemini' and get_gemini_vector_backend( ) in [ 'File Search Stores', 'Cloud Buckets', ] )
	
	supports_upload = provider == 'GPT' or provider == 'Gemini'
	supports_delete = provider == 'GPT' or provider == 'Gemini'
	supports_search = provider in [ 'GPT', 'Grok', 'Gemini' ]
	
	return {
			'provider': provider,
			'backend': backend_name,
			'supports_create': supports_create,
			'supports_upload': supports_upload,
			'supports_delete': supports_delete,
			'supports_search': supports_search,
	}

# ======================================================================================
# MODE STATE UTILITIES
# ======================================================================================

def ensure_key( key: str, default: Any ) -> None:
	"""
	
		Purpose:
		--------
		Ensure a Streamlit session-state key exists before widget instantiation.
		
		Parameters:
		-----------
		key: str
			The session-state key to initialize.
		
		default: Any
			The default value assigned only when the key does not already exist.
		
		Returns:
		--------
		None
		
	"""
	if key not in st.session_state:
		st.session_state[ key ] = default

def ensure_common_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure common provider, mode, message, usage, file, instruction, and API-key
		session-state values exist before provider-specific mode sections execute.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_key( 'api_keys', { 'GPT': None, 'Grok': None, 'Gemini': None } )
	ensure_key( 'provider', 'GPT' )
	ensure_key( 'mode', 'Chat' )
	ensure_key( 'messages', [ ] )
	ensure_key( 'chat_history', [ ] )
	ensure_key( 'files', [ ] )
	ensure_key( 'uploaded', [ ] )
	ensure_key( 'use_semantic', False )
	ensure_key( 'is_grounded', False )
	ensure_key( 'selected_prompt_id', '' )
	ensure_key( 'pending_system_prompt_name', '' )
	ensure_key( 'last_answer', '' )
	ensure_key( 'last_sources', [ ] )
	ensure_key( 'last_call_usage', {
				'prompt_tokens': 0,
				'completion_tokens': 0,
				'total_tokens': 0,
		} )
	ensure_key( 'token_usage', {
				'prompt_tokens': 0,
				'completion_tokens': 0,
				'total_tokens': 0,
		} )
	ensure_key( 'last_analysis', {
				'tables': [ ],
				'docqna_files': [ ],
				'files': [ ],
				'text': [ ],
		} )
	
	# ------------------------------------------------------------------
	# API Keys
	# ------------------------------------------------------------------
	ensure_key( 'openai_api_key', '' )
	ensure_key( 'gemini_api_key', '' )
	ensure_key( 'groq_api_key', '' )
	ensure_key( 'xai_api_key', '' )
	ensure_key( 'google_api_key', '' )
	ensure_key( 'google_cse_id', '' )
	ensure_key( 'googlemaps_api_key', '' )
	ensure_key( 'geocoding_api_key', '' )
	ensure_key( 'google_cloud_project_id', '' )
	ensure_key( 'google_cloud_location', '' )
	
	# ------------------------------------------------------------------
	# Shared Model Keys
	# ------------------------------------------------------------------
	ensure_key( 'chat_model', '' )
	ensure_key( 'text_model', '' )
	ensure_key( 'image_model', '' )
	ensure_key( 'audio_model', '' )
	ensure_key( 'embedding_model', '' )
	ensure_key( 'docqna_model', '' )
	ensure_key( 'files_model', '' )
	ensure_key( 'stores_model', '' )
	ensure_key( 'bucket_model', '' )
	ensure_key( 'tts_model', '' )
	ensure_key( 'transcription_model', '' )
	ensure_key( 'translation_model', '' )
	
	# ------------------------------------------------------------------
	# Shared Instruction Keys
	# ------------------------------------------------------------------
	ensure_key( 'instructions', '' )
	ensure_key( 'chat_system_instructions', '' )
	ensure_key( 'text_system_instructions', '' )
	ensure_key( 'image_system_instructions', '' )
	ensure_key( 'audio_system_instructions', '' )
	ensure_key( 'docqna_system_instructions', '' )
	ensure_key( 'stores_system_instructions', '' )
	ensure_key( 'bucket_system_instructions', '' )
	
	# ------------------------------------------------------------------
	# Shared Generation Keys
	# ------------------------------------------------------------------
	ensure_key( 'max_tools', 0 )
	ensure_key( 'max_tokens', 0 )
	ensure_key( 'temperature', 0.0 )
	ensure_key( 'top_percent', 0.0 )
	ensure_key( 'frequency_penalty', 0.0 )
	ensure_key( 'presence_penalty', 0.0 )
	ensure_key( 'presense_penalty', 0.0 )
	ensure_key( 'background', False )
	ensure_key( 'parallel_tools', False )
	ensure_key( 'store', False )
	ensure_key( 'stream', False )
	ensure_key( 'execution_mode', 'Standard' )
	ensure_key( 'response_format', '' )
	ensure_key( 'tool_choice', '' )
	ensure_key( 'reasoning', '' )
	ensure_key( 'stops', [ ] )
	ensure_key( 'include', [ ] )
	ensure_key( 'input', [ ] )
	ensure_key( 'tools', [ ] )

def ensure_text_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Text mode session-state keys from Buddy, Gipity, and Jeni exist before
		Text mode widgets or provider execution paths read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'text_number', 0 )
	ensure_key( 'text_max_calls', 0 )
	ensure_key( 'text_top_k', 0 )
	ensure_key( 'text_max_urls', 0 )
	ensure_key( 'text_max_searches', 0 )
	ensure_key( 'text_max_tokens', 0 )
	ensure_key( 'text_temperature', 0.0 )
	ensure_key( 'text_top_percent', 0.0 )
	ensure_key( 'text_frequency_penalty', 0.0 )
	ensure_key( 'text_presence_penalty', 0.0 )
	ensure_key( 'text_presense_penalty', 0.0 )
	ensure_key( 'text_parallel_tools', False )
	ensure_key( 'text_parallel_calls', False )
	ensure_key( 'text_background', False )
	ensure_key( 'text_store', False )
	ensure_key( 'text_stream', False )
	ensure_key( 'text_google_grounding', False )
	ensure_key( 'text_response_format', '' )
	ensure_key( 'text_tool_choice', '' )
	ensure_key( 'text_resolution', '' )
	ensure_key( 'text_media_resolution', '' )
	ensure_key( 'text_reasoning', '' )
	ensure_key( 'text_response_schema', '' )
	ensure_key( 'text_safety_profile', '' )
	ensure_key( 'text_input', '' )
	ensure_key( 'text_content', '' )
	ensure_key( 'text_previous_response_id', '' )
	ensure_key( 'text_conversation_id', '' )
	ensure_key( 'text_include', [ ] )
	ensure_key( 'text_includes', [ ] )
	ensure_key( 'text_domains', [ ] )
	ensure_key( 'text_urls', [ ] )
	ensure_key( 'text_tools', [ ] )
	ensure_key( 'text_stops', [ ] )
	ensure_key( 'text_modalities', [ ] )
	ensure_key( 'text_context', [ ] )
	ensure_key( 'text_messages', [ ] )
	ensure_key( 'text_gemini_history', [ ] )
	ensure_key( 'text_file_search_store_names', [ ] )
	ensure_key( 'selected_filestore_id', '' )
	ensure_key( 'selected_filestore_label', '' )
	ensure_key( 'text_vector_store_ids', '' )
	ensure_key( 'text_json_schema_name', 'structured_response' )
	ensure_key( 'text_json_schema', '' )
	ensure_key( 'text_json_schema_strict', True )

def ensure_image_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Image mode session-state keys from Buddy, Gipity, and Jeni exist before
		Image mode widgets or provider execution paths read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'image_mode', '' )
	ensure_key( 'image_analysis_model', '' )
	ensure_key( 'image_analysis_detail', 'auto' )
	ensure_key( 'image_max_tokens', 0 )
	ensure_key( 'image_max_calls', 0 )
	ensure_key( 'image_max_searches', 0 )
	ensure_key( 'image_number', 1 )
	ensure_key( 'image_compression', 0.0 )
	ensure_key( 'image_temperature', 0.0 )
	ensure_key( 'image_top_percent', 0.0 )
	ensure_key( 'image_frequency_penalty', 0.0 )
	ensure_key( 'image_presence_penalty', 0.0 )
	ensure_key( 'image_parallel_calls', False )
	ensure_key( 'image_background', False )
	ensure_key( 'image_store', False )
	ensure_key( 'image_stream', False )
	ensure_key( 'image_tool_choice', '' )
	ensure_key( 'image_reasoning', '' )
	ensure_key( 'image_mime_type', '' )
	ensure_key( 'image_response_format', '' )
	ensure_key( 'image_previous_response_id', '' )
	ensure_key( 'image_input', [ ] )
	ensure_key( 'image_include', [ ] )
	ensure_key( 'image_tools', [ ] )
	ensure_key( 'image_modalities', [ ] )
	ensure_key( 'image_modality', '' )
	ensure_key( 'image_messages', [ ] )
	ensure_key( 'image_context', [ ] )
	ensure_key( 'image_domains', [ ] )
	ensure_key( 'image_content', [ ] )
	ensure_key( 'image_output_bytes', None )
	ensure_key( 'image_aspect_ratio', '' )
	ensure_key( 'image_size', '' )
	ensure_key( 'image_quality', '' )
	ensure_key( 'image_backcolor', '' )
	ensure_key( 'image_detail', '' )
	ensure_key( 'image_grounded', False )
	ensure_key( 'image_image_search', False )

def ensure_audio_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Audio mode session-state keys from Buddy, Gipity, and Jeni exist before
		Audio mode widgets or provider execution paths read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'audio_max_tokens', 0 )
	ensure_key( 'audio_temperature', 0.0 )
	ensure_key( 'audio_top_percent', 0.0 )
	ensure_key( 'audio_frequency_penalty', 0.0 )
	ensure_key( 'audio_presence_penalty', 0.0 )
	ensure_key( 'audio_background', False )
	ensure_key( 'audio_store', False )
	ensure_key( 'audio_stream', False )
	ensure_key( 'audio_tool_choice', '' )
	ensure_key( 'audio_reasoning', '' )
	ensure_key( 'audio_response_format', '' )
	ensure_key( 'audio_format', '' )
	ensure_key( 'audio_input', '' )
	ensure_key( 'audio_mime_type', '' )
	ensure_key( 'audio_media_resolution', '' )
	ensure_key( 'audio_stops', [ ] )
	ensure_key( 'audio_include', [ ] )
	ensure_key( 'audio_includes', [ ] )
	ensure_key( 'audio_tools', [ ] )
	ensure_key( 'audio_context', [ ] )
	ensure_key( 'audio_modalities', [ ] )
	ensure_key( 'audio_messages', [ ] )
	ensure_key( 'audio_output_bytes', None )
	
	# ------------------------------------------------------------------
	# Audio-Specific Keys
	# ------------------------------------------------------------------
	ensure_key( 'audio_task', '' )
	ensure_key( 'audio_file', '' )
	ensure_key( 'audio_rate', int( cfg.SAMPLE_RATES[ 0 ] ) if hasattr( cfg, 'SAMPLE_RATES' )
	                                                          and cfg.SAMPLE_RATES else 44100 )
	ensure_key( 'audio_language', '' )
	ensure_key( 'audio_voice', '' )
	ensure_key( 'audio_start_time', 0.0 )
	ensure_key( 'audio_end_time', 0.0 )
	ensure_key( 'audio_loop', False )
	ensure_key( 'audio_autoplay', False )
	ensure_key( 'audio_output', '' )

def ensure_embeddings_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Embeddings mode session-state keys from Buddy, Gipity, and Jeni exist
		before Embeddings widgets or provider execution paths read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'embedding_model', '' )
	ensure_key( 'embeddings_dimensions', 0 )
	ensure_key( 'embeddings_chunk_size', 800 )
	ensure_key( 'embeddings_overlap_amount', 0 )
	ensure_key( 'embeddings_input_text', '' )
	ensure_key( 'embeddings_encoding_format', 'float' )
	ensure_key( 'embeddings_method', '' )
	ensure_key( 'embeddings_user', '' )
	ensure_key( 'embeddings', [ ] )
	ensure_key( 'embeddings_chunks', [ ] )
	ensure_key( 'embeddings_df', pd.DataFrame( ) )
	ensure_key( 'embedding_metrics', { } )
	ensure_key( 'embedding_usage', { } )

def ensure_docqna_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Document Q&A mode session-state keys from Buddy, Gipity, and Jeni exist
		before Document Q&A widgets, local retrieval, Files API, or Vector Store paths read
		them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	# ------------------------------------------------------------------
	# Document Q&A Generation Keys
	# ------------------------------------------------------------------
	ensure_key( 'docqna_max_tools', 0 )
	ensure_key( 'docqna_max_tokens', 0 )
	ensure_key( 'docqna_max_calls', 0 )
	ensure_key( 'docqna_temperature', 0.0 )
	ensure_key( 'docqna_top_percent', 0.0 )
	ensure_key( 'docqna_frequency_penalty', 0.0 )
	ensure_key( 'docqna_presence_penalty', 0.0 )
	ensure_key( 'docqna_number', 0 )
	ensure_key( 'docqna_top_k', 6 )
	ensure_key( 'docqna_max_searches', 0 )
	ensure_key( 'docqna_parallel_tools', False )
	ensure_key( 'docqna_background', False )
	ensure_key( 'docqna_store', False )
	ensure_key( 'docqna_stream', False )
	ensure_key( 'docqna_response_format', '' )
	ensure_key( 'docqna_tool_choice', '' )
	ensure_key( 'docqna_resolution', '' )
	ensure_key( 'docqna_media_resolution', '' )
	ensure_key( 'docqna_reasoning', '' )
	ensure_key( 'docqna_input', '' )
	ensure_key( 'docqna_stops', [ ] )
	ensure_key( 'docqna_modalities', [ ] )
	ensure_key( 'docqna_include', [ ] )
	ensure_key( 'docqna_domains', [ ] )
	ensure_key( 'docqna_tools', [ ] )
	ensure_key( 'docqna_context', '' )
	ensure_key( 'docqna_content', [ ] )
	
	# ------------------------------------------------------------------
	# Document Q&A Source and Retrieval Keys
	# ------------------------------------------------------------------
	ensure_key( 'docqna_source', 'Local Upload' )
	ensure_key( 'doc_source', 'uploadlocal' )
	ensure_key( 'docqna_uploaded', None )
	ensure_key( 'docqna_files', [ ] )
	ensure_key( 'docqna_active_docs', [ ] )
	ensure_key( 'active_docs', [ ] )
	ensure_key( 'docqna_bytes', None )
	ensure_key( 'doc_bytes', { } )
	ensure_key( 'docqna_texts', { } )
	ensure_key( 'docqna_chunks', [ ] )
	ensure_key( 'docqna_messages', [ ] )
	ensure_key( 'docqna_multi_mode', False )
	ensure_key( 'docqna_vec_ready', False )
	ensure_key( 'docqna_fingerprint', '' )
	ensure_key( 'docqna_chunk_count', 0 )
	ensure_key( 'docqna_fallback_rows', [ ] )
	ensure_key( 'docqna_last_hits', [ ] )
	ensure_key( 'docqna_last_sources', [ ] )
	ensure_key( 'docqna_last_answer', '' )
	ensure_key( 'docqna_index_status', 'Not indexed' )
	ensure_key( 'docqna_backend', 'local' )
	ensure_key( 'docqna_show_diagnostics', True )
	ensure_key( 'docqna_file_id', '' )
	ensure_key( 'docqna_vector_store_id', '' )
	ensure_key( 'docqna_chunk_size', 900 )
	ensure_key( 'docqna_chunk_overlap', 150 )

def ensure_files_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Files mode session-state keys from Buddy, Gipity, and Jeni exist before
		Files mode widgets or provider execution paths read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'files_max_tokens', 0 )
	ensure_key( 'files_temperature', 0.0 )
	ensure_key( 'files_top_percent', 0.0 )
	ensure_key( 'files_frequency_penalty', 0.0 )
	ensure_key( 'files_presence_penalty', 0.0 )
	ensure_key( 'files_background', False )
	ensure_key( 'files_store', False )
	ensure_key( 'files_stream', False )
	ensure_key( 'files_tool_choice', '' )
	ensure_key( 'files_reasoning', '' )
	ensure_key( 'files_response_format', '' )
	ensure_key( 'files_input', '' )
	ensure_key( 'files_media_resolution', '' )
	ensure_key( 'files_stops', [ ] )
	ensure_key( 'files_include', [ ] )
	ensure_key( 'files_includes', [ ] )
	ensure_key( 'files_tools', [ ] )
	ensure_key( 'files_context', [ ] )
	ensure_key( 'files_messages', [ ] )
	
	# ------------------------------------------------------------------
	# Files-Specific Keys
	# ------------------------------------------------------------------
	ensure_key( 'files_purpose', '' )
	ensure_key( 'files_type', '' )
	ensure_key( 'files_id', '' )
	ensure_key( 'files_url', '' )
	ensure_key( 'files_table', '' )
	ensure_key( 'files_uploaded', None )
	ensure_key( 'files_path', '' )
	ensure_key( 'files_operation', '' )

def ensure_vectorstores_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Vector Stores mode session-state keys from Buddy, Gipity, and Jeni exist
		before storage widgets or provider execution paths read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'stores_temperature', 0.0 )
	ensure_key( 'stores_top_percent', 0.0 )
	ensure_key( 'stores_max_tokens', 0 )
	ensure_key( 'stores_frequency_penalty', 0.0 )
	ensure_key( 'stores_presence_penalty', 0.0 )
	ensure_key( 'stores_max_calls', 0 )
	ensure_key( 'stores_tool_choice', '' )
	ensure_key( 'stores_response_format', '' )
	ensure_key( 'stores_reasoning', '' )
	ensure_key( 'stores_resolution', '' )
	ensure_key( 'stores_media_resolution', '' )
	ensure_key( 'stores_parallel_tools', False )
	ensure_key( 'stores_background', False )
	ensure_key( 'stores_store', False )
	ensure_key( 'stores_stream', False )
	ensure_key( 'stores_input', [ ] )
	ensure_key( 'stores_tools', [ ] )
	ensure_key( 'stores_messages', [ ] )
	ensure_key( 'stores_stops', [ ] )
	ensure_key( 'stores_include', [ ] )
	ensure_key( 'stores_includes', [ ] )
	
	# ------------------------------------------------------------------
	# Vector Store-Specific Keys
	# ------------------------------------------------------------------
	ensure_key( 'stores_id', '' )
	ensure_key( 'stores_name', '' )
	ensure_key( 'stores_file_id', '' )
	ensure_key( 'stores_file_ids', [ ] )
	ensure_key( 'stores_uploaded', None )
	ensure_key( 'stores_path', '' )
	ensure_key( 'stores_operation', '' )
	ensure_key( 'stores_collection', '' )

def ensure_file_search_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Gemini File Search Store session-state keys exist before File Search Store
		widgets or Gemini execution paths read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'filestore_model', '' )
	ensure_key( 'filestore_temperature', 0.0 )
	ensure_key( 'filestore_top_percent', 0.0 )
	ensure_key( 'filestore_max_tokens', 0 )
	ensure_key( 'filestore_frequency_penalty', 0.0 )
	ensure_key( 'filestore_presence_penalty', 0.0 )
	ensure_key( 'filestore_max_calls', 0 )
	ensure_key( 'filestore_tool_choice', '' )
	ensure_key( 'filestore_response_format', '' )
	ensure_key( 'filestore_reasoning', '' )
	ensure_key( 'filestore_parallel_tools', False )
	ensure_key( 'filestore_background', False )
	ensure_key( 'filestore_store', False )
	ensure_key( 'filestore_stream', False )
	ensure_key( 'filestore_input', [ ] )
	ensure_key( 'filestore_tools', [ ] )
	ensure_key( 'filestore_messages', [ ] )
	ensure_key( 'filestore_stops', [ ] )
	ensure_key( 'filestore_include', [ ] )
	ensure_key( 'filestore_id', '' )
	ensure_key( 'filestore_name', '' )
	ensure_key( 'filestore_selected_label', '' )

def ensure_cloudbuckets_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Google Cloud Buckets mode session-state keys from Jeni exist before Cloud
		Bucket widgets or Gemini CloudBuckets execution paths read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'bucket_model', '' )
	ensure_key( 'bucket_temperature', 0.0 )
	ensure_key( 'bucket_top_percent', 0.0 )
	ensure_key( 'bucket_max_tokens', 0 )
	ensure_key( 'bucket_frequency_penalty', 0.0 )
	ensure_key( 'bucket_presence_penalty', 0.0 )
	ensure_key( 'bucket_number', 0 )
	ensure_key( 'bucket_max_calls', 0 )
	ensure_key( 'bucket_tool_choice', '' )
	ensure_key( 'bucket_response_format', '' )
	ensure_key( 'bucket_reasoning', '' )
	ensure_key( 'bucket_resolution', '' )
	ensure_key( 'bucket_media_resolution', '' )
	ensure_key( 'bucket_parallel_tools', False )
	ensure_key( 'bucket_background', False )
	ensure_key( 'bucket_store', False )
	ensure_key( 'bucket_stream', False )
	ensure_key( 'bucket_input', [ ] )
	ensure_key( 'bucket_tools', [ ] )
	ensure_key( 'bucket_messages', [ ] )
	ensure_key( 'bucket_stops', [ ] )
	ensure_key( 'bucket_include', [ ] )
	ensure_key( 'bucket_includes', [ ] )
	
	# ------------------------------------------------------------------
	# Cloud Bucket-Specific Keys
	# ------------------------------------------------------------------
	ensure_key( 'bucket_id', '' )
	ensure_key( 'bucket_name', '' )
	ensure_key( 'bucket_object_name', '' )
	ensure_key( 'bucket_path', '' )
	ensure_key( 'bucket_uploaded', None )
	ensure_key( 'bucket_operation', '' )
	ensure_key( 'selected_bucket_id', '' )
	ensure_key( 'selected_bucket_label', '' )

def ensure_export_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Export mode session-state keys exist before Export mode widgets read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'export_format', '' )
	ensure_key( 'export_source', '' )
	ensure_key( 'export_filename', '' )
	ensure_key( 'export_content', '' )

def ensure_data_management_mode_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Data Management mode session-state keys exist before database widgets,
		table editors, chart controls, or export controls read them.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_common_mode_state( )
	
	ensure_key( 'dm_table', '' )
	ensure_key( 'dm_selected_table', '' )
	ensure_key( 'dm_query', '' )
	ensure_key( 'dm_limit', 500 )
	ensure_key( 'dm_offset', 0 )
	ensure_key( 'dm_uploaded', None )
	ensure_key( 'dm_upload_table_name', '' )
	ensure_key( 'dm_chart_type', '' )
	ensure_key( 'dm_filter_column', '' )
	ensure_key( 'dm_filter_operator', '' )
	ensure_key( 'dm_filter_value', '' )
	ensure_key( 'df_current', pd.DataFrame( ) )

def ensure_mode_state( mode_name: str | None = None ) -> None:
	"""
	
		Purpose:
		--------
		Dispatch to the correct mode-state initializer based on the active Buddy mode.
		This should be called before each mode section creates widgets.
		
		Parameters:
		-----------
		mode_name: str | None
			Optional mode name. When None, reads st.session_state[ 'mode' ].
		
		Returns:
		--------
		None
		
	"""
	current_mode = mode_name if isinstance( mode_name, str ) and mode_name.strip( ) else \
		st.session_state.get( 'mode', 'Chat' )
	
	ensure_common_mode_state( )
	
	if current_mode in [ 'Chat', 'Text' ]:
		ensure_text_mode_state( )
	elif current_mode in [ 'Images', 'Image' ]:
		ensure_image_mode_state( )
	elif current_mode in [ 'Audio' ]:
		ensure_audio_mode_state( )
	elif current_mode in [ 'Embeddings' ]:
		ensure_embeddings_mode_state( )
	elif current_mode in [ 'Document Q&A', 'Documents' ]:
		ensure_docqna_mode_state( )
	elif current_mode in [ 'Files' ]:
		ensure_files_mode_state( )
	elif current_mode in [ 'Vector Stores', 'VectorStores' ]:
		ensure_vectorstores_mode_state( )
	elif current_mode in [ 'File Search Stores', 'FileSearchStores' ]:
		ensure_file_search_mode_state( )
	elif current_mode in [ 'Google Cloud Buckets', 'Cloud Buckets', 'CloudBuckets' ]:
		ensure_cloudbuckets_mode_state( )
	elif current_mode in [ 'Data Management' ]:
		ensure_data_management_mode_state( )
	elif current_mode in [ 'Export' ]:
		ensure_export_mode_state( )

# ======================================================================================
# TEXT MODE UTILITIES
# ======================================================================================

def get_text_avatar( provider_name: str ) -> str:
	"""
	
		Purpose:
		--------
		Return the configured assistant avatar for the active text provider.
		
		Parameters:
		-----------
		provider_name: str
			Selected provider name.
		
		Returns:
		--------
		str
			Avatar string or configured avatar path.
		
	"""
	if provider_name == 'GPT':
		return getattr( cfg, 'GPT_AVATAR', getattr( cfg, 'BUDDY', '🧠' ) )
	
	if provider_name == 'Gemini':
		return getattr( cfg, 'GEMINI_AVATAR', getattr( cfg, 'BUDDY', '🧠' ) )
	
	if provider_name == 'Grok':
		return getattr( cfg, 'GROK_AVATAR', getattr( cfg, 'BUDDY', '🧠' ) )
	
	return getattr( cfg, 'BUDDY', '🧠' )

def get_text_option_list( source: Any, attr_name: str, fallback: List[ str ] ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return a list-valued option attribute from a provider wrapper, preserving a safe
		fallback when the wrapper does not expose the requested option collection.
		
		Parameters:
		-----------
		source: Any
			Provider wrapper instance.
		
		attr_name: str
			Attribute or property name to read.
		
		fallback: List[str]
			Fallback options.
		
		Returns:
		--------
		List[str]
			Resolved option list.
		
	"""
	try:
		options = getattr( source, attr_name, None )
		if isinstance( options, list ) and len( options ) > 0:
			return options
	except Exception:
		pass
	
	return fallback

def clear_text_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Text mode message and answer state without modifying model controls.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'text_messages' ] = [ ]
	st.session_state[ 'text_context' ] = [ ]
	st.session_state[ 'text_gemini_history' ] = [ ]
	st.session_state[ 'last_answer' ] = ''
	st.session_state[ 'last_sources' ] = [ ]

def clear_text_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Text mode system instructions and selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'text_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def load_text_instruction_template( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Text mode system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		prompt_text = fetch_prompt_text( cfg.DB_PATH, name )
		if prompt_text is not None:
			st.session_state[ 'text_system_instructions' ] = prompt_text

def convert_text_system_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Text mode system instructions between XML-like delimiters and Markdown
		headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text_value = st.session_state.get( 'text_system_instructions', '' )
	if not isinstance( text_value, str ) or not text_value.strip( ):
		return
	
	source = text_value.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'text_system_instructions' ] = converted

def reset_text_model_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Text mode model controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'text_model', 'text_reasoning', 'text_modalities',
	             'text_media_resolution', 'text_number' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_text_inference_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Text mode inference controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'text_temperature', 'text_top_percent', 'text_top_k',
	             'text_frequency_penalty', 'text_presence_penalty',
	             'text_presense_penalty' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_text_tool_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Text mode tool and grounding controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'text_max_calls', 'text_tool_choice', 'text_include',
	             'text_includes', 'text_tools', 'text_domains_input',
	             'text_domains', 'text_urls_input', 'text_urls',
	             'text_parallel_tools', 'text_parallel_calls',
	             'text_vector_store_ids', 'text_google_grounding',
	             'text_file_search_store_names', 'selected_filestore_id',
	             'selected_filestore_label' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_text_response_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Text mode response controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'text_stream', 'text_store', 'text_max_tokens',
	             'text_background', 'text_response_format', 'text_input',
	             'text_previous_response_id', 'text_conversation_id',
	             'text_json_schema_name', 'text_json_schema',
	             'text_json_schema_strict', 'text_response_schema',
	             'text_stops_input', 'text_stops', 'text_safety_profile' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def split_text_values( value: Any, delimiter: str=',' ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Split delimited UI text into clean string values.
		
		Parameters:
		-----------
		value: Any
			Raw UI value.
		
		delimiter: str
			Delimiter used to split the text.
		
		Returns:
		--------
		List[str]
			Clean string values.
		
	"""
	if value is None:
		return [ ]
	
	if isinstance( value, list ):
		return [ str( item ).strip( ) for item in value if str( item ).strip( ) ]
	
	if not isinstance( value, str ) or not value.strip( ):
		return [ ]
	
	return [ item.strip( ) for item in value.split( delimiter ) if item.strip( ) ]

def parse_text_vector_store_ids( value: str | List[ str ] | None ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Parse comma-delimited or list-based vector store identifiers for OpenAI file_search.
		
		Parameters:
		-----------
		value: str | List[str] | None
			Comma-delimited vector store IDs or existing list of IDs.
		
		Returns:
		--------
		List[str]
			Clean vector store IDs.
		
	"""
	return split_text_values( value=value, delimiter=',' )

def build_text_response_format( response_format: str | None, schema_name: str | None=None,
		schema_text: str | None=None, strict: bool=True ) -> Dict[ str, Any ] | None:
	"""
	
		Purpose:
		--------
		Build the OpenAI Responses API text.format object from Text mode response-format
		controls.
		
		Parameters:
		-----------
		response_format: str | None
			Selected response format.
		
		schema_name: str | None
			Schema name for json_schema output.
		
		schema_text: str | None
			JSON Schema text.
		
		strict: bool
			Strict schema flag.
		
		Returns:
		--------
		Dict[str, Any] | None
			Responses API text-format object or None.
		
	"""
	if not isinstance( response_format, str ) or not response_format.strip( ):
		return None
	
	format_name = response_format.strip( )
	
	if format_name == 'text':
		return { 'format': { 'type': 'text' } }
	
	if format_name == 'json_object':
		return { 'format': { 'type': 'json_object' } }
	
	if format_name == 'json_schema':
		if not isinstance( schema_text, str ) or not schema_text.strip( ):
			st.warning( 'JSON Schema output requires a schema. Falling back to plain text.' )
			return { 'format': { 'type': 'text' } }
		
		try:
			schema = json.loads( schema_text )
		except Exception as exc:
			st.warning( f'JSON Schema could not be parsed. Falling back to plain text: {exc}' )
			return { 'format': { 'type': 'text' } }
		
		name = schema_name if isinstance( schema_name, str ) and schema_name.strip( ) else \
			'structured_response'
		
		return { 'format': {
						'type': 'json_schema',
						'name': name.strip( ),
						'schema': schema,
						'strict': bool( strict ),
				} }
	
	return None

def build_text_tools( selected_tools: List[ str ] | None,
		vector_store_ids: List[ str ] | None=None ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Build safe OpenAI Responses API tool dictionaries for Text mode.
		
		Parameters:
		-----------
		selected_tools: List[str] | None
			Selected tool names.
		
		vector_store_ids: List[str] | None
			Vector store IDs used by file_search.
		
		Returns:
		--------
		List[Dict[str, Any]]
			OpenAI-compatible tool definitions.
		
	"""
	tools: List[ Dict[ str, Any ] ] = [ ]
	vector_ids = vector_store_ids if isinstance( vector_store_ids, list ) else [ ]
	
	if not isinstance( selected_tools, list ) or len( selected_tools ) == 0:
		return tools
	
	for name in selected_tools:
		tool_name = str( name or '' ).strip( )
		if not tool_name:
			continue
		
		if tool_name in [ 'web_search', 'web_search_preview' ]:
			tools.append( { 'type': 'web_search' } )
			continue
		
		if tool_name == 'file_search':
			if len( vector_ids ) == 0:
				st.warning( 'File Search was selected, but no vector store IDs were provided.' )
				continue
			
			tools.append( {
						'type': 'file_search',
						'vector_store_ids': vector_ids,
				} )
			continue
	
	return tools

def build_text_include( selected_include: List[ str ] | None,
		selected_tools: List[ Dict[ str, Any ] ] | None=None ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Filter Text mode include fields so they correspond to the selected OpenAI tool
		context.
		
		Parameters:
		-----------
		selected_include: List[str] | None
			UI-selected include values.
		
		selected_tools: List[Dict[str, Any]] | None
			Final tool dictionaries.
		
		Returns:
		--------
		List[str]
			Filtered include values.
		
	"""
	if not isinstance( selected_include, list ) or len( selected_include ) == 0:
		return [ ]
	
	tool_types: List[ str ] = [ ]
	if isinstance( selected_tools, list ):
		for tool in selected_tools:
			if isinstance( tool, dict ) and tool.get( 'type' ):
				tool_types.append( str( tool.get( 'type' ) ) )
	
	include_values: List[ str ] = [ ]
	for value in selected_include:
		include_name = str( value or '' ).strip( )
		if not include_name:
			continue
		
		if include_name in [ 'reasoning.encrypted_content', 'message.output_text.logprobs' ]:
			include_values.append( include_name )
			continue
		
		if include_name.startswith( 'web_search_call.' ) and 'web_search' in tool_types:
			include_values.append( include_name )
			continue
		
		if include_name == 'file_search_call.results' and 'file_search' in tool_types:
			include_values.append( include_name )
			continue
	
	return include_values

def build_text_tool_choice( tool_choice: str | None,
		selected_tools: List[ Dict[ str, Any ] ] | None = None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return a tool-choice value only when compatible with the final tool list.
		
		Parameters:
		-----------
		tool_choice: str | None
			UI-selected tool-choice value.
		
		selected_tools: List[Dict[str, Any]] | None
			Final tool dictionaries.
		
		Returns:
		--------
		str | None
			Tool-choice value or None.
		
	"""
	if not isinstance( tool_choice, str ) or not tool_choice.strip( ):
		return None
	
	choice = tool_choice.strip( )
	if choice not in [ 'auto', 'required', 'none' ]:
		return None
	
	if choice == 'none':
		return 'none'
	
	if not isinstance( selected_tools, list ) or len( selected_tools ) == 0:
		return None
	
	return choice

def build_text_context( messages: List[ Dict[ str, Any ] ] | None,
		include_last_message: bool=False ) -> List[ Dict[ str, str ] ]:
	"""
	
		Purpose:
		--------
		Build clean Text mode conversation context from Streamlit message state.
		
		Parameters:
		-----------
		messages: List[Dict[str, Any]] | None
			Text mode messages.
		
		include_last_message: bool
			Whether the final message should be included.
		
		Returns:
		--------
		List[Dict[str, str]]
			Clean conversation context.
		
	"""
	if not isinstance( messages, list ):
		return [ ]
	
	items = messages if include_last_message else messages[ :-1 ]
	context: List[ Dict[ str, str ] ] = [ ]
	for item in items:
		if not isinstance( item, dict ):
			continue
		
		role = str( item.get( 'role', '' ) or '' ).strip( )
		content = item.get( 'content', '' )
		
		if role not in [ 'user', 'assistant', 'system', 'developer' ]:
			continue
		
		if not isinstance( content, str ) or not content.strip( ):
			continue
		
		context.append( {
					'role': role,
					'content': content.strip( ),
			} )
	
	return context

def get_text_conversation_id( input_mode: str | None, conversation_id: str | None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return a conversation identifier only when Text mode explicitly selects API
		conversation state.
		
		Parameters:
		-----------
		input_mode: str | None
			Input mode selection.
		
		conversation_id: str | None
			Conversation identifier.
		
		Returns:
		--------
		str | None
			Conversation identifier or None.
		
	"""
	if input_mode != 'conversation':
		return None
	
	if not isinstance( conversation_id, str ) or not conversation_id.strip( ):
		return None
	
	return conversation_id.strip( )

def get_text_previous_response_id( input_mode: str | None, previous_id: str | None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return previous_response_id only when response chaining is selected.
		
		Parameters:
		-----------
		input_mode: str | None
			Input mode selection.
		
		previous_id: str | None
			Previous response identifier.
		
		Returns:
		--------
		str | None
			Previous response identifier or None.
		
	"""
	if input_mode in [ 'single_turn', 'conversation' ]:
		return None
	
	if not isinstance( previous_id, str ) or not previous_id.strip( ):
		return None
	
	return previous_id.strip( )

def apply_gemini_runtime_config( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Gemini initializes in API-key mode and does not accidentally route through
		Vertex AI runtime variables unless later explicitly configured.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	key = ( st.session_state.get( 'gemini_api_key' )
			or st.session_state.get( 'google_api_key' )
			or getattr( cfg, 'GEMINI_API_KEY', None )
			or getattr( cfg, 'GOOGLE_API_KEY', None )
			or os.environ.get( 'GEMINI_API_KEY' )
			or os.environ.get( 'GOOGLE_API_KEY' ) )
	
	if key:
		os.environ[ 'GEMINI_API_KEY' ] = key
		os.environ[ 'GOOGLE_API_KEY' ] = key
	
	for env_name in [ 'GOOGLE_GENAI_USE_VERTEXAI', 'GOOGLE_CLOUD_PROJECT',
	                  'GOOGLE_CLOUD_PROJECT_ID', 'GOOGLE_CLOUD_LOCATION' ]:
		os.environ.pop( env_name, None )
	
	for attr_name in [ 'GOOGLE_GENAI_USE_VERTEXAI', 'GOOGLE_CLOUD_PROJECT',
	                   'GOOGLE_CLOUD_PROJECT_ID', 'GOOGLE_CLOUD_LOCATION' ]:
		try:
			setattr( cfg, attr_name, None )
		except Exception:
			pass

# ======================================================================================
# IMAGE MODE UTILITIES
# ======================================================================================

def clear_image_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Image Mode message and output state without modifying image controls.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'image_input' ] = [ ]
	st.session_state[ 'image_messages' ] = [ ]
	st.session_state[ 'image_context' ] = [ ]
	st.session_state[ 'image_output_bytes' ] = None
	st.session_state[ 'last_answer' ] = ''

def clear_image_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Image Mode system instructions and selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'image_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def append_image_message( role: str, content: str ) -> None:
	"""
	
		Purpose:
		--------
		Append a message to Image Mode state.
		
		Parameters:
		-----------
		role: str
			Message role.
		
		content: str
			Message content.
		
		Returns:
		--------
		None
		
	"""
	if 'image_input' not in st.session_state or not isinstance(
			st.session_state[ 'image_input' ], list ):
		st.session_state[ 'image_input' ] = [ ]
	
	if 'image_messages' not in st.session_state or not isinstance(
			st.session_state[ 'image_messages' ], list ):
		st.session_state[ 'image_messages' ] = [ ]
	
	message = {
			'role': role,
			'content': content,
	}
	
	st.session_state[ 'image_input' ].append( message )
	st.session_state[ 'image_messages' ].append( message )

def load_image_instruction_template( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Image Mode system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		prompt_text = fetch_prompt_text( cfg.DB_PATH, name )
		if prompt_text is not None:
			st.session_state[ 'image_system_instructions' ] = prompt_text

def convert_image_system_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Image Mode system instructions between XML-like delimiters and Markdown
		headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text_value = st.session_state.get( 'image_system_instructions', '' )
	if not isinstance( text_value, str ) or not text_value.strip( ):
		return
	
	source = text_value.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'image_system_instructions' ] = converted

def reset_image_llm_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Image Mode model-selection controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'image_mode', 'image_model', 'image_analysis_model', 'image_number',
	             'image_modality' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_image_visual_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Image Mode visual controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'image_mime_type', 'image_size', 'image_quality', 'image_backcolor',
	             'image_compression', 'image_aspect_ratio', 'image_detail' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_image_tool_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Image Mode tool and grounding controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'image_include', 'image_tools', 'image_domains_input', 'image_domains',
	             'image_tool_choice', 'image_grounded', 'image_image_search',
	             'image_max_calls', 'image_max_searches', 'image_parallel_calls' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_image_response_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Image Mode response controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'image_temperature', 'image_top_percent', 'image_frequency_penalty',
	             'image_presence_penalty', 'image_max_tokens', 'image_store',
	             'image_stream', 'image_background', 'image_response_format',
	             'image_reasoning', 'image_previous_response_id' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def get_image_models( image: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return provider image model options for generation and editing.
		
		Parameters:
		-----------
		image: Any
			Provider Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Image model options.
		
	"""
	options = getattr( image, 'model_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	provider_name = get_provider_name( )
	
	if provider_name == 'GPT':
		return [ '' ] + list( getattr( cfg, 'GPT_GENERATION',
			[ 'gpt-image-1', 'gpt-image-1-mini' ] ) )
	
	if provider_name == 'Gemini':
		return [ '' ] + list( getattr( cfg, 'GEMINI_GENERATION',
			[ 'gemini-2.5-flash-image' ] ) )
	
	if provider_name == 'Grok':
		return [ '' ] + list( getattr( cfg, 'GROK_GENERATION', [ 'grok-2-image' ] ) )
	
	return [ '' ]

def get_image_analysis_models( image: Any = None ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return provider vision-capable image analysis model options.
		
		Parameters:
		-----------
		image: Any
			Optional Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Image analysis model options.
		
	"""
	if image is not None:
		options = getattr( image, 'analysis_model_options', None )
		if isinstance( options, list ) and len( options ) > 0:
			return [ '' ] + options
	
	provider_name = get_provider_name( )
	
	if provider_name == 'GPT':
		return [ '' ] + list( getattr( cfg, 'GPT_ANALYSIS',
			[ 'gpt-4o-mini', 'gpt-4o', 'gpt-5-mini', 'gpt-5' ] ) )
	
	if provider_name == 'Gemini':
		return [ '' ] + list( getattr( cfg, 'GEMINI_ANALYSIS',
			[ 'gemini-2.5-flash', 'gemini-2.5-flash-image' ] ) )
	
	if provider_name == 'Grok':
		return [ '' ] + list( getattr( cfg, 'GROK_ANALYSIS', [ 'grok-4' ] ) )
	
	return [ '' ]

def get_image_editing_models( image: Any = None ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return provider image editing model options.
		
		Parameters:
		-----------
		image: Any
			Optional Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Image editing model options.
		
	"""
	provider_name = get_provider_name( )
	
	if provider_name == 'GPT':
		return [ '' ] + list( getattr( cfg, 'GPT_EDITING',
			[ 'gpt-image-1', 'gpt-image-1-mini' ] ) )
	
	if provider_name == 'Gemini':
		return [ '' ] + list( getattr( cfg, 'GEMINI_EDITING',
			[ 'gemini-2.5-flash-image' ] ) )
	
	if image is not None:
		options = getattr( image, 'model_options', None )
		if isinstance( options, list ) and len( options ) > 0:
			return [ '' ] + options
	
	return [ '' ]

def get_image_size_options( image: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return image size options from the wrapper when available.
		
		Parameters:
		-----------
		image: Any
			Provider Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Image size options.
		
	"""
	options = getattr( image, 'size_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'auto', '1024x1024', '1024x1536', '1536x1024' ]

def get_image_quality_options( image: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return image quality options from the wrapper when available.
		
		Parameters:
		-----------
		image: Any
			Provider Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Image quality options.
		
	"""
	options = getattr( image, 'quality_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'auto', 'low', 'medium', 'high' ]

def get_image_mime_options( image: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return image output format options from the wrapper when available.
		
		Parameters:
		-----------
		image: Any
			Provider Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Image output format options.
		
	"""
	options = getattr( image, 'mime_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'png', 'jpeg', 'webp' ]

def get_image_background_options( image: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return image background options from the wrapper when available.
		
		Parameters:
		-----------
		image: Any
			Provider Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Image background options.
		
	"""
	options = getattr( image, 'backcolor_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'auto', 'transparent', 'opaque' ]

def get_image_detail_options( image: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return image-analysis detail options from the wrapper when available.
		
		Parameters:
		-----------
		image: Any
			Provider Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Image-analysis detail options.
		
	"""
	options = getattr( image, 'detail_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'auto', 'low', 'high', 'original' ]

def get_image_aspect_options( image: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return Gemini image aspect-ratio options when available.
		
		Parameters:
		-----------
		image: Any
			Provider Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Aspect-ratio options.
		
	"""
	options = getattr( image, 'aspect_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9' ]

def get_image_modality_options( image: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return Gemini image response modality options when available.
		
		Parameters:
		-----------
		image: Any
			Provider Images wrapper instance.
		
		Returns:
		--------
		List[str]
			Response modality options.
		
	"""
	options = getattr( image, 'modality_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	return [ '', 'text', 'image', 'auto' ]

def render_image_output( image_result: str | bytes | List[ str | bytes ] | Any | None,
		caption: str='Image output' ) -> bool:
	"""
	
		Purpose:
		--------
		Render image output returned from an Images wrapper, including PIL images,
		single bytes, URLs, strings, and list outputs.
		
		Parameters:
		-----------
		image_result: str | bytes | List[str | bytes] | Any | None
			Image result returned by the provider wrapper.
		
		caption: str
			Display caption.
		
		Returns:
		--------
		bool
			True when output was rendered; otherwise False.
		
	"""
	if image_result is None:
		return False
	
	outputs = image_result if isinstance( image_result, list ) else [ image_result ]
	rendered = False
	
	for index, item in enumerate( outputs, start=1 ):
		if item is None:
			continue
		
		output_caption = f'{caption} {index}' if len( outputs ) > 1 else caption
		
		if isinstance( item, bytes ) and len( item ) > 0:
			st.image( item, caption=output_caption, use_column_width=True )
			rendered = True
			continue
		
		if isinstance( item, str ) and item.strip( ):
			value = item.strip( )
			if value.lower( ).startswith( ('http://', 'https://') ):
				st.image( value, caption=output_caption, use_column_width=True )
			else:
				st.markdown( value )
			
			rendered = True
			continue
		
		try:
			st.image( item, caption=output_caption, use_column_width=True )
			rendered = True
		except Exception:
			st.write( item )
			rendered = True
	
	return rendered

def get_image_prompt( prompt: str | None ) -> str:
	"""
	
		Purpose:
		--------
		Normalize a chat-input prompt for Image Mode execution.
		
		Parameters:
		-----------
		prompt: str | None
			Raw prompt value.
		
		Returns:
		--------
		str
			Clean prompt text.
		
	"""
	if not isinstance( prompt, str ) or not prompt.strip( ):
		return ''
	
	return prompt.strip( )

# ======================================================================================
# AUDIO MODE UTILITIES
# ======================================================================================

def ensure_audio_runtime_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Audio mode runtime keys exist before Audio mode widgets or execution paths
		read them. File uploader widget keys are intentionally excluded because Streamlit
		does not permit file uploader values to be assigned through session_state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_audio_mode_state( )
	ensure_key( 'audio_tts_input', '' )
	ensure_key( 'audio_speed', 1.0 )
	ensure_key( 'audio_last_result', { } )
	ensure_key( 'audio_last_usage', { } )

def get_audio_task_options( ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return supported Audio mode tasks for the currently selected provider.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		List[str]
			Supported audio task names.
		
	"""
	options: List[ str ] = [ ]
	
	if provider_supports( 'Transcription' ):
		options.append( 'Transcribe' )
	
	if provider_supports( 'Translation' ):
		options.append( 'Translate' )
	
	if provider_supports( 'TTS' ):
		options.append( 'Text-to-Speech' )
	
	return options

def get_audio_option_list( source: Any, attr_name: str, fallback: List[ str ] ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return a list-valued option attribute from an audio wrapper, preserving a fallback
		when the provider does not expose the requested option collection.
		
		Parameters:
		-----------
		source: Any
			Audio provider wrapper instance.
		
		attr_name: str
			Attribute name to read.
		
		fallback: List[str]
			Fallback option list.
		
		Returns:
		--------
		List[str]
			Resolved option list.
		
	"""
	if source is not None:
		try:
			options = getattr( source, attr_name, None )
			if isinstance( options, list ) and len( options ) > 0:
				return options
		except Exception:
			pass
	
	return fallback

def get_audio_model_options( task: str | None, transcriber: Any, translator: Any, tts: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return task-specific audio model options for the selected provider.
		
		Parameters:
		-----------
		task: str | None
			Selected audio task.
		
		transcriber: Any
			Provider transcription wrapper.
		
		translator: Any
			Provider translation wrapper.
		
		tts: Any
			Provider text-to-speech wrapper.
		
		Returns:
		--------
		List[str]
			Model option list.
		
	"""
	provider_name = get_provider_name( )
	
	if task == 'Transcribe':
		return [ '' ] + get_audio_option_list( transcriber, 'model_options',
			[ 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'whisper-1' ]
			if provider_name == 'GPT' else [ 'gemini-3-flash-preview', 'gemini-2.0-flash' ] )
	
	if task == 'Translate':
		return [ '' ] + get_audio_option_list( translator, 'model_options',
			[ 'whisper-1' ] if provider_name == 'GPT'
			else [ 'gemini-3-flash-preview', 'gemini-2.0-flash' ] )
	
	if task == 'Text-to-Speech':
		return [ '' ] + get_audio_option_list( tts, 'model_options',
			[ 'gpt-4o-mini-tts', 'tts-1', 'tts-1-hd' ]
			if provider_name == 'GPT' else [ 'gemini-2.5-flash-preview-tts' ] )
	
	return [ '' ]

def get_audio_language_options( task: str | None, transcriber: Any, translator: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return task-specific language options for transcription or translation.
		
		Parameters:
		-----------
		task: str | None
			Selected audio task.
		
		transcriber: Any
			Provider transcription wrapper.
		
		translator: Any
			Provider translation wrapper.
		
		Returns:
		--------
		List[str]
			Language option list.
		
	"""
	if task == 'Transcribe':
		return [ '' ] + get_audio_option_list( transcriber, 'language_options',
			[ 'en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh' ] )
	
	if task == 'Translate':
		return [ '' ] + get_audio_option_list( translator, 'language_options',
			[ 'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese' ] )
	
	return [ '' ]

def get_audio_voice_options( tts: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return text-to-speech voice options for the selected provider.
		
		Parameters:
		-----------
		tts: Any
			Provider text-to-speech wrapper.
		
		Returns:
		--------
		List[str]
			Voice option list.
		
	"""
	return [ '' ] + get_audio_option_list( tts, 'voice_options',
		[ 'alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer' ] )

def get_audio_response_format_options( task: str | None, model: str | None,
		transcriber: Any, translator: Any, tts: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return task-specific response/output format options for Audio mode.
		
		Parameters:
		-----------
		task: str | None
			Selected audio task.
		
		model: str | None
			Selected audio model.
		
		transcriber: Any
			Provider transcription wrapper.
		
		translator: Any
			Provider translation wrapper.
		
		tts: Any
			Provider text-to-speech wrapper.
		
		Returns:
		--------
		List[str]
			Format option list.
		
	"""
	if task == 'Transcribe':
		return [ '' ] + get_audio_option_list( transcriber, 'format_options',
			[ 'json', 'text', 'srt', 'verbose_json', 'vtt' ] )
	
	if task == 'Translate':
		return [ '' ] + get_audio_option_list( translator, 'format_options',
			[ 'json', 'text', 'srt', 'verbose_json', 'vtt' ] )
	
	if task == 'Text-to-Speech':
		return [ '' ] + get_audio_option_list( tts, 'format_options',
			[ 'mp3', 'wav', 'aac', 'flac', 'opus', 'pcm' ] )
	
	return [ '' ]

def get_audio_prompt_value( task: str | None, prompt: str | None ) -> str | None:
	"""
	
		Purpose:
		--------
		Return task-specific audio instructions or prompt value.
		
		Parameters:
		-----------
		task: str | None
			Selected audio task.
		
		prompt: str | None
			Audio system instruction text.
		
		Returns:
		--------
		str | None
			Clean prompt text or None.
		
	"""
	if not isinstance( prompt, str ) or not prompt.strip( ):
		return None
	
	return prompt.strip( )

def get_audio_response_format_value( task: str | None, selected_format: str | None,
		selected_mime_type: str | None = None ) -> str | None:
	"""
	
		Purpose:
		--------
		Normalize selected audio format values before calling provider wrappers.
		
		Parameters:
		-----------
		task: str | None
			Selected audio task.
		
		selected_format: str | None
			UI-selected response format.
		
		selected_mime_type: str | None
			UI-selected MIME type.
		
		Returns:
		--------
		str | None
			Normalized format value or None.
		
	"""
	if isinstance( selected_format, str ) and selected_format.strip( ):
		value = selected_format.strip( )
		if value.startswith( 'audio/' ):
			return value.split( '/', 1 )[ 1 ]
		
		return value
	
	if isinstance( selected_mime_type, str ) and selected_mime_type.strip( ):
		value = selected_mime_type.strip( )
		if value.startswith( 'audio/' ):
			return value.split( '/', 1 )[ 1 ]
		
		return value
	
	return None

def extract_audio_usage( response: Any ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Extract audio response usage metadata using the shared usage extractor.
		
		Parameters:
		-----------
		response: Any
			Provider response object.
		
		Returns:
		--------
		Dict[str, Any]
			Usage dictionary.
		
	"""
	try:
		return _extract_usage_from_response( response )
	except Exception:
		return { }

def clear_audio_outputs( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Audio mode output state while preserving controls and messages.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'audio_output' ] = ''
	st.session_state[ 'audio_output_bytes' ] = None
	st.session_state[ 'audio_last_result' ] = { }
	st.session_state[ 'audio_last_usage' ] = { }

def clear_audio_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Audio mode messages and output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'audio_messages' ] = [ ]
	clear_audio_outputs( )

def clear_audio_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Audio mode system instructions and selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'audio_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def load_audio_instruction_template( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Audio mode system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		prompt_text = fetch_prompt_text( cfg.DB_PATH, name )
		if prompt_text is not None:
			st.session_state[ 'audio_system_instructions' ] = prompt_text

def convert_audio_system_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Audio mode system instructions between XML-like delimiters and Markdown
		headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text_value = st.session_state.get( 'audio_system_instructions', '' )
	if not isinstance( text_value, str ) or not text_value.strip( ):
		return
	
	source = text_value.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'audio_system_instructions' ] = converted

def reset_audio_llm_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Audio mode LLM controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'audio_task', 'audio_model', 'audio_language', 'audio_voice',
	             'audio_response_format', 'audio_mime_type' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_audio_response_settings( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Audio mode response controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'audio_temperature', 'audio_top_percent', 'audio_frequency_penalty',
	             'audio_presence_penalty', 'audio_max_tokens', 'audio_speed',
	             'audio_store', 'audio_stream', 'audio_background',
	             'audio_start_time', 'audio_end_time', 'audio_loop',
	             'audio_autoplay' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def save_audio_upload( upload: Any ) -> str | None:
	"""
	
		Purpose:
		--------
		Save an uploaded or recorded audio object to a temporary file and return its path.
		
		Parameters:
		-----------
		upload: Any
			Streamlit uploaded or recorded audio object.
		
		Returns:
		--------
		str | None
			Temporary file path or None.
		
	"""
	if upload is None:
		return None
	
	try:
		name = getattr( upload, 'name', 'audio.wav' )
		_, ext = os.path.splitext( name )
		ext = ext if ext else '.wav'
		
		with tempfile.NamedTemporaryFile( delete=False, suffix=ext ) as tmp:
			if hasattr( upload, 'getbuffer' ):
				tmp.write( upload.getbuffer( ) )
			elif hasattr( upload, 'read' ):
				tmp.write( upload.read( ) )
			else:
				return None
			
			return tmp.name
	except Exception:
		return None

def run_audio_file_task( task: str | None, file_path: str | None,
		transcriber: Any, translator: Any ) -> str | None:
	"""
	
		Purpose:
		--------
		Run transcription or translation against an audio file and store normalized output
		state.
		
		Parameters:
		-----------
		task: str | None
			Selected Audio mode task.
		
		file_path: str | None
			Temporary audio file path.
		
		transcriber: Any
			Provider transcription wrapper.
		
		translator: Any
			Provider translation wrapper.
		
		Returns:
		--------
		str | None
			Text output from transcription or translation.
		
	"""
	if not isinstance( task, str ) or not task.strip( ):
		st.warning( 'Select an audio task before processing audio.' )
		return None
	
	if not isinstance( file_path, str ) or not file_path.strip( ):
		st.warning( 'Upload or record audio before processing.' )
		return None
	
	prompt_value = get_audio_prompt_value(
		task=task,
		prompt=st.session_state.get( 'audio_system_instructions', '' ) )
	
	response_format = get_audio_response_format_value(
		task=task,
		selected_format=st.session_state.get( 'audio_response_format' ),
		selected_mime_type=st.session_state.get( 'audio_mime_type' ) )
	
	model = st.session_state.get( 'audio_model' )
	language = st.session_state.get( 'audio_language' )
	temperature = st.session_state.get( 'audio_temperature' )
	include = st.session_state.get( 'audio_include', [ ] )
	
	if task == 'Transcribe':
		try:
			result_text = transcriber.transcribe(
				path=file_path,
				model=model or 'gpt-4o-transcribe',
				language=language or None,
				prompt=prompt_value,
				format=response_format,
				temperature=temperature,
				include=include )
		except TypeError:
			result_text = transcriber.transcribe(
				path=file_path,
				model=model or 'gemini-3-flash-preview',
				language=language or None )
		
		st.session_state[ 'audio_output' ] = result_text or ''
		st.session_state[ 'audio_last_result' ] = getattr(
			transcriber, 'normalized_result', { } ) or { }
		st.session_state[ 'audio_last_usage' ] = extract_audio_usage(
			getattr( transcriber, 'response', None ) )
		
		return result_text
	
	if task == 'Translate':
		try:
			result_text = translator.translate(
				filepath=file_path,
				model=model or 'whisper-1',
				prompt=prompt_value,
				format=response_format,
				temperature=temperature,
				language=language or None )
		except TypeError:
			try:
				result_text = translator.translate(
					path=file_path,
					model=model or 'gemini-3-flash-preview',
					language=language or None )
			except TypeError:
				result_text = translator.translate(
					file_path,
					model=model or 'gemini-3-flash-preview',
					language=language or None )
		
		st.session_state[ 'audio_output' ] = result_text or ''
		st.session_state[ 'audio_last_result' ] = getattr(
			translator, 'normalized_result', { } ) or { }
		st.session_state[ 'audio_last_usage' ] = extract_audio_usage(
			getattr( translator, 'response', None ) )
		
		return result_text
	
	st.info( 'Use the Text-to-Speech input area to generate speech from text.' )
	return None

def run_audio_tts_task( text: str | None, tts: Any ) -> bytes | None:
	"""
	
		Purpose:
		--------
		Run text-to-speech and store generated audio bytes.
		
		Parameters:
		-----------
		text: str | None
			Text to synthesize.
		
		tts: Any
			Provider text-to-speech wrapper.
		
		Returns:
		--------
		bytes | None
			Generated audio bytes.
		
	"""
	if not isinstance( text, str ) or not text.strip( ):
		st.warning( 'Enter text before generating speech.' )
		return None
	
	model = st.session_state.get( 'audio_model' )
	voice = st.session_state.get( 'audio_voice' )
	speed = float( st.session_state.get( 'audio_speed', 1.0 ) or 1.0 )
	response_format = get_audio_response_format_value(
		task='Text-to-Speech',
		selected_format=st.session_state.get( 'audio_response_format' ),
		selected_mime_type=st.session_state.get( 'audio_mime_type' ) )
	
	instructions = get_audio_prompt_value(
		task='Text-to-Speech',
		prompt=st.session_state.get( 'audio_system_instructions', '' ) )
	
	audio_bytes = tts.create_speech(
		text=text.strip( ),
		model=model or 'gpt-4o-mini-tts',
		format=response_format or 'mp3',
		speed=speed,
		voice=voice or 'alloy',
		instruct=instructions )
	
	st.session_state[ 'audio_output_bytes' ] = audio_bytes
	st.session_state[ 'audio_output' ] = text.strip( )
	st.session_state[ 'audio_last_result' ] = {
			'text': text.strip( ),
			'format': response_format or 'mp3',
			'voice': voice or 'alloy',
			'speed': speed,
	}
	st.session_state[ 'audio_last_usage' ] = extract_audio_usage(
		getattr( tts, 'response', None ) )
	
	return audio_bytes

def render_audio_text_result( title: str, result_text: str | None ) -> None:
	"""
	
		Purpose:
		--------
		Render transcription or translation text output.
		
		Parameters:
		-----------
		title: str
			Result title.
		
		result_text: str | None
			Result text.
		
		Returns:
		--------
		None
		
	"""
	if isinstance( result_text, str ) and result_text.strip( ):
		st.text_area( title, value=result_text.strip( ), height=300, width='stretch' )
	else:
		st.warning( 'No text output was returned.' )

def render_audio_bytes( audio_bytes: bytes | None, response_format: str | None = None ) -> None:
	"""
	
		Purpose:
		--------
		Render generated audio bytes.
		
		Parameters:
		-----------
		audio_bytes: bytes | None
			Generated audio bytes.
		
		response_format: str | None
			Audio format value.
		
		Returns:
		--------
		None
		
	"""
	if not audio_bytes:
		st.warning( 'No audio output was returned.' )
		return
	
	format_value = response_format or 'mp3'
	if format_value.startswith( 'audio/' ):
		mime = format_value
	else:
		mime = f'audio/{format_value}'
	
	st.audio( audio_bytes, format=mime )

# ======================================================================================
# EMBEDDINGS MODE UTILITIES
# ======================================================================================

def get_embedding_model_options( embedding: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return embedding model options from the selected provider wrapper.
		
		Parameters:
		-----------
		embedding: Any
			Provider Embeddings wrapper instance.
		
		Returns:
		--------
		List[str]
			Embedding model options.
		
	"""
	options = getattr( embedding, 'model_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	provider_name = get_provider_name( )
	
	if provider_name == 'GPT':
		return [
				'',
				'text-embedding-3-small',
				'text-embedding-3-large',
				'text-embedding-ada-002',
		]
	
	if provider_name == 'Gemini':
		return [
				'',
				'gemini-embedding-001',
				'text-embedding-004',
		]
	
	return [ '' ]

def get_embedding_encoding_options( embedding: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return embedding encoding format options from the selected provider wrapper.
		
		Parameters:
		-----------
		embedding: Any
			Provider Embeddings wrapper instance.
		
		Returns:
		--------
		List[str]
			Encoding format options.
		
	"""
	options = getattr( embedding, 'encoding_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return options
	
	provider_name = get_provider_name( )
	
	if provider_name == 'GPT':
		return [ 'float', 'base64' ]
	
	if provider_name == 'Gemini':
		return [ 'float' ]
	
	return [ 'float' ]

def get_embedding_max_dimensions( model: str | None, embedding: Any ) -> int:
	"""
	
		Purpose:
		--------
		Return the maximum supported dimensions for the selected embedding model.
		
		Parameters:
		-----------
		model: str | None
			Selected embedding model.
		
		embedding: Any
			Provider Embeddings wrapper instance.
		
		Returns:
		--------
		int
			Maximum supported dimensions.
		
	"""
	if not isinstance( model, str ) or not model.strip( ):
		return 1536
	
	try:
		return int( embedding.get_max_dimensions( model.strip( ) ) )
	except Exception:
		pass
	
	if model == 'text-embedding-3-large':
		return 3072
	
	if model == 'gemini-embedding-001':
		return 3072
	
	return 1536

def embedding_model_supports_dimensions( model: str | None, embedding: Any ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether the selected embedding model supports a dimensions parameter.
		
		Parameters:
		-----------
		model: str | None
			Selected embedding model.
		
		embedding: Any
			Provider Embeddings wrapper instance.
		
		Returns:
		--------
		bool
			True when dimensions may be sent; otherwise False.
		
	"""
	if not isinstance( model, str ) or not model.strip( ):
		return False
	
	support = getattr( embedding, 'model_dimension_support', None )
	if isinstance( support, dict ):
		return bool( support.get( model.strip( ), False ) )
	
	return model.strip( ) in [ 'text-embedding-3-small', 'text-embedding-3-large',
			'gemini-embedding-001', ]

def normalize_embedding_dimensions( model: str | None, dimensions: int | None, embedding: Any ) -> int | None:
	"""
	
		Purpose:
		--------
		Normalize dimensions before calling a provider Embeddings API.
		
		Parameters:
		-----------
		model: str | None
			Selected embedding model.
		
		dimensions: int | None
			Requested embedding dimensions.
		
		embedding: Any
			Provider Embeddings wrapper instance.
		
		Returns:
		--------
		int | None
			Valid dimensions value or None when omitted.
		
	"""
	if not isinstance( model, str ) or not model.strip( ):
		return None
	
	try:
		value = int( dimensions or 0 )
	except Exception:
		return None
	
	if value <= 0:
		return None
	
	if not embedding_model_supports_dimensions( model, embedding ):
		return None
	
	max_dimensions = get_embedding_max_dimensions( model, embedding )
	if value > max_dimensions:
		return max_dimensions
	
	return value

def normalize_embedding_chunk_settings( chunk_size: int | None, overlap_amount: int | None ) -> Tuple[ int, int ]:
	"""
	
		Purpose:
		--------
		Normalize chunk size and overlap settings for tokenizer-aware chunking.
		
		Parameters:
		-----------
		chunk_size: int | None
			Requested chunk size in tokens.
		
		overlap_amount: int | None
			Token overlap between adjacent chunks.
		
		Returns:
		--------
		Tuple[int, int]
			Normalized chunk size and overlap amount.
		
	"""
	try:
		chunk_value = int( chunk_size or 800 )
	except Exception:
		chunk_value = 800
	
	try:
		overlap_value = int( overlap_amount or 0 )
	except Exception:
		overlap_value = 0
	
	if chunk_value <= 0:
		chunk_value = 800
	
	if chunk_value > 8192:
		chunk_value = 8192
	
	if overlap_value < 0:
		overlap_value = 0
	
	if overlap_value >= chunk_value:
		overlap_value = max( 0, chunk_value // 5 )
	
	return chunk_value, overlap_value

def chunk_text_for_embeddings( text: str, chunk_size: int=800,
		overlap_amount: int=0, encoding_name: str='cl100k_base' ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Split text into tokenizer-aware chunks for embedding APIs.
		
		Parameters:
		-----------
		text: str
			Input text to chunk.
		
		chunk_size: int
			Maximum chunk size in tokens.
		
		overlap_amount: int
			Token overlap between adjacent chunks.
		
		encoding_name: str
			Tiktoken encoding name.
		
		Returns:
		--------
		List[str]
			Text chunks.
		
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return [ ]
	
	chunk_value, overlap_value = normalize_embedding_chunk_settings(
		chunk_size=chunk_size,
		overlap_amount=overlap_amount )
	
	encoding = tiktoken.get_encoding( encoding_name )
	tokens = encoding.encode( text )
	
	if len( tokens ) == 0:
		return [ ]
	
	chunks: List[ str ] = [ ]
	start = 0
	step = max( 1, chunk_value - overlap_value )
	
	while start < len( tokens ):
		end = min( start + chunk_value, len( tokens ) )
		chunk_tokens = tokens[ start:end ]
		chunk_text_value = encoding.decode( chunk_tokens ).strip( )
		
		if chunk_text_value:
			chunks.append( chunk_text_value )
		
		if end >= len( tokens ):
			break
		
		start += step
	
	return chunks

def normalize_embedding_vectors( vectors: Any ) -> List[ Any ]:
	"""
	
		Purpose:
		--------
		Normalize provider embedding output into a list of vectors or encoded strings.
		
		Parameters:
		-----------
		vectors: Any
			Embedding output returned by the provider wrapper.
		
		Returns:
		--------
		List[Any]
			Normalized embedding outputs.
		
	"""
	if vectors is None:
		return [ ]
	
	if isinstance( vectors, str ):
		return [ vectors ]
	
	if isinstance( vectors, list ):
		if len( vectors ) == 0:
			return [ ]
		
		if all( isinstance( value, (int, float) ) for value in vectors ):
			return [ vectors ]
		
		return vectors
	
	return [ vectors ]

def build_embeddings_dataframe( chunks: List[ str ], vectors: Any, encoding_format: str='float' ) -> pd.DataFrame:
	"""
	
		Purpose:
		--------
		Build a display dataframe from embedding chunks and embedding vectors.
		
		Parameters:
		-----------
		chunks: List[str]
			Text chunks submitted to the embedding API.
		
		vectors: Any
			Embedding vectors or encoded strings returned by the wrapper.
		
		encoding_format: str
			Embedding encoding format.
		
		Returns:
		--------
		pd.DataFrame
			Display-ready embeddings dataframe.
		
	"""
	outputs = normalize_embedding_vectors( vectors )
	if len( outputs ) == 0:
		return pd.DataFrame( )
	
	rows: List[ Dict[ str, Any ] ] = [ ]
	format_value = encoding_format if isinstance( encoding_format, str ) else 'float'
	if format_value == 'base64':
		for index, item in enumerate( outputs ):
			chunk = chunks[ index ] if index < len( chunks ) else ''
			rows.append( {
						'ChunkIndex': index + 1,
						'Chunk': chunk,
						'EmbeddingBase64': item if isinstance( item, str ) else str( item ),
				} )
		
		return pd.DataFrame( rows )
	
	for index, vector in enumerate( outputs ):
		chunk = chunks[ index ] if index < len( chunks ) else ''
		
		if not isinstance( vector, list ):
			rows.append( {
						'ChunkIndex': index + 1,
						'Chunk': chunk,
						'Embedding': str( vector ),
				} )
			continue
		
		row: Dict[ str, Any ] = {
				'ChunkIndex': index + 1,
				'Chunk': chunk,
		}
		
		for dim_index, value in enumerate( vector ):
			row[ f'dim_{dim_index}' ] = value
		
		rows.append( row )
	
	return pd.DataFrame( rows )

def get_embedding_vector_dimension( vectors: Any ) -> int:
	"""
	
		Purpose:
		--------
		Return the numeric embedding vector dimension when available.
		
		Parameters:
		-----------
		vectors: Any
			Embedding output returned by the provider wrapper.
		
		Returns:
		--------
		int
			Vector dimension count, or zero for non-numeric output.
		
	"""
	outputs = normalize_embedding_vectors( vectors )
	if len( outputs ) == 0:
		return 0
	
	first = outputs[ 0 ]
	if isinstance( first, list ):
		return len( first )
	
	return 0

def extract_embedding_usage( response: Any ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Extract usage metadata from an embedding API response object.
		
		Parameters:
		-----------
		response: Any
			Embedding API response object.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized usage metadata.
		
	"""
	if response is None:
		return { }
	
	try:
		raw = getattr( response, 'usage', None )
	except Exception:
		raw = None
	
	if raw is None:
		try:
			raw = getattr( response, 'usage_metadata', None )
		except Exception:
			raw = None
	
	if raw is None:
		return { }
	
	if isinstance( raw, dict ):
		return raw
	
	if hasattr( raw, 'model_dump' ):
		try:
			return raw.model_dump( )
		except Exception:
			return { 'raw': str( raw ) }
	
	return { 'raw': str( raw ) }

def build_embedding_metrics( source_text: str, normalized_text: str, chunks: List[ str ],
		vectors: Any, usage: Dict[ str, Any ] | None=None ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Build Embeddings mode metrics for display and session-state storage.
		
		Parameters:
		-----------
		source_text: str
			Original input text.
		
		normalized_text: str
			Normalized text submitted to chunking.
		
		chunks: List[str]
			Embedding chunks.
		
		vectors: Any
			Embedding output returned by the wrapper.
		
		usage: Dict[str, Any] | None
			Optional usage metadata from the API.
		
		Returns:
		--------
		Dict[str, Any]
			Embedding metrics.
		
	"""
	source_value = source_text if isinstance( source_text, str ) else ''
	normalized_value = normalized_text if isinstance( normalized_text, str ) else ''
	outputs = normalize_embedding_vectors( vectors )
	words = normalized_value.split( )
	unique_words = set( words )
	token_total = count_tokens( normalized_value ) if normalized_value else 0
	vector_dimension = get_embedding_vector_dimension( outputs )
	
	return {
			'characters': len( source_value ),
			'normalized_characters': len( normalized_value ),
			'words': len( words ),
			'unique_words': len( unique_words ),
			'type_token_ratio': round( len( unique_words ) / len( words ), 4 )
			if len( words ) else 0.0,
			'tokens': token_total,
			'chunks': len( chunks ),
			'embeddings': len( outputs ),
			'vector_dimension': vector_dimension,
			'encoding_format': st.session_state.get( 'embeddings_encoding_format', 'float' ),
			'usage': usage if isinstance( usage, dict ) else { },
	}

def render_embedding_metrics( metrics: Dict[ str, Any ] | None ) -> None:
	"""
	
		Purpose:
		--------
		Render Embeddings mode metrics using Streamlit metric controls.
		
		Parameters:
		-----------
		metrics: Dict[str, Any] | None
			Embedding metrics dictionary.
		
		Returns:
		--------
		None
		
	"""
	if not isinstance( metrics, dict ) or len( metrics ) == 0:
		return
	
	metric_c1, metric_c2, metric_c3, metric_c4, metric_c5 = st.columns(
		[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
	
	with metric_c1:
		st.metric( 'Tokens', metrics.get( 'tokens', 0 ) )
	
	with metric_c2:
		st.metric( 'Chunks', metrics.get( 'chunks', 0 ) )
	
	with metric_c3:
		st.metric( 'Embeddings', metrics.get( 'embeddings', 0 ) )
	
	with metric_c4:
		st.metric( 'Dimensions', metrics.get( 'vector_dimension', 0 ) )
	
	with metric_c5:
		st.metric( 'Words', metrics.get( 'words', 0 ) )

def render_embeddings_dataframe( df_embeddings: pd.DataFrame ) -> None:
	"""
	
		Purpose:
		--------
		Render Embeddings mode output dataframe safely.
		
		Parameters:
		-----------
		df_embeddings: pd.DataFrame
			Embeddings output dataframe.
		
		Returns:
		--------
		None
		
	"""
	if df_embeddings is None or df_embeddings.empty:
		st.info( 'No embeddings available.' )
		return
	
	st.data_editor( df_embeddings, use_container_width=True, hide_index=True )

def reset_embeddings_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Embeddings mode configuration controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'embedding_model', 'embeddings_dimensions',
	             'embeddings_chunk_size', 'embeddings_overlap_amount',
	             'embeddings_encoding_format', 'embeddings_user' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def clear_embeddings_output( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Embeddings mode outputs while preserving configuration controls.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'embeddings' ] = [ ]
	st.session_state[ 'embeddings_chunks' ] = [ ]
	st.session_state[ 'embeddings_df' ] = pd.DataFrame( )
	st.session_state[ 'embedding_metrics' ] = { }
	st.session_state[ 'embedding_usage' ] = { }

def reset_embeddings_all( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Embeddings mode configuration, input, and output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	reset_embeddings_controls( )
	clear_embeddings_output( )
	
	if 'embeddings_input_text' in st.session_state:
		del st.session_state[ 'embeddings_input_text' ]

def create_provider_embeddings( embedding: Any, chunks: List[ str ], model: str,
		encoding_format: str, dimensions: int | None, user_value: str | None ) -> Any:
	"""
	
		Purpose:
		--------
		Create embeddings using the correct provider-specific wrapper call.
		
		Parameters:
		-----------
		embedding: Any
			Provider Embeddings wrapper instance.
		
		chunks: List[str]
			Text chunks to embed.
		
		model: str
			Embedding model name.
		
		encoding_format: str
			Requested embedding encoding format.
		
		dimensions: int | None
			Optional embedding dimension value.
		
		user_value: str | None
			Optional user identifier for providers that support it.
		
		Returns:
		--------
		Any
			Provider embedding vectors.
		
	"""
	provider_name = get_provider_name( )
	
	if provider_name == 'Gemini':
		return embedding.create(
			text=chunks,
			model=model,
			dimensions=dimensions,
			encoding_format=encoding_format,
			task_type='RETRIEVAL_DOCUMENT' )
	
	if provider_name == 'GPT':
		return embedding.create(
			text=chunks,
			model=model,
			format=encoding_format,
			dimensions=dimensions,
			user=user_value )
	
	return embedding.create(
		text=chunks,
		model=model,
		dimensions=dimensions )

# ======================================================================================
# DOCQNA UTILITIES
# ======================================================================================

def get_docqna_avatar( provider_name: str ) -> str:
	"""
	
		Purpose:
		--------
		Return the configured assistant avatar for the active Document Q&A provider.
		
		Parameters:
		-----------
		provider_name: str
			Selected provider name.
		
		Returns:
		--------
		str
			Avatar string or configured avatar path.
		
	"""
	if provider_name == 'GPT':
		return getattr( cfg, 'GPT_AVATAR', getattr( cfg, 'BUDDY', '🧠' ) )
	
	if provider_name == 'Gemini':
		return getattr( cfg, 'GEMINI_AVATAR', getattr( cfg, 'BUDDY', '🧠' ) )
	
	if provider_name == 'Grok':
		return getattr( cfg, 'GROK_AVATAR', getattr( cfg, 'BUDDY', '🧠' ) )
	
	return getattr( cfg, 'BUDDY', '🧠' )

def clear_docqna_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Document Q&A chat messages without clearing loaded documents or retrieval
		state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'docqna_messages' ] = [ ]

def clear_docqna_outputs( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Document Q&A answer, context, source, and retrieval output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'docqna_last_answer' ] = ''
	st.session_state[ 'docqna_last_hits' ] = [ ]
	st.session_state[ 'docqna_last_sources' ] = [ ]
	st.session_state[ 'docqna_context' ] = ''
	st.session_state[ 'last_answer' ] = ''
	st.session_state[ 'last_sources' ] = [ ]

def unload_docqna_documents( ) -> None:
	"""
	
		Purpose:
		--------
		Unload active Document Q&A files, extracted text, chunks, and local retrieval
		state. File uploader widget keys are intentionally not assigned or cleared.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'docqna_uploaded' ] = None
	st.session_state[ 'docqna_files' ] = [ ]
	st.session_state[ 'docqna_active_docs' ] = [ ]
	st.session_state[ 'active_docs' ] = [ ]
	st.session_state[ 'docqna_bytes' ] = None
	st.session_state[ 'doc_bytes' ] = { }
	st.session_state[ 'docqna_texts' ] = { }
	st.session_state[ 'docqna_chunks' ] = [ ]
	st.session_state[ 'docqna_vec_ready' ] = False
	st.session_state[ 'docqna_fingerprint' ] = ''
	st.session_state[ 'docqna_chunk_count' ] = 0
	st.session_state[ 'docqna_index_status' ] = 'Not indexed'
	clear_docqna_outputs( )

def reset_docqna_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Document Q&A controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'docqna_model', 'docqna_source', 'docqna_file_id',
	             'docqna_vector_store_id', 'docqna_multi_mode', 'docqna_top_k',
	             'docqna_chunk_size', 'docqna_chunk_overlap',
	             'docqna_show_diagnostics', 'docqna_temperature',
	             'docqna_top_percent', 'docqna_max_tokens',
	             'docqna_response_format', 'docqna_tool_choice',
	             'docqna_reasoning', 'docqna_file_search_store_names_input',
	             'docqna_file_search_store_names' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_docqna_all( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Document Q&A controls, loaded documents, index state, outputs, and messages.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	reset_docqna_controls( )
	unload_docqna_documents( )
	clear_docqna_messages( )

def clear_docqna_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Document Q&A system instructions and selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		None
		
	"""
	st.session_state[ 'docqna_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def load_docqna_instruction( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Document Q&A system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		prompt_text = fetch_prompt_text( cfg.DB_PATH, name )
		if prompt_text is not None:
			st.session_state[ 'docqna_system_instructions' ] = prompt_text

def convert_docqna_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Document Q&A system instructions between XML-like delimiters and Markdown
		headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text_value = st.session_state.get( 'docqna_system_instructions', '' )
	if not isinstance( text_value, str ) or not text_value.strip( ):
		return
	
	source = text_value.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'docqna_system_instructions' ] = converted

def get_docqna_sources( ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return supported Document Q&A source options for Buddy's provider-aware workflow.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		List[str]
			Document Q&A source option names.
		
	"""
	provider_name = get_provider_name( )
	
	options = [ 'Local Upload' ]
	
	if provider_name == 'GPT':
		options.extend( [ 'OpenAI File ID', 'OpenAI Vector Store ID' ] )
	
	elif provider_name == 'Gemini':
		options.extend( [ 'Gemini File Search Store' ] )
	
	elif provider_name == 'Grok':
		options.extend( [ 'xAI Collection' ] )
	
	return options

def get_docqna_extension( filename: str | None ) -> str:
	"""
	
		Purpose:
		--------
		Return a lowercase file extension for Document Q&A extraction and preview routing.
		
		Parameters:
		-----------
		filename: str | None
			File name to inspect.
		
		Returns:
		--------
		str
			Lowercase file extension including the leading period.
		
	"""
	if not isinstance( filename, str ) or not filename.strip( ):
		return ''
	
	return Path( filename ).suffix.lower( )

def compute_fingerprint( documents: List[ Dict[ str, Any ] ] ) -> str:
	"""
	
		Purpose:
		--------
		Compute a stable fingerprint for active Document Q&A files.
		
		Parameters:
		-----------
		documents: List[Dict[str, Any]]
			Active document metadata and byte payloads.
		
		Returns:
		--------
		str
			SHA-256 fingerprint for the active document set.
		
	"""
	hasher = hashlib.sha256( )
	
	if not isinstance( documents, list ):
		return ''
	
	for doc in documents:
		if not isinstance( doc, dict ):
			continue
		
		name = str( doc.get( 'name', '' ) )
		content = doc.get( 'bytes', b'' )
		hasher.update( name.encode( 'utf-8', errors='ignore' ) )
		
		if isinstance( content, bytes ):
			hasher.update( content )
	
	return hasher.hexdigest( )

def extract_pdf_text( file_bytes: bytes ) -> str:
	"""
	
		Purpose:
		--------
		Extract text from PDF bytes using PyMuPDF when available.
		
		Parameters:
		-----------
		file_bytes: bytes
			PDF byte payload.
		
		Returns:
		--------
		str
			Extracted PDF text.
		
	"""
	if not isinstance( file_bytes, bytes ) or len( file_bytes ) == 0:
		return ''
	
	try:
		import fitz
		
		pages: List[ str ] = [ ]
		with fitz.open( stream=file_bytes, filetype='pdf' ) as doc:
			for page in doc:
				pages.append( page.get_text( 'text' ) or '' )
		
		return '\n\n'.join( pages ).strip( )
	except Exception:
		return extract_text_bytes( file_bytes )

def extract_text_bytes( file_bytes: bytes ) -> str:
	"""
	
		Purpose:
		--------
		Decode plain-text-like document bytes.
		
		Parameters:
		-----------
		file_bytes: bytes
			Text byte payload.
		
		Returns:
		--------
		str
			Decoded text.
		
	"""
	if not isinstance( file_bytes, bytes ) or len( file_bytes ) == 0:
		return ''
	
	for encoding in [ 'utf-8', 'utf-8-sig', 'cp1252', 'latin-1' ]:
		try:
			return file_bytes.decode( encoding ).strip( )
		except Exception:
			continue
	
	return ''

def extract_docx_text( file_bytes: bytes ) -> str:
	"""
	
		Purpose:
		--------
		Extract text from DOCX bytes using the zipped WordprocessingML document body.
		
		Parameters:
		-----------
		file_bytes: bytes
			DOCX byte payload.
		
		Returns:
		--------
		str
			Extracted DOCX text.
		
	"""
	if not isinstance( file_bytes, bytes ) or len( file_bytes ) == 0:
		return ''
	
	try:
		with zipfile.ZipFile( io.BytesIO( file_bytes ) ) as archive:
			xml_bytes = archive.read( 'word/document.xml' )
		
		root = ET.fromstring( xml_bytes )
		namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
		paragraphs: List[ str ] = [ ]
		
		for paragraph in root.iter( f'{namespace}p' ):
			parts: List[ str ] = [ ]
			
			for node in paragraph.iter( f'{namespace}t' ):
				if node.text:
					parts.append( node.text )
			
			text = ''.join( parts ).strip( )
			if text:
				paragraphs.append( text )
		
		return '\n\n'.join( paragraphs ).strip( )
	except Exception:
		return ''

def extract_docqna_text( filename: str, file_bytes: bytes ) -> str:
	"""
	
		Purpose:
		--------
		Extract document text from supported Document Q&A file types.
		
		Parameters:
		-----------
		filename: str
			Uploaded file name.
		
		file_bytes: bytes
			Uploaded file bytes.
		
		Returns:
		--------
		str
			Extracted text.
		
	"""
	extension = get_docqna_extension( filename )
	
	if extension == '.pdf':
		return extract_pdf_text( file_bytes )
	
	if extension == '.docx':
		return extract_docx_text( file_bytes )
	
	if extension in [ '.txt', '.md', '.csv', '.json', '.xml', '.py', '.cs', '.sql',
	                  '.yaml', '.yml', '.html', '.css', '.js', '.ts' ]:
		return extract_text_bytes( file_bytes )
	
	return extract_text_bytes( file_bytes )

def normalize_docqna_text( text: str ) -> str:
	"""
	
		Purpose:
		--------
		Normalize extracted document text for local Document Q&A retrieval.
		
		Parameters:
		-----------
		text: str
			Extracted document text.
		
		Returns:
		--------
		str
			Normalized document text.
		
	"""
	if not isinstance( text, str ):
		return ''
	
	value = text.replace( '\x00', ' ' )
	value = re.sub( r'[ \t]+', ' ', value )
	value = re.sub( r'\n{3,}', '\n\n', value )
	return value.strip( )

def chunk_docqna_text( text: str, chunk_size: int = 900, chunk_overlap: int = 150 ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Split document text into overlapping word-based chunks for local retrieval.
		
		Parameters:
		-----------
		text: str
			Document text.
		
		chunk_size: int
			Maximum words per chunk.
		
		chunk_overlap: int
			Words overlapping between adjacent chunks.
		
		Returns:
		--------
		List[str]
			Document text chunks.
		
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return [ ]
	
	try:
		size = int( chunk_size )
	except Exception:
		size = 900
	
	try:
		overlap = int( chunk_overlap )
	except Exception:
		overlap = 150
	
	if size <= 0:
		size = 900
	
	if overlap < 0:
		overlap = 0
	
	if overlap >= size:
		overlap = max( 0, size // 5 )
	
	words = text.split( )
	if len( words ) == 0:
		return [ ]
	
	chunks: List[ str ] = [ ]
	step = max( 1, size - overlap )
	start = 0
	
	while start < len( words ):
		end = min( start + size, len( words ) )
		chunk = ' '.join( words[ start:end ] ).strip( )
		
		if chunk:
			chunks.append( chunk )
		
		if end >= len( words ):
			break
		
		start += step
	
	return chunks

def load_docqna_file( uploaded: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Load one or more Streamlit uploaded files into Document Q&A state. The file
		uploader widget key itself is never initialized or assigned manually.
		
		Parameters:
		-----------
		uploaded: Any
			Streamlit uploaded file object or list of uploaded file objects.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Loaded active document records.
		
	"""
	if uploaded is None:
		return [ ]
	
	files = uploaded if isinstance( uploaded, list ) else [ uploaded ]
	active_docs: List[ Dict[ str, Any ] ] = [ ]
	texts: Dict[ str, str ] = { }
	doc_bytes: Dict[ str, bytes ] = { }
	
	for item in files:
		if item is None:
			continue
		
		name = getattr( item, 'name', 'uploaded_document' )
		
		try:
			content = item.getvalue( ) if hasattr( item, 'getvalue' ) else item.read( )
		except Exception:
			content = None
		
		if not isinstance( content, bytes ) or len( content ) == 0:
			continue
		
		text = extract_docqna_text( filename=name, file_bytes=content )
		
		active_docs.append(
			{
					'name': name,
					'extension': get_docqna_extension( name ),
					'bytes': content,
					'text': text,
					'size': len( content ),
			} )
		
		texts[ name ] = text
		doc_bytes[ name ] = content
	
	st.session_state[ 'docqna_uploaded' ] = [ doc.get( 'name', '' ) for doc in active_docs ]
	st.session_state[ 'docqna_files' ] = active_docs
	st.session_state[ 'docqna_active_docs' ] = active_docs
	st.session_state[ 'active_docs' ] = [ doc.get( 'name', '' ) for doc in active_docs ]
	st.session_state[ 'docqna_texts' ] = texts
	st.session_state[ 'doc_bytes' ] = doc_bytes
	
	if len( active_docs ) > 0:
		st.session_state[ 'docqna_bytes' ] = active_docs[ 0 ].get( 'bytes' )
	else:
		st.session_state[ 'docqna_bytes' ] = None
	
	fingerprint = compute_fingerprint( active_docs )
	if fingerprint != st.session_state.get( 'docqna_fingerprint', '' ):
		st.session_state[ 'docqna_vec_ready' ] = False
		st.session_state[ 'docqna_fingerprint' ] = fingerprint
		st.session_state[ 'docqna_index_status' ] = 'Loaded; not indexed'
	
	return active_docs

def get_document_names( ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return active Document Q&A document names.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		List[str]
			Active document names.
		
	"""
	docs = st.session_state.get( 'docqna_active_docs', [ ] )
	
	if not isinstance( docs, list ):
		return [ ]
	
	return [ doc.get( 'name', '' ) for doc in docs if isinstance( doc, dict ) and doc.get( 'name' ) ]

def render_document_preview( ) -> None:
	"""
	
		Purpose:
		--------
		Render active Document Q&A document previews by file extension.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	docs = st.session_state.get( 'docqna_active_docs', [ ] )
	
	if not isinstance( docs, list ) or len( docs ) == 0:
		st.info( 'No active document loaded.' )
		return
	
	for doc in docs:
		if not isinstance( doc, dict ):
			continue
		
		name = doc.get( 'name', 'Document' )
		extension = doc.get( 'extension', '' )
		content = doc.get( 'bytes', b'' )
		text = doc.get( 'text', '' )
		
		with st.expander( label=f'Preview: {name}', icon='📄',
				expanded=False, width='stretch' ):
			st.caption(
				f'File type: {extension or "unknown"} | Size: {doc.get( "size", 0 )} bytes' )
			
			if extension == '.pdf' and isinstance( content, bytes ):
				try:
					st.pdf( content, height=420 )
				except Exception:
					st.text_area( label='Extracted Text Preview',
						value=text[ :12000 ] if isinstance( text, str ) else '',
						height=300, width='stretch', disabled=True )
			elif extension == '.md' and isinstance( text, str ):
				st.markdown( text[ :12000 ] )
			elif isinstance( text, str ) and text.strip( ):
				st.text_area( label='Extracted Text Preview',
					value=text[ :12000 ],
					height=300, width='stretch', disabled=True )
			else:
				st.warning( 'No readable text preview is available for this file.' )

def rebuild_docqna_index( ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Rebuild the local Document Q&A retrieval index from active documents.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		List[Dict[str, Any]]
			Indexed chunk records.
		
	"""
	docs = st.session_state.get( 'docqna_active_docs', [ ] )
	
	if not isinstance( docs, list ) or len( docs ) == 0:
		st.session_state[ 'docqna_chunks' ] = [ ]
		st.session_state[ 'docqna_vec_ready' ] = False
		st.session_state[ 'docqna_chunk_count' ] = 0
		st.session_state[ 'docqna_index_status' ] = 'No documents loaded'
		return [ ]
	
	chunk_records: List[ Dict[ str, Any ] ] = [ ]
	chunk_size = st.session_state.get( 'docqna_chunk_size', 900 )
	chunk_overlap = st.session_state.get( 'docqna_chunk_overlap', 150 )
	
	for doc in docs:
		if not isinstance( doc, dict ):
			continue
		
		name = doc.get( 'name', 'Document' )
		text = normalize_docqna_text( doc.get( 'text', '' ) )
		
		for index, chunk in enumerate( chunk_docqna_text(
				text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap ) ):
			chunk_records.append(
				{
						'document': name,
						'chunk_index': index + 1,
						'text': chunk,
						'tokens': count_tokens( chunk ),
				} )
	
	st.session_state[ 'docqna_chunks' ] = chunk_records
	st.session_state[ 'docqna_chunk_count' ] = len( chunk_records )
	st.session_state[ 'docqna_index_status' ] = (
			f'Indexed {len( chunk_records )} chunks'
			if len( chunk_records ) > 0 else 'No readable chunks'
	)
	st.session_state[ 'docqna_vec_ready' ] = len( chunk_records ) > 0
	
	return chunk_records

def score_chunk( query: str, chunk: str ) -> float:
	"""
	
		Purpose:
		--------
		Compute a simple lexical score for local Document Q&A retrieval.
		
		Parameters:
		-----------
		query: str
			User query.
		
		chunk: str
			Document chunk text.
		
		Returns:
		--------
		float
			Relevance score.
		
	"""
	if not isinstance( query, str ) or not isinstance( chunk, str ):
		return 0.0
	
	query_terms = {
			term.lower( )
			for term in re.findall( r'\b\w+\b', query )
			if len( term ) > 2
	}
	
	if len( query_terms ) == 0:
		return 0.0
	
	chunk_terms = re.findall( r'\b\w+\b', chunk.lower( ) )
	if len( chunk_terms ) == 0:
		return 0.0
	
	chunk_set = set( chunk_terms )
	overlap = len( query_terms.intersection( chunk_set ) )
	density = overlap / max( 1, len( query_terms ) )
	
	return float( density )

def retrieve_chunks( query: str, top_k: int = 6 ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Retrieve top local Document Q&A chunks using indexed chunk records.
		
		Parameters:
		-----------
		query: str
			User query.
		
		top_k: int
			Maximum chunks to return.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Ranked retrieval hits.
		
	"""
	chunks = st.session_state.get( 'docqna_chunks', [ ] )
	
	if not isinstance( chunks, list ) or len( chunks ) == 0:
		chunks = rebuild_docqna_index( )
	
	results: List[ Dict[ str, Any ] ] = [ ]
	
	for item in chunks:
		if not isinstance( item, dict ):
			continue
		
		text_value = item.get( 'text', '' )
		score = score_chunk( query, text_value )
		
		if score <= 0:
			continue
		
		result = dict( item )
		result[ 'score' ] = score
		results.append( result )
	
	results.sort( key=lambda row: row.get( 'score', 0.0 ), reverse=True )
	
	if len( results ) == 0 and isinstance( chunks, list ):
		results = [
				{
						**item,
						'score': 0.0,
				}
				for item in chunks[ : int( top_k ) ]
				if isinstance( item, dict )
		]
	
	return results[ : int( top_k ) ]

def build_docqna_local_prompt( query: str, hits: List[ Dict[ str, Any ] ] ) -> str:
	"""
	
		Purpose:
		--------
		Build a grounded prompt from local Document Q&A retrieval hits.
		
		Parameters:
		-----------
		query: str
			User question.
		
		hits: List[Dict[str, Any]]
			Retrieved document chunks.
		
		Returns:
		--------
		str
			Prompt with retrieved context.
		
	"""
	system = str( st.session_state.get( 'docqna_system_instructions', '' ) or '' ).strip( )
	context_blocks: List[ str ] = [ ]
	
	for hit in hits:
		if not isinstance( hit, dict ):
			continue
		
		doc_name = hit.get( 'document', 'Document' )
		chunk_index = hit.get( 'chunk_index', '' )
		text_value = hit.get( 'text', '' )
		
		if isinstance( text_value, str ) and text_value.strip( ):
			context_blocks.append(
				f'[Document: {doc_name} | Chunk: {chunk_index}]\n{text_value}'.strip( ) )
	
	context = '\n\n'.join( context_blocks ).strip( )
	prompt_parts: List[ str ] = [ ]
	
	if system:
		prompt_parts.append( system )
	
	if context:
		prompt_parts.append(
			'Use the following document excerpts to answer the question. If the excerpts do '
			'not contain the answer, say you do not have enough information.\n\n'
			f'{context}' )
	else:
		prompt_parts.append(
			'No relevant document excerpts were retrieved. If the document context does not '
			'contain the answer, say you do not have enough information.' )
	
	prompt_parts.append( f'Question:\n{query}\n\nAnswer:' )
	
	return '\n\n'.join( prompt_parts ).strip( )

def run_docqna_query( query: str ) -> str:
	"""
	
		Purpose:
		--------
		Run a local-upload Document Q&A query using retrieved document chunks and the
		selected provider chat wrapper.
		
		Parameters:
		-----------
		query: str
			User question.
		
		Returns:
		--------
		str
			Generated answer.
		
	"""
	provider_name = get_provider_name( )
	chat = get_chat_module( )
	top_k = int( st.session_state.get( 'docqna_top_k', 6 ) or 6 )
	hits = retrieve_chunks( query=query, top_k=top_k )
	
	st.session_state[ 'docqna_last_hits' ] = hits
	st.session_state[ 'docqna_last_sources' ] = [ {
					'document': hit.get( 'document' ),
					'chunk_index': hit.get( 'chunk_index' ),
					'score': hit.get( 'score' ),
					'tokens': hit.get( 'tokens' ),
			}
			for hit in hits
			if isinstance( hit, dict ) ]
	
	prompt = build_docqna_local_prompt( query=query, hits=hits )
	
	if provider_name == 'Gemini':
		apply_gemini_runtime_config( )
		result = chat.generate_text(
			prompt=prompt,
			model=st.session_state.get( 'docqna_model' ) or st.session_state.get( 'text_model' ),
			temperature=st.session_state.get( 'docqna_temperature' ),
			top_p=st.session_state.get( 'docqna_top_percent' ),
			top_k=st.session_state.get( 'docqna_top_k' ),
			max_tokens=st.session_state.get( 'docqna_max_tokens' ),
			instruct=st.session_state.get( 'docqna_system_instructions' ),
			stream=False )
		return str( result or '' ).strip( )
	
	if provider_name == 'GPT':
		result = chat.generate_text(
			prompt=prompt,
			model=st.session_state.get( 'docqna_model' ) or st.session_state.get( 'text_model' ),
			temperature=st.session_state.get( 'docqna_temperature' ),
			top_p=st.session_state.get( 'docqna_top_percent' ),
			frequency=st.session_state.get( 'docqna_frequency_penalty' ),
			presence=st.session_state.get( 'docqna_presence_penalty' ),
			max_tokens=st.session_state.get( 'docqna_max_tokens' ),
			store=st.session_state.get( 'docqna_store' ),
			stream=False,
			instruct=st.session_state.get( 'docqna_system_instructions' ) )
		return str( result or '' ).strip( )
	
	try:
		result = chat.create(
			prompt=prompt,
			model=st.session_state.get( 'docqna_model' ) or st.session_state.get( 'text_model' ),
			max_tokens=st.session_state.get( 'docqna_max_tokens' ) or 10000,
			temperature=st.session_state.get( 'docqna_temperature' ) or 0.8,
			top_p=st.session_state.get( 'docqna_top_percent' ) or 0.9,
			effort=st.session_state.get( 'docqna_reasoning' ) or 'high',
			format=st.session_state.get( 'docqna_response_format' ) or 'text',
			store=bool( st.session_state.get( 'docqna_store', True ) ),
			instruct=st.session_state.get( 'docqna_system_instructions' ) )
		return str( result or '' ).strip( )
	except Exception:
		result = chat.generate_text( prompt=prompt )
		return str( result or '' ).strip( )

def run_remote_query( query: str ) -> str:
	"""
	
		Purpose:
		--------
		Run a provider-native remote Document Q&A query using the selected Vector Stores
		resource from Buddy's Vector Stores mode, with manual source fields as fallback.
		
		Parameters:
		-----------
		query: str
			User question.
		
		Returns:
		--------
		str
			Generated answer.
		
	"""
	provider_name = get_provider_name( )
	chat = get_chat_module( )
	source = st.session_state.get( 'docqna_source', 'Local Upload' )
	
	if provider_name == 'GPT':
		selected_vector_store_ids = get_store_ids( 'GPT' )
		manual_vector_store_ids = [ ]
		
		if source == 'OpenAI Vector Store ID':
			manual_vector_store_ids = split_text_values(
				st.session_state.get( 'docqna_vector_store_id', '' ),
				delimiter=',' )
		
		vector_store_ids = merge_unique_strings(
			primary=manual_vector_store_ids,
			secondary=selected_vector_store_ids )
		
		tools: List[ Dict[ str, Any ] ] = [ ]
		
		if len( vector_store_ids ) > 0:
			tools.append(
				{
						'type': 'file_search',
						'vector_store_ids': vector_store_ids,
				} )
		
		if source == 'OpenAI File ID' and not st.session_state.get( 'docqna_file_id' ):
			return 'Enter an OpenAI file ID before asking a file-based question.'
		
		if source == 'OpenAI Vector Store ID' and len( vector_store_ids ) == 0:
			return 'Enter or select an OpenAI vector store ID before asking a vector-store question.'
		
		result = chat.generate_text(
			prompt=query,
			model=st.session_state.get( 'docqna_model' ) or st.session_state.get( 'text_model' ),
			temperature=st.session_state.get( 'docqna_temperature' ),
			top_p=st.session_state.get( 'docqna_top_percent' ),
			frequency=st.session_state.get( 'docqna_frequency_penalty' ),
			presence=st.session_state.get( 'docqna_presence_penalty' ),
			max_tokens=st.session_state.get( 'docqna_max_tokens' ),
			store=st.session_state.get( 'docqna_store' ),
			stream=False,
			instruct=st.session_state.get( 'docqna_system_instructions' ),
			tools=tools,
			include=[ 'file_search_call.results' ] if len( tools ) > 0 else [ ],
			vector_store_ids=vector_store_ids )
		
		response_obj = getattr( chat, 'response', None )
		st.session_state[ 'docqna_last_sources' ] = extract_sources( response_obj )
		st.session_state[ 'last_sources' ] = st.session_state[ 'docqna_last_sources' ]
		
		return str( result or '' ).strip( )
	
	if provider_name == 'Gemini':
		apply_gemini_runtime_config( )
		
		manual_store_names = [ ]
		
		if source == 'Gemini File Search Store':
			manual_store_names = split_text_values(
				st.session_state.get( 'docqna_file_search_store_names_input', '' ),
				delimiter=',' )
		
		selected_store_names = get_active_gemini_file_search_store_names( 'Gemini' )
		
		store_names = merge_unique_strings(
			primary=manual_store_names,
			secondary=selected_store_names )
		
		st.session_state[ 'docqna_file_search_store_names' ] = store_names
		
		if get_gemini_vector_backend( ) == 'Cloud Buckets' and len( store_names ) == 0:
			return (
					'The selected Gemini backend is Cloud Buckets. Cloud Buckets are storage '
					'objects, not Gemini File Search Store resources. Select a Gemini File '
					'Search Store backend or use Local Upload after downloading/loading the object.'
			)
		
		if source == 'Gemini File Search Store' and len( store_names ) == 0:
			return 'Enter or select at least one Gemini File Search Store resource name before asking.'
		
		result = chat.generate_text(
			prompt=query,
			model=st.session_state.get( 'docqna_model' ) or st.session_state.get( 'text_model' ),
			temperature=st.session_state.get( 'docqna_temperature' ),
			top_p=st.session_state.get( 'docqna_top_percent' ),
			top_k=st.session_state.get( 'docqna_top_k' ),
			max_tokens=st.session_state.get( 'docqna_max_tokens' ),
			instruct=st.session_state.get( 'docqna_system_instructions' ),
			file_search_store_names=store_names,
			stream=False )
		
		return str( result or '' ).strip( )
	
	if provider_name == 'Grok':
		collection_ids = get_active_grok_collection_ids( 'Grok' )
		
		if len( collection_ids ) == 0:
			return 'Select an xAI Collection in Vector Stores mode before using Grok remote Document Q&A.'
		
		vectorstores = get_vectorstores_module( 'Grok' )
		result = vectorstores.search(
			prompt=query,
			store_id=collection_ids[ 0 ] )
		
		if isinstance( result, str ) and result.strip( ):
			return result.strip( )
		
		if isinstance( result, dict ):
			return json.dumps( result, indent=2, default=str )
		
		return str( result or '' ).strip( )
	
	return run_docqna_query( query )

def route_document_query( prompt: str ) -> str:
	"""
	
		Purpose:
		--------
		Route a document question through the active Document Q&A source.
		
		Parameters:
		-----------
		prompt: str
			User question.
		
		Returns:
		--------
		str
			Assistant answer text.
		
	"""
	query = str( prompt or '' ).strip( )
	if not query:
		return 'Please enter a question about the active document source.'
	
	source = st.session_state.get( 'docqna_source', 'Local Upload' )
	
	if source == 'Local Upload':
		return run_docqna_query( query )
	
	return run_remote_query( query )

def summarize_document( ) -> str:
	"""
	
		Purpose:
		--------
		Summarize the currently active document source through Document Q&A routing.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		str
			Document summary.
		
	"""
	summary_prompt = """
		Provide a clear, structured summary of the active document source.
		
		Include:
		- Purpose
		- Key themes
		- Major conclusions
		- Important data points, if any
		- Policy or operational implications, if applicable
		
		Be precise and concise.
	"""
	return route_document_query( summary_prompt.strip( ) )

def render_hits( ) -> None:
	"""
	
		Purpose:
		--------
		Render Document Q&A retrieval hits and sources.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	hits = st.session_state.get( 'docqna_last_hits', [ ] )
	sources = st.session_state.get( 'docqna_last_sources', [ ] )
	
	if isinstance( hits, list ) and len( hits ) > 0:
		with st.expander( label='Retrieved Chunks', icon='🧩',
				expanded=False, width='stretch' ):
			df_hits = pd.DataFrame( hits )
			st.data_editor( df_hits, use_container_width=True, hide_index=True )
	
	if isinstance( sources, list ) and len( sources ) > 0:
		with st.expander( label='Document Sources', icon='📌',
				expanded=False, width='stretch' ):
			df_sources = pd.DataFrame( sources )
			st.data_editor( df_sources, use_container_width=True, hide_index=True )

def get_docqna_avatar( provider_name: str ) -> str:
	"""
	
		Purpose:
		--------
		Return the configured assistant avatar for the active Document Q&A provider.
		
		Parameters:
		-----------
		provider_name: str
			Selected provider name.
		
		Returns:
		--------
		str
			Avatar string or configured avatar path.
		
	"""
	if provider_name == 'GPT':
		return getattr( cfg, 'GPT_AVATAR', getattr( cfg, 'BUDDY', '🧠' ) )
	
	if provider_name == 'Gemini':
		return getattr( cfg, 'GEMINI_AVATAR', getattr( cfg, 'BUDDY', '🧠' ) )
	
	if provider_name == 'Grok':
		return getattr( cfg, 'GROK_AVATAR', getattr( cfg, 'BUDDY', '🧠' ) )
	
	return getattr( cfg, 'BUDDY', '🧠' )

# ======================================================================================
# FILES MODE UTILITIES
# ======================================================================================

def ensure_runtime_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure Files mode runtime keys exist before Files mode widgets or execution paths
		read them. File uploader widget keys are intentionally excluded because Streamlit
		does not permit file uploader values to be assigned through session_state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_files_mode_state( )
	ensure_key( 'files_filter_purpose', '' )
	ensure_key( 'files_table_data', [ ] )
	ensure_key( 'files_metadata', { } )
	ensure_key( 'files_content', '' )
	ensure_key( 'files_delete_result', { } )
	ensure_key( 'files_last_answer', '' )
	ensure_key( 'files_last_upload', { } )
	ensure_key( 'files_last_list', [ ] )
	ensure_key( 'files_last_operation', '' )
	ensure_key( 'files_system_instructions', '' )

def clear_files_messages( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Files mode chat messages without clearing loaded metadata or outputs.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'files_messages' ] = [ ]

def clear_files_outputs( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Files mode output state while preserving configuration controls.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'files_table_data' ] = [ ]
	st.session_state[ 'files_metadata' ] = { }
	st.session_state[ 'files_content' ] = ''
	st.session_state[ 'files_delete_result' ] = { }
	st.session_state[ 'files_last_answer' ] = ''
	st.session_state[ 'files_last_upload' ] = { }
	st.session_state[ 'files_last_list' ] = [ ]
	st.session_state[ 'files_last_operation' ] = ''
	st.session_state[ 'last_answer' ] = ''

def reset_files_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Files mode controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'files_model', 'files_purpose', 'files_filter_purpose',
	             'files_id', 'files_url', 'files_type', 'files_table',
	             'files_temperature', 'files_top_percent', 'files_max_tokens',
	             'files_response_format', 'files_reasoning', 'files_tool_choice',
	             'files_store', 'files_stream', 'files_background' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_files_all( ) -> None:
	"""
	
		Purpose:
		--------
		Reset Files mode controls, messages, and output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	reset_files_controls( )
	clear_files_outputs( )
	clear_files_messages( )

def clear_files_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Clear Files mode system instructions and selected prompt template.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'files_system_instructions' ] = ''
	st.session_state[ 'instructions' ] = ''

def load_files_instruction( ) -> None:
	"""
	
		Purpose:
		--------
		Load the selected prompt template into Files mode system instructions.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	name = st.session_state.get( 'instructions' )
	if name and name != 'No Templates Found':
		prompt_text = fetch_prompt_text( cfg.DB_PATH, name )
		if prompt_text is not None:
			st.session_state[ 'files_system_instructions' ] = prompt_text

def convert_files_instructions( ) -> None:
	"""
	
		Purpose:
		--------
		Convert Files mode system instructions between XML-like delimiters and Markdown
		headings.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	text_value = st.session_state.get( 'files_system_instructions', '' )
	if not isinstance( text_value, str ) or not text_value.strip( ):
		return
	
	source = text_value.strip( )
	if cfg.XML_BLOCK_PATTERN.search( source ):
		converted = convert_xml( source )
	else:
		converted = convert_markdown( source )
	
	st.session_state[ 'files_system_instructions' ] = converted

def get_purpose_options( files: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return provider upload-purpose options for Files mode.
		
		Parameters:
		-----------
		files: Any
			Provider Files wrapper instance.
		
		Returns:
		--------
		List[str]
			Upload purpose options.
		
	"""
	options = getattr( files, 'upload_purpose_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return options
	
	provider_name = get_provider_name( )
	
	if provider_name == 'GPT':
		return [ 'assistants', 'batch', 'fine-tune', 'vision', 'user_data' ]
	
	if provider_name == 'Gemini':
		return [ 'user_data' ]
	
	return [ 'user_data' ]

def get_filter_options( files: Any ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return provider purpose-filter options for Files mode list operations.
		
		Parameters:
		-----------
		files: Any
			Provider Files wrapper instance.
		
		Returns:
		--------
		List[str]
			Filter purpose options.
		
	"""
	options = getattr( files, 'file_purpose_options', None )
	if isinstance( options, list ) and len( options ) > 0:
		return [ '' ] + options
	
	provider_name = get_provider_name( )
	
	if provider_name == 'GPT':
		return [ '', 'assistants', 'batch', 'fine-tune', 'vision', 'user_data' ]
	
	return [ '' ]

def normalize_files_object( value: Any ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Normalize a provider file object into a serializable dictionary for display.
		
		Parameters:
		-----------
		value: Any
			Provider file object, dictionary, or scalar.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized metadata dictionary.
		
	"""
	if value is None:
		return { }
	
	if isinstance( value, dict ):
		return value
	
	if hasattr( value, 'model_dump' ):
		try:
			return value.model_dump( )
		except Exception:
			pass
	
	if hasattr( value, '__dict__' ):
		try:
			data = { }
			for key, item in vars( value ).items( ):
				if key.startswith( '_' ):
					continue
				
				if item is None or isinstance( item, (str, int, float, bool) ):
					data[ key ] = item
				else:
					data[ key ] = str( item )
			
			return data
		except Exception:
			pass
	
	return { 'value': str( value ) }

def normalize_files_list( value: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Normalize a provider file list response into display-ready dictionaries.
		
		Parameters:
		-----------
		value: Any
			Provider list response.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Normalized file metadata rows.
		
	"""
	if value is None:
		return [ ]
	
	if isinstance( value, list ):
		return [ normalize_files_object( item ) for item in value ]
	
	data = getattr( value, 'data', None )
	if isinstance( data, list ):
		return [ normalize_files_object( item ) for item in data ]
	
	files = getattr( value, 'files', None )
	if isinstance( files, list ):
		return [ normalize_files_object( item ) for item in files ]
	
	file_list = getattr( value, 'file_list', None )
	if isinstance( file_list, list ):
		return [ normalize_files_object( item ) for item in file_list ]
	
	return [ normalize_files_object( value ) ]

def get_files_id ( row: Dict[ str, Any ] ) -> str:
	"""
	
		Purpose:
		--------
		Return a file identifier from a normalized file metadata row.
		
		Parameters:
		-----------
		row: Dict[str, Any]
			Normalized file metadata row.
		
		Returns:
		--------
		str
			File identifier or empty string.
		
	"""
	if not isinstance( row, dict ):
		return ''
	
	for key in [ 'id', 'file_id', 'name', 'uri', 'resource_name' ]:
		value = row.get( key )
		if isinstance( value, str ) and value.strip( ):
			return value.strip( )
	
	return ''

def build_selector_options( rows: List[ Dict[ str, Any ] ] ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Build display labels for selecting files from listed metadata rows.
		
		Parameters:
		-----------
		rows: List[Dict[str, Any]]
			Normalized file metadata rows.
		
		Returns:
		--------
		List[str]
			Display labels.
		
	"""
	options: List[ str ] = [ ]
	
	for row in rows:
		if not isinstance( row, dict ):
			continue
		
		file_id = get_files_id ( row )
		name = ( row.get( 'filename' )
				or row.get( 'display_name' )
				or row.get( 'name' )
				or row.get( 'id' )
				or 'file' )
		
		if file_id:
			options.append( f'{name} — {file_id}' )
	
	return options

def get_option_id( option: str | None ) -> str:
	"""
	
		Purpose:
		--------
		Extract a file identifier from a UI selection label.
		
		Parameters:
		-----------
		option: str | None
			Selected label.
		
		Returns:
		--------
		str
			File identifier or empty string.
		
	"""
	if not isinstance( option, str ) or not option.strip( ):
		return ''
	
	if ' — ' in option:
		return option.rsplit( ' — ', 1 )[ 1 ].strip( )
	
	return option.strip( )

def upload_provider_file( files: Any, path: str, purpose: str | None = None ) -> Any:
	"""
	
		Purpose:
		--------
		Upload a file using the selected provider Files wrapper.
		
		Parameters:
		-----------
		files: Any
			Provider Files wrapper instance.
		
		path: str
			Temporary path to upload.
		
		purpose: str | None
			Optional provider upload purpose.
		
		Returns:
		--------
		Any
			Provider upload result.
		
	"""
	provider_name = get_provider_name( )
	
	if provider_name == 'Gemini':
		apply_gemini_runtime_config( )
		
		try:
			return files.upload( path=path )
		except TypeError:
			try:
				return files.upload( file_path=path )
			except TypeError:
				return files.upload( path )
	
	try:
		return files.upload( path=path, purpose=purpose or 'user_data' )
	except TypeError:
		try:
			return files.upload( filepath=path, purpose=purpose or 'user_data' )
		except TypeError:
			return files.upload( path, purpose or 'user_data' )

def list_provider_files( files: Any, purpose: str | None = None ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		List files using the selected provider Files wrapper.
		
		Parameters:
		-----------
		files: Any
			Provider Files wrapper instance.
		
		purpose: str | None
			Optional provider purpose filter.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Normalized file metadata rows.
		
	"""
	provider_name = get_provider_name( )
	
	if provider_name == 'Gemini':
		apply_gemini_runtime_config( )
		
		if hasattr( files, 'list_files' ):
			result = files.list_files( )
		else:
			result = files.list( )
		
		return normalize_files_list( result )
	
	try:
		result = files.list( purpose=purpose if purpose else None )
	except TypeError:
		result = files.list( )
	
	return normalize_files_list( result )

def retrieve_provider_file( files: Any, file_id: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Retrieve file metadata using the selected provider Files wrapper.
		
		Parameters:
		-----------
		files: Any
			Provider Files wrapper instance.
		
		file_id: str
			Provider file identifier.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized metadata.
		
	"""
	if not isinstance( file_id, str ) or not file_id.strip( ):
		return { }
	
	provider_name = get_provider_name( )
	
	if provider_name == 'Gemini':
		apply_gemini_runtime_config( )
		
		try:
			result = files.retrieve( file_id=file_id.strip( ) )
		except TypeError:
			result = files.retrieve( file_id.strip( ) )
		
		return normalize_files_object( result )
	
	try:
		result = files.retrieve( id=file_id.strip( ) )
	except TypeError:
		try:
			result = files.retrieve( file_id=file_id.strip( ) )
		except TypeError:
			result = files.retrieve( file_id.strip( ) )
	
	return normalize_files_object( result )

def extract_file_content( files: Any, file_id: str ) -> str:
	"""
	
		Purpose:
		--------
		Extract file content using the selected provider Files wrapper when supported.
		
		Parameters:
		-----------
		files: Any
			Provider Files wrapper instance.
		
		file_id: str
			Provider file identifier.
		
		Returns:
		--------
		str
			Extracted or normalized file content.
		
	"""
	if not isinstance( file_id, str ) or not file_id.strip( ):
		return ''
	
	if not hasattr( files, 'extract' ):
		return ''
	
	try:
		result = files.extract( id=file_id.strip( ) )
	except TypeError:
		try:
			result = files.extract( file_id=file_id.strip( ) )
		except TypeError:
			result = files.extract( file_id.strip( ) )
	
	if isinstance( result, bytes ):
		try:
			return result.decode( 'utf-8' )
		except Exception:
			return str( result )
	
	if isinstance( result, str ):
		return result
	
	if isinstance( result, dict ):
		return json.dumps( result, indent=2, default=str )
	
	return str( result )

def delete_provider_file( files: Any, file_id: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Delete a file using the selected provider Files wrapper.
		
		Parameters:
		-----------
		files: Any
			Provider Files wrapper instance.
		
		file_id: str
			Provider file identifier.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized delete result.
		
	"""
	if not isinstance( file_id, str ) or not file_id.strip( ):
		return { }
	
	provider_name = get_provider_name( )
	
	if provider_name == 'Gemini':
		apply_gemini_runtime_config( )
	
	try:
		result = files.delete( id=file_id.strip( ) )
	except TypeError:
		try:
			result = files.delete( file_id=file_id.strip( ) )
		except TypeError:
			result = files.delete( file_id.strip( ) )
	
	return normalize_files_object( result )

def analyze_provider_file( files: Any, prompt: str, file_id: str | None=None,
		model: str | None=None ) -> str:
	"""
	
		Purpose:
		--------
		Analyze, summarize, search, or survey file content through provider-supported Files
		wrapper helpers.
		
		Parameters:
		-----------
		files: Any
			Provider Files wrapper instance.
		
		prompt: str
			User question or analysis instruction.
		
		file_id: str | None
			Optional provider file identifier.
		
		model: str | None
			Optional model name.
		
		Returns:
		--------
		str
			Provider analysis answer.
		
	"""
	if not isinstance( prompt, str ) or not prompt.strip( ):
		return ''
	
	clean_prompt = prompt.strip( )
	clean_file_id = file_id.strip( ) if isinstance( file_id, str ) and file_id.strip( ) else None
	
	if clean_file_id and hasattr( files, 'summarize' ):
		try:
			return str( files.summarize(
				id=clean_file_id,
				prompt=clean_prompt,
				model=model ) or '' ).strip( )
		except TypeError:
			pass
	
	if clean_file_id and hasattr( files, 'survey' ):
		try:
			result = files.survey( id=clean_file_id )
			return json.dumps( result, indent=2, default=str )
		except Exception:
			pass
	
	if clean_file_id:
		content = extract_file_content( files=files, file_id=clean_file_id )
		if content:
			query = (
					f'{st.session_state.get( "files_system_instructions", "" )}\n\n'
					f'Use the following file content to answer the user request.\n\n'
					f'File Content:\n{content[ :12000 ]}\n\n'
					f'User Request:\n{clean_prompt}'
			).strip( )
			chat = get_chat_module( )
			try:
				answer = chat.generate_text(
					prompt=query,
					model=model or st.session_state.get( 'files_model' ),
					temperature=st.session_state.get( 'files_temperature' ),
					top_p=st.session_state.get( 'files_top_percent' ),
					max_tokens=st.session_state.get( 'files_max_tokens' ),
					instruct=st.session_state.get( 'files_system_instructions' ) )
				return str( answer or '' ).strip( )
			except Exception:
				return content[ :12000 ]
	
	return ''

# ======================================================================================
# VECTOR STORE / FILE SEARCH / CLOUD BUCKET UTILITIES
# ======================================================================================

def ensure_storage_state( ) -> None:
	"""
	
		Purpose:
		--------
		Ensure storage mode runtime keys exist before Vector Stores, File Search Stores,
		or Cloud Buckets widgets and execution paths read them. File uploader widget keys
		are intentionally excluded because Streamlit does not permit file uploader values
		to be assigned through session_state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	ensure_vectorstores_mode_state( )
	ensure_file_search_mode_state( )
	ensure_cloudbuckets_mode_state( )
	ensure_key( 'storage_operation_result', { } )
	ensure_key( 'storage_table_data', [ ] )
	ensure_key( 'storage_last_operation', '' )
	ensure_key( 'storage_selected_option', '' )
	ensure_key( 'storage_last_answer', '' )

def clear_storage_outputs( ) -> None:
	"""
	
		Purpose:
		--------
		Clear storage mode output state while preserving controls.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	st.session_state[ 'storage_operation_result' ] = { }
	st.session_state[ 'storage_table_data' ] = [ ]
	st.session_state[ 'storage_last_operation' ] = ''
	st.session_state[ 'storage_selected_option' ] = ''
	st.session_state[ 'storage_last_answer' ] = ''

def reset_storage_controls( ) -> None:
	"""
	
		Purpose:
		--------
		Reset storage mode controls through a widget-safe callback.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	for key in [ 'stores_id', 'stores_name', 'stores_file_id', 'stores_file_ids',
	             'stores_path', 'stores_operation', 'filestore_id', 'filestore_name',
	             'filestore_selected_label', 'bucket_id', 'bucket_name',
	             'bucket_object_name', 'bucket_path', 'bucket_operation',
	             'selected_bucket_id', 'selected_bucket_label' ]:
		if key in st.session_state:
			del st.session_state[ key ]

def reset_storage_all( ) -> None:
	"""
	
		Purpose:
		--------
		Reset storage controls and output state.
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
	"""
	reset_storage_controls( )
	clear_storage_outputs( )

def normalize_storage_object( value: Any ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Normalize a provider storage object into a serializable dictionary for display.
		
		Parameters:
		-----------
		value: Any
			Provider object, dictionary, or scalar value.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized metadata dictionary.
		
	"""
	if value is None:
		return { }
	
	if isinstance( value, dict ):
		return value
	
	if hasattr( value, 'model_dump' ):
		try:
			return value.model_dump( )
		except Exception:
			pass
	
	if hasattr( value, '__dict__' ):
		try:
			data: Dict[ str, Any ] = { }
			
			for key, item in vars( value ).items( ):
				if key.startswith( '_' ):
					continue
				
				if item is None or isinstance( item, (str, int, float, bool) ):
					data[ key ] = item
				else:
					data[ key ] = str( item )
			
			return data
		except Exception:
			pass
	
	return { 'value': str( value ) }

def normalize_storage_list( value: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		Normalize provider list responses into display-ready dictionaries.
		
		Parameters:
		-----------
		value: Any
			Provider list response.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Normalized metadata rows.
		
	"""
	if value is None:
		return [ ]
	
	if isinstance( value, list ):
		return [ normalize_storage_object( item ) for item in value ]
	
	data = getattr( value, 'data', None )
	if isinstance( data, list ):
		return [ normalize_storage_object( item ) for item in data ]
	
	items = getattr( value, 'items', None )
	if isinstance( items, list ):
		return [ normalize_storage_object( item ) for item in items ]
	
	buckets = getattr( value, 'buckets', None )
	if isinstance( buckets, list ):
		return [ normalize_storage_object( item ) for item in buckets ]
	
	files = getattr( value, 'files', None )
	if isinstance( files, list ):
		return [ normalize_storage_object( item ) for item in files ]
	
	return [ normalize_storage_object( value ) ]

def get_storage_rowid( row: Dict[ str, Any ] ) -> str:
	"""
	
		Purpose:
		--------
		Return a storage identifier from a normalized metadata row.
		
		Parameters:
		-----------
		row: Dict[str, Any]
			Normalized metadata row.
		
		Returns:
		--------
		str
			Storage identifier or empty string.
		
	"""
	if not isinstance( row, dict ):
		return ''
	
	for key in [ 'id', 'name', 'resource_name', 'uri', 'bucket', 'bucket_name' ]:
		value = row.get( key )
		if isinstance( value, str ) and value.strip( ):
			return value.strip( )
	
	return ''

def build_storage_selectors( rows: List[ Dict[ str, Any ] ] ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Build display labels for selecting storage resources from metadata rows.
		
		Parameters:
		-----------
		rows: List[Dict[str, Any]]
			Normalized metadata rows.
		
		Returns:
		--------
		List[str]
			Display labels.
		
	"""
	options: List[ str ] = [ ]
	for row in rows:
		if not isinstance( row, dict ):
			continue
		
		resource_id = get_storage_rowid( row )
		name = ( row.get( 'display_name' )
				or row.get( 'name' )
				or row.get( 'id' )
				or row.get( 'bucket_name' )
				or 'resource' )
		
		if resource_id:
			options.append( f'{name} — {resource_id}' )
	
	return options

def get_storage_id_from_option( option: str | None ) -> str:
	"""
	
		Purpose:
		--------
		Extract a storage resource identifier from a UI selection label.
		
		Parameters:
		-----------
		option: str | None
			Selected option label.
		
		Returns:
		--------
		str
			Storage resource identifier or empty string.
		
	"""
	if not isinstance( option, str ) or not option.strip( ):
		return ''
	
	if ' — ' in option:
		return option.rsplit( ' — ', 1 )[ 1 ].strip( )
	
	return option.strip( )

def create_openai_vector_store( vectorstores: Any, name: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Create an OpenAI vector store through the GPT VectorStores wrapper.
		
		Parameters:
		-----------
		vectorstores: Any
			GPT VectorStores wrapper instance.
		
		name: str
			Vector store name.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized vector store metadata.
		
	"""
	if not isinstance( name, str ) or not name.strip( ):
		raise ValueError( 'Vector store name is required.' )
	
	try:
		result = vectorstores.create( name=name.strip( ) )
	except TypeError:
		result = vectorstores.create( name.strip( ) )
	
	return normalize_storage_object( result )

def list_openai_vector_stores( vectorstores: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		List OpenAI vector stores through the GPT VectorStores wrapper.
		
		Parameters:
		-----------
		vectorstores: Any
			GPT VectorStores wrapper instance.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Normalized vector store rows.
		
	"""
	if hasattr( vectorstores, 'list' ):
		result = vectorstores.list( )
	else:
		result = vectorstores.list_stores( )
	
	return normalize_storage_list( result )

def retrieve_openai_vector_store( vectorstores: Any, store_id: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Retrieve OpenAI vector store metadata.
		
		Parameters:
		-----------
		vectorstores: Any
			GPT VectorStores wrapper instance.
		
		store_id: str
			Vector store identifier.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized vector store metadata.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		raise ValueError( 'Vector store ID is required.' )
	
	try:
		result = vectorstores.retrieve( id=store_id.strip( ) )
	except TypeError:
		try:
			result = vectorstores.retrieve( vector_store_id=store_id.strip( ) )
		except TypeError:
			result = vectorstores.retrieve( store_id.strip( ) )
	
	return normalize_storage_object( result )

def delete_openai_vector_store( vectorstores: Any, store_id: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Delete an OpenAI vector store.
		
		Parameters:
		-----------
		vectorstores: Any
			GPT VectorStores wrapper instance.
		
		store_id: str
			Vector store identifier.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized delete result.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		raise ValueError( 'Vector store ID is required.' )
	
	try:
		result = vectorstores.delete( id=store_id.strip( ) )
	except TypeError:
		try:
			result = vectorstores.delete( vector_store_id=store_id.strip( ) )
		except TypeError:
			result = vectorstores.delete( store_id.strip( ) )
	
	return normalize_storage_object( result )

def attach_file_to_openai_vector_store( vectorstores: Any, store_id: str,
		file_id: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Attach an existing OpenAI file to an OpenAI vector store.
		
		Parameters:
		-----------
		vectorstores: Any
			GPT VectorStores wrapper instance.
		
		store_id: str
			Vector store identifier.
		
		file_id: str
			OpenAI file identifier.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized attachment metadata.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		raise ValueError( 'Vector store ID is required.' )
	
	if not isinstance( file_id, str ) or not file_id.strip( ):
		raise ValueError( 'File ID is required.' )
	
	for method_name in [ 'attach_file', 'add_file', 'create_file', 'upload_file' ]:
		if hasattr( vectorstores, method_name ):
			method = getattr( vectorstores, method_name )
			try:
				result = method( vector_store_id=store_id.strip( ), file_id=file_id.strip( ) )
			except TypeError:
				try:
					result = method( store_id.strip( ), file_id.strip( ) )
				except TypeError:
					continue
			
			return normalize_storage_object( result )
	
	raise AttributeError( 'VectorStores wrapper does not expose a file attachment method.' )

def create_gemini_file_search_store( filestore: Any, name: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Create a Gemini File Search Store.
		
		Parameters:
		-----------
		filestore: Any
			Gemini FileSearch wrapper instance.
		
		name: str
			File Search Store display name.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized store metadata.
		
	"""
	if not isinstance( name, str ) or not name.strip( ):
		raise ValueError( 'File Search Store name is required.' )
	
	apply_gemini_runtime_config( )
	for method_name in [ 'create', 'create_store', 'create_file_search_store' ]:
		if hasattr( filestore, method_name ):
			method = getattr( filestore, method_name )
			try:
				result = method( name=name.strip( ) )
			except TypeError:
				result = method( name.strip( ) )
			
			return normalize_storage_object( result )
	
	raise AttributeError( 'Gemini FileSearch wrapper does not expose a create method.' )

def list_gemini_file_search_stores( filestore: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		List Gemini File Search Stores.
		
		Parameters:
		-----------
		filestore: Any
			Gemini FileSearch wrapper instance.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Normalized store rows.
		
	"""
	apply_gemini_runtime_config( )
	for method_name in [ 'list', 'list_stores', 'list_file_search_stores' ]:
		if hasattr( filestore, method_name ):
			result = getattr( filestore, method_name )( )
			return normalize_storage_list( result )
	
	raise AttributeError( 'Gemini FileSearch wrapper does not expose a list method.' )

def retrieve_gemini_file_search_store( filestore: Any, store_id: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Retrieve Gemini File Search Store metadata.
		
		Parameters:
		-----------
		filestore: Any
			Gemini FileSearch wrapper instance.
		
		store_id: str
			File Search Store resource name or identifier.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized store metadata.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		raise ValueError( 'File Search Store ID or resource name is required.' )
	
	apply_gemini_runtime_config( )
	for method_name in [ 'retrieve', 'get', 'get_store', 'get_file_search_store' ]:
		if hasattr( filestore, method_name ):
			method = getattr( filestore, method_name )
			try:
				result = method( name=store_id.strip( ) )
			except TypeError:
				try:
					result = method( id=store_id.strip( ) )
				except TypeError:
					result = method( store_id.strip( ) )
			
			return normalize_storage_object( result )
	
	raise AttributeError( 'Gemini FileSearch wrapper does not expose a retrieve method.' )

def delete_gemini_file_search_store( filestore: Any, store_id: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Delete a Gemini File Search Store.
		
		Parameters:
		-----------
		filestore: Any
			Gemini FileSearch wrapper instance.
		
		store_id: str
			File Search Store resource name or identifier.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized delete result.
		
	"""
	if not isinstance( store_id, str ) or not store_id.strip( ):
		raise ValueError( 'File Search Store ID or resource name is required.' )
	
	apply_gemini_runtime_config( )
	for method_name in [ 'delete', 'delete_store', 'delete_file_search_store' ]:
		if hasattr( filestore, method_name ):
			method = getattr( filestore, method_name )
			try:
				result = method( name=store_id.strip( ) )
			except TypeError:
				try:
					result = method( id=store_id.strip( ) )
				except TypeError:
					result = method( store_id.strip( ) )
			
			return normalize_storage_object( result )
	
	raise AttributeError( 'Gemini FileSearch wrapper does not expose a delete method.' )

def create_google_cloud_bucket( buckets: Any, bucket_name: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Create a Google Cloud Storage bucket through the Gemini CloudBuckets wrapper.
		
		Parameters:
		-----------
		buckets: Any
			Gemini CloudBuckets wrapper instance.
		
		bucket_name: str
			Cloud bucket name.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized bucket metadata.
		
	"""
	if not isinstance( bucket_name, str ) or not bucket_name.strip( ):
		raise ValueError( 'Bucket name is required.' )
	
	apply_gemini_runtime_config( )
	for method_name in [ 'create', 'create_bucket' ]:
		if hasattr( buckets, method_name ):
			method = getattr( buckets, method_name )
			try:
				result = method( name=bucket_name.strip( ) )
			except TypeError:
				try:
					result = method( bucket_name=bucket_name.strip( ) )
				except TypeError:
					result = method( bucket_name.strip( ) )
			
			return normalize_storage_object( result )
	
	raise AttributeError( 'CloudBuckets wrapper does not expose a create method.' )

def list_google_cloud_buckets( buckets: Any ) -> List[ Dict[ str, Any ] ]:
	"""
	
		Purpose:
		--------
		List Google Cloud Storage buckets through the Gemini CloudBuckets wrapper.
		
		Parameters:
		-----------
		buckets: Any
			Gemini CloudBuckets wrapper instance.
		
		Returns:
		--------
		List[Dict[str, Any]]
			Normalized bucket rows.
		
	"""
	apply_gemini_runtime_config( )
	for method_name in [ 'list', 'list_buckets' ]:
		if hasattr( buckets, method_name ):
			result = getattr( buckets, method_name )( )
			return normalize_storage_list( result )
	
	raise AttributeError( 'CloudBuckets wrapper does not expose a list method.' )

def upload_to_google_cloud_bucket( buckets: Any, bucket_name: str,
		object_name: str, path: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Upload a local file to a Google Cloud Storage bucket.
		
		Parameters:
		-----------
		buckets: Any
			Gemini CloudBuckets wrapper instance.
		
		bucket_name: str
			Bucket name.
		
		object_name: str
			Object name to create in the bucket.
		
		path: str
			Local file path.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized upload metadata.
		
	"""
	if not isinstance( bucket_name, str ) or not bucket_name.strip( ):
		raise ValueError( 'Bucket name is required.' )
	
	if not isinstance( object_name, str ) or not object_name.strip( ):
		raise ValueError( 'Object name is required.' )
	
	if not isinstance( path, str ) or not path.strip( ):
		raise ValueError( 'Upload path is required.' )
	
	apply_gemini_runtime_config( )
	for method_name in [ 'upload', 'upload_file', 'upload_blob' ]:
		if hasattr( buckets, method_name ):
			method = getattr( buckets, method_name )
			try:
				result = method(
					bucket_name=bucket_name.strip( ),
					object_name=object_name.strip( ),
					path=path.strip( ) )
			except TypeError:
				try:
					result = method( bucket_name.strip( ), object_name.strip( ), path.strip( ) )
				except TypeError:
					result = method( bucket_name.strip( ), path.strip( ) )
			
			return normalize_storage_object( result )
	
	raise AttributeError( 'CloudBuckets wrapper does not expose an upload method.' )

def delete_google_cloud_bucket_object( buckets: Any, bucket_name: str,
		object_name: str ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Delete an object from a Google Cloud Storage bucket.
		
		Parameters:
		-----------
		buckets: Any
			Gemini CloudBuckets wrapper instance.
		
		bucket_name: str
			Bucket name.
		
		object_name: str
			Object name.
		
		Returns:
		--------
		Dict[str, Any]
			Normalized delete result.
		
	"""
	if not isinstance( bucket_name, str ) or not bucket_name.strip( ):
		raise ValueError( 'Bucket name is required.' )
	
	if not isinstance( object_name, str ) or not object_name.strip( ):
		raise ValueError( 'Object name is required.' )
	
	apply_gemini_runtime_config( )
	for method_name in [ 'delete_object', 'delete_blob', 'delete_file' ]:
		if hasattr( buckets, method_name ):
			method = getattr( buckets, method_name )
			try:
				result = method(
					bucket_name=bucket_name.strip( ),
					object_name=object_name.strip( ) )
			except TypeError:
				result = method( bucket_name.strip( ), object_name.strip( ) )
			
			return normalize_storage_object( result )
	
	raise AttributeError( 'CloudBuckets wrapper does not expose an object delete method.' )

# ======================================================================================
# VECTOR STORES RETRIEVAL HOOK UTILITIES
# ======================================================================================

def get_retrieval_backend( provider_name: Optional[ str ] = None ) -> Dict[ str, Any ]:
	"""
	
		Purpose:
		--------
		Return the active retrieval backend selected in Buddy's visible Vector Stores mode.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		Dict[str, Any]
			Provider-safe backend metadata used by Text and Document Q&A modes.
		
	"""
	provider = get_provider_name( provider_name )
	backend = ''
	resource_id = ''
	is_retrieval_ready = False
	
	if provider == 'GPT':
		backend = 'OpenAI Vector Stores'
		resource_id = get_selected_store_id(
			manual_key='stores_manual_id',
			selected_key='stores_selected_id',
			fallback_key='stores_id' )
		is_retrieval_ready = bool( resource_id )
	
	elif provider == 'Grok':
		backend = 'xAI Collections'
		resource_id = get_selected_store_id(
			manual_key='stores_manual_id',
			selected_key='stores_selected_id',
			fallback_key='stores_id' )
		is_retrieval_ready = bool( resource_id )
	
	elif provider == 'Gemini':
		selected_backend = get_gemini_vector_backend( )
		backend = f'Gemini {selected_backend}'
		
		if selected_backend == 'File Search Stores':
			resource_id = get_selected_store_id(
				manual_key='filestore_id',
				selected_key='filestore_selected_id',
				fallback_key='filestore_id' )
			is_retrieval_ready = bool( resource_id )
		else:
			resource_id = get_selected_store_id(
				manual_key='bucket_name',
				selected_key='bucket_selected_id',
				fallback_key='bucket_name' )
			is_retrieval_ready = False
	
	return {
			'provider': provider,
			'backend': backend,
			'resource_id': resource_id,
			'is_retrieval_ready': is_retrieval_ready,
			'gemini_backend': get_gemini_vector_backend( )
			if provider == 'Gemini' else '',
	}

def get_store_ids( provider_name: Optional[ str ] = None ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return active OpenAI Vector Store IDs selected through Buddy's Vector Stores mode.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		List[str]
			OpenAI vector store IDs. Empty for non-GPT providers.
		
	"""
	provider = get_provider_name( provider_name )
	
	if provider != 'GPT':
		return [ ]
	
	backend = get_retrieval_backend( provider )
	resource_id = backend.get( 'resource_id', '' )
	
	return parse_storage_ids( resource_id )

def get_active_grok_collection_ids( provider_name: Optional[ str ] = None ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return active xAI Collection IDs selected through Buddy's Vector Stores mode.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		List[str]
			xAI collection IDs. Empty for non-Grok providers.
		
	"""
	provider = get_provider_name( provider_name )
	
	if provider != 'Grok':
		return [ ]
	
	backend = get_retrieval_backend( provider )
	resource_id = backend.get( 'resource_id', '' )
	
	return parse_storage_ids( resource_id )

def get_active_gemini_file_search_store_names( provider_name: Optional[ str ]=None ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Return active Gemini File Search Store resource names selected through Buddy's
		Vector Stores mode.
		
		Important:
		----------
		Gemini Cloud Bucket names are not returned here because Cloud Buckets are storage
		objects, not File Search Store resources.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		List[str]
			Gemini File Search Store resource names.
		
	"""
	provider = get_provider_name( provider_name )
	
	if provider != 'Gemini':
		return [ ]
	
	if get_gemini_vector_backend( ) != 'File Search Stores':
		return [ ]
	
	backend = get_retrieval_backend( provider )
	resource_id = backend.get( 'resource_id', '' )
	
	return parse_storage_ids( resource_id )

def merge_unique_strings( primary: List[ str ], secondary: List[ str ] ) -> List[ str ]:
	"""
	
		Purpose:
		--------
		Merge two string lists while preserving order and removing duplicates.
		
		Parameters:
		-----------
		primary: List[str]
			Primary string list.
		
		secondary: List[str]
			Secondary string list.
		
		Returns:
		--------
		List[str]
			Merged unique string list.
		
	"""
	merged: List[ str ] = [ ]
	
	for source in [ primary, secondary ]:
		if not isinstance( source, list ):
			continue
		
		for item in source:
			text = str( item or '' ).strip( )
			
			if text and text not in merged:
				merged.append( text )
	
	return merged

def build_provider_retrieval_summary( provider_name: Optional[ str ] = None ) -> str:
	"""
	
		Purpose:
		--------
		Build a compact retrieval summary for Text and Document Q&A diagnostics.
		
		Parameters:
		-----------
		provider_name: Optional[str]
			Optional explicit provider name.
		
		Returns:
		--------
		str
			User-facing retrieval summary.
		
	"""
	backend = get_retrieval_backend( provider_name )
	provider = backend.get( 'provider', '' )
	backend_name = backend.get( 'backend', '' )
	resource_id = backend.get( 'resource_id', '' )
	ready = backend.get( 'is_retrieval_ready', False )
	
	if not resource_id:
		return f'{provider}: no Vector Stores resource selected.'
	
	if ready:
		return f'{provider}: using {backend_name} resource {resource_id}.'
	
	return f'{provider}: selected {backend_name} resource {resource_id}, but it is not a retrieval store.'

# ---------------- TEXT ----------------
def text_model_options( chat ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'model_options', chat.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'model_options', chat.model_options )
	if _provider( ) == 'Grok':
		return _safe( 'grok', 'model_options', chat.model_options )
	return chat.model_options

# ---------------- IMAGES ----------------
def image_model_options( image ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'image_model_options', image.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'image_model_options', image.model_options )
	if _provider( ) == 'Grok':
		return _safe( 'grok', 'model_options', image.model_options )
	return image.model_options

def image_size_or_aspect_options( image ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'aspect_options', image.size_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'aspect_options', image.size_options )
	if _provider( ) == 'Grok':
		return _safe( 'grok', 'model_options', image.size_options )
	return image.size_options

# ---------------- AUDIO ----------------
def audio_model_options( transcriber ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'audio_model_options', transcriber.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'audio_model_options', transcriber.model_options )
	return transcriber.model_options

def audio_language_options( transcriber ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'language_options', transcriber.language_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'language_options', transcriber.language_options )
	return transcriber.language_options

# ---------------- EMBEDDINGS ----------------
def embedding_model_options( embed ):
	if _provider( ) == 'GPT':
		return _safe( 'gpt', 'embedding_model_options', embed.model_options )
	if _provider( ) == 'Gemini':
		return _safe( 'gemini', 'embedding_model_options', embed.model_options )
	return embed.model_options

# -------------DOC Q&A ----------------------
def route_document_query( prompt: str ) -> str:
	source = st.session_state.get( 'doc_source' )
	active_docs = st.session_state.get( 'docqna_active_docs', [ ] )
	doc_bytes = st.session_state.get( 'doc_bytes', { } )
	instructions = st.session_state.get( 'docqna_system_instructions' )
	
	if not source:
		return 'No document source selected.'
	
	if not active_docs:
		return 'No document selected.'
	
	# --------------------------------------------------
	# LOCAL DOCUMENT → Chat (single or multi)
	# --------------------------------------------------
	if source == 'uploadlocal':
		chat = get_chat_module( )
		
		# Single document
		if len( active_docs ) == 1:
			name = active_docs[ 0 ]
			file_bytes = doc_bytes.get( name )
			
			if not file_bytes:
				return 'Document content not available.'
			
			text = extract_text_from_bytes( file_bytes )
			
			full_prompt = f"""
				{instructions}
				
				Use the following document to answer the question.
				Be precise and cite relevant portions when possible.
				
				DOCUMENT:
				{text}
				
				QUESTION:
				{prompt}
				"""
			return chat.generate_text( prompt=full_prompt )
		
		# Multi-document injection
		combined_text = ""
		
		for name in active_docs:
			file_bytes = doc_bytes.get( name )
			if not file_bytes:
				continue
			
			text = extract_text_from_bytes( file_bytes )
			
			combined_text += f"\n\n===== DOCUMENT: {name} =====\n\n{text}\n"
		
		if not combined_text.strip( ):
			return 'No readable document content available.'
		
		full_prompt = f"""
			{instructions}
			
			You are analyzing multiple documents.
			
			Use the content below to answer the question.
			If multiple documents are relevant, compare them.
			Cite document names when possible.
			
			DOCUMENT SET:
			{combined_text}
			
			QUESTION:
			{prompt}
			"""
		
		return chat.generate_text( prompt=full_prompt )
	
	# --------------------------------------------------
	# FILES API → Files class
	# --------------------------------------------------
	if source == "filesapi":
		files = get_files_module( )
		
		# Single file search
		if len( active_docs ) == 1:
			return files.search( prompt, active_docs[ 0 ] )
		
		# Multi-file survey
		return files.survey( prompt )
	
	# --------------------------------------------------
	# VECTOR STORE → VectorStores class
	# --------------------------------------------------
	if source == 'vectorstore':
		vectorstores = get_vectorstores_module( )
		
		# Single store
		if len( active_docs ) == 1:
			return vectorstores.search( prompt, active_docs[ 0 ] )
		
		# Multi-store aggregation
		responses = [ ]
		for store_id in active_docs:
			result = vectorstores.search( prompt, store_id )
			if result:
				responses.append( result )
		
		if not responses:
			return 'No results found across selected vector stores.'
		
		return "\n\n".join( responses )
	
	return 'Unsupported document source.'

def extract_text_from_bytes( file_bytes: bytes ) -> str:
	"""
		Extracts text from PDF or text-based documents.
	"""
	try:
		import fitz  # PyMuPDF		
		doc = fitz.open( stream=file_bytes, filetype="pdf" )
		text = ""
		for page in doc:
			text += page.get_text( )
		return text.strip( )
	
	except Exception:
		try:
			return file_bytes.decode( errors="ignore" )
		except Exception:
			return ""

def summarize_document( ) -> str:
	"""
		Uses the routing layer to summarize the currently active document.
	"""
	doc_instructions = st.session_state.get( "doc_instructions", "" )
	summary_prompt = """
		Provide a clear, structured summary of this document.
		Include:
		- Purpose
		- Key themes
		- Major conclusions
		- Important data points (if any)
		- Policy implications (if applicable)
		
		Be precise and concise.
		"""
	if doc_instructions:
		summary_prompt = f"{doc_instructions}\n\n{summary_prompt}"
	
	return route_document_query( summary_prompt.strip( ) )

# ==============================================================================
# Page Setup
# ==============================================================================

initialize_database( )
embedder = load_embedder( )
AVATARS = { 'user': cfg.ANALYST, 'assistant': cfg.BUDDY, }
st.set_page_config( page_title=cfg.APP_TITLE, layout='wide', page_icon=cfg.FAVICON, 
	initial_sidebar_state='collapsed' )

st.caption( cfg.APP_SUBTITLE )
inject_response_css( )
init_state( )

# ======================================================================================
# Sidebar
# ======================================================================================
with st.sidebar:
	provider = st.session_state.get( 'provider'  )
	style_subheaders( )
	st.subheader( '' )
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	provider = st.selectbox( 'Select API', list( cfg.PROVIDERS.keys( ) ),
		index=list( cfg.PROVIDERS.keys( ) ).index( st.session_state.get( 'provider', 'GPT' ) ) )
	
	st.session_state[ 'provider' ] = provider
	logo_path = cfg.LOGO_MAP.get( provider )
	if logo_path and os.path.exists( logo_path ):
		logo_path = cfg.LOGO_MAP.get( provider )
		st.logo( logo_path, size='large', link=cfg.CRS )
	
	#-----API KEY Expander------------------------------
	with st.expander( label='Keys:', icon='🔑', expanded=False ):
		openai_key = st.text_input( 'OpenAI API Key', type='password',
			value=st.session_state.get( 'openai_api_key', cfg.OPENAI_API_KEY ),
			help='Overrides OPENAI_API_KEY from config.py for this session only.',
			key='sidebar_openai_api_key' )
		
		gemini_key = st.text_input( 'Gemini API Key', type='password',
			value=st.session_state.get( 'gemini_api_key', cfg.GEMINI_API_KEY ),
			help='Overrides GEMINI_API_KEY from config.py for this session only.',
			key='sidebar_gemini_api_key' )
		
		xai_key = st.text_input( 'xAI API Key', type='password',
			value=st.session_state.get( 'xai_api_key', cfg.XAI_API_KEY ),
			help='Overrides XAI_API_KEY from config.py for this session only.',
			key='sidebar_xai_api_key' )
		
		google_key = st.text_input( 'Google API Key', type='password',
			value=st.session_state.get( 'google_api_key', cfg.GOOGLE_API_KEY ),
			help='Overrides GOOGLE_API_KEY from config.py for this session only.',
			key='sidebar_google_api_key' )
		
		google_cse_id = st.text_input( 'Google CSE ID', type='password',
			value=st.session_state.get( 'google_cse_id', cfg.GOOGLE_CSE_ID ),
			help='Overrides GOOGLE_CSE_ID from config.py for this session only.',
			key='sidebar_google_cse_id' )
		
		google_cloud_project_id = st.text_input( 'Google Cloud Project ID', type='password',
			value=st.session_state.get( 'google_cloud_project_id', cfg.GOOGLE_CLOUD_PROJECT_ID ),
			help='Overrides GOOGLE_CLOUD_PROJECT_ID from config.py for this session only.',
			key='sidebar_google_cloud_project_id' )
		
		google_cloud_location = st.text_input( 'Google Cloud Location', type='password',
			value=st.session_state.get( 'google_cloud_location', cfg.GOOGLE_CLOUD_LOCATION ),
			help='Overrides GOOGLE_CLOUD_LOCATION from config.py for this session only.',
			key='sidebar_google_cloud_location' )
		
		if openai_key:
			st.session_state[ 'openai_api_key' ] = openai_key
			os.environ[ 'OPENAI_API_KEY' ] = openai_key
		
		if gemini_key:
			st.session_state[ 'gemini_api_key' ] = gemini_key
			os.environ[ 'GEMINI_API_KEY' ] = gemini_key
		
		if xai_key:
			st.session_state[ 'xai_api_key' ] = xai_key
			os.environ[ 'XAI_API_KEY' ] = xai_key
		
		if google_key:
			st.session_state[ 'google_api_key' ] = google_key
			os.environ[ 'GOOGLE_API_KEY' ] = google_key
		
		if google_cse_id:
			st.session_state[ 'google_cse_id' ] = google_cse_id
			os.environ[ 'GOOGLE_CSE_ID' ] = google_cse_id
		
		if google_cloud_project_id:
			st.session_state[ 'google_cloud_project_id' ] = google_cloud_project_id
			os.environ[ 'GOOGLE_CLOUD_PROJECT_ID' ] = google_cloud_project_id
		
		if google_cloud_location:
			st.session_state[ 'google_cloud_location' ] = google_cloud_location
			os.environ[ 'GOOGLE_CLOUD_LOCATION' ] = google_cloud_location
	
	if 'provider' not in st.session_state or st.session_state[ 'provider' ] is None:
		st.session_state[ 'provider' ] = 'GPT'
	
	if 'mode' not in st.session_state or st.session_state[ 'mode' ] is None:
		st.session_state[ 'mode' ] = 'Text'
	
	if 'messages' not in st.session_state:
		st.session_state.messages: List[ Dict[ str, Any ] ] = [ ]
	
	if 'last_call_usage' not in st.session_state:
		st.session_state.last_call_usage = { 'prompt_tokens': 0, 'completion_tokens': 0,
				'total_tokens': 0, }
	
	if 'token_usage' not in st.session_state:
		st.session_state.token_usage = { 'prompt_tokens': 0, 'completion_tokens': 0,
				'total_tokens': 0, }
	
	if 'files' not in st.session_state:
		st.session_state.files: List[ str ] = [ ]
		
		if st.button( 'Clear Chat' ):
			reset_state( )
			st.rerun( )
		
	st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
	
	if provider == 'Gemini':
		mode = st.sidebar.radio( 'Select Mode', cfg.GEMINI_MODES, index=0 )
	elif provider == 'Grok':
		mode = st.sidebar.radio( 'Select Mode', cfg.GROK_MODES, index=0 )
	else:
		mode = st.sidebar.radio( 'Select Mode', cfg.GPT_MODES, index=0 )

# ======================================================================================
# CHAT MODE
# ======================================================================================
if mode == 'Chat':
	st.subheader( "💬 Chat Completions", help=cfg.CHAT_COMPLETIONS )
	st.divider( )
	provider_module = get_provider_module( )
	provider_name = st.session_state.get( 'provider', 'GPT' )
	chat_number = st.session_state.get( 'number', 0 )
	chat_top_p = st.session_state.get( 'top_percent', 0.0 )
	chat_freq = st.session_state.get( 'frequency_penalty', 0.0 )
	chat_presense = st.session_state.get( 'presense_penalty', 0.0 )
	chat_temperature = st.session_state.get( 'temperature', 0.0 )
	chat_background = st.session_state.get( 'background', False )
	chat_stream = st.session_state.get( 'stream', False )
	chat_store = st.session_state.get( 'store', False )
	chat_model = st.session_state.get( 'chat_model', '' )
	chat_format = st.session_state.get( 'response_format', '' )
	chat_input = st.session_state.get( 'input', [ ] )
	chat_reasoning = st.session_state.get( 'reasoning', '' )
	chat_choice = st.session_state.get( 'tool_choice', '' )
	chat_parallel = st.session_state.get( 'parallel_tools', False )
	chat_messages = st.session_state.get( 'messages', [ ] )
	chat_history = st.session_state.get( 'chat_history', [ ] )
	st.session_state.setdefault( 'chat_previous_response_id', '' )
	_modes = [ 'Standard', 'Guidance Only', 'Analysis Only' ]
	_current_mode = st.session_state.get( 'execution_mode', 'Standard' )
	
	if _current_mode not in _modes:
		_current_mode = 'Standard'
		st.session_state.execution_mode = 'Standard'
	
	# ------------------------------------------------------------------
	# Sidebar — Chat Settings
	# ------------------------------------------------------------------
	with st.sidebar:
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		st.text( '⚙️  Chat Settings' )
		st.radio( 'Execution Mode', options=_modes, index=_modes.index( _current_mode ),
			key='execution_mode', )
	
	execution_mode = st.session_state.get( 'execution_mode', 'Standard' )
	intent_prefix = build_intent_prefix( execution_mode )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.25, 3.5, 0.25 ] )
	with center:
		user_input = st.chat_input( 'Have a Planning, Programming, or Budget Execution question?' )
		if user_input:
			with st.chat_message( 'user', avatar=cfg.ANALYST ):
				st.markdown( user_input )
			
			with st.chat_message( 'assistant', avatar=cfg.BUDDY ):
				try:
					chat = get_chat_module( )
					effective_input = f'{intent_prefix}{user_input}' if intent_prefix else user_input
					
					with st.spinner( 'Running prompt...' ):
						if provider_name == 'GPT':
							response = chat.completion(
								prompt_id=cfg.PROMPT_ID,
								prompt_version=cfg.PROMPT_VERSION,
								model=chat_model or None,
								user_input=effective_input,
								temperature=chat_temperature,
								format=chat_format if isinstance( chat_format, dict ) else None,
								top_p=chat_top_p,
								frequency=chat_freq,
								presence=chat_presense,
								max_tokens=st.session_state.get( 'max_tokens', 0 ) or None,
								store=chat_store,
								stream=chat_stream,
								instruct=st.session_state.get( 'chat_system_instructions', '' ) or None,
								background=chat_background,
								reasoning=chat_reasoning or None,
								include=[ 'web_search_call.action.sources',
								          'code_interpreter_call.outputs',
										'file_search_call.results', ],
								tools=[
										{
											'type': 'file_search',
											'vector_store_ids': cfg.GPT_VECTORSTORES,
										},
										{
											'type': 'web_search',
											'filters': { 'allowed_domains': cfg.GPT_DOMAINS, },
											'search_context_size': 'medium',
											'user_location': { 'type': 'approximate' },
										},
										{
											'type': 'code_interpreter',
											'container': { 'type': 'auto', 'file_ids': cfg.GPT_FILES, },
										},
								],
								tool_choice=chat_choice or None,
								is_parallel=chat_parallel,
								previous_id=st.session_state.get( 'chat_previous_response_id', '' ) or None,
							)
							st.session_state.chat_previous_response_id = getattr( response, 'id', '' ) or ''
						else:
							output_text = chat.generate_text(
								prompt=effective_input,
								model=chat_model,
								temperature=chat_temperature,
								format=chat_format if isinstance( chat_format, dict ) else None,
								top_p=chat_top_p,
								frequency=chat_freq,
								presence=chat_presense,
								max_tokens=st.session_state.get( 'max_tokens', 0 ) or None,
								store=chat_store,
								stream=chat_stream,
								instruct=st.session_state.get( 'chat_system_instructions', '' ) or None,
								background=chat_background,
								reasoning=chat_reasoning or None,
								tool_choice=chat_choice or None,
								is_parallel=chat_parallel,
							)
							response = None
					
					sources = [ ]
					analysis = { 'tables': [ ], 'files': [ ], 'text': [ ] }
					
					if response is not None:
						try:
							for item in getattr( response, 'output', [ ] ):
								item_type = getattr( item, 'type', '' )
								
								if item_type == 'web_search_call':
									action = getattr( item, 'action', None )
									raw_sources = getattr( action, 'sources', None )
									if raw_sources:
										for src in raw_sources:
											sources.append(
												{
														'type': 'web',
														'url': getattr( src, 'url', None ),
														'title': getattr( src, 'title', None ),
														'file_id': None,
														'file_name': None,
														'snippet': getattr( src, 'snippet', None ),
												}
											)
								
								elif item_type == 'file_search_call':
									results = getattr( item, 'results', None )
									if results:
										for result in results:
											sources.append(
												{
														'type': 'file',
														'url': None,
														'title': getattr( result, 'file_name', None )
														         or getattr( result, 'title', None ),
														'file_id': getattr( result, 'file_id', None )
														           or getattr( result, 'id', None ),
														'file_name': getattr( result, 'file_name', None ),
														'snippet': getattr( result, 'text', None ),
												}
											)
								
								elif item_type == 'code_interpreter_call':
									outputs = getattr( item, 'outputs', None )
									if outputs:
										for out in outputs:
											out_type = getattr( out, 'type', None )
											if out_type == 'table':
												analysis[ 'tables' ].append( normalize( out ) )
											elif out_type == 'file':
												analysis[ 'files' ].append( normalize( out ) )
											elif out_type in ('output_text', 'text'):
												text = getattr( out, 'text', None )
												if isinstance( text, str ) and text.strip( ):
													analysis[ 'text' ].append( text )
						except Exception:
							sources = [ ]
							analysis = { 'tables': [ ], 'files': [ ], 'text': [ ] }
					
					st.session_state.last_sources = sources
					st.session_state.last_analysis = analysis
					
					if response is not None:
						output_text = extract_response_text( response )
					else:
						output_text = output_text if isinstance( output_text, str ) else ''
					
					if output_text.strip( ):
						st.markdown( output_text )
					else:
						st.warning( 'No text response returned by the prompt.' )
					
					if analysis.get( 'text' ):
						with st.expander( 'Analysis Output', expanded=False ):
							for block in analysis.get( 'text', [ ] ):
								st.markdown( block )
					
					if sources:
						st.markdown( '#### Sources' )
						for i, src in enumerate( sources, 1 ):
							url = src.get( 'url' )
							title = src.get( 'title' ) or src.get( 'file_name' ) or f'Source {i}'
							
							if url:
								st.markdown( f'- [{title}]({url})' )
							elif src.get( 'file_id' ):
								st.markdown( f"- {title} _(File ID: `{src[ 'file_id' ]}`)_" )
							else:
								st.markdown( f'- {title}' )
					
					st.session_state.chat_history.append(
						{
								'role': 'user',
								'content': user_input,
						}
					)
					
					st.session_state.chat_history.append(
						{
								'role': 'assistant',
								'content': output_text,
						}
					)
					
					if response is not None:
						try:
							update_token_counters( response )
						except Exception:
							pass
				except Exception as e:
					st.error( 'An error occurred while running the prompt.' )
					st.exception( e )

# ======================================================================================
# TEXT MODE
# ======================================================================================
elif mode == 'Text':
	ensure_text_mode_state( )
	
	provider_name = get_provider_name( )
	text = get_chat_module( )
	text_avatar = get_text_avatar( provider_name )
	
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '💬 Text Generation', help=cfg.TEXT_GENERATION )
		st.divider( )
		if st.session_state.get( 'clear_instructions' ):
			st.session_state[ 'text_system_instructions' ] = ''
			st.session_state[ 'instructions_last_loaded' ] = ''
			st.session_state[ 'clear_instructions' ] = False
		
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# Model Settings
			# ------------------------------------------------------------------
			with st.expander( label='LLM Settings', icon='🧊', expanded=False, width='stretch' ):
				llm_c1, llm_c2, llm_c3, llm_c4, llm_c5, llm_c6 = st.columns(
					[ 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 ], border=True, gap='xxsmall' )
				
				with llm_c1:
					model_options = get_text_option_list( text, 'model_options', [ '' ] )
					st.selectbox( label='Select Model', options=model_options,
						key='text_model', placeholder='Options', index=None,
						help='REQUIRED. Text Generation model used by the AI' )
				
				with llm_c2:
					reasoning_options = get_text_option_list( text, 'reasoning_options', [ '' ] )
					st.selectbox( label='Reasoning', options=reasoning_options,
						key='text_reasoning', help=cfg.REASONING, index=None,
						placeholder='Options' )
				
				with llm_c3:
					if provider_name == 'Gemini':
						st.slider( label='Candidates', min_value=0, max_value=50,
							value=int( st.session_state.get( 'text_number', 0 ) ), step=1,
							help='Optional. Upper limit on responses returned by the model.',
							key='text_number' )
					else:
						st.slider( label='Number', min_value=0, max_value=10,
							value=int( st.session_state.get( 'text_number', 0 ) ), step=1,
							help='Optional. Upper limit on responses returned by the model.',
							key='text_number' )
				
				with llm_c4:
					if provider_name == 'Gemini':
						media_options = get_text_option_list( text, 'media_options', [ '' ] )
						st.selectbox( label='Media Resolution', options=media_options,
							key='text_media_resolution',
							help='Optional. Requested media resolution.',
							index=None, placeholder='Options' )
					else:
						input_options = [ 'single_turn', 'response_chain', 'conversation' ]
						st.selectbox( label='Input Mode', options=input_options,
							key='text_input',
							help='Optional. Controls stateful API input behavior.',
							index=None, placeholder='Options' )
				
				with llm_c5:
					format_options = get_text_option_list( text, 'format_options',
						[ 'text', 'json_object', 'json_schema' ] )
					st.selectbox( label='Response Format', options=format_options,
						key='text_response_format', help='Optional. Desired response format.',
						index=None, placeholder='Options' )
				
				with llm_c6:
					st.button( label='Reset', key='text_model_reset', width='stretch',
						on_click=reset_text_model_settings )
			
			# ------------------------------------------------------------------
			# Inference Settings
			# ------------------------------------------------------------------
			with st.expander( label='Inference Settings', icon='🎚️', expanded=False,
					width='stretch' ):
				prm_c1, prm_c2, prm_c3, prm_c4, prm_c5, prm_c6 = st.columns(
					[ 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 ], border=True,
					gap='xxsmall' )
				
				with prm_c1:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0,
						value=float( st.session_state.get( 'text_top_percent', 0.0 ) ),
						step=0.01, help=cfg.TOP_P, key='text_top_percent' )
				
				with prm_c2:
					st.slider( label='Temperature', min_value=-2.0, max_value=2.0,
						value=float( st.session_state.get( 'text_temperature', 0.0 ) ),
						step=0.01, help=cfg.TEMPERATURE, key='text_temperature' )
				
				with prm_c3:
					st.slider( label='Frequency Penalty', min_value=-2.0, max_value=2.0,
						value=float( st.session_state.get( 'text_frequency_penalty', 0.0 ) ),
						step=0.01, help=cfg.FREQUENCY_PENALTY, key='text_frequency_penalty' )
				
				with prm_c4:
					st.slider( label='Presence Penalty', min_value=-2.0, max_value=2.0,
						value=float( st.session_state.get( 'text_presence_penalty', 0.0 ) ),
						step=0.01, help=cfg.PRESENCE_PENALTY, key='text_presence_penalty' )
				
				with prm_c5:
					if provider_name == 'Gemini':
						st.slider( label='Top-K', min_value=0, max_value=100,
							value=int( st.session_state.get( 'text_top_k', 0 ) ), step=1,
							help=cfg.TOP_K, key='text_top_k' )
					else:
						st.caption( 'Top-K is only used by Gemini.' )
				
				with prm_c6:
					st.button( label='Reset', key='text_inference_reset', width='stretch',
						on_click=reset_text_inference_settings )
			
			# ------------------------------------------------------------------
			# Tool Settings
			# ------------------------------------------------------------------
			with st.expander( label='Tool Settings', icon='🛠️', expanded=False, width='stretch' ):
				tool_c1, tool_c2, tool_c3, tool_c4, tool_c5, tool_c6 = st.columns(
					[ 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 ], border=True,
					gap='xxsmall' )
				
				with tool_c1:
					if provider_name == 'Gemini':
						st.toggle( label='Google Grounding', key='text_google_grounding',
							help='Enable Google Search grounding for supported Gemini models.' )
					else:
						st.toggle( label='Parallel Tools', key='text_parallel_tools',
							help=cfg.PARALLEL_TOOL_CALLS )
				
				with tool_c2:
					st.slider( label='Max Tool Calls', min_value=0, max_value=10,
						value=int( st.session_state.get( 'text_max_calls', 0 ) ), step=1,
						help=cfg.MAX_TOOL_CALLS, key='text_max_calls' )
				
				with tool_c3:
					if provider_name == 'Gemini':
						st.slider( label='Max URLs', min_value=0, max_value=25,
							value=int( st.session_state.get( 'text_max_urls', 0 ) ), step=1,
							help='Optional. Maximum URL-context items.',
							key='text_max_urls' )
					else:
						st.slider( label='Max Searches', min_value=0, max_value=30,
							value=int( st.session_state.get( 'text_max_searches', 0 ) ), step=1,
							help='Optional. Maximum web search result count.',
							key='text_max_searches' )
				
				with tool_c4:
					tool_options = get_text_option_list( text, 'tool_options', [ ] )
					st.multiselect( label='Tools', options=tool_options, key='text_tools',
						help=cfg.TOOLS, placeholder='Options' )
				
				with tool_c5:
					choice_options = get_text_option_list( text, 'choice_options',
						[ 'auto', 'required', 'none' ] )
					st.selectbox( label='Tool Choice', options=choice_options,
						key='text_tool_choice', help=cfg.CHOICE, index=None,
						placeholder='Options' )
				
				with tool_c6:
					st.button( label='Reset', key='text_tools_reset', width='stretch',
						on_click=reset_text_tool_settings )
				
				if provider_name == 'GPT':
					gpt_tool_c1, gpt_tool_c2 = st.columns( [ 0.5, 0.5 ], border=True,
						gap='xxsmall' )
					
					with gpt_tool_c1:
						st.text_input( label='Allowed Websites', key='text_domains_input',
							value=','.join( st.session_state.get( 'text_domains', [ ] ) ),
							help='Optional. Comma-delimited domains for web search.',
							width='stretch', placeholder='gao.gov,omb.gov,congress.gov' )
					
					with gpt_tool_c2:
						st.text_input( label='Vector Store IDs', key='text_vector_store_ids',
							help='Optional. Comma-delimited OpenAI vector store IDs for file_search.',
							width='stretch', placeholder='vs_...' )
				
				if provider_name == 'Gemini':
					gemini_tool_c1, gemini_tool_c2 = st.columns( [ 0.5, 0.5 ], border=True,
						gap='xxsmall' )
					
					with gemini_tool_c1:
						st.text_input( label='URLs', key='text_urls_input',
							value=';'.join( st.session_state.get( 'text_urls', [ ] ) ),
							help='Optional. Semicolon-delimited URLs for Gemini URL context.',
							width='stretch',
							placeholder='https://example.com/page-1;https://example.com/page-2' )
					
					with gemini_tool_c2:
						st.text_input( label='File Search Store Names',
							key='text_file_search_store_names_input',
							value=','.join(
								st.session_state.get( 'text_file_search_store_names', [ ] ) ),
							help='Optional. Comma-delimited Gemini File Search Store resource names.',
							width='stretch' )
			
			# ------------------------------------------------------------------
			# Response Settings
			# ------------------------------------------------------------------
			with st.expander( label='Response Settings', icon='↔️', expanded=False, width='stretch' ):
				
				resp_c1, resp_c2, resp_c3, resp_c4, resp_c5, resp_c6 = st.columns(
					[ 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 ], border=True,
					gap='xxsmall' )
				
				with resp_c1:
					st.slider( label='Max Tokens', min_value=0, max_value=100000,
						value=int( st.session_state.get( 'text_max_tokens', 0 ) ),
						step=500, help=cfg.MAX_OUTPUT_TOKENS, key='text_max_tokens' )
				
				with resp_c2:
					st.toggle( label='Stream', key='text_stream', help=cfg.STREAM )
				
				with resp_c3:
					st.toggle( label='Store', key='text_store', help=cfg.STORE )
				
				with resp_c4:
					st.toggle( label='Background', key='text_background',
						help=cfg.BACKGROUND_MODE )
				
				with resp_c5:
					st.text_input( label='Stop Sequences', key='text_stops_input',
						value=','.join( st.session_state.get( 'text_stops', [ ] ) ),
						help=cfg.STOP_SEQUENCE, width='stretch',
						placeholder='Enter stop strings separated by commas' )
				
				with resp_c6:
					st.button( label='Reset', key='text_response_reset', width='stretch',
						on_click=reset_text_response_settings )
				
				if provider_name == 'GPT' and st.session_state.get(
						'text_response_format' ) == 'json_schema':
					schema_c1, schema_c2, schema_c3 = st.columns( [ 0.25, 0.55, 0.20 ],
						border=True, gap='xxsmall' )
					
					with schema_c1:
						st.text_input( label='Schema Name', key='text_json_schema_name',
							width='stretch' )
					
					with schema_c2:
						st.text_area( label='JSON Schema', key='text_json_schema',
							height=100, width='stretch' )
					
					with schema_c3:
						st.toggle( label='Strict', key='text_json_schema_strict' )
				
				if provider_name == 'Gemini':
					gemini_resp_c1, gemini_resp_c2 = st.columns( [ 0.5, 0.5 ],
						border=True, gap='xxsmall' )
					
					with gemini_resp_c1:
						st.text_area( label='Response Schema', key='text_response_schema',
							height=100, width='stretch',
							help='Optional. JSON schema used when Gemini response format supports it.' )
					
					with gemini_resp_c2:
						safety_options = get_text_option_list( text, 'safety_options', [ '' ] )
						st.selectbox( label='Safety', options=safety_options,
							key='text_safety_profile',
							help='Optional. Gemini safety profile for the request.',
							index=None, placeholder='Options' )
		
		# ------------------------------------------------------------------
		# System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			in_left, in_right = st.columns( [ 0.75, 0.25 ], border=True, gap='xxsmall' )
			with in_left:
				st.text_area( label='Enter Text', height=90, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='text_system_instructions' )
			
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions', on_change=load_text_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_text_instructions )
			
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_text_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ---------------------------------------------------
		#                   MESSAGES
		# ---------------------------------------------------
		if isinstance( st.session_state.get( 'text_messages' ), list ):
			for msg in st.session_state.text_messages:
				role = msg.get( 'role', 'assistant' )
				avatar = text_avatar if role == 'assistant' else ''
				with st.chat_message( role, avatar=avatar ):
					st.markdown( msg.get( 'content', '' ) )
		
		prompt = st.chat_input( f'Ask {provider_name} …' )
		if prompt is not None and str( prompt ).strip( ):
			prompt = str( prompt ).strip( )
			st.session_state.text_messages.append( {
						'role': 'user',
						'content': prompt,
				} )
			
			with st.chat_message( 'assistant', avatar=text_avatar ):
				with st.spinner( 'Thinking…' ):
					response_text = ''
					response_obj = None
					stream_buffer: List[ str ] = [ ]
					stream_placeholder = st.empty( )
					
					def on_stream_chunk( chunk: str ) -> None:
						"""
						
							Purpose:
							--------
							Render streamed Gemini chunks into a placeholder.
							
							Parameters:
							-----------
							chunk: str
								Chunk text from the provider stream.
							
							Returns:
							--------
							None
							
						"""
						if chunk is None:
							return
						
						stream_buffer.append( str( chunk ) )
						stream_placeholder.markdown( ''.join( stream_buffer ) + '▌' )
					
					try:
						st.session_state[ 'text_domains' ] = split_text_values(
							st.session_state.get( 'text_domains_input', '' ), delimiter=',' )
						st.session_state[ 'text_stops' ] = split_text_values(
							st.session_state.get( 'text_stops_input', '' ), delimiter=',' )
						
						if provider_name == 'GPT':
							manual_vector_store_ids = parse_text_vector_store_ids(
								st.session_state.get( 'text_vector_store_ids', '' ) )

							selected_vector_store_ids = get_store_ids( 'GPT' )
							
							vector_store_ids = merge_unique_strings(
								primary=manual_vector_store_ids,
								secondary=selected_vector_store_ids )
							
							text_tools = build_text_tools(
								selected_tools=st.session_state.get( 'text_tools', [ ] ),
								vector_store_ids=vector_store_ids )
							
							text_include = build_text_include(
								selected_include=st.session_state.get( 'text_include', [ ] ),
								selected_tools=text_tools )
							
							text_tool_choice = build_text_tool_choice(
								tool_choice=st.session_state.get( 'text_tool_choice' ),
								selected_tools=text_tools )
							
							text_format = build_text_response_format(
								response_format=st.session_state.get( 'text_response_format' ),
								schema_name=st.session_state.get( 'text_json_schema_name' ),
								schema_text=st.session_state.get( 'text_json_schema' ),
								strict=st.session_state.get( 'text_json_schema_strict', True ) )
							
							if st.session_state.get( 'text_input' ) != 'single_turn':
								text_context = build_text_context(
									messages=st.session_state.get( 'text_messages', [ ] ),
									include_last_message=False )
							else:
								text_context = [ ]
							
							st.session_state[ 'text_context' ] = text_context
							text_previous_id = get_text_previous_response_id(
								input_mode=st.session_state.get( 'text_input' ),
								previous_id=st.session_state.get( 'text_previous_response_id' ) )
							
							text_conversation_id = get_text_conversation_id(
								input_mode=st.session_state.get( 'text_input' ),
								conversation_id=st.session_state.get( 'text_conversation_id' ) )
							
							response_text = text.generate_text( prompt=prompt,
								model=st.session_state.get( 'text_model' ),
								temperature=st.session_state.get( 'text_temperature' ),
								format=text_format,
								top_p=st.session_state.get( 'text_top_percent' ),
								frequency=st.session_state.get( 'text_frequency_penalty' ),
								max_tools=st.session_state.get( 'text_max_calls' ),
								presence=st.session_state.get( 'text_presence_penalty' ),
								max_tokens=st.session_state.get( 'text_max_tokens' ),
								store=st.session_state.get( 'text_store' ), stream=False,
								instruct=st.session_state.get( 'text_system_instructions' ),
								background=False,
								reasoning=st.session_state.get( 'text_reasoning' ), include=text_include,
								tools=text_tools,
								allowed_domains=st.session_state.get( 'text_domains', [ ] ),
								previous_id=text_previous_id,
								tool_choice=text_tool_choice,
								is_parallel=st.session_state.get( 'text_parallel_tools' ),
								context=text_context,
								vector_store_ids=vector_store_ids,
								conversation_id=text_conversation_id )
							
							response_obj = getattr( text, 'response', None )
						
						elif provider_name == 'Gemini':
							apply_gemini_runtime_config( )
							
							structured_context = st.session_state.get( 'text_gemini_history', [ ] )
							if not isinstance( structured_context, list ) or len( 
									structured_context ) == 0:
								structured_context = st.session_state.get( 
									'text_messages', [ ] )[ :-1 ]
							
							grounding_enabled = bool(
								st.session_state.get( 'text_google_grounding', False ) )
							
							selected_tools = [ str( item ).strip( )
									for item in st.session_state.get( 'text_tools', [ ] )
									if str( item ).strip( ) ]
							
							if grounding_enabled and 'google_search' not in selected_tools:
								selected_tools.append( 'google_search' )
							
							st.session_state[ 'text_urls' ] = split_text_values(
								st.session_state.get( 'text_urls_input', '' ), delimiter=';' )
							
							manual_file_search_store_names = split_text_values(
								st.session_state.get( 'text_file_search_store_names_input', '' ),
								delimiter=',' )

							selected_file_search_store_names = get_active_gemini_file_search_store_names( 'Gemini' )
							
							st.session_state[ 'text_file_search_store_names' ] = merge_unique_strings(
								primary=manual_file_search_store_names,
								secondary=selected_file_search_store_names )
							
							response_text = text.generate_text( prompt=prompt,
								model=st.session_state.get( 'text_model' ),
								number=st.session_state.get( 'text_number' ),
								temperature=st.session_state.get( 'text_temperature' ),
								top_p=st.session_state.get( 'text_top_percent' ),
								top_k=st.session_state.get( 'text_top_k' ),
								frequency=st.session_state.get( 'text_frequency_penalty' ),
								presence=st.session_state.get( 'text_presence_penalty' ),
								max_tokens=st.session_state.get( 'text_max_tokens' ),
								stops=st.session_state.get( 'text_stops', [ ] ),
								instruct=st.session_state.get( 'text_system_instructions' ),
								response_format=st.session_state.get( 'text_response_format' ),
								tools=selected_tools,
								tool_choice=st.session_state.get( 'text_tool_choice' ),
								reasoning=st.session_state.get( 'text_reasoning' ),
								modalities=st.session_state.get( 'text_modalities', [ ] ),
								media_resolution=st.session_state.get( 'text_media_resolution' ),
								context=structured_context,
								content=st.session_state.get( 'text_content' ),
								urls=st.session_state.get( 'text_urls', [ ] ),
								max_urls=st.session_state.get( 'text_max_urls' ),
								response_schema=st.session_state.get( 'text_response_schema' ),
								safety_profile=st.session_state.get( 'text_safety_profile' ),
								file_search_store_names=st.session_state.get(
									'text_file_search_store_names', [ ] ),
								stream=st.session_state.get( 'text_stream', False ),
								stream_handler=on_stream_chunk if st.session_state.get(
									'text_stream', False ) else None )
							
							response_obj = getattr( text, 'content_response', None )
							
							if st.session_state.get( 'text_stream', False ):
								st.session_state[ 'text_gemini_history' ] = [ ]
							else:
								structured_history = text.get_structured_history( ) \
									if hasattr( text, 'get_structured_history' ) else [ ]
								
								if structured_history is not None and len( structured_history ) > 0:
									st.session_state[ 'text_gemini_history' ] = structured_history
						
						elif provider_name == 'Grok':
							grok_collection_ids = get_active_grok_collection_ids( 'Grok' )
							
							if len( grok_collection_ids ) > 0 and provider_supports( 'VectorStores', 'Grok' ):
								try:
									grok_vectorstores = get_vectorstores_module( 'Grok' )
									search_result = grok_vectorstores.search(
										prompt=prompt,
										store_id=grok_collection_ids[ 0 ] )
									
									if isinstance( search_result, str ) and search_result.strip( ):
										prompt = ( 'Use xAI Collection search result as retrieval context.\n\n'
												f'{search_result.strip( )}\n\n'
												f'User Question:\n{prompt}' )
								except Exception as exc:
									st.warning( f'Grok collection search was skipped: {exc}' )
							
							if hasattr( text, 'user' ):
								text.user = prompt
							
							response_obj = text.create( prompt=prompt,
								model=st.session_state.get( 'text_model' ) or 'grok-4',
								max_tokens=st.session_state.get( 'text_max_tokens' ) or 10000,
								temperature=st.session_state.get( 'text_temperature' ) or 0.8,
								top_p=st.session_state.get( 'text_top_percent' ) or 0.9,
								effort=st.session_state.get( 'text_reasoning' ) or 'high',
								format=st.session_state.get( 'text_response_format' ) or 'text',
								store=bool( st.session_state.get( 'text_store', True ) ),
								include=st.session_state.get( 'text_include', [ ] ),
								instruct=st.session_state.get( 'text_system_instructions' ) )
							
							response_text = str( response_obj or '' ).strip( )
						
						else:
							response_text = ''
							response_obj = None
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Generation Failed: {err.info}' )
						response_text = ''
						response_obj = None
					
					if response_text is not None and str( response_text ).strip( ):
						final_text = str( response_text ).strip( )
						
						if st.session_state.get( 'text_stream',
								False ) and provider_name == 'Gemini':
							stream_placeholder.markdown( final_text )
						else:
							st.markdown( final_text )
						
						st.session_state.text_messages.append( {
									'role': 'assistant',
									'content': final_text,
							} )
						st.session_state[ 'last_answer' ] = final_text
						st.session_state[ 'last_sources' ] = extract_sources( response_obj )
						
						try:
							update_token_counters( response_obj )
						except Exception:
							pass
					else:
						st.error( 'Generation Failed!.' )
		
		clear_c1, clear_c2 = st.columns( [ 0.8, 0.2 ] )
		with clear_c2:
			st.button( label='Clear Messages', width='stretch',
				on_click=clear_text_messages )

# ======================================================================================
# IMAGES MODE
# ======================================================================================
elif mode == 'Images':
	ensure_image_mode_state( )
	
	provider_name = get_provider_name( )
	image = get_images_module( )
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'image_system_instructions' ] = ''
		st.session_state[ 'clear_image_instructions' ] = False
		st.session_state[ 'clear_instructions' ] = False
	
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '📷 Images API', help=cfg.IMAGES_API )
		st.divider( )
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# LLM Settings
			# ------------------------------------------------------------------
			with st.expander( label='LLM Settings', icon='🧊', expanded=False, width='stretch' ):
				llm_c1, llm_c2, llm_c3, llm_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				with llm_c1:
					st.selectbox( label='Image Mode',
						options=[ 'Generation', 'Analysis', 'Editing' ],
						key='image_mode', help='Available image workflows.',
						index=None, placeholder='Options' )
				
				with llm_c2:
					current_image_mode = st.session_state.get( 'image_mode', '' )
					
					if current_image_mode == 'Analysis':
						model_options = get_image_analysis_models( image )
					elif current_image_mode == 'Editing':
						model_options = get_image_editing_models( image )
					else:
						model_options = get_image_models( image )
					
					st.selectbox( label='Select Model', options=model_options,
						key='image_model',
						help='Required model for the selected image workflow.',
						index=None, placeholder='Options' )
				
				with llm_c3:
					if provider_name == 'GPT':
						st.selectbox( label='Analysis Model',
							options=get_image_analysis_models( image ),
							key='image_analysis_model',
							help='Responses API vision model used for image analysis.',
							index=None, placeholder='Options' )
					elif provider_name == 'Gemini':
						st.selectbox( label='Response Modality',
							options=get_image_modality_options( image ),
							key='image_modality',
							help='Gemini image response modality.',
							index=None, placeholder='Options' )
					else:
						st.selectbox( label='Analysis Model',
							options=get_image_analysis_models( image ),
							key='image_analysis_model',
							help='Vision-capable model used for image analysis.',
							index=None, placeholder='Options' )
				
				with llm_c4:
					st.slider( label='Number', min_value=1, max_value=10,
						value=max( 1, int( st.session_state.get( 'image_number', 1 ) or 1 ) ),
						step=1, help='Number of images to request.', key='image_number' )
				
				st.button( label='Reset', key='reset_image_llm',
					width='stretch', on_click=reset_image_llm_settings )
			
			# ------------------------------------------------------------------
			# Visual Settings
			# ------------------------------------------------------------------
			with st.expander( label='Visual Settings', icon='👁️', expanded=False,
					width='stretch' ):
				vis_c1, vis_c2, vis_c3, vis_c4, vis_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				with vis_c1:
					st.selectbox( label='Output Format',
						options=get_image_mime_options( image ), key='image_mime_type',
						help='Image output format.', index=None, placeholder='Options' )
				
				with vis_c2:
					if provider_name == 'Gemini':
						st.selectbox( label='Aspect Ratio',
							options=get_image_aspect_options( image ),
							key='image_aspect_ratio',
							help='Gemini image aspect ratio.',
							index=None, placeholder='Options' )
					else:
						st.selectbox( label='Image Size',
							options=get_image_size_options( image ), key='image_size',
							help='Requested output image size.',
							index=None, placeholder='Options' )
				
				with vis_c3:
					st.selectbox( label='Image Quality',
						options=get_image_quality_options( image ), key='image_quality',
						help='Requested image quality when supported.',
						index=None, placeholder='Options' )
				
				with vis_c4:
					st.selectbox( label='Background',
						options=get_image_background_options( image ), key='image_backcolor',
						help='Requested background mode when supported.',
						index=None, placeholder='Options' )
				
				with vis_c5:
					if provider_name == 'GPT':
						st.selectbox( label='Analysis Detail',
							options=get_image_detail_options( image ),
							key='image_analysis_detail',
							help='Vision analysis detail level.',
							index=None, placeholder='Options' )
					else:
						st.selectbox( label='Resolution',
							options=get_text_option_list( image, 'media_options',
								[ '', 'media_resolution_high', 'media_resolution_medium',
								  'media_resolution_low' ] ),
							key='image_media_resolution',
							help='Media or output resolution when supported.',
							index=None, placeholder='Options' )
				
				st.slider( label='Compression', min_value=0.0, max_value=1.0,
					value=float( st.session_state.get( 'image_compression', 0.0 ) or 0.0 ),
					step=0.01, help='Optional JPEG/WebP compression value.',
					key='image_compression' )
				
				st.button( label='Reset', key='image_visual_reset',
					width='stretch', on_click=reset_image_visual_settings )
			
			# ------------------------------------------------------------------
			# Tool Settings
			# ------------------------------------------------------------------
			with st.expander( label='Tool Settings', icon='🛠️', expanded=False, width='stretch' ):
				tool_c1, tool_c2, tool_c3, tool_c4, tool_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				with tool_c1:
					if provider_name == 'Gemini':
						st.toggle( label='Google Grounding', key='image_grounded',
							help='Enable Gemini Google Search grounding when supported.' )
					else:
						st.toggle( label='Parallel Tools', key='image_parallel_calls',
							help=cfg.PARALLEL_TOOL_CALLS )
				
				with tool_c2:
					if provider_name == 'Gemini':
						st.toggle( label='Image Search', key='image_image_search',
							help='Enable Gemini image-search grounding when supported.' )
					else:
						st.slider( label='Max Tool Calls', min_value=0, max_value=10,
							value=int( st.session_state.get( 'image_max_calls', 0 ) or 0 ),
							step=1, help=cfg.MAX_TOOL_CALLS, key='image_max_calls' )
				
				with tool_c3:
					st.slider( label='Max Searches', min_value=0, max_value=30,
						value=int( st.session_state.get( 'image_max_searches', 0 ) or 0 ),
						step=1, help='Optional maximum search count.',
						key='image_max_searches' )
				
				with tool_c4:
					tool_options = get_text_option_list( image, 'tool_options', [ ] )
					st.multiselect( label='Tools', options=tool_options,
						key='image_tools', help=cfg.TOOLS, placeholder='Options' )
				
				with tool_c5:
					choice_options = get_text_option_list( image, 'choice_options',
						[ 'auto', 'required', 'none' ] )
					st.selectbox( label='Tool Choice', options=choice_options,
						key='image_tool_choice', help=cfg.CHOICE, index=None,
						placeholder='Options' )
				
				if provider_name == 'GPT':
					st.text_input( label='Allowed Websites',
						key='image_domains_input',
						value=','.join( st.session_state.get( 'image_domains', [ ] ) ),
						help='Optional. Comma-delimited domains for image-related web tools.',
						width='stretch', placeholder='gao.gov,omb.gov,congress.gov' )
				
				st.button( label='Reset', key='image_tools_reset',
					width='stretch', on_click=reset_image_tool_settings )
			
			# ------------------------------------------------------------------
			# Response Settings
			# ------------------------------------------------------------------
			with st.expander( label='Response Settings', icon='↔️', expanded=False,
					width='stretch' ):
				
				resp_c1, resp_c2, resp_c3, resp_c4, resp_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				with resp_c1:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0,
						value=float( st.session_state.get( 'image_top_percent', 0.0 ) or 0.0 ),
						step=0.01, help=cfg.TOP_P, key='image_top_percent' )
				
				with resp_c2:
					st.slider( label='Temperature', min_value=-2.0, max_value=2.0,
						value=float( st.session_state.get( 'image_temperature', 0.0 ) or 0.0 ),
						step=0.01, help=cfg.TEMPERATURE, key='image_temperature' )
				
				with resp_c3:
					st.slider( label='Max Tokens', min_value=0, max_value=100000,
						value=int( st.session_state.get( 'image_max_tokens', 0 ) or 0 ),
						step=500, help=cfg.MAX_OUTPUT_TOKENS, key='image_max_tokens' )
				
				with resp_c4:
					st.toggle( label='Store', key='image_store', help=cfg.STORE )
				
				with resp_c5:
					st.toggle( label='Background', key='image_background',
						help=cfg.BACKGROUND_MODE )
				
				st.button( label='Reset', key='image_response_reset',
					width='stretch', on_click=reset_image_response_settings )
		
		# ------------------------------------------------------------------
		# System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch' ):
			
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			in_left, in_right = st.columns( [ 0.75, 0.25 ], border=True, gap='xxsmall' )
			
			with in_left:
				st.text_area( label='Enter Text', height=90, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='image_system_instructions' )
			
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions', on_change=load_image_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_image_instructions )
			
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_image_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Workflow Tabs
		# ------------------------------------------------------------------
		tab_gen, tab_analyze, tab_edit = st.tabs( [ 'Generate', 'Analyze', 'Edit' ] )
		
		# Image Generation
		with tab_gen:
			prompt = st.chat_input( 'Enter image generation prompt...',
				key='image_generate_prompt' )
			gen_c1, gen_c2 = st.columns( [ 0.5, 0.5 ] )
			
			with gen_c1:
				if st.button( 'Generate Image', key='generate_image', width='stretch' ):
					prompt_value = get_image_prompt( prompt )
					if not prompt_value:
						st.warning( 'Enter a prompt before generating an image.' )
					else:
						append_image_message( 'user', prompt_value )
						
						with st.spinner( 'Generating…' ):
							try:
								if provider_name == 'Gemini':
									apply_gemini_runtime_config( )
									
									image_result = image.generate(
										prompt=prompt_value,
										model=st.session_state.get( 'image_model' )
										      or 'gemini-2.5-flash-image',
										aspect=st.session_state.get( 'image_aspect_ratio' ) or None,
										number=st.session_state.get( 'image_number' ),
										temperature=st.session_state.get( 'image_temperature' ),
										top_p=st.session_state.get( 'image_top_percent' ),
										frequency=st.session_state.get( 'image_frequency_penalty' ),
										presence=st.session_state.get( 'image_presence_penalty' ),
										max_tokens=st.session_state.get( 'image_max_tokens' ),
										resolution=st.session_state.get( 'image_media_resolution' ),
										instruct=st.session_state.get(
											'image_system_instructions' ),
										output_mime_type=st.session_state.get( 'image_mime_type' ),
										response_modalities=st.session_state.get(
											'image_modality' ),
										grounded=st.session_state.get( 'image_grounded', False ),
										image_search=st.session_state.get(
											'image_image_search', False ) )
								else:
									st.session_state[ 'image_domains' ] = split_text_values(
										st.session_state.get( 'image_domains_input', '' ),
										delimiter=',' )
									
									image_result = image.generate(
										prompt=prompt_value,
										number=st.session_state.get( 'image_number' ) or 1,
										model=st.session_state.get( 'image_model' )
										      or 'gpt-image-1-mini',
										size=st.session_state.get( 'image_size' ) or '1024x1024',
										quality=st.session_state.get( 'image_quality' ) or 'auto',
										fmt=st.session_state.get( 'image_mime_type' ) or 'jpeg',
										compression=st.session_state.get( 'image_compression' ),
										background=st.session_state.get( 'image_backcolor' )
										           or None )
								
								if render_image_output( image_result, caption='Generated image' ):
									append_image_message( 'assistant', 'Generated image.' )
									st.session_state[ 'image_output_bytes' ] = image_result
									st.session_state[ 'last_answer' ] = 'Generated image.'
								else:
									st.warning( 'No generated image output was returned.' )
								
								try:
									update_token_counters(
										getattr( image, 'response', None )
										or getattr( image, 'content_response', None )
										or getattr( image, 'image_response', None ) )
								except Exception:
									pass
							
							except Exception as exc:
								st.error( f'Generation Failed: {exc}' )
			
			with gen_c2:
				st.button( 'Clear Messages', key='clear_image_generation_messages',
					width='stretch', on_click=clear_image_messages )
		
		# Image Analysis
		with tab_analyze:
			uploaded_img = st.file_uploader( 'Upload Image for Analysis',
				type=[ 'png', 'jpg', 'jpeg', 'webp' ], accept_multiple_files=False,
				key='images_analysis_uploader' )
			
			tmp_path = None
			if uploaded_img:
				tmp_path = save_temp( uploaded_img )
				st.image( uploaded_img, caption='Uploaded image preview',
					use_column_width=True )
			
			prompt = st.chat_input( 'Enter image analysis prompt...', key='image_analyze_prompt' )
			ana_c1, ana_c2 = st.columns( [ 0.5, 0.5 ] )
			
			with ana_c1:
				if st.button( 'Analyze Image', key='analyze_image', width='stretch' ):
					prompt_value = get_image_prompt( prompt )
					
					if not tmp_path:
						st.warning( 'Upload an image before analysis.' )
					elif not prompt_value:
						st.warning( 'Enter a prompt before analyzing an image.' )
					else:
						append_image_message( 'user', prompt_value )
						
						with st.spinner( 'Analyzing image…' ):
							try:
								if provider_name == 'Gemini':
									apply_gemini_runtime_config( )
									
									analysis_result = image.analyze(
										prompt=prompt_value,
										path=tmp_path,
										model=st.session_state.get( 'image_model' )
										      or 'gemini-2.5-flash-image',
										aspect=st.session_state.get( 'image_aspect_ratio' )
										       or None,
										number=st.session_state.get( 'image_number' ),
										temperature=st.session_state.get( 'image_temperature' ),
										top_p=st.session_state.get( 'image_top_percent' ),
										frequency=st.session_state.get( 'image_frequency_penalty' ),
										presence=st.session_state.get( 'image_presence_penalty' ),
										max_tokens=st.session_state.get( 'image_max_tokens' ),
										resolution=st.session_state.get(
											'image_media_resolution' ),
										instruct=st.session_state.get(
											'image_system_instructions' ),
										output_mime_type=st.session_state.get(
											'image_mime_type' ),
										response_modalities=st.session_state.get(
											'image_modality' ) or 'text',
										grounded=st.session_state.get(
											'image_grounded', False ),
										image_search=st.session_state.get(
											'image_image_search', False ) )
								else:
									analysis_model = (
											st.session_state.get( 'image_analysis_model' )
											or st.session_state.get( 'image_model' )
											or 'gpt-4o-mini'
									)
									
									analysis_result = image.analyze(
										text=prompt_value,
										path=tmp_path,
										instruct=st.session_state.get(
											'image_system_instructions' ),
										model=analysis_model,
										max_tokens=st.session_state.get(
											'image_max_tokens' ),
										temperature=st.session_state.get(
											'image_temperature' ),
										include=st.session_state.get( 'image_include', [ ] ),
										store=st.session_state.get( 'image_store' ),
										stream=False,
										detail=st.session_state.get(
											'image_analysis_detail' ) or 'auto' )
								
								if analysis_result is not None and str( analysis_result ).strip( ):
									st.markdown( '**Analysis result:**' )
									st.write( analysis_result )
									append_image_message( 'assistant',
										str( analysis_result ).strip( ) )
									st.session_state[ 'last_answer' ] = str(
										analysis_result ).strip( )
								else:
									st.warning( 'No analysis output was returned.' )
								
								try:
									update_token_counters(
										getattr( image, 'response', None )
										or getattr( image, 'content_response', None )
										or getattr( image, 'image_response', None ) )
								except Exception:
									pass
							
							except Exception as exc:
								st.error( f'Analysis Failed: {exc}' )
			
			with ana_c2:
				st.button( 'Clear Messages', key='clear_image_analysis_messages',
					width='stretch', on_click=clear_image_messages )
		
		# Image Editing
		with tab_edit:
			uploaded_img = st.file_uploader( 'Upload Image for Edit',
				type=[ 'png', 'jpg', 'jpeg', 'webp' ], accept_multiple_files=False,
				key='images_edit_uploader' )
			
			tmp_path = None
			if uploaded_img:
				tmp_path = save_temp( uploaded_img )
				st.image( uploaded_img, caption='Uploaded image preview',
					use_column_width=True )
			
			prompt = st.chat_input( 'Enter image editing prompt...', key='image_edit_prompt' )
			edit_c1, edit_c2 = st.columns( [ 0.5, 0.5 ] )
			
			with edit_c1:
				if st.button( 'Edit Image', key='edit_image', width='stretch' ):
					prompt_value = get_image_prompt( prompt )
					
					if not tmp_path:
						st.warning( 'Upload an image before editing.' )
					elif not prompt_value:
						st.warning( 'Enter a prompt before editing an image.' )
					else:
						append_image_message( 'user', prompt_value )
						
						with st.spinner( 'Editing image…' ):
							try:
								if provider_name == 'Gemini':
									apply_gemini_runtime_config( )
									
									edit_result = image.edit(
										prompt=prompt_value,
										path=tmp_path,
										model=st.session_state.get( 'image_model' )
										      or 'gemini-2.5-flash-image',
										aspect=st.session_state.get( 'image_aspect_ratio' )
										       or None,
										number=st.session_state.get( 'image_number' ),
										temperature=st.session_state.get(
											'image_temperature' ),
										top_p=st.session_state.get( 'image_top_percent' ),
										frequency=st.session_state.get(
											'image_frequency_penalty' ),
										presence=st.session_state.get(
											'image_presence_penalty' ),
										max_tokens=st.session_state.get(
											'image_max_tokens' ),
										resolution=st.session_state.get(
											'image_media_resolution' ),
										instruct=st.session_state.get(
											'image_system_instructions' ),
										output_mime_type=st.session_state.get(
											'image_mime_type' ),
										response_modalities=st.session_state.get(
											'image_modality' ) or 'image',
										grounded=st.session_state.get(
											'image_grounded', False ),
										image_search=st.session_state.get(
											'image_image_search', False ) )
								else:
									edit_result = image.edit(
										prompt=prompt_value,
										path=tmp_path,
										model=st.session_state.get( 'image_model' )
										      or 'gpt-image-1-mini',
										size=st.session_state.get( 'image_size' )
										     or '1024x1024',
										quality=st.session_state.get( 'image_quality' )
										        or 'auto',
										fmt=st.session_state.get( 'image_mime_type' )
										    or 'jpeg',
										compression=st.session_state.get(
											'image_compression' ),
										background=st.session_state.get( 'image_backcolor' )
										           or None,
										number=st.session_state.get( 'image_number' ) )
								
								if render_image_output( edit_result, caption='Edited image' ):
									append_image_message( 'assistant', 'Edited image.' )
									st.session_state[ 'image_output_bytes' ] = edit_result
									st.session_state[ 'last_answer' ] = 'Edited image.'
								else:
									st.warning( 'No edited image output was returned.' )
								
								try:
									update_token_counters(
										getattr( image, 'response', None )
										or getattr( image, 'content_response', None )
										or getattr( image, 'image_response', None ) )
								except Exception:
									pass
							
							except Exception as exc:
								st.error( f'Edit Failed: {exc}' )
			
			with edit_c2:
				st.button( 'Clear Messages', key='clear_image_edit_messages',
					width='stretch', on_click=clear_image_messages )

# ======================================================================================
# AUDIO MODE
# ======================================================================================
elif mode == 'Audio':
	ensure_audio_runtime_state( )
	
	provider_name = get_provider_name( )
	transcriber = get_transcription_module( ) if provider_supports( 'Transcription' ) else None
	translator = get_translation_module( ) if provider_supports( 'Translation' ) else None
	tts = get_tts_module( ) if provider_supports( 'TTS' ) else None
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'audio_system_instructions' ] = ''
		st.session_state[ 'clear_audio_instructions' ] = False
		st.session_state[ 'clear_instructions' ] = False
	
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '🎧 Audio API', help=cfg.AUDIO_API )
		st.divider( )
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# LLM Settings
			# ------------------------------------------------------------------
			with st.expander( label='LLM Settings', icon='🧊', expanded=False, width='stretch' ):
				audio_c1, audio_c2, audio_c3, audio_c4, audio_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				with audio_c1:
					task_options = get_audio_task_options( )
					if not task_options:
						st.info( 'Audio is not supported by the selected provider.' )
						audio_task = ''
					else:
						st.selectbox( label='Task', options=task_options, key='audio_task',
							help='Select the Audio API workflow to run.',
							index=None, placeholder='Options' )
						audio_task = st.session_state.get( 'audio_task', '' )
				
				model_options = get_audio_model_options(
					audio_task, transcriber, translator, tts )
				
				if st.session_state.get( 'audio_model' ) not in model_options:
					st.session_state[ 'audio_model' ] = ''
				
				format_options = get_audio_response_format_options(
					audio_task, st.session_state.get( 'audio_model' ),
					transcriber, translator, tts )
				
				if st.session_state.get( 'audio_response_format' ) not in format_options:
					st.session_state[ 'audio_response_format' ] = ''
				
				with audio_c2:
					st.selectbox( label='Model', options=model_options,
						key='audio_model', help='Audio model for the selected task.',
						index=None, placeholder='Options' )
				
				with audio_c3:
					if audio_task in [ 'Transcribe', 'Translate' ]:
						st.selectbox( label='Language',
							options=get_audio_language_options(
								audio_task, transcriber, translator ),
							key='audio_language',
							help='Optional language hint or target language.',
							index=None, placeholder='Options' )
					elif audio_task == 'Text-to-Speech':
						st.selectbox( label='Voice',
							options=get_audio_voice_options( tts ),
							key='audio_voice',
							help='Voice used for text-to-speech.',
							index=None, placeholder='Options' )
					else:
						st.caption( 'Select a task.' )
				
				with audio_c4:
					st.selectbox( label='Format', options=format_options,
						key='audio_response_format',
						help='Response or output audio format.',
						index=None, placeholder='Options' )
				
				with audio_c5:
					st.button( label='Reset', key='audio_llm_reset',
						width='stretch', on_click=reset_audio_llm_settings )
			
			# ------------------------------------------------------------------
			# Response Settings
			# ------------------------------------------------------------------
			with st.expander( label='Response Settings', icon='↔️', expanded=False, width='stretch' ):
				
				resp_c1, resp_c2, resp_c3, resp_c4, resp_c5, resp_c6 = st.columns(
					[ 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 ], border=True, gap='xxsmall' )
				
				with resp_c1:
					st.slider( label='Temperature', min_value=-2.0, max_value=2.0,
						value=float( st.session_state.get( 'audio_temperature', 0.0 ) ),
						step=0.01, help=cfg.TEMPERATURE, key='audio_temperature' )
				
				with resp_c2:
					st.slider( label='Top-P', min_value=0.0, max_value=1.0,
						value=float( st.session_state.get( 'audio_top_percent', 0.0 ) ),
						step=0.01, help=cfg.TOP_P, key='audio_top_percent' )
				
				with resp_c3:
					st.slider( label='Max Tokens', min_value=0, max_value=100000,
						value=int( st.session_state.get( 'audio_max_tokens', 0 ) ),
						step=500, help=cfg.MAX_OUTPUT_TOKENS,
						key='audio_max_tokens' )
				
				with resp_c4:
					st.slider( label='Speed', min_value=0.25, max_value=4.0,
						value=float( st.session_state.get( 'audio_speed', 1.0 ) or 1.0 ),
						step=0.05, help='Speech playback speed for text-to-speech.',
						key='audio_speed' )
				
				with resp_c5:
					st.toggle( label='Store', key='audio_store', help=cfg.STORE )
				
				with resp_c6:
					st.button( label='Reset', key='audio_response_reset',
						width='stretch', on_click=reset_audio_response_settings )
				
				play_c1, play_c2, play_c3, play_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				with play_c1:
					st.slider( label='Start Time', min_value=0.0, max_value=3600.0,
						value=float( st.session_state.get( 'audio_start_time', 0.0 ) ),
						step=1.0, help='Playback start time.',
						key='audio_start_time' )
				
				with play_c2:
					st.slider( label='End Time', min_value=0.0, max_value=3600.0,
						value=float( st.session_state.get( 'audio_end_time', 0.0 ) ),
						step=1.0, help='Playback end time.',
						key='audio_end_time' )
				
				with play_c3:
					st.toggle( label='Loop', key='audio_loop',
						help='Loop local playback.' )
				
				with play_c4:
					st.toggle( label='Autoplay', key='audio_autoplay',
						help='Autoplay local playback when supported.' )
		
		# ------------------------------------------------------------------
		# System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			in_left, in_right = st.columns( [ 0.75, 0.25 ], border=True, gap='xxsmall' )
			
			with in_left:
				st.text_area( label='Enter Text', height=90, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='audio_system_instructions' )
			
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions', on_change=load_audio_instruction_template )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_audio_instructions )
			
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_audio_system_instructions )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Audio Work Area
		# ------------------------------------------------------------------
		left_audio, center_audio, right_audio = st.columns(
			[ 0.34, 0.33, 0.33 ], border=True, gap='small' )
		
		audio_task = st.session_state.get( 'audio_task', '' )
		tmp_path = None
		
		# ------------------------------------------------------------------
		# Upload / File Processing
		# ------------------------------------------------------------------
		with left_audio:
			st.caption( 'Audio File' )
			uploaded_audio = st.file_uploader(
				label='Upload Audio',
				type=[ 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm', 'flac',
				       'ogg', 'aac', 'aiff' ],
				accept_multiple_files=False,
				key='audio_uploaded_file' )
			
			if uploaded_audio is not None:
				tmp_path = save_audio_upload( uploaded_audio )
				st.audio( uploaded_audio )
			
			if audio_task in [ 'Transcribe', 'Translate' ]:
				if st.button( 'Process Audio', key='audio_process_file',
						width='stretch' ):
					with st.spinner( f'{audio_task} audio…' ):
						try:
							if provider_name == 'Gemini':
								apply_gemini_runtime_config( )
							
							result_text = run_audio_file_task(
								task=audio_task,
								file_path=tmp_path,
								transcriber=transcriber,
								translator=translator )
							
							render_audio_text_result( audio_task, result_text )
							
							if result_text:
								st.session_state[ 'audio_messages' ].append( {
											'role': 'assistant',
											'content': result_text,
									} )
								st.session_state[ 'last_answer' ] = result_text
							
							response_obj = None
							if audio_task == 'Transcribe' and transcriber is not None:
								response_obj = getattr( transcriber, 'response', None )
							elif audio_task == 'Translate' and translator is not None:
								response_obj = getattr( translator, 'response', None )
							
							try:
								update_token_counters( response_obj )
							except Exception:
								pass
						
						except Exception as exc:
							st.error( f'{audio_task} failed: {exc}' )
			
			elif audio_task == 'Text-to-Speech':
				st.info( 'Use the text box below to generate speech.' )
			else:
				st.info( 'Select an audio task.' )
		
		# ------------------------------------------------------------------
		# Record Audio
		# ------------------------------------------------------------------
		with center_audio:
			st.caption( 'Record Audio' )
			sample_rate = int( st.session_state.get( 'audio_rate', 44100 ) or 44100 )
			recording = st.audio_input( label='Record Audio', sample_rate=sample_rate )
			
			if recording is not None:
				st.audio( recording, format='audio/wav' )
				
				if audio_task in [ 'Transcribe', 'Translate' ]:
					if st.button( 'Process Recording', key='audio_process_recording',
							width='stretch' ):
						record_path = save_audio_upload( recording )
						
						with st.spinner( f'{audio_task} recording…' ):
							try:
								if provider_name == 'Gemini':
									apply_gemini_runtime_config( )
								
								result_text = run_audio_file_task(
									task=audio_task,
									file_path=record_path,
									transcriber=transcriber,
									translator=translator )
								
								render_audio_text_result( audio_task, result_text )
								
								if result_text:
									st.session_state[ 'audio_messages' ].append( {
												'role': 'assistant',
												'content': result_text,
										} )
									st.session_state[ 'last_answer' ] = result_text
							
							except Exception as exc:
								st.error( f'{audio_task} failed: {exc}' )
		
		# ------------------------------------------------------------------
		# Playback
		# ------------------------------------------------------------------
		with right_audio:
			st.caption( 'Local Audio File' )
			data = getattr( cfg, 'AUDIO_TEST_FILE', None )
			
			if data is not None and os.path.exists( data ):
				st.audio( data,
					start_time=float( st.session_state.get( 'audio_start_time', 0.0 ) ),
					end_time=float( st.session_state.get( 'audio_end_time', 0.0 ) ),
					format='audio/mp3', width='stretch',
					loop=bool( st.session_state.get( 'audio_loop', False ) ),
					autoplay=bool( st.session_state.get( 'audio_autoplay', False ) ) )
			else:
				st.info( 'No local audio file is configured.' )
		
		# ------------------------------------------------------------------
		# Text-to-Speech
		# ------------------------------------------------------------------
		if audio_task == 'Text-to-Speech':
			tts_text = st.text_area( label='Enter Text to Synthesize',
				key='audio_tts_input', height=150, width='stretch' )
			
			tts_c1, tts_c2 = st.columns( [ 0.5, 0.5 ] )
			
			with tts_c1:
				if st.button( 'Generate Audio', key='audio_generate_tts',
						width='stretch' ):
					with st.spinner( 'Synthesizing speech…' ):
						try:
							if provider_name == 'Gemini':
								apply_gemini_runtime_config( )
							
							audio_bytes = run_audio_tts_task( tts_text, tts )
							response_format = get_audio_response_format_value(
								task='Text-to-Speech',
								selected_format=st.session_state.get(
									'audio_response_format' ),
								selected_mime_type=st.session_state.get(
									'audio_mime_type' ) )
							
							render_audio_bytes( audio_bytes, response_format )
							
							if audio_bytes:
								st.session_state[ 'audio_messages' ].append(
									{
											'role': 'assistant',
											'content': 'Generated audio from text.',
									} )
								st.session_state[ 'last_answer' ] = 'Generated audio from text.'
							
							try:
								update_token_counters( getattr( tts, 'response', None ) )
							except Exception:
								pass
						
						except Exception as exc:
							st.error( f'Text-to-speech failed: {exc}' )
			
			with tts_c2:
				st.button( 'Clear Output', key='audio_clear_outputs',
					width='stretch', on_click=clear_audio_outputs )
		
		# ------------------------------------------------------------------
		# Output and Messages
		# ------------------------------------------------------------------
		if st.session_state.get( 'audio_output' ):
			st.text_area( 'Latest Text Output',
				value=st.session_state.get( 'audio_output', '' ),
				height=220, width='stretch', disabled=True )
		
		if st.session_state.get( 'audio_output_bytes' ):
			response_format = get_audio_response_format_value(
				task='Text-to-Speech',
				selected_format=st.session_state.get( 'audio_response_format' ),
				selected_mime_type=st.session_state.get( 'audio_mime_type' ) )
			render_audio_bytes( st.session_state.get( 'audio_output_bytes' ),
				response_format )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		if isinstance( st.session_state.get( 'audio_messages' ), list ):
			for msg in st.session_state.audio_messages:
				with st.chat_message( msg.get( 'role', 'assistant' ), avatar='' ):
					st.markdown( msg.get( 'content', '' ) )
		
		audio_chat = st.chat_input( 'Enter audio note …', key='audio_messages_input' )
		if audio_chat is not None and isinstance( audio_chat, str ) and audio_chat.strip( ):
			st.session_state.audio_messages.append( {
						'role': 'user',
						'content': audio_chat.strip( ),
				} )
		
		st.button( 'Clear Messages', key='audio_clear_messages',
			width='stretch', on_click=clear_audio_messages )

# ======================================================================================
# EMBEDDINGS MODE
# ======================================================================================
elif mode == 'Embeddings':
	ensure_embeddings_mode_state( )
	
	provider_name = get_provider_name( )
	embedding = get_embeddings_module( )
	
	# ------------------------------------------------------------------
	# Session State Type Guards
	# ------------------------------------------------------------------
	if not isinstance( st.session_state.get( 'embeddings_input_text' ), str ):
		st.session_state[ 'embeddings_input_text' ] = ''
	
	if not isinstance( st.session_state.get( 'embeddings_encoding_format' ), str ):
		st.session_state[ 'embeddings_encoding_format' ] = 'float'
	
	if not isinstance( st.session_state.get( 'embeddings_chunks' ), list ):
		st.session_state[ 'embeddings_chunks' ] = [ ]
	
	if not isinstance( st.session_state.get( 'embedding_metrics' ), dict ):
		st.session_state[ 'embedding_metrics' ] = { }
	
	if not isinstance( st.session_state.get( 'embedding_usage' ), dict ):
		st.session_state[ 'embedding_usage' ] = { }
	
	if 'embeddings_df' not in st.session_state or not isinstance(
			st.session_state.get( 'embeddings_df' ), pd.DataFrame ):
		st.session_state[ 'embeddings_df' ] = pd.DataFrame( )
	
	# ------------------------------------------------------------------
	# Main UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.subheader( '🧬 Embeddings API' )
		st.divider( )
		with st.expander( label='Configuration', icon='🧊', expanded=False, width='stretch' ):
			cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5 = st.columns(
				[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
			
			# ---------- Model ------------
			with cfg_c1:
				model_options = get_embedding_model_options( embedding )
				if st.session_state.get( 'embedding_model' ) not in model_options:
					st.session_state[ 'embedding_model' ] = ''
				
				st.selectbox( label='Model', options=model_options, key='embedding_model',
					help='Embedding model used by the selected provider.',
					index=None, placeholder='Options' )
				
				embedding_model = st.session_state.get( 'embedding_model', '' )
			
			# Normalize dimension state before the dimension slider is instantiated.
			max_dimensions = get_embedding_max_dimensions( embedding_model, embedding )
			supports_dimensions = embedding_model_supports_dimensions(
				embedding_model, embedding )
			
			try:
				current_dimensions = int(
					st.session_state.get( 'embeddings_dimensions', 0 ) or 0 )
			except Exception:
				current_dimensions = 0
			
			if not supports_dimensions:
				st.session_state[ 'embeddings_dimensions' ] = 0
			elif current_dimensions > max_dimensions:
				st.session_state[ 'embeddings_dimensions' ] = max_dimensions
			elif current_dimensions < 0:
				st.session_state[ 'embeddings_dimensions' ] = 0
			
			# ---------- Encoding Format ------------
			with cfg_c2:
				encoding_options = get_embedding_encoding_options( embedding )
				if st.session_state.get( 'embeddings_encoding_format' ) not in encoding_options:
					st.session_state[ 'embeddings_encoding_format' ] = 'float'
				
				st.selectbox( label='Encoding Format', options=encoding_options,
					key='embeddings_encoding_format',index=None, placeholder='Options' )
				
				embeddings_encoding_format = st.session_state.get(
					'embeddings_encoding_format', 'float' )
			
			# ---------- Dimensions ------------
			with cfg_c3:
				st.slider( label='Dimensions', min_value=0, max_value=max_dimensions, step=1,
					key='embeddings_dimensions', disabled=not supports_dimensions )
				
				embeddings_dimensions = st.session_state.get( 'embeddings_dimensions', 0 )
				
				if not supports_dimensions:
					st.caption( 'Dimensions are omitted for this model.' )
			
			# ---------- Chunk Size ------------
			with cfg_c4:
				try:
					current_chunk_size = int(
						st.session_state.get( 'embeddings_chunk_size', 800 ) or 800 )
				except Exception:
					current_chunk_size = 800
				
				if current_chunk_size <= 0:
					st.session_state[ 'embeddings_chunk_size' ] = 800
				elif current_chunk_size > 8192:
					st.session_state[ 'embeddings_chunk_size' ] = 8192
				
				st.slider( label='Chunk Size',
					min_value=1, max_value=8192, step=50,
					help='Maximum chunk size in tokenizer tokens.',
					key='embeddings_chunk_size' )
				
				embeddings_chunk_size = st.session_state.get( 'embeddings_chunk_size', 800 )
			
			# ---------- Overlap Amount ------------
			with cfg_c5:
				try:
					current_overlap = int(
						st.session_state.get( 'embeddings_overlap_amount', 0 ) or 0 )
				except Exception:
					current_overlap = 0
				
				if current_overlap < 0:
					st.session_state[ 'embeddings_overlap_amount' ] = 0
				elif current_overlap >= int(
						st.session_state.get( 'embeddings_chunk_size', 800 ) ):
					st.session_state[ 'embeddings_overlap_amount' ] = max(
						0,
						int( st.session_state.get( 'embeddings_chunk_size', 800 ) ) // 5 )
				
				st.slider( label='Overlap Amount', min_value=0,
					max_value=max(
						10,
						int( st.session_state.get( 'embeddings_chunk_size', 800 ) ) - 1 ),
					step=10,
					help='Token overlap between adjacent chunks.',
					key='embeddings_overlap_amount' )
				
				embeddings_overlap_amount = st.session_state.get(
					'embeddings_overlap_amount', 0 )
			
			user_c1, user_c2 = st.columns( [ 0.75, 0.25 ], border=True, gap='xxsmall' )
			with user_c1:
				st.text_input( label='User',
					key='embeddings_user',
					help='Optional end-user identifier for providers that support it.',
					width='stretch' )
			
			with user_c2:
				st.button( label='Reset Configuration',
					key='embedding_config_reset',
					width='stretch',
					on_click=reset_embeddings_controls )
		
		# ------------------------------------------------------------------
		# Input
		# ------------------------------------------------------------------
		st.text_area( label='Text to Embed', key='embeddings_input_text',
			height=260, width='stretch',
			help='Text that will be normalized, chunked, and embedded.' )
		
		action_c1, action_c2, action_c3 = st.columns(
			[ 0.34, 0.33, 0.33 ], border=True, gap='xxsmall' )
		
		# ------------------------------------------------------------------
		# Create Embeddings
		# ------------------------------------------------------------------
		with action_c1:
			if st.button( label='Create Embeddings',
					key='embedding_create',
					width='stretch' ):
				with st.spinner( 'Creating embeddings…' ):
					try:
						if provider_name == 'Gemini':
							apply_gemini_runtime_config( )
						
						source_text = st.session_state.get( 'embeddings_input_text', '' )
						model = st.session_state.get( 'embedding_model' )
						
						if not isinstance( model, str ) or not model.strip( ):
							if provider_name == 'GPT':
								model = 'text-embedding-3-small'
							elif provider_name == 'Gemini':
								model = 'gemini-embedding-001'
							else:
								model = ''
						
						encoding_format = st.session_state.get(
							'embeddings_encoding_format' ) or 'float'
						
						if not isinstance( source_text, str ) or not source_text.strip( ):
							st.warning( 'Enter text before creating embeddings.' )
						else:
							normalized_text = normalize_text( source_text )
							
							chunk_size, overlap_amount = normalize_embedding_chunk_settings(
								chunk_size=st.session_state.get(
									'embeddings_chunk_size', 800 ),
								overlap_amount=st.session_state.get(
									'embeddings_overlap_amount', 0 ) )
							
							chunks = chunk_text_for_embeddings(
								text=normalized_text,
								chunk_size=chunk_size,
								overlap_amount=overlap_amount )
							
							if len( chunks ) == 0:
								st.warning( 'No valid chunks were produced from the input text.' )
							else:
								dimensions = normalize_embedding_dimensions(
									model=model,
									dimensions=st.session_state.get(
										'embeddings_dimensions', 0 ),
									embedding=embedding )
								
								user_value = st.session_state.get( 'embeddings_user', '' )
								user_value = user_value.strip( ) if isinstance(
									user_value, str ) and user_value.strip( ) else None
								
								vectors = create_provider_embeddings(
									embedding=embedding,
									chunks=chunks,
									model=model,
									encoding_format=encoding_format,
									dimensions=dimensions,
									user_value=user_value )
								
								response_obj = (
										getattr( embedding, 'response', None )
										or getattr( embedding, 'content_response', None )
										or getattr( embedding, 'embedding_response', None )
								)
								
								usage = extract_embedding_usage( response_obj )
								
								df_embeddings = build_embeddings_dataframe(
									chunks=chunks,
									vectors=vectors,
									encoding_format=encoding_format )
								
								metrics = build_embedding_metrics(
									source_text=source_text,
									normalized_text=normalized_text,
									chunks=chunks,
									vectors=vectors,
									usage=usage )
								
								st.session_state[ 'embeddings' ] = normalize_embedding_vectors(
									vectors )
								st.session_state[ 'embeddings_chunks' ] = chunks
								st.session_state[ 'embeddings_df' ] = df_embeddings
								st.session_state[ 'embedding_metrics' ] = metrics
								st.session_state[ 'embedding_usage' ] = usage
								
								try:
									update_token_counters( response_obj )
								except Exception:
									pass
								
								st.success( 'Embeddings created successfully.' )
					
					except Exception as exc:
						err = Error( exc )
						st.error( f'Embedding creation failed: {err.info}' )
		
		# ------------------------------------------------------------------
		# Clear Output
		# ------------------------------------------------------------------
		with action_c2:
			st.button( label='Clear Output',
				key='clear_embeddings_output',
				width='stretch',
				on_click=clear_embeddings_output )
		
		# ------------------------------------------------------------------
		# Reset All
		# ------------------------------------------------------------------
		with action_c3:
			st.button( label='Reset All',
				key='reset_embeddings_all',
				width='stretch',
				on_click=reset_embeddings_all )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Metrics
		# ------------------------------------------------------------------
		metrics = st.session_state.get( 'embedding_metrics', { } )
		if isinstance( metrics, dict ) and len( metrics ) > 0:
			render_embedding_metrics( metrics )
		
		# ------------------------------------------------------------------
		# Embeddings Output
		# ------------------------------------------------------------------
		df_embeddings = st.session_state.get( 'embeddings_df', pd.DataFrame( ) )
		if isinstance( df_embeddings, pd.DataFrame ) and not df_embeddings.empty:
			st.subheader( 'Embedding Output' )
			render_embeddings_dataframe( df_embeddings )
		
		# ------------------------------------------------------------------
		# Chunks
		# ------------------------------------------------------------------
		chunks = st.session_state.get( 'embeddings_chunks', [ ] )
		if isinstance( chunks, list ) and len( chunks ) > 0:
			with st.expander( label='Chunks', icon='🧩',
					expanded=False, width='stretch' ):
				df_chunks = pd.DataFrame(
					[
							{
									'ChunkIndex': index + 1,
									'Text': chunk,
									'Tokens': count_tokens( chunk ),
							}
							for index, chunk in enumerate( chunks )
					] )
				st.data_editor( df_chunks, use_container_width=True, hide_index=True )
		
		# ------------------------------------------------------------------
		# Usage
		# ------------------------------------------------------------------
		usage = st.session_state.get( 'embedding_usage', { } )
		if isinstance( usage, dict ) and len( usage ) > 0:
			with st.expander( label='Embedding Usage', icon='📊',
					expanded=False, width='stretch' ):
				st.json( usage )

# ======================================================================================
# DOCQA MODE
# ======================================================================================
elif mode == 'Document Q&A':
	ensure_docqna_mode_state( )
	
	if not isinstance( st.session_state.get( 'docqna_messages' ), list ):
		st.session_state[ 'docqna_messages' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_active_docs' ), list ):
		st.session_state[ 'docqna_active_docs' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_files' ), list ):
		st.session_state[ 'docqna_files' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_texts' ), dict ):
		st.session_state[ 'docqna_texts' ] = { }
	
	if not isinstance( st.session_state.get( 'docqna_chunks' ), list ):
		st.session_state[ 'docqna_chunks' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_last_hits' ), list ):
		st.session_state[ 'docqna_last_hits' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_last_sources' ), list ):
		st.session_state[ 'docqna_last_sources' ] = [ ]
	
	if not isinstance( st.session_state.get( 'docqna_last_answer' ), str ):
		st.session_state[ 'docqna_last_answer' ] = ''
	
	if not isinstance( st.session_state.get( 'docqna_context' ), str ):
		st.session_state[ 'docqna_context' ] = ''
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'docqna_system_instructions' ] = ''
		st.session_state[ 'clear_instructions' ] = False
	
	provider_name = get_provider_name( )
	docqna_avatar = get_docqna_avatar( provider_name )
	
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📖 Document Q & A' )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# Source Controls
			# ------------------------------------------------------------------
			with st.expander( label='Source Controls', icon='📚', expanded=False, width='stretch' ):
				source_c1, source_c2, source_c3, source_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				with source_c1:
					source_options = get_docqna_sources( )
					if st.session_state.get( 'docqna_source' ) not in source_options:
						st.session_state[ 'docqna_source' ] = 'Local Upload'
					
					st.selectbox( label='Source',
						options=source_options,
						key='docqna_source',
						help='Document source used for Q&A.',
						index=None,
						placeholder='Options' )
				
				with source_c2:
					st.toggle( label='Multi-Document',
						key='docqna_multi_mode',
						help='Allow multiple local document uploads.' )
				
				with source_c3:
					st.slider( label='Top-K', min_value=1,
						max_value=20,
						value=int( st.session_state.get( 'docqna_top_k', 6 ) or 6 ), step=1,
						key='docqna_top_k',
						help='Number of retrieved chunks to use for local Q&A.' )
				
				with source_c4:
					st.button( label='Reset Controls',
						key='docqna_reset_controls',
						width='stretch',
						on_click=reset_docqna_controls )
				
				source_value = st.session_state.get( 'docqna_source', 'Local Upload' )
				if source_value == 'OpenAI File ID':
					st.text_input( label='OpenAI File ID', key='docqna_file_id', width='stretch',
						placeholder='file-...',
						help='OpenAI file identifier. Use Vector Store ID for retrieval-backed Q&A.' )
				
				elif source_value == 'OpenAI Vector Store ID':
					st.text_input( label='OpenAI Vector Store ID(s)', key='docqna_vector_store_id',
						width='stretch', placeholder='vs_...',
						help='Comma-delimited OpenAI vector store IDs.' )
				
				elif source_value == 'Gemini File Search Store':
					st.text_input( label='Gemini File Search Store Resource Name(s)',
						key='docqna_file_search_store_names_input', width='stretch',
						placeholder='fileSearchStores/...',
						help='Comma-delimited Gemini File Search Store resource names.' )
				
				elif source_value == 'xAI Collection':
					st.info(
						'Grok Document Q&A uses the xAI Collection selected in Vector Stores mode. '
						'Select or enter a collection ID there before asking questions.' )
			
			# ------------------------------------------------------------------
			# Model Controls
			# ------------------------------------------------------------------
			with st.expander( label='Model Controls', icon='🧊', expanded=False, width='stretch' ):
				model_c1, model_c2, model_c3, model_c4, model_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				chat = get_chat_module( )
				
				with model_c1:
					model_options = get_text_option_list( chat, 'model_options', [ '' ] )
					st.selectbox( label='Model',
						options=model_options,
						key='docqna_model',
						help='Provider model used for answering document questions.',
						index=None,
						placeholder='Options' )
				
				with model_c2:
					st.slider( label='Temperature',
						min_value=-2.0,
						max_value=2.0,
						value=float( st.session_state.get( 'docqna_temperature', 0.0 ) ),
						step=0.01,
						key='docqna_temperature',
						help=cfg.TEMPERATURE )
				
				with model_c3:
					st.slider( label='Top-P',
						min_value=0.0,
						max_value=1.0,
						value=float( st.session_state.get( 'docqna_top_percent', 0.0 ) ),
						step=0.01,
						key='docqna_top_percent',
						help=cfg.TOP_P )
				
				with model_c4:
					st.slider( label='Max Tokens',
						min_value=0,
						max_value=100000,
						value=int( st.session_state.get( 'docqna_max_tokens', 0 ) ),
						step=500,
						key='docqna_max_tokens',
						help=cfg.MAX_OUTPUT_TOKENS )
				
				with model_c5:
					st.toggle( label='Store',
						key='docqna_store',
						help=cfg.STORE )
			
			# ------------------------------------------------------------------
			# Chunk Controls
			# ------------------------------------------------------------------
			with st.expander( label='Chunk Controls', icon='🧩', expanded=False, width='stretch' ):
				chunk_c1, chunk_c2, chunk_c3, chunk_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				with chunk_c1:
					st.slider( label='Chunk Size',
						min_value=100,
						max_value=3000,
						value=int( st.session_state.get( 'docqna_chunk_size', 900 ) or 900 ),
						step=50,
						key='docqna_chunk_size',
						help='Maximum words per local retrieval chunk.' )
				
				with chunk_c2:
					st.slider( label='Chunk Overlap',
						min_value=0,
						max_value=1000,
						value=int( st.session_state.get( 'docqna_chunk_overlap', 150 ) or 150 ),
						step=25,
						key='docqna_chunk_overlap',
						help='Word overlap between local retrieval chunks.' )
				
				with chunk_c3:
					st.toggle( label='Diagnostics',
						key='docqna_show_diagnostics',
						help='Show retrieval/index diagnostics.' )
				
				with chunk_c4:
					if st.button( label='Rebuild Index',
							key='docqna_rebuild_index',
							width='stretch' ):
						rebuild_docqna_index( )
						st.success( st.session_state.get( 'docqna_index_status', 'Indexed' ) )
		
		# ------------------------------------------------------------------
		# System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			in_left, in_right = st.columns( [ 0.75, 0.25 ],
				border=True, gap='xxsmall' )
			
			with in_left:
				st.text_area( label='Enter Text',
					height=90,
					width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS,
					key='docqna_system_instructions' )
			
			with in_right:
				st.selectbox( label='Use Template',
					options=prompt_names,
					index=None,
					key='instructions',
					on_change=load_docqna_instruction )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			with btn_c1:
				st.button( label='Clear Instructions',
					width='stretch',
					on_click=clear_docqna_instructions )
			
			with btn_c2:
				st.button( label='XML <-> Markdown',
					width='stretch',
					on_click=convert_docqna_instructions )
		
		# ------------------------------------------------------------------
		# Document Loading
		# ------------------------------------------------------------------
		with st.expander( label='Document Loading', icon='📥', expanded=False, width='stretch' ):
			load_left, load_right = st.columns( [ 0.25, 0.75 ],
				border=True, gap='small' )
			
			with load_left:
				source_value = st.session_state.get( 'docqna_source', 'Local Upload' )
				
				if source_value == 'Local Upload':
					uploaded_docs = st.file_uploader(
						label='Upload',
						type=[ 'pdf', 'txt', 'md', 'docx', 'csv', 'json', 'xml',
						       'py', 'cs', 'sql', 'yaml', 'yml', 'html', 'css',
						       'js', 'ts' ],
						accept_multiple_files=bool(
							st.session_state.get( 'docqna_multi_mode', False ) ),
						label_visibility='visible',
						key='docqna_local_file_uploader' )
					
					if uploaded_docs is not None:
						docs = load_docqna_file( uploaded_docs )
						if len( docs ) > 0:
							st.success( f'Loaded {len( docs )} document(s).' )
					
					st.button( label='Unload Documents',
						key='docqna_unload_documents',
						width='stretch',
						on_click=unload_docqna_documents )
				
				else:
					st.info( 'Remote provider source selected. Use the source ID field above.' )
				
				if st.button( label='Summarize Active Source',
						key='docqna_summarize_active',
						width='stretch' ):
					with st.spinner( 'Summarizing active document source…' ):
						answer = summarize_document( )
						if isinstance( answer, str ) and answer.strip( ):
							st.session_state[ 'docqna_last_answer' ] = answer.strip( )
							st.session_state[ 'last_answer' ] = answer.strip( )
							st.session_state[ 'docqna_messages' ].append(
								{
										'role': 'assistant',
										'content': answer.strip( ),
								} )
			
			with load_right:
				source_value = st.session_state.get( 'docqna_source', 'Local Upload' )
				
				if source_value == 'Local Upload':
					render_document_preview( )
				else:
					st.json(
						{
								'provider': provider_name,
								'source': source_value,
								'file_id': st.session_state.get( 'docqna_file_id', '' ),
								'vector_store_id': st.session_state.get(
									'docqna_vector_store_id', '' ),
								'gemini_file_search_stores': split_text_values(
									st.session_state.get(
										'docqna_file_search_store_names_input', '' ),
									delimiter=',' ),
						} )
		
		# ------------------------------------------------------------------
		# Diagnostics
		# ------------------------------------------------------------------
		if bool( st.session_state.get( 'docqna_show_diagnostics', True ) ):
			with st.expander( label='Diagnostics', icon='🧪',
					expanded=False, width='stretch' ):
				diag_c1, diag_c2, diag_c3, diag_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				with diag_c1:
					st.metric( 'Documents',
						len( get_document_names( ) ) )
				
				with diag_c2:
					st.metric( 'Chunks',
						int( st.session_state.get( 'docqna_chunk_count', 0 ) or 0 ) )
				
				with diag_c3:
					st.metric( 'Indexed',
						'Yes' if st.session_state.get( 'docqna_vec_ready', False ) else 'No' )
				
				with diag_c4:
					st.metric( 'Source',
						st.session_state.get( 'docqna_source', 'Local Upload' ) )
				
				st.json(
					{
							'index_status': st.session_state.get(
								'docqna_index_status', 'Not indexed' ),
							'fingerprint': st.session_state.get(
								'docqna_fingerprint', '' ),
							'active_documents': get_document_names( ),
					} )
				
				render_hits( )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Messages
		# ------------------------------------------------------------------
		for msg in st.session_state.get( 'docqna_messages', [ ] ):
			if not isinstance( msg, dict ):
				continue
			
			role = msg.get( 'role', 'assistant' )
			avatar = docqna_avatar if role == 'assistant' else ''
			
			with st.chat_message( role, avatar=avatar ):
				st.markdown( msg.get( 'content', '' ) )
		
		prompt = st.chat_input( 'Ask a question about the active document source …' )
		
		if prompt is not None and str( prompt ).strip( ):
			prompt = str( prompt ).strip( )
			
			st.session_state[ 'docqna_messages' ].append(
				{
						'role': 'user',
						'content': prompt,
				} )
			
			with st.chat_message( 'assistant', avatar=docqna_avatar ):
				with st.spinner( 'Answering from the active document source…' ):
					try:
						answer = route_document_query( prompt )
						
						if isinstance( answer, str ) and answer.strip( ):
							st.markdown( answer )
							st.session_state[ 'docqna_messages' ].append( {
										'role': 'assistant',
										'content': answer.strip( ),
								} )
							st.session_state[ 'docqna_last_answer' ] = answer.strip( )
							st.session_state[ 'last_answer' ] = answer.strip( )
						else:
							message = 'No Document Q&A answer was returned.'
							st.warning( message )
							st.session_state[ 'docqna_messages' ].append( {
										'role': 'assistant',
										'content': message,
								} )
					except Exception as exc:
						st.error( f'Document Q&A failed: {exc}' )
		
		# ------------------------------------------------------------------
		# Last Answer and Sources
		# ------------------------------------------------------------------
		last_answer = st.session_state.get( 'docqna_last_answer', '' )
		if isinstance( last_answer, str ) and last_answer.strip( ):
			with st.expander( label='Last Document Answer', icon='🧠',
					expanded=False, width='stretch' ):
				st.markdown( last_answer )
		
		last_sources = st.session_state.get( 'docqna_last_sources', [ ] )
		if isinstance( last_sources, list ) and len( last_sources ) > 0:
			with st.expander( label='Last Document Sources', icon='📌', expanded=False,
					width='stretch' ):
				
				df_sources = pd.DataFrame( last_sources )
				st.data_editor( df_sources, use_container_width=True, hide_index=True )
		
		# ------------------------------------------------------------------
		# Reset Buttons
		# ------------------------------------------------------------------
		reset_c1, reset_c2, reset_c3 = st.columns(
			[ 0.34, 0.33, 0.33 ], border=True, gap='xxsmall' )
		
		with reset_c1:
			st.button( label='Clear Messages', key='docqna_clear_messages', width='stretch',
				on_click=clear_docqna_messages )
		
		with reset_c2:
			st.button( label='Clear Outputs', key='docqna_clear_outputs', width='stretch',
				on_click=clear_docqna_outputs )
		
		with reset_c3:
			st.button( label='Reset All', key='docqna_reset_all', width='stretch',
				on_click=reset_docqna_all )

# ======================================================================================
# FILES API MODE
# ======================================================================================
elif mode == 'Files':
	ensure_runtime_state( )
	if not isinstance( st.session_state.get( 'files_messages' ), list ):
		st.session_state[ 'files_messages' ] = [ ]
	
	if not isinstance( st.session_state.get( 'files_table_data' ), list ):
		st.session_state[ 'files_table_data' ] = [ ]
	
	if not isinstance( st.session_state.get( 'files_metadata' ), dict ):
		st.session_state[ 'files_metadata' ] = { }
	
	if not isinstance( st.session_state.get( 'files_delete_result' ), dict ):
		st.session_state[ 'files_delete_result' ] = { }
	
	if not isinstance( st.session_state.get( 'files_last_answer' ), str ):
		st.session_state[ 'files_last_answer' ] = ''
	
	if st.session_state.get( 'clear_instructions' ):
		st.session_state[ 'files_system_instructions' ] = ''
		st.session_state[ 'clear_instructions' ] = False
	
	provider_name = get_provider_name( )
	files = get_files_module( )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📁 Files API', help=cfg.FILES_API )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠',
				expanded=False, width='stretch' ):
			# ------------------------------------------------------------------
			# File Management Controls
			# ------------------------------------------------------------------
			with st.expander( label='File Management', icon='📂', expanded=False,
					width='stretch' ):
				
				mgmt_c1, mgmt_c2, mgmt_c3, mgmt_c4 = st.columns(
					[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
				
				with mgmt_c1:
					upload_purposes = get_purpose_options( files )
					if st.session_state.get( 'files_purpose' ) not in upload_purposes:
						st.session_state[ 'files_purpose' ] = upload_purposes[ 0 ] \
							if len( upload_purposes ) > 0 else 'user_data'
					
					st.selectbox( label='Upload Purpose',
						options=upload_purposes,
						key='files_purpose',
						help='Provider upload purpose.',
						index=None,
						placeholder='Options' )
				
				with mgmt_c2:
					filter_purposes = get_filter_options( files )
					if st.session_state.get( 'files_filter_purpose' ) not in filter_purposes:
						st.session_state[ 'files_filter_purpose' ] = ''
					
					st.selectbox( label='List Purpose Filter',
						options=filter_purposes,
						key='files_filter_purpose',
						help='Optional purpose filter used when listing files.',
						index=None,
						placeholder='Options' )
				
				with mgmt_c3:
					st.text_input( label='File ID',
						key='files_id',
						width='stretch',
						placeholder='file-... or Gemini file name',
						help='Provider file identifier for retrieve, extract, delete, or analysis.' )
				
				with mgmt_c4:
					st.button( label='Reset Controls',
						key='files_reset_controls',
						width='stretch',
						on_click=reset_files_controls )
			
			# ------------------------------------------------------------------
			# Analysis Controls
			# ------------------------------------------------------------------
			with st.expander( label='Analysis Controls', icon='🧊', expanded=False,
					width='stretch' ):
				
				analysis_c1, analysis_c2, analysis_c3, analysis_c4, analysis_c5 = st.columns(
					[ 0.20, 0.20, 0.20, 0.20, 0.20 ], border=True, gap='xxsmall' )
				
				chat = get_chat_module( )
				
				with analysis_c1:
					model_options = get_text_option_list( chat, 'model_options', [ '' ] )
					st.selectbox( label='Model',
						options=model_options,
						key='files_model',
						help='Model used for optional file analysis.',
						index=None,
						placeholder='Options' )
				
				with analysis_c2:
					st.slider( label='Temperature',
						min_value=-2.0,
						max_value=2.0,
						value=float( st.session_state.get( 'files_temperature', 0.0 ) ),
						step=0.01,
						key='files_temperature',
						help=cfg.TEMPERATURE )
				
				with analysis_c3:
					st.slider( label='Top-P',
						min_value=0.0,
						max_value=1.0,
						value=float( st.session_state.get( 'files_top_percent', 0.0 ) ),
						step=0.01,
						key='files_top_percent',
						help=cfg.TOP_P )
				
				with analysis_c4:
					st.slider( label='Max Tokens',
						min_value=0,
						max_value=100000,
						value=int( st.session_state.get( 'files_max_tokens', 0 ) ),
						step=500,
						key='files_max_tokens',
						help=cfg.MAX_OUTPUT_TOKENS )
				
				with analysis_c5:
					st.toggle( label='Store',
						key='files_store',
						help=cfg.STORE )
		
		# ------------------------------------------------------------------
		# System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False,
				width='stretch' ):
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ 'No Templates Found' ]
			
			in_left, in_right = st.columns( [ 0.75, 0.25 ], border=True, gap='xxsmall' )
			with in_left:
				st.text_area( label='Enter Text', height=90,
					width='stretch', help=cfg.SYSTEM_INSTRUCTIONS, key='files_system_instructions' )
			
			with in_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions', on_change=load_files_instruction )
			
			btn_c1, btn_c2 = st.columns( [ 0.8, 0.2 ] )
			
			with btn_c1:
				st.button( label='Clear Instructions', width='stretch',
					on_click=clear_files_instructions )
			
			with btn_c2:
				st.button( label='XML <-> Markdown', width='stretch',
					on_click=convert_files_instructions )
		
		# ------------------------------------------------------------------
		# File Operations
		# ------------------------------------------------------------------
		ops_left, ops_right = st.columns( [ 0.45, 0.55 ], border=True, gap='small' )
		
		with ops_left:
			st.caption( 'Upload' )
			uploaded_file = st.file_uploader( label='Upload file to provider Files API',
				type=[ 'pdf', 'txt', 'md', 'docx', 'csv', 'json', 'xml',
				       'png', 'jpg', 'jpeg', 'webp', 'py', 'cs', 'sql',
				       'yaml', 'yml', 'html', 'css', 'js', 'ts' ],
				accept_multiple_files=False, key='files_api_file_uploader' )
			
			if uploaded_file is not None:
				tmp_path = save_temp( uploaded_file )
				
				if st.button( label='Upload File', key='files_upload_button', width='stretch' ):
					if not tmp_path:
						st.warning( 'Could not save uploaded file for provider upload.' )
					else:
						with st.spinner( 'Uploading file…' ):
							try:
								result = upload_provider_file( files=files, path=tmp_path,
									purpose=st.session_state.get( 'files_purpose' ) )
								
								metadata = normalize_files_object( result )
								st.session_state[ 'files_last_upload' ] = metadata
								st.session_state[ 'files_metadata' ] = metadata
								st.session_state[ 'files_last_operation' ] = 'upload'
								
								file_id = get_files_id ( metadata )
								if file_id:
									st.session_state[ 'files_id' ] = file_id
								
								st.success( f'Uploaded file: {file_id or uploaded_file.name}' )
							except Exception as exc:
								st.error( f'Upload failed: {exc}' )
			
			st.caption( 'Operations' )
			op_c1, op_c2 = st.columns( [ 0.5, 0.5 ], gap='xxsmall' )
			with op_c1:
				if st.button( label='List Files', key='files_list_button', width='stretch' ):
					with st.spinner( 'Listing files…' ):
						try:
							rows = list_provider_files(
								files=files,
								purpose=st.session_state.get( 'files_filter_purpose' ) )
							
							st.session_state[ 'files_table_data' ] = rows
							st.session_state[ 'files_last_list' ] = rows
							st.session_state[ 'files_last_operation' ] = 'list'
							
							st.success( f'Found {len( rows )} file(s).' )
						except Exception as exc:
							st.error( f'List failed: {exc}' )
				
				if st.button( label='Retrieve Metadata', key='files_retrieve_button', width='stretch' ):
					file_id = st.session_state.get( 'files_id', '' )
					if not isinstance( file_id, str ) or not file_id.strip( ):
						st.warning( 'Enter or select a file ID before retrieving metadata.' )
					else:
						with st.spinner( 'Retrieving file metadata…' ):
							try:
								metadata = retrieve_provider_file(
									files=files,
									file_id=file_id )
								
								st.session_state[ 'files_metadata' ] = metadata
								st.session_state[ 'files_last_operation' ] = 'retrieve'
							except Exception as exc:
								st.error( f'Retrieve failed: {exc}' )
			
			with op_c2:
				if st.button( label='Extract Content',
						key='files_extract_button',
						width='stretch' ):
					file_id = st.session_state.get( 'files_id', '' )
					
					if not isinstance( file_id, str ) or not file_id.strip( ):
						st.warning( 'Enter or select a file ID before extracting content.' )
					else:
						with st.spinner( 'Extracting file content…' ):
							try:
								content = extract_file_content(
									files=files,
									file_id=file_id )
								
								st.session_state[ 'files_content' ] = content
								st.session_state[ 'files_last_operation' ] = 'extract'
								
								if not content:
									st.warning( 'No extractable content was returned.' )
							except Exception as exc:
								st.error( f'Extract failed: {exc}' )
				
				if st.button( label='Delete File',
						key='files_delete_button',
						width='stretch' ):
					file_id = st.session_state.get( 'files_id', '' )
					
					if not isinstance( file_id, str ) or not file_id.strip( ):
						st.warning( 'Enter or select a file ID before deleting.' )
					else:
						with st.spinner( 'Deleting file…' ):
							try:
								result = delete_provider_file(
									files=files,
									file_id=file_id )
								
								st.session_state[ 'files_delete_result' ] = result
								st.session_state[ 'files_last_operation' ] = 'delete'
								st.success( 'Delete request completed.' )
							except Exception as exc:
								st.error( f'Delete failed: {exc}' )
		
		with ops_right:
			st.caption( 'Results' )
			rows = st.session_state.get( 'files_table_data', [ ] )
			if isinstance( rows, list ) and len( rows ) > 0:
				df_files = pd.DataFrame( rows )
				st.data_editor( df_files, use_container_width=True, hide_index=True )
				options = build_selector_options( rows )
				if len( options ) > 0:
					selected_file = st.selectbox( label='Select Listed File',
						options=options,
						key='files_selected_option',
						index=None,
						placeholder='Options' )
					
					selected_id = get_option_id( selected_file )
					if selected_id:
						st.session_state[ 'files_id' ] = selected_id
			
			metadata = st.session_state.get( 'files_metadata', { } )
			if isinstance( metadata, dict ) and len( metadata ) > 0:
				with st.expander( label='Metadata', icon='🧾',
						expanded=False, width='stretch' ):
					st.json( metadata )
			
			content = st.session_state.get( 'files_content', '' )
			if isinstance( content, str ) and content.strip( ):
				with st.expander( label='Extracted Content', icon='📄',
						expanded=False, width='stretch' ):
					st.text_area( label='Content',
						value=content[ :20000 ],
						height=360,
						width='stretch',
						disabled=True )
			
			delete_result = st.session_state.get( 'files_delete_result', { } )
			if isinstance( delete_result, dict ) and len( delete_result ) > 0:
				with st.expander( label='Delete Result', icon='🗑️',
						expanded=False, width='stretch' ):
					st.json( delete_result )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# File Analysis Chat
		# ------------------------------------------------------------------
		for msg in st.session_state.get( 'files_messages', [ ] ):
			if not isinstance( msg, dict ):
				continue
			
			with st.chat_message( msg.get( 'role', 'assistant' ) ):
				st.markdown( msg.get( 'content', '' ) )
		
		prompt = st.chat_input( 'Ask a question about the selected file …' )
		if prompt is not None and str( prompt ).strip( ):
			prompt = str( prompt ).strip( )
			st.session_state[ 'files_messages' ].append( {
						'role': 'user',
						'content': prompt,
				} )
			
			with st.chat_message( 'assistant' ):
				with st.spinner( 'Analyzing selected file…' ):
					try:
						answer = analyze_provider_file( files=files, prompt=prompt,
							file_id=st.session_state.get( 'files_id', '' ),
							model=st.session_state.get( 'files_model' ) )
						
						if isinstance( answer, str ) and answer.strip( ):
							st.markdown( answer )
							st.session_state[ 'files_messages' ].append( {
										'role': 'assistant',
										'content': answer.strip( ),
								} )
							st.session_state[ 'files_last_answer' ] = answer.strip( )
							st.session_state[ 'last_answer' ] = answer.strip( )
							
							try:
								update_token_counters( getattr( files, 'response', None ) )
							except Exception:
								pass
						else:
							message = 'No file analysis response was returned.'
							st.warning( message )
							st.session_state[ 'files_messages' ].append( {
										'role': 'assistant',
										'content': message,
								} )
					except Exception as exc:
						st.error( f'File analysis failed: {exc}' )
		
		last_answer = st.session_state.get( 'files_last_answer', '' )
		if isinstance( last_answer, str ) and last_answer.strip( ):
			with st.expander( label='Last File Analysis', icon='🧠',
					expanded=False, width='stretch' ):
				st.markdown( last_answer )
		
		reset_c1, reset_c2, reset_c3 = st.columns( [ 0.34, 0.33, 0.33 ], border=True, gap='xxsmall' )
		with reset_c1:
			st.button( label='Clear Messages', key='clear_files_messages', width='stretch',
				on_click=clear_files_messages )
		
		with reset_c2:
			st.button( label='Clear Outputs', key='clear_files_mode_outputs', width='stretch',
				on_click=clear_files_outputs )
		
		with reset_c3:
			st.button( label='Reset All', key='reset_files_all', width='stretch',
				on_click=reset_files_all )

# ======================================================================================
# VECTOR STORES MODE
# ======================================================================================
elif mode == 'Vector Stores':
	ensure_vectorstores_mode_state( )
	provider_name = get_provider_name( )
	backend_name = get_vectorstores_backend_name( provider_name )
	backend_summary = get_storage_backend_summary( provider_name )
	
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '🧠 Vector Stores' )
		st.divider( )
		
		# ------------------------------------------------------------------
		# Routing / Backend Summary
		# ------------------------------------------------------------------
		with st.expander( label='Storage Routing', icon='🧭', expanded=True, width='stretch' ):
			route_c1, route_c2, route_c3, route_c4 = st.columns(
				[ 0.25, 0.25, 0.25, 0.25 ], border=True, gap='xxsmall' )
			
			with route_c1:
				st.markdown( f'**Provider:** {provider_name}' )
			
			with route_c2:
				if provider_name == 'Gemini':
					st.selectbox(
						label='Gemini Backend',
						options=[ 'File Search Stores', 'Cloud Buckets' ],
						key='stores_backend',
						help='Gemini backend exposed under Buddy’s Vector Stores alias.',
						index=None,
						placeholder='Options'
					)
					backend_name = get_vectorstores_backend_name( provider_name )
					backend_summary = get_storage_backend_summary( provider_name )
				else:
					st.markdown( f'**Backend:** {backend_name}' )
			
			with route_c3:
				st.markdown( f'**Concrete Backend:** {backend_name}' )
			
			with route_c4:
				st.button(
					label='Reset All',
					key='vectorstores_reset_all',
					width='stretch',
					on_click=reset_vectorstore_all
				)
			
			st.json( backend_summary )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# GPT: OpenAI Vector Stores
		# ------------------------------------------------------------------
		if provider_name == 'GPT':
			vectorstores = get_vectorstores_module( provider_name )
			
			ops_left, ops_right = st.columns( [ 0.42, 0.58 ], border=True, gap='small' )
			
			with ops_left:
				st.caption( 'OpenAI Vector Store Controls' )
				
				st.text_input( label='Vector Store Name', key='stores_name',
					width='stretch', placeholder='Federal Financial Regulations' )
				
				st.text_area(
					label='Metadata JSON',
					key='stores_metadata',
					height=90,
					width='stretch',
					placeholder='{ "domain": "appropriations" }'
				)
				
				st.text_input(
					label='Manual Vector Store ID',
					key='stores_manual_id',
					width='stretch',
					placeholder='vs_...'
				)
				
				st.text_input(
					label='File ID',
					key='stores_file_id',
					width='stretch',
					placeholder='file-...'
				)
				
				st.text_area(
					label='Batch File IDs',
					key='stores_file_ids_text',
					height=70,
					width='stretch',
					placeholder='file-a, file-b, file-c'
				)
				
				uploaded_store_file = st.file_uploader(
					label='Upload File and Attach',
					type=[
							'pdf', 'txt', 'md', 'docx', 'csv', 'json', 'xml',
							'png', 'jpg', 'jpeg', 'webp', 'py', 'cs', 'sql',
							'yaml', 'yml', 'html', 'css', 'js', 'ts'
					],
					accept_multiple_files=False,
					key='stores_file_upload'
				)
				
				action_c1, action_c2 = st.columns( [ 0.50, 0.50 ], gap='xxsmall' )
				
				with action_c1:
					if st.button( label='Create Store', key='gpt_stores_create', width='stretch' ):
						with st.spinner( 'Creating OpenAI vector store…' ):
							try:
								name = require_storage_value(
									'Vector Store Name',
									st.session_state.get( 'stores_name', '' )
								)
								metadata = parse_storage_json(
									st.session_state.get( 'stores_metadata', '' )
								)
								
								try:
									result = call_storage_method(
										vectorstores,
										[ 'create', 'create_store' ],
										name=name,
										metadata=metadata if metadata else None
									)
								except AttributeError:
									raise
								except TypeError:
									result = call_storage_method(
										vectorstores,
										[ 'create', 'create_store' ],
										name
									)
								
								normalized = set_storage_result(
									result,
									operation='create_openai_vector_store',
									result_key='stores_store_metadata'
								)
								
								identifier = get_storage_identifier( normalized )
								if identifier:
									st.session_state[ 'stores_id' ] = identifier
								
								st.success( f'Created vector store: {identifier or name}' )
							except Exception as exc:
								st.error( f'Create failed: {exc}' )
					
					if st.button(
							label='Retrieve Store',
							key='gpt_stores_retrieve',
							width='stretch' ):
						with st.spinner( 'Retrieving OpenAI vector store…' ):
							try:
								store_id = require_storage_value(
									'Vector Store ID',
									get_vectorstores_selected_id( )
								)
								
								result = call_storage_method(
									vectorstores,
									[ 'retrieve', 'retrieve_store', 'get' ],
									store_id
								)
								
								set_storage_result(
									result,
									operation='retrieve_openai_vector_store',
									result_key='stores_store_metadata'
								)
							except Exception as exc:
								st.error( f'Retrieve failed: {exc}' )
					
					if st.button(
							label='Create File Batch',
							key='gpt_stores_batch',
							width='stretch' ):
						with st.spinner( 'Creating OpenAI vector store file batch…' ):
							try:
								store_id = require_storage_value(
									'Vector Store ID',
									get_vectorstores_selected_id( )
								)
								file_ids = parse_storage_ids(
									st.session_state.get( 'stores_file_ids_text', '' )
								)
								
								if len( file_ids ) == 0:
									raise ValueError( 'At least one file ID is required.' )
								
								result = call_storage_method(
									vectorstores,
									[
											'create_file_batch',
											'batch',
											'create_batch',
											'attach_files',
											'add_files'
									],
									vector_store_id=store_id,
									file_ids=file_ids
								)
								
								set_storage_result(
									result,
									operation='create_openai_file_batch',
									result_key='stores_batch_result'
								)
								st.success( 'File batch request completed.' )
							except Exception as exc:
								st.error( f'Batch failed: {exc}' )
				
				with action_c2:
					if st.button( label='List Stores', key='gpt_stores_list', width='stretch' ):
						with st.spinner( 'Listing OpenAI vector stores…' ):
							try:
								result = call_storage_method(
									vectorstores,
									[ 'list_stores', 'list', 'list_collections' ]
								)
								
								rows = set_storage_rows( result, table_key='stores_table' )
								st.success( f'Found {len( rows )} vector store(s).' )
							except Exception as exc:
								st.error( f'List failed: {exc}' )
					
					if st.button(
							label='Delete Store',
							key='gpt_stores_delete',
							width='stretch' ):
						with st.spinner( 'Deleting OpenAI vector store…' ):
							try:
								store_id = require_storage_value(
									'Vector Store ID',
									get_vectorstores_selected_id( )
								)
								
								result = call_storage_method(
									vectorstores,
									[ 'delete', 'delete_store' ],
									store_id
								)
								
								set_storage_result(
									result,
									operation='delete_openai_vector_store',
									result_key='stores_delete_result'
								)
								st.success( 'Delete request completed.' )
							except Exception as exc:
								st.error( f'Delete failed: {exc}' )
					
					if st.button(
							label='Attach Existing File',
							key='gpt_stores_attach_file',
							width='stretch' ):
						with st.spinner( 'Attaching file to OpenAI vector store…' ):
							try:
								store_id = require_storage_value(
									'Vector Store ID',
									get_vectorstores_selected_id( )
								)
								file_id = require_storage_value(
									'File ID',
									st.session_state.get( 'stores_file_id', '' )
								)
								
								result = call_storage_method(
									vectorstores,
									[
											'attach_file',
											'add_file',
											'create_file',
											'upload_file'
									],
									vector_store_id=store_id,
									file_id=file_id
								)
								
								set_storage_result(
									result,
									operation='attach_openai_file',
									result_key='stores_file_metadata'
								)
								st.success( 'File attachment request completed.' )
							except Exception as exc:
								st.error( f'Attach failed: {exc}' )
				
				if st.button(
						label='Upload and Attach File',
						key='gpt_stores_upload_attach',
						width='stretch' ):
					with st.spinner( 'Uploading and attaching file…' ):
						try:
							store_id = require_storage_value(
								'Vector Store ID',
								get_vectorstores_selected_id( )
							)
							path = save_uploaded_storage_file( uploaded_store_file )
							
							result = call_storage_method(
								vectorstores,
								[
										'upload_and_attach',
										'upload_file',
										'attach_upload',
										'upload'
								],
								vector_store_id=store_id,
								path=path
							)
							
							set_storage_result(
								result,
								operation='upload_attach_openai_file',
								result_key='stores_upload_result'
							)
							st.success( 'Upload and attach request completed.' )
						except Exception as exc:
							st.error( f'Upload and attach failed: {exc}' )
			
			with ops_right:
				st.caption( 'OpenAI Vector Store Results' )
				
				rows = st.session_state.get( 'stores_table', [ ] )
				if isinstance( rows, list ) and len( rows ) > 0:
					st.data_editor( pd.DataFrame( rows ), use_container_width=True,
						hide_index=True )
					
					options = build_storage_selectors( rows )
					if len( options ) > 0:
						selected = st.selectbox(
							label='Select Vector Store',
							options=options,
							key='storage_selected_option',
							index=None,
							placeholder='Options'
						)
						sync_storage_selection( selected, provider_name='GPT' )
				
				result = st.session_state.get( 'storage_operation_result', { } )
				if isinstance( result, dict ) and len( result ) > 0:
					with st.expander(
							label='Operation Result',
							icon='🧾',
							expanded=True,
							width='stretch' ):
						st.json( result )
		
		# ------------------------------------------------------------------
		# Grok: xAI Collections
		# ------------------------------------------------------------------
		elif provider_name == 'Grok':
			vectorstores = get_vectorstores_module( provider_name )
			
			ops_left, ops_right = st.columns( [ 0.42, 0.58 ], border=True, gap='small' )
			
			with ops_left:
				st.caption( 'xAI Collection Controls' )
				
				st.text_input(
					label='Manual Collection ID',
					key='stores_manual_id',
					width='stretch',
					placeholder='collection_...'
				)
				
				st.text_area(
					label='Search / Survey Prompt',
					key='stores_query',
					height=110,
					width='stretch',
					placeholder='Search or survey configured xAI collections...'
				)
				
				st.text_area(
					label='Survey Collection IDs',
					key='stores_file_ids_text',
					height=70,
					width='stretch',
					placeholder='collection-a, collection-b, collection-c'
				)
				
				grok_c1, grok_c2 = st.columns( [ 0.50, 0.50 ], gap='xxsmall' )
				
				with grok_c1:
					if st.button(
							label='List Collections',
							key='grok_stores_list',
							width='stretch' ):
						with st.spinner( 'Listing configured xAI collections…' ):
							try:
								rows = get_grok_collections( vectorstores )
								set_storage_rows( rows, table_key='stores_table' )
								st.success( f'Found {len( rows )} configured collection(s).' )
							except Exception as exc:
								st.error( f'List failed: {exc}' )
					
					if st.button(
							label='Retrieve Collection',
							key='grok_stores_retrieve',
							width='stretch' ):
						with st.spinner( 'Retrieving xAI collection metadata…' ):
							try:
								collection_id = require_storage_value(
									'Collection ID',
									get_vectorstores_selected_id( )
								)
								
								result = call_storage_method(
									vectorstores,
									[
											'retrieve',
											'retrieve_collection',
											'get_collection',
											'get'
									],
									collection_id
								)
								
								set_storage_result(
									result,
									operation='retrieve_grok_collection',
									result_key='stores_store_metadata'
								)
							except Exception as exc:
								st.error( f'Retrieve failed: {exc}' )
					
					if st.button(
							label='Create Collection',
							key='grok_stores_create',
							width='stretch' ):
						warn_grok_unsupported_operation( 'collection creation' )
				
				with grok_c2:
					if st.button(
							label='Search Collection',
							key='grok_stores_search',
							width='stretch' ):
						with st.spinner( 'Searching xAI collection…' ):
							try:
								collection_id = require_storage_value(
									'Collection ID',
									get_vectorstores_selected_id( )
								)
								query = require_storage_value(
									'Search / Survey Prompt',
									st.session_state.get( 'stores_query', '' )
								)
								
								result = call_storage_method(
									vectorstores,
									[ 'search', 'search_collection', 'query_collection' ],
									prompt=query,
									store_id=collection_id
								)
								
								if isinstance( result, str ):
									st.session_state[ 'stores_answer' ] = result
									set_storage_result(
										{ 'answer': result },
										operation='search_grok_collection',
										result_key='stores_search_result'
									)
								else:
									set_storage_result(
										result,
										operation='search_grok_collection',
										result_key='stores_search_result'
									)
								
								st.success( 'Search request completed.' )
							except Exception as exc:
								st.error( f'Search failed: {exc}' )
					
					if st.button(
							label='Survey Collections',
							key='grok_stores_survey',
							width='stretch' ):
						with st.spinner( 'Surveying xAI collections…' ):
							try:
								query = require_storage_value(
									'Search / Survey Prompt',
									st.session_state.get( 'stores_query', '' )
								)
								
								store_ids = parse_storage_ids(
									st.session_state.get( 'stores_file_ids_text', '' )
								)
								
								if len( store_ids ) == 0:
									selected_id = get_vectorstores_selected_id( )
									if selected_id:
										store_ids = [ selected_id ]
								
								if len( store_ids ) == 0:
									raise ValueError(
										'At least one collection ID is required for survey.' )
								
								result = call_storage_method(
									vectorstores,
									[ 'survey', 'survey_collections' ],
									prompt=query,
									store_ids=store_ids
								)
								
								if isinstance( result, str ):
									st.session_state[ 'stores_answer' ] = result
									set_storage_result(
										{
												'answer': result,
												'store_ids': store_ids,
										},
										operation='survey_grok_collections',
										result_key='stores_survey_result'
									)
								else:
									rows = normalize_storage_rows( result )
									if len( rows ) > 0:
										set_storage_rows( rows, table_key='stores_table' )
									
									set_storage_result(
										result,
										operation='survey_grok_collections',
										result_key='stores_survey_result'
									)
								
								st.success( 'Survey completed.' )
							except Exception as exc:
								st.error( f'Survey failed: {exc}' )
					
					if st.button(
							label='Upload / Attach',
							key='grok_stores_upload',
							width='stretch' ):
						warn_grok_unsupported_operation( 'upload-to-collection' )
					
					if st.button(
							label='Delete Collection',
							key='grok_stores_delete',
							width='stretch' ):
						warn_grok_unsupported_operation( 'collection deletion' )
			
			with ops_right:
				st.caption( 'xAI Collection Results' )
				
				rows = st.session_state.get( 'stores_table', [ ] )
				if isinstance( rows, list ) and len( rows ) > 0:
					st.data_editor( pd.DataFrame( rows ), use_container_width=True,
						hide_index=True )
					
					options = build_storage_selectors( rows )
					if len( options ) > 0:
						selected = st.selectbox(
							label='Select Collection',
							options=options,
							key='storage_selected_option',
							index=None,
							placeholder='Options'
						)
						sync_storage_selection( selected, provider_name='Grok' )
				
				answer = st.session_state.get( 'stores_answer', '' )
				if isinstance( answer, str ) and answer.strip( ):
					with st.expander(
							label='Search / Survey Answer',
							icon='🧠',
							expanded=True,
							width='stretch' ):
						st.markdown( answer )
				
				result = st.session_state.get( 'storage_operation_result', { } )
				if isinstance( result, dict ) and len( result ) > 0:
					with st.expander(
							label='Operation Result',
							icon='🧾',
							expanded=True,
							width='stretch' ):
						st.json( result )
		
		# ------------------------------------------------------------------
		# Gemini: File Search Stores
		# ------------------------------------------------------------------
		elif provider_name == 'Gemini' and get_gemini_vector_backend( ) == 'File Search Stores':
			filestore = get_vectorstores_module( provider_name, backend='File Search Stores' )
			
			ops_left, ops_right = st.columns( [ 0.42, 0.58 ], border=True, gap='small' )
			
			with ops_left:
				st.caption( 'Gemini File Search Store Controls' )
				
				st.text_input(
					label='File Search Store Name',
					key='filestore_name',
					width='stretch',
					placeholder='Federal Financial Regulations'
				)
				
				st.text_input(
					label='File Search Store Resource Name',
					key='filestore_id',
					width='stretch',
					placeholder='fileSearchStores/...'
				)
				
				st.info(
					'The current Gemini FileSearch wrapper exposes create, list, retrieve, '
					'and delete. File upload/import is intentionally disabled here until '
					'the wrapper adds a media.uploadToFileSearchStore method.'
				)
				
				fs_c1, fs_c2 = st.columns( [ 0.50, 0.50 ], gap='xxsmall' )
				
				with fs_c1:
					if st.button(
							label='Create Store',
							key='gemini_fs_create',
							width='stretch' ):
						with st.spinner( 'Creating Gemini File Search Store…' ):
							try:
								name = require_storage_value(
									'File Search Store Name',
									st.session_state.get( 'filestore_name', '' )
								)
								
								result = call_storage_method(
									filestore,
									[
											'create',
											'create_store',
											'create_file_search_store'
									],
									name
								)
								
								normalized = set_storage_result(
									result,
									operation='create_gemini_file_search_store',
									result_key='filestore_operation_result'
								)
								
								identifier = get_storage_identifier( normalized )
								if identifier:
									st.session_state[ 'filestore_id' ] = identifier
								
								st.success( f'Created File Search Store: {identifier or name}' )
							except Exception as exc:
								st.error( f'Create failed: {exc}' )
					
					if st.button(
							label='Retrieve Store',
							key='gemini_fs_retrieve',
							width='stretch' ):
						with st.spinner( 'Retrieving Gemini File Search Store…' ):
							try:
								store_id = require_storage_value(
									'File Search Store Resource Name',
									get_vectorstores_selected_id( )
								)
								
								result = call_storage_method(
									filestore,
									[
											'retrieve',
											'get',
											'get_store',
											'get_file_search_store'
									],
									store_id
								)
								
								set_storage_result(
									result,
									operation='retrieve_gemini_file_search_store',
									result_key='filestore_metadata'
								)
							except Exception as exc:
								st.error( f'Retrieve failed: {exc}' )
				
				with fs_c2:
					if st.button(
							label='List Stores',
							key='gemini_fs_list',
							width='stretch' ):
						with st.spinner( 'Listing Gemini File Search Stores…' ):
							try:
								result = call_storage_method(
									filestore,
									[
											'list',
											'list_stores',
											'list_file_search_stores'
									]
								)
								
								rows = set_storage_rows( result, table_key='filestore_table' )
								st.success( f'Found {len( rows )} File Search Store(s).' )
							except Exception as exc:
								st.error( f'List failed: {exc}' )
					
					if st.button(
							label='Delete Store',
							key='gemini_fs_delete',
							width='stretch' ):
						with st.spinner( 'Deleting Gemini File Search Store…' ):
							try:
								store_id = require_storage_value(
									'File Search Store Resource Name',
									get_vectorstores_selected_id( )
								)
								
								result = call_storage_method(
									filestore,
									[
											'delete',
											'delete_store',
											'delete_file_search_store'
									],
									store_id
								)
								
								set_storage_result(
									result,
									operation='delete_gemini_file_search_store',
									result_key='filestore_delete_result'
								)
								st.success( 'Delete request completed.' )
							except Exception as exc:
								st.error( f'Delete failed: {exc}' )
			
			with ops_right:
				st.caption( 'Gemini File Search Store Results' )
				
				rows = st.session_state.get( 'filestore_table', [ ] )
				if isinstance( rows, list ) and len( rows ) > 0:
					st.data_editor( pd.DataFrame( rows ), use_container_width=True,
						hide_index=True )
					
					options = build_storage_selectors( rows )
					if len( options ) > 0:
						selected = st.selectbox(
							label='Select File Search Store',
							options=options,
							key='storage_selected_option',
							index=None,
							placeholder='Options'
						)
						sync_storage_selection(
							selected,
							provider_name='Gemini',
							backend='File Search Stores'
						)
				
				result = st.session_state.get( 'storage_operation_result', { } )
				if isinstance( result, dict ) and len( result ) > 0:
					with st.expander(
							label='Operation Result',
							icon='🧾',
							expanded=True,
							width='stretch' ):
						st.json( result )
		
		# ------------------------------------------------------------------
		# Gemini: Cloud Buckets
		# ------------------------------------------------------------------
		elif provider_name == 'Gemini' and get_gemini_vector_backend( ) == 'Cloud Buckets':
			buckets = get_vectorstores_module( provider_name, backend='Cloud Buckets' )
			
			ops_left, ops_right = st.columns( [ 0.42, 0.58 ], border=True, gap='small' )
			
			with ops_left:
				st.caption( 'Gemini Cloud Bucket Object Controls' )
				
				st.text_input(
					label='Bucket Name',
					key='bucket_name',
					width='stretch',
					placeholder='my-bucket-name'
				)
				
				st.text_input(
					label='Object Name',
					key='bucket_object_name',
					width='stretch',
					placeholder='folder/document.pdf'
				)
				
				uploaded_bucket_file = st.file_uploader(
					label='Upload Object',
					type=[
							'pdf', 'txt', 'md', 'docx', 'csv', 'json', 'xml',
							'png', 'jpg', 'jpeg', 'webp', 'py', 'cs', 'sql',
							'yaml', 'yml', 'html', 'css', 'js', 'ts'
					],
					accept_multiple_files=False,
					key='bucket_file_upload'
				)
				
				st.info(
					'Bucket creation is intentionally not exposed here because the current '
					'CloudBuckets wrapper create(...) method is not a bucket-creation method. '
					'This backend manages objects inside an existing bucket.'
				)
				
				bucket_c1, bucket_c2 = st.columns( [ 0.50, 0.50 ], gap='xxsmall' )
				
				with bucket_c1:
					if st.button(
							label='List Objects',
							key='gemini_bucket_list_objects',
							width='stretch' ):
						with st.spinner( 'Listing Google Cloud bucket objects…' ):
							try:
								bucket_name = require_storage_value(
									'Bucket Name',
									st.session_state.get( 'bucket_name', '' )
								)
								
								result = call_storage_method(
									buckets,
									[ 'list', 'list_objects', 'list_blobs' ],
									bucket_name
								)
								
								rows = set_storage_rows( result, table_key='bucket_table' )
								st.success( f'Found {len( rows )} object(s).' )
							except Exception as exc:
								st.error( f'List objects failed: {exc}' )
					
					if st.button(
							label='Retrieve Object',
							key='gemini_bucket_retrieve_object',
							width='stretch' ):
						with st.spinner( 'Retrieving Google Cloud bucket object…' ):
							try:
								bucket_name = require_storage_value(
									'Bucket Name',
									st.session_state.get( 'bucket_name', '' )
								)
								object_name = require_storage_value(
									'Object Name',
									st.session_state.get( 'bucket_object_name', '' )
								)
								
								result = call_storage_method(
									buckets,
									[
											'retrieve',
											'get',
											'download',
											'get_object',
											'get_blob'
									],
									bucket_name,
									object_name
								)
								
								set_storage_result(
									result,
									operation='retrieve_gemini_cloud_object',
									result_key='bucket_metadata'
								)
							except Exception as exc:
								st.error( f'Retrieve failed: {exc}' )
				
				with bucket_c2:
					if st.button( label='Upload Object', key='gemini_bucket_upload_object',
							width='stretch' ):
						with st.spinner( 'Uploading object to Google Cloud bucket…' ):
							try:
								bucket_name = require_storage_value( 'Bucket Name',
									st.session_state.get( 'bucket_name', '' ) )
								
								object_name = st.session_state.get( 'bucket_object_name', '' )
								if not isinstance( object_name, str ) or not object_name.strip( ):
									object_name = getattr( uploaded_bucket_file, 'name', '' )
								
								object_name = require_storage_value( 'Object Name', object_name )
								path = save_uploaded_storage_file( uploaded_bucket_file )
								result = call_storage_method( buckets,
									[ 'upload', 'upload_file', 'upload_blob', 'upload_object' ],
									path=path, bucket=bucket_name, name=object_name )
								
								set_storage_result( result, operation='upload_gemini_cloud_object',
									result_key='bucket_upload_result' )
								
								st.success( 'Object upload completed.' )
							except Exception as exc:
								st.error( f'Upload failed: {exc}' )
					
					if st.button( label='Delete Object', key='gemini_bucket_delete_object',
							width='stretch' ):
						with st.spinner( 'Deleting Google Cloud bucket object…' ):
							try:
								bucket_name = require_storage_value( 'Bucket Name',
									st.session_state.get( 'bucket_name', '' ) )
								
								object_name = require_storage_value( 'Object Name',
									st.session_state.get( 'bucket_object_name', '' ) )
								
								result = call_storage_method( buckets, [
										'delete',
										'delete_object',
										'delete_blob',
										'delete_file'
								],
									bucket_name, object_name )
								
								set_storage_result( result,
									operation='delete_gemini_cloud_object',
									result_key='bucket_delete_result' )
								
								st.success( 'Object delete request completed.' )
							except Exception as exc:
								st.error( f'Delete failed: {exc}' )
			
			with ops_right:
				st.caption( 'Gemini Cloud Bucket Object Results' )
				
				rows = st.session_state.get( 'bucket_table', [ ] )
				if isinstance( rows, list ) and len( rows ) > 0:
					st.data_editor( pd.DataFrame( rows ), use_container_width=True,
						hide_index=True )
					
					options = build_storage_selectors( rows )
					if len( options ) > 0:
						selected = st.selectbox( label='Select Object / Bucket Row',
							options=options, key='storage_selected_option', index=None,
							placeholder='Options' )
						
						sync_storage_selection( selected, provider_name='Gemini',
							backend='Cloud Buckets' )
				
				result = st.session_state.get( 'storage_operation_result', { } )
				if isinstance( result, dict ) and len( result ) > 0:
					with st.expander( label='Operation Result', icon='🧾', expanded=True,
							width='stretch' ):
						st.json( result )
		
		# ------------------------------------------------------------------
		# Unsupported Branch
		# ------------------------------------------------------------------
		else:
			st.warning( f'{provider_name} does not expose a supported Vector Stores backend in '
			            f'the current Buddy configuration.' )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Shared Output / Reset Controls
		# ------------------------------------------------------------------
		shared_c1, shared_c2 = st.columns( [ 0.50, 0.50 ], border=True, gap='xxsmall' )
		with shared_c1:
			st.button( label='Clear Outputs', key='vectorstores_clear_outputs',
				width='stretch', on_click=clear_vectorstore_outputs )
		
		with shared_c2:
			st.button( label='Reset Controls', key='vectorstores_reset_controls',
				width='stretch', on_click=reset_vectorstore_controls )
		
		last_operation = st.session_state.get( 'storage_last_operation', '' )
		if isinstance( last_operation, str ) and last_operation.strip( ):
			st.caption( f'Last operation: {last_operation}' )

# ======================================================================================
# PROMPT ENGINEERING MODE
# ======================================================================================
elif mode == 'Prompt Engineering':
	import sqlite3
	import math
	
	TABLE = 'Prompts'
	PAGE_SIZE = 10
	st.session_state.setdefault( 'pe_cascade_enabled', False )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📝 Prompt Engineering', help=cfg.PROMPT_ENGINEERING )
		st.divider( )
		st.checkbox( 'Cascade selection into System Instructions', key='pe_cascade_enabled' )
		
		# ------------------------------------------------------------------
		# Session state
		# ------------------------------------------------------------------
		st.session_state.setdefault( 'pe_page', 1 )
		st.session_state.setdefault( 'pe_search', '' )
		st.session_state.setdefault( 'pe_sort_col', 'PromptsId' )
		st.session_state.setdefault( 'pe_sort_dir', 'ASC' )
		st.session_state.setdefault( 'pe_selected_id', None )
		st.session_state.setdefault( 'pe_caption', '' )
		st.session_state.setdefault( 'pe_name', '' )
		st.session_state.setdefault( 'pe_text', '' )
		st.session_state.setdefault( 'pe_version', '' )
		st.session_state.setdefault( 'pe_id', 0 )
		
		# ------------------------------------------------------------------
		# DB helpers
		# ------------------------------------------------------------------
		def get_conn( ):
			return sqlite3.connect( cfg.DB_PATH )
		
		def reset_selection( ):
			st.session_state.pe_selected_id = None
			st.session_state.pe_caption = ''
			st.session_state.pe_name = ''
			st.session_state.pe_text = ''
			st.session_state.pe_version = ''
			st.session_state.pe_id = 0
		
		def load_prompt( pid: int ) -> None:
			with get_conn( ) as conn:
				_select = f"SELECT Caption, Name, Text, Version, ID FROM {TABLE} WHERE PromptsId=?"
				cur = conn.execute( _select, (pid,), )
				row = cur.fetchone( )
				if not row:
					return
				st.session_state.pe_caption = row[ 0 ]
				st.session_state.pe_name = row[ 1 ]
				st.session_state.pe_text = row[ 2 ]
				st.session_state.pe_version = row[ 3 ]
				st.session_state.pe_id = row[ 4 ]
		
		# ------------------------------------------------------------------
		# Filters
		# ------------------------------------------------------------------
		c1, c2, c3, c4 = st.columns( [ 4, 2, 2, 3 ] )
		
		with c1:
			st.text_input( 'Search (Name/Text contains)', key='pe_search' )
		
		with c2:
			st.selectbox( 'Sort by', [ 'PromptsId', 'Caption', 'Name', 'Text', 'Version', 'ID' ],
				key='pe_sort_col', )
		
		with c3:
			st.selectbox( 'Direction', [ 'ASC', 'DESC' ], key='pe_sort_dir' )
		
		with c4:
			st.markdown(
				"<div style='font-size:0.95rem;font-weight:600;margin-bottom:0.25rem;'>Go to ID</div>",
				unsafe_allow_html=True, )
			
			a1, a2, a3 = st.columns( [ 2, 1, 1 ] )
			
			with a1:
				jump_id = st.number_input( 'Go to ID', min_value=1,
					step=1, label_visibility='collapsed', )
			
			with a2:
				if st.button( 'Go' ):
					st.session_state.pe_selected_id = int( jump_id )
					load_prompt( int( jump_id ) )
			
			with a3:
				st.button( 'Clear', on_click=reset_selection )
		
		# ------------------------------------------------------------------
		# Load prompt table
		# ------------------------------------------------------------------
		where = ""
		params = [ ]
		if st.session_state.pe_search:
			where = 'WHERE Name LIKE ? OR Text LIKE ?'
			s = f"%{st.session_state.pe_search}%"
			params.extend( [ s, s ] )
		
		offset = (st.session_state.pe_page - 1) * PAGE_SIZE
		query = f"""
	        SELECT PromptsId, Caption, Name, Text, Version, ID
	        FROM {TABLE}
	        {where}
	        ORDER BY {st.session_state.pe_sort_col} {st.session_state.pe_sort_dir}
	        LIMIT {PAGE_SIZE} OFFSET {offset}
	    """
		
		count_query = f"SELECT COUNT(*) FROM {TABLE} {where}"
		
		with get_conn( ) as conn:
			rows = conn.execute( query, params ).fetchall( )
			total_rows = conn.execute( count_query, params ).fetchone( )[ 0 ]
		
		total_pages = max( 1, math.ceil( total_rows / PAGE_SIZE ) )
		
		# ------------------------------------------------------------------
		# Prompt table
		# ------------------------------------------------------------------
		table_rows = [ ]
		for r in rows:
			table_rows.append(
			{
					'Selected': r[ 0 ] == st.session_state.pe_selected_id,
					'PromptsId': r[ 0 ],
					'Caption': r[ 1 ],
					'Name': r[ 2 ],
					'Text': r[ 3 ],
					'Version': r[ 4 ],
					'ID': r[ 5 ],
			} )
		
		edited = st.data_editor( table_rows, hide_index=True, use_container_width=True,
			key="prompt_table", )
		
		# ------------------------------------------------------------------
		# SELECTION PROCESSING (must run BEFORE widgets below)
		# ------------------------------------------------------------------
		selected = [ r for r in edited if isinstance( r, dict ) and r.get( 'Selected' ) ]
		if len( selected ) == 1:
			pid = int( selected[ 0 ][ 'PromptsId' ] )
			if pid != st.session_state.pe_selected_id:
				st.session_state.pe_selected_id = pid
				load_prompt( pid )
		
		elif len( selected ) == 0:
			reset_selection( )
		
		elif len( selected ) > 1:
			st.warning( 'Select exactly one prompt row.' )
		
		# ------------------------------------------------------------------
		# Paging
		# ------------------------------------------------------------------
		p1, p2, p3 = st.columns( [ 0.25, 3.5, 0.25 ] )
		with p1:
			if st.button( "◀ Prev" ) and st.session_state.pe_page > 1:
				st.session_state.pe_page -= 1
		
		with p2:
			st.markdown( f"Page **{st.session_state.pe_page}** of **{total_pages}**" )
		
		with p3:
			if st.button( "Next ▶" ) and st.session_state.pe_page < total_pages:
				st.session_state.pe_page += 1
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )

		
		# ------------------------------------------------------------------
		# Edit Prompt
		# ------------------------------------------------------------------
		with st.expander( "🖊️ Edit Prompt", expanded=False ):
			st.text_input( "PromptsId", value=st.session_state.pe_selected_id or "",
				disabled=True, )
			st.text_input( 'Name', key='pe_name' )
	
			st.text_area( 'Text', key='pe_text', height=260 )
			st.text_input( 'Version', key='pe_version' )
			c1, c2, c3 = st.columns( 3 )
			
			with c1:
				if st.button( '💾 Save Changes'
				if st.session_state.pe_selected_id
				else '➕ Create Prompt' ):
					with get_conn( ) as conn:
						if st.session_state.pe_selected_id:
							conn.execute(
								f"""
	                            UPDATE {TABLE}
	                            SET Caption=?, Name=?, Text=?, Version=?, ID=?
	                            WHERE PromptsId=?
	                            """,
								(
										st.session_state.pe_caption,
										st.session_state.pe_name,
										st.session_state.pe_text,
										st.session_state.pe_version,
										st.session_state.pe_id,
										st.session_state.pe_selected_id
								), )
						else:
							conn.execute(
								f"""
	                            INSERT INTO {TABLE} (Caption, Name, Text, Version, ID)
	                            VALUES (?, ?, ?, ? , ?)
	                            """,
								(
										st.session_state.pe_caption,
										st.session_state.pe_name,
										st.session_state.pe_text,
										st.session_state.pe_version,
										st.session_state.pe_id
								),
							)
						conn.commit( )
					
					st.success( 'Saved.' )
					reset_selection( )
			
			with c2:
				if st.session_state.pe_selected_id and st.button( 'Delete' ):
					with get_conn( ) as conn:
						conn.execute(
							f'DELETE FROM {TABLE} WHERE PromptsId=?',
							(st.session_state.pe_selected_id,), )
						conn.commit( )
					reset_selection( )
					st.success( 'Deleted.' )
			
			with c3:
				st.button( '🧹 Clear Selection', on_click=reset_selection )
				
# ==============================================================================
# EXPORT MODE
# ==============================================================================
elif mode == 'Data Export':
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '📭  Export' )
		st.divider( )
		
		# -----------------------------------
		# Prompt export (System Instructions)
		st.caption( 'System Prompt' )
		export_format = st.radio( 'Export Format', options=[ 'XML-Delimited', 'Markdown' ],
			horizontal=True, help='Choose how system instructions should be exported.' )
		prompt_text: str = st.session_state.get( 'system_prompt', '' )
		if export_format == 'Markdown':
			try:
				export_text: str = convert_xml( prompt_text )
				export_filename: str = 'Buddy_Instructions.md'
			except Exception as exc:
				st.error( f'Markdown conversion failed: {exc}' )
				export_text = ''
				export_filename = ''
		else:
			export_text = prompt_text
			export_filename = 'Buddy_System_Instructions.xml'
		
		st.download_button( label='Download System Instructions', data=export_text,
			file_name=export_filename, mime='text/plain', disabled=not bool( export_text.strip( ) ) )
		
		# -----------------------------
		# Existing chat history export
		st.divider( )
		st.markdown( '###### Chat History' )
		
		hist = load_history( )
		md_history = '\n\n'.join(
			[ f'**{role.upper( )}**\n{content}' for role, content in hist ]
		)
		
		st.download_button( 'Download Chat History (Markdown)', md_history,
			'buddy_chat.md', mime='text/markdown' )
		
		buf = io.BytesIO( )
		pdf = canvas.Canvas( buf, pagesize=LETTER )
		y = 750
		
		for role, content in hist:
			pdf.drawString( 40, y, f'{role.upper( )}: {content[ :90 ]}' )
			y -= 14
			if y < 50:
				pdf.showPage( )
				y = 750
		
		pdf.save( )
		
		st.download_button( 'Download Chat History (PDF)', buf.getvalue( ),
			'buddy_chat.pdf', mime='application/pdf' )

# ==============================================================================
# DATA MANAGEMENT MODE
# ==============================================================================
elif mode == 'Data Management':
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.subheader( '🏛️ Data Management', help=cfg.DATA_MANAGEMENT )
		st.divider( )
		tabs = st.tabs( [ '📥 Import', '🗂 Browse', '💉 CRUD', '📊 Explore', '🔎 Filter',
				'🧮 Aggregate', '📈 Visualize', '⚙ Admin', '🧠 SQL' ] )
		
		tables = list_tables( )
		if not tables:
			st.info( "No tables available." )
		else:
			table = st.selectbox( "Table", tables )
			df_full = read_table( table )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ] )
			overwrite = st.checkbox( 'Overwrite existing tables', value=True )
			if uploaded_file:
				try:
					sheets = pd.read_excel( uploaded_file, sheet_name=None )
					with create_connection( ) as conn:
						conn.execute( 'BEGIN' )
						for sheet_name, df in sheets.items( ):
							table_name = create_identifier( sheet_name )
							if overwrite:
								conn.execute( f'DROP TABLE IF EXISTS "{table_name}"' )
							
							# --- Create Table ---
							columns = [ ]
							df.columns = [ create_identifier( c ) for c in df.columns ]
							for col in df.columns:
								sql_type = get_sqlite_type( df[ col ].dtype )
								columns.append( f'"{col}" {sql_type}' )
							
							create_stmt = (
									f'CREATE TABLE "{table_name}" '
									f'({", ".join( columns )});'
							)
							
							conn.execute( create_stmt )
							
							# --- Insert Data ---
							placeholders = ", ".join( [ "?" ] * len( df.columns ) )
							insert_stmt = (
									f'INSERT INTO "{table_name}" '
									f'VALUES ({placeholders});'
							)
							
							conn.executemany(
								insert_stmt,
								df.where( pd.notnull( df ), None ).values.tolist( )
							)
						
						conn.commit( )
					
					st.success( 'Import completed successfully (transaction committed).' )
					st.rerun( )
				
				except Exception as e:
					try:
						conn.rollback( )
					except:
						pass
					st.error( f'Import failed — transaction rolled back.\n\n{e}' )
			
		# ------------------------------------------------------------------------------
		# BROWSE TAB
		# ------------------------------------------------------------------------------
		with tabs[ 1 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='table_name' )
				df = read_table( table )
				st.dataframe( df, use_container_width=True )
			else:
				st.info( 'No tables available.' )
	
		# ------------------------------------------------------------------------------
		# CRUD (Schema-Aware)
		# ------------------------------------------------------------------------------
		with tabs[ 2 ]:
			tables = list_tables( )
			if not tables:
				st.info( 'No tables available.' )
			else:
				table = st.selectbox( 'Select Table', tables, key='crud_table' )
				df = read_table( table )
				schema = create_schema( table )
				
				# Build type map
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				st.subheader( 'Insert Row' )
				insert_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						insert_data[ column ] = st.number_input( column, step=1, key=f'ins_{column}' )
					
					elif 'REAL' in col_type:
						insert_data[
							column ] = st.number_input( column, format='%.6f', key=f'ins_{column}' )
					
					elif 'BOOL' in col_type:
						insert_data[ column ] = 1 if st.checkbox( column, key=f'ins_{column}' ) else 0
					
					else:
						insert_data[ column ] = st.text_input( column, key=f'ins_{column}' )
				
				if st.button( 'Insert Row' ):
					cols = list( insert_data.keys( ) )
					placeholders = ', '.join( [ '?' ] * len( cols ) )
					stmt = f'INSERT INTO "{table}" ({", ".join( cols )}) VALUES ({placeholders});'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( insert_data.values( ) ) )
						conn.commit( )
					
					st.success( 'Row inserted.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				st.subheader( 'Update Row' )
				rowid = st.number_input( 'Row ID', min_value=1, step=1 )
				update_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						val = st.number_input( column, step=1, key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'REAL' in col_type:
						val = st.number_input( column, format='%.6f', key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'BOOL' in col_type:
						val = 1 if st.checkbox( column, key=f'upd_{column}' ) else 0
						update_data[ column ] = val
					
					else:
						val = st.text_input( column, key=f"upd_{column}" )
						update_data[ column ] = val
				
				if st.button( 'Update Row' ):
					set_clause = ', '.join( [ f'{c}=?' for c in update_data ] )
					stmt = f'UPDATE {table} SET {set_clause} WHERE rowid=?;'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
						conn.commit( )
					
					st.success( 'Row updated.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				st.subheader( 'Delete Row' )
				delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1 )
				if st.button( 'Delete Row' ):
					with create_connection( ) as conn:
						conn.execute( f'DELETE FROM {table} WHERE rowid=?;', (delete_id,) )
						conn.commit( )
					
					st.success( 'Row deleted.' )
					st.rerun( )
		
		# ------------------------------------------------------------------------------
		# EXPLORE
		# ------------------------------------------------------------------------------
		with tabs[ 3 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='explore_table' )
				page_size = st.slider( 'Rows per page', 10, 500, 50 )
				page = st.number_input( 'Page', min_value=1, step=1 )
				offset = (page - 1) * page_size
				df_page = read_table( table, page_size, offset )
				st.dataframe( df_page, use_container_width=True )
			
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='filter_table' )
				df = read_table( table )
				column = st.selectbox( 'Column', df.columns )
				value = st.text_input( 'Contains' )
				if value:
					df = df[ df[ column ].astype( str ).str.contains( value ) ]
				st.dataframe( df, use_container_width=True )
			
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='agg_table' )
				df = read_table( table )
				numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_cols:
					col = st.selectbox( 'Column', numeric_cols )
					agg = st.selectbox( 'Function', [ 'SUM',  'AVG', 'COUNT' ] )
					if agg == 'SUM':
						st.metric( 'Result', df[ col ].sum( ) )
					elif agg == 'AVG':
						st.metric( 'Result', df[ col ].mean( ) )
					elif agg == 'COUNT':
						st.metric( 'Result', df[ col ].count( ) )
			
		# ------------------------------------------------------------------------------
		# VISUALIZE
		# ------------------------------------------------------------------------------
		with tabs[ 6 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='viz_table' )
				df = read_table( table )
				numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_cols:
					col = st.selectbox( 'Column', numeric_cols )
					fig = px.histogram( df, x=col )
					st.plotly_chart( fig, use_container_width=True )
			
		# ------------------------------------------------------------------------------
		# ADMIN
		# ------------------------------------------------------------------------------
		with tabs[ 7 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='admin_table' )
			
			st.divider( )
			
			st.subheader( 'Data Profiling' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='profile_table' )
				if st.button( 'Generate Profile' ):
					profile_df = create_profile_table( table )
					st.dataframe( profile_df, use_container_width=True )
					
			st.subheader( 'Drop Table' )
	
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( 'Drop Table', key='admin_drop_button' ):
					st.session_state.dm_confirm_drop = True
				
				# Step 2: Confirmation UI
				if st.session_state.dm_confirm_drop:
					st.warning( f'You are about to permanently delete table {table}. '
						'This action cannot be undone.' )
					
					col1, col2 = st.columns( 2 )
					
					if col1.button( 'Confirm Drop', key='admin_confirm_drop' ):
						try:
							drop_table( table )
							st.success( f'Table {table} dropped successfully.' )
						except Exception as e:
							st.error( f'Drop failed: {e}' )
						
						st.session_state.dm_confirm_drop = False
						st.rerun( )
					
					if col2.button( 'Cancel', key='admin_cancel_drop' ):
						st.session_state.dm_confirm_drop = False
						st.rerun( )
				
				df = read_table( table )
				col = st.selectbox( 'Create Index On', df.columns )
				
				if st.button( 'Create Index' ):
					create_index( table, col )
					st.success( 'Index created.' )
					
			st.divider( )
			
			st.subheader( 'Create Custom Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20, value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( {
						'name': col_name,
						'type': col_type,
						'not_null': not_null,
						'primary_key': primary_key,
						'auto_increment': auto_inc } )
			
			if st.button( 'Create Table' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			st.divider( )
			st.subheader( 'Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				
				# Column schema
				schema = create_schema( table )
				schema_df = pd.DataFrame(
					schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "### Columns" )
				st.dataframe( schema_df, use_container_width=True )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute( f'SELECT COUNT(*) FROM "{table}"' ).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = pd.DataFrame( indexes,
						columns=[ 'seq', 'name',  'unique',  'origin', 'partial' ] )
					st.markdown( "### Indexes" )
					st.dataframe( idx_df, use_container_width=True )
				else:
					st.info( "No indexes defined." )
			
			st.divider( )
			st.subheader( "ALTER TABLE Operations" )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='alter_table_select' )
				operation = st.selectbox( 'Operation',
					[ 'Add Column', 'Rename Column', 'Rename Table', 'Drop Column' ] )
				
				if operation == 'Add Column':
					new_col = st.text_input( 'Column Name' )
					col_type = st.selectbox( 'Column Type', [ 'INTEGER',  'REAL',  'TEXT' ] )
					
					if st.button( 'Add Column' ):
						add_column( table, new_col, col_type )
						st.success( 'Column added.' )
						st.rerun( )
				
				elif operation == 'Rename Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					old_col = st.selectbox( 'Column to Rename', col_names )
					new_col = st.text_input( 'New Column Name' )
					
					if st.button( 'Rename Column' ):
						rename_column( table, old_col, new_col )
						st.success( 'Column renamed.' )
						st.rerun( )
				
				elif operation == 'Rename Table':
					new_name = st.text_input( 'New Table Name' )
					
					if st.button( 'Rename Table' ):
						rename_table( table, new_name )
						st.success( 'Table renamed.' )
						st.rerun( )
				
				elif operation == 'Drop Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					drop_col = st.selectbox( 'Column to Drop', col_names )
					
					if st.button( 'Drop Column' ):
						drop_column( table, drop_col )
						st.success( 'Column dropped.' )
						st.rerun( )
						
		# ------------------------------------------------------------------------------
		# SQL
		# ------------------------------------------------------------------------------
		with tabs[ 8 ]:
			st.subheader( 'SQL Console' )
			query = st.text_area( 'Enter SQL Query' )
			if st.button( 'Run Query' ):
				if not is_safe_query( query ):
					st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
				else:
					try:
						start_time = time.perf_counter( )
						with create_connection( ) as conn:
							result = pd.read_sql_query( query, conn )
						
						end_time = time.perf_counter( )
						elapsed = end_time - start_time
						
						# ----------------------------------------------------------
						# Display Results
						# ----------------------------------------------------------
						st.dataframe( result, use_container_width=True )
						row_count = len( result )
						
						# ----------------------------------------------------------
						# Execution Metrics
						# ----------------------------------------------------------
						col1, col2 = st.columns( 2 )
						col1.metric( 'Rows Returned', f'{row_count:,}' )
						col2.metric( 'Execution Time (seconds)', f'{elapsed:.6f}' )
						
						# Optional slow query warning
						if elapsed > 2.0:
							st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
						
						# ----------------------------------------------------------
						# Download
						# ----------------------------------------------------------
						if not result.empty:
							csv = result.to_csv( index=False ).encode( 'utf-8' )
							st.download_button( 'Download CSV', csv,
								'query_results.csv', 'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )

# ======================================================================================
# FOOTER — SECTION
# ======================================================================================
st.markdown( """
	<style>
	.block-container {
		padding-bottom: 3rem;
	}
	</style>
	""", unsafe_allow_html=True, )

# ---- Fixed Container
st.markdown("""
	<style>
	.boo-status-bar {
		position: fixed;
		bottom: 0;
		left: 0;
		width: 100%;
		background-color: rgba(17, 17, 17, 0.95);
		border-top: 1px solid #2a2a2a;
		padding: 10px 16px;
		font-size: 0.80rem;
		color: #35618c;
		z-index: 1000;
	}
	.boo-status-inner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		max-width: 100%;
	}
	</style>
	""", unsafe_allow_html=True, )

# ======================================================================================
# FOOTER RENDERING
# ======================================================================================
_mode_to_model_key = \
{
	'Chat': 'chat_model',
	'Text': 'text_model',
	'Images': 'image_model',
	'Audio': 'audio_model',
	'TTS': 'tts_model',
	'Translation': 'translation_model',
	'Transcription': 'transcription_model',
	'Embeddings': 'embedding_model',
	'Document Q&A': 'docqna_model',
	'Files': 'files_model',
	'Vector Stores': 'stores_model',
	'Data Management': 'text_model'
}

provider_val = st.session_state.get( 'provider', '—' )
mode_val = mode or '—'
active_model = st.session_state.get( _mode_to_model_key.get( mode, "" ), None )
right_parts = [ ]
if active_model is not None:
	right_parts.append( f'Model: {active_model}' )

# ---- Rendered Variables
if mode == 'Text':
	temperature = st.session_state.get( 'text_temperature' )
	top_p = st.session_state.get( 'text_top_percent' )
	freq = st.session_state.get( 'text_frequency_penalty' )
	presence = st.session_state.get( 'text_presence_penalty' )
	number = st.session_state.get( 'text_number' )
	stream = st.session_state.get( 'text_stream' )
	parallel_tools = st.session_state.get( 'text_parallel_tools' )
	max_calls = st.session_state.get( 'text_max_tools' )
	store = st.session_state.get( 'text_store' )
	tools = st.session_state.get( 'text_tools' )
	include = st.session_state.get( 'text_include' )
	domains = st.session_state.get( 'text_domains' )
	input_mode = st.session_state.get( 'text_input' )
	tool_choice = st.session_state.get( 'text_tool_choice' )
	background = st.session_state.get( 'text_background' )
	messages = st.session_state.get( 'text_messages' )
	max_tokens = st.session_state.get( 'text_max_tokens' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature:.1%}' )
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p:.1%}' )
	if freq is not None:
		right_parts.append( f'Freq: {freq:.2f}' )
	if presence is not None:
		right_parts.append( f'Presence: {presence:.2f}' )
	if number is not None:
		right_parts.append( f'N: {number}' )
	if max_tokens is not None:
		right_parts.append( f'Max Tokens: {max_tokens}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	if parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if max_calls is not None:
		right_parts.append( f'Max Calls: {max_calls}' )
	if store:
		right_parts.append( 'Store: On' )
	if tools:
		right_parts.append( f'Tools: {len( tools )}' )
	if include:
		right_parts.append( 'Include: On' )
	if domains:
		right_parts.append( 'Domains: Set' )
	if input_mode:
		right_parts.append( 'Input: Set' )
	if tool_choice:
		right_parts.append( f'Tool Choice: On' )
	if background:
		right_parts.append( 'Background: On' )
	if messages:
		right_parts.append( 'Messages: Set' )

elif mode == 'Images':
	image_mode = st.session_state.get( 'image_mode' )
	image_size = st.session_state.get( 'image_size' )
	image_aspect = st.session_state.get( 'image_aspect' )
	image_style = st.session_state.get( 'image_style' )
	image_backcolor = st.session_state.get( 'image_backcolor' )
	image_quality = st.session_state.get( 'image_quality' )
	image_fmt = st.session_state.get( 'image_format' )
	image_reasoning = st.session_state.get( 'image_reasoning' )
	image_detail = st.session_state.get( 'image_detail' )
	image_number = st.session_state.get( 'image_number' )
	image_stream = st.session_state.get( 'image_stream' )
	image_store = st.session_state.get( 'image_store' )
	image_background = st.session_state.get( 'image_background' )
	image_include = st.session_state.get( 'image_include' )
	image_parallel_tools = st.session_state.get( 'image_parallel_tools' )
	image_max_calls = st.session_state.get( 'text_max_tools' )
	image_tools = st.session_state.get( 'image_tools' )
	
	if image_aspect is not None:
		right_parts.append( f'Aspect: {image_aspect}' )
	elif image_size is not None:
		right_parts.append( f'Size: {image_size}' )
	
	if image_mode is not None:
		right_parts.append( f'Mode: {image_mode}' )
	if image_reasoning is not None:
		right_parts.append( f'Reasoning: {image_reasoning}' )
	if image_style is not None:
		right_parts.append( f'Style: {image_style}' )
	if image_quality is not None:
		right_parts.append( f'Quality: {image_quality}' )
	if image_backcolor is not None:
		right_parts.append( f'Backcolor: {image_backcolor}' )
	if image_fmt is not None:
		right_parts.append( f'Format: {image_fmt}' )
	if image_detail is not None:
		right_parts.append( f'Detail: {image_detail}' )
	
	if image_number is not None:
		right_parts.append( f'N: {image_number}' )
	if image_parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if image_max_calls is not None:
		right_parts.append( f'Max Calls: {image_max_calls}' )
	if image_tools:
		right_parts.append( f'Tools: {len(image_tools)}' )
	if image_include:
		right_parts.append( 'Include: On' )
	if image_stream:
		right_parts.append( 'Stream: On' )
	if image_store:
		right_parts.append( 'Store: On' )
	if image_background:
		right_parts.append( 'Background: On' )

elif mode == 'Audio':
	audio_task = st.session_state.get( 'audio_task' )
	audio_format = st.session_state.get( 'audio_response_format' )
	audio_top_p = st.session_state.get( 'audio_top_percent' )
	audio_freq = st.session_state.get( 'audio_frequency_penalty' )
	audio_presence = st.session_state.get( 'audio_presense_penalty' )
	audio_number = st.session_state.get( 'audio_number' )
	audio_temperature = st.session_state.get( 'audio_temperature' )
	audio_stream = st.session_state.get( 'audio_stream' )
	audio_store = st.session_state.get( 'audio_store' )
	audio_input_mode = st.session_state.get( 'audio_input' )
	audio_reasoning = st.session_state.get( 'audio_reasoning' )
	audio_tool_choice = st.session_state.get( 'audio_tool_choice' )
	audio_messages = st.session_state.get( 'audio_messages' )
	audio_background = st.session_state.get( 'audio_background' )
	audio_file = st.session_state.get( 'audio_file' )
	audio_rate = st.session_state.get( 'audio_rate'  )
	audio_start = st.session_state.get( 'audio_start' )
	audio_end = st.session_state.get( 'audio_end'  )
	audio_loop = st.session_state.get( 'audio_loop'  )
	audio_play = st.session_state.get( 'auto_play'  )
	audio_voice = st.session_state.get( 'voice', None )
	
	if audio_task is not None:
		right_parts.append( f'Task: {audio_task}' )
	if audio_format is not None:
		right_parts.append( f'Format: {audio_format}' )
	
	if audio_temperature is not None:
		right_parts.append( f'Temp: {audio_temperature:.1%}' )
	if audio_top_p is not None:
		right_parts.append( f'Top-P: {audio_top_p:.1%}' )
	if audio_freq is not None:
		right_parts.append( f'Freq: {audio_freq:.2f}' )
	if audio_presence is not None:
		right_parts.append( f'Presence: {audio_presence:.2f}' )
	if audio_number is not None:
		right_parts.append( f'N: {audio_number}' )
	
	if audio_stream:
		right_parts.append( 'Stream: On' )
	if audio_store:
		right_parts.append( 'Store: On' )
	if audio_reasoning:
		right_parts.append( 'Reasoning: On' )
	if audio_chat:
		right_parts.append( 'Input: Set' )
	if audio_tool_choice:
		right_parts.append( f'Tool Choice: {audio_tool_choice}' )
	if audio_messages:
		right_parts.append( 'Messages: Set' )
	if audio_background:
		right_parts.append( 'Background: On' )
	
	if audio_voice:
		right_parts.append( f'Voice: {audio_voice}' )
	if audio_rate is not None:
		right_parts.append( f'Rate: {audio_rate}' )
	if (audio_start or audio_end) and audio_end >= audio_start:
		right_parts.append( f'Trim: {audio_start}s–{audio_end}s' )
	if audio_loop:
		right_parts.append( 'Loop: On' )
	if audio_play:
		right_parts.append( 'Autoplay: On' )
	if audio_file is not None:
		right_parts.append( 'File: Set' )

elif mode == 'Embeddings':
	model = st.session_state.get( 'embedding_model' )
	dimensions = st.session_state.get( 'embeddings_dimensions' )
	encoding = st.session_state.get( 'embeddings_encoding_format' )
	input_data = st.session_state.get( 'embedding_text_input' )
	
	if model is not None:
		right_parts.append( f'Model: {model}' )
	
	if dimensions is not None:
		right_parts.append( f'Dim: {dimensions}' )
	
	if encoding is not None:
		right_parts.append( f'Format: {encoding}' )
	
	if input_data:
		right_parts.append( 'Input: Set' )

elif mode == 'Files':
	files_purpose = st.session_state.get( 'files_purpose' )
	files_type = st.session_state.get( 'files_type' )
	files_id = st.session_state.get( 'files_id' )
	files_url = st.session_state.get( 'files_url' )
	
	if files_purpose is not None:
		right_parts.append( f'Purpose: {files_purpose}' )
	
	if files_type is not None:
		right_parts.append( f'Type: {files_type}' )
	
	if files_id is not None:
		right_parts.append( f'File ID: {files_id}' )
	
	if files_url is not None:
		right_parts.append( 'URL: Set' )

elif mode == 'VectorStores':
	model = st.session_state.get( 'stores_model' )
	fmt = st.session_state.get( 'stores_response_format' )
	temperature = st.session_state.get( 'stores_temperature' )
	top_p = st.session_state.get( 'stores_top_percent' )
	freq = st.session_state.get( 'stores_frequency_penalty' )
	presence = st.session_state.get( 'stores_presense_penalty' )
	number = st.session_state.get( 'stores_number' )
	stream = st.session_state.get( 'stores_stream' )
	store = st.session_state.get( 'stores_store' )
	input_data = st.session_state.get( 'stores_input' )
	reasoning = st.session_state.get( 'stores_reasoning' )
	tool_choice = st.session_state.get( 'stores_tool_choice' )
	messages = st.session_state.get( 'stores_messages' )
	background = st.session_state.get( 'stores_background' )
	
	if model is not None:
		right_parts.append( f'Model: {model}' )
	
	if fmt is not None:
		right_parts.append( f'Format: {fmt}' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature}' )
	
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p}' )
	
	if freq is not None:
		right_parts.append( f'Freq: {freq}' )
	
	if presence is not None:
		right_parts.append( f'Presence: {presence}' )
	
	if number is not None:
		right_parts.append( f'N: {number}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	
	if store:
		right_parts.append( 'Store: On' )
	
	if reasoning is not None:
		right_parts.append( f'Reasoning: {reasoning}' )
	
	if tool_choice is not None:
		right_parts.append( f'Tool Choice: {tool_choice}' )
	
	if input_data:
		right_parts.append( 'Input: Set' )
	
	if messages:
		right_parts.append( 'Messages: Set' )
	
	if background:
		right_parts.append( 'Background: On' )

right_text = ' ◽ '.join( right_parts ) if right_parts else '—'

# ---- Rendering Method
st.markdown( f"""
    <div class="boo-status-bar">
        <div class="boo-status-inner">
            <span>{provider_val} — {mode_val}</span>
            <span>{right_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True, )