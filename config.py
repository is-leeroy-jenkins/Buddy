'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

	     Buddy
	     Copyright ©  2022  Terry Eppler

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

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    config.py
  </summary>
  ******************************************************************************************
  '''
import os
import re


#------------- COMMON CONSTANTS ---------------------
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
FAVICON = r'resources/favicon.ico'
CRS = r'https://www.congress.gov/crs-appropriations-status-table'
BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"
APP_TITLE = 'Buddy'
APP_SUBTITLE = 'Budget Execution AI'
OPEN_TAG = re.compile( r"<([A-Za-z0-9_\-:.]+)>" )
CLSOE_TAG = re.compile( r"</([A-Za-z0-9_\-:.]+)>" )
MARKDOWN_HEADING_PATTERN = re.compile( r"^##\s+(?P<title>.+?)\s*$" )
XML_BLOCK_PATTERN = re.compile( r"<(?P<tag>[a-zA-Z0-9_:-]+)>(?P<body>.*?)</\1>", re.DOTALL )
DB_PATH = "stores/sqlite/Data.db"
ANALYST = '❓'
BUDDY = '🧠'
PROVIDERS = { 'GPT': 'gpt', 'Gemini': 'gemini', 'Grok': 'grok', }
PROMPT_ID = 'pmpt_697f53f7ddc881938d81f9b9d18d6136054cd88c36f94549'
PROMPT_VERSION = '12'

# -------------- API KEYS ---------------------
OPENAI_API_KEY = os.getenv( 'OPENAI_API_KEY' )
GEMINI_API_KEY = os.getenv( 'GEMINI_API_KEY' )
GOOGLE_API_KEY = os.getenv( 'GOOGLE_API_KEY' )
GOOGLE_CLOUD_PROJECT_ID = os.getenv( 'GOOGLE_CLOUD_PROJECT' )
GOOGLE_CLOUD_LOCATION = os.getenv( 'GOOGLE_CLOUD_LOCATION' )
GOOGLE_GENAI_USE_VERTEXAI = os.getenv( 'GOOGLE_GENAI_USE_VERTEXAI' )
GROQ_API_KEY = os.getenv( 'GROQ_API_KEY' )
XAI_API_KEY = os.getenv( 'XAI_API_KEY' )

#----------------- GPT CONFIG -------------------
GPT_LOGO = r'resources/buddy_logo.ico'

GPT_VECTOR_STORES = [ 'vs_712r5W5833G6aLxIYIbuvVcK',
                      'vs_697f86ad98888191b967685ae558bfc0' ]

GPT_FILES = [ 'file-Wd8G8pbLSgVjHur8Qv4mdt',
              'file-WPmTsHFYDLGHbyERqJdyqv',
              'file-DW5TuqYoEfqFfqFFsMXBvy',
              'file-U8ExiB6aJunAeT6872HtEU',
              'file-FHkNiF6Rv29eCkAWEagevT',
              'file-XsjQorjtffHTWjth8EVnkL' ]

GPT_DOMAINS = [ 'congress.gov',
                'google.com',
                'gao.gov',
                'omb.gov',
                'defense.gov' ]

GPT_MODES = [ 'Chat',
              'Text',
              'Images',
              'Audio',
              'Embeddings',
              'Documents',
              'Files',
              'Vector Store',
              'Prompt Engineering',
              'Data Export' ]

GPT_VECTORSTORE_IDS = [ 'vs_712r5W5833G6aLxIYIbuvVcK',
                        'vs_697f86ad98888191b967685ae558bfc0' ]

GPT_FILE_IDS = [ 'file-Wd8G8pbLSgVjHur8Qv4mdt',
                 'file-WPmTsHFYDLGHbyERqJdyqv',
                 'file-DW5TuqYoEfqFfqFFsMXBvy',
                 'file-U8ExiB6aJunAeT6872HtEU',
                 'file-FHkNiF6Rv29eCkAWEagevT',
                 'file-XsjQorjtffHTWjth8EVnkL' ]

GPT_WEB_DOMAINS = [ 'congress.gov',
                    'google.com',
                    'gao.gov',
                    'omb.gov',
                    'defense.gov' ]

# ---------------- GROK CONFIG ------------------
GROK_LOGO = r'resources/grok_logo.png'
GROK_MODES = [ 'Text',
               'Images',
               'Files',
               'Collections',
               'Prompt Engineering',
               'Data Export' ]

# ---------------- GEMINI CONFIG ------------------
GEMINI_LOGO = r'resources/gemini_logo.png'
GEMINI_MODES = [ 'Text',
                 'Images',
                 'Embeddings',
                 'Audio',
                 'File Stores',
                 'Prompt Engineering',
                 'Data Export' ]

#----------------- MAPS ---------------------------
MODE_CLASS_MAP = {
		'Chat': None,
		'Text': [ 'Chat' ],
		'Images': [ 'Images' ],
		'Audio': [ 'TTS',
		           'Translation',
		           'Transcription' ],
		'Embeddings': [ 'Embeddings' ],
}

CLASS_MODE_MAP = {
		'GPT': GPT_MODES,
		'Gemini': GEMINI_MODES,
		'Grok': GROK_MODES }

LOGO_MAP = {
		'GPT': GPT_LOGO,
		'Gemini': GEMINI_LOGO,
		'Grok': GROK_LOGO }



#-------- PARAMETER DEFINITIONS -------------------
TEMPERATURE = r'''A number between 0 and 2. Higher values like 0.8 will make the output more random,
while lower values like 0.2 will make it more focused and deterministic'''

TOP_PERCENT = r'''Nucleus Sampling  where the model considers the results of the tokens with
top 'P' probability mass'''

PRESENCE_PENALTY = r'''Positive values penalize new tokens increasing the model's
likelihood to talk about new topics.'''

FREQUENCY_PENALTY = r'''Positive values penalize new tokens decreasing the model's
likelihood to repeat the same line verbatim.'''

MAX_OUTPUT_TOKENS = r'''The maximum number of tokens used in generating output content'''

STOP_SEQUENCE = r'''Up to 4 string sequences where the API will stop generating further tokens.'''
