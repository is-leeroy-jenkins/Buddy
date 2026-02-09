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
CLOSE_TAG = re.compile( r"</([A-Za-z0-9_\-:.]+)>" )
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
GOOGLE_CREDENTIALS = os.getenv( 'GOOGLE_APPLICATION_CREDENTIALS')
GROQ_API_KEY = os.getenv( 'GROQ_API_KEY' )
XAI_API_KEY = os.getenv( 'XAI_API_KEY' )

#----------------- GPT CONFIG -------------------
GPT_LOGO = r'resources/buddy_logo.ico'

GPT_VECTOR_STORES = [ 'vs_712r5W5833G6aLxIYIbuvVcK',
                      'vs_697f86ad98888191b967685ae558bfc0']

GPT_FILES = [ 'file-Wd8G8pbLSgVjHur8Qv4mdt',
              'file-WPmTsHFYDLGHbyERqJdyqv',
              'file-DW5TuqYoEfqFfqFFsMXBvy',
              'file-U8ExiB6aJunAeT6872HtEU',
              'file-FHkNiF6Rv29eCkAWEagevT',
              'file-XsjQorjtffHTWjth8EVnkL',
              'file-32s641QK1Xb5QUatY3zfWF' ]

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
                 'file-XsjQorjtffHTWjth8EVnkL',
                 'file-32s641QK1Xb5QUatY3zfWF' ]

GPT_WEB_DOMAINS = [ 'congress.gov',
                    'google.com',
                    'gao.gov',
                    'omb.gov',
                    'defense.gov' ]

# ---------------- GROK CONFIG ------------------
GROK_LOGO = r'resources/grok_logo.png'
GROK_MODES = [ 'Text',
               'Images',
               'Documents',
               'Files',
               'Audio',
               'Vector Stores',
               'Prompt Engineering',
               'Data Export' ]

GROK_COLLECTIONS = [ { 'DOD Regulations': 'collection_a7973fd2-a336-4ed0-a495-4ffa947041c6'},
                     { 'DOA Regulations': 'collection_dbf8919e-5f56-435b-806b-642cd57c355e'},
                     { 'Financial Regulations': 'collection_9195d847-03a1-443c-9240-294c64dd01e2'},
                     { 'Explanatory Statements': 'collection_41dc3374-24d0-4692-819c-59e3d7b11b93' },
                     { 'Public Laws': 'collection_c1d0b83e-2f59-4f10-9cf7-51392b490fee' }, ]


# ---------------- GEMINI CONFIG ------------------
GEMINI_LOGO = r'resources/gemini_logo.png'
GEMINI_MODES = [ 'Text',
                 'Images',
                 'Files',
                 'Documents',
                 'Embeddings',
                 'Audio',
                 'Vector Stores',
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
		'Documents': [ 'Files'  ],
		'Files': [ 'Files'  ],
		'Vector Store': [ 'Files', 'VectorStores' ],
}

CLASS_MODE_MAP = {
		'GPT': GPT_MODES,
		'Gemini': GEMINI_MODES,
		'Grok': GROK_MODES }

LOGO_MAP = {
		'GPT': GPT_LOGO,
		'Gemini': GEMINI_LOGO,
		'Grok': GROK_LOGO }



#-------- DEFINITIONS -------------------
TEMPERATURE = r'''A number between 0 and 2. Higher values like 0.8 will make the output
		more random, while lower values like 0.2 will make it more focused and deterministic'''

TOP_P = r'''Optional. The maximum cumulative probability of tokens to consider when sampling.
		The model uses combined Top-k and Top-p (nucleus) sampling. Tokens are sorted based on
		their assigned probabilities so that only the most likely tokens are considered.
		Top-k sampling directly limits the maximum number of tokens to consider,
		while Nucleus sampling limits the number of tokens based on the cumulative probability.'''


PRESENCE_PENALTY = r'''Optional. Presence penalty applied to the next token's logprobs
		if the token has already been seen in the response. This penalty is binary on/off
		and not dependant on the number of times the token is used (after the first).'''

FREQUENCY_PENALTY = r'''Optional. Frequency penalty applied to the next token's logprobs,
		multiplied by the number of times each token has been seen in the respponse so far.
		A positive penalty will discourage the use of tokens that have already been used,
		proportional to the number of times the token has been used: The more a token is used,
		the more difficult it is for the model to use that token again increasing
		the vocabulary of responses.'''

LOG_PROBS = r'''Optional. Only valid if responseLogprobs=True. This sets the number of top logprobs to
		return at each decoding step in the Candidate.logprobs_result.
		The number must be in the range of [0, 20].'''

MAX_OUTPUT_TOKENS = r'''The maximum number of tokens used in generating output content'''

STOP_SEQUENCE = r'''Up to 4 string sequences where the API will stop generating further tokens.'''
