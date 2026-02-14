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
PROMPT_VERSION = '16'
LOCAL_AUDIO_PATH = r'stores/audio/conditions.mp3'

# -------------- API KEYS ---------------------
OPENAI_API_KEY = os.getenv( 'OPENAI_API_KEY' )
GEMINI_API_KEY = os.getenv( 'GEMINI_API_KEY' )
GOOGLE_API_KEY = os.getenv( 'GOOGLE_API_KEY' )
GOOGLE_CLOUD_PROJECT_ID = os.getenv( 'GOOGLE_CLOUD_PROJECT' )
GOOGLE_CLOUD_LOCATION = os.getenv( 'GOOGLE_CLOUD_LOCATION' )
GOOGLE_GENAI_USE_VERTEXAI = os.getenv( 'GOOGLE_GENAI_USE_VERTEXAI' )
GOOGLE_APPLICATION_CREDENTIALS = os.getenv( 'GOOGLE_APPLICATION_CREDENTIALS')
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
              'Document Q&A',
              'Files',
              'Vector Stores',
              'Prompt Engineering',
              'Data Management',
              'Export' ]

GPT_VECTORSTORE_IDS = [ 'vs_712r5W5833G6aLxIYIbuvVcK', 'vs_697f86ad98888191b967685ae558bfc0' ]

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
               'Document Q&A',
               'Files',
               'Vector Stores',
               'Prompt Engineering',
               'Data Management',
               'Export' ]

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
                 'Document Q&A',
                 'Embeddings',
                 'Audio',
                 'Vector Stores',
                 'Prompt Engineering',
                 'Data Management',
                 'Export' ]

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
		'Vector Stores': [ 'VectorStores' ],
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
TEMPERATURE = r'''Optional. A number between 0 and 2. Higher values like 0.8 will make the output
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

MAX_OUTPUT_TOKENS = r'''Optional. The maximum number of tokens used in generating output content'''

STOP_SEQUENCE = r'''Optional. Up to 4 string sequences where the API will stop generating further tokens.'''

STORE = 'Optional. Whether to maintain state from turn to turn, preserving reasoning and tool context '

STREAM = 'Optional. Whether to return the generated respose in asynchronous chunks'

TOOLS = '''Optional. An array of tools the model may call while generating a response. You can specify which
		tool to use by setting the tool_choice parameter. Used by the Reponses API
		and Reasoning models (GPT 5x, Ox, 4o, etc)'''

INCLUDE = r'''Optional. Specifies additional output data to include in the model response enabling reasoning
			items to be used in multi-turn conversations when using the Responses API statelessly
			and Reasoning models (GPT 5x, Ox, 4o, etc)
			'''

REASONING = r'''Optional. Reasoning models introduce reasoning tokens in addition to input and output tokens.
				The models use these reasoning tokens to “think,” breaking down the prompt and
				considering multiple approaches to generating a response. After generating reasoning tokens,
				the model produces an answer as visible completion tokens and discards
				the reasoning tokens from its context. Used by the Reasoning models (GPT 5x and Ox etc)'''

CHOICE = r'''Optional. Determines how tools are chosen when using reasoning models GPT 5x and Ox etc)'''

SYSTEM_INSTRUCTIONS = r'''Optional. Gives the model high-level instructions on how it should behave while
		generating a response, including tone, goals, and examples of correct responses. Any
		instructions provided this way will take priority over a prompt in the input parameter.'''

SAMPLE_RATES = [ 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000 ]

BACKGROUND_MODE = r'''Background mode enables you to execute long-running tasks reliably,
		without having to worry about timeouts or other connectivity issues.'''

HYPERPARAMETERS = r'''Settings used during the inference (deployment) phase to control the behavior,
		creativity, and format of a model's output allowing users to fine-tune model
		responses without retraining. '''

PROMPT_ENGINEERING = r'''Prompt engineering is the process of writing effective instructions
		for a model, such that it consistently generates content that meets your requirements.
		Because the content generated from a model is non-deterministic, prompting to get your
		desired output is a mix of art and science. However, you can apply techniques and
		best practices to get good results consistently.
		'''

TEXT_GENERATION = r'''Use a large language model to produce coherent, context-aware natural language
		output in response to user prompts, system instructions, or retrieved document context.
		When a user submits a request—whether it is a general inquiry, a structured analytical task,
		or a document-grounded question—Buddy constructs a prompt that may include system directives,
		conversation history, and optionally retrieved content from its vector store. The underlying
		model then generates text according to configurable parameters such as temperature,
		maximum tokens, and response format. This capability enables Buddy to function as
		a conversational assistant, analytical explainer, summarizer, drafting tool, and reasoning engine,
		producing structured or narrative outputs tailored to the user’s workflow. '''

CHAT_COMPLETIONS  = r'''A unified interface for interacting with advanced generative models through
		a single request–response workflow. It allows a client to send structured inputs—such as text,
		images, audio, or tool instructions—and receive model-generated outputs that may include
		natural language responses, structured data, reasoning traces, or tool call instructions.
		It supports multi-modal inputs, iterative conversations, function/tool invocation,
		streaming outputs, and configurable generation parameters (e.g., temperature, max tokens),
		making it suitable for building chat systems, automation agents, data extraction pipelines,
		and decision-support applications. '''

AUDIO_API = r'''The Audio API functionality enables the ingestion, transformation, and generation
		of spoken language as part of the broader AI workflow. It allows users to upload audio files
		for transcription, converting speech into structured text that can then be analyzed,
		summarized, embedded, or used in Document Q&A and conversational contexts. It can also
		support translation of spoken content into other languages and text-to-speech generation, p
		roducing natural-sounding audio from model-generated text. By integrating speech recognition
		and synthesis alongside text and document processing, the Audio API expands Buddy into a
		multimodal assistant capable of handling voice-driven inputs and delivering spoken outputs
		within analytical or conversational workflows.  '''

FILES_API = r''' A structured mechanism for uploading, storing, listing, retrieving, and deleting
		user-provided files that are intended for downstream processing by the application’s
		AI workflows. It serves as the persistence layer for document assets that may later
		 be used for embedding generation, Document Q&A, or other model-assisted analysis. Rather
		 than embedding raw files directly into prompts, the Files API allows the user to reference
		 stored file objects by identifier, enabling controlled access, reuse across sessions,
		 and integration with higher-level capabilities such as retrieval, structured extraction,
		 or conversational analysis. In short, it manages document lifecycle and access so that
		 file-based intelligence features operate reliably and efficiently '''

IMAGES_API = r''' Enables the generation and analysis of visual content as part of the application’s
		broader AI workflow. On the generation side, users can provide descriptive prompts to
		create images that support presentations, reports, branding, or conceptual exploration.
		On the analysis side, uploaded images can be processed to extract descriptive insights,
		captions, or structured information that can then be incorporated into downstream tasks
		such as summarization or decision support. By integrating image generation and interpretation
		alongside text, documents, and structured data, the Images API expands beyond purely textual interaction,
		allowing it to operate in a multimodal environment where visual and
		linguistic information can be processed cohesively '''

VECTORSTORES_API = r'''Specialized databases designed to store and index embeddings so they can be
        searched efficiently by semantic similarity. After documents are processed and converted
        into high-dimensional vectors, those vectors are persisted in a vector store alongside
        metadata such as document name, chunk position, or source reference. When a user submits
        a query, its embedding is generated and compared against stored vectors using similarity
        metrics to retrieve the most relevant content. This enables fast, scalable semantic search
        and underpins features like Document Q&A by ensuring that responses are grounded in the
        most contextually relevant portions of the user’s data rather than relying solely
        on generalized model knowledge. '''

EMBEDDINGS_API = r'''Creates numerical vector representations of text that capture semantic meaning in a
		high-dimensional space. When documents, prompts, or queries are processed, their textual
		content is transformed into embeddings so that semantically similar content is positioned
		close together mathematically. Buddy stores these vectors in its local vector database,
		enabling similarity search, clustering, document retrieval, and contextual grounding for
		downstream tasks like Document Q&A. By converting language into structured numerical form,
		embeddings serve as the foundation for intelligent search, relevance ranking, and
		retrieval-augmented reasoning within the application. '''

DOCUMENT_Q_AND_A = r'''A retrieval-augmented workflow that allows users to ask natural language
		questions about uploaded documents (e.g., PDFs, Word files, Excel sheets) and receive
		contextually grounded answers derived directly from those materials. The system ingests
		documents, extracts and chunks their text, generates embeddings, stores those embeddings
		in a local vector database, and retrieves the most semantically relevant passages when a
		question is asked. The retrieved context is then supplied to the language model to
		generate a precise, source-aware response. This approach enables accurate,
		citation-ready answers tied to user-provided content rather than relying solely on general
		model knowledge, effectively turning Buddy into a document-aware analytical assistant.  '''

DATA_MANAGEMENT = r'''Structured handling, organization, processing of
		user-provided data in a self-contained SQLite Database. It allows uploading of files, extracting and
		normalizing their content, chunking text for semantic processing, generating embeddings,
		storing metadata, and enabling controlled retrieval for downstream features such as Document Q&A
		and Data Analysis. Beyond ingestion, it includes version awareness, indexing, schema inspection
		(where applicable), and the ability to manage or remove stored assets safely. Document
		Management provides the foundational infrastructure that transforms raw files into structured,
		searchable, and model-ready assets, ensuring that Buddy’s intelligence features operate
		on reliable, well-governed data rather than unmanaged documents.  '''