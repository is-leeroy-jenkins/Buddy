'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                grok.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="grok.py" company="Terry D. Eppler">

	     grok.py
	     Copyright ©  2024  Terry Eppler

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
    Groq Cloud API wrapper for Streamlit with Hybrid Tool support.
  </summary>
  ******************************************************************************************
'''

import os
import base64
import requests
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
import groq
from groq import Groq
import config as cfg
from boogr import ErrorDialog, Error

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encodes a local image to a base64 string for vision API requests."""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class Endpoints:
	'''

	    Purpose:
	    ---------
	    Encapsulates all service endpoints for Groq LPU and Hybrid Tool providers.

	    Attributes:
	    -----------
	    base_url           : str - Groq API base
	    chat_completions   : str - Text and Vision endpoint
	    speech_generations : str - Hybrid TTS endpoint
	    translations       : str - Whisper translation endpoint
	    transcriptions     : str - Whisper transcription endpoint
	    image_generations  : str - Hybrid Image generation
	    image_edits        : str - Hybrid Image editing
	    embeddings         : str - Hybrid text embeddings
	    wolfram            : str - Wolfram Alpha computation
	    google_search      : str - Tavily/Google search integration

    '''
	base_url: Optional[ str ]
	chat_completions: Optional[ str ]
	speech_generations: Optional[ str ]
	translations: Optional[ str ]
	transcriptions: Optional[ str ]
	image_generations: Optional[ str ]
	image_edits: Optional[ str ]
	embeddings: Optional[ str ]
	wolfram: Optional[ str ]
	google_search: Optional[ str ]
	
	def __init__( self ):
		self.base_url = f'https://api.groq.com/'
		self.chat_completions = f'https://api.groq.com/openai/v1/chat/completions'
		self.speech_generations = f'https://api.openai.com/v1/audio/speech'
		self.translations = f'https://api.groq.com/openai/v1/audio/translations'
		self.transcriptions = f'https://api.groq.com/openai/v1/audio/transcriptions'
		self.image_generations = f'https://api.openai.com/v1/images/generations'
		self.image_edits = f'https://api.openai.com/v1/images/edits'
		self.embeddings = f'https://api.openai.com/v1/embeddings'
		self.wolfram = f'https://api.wolframalpha.com/v1/result'
		self.google_search = f'https://api.tavily.com/search'

class Header:
	'''

	    Purpose:
	    --------
	    Manages HTTP header configurations for multi-provider API requests.

	    Attributes:
	    -----------
	    content_type  : str - MIME type
	    api_key       : str - Key used
	    authorization : str - Bearer string

    '''
	content_type: Optional[ str ]
	api_key: Optional[ str ]
	authorization: Optional[ str ]
	
	def __init__( self, key: str = cfg.GROQ_API_KEY ):
		self.content_type = 'application/json'
		self.api_key = key
		self.authorization = f'Bearer {key}'
	
	def get_header( self ) -> Dict[ str, str ] | None:
		"""Returns the configured HTTP header dictionary."""
		return { 'Content-Type': self.content_type, 'Authorization': self.authorization }

class Grok:
	'''

		Purpose:
		-------
		Base configuration class for Groq AI services and shared hyper-parameters.

		Attributes:
		-----------
		api_key           : str - Groq API Key
		instructions      : str - Global system prompt
		prompt            : str - Current request prompt
		model             : str - Current model ID
		max_tokens        : int - Token limit
		temperature       : float - Randomness
		top_p             : float - Nucleus sampling
		top_k             : int - Top-k threshold
		modalities        : list - Input/Output modes
		frequency_penalty : float - Repetition control
		presence_penalty  : float - Topic control
		response_format   : dict - Schema control

	'''
	api_key: Optional[ str ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	modalities: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	response_format: Optional[ Union[ str, Dict[ str, str ] ] ]
	candidate_count: Optional[ int ]
	
	def __init__( self ):
		self.api_key = cfg.GROQ_API_KEY
		self.model = None;
		self.temperature = 0.7;
		self.top_p = 0.9;
		self.top_k = 40
		self.candidate_count = 1;
		self.frequency_penalty = 0.0;
		self.presence_penalty = 0.0
		self.max_tokens = 8192;
		self.instructions = None;
		self.prompt = None
		self.modalities = None;
		self.response_format = None

class Chat( Grok ):
	'''

	    Purpose:
	    _______
	    Class handling high-speed Text, Vision, Web Search, and Math tools.

	    Attributes:
	    -----------
	    client           : Groq - LPU Client
	    contents         : list - Multi-part messages
	    response         : any - SDK response
	    image_url        : str - Vision target
	    file_path        : str - Local document target
	    url              : str - Remote website target

	    Methods:
	    --------
	    generate_text( prompt, model )      : Base LPU generation
	    generate_image( prompt, model )     : Tool route for images
	    analyze_image( prompt, filepath )   : Vision analysis
	    summarize_document( prompt, path )  : Local RAG summarization
	    search_file( prompt, filepath )     : Targeted file query
	    web_search( prompt, model )         : Tavily-powered research
	    search_website( prompt, url, mod )  : URL-specific context scraping
	    wolfram_alpha( prompt, model )      : Computational tool access

    '''
	client: Optional[ Groq ]
	contents: Optional[ Union[ str, List[ Dict[ str, Any ] ] ] ]
	response: Optional[ Any ]
	image_url: Optional[ str ]
	file_path: Optional[ str ]
	url: Optional[ str ]
	
	def __init__( self, model: str='llama-3.3-70b-versatile', temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int=8192, instruct: str=None ):
		super( ).__init__( )
		self.model = model;
		self.top_p = top_p;
		self.temperature = temperature
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_tokens = max_tokens;
		self.instructions = instruct
		self.client = Groq( api_key=self.api_key );
		self.contents = None
		self.response = None;
		self.image_url = None;
		self.file_path = None;
		self.url = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Returns list of Groq models including optimized 70b and 8b versions."""
		return [ 'llama-3.3-70b-versatile',
		         'llama-3.3-70b-specdec',
		         'llama-3.1-70b-versatile',
		         'llama-3.1-8b-instant',
		         'mixtral-8x7b-32768',
		         'gemma2-9b-it' ]
	
	def generate_text( self, prompt: str, model: str = 'llama-3.3-70b-versatile',
			temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=8192, instruct: str=None  ) -> str | None:
		"""
			
			Purpose:
			--------
			Generates an ultra-fast text completion using Groq LPU inference.
			
			Parameters:
			-----------
			prompt: str - The primary user instruction.
			model: str - The LLM identifier.
			
			Returns:
			--------
			Optional[ str ] - Content of the model's message.
		
		"""
		try:
			throw_if( 'prompt', prompt );
			self.prompt = prompt;
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency;
			self.presence_penalty = presence
			self.max_tokens = max_tokens;
			self.instructions = instruct
			messages = [ ]
			if self.instructions is not None:
				messages.append( { "role": "system", "content": self.instructions } )
			messages.append( { "role": "user", "content": self.prompt } )
			self.response = self.client.chat.completions.create( model=self.model, messages=messages,
				temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p,
				frequency_penalty=self.frequency_penalty, presence_penalty=self.presence_penalty )
			return self.response.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt, model )';
			error = ErrorDialog( exception );
			error.show( )
	
	def generate_image( self, prompt: str, model: str='dall-e-3' ) -> str | None:
		"""Purpose: Routes image generation to the Image Tool class."""
		try:
			throw_if( 'prompt', prompt );
			self.prompt = prompt;
			self.model = model
			image_tool = Image( temperature=self.temperature, top_p=self.top_p )
			return image_tool.generate( prompt=self.prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'generate_image( self, prompt, model )';
			error = ErrorDialog( exception );
			error.show( )
	
	def analyze_image( self, prompt: str,
			filepath: str, model: str='llama-3.2-11b-vision-preview' ) -> str | None:
		"""Purpose: Vision analysis for local files."""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'filepath', filepath )
			self.prompt = prompt;
			self.file_path = filepath;
			self.model = model
			b64 = encode_image( self.file_path )
			messages = [ {
					             "role": "user",
					             "content": [ {
							                          "type": "text",
							                          "text": self.prompt },
					                          {
							                          "type": "image_url",
							                          "image_url": {
									                          "url": f"data:image/jpeg;base64,{b64}" } } ] } ]
			self.response = self.client.chat.completions.create( model=self.model, messages=messages,
				temperature=self.temperature, max_tokens=self.max_tokens )
			return self.response.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt, filepath, model )';
			error = ErrorDialog( exception );
			error.show( )
	
	def web_search( self, prompt: str, model: str='llama-3.3-70b-versatile' ) -> str | None:
		"""Purpose: Search-augmented generation using Tavily."""
		try:
			throw_if( 'prompt', prompt );
			self.prompt = prompt;
			self.model = model
			payload = {
					"api_key": cfg.TAVILY_API_KEY,
					"query": self.prompt,
					"search_depth": "advanced" }
			search_resp = requests.post( Endpoints( ).google_search, json=payload )
			context = search_resp.json( ).get( 'results', [ ] )
			synth_prompt = f"Using results: {context}, Answer: {self.prompt}"
			return self.generate_text( prompt=synth_prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'web_search( self, prompt, model )';
			error = ErrorDialog( exception );
			error.show( )
	
	def wolfram_alpha( self, prompt: str, model: str='llama-3.3-70b-versatile' ) -> str | None:
		"""Purpose: Precise computational query through Wolfram API."""
		try:
			throw_if( 'prompt', prompt );
			self.prompt = prompt;
			self.model = model
			params = { "appid": cfg.WOLFRAM_APP_ID, "i": self.prompt }
			wolf_resp = requests.get( Endpoints( ).wolfram, params=params )
			math_prompt = f"Explain result '{wolf_resp.text}' for query: {self.prompt}"
			return self.generate_text( prompt=math_prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'wolfram_alpha( self, prompt, model )';
			error = ErrorDialog( exception );
			error.show( )
	
	def summarize_document( self, prompt: str,
			filepath: str, model: str='llama-3.3-70b-versatile' ) -> str | None:
		"""Purpose: Extracts and summarizes text from a local path."""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'filepath', filepath )
			self.prompt = prompt;
			self.file_path = filepath;
			self.model = model
			with open( self.file_path, 'r', encoding='utf-8' ) as f:
				doc_text = f.read( )
			return self.generate_text( f"{self.prompt}\n\nFile: {doc_text}", model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt, filepath, model )';
			error = ErrorDialog( exception );
			error.show( )
	
	def search_file( self, prompt: str, filepath: str, model: str='llama-3.3-70b-versatile' ) -> str | None:
		"""Purpose: Answers a specific query using file content as context."""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'filepath', filepath )
			self.prompt = prompt;
			self.file_path = filepath;
			self.model = model
			with open( self.file_path, 'r', encoding='utf-8' ) as f:
				doc_text = f.read( )
			search_prompt = f"Based on: {doc_text}, Answer: {self.prompt}"
			return self.generate_text( search_prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'search_file( self, prompt, filepath, model )';
			error = ErrorDialog( exception );
			error.show( )
	
	def search_website( self, prompt: str, url: str, model: str='llama-3.3-70b-versatile' ) -> str | None:
		"""Purpose: Scrapes a URL and answers prompt based on site content."""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'url', url )
			self.prompt = prompt;
			self.url = url;
			self.model = model
			resp = requests.get( self.url, timeout=10 )
			site_prompt = f"Answer '{self.prompt}' using site text: {resp.text[ :15000 ]}"
			return self.generate_text( prompt=site_prompt, model=self.model )
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Chat'
			exception.method = 'search_website( self, prompt, url, model )';
			error = ErrorDialog( exception );
			error.show( )

class Embedding( Grok ):
	'''

		Purpose:
		--------
		Generates vector representations using Hybrid/OpenAI compatible services.

    '''
	client: Optional[ Groq ];
	response: Optional[ Any ];
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ];
	dimensions: Optional[ int ];
	input_text: Optional[ str ]
	top_percent: Optional[ float ];
	max_completion_tokens: Optional[ int ];
	contents: Optional[ List[ str ] ]
	http_options: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str='text-embedding-3-small', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0, max_tokens: int=10000 ):
		super( ).__init__( )
		self.client = Groq( api_key=self.api_key )
		self.model = model;
		self.temperature = temperature;
		self.top_percent = top_p
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_completion_tokens = max_tokens;
		self.contents = [ ]
		self.encoding_format = 'float';
		self.input_text = None;
		self.embedding = None
		self.response = None;
		self.dimensions = None;
		self.http_options = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'nomic-embed-text-v1.5',
		         'text-embedding-3-small',
		         'text-embedding-3-large',
		         'text-embedding-ada-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		return [ 'float', 'base64' ]
	
	def create( self, text: str, model: str='text-embedding-3-small', format: str='float' ) -> List[ float ] | None:
		"""Purpose: Generates text embeddings via Hybrid POST."""
		try:
			throw_if( 'text', text )
			self.input_text = text;
			self.model = model;
			self.encoding_format = format
			headers = Header( key=cfg.OPENAI_API_KEY ).get_header( )
			payload = \
			{
				'input': self.input_text,
				'model': self.model,
				'encoding_format': self.encoding_format
			}
			
			resp = requests.post( Endpoints( ).embeddings, headers=headers, json=payload )
			if resp.status_code == 200:
				self.response = resp.json( )
				self.embedding = self.response[ 'data' ][ 0 ][ 'embedding' ]
				return self.embedding
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'groq';
			exception.cause = 'Embedding'
			exception.method = 'create( self, text, model, format )';
			error = ErrorDialog( exception );
			error.show( )
	
	def count_tokens( self, text: str, coding: str='cl100k_base' ) -> Optional[ int ]:
		"""Purpose: Simple word-count estimation for token limits."""
		try:
			throw_if( 'text', text );
			self.input_text = text
			return len( self.input_text.split( ) )
		except Exception as e:
			exception = Error( e );
			exception.module = 'groq';
			exception.cause = 'Embedding'
			exception.method = 'count_tokens( self, text, coding )';
			error = ErrorDialog( exception );
			error.show( )

class TTS( Grok ):
	"""

	    Purpose
	    ___________
	    Converts text to spoken audio via Hybrid/OpenAI TTS endpoints.

    """
	speed: Optional[ float ];
	voice: Optional[ str ];
	response: Optional[ Any ]
	client: Optional[ Groq ];
	audio_path: Optional[ str ];
	response_format: Optional[ str ]
	input_text: Optional[ str ];
	store: Optional[ bool ];
	stream: Optional[ bool ]
	number: Optional[ int ];
	top_percent: Optional[ float ];
	max_completion_tokens: Optional[ int ]
	stops: Optional[ List[ str ] ];
	messages: Optional[ List[ Dict[ str, str ] ] ]
	tools: Optional[ List[ Any ] ];
	vector_store_ids: Optional[ List[ str ] ]
	descriptions: Optional[ List[ str ] ];
	assistants: Optional[ List[ Any ] ]
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=True, stream: bool=True,
			instruct: str=None ):
		super( ).__init__( )
		self.client = Groq( api_key=self.api_key );
		self.model = 'tts-1'
		self.number = number;
		self.temperature = temperature;
		self.top_percent = top_p
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_tokens = max_tokens;
		self.store = store;
		self.stream = stream
		self.instructions = instruct;
		self.audio_path = None;
		self.response = None
		self.response_format = 'mp3';
		self.speed = 1.0;
		self.voice = 'onyx'
		self.input_text = None;
		self.stops = None;
		self.messages = None;
		self.tools = None
		self.vector_store_ids = None;
		self.descriptions = None;
		self.assistants = None
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'tts-1', 'tts-1-hd' ]
	
	@property
	def voice_options( self ) -> List[ str ]:
		return [ 'alloy',
		         'echo',
		         'fable',
		         'onyx',
		         'nova',
		         'shimmer' ]
	
	@property
	def output_options( self ) -> List[ str ]:
		return [ 'mp3',
		         'opus',
		         'aac',
		         'flac',
		         'wav' ]
	
	@property
	def sample_options( self ) -> List[ int ]:
		return [ 8000,
		         16000,
		         22050,
		         24000,
		         32000,
		         44100,
		         48000 ]
	
	def create_audio( self, text: str, filepath: str, format: str='mp3',
			speed: float=1.0, model: str='tts-1' ) -> str | None:
		"""Purpose: Generates speech audio via Hybrid POST."""
		try:
			throw_if( 'text', text );
			throw_if( 'filepath', filepath )
			self.input_text = text;
			self.audio_path = filepath;
			self.model = model
			self.response_format = format;
			self.speed = speed
			headers = Header( key=cfg.OPENAI_API_KEY ).get_header( )
			payload = \
			{
				'model': self.model,
				'input': self.input_text,
				'voice': self.voice,
				'response_format': self.response_format,
				'speed': self.speed 
			}
			
			resp = requests.post( Endpoints( ).speech_generations, headers=headers, json=payload )
			if resp.status_code == 200:
				with open( self.audio_path, 'wb' ) as f:
					f.write( resp.content )
				return self.audio_path
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'TTS'
			exception.method = 'create_audio( self, text, filepath, format, speed, model )';
			error = ErrorDialog( exception );
			error.show( )

class Transcription( Grok ):
	"""

	    Purpose
	    ___________
	    High-speed audio-to-text transcription via Groq LPU Whisper.

    """
	client: Optional[ Groq ];
	audio_file: Optional[ Any ];
	transcript: Optional[ str ]
	response: Optional[ Any ];
	input_text: Optional[ str ];
	store: Optional[ bool ]
	stream: Optional[ bool ];
	number: Optional[ int ];
	top_percent: Optional[ float ]
	max_completion_tokens: Optional[ int ];
	messages: Optional[ List[ Dict[ str, str ] ] ]
	stops: Optional[ List[ str ] ]
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=False, stream: bool=True,
			instruct: str=None ):
		super( ).__init__( )
		self.client = Groq( api_key=self.api_key )
		self.number = number;
		self.temperature = temperature;
		self.top_percent = top_p
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_tokens = max_tokens;
		self.store = store;
		self.stream = stream
		self.instructions = instruct;
		self.input_text = None;
		self.audio_file = None
		self.transcript = None;
		self.response = None;
		self.messages = None;
		self.stops = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		return [ 'whisper-large-v3-turbo',
		         'whisper-large-v3',
		         'distil-whisper-large-v3-en' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		return [ 'text',
		         'json',
		         'verbose_json',
		         'vtt',
		         'srt' ]
	
	@property
	def language_options( self ):
		'''
		
			Returns:
			-------
			A List[ str ] of languages translatable to English
			
		'''
		return [
				"af",
				"am",
				"ar",
				"as",
				"az",
				"ba",
				"be",
				"bg",
				"bn",
				"bo",
				"br",
				"bs",
				"ca",
				"cs",
				"cy",
				"da",
				"de",
				"el",
				"en",
				"es",
				"et",
				"eu",
				"fa",
				"fi",
				"fo",
				"fr",
				"gl",
				"gu",
				"ha",
				"haw",
				"he",
				"hi",
				"hr",
				"ht",
				"hu",
				"hy",
				"id",
				"is",
				"it",
				"ja",
				"jw",
				"ka",
				"kk",
				"km",
				"kn",
				"ko",
				"la",
				"lb",
				"ln",
				"lo",
				"lt",
				"lv",
				"mg",
				"mi",
				"mk",
				"ml",
				"mn",
				"mr",
				"ms",
				"mt",
				"my",
				"ne",
				"nl",
				"nn",
				"no",
				"oc",
				"pa",
				"pl",
				"ps",
				"pt",
				"ro",
				"ru",
				"sa",
				"sd",
				"si",
				"sk",
				"sl",
				"sn",
				"so",
				"sq",
				"sr",
				"su",
				"sv",
				"sw",
				"ta",
				"te",
				"tg",
				"th",
				"tk",
				"tl",
				"tr",
				"tt",
				"uk",
				"ur",
				"uz",
				"vi",
				"yi",
				"yo",
				"zh"
		]
	
	def transcribe( self, path: str, model: str='whisper-large-v3-turbo' ) -> Optional[ str ]:
		"""Purpose: Local file transcription using Groq LPU."""
		try:
			throw_if( 'path', path );
			self.audio_file = path;
			self.model = model
			with open( self.audio_file, 'rb' ) as audio:
				self.response = self.client.audio.transcriptions.create( file=(self.audio_file,
				                                                               audio.read( )), model=self.model, response_format="text" )
			self.transcript = str( self.response );
			return self.transcript
		except Exception as e:
			ex = Error( e );
			ex.module = 'grok';
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path, model )';
			error = ErrorDialog( ex );
			error.show( )

class Translation( Grok ):
	"""

	    Purpose
	    ___________
	    Direct-to-English audio translation via Groq LPU Whisper.

    """
	target_language: Optional[ str ];
	client: Optional[ Groq ];
	audio_file: Optional[ Any ]
	response: Optional[ Any ];
	voice: Optional[ str ];
	store: Optional[ bool ]
	stream: Optional[ bool ];
	number: Optional[ int ];
	top_percent: Optional[ float ]
	max_completion_tokens: Optional[ int ];
	audio_path: Optional[ str ]
	messages: Optional[ List[ Dict[ str, str ] ] ];
	stops: Optional[ List[ str ] ]
	completion: Optional[ str ]
	
	def __init__( self, number: int=1, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, store: bool=False, stream: bool=True,
			instruct: str=None ):
		super( ).__init__( )
		self.client = Groq( api_key=self.api_key )
		self.model = 'whisper-large-v3';
		self.number = number;
		self.temperature = temperature
		self.top_percent = top_p;
		self.frequency_penalty = frequency
		self.presence_penalty = presence;
		self.max_tokens = max_tokens
		self.store = store;
		self.stream = stream;
		self.instructions = instruct
		self.audio_file = None;
		self.response = None;
		self.voice = None
		self.audio_path = None;
		self.messages = None;
		self.stops = None
		self.completion = None;
		self.target_language = 'English'
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'whisper-large-v3',
		         'whisper-large-v3-turbo' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		'''

			Returns:
			-------
			A List[ str ] of languages translatable to English

		'''
		return [
				"af",
				"am",
				"ar",
				"as",
				"az",
				"ba",
				"be",
				"bg",
				"bn",
				"bo",
				"br",
				"bs",
				"ca",
				"cs",
				"cy",
				"da",
				"de",
				"el",
				"en",
				"es",
				"et",
				"eu",
				"fa",
				"fi",
				"fo",
				"fr",
				"gl",
				"gu",
				"ha",
				"haw",
				"he",
				"hi",
				"hr",
				"ht",
				"hu",
				"hy",
				"id",
				"is",
				"it",
				"ja",
				"jw",
				"ka",
				"kk",
				"km",
				"kn",
				"ko",
				"la",
				"lb",
				"ln",
				"lo",
				"lt",
				"lv",
				"mg",
				"mi",
				"mk",
				"ml",
				"mn",
				"mr",
				"ms",
				"mt",
				"my",
				"ne",
				"nl",
				"nn",
				"no",
				"oc",
				"pa",
				"pl",
				"ps",
				"pt",
				"ro",
				"ru",
				"sa",
				"sd",
				"si",
				"sk",
				"sl",
				"sn",
				"so",
				"sq",
				"sr",
				"su",
				"sv",
				"sw",
				"ta",
				"te",
				"tg",
				"th",
				"tk",
				"tl",
				"tr",
				"tt",
				"uk",
				"ur",
				"uz",
				"vi",
				"yi",
				"yo",
				"zh"
		]
	
	def translate( self, path: str, model: str='whisper-large-v3' ) -> str | None:
		"""Purpose: Translates local audio directly to English text."""
		try:
			throw_if( 'path', path );
			self.audio_path = path;
			self.model = model
			with open( self.audio_path, 'rb' ) as audio:
				self.response = self.client.audio.translations.create( file=(self.audio_path,
				                                                             audio.read( )), model=self.model )
			return self.response.text
		except Exception as e:
			ex = Error( e );
			ex.module = 'grok';
			ex.cause = 'Translation'
			ex.method = 'translate( self, path, model )';
			error = ErrorDialog( ex );
			error.show( )

class Image( Grok ):
	'''

	    Purpose
	    ___________
	    Class for Groq Vision analysis and Hybrid Image Generation/Editing.

    '''
	image_url: Optional[ str ];
	quality: Optional[ str ];
	detail: Optional[ str ];
	size: Optional[ str ]
	tool_choice: Optional[ str ];
	style: Optional[ str ];
	response_format: Optional[ str ]
	client: Optional[ Groq ];
	store: Optional[ bool ];
	stream: Optional[ bool ]
	number: Optional[ int ];
	top_percent: Optional[ float ];
	max_completion_tokens: Optional[ int ]
	input: Optional[ List[ Any ] ];
	input_text: Optional[ str ];
	file_path: Optional[ str ]
	response: Optional[ Any ];
	stops: Optional[ List[ str ] ];
	messages: Optional[ List[ Dict[ str, str ] ] ]
	completion: Optional[ str ]
	
	def __init__( self, n: int = 1, temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			store: bool=False, stream: bool = False ):
		super( ).__init__( )
		self.client = Groq( api_key=self.api_key )
		self.number = n;
		self.temperature = temperature;
		self.top_percent = top_p
		self.frequency_penalty = frequency;
		self.presence_penalty = presence
		self.max_tokens = max_tokens;
		self.store = store;
		self.stream = stream
		self.tool_choice = 'auto';
		self.input_text = None;
		self.file_path = None
		self.image_url = None;
		self.quality = 'standard';
		self.size = '1024x1024'
		self.style = 'natural';
		self.response_format = 'url';
		self.input = None
		self.detail = None;
		self.response = None;
		self.stops = None;
		self.messages = None
		self.completion = None
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'llama-3.2-90b-vision-preview',
		         'llama-3.2-11b-vision-preview' ]
	
	@property
	def gen_model_options( self ) -> List[ str ]:
		return [ 'dall-e-3',
		         'dall-e-2' ]
	
	@property
	def size_options( self ) -> List[ str ]:
		return [ '1024x1024',
		         '1024x1792',
		         '1792x1024',
		         '512x512',
		         '256x256' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		return [ 'url',
		         'b64_json' ]
	
	def generate( self, prompt: str, model: str = 'dall-e-3', quality: str = 'standard',
			size: str = '1024x1024' ) -> Optional[ str ]:
		"""Purpose: Generates a new image via Hybrid service POST."""
		try:
			throw_if( 'text', prompt );
			self.input_text = prompt;
			self.model = model
			self.quality = quality;
			self.size = size
			headers = Header( key=cfg.OPENAI_API_KEY ).get_header( )
			payload = {
					"model": self.model,
					"prompt": self.input_text,
					"size": self.size,
					"quality": self.quality,
					"n": self.number }
			resp = requests.post( Endpoints( ).image_generations, headers=headers, json=payload )
			if resp.status_code == 200:
				self.response = resp.json( );
				return self.response[ 'data' ][ 0 ][ 'url' ]
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Image'
			exception.method = 'generate( self, prompt, model, quality, size )';
			error = ErrorDialog( exception );
			error.show( )
	
	def analyze( self, text: str, path: str, model: str = 'llama-3.2-90b-vision-preview' ) -> \
	Optional[ str ]:
		"""Purpose: Vision content analysis via Groq LPU."""
		try:
			throw_if( 'text', text );
			throw_if( 'path', path )
			self.input_text = text;
			self.file_path = path;
			self.model = model
			b64 = encode_image( self.file_path )
			messages = [
			{
	             "role": "user",
	             "content": [
	             {
                      "type": "text",
                      "text": self.input_text
	             },
                  {
                      "type": "image_url",
                      "image_url":
                      {
	                          "url": f"data:image/jpeg;base64,{b64}"
                      }
                  } ]
			} ]
			
			self.response = self.client.chat.completions.create( model=self.model, messages=messages )
			return self.response.choices[ 0 ].message.content
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Image'
			exception.method = 'analyze( self, text, path, model )';
			error = ErrorDialog( exception );
			error.show( )
	
	def edit( self, prompt: str, path: str, model: str = 'dall-e-2' ) -> Optional[ str ]:
		"""Purpose: Modifies a local image via Hybrid service POST."""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'path', path )
			self.input_text = prompt;
			self.file_path = path;
			self.model = model
			headers = { "Authorization": f"Bearer {cfg.OPENAI_API_KEY}" }
			files = { "image": open( self.file_path, "rb" ) }
			data = \
			{
					"prompt": self.input_text,
					"n": self.number,
					"size": self.size,
					"model": self.model
			}
			
			resp = requests.post( Endpoints( ).image_edits, headers=headers, files=files, data=data )
			if resp.status_code == 200:
				return resp.json( )[ 'data' ][ 0 ][ 'url' ]
			return None
		except Exception as e:
			exception = Error( e );
			exception.module = 'grok';
			exception.cause = 'Image'
			exception.method = 'edit( self, prompt, path, model )';
			error = ErrorDialog( exception );
			error.show( )