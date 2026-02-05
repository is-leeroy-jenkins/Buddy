'''
  ******************************************************************************************
      Assembly:                Buddy
      Filename:                missy.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="missy.py" company="Terry D. Eppler">

	     missy.py
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
    missy.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests
from boogr import Error, ErrorDialog
import config as cfg


def throw_if( name: str, value: object ) -> None:
	"""
	
		Purpose:
		--------
		Guard clause that raises a ValueError if a required argument is None.
	
		Parameters:
		-----------
		name : str
			Name of the argument being validated.
		value : object
			Value to validate.
	
		Returns:
		--------
		None
	
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Mistral( ):
	"""
	
	    Purpose:
	    --------
	    Base class for Mistral functionality. Encapsulates config, HTTP helpers, and common
	    parameters used across specialized classes.
	
	    Attributes:
	    -----------
	    base_url : str
	        API root for Mistral.
	    headers : Dict[str, str]
	        Default headers used for requests.
	    timeout : int
	        Default request timeout (seconds).
	    api_key : Optional[str]
	        API key loaded from config.
	    modalities : Optional[List[str]]
	        Supported modalities by default.
	    stops : Optional[List[str]]
	        Default stop tokens.
	    response_format : Optional[str]
	        Default response format.
	    number : Optional[int]
	        Default number of completions to request.
	    temperature : Optional[float]
	        Default sampling temperature.
	    top_percent : Optional[float]
	        Default nucleus sampling value.
	    frequency_penalty : Optional[float]
	        Default frequency penalty.
	    presence_penalty : Optional[float]
	        Default presence penalty.
	    max_completion_tokens : Optional[int]
	        Default token generation limit.
			
	"""
	api_key: str
	base_url: str
	headers: Dict[ str, str ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	max_tokens: Optional[ int ]
	stream: Optional[ bool ]
	prompt: Optional[ str ]
	response: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize shared configuration for Mistral API usage.
	
			Returns:
			--------
			None
			
		"""
		self.base_url = 'https://api.mistral.ai/v1'
		self.api_key = cfg.MISTRAL_API_KEY
		self.headers = { 'Authorization': f'Bearer {self.api_key}' } if self.api_key else { }
		self.timeout = 120
		self.modalities  = [ 'text', 'images', 'audio' ]
		self.stops: Optional[ List[ str ] ] = [ '#', ';' ]
		self.response_format = None
		self.number: Optional[ int ] = None
		self.temperature: Optional[ float ] = None
		self.top_percent: Optional[ float ] = None
		self.frequency_penalty: Optional[ float ] = None
		self.presence_penalty: Optional[ float ] = None
		self.max_completion_tokens: Optional[ int ] = None

class Chat( Mistral ):
	"""
	
		Purpose:
		--------
		Class providing chat and text-generation capabilities using Mistral chat models.
		This mirrors the Chat(GPT) interface and usage patterns found in gpt.py.
	
		Attributes:
		-----------
		model : Optional[str]
			Name of the Mistral model to use.
		messages : Optional[List[Dict[str, Any]]]
			Message payload sent to the chat completion endpoint.
	
		Methods:
		--------
		generate_text(prompt, model, temperature, top_p, max_tokens, stream)
			Generate a text or chat completion.
			
	"""
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ str ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	image_url: Optional[ str ]
	search_recency: Optional[ int ]
	max_search_results: Optional[ int ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	vector_stores: Optional[ Dict[ str, str ] ]
	content: Optional[ List[ Dict[ str, Any ] ] ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	file: Optional[ Any ]
	response: Optional[ Any ]

	def __init__( self, number: int = 1, temperature: float = 0.8, top_p: float = 0.9, frequency: float = 0.0,
			presence: float = 0.0, max_tokens: int = 10000, store: bool = True, stream: bool = False,
			instruct: Optional[ str ] = None ) -> None:
			"""
				
				Purpose:
				--------
				Initialize Chat with default sampling/control parameters.
			
				Parameters:
				-----------
				number : int
					Number of completions to request.
				temperature : float
					Sampling temperature.
				top_p : float
					Nucleus sampling parameter.
				frequency : float
					Frequency penalty.
				presence : float
					Presence penalty.
				max_tokens : int
					Maximum tokens to generate.
				store : bool
					Whether to store results.
				stream : bool
					Whether to use streaming endpoints.
				instruct : Optional[str]
					System-level instructions.
			
				Returns:
				--------
				None
				
			"""
			super( ).__init__( )
			self.number = number
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_completion_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.instructions = instruct
			self.tool_choice = 'auto'
			self.model = None
			self.content = None
			self.prompt = None
			self.response = None
			self.file = None
			self.file_path = None
			self.file_ids = [ ]
			self.input = None
			self.messages = None
			self.image_url = None
			self.tools = None
			self.include = None
			self.search_recency = None
			self.max_search_results = None
			self.purpose = None
			self.vector_stores = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Representative chat-capable models for UI lists.
		
			Returns:
			--------
			List[str]
		
		"""
		return [ 'mistral-small',
		         'mistral-medium',
		         'mistral-large',
		         'mistral-instruct' ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			List of include options for responses/tools.
		
			Returns:
			--------
			List[str]
		
		"""
		return [ 'file',
		         'web_search',
		         'image',
		         'logprobs',
		         'debug' ]
	
	@property
	def purpose_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			File purpose options.
		
			Returns:
			--------
			List[str]
		
		"""
		return [ 'assistants',
		         'batch',
		         'fine-tune',
		         'vision',
		         'user_data' ]
	
	def generate_text( self, prompt: str, model: str='mistral-small', number: int=1, temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			store: bool=True, stream: bool=False, instruct: str=None ) -> str | None:
		"""
		
			Purpose:
			--------
			Generate a chat/text completion.
		
			Parameters:
			-----------
			prompt : str
				Prompt or user input text.
			model : str
				Model id.
			number : int
				Number of completions.
			temperature : float
				Sampling temperature.
			top_p : float
				Nucleus sampling value.
			frequency : float
				Frequency penalty.
			presence : float
				Presence penalty.
			max_tokens : int
				Max tokens to generate.
			store : bool
				Whether to store outputs.
			stream : bool
				Whether to stream results.
			instruct : Optional[str]
				System instructions.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'model', model )
			self.prompt = prompt;
			self.model = model;
			self.number = number
			self.temperature = temperature;
			self.top_percent = top_p
			self.frequency_penalty = frequency;
			self.presence_penalty = presence
			self.max_completion_tokens = max_tokens;
			self.store = store;
			self.stream = stream
			self.instructions = instruct
			payload: Dict[ str, Any ] = {
					'model': self.model,
					'input': self.prompt,
					'max_output_tokens': self.max_completion_tokens }
			if self.temperature is not None:
				payload[ 'temperature' ] = self.temperature
			if self.top_percent is not None:
				payload[ 'top_p' ] = self.top_percent
			if not stream:
				resp = self._post_json( '/responses', payload, stream=False )
				data = resp.json( )
				return data.get( 'output_text' ) or data.get( 'output', '' )
			resp = self._post_json( '/responses', payload, stream=True )
			
			def _gen( ) -> Generator[ str, None, None ]:
				for line in resp.iter_lines( ):
					evt = _as_sse_json( line )
					if not evt:
						continue
					out = evt.get( 'output_text' ) or evt.get( 'output' )
					if isinstance( out, str ):
						yield out
			return ''.join( list( _gen( ) ) )
		except Exception as e:
			ex = Error( e );
			ex.module = 'mistral';
			ex.cause = 'Chat';
			ex.method = 'generate_text(self, prompt)'
			error = ErrorDialog( ex )
			error.show( )
	
	def generate_image( self, prompt: str, number: int=1, model: str='mistral-image', size: str='1024x1024',
			quality: str='standard', fmt: str='url' ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Generate an image given a prompt.
		
			Parameters:
			-----------
			prompt : str
				Text prompt guiding image generation.
			number : int
				Number of images to create.
			model : str
				Image model id.
			size : str
				Size spec (e.g., '1024x1024').
			quality : str
				Quality preset.
			fmt : str
				Response format ('url'|'base64'|'.png').
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			payload = { 'model': model, 'prompt': prompt, 'n': number,
					'size': size, 'quality': quality, 'response_format': fmt }
			resp = self._post_json( '/images/generate', payload, stream=False )
			data = resp.json( )
			if fmt == 'url':
				return data.get( 'data', [ { } ] )[ 0 ].get( 'url' )
			return data
		except Exception as e:
			ex = Error( e )
			ex.module = 'mistral'
			ex.cause = 'Chat'
			ex.method = 'generate_image(self, prompt)'
			error = ErrorDialog( ex )
			error.show( )
	
	def analyze_image( self, prompt: str, url: str, model: str = 'mistral-vision', max_tokens: int = 1024 ) -> \
	Optional[ str ]:
		"""
		
			Purpose:
			--------
			Analyze an image with a textual instruction.
		
			Parameters:
			-----------
			prompt : str
				Text instruction.
			url : str
				Public image URL.
			model : str
				Vision-capable model id.
			max_tokens : int
				Max tokens.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'url', url )
			input_payload = [ { 'role': 'user',
			                    'content': [ { 'type': 'text', 'text': prompt },
					                         { 'type': 'image_url',
							                'image_url': {'url': url } } ] } ]
			payload = {'model': model,'input': input_payload,
					'max_output_tokens': max_tokens }
			resp = self._post_json( '/responses', payload, stream=False )
			data = resp.json( )
			return data.get( 'output_text' ) or data.get( 'output' )
		except Exception as e:
			ex = Error( e );
			ex.module = 'mistral';
			ex.cause = 'Chat';
			ex.method = 'analyze_image(self, prompt, url)'
			error = ErrorDialog( ex )
			error.show( )
	
	def edit_image( self, prompt: str, src_url: str, dest_path: str,
			model: str='mistral-image',size: str='1024x1024' ) -> str | None:
		"""
			
			Purpose:
			--------
			Edit an image given instructions.
		
			Parameters:
			-----------
			prompt : str
				Instructions for editing.
			src_url : str
				Source image URL or path.
			dest_path : str
				Local destination path for edited image.
			model : str
				Image model id.
			size : str
				Output size.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'src_url', src_url );
			throw_if( 'dest_path', dest_path )
			payload = { 'model': model, 'prompt': prompt,
					'image_url': src_url, 'size': size }
			resp = self._post_json( '/images/edit', payload, stream=False )
			data = resp.json( )
			content = data.get( 'data', [ { } ] )[ 0 ]
			if isinstance( content, dict ) and 'b64_json' in content:
				out = Path( dest_path )
				out.parent.mkdir( parents=True, exist_ok=True )
				out.write_bytes( bytes( content[ 'b64_json' ], 'utf-8' ) )
				return str( out )
			return content.get( 'url' ) or None
		except Exception as e:
			ex = Error( e );
			ex.module = 'mistral';
			ex.cause = 'Chat';
			ex.method = 'edit_image(self, prompt, src_url, dest_path)'
			error = ErrorDialog( ex )
			error.show( )
	
	def summarize_document( self, prompt: str, pdf_path: str,
			model: str='mistral-instruct', max_tokens: int=2048 ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Summarize a document given a local PDF path.
		
			Parameters:
			-----------
			prompt : str
				Summarization prompt.
			pdf_path : str
				Path to local PDF.
			model : str
				Model id for document reasoning.
			max_tokens : int
				Max tokens.
		
			Returns:
			--------
			Optional[str]
			
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'pdf_path', pdf_path )
			files = Files( )
			f = files.upload( file_name=Path( pdf_path ).name, content_bytes=Path( pdf_path ).read_bytes( ), purpose='user_data' )
			file_id = f.get( 'id' ) if isinstance( f, dict ) else None
			input_payload = [ { 'role': 'user', 'content': [ {  'type': 'file',
			                                                    'file': {  'file_id': file_id } },
					                               { 'type': 'text', 'text': prompt } ] } ]
			payload = { 'model': model, 'input': input_payload,
					'max_output_tokens': max_tokens }
			resp = self._post_json( '/responses', payload, stream=False )
			data = resp.json( )
			return data.get( 'output_text' ) or data.get( 'output' )
		except Exception as e:
			ex = Error( e );
			ex.module = 'mistral'
			ex.cause = 'Chat'
			ex.method = 'summarize_document(self, prompt, pdf_path)'
			error = ErrorDialog( ex )
			error.show( )
	
	def search_web( self, prompt: str, model: str = 'mistral-small', recency: int = 30,
			max_results: int = 100 ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Run a web search augmented completion.
		
			Parameters:
			-----------
			prompt : str
				Query string to search.
			model : str
				Model id.
			recency : int
				Search recency in days.
			max_results : int
				Maximum search results.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			web_options = { 'search_recency_days': recency, 'max_search_results': max_results }
			payload = { 'model': model, 'web_search_options': web_options, 'input': prompt }
			resp = self._post_json( '/responses', payload, stream=False )
			data = resp.json( )
			return data.get( 'output_text' ) or data.get( 'output' )
		except Exception as e:
			ex = Error( e );
			ex.module = 'mistral';
			ex.cause = 'Chat';
			ex.method = 'search_web(self, prompt)'
			error = ErrorDialog( ex )
			error.show( )
	
	def search_files( self, prompt: str, model: str='mistral-small',
			store_id: str=None, max_results: int=8 ) -> str:
		"""
		
			Purpose:
			--------
			Run a file-store assisted search (vector store tool integration).
		
			Parameters:
			-----------
			prompt : str
				Query text.
			model : str
				Model id.
			store_id : Optional[str]
				Vector store id.
			max_results : int
				Max returned file results.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'model', model )
			tools = [ ]
			if store_id:
				tools = [ {'text': 'file_search','vector_store_ids': [ store_id ],
						    'max_num_results': max_results } ]
			payload = {'model': model,'tools': tools,'input': prompt }
			resp = self._post_json( '/responses', payload, stream=False )
			data = resp.json( )
			return data.get( 'output_text' ) or data.get( 'output' )
		except Exception as e:
			ex = Error( e );
			ex.module = 'mistral';
			ex.cause = 'Chat';
			ex.method = 'search_files(self, prompt)'
			error = ErrorDialog( ex )
			error.show( )
	
	def upload_file( self, filepath: str, purpose: str='user_data' ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Upload a local file and return its id.
		
			Parameters:
			-----------
			filepath : str
				Local path to file.
			purpose : str
				Upload purpose.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'filepath', filepath )
			files = Files( )
			result = files.upload( file_name=Path( filepath ).name, content_bytes=Path( filepath ).read_bytes( ), purpose=purpose )
			return result.get( 'id' ) if isinstance( result, dict ) else None
		except Exception as e:
			ex = Error( e );
			ex.module = 'mistral';
			ex.cause = 'Chat';
			ex.method = 'upload_file(self, filepath, purpose)'
			error = ErrorDialog( ex )
			error.show( )
	
	def __dir__( self ) -> List[ str ]:
		return [ 'number',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'content',
		         'prompt',
		         'response',
		         'generate_text',
		         'generate_image',
		         'analyze_image',
		         'edit_image',
		         'summarize_document',
		         'search_web',
		         'search_files',
		         'upload_file' ]

class Transcription( Mistral ):
	"""
		
		Purpose:
		--------
		Audio transcription wrapper.
	
	"""
	def __init__( self ) -> None:
		"""
			
			Purpose:
			--------
			Initialize transcription helper.
	
			Returns:
			--------
			None
			
		"""
		super( ).__init__( )
		self.model = None
		self.audio_file = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Transcription models.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'whisper-1', 'mistral-transcribe', 'voxtral-large' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Output formats.
	
			Returns:
			--------
			List[str]
			
		"""
		return [ 'text',
		         'json',
		         'srt',
		         'vtt',
		         'diarized_json' ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""
		Purpose:
		--------
		Supported languages.

		Returns:
		--------
		List[str]
		"""
		return [ 'en',
		         'es',
		         'fr',
		         'de',
		         'zh',
		         'it',
		         'ja' ]
	
	def transcribe( self, path: str, model: str='whisper-1',
			language: str='en', format: st ='text' ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Transcribe audio file.
	
			Parameters:
			-----------
			path : str
				Local audio path.
			model : str
				Transcription model id.
			language : Optional[str]
				Language hint.
			format : str
				Output format.
	
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'path', path );
			throw_if( 'model', model )
			with open( path, 'rb' ) as fh:
				files = { 'file': (Path( path ).name, fh) }
				data = { 'model': model, 'language': language, 'format': format }
				resp = requests.post( url=f'{self.base_url}/audio/transcriptions',
					headers=self.headers, files=files, data=data, timeout=self.timeout )
				resp.raise_for_status( )
				return resp.json( ).get( 'text' ) or resp.json( )
		except Exception as e:
			ex = Error( e );
			ex.module = 'mistral';
			ex.cause = 'Transcription';
			ex.method = 'transcribe(self, path)'
			error = ErrorDialog( ex )
			error.show( )

class Translation( Mistral ):
	"""
	
		Purpose:
		--------
		Audio translation wrapper (non-streaming).
	
	"""
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = None
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'whisper-1', 'mistral-translate' ]
	
	@property
	def language_options( self ) -> List[ str ]:
		return [ 'en', 'es', 'fr', 'de', 'zh', 'it', 'ja' ]
	
	def translate( self, path: str, source_lang: str=None, target_lang: str='en',
			model: str='whisper-1' ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Translate speech audio to target language.
	
			Parameters:
			-----------
			path : str
				Local audio path.
			source_lang : Optional[str]
				Source language hint.
			target_lang : Optional[str]
				Target language (default 'en').
			model : str
				Translation model id.
	
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'model', model )
			with open( path, 'rb' ) as fh:
				files = { 'file': (Path( path ).name, fh) }
				data = { 'model': model, 'source_language': source_lang, 'target_language': target_lang }
				resp = requests.post( url=f'{self.base_url}/audio/translations',
					headers=self.headers, files=files, data=data, timeout=self.timeout )
				resp.raise_for_status( )
				return resp.json( ).get( 'text' ) or resp.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'mistral'
			ex.cause = 'Translation'
			ex.method = 'translate(self, path)'
			error = ErrorDialog( ex )
			error.show( )

class TTS( Mistral ):
	"""
	
		Purpose:
		--------
		Text-to-speech helper.
	
	"""
	
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = None
		self.voice = None
		self.speed = None
		self.response_format = None
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'mistral-tts', 'gpt-4o-mini-tts', 'tts-1' ]
	
	@property
	def voice_options( self ) -> List[ str ]:
		return [ 'alloy', 'ash', 'ballad', 'coral', 'echo',
		         'fable', 'onyx', 'nova', 'sage', 'shiver' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		return [ 'mp3', 'wav',  'aac', 'flac', 'opus' ]
	
	@property
	def speed_options( self ) -> List[ float ]:
		return [ 0.25,  0.5, 1.0, 1.5, 2.0 ]
	
	def create_audio( self, text: str, output_path: str, model: str='mistral-tts',
			voice: str='alloy', speed: float=1.0, fmt: str='mp3' ) -> str | None:
		"""
		
			Purpose:
			--------
			Create TTS audio file.
	
			Parameters:
			-----------
			text : str
				Input text to synthesize.
			output_path : str
				Local path to write audio to.
			model : str
				TTS model id.
			voice : str
				Voice spec.
			speed : float
				Speech rate.
			fmt : str
				Output format.
	
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'output_path', output_path )
			throw_if( 'model', model )
			payload = { 'model': model, 'voice': voice, 'speed': speed,
					'format': fmt, 'input': text }
			resp = self._post_json( '/audio/speech', payload, stream=False )
			data = resp.json( )
			content = data.get( 'data', [ { } ] )[ 0 ]
			if isinstance( content, dict ) and 'b64_audio' in content:
				out = Path( output_path )
				out.parent.mkdir( parents=True, exist_ok=True )
				out.write_bytes( bytes( content[ 'b64_audio' ], 'utf-8' ) )
				return str( out )
			return content.get( 'url' ) or None
		except Exception as e:
			ex = Error( e )
			ex.module = 'mistral'
			ex.cause = 'TTS'
			ex.method = 'create_audio(self, text, output_path)'
			error = ErrorDialog( ex )
			error.show( )
			
class Embedding( Mistral ):
	"""
		
		Purpose:
		--------
		Class providing vector embedding generation using Mistral embedding models. This mirrors
		the Embedding(GPT) interface and conventions found in gpt.py.
	
		Attributes:
		-----------
		model : Optional[str]
			Name of the Mistral embedding model to use.
		vectors : Optional[List[List[float]]]
			Generated embedding vectors.
	
		Methods:
		--------
		create(input_text, model)
			Generate embeddings for input text.
		
	"""
	model: Optional[ str ]
	vectors: Optional[ List[ List[ float ] ] ]
	
	def __init__( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize an Embedding instance for Mistral embedding generation.
	
			Returns:
			--------
			None
			
		"""
		super( ).__init__( )
		self.model = None
		self.vectors = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			List supported Mistral embedding models.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'mistral-embed', ]
	
	def create( self, input_text: List[ str ], model: str='mistral-embed' ) -> List[ List[ float ] ] | None:
		"""
		
			Purpose:
			--------
			Generate embeddings for the provided input text using the Mistral API.
	
			Parameters:
			-----------
			input_text : str | List[str]
				Input text or list of text segments to embed.
			model : str
				Name of the Mistral embedding model to use.
	
			Returns:
			--------
			List[List[float]] | None
				List of embedding vectors.
				
		"""
		try:
			throw_if( 'input_text', input_text )
			self.model = model
			payload  = { 'model': self.model, 'input': input_text, }
			response = requests.post( url=f'{self.base_url}/embeddings', headers=self.headers,
				json=payload, timeout=120, )
			response.raise_for_status( )
			data = response.json( )
			self.vectors = [ item[ 'embedding' ] for item in data.get( 'data', [ ] ) ]
			return self.vectors
		except Exception as e:
			exception = Error( e )
			exception.module = 'mistral'
			exception.cause = 'Embedding'
			exception.method = 'create(self, input_text, model)'
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Return a list of public members for introspection and UI binding.
	
			Returns:
			--------
			List[str] | None
			
		"""
		return [ 'model', 'model_options', 'vectors', 'create', ]

class TTS( Mistral ):
	"""
		
		Purpose:
		--------
		Text-to-speech helper.
	
	"""
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = None
		self.voice = None
		self.speed = None
		self.response_format = None
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'mistral-tts',
		         'gpt-4o-mini-tts',
		         'tts-1' ]
	
	@property
	def voice_options( self ) -> List[ str ]:
		return [ 'alloy',
		         'ash',
		         'ballad',
		         'coral',
		         'echo',
		         'fable',
		         'onyx',
		         'nova',
		         'sage',
		         'shiver' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		return [ 'mp3', 'wav', 'aac', 'flac', 'opus' ]
	
	@property
	def speed_options( self ) -> List[ float ]:
		return [ 0.25,
		         0.5,
		         1.0,
		         1.5,
		         2.0 ]
	
	def create_audio( self, text: str, output_path: str, model: str='mistral-tts',
			voice: str='alloy', speed: float=1.0, fmt: str='mp3' ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Create TTS audio file.
	
			Parameters:
			-----------
			text : str
				Input text to synthesize.
			output_path : str
				Local path to write audio to.
			model : str
				TTS model id.
			voice : str
				Voice spec.
			speed : float
				Speech rate.
			fmt : str
				Output format.
	
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'text', text );
			throw_if( 'output_path', output_path );
			throw_if( 'model', model )
			payload = { 'model': model, 'voice': voice,
					'speed': speed, 'format': fmt, 'input': text }
			resp = self._post_json( '/audio/speech', payload, stream=False )
			data = resp.json( )
			content = data.get( 'data', [ { } ] )[ 0 ]
			if isinstance( content, dict ) and 'b64_audio' in content:
				out = Path( output_path );
				out.parent.mkdir( parents=True, exist_ok=True )
				out.write_bytes( bytes( content[ 'b64_audio' ], 'utf-8' ) )
				return str( out )
			return content.get( 'url' ) or None
		except Exception as e:
			ex = Error( e )
			ex.module = 'mistral'
			ex.cause = 'TTS'
			ex.method = 'create_audio(self, text, output_path)'
			error = ErrorDialog( ex )
			error.show( )
			
class Image( Mistral ):
	"""
	
		Purpose:
		--------
		Provide multimodal image understanding and analysis using Mistral vision-capable chat
		models. This class isolates image reasoning from standard text chat while continuing
		to use the Chat Completions API.
	
		Attributes:
		-----------
		model : Optional[str]
			Name of the Mistral vision-capable model.
		prompt : Optional[str]
			Instruction or question associated with the image.
		image_url : Optional[str]
			URL of the image to analyze.
		response : Optional[Dict[str, Any]]
			Raw response payload from the API.
			
	"""
	model: Optional[ str ]
	prompt: Optional[ str ]
	image_url: Optional[ str ]
	response: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize a Vision instance for multimodal image analysis.
	
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.model = None
		self.prompt = None
		self.image_url = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			List supported Mistral vision-capable models.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'mistral-large-latest', 'mistral-large-2512', ]
	
	def analyze_image( self, prompt: str, image_url: str, model: str='mistral-large-latest',
			temperature: float=None, top_p: float = None, max_tokens: int=None ) -> str | None:
		"""
			
			Purpose:
			--------
			Analyze an image using a multimodal prompt via the Mistral Chat Completions API.
	
			Parameters:
			-----------
			prompt : str
				Instruction or question describing the desired image analysis.
			image_url : str
				URL pointing to the image to analyze.
			model : str
				Vision-capable Mistral model name.
			temperature : Optional[float]
				Sampling temperature override.
			top_p : Optional[float]
				Nucleus sampling override.
			max_tokens : Optional[int]
				Maximum token override.
	
			Returns:
			--------
			str | None
				Generated image analysis output.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'image_url', image_url )
			self.prompt = prompt
			self.image_url = image_url
			self.model = model
			payload = { 'model': self.model, 'messages': [
						{ 'role': 'user', 'content': [
						{ 'type': 'text', 'text': self.prompt, },
						{ 'type': 'image_url',
						  'image_url': { 'url': self.image_url, }, }, ], } ],
					'temperature': temperature if temperature is not None else self.temperature,
					'top_p': top_p if top_p is not None else self.top_p,
					'max_tokens': max_tokens if max_tokens is not None else self.max_tokens,
			}
			
			response = requests.post( url=f'{self.base_url}/chat/completions',
				headers=self.headers, json=payload, timeout=120, )
			response.raise_for_status( )
			self.response = response.json( )
			return self.response[ 'choices' ][ 0 ][ 'message' ][ 'content' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'mistral'
			exception.cause = 'Vision'
			exception.method = (
					'analyze_image(self, prompt, image_url, model, temperature, top_p, max_tokens)'
			)
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Return public members for introspection and UI binding.
	
			Returns:
			--------
			List[str] | None
		
		"""
		return [
				'model',
				'model_options',
				'prompt',
				'image_url',
				'response',
				'analyze_image',
		]


class Document( Mistral ):
	"""
	
		Purpose:
		--------
		Provide document-oriented reasoning, summarization, and structured extraction by
		applying Mistral language models to locally extracted document text.
	
		Attributes:
		-----------
		model : Optional[str]
			Name of the Mistral model used for document reasoning.
		document_text : Optional[str]
			Extracted textual content of the document.
		response : Optional[Dict[str, Any]]
			Raw response payload from the API.
		
	"""
	model: Optional[ str ]
	document_text: Optional[ str ]
	response: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ) -> None:
		"""
			
			Purpose:
			--------
			Initialize a Document instance for document-level reasoning.
	
			Returns:
			--------
			None
			
		"""
		super( ).__init__( )
		self.model = None
		self.document_text = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			List supported Mistral models appropriate for document analysis.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'mistral-large-latest', 'mistral-large-2512', ]
	
	def analyze_document( self, document_text: str, prompt: str,
			model: str='mistral-large-latest', temperature: float=None,
			top_p: float=None, max_tokens: int=None ) -> str | None:
		"""
		
			Purpose:
			--------
			Analyze extracted document text using a structured prompt and Mistral reasoning.
	
			Parameters:
			-----------
			document_text : str
				Extracted textual content of the document.
			prompt : str
				Instruction describing the desired analysis or extraction.
			model : str
				Mistral model name.
			temperature : Optional[float]
				Sampling temperature override.
			top_p : Optional[float]
				Nucleus sampling override.
			max_tokens : Optional[int]
				Maximum token override.
	
			Returns:
			--------
			str | None
				Generated document analysis output.
			
		"""
		try:
			throw_if( 'document_text', document_text )
			throw_if( 'prompt', prompt )
			self.document_text = document_text
			self.model = model
			payload = {'model': self.model,
			           'messages': [{'role': 'user','content': f'{prompt}\n\n{self.document_text}',}],
					'temperature': temperature if temperature is not None else self.temperature,
					'top_p': top_p if top_p is not None else self.top_p,
					'max_tokens': max_tokens if max_tokens is not None else self.max_tokens,}
			
			response = requests.post( url=f'{self.base_url}/chat/completions', headers=self.headers,
				json=payload, timeout=120, )
			response.raise_for_status( )
			self.response = response.json( )
			return self.response[ 'choices' ][ 0 ][ 'message' ][ 'content' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'mistral'
			exception.cause = 'Document'
			exception.method = 'analyze_document(self, text, prompt, model, temp, top_p, tokens)'
			error = ErrorDialog( exception )
			error.show( )
	
	def __dir__( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Return public members for introspection and UI binding.
	
			Returns:
			--------
			List[str] | None
		
		"""
		return [
				'model',
				'model_options',
				'document_text',
				'response',
				'analyze_document',
		]