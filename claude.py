'''
  ******************************************************************************************
      Assembly:                Buddy
      Filename:                claude.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="claude.py" company="Terry D. Eppler">

	     claude.py
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
    claude.py
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

class Claude:
	"""
		
		Purpose:
		--------
		Base class encapsulating shared configuration and HTTP behavior for Anthropic Claude.
	
		Attributes:
		-----------
		api_key : str
			Anthropic API key.
		base_url : str
			Claude API base URL.
		headers : Dict[str, str]
			Default HTTP headers.
		response : Optional[Dict[str, Any]]
			Raw API response payload.
		
	"""
	api_key: str
	base_url: str
	headers: Dict[ str, str ]
	response: Optional[ Dict[ str, Any ] ]
	number: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	max_completion_tokens: Optional[ int ]
	response_format: Optional[ str ]
	stops: Optional[ List[ str ] ]
	modalities: Optional[ List[ str ] ]
	timeout: Optional[ int ]
	
	def __init__( self ) -> None:
		"""
			
			Purpose:
			--------
			Initialize shared configuration for Claude API usage.
	
			Returns:
			--------
			None
		
		"""
		self.api_key: Optional[ str ] = getattr( cfg, 'CLAUDE_API_KEY', None )
		self.base_url: str = 'https://api.anthropic.com/v1'
		self.headers: Dict[ str, str ] = { 'x-api-key': self.api_key or '',
				'Content-Type': 'application/json', }
		self.timeout  = 120
		self.modalities  = [ 'text', 'images',  'audio' ]
		self.stops  = [ '#',  ';' ]
		self.number = None
		self.temperature = None
		self.top_p = None
		self.max_completion_tokens = None
		self.response_format = None

class Chat( Claude ):
	"""
		
		Purpose:
		--------
		Provide text-based chat and reasoning using Claude models.
	
		Attributes:
		-----------
		model : Optional[str]
			Claude model name.
		system_prompt : Optional[str]
			Optional system instruction.
		
	"""
	model: Optional[ str ];
	messages: Optional[ List[ Dict[ str, Any ] ] ];
	system_prompt: Optional[ str ]
	
	def __init__( self, number: int = 1, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 8192,
			stream: bool = False, system_prompt: Optional[ str ] = None ) -> None:
		"""
		
			Purpose:
			--------
			Initialize Chat instance with sampling defaults.
		
			Parameters:
			-----------
			number : int
				Number of completions to request.
			temperature : float
				Sampling temperature.
			top_p : float
				Nucleus sampling value.
			max_tokens : int
				Maximum tokens to generate.
			stream : bool
				Whether to request streaming output.
			system_prompt : Optional[str]
				Optional system-level instruction.
		
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.number = number
		self.temperature = temperature
		self.top_p = top_p
		self.max_completion_tokens = max_tokens
		self.stream = stream
		self.system_prompt = system_prompt
		self.model = None
		self.messages = None
		self.prompt = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Representative Claude models available for chat.
		
			Returns:
			--------
			List[str]
			
		"""
		return [ 'claude-3-opus', 'claude-3-sonnet', 'claude-2',  'claude-instant' ]
	
	def generate_text( self, prompt: str, model: str = 'claude-3-sonnet', temperature: Optional[
		float ] = None, top_p: Optional[ float ] = None, max_tokens: Optional[ int ] = None, stream: Optional[
				bool ] = None ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Generate a chat completion using Claude messages endpoint.
		
			Parameters:
			-----------
			prompt : str
				User prompt text.
			model : str
				Model id to use.
			temperature : Optional[float]
				Temperature override.
			top_p : Optional[float]
				Top-p override.
			max_tokens : Optional[int]
				Max tokens override.
			stream : Optional[bool]
				Streaming override.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'model', model )
			self.prompt = prompt;
			self.model = model
			payload: Dict[ str, Any ] = {
					'model': self.model,
					'messages': [ {
							              'role': 'user',
							              'content': self.prompt } ],
					'max_tokens': max_tokens if max_tokens is not None else self.max_completion_tokens,
			}
			if temperature is not None:
				payload[ 'temperature' ] = temperature
			if top_p is not None:
				payload[ 'top_p' ] = top_p
			if self.system_prompt:
				payload[ 'system' ] = self.system_prompt
			use_stream = stream if stream is not None else self.stream
			if not use_stream:
				resp = self._post_json( '/messages', payload, stream=False )
				data = resp.json( )
				# Anthropic responses often nest content; try common paths
				if isinstance( data, dict ):
					if 'completion' in data:
						return data.get( 'completion' )
					if 'messages' in data and isinstance( data[ 'messages' ], list ):
						first = data[ 'messages' ][ 0 ]
						return first.get( 'content' ) if isinstance( first, dict ) else None
					# fallback
					return data.get( 'output' ) or data.get( 'content' )
				return None
			# streaming (SSE)
			resp = self._post_json( '/messages', payload, stream=True )
			
			def _gen( ) -> Generator[ str, None, None ]:
				for line in resp.iter_lines( ):
					evt = _as_sse_json( line )
					if not evt:
						continue
					# attempt to extract progressive text
					if 'completion' in evt:
						yield evt[ 'completion' ]
					elif 'delta' in evt and isinstance( evt[ 'delta' ], dict ):
						yield evt[ 'delta' ].get( 'content', '' )
			
			return ''.join( list( _gen( ) ) )
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Chat'
			ex.method = 'generate_text(self, prompt)'
			error = ErrorDialog( ex )
			error.show( )
	
	def analyze_image( self, prompt: str, image_url: str,
			model: str='claude-vision', max_tokens: int=2048 ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Send a multimodal (image + text) instruction and return analysis text.
		
			Parameters:
			-----------
			prompt : str
				Instruction guiding image analysis.
			image_url : str
				Publicly accessible image URL.
			model : str
				Vision-capable Claude model.
			max_tokens : int
				Maximum tokens to generate.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'image_url', image_url )
			payload = { 'model': model, 'messages': [ { 'role': 'user',
						 'content': self.prompt or prompt,  'image_url': image_url } ],
					'max_tokens': max_tokens, }
			resp = self._post_json( '/messages', payload, stream=False )
			data = resp.json( )
			return data.get( 'completion' ) or data.get( 'output' ) or None
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Chat'
			ex.method = 'analyze_image(self, prompt, image_url)'
			error = ErrorDialog( ex )
			error.show( )
	
	def summarize_document( self, prompt: str, pdf_path: str,
			model: str='claude-3-opus', max_tokens: int=2048 ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Summarize a local document by uploading it and then invoking a completion.
		
			Parameters:
			-----------
			prompt : str
				Summarization instructions.
			pdf_path : str
				Local PDF path.
			model : str
				Claude model id.
			max_tokens : int
				Maximum tokens for summary.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'pdf_path', pdf_path )
			files = Files( )
			upload_resp = files.upload( file_name=Path( pdf_path ).name,
				content_bytes=Path( pdf_path ).read_bytes( ), purpose='user_data' )
			file_id = upload_resp.get( 'id' ) if isinstance( upload_resp, dict ) else None
			if not file_id:
				raise RuntimeError( 'File upload failed' )
			payload = {
					'model': model,
					'messages': [ { 'role': 'user','content': prompt,
							      'file': { 'file_id': file_id } } ],
					'max_tokens': max_tokens, }
			resp = self._post_json( '/messages', payload, stream=False )
			return resp.json( ).get( 'completion' ) or resp.json( ).get( 'output' )
		except Exception as e:
			ex = Error( e );
			ex.module = 'claude'
			ex.cause = 'Chat'
			ex.method = 'summarize_document(self, prompt, pdf_path)'
			error = ErrorDialog( ex )
			error.show( )
	
	def search_web( self, prompt: str, model: str='claude-3-opus',
			recency: int=30, max_results: int=100 ) -> str :
		"""
		
			Purpose:
			--------
			Perform a web-augmented chat completion (web search integration).
		
			Parameters:
			-----------
			prompt : str
				Query or instruction.
			model : str
				Model id.
			recency : int
				Recency in days for web results.
			max_results : int
				Max number of web results.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'model', model )
			web_options = { 'search_recency_days': recency, 'max_search_results': max_results }
			payload = { 'model': model, 'messages': [ {
						 'role': 'user', 'content': prompt } ], 'web_search_options': web_options }
			resp = self._post_json( '/messages', payload, stream=False )
			return resp.json( ).get( 'completion' ) or resp.json( ).get( 'output' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Chat'
			ex.method = 'search_web(self, prompt)'
			error = ErrorDialog( ex )
			error.show( )

	def upload_file( self, filepath: str, purpose: str='user_data' ) -> str:
		"""
			
			Purpose:
			--------
			Upload a local file to Claude's file endpoint.
		
			Parameters:
			-----------
			filepath : str
				Local path to file.
			purpose : str
				Purpose string for the upload.
		
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'filepath', filepath )
			files = Files( )
			result = files.upload( file_name=Path( filepath ).name,
				content_bytes=Path( filepath ).read_bytes( ), purpose=purpose )
			return result.get( 'id' ) if isinstance( result, dict ) else None
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Chat'
			ex.method = 'upload_file(self, filepath, purpose)'
			error = ErrorDialog( ex )
			error.show( )
	
	def __dir__( self ) -> List[ str ]:
		return [ 'model',
		         'model_options',
		         'generate_text',
		         'analyze_image',
		         'summarize_document',
		         'search_web',
		         'upload_file' ]

class Images( Claude ):
	"""
	
		Purpose:
		--------
		Images wrapper for generation and image analysis where Claude models provide
		vision-capabilities.
	
		Methods:
		--------
		generate(...) -> image url/base64
		analyze(...) -> textual analysis
		edit(...) -> edited image url/local path
		
	"""
	def __init__( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize Images helper.
	
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.model = None
		self.quality = None
		self.size = None
		self.style = None
		self.response_format = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Image / multimodal capable Claude model ids.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'claude-vision-1',
		         'claude-3-opus',
		         'claude-instant-vision' ]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Standard image size options.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ '256x256', '512x512', '1024x1024', '1792x1024' ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Quality presets for generation.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'low', 'medium', 'high' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Response formats supported.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'url', 'base64',  'png', 'jpeg' ]
	
	def generate( self, prompt: str, model: str='claude-instant-vision', number: int=1, size: str='1024x1024',
			quality: str='standard', style: str ='natural', fmt: str='url' ) -> str:
		"""
		
			Purpose:
			--------
			Generate images using Claude image generation capabilities if provided.
	
			Parameters:
			-----------
			prompt : str
				Text prompt for generation.
			model : str
				Model id.
			number : int
				Number of images.
			size : str
				Output size.
			quality : str
				Quality preset.
			style : str
				Style preset.
			fmt : str
				Response format.
	
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			payload = {
					'model': model,
					'prompt': prompt,
					'n': number,
					'size': size,
					'quality': quality,
					'style': style }
			resp = self._post_json( '/images/generate', payload, stream=False )
			data = resp.json( )
			if fmt == 'url':
				return data.get( 'data', [ { } ] )[ 0 ].get( 'url' )
			return data
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Images'
			ex.method = 'generate(self, prompt)'
			error = ErrorDialog( ex )
			error.show( )
	
	def analyze( self, prompt: str, image_url: str,
			model: str='claude-vision-1', max_tokens: int=2048 ) -> str:
		"""
		
			Purpose:
			--------
			Analyze a remote image given a textual instruction.
	
			Parameters:
			-----------
			prompt : str
				Instruction text.
			image_url : str
				Public image URL.
			model : str
				Vision-capable model id.
			max_tokens : int
				Maximum tokens to generate.
	
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'image_url', image_url )
			messages = [ { 'role': 'user', 'content': prompt, 'image_url': image_url } ]
			payload = { 'model': model, 'messages': messages, 'max_tokens': max_tokens }
			resp = self._post_json( '/messages', payload, stream=False )
			return resp.json( ).get( 'completion' ) or resp.json( ).get( 'output' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Images'
			ex.method = 'analyze(self, prompt, image_url)'
			error = ErrorDialog( ex )
			error.show( )
	
	def edit( self, prompt: str, src_url: str, output_path: str,
			model: str='claude-instant-vision', size: str='1024x1024' ) -> str:
		"""
			
			Purpose:
			--------
			Edit an image and save or return the edited result.
	
			Parameters:
			-----------
			prompt : str
				Editing instructions.
			src_url : str
				Source image URL.
			output_path : str
				Local destination path if returned base64 is saved.
			model : str
				Model id.
			size : str
				Output size.
	
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'prompt', prompt );
			throw_if( 'src_url', src_url );
			throw_if( 'output_path', output_path )
			payload = {
					'model': model,
					'prompt': prompt,
					'image_url': src_url,
					'size': size }
			resp = self._post_json( '/images/edit', payload, stream=False )
			data = resp.json( )
			content = data.get( 'data', [ { } ] )[ 0 ]
			if isinstance( content, dict ) and 'b64_json' in content:
				out = Path( output_path );
				out.parent.mkdir( parents=True, exist_ok=True )
				out.write_bytes( bytes( content[ 'b64_json' ], 'utf-8' ) )
				return str( out )
			return content.get( 'url' ) or None
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.method = 'edit(self, prompt, src_url, output_path)'
			error = ErrorDialog( ex )
			error.show( )
	
	def __dir__( self ) -> List[ str ]:
		return [ 'model_options',
		         'size_options',
		         'quality_options',
		         'format_options',
		         'generate',
		         'analyze',
		         'edit' ]

class Embedding( Claude ):
	"""
		
		Purpose:
		--------
		Embedding helper using Claude embedding endpoints.
		
	"""
	
	def __init__( self ) -> None:
		"""
			
			Purpose:
			--------
			Initialize embedding helper.
	
			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.model = None
		self.encoding_format = None
		self.dimensions = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""
			
			Purpose:
			--------
			Embedding model ids.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'claude-embed',
		         'claude-embed-2' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Encoding formats available.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'float32',
		         'float64',
		         'base64' ]
	
	def create( self, text: Union[
		str, List[ str ] ], model: str = 'claude-embed', encoding_format: str = 'float32',
			metadata: Optional[ Dict[ str, Any ] ] = None, output_dimension: Optional[
				int ] = None ) -> Optional[ List[ float ] ]:
		"""
		
			Purpose:
			--------
			Create embeddings for provided text or list of texts.
	
			Parameters:
			-----------
			text : Union[str, List[str]]
				Text or array of texts to embed.
			model : str
				Embedding model id.
			encoding_format : str
				Output encoding format.
			metadata : Optional[Dict[str, Any]]
				Optional metadata per input.
			output_dimension : Optional[int]
				Desired output dimension.
	
			Returns:
			--------
			Optional[List[float]]
		
		"""
		try:
			throw_if( 'text', text );
			throw_if( 'model', model )
			payload: Dict[ str, Any ] = {
					'model': model,
					'input': text }
			if encoding_format is not None:
				payload[ 'encoding_format' ] = encoding_format
			if metadata is not None:
				payload[ 'metadata' ] = metadata
			if output_dimension is not None:
				payload[ 'output_dimension' ] = output_dimension
			resp = self._post_json( '/embeddings', payload, stream=False )
			data = resp.json( )
			items = data.get( 'data', [ ] )
			if not items:
				return None
			return items[ 0 ].get( 'embedding' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Embedding'
			ex.method = 'create(self, text, model)'
			error = ErrorDialog( ex )
			error.show( )
	
	def __dir__( self ) -> List[ str ]:
		return [ 'model_options', 'encoding_options', 'create' ]

class Files( Claude ):
	"""
		
		Purpose:
		--------
		File upload / list / retrieve / delete helpers for Claude-compatible file endpoints.
	
	"""
	def upload( self, file_name: str, content_bytes: bytes, purpose: str ) -> Optional[
		Dict[ str, Any ] ]:
		"""
			
			Purpose:
			--------
			Upload raw bytes to the file endpoint.
	
			Parameters:
			-----------
			file_name : str
				Name for the uploaded file.
			content_bytes : bytes
				Raw file bytes.
			purpose : str
				Upload purpose (e.g., 'user_data').
	
			Returns:
			--------
			Optional[Dict[str, Any]]
		
		"""
		try:
			throw_if( 'file_name', file_name );
			throw_if( 'content_bytes', content_bytes );
			throw_if( 'purpose', purpose )
			url = f'{self.base_url}/files'
			files = {
					'file': (file_name, content_bytes) }
			data = {
					'purpose': purpose }
			resp = requests.post( url=url, headers={
					'x-api-key': self.api_key or '' }, files=files, data=data, timeout=self.timeout )
			resp.raise_for_status( )
			return resp.json( )
		except Exception as e:
			ex = Error( e );
			ex.module = 'claude';
			ex.cause = 'Files';
			ex.method = 'upload(self, file_name, content_bytes, purpose)'
			error = ErrorDialog( ex )
			error.show( )
	
	def list( self ) -> Optional[ Dict[ str, Any ] ]:
		"""
		
			Purpose:
			--------
			List uploaded files.
	
			Returns:
			--------
			Optional[Dict[str, Any]]
			
		"""
		try:
			return self._get( '/files' ).json( )
		except Exception as e:
			ex = Error( e );
			ex.module = 'claude';
			ex.cause = 'Files';
			ex.method = 'list(self)'
			error = ErrorDialog( ex )
			error.show( )
	
	def retrieve( self, file_id: str ) -> Optional[ Dict[ str, Any ] ]:
		"""
		
			Purpose:
			--------
			Retrieve file metadata by id.
	
			Parameters:
			-----------
			file_id : str
				File id.
	
			Returns:
			--------
			Optional[Dict[str, Any]]
		
		"""
		try:
			throw_if( 'file_id', file_id );
			return self._get( f'/files/{file_id}' ).json( )
		except Exception as e:
			ex = Error( e );
			ex.module = 'claude';
			ex.cause = 'Files';
			ex.method = 'retrieve(self, file_id)'
			error = ErrorDialog( ex )
			error.show( )
	
	def delete( self, file_id: str ) -> Optional[ Dict[ str, Any ] ]:
		"""
			
			Purpose:
			--------
			Delete an uploaded file.
	
			Parameters:
			-----------
			file_id : str
				File id.
	
			Returns:
			--------
			Optional[Dict[str, Any]]
		
		"""
		try:
			throw_if( 'file_id', file_id )
			return self._delete( f'/files/{file_id}' ).json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Files'
			ex.method = 'delete(self, file_id)'
			error = ErrorDialog( ex )
			error.show( )


class Transcription( Claude ):
	"""
		
		Purpose:
		--------
		Audio transcription helper for Claude-compatible endpoints.
		
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
			Transcription model ids.
	
			Returns:
			--------
			List[str]
		
		"""
		return [ 'whisper-1',
		         'claude-transcribe',
		         'claude-voice' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Output format options.
	
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
			Supported language codes.
	
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
	
	def transcribe( self, path: str, model: str = 'whisper-1', language: Optional[
		str ] = 'en', format: str = 'text' ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Transcribe an audio file and return the transcription text.
	
			Parameters:
			-----------
			path : str
				Local path to audio file.
			model : str
				Transcription model id.
			language : Optional[str]
				Language hint for transcription.
			format : str
				Output format.
	
			Returns:
			--------
			Optional[str]
		
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'model', model )
			with open( path, 'rb' ) as fh:
				files = {
						'file': (Path( path ).name, fh) }
				data = {
						'model': model,
						'language': language,
						'format': format }
				resp = requests.post( url=f'{self.base_url}/audio/transcriptions', headers={
						'x-api-key': self.api_key or '' }, files=files, data=data, timeout=self.timeout )
				resp.raise_for_status( )
				return resp.json( ).get( 'text' ) or resp.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Transcription'
			ex.method = 'transcribe(self, path)'
			error = ErrorDialog( ex )
			error.show( )

class Translation( Claude ):
	"""
		
		Purpose:
		--------
		Audio translation helper (speech-to-speech or speech-to-text).
	
	"""
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = None
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'whisper-1',
		         'claude-translate' ]
	
	@property
	def language_options( self ) -> List[ str ]:
		return [ 'en',
		         'es',
		         'fr',
		         'de',
		         'zh',
		         'it',
		         'ja' ]
	
	def translate( self, path: str, source_lang: Optional[ str ] = None, target_lang: Optional[
		str ] = 'en', model: str = 'whisper-1' ) -> Optional[ str ]:
		"""
			
			Purpose:
			--------
			Translate audio content to a target language.
	
			Parameters:
			-----------
			path : str
				Local audio path.
			source_lang : Optional[str]
				Source language hint.
			target_lang : Optional[str]
				Target language code (default 'en').
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
				files = {
						'file': (Path( path ).name, fh) }
				data = {
						'model': model,
						'source_language': source_lang,
						'target_language': target_lang }
				resp = requests.post( url=f'{self.base_url}/audio/translations', headers={
						'x-api-key': self.api_key or '' }, files=files, data=data, timeout=self.timeout )
				resp.raise_for_status( )
				return resp.json( ).get( 'text' ) or resp.json( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'claude'
			ex.cause = 'Translation'
			ex.method = 'translate(self, path)'
			error = ErrorDialog( ex )
			error.show( )

class TTS( Claude ):
	"""
		
		Purpose:
		--------
		Text-to-Speech (TTS) helper for Claude-compatible TTS endpoints.
	
	"""
	def __init__( self ) -> None:
		super( ).__init__( )
		self.model = None
		self.voice = None
		self.speed = None
		self.response_format = None
	
	@property
	def model_options( self ) -> List[ str ]:
		return [ 'claude-tts',
		         'claude-voice-1' ]
	
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
		return [ 'mp3',
		         'wav',
		         'aac',
		         'flac',
		         'opus' ]
	
	@property
	def speed_options( self ) -> List[ float ]:
		return [ 0.25,
		         0.5,
		         1.0,
		         1.5,
		         2.0 ]
	
	def create_audio( self, text: str, output_path: str, model: str='claude-tts',
			voice: str='alloy', speed: float=1.0, fmt: str='mp3' ) -> str:
		"""
		
			Purpose:
			--------
			Synthesize text to audio and save locally or return a URL.
	
			Parameters:
			-----------
			text : str
				Input text to synthesize.
			output_path : str
				Local file path to write audio when base64 is returned.
			model : str
				TTS model id.
			voice : str
				Voice selection.
			speed : float
				Speech rate multiplier.
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
			payload = {
					'model': model,
					'voice': voice,
					'speed': speed,
					'format': fmt,
					'input': text }
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
			ex.module = 'claude'
			ex.cause = 'TTS'
			ex.method = 'create_audio(self, text, output_path)'
			error = ErrorDialog( ex )
			error.show( )