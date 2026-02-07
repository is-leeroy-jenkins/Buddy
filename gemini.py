'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                gemini.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="gemini.py" company="Terry D. Eppler">

	     gemini.py
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
    gemini.py
  </summary>
  ******************************************************************************************
'''
import config as cfg
from boogr import ErrorDialog, Error
import os
import requests
import PIL.Image
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
from google import genai
from google.cloud import storage
from google.genai import types
from google.genai.types import (Part, GenerateContentConfig, ImageConfig, FunctionCallingConfig,
                                GenerateImagesConfig, GenerateVideosConfig, ThinkingConfig,
                                GeneratedImage, EmbedContentConfig, Content, ContentEmbedding,
                                Candidate, HttpOptions, GenerateImagesResponse, Field,
                                GenerateContentResponse, GenerateVideosResponse, Image, File)

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

class Gemini( ):
	'''

		Purpose:
		-------
		Base configuration and attribute store for Google Gemini AI functionality.

		Attributes:
		-----------
		number            : int - Default candidate count
		project_id        : str - Google Cloud Project ID
		api_key           : str - Google API Key
		cloud_location    : str - Google Cloud region
		instructions      : str - System instructions
		prompt            : str - User input prompt
		model             : str - Model identifier
		api_version       : str - API version
		max_tokens        : int - Token limit
		temperature       : float - Sampling temperature
		top_p             : float - Nucleus sampling
		top_k             : int - Top-k threshold
		content_config    : GenerateContentConfig - Content generation settings
		function_config   : FunctionCallingConfig - Tool use configuration
		thought_config    : ThinkingConfig - Reasoning settings
		genimg_config     : GenerateImagesConfig - Image generation settings
		image_config      : ImageConfig - Multimodal settings
		tool_config       : list - Collection of Tool objects for grounding
		candidate_count   : int - Response count
		modalities        : list - I/O types
		stops             : list - Stop sequences
		frequency_penalty : float - Repetition control
		presence_penalty  : float - Topic control
		response_format   : str - format string

	'''
	number: Optional[ int ]
	google_api_key: Optional[ str ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	max_output_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	content_config: Optional[ GenerateContentConfig ]
	function_config: Optional[ FunctionCallingConfig ]
	thought_config: Optional[ ThinkingConfig ]
	genimg_config: Optional[ GenerateImagesConfig ]
	image_config: Optional[ ImageConfig ]
	tool_config: Optional[ List[ types.Tool ] ]
	candidate_count: Optional[ int ]
	modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	response_format: Optional[ str ]
	
	def __init__( self ):
		self.api_key = cfg.GOOGLE_API_KEY
		self.model = None;
		self.content_config = None;
		self.image_config = None
		self.function_config = None;
		self.thought_config = None;
		self.genimg_config = None
		self.tool_config = None;
		self.api_version = None;
		self.temperature = 0.7
		self.top_p = 0.9;
		self.top_k = 40;
		self.candidate_count = 1
		self.frequency_penalty = 0.0;
		self.presence_penalty = 0.0;
		self.max_output_tokens = 2048
		self.instructions = None;
		self.prompt = None;
		self.response_format = None
		self.number = 1;
		self.modalities = None;
		self.stops = None

class Chat( Gemini ):
	'''

	    Purpose:
	    _______
	    Class handling text, vision, and tool-augmented analysis with the Google Gemini SDK.

	    Attributes:
	    -----------
	    use_vertex          : bool - Use Vertex AI (True) or API Key (False)
	    http_options        : HttpOptions - Networking and version settings
	    client              : Client - The initialized GenAI client
	    contents            : Union - Input prompt or message parts
	    content_response    : GenerateContentResponse - Result from text generation
	    image_response      : GenerateImagesResponse - Result from image generation
	    image_uri           : str - URI of processed image
	    audio_uri           : str - URI of processed audio
	    file_path           : str - Local path for document processing
	    response_modalities : list - Allowed output formats

	    Methods:
	    --------
	    generate_text( prompt, model )      : Generates text based on prompt
	    analyze_image( prompt, path, mod )  : Processes image content with text
	    summarize_document( prompt, path )  : Uploads and summarizes documents
	    web_search( prompt, model )         : Performs a search-grounded text generation
	    search_maps( prompt, model )        : Grounds responses using Google Search/Maps context

    '''
	use_vertex: Optional[ bool ]
	http_options: Optional[ HttpOptions ]
	client: Optional[ genai.Client ]
	storage_client: Optional[ storage.Client ]
	contents: Optional[ Union[ str, List[ str ] ] ]
	content_response: Optional[ GenerateContentResponse ]
	image_response: Optional[ GenerateImagesResponse ]
	image_uri: Optional[ str ]
	audio_uri: Optional[ str ]
	file_path: Optional[ str ]
	response_modalities: Optional[ List[ str ] ]
	files: Optional[ List[ str ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.number = None
		self.model = None
		self.api_version = None
		self.top_p = None
		self.temperature = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.candidate_count = None
		self.max_output_tokens = None
		self.use_vertex = None
		self.instructions = None
		self.contents = None
		self.http_options = None
		self.client = None
		self.storage_client = None
		self.response_modalities = [ 'TEXT', 'IMAGE' ]
		self.content_config = None;
		self.image_config = None;
		self.function_config = None
		self.thought_config = None;
		self.tool_config = None;
		self.content_response = None
		self.image_response = None;
		self.image_uri = None;
		self.audio_uri = None;
		self.file_path = None
		self.files = None
	
	@property
	def file_options( self ) -> List[ str ] | None:
		"""Returns list of available chat models."""
		self.files = self.get_files( )
		return self.files
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Returns list of available chat models."""
		return [ 'gemini-2.5-flash',
		         'gemini-2.5 flash image',
		         'gemini-2.5 flash-tts',
		         'gemini-2.5 flash-lite',
				 'gemini-2.0-flash',
		         'gemini-2.0-flash-lite',
		         'gemini-1.5-flash' ]
	
	@property
	def version_options( self ) -> List[ str ] | None:
		"""Returns list of available API versions."""
		return [ 'v1',
		         'v1alpha',
		         'v1beta1' ]
	
	@property
	def mime_options( self ):
		'''
			
			Returns:
			--------
			A List[ str ] of mime types
			
		'''
		return [ 'text/plain',
		         'application/json',
		         'text/x.enum' ]
	
	def generate_text( self, prompt: str, model: str='gemini-2.0-flash', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0,
			max_tokens: int=10000, stops: List[str]=None ) -> str | None:
		"""
		
			Purpose:
			-----------
			Generates a text completion based on the provided prompt and configuration.
			
			Parameters:
			-----------
			prompt: str - The text input for the model.
			model: str - The specific Gemini model identifier.
			Returns:
			--------
			Optional[ GenerateContentResponse ] - The response object or None on failure.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = prompt;
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				top_p=self.top_p, max_output_tokens=self.max_tokens,
				candidate_count=self.candidate_count, system_instruction=self.instructions,
				frequency_penalty=self.frequency_penalty, presence_penalty=self.presence_penalty )
			self.client = genai.Client( api_key=self.api_key )
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return self.content_response.text
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt, model ) -> GenerateContentResponse'
			error = ErrorDialog( exception )
			error.show( )
	
	def web_search( self, prompt: str, model: str='gemini-2.0-flash', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0,
			max_tokens: int=10000, stops: List[ str ]=None ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Generates a response grounded in Google Search results.
			
			Parameters:
			-----------
			prompt: str - The query for search-augmented generation.
			model: str - The Gemini model identifier.
			
			Returns:
			--------
			Optional[ str ] - The grounded text response.
		
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = prompt;
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.tool_config = [ types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config, system_instruction=self.instructions )
			self.client = genai.Client( api_key=self.api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'web_search( self, prompt, model ) -> Optional[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def search_maps( self, prompt: str, model: str='gemini-2.0-flash', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0,
			max_tokens: int=10000, stops: List[ str ]=None ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Uses Google Search grounding specifically for location and place-based queries.
			
			Parameters:
			-----------
			prompt: str - The location or directions query.
			model: str - The Gemini model identifier.
			Returns:
			--------
			Optional[ str ] - The grounded response containing place data.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			self.contents = f"Using Google Search and Maps data, answer: {prompt}"
			self.model = model
			self.contents = prompt;
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.tool_config = [ types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config  )
			self.client = genai.Client( api_key=self.api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'search_maps( self, prompt, model ) -> Optional[ str ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def analyze_image( self, prompt: str, filepath: str, model: str='gemini-2.0-flash',
			temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, stops: List[ str ]=None ) -> str | None:
		"""
			
			Purpose:
			--------
			Analyzes the content of a local image file using multimodal Gemini.
			
			Parameters:
			-----------
			prompt: str - Question or instruction for the analysis.
			filepath: str - Local filesystem path to the image.
			model: str - The multimodal Gemini model identifier.
			
			Returns:
			--------
			Optional[ str ] - The model's analysis text or None on failure.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepath', filepath )
			self.prompt = prompt
			self.file_path = filepath
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			img = PIL.Image.open( self.file_path )
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				top_p=self.top_p, max_output_tokens=self.max_tokens )
			self.client = genai.Client( api_key=self.api_key )
			response = self.client.models.generate_content( model=self.model,
				contents=[ img,  self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt, filepath, model ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def summarize_document( self, prompt: str, filepath: str, model: str='gemini-2.0-flash',
			temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, stops: List[ str ]=None ) -> str | None:
		"""
			
			Purpose:
			-------
			Uploads and summarizes a PDF or text document.
			
			Parameters:
			-----------
			prompt: str - Summarization instructions.
			filepath: str - Path to the document file.
			model: str - The model identifier for processing.
			Returns:
			--------
			Optional[ str ] - The document summary or None on failure.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepath', filepath )
			self.prompt = prompt
			self.file_path = filepath
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.api_key )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					doc_part = Part.from_bytes( data=f.read( ), mime_type="application/pdf" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ doc_part, self.prompt ], config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_path )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file, self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt, filepath, model ) -> str'
			error = ErrorDialog( exception )
			error.show( )

class Files( Gemini ):
	'''

		Purpose:
		--------
		Class encapsulating Gemini's FileStores API for uploading and managing remote assets.

		Attributes:
		-----------
		client       : Client - Initialized GenAI client
		file_id      : str - ID of the target file
		display_name : str - User-friendly label for the file
		mime_type    : str - Content type of the file
		file_path    : str - Local filesystem path
		file_list    : list - Collection of remote File objects
		response     : any - RAW API response object
		use_vertex   : bool - Integration flag

		Methods:
		--------
		upload( path, name )      : Uploads a local file to Gemini storage
		retrieve( file_id )       : Fetches metadata for a specific remote file
		list_files( )             : Lists all files currently in remote storage
		delete( file_id )         : Removes a file from remote storage

	'''
	api_version: Optional[ str ]
	google_api_key: Optional[ str ]
	storage_client: Optional[ storage.Client ]
	project_id: Optional[ str ]
	project_location: Optional[ str ]
	file_id: Optional[ str ]
	bucket_id: Optional[ str ]
	display_name: Optional[ str ]
	mime_type: Optional[ str ]
	file_path: Optional[ str ]
	file_list: Optional[ List[ File ] ]
	response: Optional[ Any ]
	use_vertex: Optional[ bool ]
	files: Optional[ List[ str ] ]
	
	def __init__( self, model: str='gemini-2.0-flash',
			temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, stops: List[ str ]=None ):
		super( ).__init__( )
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.project_location = cfg.GOOGLE_CLOUD_LOCATION
		self.file_path = filepath
		self.model = model
		self.top_p = top_p;
		self.temperature = temperature
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.stops = stops
		self.client = genai.Client( api_key=self.api_key )
		self.bucket_id = None
		self.file_id = None;
		self.display_name = None;
		self.mime_type = None
		self.file_path = None;
		self.file_list = [ ];
		self.response = None
		self.files = [ ]
	
	@property
	def file_options( self ) -> List[ str ] | None:
		"""Returns list of available chat models."""
		self.files = self.get_all( )
		return self.files
	
	def get_all( self, model: str = 'gemini-2.0-flash', temperature: float = 0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 10000, stops: List[ str ] = None ) -> List[ str ] | None:
		"""
			
			Purpose:
			-------
			Uploads and summarizes a PDF or text document.
			
			Parameters:
			-----------
			prompt: str - Summarization instructions.
			filepath: str - Path to the document file.
			model: str - The model identifier for processing.
			Returns:
			--------
			Optional[ str ] - The document summary or None on failure.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepath', filepath )
			self.prompt = prompt
			self.file_path = filepath
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.storage_client = storage.Client( api_key=self.google_api_key )
			bucket_name = "jeni-financial"
			prefix = "regulations/"
			bucket = self.storage_client.bucket( bucket_name )
			for blob in bucket.list_blobs( prefix=prefix ):
				url = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
				self.files.append( url )
			return self.files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'get_all( self  ) -> str'
			error = ErrorDialog( exception )
			error.show( )
			
	def upload( self, path: str, name: str = None ) -> File | None:
		"""
		
			Purpose:
			--------
			Uploads a file from a local path to Gemini's remote temporal storage.
			
			Parameters:
			-----------
			path: str - Local filesystem path to the file.
			name: str - Optional display name for the file.
			Returns:
			--------
			Optional[ File ] - Metadata object of the uploaded file.
			
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path;
			self.display_name = name
			self.response = self.client.files.upload( path=self.file_path,
				config={
						'display_name': self.display_name } )
			return self.response
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'FileStore'
			exception.method = 'upload( self, path: str, name: str ) -> Optional[ File ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def retrieve( self, file_id: str ) -> Optional[ File ]:
		"""
			
			Purpose:
			--------
			Retrieves the metadata and state of a previously uploaded file.
			
			Parameters:
			-----------
			file_id: str - The unique identifier of the remote file.
			
			Returns:
			--------
			Optional[ File ] - File metadata object.
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.response = self.client.files.get( name=self.file_id )
			return self.response
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'FileStore'
			exception.method = 'retrieve( self, file_id: str ) -> Optional[ File ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def list_files( self ) -> List[ File ] | None:
		"""
		
			Purpose:
			---------
			Returns a list of all files currently stored in the user's remote project.
			
			Returns:
			--------
			Optional[ List[ File ] ] - List of File metadata objects.
			
		"""
		try:
			self.file_list = list( self.client.files.list( ) )
			return self.file_list
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'FileStore'
			exception.method = 'list_files( self ) -> Optional[ List[ File ] ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete( self, file_id: str ) -> bool | None:
		"""
		
			Purpose:
			--------
			Deletes a specific file from remote storage to free up project quota.
			
			Parameters:
			-----------
			file_id: str - Unique identifier of the file to remove.
			
			Returns:
			--------
			bool - True if deletion was successful.
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client.files.delete( name=self.file_id )
			return True
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'FileStore'
			exception.method = 'delete( self, file_id: str ) -> bool'
			error = ErrorDialog( exception )
			error.show( )

class FileSearchStore( Gemini ):
	'''

		Purpose:
		--------
		Class encapsulating Gemini's FileStores API for uploading and managing remote assets.

		Attributes:
		-----------
		client       : Client - Initialized GenAI client
		file_id      : str - ID of the target file
		display_name : str - User-friendly label for the file
		mime_type    : str - Content type of the file
		file_path    : str - Local filesystem path
		file_list    : list - Collection of remote File objects
		response     : any - RAW API response object
		use_vertex   : bool - Integration flag

		Methods:
		--------
		upload( path, name )      : Uploads a local file to Gemini storage
		retrieve( file_id )       : Fetches metadata for a specific remote file
		list_files( )             : Lists all files currently in remote storage
		delete( file_id )         : Removes a file from remote storage

	'''
	gemini_api_key: Optional[ str ]
	client: Optional[ genai.Client ]
	file_id: Optional[ str ]
	display_name: Optional[ str ]
	mime_type: Optional[ str ]
	file_path: Optional[ str ]
	file_list: Optional[ List[ File ] ]
	response: Optional[ Any ]
	use_vertex: Optional[ bool ]
	
	def __init__( self, filepath: str, model: str='gemini-2.0-flash',
			temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, stops: List[ str ]=None ):
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.file_path = filepath
		self.model = model
		self.top_p = top_p;
		self.temperature = temperature
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.stops = stops
		self.client = genai.Client( api_key=self.gemini_api_key )
		self.file_id = None;
		self.display_name = None;
		self.mime_type = None
		self.file_path = None;
		self.file_list = [ ];
		self.response = None
	
	def upload( self, path: str, name: str = None ) -> File | None:
		"""
		
			Purpose:
			--------
			Uploads a file from a local path to Gemini's remote temporal storage.
			
			Parameters:
			-----------
			path: str - Local filesystem path to the file.
			name: str - Optional display name for the file.
			Returns:
			--------
			Optional[ File ] - Metadata object of the uploaded file.
			
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path;
			self.display_name = name
			self.response = self.client.files.upload( path=self.file_path,
				config={
						'display_name': self.display_name } )
			return self.response
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'FileStore'
			exception.method = 'upload( self, path: str, name: str ) -> Optional[ File ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def retrieve( self, file_id: str ) -> Optional[ File ]:
		"""
			
			Purpose:
			--------
			Retrieves the metadata and state of a previously uploaded file.
			
			Parameters:
			-----------
			file_id: str - The unique identifier of the remote file.
			
			Returns:
			--------
			Optional[ File ] - File metadata object.
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.response = self.client.files.get( name=self.file_id )
			return self.response
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'FileStore'
			exception.method = 'retrieve( self, file_id: str ) -> Optional[ File ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def list_files( self ) -> List[ File ] | None:
		"""
		
			Purpose:
			---------
			Returns a list of all files currently stored in the user's remote project.
			
			Returns:
			--------
			Optional[ List[ File ] ] - List of File metadata objects.
			
		"""
		try:
			self.file_list = list( self.client.files.list( ) )
			return self.file_list
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'FileStore'
			exception.method = 'list_files( self ) -> Optional[ List[ File ] ]'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete( self, file_id: str ) -> bool | None:
		"""
		
			Purpose:
			--------
			Deletes a specific file from remote storage to free up project quota.
			
			Parameters:
			-----------
			file_id: str - Unique identifier of the file to remove.
			
			Returns:
			--------
			bool - True if deletion was successful.
		
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client.files.delete( name=self.file_id )
			return True
		except Exception as e:
			exception = Error( e );
			exception.module = 'gemini'
			exception.cause = 'FileStore'
			exception.method = 'delete( self, file_id: str ) -> bool'
			error = ErrorDialog( exception )
			error.show( )

class Embeddings( Gemini ):
	'''

		Purpose:
		--------
		Class handling text embedding generation with the Google GenAI SDK.

		Attributes:
		-----------
		client              : Client - Initialized GenAI client
		response            : any - raw API response
		embedding           : list - Generated vector of floats
		encoding_format     : str - Format of the embedding response
		dimensions          : int - Size of the embedding vector
		use_vertex          : bool - Cloud integration flag
		task_type           : str - Type of task (RETRIEVAL, etc)
		http_options        : HttpOptions - Client networking settings
		embedding_config    : EmbedContentConfig - Configuration for embeddings
		contents            : list - Input strings
		input_text          : str - Current text being processed
		file_path           : str - Path to source text
		response_modalities : str - Modality configuration

		Methods:
		--------
		generate( text, model ) : Creates an embedding vector for input text

	'''
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	task_type: Optional[ str ]
	embedding_config: Optional[ types.EmbedContentConfig ]
	contents: Optional[ List[ str ] ]
	input_text: Optional[ str ]
	file_path: Optional[ str ]
	response_modalities: Optional[ str ]
	
	def __init__( self, model: str='text-embedding-004', temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int=10000 ):
		super( ).__init__( )
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.client = None
		self.embedding = None;
		self.response = None
		self.encoding_format = None
		self.input_text = None
		self.file_path = None
		self.dimensions = None
		self.task_type = None
		self.response_modalities = None
		self.embedding_config = None;
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Returns list of embedding models."""
		return [ 'text-embedding-004',
		         'text-multilingual-embedding-002' ]
	
	def generate( self, text: str, model: str='text-embedding-004', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0,
			max_tokens: int=10000 ) -> List[ float ] | None:
		"""
			
			Purpose:
			---------
			Generates a vector representation of the provided text.
			
			Parameters:
			-----------
			text: str - Input text string.
			model: str - Embedding model identifier.
			
			Returns:
			--------
			Optional[ List[ float ] ] - List of embedding values or None on failure.
		
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text;
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.embedding_config = EmbedContentConfig( task_type=self.task_type )
			self.response = self.client.models.embed_content( model=self.model,
				contents=self.input_text, config=self.embedding_config )
			self.embedding = self.response.embeddings[ 0 ].values
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embedding'
			exception.method = 'generate( self, text, model ) -> List[ float ]'
			error = ErrorDialog( exception )
			error.show( )

class TTS( Gemini ):
	"""

	    Purpose
	    ___________
	    Class for conversion of text to speech using Gemini multimodal output.

	    Attributes:
	    -----------
	    speed           : float - Audio playback speed
	    voice           : str - Persona identifier
	    response        : GenerateContentResponse - Raw response
	    client          : Client - genai instance
	    audio_path      : str - Target path
	    response_format : str - Audio format
	    input_text      : str - Original text
	    use_vertex      : bool - Integration flag

	    Methods:
	    --------
	    create_audio( text, path, format, speed, voice ) : Saves multimodal audio to file

    """
	speed: Optional[ float ]
	voice: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	client: Optional[ genai.Client ]
	audio_path: Optional[ str ]
	response_format: Optional[ str ]
	input_text: Optional[ str ]
	use_vertex: Optional[ bool ]
	
	def __init__( self, n: int=1, model: str='gemini-2.0-flash',
			temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, instruct: str=None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.api_version = version
		self.use_vertex = use_ai
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.http_options = HttpOptions( api_version=self.api_version )
		self.client = genai.Client( vertexai=self.use_vertex, api_key=self.api_key,
			http_options=self.http_options )
		self.voice = 'Puck'
		self.speed = 1.0
		self.response_format = 'MP3'
		self.audio_path = None
		self.input_text = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Returns list of models supporting audio output."""
		return [ 'gemini-2.0-flash',
		         'gemini-1.5-flash' ]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		"""Returns list of available voice personas."""
		return [ 'Achernar',
		         'Aoede',
		         'Erinome',
		         'Kore',
		         'Orus',
		         'Puck' ]
	
	def create_audio( self, text: str, filepath: str, format: str='MP3',
			speed: float=1.0, voice: str='Puck' ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Converts text to speech and writes the data to a local file.
			
			Parameters:
			-----------
			text: str - Input text string.
			filepath: str - Target local path.
			format: str - File format.
			speed: float - Playback rate.
			voice: str - Persona name.
			
			Returns:
			--------
			Optional[ str ] - Local path to the created file or None.
		
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'filepath', filepath )
			self.input_text = text
			self.audio_path = filepath
			self.response_format = format
			self.speed = speed
			self.voice = voice
			prompt = f"Read the following aloud with a {self.voice} persona: {self.input_text}"
			self.content_config = GenerateContentConfig( response_modalities=[ 'AUDIO' ],
				temperature=self.temperature )
			self.response = self.client.models.generate_content( model=self.model,
				contents=prompt, config=self.content_config )
			for part in self.response.candidates[ 0 ].content.parts:
				if part.inline_data:
					with open( self.audio_path, 'wb' ) as f:
						f.write( part.inline_data.data )
					return self.audio_path
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'create_audio( self, text, filepath, format, speed, voice ) -> str'
			error = ErrorDialog( exception )
			error.show( )

class Transcription( Gemini ):
	"""

	    Purpose
	    ___________
	    Class handling audio-to-text transcription using Gemini audio processing.

	    Attributes:
	    -----------
	    client     : Client - GenAI instance
	    transcript : str - Text result
	    file_path  : str - Path to audio file
	    use_vertex : bool - Integration flag

	    Methods:
	    --------
	    transcribe( path, model ) : Transcribes local audio file to text

    """
	client: Optional[ genai.Client ]
	transcript: Optional[ str ]
	file_path: Optional[ str ]
	use_vertex: Optional[ bool ]
	
	def __init__( self, n: int=1, model: str='gemini-2.0-flash', version: str='v1alpha',
			use_ai: bool=False, temperature: float=0.8, top_p: float=0.9, frequency: float=0.0,
			presence: float=0.0, max_tokens: int=10000, instruct: str=None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.api_version = version
		self.use_vertex = use_ai
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( vertexai=self.use_vertex, api_key=self.api_key,
			http_options=HttpOptions( api_version=self.api_version ) )
		self.transcript = None
		self.file_path = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Returns list of models supporting audio input."""
		return [ 'gemini-2.0-flash',
		         'gemini-1.5-flash' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Returns list of available target languages."""
		return [ 'English',
		         'Spanish',
		         'French',
		         'Japanese',
		         'German',
		         'Chinese' ]
	
	def transcribe( self, path: str, model: str='gemini-2.0-flash' ) -> Optional[ str ]:
		"""
			
			Purpose:
			---------
			Transcribes an audio file into text using multimodal GenAI.
			
			Parameters:
			-----------
			path: str - Local path to the source audio.
			model: str - Specific GenAI model ID.
			Returns:
			--------
			Optional[ str ] - Verbatim text transcript.
		
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.model = model
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					audio_part = Part.from_bytes( data=f.read( ), mime_type="audio/mpeg" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ audio_part,"Provide a verbatim transcription." ],
					config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_path )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file, "Provide a verbatim transcription." ],
					config=self.content_config )
			self.transcript = response.text
			return self.transcript
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path, model ) -> str'
			error = ErrorDialog( ex )
			error.show( )

class Translation( Gemini ):
	"""

	    Purpose
	    ___________
	    Class for translating text between languages using Gemini LLM.

	    Attributes:
	    -----------
	    client          : Client - genai client instance
	    target_language : str - Destination language
	    source_language : str - Source language
	    use_vertex      : bool - Cloud integration flag

	    Methods:
	    --------
	    translate( text, target, source ) : Translates text strings

    """
	client: Optional[ genai.Client ]
	target_language: Optional[ str ]
	source_language: Optional[ str ]
	use_vertex: Optional[ bool ]
	
	def __init__( self, n: int=1, model: str = 'gemini-2.0-flash', version: str='v1alpha',
			use_ai: bool=False, temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			instruct: str=None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.api_version = version
		self.use_vertex = use_ai
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( vertexai=self.use_vertex, api_key=self.api_key,
			http_options=HttpOptions( api_version=self.api_version ) )
		self.target_language = None
		self.source_language = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Returns list of translation-capable models."""
		return [ 'gemini-2.0-flash',
		         'gemini-1.5-pro' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Returns list of available target languages."""
		return [ 'English',
		         'Spanish',
		         'French',
		         'Japanese',
		         'German',
		         'Chinese' ]
	
	def translate( self, text: str, target: str, source: str='Auto' ) -> Optional[ str ]:
		"""
			
			Purpose:
			-------
			Translates text from one language to another.
			
			Parameters:
			-----------
			text: str - Text to translate.
			target: str - Target language.
			source: str - Source language.
			
			Returns:
			--------
			Optional[ str ] - Translated text.
		
		"""
		try:
			throw_if( 'text', text )
			self.target_language = target
			self.source_language = source
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			prompt = f"Translate the following from {self.source_language} to {self.target_language}: {text}"
			response = self.client.models.generate_content( model=self.model,
				contents=prompt, config=self.content_config )
			return response.text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = 'translate( self, text, target, source ) -> str'
			error = ErrorDialog( exception )
			error.show( )

class Images( Gemini ):
	"""

	    Purpose
	    ___________
	    Class for generating images from text using Google Imagen models.

	    Attributes:
	    -----------
	    client       : Client - GenAI instance
	    aspect_ratio : str - W:H ratio
	    use_vertex   : bool - Integration flag

	    Methods:
	    --------
	    generate( prompt, aspect ) : Generates Imagen asset

    """
	client: Optional[ genai.Client ]
	aspect_ratio: Optional[ str ]
	use_vertex: Optional[ bool ]
	
	def __init__( self,  n: int=1, model: str='gemini-2.0-flash', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0, ratio: str= '4:3',
			max_tokens: int=10000, instruct: str=None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.aspect_ratio = ratio
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=self.api_key )
		self.aspect_ratio = '4:3'
		self.genimg_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Returns list of image generation models."""
		return [ 'gemini-2.5-flash-image',
		         'gemini-3-pro-image-preview' ]
	
	@property
	def aspect_options( self ) -> List[ str ] | None:
		"""Returns list of allowed aspect ratios."""
		return [ '1:1',
		         '3:4',
		         '4:3',
		         '9:16',
		         '16:9' ]
	
	def generate( self, prompt: str, aspect: str='4:3', n: int=1,
			model: str='gemini-2.5-flash-image', temperature: float=0.8, top_p: float=0.9,
			frequency: float=0.0, presence: float=0.0, max_tokens: int=10000,
			instruct: str=None ) -> Optional[ Image ]:
		"""
			
			Purpose:
			-----------
			Generates a new image based on a descriptive text prompt.
			
			Parameters:
			-----------
			prompt: str - Image description.
			aspect: str - Aspect ratio.
			
			Returns:
			--------
			Optional[ Image ] - The Image data object.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			self.number = n
			self.aspect_ratio = aspect
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.instructions = instruct
			self.genimg_config = GenerateImagesConfig( aspect_ratio=self.aspect_ratio,
				number_of_images=self.number )
			response = self.client.models.generate_images( model=self.model,
				prompt=self.prompt, config=self.genimg_config )
			return response.generated_images[ 0 ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'generate( self, prompt, aspect ) -> Image'
			error = ErrorDialog( exception )
			error.show( )