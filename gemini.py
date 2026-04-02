'''
  ******************************************************************************************
      Assembly:                Jeni
      Filename:                gemini.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="gemini.py" company="Terry D. Eppler">

	     gemini.py
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
    gemini.py
  </summary>
  ******************************************************************************************
'''
from google.genai.file_search_stores import FileSearchStores
import config as cfg
import base64
from boogr import ErrorDialog, Error
import json
import os
import requests
import PIL.Image
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
from google import genai
from google.cloud import storage
from google.genai import types
from google.genai.pagers import Pager
from google.genai.types import (Part, GenerateContentConfig, ImageConfig, FunctionCallingConfig,
                                GenerateImagesConfig, GenerateVideosConfig, ThinkingConfig,
                                GeneratedImage, EmbedContentConfig, Content, ContentEmbedding,
                                Candidate, HttpOptions, GenerateImagesResponse, Field,
                                FileSearchStore,
                                GenerateContentResponse, GenerateVideosResponse, Image, File,
                                SpeakerVoiceConfig, VoiceConfig, SpeechConfig, Tool, ToolConfig,
                                GoogleSearch, UrlContext, SafetySetting, HarmCategory,
                                HarmBlockThreshold)

def throw_if( name: str, value: object ):
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""
		
		Purpose:
		---------
		Encodes a local image to a base64 string for vision API requests.
		
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

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
		response_modalities        : list - I/O types
		stops             : list - Stop sequences
		frequency_penalty : float - Repetition control
		presence_penalty  : float - Topic control
		response_format   : str - format string

	'''
	number: Optional[ int ]
	google_api_key: Optional[ str ]
	gemini_api_key: Optional[ str ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	media_resolution: Optional[ str ]
	response_modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	domains: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	response_format: Optional[ str ]
	content_response: Optional[ GenerateContentResponse ]
	image_response: Optional[ GenerateImagesResponse ]
	content_config: Optional[ GenerateContentConfig ]
	function_config: Optional[ FunctionCallingConfig ]
	thought_config: Optional[ ThinkingConfig ]
	genimg_config: Optional[ GenerateImagesConfig ]
	image_config: Optional[ ImageConfig ]
	tool_config: Optional[ List[ types.Tool ] ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ str ] ]
	
	def __init__( self ):
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.model = None
		self.api_version = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.candidate_count = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.prompt = None
		self.response_format = None
		self.number = None
		self.response_modalities = [ ]
		self.stops = [ ]
		self.tools = [ ]

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
	contents: Optional[ Union[ str, List[ str ], List[ Content ] ] ]
	image_uri: Optional[ str ]
	audio_uri: Optional[ str ]
	file_path: Optional[ str ]
	files: Optional[ List[ str ] ]
	content_block: Optional[ str ]
	context: Optional[ List[ Dict[ str, Any ] ] ]
	urls: Optional[ List[ str ] ]
	max_urls: Optional[ int ]
	response_schema: Optional[ Any ]
	safety_profile: Optional[ str ]
	safety_settings: Optional[ List[ SafetySetting ] ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite' ):
		super( ).__init__( )
		self.api_version = None
		self.client = None
		self.content_config = None
		self.image_config = None
		self.function_tool_config = None
		self.thought_config = None
		self.genimg_config = None
		self.tool_objects = None
		self.tools = [ ]
		self.response_modalities = [ ]
		self.files = [ ]
		self.http_options = { }
		self.number = None
		self.candidate_count = None
		self.model = model
		self.top_p = None
		self.top_k = None
		self.temperature = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.use_vertex = None
		self.instructions = None
		self.media_resolution = None
		self.tool_choice = None
		self.contents = None
		self.content_block = None
		self.context = [ ]
		self.client = None
		self.storage_client = None
		self.content_response = None
		self.image_response = None
		self.image_uri = None
		self.audio_uri = None
		self.file_path = None
		self.stops = [ ]
		self.response_mime_type = None
		self.response_schema = None
		self.urls = [ ]
		self.max_urls = None
		self.safety_profile = None
		self.safety_settings = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Returns list of available chat models.
			
		"""
		return [ 'gemini-2.5-flash',
		         'gemini-2.5-flash-lite',
		         'gemini-2.5-pro',
		         'gemini-3-flash-preview',
		         'gemini-3.1-flash-lite-preview',
		         'gemini-3.1-pro-preview',
		         'gemini-2.0-flash',
		         'gemini-2.0-flash-lite' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'google_search',
		         'google_maps',
		         'url_context',
		         'code_execution' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of thinking effort options

		'''
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL',
		         'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def media_options( self ):
		'''
		
		Purpose:
		--------
		Returns a List[ str ] of media resolution options.
		
		'''
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'AUTO',
		         'ANY',
		         'NONE',
		         'VALIDATED' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of the includeable options

		'''
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available modality options

		'''
		return [ 'MODALITY_UNSPECIFIED', 'TEXT', 'IMAGE', 'AUDIO' ]
	
	@property
	def format_options( self ):
		'''
			
			Returns:
			--------
			A List[ str ] of mime types
			
		'''
		return [ 'text/plain',
		         'application/json',
		         'text/x.enum' ]
	
	@property
	def safety_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of safety profile options

		'''
		return [ '',
		         'strict',
		         'balanced',
		         'permissive' ]
	
	def get_supported_tool_options( self, model: str = None ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Returns the subset of built-in Gemini tools supported by the selected model.
			
			Parameters:
			-----------
			model: str - Optional Gemini model identifier.
			
			Returns:
			--------
			List[ str ] - Supported tool names.
		
		"""
		try:
			self.model_name = str( model or self.model or '' ).strip( ).lower( )
			self.options = [ 'google_search', 'url_context', 'code_execution' ]
			
			if self._supports_google_maps( self.model_name ):
				self.options.append( 'google_maps' )
			
			return self.options
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_supported_tool_options( self, model: str=None )'
			raise exception
	
	def _resolve_api_key( self ) -> str | None:
		"""
		
			Purpose:
			--------
			Resolves the Gemini developer API key at call-time.
			
			Returns:
			--------
			Optional[ str ] - Resolved API key.
		
		"""
		try:
			self.api_key = os.environ.get( 'GEMINI_API_KEY' )
			if self.api_key and str( self.api_key ).strip( ):
				return str( self.api_key ).strip( )
			
			self.api_key = os.environ.get( 'GOOGLE_API_KEY' )
			if self.api_key and str( self.api_key ).strip( ):
				return str( self.api_key ).strip( )
			
			self.api_key = getattr( cfg, 'GEMINI_API_KEY', None )
			if self.api_key and str( self.api_key ).strip( ):
				return str( self.api_key ).strip( )
			
			self.api_key = getattr( cfg, 'GOOGLE_API_KEY', None )
			if self.api_key and str( self.api_key ).strip( ):
				return str( self.api_key ).strip( )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_resolve_api_key( self ) -> str | None'
			raise exception
	
	def _supports_google_maps( self, model: str = None ) -> bool:
		"""
		
			Purpose:
			--------
			Determines whether the selected model should expose Google Maps grounding.
			
			Parameters:
			-----------
			model: str - Gemini model identifier.
			
			Returns:
			--------
			bool
			
		"""
		try:
			self.model_name = str( model or self.model or '' ).strip( ).lower( )
			self.maps_models = {
					'gemini-3.1-pro-preview',
					'gemini-3.1-flash-lite-preview',
					'gemini-3-flash-preview',
					'gemini-2.5-pro',
					'gemini-2.5-flash',
					'gemini-2.5-flash-lite',
					'gemini-2.0-flash'
			}
			return self.model_name in self.maps_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_supports_google_maps( self, model: str=None ) -> bool'
			raise exception
	
	def _supports_computer_use( self, model: str = None ) -> bool:
		"""
		
			Purpose:
			--------
			Determines whether the selected model should expose Computer Use.
			
			Parameters:
			-----------
			model: str - Gemini model identifier.
			
			Returns:
			--------
			bool
			
		"""
		try:
			return False
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_supports_computer_use( self, model: str=None ) -> bool'
			raise exception
	
	def _normalize_positive_int( self, value: Any = None ) -> int | None:
		"""
		
			Purpose:
			--------
			Normalizes positive integer request parameters. Zero and invalid values are omitted.
			
			Parameters:
			-----------
			value: Any - Candidate integer value.
			
			Returns:
			--------
			Optional[ int ]
		
		"""
		try:
			if value is None:
				return None
			
			self.int_value = int( value )
			return self.int_value if self.int_value > 0 else None
		except Exception:
			return None
	
	def _build_tools( self, tools: List[ str ] = None ) -> List[ Tool ] | None:
		"""
		
			Purpose:
			--------
			Builds Gemini built-in tool objects from
			selected tool names.
			
			Parameters:
			-----------
			tools: List[ str ] - Tool names selected in the UI.
			
			Returns:
			--------
			Optional[ List[ Tool ] ] - Tool collection or None.
		
		"""
		try:
			self.tools = tools if tools is not None else [ ]
			self.tool_objects = [ ]
			self.supported_tools = set( self.get_supported_tool_options( self.model ) )
			
			for name in self.tools:
				if not isinstance( name, str ):
					continue
				
				self.name = str( name ).strip( )
				if not self.name or self.name not in self.supported_tools:
					continue
				
				if self.name == 'google_search':
					self.tool_objects.append( Tool( google_search=GoogleSearch( ) ) )
				
				elif self.name == 'url_context':
					self.tool_objects.append( Tool( url_context=UrlContext( ) ) )
				
				elif self.name == 'code_execution':
					self.tool_objects.append( Tool( code_execution=types.ToolCodeExecution ) )
				
				elif self.name == 'google_maps':
					self.tool_objects.append( Tool( google_maps=types.GoogleMaps( ) ) )
			
			return self.tool_objects if len( self.tool_objects ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_build_tools( self, tools: List[ str ]=None )'
			raise exception
	
	def _parse_response_schema( self, response_schema: Any = None ) -> Any:
		"""
		
			Purpose:
			--------
			Normalizes a structured-output schema passed
			as a dict, JSON string, or schema class.
			
			Parameters:
			-----------
			response_schema: Any - UI schema value.
			
			Returns:
			--------
			Any - Parsed schema object or None.
		
		"""
		try:
			if response_schema is None:
				return None
			
			if isinstance( response_schema, dict ):
				return response_schema if len( response_schema ) > 0 else None
			
			if isinstance( response_schema, str ):
				self.schema_text = response_schema.strip( )
				if not self.schema_text:
					return None
				
				return json.loads( self.schema_text )
			
			return response_schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_parse_response_schema( self, response_schema: Any=None )'
			raise exception
	
	def _build_contents( self, prompt: str, context: List[ Any ] = None,
			content: str = None ) -> str | List[ Content ]:
		"""
		
			Purpose:
			--------
			Builds Gemini contents from the current
			prompt and any prior conversational context.
			
			Parameters:
			-----------
			prompt: str - Current user prompt.
			context: List[ Any ] - Prior chat messages or Gemini Content objects.
			content: str - Optional prepended content block.
			
			Returns:
			--------
			Union[ str, List[ Content ] ] - Contents payload for Gemini.
		
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.context = context if context is not None else [ ]
			self.content_block = content
			self.contents = [ ]
			for item in self.context:
				if item is None:
					continue
				
				if isinstance( item, Content ):
					self.contents.append( item )
					continue
				
				if not isinstance( item, dict ):
					continue
				
				role = item.get( 'role', 'user' )
				text = item.get( 'content', None )
				if text is None:
					continue
				
				text = str( text ).strip( )
				if not text:
					continue
				
				if role == 'assistant':
					self.contents.append(
						Content(
							role='model',
							parts=[ Part.from_text( text=text ) ]
						)
					)
				else:
					self.contents.append(
						Content(
							role='user',
							parts=[ Part.from_text( text=text ) ]
						)
					)
			
			self.user_text = str( self.prompt ).strip( )
			if self.content_block is not None and str( self.content_block ).strip( ):
				self.user_text = f"{str( self.content_block ).strip( )}\n\n{self.user_text}"
			
			self.contents.append(
				Content(
					role='user',
					parts=[ Part.from_text( text=self.user_text ) ]
				)
			)
			
			return self.contents if len( self.contents ) > 0 else self.prompt
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('_build_contents( self, prompt: str, '
			                    'context: List[ Any ]=None, '
			                    'content: str=None )')
			raise exception
	
	def _get_response_content( self ) -> Content | None:
		"""
		
			Purpose:
			--------
			Extracts the structured Gemini model content
			from the most recent response.
			
			Returns:
			--------
			Optional[ Content ] - The model content block.
		
		"""
		try:
			if self.content_response is None:
				return None
			
			if hasattr( self.content_response, 'candidates' ) and self.content_response.candidates:
				for candidate in self.content_response.candidates:
					if hasattr( candidate, 'content' ) and candidate.content is not None:
						return candidate.content
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_get_response_content( self ) -> Content | None'
			raise exception
	
	def get_structured_history( self ) -> List[ Content ] | None:
		"""
		
			Purpose:
			--------
			Builds the full structured conversation history
			for reuse in a subsequent Gemini request.
			
			Returns:
			--------
			Optional[ List[ Content ] ] - Conversation history with model output.
		
		"""
		try:
			self.history = [ ]
			
			if self.contents is not None and isinstance( self.contents, list ):
				for item in self.contents:
					if isinstance( item, Content ):
						self.history.append( item )
			
			self.response_content = self._get_response_content( )
			if self.response_content is not None:
				self.history.append( self.response_content )
			
			return self.history if len( self.history ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_structured_history( self ) -> List[ Content ] | None'
			raise exception
	
	def _build_config( self, model: str = 'gemini-2.5-flash-lite', number: int = None,
			temperature: float = None, top_p: float = None, top_k: int = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None, response_format: str = None,
			tools: List[ str ] = None, tool_choice: str = None, reasoning: str = None,
			modalities: List[ str ] = None, media_resolution: str = None,
			response_schema: Any = None, safety_profile: str = None ) -> GenerateContentConfig:
		"""
		
			Purpose:
			--------
			Builds the GenerateContentConfig object used
			for Gemini text generation.
			
			Parameters:
			-----------
			model: str - Gemini model identifier.
			
			Returns:
			--------
			GenerateContentConfig - Configured content settings.
		
		"""
		try:
			self.model = str( model or self.model or 'gemini-2.5-flash-lite' ).strip( )
			self.number = self._normalize_positive_int( number )
			self.candidate_count = self.number
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = self._normalize_positive_int( top_k )
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = self._normalize_positive_int( max_tokens )
			self.stops = stops if stops is not None else [ ]
			self.instructions = instruct
			self.response_mime_type = response_format
			self.response_schema = self._parse_response_schema( response_schema=response_schema )
			self.safety_settings = self._build_safety_settings( safety_profile=safety_profile )
			self.tool_choice = tool_choice
			self.media_resolution = str( media_resolution ).strip( ) if media_resolution else None
			self.tool_objects = self._build_tools( tools=tools )
			self.function_tool_config = self._build_tool_config(
				tool_choice=self.tool_choice,
				tools=self.tool_objects )
			self.response_modalities = self._build_modalities( modalities=modalities )
			self.thought_config = self._build_reasoning( reasoning=reasoning )
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.top_k is not None:
				self.config_kwargs[ 'top_k' ] = self.top_k
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.candidate_count is not None:
				self.config_kwargs[ 'candidate_count' ] = self.candidate_count
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.frequency_penalty is not None:
				self.config_kwargs[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None:
				self.config_kwargs[ 'presence_penalty' ] = self.presence_penalty
			
			if self.stops is not None and len( self.stops ) > 0:
				self.config_kwargs[ 'stop_sequences' ] = self.stops
			
			if self.response_mime_type is not None and str( self.response_mime_type ).strip( ):
				self.config_kwargs[ 'response_mime_type' ] = str( self.response_mime_type ).strip( )
			
			if self.response_schema is not None:
				if isinstance( self.response_schema, dict ):
					self.config_kwargs[ 'response_json_schema' ] = self.response_schema
				else:
					self.config_kwargs[ 'response_schema' ] = self.response_schema
			
			if self.media_resolution is not None and self.media_resolution:
				self.config_kwargs[ 'media_resolution' ] = self.media_resolution
			
			if self.tool_objects is not None and len( self.tool_objects ) > 0:
				self.config_kwargs[ 'tools' ] = self.tool_objects
			
			if self.function_tool_config is not None:
				self.config_kwargs[ 'tool_config' ] = self.function_tool_config
			
			if self.safety_settings is not None and len( self.safety_settings ) > 0:
				self.config_kwargs[ 'safety_settings' ] = self.safety_settings
			
			if self.response_modalities is not None and len( self.response_modalities ) > 0:
				self.config_kwargs[ 'response_modalities' ] = self.response_modalities
			
			if self.thought_config is not None:
				self.config_kwargs[ 'thinking_config' ] = self.thought_config
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = '_build_config( self, model ) -> GenerateContentConfig'
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			number: int = None, temperature: float = None, top_p: float = None, top_k: int = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None, response_format: str = None,
			tools: List[ str ] = None, tool_choice: str = None, reasoning: str = None,
			modalities: List[ str ] = None, media_resolution: str = None,
			context: List[ Dict[ str, Any ] ] = None, content: str = None,
			urls: List[ str ] = None, max_urls: int = None, response_schema: Any = None,
			safety_profile: str = None, stream: bool = False,
			stream_handler: Any = None ) -> str | None:
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
			Optional[ str ] - The text response or None on failure.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.model = str( model or self.model or 'gemini-2.5-flash-lite' ).strip( )
			self.stream = bool( stream )
			self.urls = self._build_urls( urls=urls, max_urls=max_urls )
			self.content_block = self._append_urls_to_content( content=content, urls=self.urls )
			self.contents = self._build_contents(
				prompt=prompt,
				context=context,
				content=self.content_block )
			self.content_config = self._build_config(
				model=self.model,
				number=number,
				temperature=temperature,
				top_p=top_p,
				top_k=top_k,
				frequency=frequency,
				presence=presence,
				max_tokens=max_tokens,
				stops=stops,
				instruct=instruct,
				response_format=response_format,
				tools=tools,
				tool_choice=tool_choice,
				reasoning=reasoning,
				modalities=modalities,
				media_resolution=media_resolution,
				response_schema=response_schema,
				safety_profile=safety_profile )
			
			self.api_key = self._resolve_api_key( )
			throw_if( 'api_key', self.api_key )
			self.client = genai.Client( api_key=self.api_key )
			
			if self.stream:
				self.stream_response = self.client.models.generate_content_stream(
					model=self.model,
					contents=self.contents,
					config=self.content_config )
				
				if stream_handler is not None:
					self.text_blocks = [ ]
					for chunk in self.stream_response:
						if chunk is None:
							continue
						
						self.chunk_text = getattr( chunk, 'text', None )
						if self.chunk_text is None or not str( self.chunk_text ):
							continue
						
						self.text_blocks.append( str( self.chunk_text ) )
						stream_handler( str( self.chunk_text ) )
					
					self.output_text = ''.join( self.text_blocks ).strip( )
					return self.output_text if self.output_text else None
				
				return self.get_stream_output_text( stream_response=self.stream_response )
			
			self.content_response = self.client.models.generate_content(
				model=self.model,
				contents=self.contents,
				config=self.content_config )
			
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt, model ) -> Optional[ str ]'
			raise exception

class Images( Gemini ):
	"""

	    Purpose
	    ___________
	    Class for generating, analyzing, and editing images with the Google Gemini SDK.

	    Attributes:
	    -----------
	    client       : Client - GenAI instance
	    aspect_ratio : str - W:H ratio
	    use_vertex   : bool - Integration flag

	    Methods:
	    --------
	    generate( prompt, aspect )        : Generates an image from text
	    analyze( prompt, path, model )    : Analyzes an image using text + image input
	    edit( prompt, path, model )       : Edits an image using text + image input

    """
	client: Optional[ genai.Client ]
	aspect_ratio: Optional[ str ]
	use_vertex: Optional[ bool ]
	resolution: Optional[ str ]
	size: Optional[ str ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-image' ):
		super( ).__init__( )
		self.number = None
		self.model = model
		self.client = None
		self.instructions = None
		self.content_config = None
		self.image_config = None
		self.function_config = None
		self.thought_config = None
		self.genimg_config = None
		self.tool_config = None
		self.response_modalities = [ ]
		self.tools = [ ]
		self.stops = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.temperature = None
		self.size = None
		self.top_p = None
		self.top_k = None
		self.aspect_ratio = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.candidate_count = None
		self.max_output_tokens = None
		self.use_vertex = None
		self.media_resolution = None
		self.tool_choice = None
		self.content_response = None
		self.response = None
		self.grounding_metadata = None
		self.output_mime_type = None
		self.response_mode = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Returns list of image generation models.
			
		"""
		return [ 'gemini-2.5-flash-image',
		         'gemini-3.1-flash-image-preview' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of the includeable options

		'''
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def aspect_options( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Returns list of allowed aspect ratios.
			
		"""
		return [ '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9' ]
	
	@property
	def media_options( self ):
		'''
		
		Purpose:
		--------
		Returns a List[ str ] of media resolution options.
		
		'''
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available modality options

		'''
		return [ 'TEXT', 'IMAGE', 'TEXT_AND_IMAGE' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of thinking effort options

		'''
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL',
		         'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def size_options( self ):
		'''
			
			Purpose:
			---------
			Returns list of image sizes
			
		'''
		return [ '1K', '2K', '4K' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'google_search', 'image_search' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'AUTO', 'ANY', 'NONE', 'VALIDATED' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		'''
			
			Returns:
			--------
			A List[ str ] of mime types
			
		'''
		return [ 'text/plain',
		         'application/json',
		         'text/x.enum' ]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		'''
			
			Returns:
			--------
			A List[ str ] of mime types
			
		'''
		return [ 'image/jpeg',
		         'image/png',
		         'image/webp' ]
	
	@property
	def resolution_options( self ) -> List[ str ] | None:
		'''
			
			Purpose:
			-------
			Returns a list of resolution options
			
		'''
		return [ '1K', '2K', '4K' ]
	
	def _supports_image_size( self, model: str = None ) -> bool:
		try:
			self.model_name = str( model or self.model ).strip( )
			return self.model_name in [ 'gemini-3.1-flash-image-preview' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_supports_image_size( self, model=None ) -> bool'
			raise exception
	
	def _supports_search_grounding( self, model: str = None ) -> bool:
		try:
			self.model_name = str( model or self.model ).strip( )
			return self.model_name in [ 'gemini-3.1-flash-image-preview' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_supports_search_grounding( self, model=None ) -> bool'
			raise exception
	
	def _supports_image_search( self, model: str = None ) -> bool:
		"""
			
			Purpose:
			-----------
			Determines whether the selected image model supports Google Image Search grounding.
			
			Parameters:
			-----------
			model: str - The Gemini image model identifier.
			
			Returns:
			--------
			bool - True when Google Image Search grounding is supported; otherwise False.
			
		"""
		try:
			self.model_name = str( model or self.model ).strip( )
			return self.model_name == 'gemini-3.1-flash-image-preview'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_supports_image_search( self, model=None ) -> bool'
			raise exception
	
	def _normalize_response_modalities( self, response_modalities: Optional[ str ],
			image_only: bool = False ) -> List[ str ]:
		"""
			
			Purpose:
			-----------
			Normalizes the UI response-mode selection into Gemini response modalities.
			
			Parameters:
			-----------
			response_modalities: Optional[ str ] - UI-selected response mode.
			image_only: bool - Indicates whether the workflow defaults to image output.
			
			Returns:
			--------
			List[ str ] - Normalized Gemini response modalities.
			
		"""
		try:
			self.mode_name = str( response_modalities or '' ).strip( ).upper( )
			
			if self.mode_name == 'TEXT_AND_IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			if self.mode_name == 'TEXT':
				return [ 'TEXT' ]
			
			if self.mode_name == 'IMAGE':
				return [ 'IMAGE' ]
			
			return [ 'IMAGE' ] if image_only else [ 'TEXT' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('_normalize_response_modalities( self, response_modalities=None, '
			                    'image_only=False ) -> List[ str ]')
			raise exception
	
	def _build_grounding_tool( self, image_search: bool = False ) -> Optional[ Tool ]:
		"""
			
			Purpose:
			-----------
			Builds a Google Search grounding tool for supported image models.
			
			Parameters:
			-----------
			image_search: bool - Includes Google Image Search when supported by the model.
			
			Returns:
			--------
			Optional[ Tool ] - Search grounding tool or None.
			
		"""
		try:
			if not self._supports_search_grounding( self.model ):
				return None
			
			self.use_image_search = bool( image_search )
			self.model_name = str( self.model ).strip( )
			
			if self.use_image_search and self._supports_image_search( self.model_name ):
				try:
					return Tool(
						google_search=types.GoogleSearch(
							search_types=types.SearchTypes(
								web_search=types.WebSearch( ),
								image_search=types.ImageSearch( )
							)
						)
					)
				except Exception:
					return Tool( google_search=types.GoogleSearch( ) )
			
			return Tool( google_search=types.GoogleSearch( ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_build_grounding_tool( self, image_search=False ) -> Optional[ Tool ]'
			raise exception
	
	def _get_content_config( self, image_only: bool = False, grounded: bool = False,
			image_search: bool = False, response_modalities: Optional[ str ] = None,
			output_mime_type: Optional[ str ] = None ) -> GenerateContentConfig:
		"""
			
			Purpose:
			-----------
			Creates a Gemini GenerateContentConfig for image workflows.
			
			Parameters:
			-----------
			image_only: bool - Indicates whether only image output should be returned by default.
			grounded: bool - Indicates whether Google Search grounding should be enabled.
			image_search: bool - Indicates whether Google Image Search grounding should be used.
			response_modalities: Optional[ str ] - UI-selected response mode.
			output_mime_type: Optional[ str ] - Desired output image MIME type.
			
			Returns:
			--------
			GenerateContentConfig - Configured content generation settings.
			
		"""
		try:
			self.image_config = None
			self.tool_config = None
			self.grounding_metadata = None
			self.image_kwargs = { }
			if self.aspect_ratio:
				self.image_kwargs[ 'aspect_ratio' ] = self.aspect_ratio
			
			if self.size and self._supports_image_size( self.model ):
				self.image_kwargs[ 'image_size' ] = self.size
			
			if output_mime_type:
				self.image_kwargs[ 'output_mime_type' ] = output_mime_type
			
			if len( self.image_kwargs ) > 0:
				self.image_config = types.ImageConfig( **self.image_kwargs )
			
			if grounded:
				self.grounding_tool = self._build_grounding_tool( image_search=image_search )
				if self.grounding_tool is not None:
					self.tool_config = [ self.grounding_tool ]
			
			self.response_modalities = self._normalize_response_modalities(
				response_modalities=response_modalities,
				image_only=image_only )
			
			self.config_kwargs = {
					'temperature': self.temperature,
					'top_p': self.top_p,
					'candidate_count': self.number,
					'max_output_tokens': self.max_output_tokens,
					'system_instruction': self.instructions,
					'response_modalities': self.response_modalities
			}
			
			if self.image_config is not None:
				self.config_kwargs[ 'image_config' ] = self.image_config
			
			if self.tool_config is not None and len( self.tool_config ) > 0:
				self.config_kwargs[ 'tools' ] = self.tool_config
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('_get_content_config( self, image_only=False, grounded=False, '
			                    'image_search=False, response_modalities=None, '
			                    'output_mime_type=None ) -> GenerateContentConfig')
			raise exception
	
	def _open_image( self, path: str ) -> PIL.Image.Image:
		"""
			
			Purpose:
			-----------
			Opens a local image file for Gemini multimodal requests.
			
			Parameters:
			-----------
			path: str - Path to the local image file.
			
			Returns:
			--------
			PIL.Image.Image - Opened local image.
			
		"""
		try:
			throw_if( 'path', path )
			with PIL.Image.open( path ) as source:
				return source.copy( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_open_image( self, path ) -> PIL.Image.Image'
			raise exception
	
	def _capture_grounding_metadata( self ) -> None:
		"""
			
			Purpose:
			-----------
			Captures grounding metadata from the most recent Gemini content response.
			
			Returns:
			--------
			None
			
		"""
		try:
			self.grounding_metadata = None
			if self.content_response is None:
				return
			
			self.candidates = getattr( self.content_response, 'candidates', None )
			if self.candidates:
				for candidate in self.candidates:
					self.metadata = getattr( candidate, 'grounding_metadata', None )
					if self.metadata is None:
						self.metadata = getattr( candidate, 'groundingMetadata', None )
					
					if self.metadata is not None:
						self.grounding_metadata = self.metadata
						return
		except Exception:
			self.grounding_metadata = None
	
	def _get_first_image( self ) -> Optional[ PIL.Image.Image ]:
		"""
			
			Purpose:
			-----------
			Extracts the first returned image from a Gemini content response.
			
			Returns:
			--------
			Optional[ PIL.Image.Image ] - The first returned image, if any.
			
		"""
		try:
			if self.content_response is None:
				return None
			
			parts = getattr( self.content_response, 'parts', None )
			if parts:
				for part in parts:
					try:
						if getattr( part, 'inline_data', None ) is not None:
							return part.as_image( )
					except Exception:
						continue
			
			candidates = getattr( self.content_response, 'candidates', None )
			if candidates:
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					if content is None:
						continue
					
					candidate_parts = getattr( content, 'parts', None ) or [ ]
					for part in candidate_parts:
						try:
							if getattr( part, 'inline_data', None ) is not None:
								return part.as_image( )
						except Exception:
							continue
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_get_first_image( self ) -> Optional[ PIL.Image.Image ]'
			raise exception
	
	def _get_output_text( self ) -> Optional[ str ]:
		"""
			
			Purpose:
			-----------
			Extracts text output from a Gemini content response.
			
			Returns:
			--------
			Optional[ str ] - The returned text, if any.
			
		"""
		try:
			if self.content_response is None:
				return None
			
			text = getattr( self.content_response, 'text', None )
			if isinstance( text, str ) and text.strip( ):
				return text
			
			parts = getattr( self.content_response, 'parts', None )
			if parts:
				output = [ ]
				for part in parts:
					part_text = getattr( part, 'text', None )
					if isinstance( part_text, str ) and part_text.strip( ):
						output.append( part_text.strip( ) )
				
				if output:
					return '\n'.join( output )
			
			candidates = getattr( self.content_response, 'candidates', None )
			if candidates:
				for candidate in candidates:
					content = getattr( candidate, 'content', None )
					if content is None:
						continue
					
					output = [ ]
					for part in getattr( content, 'parts', None ) or [ ]:
						part_text = getattr( part, 'text', None )
						if isinstance( part_text, str ) and part_text.strip( ):
							output.append( part_text.strip( ) )
					
					if output:
						return '\n'.join( output )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = '_get_output_text( self ) -> Optional[ str ]'
			raise exception
	
	def generate( self, prompt: str, model: str = 'gemini-2.5-flash-image', aspect: str = None,
			number: int = None, temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			resolution: str = None, instruct: str = None, output_mime_type: str = None,
			response_modalities: str = None, grounded: bool = False,
			image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""
			
			Purpose:
			-----------
			Generates a new image based on a descriptive text prompt.
			
			Parameters:
			-----------
			prompt: str - Image description.
			aspect: str - Aspect ratio.
			resolution: str - Output image size when supported by the selected model.
			output_mime_type: str - Requested output MIME type for returned image content.
			response_modalities: str - UI-selected Gemini response mode.
			grounded: bool - Enables Google Search grounding when supported by the selected model.
			image_search: bool - Enables Google Image Search grounding when supported by the selected model.
			
			Returns:
			--------
			Optional[ PIL.Image.Image ] - The generated image.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect
			self.size = resolution
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self._get_content_config(
				image_only=True,
				grounded=grounded,
				image_search=image_search,
				response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type
			)
			self.content_response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt ],
				config=self.content_config
			)
			self.response = self.content_response
			self._capture_grounding_metadata( )
			return self._get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'generate( self, prompt, aspect ) -> Optional[ PIL.Image.Image ]'
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str = 'gemini-2.5-flash-image',
			aspect: str = None, number: int = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, resolution: str = None, instruct: str = None,
			output_mime_type: str = None, response_modalities: str = None,
			grounded: bool = False, image_search: bool = False ) -> Optional[ str ]:
		"""
			
			Purpose:
			-----------
			Analyzes a local image using a text prompt and image input.
			
			Parameters:
			-----------
			prompt: str - Analysis instruction.
			path: str - Path to the local image.
			output_mime_type: str - Reserved for API consistency; not used for text analysis output.
			response_modalities: str - UI-selected Gemini response mode.
			grounded: bool - Enables Google Search grounding when supported by the selected model.
			image_search: bool - Enables Google Image Search grounding when supported by the selected model.
			
			Returns:
			--------
			Optional[ str ] - The analysis text.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect
			self.media_resolution = resolution
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities or 'TEXT'
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self._get_content_config(
				image_only=False,
				grounded=grounded,
				image_search=image_search,
				response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type
			)
			self.content_response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self._open_image( path ) ],
				config=self.content_config
			)
			self.response = self.content_response
			self._capture_grounding_metadata( )
			return self._get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'analyze( self, prompt, path, model ) -> Optional[ str ]'
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gemini-2.5-flash-image',
			aspect: str = None, number: int = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, resolution: str = None, instruct: str = None,
			output_mime_type: str = None, response_modalities: str = None,
			grounded: bool = False, image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""
			
			Purpose:
			-----------
			Edits a local image using a text instruction and image input.
			
			Parameters:
			-----------
			prompt: str - Editing instruction.
			path: str - Path to the local image.
			aspect: str - Aspect ratio.
			resolution: str - Output image size when supported by the selected model.
			output_mime_type: str - Requested output MIME type for returned image content.
			response_modalities: str - UI-selected Gemini response mode.
			grounded: bool - Enables Google Search grounding when supported by the selected model.
			image_search: bool - Enables Google Image Search grounding when supported by the selected model.
			
			Returns:
			--------
			Optional[ PIL.Image.Image ] - The edited image.
			
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect
			self.size = resolution
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.content_config = self._get_content_config(
				image_only=True,
				grounded=grounded,
				image_search=image_search,
				response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type
			)
			self.content_response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self._open_image( path ) ],
				config=self.content_config
			)
			self.response = self.content_response
			self._capture_grounding_metadata( )
			return self._get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'edit( self, prompt, path, model ) -> Optional[ PIL.Image.Image ]'
			raise exception

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
	
	def __init__( self, model: str = 'gemini-embedding-001' ):
		super( ).__init__( )
		self.model = model
		self.temperature = None
		self.top_p = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
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
		return [ 'gemini-embedding-001',
		         'text-multilingual-embedding-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		'''
			
			Returns:
			--------
			List[ str ] of available format options

		'''
		return [ 'float', 'base64' ]
	
	@property
	def task_options( self ) -> List[ str ]:
		'''
			
			Returns:
			--------
			List[ str ] of available embedding tasks

		'''
		return [ 'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY',
		         'CLASSIFICATION', 'CLUSTERING' ]
	
	def create( self, text: str, model: str = 'gemini-embedding-001', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None ) -> List[ float ] | None:
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
			self.client = genai.Client( api_key=self.gemini_api_key )
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
			raise exception

class TTS( Gemini ):
	"""

	    Purpose
	    ___________
	    Class for conversion of text to speech using Gemini TTS output.

	    Attributes:
	    -----------
	    speed           : float - Audio playback speed
	    voice           : str - Persona identifier
	    response        : GenerateContentResponse - Raw response
	    client          : Client - genai instance
	    audio_path      : str - Target path
	    response_format : str - Audio format
	    input_text      : str - Original text

	    Methods:
	    --------
	    create_speech( text, filepath, model, format, speed, voice ) : Generates speech audio

    """
	speed: Optional[ float ]
	voice: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	voice_config: Optional[ VoiceConfig ]
	speech_config: Optional[ SpeechConfig ]
	client: Optional[ genai.Client ]
	audio_path: Optional[ str ]
	response_format: Optional[ str ]
	input_text: Optional[ str ]
	audio_bytes: Optional[ bytes ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-preview-tts' ):
		super( ).__init__( )
		self.number = None
		self.model = model
		self.temperature = None
		self.top_p = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.voice_config = None
		self.speech_config = None
		self.content_config = None
		self.client = None
		self.voice = None
		self.speed = None
		self.response = None
		self.response_format = None
		self.audio_path = None
		self.input_text = None
		self.audio_bytes = None
		self.response_modalities = [ ]
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns list of TTS-capable Gemini models.

		"""
		return [ 'gemini-2.5-flash-preview-tts',
		         'gemini-2.5-pro-preview-tts' ]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns list of available prebuilt voices.

		"""
		return [ 'Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir', 'Leda', 'Orus', 'Aoede', 'Callirhoe',
		         'Autonoe', 'Enceladus', 'Iapetus', 'Umbriel', 'Algieba', 'Despina', 'Erinome',
		         'Algenib', 'Rasalgethi', 'Laomedeia', 'Achernar', 'Alnilam', 'Schedar', 'Gacrux',
		         'Pulcherrima', 'Achird', 'Zubenelgenubi', 'Vindemiatrix', 'Sadachbia',
		         'Sadaltager', 'Sulafar' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns the supported output container formats for this wrapper.

		"""
		return [ 'audio/wav' ]
	
	def _to_wave_bytes( self, pcm_data: bytes, rate: int = 24000, channels: int = 1,
			sample_width: int = 2 ) -> bytes:
		"""

			Purpose:
			--------
			Wraps raw PCM bytes returned by Gemini TTS into a WAV container.

			Parameters:
			-----------
			pcm_data: bytes - Raw PCM audio bytes.
			rate: int - Sample rate.
			channels: int - Number of channels.
			sample_width: int - Sample width in bytes.

			Returns:
			--------
			bytes - WAV file bytes.

		"""
		import io
		import wave
		
		with io.BytesIO( ) as buffer:
			with wave.open( buffer, 'wb' ) as wf:
				wf.setnchannels( channels )
				wf.setsampwidth( sample_width )
				wf.setframerate( rate )
				wf.writeframes( pcm_data )
			
			return buffer.getvalue( )
	
	def create_speech( self, text: str, filepath: str = None,
			model: str = 'gemini-2.5-flash-preview-tts', format: str = 'audio/wav',
			speed: float = None, voice: str = None, frequency: float = None,
			presense: float = None, max_tokens: int = None, instruct: str = None,
			temperature: float = None, top_p: float = None ) -> bytes | str | None:
		"""

			Purpose:
			--------
			Converts text to speech using Gemini TTS. If filepath is provided,
			the generated WAV is written to disk; otherwise WAV bytes are returned.

			Parameters:
			-----------
			text: str - Input text string.
			filepath: str - Optional target local path.
			model: str - Gemini TTS model identifier.
			format: str - Output audio format.
			speed: float - Playback rate hint.
			voice: str - Persona name.
			frequency: float - Frequency penalty.
			presense: float - Presence penalty.
			max_tokens: int - Maximum output tokens.
			instruct: str - Optional system instruction.
			temperature: float - Sampling temperature.
			top_p: float - Nucleus sampling threshold.

			Returns:
			--------
			bytes | str | None - WAV bytes or local path to the created file.

		"""
		try:
			throw_if( 'text', text )
			self.input_text = text
			self.audio_path = filepath
			self.response_format = str( format or 'audio/wav' ).strip( )
			self.speed = speed
			self.voice = str( voice or 'Kore' ).strip( )
			self.max_tokens = max_tokens
			self.model = str( model or self.model or 'gemini-2.5-flash-preview-tts' ).strip( )
			self.frequency_penalty = frequency
			self.presence_penalty = presense
			self.instructions = instruct
			self.temperature = temperature
			self.top_p = top_p
			self.response_modalities = [ 'AUDIO' ]
			self.voice_config = VoiceConfig(
				prebuilt_voice_config=types.PrebuiltVoiceConfig( voice_name=self.voice ) )
			self.speech_config = SpeechConfig( voice_config=self.voice_config )
			self.config_kwargs = { 'response_modalities': self.response_modalities,
					'speech_config': self.speech_config }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=self.input_text,
				config=self.content_config )
			
			self.audio_bytes = None
			for part in self.response.candidates[ 0 ].content.parts:
				if getattr( part, 'inline_data', None ) is not None and part.inline_data.data:
					self.audio_bytes = self._to_wave_bytes( part.inline_data.data )
					break
			
			if self.audio_bytes is None:
				raise ValueError( 'No audio bytes were returned by Gemini TTS.' )
			
			if self.audio_path is not None and str( self.audio_path ).strip( ):
				with open( self.audio_path, 'wb' ) as f:
					f.write( self.audio_bytes )
				
				return self.audio_path
			
			return self.audio_bytes
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, text, filepath, model, format, speed, voice )'
			error = ErrorDialog( exception )
			error.show( )

class Transcription( Gemini ):
	"""

	    Purpose
	    ___________
	    Class handling audio-to-text transcription using Gemini audio understanding.

	    Attributes:
	    -----------
	    client     : Client - GenAI instance
	    transcript : str - Text result
	    file_path  : str - Path to audio file
	    response   : GenerateContentResponse - Raw response

	    Methods:
	    --------
	    transcribe( path, model ) : Transcribes local audio file to text

    """
	client: Optional[ genai.Client ]
	transcript: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float = 0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 10000, instruct: str = None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=self.gemini_api_key )
		self.transcript = None
		self.file_path = None
		self.response = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns list of models supporting audio input.

		"""
		return [ 'gemini-3-flash-preview',
		         'gemini-2.0-flash' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns list of language hints.

		"""
		return [ 'Auto',
		         'English',
		         'Spanish',
		         'French',
		         'Japanese',
		         'German',
		         'Chinese' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns supported audio mime hints.

		"""
		return [ 'audio/wav',
		         'audio/mp3',
		         'audio/x-m4a',
		         'audio/flac' ]
	
	def _build_prompt( self, language: str = None, start_time: float = None, end_time: float = None ) -> str:
		"""

			Purpose:
			--------
			Builds the transcription prompt for Gemini audio understanding.

			Parameters:
			-----------
			language: str - Optional language hint.
			start_time: float - Optional start timestamp in seconds.
			end_time: float - Optional end timestamp in seconds.

			Returns:
			--------
			str - Prompt text.

		"""
		self.prompt_parts = [ 'Generate a verbatim transcript of the speech.' ]
		
		if language is not None and str( language ).strip( ) and str( language ).strip( ) != 'Auto':
			self.prompt_parts.append(
				f'The expected spoken language is {str( language ).strip( )}.' )
		
		if start_time is not None and end_time is not None and end_time >= start_time:
			self.prompt_parts.append(
				f'Only transcribe the portion of the audio between {start_time:0.2f} seconds '
				f'and {end_time:0.2f} seconds.' )
		
		self.prompt_parts.append( 'Return only the transcript text.' )
		return ' '.join( self.prompt_parts )
	
	def transcribe( self, path: str, model: str = 'gemini-3-flash-preview',
			language: str = None, mime_type: str = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, start_time: float = None, end_time: float = None,
			instruct: str = None ) -> Optional[ str ]:
		"""

			Purpose:
			--------
			Transcribes an audio file into text using Gemini audio understanding.

			Parameters:
			-----------
			path: str - Local path to the source audio.
			model: str - Specific GenAI model ID.
			language: str - Optional language hint.
			mime_type: str - Optional mime-type override.
			temperature: float - Sampling temperature.
			top_p: float - Nucleus sampling threshold.
			frequency: float - Frequency penalty.
			presence: float - Presence penalty.
			max_tokens: int - Maximum output tokens.
			start_time: float - Optional start timestamp in seconds.
			end_time: float - Optional end timestamp in seconds.
			instruct: str - Optional system instruction.

			Returns:
			--------
			Optional[ str ] - Verbatim transcript text.

		"""
		try:
			import mimetypes
			
			throw_if( 'path', path )
			self.file_path = path
			self.model = str( model or self.model or 'gemini-3-flash-preview' ).strip( )
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_p = top_p if top_p is not None else self.top_p
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
			self.instructions = instruct if instruct is not None else self.instructions
			self.mime_type = mime_type or mimetypes.guess_type( self.file_path )[ 0 ] or 'audio/wav'
			self.prompt = self._build_prompt(
				language=language,
				start_time=start_time,
				end_time=end_time )
			
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self.uploaded_file ],
				config=self.content_config )
			self.transcript = self.response.text
			return self.transcript
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path, model, language ) -> str'
			error = ErrorDialog( ex )
			error.show( )

class Translation( Gemini ):
	"""

	    Purpose
	    ___________
	    Class for translating spoken audio into text using Gemini audio understanding.

	    Attributes:
	    -----------
	    client          : Client - genai client instance
	    target_language : str - Destination language
	    source_language : str - Source language hint
	    file_path       : str - Audio file path
	    response        : GenerateContentResponse - Raw response

	    Methods:
	    --------
	    translate( path, model, language ) : Translates speech in an audio file

    """
	client: Optional[ genai.Client ]
	target_language: Optional[ str ]
	source_language: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float = 0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 10000,
			instruct: str = None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=self.gemini_api_key )
		self.target_language = None
		self.source_language = None
		self.file_path = None
		self.response = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns list of translation-capable audio models.

		"""
		return [ 'gemini-3-flash-preview',
		         'gemini-2.0-flash' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns supported audio mime hints.

		"""
		return [ 'audio/wav',
		         'audio/mp3',
		         'audio/x-m4a',
		         'audio/flac' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns list of available target languages.

		"""
		return [ 'English',
		         'Spanish',
		         'French',
		         'Japanese',
		         'German',
		         'Chinese' ]
	
	def _build_prompt( self, target: str, source: str = 'Auto', start_time: float = None,
			end_time: float = None ) -> str:
		"""

			Purpose:
			--------
			Builds the translation prompt for Gemini audio understanding.

			Parameters:
			-----------
			target: str - Target translation language.
			source: str - Optional source-language hint.
			start_time: float - Optional start timestamp in seconds.
			end_time: float - Optional end timestamp in seconds.

			Returns:
			--------
			str - Prompt text.

		"""
		self.prompt_parts = [ f'Translate the spoken audio into {target}.' ]
		
		if source is not None and str( source ).strip( ) and str( source ).strip( ) != 'Auto':
			self.prompt_parts.append(
				f'The expected source language is {str( source ).strip( )}.' )
		
		if start_time is not None and end_time is not None and end_time >= start_time:
			self.prompt_parts.append(
				f'Only translate the portion of the audio between {start_time:0.2f} seconds '
				f'and {end_time:0.2f} seconds.' )
		
		self.prompt_parts.append( 'Return only the translated text.' )
		return ' '.join( self.prompt_parts )
	
	def translate( self, path: str, model: str = 'gemini-3-flash-preview',
			language: str = 'English', source: str = 'Auto', mime_type: str = None,
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, start_time: float = None,
			end_time: float = None, instruct: str = None ) -> Optional[ str ]:
		"""

			Purpose:
			--------
			Translates spoken audio from one language to another.

			Parameters:
			-----------
			path: str - Local path to the source audio.
			model: str - Specific GenAI model ID.
			language: str - Target language.
			source: str - Source language hint.
			mime_type: str - Optional mime-type override.
			temperature: float - Sampling temperature.
			top_p: float - Nucleus sampling threshold.
			frequency: float - Frequency penalty.
			presence: float - Presence penalty.
			max_tokens: int - Maximum output tokens.
			start_time: float - Optional start timestamp in seconds.
			end_time: float - Optional end timestamp in seconds.
			instruct: str - Optional system instruction.

			Returns:
			--------
			Optional[ str ] - Translated text.

		"""
		try:
			import mimetypes
			
			throw_if( 'path', path )
			self.file_path = path
			self.model = str( model or self.model or 'gemini-3-flash-preview' ).strip( )
			self.target_language = str( language or 'English' ).strip( )
			self.source_language = str( source or 'Auto' ).strip( )
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_p = top_p if top_p is not None else self.top_p
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
			self.instructions = instruct if instruct is not None else self.instructions
			self.mime_type = mime_type or mimetypes.guess_type( self.file_path )[ 0 ] or 'audio/wav'
			self.prompt = self._build_prompt(
				target=self.target_language,
				source=self.source_language,
				start_time=start_time,
				end_time=end_time )
			
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, self.uploaded_file ],
				config=self.content_config )
			return self.response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Translation'
			ex.method = 'translate( self, path, model, language, source ) -> str'
			error = ErrorDialog( ex )
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
	file_paths: Optional[ List[ str ] ]
	file_lists: Optional[ List[ File ] ]
	response: Optional[ Any ]
	use_vertex: Optional[ bool ]
	collections: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, str ] ]
	
	def __init__( self, model: str = 'gemini-2.0-flash' ):
		super( ).__init__( )
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.project_location = cfg.GOOGLE_CLOUD_LOCATION
		self.model = model
		self.top_p = None
		self.top_k = None
		self.temperature = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.tool_choice = None
		self.stops = [ ]
		self.response_modalities = [ ]
		self.tools = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.storage_client = None
		self.bucket_id = None
		self.file_id = None
		self.display_name = None
		self.media_resolution = None
		self.mime_type = None
		self.file_path = None
		self.file_list = [ ]
		self.response = None
		self.collections = { }
		self.documents = { }
	
	@property
	def file_options( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Returns list of available chat models.
			
		"""
		return self.files
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Returns list of available chat models.
			
		"""
		return [ 'gemini-3.5-flash',
		         'gemini-3.5 flash-lite',
		         'gemini-3.0-flash',
		         'gemini-3.0-flash-lite' ]
	
	@property
	def media_options( self ):
		'''
		
		Purpose:
		--------
		Returns a List[ str ] of media resolution options.
		
		'''
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of the includeable options

		'''
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of thinking effort options

		'''
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL',
		         'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'AUTO',
		         'ANY',
		         'NONE',
		         'VALIDATED' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'google_search',
		         'google_maps',
		         'file_search',
		         'url_context',
		         'code_execution',
		         'computer_use' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available modality options

		'''
		return [ 'MODALITY_UNSPECIFIED', 'TEXT', 'IMAGE', 'AUDIO' ]
	
	@property
	def media_options( self ):
		'''
		
		Purpose:
		--------
		Returns a List[ str ] of media resolution options.
		
		'''
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	def upload( self, filepath: str, name: str = None ) -> File | None:
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
			throw_if( 'filepath', filepath )
			throw_if( 'name', name )
			self.file_path = filepath;
			self.display_name = name
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.response = self.client.files.upload( path=self.file_path,
				config={ 'display_name': self.display_name } )
			return self.response
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'upload( self, path: str, name: str ) -> Optional[ File ]'
			raise ex
	
	def list( self, model: str = 'gemini-2.0-flash', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None ) -> List[ str ]:
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
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.storage_client = storage.Client( api_key=cfg.GOOGLE_API_KEY )
			name = "jeni-financial"
			prefix = "regulations"
			bucket = self.storage_client.bucket( bucket_name=name )
			for blob in bucket.list_blobs( prefix=prefix ):
				self.files.append( blob.name )
			return self.files
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'list_files( self ) -> Optional[ List[ File ] ]'
			raise ex
	
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
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'retrieve( self, file_id: str ) -> Optional[ File ]'
			raise ex
	
	def summarize( self, prompt: str, filepath: str, model: str = 'gemini-2.0-flash',
			temperature: float = None, top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str | None:
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
			self.instructions = instruct
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.gemini_api_key )
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
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'summarize_document( self, prompt, filepath, model ) -> str'
			raise ex
	
	def search( self, prompt: str, filepath: str, model: str = 'gemini-2.0-flash',
			temperature: float = None, top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str | None:
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
			self.instructions = instruct
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.gemini_api_key )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					doc_part = Part.from_bytes( data=f.read( ), mime_type="application/pdf" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ doc_part, self.prompt ], config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_path )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file,
					           self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'search( self, prompt, filepath, model ) -> str'
			raise ex
	
	def survey( self, prompt: str, filepaths: List[ str ], model: str = 'gemini-2.0-flash',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, stops: List[str ] = None ) -> str | None:
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
			throw_if( 'filepaths', filepaths )
			self.prompt = prompt
			self.file_paths = filepaths
			self.model = model
			self.top_p = top_p;
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=self.gemini_api_key )
			if self.use_vertex:
				with open( self.file_path, 'rb' ) as f:
					doc_part = Part.from_bytes( data=f.read( ), mime_type="application/pdf" )
				response = self.client.models.generate_content( model=self.model,
					contents=[ doc_part, self.prompt ], config=self.content_config )
			else:
				uploaded_file = self.client.files.upload( path=self.file_paths )
				response = self.client.models.generate_content( model=self.model,
					contents=[ uploaded_file, self.prompt ], config=self.content_config )
			return response.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'survey( self, prompt, filepaths, model ) -> str'
			raise ex
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str | None:
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
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config, system_instruction=self.instructions )
			self.client = genai.Client( api_key=self.gemini_api_key )
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
	
	def search_maps( self, prompt: str, model: str = 'gemini-2.5-flash-lite', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str | None:
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
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config )
			self.client = genai.Client( api_key=self.gemini_api_key )
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
			self.client = genai.Client( api_key=self.gemini_api_key )
			self.client.files.delete( name=self.file_id )
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'FileStore'
			ex.method = 'delete( self, file_id: str ) -> bool'
			raise ex

class VectorStores( Gemini ):
	'''

		Purpose:
		--------
		Encapsulate Google Cloud Storage as a Vector Store backend for the Buddy
		application. Buckets are treated as collections and objects (blobs) as
		stored vector documents or assets.

		Attributes:
		-----------
		project_id   : str | None
		bucket_name  : str | None
		object_name  : str | None
		file_path    : str | None
		client       : storage.Client | None
		bucket       : storage.Bucket | None
		response     : Any
		collections  : Dict[ str, str ] | None
		documents    : Dict[ str, str ] | None

		Methods:
		--------
		upload( path, bucket, name )
		retrieve( bucket, name )
		list( bucket )
		delete( bucket, name )

	'''
	project_id: Optional[ str ]
	bucket_name: Optional[ str ]
	object_name: Optional[ str ]
	file_path: Optional[ str ]
	file_ids: Optional[ List[ str ] ]
	store_ids: Optional[ List[ str ] ]
	client: Optional[ storage.Client ]
	bucket: Optional[ storage.Bucket ]
	response: Optional[ Any ]
	collections: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.client = storage.Client( project=self.project_id )
		self.bucket_name = None
		self.object_name = None
		self.file_path = None
		self.media_resolution = None
		self.file_ids = [ ]
		self.store_ids = [ ]
		self.stops = [ ]
		self.response_modalities = [ ]
		self.tools = [ ]
		self.domains = [ ]
		self.http_options = { }
		self.bucket = None
		self.response = None
		self.collections = \
			{
					'Federal Financial Data': 'jeni-financial/data',
					'Federal Financial Regulations': 'jeni-financial/regulations',
					'DoW Financial Data': 'jeni_dow/budget/data',
					'DoW Financial Regulations': 'jeni_dow/budget/regulations',
					'DoA Financial Data': 'jenni-doa/Financial Data',
			}
		self.documents = \
			{
					'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
					'SF133.csv': 'file-32s641QK1Xb5QUatY3zfWF',
					'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
					'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn'
			}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Returns list of available chat models."""
		return [ 'gemini-2.5-flash',
		         'gemini-2.5 flash image',
		         'gemini-2.5 flash-tts',
		         'gemini-2.5 flash-lite',
		         'gemini-2.0-flash',
		         'gemini-2.0-flash-lite' ]
	
	@property
	def media_options( self ):
		'''
		
		Purpose:
		--------
		Returns a List[ str ] of media resolution options.
		
		'''
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	def create( self, bucket: str, name: str ):
		"""

			Purpose:
			--------
			Delete an object from a GCS bucket.

			Parameters:
			-----------
			bucket : str
				GCS bucket name.
			name   : str
				Object (blob) name.

			Returns:
			--------
			bool

		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.delete( )
			return True
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'delete( self, bucket, name )'
			raise ex
	
	def upload( self, path: str, bucket: str, name: str = None ):
		"""

			Purpose:
			--------
			Upload a local file to a Google Cloud Storage bucket.

			Parameters:
			-----------
			path   : str
				Local filesystem path to the file.
			bucket : str
				Target GCS bucket name.
			name   : str | None
				Optional object name override.

			Returns:
			--------
			storage.Blob | None

		"""
		try:
			throw_if( 'path', path )
			throw_if( 'bucket', bucket )
			self.file_path = path
			self.bucket_name = bucket
			self.object_name = name or path.split( '/' )[ -1 ]
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.upload_from_filename( self.file_path )
			self.response = blob
			return blob
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'upload( self, path, bucket, name )'
			raise ex
	
	def retrieve( self, bucket: str, name: str ):
		"""
	
				Purpose:
				--------
				Retrieve metadata for a stored object in GCS.
	
				Parameters:
				-----------
				bucket : str
					GCS bucket name.
				name   : str
					Object (blob) name.
	
				Returns:
				--------
				storage.Blob | None

		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.get_blob( self.object_name )
			return blob
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'retrieve( self, bucket, name )'
			raise ex
	
	def list( self, bucket: str ):
		"""

			Purpose:
			--------
			List all objects stored in a given GCS bucket.

			Parameters:
			-----------
			bucket : str
				GCS bucket name.

			Returns:
			--------
			List[ storage.Blob ] | None

		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket_name = bucket
			self.bucket = self.client.bucket( self.bucket_name )
			blobs = list( self.bucket.list_blobs( ) )
			self.documents = { blob.name: blob.id for blob in blobs }
			return blobs
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'list( self, bucket )'
			raise ex
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str | None:
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
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config, system_instruction=self.instructions )
			self.client = genai.Client( api_key=self.gemini_api_key )
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
	
	def search_maps( self, prompt: str, model: str = 'gemini-2.5-flash-lite', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str | None:
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
			self.instructions = instruct
			self.tool_config = [
					types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config )
			self.client = genai.Client( api_key=self.gemini_api_key )
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
	
	def delete( self, bucket: str, name: str ):
		"""

			Purpose:
			--------
			Delete an object from a GCS bucket.

			Parameters:
			-----------
			bucket : str
				GCS bucket name.
			name   : str
				Object (blob) name.

			Returns:
			--------
			bool

		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.delete( )
			return True
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'VectorStores'
			ex.method = 'delete( self, bucket, name )'
			raise ex
