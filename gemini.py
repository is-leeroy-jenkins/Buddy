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
from google.genai.types import ( Part, GenerateContentConfig, ImageConfig, FunctionCallingConfig,
                                GenerateImagesConfig, GenerateVideosConfig, ThinkingConfig,
                                GeneratedImage, EmbedContentConfig, Content, ContentEmbedding,
                                Candidate, HttpOptions, GenerateImagesResponse, Field,
                                FileSearchStore, FileSearch,
                                GenerateContentResponse, GenerateVideosResponse, Image, File,
                                SpeakerVoiceConfig, VoiceConfig, SpeechConfig, Tool, ToolConfig,
                                GoogleSearch, UrlContext, SafetySetting, HarmCategory,
                                HarmBlockThreshold )

def throw_if( name: str, value: object ) -> None:
	"""
	
		Purpose:
		--------
		Raises a ValueError when a required argument is None or empty.
		
		Parameters:
		-----------
		name: str - Argument name used in the error message.
		value: object - Argument value to validate.
		
		Returns:
		--------
		None
	
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
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
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.google_api_key = cfg.GOOGLE_API_KEY
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
		self.grounding_metadata = None
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
		self.file_search_store_names = [ ]
		self.include_server_side_tool_invocations = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Returns list of available chat llm.
			
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
		         'file_search',
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
		return [ 'auto', 'any', 'none', 'validated' ]
	
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
		return [ '', 'text', 'image', 'audio' ]
	
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
	
	def get_supported_tools( self, model: str ) -> List[ str ]:
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
			throw_if( 'model', model )
			self.model_name = str( model ).strip( ).lower( )
			self.options = [ 'google_search', 'url_context', 'file_search', 'code_execution' ]
			
			if self.supports_google_maps( self.model_name ):
				self.options.append( 'google_maps' )
			
			return self.options
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_supported_tools( self, model: str=None )'
			raise exception
	
	def supports_google_maps( self, model: str ) -> bool:
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
			throw_if( 'model', model )
			self.model_name = model.strip( ).lower( )
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
			exception.method = 'supports_google_maps( self, model: str=None ) -> bool'
			raise exception
	
	def build_urls( self, urls: List[ str ], max_urls: int = 10 ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Normalizes URL context values selected or entered in the UI.
			
			Parameters:
			-----------
			urls: List[ str ] - Candidate URL values.
			max_urls: int - Optional maximum number of URLs to include.
			
			Returns:
			--------
			List[ str ] - Clean URL list.
		
		"""
		try:
			throw_if( 'max_urls', max_urls )
			self.urls = urls if urls is not None else [ ]
			for url in urls:
				if url is None:
					continue
				
				self.url = url.strip( )
				if not self.url:
					continue
				
				self.urls.append( self.url )
			
			self.max_urls = max_urls
			if self.max_urls is not None:
				self.urls = self.urls[ : self.max_urls ]
			
			return self.urls
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_urls( self, urls: List[ str ]=None, max_urls: int=None )'
			raise exception
	
	def append_urls_to_content( self, content: str, urls: List[ str ] ) -> str | None:
		"""
		
			Purpose:
			--------
			Appends URL context values to the optional content block used in generation.
			
			Parameters:
			-----------
			content: str - Optional content/context text.
			urls: List[ str ] - URLs to include in the prompt context.
			
			Returns:
			--------
			str | None - Combined context block or None.
		
		"""
		try:
			self.content_blocks = [ ]
			self.content_blocks.append( content.strip( ) )
			self.urls = urls
			if len( self.urls ) > 0:
				self.content_blocks.append( 'Reference URLs:\n' + '\n'.join( self.urls ) )
			
			return '\n\n'.join( self.content_blocks ) if len( self.content_blocks ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('append_urls_to_content( self, **kwargs ) -> str')
			raise exception
	
	def build_tools( self, tools: List[ str ] ) -> List[ Tool ] | None:
		"""
		
			Purpose:
			--------
			Builds the Google Search grounding tool for Text mode when explicitly enabled.
			
			Parameters:
			-----------
			tools: List[ str ] - Derived tool names for the request.
			file_search_store_names: List[ str ] - Ignored for Google Search-only grounding.
			
			Returns:
			--------
			List[ Tool ] | None - Google Search tool list or None.
		
		"""
		try:
			self.tools = [
					str( tool ).strip( )
					for tool in (tools or [ ])
					if str( tool ).strip( )
			]
			
			if 'google_search' not in self.tools:
				return None
			
			return [ Tool( google_search=GoogleSearch( ) ) ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, tools: List[ str ], file_search_store_names: List[ str ] )'
			raise exception
		
	def build_tool_config( self, tool_choice: str = None,
			tools: List[ Tool ] = None ) -> ToolConfig | None:
		"""
		
			Purpose:
			--------
			Builds Gemini tool configuration from the selected tool-choice mode.
			
			Parameters:
			-----------
			tool_choice: str - Tool choice mode selected in the UI.
			tools: List[ Tool ] - Configured Gemini tool objects.
			
			Returns:
			--------
			ToolConfig | None - Tool configuration or None.
		
		"""
		try:
			self.tool_choice = str( tool_choice or '' ).strip( ).lower( )
			self.tool_objects = tools if tools is not None else [ ]
			
			if not self.tool_choice:
				return None
			
			if self.tool_choice == 'auto':
				return None
			
			if len( self.tool_objects ) == 0:
				return None
			
			if self.tool_choice not in [ 'any', 'none' ]:
				return None
			
			return ToolConfig( function_calling_config=FunctionCallingConfig(
				mode=self.tool_choice.upper( ) ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_tool_config( self, **kwargs) -> ToolConfig | None')
			raise exception
	
	def build_modalities( self, modalities: List[ str ] ) -> List[ str ] | None:
		"""
			
				Purpose:
				--------
				Normalizes optional response modality values selected in the UI.
				
				Parameters:
				-----------
				modalities: List[ str ] - Candidate response modalities.
				
				Returns:
				--------
				List[ str ] | None - Clean response modality list or None.
			
			"""
		try:
			self.modalities = [ ]
			
			for modality in (modalities or [ ]):
				if modality is None:
					continue
				
				self.modality = str( modality ).strip( ).upper( )
				if self.modality in [ 'TEXT', 'IMAGE', 'AUDIO' ]:
					self.modalities.append( self.modality )
			
			return self.modalities if len( self.modalities ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_modalities( self, modalities: List[ str ] )'
			raise exception
	
	def build_reasoning( self, reasoning: str ) -> ThinkingConfig | None:
		"""
		
			Purpose:
			--------
			Builds Gemini thinking configuration when a reasoning level is selected.
			
			Parameters:
			-----------
			reasoning: str - Reasoning level selected in the UI.
			
			Returns:
			--------
			ThinkingConfig | None - Thinking configuration or None.
		
		"""
		try:
			self.reasoning = str( reasoning or '' ).strip( ).upper( )
			
			if not self.reasoning:
				return None
			
			if self.reasoning == 'THINKING_LEVEL_UNSPECIFIED':
				return None
			
			if self.reasoning not in [ 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH' ]:
				return None
			
			return ThinkingConfig( thinking_level=self.reasoning )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_reasoning( self, reasoning: str ) -> ThinkingConfig | None'
			raise exception
	
	def build_safety_settings( self, safety_profile: str ) -> List[ SafetySetting ] | None:
		"""
		
			Purpose:
			--------
			Builds Gemini safety settings from the optional UI safety profile.
			
			Parameters:
			-----------
			safety_profile: str - Safety threshold name selected in the UI.
			
			Returns:
			--------
			List[ SafetySetting ] | None - Safety settings or None.
		
		"""
		try:
			self.safety_profile = str( safety_profile or '' ).strip( ).upper( )
			
			if not self.safety_profile:
				return None
			
			self.threshold = getattr( HarmBlockThreshold, self.safety_profile, None )
			if self.threshold is None:
				return None
			
			self.categories = [ ]
			for name in [
					'HARM_CATEGORY_HATE_SPEECH',
					'HARM_CATEGORY_HARASSMENT',
					'HARM_CATEGORY_SEXUALLY_EXPLICIT',
					'HARM_CATEGORY_DANGEROUS_CONTENT',
					'HARM_CATEGORY_CIVIC_INTEGRITY' ]:
				self.category = getattr( HarmCategory, name, None )
				if self.category is not None:
					self.categories.append( self.category )
			
			if len( self.categories ) == 0:
				return None
			
			return [
					SafetySetting( category=category, threshold=self.threshold )
					for category in self.categories
			]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_safety_settings( self, safety_profile: str )'
			raise exception
	
	def get_output_text( self ) -> Optional[ str ]:
		"""
		
			Purpose:
			--------
			Extracts text output from a Gemini generate_content response.
			
			Returns:
			--------
			Optional[ str ] - Text output or None.
		
		"""
		try:
			if self.content_response is None:
				return None
			
			self.text = getattr( self.content_response, 'text', None )
			if isinstance( self.text, str ) and self.text.strip( ):
				return self.text.strip( )
			
			self.parts = getattr( self.content_response, 'parts', None )
			if self.parts:
				self.output = [ ]
				for part in self.parts:
					self.part_text = getattr( part, 'text', None )
					if isinstance( self.part_text, str ) and self.part_text.strip( ):
						self.output.append( self.part_text.strip( ) )
				
				if len( self.output ) > 0:
					return '\n'.join( self.output ).strip( )
			
			self.candidates = getattr( self.content_response, 'candidates', None )
			if self.candidates:
				self.output = [ ]
				for candidate in self.candidates:
					self.content = getattr( candidate, 'content', None )
					if self.content is None:
						continue
					
					for part in getattr( self.content, 'parts', None ) or [ ]:
						self.part_text = getattr( part, 'text', None )
						if isinstance( self.part_text, str ) and self.part_text.strip( ):
							self.output.append( self.part_text.strip( ) )
				
				if len( self.output ) > 0:
					return '\n'.join( self.output ).strip( )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_output_text( self ) -> Optional[ str ]'
			raise exception
	
	def parse_response_schema( self, response_schema: Any ) -> Any:
		"""
		
			Purpose:
			--------
			Normalizes a structured-output schema passed as a dict, JSON string, or schema class.
			
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
				return response_schema
			
			if not isinstance( response_schema, str ):
				return response_schema
			
			self.schema_text = response_schema.strip( )
			if not self.schema_text:
				return None
			
			return json.loads( self.schema_text )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'parse_response_schema( self, response_schema: Any )'
			raise exception
	
	def build_contents( self, prompt: str, content: str, context: List[ Any ] = None ) -> str | List[ Content ]:
		"""
		
			Purpose:
			--------
			Builds Gemini contents from the current prompt and any prior conversational context.
			
			Parameters:
			-----------
			prompt: str - Current user prompt.
			content: str - Optional prepended content block.
			context: List[ Any ] - Prior chat messages or Gemini Content objects.
			
			Returns:
			--------
			Union[ str, List[ Content ] ] - Contents payload for Gemini.
		
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = str( prompt ).strip( )
			self.context = context if context is not None else [ ]
			self.content_block = str( content or '' ).strip( )
			self.contents = [ ]
			
			for item in self.context:
				if item is None:
					continue
				
				if isinstance( item, Content ):
					self.contents.append( item )
					continue
				
				if not isinstance( item, dict ):
					continue
				
				role = str( item.get( 'role', 'user' ) or 'user' ).strip( )
				text = item.get( 'content', None )
				if text is None:
					continue
				
				text = str( text ).strip( )
				if not text:
					continue
				
				if role == 'assistant':
					self.contents.append( Content( role='model',
						parts=[ Part.from_text( text=text ) ] ) )
				else:
					self.contents.append( Content( role='user',
						parts=[ Part.from_text( text=text ) ] ) )
			
			self.user_text = self.prompt
			if self.content_block:
				self.user_text = f'{self.content_block}\n\n{self.user_text}'
			
			self.contents.append( Content( role='user',
				parts=[ Part.from_text( text=self.user_text ) ] ) )
			
			return self.contents
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_contents( self, prompt: str, content: str, context: List[ Any ]=None )'
			raise exception
	
	def capture_grounding_metadata( self ) -> None:
		"""
		
			Purpose:
			--------
			Captures grounding metadata from the most recent Gemini text response.
			
			Parameters:
			-----------
			None
			
			Returns:
			--------
			None
		
		"""
		try:
			self.grounding_metadata = None
			
			if self.content_response is None:
				return
			
			self.candidates = getattr( self.content_response, 'candidates', None )
			if not self.candidates:
				return
			
			for candidate in self.candidates:
				self.metadata = getattr( candidate, 'grounding_metadata', None )
				if self.metadata is None:
					self.metadata = getattr( candidate, 'groundingMetadata', None )
				
				if self.metadata is not None:
					self.grounding_metadata = self.metadata
					return
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'capture_grounding_metadata( self )'
			raise exception
	
	def get_grounding_sources( self ) -> List[ Dict[ str, str ] ]:
		"""
		
			Purpose:
			--------
			Extracts displayable source links from Gemini grounding metadata.
			
			Parameters:
			-----------
			None
			
			Returns:
			--------
			List[ Dict[ str, str ] ] - Grounding source dictionaries.
		
		"""
		try:
			self.sources = [ ]
			
			if self.grounding_metadata is None:
				return self.sources
			
			self.chunks = getattr( self.grounding_metadata, 'grounding_chunks', None )
			if self.chunks is None:
				self.chunks = getattr( self.grounding_metadata, 'groundingChunks', None )
			
			if not self.chunks:
				return self.sources
			
			for chunk in self.chunks:
				self.web = getattr( chunk, 'web', None )
				if self.web is None and isinstance( chunk, dict ):
					self.web = chunk.get( 'web' )
				
				if self.web is None:
					continue
				
				if isinstance( self.web, dict ):
					self.uri = self.web.get( 'uri' ) or self.web.get( 'url' )
					self.title = self.web.get( 'title' ) or self.uri
				else:
					self.uri = getattr( self.web, 'uri', None )
					if self.uri is None:
						self.uri = getattr( self.web, 'url', None )
					
					self.title = getattr( self.web, 'title', None ) or self.uri
				
				if self.uri:
					self.sources.append(
						{
								'title': str( self.title or self.uri ),
								'url': str( self.uri ),
								'snippet': ''
						} )
			
			return self.sources
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_grounding_sources( self ) -> List[ Dict[ str, str ] ]'
			raise exception
	
	def get_structured_history( self ) -> List[ Content ] | None:
		"""
		
			Purpose:
			--------
			Builds the full structured conversation history for reuse in a subsequent Gemini request.
			
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
			
			if self.content_response is not None:
				self.candidates = getattr( self.content_response, 'candidates', None )
				if self.candidates:
					for candidate in self.candidates:
						self.response_content = getattr( candidate, 'content', None )
						if isinstance( self.response_content, Content ):
							self.history.append( self.response_content )
							break
			
			return self.history if len( self.history ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_structured_history( self ) -> List[ Content ] | None'
			raise exception
	
	def build_config( self, model: str = 'gemini-2.5-flash-lite', number: int = None,
			temperature: float = None, top_p: float = None, top_k: int = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None, response_format: str = None,
			tools: List[ str ] = None, tool_choice: str = None, reasoning: str = None,
			modalities: List[ str ] = None, media_resolution: str = None,
			response_schema: Any = None, safety_profile: str = None,
			file_search_store_names: List[ str ] = None ) -> GenerateContentConfig:
		"""
		
			Purpose:
			--------
			Builds the GenerateContentConfig object used for Gemini text generation.
			
			Parameters:
			-----------
			model: str - Gemini model identifier.
			number: int - Candidate count.
			temperature: float - Sampling temperature.
			top_p: float - Nucleus sampling probability.
			top_k: int - Top-k token selection count.
			frequency: float - Frequency penalty.
			presence: float - Presence penalty.
			max_tokens: int - Maximum output tokens.
			stops: List[ str ] - Stop sequences.
			instruct: str - System instruction text.
			response_format: str - Response MIME type.
			tools: List[ str ] - Selected tool names.
			tool_choice: str - Tool-choice mode.
			reasoning: str - Thinking level.
			modalities: List[ str ] - Response modalities.
			media_resolution: str - Media resolution.
			response_schema: Any - Structured-output schema.
			safety_profile: str - Safety profile.
			file_search_store_names: List[ str ] - Gemini File Search Store names.
			
			Returns:
			--------
			GenerateContentConfig - Configured content settings.
		
		"""
		try:
			self.model = str( model or self.model or 'gemini-2.5-flash-lite' ).strip( )
			throw_if( 'model', self.model )
			
			self.number = number
			self.candidate_count = int( self.number or 0 )
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = int( top_k or 0 )
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = int( max_tokens or 0 )
			self.stops = stops if stops is not None else [ ]
			self.instructions = instruct
			self.file_search_store_names = file_search_store_names
			self.response_mime_type = str( response_format or '' ).strip( )
			self.response_schema = self.parse_response_schema( response_schema )
			self.safety_settings = self.build_safety_settings( safety_profile )
			self.tool_choice = tool_choice
			self.media_resolution = str( media_resolution ).strip( ) if media_resolution else None
			self.tool_objects = self.build_tools( tools=tools )
			self.function_tool_config = self.build_tool_config( tool_choice=self.tool_choice,
				tools=self.tool_objects )
			self.response_modalities = self.build_modalities( modalities=modalities )
			self.thought_config = self.build_reasoning( reasoning )
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None and float( self.top_p ) > 0:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.top_k > 0:
				self.config_kwargs[ 'top_k' ] = self.top_k
			
			if self.max_tokens > 0:
				self.config_kwargs[ 'max_output_tokens' ] = self.max_tokens
			
			if self.candidate_count > 0:
				self.config_kwargs[ 'candidate_count' ] = self.candidate_count
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.frequency_penalty is not None:
				self.config_kwargs[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None:
				self.config_kwargs[ 'presence_penalty' ] = self.presence_penalty
			
			if self.stops is not None and len( self.stops ) > 0:
				self.config_kwargs[ 'stop_sequences' ] = self.stops
			
			if self.response_mime_type:
				self.config_kwargs[ 'response_mime_type' ] = self.response_mime_type
			
			if self.response_schema is not None:
				if isinstance( self.response_schema, dict ):
					self.config_kwargs[ 'response_json_schema' ] = self.response_schema
				else:
					self.config_kwargs[ 'response_schema' ] = self.response_schema
			
			if self.media_resolution is not None:
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
			exception.method = 'build_config( self, model ) -> GenerateContentConfig'
			raise exception
	
	def generate_text( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			number: int = None, temperature: float = None, top_p: float = None,
			top_k: int = None, frequency: float = None, presence: float = None,
			max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None, response_format: str = None,
			tools: List[ str ] = None, tool_choice: str = None, reasoning: str = None,
			modalities: List[ str ] = None, media_resolution: str = None,
			context: List[ Dict[ str, Any ] ] = None, content: str = None,
			urls: List[ str ] = None, max_urls: int = None, response_schema: Any = None,
			safety_profile: str = None, file_search_store_names: List[ str ] = None,
			stream: bool = False, stream_handler: Any = None ) -> str | None:
		"""
		
			Purpose:
			--------
			Generates a text completion based on the provided prompt and configuration.
			
			Parameters:
			-----------
			prompt: str - The text input for the model.
			model: str - The Gemini model identifier.
			number: int - Candidate count.
			temperature: float - Sampling temperature.
			top_p: float - Nucleus sampling probability.
			top_k: int - Top-k token selection count.
			frequency: float - Frequency penalty.
			presence: float - Presence penalty.
			max_tokens: int - Maximum output tokens.
			stops: List[ str ] - Stop sequences.
			instruct: str - System instruction text.
			response_format: str - Response MIME type.
			tools: List[ str ] - Selected Gemini tools.
			tool_choice: str - Tool-choice mode.
			reasoning: str - Thinking level.
			modalities: List[ str ] - Response modalities.
			media_resolution: str - Media resolution.
			context: List[ Dict[ str, Any ] ] - Conversation context.
			content: str - Additional context block.
			urls: List[ str ] - URLs used with URL context.
			max_urls: int - Maximum URLs to include.
			response_schema: Any - Structured-output schema.
			safety_profile: str - Safety profile.
			file_search_store_names: List[ str ] - File Search Store resource names.
			stream: bool - Whether to stream the response.
			stream_handler: Any - Optional callback for streaming chunks.
			
			Returns:
			--------
			str | None - The generated text response or None.
		
		"""
		try:
			throw_if( 'prompt', prompt )
			self.model = str( model or self.model or 'gemini-2.5-flash-lite' ).strip( )
			throw_if( 'model', self.model )
			
			self.gemini_api_key = (
					self.gemini_api_key
					or self.google_api_key
					or os.environ.get( 'GEMINI_API_KEY' )
					or os.environ.get( 'GOOGLE_API_KEY' )
			)
			throw_if( 'gemini_api_key', self.gemini_api_key )
			
			self.stream = bool( stream )
			self.urls = self.build_urls( urls=urls, max_urls=max_urls )
			self.content_block = self.append_urls_to_content( content=content, urls=self.urls )
			self.contents = self.build_contents( prompt=prompt, context=context,
				content=self.content_block )
			self.content_config = self.build_config( model=self.model, number=number,
				temperature=temperature, top_p=top_p, top_k=top_k, frequency=frequency,
				presence=presence, max_tokens=max_tokens, stops=stops, instruct=instruct,
				response_format=response_format, tools=tools, tool_choice=tool_choice,
				reasoning=reasoning, modalities=modalities, media_resolution=media_resolution,
				response_schema=response_schema, safety_profile=safety_profile,
				file_search_store_names=file_search_store_names )
			
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			if self.stream:
				self.stream_response = self.client.models.generate_content_stream(
					model=self.model, contents=self.contents, config=self.content_config )
				
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
				
				return self.stream_response
			
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config )
			self.capture_grounding_metadata( )
			
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
	
	def __init__( self, model: str='gemini-2.5-flash-image' ):
		super( ).__init__( )
		self.number = None
		self.model = model
		self.client = None
		self.instructions = None
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
			Returns list of image generation llm.
			
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
		return [ 'text', 'image', 'text_and_image' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of thinking effort options

		'''
		return [ 'unspecified', 'minimal',
		         'low', 'medium', 'high' ]
	
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
		return [ 'auto', 'any', 'none', 'validated' ]
	
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
		
	def supports_image_size( self, model: str='gemini-2.5-flash-image' ) -> bool:
		"""
			
			Purpose:
			-----------
			Determines whether the selected Gemini image model supports explicit image-size
			configuration through types.ImageConfig.image_size.
			
			Parameters:
			-----------
			model: str - Gemini image model identifier selected by the UI.
			
			Returns:
			--------
			bool - True when the model supports image_size; otherwise False.
			
		"""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			self.image_size_models = [ 'gemini-3.1-flash-image-preview',
					'gemini-3-pro-image-preview' ]
			return self.model_name in self.image_size_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_size( self, model: str ) -> bool'
			raise exception
	
	def supports_search_grounding( self, model: str='gemini-2.5-flash-image' ) -> bool:
		"""
			
			Purpose:
			-----------
			Determines whether the selected Gemini image model should expose standard Google
			Search grounding in Image mode.
			
			Parameters:
			-----------
			model: str - Gemini image model identifier selected by the UI.
			
			Returns:
			--------
			bool - True when Google Search grounding is supported; otherwise False.
			
		"""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			self.search_grounding_models = [ 'gemini-3.1-flash-image-preview',
					'gemini-3-pro-image-preview' ]
			return self.model_name in self.search_grounding_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_search_grounding( self, model: str ) -> bool'
			raise exception
	
	def supports_image_search( self, model: str='gemini-2.5-flash-image' ) -> bool:
		"""
			
			Purpose:
			-----------
			Determines whether the selected Gemini image model supports Google Image Search
			grounding through the google_search.search_types.image_search configuration.
			
			Parameters:
			-----------
			model: str - Gemini image model identifier selected by the UI.
			
			Returns:
			--------
			bool - True when Google Image Search grounding is supported; otherwise False.
			
		"""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			return self.model_name == 'gemini-3.1-flash-image-preview'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_search( self, model: str ) -> bool'
			raise exception
	
	def normalize_response_modalities( self, response_modalities: Optional[ str ],
			image_only: bool=False ) -> List[ str ]:
		"""
			
			Purpose:
			-----------
			Normalizes the UI response-mode selection into a Gemini-compatible response
			modalities list.
			
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
			
			if self.mode_name == 'TEXT,IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			if self.mode_name == 'TEXT, IMAGE':
				return [ 'TEXT', 'IMAGE' ]
			
			return [ 'IMAGE' ] if image_only else [ 'TEXT' ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = (
					'normalize_response_modalities( self, response_modalities: Optional[str], '
					'image_only: bool=False ) -> List[str]')
			raise exception
	
	def build_grounding_tool( self, image_search: bool=False ) -> Optional[ Tool ]:
		"""
			
			Purpose:
			-----------
			Builds a Gemini Google Search grounding tool for image workflows.
			
			Parameters:
			-----------
			image_search: bool - Indicates whether Google Image Search should be requested
			when supported by the selected model.
			
			Returns:
			--------
			Optional[ Tool ] - Configured Google Search tool, or None when unsupported.
			
		"""
		try:
			if not self.supports_search_grounding( self.model ):
				return None
			
			self.use_image_search = bool( image_search )
			self.model_name = str( self.model or '' ).strip( ).lower( )
			if self.use_image_search and self.supports_image_search( self.model_name ):
				return Tool( google_search=types.GoogleSearch( search_types=types.SearchTypes(
							web_search=types.WebSearch( ), image_search=types.ImageSearch( ) ) ) )
			
			return Tool( google_search=types.GoogleSearch( ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'build_grounding_tool( self, image_search: bool=False ) -> Optional[Tool]'
			raise exception
	
	def get_content_config( self, response_modalities: Optional[ str ], image_only: bool=False,
			image_search: bool=False, grounded: bool=False,
			output_mime_type: Optional[ str ]=None ) -> GenerateContentConfig:
		"""
			
			Purpose:
			-----------
			Creates a Gemini GenerateContentConfig for image-generation, image-analysis, and
			image-editing workflows using SDK-compatible request fields.
			
			Parameters:
			-----------
			response_modalities: Optional[str] - UI-selected response mode.
			image_only: bool - Indicates whether the workflow defaults to image output.
			image_search: bool - Indicates whether Google Image Search grounding should be used.
			grounded: bool - Indicates whether Google Search grounding should be enabled.
			output_mime_type: Optional[str] - Local output preference retained for UI use; not
			passed into types.ImageConfig.
			
			Returns:
			--------
			GenerateContentConfig - Configured Gemini content-generation settings.
			
		"""
		try:
			self.image_only = image_only
			self.image_config = None
			self.tool_config = None
			self.grounding_metadata = None
			self.output_mime_type = str( output_mime_type or '' ).strip( ) or None
			self.image_kwargs = { }
			self.aspect_value = str( self.aspect_ratio or '' ).strip( )
			if self.aspect_value:
				self.image_kwargs[ 'aspect_ratio' ] = self.aspect_value
			
			self.size_value = str( self.size or '' ).strip( )
			if self.size_value and self.supports_image_size( self.model ):
				self.image_kwargs[ 'image_size' ] = self.size_value
			
			if len( self.image_kwargs ) > 0:
				self.image_config = types.ImageConfig( **self.image_kwargs )
			
			if grounded:
				self.grounding_tool = self.build_grounding_tool( image_search=image_search )
				if self.grounding_tool is not None:
					self.tool_config = [ self.grounding_tool ]
			
			self.response_modalities = self.normalize_response_modalities(
				response_modalities=response_modalities, image_only=image_only )
			
			self.config_kwargs = { 'response_modalities': self.response_modalities }
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.number is not None and int( self.number or 0 ) > 0:
				self.config_kwargs[ 'candidate_count' ] = int( self.number )
			
			if self.max_output_tokens is not None and int( self.max_output_tokens or 0 ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( self.max_output_tokens )
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ] = str( self.instructions ).strip( )
			
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
			exception.method = 'get_content_config( self, **kwargs ) -> GenerateContentConfig'
			raise exception
	
	def open_image( self, path: str ) -> PIL.Image.Image:
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
			exception.method = 'open_image( self, path ) -> PIL.Image.Image'
			raise exception
	
	def capture_metadata( self ) -> None:
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
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'capture_metadata( self )'
			raise exception
	
	def get_first_image( self ) -> Optional[ PIL.Image.Image ]:
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
			exception.method = 'get_first_image( self ) -> Optional[ PIL.Image.Image ]'
			raise exception
	
	def get_output_text( self ) -> Optional[ str ]:
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
			exception.method = 'get_output_text( self ) -> Optional[ str ]'
			raise exception
	
	def generate( self, prompt: str, model: str='gemini-2.5-flash-image', aspect: str=None,
			number: int=None, temperature: float=None, top_p: float=None,
			frequency: float=None, presence: float=None, max_tokens: int=None,
			resolution: str=None, instruct: str=None, output_mime_type: str=None,
			response_modalities: str=None, grounded: bool=False,
			image_search: bool=False ) -> Optional[ PIL.Image.Image ]:
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
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.content_config = self.get_content_config( image_only=True, grounded=grounded,
				image_search=image_search, response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type )
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt ], config=self.content_config )
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'generate( self, prompt, aspect ) -> Optional[ PIL.Image.Image ]'
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str='gemini-2.5-flash-image',
			aspect: str=None, number: int=None, temperature: float=None,
			top_p: float=None, frequency: float=None, presence: float=None,
			max_tokens: int=None, resolution: str=None, instruct: str=None,
			output_mime_type: str=None, response_modalities: str=None,
			grounded: bool=False, image_search: bool=False ) -> Optional[ str ]:
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
			self.response_mode = response_modalities or 'text'
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.content_config = self.get_content_config( image_only=False,
				grounded=grounded, image_search=image_search, response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type )
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.open_image( path ) ], config=self.content_config )
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'analyze( self, prompt, path, model ) -> Optional[ str ]'
			raise exception
	
	def edit( self, prompt: str, path: str, model: str='gemini-2.5-flash-image',
			aspect: str=None, number: int=None, temperature: float=None,
			top_p: float=None, frequency: float=None, presence: float=None,
			max_tokens: int=None, resolution: str=None, instruct: str=None,
			output_mime_type: str=None, response_modalities: str=None,
			grounded: bool=False, image_search: bool=False ) -> Optional[ PIL.Image.Image ]:
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
			self.response_mode = response_modalities or 'image'
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.content_config = self.get_content_config( image_only=True,
				grounded=grounded, image_search=image_search, response_modalities=self.response_mode,
				output_mime_type=self.output_mime_type )
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.open_image( path ) ], config=self.content_config )
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_first_image( )
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
		client              : Client - Initialized GenAI client.
		response            : Any - Raw API response.
		embedding           : List[ float ] | List[ List[ float ] ] - Generated vectors.
		encoding_format     : str - UI-selected embedding output format.
		dimensions          : int - Optional embedding output dimensionality.
		task_type           : str - Optional embedding task type.
		title               : str - Optional retrieval-document title.
		embedding_config    : EmbedContentConfig - Configuration for embeddings.
		contents            : str | List[ str ] - Input text or text chunks.
		input_text          : str | List[ str ] - Current text input.
		file_path           : str - Path to source text.

		Methods:
		--------
		create( text, model ) : Creates one or more embedding vectors.

	'''
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	embedding: Optional[ List[ float ] | List[ List[ float ] ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	task_type: Optional[ str ]
	title: Optional[ str ]
	embedding_config: Optional[ types.EmbedContentConfig ]
	contents: Optional[ str | List[ str ] ]
	input_text: Optional[ str | List[ str ] ]
	file_path: Optional[ str ]
	response_modalities: Optional[ str ]
	
	def __init__( self, model: str='gemini-embedding-001' ):
		super( ).__init__( )
		self.model = model
		self.client = None
		self.embedding = None
		self.embeddings = None
		self.response = None
		self.encoding_format = None
		self.input_text = None
		self.contents = None
		self.file_path = None
		self.dimensions = None
		self.task_type = None
		self.title = None
		self.response_modalities = None
		self.embedding_config = None
		self.content_config = None
		self.api_key = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Returns supported Gemini embedding model options.
			
			Returns:
			--------
			List[ str ] | None - Available embedding model names.
		
		"""
		return [ 'gemini-embedding-001',
		         'gemini-embedding-2',
		         'gemini-embedding-2-preview',
		         'text-embedding-004',
		         'text-multilingual-embedding-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		'''
			
			Purpose:
			--------
			Returns available embedding output format options retained for UI compatibility.
			
			Returns:
			--------
			List[ str ] - Available format options.

		'''
		return [ 'float', 'base64' ]
	
	@property
	def task_options( self ) -> List[ str ]:
		'''
			
			Purpose:
			--------
			Returns available embedding task options.
			
			Returns:
			--------
			List[ str ] - Available embedding task types.

		'''
		return [ '',
		         'RETRIEVAL_QUERY',
		         'RETRIEVAL_DOCUMENT',
		         'SEMANTIC_SIMILARITY',
		         'CLASSIFICATION',
		         'CLUSTERING',
		         'QUESTION_ANSWERING',
		         'FACT_VERIFICATION',
		         'CODE_RETRIEVAL_QUERY' ]
	
	def normalize_dimensions( self, dimensions: int ) -> int | None:
		"""
		
			Purpose:
			--------
			Normalizes the optional embedding output dimensionality.
			
			Parameters:
			-----------
			dimensions: int - Requested output dimensionality.
			
			Returns:
			--------
			int | None - Positive dimensionality or None.
		
		"""
		try:
			throw_if( 'dimensions', dimensions)
			self.dimensions = dimensions
			if self.dimensions <= 0:
				return None
			
			return self.dimensions
		except Exception:
			return None
	
	def normalize_contents( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""
		
			Purpose:
			--------
			Normalizes embedding input into either one text string or a list of text chunks.
			
			Parameters:
			-----------
			text: str | List[ str ] - Input text or text chunks.
			
			Returns:
			--------
			str | List[ str ] - Normalized embedding content.
		
		"""
		try:
			throw_if( 'text', text )
			
			if isinstance( text, list ):
				self.contents = [ ]
				for item in text:
					if item is None:
						continue
					
					self.item = str( item ).strip( )
					if self.item:
						self.contents.append( self.item )
				
				throw_if( 'text', self.contents )
				return self.contents
			
			self.contents = str( text ).strip( )
			throw_if( 'text', self.contents )
			return self.contents
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'normalize_contents( self, text: str | List[ str ] )'
			raise exception
	
	def build_embedding_config( self, model: str='gemini-embedding-001',
			dimensions: int=None, task_type: str=None, title: str=None ) -> EmbedContentConfig:
		"""
		
			Purpose:
			--------
			Builds the Gemini embedding configuration for the selected model.
			
			Parameters:
			-----------
			model: str - Gemini embedding model identifier.
			dimensions: int - Optional output dimensionality.
			task_type: str - Optional embedding task type.
			title: str - Optional retrieval-document title.
			
			Returns:
			--------
			EmbedContentConfig - Embedding configuration object.
		
		"""
		try:
			self.model = model
			self.dimensions = dimensions
			self.task_type = task_type.strip( ).upper( )
			self.title = title.strip( )
			self.config_kwargs = { }
			
			if self.dimensions is not None:
				self.config_kwargs[ 'output_dimensionality' ]=self.dimensions
			
			if self.task_type and 'gemini-embedding-2' not in self.model:
				self.config_kwargs[ 'task_type' ]=self.task_type
			
			if self.title and self.task_type == 'RETRIEVAL_DOCUMENT' \
					and 'gemini-embedding-2' not in self.model:
				self.config_kwargs[ 'title' ]=self.title
			
			self.embedding_config = EmbedContentConfig( **self.config_kwargs )
			return self.embedding_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'build_embedding_config( self, model, dimensions, task_type, title )'
			raise exception
	
	def extract_embeddings( self ) -> List[ float ] | List[ List[ float ] ] | None:
		"""
		
			Purpose:
			--------
			Extracts embedding vectors from a Gemini embed_content response.
			
			Returns:
			--------
			List[ float ] | List[ List[ float ] ] | None - One vector, multiple vectors, or None.
		
		"""
		try:
			if self.response is None:
				return None
			
			if not hasattr( self.response, 'embeddings' ):
				return None
			
			self.embeddings = [ ]
			for item in self.response.embeddings:
				if item is None:
					continue
				
				if hasattr( item, 'values' ) and item.values is not None:
					self.embeddings.append( list( item.values ) )
			
			if len( self.embeddings ) == 0:
				return None
			
			if len( self.embeddings ) == 1 and isinstance( self.input_text, str ):
				self.embedding = self.embeddings[ 0 ]
				return self.embedding
			
			self.embedding = self.embeddings
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'extract_embeddings( self )'
			raise exception
		
	def create( self, text: str | List[ str ], model: str='gemini-embedding-001',
			dimensions: int=None, task_type: str=None, title: str=None,
			encoding_format: str='float' ) -> List[ float ] | List[ List[ float ] ] | None:
		"""
			
			Purpose:
			--------
			Generates one or more vector representations of the provided text.
			
			Parameters:
			-----------
			text: str | List[ str ] - Input text string or chunk list.
			model: str - Embedding model identifier.
			dimensions: int - Optional output dimensionality.
			task_type: str - Optional embedding task type.
			title: str - Optional retrieval-document title.
			encoding_format: str - UI-retained encoding format value.
			
			Returns:
			--------
			List[ float ] | List[ List[ float ] ] | None - Embedding vector or vectors.
		
		"""
		try:
			throw_if( 'text', text )
			self.api_key = cfg.GEMINI_API_KEY
			throw_if( 'api_key', self.api_key )
			self.dimensions = dimensions
			self.task_type = task_type
			self.title = title
			self.encoding_format = encoding_format
			self.input_text = self.normalize_contents( text=text )
			self.model = model.strip( )
			self.encoding_format = encoding_format
			self.embedding_config = self.build_embedding_config( model=self.model,
				dimensions=self.imensions, task_type=self.task_type, title=self.title )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.models.embed_content( model=self.model,
				contents=self.input_text, config=self.embedding_config )
			
			return self.extract_embeddings( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'create( self, text, model ) -> List[ float ] | List[ List[ float ] ]'
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
	
	def __init__( self, model: str='gemini-2.5-flash-preview-tts' ):
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
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

			Returns:
			--------
			List[str] | None - Supported Gemini TTS model identifiers.

		"""
		return [ 'gemini-3.1-flash-tts-preview', 'gemini-2.5-flash-preview-tts',
				'gemini-2.5-pro-preview-tts' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns the supported output container formats for this wrapper.

			Returns:
			--------
			List[str] | None - Supported local output formats.

		"""
		return [ 'audio/wav' ]
	
	def to_wave_bytes( self, pcm_data: bytes, rate: int=24000, channels: int=1,
			sample_width: int=2 ) -> bytes:
		"""

			Purpose:
			--------
			Wraps raw PCM bytes returned by Gemini TTS into a WAV container.

			Parameters:
			-----------
			pcm_data: bytes - Raw PCM audio bytes.
			rate: int - Sample rate in Hz.
			channels: int - Number of audio channels.
			sample_width: int - Sample width in bytes.

			Returns:
			--------
			bytes - WAV file bytes.

		"""
		try:
			import io
			import wave
			throw_if( 'pcm_data', pcm_data )
			with io.BytesIO( ) as buffer:
				with wave.open( buffer, 'wb' ) as wf:
					wf.setnchannels( channels )
					wf.setsampwidth( sample_width )
					wf.setframerate( rate )
					wf.writeframes( pcm_data )
				
				return buffer.getvalue( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'to_wave_bytes( self, **kwargs) -> bytes'
			raise exception
	
	def normalize_voice( self, voice: Optional[ str ]=None ) -> str:
		"""

			Purpose:
			--------
			Normalizes the UI-selected voice into a valid Gemini prebuilt voice name.

			Parameters:
			-----------
			voice: Optional[str] - UI-selected voice name.

			Returns:
			--------
			str - Valid Gemini TTS voice name.

		"""
		try:
			self.voice_name = str( voice or '' ).strip( )
			self.valid_voices = set( self.voice_options or [ ] )
			if self.voice_name in self.valid_voices:
				return self.voice_name
			
			return 'Kore'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'normalize_voice( self, voice: Optional[str]=None ) -> str'
			raise exception
	
	def normalize_tts_prompt( self, text: str, speed: Optional[ float ]=None,
			instruct: Optional[ str ]=None ) -> str:
		"""

			Purpose:
			--------
			Builds a Gemini TTS prompt using natural-language delivery instructions instead
			of unsupported request parameters.

			Parameters:
			-----------
			text: str - Text to synthesize.
			speed: Optional[float] - Optional UI speed hint.
			instruct: Optional[str] - Optional user/system delivery instruction.

			Returns:
			--------
			str - Prompt text sent to the TTS model.

		"""
		try:
			throw_if( 'text', text )
			self.prompt_parts = [ ]
			
			if instruct is not None and str( instruct ).strip( ):
				self.prompt_parts.append( str( instruct ).strip( ) )
			
			if speed is not None:
				self.speed_value = float( speed )
				if self.speed_value < 0.85:
					self.prompt_parts.append( 'Read the following text at a slow, clear pace.' )
				elif self.speed_value > 1.15:
					self.prompt_parts.append(
						'Read the following text at a faster, energetic pace.' )
			
			self.prompt_parts.append( str( text ).strip( ) )
			return '\n\n'.join( self.prompt_parts )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'normalize_tts_prompt( self, **kwargs ) -> str'
			raise exception
	
	def create_speech( self, text: str, filepath: str = None,
			model: str = 'gemini-3.1-flash-tts-preview', format: str = 'audio/wav',
			speed: float = None, voice: str = None, frequency: float = None,
			presense: float = None, max_tokens: int = None, instruct: str = None,
			temperature: float = None, top_p: float = None ) -> bytes | str | None:
		"""

			Purpose:
			--------
			Converts text to speech using Gemini TTS. If filepath is provided, the generated
			WAV is written to disk; otherwise WAV bytes are returned.

			Parameters:
			-----------
			text: str - Input text string.
			filepath: str - Optional target local path.
			model: str - Gemini TTS model identifier.
			format: str - Output audio format.
			speed: float - Optional delivery pace hint converted into prompt text.
			voice: str - Gemini prebuilt voice name.
			frequency: float - UI-retained value; not sent to Gemini TTS.
			presense: float - UI-retained value; not sent to Gemini TTS.
			max_tokens: int - Maximum output token budget.
			instruct: str - Optional delivery/system instruction.
			temperature: float - Sampling temperature.
			top_p: float - Nucleus sampling threshold.

			Returns:
			--------
			bytes | str | None - WAV bytes or local path to the created file.

		"""
		try:
			throw_if( 'text', text )
			self.input_text = self.normalize_tts_prompt(
				text=text,
				speed=speed,
				instruct=instruct )
			self.audio_path = filepath
			self.response_format = str( format or 'audio/wav' ).strip( )
			self.speed = speed
			self.voice = self.normalize_voice( voice )
			self.frequency_penalty = frequency
			self.presence_penalty = presense
			self.max_tokens = max_tokens
			self.model = str( model or self.model or 'gemini-3.1-flash-tts-preview' ).strip( )
			self.temperature = temperature
			self.top_p = top_p
			self.response_modalities = [ 'AUDIO' ]
			
			if self.response_format != 'audio/wav':
				raise ValueError( 'Gemini TTS wrapper currently supports local WAV output only.' )
			
			if self.model not in self.model_options:
				raise ValueError( f'Unsupported Gemini TTS model: {self.model}' )
			
			self.voice_config = VoiceConfig(
				prebuilt_voice_config=types.PrebuiltVoiceConfig(
					voice_name=self.voice ) )
			self.speech_config = SpeechConfig( voice_config=self.voice_config )
			self.config_kwargs = {
					'response_modalities': self.response_modalities,
					'speech_config': self.speech_config
			}
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None and int( self.max_tokens or 0 ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( self.max_tokens )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=self.input_text,
				config=self.content_config )
			
			self.audio_bytes = None
			for part in self.response.candidates[ 0 ].content.parts:
				if getattr( part, 'inline_data', None ) is not None and part.inline_data.data:
					self.audio_bytes = self.to_wave_bytes( part.inline_data.data )
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
			exception.method = (
					'create_speech( self, text: str, filepath: str=None, '
					'model: str="gemini-3.1-flash-tts-preview", format: str="audio/wav", '
					'speed: float=None, voice: str=None, frequency: float=None, '
					'presense: float=None, max_tokens: int=None, instruct: str=None, '
					'temperature: float=None, top_p: float=None ) -> bytes | str | None')
			error = ErrorDialog( exception )
			error.show( )
			return None

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
	
	def __init__( self, n: int=1, model: str='gemini-3-flash-preview', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0,
			max_tokens: int=10000, instruct: str=None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
		self.transcript = None
		self.file_path = None
		self.response = None
		self.content_config = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns list of llm supporting audio input.

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
			Returns supported Gemini audio input MIME types.

			Returns:
			--------
			List[str] | None - Supported audio MIME types.

		"""
		return [
				'audio/wav',
				'audio/mp3',
				'audio/aiff',
				'audio/aac',
				'audio/ogg',
				'audio/flac'
		]
	
	def normalize_mime_type( self, path: str, mime_type: str = None ) -> str:
		"""

			Purpose:
			--------
			Normalizes UI-provided or filename-derived audio MIME types into Gemini-supported
			audio MIME types.

			Parameters:
			-----------
			path: str - Local audio file path.
			mime_type: str - Optional UI-selected MIME type.

			Returns:
			--------
			str - Gemini-supported audio MIME type.

		"""
		try:
			import mimetypes
			self.raw_mime_type = str( mime_type or '' ).strip( )
			if not self.raw_mime_type:
				self.raw_mime_type = mimetypes.guess_type( path )[ 0 ] or ''
			
			self.mime_aliases = {
					'audio/mpeg': 'audio/mp3',
					'audio/x-mp3': 'audio/mp3',
					'audio/x-wav': 'audio/wav',
					'audio/wave': 'audio/wav',
					'audio/x-m4a': 'audio/aac',
					'audio/m4a': 'audio/aac',
					'audio/mp4': 'audio/aac',
					'audio/x-aiff': 'audio/aiff',
					'audio/aif': 'audio/aiff',
					'audio/x-flac': 'audio/flac'
			}
			self.mime_type = self.mime_aliases.get( self.raw_mime_type, self.raw_mime_type )
			
			if self.mime_type in self.format_options:
				return self.mime_type
			
			self.suffix = str( Path( path ).suffix or '' ).strip( ).lower( )
			self.extension_map = {
					'.wav': 'audio/wav',
					'.mp3': 'audio/mp3',
					'.aiff': 'audio/aiff',
					'.aif': 'audio/aiff',
					'.aac': 'audio/aac',
					'.m4a': 'audio/aac',
					'.ogg': 'audio/ogg',
					'.flac': 'audio/flac'
			}
			
			if self.suffix in self.extension_map:
				return self.extension_map[ self.suffix ]
			
			return 'audio/wav'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = 'normalize_mime_type( self, path: str, mime_type: str=None ) -> str'
			raise exception
	
	def build_prompt( self, language: str=None, start_time: float=None,
			end_time: float=None ) -> str:
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
	
	def transcribe( self, path: str, model: str='gemini-3-flash-preview',
			language: str=None, mime_type: str=None, temperature: float=None,
			top_p: float=None, frequency: float=None, presence: float=None,
			max_tokens: int=None, start_time: float=None, end_time: float=None,
			instruct: str=None ) -> Optional[ str ]:
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
			self.mime_type = self.normalize_mime_type( path=self.file_path, mime_type=mime_type )
			self.prompt = self.build_prompt( language=language, start_time=start_time,
				end_time=end_time )
			
			self.config_kwargs = { }
			
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ]=self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ]=self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ]=self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ]=str( self.instructions ).strip( )
			
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
	
	def __init__( self, n: int=1, model: str='gemini-3-flash-preview', temperature: float=0.8,
			top_p: float=0.9, frequency: float=0.0, presence: float=0.0,
			max_tokens: int=10000,
			instruct: str=None ):
		super( ).__init__( )
		self.number = n
		self.model = model
		self.temperature = temperature
		self.top_p = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.instructions = instruct
		self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
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
			Returns list of translation-capable audio llm.

		"""
		return [ 'gemini-3-flash-preview',
		         'gemini-2.0-flash' ]
		
	@property
	def format_options( self ) -> List[ str ] | None:
		"""

			Purpose:
			--------
			Returns supported Gemini audio input MIME types.

			Returns:
			--------
			List[str] | None - Supported audio MIME types.

		"""
		return [
				'audio/wav',
				'audio/mp3',
				'audio/aiff',
				'audio/aac',
				'audio/ogg',
				'audio/flac'
		]
	
	def normalize_mime_type( self, path: str, mime_type: str = None ) -> str:
		"""

			Purpose:
			--------
			Normalizes UI-provided or filename-derived audio MIME types into Gemini-supported
			audio MIME types.

			Parameters:
			-----------
			path: str - Local audio file path.
			mime_type: str - Optional UI-selected MIME type.

			Returns:
			--------
			str - Gemini-supported audio MIME type.

		"""
		try:
			import mimetypes
			
			self.raw_mime_type = str( mime_type or '' ).strip( )
			if not self.raw_mime_type:
				self.raw_mime_type = mimetypes.guess_type( path )[ 0 ] or ''
			
			self.mime_aliases = {
					'audio/mpeg': 'audio/mp3',
					'audio/x-mp3': 'audio/mp3',
					'audio/x-wav': 'audio/wav',
					'audio/wave': 'audio/wav',
					'audio/x-m4a': 'audio/aac',
					'audio/m4a': 'audio/aac',
					'audio/mp4': 'audio/aac',
					'audio/x-aiff': 'audio/aiff',
					'audio/aif': 'audio/aiff',
					'audio/x-flac': 'audio/flac'
			}
			self.mime_type = self.mime_aliases.get( self.raw_mime_type, self.raw_mime_type )
			
			if self.mime_type in self.format_options:
				return self.mime_type
			
			self.suffix = str( Path( path ).suffix or '' ).strip( ).lower( )
			self.extension_map = {
					'.wav': 'audio/wav',
					'.mp3': 'audio/mp3',
					'.aiff': 'audio/aiff',
					'.aif': 'audio/aiff',
					'.aac': 'audio/aac',
					'.m4a': 'audio/aac',
					'.ogg': 'audio/ogg',
					'.flac': 'audio/flac'
			}
			
			if self.suffix in self.extension_map:
				return self.extension_map[ self.suffix ]
			
			return 'audio/wav'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = 'normalize_mime_type( self, path: str, mime_type: str=None ) -> str'
			raise exception
	
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
	
	def build_prompt( self, target: str, source: str='Auto', start_time: float=None,
			end_time: float=None ) -> str:
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
	
	def translate( self, path: str, model: str='gemini-3-flash-preview',
			language: str='English', source: str='Auto', mime_type: str=None,
			temperature: float=None, top_p: float=None, frequency: float=None,
			presence: float=None, max_tokens: int=None, start_time: float=None,
			end_time: float=None, instruct: str=None ) -> Optional[ str ]:
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
			self.mime_type = self.normalize_mime_type( path=self.file_path, mime_type=mime_type )
			self.prompt = self.build_prompt( target=self.target_language, source=self.source_language,
				start_time=start_time, end_time=end_time )
			
			self.config_kwargs = { }
			if self.temperature is not None:
				self.config_kwargs[ 'temperature' ]=self.temperature
			
			if self.top_p is not None:
				self.config_kwargs[ 'top_p' ]=self.top_p
			
			if self.max_tokens is not None:
				self.config_kwargs[ 'max_output_tokens' ]=self.max_tokens
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.config_kwargs[ 'system_instruction' ]=str( self.instructions ).strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			self.response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.uploaded_file ], config=self.content_config )
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
	
	def __init__( self, model: str='gemini-2.0-flash' ):
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
		self.files = [ ]
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
			Returns list of available chat llm.
			
		"""
		return self.files
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""
			
			Purpose:
			--------
			Returns list of available chat llm.
			
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
	
	def upload( self, filepath: str, name: str=None ) -> File | None:
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
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.files.upload( path=self.file_path,
				config={ 'display_name': self.display_name } )
			return self.response
		except Exception as e:
			ex = Error( e );
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'upload( self, path: str, name: str ) -> Optional[ File ]'
			raise ex
		
	def list( self, model: str = 'gemini-3.0-flash', top_p: float = 0.8, top_k: int = 50,
			temperature: float = 0.5, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 8192, tool_choice: str = 'auto', stops: List[ str ] = None,
			tools: List[ str ] = None, domains: List[ str ] = None,
			modalities: List[ str ] = None,
			media_resolution: str = 'media_resolution_medium' ) -> Any | None:
		"""
		
			Purpose:
			--------
			Lists files from the existing Google Cloud Storage-backed file listing path.
			
			Parameters:
			-----------
			model: str - Gemini model identifier retained for UI compatibility.
			top_p: float - Nucleus sampling value retained for UI compatibility.
			top_k: int - Top-k token selection count retained for UI compatibility.
			temperature: float - Sampling temperature retained for UI compatibility.
			frequency: float - Frequency penalty retained for UI compatibility.
			presence: float - Presence penalty retained for UI compatibility.
			max_tokens: int - Maximum output tokens retained for UI compatibility.
			tool_choice: str - Tool-choice value retained for UI compatibility.
			stops: List[ str ] - Stop sequences retained for UI compatibility.
			tools: List[ str ] - Tool names retained for UI compatibility.
			domains: List[ str ] - Domain filters retained for UI compatibility.
			modalities: List[ str ] - Modalities retained for UI compatibility.
			media_resolution: str - Media resolution retained for UI compatibility.
			
			Returns:
			--------
			Any | None - List of file names or None.
			
		"""
		try:
			self.files = [ ]
			self.model = model
			self.top_p = top_p
			self.top_k = top_k
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.tool_choice = tool_choice
			self.stops = stops if stops is not None else [ ]
			self.tools = tools if tools is not None else [ ]
			self.domains = domains if domains is not None else [ ]
			self.response_modalities = modalities if modalities is not None else [ ]
			self.media_resolution = media_resolution
			
			self.storage_client = storage.Client( )
			name = 'jeni-financial'
			prefix = 'regulations'
			bucket = self.storage_client.bucket( bucket_name=name )
			
			for blob in bucket.list_blobs( prefix=prefix ):
				self.files.append( blob.name )
			
			self.file_list = self.files
			return self.files
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'list( self ) -> Any | None'
			raise ex
		
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
	
	def summarize( self, prompt: str, filepath: str, model: str='gemini-2.0-flash',
			temperature: float=None, top_p: float=None, frequency: float=None,
			presence: float=None, max_tokens: int=None, stops: List[ str ]=None,
			instruct: str=None ) -> str | None:
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
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
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
	
	def search( self, prompt: str, filepath: str, model: str='gemini-2.0-flash',
			temperature: float=None, top_p: float=None, frequency: float=None,
			presence: float=None, max_tokens: int=None, stops: List[ str ]=None,
			instruct: str=None ) -> str | None:
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
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
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
	
	def survey( self, prompt: str, filepaths: List[ str ], model: str='gemini-2.0-flash',
			temperature: float=None, top_p: float=None, frequency: float=None,
			presence: float=None, max_tokens: int=None,
			stops: List[ str ]=None ) -> str | None:
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
			self.top_p = top_p
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops
			self.content_config = GenerateContentConfig( temperature=self.temperature )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
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
	
	def web_search( self, prompt: str, model: str='gemini-2.5-flash-lite',
			temperature: float=None, top_p: float=None, frequency: float=None, presence: float=None,
			max_tokens: int=None, stops: List[ str ]=None, instruct: str=None ) -> str | None:
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
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
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
	
	def search_maps( self, prompt: str, model: str='gemini-2.5-flash-lite',
			temperature: float=None, top_p: float=None, frequency: float=None,
			presence: float=None,
			max_tokens: int=None, stops: List[ str ]=None, instruct: str=None ) -> str | None:
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
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
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
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.client.files.delete( name=self.file_id )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'FileStore'
			ex.method = 'delete( self, file_id: str ) -> bool'
			raise ex

class FileSearch( Gemini ):
	"""

		Purpose:
		--------
		Encapsulate Gemini File Search Store management for the Jeni application.

		Attributes:
		-----------
		client       : genai.Client | None
		response     : Any
		store_id     : str | None
		store_name   : str | None
		collections  : Dict[ str, str ]
		stores       : List[ FileSearchStore ]

		Methods:
		--------
		create( name )
		retrieve( store_id )
		list( )
		delete( store_id )

	"""
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	store_id: Optional[ str ]
	store_name: Optional[ str ]
	collections: Optional[ Dict[ str, str ] ]
	stores: Optional[ List[ FileSearchStore ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.client = None
		self.response = None
		self.store_id = None
		self.store_name = None
		self.collections = { }
		self.stores = [ ]
		self.refresh_collections( )
	
	def refresh_collections( self ) -> Dict[ str, str ]:
		"""

			Purpose:
			--------
			Refresh the in-memory mapping of display names to File Search Store resource names.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[ str, str ]

		"""
		try:
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.collections = { }
			self.stores = [ ]
			for store in self.client.file_search_stores.list( ):
				self.stores.append( store )
				self.display_name = getattr( store, 'display_name', None )
				self.resource_name = getattr( store, 'name', None )
				
				if self.resource_name is None:
					continue
				
				self.label = str( self.display_name ).strip( ) if self.display_name else str(
					self.resource_name ).strip( )
				self.collections[ self.label ]=str( self.resource_name ).strip( )
			
			return self.collections
		except Exception:
			self.collections = { }
			self.stores = [ ]
			return self.collections
	
	def create( self, name: str ) -> FileSearchStore | Any:
		"""

			Purpose:
			--------
			Create a new Gemini File Search Store.

			Parameters:
			-----------
			name: str
				Display name for the File Search Store.

			Returns:
			--------
			FileSearchStore | Any

		"""
		try:
			throw_if( 'name', name )
			self.store_name = str( name ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.file_search_stores.create( config={ 'display_name': self.store_name } )
			self.refresh_collections( )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'create( self, name: str ) -> FileSearchStore | Any'
			raise exception
	
	def retrieve( self, store_id: str ) -> FileSearchStore | Any:
		"""

			Purpose:
			--------
			Retrieve a specific Gemini File Search Store by resource name.

			Parameters:
			-----------
			store_id: str
				Resource name in the form fileSearchStores/<id>.

			Returns:
			--------
			FileSearchStore | Any

		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.file_search_stores.get( name=self.store_id )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'retrieve( self, store_id: str ) -> FileSearchStore | Any'
			raise exception
	
	def list( self ) -> List[ FileSearchStore ] | Any:
		"""

			Purpose:
			--------
			List all Gemini File Search Stores available to the current project.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ FileSearchStore ] | Any

		"""
		try:
			self.refresh_collections( )
			return self.stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'list( self ) -> List[ FileSearchStore ] | Any'
			raise exception
	
	def delete( self, store_id: str, force: bool=True ) -> bool | Any:
		"""

			Purpose:
			--------
			Delete a Gemini File Search Store.

			Parameters:
			-----------
			store_id: str
				Resource name in the form fileSearchStores/<id>.
			force: bool
				If True, delete contained documents and related objects as well.

			Returns:
			--------
			bool | Any

		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.client.file_search_stores.delete( name=self.store_id,
				config={ 'force': bool( force ) } )
			self.refresh_collections( )
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'delete( self, store_id: str, force: bool=True ) -> bool | Any'
			raise exception
		
class CloudBuckets( Gemini ):
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
					'DoW Financial Data': 'jeni-dow/budget/data',
					'DoW Financial Regulations': 'jeni-dow/budget/regulations',
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
		"""Returns list of available chat llm."""
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
	
	def upload( self, path: str, bucket: str, name: str=None ):
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
	
	def web_search( self, prompt: str, model: str='gemini-2.5-flash-lite',
			temperature: float=None, top_p: float=None, frequency: float=None, presence: float=None,
			max_tokens: int=None, stops: List[ str ]=None, instruct: str=None ) -> str | None:
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
			self.tool_config = [ types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config, system_instruction=self.instructions )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
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
	
	def search_maps( self, prompt: str, model: str='gemini-2.5-flash-lite',
			temperature: float=None, top_p: float=None, frequency: float=None, presence: float=None,
			max_tokens: int=None, stops: List[ str ]=None, instruct: str=None ) -> str | None:
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
			self.tool_config = [ types.Tool( google_search_retrieval=types.GoogleSearchRetrieval( ) ) ]
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=self.tool_config )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
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