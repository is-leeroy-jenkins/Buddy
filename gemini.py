"""Gemini provider integration for Buddy.

Purpose:
    Provides Google Gemini, Google GenAI, file-search, cloud-storage, embedding, image,
    text, audio, transcription, translation, and file-management wrappers used by the
    Buddy Streamlit application and its MkDocs API reference.
"""
from google.genai.file_search_stores import FileSearchStores
import config as cfg
import base64
from boogr import ErrorDialog, Error, Logger
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
                                FileSearchStore, FileSearch,
                                GenerateContentResponse, GenerateVideosResponse, Image, File,
                                SpeakerVoiceConfig, VoiceConfig, SpeechConfig, Tool, ToolConfig,
                                GoogleSearch, UrlContext, SafetySetting, HarmCategory,
                                HarmBlockThreshold)

def throw_if( name: str, value: object ) -> None:
	"""Throw if.
	
	Purpose:
	    Validates required values before provider request construction.
	
	Args:
	    name (str): Name supplied to the Gemini workflow.
	    value (object): Value supplied to the Gemini workflow.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encode image.
	
	Purpose:
	    Encodes local binary content into a text representation required by Gemini request payloads.
	
	Args:
	    image_path (str): Image path supplied to the Gemini workflow.
	
	Returns:
	    str: Result produced by the Gemini workflow.
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class Gemini( ):
	"""Gemini workflow wrapper.
	
	Purpose:
	    Provides shared Gemini configuration state, API-key storage, generation defaults, and option containers used by provider-specific capability wrappers.
	
	Attributes:
	    number: Runtime attribute used by the Gemini workflow.
	    google_api_key: Runtime attribute used by the Gemini workflow.
	    gemini_api_key: Runtime attribute used by the Gemini workflow.
	    instructions: Runtime attribute used by the Gemini workflow.
	    prompt: Runtime attribute used by the Gemini workflow.
	    model: Runtime attribute used by the Gemini workflow.
	    api_version: Runtime attribute used by the Gemini workflow.
	    max_tokens: Runtime attribute used by the Gemini workflow.
	    temperature: Runtime attribute used by the Gemini workflow.
	    top_p: Runtime attribute used by the Gemini workflow.
	    top_k: Runtime attribute used by the Gemini workflow.
	    candidate_count: Runtime attribute used by the Gemini workflow.
	    media_resolution: Runtime attribute used by the Gemini workflow.
	    response_modalities: Runtime attribute used by the Gemini workflow.
	    stops: Runtime attribute used by the Gemini workflow.
	    domains: Runtime attribute used by the Gemini workflow.
	    frequency_penalty: Runtime attribute used by the Gemini workflow.
	    presence_penalty: Runtime attribute used by the Gemini workflow.
	    response_format: Runtime attribute used by the Gemini workflow.
	    content_response: Runtime attribute used by the Gemini workflow.
	    image_response: Runtime attribute used by the Gemini workflow.
	    content_config: Runtime attribute used by the Gemini workflow.
	    function_config: Runtime attribute used by the Gemini workflow.
	    thought_config: Runtime attribute used by the Gemini workflow.
	    genimg_config: Runtime attribute used by the Gemini workflow.
	    image_config: Runtime attribute used by the Gemini workflow.
	    tool_config: Runtime attribute used by the Gemini workflow.
	    tool_choice: Runtime attribute used by the Gemini workflow.
	    tools: Runtime attribute used by the Gemini workflow.
	"""
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
		"""Initialize instance.
		
		Purpose:
		    Initializes Gemini state with default configuration values and runtime attributes used by later Gemini provider calls.
		"""
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
	"""Chat workflow wrapper.
	
	Purpose:
	    Builds and executes Gemini text, multimodal, grounded-search, URL-context, file-search, and structured-output workflows for the Streamlit application.
	
	Attributes:
	    use_vertex: Runtime attribute used by the Chat workflow.
	    http_options: Runtime attribute used by the Chat workflow.
	    client: Runtime attribute used by the Chat workflow.
	    storage_client: Runtime attribute used by the Chat workflow.
	    contents: Runtime attribute used by the Chat workflow.
	    image_uri: Runtime attribute used by the Chat workflow.
	    audio_uri: Runtime attribute used by the Chat workflow.
	    file_path: Runtime attribute used by the Chat workflow.
	    files: Runtime attribute used by the Chat workflow.
	    content_block: Runtime attribute used by the Chat workflow.
	    context: Runtime attribute used by the Chat workflow.
	    urls: Runtime attribute used by the Chat workflow.
	    max_urls: Runtime attribute used by the Chat workflow.
	    response_schema: Runtime attribute used by the Chat workflow.
	    safety_profile: Runtime attribute used by the Chat workflow.
	    safety_settings: Runtime attribute used by the Chat workflow.
	"""
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
		"""Initialize instance.
		
		Purpose:
		    Initializes Chat state with default configuration values and runtime attributes used by later Gemini provider calls.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		"""
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
		"""Model options.
		
		Purpose:
		    Returns the Gemini model names exposed by the related Streamlit selector without mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
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
		"""Tool options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'google_search',
		         'google_maps',
		         'url_context',
		         'file_search',
		         'code_execution' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL',
		         'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def media_options( self ):
		"""Media options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector.
		
		Returns:
		    list[str] | None: Option values exposed to the application UI.
		"""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'auto', 'any', 'none', 'validated' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ '', 'text', 'image', 'audio' ]
	
	@property
	def format_options( self ):
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector.
		
		Returns:
		    list[str] | None: Option values exposed to the application UI.
		"""
		return [ 'text/plain',
		         'application/json',
		         'text/x.enum' ]
	
	def get_supported_tools( self, model: str ) -> List[ str ]:
		"""Get supported tools.
		
		Purpose:
		    Retrieves normalized Gemini provider state or response data for display, reuse, or downstream request construction.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		
		Returns:
		    List[str]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def supports_google_maps( self, model: str ) -> bool:
		"""Supports google maps.
		
		Purpose:
		    Determines whether the selected Gemini model supports a provider-specific feature.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		
		Returns:
		    bool: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_urls( self, urls: List[ str ], max_urls: int = 10 ) -> List[ str ]:
		"""Build urls.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    urls (List[str]): Urls supplied to the Gemini workflow.
		    max_urls (int): Max urls supplied to the Gemini workflow.
		
		Returns:
		    List[str]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			self.max_urls = int( max_urls or 0 )
			self.source_urls = urls if isinstance( urls, list ) else [ ]
			self.urls = [ ]
			
			for url in self.source_urls:
				if url is None:
					continue
				
				self.url = str( url ).strip( )
				if not self.url:
					continue
				
				self.urls.append( self.url )
			
			if self.max_urls > 0:
				self.urls = self.urls[ : self.max_urls ]
			
			return self.urls
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_urls( self, urls: List[ str ], max_urls: int=10 )'
			Logger( ).write( exception )
			raise exception
	
	def append_urls_to_content( self, content: str, urls: List[ str ] ) -> str | None:
		"""Append urls to content.
		
		Purpose:
		    Appends optional context values to request content while preserving existing prompt text.
		
		Args:
		    content (str): Content supplied to the Gemini workflow.
		    urls (List[str]): Urls supplied to the Gemini workflow.
		
		Returns:
		    str | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			self.content_blocks = [ ]
			self.urls = urls if isinstance( urls, list ) else [ ]
			
			if isinstance( content, str ) and content.strip( ):
				self.content_blocks.append( content.strip( ) )
			elif isinstance( content, list ) and len( content ) > 0:
				self.content_text = '\n'.join(
					str( item ).strip( )
					for item in content
					if item is not None and str( item ).strip( )
				)
				
				if self.content_text:
					self.content_blocks.append( self.content_text )
			elif content is not None and str( content ).strip( ):
				self.content_blocks.append( str( content ).strip( ) )
			
			if len( self.urls ) > 0:
				self.content_blocks.append( 'Reference URLs:\n' + '\n'.join( self.urls ) )
			
			return '\n\n'.join( self.content_blocks ) if len( self.content_blocks ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'append_urls_to_content( self, content: str, urls: List[ str ] )'
			Logger( ).write( exception )
			raise exception
	
	def build_tool_config( self, tool_choice: str = None,
			tools: List[ Tool ] = None ) -> ToolConfig | None:
		"""Build tool config.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    tool_choice (str): Tool choice supplied to the Gemini workflow.
		    tools (List[Tool]): Tools supplied to the Gemini workflow.
		
		Returns:
		    ToolConfig | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_modalities( self, modalities: List[ str ] ) -> List[ str ] | None:
		"""Build modalities.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    modalities (List[str]): Modalities supplied to the Gemini workflow.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_reasoning( self, reasoning: str ) -> ThinkingConfig | None:
		"""Build reasoning.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    reasoning (str): Reasoning supplied to the Gemini workflow.
		
		Returns:
		    ThinkingConfig | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_safety_settings( self, safety_profile: str ) -> List[ SafetySetting ] | None:
		"""Build safety settings.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    safety_profile (str): Safety profile supplied to the Gemini workflow.
		
		Returns:
		    List[SafetySetting] | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> Optional[ str ]:
		"""Get output text.
		
		Purpose:
		    Retrieves normalized Gemini provider state or response data for display, reuse, or downstream request construction.
		
		Returns:
		    Optional[str]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def parse_response_schema( self, response_schema: Any ) -> Any:
		"""Parse response schema.
		
		Purpose:
		    Parses structured input into the format expected by Gemini request construction.
		
		Args:
		    response_schema (Any): Response schema supplied to the Gemini workflow.
		
		Returns:
		    Any: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_contents( self, prompt: str, content: str, context: List[ Any ] = None ) -> str | \
	                                                                                      List[
		                                                                                      Content ]:
		"""Build contents.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    prompt (str): Prompt supplied to the Gemini workflow.
		    content (str): Content supplied to the Gemini workflow.
		    context (List[Any]): Context supplied to the Gemini workflow.
		
		Returns:
		    str | List[Content]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def capture_grounding_metadata( self ) -> None:
		"""Capture grounding metadata.
		
		Purpose:
		    Captures response metadata from the most recent Gemini provider response and stores it for later source extraction.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def get_grounding_sources( self ) -> List[ Dict[ str, str ] ]:
		"""Get grounding sources.
		
		Purpose:
		    Retrieves normalized Gemini provider state or response data for display, reuse, or downstream request construction.
		
		Returns:
		    List[Dict[str, str]]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def get_structured_history( self ) -> List[ Content ] | None:
		"""Get structured history.
		
		Purpose:
		    Retrieves normalized Gemini provider state or response data for display, reuse, or downstream request construction.
		
		Returns:
		    List[Content] | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: List[ str ] = None,
			file_search_store_names: List[ str ] = None ) -> List[ Tool ] | None:
		"""Build tools.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    tools (List[str]): Tools supplied to the Gemini workflow.
		    file_search_store_names (List[str]): File search store names supplied to the Gemini workflow.
		
		Returns:
		    List[Tool] | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		
		try:
			self.tools = [
					str( tool ).strip( )
					for tool in (tools or [ ])
					if tool is not None and str( tool ).strip( )
			]
			
			self.file_search_store_names = [
					str( name ).strip( )
					for name in (file_search_store_names or [ ])
					if name is not None and str( name ).strip( )
			]
			
			if len( self.file_search_store_names ) > 0:
				return [
						Tool(
							file_search=types.FileSearch(
								file_search_store_names=self.file_search_store_names ) )
				]
			
			if 'google_search' not in self.tools:
				return None
			
			return [ Tool( google_search=GoogleSearch( ) ) ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, tools, file_search_store_names )'
			Logger( ).write( exception )
			raise exception
	
	def build_config( self, model: str = 'gemini-2.5-flash-lite', number: int = None,
			temperature: float = None, top_p: float = None, top_k: int = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None, response_format: str = None,
			tools: List[ str ] = None, tool_choice: str = None, reasoning: str = None,
			modalities: List[ str ] = None, media_resolution: str = None,
			response_schema: Any = None, safety_profile: str = None,
			file_search_store_names: List[ str ] = None ) -> GenerateContentConfig:
		"""Build config.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		    number (int): Number supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    top_k (int): Top k supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    stops (List[str]): Stops supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		    response_format (str): Response format supplied to the Gemini workflow.
		    tools (List[str]): Tools supplied to the Gemini workflow.
		    tool_choice (str): Tool choice supplied to the Gemini workflow.
		    reasoning (str): Reasoning supplied to the Gemini workflow.
		    modalities (List[str]): Modalities supplied to the Gemini workflow.
		    media_resolution (str): Media resolution supplied to the Gemini workflow.
		    response_schema (Any): Response schema supplied to the Gemini workflow.
		    safety_profile (str): Safety profile supplied to the Gemini workflow.
		    file_search_store_names (List[str]): File search store names supplied to the Gemini workflow.
		
		Returns:
		    GenerateContentConfig: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			self.file_search_store_names = [
					str( name ).strip( )
					for name in (file_search_store_names or [ ])
					if name is not None and str( name ).strip( )
			]
			self.response_mime_type = str( response_format or '' ).strip( )
			self.response_schema = self.parse_response_schema( response_schema )
			self.safety_settings = self.build_safety_settings( safety_profile )
			self.tool_choice = tool_choice
			self.media_resolution = str( media_resolution ).strip( ) if media_resolution else None
			self.tool_objects = self.build_tools(
				tools=tools,
				file_search_store_names=self.file_search_store_names )
			self.function_tool_config = self.build_tool_config(
				tool_choice=self.tool_choice,
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
			
			if self.function_tool_config is not None and len( self.file_search_store_names ) == 0:
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
			Logger( ).write( exception )
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
		"""Generate text.
		
		Purpose:
		    Executes a Gemini generation workflow using validated request settings, captures the provider response, and returns displayable output.
		
		Args:
		    prompt (str): Prompt supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    number (int): Number supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    top_k (int): Top k supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    stops (List[str]): Stops supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		    response_format (str): Response format supplied to the Gemini workflow.
		    tools (List[str]): Tools supplied to the Gemini workflow.
		    tool_choice (str): Tool choice supplied to the Gemini workflow.
		    reasoning (str): Reasoning supplied to the Gemini workflow.
		    modalities (List[str]): Modalities supplied to the Gemini workflow.
		    media_resolution (str): Media resolution supplied to the Gemini workflow.
		    context (List[Dict[str, Any]]): Context supplied to the Gemini workflow.
		    content (str): Content supplied to the Gemini workflow.
		    urls (List[str]): Urls supplied to the Gemini workflow.
		    max_urls (int): Max urls supplied to the Gemini workflow.
		    response_schema (Any): Response schema supplied to the Gemini workflow.
		    safety_profile (str): Safety profile supplied to the Gemini workflow.
		    file_search_store_names (List[str]): File search store names supplied to the Gemini workflow.
		    stream (bool): Stream supplied to the Gemini workflow.
		    stream_handler (Any): Stream handler supplied to the Gemini workflow.
		
		Returns:
		    str | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception

class Images( Gemini ):
	"""Images workflow wrapper.
	
	Purpose:
	    Builds and executes Gemini image-generation and image-analysis workflows while preserving selected model, prompt, and image configuration state.
	
	Attributes:
	    client: Runtime attribute used by the Images workflow.
	    aspect_ratio: Runtime attribute used by the Images workflow.
	    use_vertex: Runtime attribute used by the Images workflow.
	    resolution: Runtime attribute used by the Images workflow.
	    size: Runtime attribute used by the Images workflow.
	"""
	client: Optional[ genai.Client ]
	aspect_ratio: Optional[ str ]
	use_vertex: Optional[ bool ]
	resolution: Optional[ str ]
	size: Optional[ str ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-image' ):
		"""Initialize instance.
		
		Purpose:
		    Initializes Images state with default configuration values and runtime attributes used by later Gemini provider calls.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		"""
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
		"""Model options.
		
		Purpose:
		    Returns the Gemini model names exposed by the related Streamlit selector without mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'gemini-2.5-flash-image',
		         'gemini-3.1-flash-image-preview' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'file_search_call.results',
		         'message.input_image.image_url',
		         'message.output_text.logprobs',
		         'reasoning.encrypted_content' ]
	
	@property
	def aspect_options( self ) -> List[ str ] | None:
		"""Aspect options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9' ]
	
	@property
	def media_options( self ):
		"""Media options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    list[str] | None: Option values exposed to the application UI.
		"""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'text', 'image', 'text_and_image' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'unspecified', 'minimal',
		         'low', 'medium', 'high' ]
	
	@property
	def size_options( self ):
		"""Size options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    list[str] | None: Option values exposed to the application UI.
		"""
		return [ '1K', '2K', '4K' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'google_search', 'image_search' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'auto', 'any', 'none', 'validated' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'text/plain',
		         'application/json',
		         'text/x.enum' ]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'image/jpeg',
		         'image/png',
		         'image/webp' ]
	
	@property
	def resolution_options( self ) -> List[ str ] | None:
		"""Resolution options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ '1K', '2K', '4K' ]
	
	def supports_image_size( self, model: str = 'gemini-2.5-flash-image' ) -> bool:
		"""Supports image size.
		
		Purpose:
		    Determines whether the selected Gemini model supports a provider-specific feature.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		
		Returns:
		    bool: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			self.image_size_models = [ 'gemini-3.1-flash-image-preview',
			                           'gemini-3-pro-image-preview', ]
			
			return self.model_name in self.image_size_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_size( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_search_grounding( self, model: str = 'gemini-2.5-flash-image' ) -> bool:
		"""Supports search grounding.
		
		Purpose:
		    Determines whether the selected Gemini model supports a provider-specific feature.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		
		Returns:
		    bool: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def supports_image_search( self, model: str = 'gemini-2.5-flash-image' ) -> bool:
		"""Supports image search.
		
		Purpose:
		    Determines whether the selected Gemini model supports a provider-specific feature.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		
		Returns:
		    bool: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			self.model_name = str( model or '' ).strip( ).lower( )
			return self.model_name == 'gemini-3.1-flash-image-preview'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_search( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def normalize_image_size( self, resolution: str = None,
			model: str = 'gemini-2.5-flash-image' ) -> str:
		"""Normalize image size.
		
		Purpose:
		    Provides normalize image size behavior for the Images workflow while preserving provider request and response state.
		
		Args:
		    resolution (str): Resolution supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			if not self.supports_image_size( model ):
				return None
			
			self.resolution_value = str( resolution or '' ).strip( )
			if not self.resolution_value:
				return None
			
			self.resolution_map = {
					'media_resolution_low': '512',
					'media_resolution_medium': '1K',
					'media_resolution_high': '2K',
					'low': '512',
					'medium': '1K',
					'high': '2K',
					'512': '512',
					'1K': '1K',
					'2K': '2K',
					'4K': '4K',
			}
			
			return self.resolution_map.get( self.resolution_value )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'normalize_image_size( self, resolution: str=None, model: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_response_modalities( self, response_modalities: Optional[ str ],
			image_only: bool = False ) -> List[ str ]:
		"""Normalize response modalities.
		
		Purpose:
		    Provides normalize response modalities behavior for the Images workflow while preserving provider request and response state.
		
		Args:
		    response_modalities (Optional[str]): Response modalities supplied to the Gemini workflow.
		    image_only (bool): Image only supplied to the Gemini workflow.
		
		Returns:
		    List[str]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_grounding_tool( self, image_search: bool = False ) -> Optional[ Tool ]:
		"""Build grounding tool.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    image_search (bool): Image search supplied to the Gemini workflow.
		
		Returns:
		    Optional[Tool]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def get_content_config( self, response_modalities: Optional[ str ], image_only: bool = False,
			image_search: bool = False, grounded: bool = False,
			output_mime_type: Optional[ str ] = None ) -> GenerateContentConfig:
		"""Get content config.
		
		Purpose:
		    Retrieves normalized Gemini provider state or response data for display, reuse, or downstream request construction.
		
		Args:
		    response_modalities (Optional[str]): Response modalities supplied to the Gemini workflow.
		    image_only (bool): Image only supplied to the Gemini workflow.
		    image_search (bool): Image search supplied to the Gemini workflow.
		    grounded (bool): Grounded supplied to the Gemini workflow.
		    output_mime_type (Optional[str]): Output mime type supplied to the Gemini workflow.
		
		Returns:
		    GenerateContentConfig: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			
			self.size_value = self.normalize_image_size( resolution=self.size, model=self.model )
			if self.size_value:
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
			Logger( ).write( exception )
			raise exception
	
	def open_image( self, path: str ) -> PIL.Image.Image:
		"""Open image.
		
		Purpose:
		    Provides open image behavior for the Images workflow while preserving provider request and response state.
		
		Args:
		    path (str): Path supplied to the Gemini workflow.
		
		Returns:
		    PIL.Image.Image: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def capture_metadata( self ) -> None:
		"""Capture metadata.
		
		Purpose:
		    Captures response metadata from the most recent Gemini provider response and stores it for later source extraction.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def get_first_image( self ) -> Optional[ PIL.Image.Image ]:
		"""Get first image.
		
		Purpose:
		    Retrieves normalized Gemini provider state or response data for display, reuse, or downstream request construction.
		
		Returns:
		    Optional[PIL.Image.Image]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> Optional[ str ]:
		"""Get output text.
		
		Purpose:
		    Retrieves normalized Gemini provider state or response data for display, reuse, or downstream request construction.
		
		Returns:
		    Optional[str]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, model: str = 'gemini-2.5-flash-image', aspect: str = None,
			number: int = None, temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			resolution: str = None, instruct: str = None, output_mime_type: str = None,
			response_modalities: str = None, grounded: bool = False,
			image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""Generate.
		
		Purpose:
		    Provides generate behavior for the Images workflow while preserving provider request and response state.
		
		Args:
		    prompt (str): Prompt supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    aspect (str): Aspect supplied to the Gemini workflow.
		    number (int): Number supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    resolution (str): Resolution supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		    output_mime_type (str): Output mime type supplied to the Gemini workflow.
		    response_modalities (str): Response modalities supplied to the Gemini workflow.
		    grounded (bool): Grounded supplied to the Gemini workflow.
		    image_search (bool): Image search supplied to the Gemini workflow.
		
		Returns:
		    Optional[PIL.Image.Image]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str = 'gemini-2.5-flash-image',
			aspect: str = None, number: int = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, resolution: str = None, instruct: str = None,
			output_mime_type: str = None, response_modalities: str = None,
			grounded: bool = False, image_search: bool = False ) -> Optional[ str ]:
		"""Analyze.
		
		Purpose:
		    Provides analyze behavior for the Images workflow while preserving provider request and response state.
		
		Args:
		    prompt (str): Prompt supplied to the Gemini workflow.
		    path (str): Path supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    aspect (str): Aspect supplied to the Gemini workflow.
		    number (int): Number supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    resolution (str): Resolution supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		    output_mime_type (str): Output mime type supplied to the Gemini workflow.
		    response_modalities (str): Response modalities supplied to the Gemini workflow.
		    grounded (bool): Grounded supplied to the Gemini workflow.
		    image_search (bool): Image search supplied to the Gemini workflow.
		
		Returns:
		    Optional[str]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
				grounded=grounded, image_search=image_search,
				response_modalities=self.response_mode,
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
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gemini-2.5-flash-image',
			aspect: str = None, number: int = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, resolution: str = None, instruct: str = None,
			output_mime_type: str = None, response_modalities: str = None,
			grounded: bool = False, image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""Edit.
		
		Purpose:
		    Provides edit behavior for the Images workflow while preserving provider request and response state.
		
		Args:
		    prompt (str): Prompt supplied to the Gemini workflow.
		    path (str): Path supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    aspect (str): Aspect supplied to the Gemini workflow.
		    number (int): Number supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    resolution (str): Resolution supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		    output_mime_type (str): Output mime type supplied to the Gemini workflow.
		    response_modalities (str): Response modalities supplied to the Gemini workflow.
		    grounded (bool): Grounded supplied to the Gemini workflow.
		    image_search (bool): Image search supplied to the Gemini workflow.
		
		Returns:
		    Optional[PIL.Image.Image]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
				grounded=grounded, image_search=image_search,
				response_modalities=self.response_mode,
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
			Logger( ).write( exception )
			raise exception

class Embeddings( Gemini ):
	"""Embeddings workflow wrapper.
	
	Purpose:
	    Builds Gemini embedding requests and stores embedding configuration used by semantic-search and vector workflows.
	
	Attributes:
	    client: Runtime attribute used by the Embeddings workflow.
	    response: Runtime attribute used by the Embeddings workflow.
	    embedding: Runtime attribute used by the Embeddings workflow.
	    encoding_format: Runtime attribute used by the Embeddings workflow.
	    dimensions: Runtime attribute used by the Embeddings workflow.
	    task_type: Runtime attribute used by the Embeddings workflow.
	    title: Runtime attribute used by the Embeddings workflow.
	    embedding_config: Runtime attribute used by the Embeddings workflow.
	    contents: Runtime attribute used by the Embeddings workflow.
	    input_text: Runtime attribute used by the Embeddings workflow.
	    file_path: Runtime attribute used by the Embeddings workflow.
	    response_modalities: Runtime attribute used by the Embeddings workflow.
	"""
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
	
	def __init__( self, model: str = 'gemini-embedding-001' ):
		"""Initialize instance.
		
		Purpose:
		    Initializes Embeddings state with default configuration values and runtime attributes used by later Gemini provider calls.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		"""
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
		"""Model options.
		
		Purpose:
		    Returns the Gemini model names exposed by the related Streamlit selector without mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'gemini-embedding-001',
		         'gemini-embedding-2',
		         'gemini-embedding-2-preview',
		         'text-embedding-004',
		         'text-multilingual-embedding-002' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		"""Encoding options.
		
		Purpose:
		    Returns the configured option values exposed by the Embeddings workflow selector.
		
		Returns:
		    List[str]: Result produced by the Gemini workflow.
		"""
		return [ 'float', 'base64' ]
	
	@property
	def task_options( self ) -> List[ str ]:
		"""Task options.
		
		Purpose:
		    Returns the configured option values exposed by the Embeddings workflow selector.
		
		Returns:
		    List[str]: Result produced by the Gemini workflow.
		"""
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
		"""Normalize dimensions.
		
		Purpose:
		    Provides normalize dimensions behavior for the Embeddings workflow while preserving provider request and response state.
		
		Args:
		    dimensions (int): Dimensions supplied to the Gemini workflow.
		
		Returns:
		    int | None: Result produced by the Gemini workflow.
		"""
		try:
			throw_if( 'dimensions', dimensions )
			self.dimensions = dimensions
			if self.dimensions <= 0:
				return None
			
			return self.dimensions
		except Exception:
			return None
	
	def normalize_contents( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""Normalize contents.
		
		Purpose:
		    Provides normalize contents behavior for the Embeddings workflow while preserving provider request and response state.
		
		Args:
		    text (str | List[str]): Text supplied to the Gemini workflow.
		
		Returns:
		    str | List[str]: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def extract_embeddings( self ) -> List[ float ] | List[ List[ float ] ] | None:
		"""Extract embeddings.
		
		Purpose:
		    Provides extract embeddings behavior for the Embeddings workflow while preserving provider request and response state.
		
		Returns:
		    List[float] | List[List[float]] | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_embedding_config( self, model: str = 'gemini-embedding-001',
			dimensions: int = None, task_type: str = None,
			title: str = None ) -> EmbedContentConfig:
		"""Build embedding config.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		    dimensions (int): Dimensions supplied to the Gemini workflow.
		    task_type (str): Task type supplied to the Gemini workflow.
		    title (str): Title supplied to the Gemini workflow.
		
		Returns:
		    EmbedContentConfig: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			self.model = str( model or 'gemini-embedding-001' ).strip( )
			self.dimensions = self.normalize_dimensions( dimensions )
			self.task_type = str( task_type or '' ).strip( ).upper( )
			self.title = str( title or '' ).strip( )
			self.config_kwargs = { }
			
			if self.dimensions is not None:
				self.config_kwargs[ 'output_dimensionality' ] = self.dimensions
			
			if self.task_type and 'gemini-embedding-2' not in self.model:
				self.config_kwargs[ 'task_type' ] = self.task_type
			
			if self.title and self.task_type == 'RETRIEVAL_DOCUMENT' \
					and 'gemini-embedding-2' not in self.model:
				self.config_kwargs[ 'title' ] = self.title
			
			self.embedding_config = EmbedContentConfig( **self.config_kwargs )
			return self.embedding_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'build_embedding_config( self, model, dimensions, task_type, title )'
			Logger( ).write( exception )
			raise exception
	
	def create( self, text: str | List[ str ], model: str = 'gemini-embedding-001',
			dimensions: int = None, task_type: str = None, title: str = None,
			encoding_format: str = 'float' ) -> List[ float ] | List[ List[ float ] ] | None:
		"""Create.
		
		Purpose:
		    Provides create behavior for the Embeddings workflow while preserving provider request and response state.
		
		Args:
		    text (str | List[str]): Text supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    dimensions (int): Dimensions supplied to the Gemini workflow.
		    task_type (str): Task type supplied to the Gemini workflow.
		    title (str): Title supplied to the Gemini workflow.
		    encoding_format (str): Encoding format supplied to the Gemini workflow.
		
		Returns:
		    List[float] | List[List[float]] | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			throw_if( 'text', text )
			self.api_key = cfg.GEMINI_API_KEY
			throw_if( 'api_key', self.api_key )
			self.model = str( model or 'gemini-embedding-001' ).strip( )
			throw_if( 'model', self.model )
			self.dimensions = self.normalize_dimensions( dimensions )
			self.task_type = str( task_type or '' ).strip( ).upper( )
			self.title = str( title or '' ).strip( )
			self.encoding_format = encoding_format or 'float'
			self.input_text = self.normalize_contents( text=text )
			self.embedding_config = self.build_embedding_config(
				model=self.model,
				dimensions=self.dimensions,
				task_type=self.task_type,
				title=self.title )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.models.embed_content(
				model=self.model,
				contents=self.input_text,
				config=self.embedding_config )
			
			return self.extract_embeddings( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'create( self, text, model ) -> List[ float ] | List[ List[ float ] ]'
			Logger( ).write( exception )
			raise exception

class TTS( Gemini ):
	"""TTS workflow wrapper.
	
	Purpose:
	    Builds Gemini text-to-speech requests and stores voice, speech, and output configuration for audio generation workflows.
	
	Attributes:
	    speed: Runtime attribute used by the TTS workflow.
	    voice: Runtime attribute used by the TTS workflow.
	    response: Runtime attribute used by the TTS workflow.
	    voice_config: Runtime attribute used by the TTS workflow.
	    speech_config: Runtime attribute used by the TTS workflow.
	    client: Runtime attribute used by the TTS workflow.
	    audio_path: Runtime attribute used by the TTS workflow.
	    response_format: Runtime attribute used by the TTS workflow.
	    input_text: Runtime attribute used by the TTS workflow.
	    audio_bytes: Runtime attribute used by the TTS workflow.
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
		"""Initialize instance.
		
		Purpose:
		    Initializes TTS state with default configuration values and runtime attributes used by later Gemini provider calls.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		"""
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
		"""Model options.
		
		Purpose:
		    Returns the Gemini model names exposed by the related Streamlit selector without mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'gemini-3.1-flash-tts-preview', 'gemini-2.5-flash-preview-tts',
		         'gemini-2.5-pro-preview-tts' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'audio/wav' ]
	
	def to_wave_bytes( self, pcm_data: bytes, rate: int = 24000, channels: int = 1,
			sample_width: int = 2 ) -> bytes:
		"""To wave bytes.
		
		Purpose:
		    Provides to wave bytes behavior for the TTS workflow while preserving provider request and response state.
		
		Args:
		    pcm_data (bytes): Pcm data supplied to the Gemini workflow.
		    rate (int): Rate supplied to the Gemini workflow.
		    channels (int): Channels supplied to the Gemini workflow.
		    sample_width (int): Sample width supplied to the Gemini workflow.
		
		Returns:
		    bytes: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_voice( self, voice: Optional[ str ] = None ) -> str:
		"""Normalize voice.
		
		Purpose:
		    Provides normalize voice behavior for the TTS workflow while preserving provider request and response state.
		
		Args:
		    voice (Optional[str]): Voice supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_tts_prompt( self, text: str, speed: Optional[ float ] = None,
			instruct: Optional[ str ] = None ) -> str:
		"""Normalize tts prompt.
		
		Purpose:
		    Provides normalize tts prompt behavior for the TTS workflow while preserving provider request and response state.
		
		Args:
		    text (str): Text supplied to the Gemini workflow.
		    speed (Optional[float]): Speed supplied to the Gemini workflow.
		    instruct (Optional[str]): Instruct supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def create_speech( self, text: str, filepath: str = None,
			model: str = 'gemini-3.1-flash-tts-preview', format: str = 'audio/wav',
			speed: float = None, voice: str = None, frequency: float = None,
			presense: float = None, max_tokens: int = None, instruct: str = None,
			temperature: float = None, top_p: float = None ) -> bytes | str | None:
		"""Create speech.
		
		Purpose:
		    Creates the requested Gemini or Google Cloud resource using validated names, paths, or configuration values.
		
		Args:
		    text (str): Text supplied to the Gemini workflow.
		    filepath (str): Filepath supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    format (str): Format supplied to the Gemini workflow.
		    speed (float): Speed supplied to the Gemini workflow.
		    voice (str): Voice supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presense (float): Presense supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		
		Returns:
		    bytes | str | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			exception.method = 'create_speech( self, text: str, *args ) -> bytes | str | None'
			error = ErrorDialog( exception )
			error.show( )
			return None

class Transcription( Gemini ):
	"""Transcription workflow wrapper.
	
	Purpose:
	    Builds Gemini transcription requests from uploaded audio and stores request configuration for speech-to-text workflows.
	
	Attributes:
	    client: Runtime attribute used by the Transcription workflow.
	    transcript: Runtime attribute used by the Transcription workflow.
	    file_path: Runtime attribute used by the Transcription workflow.
	    response: Runtime attribute used by the Transcription workflow.
	"""
	client: Optional[ genai.Client ]
	transcript: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float = 0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 10000, instruct: str = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes Transcription state with default configuration values and runtime attributes used by later Gemini provider calls.
		
		Args:
		    n (int): N supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		"""
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
		"""Model options.
		
		Purpose:
		    Returns the Gemini model names exposed by the related Streamlit selector without mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'gemini-3-flash-preview',
		         'gemini-2.0-flash' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.
		
		Purpose:
		    Returns the configured option values exposed by the Transcription workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
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
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the Transcription workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
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
		"""Normalize mime type.
		
		Purpose:
		    Provides normalize mime type behavior for the Transcription workflow while preserving provider request and response state.
		
		Args:
		    path (str): Path supplied to the Gemini workflow.
		    mime_type (str): Mime type supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_prompt( self, language: str = None, start_time: float = None,
			end_time: float = None ) -> str:
		"""Build prompt.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    language (str): Language supplied to the Gemini workflow.
		    start_time (float): Start time supplied to the Gemini workflow.
		    end_time (float): End time supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
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
		"""Transcribe.
		
		Purpose:
		    Executes Gemini transcription using validated audio input and model configuration.
		
		Args:
		    path (str): Path supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    language (str): Language supplied to the Gemini workflow.
		    mime_type (str): Mime type supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    start_time (float): Start time supplied to the Gemini workflow.
		    end_time (float): End time supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		
		Returns:
		    Optional[str]: Result produced by the Gemini workflow.
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
	"""Translation workflow wrapper.
	
	Purpose:
	    Builds Gemini translation requests from audio/text inputs and stores language and model configuration for translation workflows.
	
	Attributes:
	    client: Runtime attribute used by the Translation workflow.
	    target_language: Runtime attribute used by the Translation workflow.
	    source_language: Runtime attribute used by the Translation workflow.
	    file_path: Runtime attribute used by the Translation workflow.
	    response: Runtime attribute used by the Translation workflow.
	"""
	client: Optional[ genai.Client ]
	target_language: Optional[ str ]
	source_language: Optional[ str ]
	file_path: Optional[ str ]
	response: Optional[ GenerateContentResponse ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3-flash-preview', temperature: float = 0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 10000,
			instruct: str = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes Translation state with default configuration values and runtime attributes used by later Gemini provider calls.
		
		Args:
		    n (int): N supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		"""
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
		"""Model options.
		
		Purpose:
		    Returns the Gemini model names exposed by the related Streamlit selector without mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'gemini-3-flash-preview',
		         'gemini-2.0-flash' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the Translation workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
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
		"""Normalize mime type.
		
		Purpose:
		    Provides normalize mime type behavior for the Translation workflow while preserving provider request and response state.
		
		Args:
		    path (str): Path supplied to the Gemini workflow.
		    mime_type (str): Mime type supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.
		
		Purpose:
		    Returns the configured option values exposed by the Translation workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'English',
		         'Spanish',
		         'French',
		         'Japanese',
		         'German',
		         'Chinese' ]
	
	def build_prompt( self, target: str, source: str = 'Auto', start_time: float = None,
			end_time: float = None ) -> str:
		"""Build prompt.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    target (str): Target supplied to the Gemini workflow.
		    source (str): Source supplied to the Gemini workflow.
		    start_time (float): Start time supplied to the Gemini workflow.
		    end_time (float): End time supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
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
		"""Translate.
		
		Purpose:
		    Executes Gemini translation using validated source content and language settings.
		
		Args:
		    path (str): Path supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    language (str): Language supplied to the Gemini workflow.
		    source (str): Source supplied to the Gemini workflow.
		    mime_type (str): Mime type supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    start_time (float): Start time supplied to the Gemini workflow.
		    end_time (float): End time supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		
		Returns:
		    Optional[str]: Result produced by the Gemini workflow.
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
			self.prompt = self.build_prompt( target=self.target_language,
				source=self.source_language,
				start_time=start_time, end_time=end_time )
			
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
	"""Files workflow wrapper.
	
	Purpose:
	    Manages Gemini file upload, retrieval, listing, deletion, and metadata workflows used by document and multimodal provider operations.
	
	Attributes:
	    client: Runtime attribute used by the Files workflow.
	    file_id: Runtime attribute used by the Files workflow.
	    file_path: Runtime attribute used by the Files workflow.
	    display_name: Runtime attribute used by the Files workflow.
	    model: Runtime attribute used by the Files workflow.
	    prompt: Runtime attribute used by the Files workflow.
	    response: Runtime attribute used by the Files workflow.
	    output_text: Runtime attribute used by the Files workflow.
	    file_list: Runtime attribute used by the Files workflow.
	    files: Runtime attribute used by the Files workflow.
	    documents: Runtime attribute used by the Files workflow.
	    content_config: Runtime attribute used by the Files workflow.
	    config_kwargs: Runtime attribute used by the Files workflow.
	"""
	client: Optional[ genai.Client ]
	file_id: Optional[ str ]
	file_path: Optional[ str ]
	display_name: Optional[ str ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	response: Optional[ Any ]
	output_text: Optional[ str ]
	file_list: Optional[ List[ Any ] ]
	files: Optional[ List[ str ] ]
	documents: Optional[ Dict[ str, str ] ]
	content_config: Optional[ GenerateContentConfig ]
	config_kwargs: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite' ) -> None:
		"""Initialize instance.
		
		Purpose:
		    Initializes Files state with default configuration values and runtime attributes used by later Gemini provider calls.
		
		Args:
		    model (str): Model supplied to the Gemini workflow.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.model = model
		self.client = None
		self.file_id = None
		self.file_path = None
		self.display_name = None
		self.prompt = None
		self.response = None
		self.output_text = None
		self.file_list = [ ]
		self.files = [ ]
		self.documents = { }
		self.content_config = None
		self.config_kwargs = { }
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns the Gemini model names exposed by the related Streamlit selector without mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [
				'gemini-3.1-flash-lite-preview',
				'gemini-3.1-pro-preview',
				'gemini-3-flash-preview',
				'gemini-2.5-flash',
				'gemini-2.5-flash-lite',
				'gemini-2.5-pro',
				'gemini-2.0-flash',
				'gemini-2.0-flash-lite',
		]
	
	@property
	def file_options( self ) -> List[ str ] | None:
		"""File options.
		
		Purpose:
		    Returns the configured option values exposed by the Files workflow selector.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return self.files
	
	def initialize_client( self ) -> genai.Client:
		"""Initialize client.
		
		Purpose:
		    Provides initialize client behavior for the Files workflow while preserving provider request and response state.
		
		Returns:
		    genai.Client: Result produced by the Gemini workflow.
		"""
		try:
			throw_if( 'gemini_api_key', cfg.GEMINI_API_KEY )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			return self.client
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'initialize_client( self ) -> genai.Client'
			raise ex
	
	def normalize_file_id( self, file_id: str = None, id: str = None,
			name: str = None ) -> str:
		"""Normalize file id.
		
		Purpose:
		    Provides normalize file id behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    file_id (str): File id supplied to the Gemini workflow.
		    id (str): Id supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
		"""
		try:
			if isinstance( file_id, str ) and file_id.strip( ):
				return file_id.strip( )
			
			if isinstance( id, str ) and id.strip( ):
				return id.strip( )
			
			if isinstance( name, str ) and name.strip( ):
				return name.strip( )
			
			throw_if( 'file_id', file_id )
			return ''
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'normalize_file_id( self, file_id, id, name ) -> str'
			raise ex
	
	def normalize_path( self, path: str = None, filepath: str = None ) -> str:
		"""Normalize path.
		
		Purpose:
		    Provides normalize path behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    path (str): Path supplied to the Gemini workflow.
		    filepath (str): Filepath supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
		"""
		try:
			if isinstance( path, str ) and path.strip( ):
				return path.strip( )
			
			if isinstance( filepath, str ) and filepath.strip( ):
				return filepath.strip( )
			
			throw_if( 'path', path )
			return ''
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'normalize_path( self, path, filepath ) -> str'
			raise ex
	
	def normalize_file_object( self, file: Any ) -> Dict[ str, Any ]:
		"""Normalize file object.
		
		Purpose:
		    Provides normalize file object behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    file (Any): File supplied to the Gemini workflow.
		
		Returns:
		    Dict[str, Any]: Result produced by the Gemini workflow.
		"""
		try:
			if file is None:
				return { }
			
			if hasattr( file, 'model_dump' ):
				return file.model_dump( )
			
			return {
					'name': getattr( file, 'name', None ),
					'display_name': getattr( file, 'display_name', None ),
					'mime_type': getattr( file, 'mime_type', None ),
					'size_bytes': getattr( file, 'size_bytes', None ),
					'create_time': getattr( file, 'create_time', None ),
					'update_time': getattr( file, 'update_time', None ),
					'expiration_time': getattr( file, 'expiration_time', None ),
					'uri': getattr( file, 'uri', None ),
					'state': getattr( file, 'state', None ),
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'normalize_file_object( self, file ) -> Dict[ str, Any ]'
			raise ex
	
	def extract_output_text( self ) -> str | None:
		"""Extract output text.
		
		Purpose:
		    Provides extract output text behavior for the Files workflow while preserving provider request and response state.
		
		Returns:
		    str | None: Result produced by the Gemini workflow.
		"""
		try:
			if self.response is None:
				return None
			
			self.output_text = getattr( self.response, 'text', None )
			if isinstance( self.output_text, str ) and self.output_text.strip( ):
				return self.output_text.strip( )
			
			return str( self.response )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'extract_output_text( self ) -> str | None'
			raise ex
	
	def build_generation_config( self, temperature: float = None, top_p: float = None,
			top_k: int = None, max_tokens: int = None,
			stops: List[ str ] = None, instruct: str = None ) -> GenerateContentConfig:
		"""Build generation config.
		
		Purpose:
		    Builds normalized Gemini request configuration from validated inputs and stores the resulting state on the instance for provider execution.
		
		Args:
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    top_k (int): Top k supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    stops (List[str]): Stops supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		
		Returns:
		    GenerateContentConfig: Result produced by the Gemini workflow.
		"""
		try:
			self.config_kwargs = { }
			
			if temperature is not None:
				self.config_kwargs[ 'temperature' ] = temperature
			
			if top_p is not None and float( top_p ) > 0:
				self.config_kwargs[ 'top_p' ] = top_p
			
			if top_k is not None and int( top_k ) > 0:
				self.config_kwargs[ 'top_k' ] = int( top_k )
			
			if max_tokens is not None and int( max_tokens ) > 0:
				self.config_kwargs[ 'max_output_tokens' ] = int( max_tokens )
			
			if isinstance( stops, list ) and len( stops ) > 0:
				self.config_kwargs[ 'stop_sequences' ] = stops
			
			if isinstance( instruct, str ) and instruct.strip( ):
				self.config_kwargs[ 'system_instruction' ] = instruct.strip( )
			
			self.content_config = GenerateContentConfig( **self.config_kwargs )
			return self.content_config
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'build_generation_config( self, **kwargs ) -> GenerateContentConfig'
			raise ex
	
	def upload( self, path: str = None, filepath: str = None,
			display_name: str = None, name: str = None ) -> File | Any:
		"""Upload.
		
		Purpose:
		    Provides upload behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    path (str): Path supplied to the Gemini workflow.
		    filepath (str): Filepath supplied to the Gemini workflow.
		    display_name (str): Display name supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    File | Any: Result produced by the Gemini workflow.
		"""
		try:
			self.initialize_client( )
			self.file_path = self.normalize_path( path=path, filepath=filepath )
			self.display_name = display_name if isinstance(
				display_name, str ) and display_name.strip( ) else name
			
			if not isinstance( self.display_name, str ) or not self.display_name.strip( ):
				self.display_name = Path( self.file_path ).name
			
			self.response = self.client.files.upload(
				file=self.file_path,
				config={ 'display_name': self.display_name.strip( ) } )
			
			self.file_id = getattr( self.response, 'name', None )
			if isinstance( self.file_id, str ) and self.file_id.strip( ):
				self.documents[ self.display_name ] = self.file_id
			
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'upload( self, path: str=None, filepath: str=None ) -> File | Any'
			raise ex
	
	def list( self, page_size: int = None ) -> List[ Any ]:
		"""List.
		
		Purpose:
		    Provides list behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    page_size (int): Page size supplied to the Gemini workflow.
		
		Returns:
		    List[Any]: Result produced by the Gemini workflow.
		"""
		try:
			self.initialize_client( )
			self.file_list = [ ]
			
			for file in self.client.files.list( ):
				self.file_list.append( file )
			
			self.files = [
					getattr( file, 'name', '' )
					for file in self.file_list
					if getattr( file, 'name', None )
			]
			
			self.documents = { }
			for file in self.file_list:
				resource_name = getattr( file, 'name', None )
				display_name = getattr( file, 'display_name', None ) or resource_name
				
				if resource_name:
					self.documents[ display_name ] = resource_name
			
			return self.file_list
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'list( self, page_size: int=None ) -> List[ Any ]'
			raise ex
	
	def list_files( self, page_size: int = None ) -> List[ Any ]:
		"""List files.
		
		Purpose:
		    Lists Gemini or Google Cloud resources and returns normalized metadata for UI display or downstream selection.
		
		Args:
		    page_size (int): Page size supplied to the Gemini workflow.
		
		Returns:
		    List[Any]: Result produced by the Gemini workflow.
		"""
		try:
			return self.list( page_size=page_size )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'list_files( self, page_size: int=None ) -> List[ Any ]'
			raise ex
	
	def retrieve( self, file_id: str = None, id: str = None, name: str = None ) -> File | Any:
		"""Retrieve.
		
		Purpose:
		    Provides retrieve behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    file_id (str): File id supplied to the Gemini workflow.
		    id (str): Id supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    File | Any: Result produced by the Gemini workflow.
		"""
		try:
			self.initialize_client( )
			self.file_id = self.normalize_file_id( file_id=file_id, id=id, name=name )
			self.response = self.client.files.get( name=self.file_id )
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'retrieve( self, file_id: str=None, id: str=None ) -> File | Any'
			raise ex
	
	def extract( self, file_id: str = None, id: str = None, name: str = None ) -> str:
		"""Extract.
		
		Purpose:
		    Provides extract behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    file_id (str): File id supplied to the Gemini workflow.
		    id (str): Id supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    str: Result produced by the Gemini workflow.
		"""
		try:
			file = self.retrieve( file_id=file_id, id=id, name=name )
			metadata = self.normalize_file_object( file )
			return json.dumps( metadata, indent=2, default=str )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'extract( self, file_id: str=None, id: str=None ) -> str'
			raise ex
	
	def delete( self, file_id: str = None, id: str = None, name: str = None ) -> Dict[ str, Any ]:
		"""Delete.
		
		Purpose:
		    Provides delete behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    file_id (str): File id supplied to the Gemini workflow.
		    id (str): Id supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    Dict[str, Any]: Result produced by the Gemini workflow.
		"""
		try:
			self.initialize_client( )
			self.file_id = self.normalize_file_id( file_id=file_id, id=id, name=name )
			self.client.files.delete( name=self.file_id )
			
			return {
					'deleted': True,
					'name': self.file_id,
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'delete( self, file_id: str=None, id: str=None ) -> Dict[ str, Any ]'
			raise ex
	
	def summarize( self, id: str = None, file_id: str = None, path: str = None,
			filepath: str = None, prompt: str = None, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None, top_p: float = None, top_k: int = None,
			max_tokens: int = None, stops: List[ str ] = None,
			instruct: str = None ) -> str | None:
		"""Summarize.
		
		Purpose:
		    Provides summarize behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    id (str): Id supplied to the Gemini workflow.
		    file_id (str): File id supplied to the Gemini workflow.
		    path (str): Path supplied to the Gemini workflow.
		    filepath (str): Filepath supplied to the Gemini workflow.
		    prompt (str): Prompt supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    top_k (int): Top k supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    stops (List[str]): Stops supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		
		Returns:
		    str | None: Result produced by the Gemini workflow.
		"""
		try:
			self.initialize_client( )
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else \
				'Summarize this file.'
			self.model = model if isinstance( model, str ) and model.strip( ) else \
				'gemini-2.5-flash-lite'
			
			self.content_config = self.build_generation_config(
				temperature=temperature,
				top_p=top_p,
				top_k=top_k,
				max_tokens=max_tokens,
				stops=stops,
				instruct=instruct )
			
			if isinstance( id, str ) and id.strip( ) or isinstance( file_id,
					str ) and file_id.strip( ):
				self.file_id = self.normalize_file_id( file_id=file_id, id=id )
				file = self.client.files.get( name=self.file_id )
			else:
				self.file_path = self.normalize_path( path=path, filepath=filepath )
				file = self.client.files.upload( file=self.file_path )
				self.file_id = getattr( file, 'name', None )
			
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=[ self.prompt, file ],
				config=self.content_config )
			
			return self.extract_output_text( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'summarize( self, id: str=None, prompt: str=None ) -> str | None'
			raise ex
	
	def search( self, id: str = None, file_id: str = None, query: str = None,
			prompt: str = None, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None, top_p: float = None, top_k: int = None,
			max_tokens: int = None, stops: List[ str ] = None,
			instruct: str = None ) -> str | None:
		"""Search.
		
		Purpose:
		    Provides search behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    id (str): Id supplied to the Gemini workflow.
		    file_id (str): File id supplied to the Gemini workflow.
		    query (str): Query supplied to the Gemini workflow.
		    prompt (str): Prompt supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    top_k (int): Top k supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    stops (List[str]): Stops supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		
		Returns:
		    str | None: Result produced by the Gemini workflow.
		"""
		try:
			question = query if isinstance( query, str ) and query.strip( ) else prompt
			throw_if( 'query', question )
			
			return self.summarize(
				id=id,
				file_id=file_id,
				prompt=question,
				model=model,
				temperature=temperature,
				top_p=top_p,
				top_k=top_k,
				max_tokens=max_tokens,
				stops=stops,
				instruct=instruct )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'search( self, id: str=None, query: str=None ) -> str | None'
			raise ex
	
	def survey( self, id: str = None, file_id: str = None, name: str = None ) -> Dict[ str, Any ]:
		"""Survey.
		
		Purpose:
		    Provides survey behavior for the Files workflow while preserving provider request and response state.
		
		Args:
		    id (str): Id supplied to the Gemini workflow.
		    file_id (str): File id supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    Dict[str, Any]: Result produced by the Gemini workflow.
		"""
		try:
			file = self.retrieve( file_id=file_id, id=id, name=name )
			return self.normalize_file_object( file )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'Files'
			ex.method = 'survey( self, id: str=None, file_id: str=None ) -> Dict[ str, Any ]'
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Provides dir behavior for the Files workflow while preserving provider request and response state.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [
				'client',
				'file_id',
				'file_path',
				'display_name',
				'model',
				'prompt',
				'response',
				'output_text',
				'file_list',
				'files',
				'documents',
				'content_config',
				'config_kwargs',
				'model_options',
				'file_options',
				'initialize_client',
				'normalize_file_id',
				'normalize_path',
				'normalize_file_object',
				'extract_output_text',
				'build_generation_config',
				'upload',
				'list',
				'list_files',
				'retrieve',
				'extract',
				'delete',
				'summarize',
				'search',
				'survey',
		]

class FileSearch( Gemini ):
	"""FileSearch workflow wrapper.
	
	Purpose:
	    Manages Gemini file-search store operations used to create, inspect, and connect searchable document stores.
	
	Attributes:
	    client: Runtime attribute used by the FileSearch workflow.
	    response: Runtime attribute used by the FileSearch workflow.
	    store_id: Runtime attribute used by the FileSearch workflow.
	    store_name: Runtime attribute used by the FileSearch workflow.
	    collections: Runtime attribute used by the FileSearch workflow.
	    stores: Runtime attribute used by the FileSearch workflow.
	"""
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	store_id: Optional[ str ]
	store_name: Optional[ str ]
	collections: Optional[ Dict[ str, str ] ]
	stores: Optional[ List[ FileSearchStore ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes FileSearch state with default configuration values and runtime attributes used by later Gemini provider calls.
		"""
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
		"""Refresh collections.
		
		Purpose:
		    Provides refresh collections behavior for the FileSearch workflow while preserving provider request and response state.
		
		Returns:
		    Dict[str, str]: Result produced by the Gemini workflow.
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
				self.collections[ self.label ] = str( self.resource_name ).strip( )
			
			return self.collections
		except Exception:
			self.collections = { }
			self.stores = [ ]
			return self.collections
	
	def create( self, name: str ) -> FileSearchStore | Any:
		"""Create.
		
		Purpose:
		    Provides create behavior for the FileSearch workflow while preserving provider request and response state.
		
		Args:
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    FileSearchStore | Any: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			throw_if( 'name', name )
			self.store_name = str( name ).strip( )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.response = self.client.file_search_stores.create(
				config={ 'display_name': self.store_name } )
			self.refresh_collections( )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'create( self, name: str ) -> FileSearchStore | Any'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str ) -> FileSearchStore | Any:
		"""Retrieve.
		
		Purpose:
		    Provides retrieve behavior for the FileSearch workflow while preserving provider request and response state.
		
		Args:
		    store_id (str): Store id supplied to the Gemini workflow.
		
		Returns:
		    FileSearchStore | Any: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def list( self ) -> List[ FileSearchStore ] | Any:
		"""List.
		
		Purpose:
		    Provides list behavior for the FileSearch workflow while preserving provider request and response state.
		
		Returns:
		    List[FileSearchStore] | Any: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			self.refresh_collections( )
			return self.stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'list( self ) -> List[ FileSearchStore ] | Any'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str, force: bool = True ) -> bool | Any:
		"""Delete.
		
		Purpose:
		    Provides delete behavior for the FileSearch workflow while preserving provider request and response state.
		
		Args:
		    store_id (str): Store id supplied to the Gemini workflow.
		    force (bool): Force supplied to the Gemini workflow.
		
		Returns:
		    bool | Any: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def upload_to_store( self, store_id: str = None, name: str = None, path: str = None,
			filepath: str = None, display_name: str = None,
			mime_type: str = None, metadata: Dict[ str, Any ] = None,
			chunking_config: Dict[ str, Any ] = None ) -> Any:
		"""Upload to store.
		
		Purpose:
		    Uploads a local file or object to Gemini or Google Cloud storage and stores the returned resource metadata.
		
		Args:
		    store_id (str): Store id supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		    path (str): Path supplied to the Gemini workflow.
		    filepath (str): Filepath supplied to the Gemini workflow.
		    display_name (str): Display name supplied to the Gemini workflow.
		    mime_type (str): Mime type supplied to the Gemini workflow.
		    metadata (Dict[str, Any]): Metadata supplied to the Gemini workflow.
		    chunking_config (Dict[str, Any]): Chunking config supplied to the Gemini workflow.
		
		Returns:
		    Any: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			self.store_id = store_id if isinstance( store_id, str ) and store_id.strip( ) else name
			self.file_path = path if isinstance( path, str ) and path.strip( ) else filepath
			throw_if( 'store_id', self.store_id )
			throw_if( 'path', self.file_path )
			self.client = genai.Client( api_key=cfg.GEMINI_API_KEY )
			self.config = { }
			
			if isinstance( display_name, str ) and display_name.strip( ):
				self.config[ 'display_name' ] = display_name.strip( )
			
			if isinstance( mime_type, str ) and mime_type.strip( ):
				self.config[ 'mime_type' ] = mime_type.strip( )
			
			if isinstance( metadata, dict ) and len( metadata ) > 0:
				self.config[ 'metadata' ] = metadata
			
			if isinstance( chunking_config, dict ) and len( chunking_config ) > 0:
				self.config[ 'chunking_config' ] = chunking_config
			
			if len( self.config ) > 0:
				self.response = self.client.file_search_stores.upload_to_file_search_store(
					file_search_store_name=self.store_id.strip( ),
					file=self.file_path,
					config=self.config )
			else:
				self.response = self.client.file_search_stores.upload_to_file_search_store(
					file_search_store_name=self.store_id.strip( ),
					file=self.file_path )
			
			self.refresh_collections( )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'upload_to_store( self, store_id: str=None, path: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def upload( self, store_id: str = None, name: str = None, path: str = None,
			filepath: str = None, display_name: str = None,
			mime_type: str = None, metadata: Dict[ str, Any ] = None,
			chunking_config: Dict[ str, Any ] = None ) -> Any:
		"""Upload.
		
		Purpose:
		    Provides upload behavior for the FileSearch workflow while preserving provider request and response state.
		
		Args:
		    store_id (str): Store id supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		    path (str): Path supplied to the Gemini workflow.
		    filepath (str): Filepath supplied to the Gemini workflow.
		    display_name (str): Display name supplied to the Gemini workflow.
		    mime_type (str): Mime type supplied to the Gemini workflow.
		    metadata (Dict[str, Any]): Metadata supplied to the Gemini workflow.
		    chunking_config (Dict[str, Any]): Chunking config supplied to the Gemini workflow.
		
		Returns:
		    Any: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			return self.upload_to_store(
				store_id=store_id,
				name=name,
				path=path,
				filepath=filepath,
				display_name=display_name,
				mime_type=mime_type,
				metadata=metadata,
				chunking_config=chunking_config )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'upload( self, store_id: str=None, path: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def upload_file( self, store_id: str = None, name: str = None, path: str = None,
			filepath: str = None, display_name: str = None,
			mime_type: str = None, metadata: Dict[ str, Any ] = None,
			chunking_config: Dict[ str, Any ] = None ) -> Any:
		"""Upload file.
		
		Purpose:
		    Uploads a local file or object to Gemini or Google Cloud storage and stores the returned resource metadata.
		
		Args:
		    store_id (str): Store id supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		    path (str): Path supplied to the Gemini workflow.
		    filepath (str): Filepath supplied to the Gemini workflow.
		    display_name (str): Display name supplied to the Gemini workflow.
		    mime_type (str): Mime type supplied to the Gemini workflow.
		    metadata (Dict[str, Any]): Metadata supplied to the Gemini workflow.
		    chunking_config (Dict[str, Any]): Chunking config supplied to the Gemini workflow.
		
		Returns:
		    Any: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			return self.upload_to_store(
				store_id=store_id,
				name=name,
				path=path,
				filepath=filepath,
				display_name=display_name,
				mime_type=mime_type,
				metadata=metadata,
				chunking_config=chunking_config )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'upload_file( self, store_id: str=None, path: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def import_file( self, store_id: str = None, name: str = None, path: str = None,
			filepath: str = None, display_name: str = None,
			mime_type: str = None, metadata: Dict[ str, Any ] = None,
			chunking_config: Dict[ str, Any ] = None ) -> Any:
		"""Import file.
		
		Purpose:
		    Provides import file behavior for the FileSearch workflow while preserving provider request and response state.
		
		Args:
		    store_id (str): Store id supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		    path (str): Path supplied to the Gemini workflow.
		    filepath (str): Filepath supplied to the Gemini workflow.
		    display_name (str): Display name supplied to the Gemini workflow.
		    mime_type (str): Mime type supplied to the Gemini workflow.
		    metadata (Dict[str, Any]): Metadata supplied to the Gemini workflow.
		    chunking_config (Dict[str, Any]): Chunking config supplied to the Gemini workflow.
		
		Returns:
		    Any: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			return self.upload_to_store(
				store_id=store_id,
				name=name,
				path=path,
				filepath=filepath,
				display_name=display_name,
				mime_type=mime_type,
				metadata=metadata,
				chunking_config=chunking_config )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'import_file( self, store_id: str=None, path: str=None )'
			Logger( ).write( exception )
			raise exception

class CloudBuckets( Gemini ):
	"""CloudBuckets workflow wrapper.
	
	Purpose:
	    Manages Google Cloud Storage bucket and object workflows used by Gemini file and document processing paths.
	
	Attributes:
	    project_id: Runtime attribute used by the CloudBuckets workflow.
	    bucket_name: Runtime attribute used by the CloudBuckets workflow.
	    object_name: Runtime attribute used by the CloudBuckets workflow.
	    file_path: Runtime attribute used by the CloudBuckets workflow.
	    file_ids: Runtime attribute used by the CloudBuckets workflow.
	    store_ids: Runtime attribute used by the CloudBuckets workflow.
	    client: Runtime attribute used by the CloudBuckets workflow.
	    bucket: Runtime attribute used by the CloudBuckets workflow.
	    response: Runtime attribute used by the CloudBuckets workflow.
	    collections: Runtime attribute used by the CloudBuckets workflow.
	    documents: Runtime attribute used by the CloudBuckets workflow.
	"""
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
		"""Initialize instance.
		
		Purpose:
		    Initializes CloudBuckets state with default configuration values and runtime attributes used by later Gemini provider calls.
		"""
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
		"""Model options.
		
		Purpose:
		    Returns the Gemini model names exposed by the related Streamlit selector without mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the Gemini workflow.
		"""
		return [ 'gemini-2.5-flash',
		         'gemini-2.5 flash image',
		         'gemini-2.5 flash-tts',
		         'gemini-2.5 flash-lite',
		         'gemini-2.0-flash',
		         'gemini-2.0-flash-lite' ]
	
	@property
	def media_options( self ):
		"""Media options.
		
		Purpose:
		    Returns the configured option values exposed by the CloudBuckets workflow selector.
		
		Returns:
		    list[str] | None: Option values exposed to the application UI.
		"""
		return [ 'media_resolution_high',
		         'media_resolution_medium',
		         'media_resolution_low' ]
	
	def create( self, bucket: str, name: str ):
		"""Create.
		
		Purpose:
		    Provides create behavior for the CloudBuckets workflow while preserving provider request and response state.
		
		Args:
		    bucket (str): Bucket supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    object: Result returned by the Gemini workflow.
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
		"""Upload.
		
		Purpose:
		    Provides upload behavior for the CloudBuckets workflow while preserving provider request and response state.
		
		Args:
		    path (str): Path supplied to the Gemini workflow.
		    bucket (str): Bucket supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    object: Result returned by the Gemini workflow.
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
		"""Retrieve.
		
		Purpose:
		    Provides retrieve behavior for the CloudBuckets workflow while preserving provider request and response state.
		
		Args:
		    bucket (str): Bucket supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    object: Result returned by the Gemini workflow.
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
		
		def delete( self, bucket: str = None, name: str = None,
				bucket_name: str = None, object_name: str = None ) -> bool:
			"""Delete.
			
			Purpose:
			    Provides delete behavior for the CloudBuckets workflow while preserving provider request and response state.
			
			Args:
			    bucket (str): Bucket supplied to the Gemini workflow.
			    name (str): Name supplied to the Gemini workflow.
			    bucket_name (str): Bucket name supplied to the Gemini workflow.
			    object_name (str): Object name supplied to the Gemini workflow.
			
			Returns:
			    bool: Result produced by the Gemini workflow.
			"""
		
		try:
			self.bucket_name = bucket_name if isinstance( bucket_name, str ) and \
			                                  bucket_name.strip( ) else bucket
			self.object_name = object_name if isinstance( object_name, str ) and \
			                                  object_name.strip( ) else name
			throw_if( 'bucket', self.bucket_name )
			throw_if( 'name', self.object_name )
			self.bucket = self.client.bucket( self.bucket_name )
			blob = self.bucket.blob( self.object_name )
			blob.delete( )
			return True
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'delete( self, bucket: str=None, name: str=None ) -> bool'
			raise ex
	
	def delete_object( self, bucket: str = None, name: str = None,
			bucket_name: str = None, object_name: str = None ) -> bool:
		"""Delete object.
		
		Purpose:
		    Deletes the requested Gemini or Google Cloud resource after validating the supplied identifier.
		
		Args:
		    bucket (str): Bucket supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		    bucket_name (str): Bucket name supplied to the Gemini workflow.
		    object_name (str): Object name supplied to the Gemini workflow.
		
		Returns:
		    bool: Result produced by the Gemini workflow.
		"""
		try:
			return self.delete(
				bucket=bucket,
				name=name,
				bucket_name=bucket_name,
				object_name=object_name )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'delete_object( self, bucket: str=None, name: str=None ) -> bool'
			raise ex
	
	def delete_blob( self, bucket: str = None, name: str = None,
			bucket_name: str = None, object_name: str = None ) -> bool:
		"""Delete blob.
		
		Purpose:
		    Deletes the requested Gemini or Google Cloud resource after validating the supplied identifier.
		
		Args:
		    bucket (str): Bucket supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		    bucket_name (str): Bucket name supplied to the Gemini workflow.
		    object_name (str): Object name supplied to the Gemini workflow.
		
		Returns:
		    bool: Result produced by the Gemini workflow.
		"""
		try:
			return self.delete(
				bucket=bucket,
				name=name,
				bucket_name=bucket_name,
				object_name=object_name )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'delete_blob( self, bucket: str=None, name: str=None ) -> bool'
			raise ex
	
	def delete_file( self, bucket: str = None, name: str = None,
			bucket_name: str = None, object_name: str = None ) -> bool:
		"""Delete file.
		
		Purpose:
		    Deletes the requested Gemini or Google Cloud resource after validating the supplied identifier.
		
		Args:
		    bucket (str): Bucket supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		    bucket_name (str): Bucket name supplied to the Gemini workflow.
		    object_name (str): Object name supplied to the Gemini workflow.
		
		Returns:
		    bool: Result produced by the Gemini workflow.
		"""
		try:
			return self.delete(
				bucket=bucket,
				name=name,
				bucket_name=bucket_name,
				object_name=object_name )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gemini'
			ex.cause = 'CloudBuckets'
			ex.method = 'delete_file( self, bucket: str=None, name: str=None ) -> bool'
			raise ex
	
	def list( self, bucket: str ):
		"""List.
		
		Purpose:
		    Provides list behavior for the CloudBuckets workflow while preserving provider request and response state.
		
		Args:
		    bucket (str): Bucket supplied to the Gemini workflow.
		
		Returns:
		    object: Result returned by the Gemini workflow.
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
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None,
			max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str | None:
		"""Web search.
		
		Purpose:
		    Provides web search behavior for the CloudBuckets workflow while preserving provider request and response state.
		
		Args:
		    prompt (str): Prompt supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    stops (List[str]): Stops supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		
		Returns:
		    str | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
	
	def search_maps( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
			temperature: float = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
			instruct: str = None ) -> str | None:
		"""Search maps.
		
		Purpose:
		    Provides search maps behavior for the CloudBuckets workflow while preserving provider request and response state.
		
		Args:
		    prompt (str): Prompt supplied to the Gemini workflow.
		    model (str): Model supplied to the Gemini workflow.
		    temperature (float): Temperature supplied to the Gemini workflow.
		    top_p (float): Top p supplied to the Gemini workflow.
		    frequency (float): Frequency supplied to the Gemini workflow.
		    presence (float): Presence supplied to the Gemini workflow.
		    max_tokens (int): Max tokens supplied to the Gemini workflow.
		    stops (List[str]): Stops supplied to the Gemini workflow.
		    instruct (str): Instruct supplied to the Gemini workflow.
		
		Returns:
		    str | None: Result produced by the Gemini workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
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
	
	def delete( self, bucket: str, name: str ):
		"""Delete.
		
		Purpose:
		    Provides delete behavior for the CloudBuckets workflow while preserving provider request and response state.
		
		Args:
		    bucket (str): Bucket supplied to the Gemini workflow.
		    name (str): Name supplied to the Gemini workflow.
		
		Returns:
		    object: Result returned by the Gemini workflow.
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
