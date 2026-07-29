'''
  ******************************************************************************************
      Assembly:                Buddy
      Filename:                gemini.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        12-27-2025
  ******************************************************************************************
  <copyright file="gemini.py" company="Terry D. Eppler">

	     gemini.py
	     Copyright ©  2025 Terry Eppler

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
    Provides Google Gemini, Google GenAI, file-search, cloud-storage, embedding, image,
    text, audio, transcription, translation, and file-management wrappers used by the
    Buddy Streamlit application and its MkDocs API reference.
  </summary>
  *****************************************************************************************
'''
from google.genai.file_search_stores import FileSearchStores
import config as cfg
import base64
from boogr import Error, Logger
import io
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
	    Validates that a required argument contains a usable value before the surrounding workflow
	    continues. This guard centralizes early validation so provider wrappers and UI routines fail
	    with consistent, readable error messages.
	
	Args:
	    name (str): Name value used by the operation.
	    value (object): Value value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encode image.
	
	Purpose:
	    Performs the encode_image workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider, and
	    data-processing paths can call it consistently.
	
	Args:
	    image_path (str): Image path value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class Gemini( ):
	"""Gemini class.
	
	Purpose:
	    Defines the Gemini component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit workflows can
	    call a consistent interface.
	
	Attributes:
	    number (Optional[int]): Stores number for the component runtime state.
	    google_api_key (Optional[str]): Stores google api key for the component runtime state.
	    gemini_api_key (Optional[str]): Stores gemini api key for the component runtime state.
	    instructions (Optional[str]): Stores instructions for the component runtime state.
	    prompt (Optional[str]): Stores prompt for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    api_version (Optional[str]): Stores api version for the component runtime state.
	    max_tokens (Optional[int]): Stores max tokens for the component runtime state.
	    temperature (Optional[float]): Stores temperature for the component runtime state.
	    top_p (Optional[float]): Stores top p for the component runtime state.
	    top_k (Optional[int]): Stores top k for the component runtime state.
	    candidate_count (Optional[int]): Stores candidate count for the component runtime state.
	    media_resolution (Optional[str]): Stores media resolution for the component runtime state.
	    response_modalities (Optional[List[str]]): Stores response modalities for the component runtime state.
	    stops (Optional[List[str]]): Stores stops for the component runtime state.
	    domains (Optional[List[str]]): Stores domains for the component runtime state.
	    frequency_penalty (Optional[float]): Stores frequency penalty for the component runtime state.
	    presence_penalty (Optional[float]): Stores presence penalty for the component runtime state.
	    response_format (Optional[str]): Stores response format for the component runtime state.
	    content_response (Optional[GenerateContentResponse]): Stores content response for the component runtime state.
	    image_response (Optional[GenerateImagesResponse]): Stores image response for the component runtime state.
	    content_config (Optional[GenerateContentConfig]): Stores content config for the component runtime state.
	    function_config (Optional[FunctionCallingConfig]): Stores function config for the component runtime state.
	    thought_config (Optional[ThinkingConfig]): Stores thought config for the component runtime state.
	    genimg_config (Optional[GenerateImagesConfig]): Stores genimg config for the component runtime state.
	    image_config (Optional[ImageConfig]): Stores image config for the component runtime state.
	    tool_config (Optional[List[types.Tool]]): Stores tool config for the component runtime state.
	    tool_choice (Optional[str]): Stores tool choice for the component runtime state.
	    tools (Optional[List[str]]): Stores tools for the component runtime state."""
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
	"""Provide Gemini text-generation workflow support.
	
	Purpose:
		Provides synchronous and streaming text generation through the Google Gen AI SDK.
		The class constructs Gemini content, configuration, safety, reasoning, structured-output,
		URL-context, Google Search, File Search, and code-execution objects from arguments assigned
		to object members.
	
	Attributes:
		client (Optional[genai.Client]): Google Gen AI client.
		model (str): Gemini model used by the current request.
		prompt (str): User prompt used by the current request.
		contents (List[Content]): Provider-ready request contents.
		content_config (Optional[GenerateContentConfig]): Provider generation configuration.
		content_response (Optional[GenerateContentResponse]): Latest synchronous response.
		stream_response (Any): Latest streaming response iterator.
		output_text (str): Text extracted from the latest response.
		context (List[Dict[str, Any]]): Application conversation history.
		tools (List[str]): Selected Gemini built-in tool names.
		tool_objects (List[Tool]): Provider-ready Gemini tool objects.
		file_search_store_names (List[str]): File Search store resource names.
		grounding_metadata (Any): Grounding metadata from the latest response.
	"""
	client: Optional[ genai.Client ]
	model: str
	prompt: str
	contents: List[ Content ]
	content_config: Optional[ GenerateContentConfig ]
	content_response: Optional[ GenerateContentResponse ]
	stream_response: Any
	output_text: str
	context: List[ Dict[ str, Any ] ]
	tools: List[ str ]
	tool_objects: List[ Tool ]
	file_search_store_names: List[ str ]
	grounding_metadata: Any
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Gemini text-generation configuration and runtime state without executing
			a provider request.
		
		Args:
			model (str): Default Gemini text-generation model.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.client = None
		self.model = model
		self.prompt = ''
		self.number = 1
		self.candidate_count = 1
		self.temperature = 0.0
		self.top_p = 0.0
		self.top_k = 0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_tokens = 0
		self.instructions = ''
		self.stops = [ ]
		self.response_mime_type = ''
		self.response_schema = None
		self.reasoning = ''
		self.thought_config = None
		self.media_resolution = ''
		self.response_modalities = [ ]
		self.tools = [ ]
		self.tool_objects = [ ]
		self.tool_choice = ''
		self.file_search_store_names = [ ]
		self.safety_profile = ''
		self.safety_settings = [ ]
		self.context = [ ]
		self.content_block = ''
		self.urls = [ ]
		self.max_urls = 10
		self.contents = [ ]
		self.content_config = None
		self.content_response = None
		self.stream_response = None
		self.stream = False
		self.stream_handler = None
		self.output_text = ''
		self.grounding_metadata = None
		self.config_values = { }
		self.sources = [ ]
		self.history = [ ]
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get model options.
		
		Purpose:
			Returns Gemini text-generation models exposed by the wrapper.
		
		Returns:
			List[str]: Available Gemini model identifiers.
		"""
		return [ 'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite-preview',
			'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash',
			'gemini-2.5-flash-lite', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Get tool options.
		
		Purpose:
			Returns Gemini built-in tools implemented by the wrapper.
		
		Returns:
			List[str]: Supported built-in tool names.
		"""
		return [ 'google_search', 'url_context', 'file_search', 'code_execution', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Get reasoning options.
		
		Purpose:
			Returns thinking-level values exposed by the wrapper.
		
		Returns:
			List[str]: Supported thinking-level values.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH', ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Get media-resolution options.
		
		Purpose:
			Returns media-resolution values exposed by the wrapper.
		
		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Get tool-choice options.
		
		Purpose:
			Returns function-calling modes retained for application control compatibility.
		
		Returns:
			List[str]: Supported function-calling mode values.
		"""
		return [ 'auto', 'any', 'none', 'validated', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Get include options.
		
		Purpose:
			Returns an empty collection because Gemini Generate Content does not use the OpenAI
			include-path request argument.
		
		Returns:
			List[str]: Empty include-option collection.
		"""
		return [ ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Get response-modality options.
		
		Purpose:
			Returns response modalities exposed by the wrapper.
		
		Returns:
			List[str]: Supported response-modality values.
		"""
		return [ 'text', 'image', 'audio', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get response-format options.
		
		Purpose:
			Returns MIME types supported by Gemini structured-output configuration.
		
		Returns:
			List[str]: Supported response MIME types.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum', ]
	
	def get_supported_tools( self, model: str ) -> List[ str ]:
		"""Get supported tools.
		
		Purpose:
			Returns the built-in tools implemented by the Gemini Chat wrapper.
		
		Args:
			model (str): Required Gemini model identifier.
		
		Returns:
			List[str]: Supported built-in tool names.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			return self.tool_options
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_supported_tools( self, model: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def build_urls( self, urls: Optional[ List[ str ] ] = None, max_urls: int = 10 ) -> List[
		str ]:
		"""Build URL collection.
		
		Purpose:
			Builds a bounded collection of nonempty reference URLs without modifying the
			caller-supplied collection.
		
		Args:
			urls (Optional[List[str]]): Optional reference URLs.
			max_urls (int): Maximum number of URLs retained.
		
		Returns:
			List[str]: Bounded reference URL collection.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.url_values = urls if urls is not None else [ ]
			self.max_urls = max_urls
			self.urls = [ ]
			
			for item in self.url_values:
				if item is None:
					continue
				
				self.url = str( item ).strip( )
				
				if self.url:
					self.urls.append( self.url )
			
			if self.max_urls > 0:
				self.urls = self.urls[ :self.max_urls ]
			
			return self.urls
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_urls( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def append_urls_to_content( self, content: str, urls: List[ str ] ) -> str:
		"""Append URLs to content.
		
		Purpose:
			Appends reference URLs to optional supplemental request content.
		
		Args:
			content (str): Supplemental request content.
			urls (List[str]): Reference URLs appended to the content.
		
		Returns:
			str: Combined supplemental content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.content_block = content
			self.urls = urls
			self.content_blocks = [ ]
			
			if self.content_block:
				self.content_blocks.append( self.content_block.strip( ) )
			
			if self.urls:
				self.content_blocks.append( 'Reference URLs:\n' + '\n'.join( self.urls ) )
			
			return '\n\n'.join( self.content_blocks )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'append_urls_to_content( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def build_modalities( self, modalities: Optional[ List[ str ] ] = None ) -> Optional[
		List[ str ] ]:
		"""Build response modalities.
		
		Purpose:
			Builds provider-ready uppercase response modality values.
		
		Args:
			modalities (Optional[List[str]]): Requested response modalities.
		
		Returns:
			Optional[List[str]]: Provider-ready modalities or None when none are selected.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.requested_modalities = (modalities if modalities is not None else [ ])
			self.response_modalities = [ ]
			for item in self.requested_modalities:
				self.modality = str( item ).strip( ).upper( )
				if self.modality in [ 'TEXT', 'IMAGE', 'AUDIO' ]:
					self.response_modalities.append( self.modality )
			
			if self.response_modalities:
				return self.response_modalities
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_modalities( self, **kwargs ) -> Optional[ List[ str ] ]'
			Logger( ).write( exception )
			raise exception
	
	def build_reasoning( self, reasoning: str = '' ) -> Optional[ ThinkingConfig ]:
		"""Build thinking configuration.
		
		Purpose:
			Builds a Gemini thinking configuration from the selected thinking level.
		
		Args:
			reasoning (str): Requested Gemini thinking level.
		
		Returns:
			Optional[ThinkingConfig]: Thinking configuration or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.reasoning = reasoning.strip( ).upper( )
			self.thought_config = None
			
			if not self.reasoning:
				return self.thought_config
			
			if self.reasoning == 'THINKING_LEVEL_UNSPECIFIED':
				return self.thought_config
			
			if self.reasoning not in [ 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH', ]:
				return self.thought_config
			
			self.thought_config = ThinkingConfig( thinking_level=self.reasoning, )
			return self.thought_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_reasoning( self, reasoning: str = "" ) -> ThinkingConfig'
			Logger( ).write( exception )
			raise exception
	
	def build_safety_settings( self, safety_profile: str = '' ) -> Optional[
		List[ SafetySetting ] ]:
		"""Build safety settings.
		
		Purpose:
			Builds consistent safety thresholds for supported Gemini harm categories.
		
		Args:
			safety_profile (str): Harm-block threshold enumeration name.
		
		Returns:
			Optional[List[SafetySetting]]: Safety settings or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.safety_profile = safety_profile.strip( ).upper( )
			self.safety_settings = [ ]
			
			if not self.safety_profile:
				return None
			
			self.threshold = getattr( HarmBlockThreshold, self.safety_profile, None, )
			if self.threshold is None:
				return None
			
			self.category_names = [ 'HARM_CATEGORY_HATE_SPEECH', 'HARM_CATEGORY_HARASSMENT',
				'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT',
				'HARM_CATEGORY_CIVIC_INTEGRITY', ]
			
			for name in self.category_names:
				self.category = getattr( HarmCategory, name, None, )
				if self.category is not None:
					self.safety_settings.append(
						SafetySetting( category=self.category, threshold=self.threshold, ) )
			
			if self.safety_settings:
				return self.safety_settings
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_safety_settings( self, **kwargs ) -> List[ SafetySetting ]'
			Logger( ).write( exception )
			raise exception
	
	def parse_response_schema( self, response_schema: Any ) -> Any:
		"""Parse response schema.
		
		Purpose:
			Converts a JSON schema string into a dictionary while preserving provider-ready
			dictionaries and typed schemas.
		
		Args:
			response_schema (Any): Response schema supplied by the application.
		
		Returns:
			Any: Parsed or unchanged response schema.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.response_schema = response_schema
			
			if self.response_schema is None:
				return None
			
			if not isinstance( self.response_schema, str ):
				return self.response_schema
			
			self.schema_text = self.response_schema.strip( )
			if not self.schema_text:
				return None
			
			return json.loads( self.schema_text )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('parse_response_schema( self, response_schema: Any ) -> Any')
			Logger( ).write( exception )
			raise exception
	
	def build_contents( self, prompt: str, content: str = '',
		context: Optional[ List[ Dict[ str, Any ] ] ] = None ) -> List[ Content ]:
		"""Build request contents.
		
		Purpose:
			Builds Gemini Content objects from application history, supplemental content, and
			the current required user prompt.
		
		Args:
			prompt (str): Required user prompt.
			content (str): Optional supplemental content.
			context (Optional[List[Dict[str, Any]]]): Prior conversation history.
		
		Returns:
			List[Content]: Provider-ready Gemini content objects.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.content_block = content
			self.context = context if context is not None else [ ]
			self.contents = [ ]
			
			for item in self.context:
				if isinstance( item, Content ):
					self.contents.append( item )
					continue
				
				if not isinstance( item, dict ):
					continue
				
				self.role = str( item.get( 'role', 'user' ) ).strip( ).lower( )
				self.message_text = item.get( 'content', '' )
				if not self.message_text:
					continue
				
				self.message_text = str( self.message_text ).strip( )
				if not self.message_text:
					continue
				
				if self.role == 'assistant':
					self.provider_role = 'model'
				else:
					self.provider_role = 'user'
				
				self.contents.append( Content( role=self.provider_role,
					parts=[ Part.from_text( text=self.message_text, ), ], ) )
			
			self.user_text = self.prompt
			if self.content_block:
				self.user_text = (f'{self.content_block}\n\n{self.prompt}')
			
			self.contents.append(
				Content( role='user', parts=[ Part.from_text( text=self.user_text, ), ], ) )
			return self.contents
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_contents( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: Optional[ List[ str ] ] = None,
		file_search_store_names: Optional[ List[ str ] ] = None ) -> Optional[ List[ Tool ] ]:
		"""Build Gemini tools.
		
		Purpose:
			Builds provider-ready Google Search, URL Context, File Search, and code-execution
			tools.
		
		Args:
			tools (Optional[List[str]]): Selected built-in tool names.
			file_search_store_names (Optional[List[str]]): File Search store resource names.
		
		Returns:
			Optional[List[Tool]]: Provider-ready tool objects or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.selected_tools = tools if tools is not None else [ ]
			self.file_search_store_names = (
				file_search_store_names if file_search_store_names is not None else [ ])
			self.tools = [ ]
			self.tool_objects = [ ]
			for item in self.selected_tools:
				self.tool_name = str( item ).strip( )
				if not self.tool_name:
					continue
				
				if self.tool_name not in self.tools:
					self.tools.append( self.tool_name )
			
			if 'google_search' in self.tools:
				self.tool_objects.append( Tool( google_search=GoogleSearch( ), ) )
			
			if 'url_context' in self.tools:
				self.tool_objects.append( Tool( url_context=UrlContext( ), ) )
			
			if 'file_search' in self.tools:
				throw_if( 'file_search_store_names', self.file_search_store_names, )
				self.tool_objects.append( Tool( file_search=FileSearch(
					file_search_store_names=(self.file_search_store_names), ), ) )
			
			if 'code_execution' in self.tools:
				self.tool_objects.append( Tool( code_execution=types.ToolCodeExecution( ), ) )
			
			if self.tool_objects:
				return self.tool_objects
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def build_tool_config( self, tool_choice: str, tools: List[ Tool ] ) -> Optional[ ToolConfig ]:
		"""Build function-calling configuration.
		
		Purpose:
			Builds a function-calling mode when provider tools include function declarations.
			Built-in Gemini tools do not require a function-calling configuration.
		
		Args:
			tool_choice (str): Requested function-calling mode.
			tools (List[Tool]): Provider-ready tools.
		
		Returns:
			Optional[ToolConfig]: Function-calling configuration or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.tool_choice = tool_choice.strip( ).upper( )
			self.tool_objects = tools
			
			if not self.tool_choice:
				return None
			
			if self.tool_choice == 'AUTO':
				return None
			
			if not self.tool_objects:
				return None
			
			self.has_function_declarations = False
			
			for tool in self.tool_objects:
				self.declarations = getattr( tool, 'function_declarations', None, )
				
				if self.declarations:
					self.has_function_declarations = True
					break
			
			if not self.has_function_declarations:
				return None
			
			if self.tool_choice not in [ 'ANY', 'NONE', 'VALIDATED', ]:
				return None
			
			return ToolConfig(
				function_calling_config=FunctionCallingConfig( mode=self.tool_choice, ), )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_tool_config( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def build_config( self, model: str, number: int = 1, temperature: float = 0.0,
		top_p: float = 0.0, top_k: int = 0, frequency: float = 0.0, presence: float = 0.0,
		max_tokens: int = 0, stops: Optional[ List[ str ] ] = None, instruct: str = '',
		response_format: str = '', tools: Optional[ List[ str ] ] = None, tool_choice: str = '',
		reasoning: str = '', modalities: Optional[ List[ str ] ]=None, media_resolution: str='',
		response_schema: Any = None, safety_profile: str = '',
		file_search_store_names: Optional[ List[ str ] ] = None ) -> GenerateContentConfig:
		"""Build generation configuration.
		
		Purpose:
			Builds a provider-ready GenerateContentConfig exclusively from arguments assigned
			to object members.
		
		Args:
			model (str): Required Gemini model identifier.
			number (int): Candidate count.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			stops (Optional[List[str]]): Stop sequences.
			instruct (str): System instruction.
			response_format (str): Response MIME type.
			tools (Optional[List[str]]): Selected built-in tools.
			tool_choice (str): Function-calling mode.
			reasoning (str): Thinking level.
			modalities (Optional[List[str]]): Response modalities.
			media_resolution (str): Media-resolution value.
			response_schema (Any): Structured-output schema.
			safety_profile (str): Harm-block threshold name.
			file_search_store_names (Optional[List[str]]): File Search stores.
		
		Returns:
			GenerateContentConfig: Provider-ready generation configuration.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.number = number
			self.candidate_count = number
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops if stops is not None else [ ]
			self.instructions = instruct
			self.response_mime_type = response_format
			self.tool_choice = tool_choice
			self.reasoning = reasoning
			self.media_resolution = media_resolution
			self.response_schema = self.parse_response_schema( response_schema )
			self.tool_objects = self.build_tools( tools, file_search_store_names, )
			self.function_tool_config = self.build_tool_config( self.tool_choice,
				self.tool_objects if self.tool_objects is not None else [ ], )
			self.response_modalities = self.build_modalities( modalities )
			self.thought_config = self.build_reasoning( self.reasoning )
			self.safety_settings = self.build_safety_settings( safety_profile )
			self.config_values = { }
			if self.candidate_count > 0:
				self.config_values[ 'candidate_count' ] = (self.candidate_count)
			
			self.config_values[ 'temperature' ] = self.temperature
			if self.top_p > 0:
				self.config_values[ 'top_p' ] = self.top_p
			
			if self.top_k > 0:
				self.config_values[ 'top_k' ] = self.top_k
			
			if self.frequency_penalty != 0:
				self.config_values[ 'frequency_penalty' ] = (self.frequency_penalty)
			
			if self.presence_penalty != 0:
				self.config_values[ 'presence_penalty' ] = (self.presence_penalty)
			
			if self.max_tokens > 0:
				self.config_values[ 'max_output_tokens' ] = (self.max_tokens)
			
			if self.stops:
				self.config_values[ 'stop_sequences' ] = self.stops
			
			if self.instructions:
				self.config_values[ 'system_instruction' ] = (self.instructions)
			
			if self.response_mime_type:
				self.config_values[ 'response_mime_type' ] = (self.response_mime_type)
			
			if self.response_schema is not None:
				if isinstance( self.response_schema, dict ):
					self.config_values[ 'response_json_schema' ] = (self.response_schema)
				else:
					self.config_values[ 'response_schema' ] = (self.response_schema)
			
			if self.media_resolution:
				self.config_values[ 'media_resolution' ] = (self.media_resolution)
			
			if self.tool_objects:
				self.config_values[ 'tools' ] = self.tool_objects
			
			if self.function_tool_config is not None:
				self.config_values[ 'tool_config' ] = (self.function_tool_config)
			
			if self.safety_settings:
				self.config_values[ 'safety_settings' ] = (self.safety_settings)
			
			if self.response_modalities:
				self.config_values[ 'response_modalities' ] = (self.response_modalities)
			
			if self.thought_config is not None:
				self.config_values[ 'thinking_config' ] = (self.thought_config)
			
			self.content_config = GenerateContentConfig( **self.config_values )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'build_config( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get output text.
		
		Purpose:
			Extracts generated text from the latest synchronous Gemini response.
		
		Returns:
			str: Generated response text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_text = ''
			
			if self.content_response is None:
				return self.output_text
			
			self.response_text = getattr( self.content_response, 'text', '', )
			
			if self.response_text:
				self.output_text = str( self.response_text )
				return self.output_text
			
			self.candidates = getattr( self.content_response, 'candidates', [ ], ) or [ ]
			self.text_blocks = [ ]
			for candidate in self.candidates:
				self.response_content = getattr( candidate, 'content', None, )
				if self.response_content is None:
					continue
				
				self.parts = getattr( self.response_content, 'parts', [ ], ) or [ ]
				for part in self.parts:
					self.part_text = getattr( part, 'text', '', )
					if self.part_text:
						self.text_blocks.append( str( self.part_text ) )
			
			self.output_text = ''.join( self.text_blocks ).strip( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_output_text( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def capture_grounding_metadata( self ) -> None:
		"""Capture grounding metadata.
		
		Purpose:
			Captures grounding metadata from the latest synchronous Gemini response.
		
		Returns:
			None: This method updates object state.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.grounding_metadata = None
			
			if self.content_response is None:
				return
			
			self.candidates = getattr( self.content_response, 'candidates', [ ], ) or [ ]
			
			for candidate in self.candidates:
				self.metadata = getattr( candidate, 'grounding_metadata', None, )
				
				if self.metadata is not None:
					self.grounding_metadata = self.metadata
					return
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'capture_grounding_metadata( self ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def get_grounding_sources( self ) -> List[ Dict[ str, str ] ]:
		"""Get grounding sources.
		
		Purpose:
			Extracts web source titles and URLs from captured Gemini grounding metadata.
		
		Returns:
			List[Dict[str, str]]: Grounding source records.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.sources = [ ]
			if self.grounding_metadata is None:
				return self.sources
			
			self.chunks = getattr( self.grounding_metadata, 'grounding_chunks', [ ], ) or [ ]
			for chunk in self.chunks:
				self.web_source = getattr( chunk, 'web', None, )
				if self.web_source is None:
					continue
				
				self.uri = getattr( self.web_source, 'uri', '', )
				self.title = getattr( self.web_source, 'title', '', )
				if self.uri:
					self.sources.append( { 'title': str( self.title or self.uri ),
						'url': str( self.uri ), 'snippet': '', } )
			
			return self.sources
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_grounding_sources( self ) -> List[ Dict[ str, str ] ]'
			Logger( ).write( exception )
			raise exception
	
	def get_structured_history( self ) -> Optional[ List[ Content ] ]:
		"""Get structured history.
		
		Purpose:
			Returns the request contents and latest model response as Gemini Content objects.
		
		Returns:
			Optional[List[Content]]: Structured history or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.history = [ ]
			for item in self.contents:
				if isinstance( item, Content ):
					self.history.append( item )
			
			if self.content_response is not None:
				self.candidates = getattr( self.content_response, 'candidates', [ ], ) or [ ]
				for candidate in self.candidates:
					self.response_content = getattr( candidate, 'content', None, )
					if isinstance( self.response_content, Content ):
						self.history.append( self.response_content )
						break
			
			if self.history:
				return self.history
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_structured_history( self ) -> Optional[ List[ Content ] ]'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str, number: int = 1, temperature: float = 0.0,
		top_p: float = 0.0, top_k: int = 0, frequency: float = 0.0, presence: float = 0.0,
		max_tokens: int = 0, stops: Optional[ List[ str ] ] = None, instruct: str = '',
		response_format: str = '', tools: Optional[ List[ str ] ] = None, tool_choice: str = '',
		reasoning: str = '', modalities: Optional[ List[ str ] ] = None, media_resolution: str='',
		context: Optional[ List[ Dict[ str, Any ] ] ] = None, content: str = '',
		urls: Optional[ List[ str ] ] = None, max_urls: int = 10, response_schema: Any = None,
		safety_profile: str = '', file_search_store_names: Optional[ List[ str ] ] = None,
		stream: bool = False, stream_handler: Any = None ) -> Any:
		"""Generate text.
		
		Purpose:
			Executes synchronous or streaming Gemini text generation using content and
			configuration objects built from assigned wrapper members.
		
		Args:
			prompt (str): Required user prompt.
			model (str): Required Gemini model identifier.
			number (int): Candidate count.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			stops (Optional[List[str]]): Stop sequences.
			instruct (str): System instruction.
			response_format (str): Response MIME type.
			tools (Optional[List[str]]): Selected built-in tools.
			tool_choice (str): Function-calling mode.
			reasoning (str): Thinking level.
			modalities (Optional[List[str]]): Response modalities.
			media_resolution (str): Media-resolution value.
			context (Optional[List[Dict[str, Any]]]): Conversation history.
			content (str): Supplemental request content.
			urls (Optional[List[str]]): Reference URLs.
			max_urls (int): Maximum number of reference URLs.
			response_schema (Any): Structured-output schema.
			safety_profile (str): Harm-block threshold name.
			file_search_store_names (Optional[List[str]]): File Search stores.
			stream (bool): Indicates whether streaming is enabled.
			stream_handler (Any): Optional callable receiving each text delta.
		
		Returns:
			Any: Generated text or the provider stream iterator when no handler is supplied.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops if stops is not None else [ ]
			self.instructions = instruct
			self.response_mime_type = response_format
			self.tools = tools if tools is not None else [ ]
			self.tool_choice = tool_choice
			self.reasoning = reasoning
			self.response_modalities = (modalities if modalities is not None else [ ])
			self.media_resolution = media_resolution
			self.context = context if context is not None else [ ]
			self.content_block = content
			self.urls = urls if urls is not None else [ ]
			self.max_urls = max_urls
			self.response_schema = response_schema
			self.safety_profile = safety_profile
			self.file_search_store_names = (
				file_search_store_names if file_search_store_names is not None else [ ])
			self.stream = stream
			self.stream_handler = stream_handler
			self.urls = self.build_urls( self.urls, self.max_urls, )
			self.content_block = self.append_urls_to_content( self.content_block, self.urls, )
			self.contents = self.build_contents( self.prompt, self.content_block, self.context, )
			self.content_config = self.build_config( self.model, self.number, self.temperature,
				self.top_p, self.top_k, self.frequency_penalty, self.presence_penalty,
				self.max_tokens, self.stops, self.instructions, self.response_mime_type, self.tools,
				self.tool_choice, self.reasoning, self.response_modalities, self.media_resolution,
				self.response_schema, self.safety_profile, self.file_search_store_names, )
			self.client = genai.Client( api_key=self.gemini_api_key, )
			if self.stream:
				self.stream_response = (
					self.client.models.generate_content_stream( model=self.model,
						contents=self.contents, config=self.content_config, ))
				
				if self.stream_handler is None:
					return self.stream_response
				
				self.text_blocks = [ ]
				for chunk in self.stream_response:
					self.chunk_text = getattr( chunk, 'text', '', )
					if not self.chunk_text:
						continue
					
					self.chunk_text = str( self.chunk_text )
					self.text_blocks.append( self.chunk_text )
					self.stream_handler( self.chunk_text )
				
				self.output_text = ''.join( self.text_blocks ).strip( )
				return self.output_text
			
			self.content_response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config, )
			self.capture_grounding_metadata( )
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Gemini Chat wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'gemini_api_key', 'google_api_key', 'client', 'model', 'prompt', 'number',
			'candidate_count', 'temperature', 'top_p', 'top_k', 'frequency_penalty',
			'presence_penalty', 'max_tokens', 'instructions', 'stops', 'response_mime_type',
			'response_schema', 'reasoning', 'thought_config', 'media_resolution',
			'response_modalities', 'tools', 'tool_objects', 'tool_choice',
			'file_search_store_names', 'safety_profile', 'safety_settings', 'context',
			'content_block', 'urls', 'max_urls', 'contents', 'content_config', 'content_response',
			'stream_response', 'stream', 'stream_handler', 'output_text', 'grounding_metadata',
			'model_options', 'tool_options', 'reasoning_options', 'media_options',
			'choice_options', 'include_options', 'modality_options', 'format_options',
			'get_supported_tools', 'build_urls', 'append_urls_to_content', 'build_modalities',
			'build_reasoning', 'build_safety_settings', 'parse_response_schema', 'build_contents',
			'build_tools', 'build_tool_config', 'build_config', 'get_output_text',
			'capture_grounding_metadata', 'get_grounding_sources', 'get_structured_history',
			'generate_text', ]

class Images( Gemini ):
	"""Provide Gemini image workflow support.
	
	Purpose:
		Provides image generation, image analysis, and conversational image editing through
		the Google Gen AI SDK. The class assigns accepted method arguments to object members,
		builds provider-native image and generation configurations, executes Gemini requests,
		and extracts generated images, response text, and grounding metadata.
	
	Attributes:
		client (Optional[genai.Client]): Google Gen AI client.
		model (str): Gemini model used by the current operation.
		prompt (str): Prompt used by the current operation.
		file_path (str): Local source-image path.
		aspect_ratio (str): Requested output-image aspect ratio.
		image_size (str): Requested output-image resolution.
		output_mime_type (str): Requested generated-image MIME type.
		response_modalities (List[str]): Requested response modalities.
		temperature (float): Sampling temperature.
		top_p (float): Nucleus-sampling value.
		max_output_tokens (int): Maximum output-token count.
		instructions (str): Optional system instruction.
		grounded (bool): Indicates whether Google Search grounding is enabled.
		image_search (bool): Indicates whether Google Image Search is enabled.
		image_config (Optional[ImageConfig]): Provider image configuration.
		content_config (Optional[GenerateContentConfig]): Provider request configuration.
		content_response (Optional[GenerateContentResponse]): Latest provider response.
		output_image (Optional[PIL.Image.Image]): First generated image.
		output_text (str): Text extracted from the latest response.
		grounding_metadata (Any): Grounding metadata from the latest response.
	"""
	client: Optional[ genai.Client ]
	model: str
	prompt: str
	file_path: str
	aspect_ratio: str
	image_size: str
	output_mime_type: str
	response_modalities: List[ str ]
	temperature: float
	top_p: float
	max_output_tokens: int
	instructions: str
	grounded: bool
	image_search: bool
	image_config: Optional[ ImageConfig ]
	content_config: Optional[ GenerateContentConfig ]
	content_response: Optional[ GenerateContentResponse ]
	output_image: Optional[ PIL.Image.Image ]
	output_text: str
	grounding_metadata: Any
	
	def __init__( self, model: str = 'gemini-2.5-flash-image' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Gemini image configuration and runtime state without executing a
			provider request.
		
		Args:
			model (str): Default Gemini image model.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.client = None
		self.model = model
		self.prompt = ''
		self.file_path = ''
		self.aspect_ratio = ''
		self.image_size = ''
		self.output_mime_type = ''
		self.response_modalities = [ ]
		self.temperature = 0.0
		self.top_p = 0.0
		self.max_output_tokens = 0
		self.instructions = ''
		self.grounded = False
		self.image_search = False
		self.image_config = None
		self.content_config = None
		self.content_response = None
		self.response = None
		self.output_image = None
		self.output_text = ''
		self.grounding_metadata = None
		self.image_input = None
		self.tool = None
		self.tools = [ ]
		self.config_values = { }
		self.parts = [ ]
		self.candidates = [ ]
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get image-model options.
		
		Purpose:
			Returns Gemini models exposed for image generation and conversational editing.
		
		Returns:
			List[str]: Supported Gemini image model identifiers.
		"""
		return [ 'gemini-3.1-flash-image', 'gemini-3.1-flash-lite-image',
			'gemini-3-pro-image-preview', 'gemini-2.5-flash-image', ]
	
	@property
	def analysis_model_options( self ) -> List[ str ]:
		"""Get image-analysis model options.
		
		Purpose:
			Returns Gemini multimodal models exposed for image analysis.
		
		Returns:
			List[str]: Supported Gemini image-analysis model identifiers.
		"""
		return [ 'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite-preview',
			'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash',
			'gemini-2.5-flash-lite',
			'gemini-2.0-flash', 'gemini-2.0-flash-lite', ]
	
	@property
	def aspect_options( self ) -> List[ str ]:
		"""Get aspect-ratio options.
		
		Purpose:
			Returns output-image aspect ratios exposed by the wrapper.
		
		Returns:
			List[str]: Supported aspect-ratio values.
		"""
		return [ '1:1', '1:4', '1:8', '2:3', '3:2', '3:4', '4:1', '4:3', '4:5', '5:4', '8:1',
			'9:16', '16:9', '21:9', ]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Get image-size options.
		
		Purpose:
			Returns image-resolution values exposed by the wrapper.
		
		Returns:
			List[str]: Supported image-size values.
		"""
		return [ '512', '1K', '2K', '4K', ]
	
	@property
	def resolution_options( self ) -> List[ str ]:
		"""Get image-resolution options.
		
		Purpose:
			Returns the same provider image-resolution values exposed by size options.
		
		Returns:
			List[str]: Supported image-resolution values.
		"""
		return self.size_options
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Get response-modality options.
		
		Purpose:
			Returns response-modality selections supported by Gemini image workflows.
		
		Returns:
			List[str]: Supported response-modality selections.
		"""
		return [ 'image', 'text', 'text_and_image', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get output MIME-type options.
		
		Purpose:
			Returns generated-image MIME types exposed by the wrapper.
		
		Returns:
			List[str]: Supported image MIME types.
		"""
		return [ 'image/png', 'image/jpeg', 'image/webp', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Get grounding-tool options.
		
		Purpose:
			Returns Google Search grounding tools implemented by the wrapper.
		
		Returns:
			List[str]: Supported grounding-tool names.
		"""
		return [ 'google_search', 'image_search', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Get include options.
		
		Purpose:
			Returns an empty collection because Gemini image requests do not use OpenAI
			include-path arguments.
		
		Returns:
			List[str]: Empty include-option collection.
		"""
		return [ ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Get tool-choice options.
		
		Purpose:
			Returns an empty collection because Gemini image grounding tools do not use the
			function-calling choice control.
		
		Returns:
			List[str]: Empty tool-choice collection.
		"""
		return [ ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Get reasoning options.
		
		Purpose:
			Returns an empty collection because reasoning for Gemini image models is managed
			by the provider rather than this wrapper.
		
		Returns:
			List[str]: Empty reasoning-option collection.
		"""
		return [ ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get response-format options.
		
		Purpose:
			Returns generated-image MIME types exposed by the wrapper.
		
		Returns:
			List[str]: Supported response-format values.
		"""
		return self.mime_options
	
	def supports_image_size( self, model: str ) -> bool:
		"""Determine image-size support.
		
		Purpose:
			Determines whether a required Gemini image model supports an explicit image-size
			configuration.
		
		Args:
			model (str): Required Gemini image model identifier.
		
		Returns:
			bool: True when the model supports explicit image sizes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			return self.model in [ 'gemini-3.1-flash-image', 'gemini-3.1-flash-lite-image',
				'gemini-3-pro-image-preview', ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('supports_image_size( self, model: str ) -> bool')
			Logger( ).write( exception )
			raise exception
	
	def supports_search_grounding( self, model: str ) -> bool:
		"""Determine search-grounding support.
		
		Purpose:
			Determines whether a required Gemini image model supports Google Search grounding.
		
		Args:
			model (str): Required Gemini image model identifier.
		
		Returns:
			bool: True when Google Search grounding is supported.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			return self.model in [ 'gemini-3.1-flash-image', 'gemini-3-pro-image-preview', ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('supports_search_grounding( self, model: str ) -> bool')
			Logger( ).write( exception )
			raise exception
	
	def supports_image_search( self, model: str ) -> bool:
		"""Determine Image Search support.
		
		Purpose:
			Determines whether a required Gemini image model supports Google Image Search
			grounding.
		
		Args:
			model (str): Required Gemini image model identifier.
		
		Returns:
			bool: True when Google Image Search grounding is supported.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			return self.model == 'gemini-3.1-flash-image'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('supports_image_search( self, model: str ) -> bool')
			Logger( ).write( exception )
			raise exception
	
	def build_response_modalities( self, response_modalities: str, image_only: bool = False ) -> \
	List[ str ]:
		"""Build response modalities.
		
		Purpose:
			Builds provider-ready Gemini response modalities from a required application
			selection.
		
		Args:
			response_modalities (str): Required application response-modality selection.
			image_only (bool): Indicates whether image output must be requested.
		
		Returns:
			List[str]: Provider-ready response modalities.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response_modalities', response_modalities )
			self.response_mode = response_modalities
			self.image_only = image_only
			self.response_modalities = [ ]
			if self.response_mode == 'text_and_image':
				self.response_modalities = [ 'TEXT', 'IMAGE', ]
			elif self.response_mode == 'text':
				self.response_modalities = [ 'TEXT', ]
			elif self.response_mode == 'image':
				self.response_modalities = [ 'IMAGE', ]
			
			if self.image_only:
				self.response_modalities = [ 'IMAGE', ]
			
			throw_if( 'response_modalities', self.response_modalities, )
			return self.response_modalities
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('build_response_modalities( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def build_grounding_tool( self, model: str, image_search: bool = False ) -> Optional[ Tool ]:
		"""Build grounding tool.
		
		Purpose:
			Builds a Google Search grounding tool for a required compatible Gemini image model.
		
		Args:
			model (str): Required Gemini image model identifier.
			image_search (bool): Indicates whether Google Image Search is enabled.
		
		Returns:
			Optional[Tool]: Provider-ready grounding tool or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.image_search = image_search
			self.tool = None
			if not self.supports_search_grounding( self.model ):
				return self.tool
			
			if self.image_search:
				if self.supports_image_search( self.model ):
					self.tool = Tool( google_search=GoogleSearch(
						search_types=types.SearchTypes( web_search=types.WebSearch( ),
							image_search=types.ImageSearch( ), ), ), )
					return self.tool
			
			self.tool = Tool( google_search=GoogleSearch( ), )
			return self.tool
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('build_grounding_tool( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def build_image_config( self, model: str, aspect_ratio: str = '', image_size: str = '',
		output_mime_type: str = '' ) -> Optional[ ImageConfig ]:
		"""Build image configuration.
		
		Purpose:
			Builds provider-native output-image configuration for a required Gemini image model.
		
		Args:
			model (str): Required Gemini image model identifier.
			aspect_ratio (str): Optional output-image aspect ratio.
			image_size (str): Optional output-image resolution.
			output_mime_type (str): Optional output-image MIME type.
		
		Returns:
			Optional[ImageConfig]: Provider image configuration or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.aspect_ratio = aspect_ratio
			self.image_size = image_size
			self.output_mime_type = output_mime_type
			self.image_values = { }
			if self.aspect_ratio:
				self.image_values[ 'aspect_ratio' ] = (self.aspect_ratio)
			
			if self.image_size:
				if self.supports_image_size( self.model ):
					self.image_values[ 'image_size' ] = (self.image_size)
			
			if self.output_mime_type:
				self.image_values[ 'output_mime_type' ] = (self.output_mime_type)
			
			if not self.image_values:
				self.image_config = None
				return self.image_config
			
			self.image_config = ImageConfig( **self.image_values )
			return self.image_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('build_image_config( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def build_content_config( self, model: str, response_modalities: str, image_only: bool = False,
		aspect_ratio: str = '', image_size: str = '', output_mime_type: str = '',
		temperature: float = 0.0, top_p: float = 0.0, max_tokens: int = 0, instruct: str = '',
		grounded: bool = False, image_search: bool = False ) -> GenerateContentConfig:
		"""Build content configuration.
		
		Purpose:
			Builds a provider-ready GenerateContentConfig from arguments assigned to object
			members.
		
		Args:
			model (str): Required Gemini model identifier.
			response_modalities (str): Required response-modality selection.
			image_only (bool): Indicates whether image output must be requested.
			aspect_ratio (str): Optional output-image aspect ratio.
			image_size (str): Optional output-image resolution.
			output_mime_type (str): Optional output-image MIME type.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction.
			grounded (bool): Indicates whether Google Search grounding is enabled.
			image_search (bool): Indicates whether Google Image Search is enabled.
		
		Returns:
			GenerateContentConfig: Provider-ready generation configuration.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			throw_if( 'response_modalities', response_modalities, )
			self.model = model
			self.response_mode = response_modalities
			self.image_only = image_only
			self.aspect_ratio = aspect_ratio
			self.image_size = image_size
			self.output_mime_type = output_mime_type
			self.temperature = temperature
			self.top_p = top_p
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.grounded = grounded
			self.image_search = image_search
			self.response_modalities = (
				self.build_response_modalities( self.response_mode, self.image_only, ))
			self.image_config = self.build_image_config( self.model, self.aspect_ratio,
				self.image_size, self.output_mime_type, )
			self.tools = [ ]
			if self.grounded:
				self.tool = self.build_grounding_tool( self.model, self.image_search, )
				if self.tool is not None:
					self.tools.append( self.tool )
			
			self.config_values = { 'response_modalities': self.response_modalities,
				'temperature': self.temperature, }
			
			if self.top_p > 0:
				self.config_values[ 'top_p' ] = self.top_p
			
			if self.max_output_tokens > 0:
				self.config_values[ 'max_output_tokens' ] = (self.max_output_tokens)
			
			if self.instructions:
				self.config_values[ 'system_instruction' ] = (self.instructions)
			
			if self.image_config is not None:
				self.config_values[ 'image_config' ] = (self.image_config)
			
			if self.tools:
				self.config_values[ 'tools' ] = self.tools
			
			self.content_config = GenerateContentConfig( **self.config_values )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('build_content_config( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def open_image( self, path: str ) -> PIL.Image.Image:
		"""Open an image.
		
		Purpose:
			Loads a required local image and returns an independent PIL image object.
		
		Args:
			path (str): Required local image path.
		
		Returns:
			PIL.Image.Image: Loaded image.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			with PIL.Image.open( self.file_path ) as source:
				self.image_input = source.copy( )
			
			return self.image_input
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('open_image( self, path: str ) -> PIL.Image.Image')
			Logger( ).write( exception )
			raise exception
	
	def capture_metadata( self ) -> None:
		"""Capture grounding metadata.
		
		Purpose:
			Captures grounding metadata from the latest Gemini response.
		
		Returns:
			None: This method updates object state.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.grounding_metadata = None
			if self.content_response is None:
				return
			
			self.candidates = getattr( self.content_response, 'candidates', [ ], ) or [ ]
			for candidate in self.candidates:
				self.metadata = getattr( candidate, 'grounding_metadata', None, )
				if self.metadata is not None:
					self.grounding_metadata = self.metadata
					return
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('capture_metadata( self ) -> None')
			Logger( ).write( exception )
			raise exception
	
	def get_first_image( self ) -> Optional[ PIL.Image.Image ]:
		"""Get first generated image.
		
		Purpose:
			Extracts the first generated inline image from the latest Gemini response.
		
		Returns:
			Optional[PIL.Image.Image]: First generated image or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_image = None
			if self.content_response is None:
				return self.output_image
			
			self.candidates = getattr( self.content_response, 'candidates', [ ], ) or [ ]
			for candidate in self.candidates:
				self.content = getattr( candidate, 'content', None, )
				if self.content is None:
					continue
				
				self.parts = getattr( self.content, 'parts', [ ], ) or [ ]
				for part in self.parts:
					self.inline_data = getattr( part, 'inline_data', None, )
					if self.inline_data is None:
						continue
					
					self.image_data = getattr( self.inline_data, 'data', None, )
					if not self.image_data:
						continue
					
					self.output_image = PIL.Image.open( io.BytesIO( self.image_data ) )
					self.output_image.load( )
					return self.output_image
			
			return self.output_image
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('get_first_image( self ) -> Optional[ PIL.Image.Image ]')
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get response text.
		
		Purpose:
			Extracts text from the latest Gemini image or image-analysis response.
		
		Returns:
			str: Extracted response text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_text = ''
			
			if self.content_response is None:
				return self.output_text
			
			self.response_text = getattr( self.content_response, 'text', '', )
			if self.response_text:
				self.output_text = str( self.response_text )
				return self.output_text
			
			self.text_parts = [ ]
			self.candidates = getattr( self.content_response, 'candidates', [ ], ) or [ ]
			for candidate in self.candidates:
				self.content = getattr( candidate, 'content', None, )
				if self.content is None:
					continue
				
				self.parts = getattr( self.content, 'parts', [ ], ) or [ ]
				for part in self.parts:
					self.part_text = getattr( part, 'text', '', )
					if self.part_text:
						self.text_parts.append( str( self.part_text ) )
			
			self.output_text = ''.join( self.text_parts ).strip( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('get_output_text( self ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, model: str, aspect_ratio: str = '1:1', image_size: str = '1K',
		output_mime_type: str = 'image/png', response_modalities: str = 'image',
		temperature: float = 0.0, top_p: float = 0.0, max_tokens: int = 0, instruct: str = '',
		grounded: bool = False, image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""Generate an image.
		
		Purpose:
			Generates an image from a required prompt using a required Gemini image model.
		
		Args:
			prompt (str): Required image-generation prompt.
			model (str): Required Gemini image model.
			aspect_ratio (str): Output-image aspect ratio.
			image_size (str): Output-image resolution.
			output_mime_type (str): Output-image MIME type.
			response_modalities (str): Response-modality selection.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction.
			grounded (bool): Indicates whether Google Search grounding is enabled.
			image_search (bool): Indicates whether Google Image Search is enabled.
		
		Returns:
			Optional[PIL.Image.Image]: Generated image or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'response_modalities', response_modalities, )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.prompt = prompt
			self.model = model
			self.aspect_ratio = aspect_ratio
			self.image_size = image_size
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities
			self.temperature = temperature
			self.top_p = top_p
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.grounded = grounded
			self.image_search = image_search
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.content_config = self.build_content_config( self.model, self.response_mode, True,
				self.aspect_ratio, self.image_size, self.output_mime_type, self.temperature,
				self.top_p, self.max_output_tokens, self.instructions, self.grounded,
				self.image_search, )
			self.content_response = (
				self.client.models.generate_content( model=self.model, contents=[ self.prompt, ],
					config=self.content_config, ))
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'generate( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str, temperature: float = 0.0,
		top_p: float = 0.0, max_tokens: int = 0, instruct: str = '', media_resolution: str = '',
		grounded: bool = False ) -> str:
		"""Analyze an image.
		
		Purpose:
			Analyzes a required local image using a required Gemini multimodal model.
		
		Args:
			prompt (str): Required image-analysis prompt.
			path (str): Required local image path.
			model (str): Required Gemini multimodal model.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction.
			media_resolution (str): Optional input-media resolution.
			grounded (bool): Indicates whether Google Search grounding is enabled.
		
		Returns:
			str: Image-analysis response text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.prompt = prompt
			self.file_path = path
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.media_resolution = media_resolution
			self.grounded = grounded
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.image_input = self.open_image( self.file_path )
			self.config_values = { 'response_modalities': [ 'TEXT', ],
				'temperature': self.temperature, }
			
			if self.top_p > 0:
				self.config_values[ 'top_p' ] = self.top_p
			
			if self.max_output_tokens > 0:
				self.config_values[ 'max_output_tokens' ] = (self.max_output_tokens)
			
			if self.instructions:
				self.config_values[ 'system_instruction' ] = (self.instructions)
			
			if self.media_resolution:
				self.config_values[ 'media_resolution' ] = (self.media_resolution)
			
			if self.grounded:
				self.tool = self.build_grounding_tool( self.model, False, )
				
				if self.tool is not None:
					self.config_values[ 'tools' ] = [ self.tool, ]
			
			self.content_config = GenerateContentConfig( **self.config_values )
			self.content_response = (self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.image_input, ], config=self.content_config, ))
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'analyze( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str, aspect_ratio: str = '',
		image_size: str = '', output_mime_type: str = 'image/png',
		response_modalities: str = 'image', temperature: float = 0.0, top_p: float = 0.0,
		max_tokens: int = 0, instruct: str = '', grounded: bool = False,
		image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""Edit an image.
		
		Purpose:
			Edits a required local image through conversational Gemini image generation using
			a required editing prompt and image model.
		
		Args:
			prompt (str): Required image-editing instruction.
			path (str): Required local source-image path.
			model (str): Required Gemini image model.
			aspect_ratio (str): Optional output-image aspect ratio.
			image_size (str): Optional output-image resolution.
			output_mime_type (str): Output-image MIME type.
			response_modalities (str): Response-modality selection.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction.
			grounded (bool): Indicates whether Google Search grounding is enabled.
			image_search (bool): Indicates whether Google Image Search is enabled.
		
		Returns:
			Optional[PIL.Image.Image]: Edited image or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'response_modalities', response_modalities, )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.prompt = prompt
			self.file_path = path
			self.model = model
			self.aspect_ratio = aspect_ratio
			self.image_size = image_size
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities
			self.temperature = temperature
			self.top_p = top_p
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.grounded = grounded
			self.image_search = image_search
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.image_input = self.open_image( self.file_path )
			self.content_config = self.build_content_config( self.model, self.response_mode, True,
				self.aspect_ratio, self.image_size, self.output_mime_type, self.temperature,
				self.top_p, self.max_output_tokens, self.instructions, self.grounded,
				self.image_search, )
			self.content_response = (self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.image_input, ], config=self.content_config, ))
			self.response = self.content_response
			self.capture_metadata( )
			return self.get_first_image( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'edit( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Gemini Images wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'gemini_api_key', 'client', 'model', 'prompt', 'file_path', 'aspect_ratio',
			'image_size', 'output_mime_type', 'response_modalities', 'temperature', 'top_p',
			'max_output_tokens', 'instructions', 'grounded', 'image_search', 'image_config',
			'content_config', 'content_response', 'response', 'output_image', 'output_text',
			'grounding_metadata', 'model_options', 'analysis_model_options', 'aspect_options',
			'size_options', 'resolution_options', 'modality_options', 'mime_options',
			'tool_options', 'include_options', 'choice_options', 'reasoning_options',
			'format_options', 'supports_image_size', 'supports_search_grounding',
			'supports_image_search', 'build_response_modalities', 'build_grounding_tool',
			'build_image_config', 'build_content_config', 'open_image', 'capture_metadata',
			'get_first_image', 'get_output_text', 'generate', 'analyze', 'edit', ]

class Embeddings( Gemini ):
	"""Provide Gemini embedding workflow support.
	
	Purpose:
		Provides embedding generation through the Google Gen AI SDK for individual text values
		and batches of text. The class assigns accepted method arguments to object members,
		constructs provider-native EmbedContentConfig objects, executes embedding requests, and
		extracts floating-point vectors and usage metadata from provider responses.
	
	Attributes:
		client (Optional[genai.Client]): Google Gen AI client.
		model (str): Gemini embedding model used by the current request.
		input_text (str | List[str]): Original embedding input.
		contents (str | List[str]): Provider-ready embedding content.
		dimensions (int): Requested output dimensionality.
		task_type (str): Embedding task type.
		title (str): Optional retrieval-document title.
		embedding_config (Optional[EmbedContentConfig]): Provider embedding configuration.
		response (Any): Latest provider embedding response.
		embedding (Optional[List[float] | List[List[float]]]): Extracted embedding output.
		embeddings (List[List[float]]): Extracted embedding collection.
		usage (Any): Usage metadata from the latest provider response.
	"""
	client: Optional[ genai.Client ]
	model: str
	input_text: str | List[ str ]
	contents: str | List[ str ]
	dimensions: int
	task_type: str
	title: str
	embedding_config: Optional[ EmbedContentConfig ]
	response: Any
	embedding: Optional[ List[ float ] | List[ List[ float ] ] ]
	embeddings: List[ List[ float ] ]
	usage: Any
	
	def __init__( self, model: str = 'gemini-embedding-001' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Gemini embedding configuration and runtime state without executing a
			provider request.
		
		Args:
			model (str): Default Gemini embedding model.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.client = None
		self.model = model
		self.input_text = ''
		self.contents = ''
		self.dimensions = 0
		self.task_type = ''
		self.title = ''
		self.embedding_config = None
		self.response = None
		self.embedding = None
		self.embeddings = [ ]
		self.usage = None
		self.config_values = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get embedding-model options.
		
		Purpose:
			Returns Gemini embedding models exposed by the wrapper.
		
		Returns:
			List[str]: Supported Gemini embedding model identifiers.
		"""
		return [ 'gemini-embedding-2', 'gemini-embedding-2-preview', 'gemini-embedding-001',
			'text-embedding-004', 'text-multilingual-embedding-002', ]
	
	@property
	def task_options( self ) -> List[ str ]:
		"""Get task-type options.
		
		Purpose:
			Returns Gemini embedding task types exposed by the wrapper.
		
		Returns:
			List[str]: Supported embedding task-type values.
		"""
		return [ '', 'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY',
			'CLASSIFICATION', 'CLUSTERING', 'QUESTION_ANSWERING', 'FACT_VERIFICATION',
			'CODE_RETRIEVAL_QUERY', ]
	
	@property
	def dimension_options( self ) -> List[ int ]:
		"""Get output-dimensionality options.
		
		Purpose:
			Returns commonly used output dimensions exposed by the application. A value of zero
			uses the selected model's default dimensionality.
		
		Returns:
			List[int]: Available output-dimensionality values.
		"""
		return [ 0, 128, 256, 512, 768, 1536, 3072, ]
	
	def supports_dimensions( self, model: str ) -> bool:
		"""Determine output-dimensionality support.
		
		Purpose:
			Determines whether a required embedding model supports the provider's
			output_dimensionality configuration.
		
		Args:
			model (str): Required Gemini embedding model identifier.
		
		Returns:
			bool: True when the selected model supports configurable output dimensions.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			
			return self.model in [ 'gemini-embedding-2', 'gemini-embedding-2-preview',
				'gemini-embedding-001', 'text-embedding-004', 'text-multilingual-embedding-002', ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('supports_dimensions( self, model: str ) -> bool')
			Logger( ).write( exception )
			raise exception
	
	def build_contents( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""Build embedding contents.
		
		Purpose:
			Builds provider-ready embedding content from required text or a required collection
			of text values.
		
		Args:
			text (str | List[str]): Required text input or batch of text inputs.
		
		Returns:
			str | List[str]: Provider-ready embedding content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text
			
			if isinstance( self.input_text, list ):
				self.contents = [ ]
				
				for item in self.input_text:
					if item is None:
						continue
					
					self.content_text = str( item ).strip( )
					
					if self.content_text:
						self.contents.append( self.content_text )
				
				throw_if( 'contents', self.contents )
				return self.contents
			
			self.contents = str( self.input_text ).strip( )
			throw_if( 'contents', self.contents )
			return self.contents
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('build_contents( self, text: str | List[ str ] ) -> '
			                    'str | List[ str ]')
			Logger( ).write( exception )
			raise exception
	
	def build_embedding_config( self, model: str, dimensions: int = 0, task_type: str = '',
		title: str = '' ) -> EmbedContentConfig:
		"""Build embedding configuration.
		
		Purpose:
			Builds a provider-native EmbedContentConfig from arguments assigned to object
			members.
		
		Args:
			model (str): Required Gemini embedding model identifier.
			dimensions (int): Requested output dimensionality, where zero uses the model default.
			task_type (str): Optional embedding task type.
			title (str): Optional retrieval-document title.
		
		Returns:
			EmbedContentConfig: Provider-ready embedding configuration.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.dimensions = dimensions
			self.task_type = task_type.strip( ).upper( )
			self.title = title.strip( )
			self.config_values = { }
			
			if self.dimensions > 0:
				if self.supports_dimensions( self.model ):
					self.config_values[ 'output_dimensionality' ] = (self.dimensions)
			
			if self.task_type:
				self.config_values[ 'task_type' ] = (self.task_type)
			
			if self.title:
				if self.task_type == 'RETRIEVAL_DOCUMENT':
					self.config_values[ 'title' ] = (self.title)
			
			self.embedding_config = EmbedContentConfig( **self.config_values )
			return self.embedding_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('build_embedding_config( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def extract_embeddings( self ) -> (List[ float ] | List[ List[ float ] ] | None):
		"""Extract embeddings.
		
		Purpose:
			Extracts floating-point embedding vectors from the latest provider response.
		
		Returns:
			List[float] | List[List[float]] | None: A single vector, multiple vectors, or None
			when the response contains no embeddings.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.embedding = None
			self.embeddings = [ ]
			
			if self.response is None:
				return None
			
			self.response_embeddings = getattr( self.response, 'embeddings', [ ], ) or [ ]
			
			if not self.response_embeddings:
				self.response_embedding = getattr( self.response, 'embedding', None, )
				
				if self.response_embedding is not None:
					self.response_embeddings = [ self.response_embedding, ]
			
			for item in self.response_embeddings:
				if item is None:
					continue
				
				self.values = getattr( item, 'values', None, )
				
				if self.values is None:
					if isinstance( item, dict ):
						self.values = item.get( 'values', None, )
				
				if self.values is None:
					continue
				
				self.embeddings.append( list( self.values ) )
			
			if not self.embeddings:
				return None
			
			if isinstance( self.contents, str ):
				self.embedding = self.embeddings[ 0 ]
				return self.embedding
			
			self.embedding = self.embeddings
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('extract_embeddings( self ) -> '
			                    'List[ float ] | List[ List[ float ] ] | None')
			Logger( ).write( exception )
			raise exception
	
	def create( self, text: str | List[ str ], model: str, dimensions: int = 0, task_type: str =
	'',
		title: str = '' ) -> List[ float ] | List[ List[ float ] ] | None:
		"""Create embeddings.
		
		Purpose:
			Creates one or more embedding vectors from required text input using a required
			Gemini embedding model and optional output dimensionality, task type, and
			retrieval-document title.
		
		Args:
			text (str | List[str]): Required text input or batch of text inputs.
			model (str): Required Gemini embedding model identifier.
			dimensions (int): Requested output dimensionality, where zero uses the model default.
			task_type (str): Optional embedding task type.
			title (str): Optional title used with RETRIEVAL_DOCUMENT.
		
		Returns:
			List[float] | List[List[float]] | None: A single embedding, multiple embeddings, or
			None when the provider returns no embedding vectors.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'model', model )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.input_text = text
			self.model = model
			self.dimensions = dimensions
			self.task_type = task_type
			self.title = title
			self.contents = self.build_contents( self.input_text )
			self.embedding_config = self.build_embedding_config( self.model, self.dimensions,
				self.task_type, self.title, )
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.response = self.client.models.embed_content( model=self.model,
				contents=self.contents, config=self.embedding_config, )
			self.usage = getattr( self.response, 'usage_metadata', None, )
			
			if self.usage is None:
				self.usage = getattr( self.response, 'usageMetadata', None, )
			
			return self.extract_embeddings( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'create( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_usage( self ) -> Any:
		"""Get embedding usage.
		
		Purpose:
			Returns usage metadata from the latest Gemini embedding response.
		
		Returns:
			Any: Provider usage metadata or None when unavailable.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			if self.response is None:
				return None
			
			self.usage = getattr( self.response, 'usage_metadata', None, )
			
			if self.usage is None:
				self.usage = getattr( self.response, 'usageMetadata', None, )
			
			return self.usage
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'get_usage( self ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Gemini Embeddings wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'gemini_api_key', 'client', 'model', 'input_text', 'contents', 'dimensions',
			'task_type', 'title', 'embedding_config', 'response', 'embedding', 'embeddings',
			'usage', 'model_options', 'task_options', 'dimension_options', 'supports_dimensions',
			'build_contents', 'build_embedding_config', 'extract_embeddings', 'create',
			'get_usage', ]

class TTS( Gemini ):
	"""Provide Gemini text-to-speech workflow support.
	
	Purpose:
		Provides single-speaker speech generation through the Google Gen AI SDK. The class
		assigns accepted method arguments to object members, constructs provider-native voice,
		speech, and content configuration objects, converts the returned raw PCM audio into WAV
		bytes, and optionally writes the generated audio to a local file.
	
	Attributes:
		client (Optional[genai.Client]): Google Gen AI client.
		model (str): Gemini text-to-speech model used by the current request.
		input_text (str): Text and delivery instructions sent to the provider.
		voice (str): Prebuilt Gemini voice used by the current request.
		speed (float): Requested delivery speed represented through prompt instructions.
		response_format (str): Audio format returned by the wrapper.
		instructions (str): Optional speech-delivery instructions.
		audio_path (str): Optional local output path.
		sample_rate (int): PCM sample rate used to construct WAV output.
		channels (int): Number of PCM audio channels.
		sample_width (int): PCM sample width in bytes.
		temperature (float): Sampling temperature.
		top_p (float): Nucleus-sampling value.
		max_tokens (int): Maximum output-token count.
		voice_config (Optional[VoiceConfig]): Provider voice configuration.
		speech_config (Optional[SpeechConfig]): Provider speech configuration.
		content_config (Optional[GenerateContentConfig]): Provider generation configuration.
		response (Optional[GenerateContentResponse]): Latest provider response.
		audio_bytes (bytes): Generated WAV audio bytes.
	"""
	client: Optional[ genai.Client ]
	model: str
	input_text: str
	voice: str
	speed: float
	response_format: str
	instructions: str
	audio_path: str
	sample_rate: int
	channels: int
	sample_width: int
	temperature: float
	top_p: float
	max_tokens: int
	voice_config: Optional[ VoiceConfig ]
	speech_config: Optional[ SpeechConfig ]
	content_config: Optional[ GenerateContentConfig ]
	response: Optional[ GenerateContentResponse ]
	audio_bytes: bytes
	
	def __init__( self, model: str = 'gemini-2.5-flash-preview-tts' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Gemini text-to-speech configuration and runtime state without executing
			a provider request.
		
		Args:
			model (str): Default Gemini text-to-speech model.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.client = None
		self.model = model
		self.input_text = ''
		self.voice = 'Kore'
		self.speed = 1.0
		self.response_format = 'audio/wav'
		self.instructions = ''
		self.audio_path = ''
		self.sample_rate = 24000
		self.channels = 1
		self.sample_width = 2
		self.temperature = 0.0
		self.top_p = 0.0
		self.max_tokens = 0
		self.response_modalities = [ 'AUDIO', ]
		self.voice_config = None
		self.speech_config = None
		self.content_config = None
		self.response = None
		self.audio_bytes = b''
		self.config_values = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get text-to-speech model options.
		
		Purpose:
			Returns Gemini models exposed for text-to-speech generation.
		
		Returns:
			List[str]: Supported Gemini text-to-speech model identifiers.
		"""
		return [ 'gemini-3.1-flash-tts-preview', 'gemini-2.5-flash-preview-tts',
			'gemini-2.5-pro-preview-tts', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get audio-format options.
		
		Purpose:
			Returns the WAV output format implemented by the wrapper.
		
		Returns:
			List[str]: Supported audio-format values.
		"""
		return [ 'audio/wav', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get audio MIME-type options.
		
		Purpose:
			Returns the WAV MIME type implemented by the wrapper.
		
		Returns:
			List[str]: Supported audio MIME types.
		"""
		return self.format_options
	
	@property
	def voice_options( self ) -> List[ str ]:
		"""Get voice options.
		
		Purpose:
			Returns prebuilt Gemini voices exposed for single-speaker speech generation.
		
		Returns:
			List[str]: Supported prebuilt voice names.
		"""
		return [ 'Achernar', 'Achird', 'Aoede', 'Algenib', 'Algieba', 'Alnilam', 'Autonoe',
			'Callirrhoe', 'Charon', 'Despina', 'Enceladus', 'Erinome', 'Fenrir', 'Gacrux',
			'Iapetus', 'Kore', 'Laomedeia', 'Leda', 'Orus', 'Puck', 'Pulcherrima', 'Rasalgethi',
			'Sadachbia', 'Sadaltager', 'Schedar', 'Sulafat', 'Umbriel', 'Vindemiatrix', 'Zephyr',
			'Zubenelgenubi', ]
	
	@property
	def speed_options( self ) -> List[ float ]:
		"""Get speech-speed options.
		
		Purpose:
			Returns speech-speed selections represented through natural-language delivery
			instructions.
		
		Returns:
			List[float]: Available speech-speed selections.
		"""
		return [ 0.75, 1.0, 1.25, 1.50, ]
	
	def to_wave_bytes( self, pcm_data: bytes, rate: int = 24000, channels: int = 1,
		sample_width: int = 2 ) -> bytes:
		"""Convert PCM data to WAV bytes.
		
		Purpose:
			Wraps required raw PCM audio in a WAV container using the assigned sample rate,
			channel count, and sample width.
		
		Args:
			pcm_data (bytes): Required raw PCM audio bytes.
			rate (int): PCM sample rate.
			channels (int): Number of audio channels.
			sample_width (int): Sample width in bytes.
		
		Returns:
			bytes: WAV-formatted audio bytes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			import io
			import wave
			
			throw_if( 'pcm_data', pcm_data )
			self.pcm_data = pcm_data
			self.sample_rate = rate
			self.channels = channels
			self.sample_width = sample_width
			
			with io.BytesIO( ) as buffer:
				with wave.open( buffer, 'wb' ) as wave_file:
					wave_file.setnchannels( self.channels )
					wave_file.setsampwidth( self.sample_width )
					wave_file.setframerate( self.sample_rate )
					wave_file.writeframes( self.pcm_data )
				
				self.audio_bytes = buffer.getvalue( )
			
			return self.audio_bytes
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'to_wave_bytes( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def create_speech( self, text: str, model: str = 'gemini-2.5-flash-preview-tts',
		format: str = 'audio/wav', voice: str = 'Kore', speed: float = 1.0, instruct: str = '',
		file_path: str = '', temperature: float = 0.0, top_p: float = 0.0, max_tokens: int = 0,
		sample_rate: int = 24000 ) -> bytes:
		"""Create speech.
		
		Purpose:
			Generates single-speaker speech from required text using the selected Gemini TTS
			model and prebuilt voice. Speech speed and delivery instructions are incorporated
			into the text prompt because Gemini TTS controls style through natural-language
			instructions.
		
		Args:
			text (str): Required text converted to speech.
			model (str): Gemini text-to-speech model.
			format (str): Wrapper audio-output format.
			voice (str): Prebuilt Gemini voice.
			speed (float): Requested delivery speed.
			instruct (str): Optional speech-delivery instructions.
			file_path (str): Optional local output path.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			max_tokens (int): Maximum output-token count.
			sample_rate (int): PCM sample rate used to construct WAV output.
		
		Returns:
			bytes: Generated WAV audio bytes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'model', model )
			throw_if( 'format', format )
			throw_if( 'voice', voice )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.raw_text = text
			self.model = model
			self.response_format = format
			self.voice = voice
			self.speed = speed
			self.instructions = instruct
			self.audio_path = file_path
			self.temperature = temperature
			self.top_p = top_p
			self.max_tokens = max_tokens
			self.sample_rate = sample_rate
			self.channels = 1
			self.sample_width = 2
			self.response_modalities = [ 'AUDIO', ]
			
			if self.response_format != 'audio/wav':
				raise ValueError( 'Gemini TTS supports WAV output in this wrapper.' )
			
			if self.model not in self.model_options:
				raise ValueError( f'Unsupported Gemini TTS model: {self.model}' )
			
			if self.voice not in self.voice_options:
				raise ValueError( f'Unsupported Gemini TTS voice: {self.voice}' )
			
			self.prompt_parts = [ ]
			
			if self.instructions:
				self.prompt_parts.append( self.instructions.strip( ) )
			
			if self.speed < 0.90:
				self.prompt_parts.append( 'Speak slowly and clearly.' )
			elif self.speed > 1.10:
				self.prompt_parts.append( 'Speak at a faster, energetic pace.' )
			
			self.prompt_parts.append( self.raw_text.strip( ) )
			self.input_text = '\n\n'.join( self.prompt_parts )
			throw_if( 'input_text', self.input_text )
			self.voice_config = VoiceConfig(
				prebuilt_voice_config=types.PrebuiltVoiceConfig( voice_name=self.voice, ), )
			self.speech_config = SpeechConfig( voice_config=self.voice_config, )
			self.config_values = { 'response_modalities': self.response_modalities,
				'speech_config': self.speech_config, 'temperature': self.temperature, }
			
			if self.top_p > 0:
				self.config_values[ 'top_p' ] = self.top_p
			
			if self.max_tokens > 0:
				self.config_values[ 'max_output_tokens' ] = (self.max_tokens)
			
			self.content_config = GenerateContentConfig( **self.config_values )
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.response = self.client.models.generate_content( model=self.model,
				contents=self.input_text, config=self.content_config, )
			self.audio_bytes = b''
			self.candidates = getattr( self.response, 'candidates', [ ], ) or [ ]
			
			for candidate in self.candidates:
				self.content = getattr( candidate, 'content', None, )
				
				if self.content is None:
					continue
				
				self.parts = getattr( self.content, 'parts', [ ], ) or [ ]
				
				for part in self.parts:
					self.inline_data = getattr( part, 'inline_data', None, )
					
					if self.inline_data is None:
						continue
					
					self.pcm_data = getattr( self.inline_data, 'data', b'', )
					
					if not self.pcm_data:
						continue
					
					self.audio_bytes = self.to_wave_bytes( self.pcm_data, self.sample_rate,
						self.channels, self.sample_width, )
					break
				
				if self.audio_bytes:
					break
			
			throw_if( 'audio_bytes', self.audio_bytes )
			
			if self.audio_path:
				with open( self.audio_path, 'wb' ) as target:
					target.write( self.audio_bytes )
			
			return self.audio_bytes
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Gemini text-to-speech wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'gemini_api_key', 'client', 'model', 'input_text', 'voice', 'speed',
			'response_format', 'instructions', 'audio_path', 'sample_rate', 'channels',
			'sample_width', 'temperature', 'top_p', 'max_tokens', 'response_modalities',
			'voice_config', 'speech_config', 'content_config', 'response', 'audio_bytes',
			'model_options', 'format_options', 'mime_options', 'voice_options', 'speed_options',
			'to_wave_bytes', 'create_speech', ]

class Transcription( Gemini ):
	"""Provide Gemini audio-transcription workflow support.
	
	Purpose:
		Provides audio transcription through Gemini multimodal audio understanding. The class
		uploads a required local audio file, builds a transcription prompt from the requested
		language and time range, constructs provider-native generation configuration, executes
		the Gemini request, and returns the resulting transcript text.
	
	Attributes:
		client (Optional[genai.Client]): Google Gen AI client.
		model (str): Gemini model used by the current transcription request.
		file_path (str): Local audio-file path used by the current request.
		language (str): Expected language of the source audio.
		mime_type (str): Audio MIME type associated with the local file.
		prompt (str): Transcription prompt sent to Gemini.
		temperature (float): Sampling temperature.
		top_p (float): Nucleus-sampling value.
		top_k (int): Top-k sampling value.
		frequency_penalty (float): Frequency penalty.
		presence_penalty (float): Presence penalty.
		max_tokens (int): Maximum output-token count.
		instructions (str): Optional system instruction.
		start_time (float): Optional transcription start time in seconds.
		end_time (float): Optional transcription end time in seconds.
		content_config (Optional[GenerateContentConfig]): Provider generation configuration.
		uploaded_file (Optional[File]): Audio file uploaded through the Gemini Files API.
		response (Optional[GenerateContentResponse]): Latest provider response.
		transcript (str): Transcript extracted from the latest response.
	"""
	client: Optional[ genai.Client ]
	model: str
	file_path: str
	language: str
	mime_type: str
	prompt: str
	temperature: float
	top_p: float
	top_k: int
	frequency_penalty: float
	presence_penalty: float
	max_tokens: int
	instructions: str
	start_time: float
	end_time: float
	content_config: Optional[ GenerateContentConfig ]
	uploaded_file: Optional[ File ]
	response: Optional[ GenerateContentResponse ]
	transcript: str
	
	def __init__( self,
		model: str = 'gemini-3-flash-preview' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Gemini audio-transcription configuration and runtime state without
			executing a provider request.
		
		Args:
			model (str): Default Gemini model used for audio transcription.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.client = None
		self.model = model
		self.file_path = ''
		self.language = 'Auto'
		self.mime_type = ''
		self.prompt = ''
		self.temperature = 0.0
		self.top_p = 0.0
		self.top_k = 0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_tokens = 0
		self.instructions = ''
		self.start_time = 0.0
		self.end_time = 0.0
		self.content_config = None
		self.uploaded_file = None
		self.response = None
		self.transcript = ''
		self.config_values = { }
		self.prompt_parts = [ ]
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get transcription-model options.
		
		Purpose:
			Returns Gemini multimodal models exposed for audio transcription.
		
		Returns:
			List[str]: Supported Gemini transcription model identifiers.
		"""
		return [
			'gemini-3.1-pro-preview',
			'gemini-3.1-flash-lite-preview',
			'gemini-3-flash-preview',
			'gemini-2.5-pro',
			'gemini-2.5-flash',
			'gemini-2.5-flash-lite',
			'gemini-2.0-flash',
			'gemini-2.0-flash-lite',
		]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Get language options.
		
		Purpose:
			Returns source-language hints exposed by the transcription wrapper.
		
		Returns:
			List[str]: Available source-language selections.
		"""
		return [
			'Auto',
			'English',
			'Spanish',
			'French',
			'German',
			'Italian',
			'Portuguese',
			'Dutch',
			'Polish',
			'Russian',
			'Ukrainian',
			'Turkish',
			'Arabic',
			'Hindi',
			'Japanese',
			'Korean',
			'Chinese',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get supported audio MIME types.
		
		Purpose:
			Returns audio MIME types exposed by the Gemini transcription workflow.
		
		Returns:
			List[str]: Supported audio MIME-type values.
		"""
		return [
			'audio/wav',
			'audio/mpeg',
			'audio/mp3',
			'audio/aiff',
			'audio/aac',
			'audio/ogg',
			'audio/flac',
			'audio/mp4',
			'audio/webm',
		]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get supported audio MIME types.
		
		Purpose:
			Returns the audio MIME types exposed by the transcription wrapper.
		
		Returns:
			List[str]: Supported audio MIME-type values.
		"""
		return self.format_options
	
	def build_prompt( self, language: str = 'Auto',
		start_time: float = 0.0, end_time: float = 0.0,
		prompt: str = '' ) -> str:
		"""Build transcription prompt.
		
		Purpose:
			Builds a Gemini transcription instruction from the selected language, optional time
			range, and optional caller-supplied transcription guidance.
		
		Args:
			language (str): Expected language of the source audio.
			start_time (float): Optional transcription start time in seconds.
			end_time (float): Optional transcription end time in seconds.
			prompt (str): Optional transcription guidance or vocabulary context.
		
		Returns:
			str: Provider-ready transcription prompt.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.language = language
			self.start_time = start_time
			self.end_time = end_time
			self.prompt = prompt
			self.prompt_parts = [ ]
			
			if self.prompt:
				self.prompt_parts.append(
					self.prompt.strip( )
				)
			
			self.prompt_parts.append(
				'Generate a complete, verbatim transcript of the spoken audio.'
			)
			
			if self.language:
				if self.language.strip( ).lower( ) != 'auto':
					self.prompt_parts.append(
						f'The expected spoken language is '
						f'{self.language.strip( )}.'
					)
			
			if self.end_time > self.start_time:
				self.prompt_parts.append(
					f'Transcribe only the audio between '
					f'{self.start_time:0.2f} seconds and '
					f'{self.end_time:0.2f} seconds.'
				)
			
			self.prompt_parts.append(
				'Preserve the spoken wording, speaker changes, punctuation, '
				'numbers, names, and technical terms accurately.'
			)
			self.prompt_parts.append(
				'Return only the transcript text.'
			)
			self.prompt = '\n\n'.join(
				self.prompt_parts
			)
			throw_if( 'prompt', self.prompt )
			return self.prompt
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = 'build_prompt( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get transcript text.
		
		Purpose:
			Extracts transcript text from the latest Gemini response.
		
		Returns:
			str: Extracted transcript text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.transcript = ''
			
			if self.response is None:
				return self.transcript
			
			self.response_text = getattr(
				self.response,
				'text',
				'',
			)
			
			if self.response_text:
				self.transcript = str(
					self.response_text
				).strip( )
				return self.transcript
			
			self.text_parts = [ ]
			self.candidates = getattr(
				self.response,
				'candidates',
				[ ],
			) or [ ]
			
			for candidate in self.candidates:
				self.content = getattr(
					candidate,
					'content',
					None,
				)
				
				if self.content is None:
					continue
				
				self.parts = getattr(
					self.content,
					'parts',
					[ ],
				) or [ ]
				
				for part in self.parts:
					self.part_text = getattr(
						part,
						'text',
						'',
					)
					
					if self.part_text:
						self.text_parts.append(
							str( self.part_text )
						)
			
			self.transcript = ''.join(
				self.text_parts
			).strip( )
			return self.transcript
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = 'get_output_text( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, path: str, model: str,
		language: str = 'Auto', mime_type: str = '',
		temperature: float = 0.0, top_p: float = 0.0,
		top_k: int = 0, frequency: float = 0.0,
		presence: float = 0.0, max_tokens: int = 0,
		start_time: float = 0.0, end_time: float = 0.0,
		instruct: str = '', prompt: str = '' ) -> str:
		"""Transcribe audio.
		
		Purpose:
			Uploads a required local audio file and generates a transcript through Gemini
			multimodal audio understanding using the selected model, source-language hint,
			time range, generation controls, and optional transcription guidance.
		
		Args:
			path (str): Required local audio-file path.
			model (str): Required Gemini multimodal model.
			language (str): Expected language of the source audio.
			mime_type (str): Optional audio MIME type retained with the request state.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			start_time (float): Optional transcription start time in seconds.
			end_time (float): Optional transcription end time in seconds.
			instruct (str): Optional system instruction.
			prompt (str): Optional transcription guidance or vocabulary context.
		
		Returns:
			str: Generated transcript text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if(
				'GEMINI_API_KEY',
				self.gemini_api_key,
			)
			self.file_path = path
			self.model = model
			self.language = language
			self.mime_type = mime_type
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.start_time = start_time
			self.end_time = end_time
			self.instructions = instruct
			self.prompt = prompt
			self.prompt = self.build_prompt(
				self.language,
				self.start_time,
				self.end_time,
				self.prompt,
			)
			self.config_values = {
				'temperature': self.temperature,
			}
			
			if self.top_p > 0:
				self.config_values[ 'top_p' ] = self.top_p
			
			if self.top_k > 0:
				self.config_values[ 'top_k' ] = self.top_k
			
			if self.frequency_penalty != 0:
				self.config_values[ 'frequency_penalty' ] = (
					self.frequency_penalty
				)
			
			if self.presence_penalty != 0:
				self.config_values[ 'presence_penalty' ] = (
					self.presence_penalty
				)
			
			if self.max_tokens > 0:
				self.config_values[ 'max_output_tokens' ] = (
					self.max_tokens
				)
			
			if self.instructions:
				self.config_values[ 'system_instruction' ] = (
					self.instructions
				)
			
			self.content_config = GenerateContentConfig(
				**self.config_values
			)
			self.client = genai.Client(
				api_key=self.gemini_api_key,
			)
			self.uploaded_file = self.client.files.upload(
				file=self.file_path,
			)
			throw_if(
				'uploaded_file',
				self.uploaded_file,
			)
			self.response = self.client.models.generate_content(
				model=self.model,
				contents=[
					self.prompt,
					self.uploaded_file,
				],
				config=self.content_config,
			)
			self.transcript = self.get_output_text( )
			throw_if( 'transcript', self.transcript )
			return self.transcript
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = 'transcribe( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Gemini Transcription wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [
			'gemini_api_key',
			'client',
			'model',
			'file_path',
			'language',
			'mime_type',
			'prompt',
			'temperature',
			'top_p',
			'top_k',
			'frequency_penalty',
			'presence_penalty',
			'max_tokens',
			'instructions',
			'start_time',
			'end_time',
			'content_config',
			'uploaded_file',
			'response',
			'transcript',
			'model_options',
			'language_options',
			'format_options',
			'mime_options',
			'build_prompt',
			'get_output_text',
			'transcribe',
		]

class Translation( Gemini ):
	"""Provide Gemini audio-translation workflow support.
	
	Purpose:
		Provides spoken-audio translation through Gemini multimodal audio understanding. The
		class uploads a required local audio file, builds a translation prompt from the required
		target language and optional source language and time range, constructs provider-native
		generation configuration, executes the Gemini request, and returns translated text.
	
	Attributes:
		client (Optional[genai.Client]): Google Gen AI client.
		model (str): Gemini model used by the current translation request.
		file_path (str): Local audio-file path used by the current request.
		target_language (str): Required target language.
		source_language (str): Expected source language.
		mime_type (str): Optional audio MIME type supplied during upload.
		prompt (str): Translation prompt sent to Gemini.
		temperature (float): Sampling temperature.
		top_p (float): Nucleus-sampling value.
		top_k (int): Top-k sampling value.
		frequency_penalty (float): Frequency penalty.
		presence_penalty (float): Presence penalty.
		max_tokens (int): Maximum output-token count.
		instructions (str): Optional system instruction.
		start_time (float): Optional translation start time in seconds.
		end_time (float): Optional translation end time in seconds.
		content_config (Optional[GenerateContentConfig]): Provider generation configuration.
		uploaded_file (Optional[File]): Audio file uploaded through the Gemini Files API.
		response (Optional[GenerateContentResponse]): Latest provider response.
		translation (str): Translated text extracted from the latest response.
	"""
	client: Optional[ genai.Client ]
	model: str
	file_path: str
	target_language: str
	source_language: str
	mime_type: str
	prompt: str
	temperature: float
	top_p: float
	top_k: int
	frequency_penalty: float
	presence_penalty: float
	max_tokens: int
	instructions: str
	start_time: float
	end_time: float
	content_config: Optional[ GenerateContentConfig ]
	uploaded_file: Optional[ File ]
	response: Optional[ GenerateContentResponse ]
	translation: str
	
	def __init__( self, model: str = 'gemini-3-flash-preview' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Gemini audio-translation configuration and runtime state without
			executing a provider request.
		
		Args:
			model (str): Default Gemini model used for audio translation.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.client = None
		self.model = model
		self.file_path = ''
		self.target_language = 'English'
		self.source_language = 'Auto'
		self.mime_type = ''
		self.prompt = ''
		self.temperature = 0.0
		self.top_p = 0.0
		self.top_k = 0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_tokens = 0
		self.instructions = ''
		self.start_time = 0.0
		self.end_time = 0.0
		self.content_config = None
		self.uploaded_file = None
		self.response = None
		self.translation = ''
		self.config_values = { }
		self.prompt_parts = [ ]
		self.upload_config = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get translation-model options.
		
		Purpose:
			Returns Gemini multimodal models exposed for spoken-audio translation.
		
		Returns:
			List[str]: Supported Gemini translation model identifiers.
		"""
		return [ 'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite-preview',
			'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash',
			'gemini-2.5-flash-lite',
			'gemini-2.0-flash', 'gemini-2.0-flash-lite', ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Get language options.
		
		Purpose:
			Returns source and target languages exposed by the audio-translation wrapper.
		
		Returns:
			List[str]: Available language selections.
		"""
		return [ 'Auto', 'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
			'Dutch',
			'Polish', 'Russian', 'Ukrainian', 'Turkish', 'Arabic', 'Hindi', 'Japanese', 'Korean',
			'Chinese', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get supported audio MIME types.
		
		Purpose:
			Returns audio MIME types exposed by the Gemini translation workflow.
		
		Returns:
			List[str]: Supported audio MIME-type values.
		"""
		return [ 'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/aiff', 'audio/aac', 'audio/ogg',
			'audio/flac', 'audio/mp4', 'audio/webm', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get supported audio MIME types.
		
		Purpose:
			Returns audio MIME types exposed by the Gemini translation wrapper.
		
		Returns:
			List[str]: Supported audio MIME-type values.
		"""
		return self.format_options
	
	def build_prompt( self, target: str, source: str = 'Auto', start_time: float = 0.0,
		end_time: float = 0.0, prompt: str = '' ) -> str:
		"""Build translation prompt.
		
		Purpose:
			Builds a Gemini audio-translation instruction from the required target language,
			optional source language, optional time range, and caller-supplied guidance.
		
		Args:
			target (str): Required target language.
			source (str): Expected source language.
			start_time (float): Optional translation start time in seconds.
			end_time (float): Optional translation end time in seconds.
			prompt (str): Optional translation guidance or vocabulary context.
		
		Returns:
			str: Provider-ready translation prompt.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'target', target )
			self.target_language = target
			self.source_language = source
			self.start_time = start_time
			self.end_time = end_time
			self.prompt = prompt
			self.prompt_parts = [ ]
			
			if self.prompt:
				self.prompt_parts.append( self.prompt.strip( ) )
			
			self.prompt_parts.append( f'Translate all spoken audio into '
			                          f'{self.target_language.strip( )}.' )
			
			if self.source_language:
				if self.source_language.strip( ).lower( ) != 'auto':
					self.prompt_parts.append( f'The expected source language is '
					                          f'{self.source_language.strip( )}.' )
			
			if self.end_time > self.start_time:
				self.prompt_parts.append( f'Translate only the audio between '
				                          f'{self.start_time:0.2f} seconds and '
				                          f'{self.end_time:0.2f} seconds.' )
			
			self.prompt_parts.append( 'Preserve the meaning, names, numbers, technical terms, '
			                          'tone, and speaker changes accurately.' )
			self.prompt_parts.append( 'Return only the translated text.' )
			self.prompt = '\n\n'.join( self.prompt_parts )
			throw_if( 'prompt', self.prompt )
			return self.prompt
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = 'build_prompt( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get translated text.
		
		Purpose:
			Extracts translated text from the latest Gemini response.
		
		Returns:
			str: Extracted translated text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.translation = ''
			
			if self.response is None:
				return self.translation
			
			self.response_text = getattr( self.response, 'text', '', )
			
			if self.response_text:
				self.translation = str( self.response_text ).strip( )
				return self.translation
			
			self.text_parts = [ ]
			self.candidates = getattr( self.response, 'candidates', [ ], ) or [ ]
			
			for candidate in self.candidates:
				self.content = getattr( candidate, 'content', None, )
				
				if self.content is None:
					continue
				
				self.parts = getattr( self.content, 'parts', [ ], ) or [ ]
				
				for part in self.parts:
					self.part_text = getattr( part, 'text', '', )
					
					if self.part_text:
						self.text_parts.append( str( self.part_text ) )
			
			self.translation = ''.join( self.text_parts ).strip( )
			return self.translation
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = 'get_output_text( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def translate( self, path: str, model: str, language: str, source: str = 'Auto',
		mime_type: str = '', temperature: float = 0.0, top_p: float = 0.0, top_k: int = 0,
		frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 0, start_time: float =
		0.0,
		end_time: float = 0.0, instruct: str = '', prompt: str = '' ) -> str:
		"""Translate audio.
		
		Purpose:
			Uploads a required local audio file and translates its spoken content into a
			required target language through Gemini multimodal audio understanding.
		
		Args:
			path (str): Required local audio-file path.
			model (str): Required Gemini multimodal model.
			language (str): Required target language.
			source (str): Expected source language.
			mime_type (str): Optional audio MIME type supplied during file upload.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			start_time (float): Optional translation start time in seconds.
			end_time (float): Optional translation end time in seconds.
			instruct (str): Optional system instruction.
			prompt (str): Optional translation guidance or vocabulary context.
		
		Returns:
			str: Generated translated text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'language', language )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.file_path = path
			self.model = model
			self.target_language = language
			self.source_language = source
			self.mime_type = mime_type
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.start_time = start_time
			self.end_time = end_time
			self.instructions = instruct
			self.prompt = prompt
			self.prompt = self.build_prompt( self.target_language, self.source_language,
				self.start_time, self.end_time, self.prompt, )
			self.config_values = { 'temperature': self.temperature, }
			
			if self.top_p > 0:
				self.config_values[ 'top_p' ] = self.top_p
			
			if self.top_k > 0:
				self.config_values[ 'top_k' ] = self.top_k
			
			if self.frequency_penalty != 0:
				self.config_values[ 'frequency_penalty' ] = (self.frequency_penalty)
			
			if self.presence_penalty != 0:
				self.config_values[ 'presence_penalty' ] = (self.presence_penalty)
			
			if self.max_tokens > 0:
				self.config_values[ 'max_output_tokens' ] = (self.max_tokens)
			
			if self.instructions:
				self.config_values[ 'system_instruction' ] = (self.instructions)
			
			self.content_config = GenerateContentConfig( **self.config_values )
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.upload_config = None
			
			if self.mime_type:
				self.upload_config = types.UploadFileConfig( mime_type=self.mime_type, )
			
			if self.upload_config is not None:
				self.uploaded_file = self.client.files.upload( file=self.file_path,
					config=self.upload_config, )
			else:
				self.uploaded_file = self.client.files.upload( file=self.file_path, )
			
			throw_if( 'uploaded_file', self.uploaded_file, )
			self.response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.uploaded_file, ], config=self.content_config, )
			self.translation = self.get_output_text( )
			throw_if( 'translation', self.translation, )
			return self.translation
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = 'translate( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Gemini Translation wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'gemini_api_key', 'client', 'model', 'file_path', 'target_language',
			'source_language', 'mime_type', 'prompt', 'temperature', 'top_p', 'top_k',
			'frequency_penalty', 'presence_penalty', 'max_tokens', 'instructions', 'start_time',
			'end_time', 'content_config', 'uploaded_file', 'response', 'translation',
			'model_options', 'language_options', 'format_options', 'mime_options', 'build_prompt',
			'get_output_text', 'translate', ]

class Files( Gemini ):
	"""Provide Gemini Files API workflow support.
	
	Purpose:
		Provides file upload, listing, retrieval, deletion, conditional download, file analysis,
		file search, and multi-file survey operations through the Google Gen AI SDK. The class
		assigns accepted arguments to object members before constructing provider requests and
		returns provider file objects, file metadata, downloaded bytes, or generated text.
	
	Attributes:
		client (Optional[genai.Client]): Google Gen AI client.
		model (str): Gemini model used for file analysis.
		file_path (str): Local file path used by an upload operation.
		file_id (str): Gemini file resource name used by the current operation.
		filename (str): Filename associated with the current operation.
		display_name (str): Display name assigned during upload.
		mime_type (str): MIME type assigned during upload.
		file (Optional[File]): Latest Gemini file object.
		files (List[File]): Latest collection of Gemini file objects.
		response (Any): Latest provider response.
		content (Any): Latest downloaded content or metadata result.
		output_text (str): Text returned by the latest file-analysis request.
		content_config (Optional[GenerateContentConfig]): File-analysis configuration.
	"""
	client: Optional[ genai.Client ]
	model: str
	file_path: str
	file_id: str
	filename: str
	display_name: str
	mime_type: str
	file: Optional[ File ]
	files: List[ File ]
	response: Any
	content: Any
	output_text: str
	content_config: Optional[ GenerateContentConfig ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Gemini Files API configuration and runtime state without executing a
			provider request.
		
		Args:
			model (str): Default Gemini model used for file analysis.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.client = None
		self.model = model
		self.file_path = ''
		self.file_id = ''
		self.filename = ''
		self.display_name = ''
		self.mime_type = ''
		self.file = None
		self.files = [ ]
		self.response = None
		self.content = None
		self.output_text = ''
		self.prompt = ''
		self.instructions = ''
		self.temperature = 0.0
		self.top_p = 0.0
		self.top_k = 0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_tokens = 0
		self.response_format = ''
		self.content_config = None
		self.upload_config = None
		self.config_values = { }
		self.metadata = { }
		self.results = [ ]
		self.file_objects = [ ]
		self.file_paths = [ ]
		self.contents = [ ]
	
	@property
	def file_options( self ) -> List[ str ]:
		"""Get file-extension options.
		
		Purpose:
			Returns common file extensions supported by Gemini file-input workflows.
		
		Returns:
			List[str]: Supported file-extension values.
		"""
		return [ 'pdf', 'txt', 'md', 'csv', 'json', 'xml', 'html', 'doc', 'docx', 'xls', 'xlsx',
			'ppt', 'pptx', 'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tiff', 'wav', 'mp3', 'm4a',
			'aac', 'flac', 'ogg', 'mp4', 'mpeg', 'mov', 'webm', ]
	
	@property
	def purpose_options( self ) -> List[ str ]:
		"""Get file-purpose options.
		
		Purpose:
			Returns an empty collection because the Gemini Files API does not require OpenAI-style
			upload-purpose values.
		
		Returns:
			List[str]: Empty purpose collection.
		"""
		return [ ]
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get file-analysis model options.
		
		Purpose:
			Returns Gemini models exposed for file analysis and document question answering.
		
		Returns:
			List[str]: Supported Gemini model identifiers.
		"""
		return [ 'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite-preview',
			'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash',
			'gemini-2.5-flash-lite',
			'gemini-2.0-flash', 'gemini-2.0-flash-lite', ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Get media-resolution options.
		
		Purpose:
			Returns media-resolution values exposed for file-analysis requests.
		
		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get response-format options.
		
		Purpose:
			Returns response MIME types supported by Gemini file-analysis requests.
		
		Returns:
			List[str]: Supported response MIME types.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Get include options.
		
		Purpose:
			Returns an empty collection because Gemini file requests do not use OpenAI
			include-path arguments.
		
		Returns:
			List[str]: Empty include-option collection.
		"""
		return [ ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Get reasoning options.
		
		Purpose:
			Returns Gemini thinking-level values exposed for file-analysis requests.
		
		Returns:
			List[str]: Supported thinking-level values.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Get tool-choice options.
		
		Purpose:
			Returns an empty collection because direct Gemini file analysis does not require a
			tool-choice argument.
		
		Returns:
			List[str]: Empty tool-choice collection.
		"""
		return [ ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Get tool options.
		
		Purpose:
			Returns optional grounding tools supported by file-analysis requests.
		
		Returns:
			List[str]: Supported tool names.
		"""
		return [ 'google_search', 'code_execution', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Get response-modality options.
		
		Purpose:
			Returns the text modality used by file-analysis requests.
		
		Returns:
			List[str]: Supported response modalities.
		"""
		return [ 'text', ]
	
	def get_file_metadata( self, file: File ) -> Dict[ str, Any ]:
		"""Get file metadata.
		
		Purpose:
			Extracts application-facing metadata from a required Gemini File object.
		
		Args:
			file (File): Required Gemini File object.
		
		Returns:
			Dict[str, Any]: Application-facing file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file', file )
			self.file = file
			self.metadata = { 'id': getattr( self.file, 'name', '', ),
				'name': getattr( self.file, 'name', '', ),
				'display_name': getattr( self.file, 'display_name', '', ),
				'filename': getattr( self.file, 'display_name', '', ),
				'mime_type': getattr( self.file, 'mime_type', '', ),
				'size_bytes': getattr( self.file, 'size_bytes', 0, ),
				'create_time': getattr( self.file, 'create_time', None, ),
				'expiration_time': getattr( self.file, 'expiration_time', None, ),
				'update_time': getattr( self.file, 'update_time', None, ),
				'uri': getattr( self.file, 'uri', '', ),
				'download_uri': getattr( self.file, 'download_uri', '', ),
				'state': getattr( self.file, 'state', None, ),
				'source': getattr( self.file, 'source', None, ),
				'error': getattr( self.file, 'error', None, ), }
			return self.metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('get_file_metadata( self, file: File ) -> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get output text.
		
		Purpose:
			Extracts generated text from the latest Gemini file-analysis response.
		
		Returns:
			str: Generated response text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_text = ''
			
			if self.response is None:
				return self.output_text
			
			self.response_text = getattr( self.response, 'text', '', )
			
			if self.response_text:
				self.output_text = str( self.response_text ).strip( )
				return self.output_text
			
			self.text_parts = [ ]
			self.candidates = getattr( self.response, 'candidates', [ ], ) or [ ]
			
			for candidate in self.candidates:
				self.response_content = getattr( candidate, 'content', None, )
				
				if self.response_content is None:
					continue
				
				self.parts = getattr( self.response_content, 'parts', [ ], ) or [ ]
				
				for part in self.parts:
					self.part_text = getattr( part, 'text', '', )
					
					if self.part_text:
						self.text_parts.append( str( self.part_text ) )
			
			self.output_text = ''.join( self.text_parts ).strip( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'get_output_text( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def upload( self, path: str, display_name: str = '', mime_type: str = '' ) -> File:
		"""Upload a file.
		
		Purpose:
			Uploads a required local file to the Gemini Files API using an optional display name
			and MIME type.
		
		Args:
			path (str): Required local file path.
			display_name (str): Optional uploaded-file display name.
			mime_type (str): Optional uploaded-file MIME type.
		
		Returns:
			File: Uploaded Gemini File object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.file_path = path
			self.filename = Path( self.file_path ).name
			self.display_name = (display_name if display_name else self.filename)
			self.mime_type = mime_type
			self.upload_config = None
			
			if self.display_name or self.mime_type:
				self.upload_values = { }
				
				if self.display_name:
					self.upload_values[ 'display_name' ] = (self.display_name)
				
				if self.mime_type:
					self.upload_values[ 'mime_type' ] = (self.mime_type)
				
				self.upload_config = types.UploadFileConfig( **self.upload_values )
			
			self.client = genai.Client( api_key=self.gemini_api_key, )
			
			if self.upload_config is None:
				self.file = self.client.files.upload( file=self.file_path, )
			else:
				self.file = self.client.files.upload( file=self.file_path,
					config=self.upload_config, )
			
			throw_if( 'file', self.file )
			self.file_id = getattr( self.file, 'name', '', )
			return self.file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'upload( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def upload_file( self, path: str, display_name: str = '', mime_type: str = '' ) -> File:
		"""Upload a file.
		
		Purpose:
			Provides the application-compatible upload-file alias.
		
		Args:
			path (str): Required local file path.
			display_name (str): Optional uploaded-file display name.
			mime_type (str): Optional uploaded-file MIME type.
		
		Returns:
			File: Uploaded Gemini File object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.file_path = path
			self.display_name = display_name
			self.mime_type = mime_type
			return self.upload( self.file_path, self.display_name, self.mime_type, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'upload_file( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def files_upload( self, path: str, display_name: str = '', mime_type: str = '' ) -> File:
		"""Upload a file.
		
		Purpose:
			Provides the application-compatible files-upload alias.
		
		Args:
			path (str): Required local file path.
			display_name (str): Optional uploaded-file display name.
			mime_type (str): Optional uploaded-file MIME type.
		
		Returns:
			File: Uploaded Gemini File object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.file_path = path
			self.display_name = display_name
			self.mime_type = mime_type
			return self.upload( self.file_path, self.display_name, self.mime_type, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'files_upload( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list( self ) -> List[ Dict[ str, Any ] ]:
		"""List files.
		
		Purpose:
			Lists Gemini files and returns application-facing metadata rows.
		
		Returns:
			List[Dict[str, Any]]: Gemini file metadata rows.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.response = self.client.files.list( )
			self.files = [ item for item in self.response ]
			self.results = [ self.get_file_metadata( item ) for item in self.files ]
			return self.results
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'list( self ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def list_files( self ) -> List[ Dict[ str, Any ] ]:
		"""List files.
		
		Purpose:
			Provides the application-compatible file-list alias.
		
		Returns:
			List[Dict[str, Any]]: Gemini file metadata rows.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			return self.list( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('list_files( self ) -> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, file_id: str ) -> File:
		"""Retrieve a file.
		
		Purpose:
			Retrieves a required Gemini file resource by name.
		
		Args:
			file_id (str): Required Gemini file resource name.
		
		Returns:
			File: Retrieved Gemini File object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.file_id = file_id
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.file = self.client.files.get( name=self.file_id, )
			throw_if( 'file', self.file )
			return self.file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'retrieve( self, file_id: str ) -> File'
			Logger( ).write( exception )
			raise exception
	
	def extract( self, file_id: str ) -> Any:
		"""Extract file content or metadata.
		
		Purpose:
			Downloads content for a required Gemini file when the file exposes a download URI.
			Uploaded Gemini prompt files generally cannot be downloaded, so their metadata is
			returned instead.
		
		Args:
			file_id (str): Required Gemini file resource name.
		
		Returns:
			Any: Downloaded bytes or application-facing file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.file = self.retrieve( self.file_id )
			self.download_uri = getattr( self.file, 'download_uri', '', )
			
			if not self.download_uri:
				self.content = self.get_file_metadata( self.file )
				return self.content
			
			self.content = self.client.files.download( file=self.file, )
			return self.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'extract( self, file_id: str ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def download( self, file_id: str ) -> Any:
		"""Download file content.
		
		Purpose:
			Provides the application-compatible download alias.
		
		Args:
			file_id (str): Required Gemini file resource name.
		
		Returns:
			Any: Downloaded bytes or application-facing file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.file_id = file_id
			return self.extract( self.file_id )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'download( self, file_id: str ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def content( self, file_id: str ) -> Any:
		"""Get file content.
		
		Purpose:
			Provides the application-compatible file-content alias.
		
		Args:
			file_id (str): Required Gemini file resource name.
		
		Returns:
			Any: Downloaded bytes or application-facing file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.file_id = file_id
			return self.extract( self.file_id )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'content( self, file_id: str ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, file_id: str ) -> bool:
		"""Delete a file.
		
		Purpose:
			Deletes a required Gemini file resource.
		
		Args:
			file_id (str): Required Gemini file resource name.
		
		Returns:
			bool: True when the delete request completes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.file_id = file_id
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.response = self.client.files.delete( name=self.file_id, )
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'delete( self, file_id: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def build_content_config( self, model: str, temperature: float = 0.0, top_p: float = 0.0,
		top_k: int = 0, frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 0,
		instruct: str = '', response_format: str = '' ) -> GenerateContentConfig:
		"""Build file-analysis configuration.
		
		Purpose:
			Builds provider-native file-analysis configuration from arguments assigned to object
			members.
		
		Args:
			model (str): Required Gemini model identifier.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction.
			response_format (str): Optional response MIME type.
		
		Returns:
			GenerateContentConfig: Provider-ready generation configuration.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.instructions = instruct
			self.response_format = response_format
			self.config_values = { 'temperature': self.temperature, }
			
			if self.top_p > 0:
				self.config_values[ 'top_p' ] = self.top_p
			
			if self.top_k > 0:
				self.config_values[ 'top_k' ] = self.top_k
			
			if self.frequency_penalty != 0:
				self.config_values[ 'frequency_penalty' ] = (self.frequency_penalty)
			
			if self.presence_penalty != 0:
				self.config_values[ 'presence_penalty' ] = (self.presence_penalty)
			
			if self.max_tokens > 0:
				self.config_values[ 'max_output_tokens' ] = (self.max_tokens)
			
			if self.instructions:
				self.config_values[ 'system_instruction' ] = (self.instructions)
			
			if self.response_format:
				self.config_values[ 'response_mime_type' ] = (self.response_format)
			
			self.content_config = GenerateContentConfig( **self.config_values )
			return self.content_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'build_content_config( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def summarize( self, prompt: str, file_id: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = 0.0, top_p: float = 0.0, top_k: int = 0, frequency: float = 0.0,
		presence: float = 0.0, max_tokens: int = 0, instruct: str = '',
		response_format: str = '' ) -> str:
		"""Analyze a file.
		
		Purpose:
			Answers a required prompt using a required Gemini file resource.
		
		Args:
			prompt (str): Required file-analysis prompt.
			file_id (str): Required Gemini file resource name.
			model (str): Gemini model used for file analysis.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction.
			response_format (str): Optional response MIME type.
		
		Returns:
			str: Generated file-analysis text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'file_id', file_id )
			throw_if( 'model', model )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.prompt = prompt
			self.file_id = file_id
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.instructions = instruct
			self.response_format = response_format
			self.file = self.retrieve( self.file_id )
			self.content_config = self.build_content_config( self.model, self.temperature,
				self.top_p, self.top_k, self.frequency_penalty, self.presence_penalty,
				self.max_tokens, self.instructions, self.response_format, )
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.response = self.client.models.generate_content( model=self.model,
				contents=[ self.prompt, self.file, ], config=self.content_config, )
			self.output_text = self.get_output_text( )
			throw_if( 'output_text', self.output_text )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'summarize( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def search( self, prompt: str, file_id: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = 0.0, top_p: float = 0.0, top_k: int = 0, frequency: float = 0.0,
		presence: float = 0.0, max_tokens: int = 0, instruct: str = '',
		response_format: str = '' ) -> str:
		"""Search a file.
		
		Purpose:
			Answers a required question using a required Gemini file resource.
		
		Args:
			prompt (str): Required question about the file.
			file_id (str): Required Gemini file resource name.
			model (str): Gemini model used for analysis.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction.
			response_format (str): Optional response MIME type.
		
		Returns:
			str: Generated answer.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'file_id', file_id )
			self.prompt = prompt
			self.file_id = file_id
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.instructions = instruct
			self.response_format = response_format
			return self.summarize( self.prompt, self.file_id, self.model, self.temperature,
				self.top_p, self.top_k, self.frequency_penalty, self.presence_penalty,
				self.max_tokens, self.instructions, self.response_format, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'search( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, prompt: str, file_ids: List[ str ], model: str = 'gemini-2.5-flash-lite',
		temperature: float = 0.0, top_p: float = 0.0, top_k: int = 0, frequency: float = 0.0,
		presence: float = 0.0, max_tokens: int = 0, instruct: str = '',
		response_format: str = '' ) -> str:
		"""Survey multiple files.
		
		Purpose:
			Answers a required prompt using multiple required Gemini file resources.
		
		Args:
			prompt (str): Required multi-file analysis prompt.
			file_ids (List[str]): Required Gemini file resource names.
			model (str): Gemini model used for analysis.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction.
			response_format (str): Optional response MIME type.
		
		Returns:
			str: Generated multi-file analysis.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'file_ids', file_ids )
			throw_if( 'model', model )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.prompt = prompt
			self.file_ids = file_ids
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.instructions = instruct
			self.response_format = response_format
			self.file_objects = [ self.retrieve( item ) for item in self.file_ids ]
			self.content_config = self.build_content_config( self.model, self.temperature,
				self.top_p, self.top_k, self.frequency_penalty, self.presence_penalty,
				self.max_tokens, self.instructions, self.response_format, )
			self.contents = [ self.prompt, ]
			self.contents.extend( self.file_objects )
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.response = self.client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config, )
			self.output_text = self.get_output_text( )
			throw_if( 'output_text', self.output_text )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'survey( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = 0.0 ) -> str:
		"""Search the web.
		
		Purpose:
			Answers a required prompt using Gemini Google Search grounding.
		
		Args:
			prompt (str): Required grounded-search prompt.
			model (str): Gemini model used for generation.
			temperature (float): Sampling temperature.
		
		Returns:
			str: Grounded response text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.prompt = prompt
			self.model = model
			self.temperature = temperature
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.content_config = GenerateContentConfig( temperature=self.temperature,
				tools=[ Tool( google_search=GoogleSearch( ), ), ], )
			self.response = self.client.models.generate_content( model=self.model,
				contents=self.prompt, config=self.content_config, )
			self.output_text = self.get_output_text( )
			throw_if( 'output_text', self.output_text )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'web_search( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Gemini Files wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'gemini_api_key', 'client', 'model', 'file_path', 'file_id', 'filename',
			'display_name', 'mime_type', 'file', 'files', 'response', 'content', 'output_text',
			'content_config', 'file_options', 'purpose_options', 'model_options', 'media_options',
			'format_options', 'include_options', 'reasoning_options', 'choice_options',
			'tool_options', 'modality_options', 'get_file_metadata', 'get_output_text', 'upload',
			'upload_file', 'files_upload', 'list', 'list_files', 'retrieve', 'extract', 'download',
			'content', 'delete', 'build_content_config', 'summarize', 'search', 'survey',
			'web_search', ]

class FileSearch( Gemini ):
	"""Provide Gemini File Search Store workflow support.
	
	Purpose:
		Provides File Search Store creation, listing, retrieval, deletion, document importing,
		and grounded search through the Google Gen AI SDK. The class assigns accepted method
		arguments to object members before constructing provider-native store, file-import,
		tool, and generation requests.
	
	Attributes:
		client (Optional[genai.Client]): Google Gen AI client.
		store_id (str): File Search Store resource name used by the current operation.
		store_name (str): Display name used when creating a File Search Store.
		embedding_model (str): Embedding model assigned when creating a store.
		file_path (str): Local file path used by the current upload operation.
		file_name (str): Gemini Files API resource name imported into a store.
		display_name (str): Display name assigned to an uploaded file.
		mime_type (str): Optional MIME type assigned to an uploaded file.
		model (str): Gemini model used by the current File Search query.
		query_text (str): Query submitted to the selected File Search Store.
		instructions (str): Optional system instruction used by the query.
		response_format (str): Optional response MIME type.
		temperature (float): Sampling temperature.
		top_p (float): Nucleus-sampling value.
		frequency_penalty (float): Frequency penalty.
		presence_penalty (float): Presence penalty.
		max_tokens (int): Maximum output-token count.
		response (Any): Latest provider response.
		operation (Any): Latest asynchronous import operation.
		uploaded_file (Optional[File]): Latest uploaded Gemini file.
		file_search_store (Optional[FileSearchStore]): Latest File Search Store.
		stores (List[FileSearchStore]): Latest File Search Store collection.
		collections (Dict[str, str]): Display names mapped to store resource names.
		output_text (str): Text extracted from the latest grounded query.
		grounding_metadata (Any): Grounding metadata from the latest query.
	"""
	client: Optional[ genai.Client ]
	store_id: str
	store_name: str
	embedding_model: str
	file_path: str
	file_name: str
	display_name: str
	mime_type: str
	model: str
	query_text: str
	instructions: str
	response_format: str
	temperature: float
	top_p: float
	frequency_penalty: float
	presence_penalty: float
	max_tokens: int
	response: Any
	operation: Any
	uploaded_file: Optional[ File ]
	file_search_store: Optional[ FileSearchStore ]
	stores: List[ FileSearchStore ]
	collections: Dict[ str, str ]
	output_text: str
	grounding_metadata: Any
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite',
		embedding_model: str = 'models/gemini-embedding-001' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Gemini File Search Store configuration and runtime state without
			executing a provider request.
		
		Args:
			model (str): Default Gemini model used for grounded File Search queries.
			embedding_model (str): Default embedding model used when creating stores.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.client = None
		self.store_id = ''
		self.store_name = ''
		self.embedding_model = embedding_model
		self.file_path = ''
		self.file_name = ''
		self.display_name = ''
		self.mime_type = ''
		self.model = model
		self.query_text = ''
		self.instructions = ''
		self.response_format = ''
		self.temperature = 0.0
		self.top_p = 0.0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_tokens = 0
		self.response = None
		self.operation = None
		self.uploaded_file = None
		self.file_search_store = None
		self.stores = [ ]
		self.collections = { }
		self.output_text = ''
		self.grounding_metadata = None
		self.file_config = { }
		self.import_config = { }
		self.store_config = { }
		self.config_values = { }
		self.content_config = None
		self.file_search_tool = None
		self.candidates = [ ]
		self.metadata_filter = ''
		self.custom_metadata = [ ]
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get query-model options.
		
		Purpose:
			Returns Gemini models exposed for grounded File Search queries.
		
		Returns:
			List[str]: Supported Gemini model identifiers.
		"""
		return [ 'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite-preview',
			'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash',
			'gemini-2.5-flash-lite', ]
	
	@property
	def embedding_model_options( self ) -> List[ str ]:
		"""Get embedding-model options.
		
		Purpose:
			Returns embedding models exposed when creating File Search Stores.
		
		Returns:
			List[str]: Supported File Search embedding-model resource names.
		"""
		return [ 'models/gemini-embedding-001', 'models/gemini-embedding-2', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get response-format options.
		
		Purpose:
			Returns response MIME types exposed for File Search queries.
		
		Returns:
			List[str]: Supported response MIME types.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Get reasoning options.
		
		Purpose:
			Returns an empty collection because this wrapper does not expose a separate
			reasoning argument for File Search queries.
		
		Returns:
			List[str]: Empty reasoning-option collection.
		"""
		return [ ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Get tool-choice options.
		
		Purpose:
			Returns an empty collection because the File Search tool is explicitly assigned by
			the wrapper.
		
		Returns:
			List[str]: Empty tool-choice collection.
		"""
		return [ ]
	
	def refresh_collections( self ) -> Dict[ str, str ]:
		"""Refresh File Search Store collections.
		
		Purpose:
			Lists available Gemini File Search Stores and rebuilds the application-facing mapping
			of display names to globally scoped store resource names.
		
		Returns:
			Dict[str, str]: Display names mapped to File Search Store resource names.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.stores = [ store for store in self.client.file_search_stores.list( ) ]
			self.collections = { }
			
			for store in self.stores:
				self.store_id = getattr( store, 'name', '', )
				self.store_name = getattr( store, 'display_name', '', )
				
				if not self.store_id:
					continue
				
				if not self.store_name:
					self.store_name = self.store_id
				
				self.collections[ str( self.store_name ).strip( ) ] = str( self.store_id ).strip( )
			
			return self.collections
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('refresh_collections( self ) -> Dict[ str, str ]')
			Logger( ).write( exception )
			raise exception
	
	def create( self, name: str,
		embedding_model: str = 'models/gemini-embedding-001' ) -> FileSearchStore:
		"""Create a File Search Store.
		
		Purpose:
			Creates a Gemini File Search Store with a required display name and selected embedding
			model.
		
		Args:
			name (str): Required File Search Store display name.
			embedding_model (str): Embedding model assigned to the store.
		
		Returns:
			FileSearchStore: Created Gemini File Search Store.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'name', name )
			throw_if( 'embedding_model', embedding_model )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.store_name = name
			self.embedding_model = embedding_model
			self.store_config = { 'display_name': self.store_name,
				'embedding_model': self.embedding_model, }
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.file_search_store = (
				self.client.file_search_stores.create( config=self.store_config, ))
			throw_if( 'file_search_store', self.file_search_store, )
			self.store_id = getattr( self.file_search_store, 'name', '', )
			self.refresh_collections( )
			return self.file_search_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'create( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def create_store( self, name: str,
		embedding_model: str = 'models/gemini-embedding-001' ) -> FileSearchStore:
		"""Create a File Search Store.
		
		Purpose:
			Provides the application-compatible alias for File Search Store creation.
		
		Args:
			name (str): Required File Search Store display name.
			embedding_model (str): Embedding model assigned to the store.
		
		Returns:
			FileSearchStore: Created Gemini File Search Store.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.store_name = name
			self.embedding_model = embedding_model
			return self.create( self.store_name, self.embedding_model, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'create_store( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list( self ) -> List[ FileSearchStore ]:
		"""List File Search Stores.
		
		Purpose:
			Lists available Gemini File Search Stores.
		
		Returns:
			List[FileSearchStore]: Available File Search Stores.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.refresh_collections( )
			return self.stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'list( self ) -> List[ FileSearchStore ]'
			Logger( ).write( exception )
			raise exception
	
	def list_stores( self ) -> List[ FileSearchStore ]:
		"""List File Search Stores.
		
		Purpose:
			Provides the application-compatible alias for File Search Store listing.
		
		Returns:
			List[FileSearchStore]: Available File Search Stores.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			return self.list( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('list_stores( self ) -> List[ FileSearchStore ]')
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str ) -> FileSearchStore:
		"""Retrieve a File Search Store.
		
		Purpose:
			Retrieves a required Gemini File Search Store resource.
		
		Args:
			store_id (str): Required File Search Store resource name.
		
		Returns:
			FileSearchStore: Retrieved File Search Store.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.store_id = store_id
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.file_search_store = (self.client.file_search_stores.get( name=self.store_id, ))
			throw_if( 'file_search_store', self.file_search_store, )
			return self.file_search_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('retrieve( self, store_id: str ) -> FileSearchStore')
			Logger( ).write( exception )
			raise exception
	
	def retrieve_store( self, store_id: str ) -> FileSearchStore:
		"""Retrieve a File Search Store.
		
		Purpose:
			Provides the application-compatible alias for File Search Store retrieval.
		
		Args:
			store_id (str): Required File Search Store resource name.
		
		Returns:
			FileSearchStore: Retrieved File Search Store.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.store_id = store_id
			return self.retrieve( self.store_id )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('retrieve_store( self, store_id: str ) -> FileSearchStore')
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str, force: bool = True ) -> bool:
		"""Delete a File Search Store.
		
		Purpose:
			Deletes a required Gemini File Search Store and optionally forces deletion of its
			indexed documents.
		
		Args:
			store_id (str): Required File Search Store resource name.
			force (bool): Indicates whether indexed documents are deleted with the store.
		
		Returns:
			bool: True when the delete request completes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.store_id = store_id
			self.force = force
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.client.file_search_stores.delete( name=self.store_id,
				config={ 'force': self.force, }, )
			self.refresh_collections( )
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'delete( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def delete_store( self, store_id: str, force: bool = True ) -> bool:
		"""Delete a File Search Store.
		
		Purpose:
			Provides the application-compatible alias for File Search Store deletion.
		
		Args:
			store_id (str): Required File Search Store resource name.
			force (bool): Indicates whether indexed documents are deleted with the store.
		
		Returns:
			bool: True when the delete request completes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.store_id = store_id
			self.force = force
			return self.delete( self.store_id, self.force, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'delete_store( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def upload_file( self, path: str, store_id: str, display_name: str = '', mime_type: str = '',
		custom_metadata: Optional[ List[ Dict[ str, Any ] ] ] = None ) -> Any:
		"""Upload a file to a File Search Store.
		
		Purpose:
			Uploads a required local file through the Gemini Files API and imports the resulting
			file resource into a required File Search Store.
		
		Args:
			path (str): Required local file path.
			store_id (str): Required File Search Store resource name.
			display_name (str): Optional file display name used in citations.
			mime_type (str): Optional uploaded-file MIME type.
			custom_metadata (Optional[List[Dict[str, Any]]]): Optional indexed-file metadata.
		
		Returns:
			Any: Gemini file-import operation.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'store_id', store_id )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.file_path = path
			self.store_id = store_id
			self.display_name = (display_name if display_name else Path( self.file_path ).name)
			self.mime_type = mime_type
			self.custom_metadata = (custom_metadata if custom_metadata is not None else [ ])
			self.file_config = { 'display_name': self.display_name, }
			
			if self.mime_type:
				self.file_config[ 'mime_type' ] = self.mime_type
			
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.uploaded_file = self.client.files.upload( file=self.file_path,
				config=self.file_config, )
			throw_if( 'uploaded_file', self.uploaded_file, )
			self.file_name = getattr( self.uploaded_file, 'name', '', )
			throw_if( 'file_name', self.file_name )
			self.import_config = { }
			
			if self.custom_metadata:
				self.import_config[ 'custom_metadata' ] = (self.custom_metadata)
			
			if self.import_config:
				self.operation = (self.client.file_search_stores.import_file(
					file_search_store_name=self.store_id, file_name=self.file_name,
					config=self.import_config, ))
			else:
				self.operation = (self.client.file_search_stores.import_file(
					file_search_store_name=self.store_id, file_name=self.file_name, ))
			
			throw_if( 'operation', self.operation )
			self.response = self.operation
			return self.operation
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'upload_file( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def upload( self, path: str, store_id: str, display_name: str = '', mime_type: str = '',
		custom_metadata: Optional[ List[ Dict[ str, Any ] ] ] = None ) -> Any:
		"""Upload a file to a File Search Store.
		
		Purpose:
			Provides the application-compatible alias for File Search Store file importing.
		
		Args:
			path (str): Required local file path.
			store_id (str): Required File Search Store resource name.
			display_name (str): Optional file display name.
			mime_type (str): Optional uploaded-file MIME type.
			custom_metadata (Optional[List[Dict[str, Any]]]): Optional indexed-file metadata.
		
		Returns:
			Any: Gemini file-import operation.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.file_path = path
			self.store_id = store_id
			self.display_name = display_name
			self.mime_type = mime_type
			self.custom_metadata = (custom_metadata if custom_metadata is not None else [ ])
			return self.upload_file( self.file_path, self.store_id, self.display_name,
				self.mime_type, self.custom_metadata, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'upload( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get File Search response text.
		
		Purpose:
			Extracts generated text from the latest Gemini File Search response.
		
		Returns:
			str: Generated grounded response text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_text = ''
			
			if self.response is None:
				return self.output_text
			
			self.response_text = getattr( self.response, 'text', '', )
			
			if self.response_text:
				self.output_text = str( self.response_text ).strip( )
				return self.output_text
			
			self.text_parts = [ ]
			self.candidates = getattr( self.response, 'candidates', [ ], ) or [ ]
			
			for candidate in self.candidates:
				self.response_content = getattr( candidate, 'content', None, )
				
				if self.response_content is None:
					continue
				
				self.parts = getattr( self.response_content, 'parts', [ ], ) or [ ]
				
				for part in self.parts:
					self.part_text = getattr( part, 'text', '', )
					
					if self.part_text:
						self.text_parts.append( str( self.part_text ) )
			
			self.output_text = ''.join( self.text_parts ).strip( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'get_output_text( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def capture_grounding_metadata( self ) -> None:
		"""Capture grounding metadata.
		
		Purpose:
			Captures citation and retrieved-context metadata from the latest File Search response.
		
		Returns:
			None: This method updates object state.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.grounding_metadata = None
			
			if self.response is None:
				return
			
			self.candidates = getattr( self.response, 'candidates', [ ], ) or [ ]
			
			for candidate in self.candidates:
				self.grounding_metadata = getattr( candidate, 'grounding_metadata', None, )
				
				if self.grounding_metadata is not None:
					return
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('capture_grounding_metadata( self ) -> None')
			Logger( ).write( exception )
			raise exception
	
	def search( self, store_id: str, query: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = 0.0, top_p: float = 0.0, frequency: float = 0.0, presence: float =
		0.0,
		max_tokens: int = 0, response_format: str = '', instruct: str = '',
		metadata_filter: str = '' ) -> str:
		"""Search a File Search Store.
		
		Purpose:
			Answers a required query using a required Gemini File Search Store as grounded
			retrieval context.
		
		Args:
			store_id (str): Required File Search Store resource name.
			query (str): Required File Search query.
			model (str): Gemini model used to generate the grounded response.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			response_format (str): Optional response MIME type.
			instruct (str): Optional system instruction.
			metadata_filter (str): Optional indexed-file metadata filter.
		
		Returns:
			str: Generated grounded response text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'query', query )
			throw_if( 'model', model )
			throw_if( 'GEMINI_API_KEY', self.gemini_api_key, )
			self.store_id = store_id
			self.query_text = query
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.response_format = response_format
			self.instructions = instruct
			self.metadata_filter = metadata_filter
			self.file_search_values = { 'file_search_store_names': [ self.store_id, ], }
			
			if self.metadata_filter:
				self.file_search_values[ 'metadata_filter' ] = (self.metadata_filter)
			
			self.file_search_tool = Tool( file_search=FileSearch( **self.file_search_values ), )
			self.config_values = { 'tools': [ self.file_search_tool, ],
				'temperature': self.temperature, }
			
			if self.top_p > 0:
				self.config_values[ 'top_p' ] = self.top_p
			
			if self.frequency_penalty != 0:
				self.config_values[ 'frequency_penalty' ] = (self.frequency_penalty)
			
			if self.presence_penalty != 0:
				self.config_values[ 'presence_penalty' ] = (self.presence_penalty)
			
			if self.max_tokens > 0:
				self.config_values[ 'max_output_tokens' ] = (self.max_tokens)
			
			if self.response_format:
				self.config_values[ 'response_mime_type' ] = (self.response_format)
			
			if self.instructions:
				self.config_values[ 'system_instruction' ] = (self.instructions)
			
			self.content_config = GenerateContentConfig( **self.config_values )
			self.client = genai.Client( api_key=self.gemini_api_key, )
			self.response = self.client.models.generate_content( model=self.model,
				contents=self.query_text, config=self.content_config, )
			self.capture_grounding_metadata( )
			self.output_text = self.get_output_text( )
			throw_if( 'output_text', self.output_text )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'search( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def query( self, store_id: str, query: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = 0.0, top_p: float = 0.0, frequency: float = 0.0, presence: float =
		0.0,
		max_tokens: int = 0, response_format: str = '', instruct: str = '',
		metadata_filter: str = '' ) -> str:
		"""Search a File Search Store.
		
		Purpose:
			Provides the application-compatible alias for grounded File Search Store queries.
		
		Args:
			store_id (str): Required File Search Store resource name.
			query (str): Required File Search query.
			model (str): Gemini model used to generate the response.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			response_format (str): Optional response MIME type.
			instruct (str): Optional system instruction.
			metadata_filter (str): Optional indexed-file metadata filter.
		
		Returns:
			str: Generated grounded response text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.store_id = store_id
			self.query_text = query
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.response_format = response_format
			self.instructions = instruct
			self.metadata_filter = metadata_filter
			return self.search( self.store_id, self.query_text, self.model, self.temperature,
				self.top_p, self.frequency_penalty, self.presence_penalty, self.max_tokens,
				self.response_format, self.instructions, self.metadata_filter, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'query( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Gemini FileSearch wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'gemini_api_key', 'client', 'store_id', 'store_name', 'embedding_model',
			'file_path', 'file_name', 'display_name', 'mime_type', 'model', 'query_text',
			'instructions', 'response_format', 'temperature', 'top_p', 'frequency_penalty',
			'presence_penalty', 'max_tokens', 'response', 'operation', 'uploaded_file',
			'file_search_store', 'stores', 'collections', 'output_text', 'grounding_metadata',
			'model_options', 'embedding_model_options', 'format_options', 'reasoning_options',
			'choice_options', 'refresh_collections', 'create', 'create_store', 'list',
			'list_stores', 'retrieve', 'retrieve_store', 'delete', 'delete_store', 'upload_file',
			'upload', 'get_output_text', 'capture_grounding_metadata', 'search', 'query', ]

class CloudBuckets( Gemini ):
	"""Provide Google Cloud Storage bucket workflow support.
	
	Purpose:
		Provides Google Cloud Storage bucket creation, retrieval, listing, deletion, object
		upload, object retrieval, object deletion, and Gemini-based bucket querying. Bucket
		queries construct Vertex AI Gemini input from Google Cloud Storage object URIs so
		supported bucket content can be analyzed without first copying each object into the
		Gemini Files API.
	
	Attributes:
		project_id (str): Google Cloud project identifier.
		location (str): Google Cloud region used by Vertex AI.
		bucket_name (str): Bucket name used by the current operation.
		object_name (str): Object name used by the current operation.
		file_path (str): Local file path used by an upload operation.
		model (str): Gemini model used by the current query.
		query_text (str): Query submitted against bucket content.
		instructions (str): Optional Gemini system instruction.
		temperature (float): Sampling temperature.
		top_p (float): Nucleus-sampling value.
		frequency_penalty (float): Frequency penalty.
		presence_penalty (float): Presence penalty.
		max_tokens (int): Maximum output-token count.
		response_format (str): Optional response MIME type.
		max_files (int): Maximum number of bucket objects included in a query.
		storage_client (Optional[storage.Client]): Google Cloud Storage client.
		genai_client (Optional[genai.Client]): Vertex AI Google Gen AI client.
		bucket (Optional[storage.Bucket]): Bucket used by the current operation.
		blob (Optional[storage.Blob]): Object used by the current operation.
		blobs (List[storage.Blob]): Objects returned by the latest list operation.
		response (Any): Latest provider response.
		output_text (str): Text extracted from the latest Gemini response.
		collections (Dict[str, str]): Application-facing bucket collection mappings.
		documents (Dict[str, str]): Application-facing document mappings.
	"""
	project_id: str
	location: str
	bucket_name: str
	object_name: str
	file_path: str
	model: str
	query_text: str
	instructions: str
	temperature: float
	top_p: float
	frequency_penalty: float
	presence_penalty: float
	max_tokens: int
	response_format: str
	max_files: int
	storage_client: Optional[ storage.Client ]
	genai_client: Optional[ genai.Client ]
	bucket: Optional[ storage.Bucket ]
	blob: Optional[ storage.Blob ]
	blobs: List[ storage.Blob ]
	response: Any
	output_text: str
	collections: Dict[ str, str ]
	documents: Dict[ str, str ]
	
	def __init__( self, project_id: str = '', location: str = 'us-central1',
		model: str = 'gemini-2.5-flash-lite' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Google Cloud Storage and Gemini bucket-query configuration without
			executing a provider request.
		
		Args:
			project_id (str): Google Cloud project identifier. An empty value uses config.py.
			location (str): Google Cloud region used by Vertex AI.
			model (str): Default Gemini model used for bucket queries.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.project_id = (project_id if project_id else cfg.GOOGLE_CLOUD_PROJECT_ID)
		self.location = location
		self.model = model
		self.bucket_name = ''
		self.object_name = ''
		self.file_path = ''
		self.query_text = ''
		self.instructions = ''
		self.temperature = 0.0
		self.top_p = 0.0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_tokens = 0
		self.response_format = ''
		self.max_files = 20
		self.prefix = ''
		self.content_type = ''
		self.storage_client = None
		self.genai_client = None
		self.client = None
		self.bucket = None
		self.blob = None
		self.blobs = [ ]
		self.response = None
		self.output_text = ''
		self.content_config = None
		self.config_values = { }
		self.contents = [ ]
		self.object_parts = [ ]
		self.metadata = { }
		self.results = [ ]
		self.collections = { 'Federal Financial Data': 'jeni-financial/data',
			'Federal Financial Regulations': 'jeni-financial/regulations',
			'DoW Financial Data': 'jeni-dow/budget/data',
			'DoW Financial Regulations': 'jeni-dow/budget/regulations',
			'DoA Financial Data': 'jenni-doa/Financial Data', }
		self.documents = { 'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
			'SF133.csv': 'file-32s641QK1Xb5QUatY3zfWF',
			'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
			'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn', }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get model options.
		
		Purpose:
			Returns Gemini models exposed for Google Cloud Storage bucket queries.
		
		Returns:
			List[str]: Supported Gemini model identifiers.
		"""
		return [ 'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite-preview',
			'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash',
			'gemini-2.5-flash-lite',
			'gemini-2.0-flash', 'gemini-2.0-flash-lite', ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Get media-resolution options.
		
		Purpose:
			Returns media-resolution values exposed for bucket-content analysis.
		
		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get response-format options.
		
		Purpose:
			Returns response MIME types exposed for bucket queries.
		
		Returns:
			List[str]: Supported response MIME types.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Get reasoning options.
		
		Purpose:
			Returns an empty collection because bucket queries do not expose a separate reasoning
			control in this wrapper.
		
		Returns:
			List[str]: Empty reasoning-option collection.
		"""
		return [ ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Get tool-choice options.
		
		Purpose:
			Returns an empty collection because bucket queries do not use function-calling tool
			selection.
		
		Returns:
			List[str]: Empty tool-choice collection.
		"""
		return [ ]
	
	def get_storage_client( self, project_id: str ) -> storage.Client:
		"""Get Storage client.
		
		Purpose:
			Creates a Google Cloud Storage client for a required project identifier.
		
		Args:
			project_id (str): Required Google Cloud project identifier.
		
		Returns:
			storage.Client: Configured Google Cloud Storage client.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'project_id', project_id )
			self.project_id = project_id
			self.storage_client = storage.Client( project=self.project_id, )
			self.client = self.storage_client
			return self.storage_client
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('get_storage_client( self, project_id: str ) -> '
			                    'storage.Client')
			Logger( ).write( exception )
			raise exception
	
	def get_genai_client( self, project_id: str, location: str ) -> genai.Client:
		"""Get Gemini client.
		
		Purpose:
			Creates a Vertex AI Google Gen AI client for a required project and location.
		
		Args:
			project_id (str): Required Google Cloud project identifier.
			location (str): Required Vertex AI region.
		
		Returns:
			genai.Client: Configured Vertex AI Google Gen AI client.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'project_id', project_id )
			throw_if( 'location', location )
			self.project_id = project_id
			self.location = location
			self.genai_client = genai.Client( vertexai=True, project=self.project_id,
				location=self.location, )
			return self.genai_client
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'get_genai_client( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_bucket_metadata( self, bucket: storage.Bucket ) -> Dict[ str, Any ]:
		"""Get bucket metadata.
		
		Purpose:
			Extracts application-facing metadata from a required Google Cloud Storage bucket.
		
		Args:
			bucket (storage.Bucket): Required Google Cloud Storage bucket.
		
		Returns:
			Dict[str, Any]: Application-facing bucket metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket = bucket
			self.metadata = { 'id': getattr( self.bucket, 'id', '', ),
				'name': getattr( self.bucket, 'name', '', ),
				'project_number': getattr( self.bucket, 'project_number', None, ),
				'location': getattr( self.bucket, 'location', '', ),
				'location_type': getattr( self.bucket, 'location_type', '', ),
				'storage_class': getattr( self.bucket, 'storage_class', '', ),
				'time_created': getattr( self.bucket, 'time_created', None, ),
				'updated': getattr( self.bucket, 'updated', None, ),
				'versioning_enabled': getattr( self.bucket, 'versioning_enabled', False, ),
				'requester_pays': getattr( self.bucket, 'requester_pays', False, ),
				'retention_period': getattr( self.bucket, 'retention_period', None, ),
				'labels': getattr( self.bucket, 'labels', None, ),
				'self_link': getattr( self.bucket, 'self_link', '', ), }
			return self.metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('get_bucket_metadata( self, bucket: storage.Bucket ) -> '
			                    'Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def get_blob_metadata( self, blob: storage.Blob ) -> Dict[ str, Any ]:
		"""Get object metadata.
		
		Purpose:
			Extracts application-facing metadata from a required Google Cloud Storage object.
		
		Args:
			blob (storage.Blob): Required Google Cloud Storage object.
		
		Returns:
			Dict[str, Any]: Application-facing object metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'blob', blob )
			self.blob = blob
			self.metadata = { 'id': getattr( self.blob, 'id', '', ),
				'name': getattr( self.blob, 'name', '', ),
				'bucket': getattr( getattr( self.blob, 'bucket', None, ), 'name', '', ),
				'content_type': getattr( self.blob, 'content_type', '', ),
				'size': getattr( self.blob, 'size', 0, ),
				'generation': getattr( self.blob, 'generation', None, ),
				'metageneration': getattr( self.blob, 'metageneration', None, ),
				'md5_hash': getattr( self.blob, 'md5_hash', '', ),
				'crc32c': getattr( self.blob, 'crc32c', '', ),
				'time_created': getattr( self.blob, 'time_created', None, ),
				'updated': getattr( self.blob, 'updated', None, ),
				'storage_class': getattr( self.blob, 'storage_class', '', ),
				'metadata': getattr( self.blob, 'metadata', None, ),
				'uri': (f'gs://{self.bucket_name}/'
				        f'{getattr( self.blob, "name", "" )}'), }
			return self.metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('get_blob_metadata( self, blob: storage.Blob ) -> '
			                    'Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def create( self, name: str, project_id: str = '', location: str = 'US' ) -> storage.Bucket:
		"""Create a bucket.
		
		Purpose:
			Creates a Google Cloud Storage bucket with a required globally unique name.
		
		Args:
			name (str): Required Google Cloud Storage bucket name.
			project_id (str): Google Cloud project identifier. An empty value uses object state.
			location (str): Google Cloud Storage bucket location.
		
		Returns:
			storage.Bucket: Created Google Cloud Storage bucket.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'name', name )
			self.bucket_name = name
			self.project_id = (project_id if project_id else self.project_id)
			self.location = location
			throw_if( 'project_id', self.project_id )
			throw_if( 'location', self.location )
			self.storage_client = self.get_storage_client( self.project_id )
			self.bucket = storage.Bucket( client=self.storage_client, name=self.bucket_name, )
			self.response = self.storage_client.create_bucket( self.bucket,
				location=self.location, )
			self.bucket = self.response
			return self.bucket
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'create( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def create_bucket( self, name: str, project_id: str = '',
		location: str = 'US' ) -> storage.Bucket:
		"""Create a bucket.
		
		Purpose:
			Provides the application-compatible alias for bucket creation.
		
		Args:
			name (str): Required Google Cloud Storage bucket name.
			project_id (str): Google Cloud project identifier.
			location (str): Google Cloud Storage bucket location.
		
		Returns:
			storage.Bucket: Created Google Cloud Storage bucket.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.bucket_name = name
			self.project_id = (project_id if project_id else self.project_id)
			self.location = location
			return self.create( self.bucket_name, self.project_id, self.location, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'create_bucket( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def upload_file( self, path: str, bucket: str, object_name: str = '', content_type: str = '',
		project_id: str = '' ) -> storage.Blob:
		"""Upload an object.
		
		Purpose:
			Uploads a required local file to a required Google Cloud Storage bucket.
		
		Args:
			path (str): Required local file path.
			bucket (str): Required Google Cloud Storage bucket name.
			object_name (str): Optional destination object name.
			content_type (str): Optional uploaded-object content type.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			storage.Blob: Uploaded Google Cloud Storage object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'bucket', bucket )
			self.file_path = path
			self.bucket_name = bucket
			self.object_name = (object_name if object_name else Path( self.file_path ).name)
			self.content_type = content_type
			self.project_id = (project_id if project_id else self.project_id)
			throw_if( 'project_id', self.project_id )
			self.storage_client = self.get_storage_client( self.project_id )
			self.bucket = self.storage_client.get_bucket( self.bucket_name )
			self.blob = self.bucket.blob( self.object_name )
			
			if self.content_type:
				self.blob.upload_from_filename( self.file_path, content_type=self.content_type, )
			else:
				self.blob.upload_from_filename( self.file_path, )
			
			self.blob.reload( )
			self.response = self.blob
			return self.blob
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'upload_file( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def upload( self, path: str, bucket: str, object_name: str = '', content_type: str = '',
		project_id: str = '' ) -> storage.Blob:
		"""Upload an object.
		
		Purpose:
			Provides the application-compatible alias for Google Cloud Storage object upload.
		
		Args:
			path (str): Required local file path.
			bucket (str): Required Google Cloud Storage bucket name.
			object_name (str): Optional destination object name.
			content_type (str): Optional uploaded-object content type.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			storage.Blob: Uploaded Google Cloud Storage object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.file_path = path
			self.bucket_name = bucket
			self.object_name = object_name
			self.content_type = content_type
			self.project_id = (project_id if project_id else self.project_id)
			return self.upload_file( self.file_path, self.bucket_name, self.object_name,
				self.content_type, self.project_id, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'upload( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def files_upload( self, path: str, bucket: str, object_name: str = '', content_type: str = '',
		project_id: str = '' ) -> storage.Blob:
		"""Upload an object.
		
		Purpose:
			Provides the application-compatible files-upload alias.
		
		Args:
			path (str): Required local file path.
			bucket (str): Required Google Cloud Storage bucket name.
			object_name (str): Optional destination object name.
			content_type (str): Optional uploaded-object content type.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			storage.Blob: Uploaded Google Cloud Storage object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.file_path = path
			self.bucket_name = bucket
			self.object_name = object_name
			self.content_type = content_type
			self.project_id = (project_id if project_id else self.project_id)
			return self.upload_file( self.file_path, self.bucket_name, self.object_name,
				self.content_type, self.project_id, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'files_upload( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, bucket: str, object_name: str = '', project_id: str = '' ) -> Any:
		"""Retrieve a bucket or object.
		
		Purpose:
			Retrieves metadata for a required Google Cloud Storage bucket or for an optional
			object within that bucket.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			object_name (str): Optional object name.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			Any: Retrieved Google Cloud Storage bucket or object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket_name = bucket
			self.object_name = object_name
			self.project_id = (project_id if project_id else self.project_id)
			throw_if( 'project_id', self.project_id )
			self.storage_client = self.get_storage_client( self.project_id )
			self.bucket = self.storage_client.get_bucket( self.bucket_name )
			
			if not self.object_name:
				self.response = self.bucket
				return self.bucket
			
			self.blob = self.bucket.get_blob( self.object_name )
			throw_if( 'blob', self.blob )
			self.response = self.blob
			return self.blob
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'retrieve( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_bucket( self, bucket: str, project_id: str = '' ) -> storage.Bucket:
		"""Retrieve a bucket.
		
		Purpose:
			Provides the application-compatible alias for Google Cloud Storage bucket retrieval.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			storage.Bucket: Retrieved Google Cloud Storage bucket.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.bucket_name = bucket
			self.project_id = (project_id if project_id else self.project_id)
			self.bucket = self.retrieve( self.bucket_name, '', self.project_id, )
			return self.bucket
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'retrieve_bucket( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get( self, bucket: str, object_name: str = '', project_id: str = '' ) -> Any:
		"""Retrieve a bucket or object.
		
		Purpose:
			Provides the application-compatible alias for bucket or object retrieval.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			object_name (str): Optional object name.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			Any: Retrieved Google Cloud Storage bucket or object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.bucket_name = bucket
			self.object_name = object_name
			self.project_id = (project_id if project_id else self.project_id)
			return self.retrieve( self.bucket_name, self.object_name, self.project_id, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'get( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list( self, bucket: str, prefix: str = '', project_id: str = '' ) -> List[ storage.Blob ]:
		"""List bucket objects.
		
		Purpose:
			Lists objects in a required Google Cloud Storage bucket and optionally limits the
			result to an object-name prefix.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			prefix (str): Optional object-name prefix.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			List[storage.Blob]: Google Cloud Storage objects.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket_name = bucket
			self.prefix = prefix
			self.project_id = (project_id if project_id else self.project_id)
			throw_if( 'project_id', self.project_id )
			self.storage_client = self.get_storage_client( self.project_id )
			self.bucket = self.storage_client.get_bucket( self.bucket_name )
			self.blobs = list( self.storage_client.list_blobs( self.bucket_name,
				prefix=self.prefix if self.prefix else None, ) )
			self.response = self.blobs
			return self.blobs
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'list( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list_objects( self, bucket: str, prefix: str = '', project_id: str = '' ) -> List[
		Dict[ str, Any ] ]:
		"""List object metadata.
		
		Purpose:
			Lists application-facing metadata for objects in a required Google Cloud Storage
			bucket.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			prefix (str): Optional object-name prefix.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			List[Dict[str, Any]]: Google Cloud Storage object metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.bucket_name = bucket
			self.prefix = prefix
			self.project_id = (project_id if project_id else self.project_id)
			self.blobs = self.list( self.bucket_name, self.prefix, self.project_id, )
			self.results = [ self.get_blob_metadata( item ) for item in self.blobs ]
			return self.results
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'list_objects( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, bucket: str, object_name: str = '', project_id: str = '' ) -> bool:
		"""Delete a bucket or object.
		
		Purpose:
			Deletes an optional object from a required Google Cloud Storage bucket. When no
			object name is supplied, deletes the bucket.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			object_name (str): Optional object name.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			bool: True when the deletion request completes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket_name = bucket
			self.object_name = object_name
			self.project_id = (project_id if project_id else self.project_id)
			throw_if( 'project_id', self.project_id )
			self.storage_client = self.get_storage_client( self.project_id )
			self.bucket = self.storage_client.get_bucket( self.bucket_name )
			
			if self.object_name:
				self.blob = self.bucket.blob( self.object_name )
				self.blob.delete( )
				self.response = True
				return True
			
			self.bucket.delete( )
			self.response = True
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'delete( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def delete_bucket( self, bucket: str, project_id: str = '' ) -> bool:
		"""Delete a bucket.
		
		Purpose:
			Provides the application-compatible alias for Google Cloud Storage bucket deletion.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			bool: True when the deletion request completes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.bucket_name = bucket
			self.project_id = (project_id if project_id else self.project_id)
			return self.delete( self.bucket_name, '', self.project_id, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'delete_bucket( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def remove( self, bucket: str, object_name: str = '', project_id: str = '' ) -> bool:
		"""Delete a bucket or object.
		
		Purpose:
			Provides the application-compatible alias for bucket or object deletion.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			object_name (str): Optional object name.
			project_id (str): Google Cloud project identifier.
		
		Returns:
			bool: True when the deletion request completes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.bucket_name = bucket
			self.object_name = object_name
			self.project_id = (project_id if project_id else self.project_id)
			return self.delete( self.bucket_name, self.object_name, self.project_id, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'remove( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get query output text.
		
		Purpose:
			Extracts generated text from the latest Gemini bucket-query response.
		
		Returns:
			str: Generated query response text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_text = ''
			
			if self.response is None:
				return self.output_text
			
			self.response_text = getattr( self.response, 'text', '', )
			
			if self.response_text:
				self.output_text = str( self.response_text ).strip( )
				return self.output_text
			
			self.text_parts = [ ]
			self.candidates = getattr( self.response, 'candidates', [ ], ) or [ ]
			
			for candidate in self.candidates:
				self.response_content = getattr( candidate, 'content', None, )
				
				if self.response_content is None:
					continue
				
				self.parts = getattr( self.response_content, 'parts', [ ], ) or [ ]
				
				for part in self.parts:
					self.part_text = getattr( part, 'text', '', )
					
					if self.part_text:
						self.text_parts.append( str( self.part_text ) )
			
			self.output_text = ''.join( self.text_parts ).strip( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'get_output_text( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def search( self, bucket: str, query: str, model: str, project_id: str = '',
		location: str = 'us-central1', prefix: str = '', max_files: int = 20,
		temperature: float = 0.0, top_p: float = 0.0, frequency: float = 0.0, presence: float =
		0.0,
		max_tokens: int = 0, response_format: str = '', instruct: str = '' ) -> str:
		"""Query bucket content.
		
		Purpose:
			Answers a required query using supported objects from a required Google Cloud Storage
			bucket as Gemini multimodal input.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			query (str): Required question about bucket content.
			model (str): Required Gemini model identifier.
			project_id (str): Google Cloud project identifier.
			location (str): Vertex AI region.
			prefix (str): Optional object-name prefix.
			max_files (int): Maximum number of bucket objects included.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			response_format (str): Optional response MIME type.
			instruct (str): Optional system instruction.
		
		Returns:
			str: Generated answer based on bucket content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'query', query )
			throw_if( 'model', model )
			self.bucket_name = bucket
			self.query_text = query
			self.model = model
			self.project_id = (project_id if project_id else self.project_id)
			self.location = location
			self.prefix = prefix
			self.max_files = max_files
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.response_format = response_format
			self.instructions = instruct
			throw_if( 'project_id', self.project_id )
			throw_if( 'location', self.location )
			self.blobs = self.list( self.bucket_name, self.prefix, self.project_id, )
			
			if self.max_files > 0:
				self.blobs = self.blobs[ :self.max_files ]
			
			throw_if( 'blobs', self.blobs )
			self.object_parts = [ ]
			
			for item in self.blobs:
				self.object_name = getattr( item, 'name', '', )
				self.content_type = getattr( item, 'content_type', '', )
				
				if not self.object_name:
					continue
				
				if not self.content_type:
					continue
				
				self.object_parts.append( Part.from_uri( file_uri=(f'gs://{self.bucket_name}/'
				                                                   f'{self.object_name}'),
					mime_type=self.content_type, ) )
			
			throw_if( 'object_parts', self.object_parts )
			self.contents = [ Part.from_text( text=self.query_text, ), ]
			self.contents.extend( self.object_parts )
			self.config_values = { 'temperature': self.temperature, }
			
			if self.top_p > 0:
				self.config_values[ 'top_p' ] = self.top_p
			
			if self.frequency_penalty != 0:
				self.config_values[ 'frequency_penalty' ] = (self.frequency_penalty)
			
			if self.presence_penalty != 0:
				self.config_values[ 'presence_penalty' ] = (self.presence_penalty)
			
			if self.max_tokens > 0:
				self.config_values[ 'max_output_tokens' ] = (self.max_tokens)
			
			if self.response_format:
				self.config_values[ 'response_mime_type' ] = (self.response_format)
			
			if self.instructions:
				self.config_values[ 'system_instruction' ] = (self.instructions)
			
			self.content_config = GenerateContentConfig( **self.config_values )
			self.genai_client = self.get_genai_client( self.project_id, self.location, )
			self.response = self.genai_client.models.generate_content( model=self.model,
				contents=self.contents, config=self.content_config, )
			self.output_text = self.get_output_text( )
			throw_if( 'output_text', self.output_text )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'search( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def query( self, bucket: str, query: str, model: str, project_id: str = '',
		location: str = 'us-central1', prefix: str = '', max_files: int = 20,
		temperature: float = 0.0, top_p: float = 0.0, frequency: float = 0.0, presence: float =
		0.0,
		max_tokens: int = 0, response_format: str = '', instruct: str = '' ) -> str:
		"""Query bucket content.
		
		Purpose:
			Provides the application-compatible alias for Gemini bucket-content queries.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			query (str): Required question about bucket content.
			model (str): Required Gemini model identifier.
			project_id (str): Google Cloud project identifier.
			location (str): Vertex AI region.
			prefix (str): Optional object-name prefix.
			max_files (int): Maximum number of bucket objects included.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			response_format (str): Optional response MIME type.
			instruct (str): Optional system instruction.
		
		Returns:
			str: Generated answer based on bucket content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.bucket_name = bucket
			self.query_text = query
			self.model = model
			self.project_id = (project_id if project_id else self.project_id)
			self.location = location
			self.prefix = prefix
			self.max_files = max_files
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.response_format = response_format
			self.instructions = instruct
			return self.search( self.bucket_name, self.query_text, self.model, self.project_id,
				self.location, self.prefix, self.max_files, self.temperature, self.top_p,
				self.frequency_penalty, self.presence_penalty, self.max_tokens,
				self.response_format, self.instructions, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'query( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def ask( self, bucket: str, query: str, model: str, project_id: str = '',
		location: str = 'us-central1', prefix: str = '', max_files: int = 20,
		temperature: float = 0.0, top_p: float = 0.0, frequency: float = 0.0, presence: float =
		0.0,
		max_tokens: int = 0, response_format: str = '', instruct: str = '' ) -> str:
		"""Query bucket content.
		
		Purpose:
			Provides the application-compatible ask alias for Gemini bucket-content queries.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			query (str): Required question about bucket content.
			model (str): Required Gemini model identifier.
			project_id (str): Google Cloud project identifier.
			location (str): Vertex AI region.
			prefix (str): Optional object-name prefix.
			max_files (int): Maximum number of bucket objects included.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			response_format (str): Optional response MIME type.
			instruct (str): Optional system instruction.
		
		Returns:
			str: Generated answer based on bucket content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.bucket_name = bucket
			self.query_text = query
			self.model = model
			self.project_id = (project_id if project_id else self.project_id)
			self.location = location
			self.prefix = prefix
			self.max_files = max_files
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.response_format = response_format
			self.instructions = instruct
			return self.search( self.bucket_name, self.query_text, self.model, self.project_id,
				self.location, self.prefix, self.max_files, self.temperature, self.top_p,
				self.frequency_penalty, self.presence_penalty, self.max_tokens,
				self.response_format, self.instructions, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'ask( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, bucket: str, prompt: str, model: str, project_id: str = '',
		location: str = 'us-central1', prefix: str = '', max_files: int = 20,
		temperature: float = 0.0, top_p: float = 0.0, frequency: float = 0.0, presence: float =
		0.0,
		max_tokens: int = 0, response_format: str = '', instruct: str = '' ) -> str:
		"""Generate text from bucket content.
		
		Purpose:
			Provides the application-compatible generate-text alias for Gemini bucket-content
			queries.
		
		Args:
			bucket (str): Required Google Cloud Storage bucket name.
			prompt (str): Required question about bucket content.
			model (str): Required Gemini model identifier.
			project_id (str): Google Cloud project identifier.
			location (str): Vertex AI region.
			prefix (str): Optional object-name prefix.
			max_files (int): Maximum number of bucket objects included.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			response_format (str): Optional response MIME type.
			instruct (str): Optional system instruction.
		
		Returns:
			str: Generated answer based on bucket content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.bucket_name = bucket
			self.query_text = prompt
			self.model = model
			self.project_id = (project_id if project_id else self.project_id)
			self.location = location
			self.prefix = prefix
			self.max_files = max_files
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.response_format = response_format
			self.instructions = instruct
			return self.search( self.bucket_name, self.query_text, self.model, self.project_id,
				self.location, self.prefix, self.max_files, self.temperature, self.top_p,
				self.frequency_penalty, self.presence_penalty, self.max_tokens,
				self.response_format, self.instructions, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = 'generate_text( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Gemini CloudBuckets wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'project_id', 'location', 'bucket_name', 'object_name', 'file_path', 'model',
			'query_text', 'instructions', 'temperature', 'top_p', 'frequency_penalty',
			'presence_penalty', 'max_tokens', 'response_format', 'max_files', 'storage_client',
			'genai_client', 'client', 'bucket', 'blob', 'blobs', 'response', 'output_text',
			'collections', 'documents', 'model_options', 'media_options', 'format_options',
			'reasoning_options', 'choice_options', 'get_storage_client', 'get_genai_client',
			'get_bucket_metadata', 'get_blob_metadata', 'create', 'create_bucket', 'upload_file',
			'upload', 'files_upload', 'retrieve', 'retrieve_bucket', 'get', 'list', 'list_objects',
			'delete', 'delete_bucket', 'remove', 'get_output_text', 'search', 'query', 'ask',
			'generate_text', ]
