"""Gemini provider wrappers for the Jeni Streamlit application.

Purpose:
    Provides application-facing wrapper classes for Gemini text generation, image generation
    and analysis, embeddings, transcription, translation, text-to-speech, Gemini file
    operations, file-search stores, and Google Cloud Storage bucket workflows.

    The module centralizes provider request construction, option lists, response extraction,
    and error logging so the Streamlit application can call stable Python interfaces instead
    of provider-specific SDK objects directly.

Classes:
    Gemini: Shared base class for Gemini wrapper configuration and runtime state.
    Chat: Text-generation wrapper for Gemini chat and grounding workflows.
    Images: Image generation, analysis, and editing wrapper.
    Embeddings: Text embedding wrapper.
    TTS: Text-to-speech wrapper.
    Transcription: Audio transcription wrapper.
    Translation: Audio translation wrapper.
    Files: Gemini file and document workflow wrapper.
    FileSearch: Gemini file-search store wrapper.
    CloudBuckets: Google Cloud Storage bucket wrapper.

Copyright:
    Copyright © 2026 Terry Eppler.

License:
    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the software, subject to inclusion of
    the copyright notice and permission notice in substantial portions of the software.

Contact:
    Terry Eppler, terryeppler@gmail.com or eppler.terry@epa.gov.
"""
from __future__ import annotations
from google.genai.file_search_stores import FileSearchStores
import config as cfg
import base64
from boogr import Error, Logger
import io
import json
import os
import requests
import PIL.Image
from typing import Any, Callable, Dict, List, Optional, Union, Set, Tuple
from pathlib import Path
from google.cloud.storage.blob import Blob
from google import genai
from google.cloud import storage
from google.genai import types
from google.genai.pagers import Pager
from google.genai.types import (Part, GenerateContentConfig, ImageConfig, FunctionCallingConfig,
                                GenerateImagesConfig, GenerateVideosConfig, ThinkingConfig,
                                GeneratedImage, EmbedContentConfig, Content, ContentEmbedding,
                                Candidate, HttpOptions, GenerateImagesResponse, Field,
                                FileSearchStore, FileSearch, GenerateContentResponse,
                                GenerateVideosResponse, Image, File, SpeakerVoiceConfig,
                                VoiceConfig, SpeechConfig, Tool, ToolConfig, GoogleSearch,
                                UrlContext, SafetySetting, HarmCategory, HarmBlockThreshold,
                                EmbedContentResponse)

def throw_if( name: str, value: object ) -> None:
	"""Validate that a required argument has a usable value.

	Purpose:
		Provides a shared argument guard for provider calls, file operations, and configuration
		builders. The helper raises a clear ``ValueError`` when a required value is missing,
		blank, or an empty collection.

	Args:
		name (str): Name of the argument being validated.
		value (object): Runtime value to validate.

	Raises:
		ValueError: Raised when ``value`` is ``None``, blank, or an empty collection.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encode a local image file as base64 text.

	Purpose:
		Reads a local image file and converts its binary content into a base64-encoded string.
		This supports workflows that need inline image data instead of a file handle or URI.

	Args:
		image_path (str): Path to the local image file.

	Returns:
		Base64-encoded image content.
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( "utf-8" )

class Gemini( ):
	"""Shared configuration state for Gemini provider wrappers.

	Purpose:
		Initializes the API credentials, model parameters, generation settings, response
		configuration, and tool selections shared by the specialized Gemini wrapper classes.
		The base class performs no provider requests and contains no workflow-specific processing.

	Attributes:
		google_api_key (Optional[str]): Google API key loaded from application configuration.
		gemini_api_key (Optional[str]): Gemini API key loaded from application configuration.
		model (Optional[str]): Active Gemini model identifier.
		api_version (Optional[str]): API version selected by a specialized wrapper.
		temperature (Optional[float]): Sampling temperature.
		top_p (Optional[float]): Top-p sampling value.
		top_k (Optional[int]): Top-k sampling value.
		candidate_count (Optional[int]): Requested candidate count.
		frequency_penalty (Optional[float]): Frequency-penalty value.
		presence_penalty (Optional[float]): Presence-penalty value.
		max_tokens (Optional[int]): Maximum output-token count.
		instructions (Optional[str]): System instruction text.
		prompt (Optional[str]): Active user prompt.
		response_format (Optional[str]): Requested response format.
		number (Optional[int]): Requested number of generated results.
		response_modalities (List[str]): Requested response modalities.
		stops (List[str]): Stop sequences.
		domains (List[str]): Domain restrictions used by supported search workflows.
		tools (List[str]): Selected provider tools.
		tool_choice (Optional[str]): Tool-selection behavior.
		content_response (Optional[GenerateContentResponse]): Latest content response.
		image_response (Optional[GenerateImagesResponse]): Latest image-generation response.
		content_config (Optional[GenerateContentConfig]): Content-generation configuration.
		function_config (Optional[FunctionCallingConfig]): Function-calling configuration.
		thought_config (Optional[ThinkingConfig]): Thinking configuration.
		genimg_config (Optional[GenerateImagesConfig]): Image-generation configuration.
		image_config (Optional[ImageConfig]): Image-output configuration.
		tool_config (Optional[List[types.Tool]]): Provider tool configuration.
	"""
	
	google_api_key: Optional[ str ]
	gemini_api_key: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_tokens: Optional[ int ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	number: Optional[ int ]
	response_modalities: List[ str ]
	stops: List[ str ]
	domains: List[ str ]
	tools: List[ str ]
	tool_choice: Optional[ str ]
	content_response: Optional[ GenerateContentResponse ]
	image_response: Optional[ GenerateImagesResponse ]
	content_config: Optional[ GenerateContentConfig ]
	function_config: Optional[ FunctionCallingConfig ]
	thought_config: Optional[ ThinkingConfig ]
	genimg_config: Optional[ GenerateImagesConfig ]
	image_config: Optional[ ImageConfig ]
	tool_config: Optional[ List[ types.Tool ] ]
	
	def __init__( self ) -> None:
		"""Initialize shared Gemini wrapper state.

		Purpose:
			Initializes common credentials, model parameters, request configuration, tool
			selections, and response placeholders used by the specialized Gemini wrappers. The
			constructor performs local state assignment only.

		Returns:
			None: This method initializes object state through side effects.
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
		self.domains = [ ]
		self.tools = [ ]
		self.tool_choice = None
		self.content_response = None
		self.image_response = None
		self.content_config = None
		self.function_config = None
		self.thought_config = None
		self.genimg_config = None
		self.image_config = None
		self.tool_config = None

class Chat( Gemini ):
	"""Gemini text-generation wrapper.

	Purpose:
		Executes text-generation, grounding, structured-output, streaming, URL Context,
		File Search, Google Search, Google Maps, and Code Execution workflows through the
		Gemini Interactions API. The class preserves the application-facing contract used by
		Jeni while maintaining conversation history through the existing client-managed
		application state.

	Attributes:
		use_vertex (bool): Whether Vertex AI configuration is enabled.
		http_options (HttpOptions): Gemini client HTTP configuration.
		client (Optional[genai.Client]): Gemini SDK client.
		storage_client (Optional[storage.Client]): Optional Google Cloud Storage client.
		contents (Optional[List[Dict[str, Any]]]): Complete Interactions input timeline.
		input_steps (List[Dict[str, Any]]): Interactions input steps submitted to Gemini.
		image_uri (Optional[str]): Optional image URI retained for interface compatibility.
		audio_uri (Optional[str]): Optional audio URI retained for interface compatibility.
		file_path (Optional[str]): Optional local file path retained for compatibility.
		files (List[str]): File identifiers retained for interface compatibility.
		content_block (str): Additional content prepended to the active prompt.
		context (List[Dict[str, Any]]): Existing application-managed conversation history.
		urls (List[str]): URL-context values.
		max_urls (int): Maximum URL values included in the prompt.
		response_schema (Optional[Dict[str, Any]]): Parsed structured-output schema.
		safety_profile (str): Safety-profile value retained for UI compatibility.
		safety_settings (Optional[List[SafetySetting]]): Optional Gemini safety settings.
		file_search_store_names (List[str]): File Search Store resource names.
		interaction (Optional[Any]): Most recent Gemini Interaction.
		interaction_id (Optional[str]): Identifier of the most recent Interaction.
		steps (List[Any]): Steps returned by the most recent Interaction.
		response (Optional[Any]): Raw response used by application token accounting.
		output_text (str): Generated response text.
		grounding_sources (List[Dict[str, Any]]): Normalized grounding citations.
		generation_config (Dict[str, Any]): Interactions generation configuration.
		interaction_response_format (Optional[Any]): Interactions response-format value.
		tool_objects (List[Dict[str, Any]]): Interactions server-side tool definitions.
		stream (bool): Whether streaming is enabled.
		stream_handler (Optional[Callable[[str], None]]): Text-delta callback.
	"""
	
	use_vertex: bool
	http_options: HttpOptions
	client: Optional[ genai.Client ]
	storage_client: Optional[ storage.Client ]
	contents: Optional[ List[ Dict[ str, Any ] ] ]
	input_steps: List[ Dict[ str, Any ] ]
	image_uri: Optional[ str ]
	audio_uri: Optional[ str ]
	file_path: Optional[ str ]
	files: List[ str ]
	content_block: str
	context: List[ Dict[ str, Any ] ]
	urls: List[ str ]
	max_urls: int
	response_schema: Optional[ Dict[ str, Any ] ]
	safety_profile: str
	safety_settings: Optional[ List[ SafetySetting ] ]
	file_search_store_names: List[ str ]
	interaction: Optional[ Any ]
	interaction_id: Optional[ str ]
	steps: List[ Any ]
	response: Optional[ Any ]
	output_text: str
	grounding_sources: List[ Dict[ str, Any ] ]
	generation_config: Dict[ str, Any ]
	interaction_response_format: Optional[ Any ]
	tool_objects: List[ Dict[ str, Any ] ]
	stream: bool
	stream_handler: Optional[ Callable[ [ str ], None ] ]
	
	def __init__( self, model: str = 'gemini-2.5-flash-lite' ) -> None:
		"""Initialize the Chat wrapper.

		Purpose:
			Initializes text-generation configuration, Interactions request state, grounding
			state, conversation-history state, and response placeholders. The constructor
			performs local state assignment only.

		Args:
			model (str): Default Gemini text-generation model.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.api_version = 'v1beta'
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.use_vertex = False
		self.client = None
		self.storage_client = None
		self.model = model
		self.prompt = None
		self.instructions = None
		self.number = 1
		self.candidate_count = 1
		self.temperature = 0.0
		self.top_p = 0.0
		self.top_k = 0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_tokens = 0
		self.stops = [ ]
		self.response_format = None
		self.response_schema = None
		self.response_modalities = [ ]
		self.media_resolution = None
		self.tool_choice = None
		self.tools = [ ]
		self.tool_objects = [ ]
		self.generation_config = { }
		self.interaction_response_format = None
		self.safety_profile = ''
		self.safety_settings = None
		self.contents = None
		self.input_steps = [ ]
		self.content_block = ''
		self.context = [ ]
		self.urls = [ ]
		self.max_urls = 0
		self.files = [ ]
		self.file_search_store_names = [ ]
		self.image_uri = None
		self.audio_uri = None
		self.file_path = None
		self.interaction = None
		self.interaction_id = None
		self.steps = [ ]
		self.response = None
		self.content_response = None
		self.output_text = ''
		self.grounding_metadata = None
		self.grounding_sources = [ ]
		self.stream = False
		self.stream_handler = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported Gemini text-generation models.

		Purpose:
			Provides the text-generation model identifiers exposed by the Jeni Text,
			Document Q&A, File Search Stores, and Google Cloud Buckets interfaces.

		Returns:
			List[str]: Supported Gemini text-generation model identifiers.
		"""
		return [ 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro',
			'gemini-3.1-flash-lite', 'gemini-3.1-pro-preview', 'gemini-3.5-flash',
			'gemini-3.6-flash', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Return supported Interactions server-side tools.

		Purpose:
			Provides the server-side tools exposed by the Jeni Text interface.

		Returns:
			List[str]: Supported Interactions tool identifiers.
		"""
		return [ 'google_search', 'google_maps', 'url_context', 'file_search', 'code_execution', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Return supported thinking levels.

		Purpose:
			Provides the thinking-level values exposed by the Jeni model controls.

		Returns:
			List[str]: Supported thinking-level values.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH', ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Return supported media-resolution values.

		Purpose:
			Preserves the media-resolution option contract used by the Jeni interface.

		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Return supported tool-choice values.

		Purpose:
			Provides the tool-selection values accepted by the Interactions API.

		Returns:
			List[str]: Supported tool-choice values.
		"""
		return [ 'auto', 'any', 'none', 'validated', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Return compatibility include options.

		Purpose:
			Preserves the option values consumed by existing Jeni controls while citations
			and tool execution are retrieved from typed Interaction steps.

		Returns:
			List[str]: Existing Jeni include-option values.
		"""
		return [ 'file_search_call.results', 'message.input_image.image_url',
			'message.output_text.logprobs', 'reasoning.encrypted_content', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Return supported response modalities.

		Purpose:
			Provides response-modality values used by the Jeni Text interface.

		Returns:
			List[str]: Supported response modalities.
		"""
		return [ '', 'text', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Return supported text response MIME types.

		Purpose:
			Provides text and structured-output MIME types used by the Text interface.

		Returns:
			List[str]: Supported response MIME types.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum', ]
	
	def get_supported_tools( self, model: str ) -> List[ str ]:
		"""Return tools supported by the selected model.

		Purpose:
			Builds the tool list consumed by the Jeni Text controls and conditionally includes
			Google Maps for models that expose that capability.

		Args:
			model (str): Gemini model identifier.

		Returns:
			List[str]: Tool identifiers supported by the selected model.

		Raises:
			Error: Raised when validation or tool-list construction fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.options = [ 'google_search', 'url_context', 'file_search', 'code_execution', ]
			
			if self.supports_google_maps( self.model ):
				self.options.append( 'google_maps' )
			
			return self.options
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'get_supported_tools( self, model: str ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def supports_google_maps( self, model: str ) -> bool:
		"""Return whether a model supports Google Maps grounding.

		Purpose:
			Centralizes Google Maps feature gating for the Jeni Text controls.

		Args:
			model (str): Gemini model identifier.

		Returns:
			bool: True when Google Maps may be exposed; otherwise False.

		Raises:
			Error: Raised when validation or model comparison fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.model_name = self.model.strip( ).lower( )
			self.maps_models = { 'gemini-3.5-flash', 'gemini-3.6-flash', }
			return self.model_name in self.maps_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'supports_google_maps( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def normalize_value( self, value: Any ) -> Any:
		"""Convert an SDK value into standard Python data.

		Purpose:
			Recursively converts SDK response models, dictionaries, and sequences into
			ordinary Python values used by history and citation processing.

		Args:
			value (Any): SDK or Python value to normalize.

		Returns:
			Any: Equivalent standard Python value.

		Raises:
			Error: Raised when value normalization fails.
		"""
		try:
			self.value = value
			
			if self.value is None or isinstance( self.value, (str, int, float, bool) ):
				return self.value
			
			if isinstance( self.value, dict ):
				return { key: self.normalize_value( item ) for key, item in self.value.items( ) }
			
			if isinstance( self.value, (list, tuple, set) ):
				return [ self.normalize_value( item ) for item in self.value ]
			
			if hasattr( self.value, 'model_dump' ):
				self.value_data = self.value.model_dump( exclude_none=True )
				return self.normalize_value( self.value_data )
			
			if hasattr( self.value, 'to_dict' ):
				self.value_data = self.value.to_dict( )
				return self.normalize_value( self.value_data )
			
			return str( self.value )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'normalize_value( self, value: Any ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def normalize_context( self, context: Optional[ List[ Dict[ str, Any ] ] ] = None ) -> List[
		Dict[ str, Any ] ]:
		"""Convert application history into Interactions steps.

		Purpose:
			Preserves existing Interactions steps and converts Jeni role/content dictionaries
			into ``user_input`` and ``model_output`` steps for stateless conversation history.

		Args:
			context (Optional[List[Dict[str, Any]]]): Existing conversation history.

		Returns:
			List[Dict[str, Any]]: Interactions-compatible history steps.

		Raises:
			Error: Raised when conversation history cannot be normalized.
		"""
		try:
			self.context = context if isinstance( context, list ) else [ ]
			self.history_steps: List[ Dict[ str, Any ] ] = [ ]
			
			for message in self.context:
				if not isinstance( message, dict ):
					continue
				
				self.context_type = str( message.get( 'type', '' ) or '' ).strip( )
				if self.context_type:
					self.context_step = self.normalize_value( message )
					if isinstance( self.context_step, dict ) and self.context_step:
						self.history_steps.append( self.context_step )
					
					continue
				
				self.message_role = str( message.get( 'role', '' ) or '' ).strip( ).lower( )
				self.message_content = message.get( 'content', '' )
				
				if isinstance( self.message_content, list ):
					self.message_text = '\n'.join(
						str( item ).strip( ) for item in self.message_content if
							item is not None and str( item ).strip( ) )
				else:
					self.message_text = str( self.message_content or '' ).strip( )
				
				if not self.message_text:
					continue
				
				if self.message_role in ('assistant', 'model'):
					self.step_type = 'model_output'
				else:
					self.step_type = 'user_input'
				
				self.history_steps.append( { 'type': self.step_type,
					'content': [ { 'type': 'text', 'text': self.message_text, }, ], } )
			
			self.context = self.history_steps
			return self.context
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = (
				'normalize_context( self, context: Optional[ List[ Dict[ str, Any ] ] ] ) '
				'-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def build_urls( self, urls: Optional[ List[ str ] ] = None, max_urls: int = 0 ) -> List[ str ]:
		"""Build the normalized URL list.

		Purpose:
			Removes blank and duplicate URL values while preserving order and applies the
			configured maximum before URL values are supplied to the active request.

		Args:
			urls (Optional[List[str]]): Candidate URL values.
			max_urls (int): Maximum URLs to retain; zero retains all values.

		Returns:
			List[str]: Normalized URL values.

		Raises:
			Error: Raised when URL normalization fails.
		"""
		try:
			self.urls = urls if isinstance( urls, list ) else [ ]
			self.max_urls = max_urls
			self.normalized_urls: List[ str ] = [ ]
			
			for url in self.urls:
				if url is None:
					continue
				
				self.url = str( url ).strip( )
				if not self.url:
					continue
				
				if self.url not in self.normalized_urls:
					self.normalized_urls.append( self.url )
			
			if self.max_urls > 0:
				self.normalized_urls = self.normalized_urls[ :self.max_urls ]
			
			self.urls = self.normalized_urls
			return self.urls
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_urls( self, urls: Optional[ List[ str ] ], max_urls: int ) '
			                    '-> List[ str ]')
			Logger( ).write( exception )
			raise exception
	
	def build_input( self, prompt: str, content: str = '',
		context: Optional[ List[ Dict[ str, Any ] ] ] = None, urls: Optional[ List[ str ] ] = None,
		max_urls: int = 0 ) -> List[ Dict[ str, Any ] ]:
		"""Build the complete Interactions input timeline.

		Purpose:
			Combines existing client-managed conversation history with optional content,
			reference URLs, and the current user prompt.

		Args:
			prompt (str): Current user prompt.
			content (str): Optional content prepended to the prompt.
			context (Optional[List[Dict[str, Any]]]): Existing conversation history.
			urls (Optional[List[str]]): URL-context values.
			max_urls (int): Maximum number of URLs.

		Returns:
			List[Dict[str, Any]]: Complete Interactions input timeline.

		Raises:
			Error: Raised when validation or input construction fails.
			ValueError: Raised when ``prompt`` is missing.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.content_block = content
			self.context = context if isinstance( context, list ) else [ ]
			self.urls = urls if isinstance( urls, list ) else [ ]
			self.max_urls = max_urls
			self.input_steps = self.normalize_context( self.context )
			self.urls = self.build_urls( self.urls, self.max_urls )
			self.current_parts: List[ str ] = [ ]
			
			if self.content_block and self.content_block.strip( ):
				self.current_parts.append( self.content_block.strip( ) )
			
			if self.urls:
				self.current_parts.append(
					'Reference URLs:\n' + '\n'.join( f'- {url}' for url in self.urls ) )
			
			self.current_parts.append( self.prompt.strip( ) )
			self.current_text = '\n\n'.join( part for part in self.current_parts if part ).strip( )
			
			self.input_steps.append( { 'type': 'user_input',
				'content': [ { 'type': 'text', 'text': self.current_text, }, ], } )
			
			self.contents = self.input_steps
			return self.input_steps
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_input( self, prompt: str, content: str, '
			                    'context: Optional[ List[ Dict[ str, Any ] ] ], '
			                    'urls: Optional[ List[ str ] ], max_urls: int ) '
			                    '-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def build_generation_config( self, temperature: float = 0.0, top_p: float = 0.0, top_k: int
	= 0,
		max_tokens: int = 0, stops: Optional[ List[ str ] ] = None, reasoning: str = '',
		tool_choice: Optional[ str ] = None ) -> Dict[ str, Any ]:
		"""Build the Interactions generation configuration.

		Purpose:
			Converts supported Jeni inference controls into the Interactions generation
			configuration without submitting blank or zero-valued optional controls.

		Args:
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			top_k (int): Top-k sampling value.
			max_tokens (int): Maximum output-token count.
			stops (Optional[List[str]]): Stop sequences.
			reasoning (str): Thinking-level value.
			tool_choice (Optional[str]): Tool-selection behavior.

		Returns:
			Dict[str, Any]: Interactions generation configuration.

		Raises:
			Error: Raised when generation configuration construction fails.
		"""
		try:
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.max_tokens = max_tokens
			self.stops = stops if isinstance( stops, list ) else [ ]
			self.reasoning = reasoning
			self.tool_choice = tool_choice
			self.generation_config = { }
			
			if self.temperature > 0.0:
				self.generation_config[ 'temperature' ] = self.temperature
			
			if self.top_p > 0.0:
				self.generation_config[ 'top_p' ] = self.top_p
			
			if self.top_k > 0:
				self.generation_config[ 'top_k' ] = self.top_k
			
			if self.max_tokens > 0:
				self.generation_config[ 'max_output_tokens' ] = self.max_tokens
			
			self.stop_sequences = [ str( value ).strip( ) for value in self.stops if
				value is not None and str( value ).strip( ) ]
			
			if self.stop_sequences:
				self.generation_config[ 'stop_sequences' ] = self.stop_sequences
			
			self.thinking_level = str( self.reasoning or '' ).strip( ).lower( )
			
			if self.thinking_level not in ('', 'thinking_level_unspecified', 'unspecified',):
				self.generation_config[ 'thinking_level' ] = self.thinking_level
			
			self.tool_choice_value = str( self.tool_choice or '' ).strip( ).lower( )
			
			if self.tool_choice_value:
				self.generation_config[ 'tool_choice' ] = self.tool_choice_value
			
			return self.generation_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_generation_config( self, temperature: float, top_p: float, '
			                    'top_k: int, max_tokens: int, stops: Optional[ List[ str ] ], '
			                    'reasoning: str, tool_choice: Optional[ str ] ) '
			                    '-> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def parse_response_schema( self, response_schema: Any = None ) -> Optional[ Dict[ str, Any ] ]:
		"""Parse an optional structured-output schema.

		Purpose:
			Accepts a dictionary or JSON string and converts it into the JSON Schema mapping
			required by the Interactions API.

		Args:
			response_schema (Any): Optional JSON Schema dictionary or JSON string.

		Returns:
			Optional[Dict[str, Any]]: Parsed JSON Schema, or None when absent.

		Raises:
			Error: Raised when a nonblank schema cannot be parsed.
		"""
		try:
			self.response_schema = response_schema
			
			if self.response_schema is None:
				return None
			
			if isinstance( self.response_schema, dict ):
				return self.response_schema
			
			self.schema_text = str( self.response_schema ).strip( )
			if not self.schema_text:
				self.response_schema = None
				return None
			
			self.schema_value = json.loads( self.schema_text )
			if not isinstance( self.schema_value, dict ):
				raise ValueError( 'The response schema must contain a JSON object.' )
			
			self.response_schema = self.schema_value
			return self.response_schema
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('parse_response_schema( self, response_schema: Any ) '
			                    '-> Optional[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def build_response_format( self, response_format: str = '', response_schema: Any = None,
		modalities: Optional[ List[ str ] ] = None ) -> Optional[ Any ]:
		"""Build the Interactions response format.

		Purpose:
			Converts the Jeni MIME type, response modalities, and optional JSON Schema into
			the polymorphic response-format structure used by the current Interactions API.

		Args:
			response_format (str): Requested text MIME type.
			response_schema (Any): Optional JSON Schema mapping or JSON string.
			modalities (Optional[List[str]]): Requested response modalities.

		Returns:
			Optional[Any]: Interactions response-format value.

		Raises:
			Error: Raised when response-format construction fails.
		"""
		try:
			self.response_format = response_format
			self.response_schema = response_schema
			self.response_modalities = (modalities if isinstance( modalities, list ) else [ ])
			self.response_schema = self.parse_response_schema( self.response_schema )
			self.response_mime_type = str( self.response_format or '' ).strip( )
			self.normalized_modalities = [ str( value ).strip( ).lower( ) for value in
				self.response_modalities if value is not None and str( value ).strip( ) ]
			
			if self.normalized_modalities and 'text' not in self.normalized_modalities:
				self.interaction_response_format = None
				return None
			
			if self.response_schema is not None:
				self.interaction_response_format = [
					{ 'type': 'text', 'mime_type': 'application/json',
						'schema': self.response_schema, }, ]
				return self.interaction_response_format
			
			if self.response_mime_type in ('application/json', 'text/x.enum'):
				self.interaction_response_format = [
					{ 'type': 'text', 'mime_type': self.response_mime_type, }, ]
				return self.interaction_response_format
			
			self.interaction_response_format = None
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_response_format( self, response_format: str, '
			                    'response_schema: Any, modalities: Optional[ List[ str ] ] ) '
			                    '-> Optional[ Any ]')
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: Optional[ List[ str ] ] = None,
		urls: Optional[ List[ str ] ] = None,
		file_search_store_names: Optional[ List[ str ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""Build Interactions server-side tools.

		Purpose:
			Translates Jeni tool identifiers into Interactions tool declarations. URL Context
			is enabled when selected or when URLs are supplied. File Search is enabled when
			File Search Store resource names are supplied.

		Args:
			tools (Optional[List[str]]): Selected tool identifiers.
			urls (Optional[List[str]]): Normalized URL-context values.
			file_search_store_names (Optional[List[str]]): File Search Store names.

		Returns:
			List[Dict[str, Any]]: Interactions server-side tool declarations.

		Raises:
			Error: Raised when tool construction fails.
		"""
		try:
			self.tools = tools if isinstance( tools, list ) else [ ]
			self.urls = urls if isinstance( urls, list ) else [ ]
			self.file_search_store_names = (
				file_search_store_names if isinstance( file_search_store_names, list ) else [ ])
			self.selected_tools = [ str( value ).strip( ).lower( ) for value in self.tools if
				value is not None and str( value ).strip( ) ]
			self.file_search_store_names = [ str( value ).strip( ) for value in
				self.file_search_store_names if value is not None and str( value ).strip( ) ]
			self.tool_objects = [ ]
			
			if 'google_search' in self.selected_tools:
				self.tool_objects.append( { 'type': 'google_search', } )
			
			if ('google_maps' in self.selected_tools and self.supports_google_maps( self.model )):
				self.tool_objects.append( { 'type': 'google_maps', } )
			
			if 'url_context' in self.selected_tools or self.urls:
				self.tool_objects.append( { 'type': 'url_context', } )
			
			if 'code_execution' in self.selected_tools:
				self.tool_objects.append( { 'type': 'code_execution', } )
			
			if self.file_search_store_names:
				self.tool_objects.append( { 'type': 'file_search',
					'file_search_store_names': self.file_search_store_names, } )
			
			return self.tool_objects
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('build_tools( self, tools: Optional[ List[ str ] ], '
			                    'urls: Optional[ List[ str ] ], '
			                    'file_search_store_names: Optional[ List[ str ] ] ) '
			                    '-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def append_source( self, source: Dict[ str, Any ], default_type: str ) -> None:
		"""Append one normalized grounding source.

		Purpose:
			Converts provider citation and tool-result fields into the stable source shape
			consumed by Jeni and prevents duplicate records.

		Args:
			source (Dict[str, Any]): Provider citation or result mapping.
			default_type (str): Source type used when omitted by the mapping.

		Returns:
			None: This method updates source state through side effects.

		Raises:
			Error: Raised when validation or source normalization fails.
			ValueError: Raised when a required argument is missing.
		"""
		try:
			throw_if( 'source', source )
			throw_if( 'default_type', default_type )
			self.source = source
			self.default_type = default_type
			self.source_type = str(
				self.source.get( 'type', self.default_type ) or self.default_type ).strip( )
			self.source_title = str(
				self.source.get( 'title' ) or self.source.get( 'display_name' ) or self.source.get(
					'file_name' ) or self.source.get( 'name' ) or '' ).strip( )
			self.source_url = str(
				self.source.get( 'uri' ) or self.source.get( 'url' ) or self.source.get(
					'source_url' ) or '' ).strip( )
			self.source_text = str(
				self.source.get( 'text' ) or self.source.get( 'snippet' ) or self.source.get(
					'quote' ) or self.source.get( 'search_suggestions' ) or '' ).strip( )
			self.source_file_id = str(
				self.source.get( 'file_id' ) or self.source.get( 'file_name' ) or self.source.get(
					'document_name' ) or '' ).strip( )
			
			if not any( [ self.source_title, self.source_url, self.source_text,
				self.source_file_id, ] ):
				return
			
			self.source_key = (self.source_type, self.source_url or self.source_file_id,
				self.source_text,)
			
			if self.source_key in self.source_keys:
				return
			
			self.source_keys.add( self.source_key )
			self.source_values.append(
				{ 'type': self.source_type, 'title': self.source_title or None,
					'snippet': self.source_text or None, 'url': self.source_url or None,
					'files_id': self.source_file_id or None, 'metadata': self.source, } )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('append_source( self, source: Dict[ str, Any ], '
			                    'default_type: str ) -> None')
			Logger( ).write( exception )
			raise exception
	
	def extract_grounding_sources( self, interaction: Any ) -> List[ Dict[ str, Any ] ]:
		"""Extract grounding citations from an Interaction.

		Purpose:
			Collects model-output annotations and supported server-side tool-result data
			from the current Interaction steps.

		Args:
			interaction (Any): Completed Gemini Interaction.

		Returns:
			List[Dict[str, Any]]: Normalized grounding-source records.

		Raises:
			Error: Raised when validation or citation extraction fails.
			ValueError: Raised when ``interaction`` is missing.
		"""
		try:
			throw_if( 'interaction', interaction )
			self.source_interaction = interaction
			self.source_values: List[ Dict[ str, Any ] ] = [ ]
			self.source_keys: Set[ Tuple[ str, str, str ] ] = set( )
			self.source_steps = getattr( self.source_interaction, 'steps', None ) or [ ]
			
			for step in self.source_steps:
				self.source_step_type = str( getattr( step, 'type', '' ) or '' ).strip( )
				
				if self.source_step_type == 'model_output':
					self.source_content = getattr( step, 'content', None ) or [ ]
					
					for block in self.source_content:
						self.annotations = getattr( block, 'annotations', None ) or [ ]
						
						for annotation in self.annotations:
							self.annotation_value = self.normalize_value( annotation )
							
							if not isinstance( self.annotation_value, dict ):
								continue
							
							self.append_source( self.annotation_value, str(
								self.annotation_value.get( 'type', 'citation' ) or 'citation' ) )
				
				elif self.source_step_type in ('google_search_result', 'file_search_result',
					'url_context_result', 'google_maps_result', 'code_execution_result',):
					self.result_value = self.normalize_value( getattr( step, 'result', None ) )
					
					if isinstance( self.result_value, list ):
						for result in self.result_value:
							if isinstance( result, dict ) and result:
								self.append_source( result, self.source_step_type )
					
					elif isinstance( self.result_value, dict ) and self.result_value:
						self.append_source( self.result_value, self.source_step_type )
			
			return self.source_values
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('extract_grounding_sources( self, interaction: Any ) '
			                    '-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def capture_interaction( self, interaction: Any ) -> None:
		"""Capture a completed Gemini Interaction.

		Purpose:
			Stores the raw response, identifier, output steps, generated text, and grounding
			sources on the members consumed by the Jeni application.

		Args:
			interaction (Any): Completed Gemini Interaction.

		Returns:
			None: This method updates response state through side effects.

		Raises:
			Error: Raised when validation or response extraction fails.
			ValueError: Raised when ``interaction`` is missing.
		"""
		try:
			throw_if( 'interaction', interaction )
			self.interaction = interaction
			self.response = self.interaction
			self.content_response = self.interaction
			self.interaction_id = getattr( self.interaction, 'id', None )
			self.steps = list( getattr( self.interaction, 'steps', None ) or [ ] )
			self.output_text = str( getattr( self.interaction, 'output_text', '' ) or '' ).strip( )
			self.grounding_sources = self.extract_grounding_sources( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('capture_interaction( self, interaction: Any ) -> None')
			Logger( ).write( exception )
			raise exception
	
	def get_grounding_sources( self ) -> List[ Dict[ str, Any ] ]:
		"""Return sources from the most recent Interaction.

		Purpose:
			Provides normalized grounding-source records consumed by Jeni response renderers.

		Returns:
			List[Dict[str, Any]]: Grounding sources from the latest response.
		"""
		return list( self.grounding_sources )
	
	def get_structured_history( self ) -> List[ Dict[ str, Any ] ]:
		"""Return the complete stateless conversation history.

		Purpose:
			Combines the submitted input timeline with the output steps returned by the latest
			Interaction so Jeni can preserve client-managed conversation state.

		Returns:
			List[Dict[str, Any]]: Complete Interactions-compatible history.

		Raises:
			Error: Raised when output steps cannot be normalized.
		"""
		try:
			self.structured_history: List[ Dict[ str, Any ] ] = [ ]
			
			for step in self.input_steps:
				self.input_step = self.normalize_value( step )
				if isinstance( self.input_step, dict ) and self.input_step:
					self.structured_history.append( self.input_step )
			
			for step in self.steps:
				self.output_step = self.normalize_value( step )
				if isinstance( self.output_step, dict ) and self.output_step:
					self.structured_history.append( self.output_step )
			
			return self.structured_history
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('get_structured_history( self ) '
			                    '-> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def generate_text_stream( self ) -> str:
		"""Execute the prepared streaming Interactions request.

		Purpose:
			Submits the active request, forwards text deltas to the application callback,
			accumulates the complete generated text, and captures final usage metadata.

		Returns:
			str: Complete generated text.

		Raises:
			Error: Raised when streaming execution or event handling fails.
		"""
		try:
			self.request = { 'model': self.model, 'input': self.input_steps, 'stream': True,
				'store': False, }
			
			if self.instructions:
				self.request[ 'system_instruction' ] = self.instructions
			
			if self.tool_objects:
				self.request[ 'tools' ] = self.tool_objects
			
			if self.generation_config:
				self.request[ 'generation_config' ] = self.generation_config
			
			if self.interaction_response_format is not None:
				self.request[ 'response_format' ] = (self.interaction_response_format)
			
			self.stream_response = self.client.interactions.create( **self.request )
			self.text_chunks: List[ str ] = [ ]
			self.completed_interaction = None
			
			for event in self.stream_response:
				self.event_type = str( getattr( event, 'event_type', '' ) or '' ).strip( )
				
				if self.event_type == 'step.delta':
					self.delta = getattr( event, 'delta', None )
					self.delta_type = str( getattr( self.delta, 'type', '' ) or '' ).strip( )
					
					if self.delta_type == 'text':
						self.delta_text = str( getattr( self.delta, 'text', '' ) or '' )
						
						if self.delta_text:
							self.text_chunks.append( self.delta_text )
							
							if callable( self.stream_handler ):
								self.stream_handler( self.delta_text )
				
				elif self.event_type == 'interaction.completed':
					self.completed_interaction = getattr( event, 'interaction', None )
				
				elif self.event_type == 'error':
					self.stream_error = getattr( event, 'error', None )
					self.stream_message = str( getattr( self.stream_error, 'message',
						'' ) or self.stream_error or 'Gemini streaming request failed.' )
					raise RuntimeError( self.stream_message )
			
			self.output_text = ''.join( self.text_chunks ).strip( )
			self.steps = [ ]
			
			if self.output_text:
				self.steps.append( { 'type': 'model_output',
					'content': [ { 'type': 'text', 'text': self.output_text, }, ], } )
			
			if self.completed_interaction is not None:
				self.interaction = self.completed_interaction
				self.response = self.completed_interaction
				self.content_response = self.completed_interaction
				self.interaction_id = getattr( self.completed_interaction, 'id', None )
			else:
				self.response = self.stream_response
				self.content_response = self.stream_response
			
			self.grounding_sources = [ ]
			
			if not self.output_text:
				raise ValueError( 'Gemini returned an empty streaming response.' )
			
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = 'generate_text_stream( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str, number: int = 1, temperature: float = 0.0,
		top_p: float = 0.0, top_k: int = 0, frequency: float = 0.0, presence: float = 0.0,
		max_tokens: int = 0, stops: Optional[ List[ str ] ] = None, instruct: str = '',
		response_format: str = '', tools: Optional[ List[ str ] ] = None,
		tool_choice: Optional[ str ] = None, reasoning: str = '',
		modalities: Optional[ List[ str ] ] = None, media_resolution: str = '',
		context: Optional[ List[ Dict[ str, Any ] ] ] = None, content: str = '',
		urls: Optional[ List[ str ] ] = None, max_urls: int = 0, response_schema: Any = '',
		safety_profile: str = '', file_search_store_names: Optional[ List[ str ] ] = None,
		stream: bool = False, stream_handler: Optional[ Callable[ [ str ], None ] ] = None ) -> (
			str):
		"""Generate text through the Gemini Interactions API.

		Purpose:
			Preserves Jeni's text-generation method contract while routing model execution
			through the Interactions API. Existing client-managed history is converted into
			Interactions steps and returned through ``get_structured_history()`` after the call.

		Args:
			prompt (str): Current user prompt.
			model (str): Gemini model identifier.
			number (int): Candidate-count value retained for compatibility.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			top_k (int): Top-k sampling value.
			frequency (float): Frequency penalty retained for compatibility.
			presence (float): Presence penalty retained for compatibility.
			max_tokens (int): Maximum output-token count.
			stops (Optional[List[str]]): Stop sequences.
			instruct (str): System instruction text.
			response_format (str): Requested response MIME type.
			tools (Optional[List[str]]): Server-side tool identifiers.
			tool_choice (Optional[str]): Tool-selection behavior.
			reasoning (str): Thinking-level value.
			modalities (Optional[List[str]]): Requested output modalities.
			media_resolution (str): Media-resolution value retained for compatibility.
			context (Optional[List[Dict[str, Any]]]): Client-managed conversation history.
			content (str): Additional application content.
			urls (Optional[List[str]]): URL-context values.
			max_urls (int): Maximum number of URLs.
			response_schema (Any): Optional JSON Schema mapping or JSON string.
			safety_profile (str): Safety-profile value retained for compatibility.
			file_search_store_names (Optional[List[str]]): File Search Store names.
			stream (bool): Whether to stream response text.
			stream_handler (Optional[Callable[[str], None]]): Text-delta callback.

		Returns:
			str: Generated response text.

		Raises:
			Error: Raised when validation, request construction, or execution fails.
			ValueError: Raised when a required prompt, model, or API key is missing.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.candidate_count = self.number
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops if isinstance( stops, list ) else [ ]
			self.instructions = instruct
			self.response_format = response_format
			self.tools = tools if isinstance( tools, list ) else [ ]
			self.tool_choice = tool_choice
			self.reasoning = reasoning
			self.response_modalities = (modalities if isinstance( modalities, list ) else [ ])
			self.media_resolution = media_resolution
			self.context = context if isinstance( context, list ) else [ ]
			self.content_block = content
			self.urls = urls if isinstance( urls, list ) else [ ]
			self.max_urls = max_urls
			self.response_schema = response_schema
			self.safety_profile = safety_profile
			self.file_search_store_names = (
				file_search_store_names if isinstance( file_search_store_names, list ) else [ ])
			self.stream = stream
			self.stream_handler = stream_handler
			self.api_key = (os.getenv( 'GEMINI_API_KEY' ) or os.getenv(
				'GOOGLE_API_KEY' ) or self.gemini_api_key or self.google_api_key)
			throw_if( 'api_key', self.api_key )
			
			self.client = genai.Client( api_key=self.api_key, http_options=self.http_options )
			
			self.input_steps = self.build_input( prompt=self.prompt, content=self.content_block,
				context=self.context, urls=self.urls, max_urls=self.max_urls )
			
			self.generation_config = self.build_generation_config( temperature=self.temperature,
				top_p=self.top_p, top_k=self.top_k, max_tokens=self.max_tokens, stops=self.stops,
				reasoning=self.reasoning, tool_choice=self.tool_choice )
			
			self.interaction_response_format = self.build_response_format(
				response_format=self.response_format, response_schema=self.response_schema,
				modalities=self.response_modalities )
			
			self.tool_objects = self.build_tools( tools=self.tools, urls=self.urls,
				file_search_store_names=self.file_search_store_names )
			
			if self.stream:
				return self.generate_text_stream( )
			
			self.request = { 'model': self.model, 'input': self.input_steps, 'stream': False,
				'store': False, }
			
			if self.instructions:
				self.request[ 'system_instruction' ] = self.instructions
			
			if self.tool_objects:
				self.request[ 'tools' ] = self.tool_objects
			
			if self.generation_config:
				self.request[ 'generation_config' ] = self.generation_config
			
			if self.interaction_response_format is not None:
				self.request[ 'response_format' ] = (self.interaction_response_format)
			
			self.interaction = self.client.interactions.create( **self.request )
			self.capture_interaction( self.interaction )
			
			if not self.output_text:
				raise ValueError( 'Gemini returned an empty Interactions response.' )
			
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Chat'
			exception.method = ('generate_text( self, prompt: str, model: str, number: int, '
			                    'temperature: float, top_p: float, top_k: int, frequency: float, '
			                    'presence: float, max_tokens: int, stops: Optional[List[str]], '
			                    'instruct: str, response_format: str, tools: Optional[List[str]], '
			                    'tool_choice: Optional[str], reasoning: str, modalities: Optional['
			                    'List[str]], media_resolution: str, context: Optional[List[Dict['
			                    'str, Any]]], content: str, urls: Optional[List[str]], max_urls: '
			                    'int, response_schema: Any, safety_profile: str, '
			                    'file_search_store_names: Optional[List[str]], stream: bool, '
			                    'stream_handler: Optional[Callable[[str], None]] ) -> str')
			Logger( ).write( exception )
			raise exception

class Images( Gemini ):
	"""Gemini image-generation, analysis, and editing wrapper.

	Purpose:
		Executes image generation, image understanding, and image editing through the Gemini
		Interactions API. The class converts local images into Interactions image-content
		blocks, builds image and text response formats, configures supported Google Search
		grounding, captures generated image or text output, and preserves the application-facing
		method contracts consumed by the Jeni Images mode.

	Attributes:
		client (Optional[genai.Client]): Active Gemini client.
		http_options (HttpOptions): Gemini client HTTP configuration.
		aspect_ratio (Optional[str]): Requested output-image aspect ratio.
		size (Optional[str]): Requested output-image size.
		resolution (Optional[str]): Media-resolution value retained for UI compatibility.
		max_output_tokens (Optional[int]): Maximum text-output token count.
		output_mime_type (Optional[str]): Requested output-image MIME type.
		response_mode (Optional[str]): Requested response-modality selection.
		response_format_value (Optional[Any]): Interactions response-format configuration.
		generation_config (Dict[str, Any]): Interactions generation configuration.
		input_content (List[Dict[str, Any]]): Multimodal Interactions input.
		tool_objects (List[Dict[str, Any]]): Interactions Google Search tool declarations.
		interaction (Optional[Any]): Most recent Gemini Interaction.
		response (Optional[Any]): Raw response used by application token accounting.
		output_text (str): Text extracted from the latest Interaction.
		output_image_content (Optional[Any]): Image content from the latest Interaction.
		grounding_metadata (Optional[Any]): Grounding data retained for UI compatibility.
		grounding_sources (List[Dict[str, Any]]): Normalized grounding records.
	"""
	
	client: Optional[ genai.Client ]
	http_options: HttpOptions
	aspect_ratio: Optional[ str ]
	size: Optional[ str ]
	resolution: Optional[ str ]
	max_output_tokens: Optional[ int ]
	output_mime_type: Optional[ str ]
	response_mode: Optional[ str ]
	response_format_value: Optional[ Any ]
	generation_config: Dict[ str, Any ]
	input_content: List[ Dict[ str, Any ] ]
	tool_objects: List[ Dict[ str, Any ] ]
	interaction: Optional[ Any ]
	response: Optional[ Any ]
	output_text: str
	output_image_content: Optional[ Any ]
	grounding_metadata: Optional[ Any ]
	grounding_sources: List[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'gemini-3.1-flash-image' ) -> None:
		"""Initialize the Images wrapper.

		Purpose:
			Initializes image-model configuration, Interactions request state, output
			configuration, and response placeholders. The constructor performs local state
			assignment only and does not create a client or submit a provider request.

		Args:
			model (str): Default Gemini image model.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.model = model
		self.api_version = 'v1beta'
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.client = None
		self.number = 1
		self.instructions = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.candidate_count = None
		self.max_tokens = None
		self.max_output_tokens = None
		self.aspect_ratio = None
		self.size = None
		self.resolution = None
		self.media_resolution = None
		self.output_mime_type = None
		self.response_mode = None
		self.response_modalities = [ ]
		self.tools = [ ]
		self.tool_choice = None
		self.input_content = [ ]
		self.response_format_value = None
		self.generation_config = { }
		self.tool_objects = [ ]
		self.interaction = None
		self.response = None
		self.content_response = None
		self.image_response = None
		self.output_text = ''
		self.output_image_content = None
		self.grounding_metadata = None
		self.grounding_sources = [ ]
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported Gemini image models.

		Purpose:
			Provides image-generation and image-editing model identifiers exposed by the Jeni
			Images mode.

		Returns:
			List[str]: Supported Gemini image-model identifiers.
		"""
		return [ 'gemini-3.1-flash-lite-image', 'gemini-3.1-flash-image', 'gemini-3-pro-image', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Return compatibility include options.

		Purpose:
			Preserves the option contract consumed by existing Jeni controls.

		Returns:
			List[str]: Existing include-option values.
		"""
		return [ 'file_search_call.results', 'message.input_image.image_url',
			'message.output_text.logprobs', 'reasoning.encrypted_content', ]
	
	@property
	def aspect_options( self ) -> List[ str ]:
		"""Return supported output-image aspect ratios.

		Purpose:
			Provides aspect-ratio values accepted by Gemini image response formats.

		Returns:
			List[str]: Supported aspect ratios.
		"""
		return [ '1:1', '1:4', '1:8', '2:3', '3:2', '3:4', '4:1', '4:3', '4:5', '5:4', '8:1',
			'9:16', '16:9', '21:9', ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Return supported media-resolution values.

		Purpose:
			Provides media-resolution values retained by the Jeni image controls.

		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Return supported image response modes.

		Purpose:
			Provides text-only, image-only, and combined text-and-image output selections.

		Returns:
			List[str]: Supported response-mode values.
		"""
		return [ 'text', 'image', 'text_and_image', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Return supported thinking levels.

		Purpose:
			Provides thinking-level values accepted by supported image models.

		Returns:
			List[str]: Supported thinking-level values.
		"""
		return [ 'unspecified', 'minimal', 'low', 'medium', 'high', ]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Return supported output-image sizes.

		Purpose:
			Provides output-image sizes accepted through the Interactions response format.

		Returns:
			List[str]: Supported image-size values.
		"""
		return [ '512', '1K', '2K', '4K', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Return supported image grounding tools.

		Purpose:
			Provides Google Web Search and Google Image Search selections used by the Jeni
			Images mode.

		Returns:
			List[str]: Supported image tool identifiers.
		"""
		return [ 'google_search', 'image_search', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Return supported tool-choice values.

		Purpose:
			Preserves the existing Jeni tool-choice option contract.

		Returns:
			List[str]: Supported tool-choice values.
		"""
		return [ 'auto', 'any', 'none', 'validated', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Return supported text-output MIME types.

		Purpose:
			Provides text MIME types retained by the Jeni image controls.

		Returns:
			List[str]: Supported text-output MIME types.
		"""
		return [ 'text/plain', 'application/json', 'text/x.enum', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Return supported generated-image MIME types.

		Purpose:
			Provides image MIME types accepted by the Interactions image response format.

		Returns:
			List[str]: Supported generated-image MIME types.
		"""
		return [ 'image/jpeg', 'image/png', 'image/webp', ]
	
	@property
	def resolution_options( self ) -> List[ str ]:
		"""Return supported output-image resolution options.

		Purpose:
			Provides the output-size values exposed by the Jeni Images mode.

		Returns:
			List[str]: Supported output-image sizes.
		"""
		return list( self.size_options )
	
	def supports_image_size( self, model: str ) -> bool:
		"""Return whether the selected model supports explicit image size.

		Purpose:
			Centralizes output-image-size feature gating for the Jeni image controls.

		Args:
			model (str): Gemini image-model identifier.

		Returns:
			bool: True when the model supports explicit image size.

		Raises:
			Error: Raised when validation or model comparison fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.model_name = self.model.strip( ).lower( )
			self.image_size_models = { 'gemini-3.1-flash-lite-image', 'gemini-3.1-flash-image',
				'gemini-3-pro-image', }
			return self.model_name in self.image_size_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_size( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_search_grounding( self, model: str ) -> bool:
		"""Return whether the model supports Google Search grounding.

		Purpose:
			Centralizes Google Search grounding feature gating for image generation.

		Args:
			model (str): Gemini image-model identifier.

		Returns:
			bool: True when the model supports Google Search grounding.

		Raises:
			Error: Raised when validation or model comparison fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.model_name = self.model.strip( ).lower( )
			self.search_grounding_models = { 'gemini-3.1-flash-image', 'gemini-3-pro-image', }
			return self.model_name in self.search_grounding_models
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_search_grounding( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def supports_image_search( self, model: str ) -> bool:
		"""Return whether the model supports Google Image Search grounding.

		Purpose:
			Restricts Google Image Search grounding to the model documented for that feature.

		Args:
			model (str): Gemini image-model identifier.

		Returns:
			bool: True when Google Image Search is supported.

		Raises:
			Error: Raised when validation or model comparison fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.model_name = self.model.strip( ).lower( )
			return self.model_name == 'gemini-3.1-flash-image'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'supports_image_search( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def normalize_response_modalities( self, response_modalities: Optional[ str ],
		image_only: bool = False ) -> List[ str ]:
		"""Normalize the requested response modes.

		Purpose:
			Converts the Jeni response-mode value into text and image modality identifiers used
			to build the Interactions response format.

		Args:
			response_modalities (Optional[str]): Jeni response-mode value.
			image_only (bool): Whether image output must be included.

		Returns:
			List[str]: Normalized response modes.

		Raises:
			Error: Raised when response-mode normalization fails.
		"""
		try:
			self.response_mode = response_modalities
			self.image_only = image_only
			self.mode_name = str( self.response_mode or '' ).strip( ).lower( )
			if self.mode_name == 'text_and_image':
				return [ 'text', 'image', ]
			
			if self.mode_name == 'text':
				return [ 'text', ]
			
			if self.mode_name == 'image':
				return [ 'image', ]
			
			if self.image_only:
				return [ 'image', ]
			
			return [ 'text', ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('normalize_response_modalities( self, response_modalities: '
			                    'Optional[str], image_only: bool ) -> List[str]')
			Logger( ).write( exception )
			raise exception
	
	def get_image_mime_type( self, path: str ) -> str:
		"""Return the MIME type for a local image.

		Purpose:
			Uses the local file suffix to produce the image MIME type required by an
			Interactions image-content block.

		Args:
			path (str): Local image path.

		Returns:
			str: Image MIME type.

		Raises:
			Error: Raised when validation or MIME-type resolution fails.
			ValueError: Raised when ``path`` is missing.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.file_suffix = Path( self.file_path ).suffix.lower( )
			if self.file_suffix in ('.jpg', '.jpeg'):
				return 'image/jpeg'
			
			if self.file_suffix == '.webp':
				return 'image/webp'
			
			if self.file_suffix == '.gif':
				return 'image/gif'
			
			return 'image/png'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'get_image_mime_type( self, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def build_image_input( self, prompt: str, path: Optional[ str ] = None ) -> List[
		Dict[ str, Any ] ]:
		"""Build Interactions text and image input.

		Purpose:
			Creates a text content block and, when supplied, a base64-encoded local image
			content block for image analysis or editing.

		Args:
			prompt (str): Image-generation, analysis, or editing instruction.
			path (Optional[str]): Optional local image path.

		Returns:
			List[Dict[str, Any]]: Interactions-compatible input content.

		Raises:
			Error: Raised when validation, file reading, or input construction fails.
			ValueError: Raised when ``prompt`` is missing.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.file_path = path
			self.input_content = [ { 'type': 'text', 'text': self.prompt.strip( ), }, ]
			if self.file_path:
				self.image_bytes = Path( self.file_path ).read_bytes( )
				throw_if( 'image_bytes', self.image_bytes )
				self.image_data = base64.b64encode( self.image_bytes ).decode( 'utf-8' )
				self.image_mime_type = self.get_image_mime_type( self.file_path )
				self.input_content.append( { 'type': 'image', 'data': self.image_data,
					'mime_type': self.image_mime_type, } )
			
			return self.input_content
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('build_image_input( self, prompt: str, path: Optional[str] ) -> '
			                    'List[Dict[str, Any]]')
			Logger( ).write( exception )
			raise exception
	
	def build_generation_config( self, temperature: Optional[ float ] = None,
		top_p: Optional[ float ] = None, max_tokens: Optional[ int ] = None ) -> Dict[ str, Any ]:
		"""Build the Interactions image generation configuration.

		Purpose:
			Converts supported Jeni inference values into the Interactions generation
			configuration.

		Args:
			temperature (Optional[float]): Sampling temperature.
			top_p (Optional[float]): Top-p sampling value.
			max_tokens (Optional[int]): Maximum output-token count.

		Returns:
			Dict[str, Any]: Interactions generation configuration.

		Raises:
			Error: Raised when generation configuration construction fails.
		"""
		try:
			self.temperature = temperature
			self.top_p = top_p
			self.max_output_tokens = max_tokens
			self.generation_config = { }
			if self.temperature is not None:
				self.generation_config[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.generation_config[ 'top_p' ] = self.top_p
			
			if self.max_output_tokens is not None and self.max_output_tokens > 0:
				self.generation_config[ 'max_output_tokens' ] = (self.max_output_tokens)
			
			return self.generation_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('build_generation_config( self, temperature: Optional[float], '
			                    'top_p: Optional[float], max_tokens: Optional[int] ) -> Dict[str, '
			                    'Any]')
			Logger( ).write( exception )
			raise exception
	
	def build_response_format( self, response_modalities: Optional[ str ], image_only: bool =
	False,
		aspect: Optional[ str ] = None, resolution: Optional[ str ] = None,
		output_mime_type: Optional[ str ] = None ) -> Any:
		"""Build the Interactions image response format.

		Purpose:
			Constructs text, image, or combined response formats and applies supported image
			MIME type, aspect ratio, and output-size settings to the image format.

		Args:
			response_modalities (Optional[str]): Jeni response-mode value.
			image_only (bool): Whether image output must be included.
			aspect (Optional[str]): Requested output-image aspect ratio.
			resolution (Optional[str]): Requested output-image size.
			output_mime_type (Optional[str]): Requested generated-image MIME type.

		Returns:
			Any: Interactions response-format object or list.

		Raises:
			Error: Raised when response-format construction fails.
		"""
		try:
			self.response_mode = response_modalities
			self.image_only = image_only
			self.aspect_ratio = aspect
			self.size = resolution
			self.output_mime_type = output_mime_type
			self.response_modalities = self.normalize_response_modalities(
				response_modalities=self.response_mode, image_only=self.image_only )
			self.response_formats: List[ Dict[ str, Any ] ] = [ ]
			for modality in self.response_modalities:
				if modality == 'text':
					self.response_formats.append( { 'type': 'text', } )
				
				elif modality == 'image':
					self.image_format: Dict[ str, Any ] = { 'type': 'image', }
					if self.output_mime_type:
						self.image_format[ 'mime_type' ] = (self.output_mime_type)
					
					if self.aspect_ratio:
						self.image_format[ 'aspect_ratio' ] = (self.aspect_ratio)
					
					if (self.size and self.supports_image_size( self.model )):
						self.image_format[ 'image_size' ] = self.size
					
					self.response_formats.append( self.image_format )
			
			if len( self.response_formats ) == 1:
				self.response_format_value = self.response_formats[ 0 ]
			else:
				self.response_format_value = self.response_formats
			
			return self.response_format_value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('build_response_format( self, response_modalities: Optional[str], '
			                    'image_only: bool, aspect: Optional[str], resolution: Optional['
			                    'str], output_mime_type: Optional[str] ) -> Any')
			Logger( ).write( exception )
			raise exception
	
	def build_grounding_tools( self, grounded: bool = False, image_search: bool = False ) -> List[
		Dict[ str, Any ] ]:
		"""Build image grounding tool declarations.

		Purpose:
			Creates the Interactions Google Search tool declaration for web grounding and
			optionally enables Google Image Search when supported by the selected model.

		Args:
			grounded (bool): Whether Google Web Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			List[Dict[str, Any]]: Interactions tool declarations.

		Raises:
			Error: Raised when grounding-tool construction fails.
		"""
		try:
			self.grounded = grounded
			self.image_search = image_search
			self.tool_objects = [ ]
			if not self.grounded and not self.image_search:
				return self.tool_objects
			
			if not self.supports_search_grounding( self.model ):
				return self.tool_objects
			
			self.search_types: List[ str ] = [ ]
			if self.grounded:
				self.search_types.append( 'web_search' )
			
			if (self.image_search and self.supports_image_search( self.model )):
				self.search_types.append( 'image_search' )
			
			self.search_tool: Dict[ str, Any ] = { 'type': 'google_search', }
			
			if self.search_types:
				self.search_tool[ 'search_types' ] = self.search_types
			
			self.tool_objects.append( self.search_tool )
			return self.tool_objects
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('build_grounding_tools( self, grounded: bool, image_search: bool ) '
			                    '-> List[Dict[str, Any]]')
			Logger( ).write( exception )
			raise exception
	
	def extract_image( self, interaction: Any ) -> Optional[ PIL.Image.Image ]:
		"""Extract a generated image from an Interaction.

		Purpose:
			Reads the SDK ``output_image`` convenience property, decodes its base64 content,
			and returns an independent Pillow image object.

		Args:
			interaction (Any): Completed Gemini Interaction.

		Returns:
			Optional[PIL.Image.Image]: Generated image, or None when no image was returned.

		Raises:
			Error: Raised when validation or image extraction fails.
			ValueError: Raised when ``interaction`` is missing.
		"""
		try:
			throw_if( 'interaction', interaction )
			self.interaction = interaction
			self.output_image_content = getattr( self.interaction, 'output_image', None )
			if self.output_image_content is None:
				return None
			
			self.output_image_data = getattr( self.output_image_content, 'data', None )
			if not self.output_image_data:
				return None
			
			self.decoded_image = base64.b64decode( self.output_image_data )
			with PIL.Image.open( io.BytesIO( self.decoded_image ) ) as source:
				return source.copy( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'extract_image( self, interaction: Any ) -> Optional[ Image ]'
			Logger( ).write( exception )
			raise exception
	
	def extract_text( self, interaction: Any ) -> Optional[ str ]:
		"""Extract generated text from an Interaction.

		Purpose:
			Reads the Interactions SDK ``output_text`` convenience property and returns the
			normalized text expected by the Jeni image-analysis workflow.

		Args:
			interaction (Any): Completed Gemini Interaction.

		Returns:
			Optional[str]: Generated text, or None when no text was returned.

		Raises:
			Error: Raised when validation or text extraction fails.
			ValueError: Raised when ``interaction`` is missing.
		"""
		try:
			throw_if( 'interaction', interaction )
			self.interaction = interaction
			self.output_text = str( getattr( self.interaction, 'output_text', '' ) or '' ).strip( )
			return self.output_text or None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('extract_text( self, interaction: Any ) -> Optional[ str ]')
			Logger( ).write( exception )
			raise exception
	
	def capture_metadata( self ) -> None:
		"""Capture grounding information from the latest Interaction.

		Purpose:
			Retains the latest Interaction steps and Google Search result metadata for
			application display and diagnostics.

		Returns:
			None: This method updates response state through side effects.

		Raises:
			Error: Raised when grounding metadata cannot be captured.
		"""
		try:
			self.grounding_metadata = None
			self.grounding_sources = [ ]
			
			if self.interaction is None:
				return
			
			self.steps = getattr( self.interaction, 'steps', None ) or [ ]
			for step in self.steps:
				self.step_type = str( getattr( step, 'type', '' ) or '' ).strip( )
				if self.step_type != 'google_search_result':
					continue
				
				self.result = getattr( step, 'result', None )
				self.grounding_sources.append( { 'type': self.step_type, 'result': self.result, } )
			
			if self.grounding_sources:
				self.grounding_metadata = self.grounding_sources
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = 'capture_metadata( self ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def execute_interaction( self, prompt: str, model: str, path: Optional[ str ] = None,
		aspect: Optional[ str ] = None, number: Optional[ int ] = None,
		temperature: Optional[ float ] = None, top_p: Optional[ float ] = None,
		frequency: Optional[ float ] = None, presence: Optional[ float ] = None,
		max_tokens: Optional[ int ] = None, resolution: Optional[ str ] = None,
		instruct: Optional[ str ] = None, output_mime_type: Optional[ str ] = None,
		response_modalities: Optional[ str ] = None, image_only: bool = False,
		grounded: bool = False, image_search: bool = False ) -> Any:
		"""Execute an image Interaction.

		Purpose:
			Validates required inputs, creates the Gemini client, builds multimodal input,
			generation settings, output formats, and grounding tools, submits the request, and
			captures the response state shared by generation, analysis, and editing workflows.

		Args:
			prompt (str): Image-generation, analysis, or editing instruction.
			model (str): Gemini image-model identifier.
			path (Optional[str]): Optional local image path.
			aspect (Optional[str]): Requested output-image aspect ratio.
			number (Optional[int]): Requested result count retained for compatibility.
			temperature (Optional[float]): Sampling temperature.
			top_p (Optional[float]): Top-p sampling value.
			frequency (Optional[float]): Frequency penalty retained for compatibility.
			presence (Optional[float]): Presence penalty retained for compatibility.
			max_tokens (Optional[int]): Maximum output-token count.
			resolution (Optional[str]): Requested generated-image size.
			instruct (Optional[str]): System instruction text.
			output_mime_type (Optional[str]): Requested generated-image MIME type.
			response_modalities (Optional[str]): Requested response-mode value.
			image_only (bool): Whether image output must be included.
			grounded (bool): Whether Google Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			Any: Completed Gemini Interaction.

		Raises:
			Error: Raised when validation, request construction, or provider execution fails.
			ValueError: Raised when ``prompt``, ``model``, or the API key is missing.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = prompt
			self.model = model
			self.file_path = path
			self.aspect_ratio = aspect
			self.number = number
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.size = resolution
			self.instructions = instruct
			self.output_mime_type = output_mime_type
			self.response_mode = response_modalities
			self.image_only = image_only
			self.grounded = grounded
			self.image_search = image_search
			self.api_key = self.gemini_api_key or self.google_api_key
			self.client = genai.Client( api_key=self.api_key, http_options=self.http_options )
			self.input_content = self.build_image_input( prompt=self.prompt, path=self.file_path )
			self.generation_config = self.build_generation_config( temperature=self.temperature,
				top_p=self.top_p, max_tokens=self.max_output_tokens )
			
			self.response_format_value = self.build_response_format(
				response_modalities=self.response_mode, image_only=self.image_only,
				aspect=self.aspect_ratio, resolution=self.size,
				output_mime_type=self.output_mime_type )
			
			self.tool_objects = self.build_grounding_tools( grounded=self.grounded,
				image_search=self.image_search )
			
			self.request: Dict[ str, Any ] = { 'model': self.model, 'input': self.input_content,
				'response_format': self.response_format_value, 'store': False, }
			
			if self.instructions:
				self.request[ 'system_instruction' ] = self.instructions
			
			if self.generation_config:
				self.request[ 'generation_config' ] = (self.generation_config)
			
			if self.tool_objects:
				self.request[ 'tools' ] = self.tool_objects
			
			self.interaction = self.client.interactions.create( **self.request )
			self.response = self.interaction
			self.content_response = self.interaction
			self.image_response = self.interaction
			self.capture_metadata( )
			return self.interaction
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('execute_interaction( self, prompt: str, model: str, '
			                    'path: Optional[str], aspect: Optional[str], number: Optional['
			                    'int], temperature: Optional[float], top_p: Optional[float], '
			                    'frequency: Optional[float], presence: Optional[float], '
			                    'max_tokens: Optional[int], resolution: Optional[str], instruct: '
			                    'Optional[str], output_mime_type: Optional[str], '
			                    'response_modalities: Optional[str], image_only: bool, grounded: '
			                    'bool, image_search: bool ) -> Any')
			Logger( ).write( exception )
			raise exception
	
	def get_first_image( self ) -> Optional[ PIL.Image.Image ]:
		"""Return the image from the most recent Interaction.

		Purpose:
			Preserves the existing Jeni helper contract by extracting the generated image from
			the latest Interaction.

		Returns:
			Optional[PIL.Image.Image]: Generated image, or None when unavailable.

		Raises:
			Error: Raised when image extraction fails.
		"""
		try:
			if self.interaction is None:
				return None
			
			return self.extract_image( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('get_first_image( self ) -> Optional[ PIL.Image.Image ]')
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> Optional[ str ]:
		"""Return text from the most recent Interaction.

		Purpose:
			Preserves the existing Jeni helper contract by extracting generated text from the
			latest Interaction.

		Returns:
			Optional[str]: Generated text, or None when unavailable.

		Raises:
			Error: Raised when text extraction fails.
		"""
		try:
			if self.interaction is None:
				return None
			
			return self.extract_text( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('get_output_text( self ) -> Optional[ str ]')
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, model: str = 'gemini-3.1-flash-image', aspect: str = None,
		number: int = None, temperature: float = None, top_p: float = None, frequency: float =
		None,
		presence: float = None, max_tokens: int = None, resolution: str = None,
		instruct: str = None, output_mime_type: str = None, response_modalities: str = None,
		grounded: bool = False, image_search: bool = False ) -> Optional[ PIL.Image.Image ]:
		"""Generate an image through the Gemini Interactions API.

		Purpose:
			Submits a text-to-image Interaction and returns the generated Pillow image expected
			by the Jeni Image Generation workflow.

		Args:
			prompt (str): Image-generation instruction.
			model (str): Gemini image-model identifier.
			aspect (str): Requested output-image aspect ratio.
			number (int): Requested result count retained for compatibility.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for compatibility.
			presence (float): Presence penalty retained for compatibility.
			max_tokens (int): Maximum output-token count.
			resolution (str): Requested generated-image size.
			instruct (str): System instruction text.
			output_mime_type (str): Requested generated-image MIME type.
			response_modalities (str): Requested response-mode value.
			grounded (bool): Whether Google Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			Optional[PIL.Image.Image]: Generated image, or None when no image is returned.

		Raises:
			Error: Raised when request execution or image extraction fails.
			ValueError: Raised when a required argument is missing.
		"""
		try:
			self.interaction = self.execute_interaction( prompt=prompt, model=model, path=None,
				aspect=aspect, number=number, temperature=temperature, top_p=top_p,
				frequency=frequency, presence=presence, max_tokens=max_tokens,
				resolution=resolution, instruct=instruct, output_mime_type=output_mime_type,
				response_modalities=response_modalities, image_only=True, grounded=grounded,
				image_search=image_search )
			
			return self.extract_image( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('generate( self, prompt: str, model: str, aspect: str, number: int, '
			                    'temperature: float, top_p: float, frequency: float, presence: '
			                    'float, max_tokens: int, resolution: str, instruct: str, '
			                    'output_mime_type: str, response_modalities: str, grounded: bool, '
			                    'image_search: bool ) -> Optional[PIL.Image.Image]')
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str = 'gemini-3.1-flash-image',
		aspect: str = None, number: int = None, temperature: float = None, top_p: float = None,
		frequency: float = None, presence: float = None, max_tokens: int = None,
		resolution: str = None, instruct: str = None, output_mime_type: str = None,
		response_modalities: str = None, grounded: bool = False, image_search: bool = False ) -> \
	Optional[ str ]:
		"""Analyze an image through the Gemini Interactions API.

		Purpose:
			Submits text and a local image as multimodal Interaction input and returns the
			generated text expected by the Jeni Image Analysis workflow.

		Args:
			prompt (str): Image-analysis instruction.
			path (str): Local image path.
			model (str): Gemini image-model identifier.
			aspect (str): Aspect-ratio value retained for compatibility.
			number (int): Requested result count retained for compatibility.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for compatibility.
			presence (float): Presence penalty retained for compatibility.
			max_tokens (int): Maximum output-token count.
			resolution (str): Media-resolution value retained for compatibility.
			instruct (str): System instruction text.
			output_mime_type (str): Output MIME type retained for compatibility.
			response_modalities (str): Requested response-mode value.
			grounded (bool): Whether Google Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			Optional[str]: Generated image analysis, or None when no text is returned.

		Raises:
			Error: Raised when request execution or text extraction fails.
			ValueError: Raised when a required argument is missing.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.interaction = self.execute_interaction( prompt=prompt, model=model,
				path=self.file_path, aspect=aspect, number=number, temperature=temperature,
				top_p=top_p, frequency=frequency, presence=presence, max_tokens=max_tokens,
				resolution=resolution, instruct=instruct, output_mime_type=output_mime_type,
				response_modalities=response_modalities or 'text', image_only=False,
				grounded=grounded, image_search=image_search )
			
			return self.extract_text( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('analyze( self, prompt: str, path: str, model: str, aspect: str, '
			                    'number: int, temperature: float, top_p: float, frequency: float, '
			                    'presence: float, max_tokens: int, resolution: str, instruct: str, '
			                    'output_mime_type: str, response_modalities: str, grounded: bool, '
			                    'image_search: bool ) -> Optional[str]')
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gemini-3.1-flash-image',
		aspect: str = None, number: int = None, temperature: float = None, top_p: float = None,
		frequency: float = None, presence: float = None, max_tokens: int = None,
		resolution: str = None, instruct: str = None, output_mime_type: str = None,
		response_modalities: str = None, grounded: bool = False, image_search: bool = False ) -> \
	Optional[ PIL.Image.Image ]:
		"""Edit an image through the Gemini Interactions API.

		Purpose:
			Submits an editing instruction and local image as multimodal Interaction input and
			returns the generated Pillow image expected by the Jeni Image Editing workflow.

		Args:
			prompt (str): Image-editing instruction.
			path (str): Local source-image path.
			model (str): Gemini image-model identifier.
			aspect (str): Requested output-image aspect ratio.
			number (int): Requested result count retained for compatibility.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for compatibility.
			presence (float): Presence penalty retained for compatibility.
			max_tokens (int): Maximum output-token count.
			resolution (str): Requested generated-image size.
			instruct (str): System instruction text.
			output_mime_type (str): Requested generated-image MIME type.
			response_modalities (str): Requested response-mode value.
			grounded (bool): Whether Google Search grounding is enabled.
			image_search (bool): Whether Google Image Search grounding is enabled.

		Returns:
			Optional[PIL.Image.Image]: Edited image, or None when no image is returned.

		Raises:
			Error: Raised when request execution or image extraction fails.
			ValueError: Raised when a required argument is missing.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.interaction = self.execute_interaction( prompt=prompt, model=model,
				path=self.file_path, aspect=aspect, number=number, temperature=temperature,
				top_p=top_p, frequency=frequency, presence=presence, max_tokens=max_tokens,
				resolution=resolution, instruct=instruct, output_mime_type=output_mime_type,
				response_modalities=response_modalities or 'image', image_only=True,
				grounded=grounded, image_search=image_search )
			
			return self.extract_image( self.interaction )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Images'
			exception.method = ('edit( self, prompt: str, path: str, model: str, aspect: str, '
			                    'number: int, temperature: float, top_p: float, frequency: float, '
			                    'presence: float, max_tokens: int, resolution: str, instruct: str, '
			                    'output_mime_type: str, response_modalities: str, grounded: bool, '
			                    'image_search: bool ) -> Optional[PIL.Image.Image]')
			Logger( ).write( exception )
			raise exception

class Embeddings( Gemini ):
	"""Gemini embedding wrapper.

	Purpose:
		Creates numerical vector representations of text through the Gemini embeddings API.
		The class validates and normalizes text input, builds provider-supported embedding
		configuration, executes the specialized ``models.embed_content()`` operation, and
		extracts vectors from the returned embedding response.

	Attributes:
		client (Optional[genai.Client]): Active Google Gen AI client.
		response (Optional[EmbedContentResponse]): Most recent embedding response.
		embedding (Optional[List[float] | List[List[float]]]): Extracted embedding result.
		embeddings (Optional[List[List[float]]]): Extracted embedding vectors.
		encoding_format (str): Application-facing embedding encoding selection.
		dimensions (Optional[int]): Requested output dimensionality.
		task_type (Optional[str]): Embedding task type.
		title (Optional[str]): Retrieval-document title.
		embedding_config (Optional[EmbedContentConfig]): Embedding request configuration.
		contents (Optional[str | List[str]]): Normalized provider input.
		input_text (Optional[str | List[str]]): Original normalized text input.
	"""
	
	client: Optional[ genai.Client ]
	response: Optional[ EmbedContentResponse ]
	embedding: Optional[ List[ float ] | List[ List[ float ] ] ]
	embeddings: Optional[ List[ List[ float ] ] ]
	encoding_format: str
	dimensions: Optional[ int ]
	task_type: Optional[ str ]
	title: Optional[ str ]
	embedding_config: Optional[ EmbedContentConfig ]
	contents: Optional[ str | List[ str ] ]
	input_text: Optional[ str | List[ str ] ]
	
	def __init__( self, model: str = 'gemini-embedding-2' ) -> None:
		"""Initialize the embeddings wrapper.

		Purpose:
			Initializes the embedding model, request configuration, input state, response state,
			and extracted-vector placeholders. The constructor performs local assignment only.

		Args:
			model (str): Default Gemini embedding model.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.model = model
		self.client = None
		self.response = None
		self.embedding = None
		self.embeddings = None
		self.encoding_format = 'float'
		self.dimensions = None
		self.task_type = None
		self.title = None
		self.embedding_config = None
		self.contents = None
		self.input_text = None
		self.api_key = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported Gemini embedding models.

		Purpose:
			Provides the active embedding model identifiers exposed by the Jeni Embeddings mode.

		Returns:
			List[str]: Supported Gemini embedding model identifiers.
		"""
		return [ 'gemini-embedding-2', 'gemini-embedding-001' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		"""Return supported embedding encodings.

		Purpose:
			Provides the native numerical encoding returned by the Gemini embeddings API.

		Returns:
			List[str]: Supported embedding encoding values.
		"""
		return [ 'float' ]
	
	@property
	def task_options( self ) -> List[ str ]:
		"""Return supported embedding task types.

		Purpose:
			Provides the embedding task types accepted for text embedding requests.

		Returns:
			List[str]: Supported embedding task types.
		"""
		return [ '', 'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY',
			'CLASSIFICATION', 'CLUSTERING', 'QUESTION_ANSWERING', 'FACT_VERIFICATION',
			'CODE_RETRIEVAL_QUERY', ]
	
	def normalize_dimensions( self, dimensions: int = 0 ) -> Optional[ int ]:
		"""Normalize the requested output dimensionality.

		Purpose:
			Converts a zero-valued UI selection into an omitted provider setting and validates
			positive dimensionality values against the range exposed by the application.

		Args:
			dimensions (int): Requested output dimensionality.

		Returns:
			Optional[int]: Positive output dimensionality, or None when omitted.

		Raises:
			Error: Raised when dimensionality validation fails.
			ValueError: Raised when dimensionality exceeds the supported application range.
		"""
		try:
			self.dimensions = dimensions
			if self.dimensions <= 0:
				self.dimensions = None
				return None
			
			if self.dimensions > 2048:
				raise ValueError( 'Embedding dimensions must be between 1 and 2048.' )
			
			return self.dimensions
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('normalize_dimensions( self, dimensions: int ) -> Optional[ int ]')
			Logger( ).write( exception )
			raise exception
	
	def normalize_contents( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""Normalize embedding input.

		Purpose:
			Removes blank values while preserving whether the caller supplied one string or a
			list of independent strings.

		Args:
			text (str | List[str]): Text or text values to embed.

		Returns:
			str | List[str]: Normalized embedding input.

		Raises:
			Error: Raised when input validation or normalization fails.
			ValueError: Raised when no usable text remains.
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text
			if isinstance( self.input_text, list ):
				self.contents = [ str( item ).strip( ) for item in self.input_text if
					item is not None and str( item ).strip( ) ]
				throw_if( 'text', self.contents )
				return self.contents
			
			self.contents = str( self.input_text ).strip( )
			throw_if( 'text', self.contents )
			return self.contents
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('normalize_contents( self, text: str | List[str] ) -> str | List['
			                    'str]')
			Logger( ).write( exception )
			raise exception
	
	def build_embedding_config( self, model: str, dimensions: int = 0, task_type: str = '',
		title: str = '' ) -> EmbedContentConfig:
		"""Build the embedding request configuration.

		Purpose:
			Constructs the provider configuration from output dimensionality, task type, and
			retrieval-document title settings supported by the selected embedding model.

		Args:
			model (str): Gemini embedding model identifier.
			dimensions (int): Requested output dimensionality.
			task_type (str): Optional embedding task type.
			title (str): Optional retrieval-document title.

		Returns:
			EmbedContentConfig: Provider embedding configuration.

		Raises:
			Error: Raised when validation or configuration construction fails.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.dimensions = self.normalize_dimensions( dimensions )
			self.task_type = str( task_type or '' ).strip( ).upper( )
			self.title = str( title or '' ).strip( )
			self.config_values: Dict[ str, Any ] = { }
			if self.dimensions is not None:
				self.config_values[ 'output_dimensionality' ] = self.dimensions
			
			if self.task_type:
				self.config_values[ 'task_type' ] = self.task_type
			
			if self.title and self.task_type == 'RETRIEVAL_DOCUMENT':
				self.config_values[ 'title' ] = self.title
			
			self.embedding_config = EmbedContentConfig( **self.config_values )
			return self.embedding_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('build_embedding_config( self, model: str, dimensions: int, '
			                    'task_type: str, title: str ) -> EmbedContentConfig')
			Logger( ).write( exception )
			raise exception
	
	def extract_embeddings( self ) -> Optional[ List[ float ] | List[ List[ float ] ] ]:
		"""Extract vectors from the embedding response.

		Purpose:
			Converts SDK embedding objects into ordinary lists of floating-point values and
			preserves the single-vector return shape for a single string input.

		Returns:
			Optional[List[float] | List[List[float]]]: Extracted embedding vector or vectors.

		Raises:
			Error: Raised when response extraction fails.
		"""
		try:
			if self.response is None:
				return None
			
			self.response_embeddings = getattr( self.response, 'embeddings', None )
			if not self.response_embeddings:
				return None
			
			self.embeddings = [ ]
			for item in self.response_embeddings:
				if item is None:
					continue
				
				self.values = getattr( item, 'values', None )
				if self.values is not None:
					self.embeddings.append( [ float( value ) for value in self.values ] )
			
			if not self.embeddings:
				return None
			
			if len( self.embeddings ) == 1 and isinstance( self.input_text, str ):
				self.embedding = self.embeddings[ 0 ]
			else:
				self.embedding = self.embeddings
			
			return self.embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = 'extract_embeddings( self ) -> Optional[ List[ float ] ]'
			Logger( ).write( exception )
			raise exception
	
	def create( self, text: str | List[ str ], model: str = 'gemini-embedding-2',
		dimensions: int = 0, task_type: str = '', title: str = '',
		encoding_format: str = 'float' ) -> Optional[ List[ float ] | List[ List[ float ] ] ]:
		"""Create text embeddings.

		Purpose:
			Validates and normalizes text input, constructs the provider configuration, executes
			the specialized Gemini embeddings operation, and returns the extracted vector data.

		Args:
			text (str | List[str]): Text or independent text values to embed.
			model (str): Gemini embedding model identifier.
			dimensions (int): Requested output dimensionality.
			task_type (str): Optional embedding task type.
			title (str): Optional retrieval-document title.
			encoding_format (str): Application-facing encoding selection.

		Returns:
			Optional[List[float] | List[List[float]]]: Generated embedding vector or vectors.

		Raises:
			Error: Raised when validation, provider execution, or extraction fails.
			ValueError: Raised when required input, model, API key, or encoding is invalid.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'model', model )
			self.input_text = text
			self.model = model
			self.dimensions = dimensions
			self.task_type = task_type
			self.title = title
			self.encoding_format = str( encoding_format or 'float' ).strip( ).lower( )
			if self.encoding_format != 'float':
				raise ValueError( 'The Gemini embeddings API returns float vectors only.' )
			
			self.contents = self.normalize_contents( self.input_text )
			self.embedding_config = self.build_embedding_config( model=self.model,
				dimensions=self.dimensions, task_type=self.task_type, title=self.title )
			
			self.api_key = self.gemini_api_key or self.google_api_key
			throw_if( 'api_key', self.api_key )
			
			self.client = genai.Client( api_key=self.api_key )
			self.response = self.client.models.embed_content( model=self.model,
				contents=self.contents, config=self.embedding_config )
			
			return self.extract_embeddings( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Embeddings'
			exception.method = ('create( self, text: str | List[str], model: str, dimensions: int, '
			                    'task_type: str, title: str, encoding_format: str ) -> Optional['
			                    'List[float] | List[List[float]]]')
			Logger( ).write( exception )
			raise exception

class TTS( Gemini ):
	"""Gemini text-to-speech wrapper.

	Purpose:
		Converts text into single-speaker audio through the Gemini Interactions API. The class
		builds controllable speech prompts, applies the selected prebuilt voice, retrieves PCM
		audio from the completed Interaction, wraps the PCM data in a WAV container, and
		optionally writes the generated audio to disk.

	Attributes:
		client (Optional[genai.Client]): Active Gemini SDK client.
		http_options (HttpOptions): Gemini client HTTP configuration.
		speed (Optional[float]): Requested speech-rate control.
		voice (Optional[str]): Selected prebuilt Gemini voice.
		response (Optional[Any]): Most recent Gemini Interaction.
		interaction (Optional[Any]): Most recent Gemini Interaction.
		audio_path (Optional[str]): Optional output WAV path.
		response_format (Optional[str]): Application-facing audio format.
		input_text (Optional[str]): Complete text and performance instruction sent to Gemini.
		audio_bytes (Optional[bytes]): Generated WAV audio.
		pcm_bytes (Optional[bytes]): Raw PCM audio returned by Gemini.
		generation_config (Dict[str, Any]): Interactions speech-generation configuration.
	"""
	
	client: Optional[ genai.Client ]
	http_options: HttpOptions
	speed: Optional[ float ]
	voice: Optional[ str ]
	response: Optional[ Any ]
	interaction: Optional[ Any ]
	audio_path: Optional[ str ]
	response_format: Optional[ str ]
	input_text: Optional[ str ]
	audio_bytes: Optional[ bytes ]
	pcm_bytes: Optional[ bytes ]
	generation_config: Dict[ str, Any ]
	
	def __init__( self, model: str = 'gemini-3.1-flash-tts-preview' ) -> None:
		"""Initialize the TTS wrapper.

		Purpose:
			Initializes model configuration, speech settings, request state, and audio-output
			placeholders. The constructor performs local state assignment only and does not
			create a provider client or submit a request.

		Args:
			model (str): Default Gemini text-to-speech model.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.model = model
		self.api_version = 'v1beta'
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.client = None
		self.number = None
		self.temperature = None
		self.top_p = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.voice = None
		self.speed = None
		self.response = None
		self.interaction = None
		self.response_format = None
		self.audio_path = None
		self.input_text = None
		self.audio_bytes = None
		self.pcm_bytes = None
		self.generation_config = { }
		self.response_modalities = [ 'audio' ]
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported Gemini text-to-speech models.

		Purpose:
			Provides the Gemini TTS model identifiers exposed by the Jeni Audio mode.

		Returns:
			List[str]: Supported Gemini text-to-speech model identifiers.
		"""
		return [ 'gemini-3.1-flash-tts-preview', 'gemini-2.5-flash-preview-tts',
			'gemini-2.5-pro-preview-tts', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Return supported application audio formats.

		Purpose:
			Provides the local WAV output format supported by the Jeni Audio mode.

		Returns:
			List[str]: Supported application audio formats.
		"""
		return [ 'audio/wav', ]
	
	@property
	def voice_options( self ) -> List[ str ]:
		"""Return supported prebuilt Gemini voices.

		Purpose:
			Provides the prebuilt single-speaker voices exposed by the Jeni Audio controls.

		Returns:
			List[str]: Supported Gemini prebuilt voice names.
		"""
		return [ 'Achernar', 'Achird', 'Algenib', 'Algieba', 'Alnilam', 'Aoede', 'Autonoe',
			'Callirrhoe', 'Charon', 'Despina', 'Enceladus', 'Erinome', 'Fenrir', 'Gacrux',
			'Iapetus', 'Kore', 'Laomedeia', 'Leda', 'Orus', 'Puck', 'Pulcherrima', 'Rasalgethi',
			'Sadachbia', 'Sadaltager', 'Schedar', 'Sulafat', 'Umbriel', 'Vindemiatrix', 'Zephyr',
			'Zubenelgenubi', ]
	
	def to_wave_bytes( self, pcm_data: bytes, rate: int = 24000, channels: int = 1,
		sample_width: int = 2 ) -> bytes:
		"""Wrap PCM audio in a WAV container.

		Purpose:
			Writes raw Gemini PCM output into an in-memory WAV file using the channel count,
			sampling rate, and sample width documented for Gemini TTS output.

		Args:
			pcm_data (bytes): Raw PCM audio.
			rate (int): Audio sample rate in hertz.
			channels (int): Audio channel count.
			sample_width (int): Sample width in bytes.

		Returns:
			bytes: Complete WAV file bytes.

		Raises:
			Error: Raised when validation or WAV encoding fails.
			ValueError: Raised when ``pcm_data`` is missing.
		"""
		try:
			import io
			import wave
			
			throw_if( 'pcm_data', pcm_data )
			self.pcm_data = pcm_data
			self.sample_rate = rate
			self.channel_count = channels
			self.sample_width = sample_width
			
			with io.BytesIO( ) as buffer:
				with wave.open( buffer, 'wb' ) as wave_file:
					wave_file.setnchannels( self.channel_count )
					wave_file.setsampwidth( self.sample_width )
					wave_file.setframerate( self.sample_rate )
					wave_file.writeframes( self.pcm_data )
				
				return buffer.getvalue( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = ('to_wave_bytes( self, pcm_data: bytes, rate: int, channels: int, '
			                    'sample_width: int ) -> bytes')
			Logger( ).write( exception )
			raise exception
	
	def normalize_voice( self, voice: Optional[ str ] = None ) -> str:
		"""Normalize the selected Gemini voice.

		Purpose:
			Returns the selected prebuilt voice when supported and otherwise uses Kore as the
			stable default voice.

		Args:
			voice (Optional[str]): Candidate Gemini voice name.

		Returns:
			str: Supported Gemini voice name.

		Raises:
			Error: Raised when voice normalization fails.
		"""
		try:
			self.voice = voice
			self.voice_name = str( self.voice or '' ).strip( )
			
			if self.voice_name in set( self.voice_options ):
				return self.voice_name
			
			return 'Kore'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = ('normalize_voice( self, voice: Optional[ str ] ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def normalize_tts_prompt( self, text: str, speed: Optional[ float ] = None,
		instruct: Optional[ str ] = None ) -> str:
		"""Build the controllable TTS prompt.

		Purpose:
			Combines optional system-style performance instructions, speech-rate guidance, and
			the exact source text into the single text input accepted by Gemini TTS models.

		Args:
			text (str): Text to synthesize.
			speed (Optional[float]): Relative speech-rate control.
			instruct (Optional[str]): Optional performance instruction.

		Returns:
			str: Complete Gemini TTS prompt.

		Raises:
			Error: Raised when validation or prompt construction fails.
			ValueError: Raised when ``text`` is missing.
		"""
		try:
			throw_if( 'text', text )
			self.text = text
			self.speed = speed
			self.instructions = instruct
			self.prompt_parts: List[ str ] = [ ]
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.prompt_parts.append( str( self.instructions ).strip( ) )
			
			if self.speed is not None:
				if self.speed < 0.9:
					self.prompt_parts.append( 'Read the following text slowly and clearly.' )
				
				elif self.speed > 1.1:
					self.prompt_parts.append(
						'Read the following text at a faster, energetic pace.' )
			
			self.prompt_parts.append( str( self.text ).strip( ) )
			self.input_text = '\n\n'.join( self.prompt_parts )
			return self.input_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = ('normalize_tts_prompt( self, text: str, '
			                    'speed: Optional[ float ], instruct: Optional[ str ] ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def build_generation_config( self, voice: str, temperature: Optional[ float ] = None,
		top_p: Optional[ float ] = None, max_tokens: Optional[ int ] = None ) -> Dict[ str, Any ]:
		"""Build the Interactions speech-generation configuration.

		Purpose:
			Constructs the single-speaker speech configuration and adds supported inference
			settings without submitting legacy Generate Content configuration objects.

		Args:
			voice (str): Supported Gemini voice name.
			temperature (Optional[float]): Sampling temperature.
			top_p (Optional[float]): Top-p sampling value.
			max_tokens (Optional[int]): Maximum output-token count.

		Returns:
			Dict[str, Any]: Interactions generation configuration.

		Raises:
			Error: Raised when validation or configuration construction fails.
			ValueError: Raised when ``voice`` is missing.
		"""
		try:
			throw_if( 'voice', voice )
			self.voice = voice
			self.temperature = temperature
			self.top_p = top_p
			self.max_tokens = max_tokens
			self.generation_config = { 'speech_config': [ { 'voice': self.voice, }, ], }
			
			if self.temperature is not None:
				self.generation_config[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.generation_config[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None and self.max_tokens > 0:
				self.generation_config[ 'max_output_tokens' ] = (self.max_tokens)
			
			return self.generation_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = ('build_generation_config( self, voice: str, '
			                    'temperature: Optional[ float ], top_p: Optional[ float ], '
			                    'max_tokens: Optional[ int ] ) -> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def create_speech( self, text: str, filepath: str = None,
		model: str = 'gemini-3.1-flash-tts-preview', format: str = 'audio/wav', speed: float =
		None,
		voice: str = None, frequency: float = None, presense: float = None, max_tokens: int = None,
		instruct: str = None, temperature: float = None,
		top_p: float = None ) -> bytes | str | None:
		"""Generate speech through the Gemini Interactions API.

		Purpose:
			Builds the controllable TTS prompt, submits an audio-only Interaction, decodes the
			returned base64 PCM audio, wraps it in a WAV container, and returns the WAV bytes or
			the path written by the method.

		Args:
			text (str): Text to synthesize.
			filepath (str): Optional output WAV path.
			model (str): Gemini text-to-speech model.
			format (str): Application-facing audio format.
			speed (float): Relative speech-rate control.
			voice (str): Prebuilt Gemini voice.
			frequency (float): Frequency penalty retained for UI compatibility.
			presense (float): Presence penalty retained for UI compatibility.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional speech-performance instruction.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.

		Returns:
			bytes | str | None: WAV bytes when no path is supplied, or the written file path.

		Raises:
			Error: Raised when validation, request execution, decoding, or file writing fails.
			ValueError: Raised when required input is missing or no audio is returned.
		"""
		try:
			throw_if( 'text', text )
			self.text = text
			self.audio_path = filepath
			self.model = model
			self.response_format = format
			self.speed = speed
			self.voice = voice
			self.frequency_penalty = frequency
			self.presence_penalty = presense
			self.max_tokens = max_tokens
			self.instructions = instruct
			self.temperature = temperature
			self.top_p = top_p
			
			if self.response_format != 'audio/wav':
				raise ValueError( 'Gemini TTS wrapper supports WAV output only.' )
			
			if self.model not in self.model_options:
				raise ValueError( f'Unsupported Gemini TTS model: {self.model}' )
			
			self.input_text = self.normalize_tts_prompt( text=self.text, speed=self.speed,
				instruct=self.instructions )
			
			self.voice = self.normalize_voice( self.voice )
			
			self.generation_config = self.build_generation_config( voice=self.voice,
				temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens )
			
			self.api_key = (os.getenv( 'GEMINI_API_KEY' ) or os.getenv(
				'GOOGLE_API_KEY' ) or self.gemini_api_key or self.google_api_key)
			throw_if( 'api_key', self.api_key )
			
			self.client = genai.Client( api_key=self.api_key, http_options=self.http_options )
			
			self.interaction = self.client.interactions.create( model=self.model,
				input=self.input_text, response_format={ 'type': 'audio', },
				generation_config=self.generation_config, store=False )
			
			self.response = self.interaction
			self.content_response = self.interaction
			self.output_audio = getattr( self.interaction, 'output_audio', None )
			
			if self.output_audio is None:
				raise ValueError( 'No audio output was returned by Gemini TTS.' )
			
			self.encoded_audio = getattr( self.output_audio, 'data', None )
			
			if not self.encoded_audio:
				raise ValueError( 'Gemini TTS returned an empty audio payload.' )
			
			self.pcm_bytes = base64.b64decode( self.encoded_audio )
			self.audio_bytes = self.to_wave_bytes( pcm_data=self.pcm_bytes )
			
			if self.audio_path is not None and str( self.audio_path ).strip( ):
				self.audio_path = str( self.audio_path ).strip( )
				
				with open( self.audio_path, 'wb' ) as audio_file:
					audio_file.write( self.audio_bytes )
				
				return self.audio_path
			
			return self.audio_bytes
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'TTS'
			exception.method = ('create_speech( self, text: str, filepath: str, model: str, '
			                    'format: str, speed: float, voice: str, frequency: float, '
			                    'presense: float, max_tokens: int, instruct: str, temperature: '
			                    'float, top_p: float ) -> bytes | str | None')
			Logger( ).write( exception )
			raise exception

class Transcription( Gemini ):
	"""Gemini audio-transcription wrapper.

	Purpose:
		Transcribes local audio files into text through the Gemini Interactions API. The class
		normalizes audio MIME types, builds transcription instructions with optional language
		and time-range constraints, uploads the local audio through the Gemini Files API, and
		submits the uploaded audio as multimodal Interactions input.

	Attributes:
		client (Optional[genai.Client]): Active Gemini SDK client.
		http_options (HttpOptions): Gemini client HTTP configuration.
		transcript (Optional[str]): Text returned by the transcription request.
		file_path (Optional[str]): Local audio-file path.
		mime_type (Optional[str]): Normalized audio MIME type.
		uploaded_file (Optional[File]): File resource uploaded to Gemini.
		interaction (Optional[Any]): Most recent Gemini Interaction.
		response (Optional[Any]): Raw Interaction consumed by application token accounting.
		generation_config (Dict[str, Any]): Interactions generation configuration.
	"""
	
	client: Optional[ genai.Client ]
	http_options: HttpOptions
	transcript: Optional[ str ]
	file_path: Optional[ str ]
	mime_type: Optional[ str ]
	uploaded_file: Optional[ File ]
	interaction: Optional[ Any ]
	response: Optional[ Any ]
	generation_config: Dict[ str, Any ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3.6-flash', temperature: float = 0.8,
		top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 10000,
		instruct: str = None ) -> None:
		"""Initialize the Transcription wrapper.

		Purpose:
			Initializes transcription settings, local file state, request configuration, and
			response placeholders. The constructor performs local state assignment only and does
			not create a provider client or submit a request.

		Args:
			n (int): Candidate-count value retained for interface compatibility.
			model (str): Default Gemini audio-understanding model.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for interface compatibility.
			presence (float): Presence penalty retained for interface compatibility.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction text.

		Returns:
			None: This method initializes object state through side effects.
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
		self.api_version = 'v1beta'
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.client = None
		self.transcript = None
		self.file_path = None
		self.mime_type = None
		self.uploaded_file = None
		self.interaction = None
		self.response = None
		self.content_response = None
		self.generation_config = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported audio-transcription models.

		Purpose:
			Provides the Gemini audio-understanding model identifiers exposed by the Jeni Audio
			mode.

		Returns:
			List[str]: Supported Gemini transcription-model identifiers.
		"""
		return [ 'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.1-pro-preview',
			'gemini-2.5-flash', 'gemini-2.5-pro', ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Return supported transcription language hints.

		Purpose:
			Provides optional spoken-language hints consumed by the Jeni Audio controls.

		Returns:
			List[str]: Supported language-hint values.
		"""
		return [ '', 'Auto Detect', 'English', 'Spanish', 'French', 'German', 'Italian',
			'Portuguese', 'Dutch', 'Russian', 'Ukrainian', 'Polish', 'Arabic', 'Hebrew', 'Hindi',
			'Bengali', 'Urdu', 'Chinese', 'Japanese', 'Korean', 'Vietnamese', 'Thai', 'Indonesian',
			'Filipino', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Return supported audio MIME types.

		Purpose:
			Provides audio MIME-type values accepted by the Jeni Audio mode.

		Returns:
			List[str]: Supported audio MIME types.
		"""
		return [ 'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/mp4', 'audio/x-m4a', 'audio/aac',
			'audio/ogg', 'audio/flac', 'audio/webm', ]
	
	def normalize_mime_type( self, path: str, mime_type: str = None ) -> str:
		"""Normalize an audio MIME type.

		Purpose:
			Uses an explicitly supplied MIME type when available and otherwise derives the MIME
			type from the local file extension.

		Args:
			path (str): Local audio-file path.
			mime_type (str): Optional explicit audio MIME type.

		Returns:
			str: Normalized audio MIME type.

		Raises:
			Error: Raised when validation or MIME-type normalization fails.
			ValueError: Raised when ``path`` is missing.
		"""
		try:
			import mimetypes
			
			throw_if( 'path', path )
			self.file_path = path
			self.mime_type = mime_type
			
			if self.mime_type is not None and str( self.mime_type ).strip( ):
				return str( self.mime_type ).strip( )
			
			self.guessed_mime_type = mimetypes.guess_type( self.file_path )[ 0 ]
			
			if self.guessed_mime_type:
				return self.guessed_mime_type
			
			self.file_suffix = Path( self.file_path ).suffix.lower( )
			self.mime_types = { '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.mpeg': 'audio/mpeg',
				'.mp4': 'audio/mp4', '.m4a': 'audio/x-m4a', '.aac': 'audio/aac',
				'.ogg': 'audio/ogg', '.oga': 'audio/ogg', '.flac': 'audio/flac',
				'.webm': 'audio/webm', }
			
			return self.mime_types.get( self.file_suffix, 'application/octet-stream' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = ('normalize_mime_type( self, path: str, mime_type: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def build_prompt( self, language: str = None, start_time: float = 0.0,
		end_time: float = 0.0 ) -> str:
		"""Build the transcription prompt.

		Purpose:
			Constructs the transcription instruction and adds optional spoken-language and
			time-range constraints without altering the source audio.

		Args:
			language (str): Optional spoken-language hint.
			start_time (float): Optional starting timestamp in seconds.
			end_time (float): Optional ending timestamp in seconds.

		Returns:
			str: Complete transcription prompt.

		Raises:
			Error: Raised when prompt construction fails.
		"""
		try:
			self.language = language
			self.start_time = start_time
			self.end_time = end_time
			self.prompt_parts: List[ str ] = [
				'Transcribe the spoken content in this audio accurately.',
				'Return only the transcript unless the supplied instructions require '
				'additional formatting.', ]
			
			self.language_name = str( self.language or '' ).strip( )
			
			if self.language_name and self.language_name.lower( ) != 'auto detect':
				self.prompt_parts.append( f'The expected spoken language is '
				                          f'{self.language_name}.' )
			
			if self.start_time > 0.0 and self.end_time > self.start_time:
				self.prompt_parts.append(
					f'Transcribe only the segment from {self.start_time:.3f} seconds '
					f'through {self.end_time:.3f} seconds.' )
			
			elif self.start_time > 0.0:
				self.prompt_parts.append( f'Begin transcription at {self.start_time:.3f} '
				                          f'seconds.' )
			
			elif self.end_time > 0.0:
				self.prompt_parts.append( f'Stop transcription at {self.end_time:.3f} seconds.' )
			
			self.prompt = '\n'.join( self.prompt_parts )
			return self.prompt
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = ('build_prompt( self, language: str, start_time: float, '
			                    'end_time: float ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def build_generation_config( self, temperature: float, top_p: float, max_tokens: int ) -> Dict[
		str, Any ]:
		"""Build the transcription generation configuration.

		Purpose:
			Converts supported Jeni inference controls into the Interactions generation
			configuration.

		Args:
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			max_tokens (int): Maximum output-token count.

		Returns:
			Dict[str, Any]: Interactions generation configuration.

		Raises:
			Error: Raised when configuration construction fails.
		"""
		try:
			self.temperature = temperature
			self.top_p = top_p
			self.max_tokens = max_tokens
			self.generation_config = { }
			if self.temperature is not None:
				self.generation_config[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.generation_config[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None and self.max_tokens > 0:
				self.generation_config[ 'max_output_tokens' ] = (self.max_tokens)
			
			return self.generation_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = ('build_generation_config( self, temperature: float, top_p: float, '
			                    'max_tokens: int ) -> Dict[str, Any]')
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, path: str, model: str = 'gemini-3.6-flash', language: str = None,
		mime_type: str = None, temperature: float = None, top_p: float = None,
		frequency: float = None, presence: float = None, max_tokens: int = None,
		start_time: float = 0.0, end_time: float = 0.0, instruct: str = None ) -> str:
		"""Transcribe audio through the Gemini Interactions API.

		Purpose:
			Uploads the local audio file, constructs a multimodal Interactions request using the
			uploaded file URI and MIME type, and returns the generated transcript.

		Args:
			path (str): Local audio-file path.
			model (str): Gemini audio-understanding model.
			language (str): Optional spoken-language hint.
			mime_type (str): Optional explicit audio MIME type.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for UI compatibility.
			presence (float): Presence penalty retained for UI compatibility.
			max_tokens (int): Maximum output-token count.
			start_time (float): Optional starting timestamp in seconds.
			end_time (float): Optional ending timestamp in seconds.
			instruct (str): Optional system instruction text.

		Returns:
			str: Generated transcript.

		Raises:
			Error: Raised when validation, upload, request execution, or response extraction
				fails.
			ValueError: Raised when required input is missing or the transcript is empty.
		"""
		try:
			throw_if( 'path', path )
			self.file_path = path
			self.model = str( model or self.model or 'gemini-3.6-flash' ).strip( )
			throw_if( 'model', self.model )
			self.language = language
			self.mime_type = mime_type
			self.temperature = (temperature if temperature is not None else self.temperature)
			self.top_p = top_p if top_p is not None else self.top_p
			self.frequency_penalty = (
				frequency if frequency is not None else self.frequency_penalty)
			self.presence_penalty = (presence if presence is not None else self.presence_penalty)
			self.max_tokens = (max_tokens if max_tokens is not None else self.max_tokens)
			self.start_time = start_time
			self.end_time = end_time
			self.instructions = (instruct if instruct is not None else self.instructions)
			self.mime_type = self.normalize_mime_type( path=self.file_path,
				mime_type=self.mime_type )
			self.prompt = self.build_prompt( language=self.language, start_time=self.start_time,
				end_time=self.end_time )
			self.generation_config = self.build_generation_config( temperature=self.temperature,
				top_p=self.top_p, max_tokens=self.max_tokens )
			self.api_key = (os.getenv( 'GEMINI_API_KEY' ) or os.getenv(
				'GOOGLE_API_KEY' ) or self.gemini_api_key or self.google_api_key)
			throw_if( 'api_key', self.api_key )
			
			self.client = genai.Client( api_key=self.api_key, http_options=self.http_options )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			self.file_uri = str( getattr( self.uploaded_file, 'uri', '' ) or '' ).strip( )
			throw_if( 'file_uri', self.file_uri )
			
			self.uploaded_mime_type = str(
				getattr( self.uploaded_file, 'mime_type', '' ) or self.mime_type ).strip( )
			
			self.request: Dict[ str, Any ] = { 'model': self.model,
				'input': [ { 'type': 'text', 'text': self.prompt, },
					{ 'type': 'audio', 'uri': self.file_uri,
						'mime_type': self.uploaded_mime_type, }, ],
				'response_format': { 'type': 'text', }, 'store': False, }
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.request[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.generation_config:
				self.request[ 'generation_config' ] = (self.generation_config)
			
			self.interaction = self.client.interactions.create( **self.request )
			self.response = self.interaction
			self.content_response = self.interaction
			self.transcript = str( getattr( self.interaction, 'output_text', '' ) or '' ).strip( )
			
			if not self.transcript:
				raise ValueError( 'Gemini returned an empty transcription.' )
			
			return self.transcript
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Transcription'
			exception.method = ('transcribe( self, path: str, model: str, language: str, mime_type: '
			                    'str, temperature: float, top_p: float, frequency: float, '
			                    'presence: float, max_tokens: int, start_time: float, end_time: '
			                    'float, instruct: str ) -> str')
			Logger( ).write( exception )
			raise exception

class Translation( Gemini ):
	"""Gemini audio-translation wrapper.

	Purpose:
		Translates spoken audio into target-language text through the Gemini Interactions API.
		The class normalizes audio MIME types, builds translation instructions with source and
		target language hints, uploads the local audio through the Gemini Files API, and submits
		the uploaded audio as multimodal Interactions input.

	Attributes:
		client (Optional[genai.Client]): Active Gemini SDK client.
		http_options (HttpOptions): Gemini client HTTP configuration.
		target_language (Optional[str]): Requested translation target language.
		source_language (Optional[str]): Optional spoken source-language hint.
		file_path (Optional[str]): Local audio-file path.
		mime_type (Optional[str]): Normalized audio MIME type.
		translation (Optional[str]): Text returned by the translation request.
		uploaded_file (Optional[File]): File resource uploaded to Gemini.
		interaction (Optional[Any]): Most recent Gemini Interaction.
		response (Optional[Any]): Raw Interaction consumed by application token accounting.
		generation_config (Dict[str, Any]): Interactions generation configuration.
	"""
	
	client: Optional[ genai.Client ]
	http_options: HttpOptions
	target_language: Optional[ str ]
	source_language: Optional[ str ]
	file_path: Optional[ str ]
	mime_type: Optional[ str ]
	translation: Optional[ str ]
	uploaded_file: Optional[ File ]
	interaction: Optional[ Any ]
	response: Optional[ Any ]
	generation_config: Dict[ str, Any ]
	
	def __init__( self, n: int = 1, model: str = 'gemini-3.6-flash', temperature: float = 0.8,
		top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 10000,
		instruct: str = None ) -> None:
		"""Initialize the Translation wrapper.

		Purpose:
			Initializes translation settings, language state, local file state, request
			configuration, and response placeholders. The constructor performs local state
			assignment only.

		Args:
			n (int): Candidate-count value retained for interface compatibility.
			model (str): Default Gemini audio-understanding model.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for interface compatibility.
			presence (float): Presence penalty retained for interface compatibility.
			max_tokens (int): Maximum output-token count.
			instruct (str): Optional system instruction text.

		Returns:
			None: This method initializes object state through side effects.
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
		self.api_version = 'v1beta'
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.client = None
		self.target_language = None
		self.source_language = None
		self.file_path = None
		self.mime_type = None
		self.translation = None
		self.uploaded_file = None
		self.interaction = None
		self.response = None
		self.content_response = None
		self.generation_config = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported audio-translation models.

		Purpose:
			Provides the Gemini audio-understanding model identifiers exposed by the Jeni Audio
			mode.

		Returns:
			List[str]: Supported Gemini translation-model identifiers.
		"""
		return [ 'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.1-pro-preview',
			'gemini-2.5-flash', 'gemini-2.5-pro', ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Return supported translation languages.

		Purpose:
			Provides source and target language values consumed by the Jeni Audio controls.

		Returns:
			List[str]: Supported translation language values.
		"""
		return [ 'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Dutch',
			'Russian', 'Ukrainian', 'Polish', 'Arabic', 'Hebrew', 'Hindi', 'Bengali', 'Urdu',
			'Chinese', 'Japanese', 'Korean', 'Vietnamese', 'Thai', 'Indonesian', 'Filipino', ]
	
	def normalize_mime_type( self, path: str, mime_type: str = None ) -> str:
		"""Normalize an audio MIME type.

		Purpose:
			Uses an explicitly supplied MIME type when available and otherwise derives the MIME
			type from the local file extension.

		Args:
			path (str): Local audio-file path.
			mime_type (str): Optional explicit audio MIME type.

		Returns:
			str: Normalized audio MIME type.

		Raises:
			Error: Raised when validation or MIME-type normalization fails.
			ValueError: Raised when ``path`` is missing.
		"""
		try:
			import mimetypes
			
			throw_if( 'path', path )
			self.file_path = path
			self.mime_type = mime_type
			if self.mime_type is not None and str( self.mime_type ).strip( ):
				return str( self.mime_type ).strip( )
			
			self.guessed_mime_type = mimetypes.guess_type( self.file_path )[ 0 ]
			if self.guessed_mime_type:
				return self.guessed_mime_type
			
			self.file_suffix = Path( self.file_path ).suffix.lower( )
			self.mime_types = { '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.mpeg': 'audio/mpeg',
				'.mp4': 'audio/mp4', '.m4a': 'audio/x-m4a', '.aac': 'audio/aac',
				'.ogg': 'audio/ogg', '.oga': 'audio/ogg', '.flac': 'audio/flac',
				'.webm': 'audio/webm', }
			
			return self.mime_types.get( self.file_suffix, 'application/octet-stream' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = ('normalize_mime_type( self, path: str, mime_type: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def build_prompt( self, target_language: str, source_language: str = None,
		start_time: float = 0.0, end_time: float = 0.0 ) -> str:
		"""Build the audio-translation prompt.

		Purpose:
			Constructs a target-language translation instruction and adds optional source-language
			and time-range constraints.

		Args:
			target_language (str): Requested translation target language.
			source_language (str): Optional spoken source-language hint.
			start_time (float): Optional starting timestamp in seconds.
			end_time (float): Optional ending timestamp in seconds.

		Returns:
			str: Complete translation prompt.

		Raises:
			Error: Raised when validation or prompt construction fails.
			ValueError: Raised when ``target_language`` is missing.
		"""
		try:
			throw_if( 'target_language', target_language )
			self.target_language = target_language
			self.source_language = source_language
			self.start_time = start_time
			self.end_time = end_time
			self.prompt_parts: List[ str ] = [ f'Translate all spoken content in this audio into '
			                                   f'{self.target_language}.',
				'Return only the translated text unless the supplied instructions require '
				'additional formatting.',
				'Preserve the original meaning, tone, names, numbers, and technical '
				'terminology.', ]
			
			self.source_language_name = str( self.source_language or '' ).strip( )
			if self.source_language_name and self.source_language_name.lower( ) != 'auto detect':
				self.prompt_parts.append( f'The expected source language is '
				                          f'{self.source_language_name}.' )
			
			if self.start_time > 0.0 and self.end_time > self.start_time:
				self.prompt_parts.append(
					f'Translate only the segment from {self.start_time:.3f} seconds '
					f'through {self.end_time:.3f} seconds.' )
			
			elif self.start_time > 0.0:
				self.prompt_parts.append( f'Begin translation at {self.start_time:.3f} seconds.' )
			
			elif self.end_time > 0.0:
				self.prompt_parts.append( f'Stop translation at {self.end_time:.3f} seconds.' )
			
			self.prompt = '\n'.join( self.prompt_parts )
			return self.prompt
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = ('build_prompt( self, target_language: str, source_language: str, '
			                    'start_time: float, end_time: float ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def build_generation_config( self, temperature: float, top_p: float, max_tokens: int ) -> Dict[
		str, Any ]:
		"""Build the translation generation configuration.

		Purpose:
			Converts supported Jeni inference controls into the Interactions generation
			configuration.

		Args:
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			max_tokens (int): Maximum output-token count.

		Returns:
			Dict[str, Any]: Interactions generation configuration.

		Raises:
			Error: Raised when configuration construction fails.
		"""
		try:
			self.temperature = temperature
			self.top_p = top_p
			self.max_tokens = max_tokens
			self.generation_config = { }
			if self.temperature is not None:
				self.generation_config[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.generation_config[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None and self.max_tokens > 0:
				self.generation_config[ 'max_output_tokens' ] = (self.max_tokens)
			
			return self.generation_config
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = ('build_generation_config( self, temperature: float, top_p: float, '
			                    'max_tokens: int ) -> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def translate( self, path: str, target_language: str, model: str = 'gemini-3.6-flash',
		source_language: str = None, mime_type: str = None, temperature: float = None,
		top_p: float = None, frequency: float = None, presence: float = None,
		max_tokens: int = None, start_time: float = 0.0, end_time: float = 0.0,
		instruct: str = None ) -> str:
		"""Translate audio through the Gemini Interactions API.

		Purpose:
			Uploads the local audio file, constructs a multimodal Interactions request using the
			uploaded file URI and MIME type, and returns the generated target-language text.

		Args:
			path (str): Local audio-file path.
			target_language (str): Requested translation target language.
			model (str): Gemini audio-understanding model.
			source_language (str): Optional spoken source-language hint.
			mime_type (str): Optional explicit audio MIME type.
			temperature (float): Sampling temperature.
			top_p (float): Top-p sampling value.
			frequency (float): Frequency penalty retained for UI compatibility.
			presence (float): Presence penalty retained for UI compatibility.
			max_tokens (int): Maximum output-token count.
			start_time (float): Optional starting timestamp in seconds.
			end_time (float): Optional ending timestamp in seconds.
			instruct (str): Optional system instruction text.

		Returns:
			str: Generated target-language translation.

		Raises:
			Error: Raised when validation, upload, request execution, or response extraction
				fails.
			ValueError: Raised when required input is missing or the translation is empty.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'target_language', target_language )
			self.file_path = path
			self.target_language = target_language
			self.model = str( model or self.model or 'gemini-3.6-flash' ).strip( )
			throw_if( 'model', self.model )
			self.source_language = source_language
			self.mime_type = mime_type
			self.temperature = (temperature if temperature is not None else self.temperature)
			self.top_p = top_p if top_p is not None else self.top_p
			self.frequency_penalty = (
				frequency if frequency is not None else self.frequency_penalty)
			self.presence_penalty = (presence if presence is not None else self.presence_penalty)
			self.max_tokens = (max_tokens if max_tokens is not None else self.max_tokens)
			self.start_time = start_time
			self.end_time = end_time
			self.instructions = (instruct if instruct is not None else self.instructions)
			self.mime_type = self.normalize_mime_type( path=self.file_path,
				mime_type=self.mime_type )
			self.prompt = self.build_prompt( target_language=self.target_language,
				source_language=self.source_language, start_time=self.start_time,
				end_time=self.end_time )
			self.generation_config = self.build_generation_config( temperature=self.temperature,
				top_p=self.top_p, max_tokens=self.max_tokens )
			self.api_key = self.gemini_api_key or self.google_api_key
			self.client = genai.Client( api_key=self.api_key, http_options=self.http_options )
			self.uploaded_file = self.client.files.upload( file=self.file_path )
			self.file_uri = str( getattr( self.uploaded_file, 'uri', '' ) or '' ).strip( )
			throw_if( 'file_uri', self.file_uri )
			self.uploaded_mime_type = str(
				getattr( self.uploaded_file, 'mime_type', '' ) or self.mime_type ).strip( )
			
			self.request: Dict[ str, Any ] = { 'model': self.model,
				'input': [ { 'type': 'text', 'text': self.prompt, },
					{ 'type': 'audio', 'uri': self.file_uri,
						'mime_type': self.uploaded_mime_type, }, ],
				'response_format': { 'type': 'text', }, 'store': False, }
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.request[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.generation_config:
				self.request[ 'generation_config' ] = (self.generation_config)
			
			self.interaction = self.client.interactions.create( **self.request )
			self.response = self.interaction
			self.content_response = self.interaction
			self.translation = str( getattr( self.interaction, 'output_text', '' ) or '' ).strip( )
			if not self.translation:
				raise ValueError( 'Gemini returned an empty audio translation.' )
			
			return self.translation
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Translation'
			exception.method = ('translate( self, path: str, target_language: str, model: str, '
			                    'source_language: str, mime_type: str, temperature: float, '
			                    'top_p: float, frequency: float, presence: float, max_tokens: int, '
			                    'start_time: float, end_time: float, instruct: str ) -> str')
			Logger( ).write( exception )
			raise exception

class Files( Gemini ):
	"""Gemini file and document workflow wrapper.

	Purpose:
		Manages Gemini Files API resources and executes document, web-search, and Google Maps
		model workflows. File lifecycle operations remain on the specialized Files API, while
		document reasoning and grounded generation use the Gemini Interactions API.

	Attributes:
		client (Optional[genai.Client]): Active Google Gen AI client.
		storage_client (Optional[storage.Client]): Google Cloud Storage client.
		project_id (Optional[str]): Google Cloud project identifier.
		project_location (Optional[str]): Google Cloud location.
		file_id (Optional[str]): Active Gemini file resource name.
		bucket_id (Optional[str]): Active Google Cloud Storage bucket name.
		display_name (Optional[str]): Uploaded file display name.
		mime_type (Optional[str]): Active file MIME type.
		file_path (Optional[str]): Active local file path.
		file_list (List[str]): Google Cloud Storage object names returned by the latest list
		operation.
		file_paths (List[str]): Local paths used by a multi-document request.
		file_lists (List[File]): Uploaded Gemini file resources used by a multi-document request.
		response (Optional[Any]): Most recent Files API or Interactions response.
		interaction (Optional[Any]): Most recent Gemini Interaction.
		use_vertex (bool): Compatibility flag retained for application configuration.
		collections (Dict[str, str]): Compatibility collection mapping.
		documents (Dict[str, str]): Compatibility document mapping.
	"""
	
	client: Optional[ genai.Client ]
	storage_client: Optional[ storage.Client ]
	project_id: Optional[ str ]
	project_location: Optional[ str ]
	file_id: Optional[ str ]
	bucket_id: Optional[ str ]
	display_name: Optional[ str ]
	mime_type: Optional[ str ]
	file_path: Optional[ str ]
	file_list: List[ str ]
	file_paths: List[ str ]
	file_lists: List[ File ]
	response: Optional[ Any ]
	interaction: Optional[ Any ]
	use_vertex: bool
	collections: Dict[ str, str ]
	documents: Dict[ str, str ]
	
	def __init__( self, model: str = 'gemini-3.6-flash' ) -> None:
		"""Initialize the Files wrapper.

		Purpose:
			Initializes file lifecycle, document request, storage, and response state. The
			constructor performs local assignment only.

		Args:
			model (str): Default Gemini model used for document workflows.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.project_location = cfg.GOOGLE_CLOUD_LOCATION
		self.model = model
		self.api_version = 'v1beta'
		self.http_options = types.HttpOptions( api_version=self.api_version )
		self.client = None
		self.storage_client = None
		self.bucket_id = None
		self.file_id = None
		self.display_name = None
		self.mime_type = None
		self.file_path = None
		self.file_list = [ ]
		self.file_paths = [ ]
		self.file_lists = [ ]
		self.files = [ ]
		self.response = None
		self.interaction = None
		self.use_vertex = False
		self.collections = { }
		self.documents = { }
		self.contents = None
		self.generation_config = { }
		self.tool_config = [ ]
		self.grounding_sources = [ ]
	
	@property
	def file_options( self ) -> List[ str ]:
		"""Return cached file resource names.

		Returns:
			List[str]: Cached Gemini file resource names.
		"""
		return list( self.files )
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported document models.

		Returns:
			List[str]: Supported Gemini model identifiers.
		"""
		return [ 'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.5-flash-lite',
			'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite', 'gemini-2.5-pro',
			'gemini-2.5-flash',
			'gemini-2.5-flash-lite' ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Return compatibility include options.

		Returns:
			List[str]: Existing application include-option values.
		"""
		return [ 'file_search_call.results', 'message.input_image.image_url',
			'message.output_text.logprobs', 'reasoning.encrypted_content' ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Return supported thinking levels.

		Returns:
			List[str]: Supported thinking-level values.
		"""
		return [ 'THINKING_LEVEL_UNSPECIFIED', 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH' ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Return supported tool-choice values.

		Returns:
			List[str]: Supported tool-choice values.
		"""
		return [ 'AUTO', 'ANY', 'NONE', 'VALIDATED' ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Return supported Interactions tools.

		Returns:
			List[str]: Supported tool identifiers.
		"""
		return [ 'google_search', 'google_maps', 'url_context', 'code_execution' ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Return supported response modalities.

		Returns:
			List[str]: Supported response modality values.
		"""
		return [ 'TEXT' ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Return supported media-resolution values.

		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low' ]
	
	def create_client( self ) -> genai.Client:
		"""Create the Google Gen AI client.

		Returns:
			genai.Client: Configured provider client.

		Raises:
			Error: Raised when API-key validation or client creation fails.
		"""
		try:
			self.api_key = (os.getenv( 'GEMINI_API_KEY' ) or os.getenv(
				'GOOGLE_API_KEY' ) or self.gemini_api_key or self.google_api_key)
			throw_if( 'api_key', self.api_key )
			self.client = genai.Client( api_key=self.api_key, http_options=self.http_options )
			return self.client
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = 'create_client( self ) -> genai.Client'
			Logger( ).write( exception )
			raise exception
	
	def normalize_mime_type( self, filepath: str ) -> str:
		"""Resolve the MIME type for a local file.

		Args:
			filepath (str): Local file path.

		Returns:
			str: Resolved MIME type.

		Raises:
			Error: Raised when validation or MIME-type resolution fails.
		"""
		try:
			import mimetypes
			
			throw_if( 'filepath', filepath )
			self.file_path = filepath
			self.mime_type = mimetypes.guess_type( self.file_path )[ 0 ]
			
			if self.mime_type:
				return self.mime_type
			
			self.suffix = Path( self.file_path ).suffix.lower( )
			self.mime_types = { '.pdf': 'application/pdf', '.txt': 'text/plain',
				'.md': 'text/markdown',
				'.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
				'.csv': 'text/csv', '.json': 'application/json', '.png': 'image/png',
				'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', }
			return self.mime_types.get( self.suffix, 'application/octet-stream' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('normalize_mime_type( self, filepath: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def upload( self, filepath: str, name: str = None ) -> File:
		"""Upload a local file through the Gemini Files API.

		Args:
			filepath (str): Local file path.
			name (str): Optional display name.

		Returns:
			File: Uploaded Gemini file resource.

		Raises:
			Error: Raised when validation or upload fails.
		"""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = filepath
			self.display_name = name
			self.client = self.create_client( )
			self.upload_config = None
			
			if self.display_name is not None and str( self.display_name ).strip( ):
				self.upload_config = types.UploadFileConfig(
					display_name=str( self.display_name ).strip( ) )
			
			if self.upload_config is None:
				self.response = self.client.files.upload( file=self.file_path )
			else:
				self.response = self.client.files.upload( file=self.file_path,
					config=self.upload_config )
			
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('upload( self, filepath: str, name: str ) -> File')
			Logger( ).write( exception )
			raise exception
	
	def list( self, model: str = 'gemini-3.6-flash', top_p: float = 0.8, top_k: int = 50,
		temperature: float = 0.5, frequency: float = 0.0, presence: float = 0.0,
		max_tokens: int = 8192, tool_choice: str = 'auto', stops: List[ str ] = None,
		tools: List[ str ] = None, domains: List[ str ] = None, modalities: List[ str ] = None,
		media_resolution: str = 'media_resolution_medium' ) -> List[ str ]:
		"""List configured Google Cloud Storage document objects.

		Purpose:
			Preserves the existing Files wrapper contract by listing objects from the
			``jeni-financial`` bucket under the ``regulations`` prefix. The model and
			generation arguments remain accepted because they are part of the existing
			application-facing signature, but they are not provider inputs for this storage
			operation.

		Args:
			model (str): Model value retained by the wrapper.
			top_p (float): Top-P value retained by the wrapper.
			top_k (int): Top-K value retained by the wrapper.
			temperature (float): Temperature value retained by the wrapper.
			frequency (float): Frequency-penalty value retained by the wrapper.
			presence (float): Presence-penalty value retained by the wrapper.
			max_tokens (int): Maximum-token value retained by the wrapper.
			tool_choice (str): Tool-choice value retained by the wrapper.
			stops (List[str]): Stop sequences retained by the wrapper.
			tools (List[str]): Tool identifiers retained by the wrapper.
			domains (List[str]): Domain values retained by the wrapper.
			modalities (List[str]): Response modalities retained by the wrapper.
			media_resolution (str): Media-resolution value retained by the wrapper.

		Returns:
			List[str]: Object names under the configured regulations prefix.

		Raises:
			Error: Raised when the Google Cloud Storage listing fails.
		"""
		try:
			self.model = model
			self.top_p = top_p
			self.top_k = top_k
			self.temperature = temperature
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.tool_choice = tool_choice
			self.stops = stops if isinstance( stops, list ) else [ ]
			self.tools = tools if isinstance( tools, list ) else [ ]
			self.domains = domains if isinstance( domains, list ) else [ ]
			self.response_modalities = (modalities if isinstance( modalities, list ) else [ ])
			self.media_resolution = media_resolution
			self.bucket_id = 'jeni-financial'
			self.prefix = 'regulations'
			self.storage_client = storage.Client( )
			self.bucket = self.storage_client.bucket( bucket_name=self.bucket_id )
			self.files = [ blob.name for blob in self.bucket.list_blobs( prefix=self.prefix ) ]
			self.file_list = list( self.files )
			self.response = self.file_list
			return self.file_list
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('list( self, model: str, top_p: float, top_k: int, '
			                    'temperature: float, frequency: float, presence: float, '
			                    'max_tokens: int, tool_choice: str, stops: List[ str ], '
			                    'tools: List[ str ], domains: List[ str ], '
			                    'modalities: List[ str ], media_resolution: str ) '
			                    '-> List[ str ]')
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, file_id: str ) -> File:
		"""Retrieve a Gemini file resource.

		Args:
			file_id (str): Gemini file resource name.

		Returns:
			File: Retrieved Gemini file resource.

		Raises:
			Error: Raised when validation or retrieval fails.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = self.create_client( )
			self.response = self.client.files.get( name=self.file_id )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('retrieve( self, file_id: str ) -> File')
			Logger( ).write( exception )
			raise exception
	
	def build_generation_config( self, temperature: float = None, top_p: float = None,
		max_tokens: int = None, stops: List[ str ] = None ) -> Dict[ str, Any ]:
		"""Build an Interactions generation configuration.

		Args:
			temperature (float): Sampling temperature.
			top_p (float): Top-P sampling value.
			max_tokens (int): Maximum output-token count.
			stops (List[str]): Stop sequences.

		Returns:
			Dict[str, Any]: Interactions generation configuration.
		"""
		self.temperature = temperature
		self.top_p = top_p
		self.max_tokens = max_tokens
		self.stops = stops if isinstance( stops, list ) else [ ]
		self.generation_config = { }
		
		if self.temperature is not None:
			self.generation_config[ 'temperature' ] = self.temperature
		
		if self.top_p is not None:
			self.generation_config[ 'top_p' ] = self.top_p
		
		if self.max_tokens is not None and self.max_tokens > 0:
			self.generation_config[ 'max_output_tokens' ] = (self.max_tokens)
		
		self.stop_sequences = [ str( item ).strip( ) for item in self.stops if
			item is not None and str( item ).strip( ) ]
		
		if self.stop_sequences:
			self.generation_config[ 'stop_sequences' ] = (self.stop_sequences)
		
		return self.generation_config
	
	def build_document_block( self, filepath: str ) -> Dict[ str, Any ]:
		"""Build an inline Interactions document block.

		Args:
			filepath (str): Local document path.

		Returns:
			Dict[str, Any]: Interactions document content block.

		Raises:
			Error: Raised when validation or file encoding fails.
		"""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = filepath
			self.mime_type = self.normalize_mime_type( self.file_path )
			self.file_bytes = Path( self.file_path ).read_bytes( )
			throw_if( 'file_bytes', self.file_bytes )
			self.file_data = base64.b64encode( self.file_bytes ).decode( 'utf-8' )
			return { 'type': 'document', 'data': self.file_data, 'mime_type': self.mime_type, }
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('build_document_block( self, filepath: str ) '
			                    '-> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def execute_document_interaction( self, prompt: str, filepaths: List[ str ], model: str,
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str:
		"""Execute a document Interaction.

		Args:
			prompt (str): Document instruction or question.
			filepaths (List[str]): Local document paths.
			model (str): Gemini model identifier.
			temperature (float): Sampling temperature.
			top_p (float): Top-P sampling value.
			frequency (float): Compatibility frequency-penalty value.
			presence (float): Compatibility presence-penalty value.
			max_tokens (int): Maximum output-token count.
			stops (List[str]): Stop sequences.
			instruct (str): Optional system instruction.

		Returns:
			str: Generated document response.

		Raises:
			Error: Raised when validation, request construction, or execution fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepaths', filepaths )
			throw_if( 'model', model )
			self.prompt = prompt
			self.file_paths = filepaths
			self.model = model
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops if isinstance( stops, list ) else [ ]
			self.instructions = instruct
			self.contents = [ { 'type': 'text', 'text': self.prompt }, ]
			
			for filepath in self.file_paths:
				self.contents.append( self.build_document_block( filepath ) )
			
			self.generation_config = self.build_generation_config( temperature=self.temperature,
				top_p=self.top_p, max_tokens=self.max_tokens, stops=self.stops )
			self.client = self.create_client( )
			self.request = { 'model': self.model, 'input': self.contents,
				'response_format': { 'type': 'text' }, 'store': False, }
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.request[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.generation_config:
				self.request[ 'generation_config' ] = (self.generation_config)
			
			self.interaction = self.client.interactions.create( **self.request )
			self.response = self.interaction
			self.content_response = self.interaction
			self.output_text = str( getattr( self.interaction, 'output_text', '' ) or '' ).strip( )
			throw_if( 'output_text', self.output_text )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('execute_document_interaction( self, prompt: str, '
			                    'filepaths: List[ str ], model: str, temperature: float, '
			                    'top_p: float, frequency: float, presence: float, '
			                    'max_tokens: int, stops: List[ str ], instruct: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def summarize( self, prompt: str, filepath: str, model: str = 'gemini-3.6-flash',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str:
		"""Summarize a local document through Interactions."""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = filepath
			return self.execute_document_interaction( prompt=prompt, filepaths=[ self.file_path ],
				model=model, temperature=temperature, top_p=top_p, frequency=frequency,
				presence=presence, max_tokens=max_tokens, stops=stops, instruct=instruct )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('summarize( self, prompt: str, filepath: str, model: str, '
			                    'temperature: float, top_p: float, frequency: float, '
			                    'presence: float, max_tokens: int, stops: List[ str ], '
			                    'instruct: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def search( self, prompt: str, filepath: str, model: str = 'gemini-3.6-flash',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str:
		"""Answer a question about a local document through Interactions."""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = filepath
			return self.execute_document_interaction( prompt=prompt, filepaths=[ self.file_path ],
				model=model, temperature=temperature, top_p=top_p, frequency=frequency,
				presence=presence, max_tokens=max_tokens, stops=stops, instruct=instruct )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('search( self, prompt: str, filepath: str, model: str, '
			                    'temperature: float, top_p: float, frequency: float, '
			                    'presence: float, max_tokens: int, stops: List[ str ], '
			                    'instruct: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def survey( self, prompt: str, filepaths: List[ str ], model: str = 'gemini-3.6-flash',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None ) -> str:
		"""Analyze multiple local documents through Interactions."""
		try:
			return self.execute_document_interaction( prompt=prompt, filepaths=filepaths,
				model=model, temperature=temperature, top_p=top_p, frequency=frequency,
				presence=presence, max_tokens=max_tokens, stops=stops, instruct=None )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('survey( self, prompt: str, filepaths: List[ str ], '
			                    'model: str, temperature: float, top_p: float, '
			                    'frequency: float, presence: float, max_tokens: int, '
			                    'stops: List[ str ] ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def execute_grounded_interaction( self, prompt: str, model: str, tool_type: str,
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str:
		"""Execute a Google Search or Google Maps Interaction."""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'tool_type', tool_type )
			self.prompt = prompt
			self.model = model
			self.tool_type = tool_type
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops if isinstance( stops, list ) else [ ]
			self.instructions = instruct
			
			if self.tool_type not in ('google_search', 'google_maps'):
				raise ValueError( f'Unsupported grounding tool: {self.tool_type}' )
			
			self.generation_config = self.build_generation_config( temperature=self.temperature,
				top_p=self.top_p, max_tokens=self.max_tokens, stops=self.stops )
			self.client = self.create_client( )
			self.request = { 'model': self.model, 'input': self.prompt,
				'tools': [ { 'type': self.tool_type } ], 'response_format': { 'type': 'text' },
				'store': False, }
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.request[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.generation_config:
				self.request[ 'generation_config' ] = (self.generation_config)
			
			self.interaction = self.client.interactions.create( **self.request )
			self.response = self.interaction
			self.content_response = self.interaction
			self.output_text = str( getattr( self.interaction, 'output_text', '' ) or '' ).strip( )
			throw_if( 'output_text', self.output_text )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('execute_grounded_interaction( self, prompt: str, model: str, '
			                    'tool_type: str, temperature: float, top_p: float, '
			                    'frequency: float, presence: float, max_tokens: int, '
			                    'stops: List[ str ], instruct: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def web_search( self, prompt: str, model: str = 'gemini-3.6-flash', temperature: float = None,
		top_p: float = None, frequency: float = None, presence: float = None,
		max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str:
		"""Generate a Google Search-grounded response."""
		try:
			return self.execute_grounded_interaction( prompt=prompt, model=model,
				tool_type='google_search', temperature=temperature, top_p=top_p,
				frequency=frequency, presence=presence, max_tokens=max_tokens, stops=stops,
				instruct=instruct )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('web_search( self, prompt: str, model: str, '
			                    'temperature: float, top_p: float, frequency: float, '
			                    'presence: float, max_tokens: int, stops: List[ str ], '
			                    'instruct: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def search_maps( self, prompt: str, model: str = 'gemini-3.6-flash', temperature: float = None,
		top_p: float = None, frequency: float = None, presence: float = None,
		max_tokens: int = None, stops: List[ str ] = None, instruct: str = None ) -> str:
		"""Generate a Google Maps-grounded response."""
		try:
			return self.execute_grounded_interaction( prompt=prompt, model=model,
				tool_type='google_maps', temperature=temperature, top_p=top_p, frequency=frequency,
				presence=presence, max_tokens=max_tokens, stops=stops, instruct=instruct )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('search_maps( self, prompt: str, model: str, '
			                    'temperature: float, top_p: float, frequency: float, '
			                    'presence: float, max_tokens: int, stops: List[ str ], '
			                    'instruct: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def delete( self, file_id: str ) -> bool:
		"""Delete a Gemini file resource.

		Args:
			file_id (str): Gemini file resource name.

		Returns:
			bool: True after successful deletion.

		Raises:
			Error: Raised when validation or deletion fails.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.client = self.create_client( )
			self.response = self.client.files.delete( name=self.file_id )
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'Files'
			exception.method = ('delete( self, file_id: str ) -> bool')
			Logger( ).write( exception )
			raise exception

class FileSearch( Gemini ):
	"""Gemini File Search Store wrapper.

	Purpose:
		Manages File Search Store resources used by retrieval-augmented workflows. The class
		creates, retrieves, lists, deletes, and refreshes store mappings without performing
		provider operations from the constructor.

	Attributes:
		client (Optional[genai.Client]): Active Google Gen AI client.
		response (Optional[Any]): Most recent File Search Store response.
		store_id (Optional[str]): Active File Search Store resource name.
		store_name (Optional[str]): Active File Search Store display name.
		collections (Dict[str, str]): Display-name to resource-name mapping.
		stores (List[FileSearchStore]): Cached File Search Store resources.
	"""
	
	client: Optional[ genai.Client ]
	response: Optional[ Any ]
	store_id: Optional[ str ]
	store_name: Optional[ str ]
	collections: Dict[ str, str ]
	stores: List[ FileSearchStore ]
	
	def __init__( self ) -> None:
		"""Initialize the File Search Store wrapper.

		Purpose:
			Initializes local File Search Store state without creating a provider client or
			performing network operations.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.client = None
		self.response = None
		self.store_id = None
		self.store_name = None
		self.collections = { }
		self.stores = [ ]
	
	def create_client( self ) -> genai.Client:
		"""Create the Google Gen AI client.

		Returns:
			genai.Client: Configured provider client.

		Raises:
			Error: Raised when API-key validation or client creation fails.
		"""
		try:
			self.api_key = (os.getenv( 'GEMINI_API_KEY' ) or os.getenv(
				'GOOGLE_API_KEY' ) or self.gemini_api_key or self.google_api_key)
			throw_if( 'api_key', self.api_key )
			self.client = genai.Client( api_key=self.api_key )
			return self.client
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = 'create_client( self ) -> genai.Client'
			Logger( ).write( exception )
			raise exception
	
	def refresh_collections( self ) -> Dict[ str, str ]:
		"""Refresh the File Search Store mapping.

		Returns:
			Dict[str, str]: Display-name to resource-name mapping.

		Raises:
			Error: Raised when the File Search Store listing fails.
		"""
		try:
			self.client = self.create_client( )
			self.collections = { }
			self.stores = list( self.client.file_search_stores.list( ) )
			
			for store in self.stores:
				self.display_name = getattr( store, 'display_name', None )
				self.resource_name = getattr( store, 'name', None )
				
				if not self.resource_name:
					continue
				
				self.label = (str( self.display_name ).strip( ) if self.display_name else str(
					self.resource_name ).strip( ))
				self.collections[ self.label ] = str( self.resource_name ).strip( )
			
			return self.collections
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('refresh_collections( self ) -> Dict[ str, str ]')
			Logger( ).write( exception )
			raise exception
	
	def create( self, name: str ) -> FileSearchStore:
		"""Create a File Search Store.

		Args:
			name (str): File Search Store display name.

		Returns:
			FileSearchStore: Created File Search Store.

		Raises:
			Error: Raised when validation or creation fails.
		"""
		try:
			throw_if( 'name', name )
			self.store_name = str( name ).strip( )
			self.client = self.create_client( )
			self.response = self.client.file_search_stores.create(
				config={ 'display_name': self.store_name } )
			self.refresh_collections( )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('create( self, name: str ) -> FileSearchStore')
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str ) -> FileSearchStore:
		"""Retrieve a File Search Store.

		Args:
			store_id (str): File Search Store resource name.

		Returns:
			FileSearchStore: Retrieved File Search Store.

		Raises:
			Error: Raised when validation or retrieval fails.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.client = self.create_client( )
			self.response = self.client.file_search_stores.get( name=self.store_id )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('retrieve( self, store_id: str ) -> FileSearchStore')
			Logger( ).write( exception )
			raise exception
	
	def list( self ) -> List[ FileSearchStore ]:
		"""List File Search Stores.

		Returns:
			List[FileSearchStore]: Available File Search Store resources.

		Raises:
			Error: Raised when listing fails.
		"""
		try:
			self.refresh_collections( )
			return list( self.stores )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('list( self ) -> List[ FileSearchStore ]')
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str, force: bool = True ) -> bool:
		"""Delete a File Search Store.

		Args:
			store_id (str): File Search Store resource name.
			force (bool): Whether to delete contained documents.

		Returns:
			bool: True after successful deletion.

		Raises:
			Error: Raised when validation or deletion fails.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = str( store_id ).strip( )
			self.force = force
			self.client = self.create_client( )
			self.client.file_search_stores.delete( name=self.store_id,
				config={ 'force': bool( self.force ) } )
			self.refresh_collections( )
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'FileSearch'
			exception.method = ('delete( self, store_id: str, force: bool ) -> bool')
			Logger( ).write( exception )
			raise exception

class CloudBuckets( Gemini ):
	"""Google Cloud Storage bucket wrapper.

	Purpose:
		Manages Google Cloud Storage objects and executes Google Search and Google Maps
		grounded model requests through the Gemini Interactions API.

	Attributes:
		project_id (Optional[str]): Google Cloud project identifier.
		bucket_name (Optional[str]): Active bucket name.
		object_name (Optional[str]): Active object name.
		file_path (Optional[str]): Active local file path.
		file_ids (List[str]): Cached file identifiers.
		store_ids (List[str]): Cached store identifiers.
		client (Optional[storage.Client]): Active Google Cloud Storage client.
		bucket (Optional[storage.Bucket]): Active bucket.
		response (Optional[Any]): Most recent storage or Interactions response.
		interaction (Optional[Any]): Most recent Gemini Interaction.
		collections (Dict[str, str]): Named bucket and prefix mappings.
		documents (Dict[str, str]): Named document mappings.
	"""
	
	project_id: Optional[ str ]
	bucket_name: Optional[ str ]
	object_name: Optional[ str ]
	file_path: Optional[ str ]
	file_ids: List[ str ]
	store_ids: List[ str ]
	client: Optional[ storage.Client ]
	bucket: Optional[ storage.Bucket ]
	response: Optional[ Any ]
	interaction: Optional[ Any ]
	collections: Dict[ str, str ]
	documents: Dict[ str, str ]
	
	def __init__( self ) -> None:
		"""Initialize the Google Cloud Storage wrapper.

		Purpose:
			Initializes local bucket, object, collection, and response state without creating
			clients or performing network operations.

		Returns:
			None: This method initializes object state through side effects.
		"""
		super( ).__init__( )
		self.project_id = cfg.GOOGLE_CLOUD_PROJECT_ID
		self.bucket_name = None
		self.object_name = None
		self.file_path = None
		self.file_ids = [ ]
		self.store_ids = [ ]
		self.client = None
		self.bucket = None
		self.response = None
		self.interaction = None
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
		"""Return supported grounded-generation models.

		Returns:
			List[str]: Supported Gemini model identifiers.
		"""
		return [ 'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.1-pro-preview',
			'gemini-2.5-flash', 'gemini-2.5-flash-lite' ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Return supported media-resolution values.

		Returns:
			List[str]: Supported media-resolution values.
		"""
		return [ 'media_resolution_high', 'media_resolution_medium', 'media_resolution_low' ]
	
	def create_storage_client( self ) -> storage.Client:
		"""Create the Google Cloud Storage client.

		Returns:
			storage.Client: Configured storage client.

		Raises:
			Error: Raised when client creation fails.
		"""
		try:
			self.client = storage.Client( project=self.project_id )
			return self.client
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('create_storage_client( self ) -> storage.Client')
			Logger( ).write( exception )
			raise exception
	
	def create( self, bucket: str, name: str ) -> Blob:
		"""Create an empty object in a bucket.

		Args:
			bucket (str): Bucket name.
			name (str): Object name.

		Returns:
			Blob: Created object reference.

		Raises:
			Error: Raised when validation or object creation fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.client = self.create_storage_client( )
			self.bucket = self.client.bucket( self.bucket_name )
			self.blob = self.bucket.blob( self.object_name )
			self.blob.upload_from_string( b'' )
			self.response = self.blob
			return self.blob
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('create( self, bucket: str, name: str ) -> Blob')
			Logger( ).write( exception )
			raise exception
	
	def upload( self, path: str, bucket: str, name: str = None ) -> Blob:
		"""Upload a local file to a bucket.

		Args:
			path (str): Local file path.
			bucket (str): Bucket name.
			name (str): Optional object name.

		Returns:
			Blob: Uploaded object.

		Raises:
			Error: Raised when validation or upload fails.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'bucket', bucket )
			self.file_path = path
			self.bucket_name = bucket
			self.object_name = (name if name else Path( self.file_path ).name)
			self.client = self.create_storage_client( )
			self.bucket = self.client.bucket( self.bucket_name )
			self.blob = self.bucket.blob( self.object_name )
			self.blob.upload_from_filename( self.file_path )
			self.response = self.blob
			return self.blob
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('upload( self, path: str, bucket: str, '
			                    'name: str ) -> Blob')
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, bucket: str, name: str ) -> Optional[ Blob ]:
		"""Retrieve object metadata.

		Args:
			bucket (str): Bucket name.
			name (str): Object name.

		Returns:
			Optional[Blob]: Matching object, or None when absent.

		Raises:
			Error: Raised when validation or retrieval fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.client = self.create_storage_client( )
			self.bucket = self.client.bucket( self.bucket_name )
			self.response = self.bucket.get_blob( self.object_name )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('retrieve( self, bucket: str, name: str ) '
			                    '-> Optional[ Blob ]')
			Logger( ).write( exception )
			raise exception
	
	def list( self, bucket: str ) -> List[ Blob ]:
		"""List bucket objects.

		Args:
			bucket (str): Bucket name.

		Returns:
			List[Blob]: Bucket objects.

		Raises:
			Error: Raised when validation or listing fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			self.bucket_name = bucket
			self.client = self.create_storage_client( )
			self.bucket = self.client.bucket( self.bucket_name )
			self.blobs = list( self.bucket.list_blobs( ) )
			self.documents = { blob.name: blob.id for blob in self.blobs }
			self.response = self.blobs
			return self.blobs
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('list( self, bucket: str ) -> List[ Blob ]')
			Logger( ).write( exception )
			raise exception
	
	def execute_grounded_interaction( self, prompt: str, model: str, tool_type: str,
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str:
		"""Execute a grounded Interactions request.

		Args:
			prompt (str): User prompt.
			model (str): Gemini model identifier.
			tool_type (str): Grounding tool type.
			temperature (float): Sampling temperature.
			top_p (float): Top-P sampling value.
			frequency (float): Compatibility frequency-penalty value.
			presence (float): Compatibility presence-penalty value.
			max_tokens (int): Maximum output-token count.
			stops (List[str]): Stop sequences.
			instruct (str): Optional system instruction.

		Returns:
			str: Generated grounded response.

		Raises:
			Error: Raised when validation or execution fails.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'tool_type', tool_type )
			self.prompt = prompt
			self.model = model
			self.tool_type = tool_type
			self.temperature = temperature
			self.top_p = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.stops = stops if isinstance( stops, list ) else [ ]
			self.instructions = instruct
			
			if self.tool_type not in ('google_search', 'google_maps'):
				raise ValueError( f'Unsupported grounding tool: {self.tool_type}' )
			
			self.generation_config = { }
			
			if self.temperature is not None:
				self.generation_config[ 'temperature' ] = self.temperature
			
			if self.top_p is not None:
				self.generation_config[ 'top_p' ] = self.top_p
			
			if self.max_tokens is not None and self.max_tokens > 0:
				self.generation_config[ 'max_output_tokens' ] = self.max_tokens
			
			self.stop_sequences = [ str( item ).strip( ) for item in self.stops if
				item is not None and str( item ).strip( ) ]
			
			if self.stop_sequences:
				self.generation_config[ 'stop_sequences' ] = self.stop_sequences
			
			self.api_key = (os.getenv( 'GEMINI_API_KEY' ) or os.getenv(
				'GOOGLE_API_KEY' ) or self.gemini_api_key or self.google_api_key)
			throw_if( 'api_key', self.api_key )
			self.genai_client = genai.Client( api_key=self.api_key,
				http_options=types.HttpOptions( api_version='v1beta' ) )
			self.request = { 'model': self.model, 'input': self.prompt,
				'tools': [ { 'type': self.tool_type } ], 'response_format': { 'type': 'text' },
				'store': False, }
			
			if self.instructions is not None and str( self.instructions ).strip( ):
				self.request[ 'system_instruction' ] = str( self.instructions ).strip( )
			
			if self.generation_config:
				self.request[ 'generation_config' ] = (self.generation_config)
			
			self.interaction = self.genai_client.interactions.create( **self.request )
			self.response = self.interaction
			self.content_response = self.interaction
			self.output_text = str( getattr( self.interaction, 'output_text', '' ) or '' ).strip( )
			throw_if( 'output_text', self.output_text )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('execute_grounded_interaction( self, prompt: str, '
			                    'model: str, tool_type: str, temperature: float, '
			                    'top_p: float, frequency: float, presence: float, '
			                    'max_tokens: int, stops: List[ str ], '
			                    'instruct: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def web_search( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str:
		"""Generate a Google Search-grounded response."""
		return self.execute_grounded_interaction( prompt=prompt, model=model,
			tool_type='google_search', temperature=temperature, top_p=top_p, frequency=frequency,
			presence=presence, max_tokens=max_tokens, stops=stops, instruct=instruct )
	
	def search_maps( self, prompt: str, model: str = 'gemini-2.5-flash-lite',
		temperature: float = None, top_p: float = None, frequency: float = None,
		presence: float = None, max_tokens: int = None, stops: List[ str ] = None,
		instruct: str = None ) -> str:
		"""Generate a Google Maps-grounded response."""
		return self.execute_grounded_interaction( prompt=prompt, model=model,
			tool_type='google_maps', temperature=temperature, top_p=top_p, frequency=frequency,
			presence=presence, max_tokens=max_tokens, stops=stops, instruct=instruct )
	
	def delete( self, bucket: str, name: str ) -> bool:
		"""Delete a bucket object.

		Args:
			bucket (str): Bucket name.
			name (str): Object name.

		Returns:
			bool: True after successful deletion.

		Raises:
			Error: Raised when validation or deletion fails.
		"""
		try:
			throw_if( 'bucket', bucket )
			throw_if( 'name', name )
			self.bucket_name = bucket
			self.object_name = name
			self.client = self.create_storage_client( )
			self.bucket = self.client.bucket( self.bucket_name )
			self.blob = self.bucket.blob( self.object_name )
			self.blob.delete( )
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'gemini'
			exception.cause = 'CloudBuckets'
			exception.method = ('delete( self, bucket: str, name: str ) -> bool')
			Logger( ).write( exception )
			raise exception
