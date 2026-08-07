'''
  ******************************************************************************************
      Assembly:                Buddy
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
from google.genai.types import ListFilesResponse
import config as cfg
from boogr import Error, Logger
import config as cfg
from openai import OpenAI
from xai_sdk.aio.image import ImageResponse
from xai_sdk import Client
from xai_sdk.tools import web_search, x_search, collections_search, code_execution
from xai_sdk.chat import user, system, image, file

def encode_image( image_path: str ) -> str:
	"""Encode image.
	
	Purpose:
	    Performs the encode_image workflow using the inputs supplied by the caller and the current
	    runtime configuration. The function keeps this behavior isolated so related UI, provider,
	    and
	    data-processing paths can call it consistently.
	
	Args:
	    image_path (str): Image path value used by the operation.
	
	Returns:
	    str: Return value produced by the operation."""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

def throw_if( name: str, value: object ) -> None:
	"""Throw if.
	
	Purpose:
	    Validates that a required argument contains a usable value before the surrounding workflow
	    continues. This guard centralizes early validation so provider wrappers and UI routines
	    fail
	    with consistent, readable error messages.
	
	Args:
	    name (str): Name value used by the operation.
	    value (object): Value value used by the operation.
	
	Returns:
	    None: This function performs its work through side effects and does not return a value."""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

class Grok( ):
	"""Grok class.
	
	Purpose:
	    Defines the Grok component used by the Boo application. The class groups related provider
	    configuration, runtime state, helper methods, and API-facing behavior so Streamlit
	    workflows can
	    call a consistent interface.
	
	Attributes:
	    api_key (Optional[str]): Stores api key for the component runtime state.
	    timeout (Optional[float]): Stores timeout for the component runtime state.
	    base_url (Optional[str]): Stores base url for the component runtime state.
	    model (Optional[str]): Stores model for the component runtime state.
	    store_messages (Optional[bool]): Stores store messages for the component runtime state.
	    response_format (Optional[str]): Stores response format for the component runtime state.
	    temperature (Optional[float]): Stores temperature for the component runtime state.
	    top_percent (Optional[float]): Stores top percent for the component runtime state.
	    frequency_penalty (Optional[float]): Stores frequency penalty for the component runtime
	        state.
	    presence_penalty (Optional[float]): Stores presence penalty for the component runtime
	        state.
	    max_output_tokens (Optional[int]): Stores max output tokens for the component runtime
	        state.
	    tool_choice (Optional[str]): Stores tool choice for the component runtime state.
	    tools (Optional[List[str]]): Stores tools for the component runtime state.
	    stops (Optional[List[str]]): Stores stops for the component runtime state.
	    instructions (Optional[str]): Stores instructions for the component runtime state.
	    content (Optional[str]): Stores content for the component runtime state.
	    messages (Optional[List[Dict[str, Any]]]): Stores messages for the component runtime state.
	    stores (Optional[Dict[str, str]]): Stores stores for the component runtime state.
	    files (Optional[Dict[str, str]]): Stores files for the component runtime state."""
	api_key: Optional[ str ]
	timeout: Optional[ float ]
	base_url: Optional[ str ]
	model: Optional[ str ]
	store_messages: Optional[ bool ]
	response_format: Optional[ str ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_output_tokens: Optional[ int ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	instructions: Optional[ str ]
	content: Optional[ str ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	stores: Optional[ Dict[ str, str ] ]
	files: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes the Grok object with its default configuration, runtime state, provider
		    settings,
		    and compatibility fields. This constructor prepares the instance for later method
		    calls without
		    performing external work beyond local attribute assignment."""
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.timeout = None
		self.instructions = None
		self.content = None
		self.store_messages = None
		self.model = None
		self.max_output_tokens = None
		self.temperature = None
		self.top_percent = None
		self.tool_choice = None
		self.tools = [ ]
		self.frequency_penalty = None
		self.presence_penalty = None
		self.response_format = None
		self.messages = [ ]
		self.stops = [ ]
		self.collections = None
		self.files = None

class Chat( Grok ):
	"""Provide xAI Grok text-generation workflow support.
	
	Purpose:
		Provides synchronous and streaming text generation through the xAI Python SDK. The
		class builds provider-native chat, tool, conversation-history, structured-output, and
		reasoning configuration from arguments assigned to object members before executing an
		xAI request.
	
	Attributes:
		client (Optional[Client]): xAI SDK client.
		chat (Any): Provider chat object used by the current request.
		model (str): Grok model used by the current request.
		prompt (str): User prompt used by the current request.
		temperature (float): Sampling temperature.
		top_percent (float): Nucleus-sampling value.
		frequency_penalty (float): Frequency penalty.
		presence_penalty (float): Presence penalty.
		max_output_tokens (int): Maximum output-token count.
		stops (List[str]): Stop sequences.
		store_messages (bool): Indicates whether xAI stores request messages.
		stream (bool): Indicates whether streaming is enabled.
		response_format (Any): Provider structured-output configuration.
		context (List[Dict[str, Any]]): Prior conversation messages.
		instructions (str): Optional system instruction.
		include (List[str]): Optional provider response inclusions.
		tool_choice (str): Provider tool-selection mode.
		previous_id (str): Previous stored response identifier.
		previous_response_id (str): Previous stored response identifier.
		parallel_tools (bool): Indicates whether parallel tool calls are enabled.
		max_tools (int): Maximum server-side tool turns.
		tools (List[Any]): Application-selected tool names or tool definitions.
		tool_objects (List[Any]): Provider-ready tool objects.
		reasoning (str): Requested reasoning-effort level.
		allowed_domains (List[str]): Domains permitted for Web Search.
		vector_store_ids (List[str]): Collection identifiers used by Collections Search.
		output_text (str): Text extracted from the latest response.
		response (Any): Latest xAI response.
		usage (Any): Usage metadata from the latest response.
	"""
	client: Optional[ Client ]
	chat: Any
	model: str
	prompt: str
	temperature: float
	top_percent: float
	frequency_penalty: float
	presence_penalty: float
	max_output_tokens: int
	stops: List[ str ]
	store_messages: bool
	stream: bool
	response_format: Any
	context: List[ Dict[ str, Any ] ]
	instructions: str
	include: List[ str ]
	tool_choice: str
	previous_id: str
	previous_response_id: str
	parallel_tools: bool
	max_tools: int
	tools: List[ Any ]
	tool_objects: List[ Any ]
	reasoning: str
	allowed_domains: List[ str ]
	vector_store_ids: List[ str ]
	output_text: str
	response: Any
	usage: Any
	
	def __init__( self, model: str = 'grok-4.20' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes xAI Grok text-generation configuration and runtime state without
			executing a provider request.
		
		Args:
			model (str): Default Grok text-generation model.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.timeout = 3600
		self.client = None
		self.chat = None
		self.model = model
		self.prompt = ''
		self.temperature = 0.0
		self.top_percent = 0.0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_output_tokens = 0
		self.stops = [ ]
		self.store_messages = False
		self.stream = False
		self.response_format = None
		self.response_schema = None
		self.context = [ ]
		self.instructions = ''
		self.include = [ ]
		self.tool_choice = ''
		self.previous_id = ''
		self.previous_response_id = ''
		self.parallel_tools = False
		self.max_tools = 0
		self.tools = [ ]
		self.tool_objects = [ ]
		self.reasoning = ''
		self.allowed_domains = [ ]
		self.vector_store_ids = [ ]
		self.output_text = ''
		self.response = None
		self.usage = None
		self.chat_values = { }
		self.parts = [ ]
		self.collections = cfg.GROK_COLLECTIONS
		self.files = getattr( cfg, 'GROK_DOCUMENTS', { }, )
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get model options.
		
		Purpose:
			Returns Grok text-generation models exposed by the wrapper.
		
		Returns:
			List[str]: Available Grok model identifiers.
		"""
		return [ 'grok-4.20', 'grok-4.20-reasoning', 'grok-4.20-multi-agent', 'grok-4.5', 'grok-4',
			'grok-4-latest', 'grok-4-fast-reasoning', 'grok-4-fast-non-reasoning',
			'grok-code-fast-1', 'grok-3', 'grok-3-mini', 'grok-3-fast', 'grok-3-mini-fast', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Get include options.
		
		Purpose:
			Returns optional xAI SDK response inclusions exposed by the wrapper.
		
		Returns:
			List[str]: Supported include values.
		"""
		return [ 'verbose_streaming', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Get tool options.
		
		Purpose:
			Returns xAI server-side tools implemented by the wrapper.
		
		Returns:
			List[str]: Supported server-side tool names.
		"""
		return [ 'web_search', 'x_search', 'collections_search', 'code_execution', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Get tool-choice options.
		
		Purpose:
			Returns xAI tool-selection modes exposed by the wrapper.
		
		Returns:
			List[str]: Supported tool-selection values.
		"""
		return [ 'auto', 'required', 'none', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get response-format options.
		
		Purpose:
			Returns response-format selections supported by the wrapper.
		
		Returns:
			List[str]: Supported response-format values.
		"""
		return [ 'text', 'json_object', 'json_schema', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Get reasoning options.
		
		Purpose:
			Returns reasoning-effort values exposed by the wrapper.
		
		Returns:
			List[str]: Supported reasoning-effort values.
		"""
		return [ 'none', 'low', 'medium', 'high', 'xhigh', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Get modality options.
		
		Purpose:
			Returns the response modality supported by the Grok Chat wrapper.
		
		Returns:
			List[str]: Supported response modalities.
		"""
		return [ 'text', ]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Get media options.
		
		Purpose:
			Returns the media-detail selection retained by the application interface.
		
		Returns:
			List[str]: Supported media-detail selections.
		"""
		return [ 'auto', ]
	
	def supports_reasoning_model( self, model: str ) -> bool:
		"""Determine reasoning-model support.
		
		Purpose:
			Determines whether a required Grok model accepts an explicit reasoning-effort
			configuration.
		
		Args:
			model (str): Required Grok model identifier.
		
		Returns:
			bool: True when the model supports explicit reasoning effort.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			return self.model in [ 'grok-4.20-reasoning', 'grok-4.20-multi-agent', 'grok-4.5',
				'grok-4-fast-reasoning', 'grok-3-mini', 'grok-3-mini-fast', ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = ('supports_reasoning_model( self, model: str ) -> bool')
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: Optional[ List[ Any ] ] = None,
		allowed_domains: Optional[ List[ str ] ] = None,
		vector_store_ids: Optional[ List[ str ] ] = None ) -> List[ Any ]:
		"""Build provider tools.
		
		Purpose:
			Builds provider-native Web Search, X Search, Collections Search, and code-execution
			tools from application selections.
		
		Args:
			tools (Optional[List[Any]]): Selected tool names or provider tool objects.
			allowed_domains (Optional[List[str]]): Domains permitted for Web Search.
			vector_store_ids (Optional[List[str]]): Collection identifiers used by Collections
				Search.
		
		Returns:
			List[Any]: Provider-ready xAI tool objects.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.tools = (tools if tools is not None else [ ])
			self.allowed_domains = (allowed_domains if allowed_domains is not None else [ ])
			self.vector_store_ids = (vector_store_ids if vector_store_ids is not None else [ ])
			self.tool_objects = [ ]
			for selected_tool in self.tools:
				if isinstance( selected_tool, str ):
					self.tool_name = selected_tool.strip( )
				elif isinstance( selected_tool, dict ):
					self.tool_name = str( selected_tool.get( 'type', '', ) ).strip( )
				else:
					self.tool_objects.append( selected_tool )
					continue
				
				if not self.tool_name:
					continue
				
				if self.tool_name == 'web_search':
					if self.allowed_domains:
						self.tool_objects.append(
							web_search( allowed_domains=self.allowed_domains, ) )
					else:
						self.tool_objects.append( web_search( ) )
					
					continue
				
				if self.tool_name == 'x_search':
					self.tool_objects.append( x_search( ) )
					continue
				
				if self.tool_name == 'collections_search':
					throw_if( 'vector_store_ids', self.vector_store_ids, )
					self.tool_objects.append(
						collections_search( collection_ids=self.vector_store_ids, ) )
					continue
				
				if self.tool_name == 'code_execution':
					self.tool_objects.append( code_execution( ) )
			
			return self.tool_objects
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def build_response_format( self, format: Any = None, response_schema: Any = None ) -> Any:
		"""Build response format.
		
		Purpose:
			Builds the provider response-format value for plain text, JSON object, or
			schema-constrained output.
		
		Args:
			format (Any): Requested response-format selection or provider configuration.
			response_schema (Any): Optional JSON schema or Pydantic model.
		
		Returns:
			Any: Provider-ready response-format value or None.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.response_format = format
			self.response_schema = response_schema
			if self.response_format is None:
				return None
			
			if not isinstance( self.response_format, str ):
				return self.response_format
			
			self.format_name = self.response_format.strip( ).lower( )
			if not self.format_name:
				return None
			
			if self.format_name == 'text':
				return None
			
			if self.format_name == 'json_object':
				return { 'type': 'json_object', }
			
			if self.format_name == 'json_schema':
				throw_if( 'response_schema', self.response_schema, )
				
				return { 'type': 'json_schema', 'json_schema': self.response_schema, }
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_response_format( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def build_chat( self, model: str, temperature: float = 0.0, top_p: float = 0.0,
		frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 0,
		stops: Optional[ List[ str ] ] = None, store: bool = False,
		include: Optional[ List[ str ] ] = None, tools: Optional[ List[ Any ] ] = None,
		allowed_domains: Optional[ List[ str ] ] = None,
		vector_store_ids: Optional[ List[ str ] ] = None, max_tools: int = 0, tool_choice: str='',
		is_parallel: bool = False, previous_id: str = '', reasoning: str = '', format: Any = None,
		response_schema: Any = None ) -> Any:
		"""Build provider chat.
		
		Purpose:
			Builds the xAI chat object from arguments assigned to object members and
			provider-native tool and response-format configuration.
		
		Args:
			model (str): Required Grok model identifier.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			stops (Optional[List[str]]): Stop sequences.
			store (bool): Indicates whether xAI stores request messages.
			include (Optional[List[str]]): Optional provider response inclusions.
			tools (Optional[List[Any]]): Selected tools.
			allowed_domains (Optional[List[str]]): Domains permitted for Web Search.
			vector_store_ids (Optional[List[str]]): Collection identifiers used by Collections
				Search.
			max_tools (int): Maximum server-side tool turns.
			tool_choice (str): Tool-selection mode.
			is_parallel (bool): Indicates whether parallel tool calls are enabled.
			previous_id (str): Previous stored response identifier.
			reasoning (str): Reasoning-effort level.
			format (Any): Response-format selection or provider configuration.
			response_schema (Any): Optional structured-output schema.
		
		Returns:
			Any: Configured xAI chat object.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.stops = (stops if stops is not None else [ ])
			self.store_messages = store
			self.include = (include if include is not None else [ ])
			self.max_tools = max_tools
			self.tool_choice = tool_choice
			self.parallel_tools = is_parallel
			self.previous_id = previous_id
			self.previous_response_id = previous_id
			self.reasoning = reasoning.strip( ).lower( )
			self.response_format = self.build_response_format( format, response_schema, )
			self.tool_objects = self.build_tools( tools, allowed_domains, vector_store_ids, )
			self.chat_values = { 'model': self.model, 'store_messages': self.store_messages, }
			if self.temperature > 0:
				self.chat_values[ 'temperature' ] = (self.temperature)
			
			if self.top_percent > 0:
				self.chat_values[ 'top_p' ] = (self.top_percent)
			
			if self.frequency_penalty != 0:
				self.chat_values[ 'frequency_penalty' ] = (self.frequency_penalty)
			
			if self.presence_penalty != 0:
				self.chat_values[ 'presence_penalty' ] = (self.presence_penalty)
			
			if self.max_output_tokens > 0:
				self.chat_values[ 'max_tokens' ] = (self.max_output_tokens)
			
			if self.stops:
				self.chat_values[ 'stop' ] = self.stops
			
			if self.include:
				self.chat_values[ 'include' ] = self.include
			
			if self.tool_objects:
				self.chat_values[ 'tools' ] = (self.tool_objects)
			
			if self.max_tools > 0:
				self.chat_values[ 'max_turns' ] = (self.max_tools)
			
			if self.tool_choice:
				self.chat_values[ 'tool_choice' ] = (self.tool_choice)
			
			if self.parallel_tools:
				self.chat_values[ 'parallel_tool_calls' ] = (self.parallel_tools)
			
			if self.previous_response_id:
				self.chat_values[ 'previous_response_id' ] = (self.previous_response_id)
			
			if self.reasoning:
				if self.reasoning != 'none':
					if self.supports_reasoning_model( self.model ):
						self.chat_values[ 'reasoning_effort' ] = (self.reasoning)
			
			if self.response_format is not None:
				self.chat_values[ 'response_format' ] = (self.response_format)
			
			self.client = Client( api_key=self.api_key, timeout=self.timeout, )
			self.chat = self.client.chat.create( **self.chat_values )
			return self.chat
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_chat( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def append_context( self, context: Optional[ List[ Dict[ str, Any ] ] ] = None ) -> None:
		"""Append conversation context.
		
		Purpose:
			Appends valid system, user, and assistant history messages to the current xAI chat.
		
		Args:
			context (Optional[List[Dict[str, Any]]]): Prior conversation messages.
		
		Returns:
			None: This method updates the current chat.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			from xai_sdk.chat import assistant
			self.context = (context if context is not None else [ ])
			for item in self.context:
				if not isinstance( item, dict ):
					continue
				
				self.role = str( item.get( 'role', '', ) ).strip( ).lower( )
				self.message_content = str( item.get( 'content', '', ) ).strip( )
				if not self.message_content:
					continue
				
				if self.role == 'system':
					self.chat.append( system( self.message_content ) )
					continue
				
				if self.role == 'user':
					self.chat.append( user( self.message_content ) )
					continue
				
				if self.role == 'assistant':
					self.chat.append( assistant( self.message_content ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'append_context( self, **kwargs ) -> None'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get output text.
		
		Purpose:
			Extracts generated text from the latest xAI response.
		
		Returns:
			str: Generated response text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_text = ''
			if self.response is None:
				return self.output_text
			
			self.response_content = getattr( self.response, 'content', '', )
			if self.response_content:
				self.output_text = str( self.response_content ).strip( )
			
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'get_output_text( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str, temperature: float = 0.0, format: Any = None,
		top_p: float = 0.0, frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 0,
		stops: Optional[ List[ str ] ] = None, store: bool = False, stream: bool = False,
		instruct: str = '', reasoning: str = '', include: Optional[ List[ str ] ] = None,
		tools: Optional[ List[ Any ] ] = None, allowed_domains: Optional[ List[ str ] ] = None,
		previous_id: str = '', tool_choice: str = '', is_parallel: bool = False,
		context: Optional[ List[ Dict[ str, Any ] ] ] = None,
		vector_store_ids: Optional[ List[ str ] ] = None, max_tools: int = 0,
		response_schema: Any = None, stream_handler: Any = None ) -> str:
		"""Generate text.
		
		Purpose:
			Executes synchronous or streaming Grok text generation using a required prompt,
			required model, optional conversation history, provider-native server-side tools,
			stored-response continuation, structured output, and reasoning configuration.
		
		Args:
			prompt (str): Required user prompt.
			model (str): Required Grok model identifier.
			temperature (float): Sampling temperature.
			format (Any): Response-format selection or provider configuration.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			stops (Optional[List[str]]): Stop sequences.
			store (bool): Indicates whether xAI stores request messages.
			stream (bool): Indicates whether streaming is enabled.
			instruct (str): Optional system instruction.
			reasoning (str): Optional reasoning-effort level.
			include (Optional[List[str]]): Optional provider response inclusions.
			tools (Optional[List[Any]]): Selected server-side tools.
			allowed_domains (Optional[List[str]]): Domains permitted for Web Search.
			previous_id (str): Previous stored response identifier.
			tool_choice (str): Tool-selection mode.
			is_parallel (bool): Indicates whether parallel tool calls are enabled.
			context (Optional[List[Dict[str, Any]]]): Prior conversation messages.
			vector_store_ids (Optional[List[str]]): Collection identifiers used by Collections
				Search.
			max_tools (int): Maximum server-side tool turns.
			response_schema (Any): Optional structured-output schema.
			stream_handler (Any): Optional callable receiving each streaming text delta.
		
		Returns:
			str: Generated response text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.prompt = prompt
			self.model = model
			self.temperature = temperature
			self.response_format = format
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.stops = (stops if stops is not None else [ ])
			self.store_messages = store
			self.stream = stream
			self.instructions = instruct
			self.reasoning = reasoning
			self.include = (include if include is not None else [ ])
			self.tools = (tools if tools is not None else [ ])
			self.allowed_domains = (allowed_domains if allowed_domains is not None else [ ])
			self.previous_id = previous_id
			self.previous_response_id = previous_id
			self.tool_choice = tool_choice
			self.parallel_tools = is_parallel
			self.context = (context if context is not None else [ ])
			self.vector_store_ids = (vector_store_ids if vector_store_ids is not None else [ ])
			self.max_tools = max_tools
			self.response_schema = response_schema
			self.stream_handler = stream_handler
			self.chat = self.build_chat( self.model, self.temperature, self.top_percent,
				self.frequency_penalty, self.presence_penalty, self.max_output_tokens, self.stops,
				self.store_messages, self.include, self.tools, self.allowed_domains,
				self.vector_store_ids, self.max_tools, self.tool_choice, self.parallel_tools,
				self.previous_response_id, self.reasoning, self.response_format,
				self.response_schema, )
			
			if self.instructions:
				self.chat.append( system( self.instructions ) )
			
			self.append_context( self.context )
			self.chat.append( user( self.prompt ) )
			if self.stream:
				self.parts = [ ]
				self.response = None
				for response, chunk in self.chat.stream( ):
					self.response = response
					self.chunk_content = getattr( chunk, 'content', '', )
					
					if not self.chunk_content:
						continue
					
					self.chunk_content = str( self.chunk_content )
					self.parts.append( self.chunk_content )
					if self.stream_handler is not None:
						self.stream_handler( self.chunk_content )
				
				self.output_text = ''.join( self.parts ).strip( )
				if not self.output_text:
					self.output_text = self.get_output_text( )
				
				if self.response is not None:
					self.previous_id = str( getattr( self.response, 'id', '', ) or '' )
					self.previous_response_id = self.previous_id
				
				return self.output_text
			
			self.response = self.chat.sample( )
			self.previous_id = str( getattr( self.response, 'id', '', ) or '' )
			self.previous_response_id = self.previous_id
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_usage( self ) -> Any:
		"""Get usage.
		
		Purpose:
			Returns usage metadata from the latest xAI response.
		
		Returns:
			Any: Provider usage metadata or None when unavailable.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			if self.response is None:
				return None
			
			self.usage = getattr( self.response, 'usage', None, )
			return self.usage
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'get_usage( self ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Grok Chat wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'base_url', 'timeout', 'client', 'chat', 'model', 'prompt',
			'temperature', 'top_percent', 'frequency_penalty', 'presence_penalty',
			'max_output_tokens', 'stops', 'store_messages', 'stream', 'response_format',
			'response_schema', 'context', 'instructions', 'include', 'tool_choice', 'previous_id',
			'previous_response_id', 'parallel_tools', 'max_tools', 'tools', 'tool_objects',
			'reasoning', 'allowed_domains', 'vector_store_ids', 'output_text', 'response', 'usage',
			'collections', 'files', 'model_options', 'include_options', 'tool_options',
			'choice_options', 'format_options', 'reasoning_options', 'modality_options',
			'media_options', 'supports_reasoning_model', 'build_tools', 'build_response_format',
			'build_chat', 'append_context', 'get_output_text', 'generate_text', 'get_usage', ]

class Images( Grok ):
	"""Provide Grok image workflow support.
	
	Purpose:
		Provides image generation, image editing, and image analysis through the xAI Python
		SDK. The class assigns accepted arguments to object members, constructs provider-native
		image or multimodal-chat requests, executes the selected operation, and extracts image
		URLs, decoded image bytes, or analysis text from provider responses.
	
	Attributes:
		client (Optional[Client]): xAI SDK client.
		chat (Any): Multimodal chat used for image analysis.
		model (str): Grok model used by the current operation.
		prompt (str): Prompt used by the current operation.
		number (int): Number of images requested.
		aspect_ratio (str): Requested output-image aspect ratio.
		image_path (str): Local source-image path.
		image_url (str): Provider-ready source-image URL or data URI.
		detail (str): Image-detail level used for analysis.
		response (Any): Latest xAI response.
		output (Any): Extracted image output or collection of outputs.
		output_text (str): Text extracted from an image-analysis response.
	"""
	client: Optional[ Client ]
	chat: Any
	model: str
	prompt: str
	number: int
	aspect_ratio: str
	image_path: str
	image_url: str
	detail: str
	response: Any
	output: Any
	output_text: str
	
	def __init__( self, model: str = 'grok-imagine-image-quality' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes Grok image configuration and runtime state without executing a provider
			request.
		
		Args:
			model (str): Default Grok image-generation model.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.timeout = 3600
		self.client = None
		self.chat = None
		self.model = model
		self.prompt = ''
		self.number = 1
		self.aspect_ratio = 'auto'
		self.image_path = ''
		self.image_url = ''
		self.file_path = ''
		self.detail = 'auto'
		self.response = None
		self.output = None
		self.output_text = ''
		self.outputs = [ ]
		self.encoded_image = ''
		self.mime_type = ''
		self.image_data = b''
		self.response_content = ''
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get image-generation model options.
		
		Purpose:
			Returns Grok models exposed for image generation and editing.
		
		Returns:
			List[str]: Supported Grok image model identifiers.
		"""
		return [ 'grok-imagine-image-quality', 'grok-imagine-image', ]
	
	@property
	def analysis_model_options( self ) -> List[ str ]:
		"""Get image-analysis model options.
		
		Purpose:
			Returns Grok multimodal models exposed for image understanding.
		
		Returns:
			List[str]: Supported Grok image-analysis model identifiers.
		"""
		return [ 'grok-4.20-reasoning', 'grok-4.20', 'grok-4.5', 'grok-4', 'grok-4-latest',
			'grok-4-fast-reasoning', 'grok-4-fast-non-reasoning', 'grok-3', 'grok-3-mini',
			'grok-3-fast', 'grok-3-mini-fast', ]
	
	@property
	def aspect_options( self ) -> List[ str ]:
		"""Get aspect-ratio options.
		
		Purpose:
			Returns output-image aspect ratios exposed by the wrapper.
		
		Returns:
			List[str]: Supported aspect-ratio values.
		"""
		return [ 'auto', '1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3', '2:1', '1:2', '19.5:9',
			'9:19.5', '20:9', '9:20', ]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Get image-size options.
		
		Purpose:
			Returns an automatic size selection because the xAI Python SDK image request does
			not expose an independent pixel-size parameter.
		
		Returns:
			List[str]: Available image-size selections.
		"""
		return [ 'auto', ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		"""Get image-quality options.
		
		Purpose:
			Returns model-based quality selections exposed by the application.
		
		Returns:
			List[str]: Available image-quality selections.
		"""
		return [ 'auto', 'quality', ]
	
	@property
	def style_options( self ) -> List[ str ]:
		"""Get image-style options.
		
		Purpose:
			Returns an empty collection because xAI image style is controlled through the
			prompt rather than a separate request argument.
		
		Returns:
			List[str]: Empty style-option collection.
		"""
		return [ ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get response-format options.
		
		Purpose:
			Returns image response representations that may be returned by the xAI SDK.
		
		Returns:
			List[str]: Supported response representations.
		"""
		return [ 'url', 'b64_json', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get source-image MIME-type options.
		
		Purpose:
			Returns MIME types supported for local source images converted to data URIs.
		
		Returns:
			List[str]: Supported source-image MIME types.
		"""
		return [ 'image/jpeg', 'image/png', 'image/webp', ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		"""Get image-detail options.
		
		Purpose:
			Returns multimodal image-detail levels exposed for image analysis.
		
		Returns:
			List[str]: Supported image-detail values.
		"""
		return [ 'auto', 'low', 'high', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Get response-modality options.
		
		Purpose:
			Returns image and text modalities represented by the wrapper operations.
		
		Returns:
			List[str]: Supported modality values.
		"""
		return [ 'image', 'text', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Get include options.
		
		Purpose:
			Returns an empty collection because xAI image operations do not use response include
			paths.
		
		Returns:
			List[str]: Empty include-option collection.
		"""
		return [ ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Get tool options.
		
		Purpose:
			Returns an empty collection because direct xAI image operations do not use chat
			server-side tools.
		
		Returns:
			List[str]: Empty tool-option collection.
		"""
		return [ ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Get tool-choice options.
		
		Purpose:
			Returns an empty collection because direct xAI image operations do not use tool
			selection.
		
		Returns:
			List[str]: Empty tool-choice collection.
		"""
		return [ ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Get reasoning options.
		
		Purpose:
			Returns an empty collection because image-generation reasoning is managed by the
			selected image model.
		
		Returns:
			List[str]: Empty reasoning-option collection.
		"""
		return [ ]
	
	def get_mime_type( self, path: str ) -> str:
		"""Get image MIME type.
		
		Purpose:
			Determines the MIME type of a required local source image from its file extension.
		
		Args:
			path (str): Required local image path.
		
		Returns:
			str: Source-image MIME type.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			self.image_path = path
			self.suffix = Path( self.image_path ).suffix.lower( )
			if self.suffix in [ '.jpg', '.jpeg', ]:
				self.mime_type = 'image/jpeg'
			elif self.suffix == '.png':
				self.mime_type = 'image/png'
			elif self.suffix == '.webp':
				self.mime_type = 'image/webp'
			else:
				self.mime_type = ''
			
			throw_if( 'mime_type', self.mime_type )
			return self.mime_type
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Images'
			exception.method = 'get_mime_type( self, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def build_image_url( self, path: str ) -> str:
		"""Build source-image data URI.
		
		Purpose:
			Reads a required local source image and converts it into a provider-ready base64
			data URI.
		
		Args:
			path (str): Required local image path.
		
		Returns:
			str: Provider-ready image data URI.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			self.image_path = path
			self.file_path = path
			self.mime_type = self.get_mime_type( self.image_path )
			with open( self.image_path, 'rb' ) as source:
				self.image_data = source.read( )
			
			throw_if( 'image_data', self.image_data )
			self.encoded_image = base64.b64encode( self.image_data ).decode( 'utf-8' )
			self.image_url = (f'data:{self.mime_type};base64,'
			                  f'{self.encoded_image}')
			return self.image_url
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Images'
			exception.method = 'build_image_url( self, path: str ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def extract_image_output( self, response: Any ) -> Any:
		"""Extract image output.
		
		Purpose:
			Extracts an image URL or decoded image bytes from a required xAI image response.
		
		Args:
			response (Any): Required xAI image response.
		
		Returns:
			Any: Image URL, decoded image bytes, or the original provider response.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response', response )
			self.response = response
			self.response_url = getattr( self.response, 'url', '', )
			if self.response_url:
				self.output = self.response_url
				return self.output
			
			self.response_base64 = getattr( self.response, 'b64_json', '', )
			if self.response_base64:
				self.output = base64.b64decode( self.response_base64 )
				return self.output
			
			self.output = self.response
			return self.output
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Images'
			exception.method = ('extract_image_output( self, response: Any ) -> Any')
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, model: str, number: int = 1,
		aspect_ratio: str = 'auto' ) -> Any:
		"""Generate images.
		
		Purpose:
			Generates one or more images from a required prompt using a required Grok image
			model and optional output aspect ratio.
		
		Args:
			prompt (str): Required image-generation prompt.
			model (str): Required Grok image model.
			number (int): Number of images requested.
			aspect_ratio (str): Output-image aspect ratio.
		
		Returns:
			Any: Generated image output or collection of image outputs.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.aspect_ratio = aspect_ratio
			self.client = Client( api_key=self.api_key, timeout=self.timeout, )
			if self.number > 1:
				if self.aspect_ratio:
					if self.aspect_ratio != 'auto':
						self.response = self.client.image.sample_batch( prompt=self.prompt,
							model=self.model, n=self.number, aspect_ratio=self.aspect_ratio, )
					else:
						self.response = self.client.image.sample_batch( prompt=self.prompt,
							model=self.model, n=self.number, )
				else:
					self.response = self.client.image.sample_batch( prompt=self.prompt,
						model=self.model, n=self.number, )
				
				self.outputs = [ ]
				for item in self.response:
					self.outputs.append( self.extract_image_output( item ) )
				
				self.output = self.outputs
				return self.output
			
			if self.aspect_ratio:
				if self.aspect_ratio != 'auto':
					self.response = self.client.image.sample( prompt=self.prompt, model=self.model,
						aspect_ratio=self.aspect_ratio, )
				else:
					self.response = self.client.image.sample( prompt=self.prompt,
						model=self.model, )
			else:
				self.response = self.client.image.sample( prompt=self.prompt, model=self.model, )
			
			return self.extract_image_output( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Images'
			exception.method = 'generate( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, prompt: str, path: str, model: str, detail: str = 'auto' ) -> str:
		"""Analyze an image.
		
		Purpose:
			Analyzes a required local image using a required Grok multimodal model and returns
			the generated textual analysis.
		
		Args:
			prompt (str): Required image-analysis prompt.
			path (str): Required local image path.
			model (str): Required Grok multimodal model.
			detail (str): Image-detail level.
		
		Returns:
			str: Generated image-analysis text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.prompt = prompt
			self.image_path = path
			self.file_path = path
			self.model = model
			self.detail = detail
			self.image_url = self.build_image_url( self.image_path )
			self.client = Client( api_key=self.api_key, timeout=self.timeout, )
			self.chat = self.client.chat.create( model=self.model, )
			self.chat.append( user( self.prompt, image( self.image_url, detail=self.detail, ), ) )
			self.response = self.chat.sample( )
			self.response_content = getattr( self.response, 'content', '', )
			self.output_text = str( self.response_content or '' ).strip( )
			throw_if( 'output_text', self.output_text )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Images'
			exception.method = 'analyze( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, model: str, path: str = '', image_url: str = '',
		aspect_ratio: str = 'auto', number: int = 1 ) -> Any:
		"""Edit an image.
		
		Purpose:
			Edits a required source image using a required prompt and Grok image model. The
			source may be supplied as a local path or public image URL.
		
		Args:
			prompt (str): Required image-editing instruction.
			model (str): Required Grok image model.
			path (str): Optional local source-image path.
			image_url (str): Optional public source-image URL.
			aspect_ratio (str): Optional output-image aspect ratio.
			number (int): Number of edited images requested.
		
		Returns:
			Any: Edited image output or collection of edited-image outputs.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.prompt = prompt
			self.model = model
			self.image_path = path
			self.file_path = path
			self.image_url = image_url
			self.aspect_ratio = aspect_ratio
			self.number = number
			if not self.image_url:
				throw_if( 'path', self.image_path )
				self.image_url = self.build_image_url( self.image_path )
			
			throw_if( 'image_url', self.image_url )
			self.client = Client( api_key=self.api_key, timeout=self.timeout, )
			if self.number > 1:
				if self.aspect_ratio:
					if self.aspect_ratio != 'auto':
						self.response = self.client.image.sample_batch( prompt=self.prompt,
							model=self.model, n=self.number, image_url=self.image_url,
							aspect_ratio=self.aspect_ratio, )
					else:
						self.response = self.client.image.sample_batch( prompt=self.prompt,
							model=self.model, n=self.number, image_url=self.image_url, )
				else:
					self.response = self.client.image.sample_batch( prompt=self.prompt,
						model=self.model, n=self.number, image_url=self.image_url, )
				
				self.outputs = [ ]
				for item in self.response:
					self.outputs.append( self.extract_image_output( item ) )
				
				self.output = self.outputs
				return self.output
			
			if self.aspect_ratio:
				if self.aspect_ratio != 'auto':
					self.response = self.client.image.sample( prompt=self.prompt, model=self.model,
						image_url=self.image_url, aspect_ratio=self.aspect_ratio, )
				else:
					self.response = self.client.image.sample( prompt=self.prompt, model=self.model,
						image_url=self.image_url, )
			else:
				self.response = self.client.image.sample( prompt=self.prompt, model=self.model,
					image_url=self.image_url, )
			
			return self.extract_image_output( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Images'
			exception.method = 'edit( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Grok Images wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'base_url', 'timeout', 'client', 'chat', 'model', 'prompt', 'number',
			'aspect_ratio', 'image_path', 'image_url', 'file_path', 'detail', 'response', 'output',
			'outputs', 'output_text', 'mime_type', 'model_options', 'analysis_model_options',
			'aspect_options', 'size_options', 'quality_options', 'style_options', 'format_options',
			'mime_options', 'detail_options', 'modality_options', 'include_options',
			'tool_options',
			'choice_options', 'reasoning_options', 'get_mime_type', 'build_image_url',
			'extract_image_output', 'generate', 'analyze', 'edit', ]

class Files( Grok ):
	"""Provide xAI file-management and file-analysis workflows.
	
	Purpose:
		Provides file upload, listing, retrieval, content download, deletion, summarization,
		question answering, and file surveying through the xAI Files and Chat APIs. The class
		assigns accepted arguments to object members before constructing provider requests and
		returns stable application-facing metadata, content, and generated text.
	
	Attributes:
		client (Optional[Client]): xAI SDK client used for file-enabled chat requests.
		api_key (str): xAI API key.
		base_url (str): xAI REST API base URL.
		file_path (str): Local file path used by the current operation.
		file_name (str): Filename assigned during upload.
		file_id (str): xAI file identifier used by the current operation.
		file_ids (List[str]): File identifiers retained by the wrapper.
		purpose (str): Compatibility purpose value stored with uploaded files.
		expires_after (int): Optional file-expiration duration in seconds.
		model (str): Grok model used for file analysis.
		prompt (str): Prompt used by the current file-analysis request.
		instructions (str): Optional system instruction.
		temperature (float): Sampling temperature.
		top_percent (float): Nucleus-sampling value.
		frequency_penalty (float): Frequency penalty.
		presence_penalty (float): Presence penalty.
		max_output_tokens (int): Maximum output-token count.
		store_messages (bool): Indicates whether xAI stores chat messages.
		stream (bool): Indicates whether response streaming is enabled.
		include (List[str]): Optional xAI streaming inclusions.
		previous_id (str): Previous stored response identifier.
		response (Any): Latest provider response.
		file_content (Any): Latest downloaded file content.
		output_text (str): Text extracted from the latest file-analysis response.
		limit (int): Maximum number of files requested by a list operation.
		pagination_token (str): Pagination token supplied to a list operation.
		next_token (str): Pagination token returned by the latest list operation.
		documents (Dict[str, str]): Configured document labels mapped to file identifiers.
	"""
	client: Optional[ Client ]
	api_key: str
	base_url: str
	file_path: str
	file_name: str
	file_id: str
	file_ids: List[ str ]
	purpose: str
	expires_after: int
	model: str
	prompt: str
	instructions: str
	temperature: float
	top_percent: float
	frequency_penalty: float
	presence_penalty: float
	max_output_tokens: int
	store_messages: bool
	stream: bool
	include: List[ str ]
	previous_id: str
	response: Any
	file_content: Any
	output_text: str
	limit: int
	pagination_token: str
	next_token: str
	documents: Dict[ str, str ]
	
	def __init__( self, model: str = 'grok-4.20' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes xAI file-management and file-analysis state without executing a
			provider request.
		
		Args:
			model (str): Default Grok model used for file analysis.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1', )
		self.timeout = 3600
		self.client = None
		self.chat = None
		self.file_path = ''
		self.file_name = ''
		self.file_id = ''
		self.file_ids = [ ]
		self.purpose = 'assistants'
		self.expires_after = 0
		self.model = model
		self.prompt = ''
		self.instructions = ''
		self.temperature = 0.0
		self.top_percent = 0.0
		self.frequency_penalty = 0.0
		self.presence_penalty = 0.0
		self.max_output_tokens = 0
		self.store_messages = False
		self.stream = False
		self.include = [ ]
		self.previous_id = ''
		self.previous_response_id = ''
		self.response = None
		self.file_content = None
		self.output_text = ''
		self.limit = 100
		self.pagination_token = ''
		self.next_token = ''
		self.download_format = ''
		self.params = { }
		self.headers = { }
		self.metadata = { }
		self.results = [ ]
		self.parts = [ ]
		self.documents = getattr( cfg, 'GROK_DOCUMENTS', { }, )
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get file-analysis model options.
		
		Purpose:
			Returns Grok models exposed for file summarization and question answering.
		
		Returns:
			List[str]: Supported Grok model identifiers.
		"""
		return [ 'grok-4.20-reasoning', 'grok-4.20', 'grok-4.5', 'grok-4', 'grok-4-latest',
			'grok-4-fast-reasoning', 'grok-4-fast-non-reasoning', 'grok-code-fast-1', 'grok-3',
			'grok-3-mini', 'grok-3-fast', 'grok-3-mini-fast', ]
	
	@property
	def purpose_options( self ) -> List[ str ]:
		"""Get file-purpose options.
		
		Purpose:
			Returns compatibility purpose values accepted and stored by the xAI Files API.
		
		Returns:
			List[str]: Available file-purpose values.
		"""
		return [ 'assistants', 'batch', 'fine-tune', 'user_data', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get output-format options.
		
		Purpose:
			Returns the textual output format implemented by file-analysis workflows.
		
		Returns:
			List[str]: Supported output formats.
		"""
		return [ 'text', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Get tool options.
		
		Purpose:
			Returns server-side tools that may be used with file-enabled chat requests.
		
		Returns:
			List[str]: Supported server-side tool names.
		"""
		return [ 'code_execution', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Get include options.
		
		Purpose:
			Returns optional xAI streaming response inclusions.
		
		Returns:
			List[str]: Supported include values.
		"""
		return [ 'verbose_streaming', ]
	
	def get_headers( self ) -> Dict[ str, str ]:
		"""Get request headers.
		
		Purpose:
			Builds authentication headers for xAI Files API requests.
		
		Returns:
			Dict[str, str]: Provider request headers.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'XAI_API_KEY', self.api_key )
			self.headers = { 'Authorization': f'Bearer {self.api_key}', }
			return self.headers
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = ('get_headers( self ) -> Dict[ str, str ]')
			Logger( ).write( exception )
			raise exception
	
	def normalize_metadata( self, file_data: Dict[ str, Any ] ) -> Dict[ str, Any ]:
		"""Normalize file metadata.
		
		Purpose:
			Converts a required xAI file response into a stable application-facing metadata
			record.
		
		Args:
			file_data (Dict[str, Any]): Required provider file metadata.
		
		Returns:
			Dict[str, Any]: Application-facing file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_data', file_data )
			self.metadata = file_data
			self.file_id = str( self.metadata.get( 'id', '', ) or '' )
			self.file_name = str(
				self.metadata.get( 'filename', self.metadata.get( 'name', '', ), ) or '' )
			
			return { 'id': self.file_id, 'name': self.file_name, 'filename': self.file_name,
				'object': self.metadata.get( 'object', 'file', ),
				'purpose': self.metadata.get( 'purpose', '', ),
				'bytes': self.metadata.get( 'bytes', self.metadata.get( 'size_bytes', 0, ), ),
				'created_at': self.metadata.get( 'created_at', None, ),
				'expires_at': self.metadata.get( 'expires_at', None, ),
				'content_type': self.metadata.get( 'content_type', '', ),
				'public_url': self.metadata.get( 'public_url', '', ),
				'public_url_expires_at': self.metadata.get( 'public_url_expires_at', None, ),
				'metadata': self.metadata, }
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'normalize_metadata( self, **kwargs ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def upload( self, file_path: str, file_name: str = '', purpose: str = 'assistants',
		expires_after: int = 0 ) -> Dict[ str, Any ]:
		"""Upload a file.
		
		Purpose:
			Uploads a required local file to xAI storage with an optional filename,
			compatibility purpose, and expiration duration.
		
		Args:
			file_path (str): Required local file path.
			file_name (str): Optional uploaded filename.
			purpose (str): Compatibility purpose value stored with the file.
			expires_after (int): Optional expiration duration in seconds.
		
		Returns:
			Dict[str, Any]: Uploaded file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_path', file_path )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.file_path = file_path
			self.file_name = (file_name.strip( ) if file_name else Path( self.file_path ).name)
			self.purpose = purpose
			self.expires_after = expires_after
			self.headers = self.get_headers( )
			self.params = { 'purpose': self.purpose, }
			if self.expires_after > 0:
				self.params[ 'expires_after' ] = str( self.expires_after )
			
			with open( self.file_path, 'rb' ) as source:
				self.response = requests.post( url=(f'{self.base_url.rstrip( "/" )}'
				                                    f'/files'), headers=self.headers,
					data=self.params, files={ 'file': (self.file_name, source,), },
					timeout=self.timeout, )
			
			self.response.raise_for_status( )
			self.metadata = self.response.json( )
			return self.normalize_metadata( self.metadata )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'upload( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list( self, limit: int = 100, pagination_token: str = '' ) -> List[ Dict[ str, Any ] ]:
		"""List files.
		
		Purpose:
			Lists uploaded xAI files using an optional result limit and pagination token.
		
		Args:
			limit (int): Maximum number of files requested.
			pagination_token (str): Optional pagination token.
		
		Returns:
			List[Dict[str, Any]]: Application-facing file metadata records.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'XAI_API_KEY', self.api_key )
			self.limit = limit
			self.pagination_token = pagination_token
			self.headers = self.get_headers( )
			self.params = { 'limit': self.limit, }
			if self.pagination_token:
				self.params[ 'pagination_token' ] = (self.pagination_token)
			
			self.response = requests.get( url=(f'{self.base_url.rstrip( "/" )}/files'),
				headers=self.headers, params=self.params, timeout=self.timeout, )
			self.response.raise_for_status( )
			self.payload = self.response.json( )
			self.file_data = self.payload.get( 'data', [ ], )
			self.next_token = str(
				self.payload.get( 'next_page', self.payload.get( 'next_token', '', ), ) or '' )
			self.results = [ self.normalize_metadata( item ) for item in self.file_data ]
			self.file_ids = [ item.get( 'id', '', ) for item in self.results if item.get( 'id', '', ) ]
			return self.results
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'list( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list_files( self, limit: int=100, pagination_token: str='' ) -> List[ Dict[ str, Any ] ]:
		"""List files.
		
		Purpose:
			Provides the application-compatible alias for xAI file listing.
		
		Args:
			limit (int): Maximum number of files requested.
			pagination_token (str): Optional pagination token.
		
		Returns:
			List[Dict[str, Any]]: Application-facing file metadata records.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.limit = limit
			self.pagination_token = pagination_token
			return self.list( self.limit, self.pagination_token, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'list_files( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, file_id: str ) -> Dict[ str, Any ]:
		"""Retrieve file metadata.
		
		Purpose:
			Retrieves metadata for a required xAI file identifier.
		
		Args:
			file_id (str): Required xAI file identifier.
		
		Returns:
			Dict[str, Any]: Application-facing file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.file_id = file_id
			self.headers = self.get_headers( )
			self.response = requests.get( url=(f'{self.base_url.rstrip( "/" )}/files/{self.file_id}'),
				headers=self.headers, timeout=self.timeout, )
			self.response.raise_for_status( )
			self.metadata = self.response.json( )
			return self.normalize_metadata( self.metadata )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'retrieve( self, file_id: str ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def extract( self, file_id: str ) -> bytes:
		"""Download file content.
		
		Purpose:
			Downloads the original content of a required xAI file.
		
		Args:
			file_id (str): Required xAI file identifier.
		
		Returns:
			bytes: Downloaded file content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.file_id = file_id
			self.headers = self.get_headers( )
			self.response = requests.get( url=(f'{self.base_url.rstrip( "/" )}/files/{self.file_id}/content'),
				headers=self.headers, timeout=self.timeout, )
			self.response.raise_for_status( )
			self.file_content = self.response.content
			return self.file_content
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'extract( self, file_id: str ) -> bytes'
			Logger( ).write( exception )
			raise exception
	
	def download( self, file_id: str ) -> bytes:
		"""Download file content.
		
		Purpose:
			Provides the application-compatible alias for xAI file-content download.
		
		Args:
			file_id (str): Required xAI file identifier.
		
		Returns:
			bytes: Downloaded file content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.file_id = file_id
			return self.extract( self.file_id )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'download( self, file_id: str ) -> bytes'
			Logger( ).write( exception )
			raise exception
	
	def content( self, file_id: str ) -> bytes:
		"""Get file content.
		
		Purpose:
			Provides the application-compatible alias for xAI file-content retrieval.
		
		Args:
			file_id (str): Required xAI file identifier.
		
		Returns:
			bytes: Downloaded file content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.file_id = file_id
			return self.extract( self.file_id )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = ('content( self, file_id: str ) -> bytes')
			Logger( ).write( exception )
			raise exception
	
	def delete( self, file_id: str ) -> Dict[ str, Any ]:
		"""Delete a file.
		
		Purpose:
			Deletes a required file from xAI storage.
		
		Args:
			file_id (str): Required xAI file identifier.
		
		Returns:
			Dict[str, Any]: File-deletion result.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.file_id = file_id
			self.headers = self.get_headers( )
			self.response = requests.delete( url=(f'{self.base_url.rstrip( "/" )}/files/{self.file_id}'),
				headers=self.headers, timeout=self.timeout, )
			self.response.raise_for_status( )
			if self.response.content:
				self.metadata = self.response.json( )
			else:
				self.metadata = { 'id': self.file_id, 'deleted': True, 'object': 'file.deleted', }
			
			return { 'id': self.metadata.get( 'id', self.file_id, ),
				'deleted': self.metadata.get( 'deleted', True, ),
				'object': self.metadata.get( 'object', 'file.deleted', ),
				'metadata': self.metadata, }
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'delete( self, file_id: str ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get output text.
		
		Purpose:
			Extracts generated text from the latest xAI file-analysis response.
		
		Returns:
			str: Generated response text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_text = ''
			if self.response is None:
				return self.output_text
			
			self.response_content = getattr( self.response, 'content', '', )
			if self.response_content:
				self.output_text = str( self.response_content ).strip( )
			
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = ('get_output_text( self ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def summarize( self, file_id: str, prompt: str, model: str, instruct: str = '',
		temperature: float = 0.0, top_p: float = 0.0, frequency: float = 0.0, presence: float=0.0,
		max_tokens: int = 0, store: bool = False, stream: bool = False,
		include: Optional[ List[ str ] ] = None, previous_id: str = '',
		stream_handler: Any = None ) -> str:
		"""Analyze a file.
		
		Purpose:
			Generates a response to a required prompt using a required xAI file attachment and
			Grok model.
		
		Args:
			file_id (str): Required xAI file identifier.
			prompt (str): Required file-analysis prompt.
			model (str): Required Grok model identifier.
			instruct (str): Optional system instruction.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			store (bool): Indicates whether xAI stores chat messages.
			stream (bool): Indicates whether response streaming is enabled.
			include (Optional[List[str]]): Optional streaming response inclusions.
			previous_id (str): Previous stored response identifier.
			stream_handler (Any): Optional callable receiving each streaming text delta.
		
		Returns:
			str: Generated file-analysis response.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.file_id = file_id
			self.prompt = prompt
			self.model = model
			self.instructions = instruct
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.store_messages = store
			self.stream = stream
			self.include = (include if include is not None else [ ])
			self.previous_id = previous_id
			self.previous_response_id = previous_id
			self.stream_handler = stream_handler
			self.chat_values = { 'model': self.model, 'store_messages': self.store_messages, }
			if self.temperature > 0:
				self.chat_values[ 'temperature' ] = (self.temperature)
			
			if self.top_percent > 0:
				self.chat_values[ 'top_p' ] = (self.top_percent)
			
			if self.frequency_penalty != 0:
				self.chat_values[ 'frequency_penalty' ] = (self.frequency_penalty)
			
			if self.presence_penalty != 0:
				self.chat_values[ 'presence_penalty' ] = (self.presence_penalty)
			
			if self.max_output_tokens > 0:
				self.chat_values[ 'max_tokens' ] = (self.max_output_tokens)
			
			if self.include:
				self.chat_values[ 'include' ] = self.include
			
			if self.previous_response_id:
				self.chat_values[ 'previous_response_id' ] = (self.previous_response_id)
			
			self.client = Client( api_key=self.api_key, timeout=self.timeout, )
			self.chat = self.client.chat.create( **self.chat_values )
			if self.instructions:
				self.chat.append( system( self.instructions ) )
			
			self.chat.append( user( self.prompt, file( file_id=self.file_id, ), ) )
			
			if self.stream:
				self.parts = [ ]
				self.response = None
				for response, chunk in self.chat.stream( ):
					self.response = response
					self.chunk_content = getattr( chunk, 'content', '', )
					if not self.chunk_content:
						continue
					
					self.chunk_content = str( self.chunk_content )
					self.parts.append( self.chunk_content )
					
					if self.stream_handler is not None:
						self.stream_handler( self.chunk_content )
				
				self.output_text = ''.join( self.parts ).strip( )
				
				if not self.output_text:
					self.output_text = self.get_output_text( )
				
				return self.output_text
			
			self.response = self.chat.sample( )
			self.previous_id = str( getattr( self.response, 'id', '', ) or '' )
			self.previous_response_id = self.previous_id
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'summarize( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def search( self, file_id: str, query: str, model: str, instruct: str = '',
		temperature: float = 0.0, top_p: float = 0.0, frequency: float=0.0, presence: float=0.0,
		max_tokens: int = 0, store: bool = False, stream: bool = False,
		include: Optional[ List[ str ] ] = None, previous_id: str = '',
		stream_handler: Any = None ) -> str:
		"""Search a file.
		
		Purpose:
			Answers a required question using a required xAI file attachment.
		
		Args:
			file_id (str): Required xAI file identifier.
			query (str): Required question about the file.
			model (str): Required Grok model identifier.
			instruct (str): Optional system instruction.
			temperature (float): Sampling temperature.
			top_p (float): Nucleus-sampling value.
			frequency (float): Frequency penalty.
			presence (float): Presence penalty.
			max_tokens (int): Maximum output-token count.
			store (bool): Indicates whether xAI stores chat messages.
			stream (bool): Indicates whether response streaming is enabled.
			include (Optional[List[str]]): Optional streaming response inclusions.
			previous_id (str): Previous stored response identifier.
			stream_handler (Any): Optional callable receiving each streaming text delta.
		
		Returns:
			str: Generated answer based on the attached file.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_id', file_id )
			throw_if( 'query', query )
			throw_if( 'model', model )
			self.file_id = file_id
			self.query_text = query
			self.model = model
			self.instructions = instruct
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.store_messages = store
			self.stream = stream
			self.include = (include if include is not None else [ ])
			self.previous_id = previous_id
			self.stream_handler = stream_handler
			self.prompt = ('Answer the following question using the attached file. '
			               'Base the answer on the file content and identify any '
			               'information the file does not provide.\n\n'
			               f'Question: {self.query_text}')
			
			return self.summarize( self.file_id, self.prompt, self.model, self.instructions,
				self.temperature, self.top_percent, self.frequency_penalty, self.presence_penalty,
				self.max_output_tokens, self.store_messages, self.stream, self.include,
				self.previous_id, self.stream_handler, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'search( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, file_id: str, max_chars: int = 4000 ) -> Dict[ str, Any ]:
		"""Survey a file.
		
		Purpose:
			Retrieves file metadata and content and returns a bounded textual preview for
			application inspection.
		
		Args:
			file_id (str): Required xAI file identifier.
			max_chars (int): Maximum number of preview characters.
		
		Returns:
			Dict[str, Any]: File metadata, preview text, and file identifier.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.max_chars = max_chars
			self.metadata = self.retrieve( self.file_id )
			self.file_content = self.extract( self.file_id )
			if isinstance( self.file_content, bytes ):
				self.content_text = self.file_content.decode( 'utf-8', errors='replace', )
			else:
				self.content_text = str( self.file_content )
			
			if self.max_chars > 0:
				self.preview = self.content_text[ :self.max_chars ]
			else:
				self.preview = self.content_text
			
			return { 'metadata': self.metadata, 'preview': self.preview, 'file_id': self.file_id, }
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Files'
			exception.method = 'survey( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Grok Files wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'base_url', 'timeout', 'client', 'chat', 'file_path', 'file_name',
			'file_id', 'file_ids', 'purpose', 'expires_after', 'model', 'prompt', 'instructions',
			'temperature', 'top_percent', 'frequency_penalty', 'presence_penalty',
			'max_output_tokens', 'store_messages', 'stream', 'include', 'previous_id',
			'previous_response_id', 'response', 'file_content', 'output_text', 'limit',
			'pagination_token', 'next_token', 'documents', 'model_options', 'purpose_options',
			'format_options', 'tool_options', 'include_options', 'get_headers',
			'normalize_metadata', 'upload', 'list', 'list_files', 'retrieve', 'extract',
			'download',
			'content', 'delete', 'get_output_text', 'summarize', 'search', 'survey', ]

class TTS( Grok ):
	"""Provide xAI text-to-speech workflow support.
	
	Purpose:
		Provides batch speech synthesis through the xAI Text-to-Speech REST API. The class
		assigns accepted arguments to object members, constructs the provider-native output
		format and request payload, executes the synthesis request, extracts the returned audio,
		and optionally writes the audio bytes to a local file.
	
	Attributes:
		api_key (str): xAI API key.
		base_url (str): xAI REST API base URL.
		text (str): Text converted to speech.
		language (str): BCP-47 language code used for synthesis.
		voice_id (str): Built-in or custom xAI voice identifier.
		output_format (str | Dict[str, Any]): Requested audio-output configuration.
		codec (str): Requested audio codec.
		speed (float): Speech-speed multiplier.
		optimize_streaming_latency (int): Latency-optimization level.
		text_normalization (bool): Indicates whether written text is normalized before synthesis.
		sample_rate (int): Requested audio sample rate.
		bit_rate (int): Requested MP3 bit rate.
		with_timestamps (bool): Indicates whether character timing metadata is requested.
		audio_path (str): Optional local output path.
		response (Any): Latest HTTP response.
		audio (bytes): Generated audio bytes.
		audio_timestamps (Any): Character timing metadata returned by the provider.
		duration (float): Generated audio duration in seconds.
		content_type (str): MIME type of the generated audio.
		params (Dict[str, Any]): Provider request payload.
	"""
	api_key: str
	base_url: str
	text: str
	language: str
	voice_id: str
	output_format: str | Dict[ str, Any ]
	codec: str
	speed: float
	optimize_streaming_latency: int
	text_normalization: bool
	sample_rate: int
	bit_rate: int
	with_timestamps: bool
	audio_path: str
	response: Any
	audio: bytes
	audio_timestamps: Any
	duration: float
	content_type: str
	params: Dict[ str, Any ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes xAI text-to-speech configuration and runtime state without executing a
			provider request.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1', )
		self.timeout = 3600
		self.text = ''
		self.language = 'en'
		self.voice_id = 'eve'
		self.output_format = 'mp3'
		self.codec = 'mp3'
		self.speed = 1.0
		self.optimize_streaming_latency = 0
		self.text_normalization = False
		self.sample_rate = 24000
		self.bit_rate = 128000
		self.with_timestamps = False
		self.audio_path = ''
		self.filepath = ''
		self.response = None
		self.audio = b''
		self.audio_timestamps = None
		self.duration = 0.0
		self.content_type = ''
		self.params = { }
		self.output_format_payload = { }
		self.result = { }
	
	@property
	def voice_options( self ) -> List[ str ]:
		"""Get voice options.
		
		Purpose:
			Returns the standard built-in xAI voices exposed by the wrapper. Custom voice
			identifiers may also be supplied directly to create_speech().
		
		Returns:
			List[str]: Standard built-in xAI voice identifiers.
		"""
		return [ 'eve', 'ara', 'rex', 'sal', 'leo', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get audio-format options.
		
		Purpose:
			Returns audio codecs supported by the xAI Text-to-Speech API.
		
		Returns:
			List[str]: Supported audio-codec values.
		"""
		return [ 'mp3', 'wav', 'pcm', 'mulaw', 'alaw', ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Get language options.
		
		Purpose:
			Returns documented language codes exposed by the xAI Text-to-Speech API.
		
		Returns:
			List[str]: Supported language-code values.
		"""
		return [ 'auto', 'en', 'ar-EG', 'ar-SA', 'ar-AE', 'bn', 'zh', 'fr', 'de', 'hi', 'id', 'it',
			'ja', 'ko', 'pt-BR', 'pt-PT', 'ru', 'es-MX', 'es-ES', 'tr', 'vi', ]
	
	@property
	def sample_rate_options( self ) -> List[ int ]:
		"""Get sample-rate options.
		
		Purpose:
			Returns documented audio sample rates supported by xAI speech synthesis.
		
		Returns:
			List[int]: Supported sample rates in hertz.
		"""
		return [ 8000, 16000, 22050, 24000, 44100, 48000, ]
	
	@property
	def bit_rate_options( self ) -> List[ int ]:
		"""Get bit-rate options.
		
		Purpose:
			Returns documented MP3 bit rates supported by xAI speech synthesis.
		
		Returns:
			List[int]: Supported MP3 bit rates in bits per second.
		"""
		return [ 32000, 64000, 96000, 128000, 192000, ]
	
	def build_output_format( self, output_format: str | Dict[ str, Any ] = 'mp3',
		sample_rate: int = 24000, bit_rate: int = 128000 ) -> Dict[ str, Any ]:
		"""Build output-format configuration.
		
		Purpose:
			Builds the provider-native audio-output configuration from the selected codec,
			sample rate, and MP3 bit rate.
		
		Args:
			output_format (str | Dict[str, Any]): Audio codec or complete provider output
				configuration.
			sample_rate (int): Requested audio sample rate.
			bit_rate (int): Requested MP3 bit rate.
		
		Returns:
			Dict[str, Any]: Provider-ready output-format configuration.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_format = output_format
			self.sample_rate = sample_rate
			self.bit_rate = bit_rate
			self.output_format_payload = { }
			if isinstance( self.output_format, dict ):
				self.output_format_payload = { key: value for key, value in
					self.output_format.items( ) if value is not None and value != '' }
				self.codec = str(
					self.output_format_payload.get( 'codec', 'mp3', ) ).strip( ).lower( )
			else:
				self.codec = str( self.output_format ).strip( ).lower( )
				throw_if( 'output_format', self.codec )
				self.output_format_payload = { 'codec': self.codec, }
			
			self.output_format_payload[ 'sample_rate' ] = (self.sample_rate)
			
			if self.codec == 'mp3':
				self.output_format_payload[ 'bit_rate' ] = (self.bit_rate)
			else:
				self.output_format_payload.pop( 'bit_rate', None, )
			
			return self.output_format_payload
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'TTS'
			exception.method = 'build_output_format( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def extract_audio( self ) -> bytes:
		"""Extract generated audio.
		
		Purpose:
			Extracts audio bytes and available response metadata from the latest xAI
			Text-to-Speech response. Both direct binary responses and JSON responses containing
			base64-encoded audio are supported.
		
		Returns:
			bytes: Generated audio bytes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response', self.response )
			self.audio = b''
			self.audio_timestamps = None
			self.duration = 0.0
			self.content_type = str( self.response.headers.get( 'Content-Type', '', ) )
			if 'application/json' in self.content_type.lower( ):
				self.result = self.response.json( )
				self.encoded_audio = self.result.get( 'audio', '', )
				if self.encoded_audio:
					self.audio = base64.b64decode( self.encoded_audio )
				
				self.audio_timestamps = self.result.get( 'audio_timestamps', None, )
				self.duration = float( self.result.get( 'duration', 0.0, ) or 0.0 )
				self.content_type = str(
					self.result.get( 'content_type', self.content_type, ) or self.content_type )
			else:
				self.audio = self.response.content
			
			throw_if( 'audio', self.audio )
			return self.audio
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'TTS'
			exception.method = 'extract_audio( self ) -> bytes'
			Logger( ).write( exception )
			raise exception
	
	def create_speech( self, text: str, language: str = 'en', voice_id: str = 'eve',
		output_format: str | Dict[ str, Any ] = 'mp3', speed: float = 1.0,
		optimize_streaming_latency: int = 0, text_normalization: bool = False,
		sample_rate: int = 24000, bit_rate: int = 128000, filepath: str = '', audio_path: str = '',
		with_timestamps: bool = False ) -> bytes:
		"""Create speech.
		
		Purpose:
			Converts required text into speech through the xAI batch Text-to-Speech endpoint
			using the selected language, built-in or custom voice, output codec, sample rate,
			MP3 bit rate, speed, normalization, latency, and timestamp controls.
		
		Args:
			text (str): Required text converted to speech.
			language (str): BCP-47 language code or auto.
			voice_id (str): Built-in or custom xAI voice identifier.
			output_format (str | Dict[str, Any]): Audio codec or complete provider output
				configuration.
			speed (float): Speech-speed multiplier.
			optimize_streaming_latency (int): Latency-optimization level.
			text_normalization (bool): Indicates whether written text is normalized before
				synthesis.
			sample_rate (int): Requested audio sample rate.
			bit_rate (int): Requested MP3 bit rate.
			filepath (str): Optional local output path.
			audio_path (str): Compatibility alias for the optional local output path.
			with_timestamps (bool): Indicates whether character timing metadata is requested.
		
		Returns:
			bytes: Generated audio bytes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'language', language )
			throw_if( 'voice_id', voice_id )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.text = text
			self.language = language
			self.voice_id = voice_id
			self.output_format = output_format
			self.speed = speed
			self.optimize_streaming_latency = (optimize_streaming_latency)
			self.text_normalization = text_normalization
			self.sample_rate = sample_rate
			self.bit_rate = bit_rate
			self.filepath = filepath
			self.audio_path = audio_path
			self.with_timestamps = with_timestamps
			
			if not self.filepath:
				self.filepath = self.audio_path
			
			self.audio_path = self.filepath
			self.output_format_payload = self.build_output_format( self.output_format,
				self.sample_rate, self.bit_rate, )
			self.params = { 'text': self.text, 'language': self.language, 'voice_id': self.voice_id,
				'output_format': self.output_format_payload, 'speed': self.speed,
				'optimize_streaming_latency': (self.optimize_streaming_latency),
				'text_normalization': self.text_normalization,
				'with_timestamps': self.with_timestamps, }
			self.response = requests.post( url=(f'{self.base_url.rstrip( "/" )}/tts'),
				headers={ 'Authorization': f'Bearer {self.api_key}',
					'Content-Type': 'application/json', }, json=self.params,
				timeout=self.timeout, )
			self.response.raise_for_status( )
			self.audio = self.extract_audio( )
			if self.audio_path:
				with open( self.audio_path, 'wb' ) as target:
					target.write( self.audio )
			
			return self.audio
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Grok text-to-speech wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'base_url', 'timeout', 'text', 'language', 'voice_id', 'output_format',
			'codec', 'speed', 'optimize_streaming_latency', 'text_normalization', 'sample_rate',
			'bit_rate', 'with_timestamps', 'audio_path', 'filepath', 'response', 'audio',
			'audio_timestamps', 'duration', 'content_type', 'params', 'voice_options',
			'format_options', 'language_options', 'sample_rate_options', 'bit_rate_options',
			'build_output_format', 'extract_audio', 'create_speech', ]

class Translation( Grok ):
	"""Provide xAI audio-translation workflow support.
	
	Purpose:
		Provides spoken-audio translation by transcribing a required audio file through the
		xAI Speech-to-Text API and translating the resulting transcript through a required
		Grok text model. The class assigns accepted arguments to object members before
		constructing either provider request.
	
	Attributes:
		api_key (str): xAI API key.
		base_url (str): xAI REST API base URL.
		audio_path (str): Local audio-file path used by the current request.
		file_name (str): Source audio filename.
		mime_type (str): Source audio MIME type.
		source_language (str): Optional language code used for transcript formatting.
		target_language (str): Required translation target language.
		text_format (bool): Indicates whether inverse text normalization is enabled.
		keyterm (str): Optional transcription-bias term.
		model (str): Grok text model used for translation.
		prompt (str): Translation prompt sent to the Grok model.
		response (Any): Latest provider response.
		transcript (str): Text returned by the Speech-to-Text API.
		translation (str): Text returned by the Grok translation request.
		result (Dict[str, Any]): Parsed Speech-to-Text response.
		params (List[Tuple[str, str]]): Speech-to-Text multipart parameters.
		chat (Any): xAI chat used for translation.
		client (Optional[Client]): xAI SDK client.
		duration (float): Source audio duration returned by Speech-to-Text.
		words (List[Dict[str, Any]]): Word-level transcription segments.
	"""
	api_key: str
	base_url: str
	audio_path: str
	file_name: str
	mime_type: str
	source_language: str
	target_language: str
	text_format: bool
	keyterm: str
	model: str
	prompt: str
	response: Any
	transcript: str
	translation: str
	result: Dict[ str, Any ]
	params: List[ tuple[ str, str ] ]
	chat: Any
	client: Optional[ Client ]
	duration: float
	words: List[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'grok-4.20' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes xAI audio-translation configuration and runtime state without
			executing a provider request.
		
		Args:
			model (str): Default Grok text model used for translation.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1', )
		self.timeout = 3600
		self.audio_path = ''
		self.file_name = ''
		self.mime_type = ''
		self.source_language = ''
		self.target_language = 'en'
		self.text_format = False
		self.keyterm = ''
		self.model = model
		self.prompt = ''
		self.instructions = ''
		self.response = None
		self.transcript = ''
		self.translation = ''
		self.result = { }
		self.params = [ ]
		self.chat = None
		self.client = None
		self.duration = 0.0
		self.words = [ ]
		self.response_content = ''
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get translation-model options.
		
		Purpose:
			Returns Grok text models exposed for transcript translation.
		
		Returns:
			List[str]: Supported Grok text model identifiers.
		"""
		return [ 'grok-4.20', 'grok-4.20-reasoning', 'grok-4.20-multi-agent', 'grok-4.5', 'grok-4',
			'grok-4-latest', 'grok-4-fast-reasoning', 'grok-4-fast-non-reasoning', 'grok-3',
			'grok-3-mini', 'grok-3-fast', 'grok-3-mini-fast', ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Get language options.
		
		Purpose:
			Returns language codes supported for xAI Speech-to-Text formatting and transcript
			translation selection.
		
		Returns:
			List[str]: Supported language-code values.
		"""
		return [ 'ar', 'cs', 'da', 'de', 'en', 'es', 'fa', 'fil', 'fr', 'hi', 'id', 'it', 'ja',
			'ko', 'mk', 'ms', 'nl', 'pl', 'pt', 'ro', 'ru', 'sv', 'th', 'tr', 'vi', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get audio MIME-type options.
		
		Purpose:
			Returns common container MIME types accepted by the xAI Speech-to-Text endpoint.
		
		Returns:
			List[str]: Supported audio MIME-type values.
		"""
		return [ 'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/ogg',
			'audio/webm', 'audio/mp4', 'audio/aac', 'audio/m4a', ]
	
	@property
	def format_options( self ) -> List[ bool ]:
		"""Get text-formatting options.
		
		Purpose:
			Returns the Boolean values supported by the Speech-to-Text inverse text
			normalization argument.
		
		Returns:
			List[bool]: Available text-formatting values.
		"""
		return [ False, True, ]
	
	def get_mime_type( self, path: str ) -> str:
		"""Get audio MIME type.
		
		Purpose:
			Determines the MIME type of a required local audio file from its extension.
		
		Args:
			path (str): Required local audio-file path.
		
		Returns:
			str: Audio MIME type.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			self.audio_path = path
			self.suffix = Path( self.audio_path ).suffix.lower( )
			self.mime_type = { '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac',
				'.ogg': 'audio/ogg', '.webm': 'audio/webm', '.m4a': 'audio/mp4',
				'.mp4': 'audio/mp4', '.aac': 'audio/aac', }.get( self.suffix,
				'application/octet-stream', )
			return self.mime_type
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Translation'
			exception.method = ('get_mime_type( self, path: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, path: str, source_language: str = '', text_format: bool = False,
		mime_type: str = '', keyterm: str = '' ) -> str:
		"""Transcribe source audio.
		
		Purpose:
			Transcribes a required local audio file through the xAI Speech-to-Text endpoint
			using optional language-formatting and keyterm controls.
		
		Args:
			path (str): Required local audio-file path.
			source_language (str): Optional language code used with text formatting.
			text_format (bool): Indicates whether inverse text normalization is enabled.
			mime_type (str): Optional source-audio MIME type.
			keyterm (str): Optional term used to bias transcription.
		
		Returns:
			str: Generated source transcript.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.audio_path = path
			self.source_language = source_language
			self.text_format = text_format
			self.mime_type = mime_type
			self.keyterm = keyterm
			self.file_name = Path( self.audio_path ).name
			if not self.mime_type:
				self.mime_type = self.get_mime_type( self.audio_path )
			
			self.params = [ ('format', str( self.text_format ).lower( ),), ]
			if self.source_language:
				self.params.append( ('language', self.source_language,) )
			
			if self.keyterm:
				self.params.append( ('keyterm', self.keyterm,) )
			
			with open( self.audio_path, 'rb' ) as source:
				self.response = requests.post( url=(f'{self.base_url.rstrip( "/" )}'
				                                    f'/stt'),
					headers={ 'Authorization': (f'Bearer {self.api_key}'), }, data=self.params,
					files={ 'file': (self.file_name, source, self.mime_type,), },
					timeout=self.timeout, )
			
			self.response.raise_for_status( )
			self.result = self.response.json( )
			self.transcript = str( self.result.get( 'text', '', ) or '' ).strip( )
			self.duration = float( self.result.get( 'duration', 0.0, ) or 0.0 )
			self.words = self.result.get( 'words', [ ], ) or [ ]
			throw_if( 'transcript', self.transcript )
			return self.transcript
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Translation'
			exception.method = 'transcribe( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get translated text.
		
		Purpose:
			Extracts translated text from the latest Grok chat response.
		
		Returns:
			str: Generated translated text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.translation = ''
			if self.response is None:
				return self.translation
			
			self.response_content = getattr( self.response, 'content', '', )
			if self.response_content:
				self.translation = str( self.response_content ).strip( )
			
			return self.translation
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Translation'
			exception.method = 'get_output_text( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def translate( self, path: str, target_language: str, model: str, source_language: str = '',
		text_format: bool = False, mime_type: str = '', keyterm: str = '',
		instruct: str = '' ) -> str:
		"""Translate spoken audio.
		
		Purpose:
			Transcribes a required local audio file through xAI Speech-to-Text and translates
			the transcript into a required target language through a required Grok text model.
		
		Args:
			path (str): Required local audio-file path.
			target_language (str): Required target language or language code.
			model (str): Required Grok text model identifier.
			source_language (str): Optional source-language formatting code.
			text_format (bool): Indicates whether inverse text normalization is enabled.
			mime_type (str): Optional source-audio MIME type.
			keyterm (str): Optional term used to bias transcription.
			instruct (str): Optional translation system instruction.
		
		Returns:
			str: Generated translated text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'target_language', target_language, )
			throw_if( 'model', model )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.audio_path = path
			self.target_language = target_language
			self.model = model
			self.source_language = source_language
			self.text_format = text_format
			self.mime_type = mime_type
			self.keyterm = keyterm
			self.instructions = instruct
			self.transcript = self.transcribe( self.audio_path, self.source_language,
				self.text_format, self.mime_type, self.keyterm, )
			self.prompt = (f'Translate the following transcript into '
			               f'{self.target_language}. Preserve the meaning, '
			               f'tone, names, numbers, technical terms, and '
			               f'speaker distinctions. Return only the translated '
			               f'text.\n\n{self.transcript}')
			self.client = Client( api_key=self.api_key, timeout=self.timeout, )
			self.chat = self.client.chat.create( model=self.model, )
			
			if self.instructions:
				self.chat.append( system( self.instructions ) )
			
			self.chat.append( user( self.prompt ) )
			self.response = self.chat.sample( )
			self.translation = self.get_output_text( )
			throw_if( 'translation', self.translation, )
			return self.translation
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Translation'
			exception.method = 'translate( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Grok Translation wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'base_url', 'timeout', 'audio_path', 'file_name', 'mime_type',
			'source_language', 'target_language', 'text_format', 'keyterm', 'model', 'prompt',
			'instructions', 'response', 'transcript', 'translation', 'result', 'params', 'chat',
			'client', 'duration', 'words', 'model_options', 'language_options', 'mime_options',
			'format_options', 'get_mime_type', 'transcribe', 'get_output_text', 'translate', ]

class Transcription( Grok ):
	"""Provide xAI speech-to-text workflow support.
	
	Purpose:
		Provides batch audio transcription through the xAI Speech-to-Text REST API. The class
		assigns accepted arguments to object members, constructs the provider multipart request,
		uploads the required local audio file, and extracts transcript text, duration, detected
		language, word-level timestamps, and channel-level results from the provider response.
	
	Attributes:
		api_key (str): xAI API key.
		base_url (str): xAI REST API base URL.
		audio_path (str): Local audio-file path used by the current request.
		file_name (str): Source audio filename.
		mime_type (str): Source audio MIME type.
		language (str): Optional source-language hint.
		text_format (bool): Indicates whether inverse text normalization is enabled.
		keyterm (str): Optional transcription-bias term.
		response (Any): Latest provider response.
		transcript (str): Transcript extracted from the latest response.
		result (Dict[str, Any]): Parsed Speech-to-Text response.
		words (List[Dict[str, Any]]): Word-level transcription results.
		channels (List[Dict[str, Any]]): Channel-level transcription results.
		duration (float): Audio duration returned by the provider.
		params (List[Tuple[str, str]]): Multipart Speech-to-Text parameters.
	"""
	api_key: str
	base_url: str
	audio_path: str
	file_name: str
	mime_type: str
	language: str
	text_format: bool
	keyterm: str
	response: Any
	transcript: str
	result: Dict[ str, Any ]
	words: List[ Dict[ str, Any ] ]
	channels: List[ Dict[ str, Any ] ]
	duration: float
	params: List[ tuple[ str, str ] ]
	
	def __init__( self ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes xAI Speech-to-Text configuration and runtime state without executing a
			provider request.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1', )
		self.timeout = 3600
		self.audio_path = ''
		self.filepath = ''
		self.file_name = ''
		self.mime_type = ''
		self.language = ''
		self.text_format = False
		self.output_format = False
		self.keyterm = ''
		self.response = None
		self.transcript = ''
		self.result = { }
		self.words = [ ]
		self.channels = [ ]
		self.duration = 0.0
		self.params = [ ]
		self.content_type = ''
	
	@property
	def format_options( self ) -> List[ bool ]:
		"""Get text-formatting options.
		
		Purpose:
			Returns the Boolean values accepted by the xAI inverse text-normalization argument.
		
		Returns:
			List[bool]: Available text-formatting values.
		"""
		return [ False, True, ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Get language options.
		
		Purpose:
			Returns language codes supported by the xAI Speech-to-Text workflow.
		
		Returns:
			List[str]: Supported language-code values.
		"""
		return [ '', 'ar', 'cs', 'da', 'de', 'en', 'es', 'fa', 'fil', 'fr', 'hi', 'id', 'it', 'ja',
			'ko', 'mk', 'ms', 'nl', 'pl', 'pt', 'ro', 'ru', 'sv', 'th', 'tr', 'vi', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get audio MIME-type options.
		
		Purpose:
			Returns common audio MIME types accepted by the xAI Speech-to-Text endpoint.
		
		Returns:
			List[str]: Supported audio MIME-type values.
		"""
		return [ 'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/ogg',
			'audio/webm', 'audio/mp4', 'audio/aac', 'audio/m4a', ]
	
	def get_mime_type( self, path: str ) -> str:
		"""Get audio MIME type.
		
		Purpose:
			Determines the MIME type of a required local audio file from its extension.
		
		Args:
			path (str): Required local audio-file path.
		
		Returns:
			str: Audio MIME type.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			self.audio_path = path
			self.suffix = Path( self.audio_path ).suffix.lower( )
			self.mime_type = { '.mp3': 'audio/mpeg', '.mpeg': 'audio/mpeg', '.wav': 'audio/wav',
				'.flac': 'audio/flac', '.ogg': 'audio/ogg', '.oga': 'audio/ogg',
				'.webm': 'audio/webm', '.m4a': 'audio/mp4', '.mp4': 'audio/mp4',
				'.aac': 'audio/aac', }.get( self.suffix, 'application/octet-stream', )
			return self.mime_type
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Transcription'
			exception.method = ('get_mime_type( self, path: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, path: str, language: str = '', format: bool = False, mime_type: str = '',
		keyterm: str = '' ) -> str:
		"""Transcribe audio.
		
		Purpose:
			Uploads a required local audio file to the xAI batch Speech-to-Text endpoint and
			returns the generated transcript. Optional language, inverse text normalization,
			MIME type, and keyterm-bias settings are included when supplied.
		
		Args:
			path (str): Required local audio-file path.
			language (str): Optional source-language hint.
			format (bool): Indicates whether inverse text normalization is enabled.
			mime_type (str): Optional source-audio MIME type.
			keyterm (str): Optional term used to bias transcription.
		
		Returns:
			str: Generated transcript text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.audio_path = path
			self.filepath = path
			self.language = language
			self.text_format = format
			self.output_format = format
			self.mime_type = mime_type
			self.keyterm = keyterm
			self.file_name = Path( self.audio_path ).name
			
			if not self.mime_type:
				self.mime_type = self.get_mime_type( self.audio_path )
			
			self.params = [ ('format', str( self.text_format ).lower( ),), ]
			
			if self.language:
				self.params.append( ('language', self.language,) )
			
			if self.keyterm:
				self.params.append( ('keyterm', self.keyterm,) )
			
			with open( self.audio_path, 'rb' ) as source:
				self.response = requests.post( url=(f'{self.base_url.rstrip( "/" )}'
				                                    f'/stt'),
					headers={ 'Authorization': (f'Bearer {self.api_key}'), }, data=self.params,
					files={ 'file': (self.file_name, source, self.mime_type,), },
					timeout=self.timeout, )
			
			self.response.raise_for_status( )
			self.content_type = str( self.response.headers.get( 'Content-Type', '', ) )
			
			if 'application/json' in self.content_type.lower( ):
				self.result = self.response.json( )
				self.transcript = str( self.result.get( 'text', '', ) or '' ).strip( )
				self.language = str(
					self.result.get( 'language', self.language, ) or self.language )
				self.duration = float( self.result.get( 'duration', 0.0, ) or 0.0 )
				self.words = self.result.get( 'words', [ ], ) or [ ]
				self.channels = self.result.get( 'channels', [ ], ) or [ ]
			else:
				self.transcript = self.response.text.strip( )
				self.result = { 'text': self.transcript, 'language': self.language,
					'duration': self.duration, 'words': self.words, 'channels': self.channels, }
			
			throw_if( 'transcript', self.transcript )
			return self.transcript
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Transcription'
			exception.method = 'transcribe( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_result( self ) -> Dict[ str, Any ]:
		"""Get transcription result.
		
		Purpose:
			Returns the complete parsed response from the latest xAI Speech-to-Text request.
		
		Returns:
			Dict[str, Any]: Parsed transcription response.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Transcription'
			exception.method = ('get_result( self ) -> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def get_words( self ) -> List[ Dict[ str, Any ] ]:
		"""Get word-level results.
		
		Purpose:
			Returns word-level transcription timestamps from the latest xAI response.
		
		Returns:
			List[Dict[str, Any]]: Word-level transcription results.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			return self.words
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Transcription'
			exception.method = ('get_words( self ) -> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def get_channels( self ) -> List[ Dict[ str, Any ] ]:
		"""Get channel-level results.
		
		Purpose:
			Returns channel-level transcription results from the latest xAI response.
		
		Returns:
			List[Dict[str, Any]]: Channel-level transcription results.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			return self.channels
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Transcription'
			exception.method = ('get_channels( self ) -> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Grok Transcription wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'base_url', 'timeout', 'audio_path', 'filepath', 'file_name',
			'mime_type', 'language', 'text_format', 'output_format', 'keyterm', 'response',
			'transcript', 'result', 'words', 'channels', 'duration', 'params', 'content_type',
			'format_options', 'language_options', 'mime_options', 'get_mime_type', 'transcribe',
			'get_result', 'get_words', 'get_channels', ]

class Collections( Grok ):
	"""Provide xAI Collections workflow support.
	
	Purpose:
		Provides xAI collection creation, listing, retrieval, updating, deletion, document
		management, and semantic search. The class uses the xAI Management API key for
		collection and indexed-document administration and the standard xAI API key for
		semantic collection searches.
	
	Attributes:
		client (Optional[Client]): xAI SDK client used for collection searches.
		api_key (str): Standard xAI API key.
		management_key (str): xAI Management API key.
		base_url (str): Standard xAI REST API base URL.
		management_base_url (str): xAI Management API base URL.
		model (str): Grok model retained by search workflows.
		prompt (str): Semantic-search query.
		name (str): Collection name used by the current operation.
		description (str): Collection description used by the current operation.
		file_path (str): Local document path used by an upload operation.
		file_name (str): Document filename used by the current operation.
		file_id (str): xAI file identifier used by the current operation.
		file_ids (List[str]): File identifiers used by a batch operation.
		store_id (str): Application-facing collection identifier.
		store_ids (List[str]): Application-facing collection identifiers.
		collection_id (str): Provider collection identifier.
		collection_ids (List[str]): Provider collection identifiers.
		request (Dict[str, Any]): Latest request data.
		response (Any): Latest provider response.
		result (Any): Normalized result from the latest operation.
		params (Dict[str, Any]): Query-string parameters.
		payload (Dict[str, Any]): JSON request body.
		headers (Dict[str, str]): HTTP request headers.
		team_id (str): Optional xAI team identifier.
		limit (int): Maximum number of resources requested.
		order (str): Result ordering.
		sort_by (str): Result sort field.
		pagination_token (str): Pagination token.
		next_token (str): Pagination token returned by the provider.
		filter (str): Provider filter expression.
		collections (Dict[str, str]): Configured collection labels and identifiers.
		documents (Dict[str, str]): Configured document labels and identifiers.
	"""
	client: Optional[ Client ]
	api_key: str
	management_key: str
	base_url: str
	management_base_url: str
	model: str
	prompt: str
	name: str
	description: str
	file_path: str
	file_name: str
	file_id: str
	file_ids: List[ str ]
	store_id: str
	store_ids: List[ str ]
	collection_id: str
	collection_ids: List[ str ]
	request: Dict[ str, Any ]
	response: Any
	result: Any
	params: Dict[ str, Any ]
	payload: Dict[ str, Any ]
	headers: Dict[ str, str ]
	team_id: str
	limit: int
	order: str
	sort_by: str
	pagination_token: str
	next_token: str
	filter: str
	collections: Dict[ str, str ]
	documents: Dict[ str, str ]
	
	def __init__( self, model: str = 'grok-4.20' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes xAI collection-management and semantic-search state without executing a
			provider request.
		
		Args:
			model (str): Default Grok model retained by collection-search workflows.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.management_key = cfg.XAI_MANAGEMENT_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1', )
		self.management_base_url = getattr( cfg, 'XAI_MANAGEMENT_BASE_URL',
			'https://management-api.x.ai/v1', )
		self.timeout = 3600
		self.client = None
		self.model = model
		self.prompt = ''
		self.response_format = ''
		self.number = 1
		self.content = ''
		self.name = ''
		self.description = ''
		self.file_path = ''
		self.file_name = ''
		self.file_id = ''
		self.file_ids = [ ]
		self.store_id = ''
		self.store_ids = [ ]
		self.collection_id = ''
		self.collection_ids = [ ]
		self.request = { }
		self.response = None
		self.result = None
		self.params = { }
		self.payload = { }
		self.headers = { }
		self.team_id = ''
		self.limit = 100
		self.order = 'desc'
		self.sort_by = 'collection_name'
		self.pagination_token = ''
		self.next_token = ''
		self.filter = ''
		self.collections = cfg.GROK_COLLECTIONS
		self.documents = getattr( cfg, 'GROK_DOCUMENTS', { }, )
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get model options.
		
		Purpose:
			Returns Grok models exposed for collection-grounded generation workflows.
		
		Returns:
			List[str]: Supported Grok model identifiers.
		"""
		return [ 'grok-4.20', 'grok-4.20-reasoning', 'grok-4.20-multi-agent', 'grok-4.5', 'grok-4',
			'grok-4-latest', 'grok-4-fast-reasoning', 'grok-4-fast-non-reasoning', 'grok-3',
			'grok-3-mini', 'grok-3-fast', 'grok-3-mini-fast', ]
	
	@property
	def order_options( self ) -> List[ str ]:
		"""Get ordering options.
		
		Purpose:
			Returns ordering values supported by xAI collection and document list operations.
		
		Returns:
			List[str]: Supported ordering values.
		"""
		return [ 'asc', 'desc', ]
	
	@property
	def collection_sort_options( self ) -> List[ str ]:
		"""Get collection sort options.
		
		Purpose:
			Returns fields supported for sorting collection-list results.
		
		Returns:
			List[str]: Supported collection sort fields.
		"""
		return [ 'collection_name', 'created_at', 'documents_count', ]
	
	@property
	def document_sort_options( self ) -> List[ str ]:
		"""Get document sort options.
		
		Purpose:
			Returns fields supported for sorting collection-document results.
		
		Returns:
			List[str]: Supported document sort fields.
		"""
		return [ 'name', 'created_at', 'size_bytes', 'status', ]
	
	def get_collection_id( self, store_id: str ) -> str:
		"""Get provider collection identifier.
		
		Purpose:
			Resolves a required application-facing store identifier or configured collection
			label to the corresponding xAI collection identifier.
		
		Args:
			store_id (str): Required store identifier or configured collection label.
		
		Returns:
			str: Provider collection identifier.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.collection_id = str(
				self.collections.get( self.store_id, self.store_id, ) ).strip( )
			throw_if( 'collection_id', self.collection_id, )
			return self.collection_id
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('get_collection_id( self, store_id: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def get_collection_name( self, collection_id: str ) -> str:
		"""Get configured collection name.
		
		Purpose:
			Resolves a required provider collection identifier to its configured
			application-facing label when one is available.
		
		Args:
			collection_id (str): Required provider collection identifier.
		
		Returns:
			str: Configured collection label or the original identifier.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'collection_id', collection_id, )
			self.collection_id = collection_id
			
			for label, identifier in self.collections.items( ):
				if identifier == self.collection_id:
					return label
			
			return self.collection_id
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('get_collection_name( self, collection_id: str ) -> str')
			Logger( ).write( exception )
			raise exception
	
	def build_management_headers( self ) -> Dict[ str, str ]:
		"""Build Management API headers.
		
		Purpose:
			Builds authenticated JSON headers for xAI collection-management requests.
		
		Returns:
			Dict[str, str]: Management API request headers.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'XAI_MANAGEMENT_KEY', self.management_key, )
			self.headers = { 'Authorization': (f'Bearer {self.management_key}'),
				'Content-Type': 'application/json', }
			return self.headers
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('build_management_headers( self ) -> Dict[ str, str ]')
			Logger( ).write( exception )
			raise exception
	
	def execute_management_request( self, method: str, path: str,
		params: Optional[ Dict[ str, Any ] ] = None,
		payload: Optional[ Dict[ str, Any ] ] = None ) -> Any:
		"""Execute Management API request.
		
		Purpose:
			Executes an authenticated xAI collection-management request and returns its decoded
			JSON body when present.
		
		Args:
			method (str): Required HTTP method.
			path (str): Required Management API resource path.
			params (Optional[Dict[str, Any]]): Optional query-string parameters.
			payload (Optional[Dict[str, Any]]): Optional JSON request body.
		
		Returns:
			Any: Decoded JSON response or an empty dictionary.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'method', method )
			throw_if( 'path', path )
			self.method = method.upper( )
			self.resource_path = path
			self.params = (params if params is not None else { })
			self.payload = (payload if payload is not None else { })
			self.headers = self.build_management_headers( )
			self.response = requests.request( method=self.method,
				url=(f'{self.management_base_url.rstrip( "/" )}/'
				     f'{self.resource_path.lstrip( "/" )}'), headers=self.headers,
				params=self.params if self.params else None,
				json=self.payload if self.payload else None, timeout=self.timeout, )
			self.response.raise_for_status( )
			
			if not self.response.content:
				self.result = { }
				return self.result
			
			self.result = self.response.json( )
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('execute_management_request( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def normalize_collection( self, collection: Dict[ str, Any ] ) -> Dict[ str, Any ]:
		"""Normalize collection metadata.
		
		Purpose:
			Converts required xAI collection metadata into a stable application-facing record.
		
		Args:
			collection (Dict[str, Any]): Required provider collection metadata.
		
		Returns:
			Dict[str, Any]: Application-facing collection metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'collection', collection )
			self.collection = collection
			self.collection_id = str(
				self.collection.get( 'collection_id', self.collection.get( 'id', '', ), ) or '' )
			self.name = str( self.collection.get( 'collection_name',
				self.collection.get( 'name', '', ), ) or '' )
			
			return { 'id': self.collection_id, 'collection_id': self.collection_id,
				'name': self.name, 'collection_name': self.name,
				'description': self.collection.get( 'collection_description',
					self.collection.get( 'description', '', ), ),
				'created_at': self.collection.get( 'created_at', None, ),
				'documents_count': self.collection.get( 'documents_count', 0, ),
				'collection_type': self.collection.get( 'collection_type', '', ),
				'index_configuration': self.collection.get( 'index_configuration', { }, ),
				'chunk_configuration': self.collection.get( 'chunk_configuration', { }, ),
				'field_definitions': self.collection.get( 'field_definitions', [ ], ),
				'metadata': self.collection, }
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('normalize_collection( self, collection: '
			                    'Dict[ str, Any ] ) -> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def normalize_collection_list( self, payload: Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]:
		"""Normalize collection list.
		
		Purpose:
			Converts a required xAI collection-list response into application-facing records
			and updates the configured name-to-identifier mapping.
		
		Args:
			payload (Dict[str, Any]): Required provider collection-list response.
		
		Returns:
			List[Dict[str, Any]]: Application-facing collection metadata records.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'payload', payload )
			self.payload = payload
			self.next_token = str( self.payload.get( 'pagination_token', '', ) or '' )
			self.collection_data = self.payload.get( 'collections', [ ], ) or [ ]
			self.results = [ self.normalize_collection( item ) for item in self.collection_data ]
			
			for item in self.results:
				if item[ 'name' ] and item[ 'id' ]:
					self.collections[ item[ 'name' ] ] = item[ 'id' ]
			
			return self.results
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('normalize_collection_list( self, payload: '
			                    'Dict[ str, Any ] ) -> List[ Dict[ str, Any ] ]')
			Logger( ).write( exception )
			raise exception
	
	def get_text_output( self, response: Any ) -> Any:
		"""Get collection-search output.
		
		Purpose:
			Extracts textual content or semantic-search matches from a required xAI collection
			search response.
		
		Args:
			response (Any): Required xAI collection-search response.
		
		Returns:
			Any: Search text, semantic matches, or the original provider response.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response', response )
			self.response = response
			self.output_text = getattr( self.response, 'content', '', )
			
			if self.output_text:
				return str( self.output_text ).strip( )
			
			self.matches = getattr( self.response, 'matches', None, )
			
			if self.matches is not None:
				return self.matches
			
			if isinstance( self.response, dict ):
				if 'matches' in self.response:
					return self.response[ 'matches' ]
				
				if 'content' in self.response:
					return self.response[ 'content' ]
			
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('get_text_output( self, response: Any ) -> Any')
			Logger( ).write( exception )
			raise exception
	
	def create( self, name: str, description: str = '' ) -> Dict[ str, Any ]:
		"""Create a collection.
		
		Purpose:
			Creates an xAI collection with a required name and optional description.
		
		Args:
			name (str): Required collection name.
			description (str): Optional collection description.
		
		Returns:
			Dict[str, Any]: Created collection metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'name', name )
			self.name = name
			self.description = description
			self.payload = { 'collection_name': self.name, }
			
			if self.description:
				self.payload[ 'collection_description' ] = (self.description)
			
			self.result = self.execute_management_request( 'POST', '/collections',
				payload=self.payload, )
			self.result = self.normalize_collection( self.result )
			self.collection_id = self.result[ 'id' ]
			
			if self.name and self.collection_id:
				self.collections[ self.name ] = self.collection_id
			
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = 'create( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list( self, limit: int = 100, order: str = 'desc', sort_by: str = 'collection_name',
		pagination_token: str = '', filter: str = '', team_id: str = '' ) -> List[
		Dict[ str, Any ] ]:
		"""List collections.
		
		Purpose:
			Lists xAI collections using pagination, ordering, sorting, filtering, and optional
			team scope.
		
		Args:
			limit (int): Maximum number of collections requested.
			order (str): Result ordering.
			sort_by (str): Collection sort field.
			pagination_token (str): Optional pagination token.
			filter (str): Optional provider filter expression.
			team_id (str): Optional team identifier.
		
		Returns:
			List[Dict[str, Any]]: Application-facing collection metadata records.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.limit = limit
			self.order = order
			self.sort_by = sort_by
			self.pagination_token = pagination_token
			self.filter = filter
			self.team_id = team_id
			self.params = { 'limit': self.limit, 'order': self.order, 'sort_by': self.sort_by, }
			
			if self.pagination_token:
				self.params[ 'pagination_token' ] = (self.pagination_token)
			
			if self.filter:
				self.params[ 'filter' ] = self.filter
			
			if self.team_id:
				self.params[ 'team_id' ] = self.team_id
			
			self.result = self.execute_management_request( 'GET', '/collections',
				params=self.params, )
			return self.normalize_collection_list( self.result )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = 'list( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str, team_id: str = '' ) -> Dict[ str, Any ]:
		"""Retrieve a collection.
		
		Purpose:
			Retrieves metadata for a required xAI collection.
		
		Args:
			store_id (str): Required collection identifier or configured label.
			team_id (str): Optional team identifier.
		
		Returns:
			Dict[str, Any]: Application-facing collection metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.team_id = team_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.params = { }
			
			if self.team_id:
				self.params[ 'team_id' ] = self.team_id
			
			self.result = self.execute_management_request( 'GET',
				f'/collections/{self.collection_id}', params=self.params, )
			return self.normalize_collection( self.result )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def update( self, store_id: str, name: str = '', description: str = '' ) -> Dict[ str, Any ]:
		"""Update a collection.
		
		Purpose:
			Updates the name or description of a required xAI collection.
		
		Args:
			store_id (str): Required collection identifier or configured label.
			name (str): Optional replacement collection name.
			description (str): Optional replacement collection description.
		
		Returns:
			Dict[str, Any]: Updated collection metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.name = name
			self.description = description
			self.collection_id = self.get_collection_id( self.store_id )
			self.payload = { }
			
			if self.name:
				self.payload[ 'collection_name' ] = self.name
			
			if self.description:
				self.payload[ 'collection_description' ] = (self.description)
			
			throw_if( 'payload', self.payload )
			self.result = self.execute_management_request( 'PUT',
				f'/collections/{self.collection_id}', payload=self.payload, )
			self.result = self.normalize_collection( self.result )
			
			if self.name:
				self.collections[ self.name ] = self.collection_id
			
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = 'update( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str, team_id: str = '' ) -> bool:
		"""Delete a collection.
		
		Purpose:
			Deletes a required xAI collection.
		
		Args:
			store_id (str): Required collection identifier or configured label.
			team_id (str): Optional team identifier.
		
		Returns:
			bool: True when the deletion request completes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.team_id = team_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.params = { }
			
			if self.team_id:
				self.params[ 'team_id' ] = self.team_id
			
			self.execute_management_request( 'DELETE', f'/collections/{self.collection_id}',
				params=self.params, )
			
			for label, identifier in list( self.collections.items( ) ):
				if identifier == self.collection_id:
					del self.collections[ label ]
			
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = 'delete( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def add_document( self, store_id: str, file_id: str,
		fields: Optional[ Dict[ str, Any ] ] = None ) -> Any:
		"""Add a document to a collection.
		
		Purpose:
			Adds a required existing xAI file to a required collection and optionally assigns
			document metadata fields.
		
		Args:
			store_id (str): Required collection identifier or configured label.
			file_id (str): Required xAI file identifier.
			fields (Optional[Dict[str, Any]]): Optional collection document fields.
		
		Returns:
			Any: Provider document-addition result.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			self.store_id = store_id
			self.file_id = file_id
			self.fields = (fields if fields is not None else { })
			self.collection_id = self.get_collection_id( self.store_id )
			self.payload = { }
			
			if self.fields:
				self.payload[ 'fields' ] = self.fields
			
			self.result = self.execute_management_request( 'POST',
				(f'/collections/{self.collection_id}/'
				 f'documents/{self.file_id}'), payload=self.payload, )
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = 'add_document( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list_documents( self, store_id: str, limit: int = 100, order: str = 'desc',
		sort_by: str = 'name', pagination_token: str = '', filter: str = '', team_id: str = '' ) \
			-> \
			List[ Dict[ str, Any ] ]:
		"""List collection documents.
		
		Purpose:
			Lists documents in a required xAI collection using pagination, ordering, sorting,
			filtering, and optional team scope.
		
		Args:
			store_id (str): Required collection identifier or configured label.
			limit (int): Maximum number of documents requested.
			order (str): Result ordering.
			sort_by (str): Document sort field.
			pagination_token (str): Optional pagination token.
			filter (str): Optional document filter expression.
			team_id (str): Optional team identifier.
		
		Returns:
			List[Dict[str, Any]]: Collection document metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.limit = limit
			self.order = order
			self.sort_by = sort_by
			self.pagination_token = pagination_token
			self.filter = filter
			self.team_id = team_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.params = { 'limit': self.limit, 'order': self.order, 'sort_by': self.sort_by, }
			
			if self.pagination_token:
				self.params[ 'pagination_token' ] = (self.pagination_token)
			
			if self.filter:
				self.params[ 'filter' ] = self.filter
			
			if self.team_id:
				self.params[ 'team_id' ] = self.team_id
			
			self.result = self.execute_management_request( 'GET',
				(f'/collections/{self.collection_id}/'
				 f'documents'), params=self.params, )
			self.next_token = str( self.result.get( 'pagination_token', '', ) or '' )
			return self.result.get( 'documents', [ ], ) or [ ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = 'list_documents( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_document( self, store_id: str, file_id: str, team_id: str = '' ) -> Dict[
		str, Any ]:
		"""Retrieve collection document.
		
		Purpose:
			Retrieves metadata for a required document in a required xAI collection.
		
		Args:
			store_id (str): Required collection identifier or configured label.
			file_id (str): Required xAI file identifier.
			team_id (str): Optional team identifier.
		
		Returns:
			Dict[str, Any]: Collection document metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			self.store_id = store_id
			self.file_id = file_id
			self.team_id = team_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.params = { }
			
			if self.team_id:
				self.params[ 'team_id' ] = self.team_id
			
			self.result = self.execute_management_request( 'GET',
				(f'/collections/{self.collection_id}/'
				 f'documents/{self.file_id}'), params=self.params, )
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('retrieve_document( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def regenerate_document( self, store_id: str, file_id: str, team_id: str = '' ) -> Any:
		"""Regenerate document index.
		
		Purpose:
			Regenerates semantic indices for a required document in a required xAI collection.
		
		Args:
			store_id (str): Required collection identifier or configured label.
			file_id (str): Required xAI file identifier.
			team_id (str): Optional team identifier.
		
		Returns:
			Any: Provider regeneration result.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			self.store_id = store_id
			self.file_id = file_id
			self.team_id = team_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.params = { }
			
			if self.team_id:
				self.params[ 'team_id' ] = self.team_id
			
			self.result = self.execute_management_request( 'PATCH',
				(f'/collections/{self.collection_id}/'
				 f'documents/{self.file_id}'), params=self.params, )
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('regenerate_document( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def remove_document( self, store_id: str, file_id: str, team_id: str = '' ) -> bool:
		"""Remove a collection document.
		
		Purpose:
			Removes a required document from a required xAI collection without deleting the
			underlying xAI file.
		
		Args:
			store_id (str): Required collection identifier or configured label.
			file_id (str): Required xAI file identifier.
			team_id (str): Optional team identifier.
		
		Returns:
			bool: True when the removal request completes.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			self.store_id = store_id
			self.file_id = file_id
			self.team_id = team_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.params = { }
			
			if self.team_id:
				self.params[ 'team_id' ] = self.team_id
			
			self.execute_management_request( 'DELETE', (f'/collections/{self.collection_id}/'
			                                            f'documents/{self.file_id}'),
				params=self.params, )
			return True
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('remove_document( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def batch_get_documents( self, store_id: str, file_ids: List[ str ], team_id: str = '' ) -> \
			List[ Dict[ str, Any ] ]:
		"""Retrieve document metadata in a batch.
		
		Purpose:
			Retrieves metadata for required file identifiers in a required xAI collection.
		
		Args:
			store_id (str): Required collection identifier or configured label.
			file_ids (List[str]): Required xAI file identifiers.
			team_id (str): Optional team identifier.
		
		Returns:
			List[Dict[str, Any]]: Requested collection document metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_ids', file_ids )
			self.store_id = store_id
			self.file_ids = file_ids
			self.team_id = team_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.params = { 'file_ids': self.file_ids, }
			
			if self.team_id:
				self.params[ 'team_id' ] = self.team_id
			
			self.result = self.execute_management_request( 'GET',
				(f'/collections/{self.collection_id}/'
				 f'documents:batchGet'), params=self.params, )
			return self.result.get( 'documents', [ ], ) or [ ]
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = ('batch_get_documents( self, **kwargs )')
			Logger( ).write( exception )
			raise exception
	
	def search( self, prompt: str, store_id: str, model: str, filter: str = '' ) -> Any:
		"""Search a collection.
		
		Purpose:
			Performs semantic retrieval for a required query against a required xAI collection.
		
		Args:
			prompt (str): Required semantic-search query.
			store_id (str): Required collection identifier or configured label.
			model (str): Required Grok model retained by the operation.
			filter (str): Optional document metadata filter.
		
		Returns:
			Any: Semantic-search matches returned by xAI.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_id', store_id )
			throw_if( 'model', model )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.prompt = prompt
			self.store_id = store_id
			self.model = model
			self.filter = filter
			self.collection_id = self.get_collection_id( self.store_id )
			self.collection_ids = [ self.collection_id, ]
			self.client = Client( api_key=self.api_key, management_api_key=self.management_key,
				timeout=self.timeout, )
			
			if self.filter:
				self.response = self.client.collections.search( query=self.prompt,
					collection_ids=self.collection_ids, filter=self.filter, )
			else:
				self.response = self.client.collections.search( query=self.prompt,
					collection_ids=self.collection_ids, )
			
			return self.get_text_output( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = 'search( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, prompt: str, store_ids: List[ str ], model: str, filter: str = '' ) -> Any:
		"""Search multiple collections.
		
		Purpose:
			Performs semantic retrieval for a required query across multiple required xAI
			collections.
		
		Args:
			prompt (str): Required semantic-search query.
			store_ids (List[str]): Required collection identifiers or configured labels.
			model (str): Required Grok model retained by the operation.
			filter (str): Optional document metadata filter.
		
		Returns:
			Any: Semantic-search matches returned by xAI.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_ids', store_ids )
			throw_if( 'model', model )
			throw_if( 'XAI_API_KEY', self.api_key )
			self.prompt = prompt
			self.store_ids = store_ids
			self.model = model
			self.filter = filter
			self.collection_ids = [ self.get_collection_id( item ) for item in self.store_ids ]
			self.client = Client( api_key=self.api_key, management_api_key=self.management_key,
				timeout=self.timeout, )
			
			if self.filter:
				self.response = self.client.collections.search( query=self.prompt,
					collection_ids=self.collection_ids, filter=self.filter, )
			else:
				self.response = self.client.collections.search( query=self.prompt,
					collection_ids=self.collection_ids, )
			
			return self.get_text_output( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'VectorStores'
			exception.method = 'survey( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the Grok VectorStores wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'management_key', 'base_url', 'management_base_url', 'timeout',
			'client', 'model', 'prompt', 'response_format', 'number', 'content', 'name',
			'description', 'file_path', 'file_name', 'file_id', 'file_ids', 'store_id',
			'store_ids',
			'collection_id', 'collection_ids', 'request', 'response', 'result', 'params',
			'payload',
			'headers', 'team_id', 'limit', 'order', 'sort_by', 'pagination_token', 'next_token',
			'filter', 'collections', 'documents', 'model_options', 'order_options',
			'collection_sort_options', 'document_sort_options', 'get_collection_id',
			'get_collection_name', 'build_management_headers', 'execute_management_request',
			'normalize_collection', 'normalize_collection_list', 'get_text_output', 'create',
			'list', 'retrieve', 'update', 'delete', 'add_document', 'list_documents',
			'retrieve_document', 'regenerate_document', 'remove_document', 'batch_get_documents',
			'search', 'survey', ]
