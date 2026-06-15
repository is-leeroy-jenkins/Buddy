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
from xai_sdk.chat import user, system, image, file

def encode_image( image_path: str ) -> str:
	"""Encode image.
	
	Purpose:
	    Encodes local binary content into a text representation required by xAI request payloads.
	
	Args:
	    image_path (str): Image path supplied to the xAI workflow.
	
	Returns:
	    str: Result produced by the xAI workflow.
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

def throw_if( name: str, value: object ) -> None:
	"""Throw if.
	
	Purpose:
	    Validates required values before provider request construction.
	
	Args:
	    name (str): Name supplied to the xAI workflow.
	    value (object): Value supplied to the xAI workflow.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

class Grok( ):
	"""Grok workflow wrapper.
	
	Purpose:
	    Provides shared xAI configuration state, API-key storage, request defaults, and common
	    runtime containers used by Grok provider workflows.
	
	Attributes:
	    api_key: Runtime attribute used by the Grok workflow.
	    timeout: Runtime attribute used by the Grok workflow.
	    base_url: Runtime attribute used by the Grok workflow.
	    model: Runtime attribute used by the Grok workflow.
	    store_messages: Runtime attribute used by the Grok workflow.
	    response_format: Runtime attribute used by the Grok workflow.
	    temperature: Runtime attribute used by the Grok workflow.
	    top_percent: Runtime attribute used by the Grok workflow.
	    frequency_penalty: Runtime attribute used by the Grok workflow.
	    presence_penalty: Runtime attribute used by the Grok workflow.
	    max_output_tokens: Runtime attribute used by the Grok workflow.
	    tool_choice: Runtime attribute used by the Grok workflow.
	    tools: Runtime attribute used by the Grok workflow.
	    stops: Runtime attribute used by the Grok workflow.
	    instructions: Runtime attribute used by the Grok workflow.
	    content: Runtime attribute used by the Grok workflow.
	    messages: Runtime attribute used by the Grok workflow.
	    stores: Runtime attribute used by the Grok workflow.
	    files: Runtime attribute used by the Grok workflow.
	"""
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
		    Initializes Grok state with default configuration values and runtime attributes used
		    by later xAI provider calls.
		"""
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
	"""Chat workflow wrapper.
	
	Purpose:
	    Builds and executes xAI text, retrieval-augmented, collection-search, web-search,
	    X-search, code-execution, and tool-enabled chat workflows.
	
	Attributes:
	    include: Runtime attribute used by the Chat workflow.
	    tool_choice: Runtime attribute used by the Chat workflow.
	    previous_id: Runtime attribute used by the Chat workflow.
	    previous_response_id: Runtime attribute used by the Chat workflow.
	    conversation_id: Runtime attribute used by the Chat workflow.
	    parallel_tools: Runtime attribute used by the Chat workflow.
	    max_tools: Runtime attribute used by the Chat workflow.
	    input: Runtime attribute used by the Chat workflow.
	    tools: Runtime attribute used by the Chat workflow.
	    reasoning: Runtime attribute used by the Chat workflow.
	    allowed_domains: Runtime attribute used by the Chat workflow.
	    max_search_results: Runtime attribute used by the Chat workflow.
	    output_text: Runtime attribute used by the Chat workflow.
	    collections: Runtime attribute used by the Chat workflow.
	    files: Runtime attribute used by the Chat workflow.
	    content: Runtime attribute used by the Chat workflow.
	    vector_store_ids: Runtime attribute used by the Chat workflow.
	    file_ids: Runtime attribute used by the Chat workflow.
	    response: Runtime attribute used by the Chat workflow.
	    file_path: Runtime attribute used by the Chat workflow.
	"""
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	previous_id: Optional[ str ]
	previous_response_id: Optional[ str ]
	conversation_id: Optional[ str ]
	parallel_tools: Optional[ bool ]
	max_tools: Optional[ int ]
	input: Optional[ List[ Dict[ str, Any ] ] | str ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	allowed_domains: Optional[ List[ str ] ]
	max_search_results: Optional[ int ]
	output_text: Optional[ str ]
	collections: Optional[ Dict[ str, str ] ]
	files: Optional[ Dict[ str, str ] ]
	content: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	response: Optional[ Any ]
	file_path: Optional[ str ]
	
	def __init__( self, model: str = 'grok-4.20', prompt: str = None, temperature: float = None,
			top_p: float = None, presense: float = None, presence: float = None, store: bool =
			None,
			stream: bool = None, stops: List[ str ] = None,
			response_format: Dict[ str, Any ] = None,
			number: int = None, instruct: str = None, context: List[ Dict[ str, str ] ] = None,
			allowed_domains: List[ str ] = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None, max_tools: int = None,
			tool_choice: str = None, file_path: str = None, background: bool = None,
			is_parallel: bool = None, max_tokens: int = None, frequency: float = None,
			input: List[ Dict[ str, Any ] ] = None, file_ids: List[ str ] = None,
			previous_id: str = None, conversation_id: str = None,
			reasoning: Dict[ str, str ] | str = None, output_text: str = None,
			max_search_results: int = None, content: str = None,
			vector_store_ids: List[ str ] = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes Chat state with default configuration values and runtime attributes used
		    by later xAI provider calls.
		
		Args:
		    model (str): Model supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    presense (float): Presense supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    stops (List[str]): Stops supplied to the xAI workflow.
		    response_format (Dict[str, Any]): Response format supplied to the xAI workflow.
		    number (int): Number supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    context (List[Dict[str, str]]): Context supplied to the xAI workflow.
		    allowed_domains (List[str]): Allowed domains supplied to the xAI workflow.
		    include (List[str]): Include supplied to the xAI workflow.
		    tools (List[Dict[str, Any]]): Tools supplied to the xAI workflow.
		    max_tools (int): Max tools supplied to the xAI workflow.
		    tool_choice (str): Tool choice supplied to the xAI workflow.
		    file_path (str): File path supplied to the xAI workflow.
		    background (bool): Background supplied to the xAI workflow.
		    is_parallel (bool): Is parallel supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    input (List[Dict[str, Any]]): Input supplied to the xAI workflow.
		    file_ids (List[str]): File ids supplied to the xAI workflow.
		    previous_id (str): Previous id supplied to the xAI workflow.
		    conversation_id (str): Conversation id supplied to the xAI workflow.
		    reasoning (Dict[str, str] | str): Reasoning supplied to the xAI workflow.
		    output_text (str): Output text supplied to the xAI workflow.
		    max_search_results (int): Max search results supplied to the xAI workflow.
		    content (str): Content supplied to the xAI workflow.
		    vector_store_ids (List[str]): Vector store ids supplied to the xAI workflow.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.client = None
		self.model = model
		self.prompt = prompt
		self.number = number
		self.response_format = response_format if response_format is not None else { }
		self.temperature = temperature
		self.top_percent = top_p
		self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
		self.frequency_penalty = frequency
		self.presence_penalty = presence if presence is not None else presense
		self.max_output_tokens = max_tokens
		self.context = context if context is not None else [ ]
		self.stream = stream
		self.store_messages = store
		self.instructions = instruct
		self.stops = stops if stops is not None else [ ]
		self.background = background
		self.input = input if input is not None else [ ]
		self.include = include if include is not None else [ ]
		self.output_text = output_text
		self.max_tools = max_tools
		self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
		self.file_ids = file_ids if file_ids is not None else [ ]
		self.tools = tools if tools is not None else [ ]
		self.previous_id = previous_id
		self.previous_response_id = previous_id
		self.conversation_id = conversation_id
		self.reasoning = reasoning
		self.parallel_tools = is_parallel
		self.tool_choice = tool_choice
		self.response = None
		self.file_path = file_path
		self.content = content
		self.max_search_results = max_search_results
		self.request = { }
		self.messages = [ ]
		self.stream_requested = False
		self.background_requested = False
		self.collections = {
				'Federal Financial Regulations': 'collection_9195d847-03a1-443c-9240-294c64dd01e2',
				'Federal Financial Data': 'collection_e28cdcc2-a9e5-430a-bdf5-94fbaf44b6a4',
				'Explanatory Statements': 'collection_41dc3374-24d0-4692-819c-59e3d7b11b93',
				'Public Laws': 'collection_c1d0b83e-2f59-4f10-9cf7-51392b490fee',
		}
		self.files = {
				'Outlays.csv': 'file_b0a448b3-904a-40c7-bae1-64df657fde1c',
				'Authority.csv': 'file_c6ad236f-0c52-45f4-8883-d3be032d07c2',
				'Balances.csv': 'file_0f63d120-406f-49e6-97e5-7855f2cb26b5',
		}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'grok-4.20',
				'grok-4.20-reasoning',
				'grok-4.20-multi-agent',
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
				'grok-code-fast-1',
				'grok-3',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-mini-fast',
		]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'web_search_call_output',
				'x_search_call_output',
				'code_execution_call_output',
				'collections_search_call_output',
				'attachment_search_call_output',
				'mcp_call_output',
				'inline_citations',
				'verbose_streaming',
		]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'web_search',
				'x_search',
				'collections_search',
				'code_execution',
		]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [ 'auto', 'required', 'none', ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'text',
				'json_object',
				'json_schema',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'none',
				'low',
				'medium',
				'high',
				'xhigh',
		]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [ 'text', ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		"""Media options.
		
		Purpose:
		    Returns the configured option values exposed by the Chat workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [ 'auto', ]
	
	def build_reasoning( self, reasoning: str | Dict[ str, str ] = None ) -> Dict[
		                                                                         str, str ] | None:
		"""Build reasoning.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Args:
		    reasoning (str | Dict[str, str]): Reasoning supplied to the xAI workflow.
		
		Returns:
		    Dict[str, str] | None: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			if reasoning is None:
				return None
			
			if isinstance( reasoning, dict ):
				value = reasoning.get( 'effort' )
				if isinstance( value, str ) and value.strip( ) in self.reasoning_options:
					if value.strip( ) == 'none':
						return None
					
					return { 'effort': value.strip( ) }
				
				return None
			
			if isinstance( reasoning, str ) and reasoning.strip( ):
				value = reasoning.strip( )
				if value == 'none':
					return None
				
				if value in self.reasoning_options:
					return { 'effort': value }
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_reasoning( self, reasoning )'
			Logger( ).write( exception )
			raise exception
	
	def build_input( self, prompt: str, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""Build input.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    context (List[Dict[str, str]]): Context supplied to the xAI workflow.
		    input_data (List[Dict[str, Any]]): Input data supplied to the xAI workflow.
		
		Returns:
		    List[Dict[str, Any]]: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.messages = [ ]
			
			if input_data is not None and len( input_data ) > 0:
				self.messages.extend( input_data )
			elif context is not None and len( context ) > 0:
				for item in context:
					if not isinstance( item, dict ):
						continue
					
					role = str( item.get( 'role', '' ) or '' ).strip( )
					content = item.get( 'content', '' )
					
					if role not in [ 'user', 'assistant', 'system', 'developer' ]:
						continue
					
					if not isinstance( content, str ) or not content.strip( ):
						continue
					
					self.messages.append(
						{
								'role': role,
								'content': [
										{
												'type': 'input_text',
												'text': content.strip( ),
										},
								],
						} )
			
			self.messages.append(
				{
						'role': 'user',
						'content': [
								{
										'type': 'input_text',
										'text': prompt,
								},
						],
				} )
			
			return self.messages
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_input( self, prompt, context, input_data )'
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: List[ Any ] = None, allowed_domains: List[ str ] = None,
			vector_store_ids: List[ str ] = None ) -> List[ Dict[ str, Any ] ] | None:
		"""Build tools.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Args:
		    tools (List[Any]): Tools supplied to the xAI workflow.
		    allowed_domains (List[str]): Allowed domains supplied to the xAI workflow.
		    vector_store_ids (List[str]): Vector store ids supplied to the xAI workflow.
		
		Returns:
		    List[Dict[str, Any]] | None: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			if tools is None or len( tools ) == 0:
				return None
			
			self.built_tools = [ ]
			for tool in tools:
				if isinstance( tool, dict ):
					tool_type = str( tool.get( 'type', '' ) or '' ).strip( )
				else:
					tool_type = str( tool or '' ).strip( )
				
				if not tool_type:
					continue
				
				if tool_type == 'web_search':
					built_tool = { 'type': 'web_search' }
					if len( self.allowed_domains ) > 0:
						built_tool[ 'allowed_domains' ] = self.allowed_domains
					
					self.built_tools.append( built_tool )
					continue
				
				if tool_type == 'x_search':
					self.built_tools.append( { 'type': 'x_search' } )
					continue
				
				if tool_type in [ 'collections_search', 'file_search' ]:
					if len( self.vector_store_ids ) == 0:
						continue
					
					self.built_tools.append(
						{
								'type': 'collections_search',
								'collection_ids': self.vector_store_ids,
						} )
					continue
				
				if tool_type in [ 'code_execution', 'code_interpreter' ]:
					self.built_tools.append( { 'type': 'code_execution' } )
					continue
			
			return self.built_tools if len( self.built_tools ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, tools, allowed_domains, vector_store_ids )'
			Logger( ).write( exception )
			raise exception
	
	def build_tool_choice( self, tool_choice: str = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> str | None:
		"""Build tool choice.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Args:
		    tool_choice (str): Tool choice supplied to the xAI workflow.
		    tools (List[Dict[str, Any]]): Tools supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			if not isinstance( tool_choice, str ) or not tool_choice.strip( ):
				return None
			
			choice = tool_choice.strip( )
			if choice not in self.choice_options:
				return None
			
			if choice == 'none':
				return 'none'
			
			if tools is None or len( tools ) == 0:
				return None
			
			return choice
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_tool_choice( self, tool_choice, tools )'
			Logger( ).write( exception )
			raise exception
	
	def build_include( self, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> List[ str ] | None:
		"""Build include.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Args:
		    include (List[str]): Include supplied to the xAI workflow.
		    tools (List[Dict[str, Any]]): Tools supplied to the xAI workflow.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			if include is None or len( include ) == 0:
				return None
			
			tool_types = [ ]
			if isinstance( tools, list ):
				for tool in tools:
					if isinstance( tool, dict ) and tool.get( 'type' ):
						tool_types.append( str( tool.get( 'type' ) ) )
			
			allowed = [ ]
			for value in include:
				if not isinstance( value, str ) or not value.strip( ):
					continue
				
				name = value.strip( )
				if name in [ 'inline_citations', 'verbose_streaming', 'mcp_call_output' ]:
					allowed.append( name )
					continue
				
				if name == 'web_search_call_output' and 'web_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'x_search_call_output' and 'x_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'code_execution_call_output' and 'code_execution' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'collections_search_call_output' and 'collections_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'attachment_search_call_output' and 'collections_search' in tool_types:
					allowed.append( name )
					continue
			
			return allowed if len( allowed ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_include( self, include, tools )'
			Logger( ).write( exception )
			raise exception
	
	def build_text_format( self, format: Dict[ str, Any ] | str = None,
			response_schema: Any = None ) -> Dict[ str, Any ] | None:
		"""Build text format.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Args:
		    format (Dict[str, Any] | str): Format supplied to the xAI workflow.
		    response_schema (Any): Response schema supplied to the xAI workflow.
		
		Returns:
		    Dict[str, Any] | None: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			if format is None:
				return None
			
			if isinstance( format, dict ) and len( format ) > 0:
				if 'format' in format and isinstance( format.get( 'format' ), dict ):
					return format
				
				if 'type' in format:
					return { 'format': format }
				
				return None
			
			if isinstance( format, str ) and format.strip( ):
				value = format.strip( )
				if value == 'text':
					return { 'format': { 'type': 'text' } }
				
				if value == 'json_object':
					return { 'format': { 'type': 'json_object' } }
				
				if value == 'json_schema' and isinstance( response_schema, dict ):
					return { 'format': { 'type': 'json_schema', 'json_schema': response_schema } }
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_text_format( self, format, response_schema )'
			Logger( ).write( exception )
			raise exception
	
	def build_request( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None,
			background: bool = False, reasoning: str = None, include: List[ str ] = None,
			tools: List[ Any ] = None, allowed_domains: List[ str ] = None,
			previous_id: str = None, tool_choice: str = None, is_parallel: bool = None,
			context: List[ Dict[ str, str ] ] = None, input_data: List[ Dict[ str, Any ] ] = None,
			vector_store_ids: List[ str ] = None, conversation_id: str = None,
			response_schema: Any = None ) -> Dict[ str, Any ]:
		"""Build request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    format (Dict[str, Any]): Format supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    max_tools (int): Max tools supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    background (bool): Background supplied to the xAI workflow.
		    reasoning (str): Reasoning supplied to the xAI workflow.
		    include (List[str]): Include supplied to the xAI workflow.
		    tools (List[Any]): Tools supplied to the xAI workflow.
		    allowed_domains (List[str]): Allowed domains supplied to the xAI workflow.
		    previous_id (str): Previous id supplied to the xAI workflow.
		    tool_choice (str): Tool choice supplied to the xAI workflow.
		    is_parallel (bool): Is parallel supplied to the xAI workflow.
		    context (List[Dict[str, str]]): Context supplied to the xAI workflow.
		    input_data (List[Dict[str, Any]]): Input data supplied to the xAI workflow.
		    vector_store_ids (List[str]): Vector store ids supplied to the xAI workflow.
		    conversation_id (str): Conversation id supplied to the xAI workflow.
		    response_schema (Any): Response schema supplied to the xAI workflow.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.model = model
			self.prompt = prompt
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.store_messages = store
			self.stream = stream
			self.background = background
			self.instructions = instruct
			self.response_format = self.build_text_format( format,
				response_schema=response_schema )
			self.max_tools = max_tools
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			self.previous_id = previous_id if isinstance( previous_id, str ) else None
			self.previous_response_id = self.previous_id
			self.conversation_id = conversation_id if isinstance( conversation_id, str ) else None
			self.parallel_tools = is_parallel
			self.reasoning = self.build_reasoning( reasoning )
			self.tools = self.build_tools( tools=tools, allowed_domains=allowed_domains,
				vector_store_ids=self.vector_store_ids )
			self.tool_choice = self.build_tool_choice( tool_choice=tool_choice, tools=self.tools )
			self.include = self.build_include( include=include, tools=self.tools )
			self.input = self.build_input( prompt=prompt, context=context, input_data=input_data )
			self.request = {
					'model': self.model,
					'input': self.input,
			}
			
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			if self.reasoning is not None and self.model == 'grok-4.20-multi-agent':
				self.request[ 'reasoning' ] = self.reasoning
			
			if isinstance( self.max_output_tokens, int ) and self.max_output_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_output_tokens
			
			if self.temperature is not None:
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None:
				self.request[ 'top_p' ] = self.top_percent
			
			if self.frequency_penalty is not None:
				self.request[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None:
				self.request[ 'presence_penalty' ] = self.presence_penalty
			
			if self.store_messages is not None:
				self.request[ 'store' ] = self.store_messages
			
			# Stream and background are retained on self for layout/UI parity. This path returns
			# final text.
			if self.include is not None and len( self.include ) > 0:
				self.request[ 'include' ] = self.include
			
			if self.tools is not None and len( self.tools ) > 0:
				self.request[ 'tools' ] = self.tools
			
			if self.tool_choice:
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.parallel_tools is not None and self.tools is not None:
				self.request[ 'parallel_tool_calls' ] = self.parallel_tools
			
			if self.previous_id and self.previous_id.strip( ):
				self.request[ 'previous_response_id' ] = self.previous_id.strip( )
			
			if self.conversation_id and self.conversation_id.strip( ):
				self.request[ 'conversation' ] = self.conversation_id.strip( )
			
			if isinstance( self.max_tools, int ) and self.max_tools > 0 and self.tools is not None:
				self.request[ 'max_tool_calls' ] = self.max_tools
			
			if self.response_format is not None and len( self.response_format ) > 0:
				self.request[ 'text' ] = self.response_format
			
			return self.request
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'build_request( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str | None:
		"""Get output text.
		
		Purpose:
		    Retrieves normalized xAI provider state or response data for display, reuse,
		    or downstream request construction.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			if self.response is None:
				return None
			
			self.output_text = getattr( self.response, 'output_text', None )
			if self.output_text:
				return self.output_text
			
			if hasattr( self.response, 'output' ) and self.response.output:
				text_parts = [ ]
				for item in self.response.output:
					if getattr( item, 'type', None ) != 'message':
						continue
					
					if not hasattr( item, 'content' ) or item.content is None:
						continue
					
					for block in item.content:
						if getattr( block, 'type', None ) == 'output_text':
							text = getattr( block, 'text', None )
							if text:
								text_parts.append( text )
				
				if len( text_parts ) > 0:
					self.output_text = ''.join( text_parts ).strip( )
					return self.output_text
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'get_output_text( self ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def get_usage( self ) -> Any:
		"""Get usage.
		
		Purpose:
		    Retrieves normalized xAI provider state or response data for display, reuse,
		    or downstream request construction.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			if self.response is None:
				return None
			
			return getattr( self.response, 'usage', None )
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'get_usage( self ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, top_k: int = None,
			frequency: float = None, max_tools: int = None, presence: float = None,
			max_tokens: int = None, store: bool = None, stream: bool = None,
			instruct: str = None, background: bool = False, reasoning: str = None,
			include: List[ str ] = None, tools: List[ Any ] = None,
			allowed_domains: List[ str ] = None, previous_id: str = None,
			tool_choice: str = None, is_parallel: bool = None,
			context: List[ Dict[ str, str ] ] = None, input_data: List[ Dict[ str, Any ] ] = None,
			vector_store_ids: List[ str ] = None, conversation_id: str = None,
			response_format: Dict[ str, Any ] | str = None, response_schema: Any = None,
			number: int = None, modalities: List[ str ] = None, media_resolution: str = None,
			content: str = None, urls: List[ str ] = None, max_urls: int = None,
			safety_profile: str = None, **kwargs: Any ) -> str | None:
		"""Generate text.
		
		Purpose:
		    Executes an xAI generation workflow using validated request settings, captures the
		    provider response, and returns displayable output.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    format (Dict[str, Any]): Format supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    top_k (int): Top k supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    max_tools (int): Max tools supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    background (bool): Background supplied to the xAI workflow.
		    reasoning (str): Reasoning supplied to the xAI workflow.
		    include (List[str]): Include supplied to the xAI workflow.
		    tools (List[Any]): Tools supplied to the xAI workflow.
		    allowed_domains (List[str]): Allowed domains supplied to the xAI workflow.
		    previous_id (str): Previous id supplied to the xAI workflow.
		    tool_choice (str): Tool choice supplied to the xAI workflow.
		    is_parallel (bool): Is parallel supplied to the xAI workflow.
		    context (List[Dict[str, str]]): Context supplied to the xAI workflow.
		    input_data (List[Dict[str, Any]]): Input data supplied to the xAI workflow.
		    vector_store_ids (List[str]): Vector store ids supplied to the xAI workflow.
		    conversation_id (str): Conversation id supplied to the xAI workflow.
		    response_format (Dict[str, Any] | str): Response format supplied to the xAI workflow.
		    response_schema (Any): Response schema supplied to the xAI workflow.
		    number (int): Number supplied to the xAI workflow.
		    modalities (List[str]): Modalities supplied to the xAI workflow.
		    media_resolution (str): Media resolution supplied to the xAI workflow.
		    content (str): Content supplied to the xAI workflow.
		    urls (List[str]): Urls supplied to the xAI workflow.
		    max_urls (int): Max urls supplied to the xAI workflow.
		    safety_profile (str): Safety profile supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.client = OpenAI( api_key=cfg.XAI_API_KEY, base_url=self.base_url )
			self.number = number
			self.top_k = top_k
			self.modalities = modalities if modalities is not None else [ ]
			self.media_resolution = media_resolution
			self.content = content
			self.urls = urls if urls is not None else [ ]
			self.max_urls = max_urls
			self.safety_profile = safety_profile
			self.extra_kwargs = kwargs or { }
			self.stream_requested = bool( stream )
			self.background_requested = bool( background )
			self.request = self.build_request( prompt=prompt, model=model,
				temperature=temperature, format=response_format or format, top_p=top_p,
				frequency=frequency, max_tools=max_tools, presence=presence,
				max_tokens=max_tokens, store=store, stream=False, instruct=instruct,
				background=False, reasoning=reasoning, include=include, tools=tools,
				allowed_domains=allowed_domains, previous_id=previous_id,
				tool_choice=tool_choice, is_parallel=is_parallel, context=context,
				input_data=input_data, vector_store_ids=vector_store_ids,
				conversation_id=conversation_id, response_schema=response_schema )
			self.response = self.client.responses.create( **self.request )
			self.previous_id = getattr( self.response, 'id', None )
			self.previous_response_id = self.previous_id
			self.output_text = self.get_output_text( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def get_grounding_sources( self ) -> List[ Dict[ str, Any ] ]:
		"""Get grounding sources.
		
		Purpose:
		    Retrieves normalized xAI provider state or response data for display, reuse,
		    or downstream request construction.
		
		Returns:
		    List[Dict[str, Any]]: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			if self.response is None:
				return [ ]
			
			self.sources = [ ]
			output = getattr( self.response, 'output', None )
			if isinstance( output, list ):
				for item in output:
					item_type = getattr( item, 'type', None )
					
					if item_type in [ 'web_search_call', 'x_search_call' ]:
						action = getattr( item, 'action', None )
						raw_sources = getattr( action, 'sources', None ) if action else None
						if isinstance( raw_sources, list ):
							for source in raw_sources:
								self.sources.append(
									{
											'title': getattr( source, 'title', None ),
											'url': getattr( source, 'url', None ),
											'snippet': getattr( source, 'snippet', None ),
											'file_id': None,
									} )
					
					if item_type in [ 'collections_search_call', 'file_search_call' ]:
						results = getattr( item, 'results', None )
						if isinstance( results, list ):
							for result in results:
								self.sources.append(
									{
											'title': getattr( result, 'file_name',
												None ) or getattr(
												result, 'title', None ),
											'url': None,
											'snippet': getattr( result, 'text', None ),
											'file_id': getattr( result, 'file_id', None ),
									} )
			
			return self.sources
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = 'get_grounding_sources( self ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def answer_document( self, prompt: str, document_text: str, model: str,
			instructions: str = None, temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			store: bool = None, include: List[ str ] = None, tools: List[ str ] = None,
			tool_choice: str = None, reasoning: str = None,
			context: List[ Dict[ str, str ] ] = None,
			vector_store_ids: List[ str ] = None ) -> str | None:
		"""Answer document.
		
		Purpose:
		    Provides answer document behavior for the Chat workflow while preserving provider
		    request and response state.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    document_text (str): Document text supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    instructions (str): Instructions supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    include (List[str]): Include supplied to the xAI workflow.
		    tools (List[str]): Tools supplied to the xAI workflow.
		    tool_choice (str): Tool choice supplied to the xAI workflow.
		    reasoning (str): Reasoning supplied to the xAI workflow.
		    context (List[Dict[str, str]]): Context supplied to the xAI workflow.
		    vector_store_ids (List[str]): Vector store ids supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		
		Raises:
		    Error: Re-raised after validation or provider execution errors are wrapped and logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'document_text', document_text )
			throw_if( 'model', model )
			
			self.prompt = prompt
			self.content = document_text
			self.model = model
			self.instructions = instructions
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.store_messages = store
			self.include = include if include is not None else [ ]
			self.tool_choice = tool_choice
			self.reasoning = reasoning
			self.context = context if context is not None else [ ]
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			
			selected_tools = [ ]
			if tools is not None:
				for item in tools:
					if isinstance( item, dict ):
						selected_tools.append( item )
						continue
					
					if isinstance( item, str ) and item.strip( ):
						selected_tools.append( { 'type': item.strip( ) } )
			
			self.tools = selected_tools
			
			if isinstance( self.tool_choice, list ):
				self.tool_choice = self.tool_choice[ 0 ] if len( self.tool_choice ) > 0 else None
			
			document_prompt = (
					f'Document Context:\n'
					f'{self.content}\n\n'
					f'User Question:\n'
					f'{self.prompt}'
			)
			
			self.output_text = self.generate_text(
				prompt=document_prompt,
				model=self.model,
				temperature=self.temperature,
				top_p=self.top_percent,
				frequency=self.frequency_penalty,
				presence=self.presence_penalty,
				max_tokens=self.max_output_tokens,
				store=self.store_messages,
				stream=False,
				instruct=self.instructions,
				reasoning=self.reasoning,
				include=self.include,
				tools=self.tools,
				tool_choice=self.tool_choice,
				context=self.context,
				vector_store_ids=self.vector_store_ids )
			
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'grok'
			exception.cause = 'Chat'
			exception.method = ('answer_document( self, prompt: str, document_text: str, model: str '
			                    ') -> str | None')
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Provides dir behavior for the Chat workflow while preserving provider request and
		    response state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'api_key',
				'base_url',
				'client',
				'model',
				'prompt',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_output_tokens',
				'stops',
				'store_messages',
				'stream',
				'background',
				'number',
				'response_format',
				'context',
				'instructions',
				'include',
				'tool_choice',
				'previous_id',
				'previous_response_id',
				'conversation_id',
				'parallel_tools',
				'max_tools',
				'input',
				'tools',
				'reasoning',
				'allowed_domains',
				'max_search_results',
				'output_text',
				'vector_store_ids',
				'file_ids',
				'response',
				'file_path',
				'model_options',
				'include_options',
				'tool_options',
				'choice_options',
				'format_options',
				'reasoning_options',
				'modality_options',
				'media_options',
				'build_reasoning',
				'build_input',
				'build_tools',
				'build_tool_choice',
				'build_include',
				'build_text_format',
				'build_request',
				'get_output_text',
				'get_usage',
				'generate_text',
				'get_grounding_sources',
				'answer_documents'
		]

class TTS( Grok ):
	"""TTS workflow wrapper.
	
	Purpose:
	    Builds text-to-speech request state for audio generation workflows exposed by the
	    application.
	
	Attributes:
	    client: Runtime attribute used by the TTS workflow.
	    model: Runtime attribute used by the TTS workflow.
	    input_text: Runtime attribute used by the TTS workflow.
	    voice: Runtime attribute used by the TTS workflow.
	    language: Runtime attribute used by the TTS workflow.
	    response_format: Runtime attribute used by the TTS workflow.
	    sample_rate: Runtime attribute used by the TTS workflow.
	    bit_rate: Runtime attribute used by the TTS workflow.
	    speed: Runtime attribute used by the TTS workflow.
	    audio_path: Runtime attribute used by the TTS workflow.
	    audio_bytes: Runtime attribute used by the TTS workflow.
	    request: Runtime attribute used by the TTS workflow.
	    response: Runtime attribute used by the TTS workflow.
	"""
	client: Optional[ Any ]
	model: Optional[ str ]
	input_text: Optional[ str ]
	voice: Optional[ str ]
	language: Optional[ str ]
	response_format: Optional[ str ]
	sample_rate: Optional[ int ]
	bit_rate: Optional[ int ]
	speed: Optional[ float ]
	audio_path: Optional[ str ]
	audio_bytes: Optional[ bytes ]
	request: Optional[ Dict[ str, Any ] ]
	response: Optional[ Any ]
	
	def __init__( self, model: str = 'xai-tts' ):
		"""Initialize instance.
		
		Purpose:
		    Initializes TTS state with default configuration values and runtime attributes used by
		    later xAI provider calls.
		
		Args:
		    model (str): Model supplied to the xAI workflow.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.client = None
		self.model = model
		self.number = None
		self.input_text = None
		self.prompt = None
		self.language = 'auto'
		self.voice = 'eve'
		self.response_format = 'mp3'
		self.sample_rate = None
		self.bit_rate = None
		self.speed = 1.0
		self.instructions = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_completion_tokens = None
		self.store = None
		self.stream = None
		self.audio_path = None
		self.file_path = None
		self.request = { }
		self.response = None
		self.audio_bytes = None
		self.output_format = None
		self.optimize_streaming_latency = None
		self.text_normalization = None
		self.extra_kwargs = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Model options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'xai-tts',
		]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		"""Voice options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'eve',
				'ara',
				'rex',
				'sal',
				'leo',
		]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Language options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'auto',
				'en',
				'ar-EG',
				'ar-SA',
				'ar-AE',
				'bn',
				'zh',
				'fr',
				'de',
				'hi',
				'id',
				'it',
				'ja',
				'ko',
				'pt-BR',
				'pt-PT',
				'ru',
				'es-MX',
				'es-ES',
				'tr',
				'vi',
		]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'mp3',
				'wav',
				'pcm',
				'mulaw',
				'alaw',
		]
	
	@property
	def response_format_options( self ) -> List[ str ] | None:
		"""Response format options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return self.format_options
	
	@property
	def output_format_options( self ) -> List[ str ] | None:
		"""Output format options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return self.format_options
	
	@property
	def speed_options( self ) -> List[ float ] | None:
		"""Speed options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector without
		    mutating provider state.
		
		Returns:
		    List[float] | None: Result produced by the xAI workflow.
		"""
		return [
				0.25,
				0.50,
				0.75,
				1.00,
				1.25,
				1.50,
				2.00,
				3.00,
				4.00,
		]
	
	@property
	def sample_rate_options( self ) -> List[ int ] | None:
		"""Sample rate options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector without
		    mutating provider state.
		
		Returns:
		    List[int] | None: Result produced by the xAI workflow.
		"""
		return [
				8000,
				16000,
				22050,
				24000,
				44100,
				48000,
		]
	
	@property
	def bit_rate_options( self ) -> List[ int ] | None:
		"""Bit rate options.
		
		Purpose:
		    Returns the configured option values exposed by the TTS workflow selector without
		    mutating provider state.
		
		Returns:
		    List[int] | None: Result produced by the xAI workflow.
		"""
		return [
				32000,
				64000,
				96000,
				128000,
				192000,
		]
	
	def validate_voice( self, voice: str = None ) -> str:
		"""Validate voice.
		
		Purpose:
		    Provides validate voice behavior for the TTS workflow while preserving provider
		    request and response state.
		
		Args:
		    voice (str): Voice supplied to the xAI workflow.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			value = str( voice or 'eve' ).strip( ).lower( )
			if value not in self.voice_options:
				return 'eve'
			
			return value
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'validate_voice( self, voice: str=None ) -> str'
			raise ex
	
	def validate_language( self, language: str = None ) -> str:
		"""Validate language.
		
		Purpose:
		    Provides validate language behavior for the TTS workflow while preserving provider
		    request and response state.
		
		Args:
		    language (str): Language supplied to the xAI workflow.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			value = str( language or 'auto' ).strip( )
			valid_values = self.language_options
			
			if value not in valid_values:
				return 'auto'
			
			return value
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'validate_language( self, language: str=None ) -> str'
			raise ex
	
	def validate_format( self, format: str = None ) -> str:
		"""Validate format.
		
		Purpose:
		    Provides validate format behavior for the TTS workflow while preserving provider
		    request and response state.
		
		Args:
		    format (str): Format supplied to the xAI workflow.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			value = str( format or 'mp3' ).strip( ).lower( )
			
			if value == 'mu-law':
				value = 'mulaw'
			
			if value == 'a-law':
				value = 'alaw'
			
			if value not in self.format_options:
				return 'mp3'
			
			return value
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'validate_format( self, format: str=None ) -> str'
			raise ex
	
	def validate_sample_rate( self, sample_rate: int = None ) -> int | None:
		"""Validate sample rate.
		
		Purpose:
		    Provides validate sample rate behavior for the TTS workflow while preserving provider
		    request and response state.
		
		Args:
		    sample_rate (int): Sample rate supplied to the xAI workflow.
		
		Returns:
		    int | None: Result produced by the xAI workflow.
		"""
		try:
			if sample_rate is None:
				return None
			
			value = int( sample_rate )
			if value not in self.sample_rate_options:
				return None
			
			return value
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'validate_sample_rate( self, sample_rate: int=None ) -> int | None'
			raise ex
	
	def validate_bit_rate( self, bit_rate: int = None ) -> int | None:
		"""Validate bit rate.
		
		Purpose:
		    Provides validate bit rate behavior for the TTS workflow while preserving provider
		    request and response state.
		
		Args:
		    bit_rate (int): Bit rate supplied to the xAI workflow.
		
		Returns:
		    int | None: Result produced by the xAI workflow.
		"""
		try:
			if bit_rate is None:
				return None
			
			value = int( bit_rate )
			if value not in self.bit_rate_options:
				return None
			
			return value
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'validate_bit_rate( self, bit_rate: int=None ) -> int | None'
			raise ex
	
	def validate_speed( self, speed: float = None ) -> float:
		"""Validate speed.
		
		Purpose:
		    Provides validate speed behavior for the TTS workflow while preserving provider
		    request and response state.
		
		Args:
		    speed (float): Speed supplied to the xAI workflow.
		
		Returns:
		    float: Result produced by the xAI workflow.
		"""
		try:
			value = 1.0 if speed is None else float( speed )
			
			if value < 0.25:
				return 0.25
			
			if value > 4.0:
				return 4.0
			
			return value
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'validate_speed( self, speed: float=None ) -> float'
			raise ex
	
	def build_output_format( self ) -> Dict[ str, Any ] | None:
		"""Build output format.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any] | None: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'response_format', self.response_format )
			self.output_format = { 'codec': self.response_format, }
			
			if self.sample_rate is not None:
				self.output_format[ 'sample_rate' ] = self.sample_rate
			
			if self.response_format == 'mp3' and self.bit_rate is not None:
				self.output_format[ 'bit_rate' ] = self.bit_rate
			
			return self.output_format
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'build_output_format( self ) -> Dict[ str, Any ] | None'
			raise ex
	
	def build_request( self ) -> Dict[ str, Any ]:
		"""Build request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'input_text', self.input_text )
			throw_if( 'voice', self.voice )
			throw_if( 'language', self.language )
			self.request = {
					'text': self.input_text,
					'voice_id': self.voice,
					'language': self.language,
			}
			self.output_format = self.build_output_format( )
			
			if self.output_format:
				self.request[ 'output_format' ] = self.output_format
			
			if self.optimize_streaming_latency is not None:
				self.request[ 'optimize_streaming_latency' ] = self.optimize_streaming_latency
			
			if self.text_normalization is not None:
				self.request[ 'text_normalization' ] = self.text_normalization
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'build_request( self ) -> Dict[ str, Any ]'
			raise ex
	
	def execute_request( self ) -> Any:
		"""Execute request.
		
		Purpose:
		    Provides execute request behavior for the TTS workflow while preserving provider
		    request and response state.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'api_key', self.api_key )
			throw_if( 'base_url', self.base_url )
			throw_if( 'request', self.request )
			self.response = requests.post(
				url=f'{self.base_url.rstrip( "/" )}/tts',
				headers={ 'Authorization': f'Bearer {self.api_key}',
				          'Content-Type': 'application/json', },
				json=self.request, timeout=self.timeout or 3600, )
			self.response.raise_for_status( )
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'execute_request( self ) -> Any'
			raise ex
	
	def extract_audio( self ) -> bytes | None:
		"""Extract audio.
		
		Purpose:
		    Provides extract audio behavior for the TTS workflow while preserving provider request
		    and response state.
		
		Returns:
		    bytes | None: Result produced by the xAI workflow.
		"""
		try:
			if self.response is None:
				return None
			
			self.audio_bytes = self.response.content
			if not self.audio_bytes:
				return None
			
			if self.audio_path:
				with open( self.audio_path, 'wb' ) as target:
					target.write( self.audio_bytes )
			
			return self.audio_bytes
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'extract_audio( self ) -> bytes | None'
			raise ex
	
	def create_speech( self, text: str, model: str = 'xai-tts', format: str = 'mp3',
			speed: float = 1.0, voice: str = 'eve', instruct: str = None, file_path: str = None,
			language: str = 'auto', sample_rate: int = None, bit_rate: int = None,
			optimize_streaming_latency: int = None, text_normalization: bool = None,
			**kwargs: Any ) -> bytes | None:
		"""Create speech.
		
		Purpose:
		    Creates the requested xAI resource using validated names, paths, or configuration
		    values.
		
		Args:
		    text (str): Text supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    format (str): Format supplied to the xAI workflow.
		    speed (float): Speed supplied to the xAI workflow.
		    voice (str): Voice supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    file_path (str): File path supplied to the xAI workflow.
		    language (str): Language supplied to the xAI workflow.
		    sample_rate (int): Sample rate supplied to the xAI workflow.
		    bit_rate (int): Bit rate supplied to the xAI workflow.
		    optimize_streaming_latency (int): Optimize streaming latency supplied to the xAI
		    workflow.
		    text_normalization (bool): Text normalization supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    bytes | None: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'text', text )
			self.input_text = text
			self.prompt = text
			self.model = model or 'xai-tts'
			self.response_format = self.validate_format( format )
			self.speed = self.validate_speed( speed )
			self.voice = self.validate_voice( voice )
			self.language = self.validate_language( language )
			self.instructions = instruct
			self.audio_path = file_path
			self.file_path = file_path
			self.sample_rate = self.validate_sample_rate( sample_rate )
			self.bit_rate = self.validate_bit_rate( bit_rate )
			self.optimize_streaming_latency = optimize_streaming_latency
			self.text_normalization = text_normalization
			self.extra_kwargs = kwargs or { }
			self.build_request( )
			self.execute_request( )
			return self.extract_audio( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'create_speech( self, text: str ) -> bytes | None'
			raise ex
	
	def synthesize( self, text: str, model: str = 'xai-tts', format: str = 'mp3',
			speed: float = 1.0, voice: str = 'eve', instruct: str = None, file_path: str = None,
			language: str = 'auto', **kwargs: Any ) -> bytes | None:
		"""Synthesize.
		
		Purpose:
		    Provides synthesize behavior for the TTS workflow while preserving provider request
		    and response state.
		
		Args:
		    text (str): Text supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    format (str): Format supplied to the xAI workflow.
		    speed (float): Speed supplied to the xAI workflow.
		    voice (str): Voice supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    file_path (str): File path supplied to the xAI workflow.
		    language (str): Language supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    bytes | None: Result produced by the xAI workflow.
		"""
		try:
			return self.create_speech( text=text, model=model, format=format, speed=speed,
				voice=voice, instruct=instruct, file_path=file_path, language=language,
				**kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'synthesize( self, text: str ) -> bytes | None'
			raise ex
	
	def generate( self, text: str = None, prompt: str = None, model: str = 'xai-tts',
			format: str = 'mp3', speed: float = 1.0, voice: str = 'eve', instruct: str = None,
			file_path: str = None, language: str = 'auto', **kwargs: Any ) -> bytes | None:
		"""Generate.
		
		Purpose:
		    Provides generate behavior for the TTS workflow while preserving provider request and
		    response state.
		
		Args:
		    text (str): Text supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    format (str): Format supplied to the xAI workflow.
		    speed (float): Speed supplied to the xAI workflow.
		    voice (str): Voice supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    file_path (str): File path supplied to the xAI workflow.
		    language (str): Language supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    bytes | None: Result produced by the xAI workflow.
		"""
		try:
			input_text = text or prompt
			throw_if( 'text', input_text )
			return self.create_speech( text=input_text, model=model, format=format, speed=speed,
				voice=voice, instruct=instruct, file_path=file_path, language=language,
				**kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'TTS'
			ex.method = 'generate( self, text: str ) -> bytes | None'
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Provides dir behavior for the TTS workflow while preserving provider request and
		    response state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'api_key',
				'base_url',
				'client',
				'model',
				'number',
				'input_text',
				'prompt',
				'language',
				'voice',
				'response_format',
				'sample_rate',
				'bit_rate',
				'speed',
				'instructions',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_completion_tokens',
				'store',
				'stream',
				'audio_path',
				'file_path',
				'request',
				'response',
				'audio_bytes',
				'output_format',
				'optimize_streaming_latency',
				'text_normalization',
				'extra_kwargs',
				'model_options',
				'voice_options',
				'language_options',
				'format_options',
				'response_format_options',
				'output_format_options',
				'speed_options',
				'sample_rate_options',
				'bit_rate_options',
				'validate_voice',
				'validate_language',
				'validate_format',
				'validate_sample_rate',
				'validate_bit_rate',
				'validate_speed',
				'build_output_format',
				'build_request',
				'execute_request',
				'extract_audio',
				'create_speech',
				'synthesize',
				'generate',
		]

class Transcription( Grok ):
	"""Transcription workflow wrapper.
	
	Purpose:
	    Builds speech-to-text request state from uploaded audio files and provider model settings.
	
	Attributes:
	    client: Runtime attribute used by the Transcription workflow.
	    model: Runtime attribute used by the Transcription workflow.
	    prompt: Runtime attribute used by the Transcription workflow.
	    language: Runtime attribute used by the Transcription workflow.
	    file_path: Runtime attribute used by the Transcription workflow.
	    audio_file: Runtime attribute used by the Transcription workflow.
	    messages: Runtime attribute used by the Transcription workflow.
	    response: Runtime attribute used by the Transcription workflow.
	    transcript: Runtime attribute used by the Transcription workflow.
	    request: Runtime attribute used by the Transcription workflow.
	"""
	client: Optional[ Client ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	language: Optional[ str ]
	file_path: Optional[ str ]
	audio_file: Optional[ Any ]
	messages: Optional[ List[ Any ] ]
	response: Optional[ Any ]
	transcript: Optional[ str ]
	request: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, number: int = 1, model: str = 'grok-3-mini-fast',
			temperature: float = 0.8, top_p: float = 0.9, frequency: float = 0.0,
			presence: float = 0.0, max_tokens: int = 10000, store: bool = True,
			stream: bool = False, language: str = 'en', instruct: str = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes Transcription state with default configuration values and runtime
		    attributes used by later xAI provider calls.
		
		Args:
		    number (int): Number supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    language (str): Language supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		"""
		super( ).__init__( )
		self.api_key = os.getenv( 'XAI_API_KEY' ) or cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1' )
		self.client = None
		self.number = number
		self.model = model
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_output_tokens = max_tokens
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.language = language
		self.instructions = instruct
		self.prompt = None
		self.file_path = None
		self.audio_file = None
		self.messages = [ ]
		self.request = { }
		self.response = None
		self.chat = None
		self.transcript = None
		self.response_format = None
		self.include = [ ]
		self.extra_kwargs = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Model options.
		
		Purpose:
		    Returns the configured option values exposed by the Transcription workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
				'grok-3',
				'grok-3-latest',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-mini-fast',
		]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Language options.
		
		Purpose:
		    Returns the configured option values exposed by the Transcription workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'auto',
				'en',
				'es',
				'fr',
				'de',
				'it',
				'ja',
				'ko',
				'pt',
				'zh',
				'Tagalog',
				'French',
				'Japanese',
				'German',
				'Italian',
				'Chinese',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the Transcription workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'audio/wav',
				'audio/mp3',
				'audio/mpeg',
				'audio/mp4',
				'audio/m4a',
				'audio/webm',
				'audio/ogg',
				'audio/flac',
		]
	
	@property
	def response_format_options( self ) -> List[ str ]:
		"""Response format options.
		
		Purpose:
		    Returns the configured option values exposed by the Transcription workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'text',
				'json',
		]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Include options.
		
		Purpose:
		    Returns the configured option values exposed by the Transcription workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ ]
	
	def build_prompt( self ) -> str:
		"""Build prompt.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			if isinstance( self.prompt, str ) and self.prompt.strip( ):
				return self.prompt.strip( )
			
			language = self.language or 'auto'
			return (
					'Transcribe the attached audio file accurately. '
					f'Use the language hint "{language}" when helpful. '
					'Return only the transcript unless additional instructions require otherwise.'
			)
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Transcription'
			ex.method = 'build_prompt( self )'
			raise ex
	
	def build_messages( self ) -> List[ Any ]:
		"""Build messages.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    List[Any]: Result produced by the xAI workflow.
		"""
		try:
			self.messages = [ ]
			
			if isinstance( self.instructions, str ) and self.instructions.strip( ):
				self.messages.append( system( self.instructions.strip( ) ) )
			
			self.messages.append( user( self.build_prompt( ) ) )
			return self.messages
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Transcription'
			ex.method = 'build_messages( self )'
			raise ex
	
	def build_request( self ) -> Dict[ str, Any ]:
		"""Build request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'model', self.model )
			throw_if( 'file_path', self.file_path )
			self.build_messages( )
			self.request = {
					'model': self.model,
					'messages': self.messages,
			}
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Transcription'
			ex.method = 'build_request( self )'
			raise ex
	
	def execute_request( self ) -> Any:
		"""Execute request.
		
		Purpose:
		    Provides execute request behavior for the Transcription workflow while preserving
		    provider request and response state.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'api_key', self.api_key )
			throw_if( 'file_path', self.file_path )
			self.client = Client( api_key=cfg.XAI_API_KEY )
			with open( self.file_path, 'rb' ) as self.audio_file:
				self.chat = self.client.chat.create( file=self.audio_file, **self.request )
				self.response = self.chat.sample( )
			
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Transcription'
			ex.method = 'execute_request( self )'
			raise ex
	
	def extract_transcript( self ) -> str:
		"""Extract transcript.
		
		Purpose:
		    Provides extract transcript behavior for the Transcription workflow while preserving
		    provider request and response state.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			if self.response is None:
				return ''
			
			output_text = getattr( self.response, 'output_text', None )
			if isinstance( output_text, str ) and output_text.strip( ):
				self.transcript = output_text.strip( )
				return self.transcript
			
			text = getattr( self.response, 'text', None )
			if isinstance( text, str ) and text.strip( ):
				self.transcript = text.strip( )
				return self.transcript
			
			content = getattr( self.response, 'content', None )
			if isinstance( content, str ) and content.strip( ):
				self.transcript = content.strip( )
				return self.transcript
			
			self.transcript = str( self.response ).strip( )
			return self.transcript
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Transcription'
			ex.method = 'extract_transcript( self )'
			raise ex
	
	def transcribe( self, path: str, model: str = 'grok-3-mini-fast', language: str = 'en',
			prompt: str = None, temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None,
			response_format: str = None, include: List[ str ] = None, mime_type: str = None,
			start_time: float = None, end_time: float = None, **kwargs: Any ) -> str:
		"""Transcribe.
		
		Purpose:
		    Executes xAI transcription using validated audio input and model configuration.
		
		Args:
		    path (str): Path supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    language (str): Language supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    response_format (str): Response format supplied to the xAI workflow.
		    include (List[str]): Include supplied to the xAI workflow.
		    mime_type (str): Mime type supplied to the xAI workflow.
		    start_time (float): Start time supplied to the xAI workflow.
		    end_time (float): End time supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'model', model )
			self.file_path = path
			self.model = model
			self.language = language
			self.prompt = prompt
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_percent = top_p if top_p is not None else self.top_percent
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_output_tokens = max_tokens if max_tokens is not None else (
					self.max_output_tokens)
			self.max_completion_tokens = self.max_output_tokens
			self.store = store if store is not None else self.store
			self.stream = stream if stream is not None else self.stream
			self.instructions = instruct if instruct is not None else self.instructions
			self.response_format = response_format
			self.include = include if include is not None else [ ]
			self.mime_type = mime_type
			self.start_time = start_time
			self.end_time = end_time
			self.extra_kwargs = kwargs or { }
			self.build_request( )
			self.execute_request( )
			return self.extract_transcript( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path: str, model: str ) -> str'
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Provides dir behavior for the Transcription workflow while preserving provider request
		    and response state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'api_key',
				'base_url',
				'client',
				'number',
				'model',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_output_tokens',
				'max_completion_tokens',
				'store',
				'stream',
				'language',
				'instructions',
				'prompt',
				'file_path',
				'audio_file',
				'messages',
				'request',
				'response',
				'chat',
				'transcript',
				'response_format',
				'include',
				'extra_kwargs',
				'model_options',
				'language_options',
				'format_options',
				'response_format_options',
				'include_options',
				'build_prompt',
				'build_messages',
				'build_request',
				'execute_request',
				'extract_transcript',
				'transcribe',
		]

class Translation( Grok ):
	"""Translation workflow wrapper.
	
	Purpose:
	    Builds translation request state from source content, language values, and provider model
	    settings.
	
	Attributes:
	    client: Runtime attribute used by the Translation workflow.
	    model: Runtime attribute used by the Translation workflow.
	    prompt: Runtime attribute used by the Translation workflow.
	    target_language: Runtime attribute used by the Translation workflow.
	    source_language: Runtime attribute used by the Translation workflow.
	    file_path: Runtime attribute used by the Translation workflow.
	    audio_file: Runtime attribute used by the Translation workflow.
	    messages: Runtime attribute used by the Translation workflow.
	    response: Runtime attribute used by the Translation workflow.
	    translation: Runtime attribute used by the Translation workflow.
	    request: Runtime attribute used by the Translation workflow.
	"""
	client: Optional[ Client ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	target_language: Optional[ str ]
	source_language: Optional[ str ]
	file_path: Optional[ str ]
	audio_file: Optional[ Any ]
	messages: Optional[ List[ Any ] ]
	response: Optional[ Any ]
	translation: Optional[ str ]
	request: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'grok-3-fast', temperature: float = 0.8,
			top_p: float = 0.9, frequency: float = 0.0, presence: float = 0.0,
			max_tokens: int = 10000, store: bool = True, stream: bool = False,
			instruct: str = None ):
		"""Initialize instance.
		
		Purpose:
		    Initializes Translation state with default configuration values and runtime attributes
		    used by later xAI provider calls.
		
		Args:
		    model (str): Model supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		"""
		super( ).__init__( )
		self.api_key = os.getenv( 'XAI_API_KEY' ) or cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1' )
		self.client = None
		self.model = model
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_output_tokens = max_tokens
		self.max_completion_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.prompt = None
		self.target_language = None
		self.source_language = None
		self.file_path = None
		self.audio_file = None
		self.messages = [ ]
		self.request = { }
		self.response = None
		self.chat = None
		self.translation = None
		self.response_format = None
		self.include = [ ]
		self.mime_type = None
		self.extra_kwargs = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Model options.
		
		Purpose:
		    Returns the configured option values exposed by the Translation workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
				'grok-3',
				'grok-3-latest',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-mini-fast',
		]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Language options.
		
		Purpose:
		    Returns the configured option values exposed by the Translation workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'English',
				'Spanish',
				'French',
				'German',
				'Italian',
				'Japanese',
				'Korean',
				'Portuguese',
				'Chinese',
				'Tagalog',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the Translation workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'audio/wav',
				'audio/mp3',
				'audio/mpeg',
				'audio/mp4',
				'audio/m4a',
				'audio/webm',
				'audio/ogg',
				'audio/flac',
		]
	
	@property
	def response_format_options( self ) -> List[ str ]:
		"""Response format options.
		
		Purpose:
		    Returns the configured option values exposed by the Translation workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'text',
				'json',
		]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Include options.
		
		Purpose:
		    Returns the configured option values exposed by the Translation workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ ]
	
	def build_prompt( self ) -> str:
		"""Build prompt.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'target_language', self.target_language )
			
			if isinstance( self.prompt, str ) and self.prompt.strip( ):
				base_prompt = self.prompt.strip( )
			else:
				base_prompt = 'Translate the spoken audio in the attached file.'
			
			if self.source_language and str( self.source_language ).strip( ):
				return (
						f'{base_prompt} Source language hint: {self.source_language}. '
						f'Translate the speech into {self.target_language}. '
						'Return only the translated text unless additional instructions require '
						'otherwise.'
				)
			
			return (
					f'{base_prompt} Translate the speech into {self.target_language}. '
					'Return only the translated text unless additional instructions require '
					'otherwise.'
			)
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Translation'
			ex.method = 'build_prompt( self )'
			raise ex
	
	def build_messages( self ) -> List[ Any ]:
		"""Build messages.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    List[Any]: Result produced by the xAI workflow.
		"""
		try:
			self.messages = [ ]
			
			if isinstance( self.instructions, str ) and self.instructions.strip( ):
				self.messages.append( system( self.instructions.strip( ) ) )
			
			self.messages.append( user( self.build_prompt( ) ) )
			return self.messages
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Translation'
			ex.method = 'build_messages( self )'
			raise ex
	
	def build_request( self ) -> Dict[ str, Any ]:
		"""Build request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'model', self.model )
			throw_if( 'file_path', self.file_path )
			throw_if( 'target_language', self.target_language )
			self.build_messages( )
			self.request = {
					'model': self.model,
					'messages': self.messages,
			}
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Translation'
			ex.method = 'build_request( self )'
			raise ex
	
	def execute_request( self ) -> Any:
		"""Execute request.
		
		Purpose:
		    Provides execute request behavior for the Translation workflow while preserving
		    provider request and response state.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'api_key', self.api_key )
			throw_if( 'file_path', self.file_path )
			self.client = Client( api_key=cfg.XAI_API_KEY )
			with open( self.file_path, 'rb' ) as self.audio_file:
				self.chat = self.client.chat.create( file=self.audio_file, **self.request )
				self.response = self.chat.sample( )
			
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Translation'
			ex.method = 'execute_request( self )'
			raise ex
	
	def extract_translation( self ) -> str:
		"""Extract translation.
		
		Purpose:
		    Provides extract translation behavior for the Translation workflow while preserving
		    provider request and response state.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			if self.response is None:
				return ''
			
			output_text = getattr( self.response, 'output_text', None )
			if isinstance( output_text, str ) and output_text.strip( ):
				self.translation = output_text.strip( )
				return self.translation
			
			text = getattr( self.response, 'text', None )
			if isinstance( text, str ) and text.strip( ):
				self.translation = text.strip( )
				return self.translation
			
			content = getattr( self.response, 'content', None )
			if isinstance( content, str ) and content.strip( ):
				self.translation = content.strip( )
				return self.translation
			
			self.translation = str( self.response ).strip( )
			return self.translation
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Translation'
			ex.method = 'extract_translation( self )'
			raise ex
	
	def translate( self, path: str, model: str = 'grok-3-fast', language: str = 'English',
			prompt: str = None, source_language: str = None, temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, store: bool = None, stream: bool = None,
			instruct: str = None, response_format: str = None, include: List[ str ] = None,
			mime_type: str = None, **kwargs: Any ) -> str:
		"""Translate.
		
		Purpose:
		    Executes xAI translation using validated source content and language settings.
		
		Args:
		    path (str): Path supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    language (str): Language supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    source_language (str): Source language supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    response_format (str): Response format supplied to the xAI workflow.
		    include (List[str]): Include supplied to the xAI workflow.
		    mime_type (str): Mime type supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'language', language )
			self.file_path = path
			self.model = model
			self.target_language = language
			self.source_language = source_language
			self.prompt = prompt
			self.temperature = temperature if temperature is not None else self.temperature
			self.top_percent = top_p if top_p is not None else self.top_percent
			self.frequency_penalty = frequency if frequency is not None else self.frequency_penalty
			self.presence_penalty = presence if presence is not None else self.presence_penalty
			self.max_output_tokens = max_tokens if max_tokens is not None else (
					self.max_output_tokens)
			self.max_completion_tokens = self.max_output_tokens
			self.store = store if store is not None else self.store
			self.stream = stream if stream is not None else self.stream
			self.instructions = instruct if instruct is not None else self.instructions
			self.response_format = response_format
			self.include = include if include is not None else [ ]
			self.mime_type = mime_type
			self.extra_kwargs = kwargs or { }
			self.build_request( )
			self.execute_request( )
			return self.extract_translation( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Translation'
			ex.method = 'translate( self, path: str, model: str, language: str ) -> str'
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Provides dir behavior for the Translation workflow while preserving provider request
		    and response state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'api_key',
				'base_url',
				'client',
				'model',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_output_tokens',
				'max_completion_tokens',
				'store',
				'stream',
				'instructions',
				'prompt',
				'target_language',
				'source_language',
				'file_path',
				'audio_file',
				'messages',
				'request',
				'response',
				'chat',
				'translation',
				'response_format',
				'include',
				'mime_type',
				'extra_kwargs',
				'model_options',
				'language_options',
				'format_options',
				'response_format_options',
				'include_options',
				'build_prompt',
				'build_messages',
				'build_request',
				'execute_request',
				'extract_translation',
				'translate',
		]

class Images( Grok ):
	"""Images workflow wrapper.
	
	Purpose:
	    Builds and executes xAI image-generation and image-analysis workflows while preserving
	    prompt, model, and response state.
	
	Attributes:
	    model: Runtime attribute used by the Images workflow.
	    prompt: Runtime attribute used by the Images workflow.
	    aspect_ratio: Runtime attribute used by the Images workflow.
	    resolution: Runtime attribute used by the Images workflow.
	    response_format: Runtime attribute used by the Images workflow.
	    client: Runtime attribute used by the Images workflow.
	    image_path: Runtime attribute used by the Images workflow.
	    image_url: Runtime attribute used by the Images workflow.
	    detail: Runtime attribute used by the Images workflow.
	    response: Runtime attribute used by the Images workflow.
	    request: Runtime attribute used by the Images workflow.
	    output: Runtime attribute used by the Images workflow.
	"""
	model: Optional[ str ]
	prompt: Optional[ str ]
	aspect_ratio: Optional[ str ]
	resolution: Optional[ str ]
	response_format: Optional[ str ]
	client: Optional[ OpenAI ]
	image_path: Optional[ str ]
	image_url: Optional[ str ]
	detail: Optional[ str ]
	response: Optional[ Any ]
	request: Optional[ Dict[ str, Any ] ]
	output: Optional[ Any ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes Images state with default configuration values and runtime attributes used
		    by later xAI provider calls.
		"""
		super( ).__init__( )
		self.api_key = os.getenv( 'XAI_API_KEY' ) or cfg.XAI_API_KEY
		self.base_url = getattr( cfg, 'XAI_BASE_URL', 'https://api.x.ai/v1' )
		self.client = None
		self.model = None
		self.prompt = None
		self.number = None
		self.aspect_ratio = None
		self.resolution = None
		self.size = None
		self.quality = None
		self.style = None
		self.detail = None
		self.response_format = None
		self.mime_type = None
		self.compression = None
		self.background = None
		self.response_modalities = None
		self.max_output_tokens = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.tools = [ ]
		self.tool_choice = None
		self.include = [ ]
		self.allowed_domains = [ ]
		self.store = None
		self.stream = None
		self.is_parallel = None
		self.max_tools = None
		self.max_searches = None
		self.image_path = None
		self.image_url = None
		self.mask_path = None
		self.request = { }
		self.response = None
		self.output = None
		self.extra_body = { }
		self.extra_kwargs = { }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Model options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ 'grok-imagine-image', 'grok-2-image-1212' ]
	
	@property
	def analysis_model_options( self ) -> List[ str ]:
		"""Analysis model options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'grok-4.20-reasoning',
				'grok-4.20',
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
		]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Tool options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [ ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [ ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [ 'auto', 'required', 'none' ]
	
	@property
	def aspect_options( self ) -> List[ str ]:
		"""Aspect options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'auto',
				'1:1',
				'3:4',
				'4:3',
				'9:16',
				'16:9',
				'2:3',
				'3:2',
				'9:19.5',
				'19.5:9',
				'9:20',
				'20:9',
				'1:2',
				'2:1',
		]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Size options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ '1k', '2k' ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		"""Quality options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ 'auto', 'low', 'medium', 'high' ]
	
	@property
	def style_options( self ) -> List[ str ]:
		"""Style options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ ]
	
	@property
	def backcolor_options( self ) -> List[ str ]:
		"""Backcolor options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		"""Detail options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ 'auto', 'low', 'high' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ 'url', 'b64_json' ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Mime options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ 'url', 'b64_json' ]
	
	@property
	def output_options( self ) -> List[ str ]:
		"""Output options.
		
		Purpose:
		    Returns the configured option values exposed by the Images workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [ 'url', 'b64_json' ]
	
	def initialize_client( self ) -> None:
		"""Initialize client.
		
		Purpose:
		    Provides initialize client behavior for the Images workflow while preserving provider
		    request and response state.
		"""
		try:
			throw_if( 'api_key', self.api_key )
			throw_if( 'base_url', self.base_url )
			self.client = OpenAI( api_key=cfg.XAI_API_KEY, base_url=self.base_url )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'initialize_client( self )'
			raise ex
	
	def normalize_resolution( self, value: str = None ) -> str | None:
		"""Normalize resolution.
		
		Purpose:
		    Provides normalize resolution behavior for the Images workflow while preserving
		    provider request and response state.
		
		Args:
		    value (str): Value supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			if value is None:
				return None
			
			resolution = str( value ).strip( ).lower( )
			if resolution in [ '1k', '2k' ]:
				return resolution
			
			return None
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'normalize_resolution( self, value )'
			raise ex
	
	def normalize_response_format( self, value: str = None ) -> str | None:
		"""Normalize response format.
		
		Purpose:
		    Provides normalize response format behavior for the Images workflow while preserving
		    provider request and response state.
		
		Args:
		    value (str): Value supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			if value is None:
				return None
			
			response_format = str( value ).strip( ).lower( )
			if response_format in [ 'url', 'b64_json' ]:
				return response_format
			
			if response_format == 'base64':
				return 'b64_json'
			
			return None
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'normalize_response_format( self, value )'
			raise ex
	
	def encode_image_data_uri( self, image_path: str ) -> str:
		"""Encode image data uri.
		
		Purpose:
		    Encodes local binary content into a text representation required by xAI request
		    payloads.
		
		Args:
		    image_path (str): Image path supplied to the xAI workflow.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'image_path', image_path )
			path = Path( image_path )
			
			if not path.exists( ):
				raise FileNotFoundError( f'Image file was not found: {image_path}' )
			
			suffix = path.suffix.lower( ).replace( '.', '' )
			if suffix == 'jpg':
				suffix = 'jpeg'
			
			if suffix not in [ 'jpeg', 'png', 'webp' ]:
				suffix = 'jpeg'
			
			image_bytes = path.read_bytes( )
			encoded = base64.b64encode( image_bytes ).decode( 'utf-8' )
			return f'data:image/{suffix};base64,{encoded}'
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'encode_image_data_uri( self, image_path )'
			raise ex
	
	def get_output_text( self ) -> str | None:
		"""Get output text.
		
		Purpose:
		    Retrieves normalized xAI provider state or response data for display, reuse,
		    or downstream request construction.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			if self.response is None:
				return None
			
			output_text = getattr( self.response, 'output_text', None )
			if output_text:
				return output_text
			
			output = getattr( self.response, 'output', None )
			if not isinstance( output, list ):
				return None
			
			text_parts: List[ str ] = [ ]
			for item in output:
				if getattr( item, 'type', None ) != 'message':
					continue
				
				content = getattr( item, 'content', None )
				if not isinstance( content, list ):
					continue
				
				for block in content:
					if getattr( block, 'type', None ) == 'output_text':
						text = getattr( block, 'text', None )
						if text:
							text_parts.append( text )
			
			return ''.join( text_parts ).strip( ) if text_parts else None
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'get_output_text( self )'
			raise ex
	
	def normalize_image_result( self ) -> Any:
		"""Normalize image result.
		
		Purpose:
		    Provides normalize image result behavior for the Images workflow while preserving
		    provider request and response state.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			if self.response is None:
				return None
			
			data = getattr( self.response, 'data', None )
			if isinstance( data, list ) and len( data ) > 0:
				results: List[ Any ] = [ ]
				
				for item in data:
					url = getattr( item, 'url', None )
					b64_json = getattr( item, 'b64_json', None )
					
					if url:
						results.append( url )
						continue
					
					if b64_json:
						results.append( b64_json )
						continue
					
					results.append( item )
				
				return results[ 0 ] if len( results ) == 1 else results
			
			url = getattr( self.response, 'url', None )
			if url:
				return url
			
			base64_value = getattr( self.response, 'base64', None )
			if base64_value:
				return base64_value
			
			image_bytes = getattr( self.response, 'image', None )
			if image_bytes:
				return image_bytes
			
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'normalize_image_result( self )'
			raise ex
	
	def build_generation_request( self ) -> Dict[ str, Any ]:
		"""Build generation request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'prompt', self.prompt )
			throw_if( 'model', self.model )
			self.request = {
					'model': self.model,
					'prompt': self.prompt,
			}
			self.extra_body = { }
			
			if isinstance( self.number, int ) and self.number > 0:
				self.request[ 'n' ] = self.number
			
			if self.response_format:
				self.request[ 'response_format' ] = self.response_format
			
			if self.aspect_ratio:
				self.extra_body[ 'aspect_ratio' ] = self.aspect_ratio
			
			if self.resolution:
				self.extra_body[ 'resolution' ] = self.resolution
			
			if self.extra_body:
				self.request[ 'extra_body' ] = self.extra_body
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'build_generation_request( self )'
			raise ex
	
	def build_edit_request( self ) -> Dict[ str, Any ]:
		"""Build edit request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'prompt', self.prompt )
			throw_if( 'model', self.model )
			throw_if( 'image_url', self.image_url )
			self.request = {
					'model': self.model,
					'prompt': self.prompt,
					'image': {
							'type': 'image_url',
							'url': self.image_url,
					},
			}
			
			if self.response_format:
				self.request[ 'response_format' ] = self.response_format
			
			if self.aspect_ratio:
				self.request[ 'aspect_ratio' ] = self.aspect_ratio
			
			if self.resolution:
				self.request[ 'resolution' ] = self.resolution
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'build_edit_request( self )'
			raise ex
	
	def build_analysis_request( self ) -> Dict[ str, Any ]:
		"""Build analysis request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'prompt', self.prompt )
			throw_if( 'model', self.model )
			throw_if( 'image_url', self.image_url )
			self.request = {
					'model': self.model,
					'input': [
							{
									'role': 'user',
									'content': [
											{
													'type': 'input_image',
													'image_url': self.image_url,
											},
											{
													'type': 'input_text',
													'text': self.prompt,
											},
									],
							},
					],
			}
			
			if self.detail:
				self.request[ 'input' ][ 0 ][ 'content' ][ 0 ][ 'detail' ] = self.detail
			
			if isinstance( self.max_output_tokens, int ) and self.max_output_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_output_tokens
			
			if self.temperature is not None:
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None:
				self.request[ 'top_p' ] = self.top_percent
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'build_analysis_request( self )'
			raise ex
	
	def generate( self, prompt: str, model: str = 'grok-imagine-image', number: int = None,
			size: str = None, quality: str = None, style: str = None, fmt: str = None,
			mime_type: str = None, compression: float = None, background: str = None,
			aspect_ratio: str = None, response_modalities: str = None, temperature: float = None,
			top_p: float = None, top_k: int = None, frequency: float = None, presence: float =
			None,
			max_tokens: int = None, instruct: str = None, tools: List[ Any ] = None,
			tool_choice: str = None, include: List[ str ] = None,
			allowed_domains: List[ str ] = None,
			store: bool = None, stream: bool = None, is_parallel: bool = None,
			max_tools: int = None,
			max_searches: int = None, grounded: bool = False, image_search: bool = False,
			response_format: str = None, **kwargs: Any ) -> Any:
		"""Generate.
		
		Purpose:
		    Provides generate behavior for the Images workflow while preserving provider request
		    and response state.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    number (int): Number supplied to the xAI workflow.
		    size (str): Size supplied to the xAI workflow.
		    quality (str): Quality supplied to the xAI workflow.
		    style (str): Style supplied to the xAI workflow.
		    fmt (str): Fmt supplied to the xAI workflow.
		    mime_type (str): Mime type supplied to the xAI workflow.
		    compression (float): Compression supplied to the xAI workflow.
		    background (str): Background supplied to the xAI workflow.
		    aspect_ratio (str): Aspect ratio supplied to the xAI workflow.
		    response_modalities (str): Response modalities supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    top_k (int): Top k supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    tools (List[Any]): Tools supplied to the xAI workflow.
		    tool_choice (str): Tool choice supplied to the xAI workflow.
		    include (List[str]): Include supplied to the xAI workflow.
		    allowed_domains (List[str]): Allowed domains supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    is_parallel (bool): Is parallel supplied to the xAI workflow.
		    max_tools (int): Max tools supplied to the xAI workflow.
		    max_searches (int): Max searches supplied to the xAI workflow.
		    grounded (bool): Grounded supplied to the xAI workflow.
		    image_search (bool): Image search supplied to the xAI workflow.
		    response_format (str): Response format supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.size = size
			self.resolution = self.normalize_resolution( size )
			self.quality = quality
			self.style = style
			self.response_format = self.normalize_response_format(
				response_format or fmt or mime_type )
			self.mime_type = mime_type
			self.compression = compression
			self.background = background
			self.aspect_ratio = aspect_ratio
			self.response_modalities = response_modalities
			self.temperature = temperature
			self.top_percent = top_p
			self.top_k = top_k
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.instructions = instruct
			self.tools = tools if tools is not None else [ ]
			self.tool_choice = tool_choice
			self.include = include if include is not None else [ ]
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.store = store
			self.stream = stream
			self.is_parallel = is_parallel
			self.max_tools = max_tools
			self.max_searches = max_searches
			self.grounded = grounded
			self.image_search = image_search
			self.extra_kwargs = kwargs or { }
			self.initialize_client( )
			self.build_generation_request( )
			self.response = self.client.images.generate( **self.request )
			self.output = self.normalize_image_result( )
			return self.output
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'generate( self, prompt: str, model: str )'
			raise ex
	
	def generate_image( self, prompt: str, model: str = 'grok-imagine-image',
			number: int = None, size: str = None, quality: str = None, style: str = None,
			fmt: str = None, mime_type: str = None, compression: float = None,
			background: str = None, aspect_ratio: str = None,
			response_modalities: str = None, **kwargs: Any ) -> Any:
		"""Generate image.
		
		Purpose:
		    Executes an xAI generation workflow using validated request settings, captures the
		    provider response, and returns displayable output.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    number (int): Number supplied to the xAI workflow.
		    size (str): Size supplied to the xAI workflow.
		    quality (str): Quality supplied to the xAI workflow.
		    style (str): Style supplied to the xAI workflow.
		    fmt (str): Fmt supplied to the xAI workflow.
		    mime_type (str): Mime type supplied to the xAI workflow.
		    compression (float): Compression supplied to the xAI workflow.
		    background (str): Background supplied to the xAI workflow.
		    aspect_ratio (str): Aspect ratio supplied to the xAI workflow.
		    response_modalities (str): Response modalities supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			return self.generate( prompt=prompt, model=model, number=number, size=size,
				quality=quality, style=style, fmt=fmt, mime_type=mime_type,
				compression=compression, background=background, aspect_ratio=aspect_ratio,
				response_modalities=response_modalities, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'generate_image( self, prompt: str, model: str )'
			raise ex
	
	def create( self, prompt: str, model: str = 'grok-imagine-image', resolution: str = None,
			aspect_ratio: str = None, format: str = None, number: int = None,
			quality: str = None, style: str = None, **kwargs: Any ) -> Any:
		"""Create.
		
		Purpose:
		    Provides create behavior for the Images workflow while preserving provider request and
		    response state.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    resolution (str): Resolution supplied to the xAI workflow.
		    aspect_ratio (str): Aspect ratio supplied to the xAI workflow.
		    format (str): Format supplied to the xAI workflow.
		    number (int): Number supplied to the xAI workflow.
		    quality (str): Quality supplied to the xAI workflow.
		    style (str): Style supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			return self.generate( prompt=prompt, model=model, number=number,
				size=resolution, quality=quality, style=style, fmt=format,
				aspect_ratio=aspect_ratio, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'create( self, prompt: str, model: str )'
			raise ex
	
	def create_image( self, prompt: str, model: str = 'grok-imagine-image',
			**kwargs: Any ) -> Any:
		"""Create image.
		
		Purpose:
		    Creates the requested xAI resource using validated names, paths, or configuration
		    values.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			return self.generate( prompt=prompt, model=model, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'create_image( self, prompt: str, model: str )'
			raise ex
	
	def edit( self, image_path: str = None, prompt: str = None, model: str = 'grok-imagine-image',
			aspect_ratio: str = None, resolution: str = None, quality: str = None,
			response_format: str = None, path: str = None, mask_path: str = None, mask: str = None,
			size: str = None, fmt: str = None, mime_type: str = None, **kwargs: Any ) -> Any:
		"""Edit.
		
		Purpose:
		    Provides edit behavior for the Images workflow while preserving provider request and
		    response state.
		
		Args:
		    image_path (str): Image path supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    aspect_ratio (str): Aspect ratio supplied to the xAI workflow.
		    resolution (str): Resolution supplied to the xAI workflow.
		    quality (str): Quality supplied to the xAI workflow.
		    response_format (str): Response format supplied to the xAI workflow.
		    path (str): Path supplied to the xAI workflow.
		    mask_path (str): Mask path supplied to the xAI workflow.
		    mask (str): Mask supplied to the xAI workflow.
		    size (str): Size supplied to the xAI workflow.
		    fmt (str): Fmt supplied to the xAI workflow.
		    mime_type (str): Mime type supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			self.image_path = image_path or path
			throw_if( 'image_path', self.image_path )
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = prompt
			self.model = model
			self.aspect_ratio = aspect_ratio
			self.resolution = self.normalize_resolution( resolution or size )
			self.quality = quality
			self.response_format = self.normalize_response_format(
				response_format or fmt or mime_type )
			self.mask_path = mask_path or mask
			self.extra_kwargs = kwargs or { }
			
			if str( self.image_path ).startswith( 'http://' ) or str(
					self.image_path ).startswith( 'https://' ):
				self.image_url = str( self.image_path )
			elif str( self.image_path ).startswith( 'data:image/' ):
				self.image_url = str( self.image_path )
			else:
				self.image_url = self.encode_image_data_uri( self.image_path )
			
			self.build_edit_request( )
			self.response = requests.post(
				url=f'{self.base_url.rstrip( "/" )}/images/edits',
				headers={
						'Authorization': f'Bearer {self.api_key}',
						'Content-Type': 'application/json',
				},
				json=self.request,
				timeout=self.timeout or 3600,
			)
			self.response.raise_for_status( )
			self.output = self.response.json( )
			return self.output
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'edit( self, image_path: str, prompt: str )'
			raise ex
	
	def edit_image( self, image_path: str = None, prompt: str = None,
			model: str = 'grok-imagine-image', **kwargs: Any ) -> Any:
		"""Edit image.
		
		Purpose:
		    Provides edit image behavior for the Images workflow while preserving provider request
		    and response state.
		
		Args:
		    image_path (str): Image path supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			return self.edit( image_path=image_path, prompt=prompt, model=model, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'edit_image( self, image_path: str, prompt: str )'
			raise ex
	
	def modify( self, image_path: str = None, prompt: str = None,
			model: str = 'grok-imagine-image', **kwargs: Any ) -> Any:
		"""Modify.
		
		Purpose:
		    Provides modify behavior for the Images workflow while preserving provider request and
		    response state.
		
		Args:
		    image_path (str): Image path supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			return self.edit( image_path=image_path, prompt=prompt, model=model, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'modify( self, image_path: str, prompt: str )'
			raise ex
	
	def generate_edit( self, image_path: str = None, prompt: str = None,
			model: str = 'grok-imagine-image', **kwargs: Any ) -> Any:
		"""Generate edit.
		
		Purpose:
		    Executes an xAI generation workflow using validated request settings, captures the
		    provider response, and returns displayable output.
		
		Args:
		    image_path (str): Image path supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			return self.edit( image_path=image_path, prompt=prompt, model=model, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'generate_edit( self, image_path: str, prompt: str )'
			raise ex
	
	def analyze( self, prompt: str, image_url: str = None, model: str = 'grok-4.20-reasoning',
			max_output_tokens: int = 10000, temperature: float = None, top_p: float = None,
			detail: str = 'high', image_path: str = None, path: str = None, store: bool = False,
			**kwargs: Any ) -> str | None:
		"""Analyze.
		
		Purpose:
		    Provides analyze behavior for the Images workflow while preserving provider request
		    and response state.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    image_url (str): Image url supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    max_output_tokens (int): Max output tokens supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    detail (str): Detail supplied to the xAI workflow.
		    image_path (str): Image path supplied to the xAI workflow.
		    path (str): Path supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = prompt
			self.model = model
			self.image_path = image_path or path
			self.detail = detail
			self.max_output_tokens = max_output_tokens
			self.temperature = temperature
			self.top_percent = top_p
			self.store = store
			self.extra_kwargs = kwargs or { }
			
			if image_url:
				self.image_url = image_url
			elif self.image_path:
				self.image_url = self.encode_image_data_uri( self.image_path )
			else:
				raise ValueError( 'Either image_url, image_path, or path is required.' )
			
			self.initialize_client( )
			self.build_analysis_request( )
			self.response = self.client.responses.create( **self.request )
			self.output = self.get_output_text( )
			return self.output
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'analyze( self, prompt: str, image_url: str )'
			raise ex
	
	def analyze_image( self, prompt: str, image_url: str = None, model: str =
	'grok-4.20-reasoning',
			image_path: str = None, path: str = None, **kwargs: Any ) -> str | None:
		"""Analyze image.
		
		Purpose:
		    Provides analyze image behavior for the Images workflow while preserving provider
		    request and response state.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    image_url (str): Image url supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    image_path (str): Image path supplied to the xAI workflow.
		    path (str): Path supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			return self.analyze( prompt=prompt, image_url=image_url, model=model,
				image_path=image_path, path=path, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'analyze_image( self, prompt: str, image_url: str )'
			raise ex
	
	def vision( self, prompt: str, image_url: str = None, model: str = 'grok-4.20-reasoning',
			image_path: str = None, path: str = None, **kwargs: Any ) -> str | None:
		"""Vision.
		
		Purpose:
		    Provides vision behavior for the Images workflow while preserving provider request and
		    response state.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    image_url (str): Image url supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    image_path (str): Image path supplied to the xAI workflow.
		    path (str): Path supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			return self.analyze( prompt=prompt, image_url=image_url, model=model,
				image_path=image_path, path=path, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'vision( self, prompt: str, image_url: str )'
			raise ex
	
	def describe( self, prompt: str, image_url: str = None, model: str = 'grok-4.20-reasoning',
			image_path: str = None, path: str = None, **kwargs: Any ) -> str | None:
		"""Describe.
		
		Purpose:
		    Provides describe behavior for the Images workflow while preserving provider request
		    and response state.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    image_url (str): Image url supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    image_path (str): Image path supplied to the xAI workflow.
		    path (str): Path supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			return self.analyze( prompt=prompt, image_url=image_url, model=model,
				image_path=image_path, path=path, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'describe( self, prompt: str, image_url: str )'
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Provides dir behavior for the Images workflow while preserving provider request and
		    response state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'api_key',
				'base_url',
				'client',
				'model',
				'prompt',
				'number',
				'aspect_ratio',
				'resolution',
				'size',
				'quality',
				'style',
				'detail',
				'response_format',
				'mime_type',
				'compression',
				'background',
				'response_modalities',
				'max_output_tokens',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'tools',
				'tool_choice',
				'include',
				'allowed_domains',
				'store',
				'stream',
				'is_parallel',
				'max_tools',
				'max_searches',
				'image_path',
				'image_url',
				'mask_path',
				'request',
				'response',
				'output',
				'extra_body',
				'extra_kwargs',
				'model_options',
				'analysis_model_options',
				'tool_options',
				'include_options',
				'choice_options',
				'aspect_options',
				'size_options',
				'quality_options',
				'style_options',
				'backcolor_options',
				'detail_options',
				'format_options',
				'mime_options',
				'output_options',
				'initialize_client',
				'normalize_resolution',
				'normalize_response_format',
				'encode_image_data_uri',
				'get_output_text',
				'normalize_image_result',
				'build_generation_request',
				'build_edit_request',
				'build_analysis_request',
				'generate',
				'generate_image',
				'create',
				'create_image',
				'edit',
				'edit_image',
				'modify',
				'generate_edit',
				'analyze',
				'analyze_image',
				'vision',
				'describe',
		]

class Files( Grok ):
	"""Files workflow wrapper.
	
	Purpose:
	    Manages xAI file upload, retrieval, listing, deletion, and metadata workflows used by
	    document and provider operations.
	
	Attributes:
	    client: Runtime attribute used by the Files workflow.
	    api_key: Runtime attribute used by the Files workflow.
	    base_url: Runtime attribute used by the Files workflow.
	    file_path: Runtime attribute used by the Files workflow.
	    file_name: Runtime attribute used by the Files workflow.
	    file_id: Runtime attribute used by the Files workflow.
	    file_ids: Runtime attribute used by the Files workflow.
	    purpose: Runtime attribute used by the Files workflow.
	    model: Runtime attribute used by the Files workflow.
	    prompt: Runtime attribute used by the Files workflow.
	    instructions: Runtime attribute used by the Files workflow.
	    request: Runtime attribute used by the Files workflow.
	    response: Runtime attribute used by the Files workflow.
	    content: Runtime attribute used by the Files workflow.
	    output_text: Runtime attribute used by the Files workflow.
	    documents: Runtime attribute used by the Files workflow.
	"""
	client: Optional[ OpenAI ]
	api_key: Optional[ str ]
	base_url: Optional[ str ]
	file_path: Optional[ str ]
	file_name: Optional[ str ]
	file_id: Optional[ str ]
	file_ids: Optional[ List[ str ] ]
	purpose: Optional[ str ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	instructions: Optional[ str ]
	request: Optional[ Dict[ str, Any ] ]
	response: Optional[ Any ]
	content: Optional[ Any ]
	output_text: Optional[ str ]
	documents: Optional[ Dict[ str, str ] ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes Files state with default configuration values and runtime attributes used
		    by later xAI provider calls.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.client = None
		self.model = None
		self.instructions = None
		self.prompt = None
		self.response = None
		self.request = { }
		self.content = None
		self.output_text = None
		self.file_id = None
		self.file_ids = [ ]
		self.file_path = None
		self.file_name = None
		self.file_paths = [ ]
		self.file_names = [ ]
		self.purpose = 'assistants'
		self.response_format = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_output_tokens = None
		self.store = None
		self.stream = None
		self.include = [ ]
		self.tools = [ ]
		self.tool_choice = None
		self.previous_id = None
		self.previous_response_id = None
		self.conversation_id = None
		self.limit = None
		self.next_token = None
		self.order = None
		self.sort_by = None
		self.filter = None
		self.team_id = None
		self.download_format = None
		self.page_number = None
		self.extra_kwargs = { }
		self.documents = {
				'AccountBalances.csv': 'file_4731bb8c-d8ff-48c0-9dae-3092fbcab214',
				'SF133.csv': 'file_41037cc2-e1f4-4cce-b25a-5c1d1f0172b2',
				'Authority.csv': 'file_cbde06d5-988b-483f-880c-441613bfe54f',
				'Outlays.csv': 'file_78479189-7d47-4edb-9abc-2931172430e9',
		}
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Model options.
		
		Purpose:
		    Returns the configured option values exposed by the Files workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'grok-4.20-reasoning',
				'grok-4.20',
				'grok-4',
				'grok-4-latest',
				'grok-4-fast-reasoning',
				'grok-4-fast-non-reasoning',
				'grok-code-fast-1',
				'grok-3',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-mini-fast',
		]
	
	@property
	def purpose_options( self ) -> List[ str ]:
		"""Purpose options.
		
		Purpose:
		    Returns the configured option values exposed by the Files workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'assistants',
				'batch',
				'fine-tune',
				'user_data',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		Purpose:
		    Returns the configured option values exposed by the Files workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'text',
				'json_object',
		]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Tool options.
		
		Purpose:
		    Returns the configured option values exposed by the Files workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'code_interpreter',
		]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Include options.
		
		Purpose:
		    Returns the configured option values exposed by the Files workflow selector without
		    mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'code_execution_call_output',
		]
	
	def initialize_client( self ) -> None:
		"""Initialize client.
		
		Purpose:
		    Provides initialize client behavior for the Files workflow while preserving provider
		    request and response state.
		"""
		try:
			throw_if( 'api_key', self.api_key )
			throw_if( 'base_url', self.base_url )
			self.client = OpenAI( api_key=cfg.XAI_API_KEY, base_url=self.base_url )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'initialize_client( self )'
			raise ex
	
	def build_headers( self ) -> Dict[ str, str ]:
		"""Build headers.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, str]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'api_key', self.api_key )
			return {
					'Authorization': f'Bearer {self.api_key}',
			}
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'build_headers( self )'
			raise ex
	
	def build_json_headers( self ) -> Dict[ str, str ]:
		"""Build json headers.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, str]: Result produced by the xAI workflow.
		"""
		try:
			headers = self.build_headers( )
			headers[ 'Content-Type' ] = 'application/json'
			return headers
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'build_json_headers( self )'
			raise ex
	
	def normalize_file_id( self, response: Any = None ) -> str | None:
		"""Normalize file id.
		
		Purpose:
		    Provides normalize file id behavior for the Files workflow while preserving provider
		    request and response state.
		
		Args:
		    response (Any): Response supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			value = response if response is not None else self.response
			
			if value is None:
				return None
			
			if isinstance( value, dict ):
				file_id = value.get( 'id' ) or value.get( 'file_id' )
				return str( file_id ) if file_id else None
			
			file_id = getattr( value, 'id', None ) or getattr( value, 'file_id', None )
			return str( file_id ) if file_id else None
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'normalize_file_id( self, response )'
			raise ex
	
	def get_output_text( self ) -> str | None:
		"""Get output text.
		
		Purpose:
		    Retrieves normalized xAI provider state or response data for display, reuse,
		    or downstream request construction.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			if self.response is None:
				return None
			
			output_text = getattr( self.response, 'output_text', None )
			if output_text:
				self.output_text = output_text
				return self.output_text
			
			output = getattr( self.response, 'output', None )
			if not isinstance( output, list ):
				return None
			
			parts: List[ str ] = [ ]
			for item in output:
				if getattr( item, 'type', None ) != 'message':
					continue
				
				content = getattr( item, 'content', None )
				if not isinstance( content, list ):
					continue
				
				for block in content:
					text = getattr( block, 'text', None )
					if text:
						parts.append( text )
			
			self.output_text = ''.join( parts ).strip( ) if parts else None
			return self.output_text
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'get_output_text( self )'
			raise ex
	
	def build_upload_request( self ) -> Dict[ str, Any ]:
		"""Build upload request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'file_path', self.file_path )
			path = Path( self.file_path )
			
			if not path.exists( ):
				raise FileNotFoundError( f'File was not found: {self.file_path}' )
			
			self.file_name = self.file_name or path.name
			throw_if( 'file_name', self.file_name )
			self.request = {
					'file_path': str( path ),
					'file_name': self.file_name,
					'purpose': self.purpose,
			}
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'build_upload_request( self )'
			raise ex
	
	def execute_upload( self ) -> Any:
		"""Execute upload.
		
		Purpose:
		    Provides execute upload behavior for the Files workflow while preserving provider
		    request and response state.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			self.initialize_client( )
			throw_if( 'file_path', self.file_path )
			throw_if( 'purpose', self.purpose )
			with open( self.file_path, 'rb' ) as file_stream:
				self.response = self.client.files.create( file=file_stream, purpose=self.purpose )
			
			self.file_id = self.normalize_file_id( self.response )
			if self.file_id and self.file_id not in self.file_ids:
				self.file_ids.append( self.file_id )
			
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'execute_upload( self )'
			raise ex
	
	def build_list_request( self ) -> Dict[ str, Any ]:
		"""Build list request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			self.request = { }
			
			if self.team_id:
				self.request[ 'team_id' ] = self.team_id
			
			if isinstance( self.limit, int ) and self.limit > 0:
				self.request[ 'limit' ] = self.limit
			
			if self.next_token:
				self.request[ 'next_token' ] = self.next_token
			
			if self.order:
				self.request[ 'order' ] = self.order
			
			if self.sort_by:
				self.request[ 'sort_by' ] = self.sort_by
			
			if self.filter:
				self.request[ 'filter' ] = self.filter
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'build_list_request( self )'
			raise ex
	
	def execute_list( self ) -> Any:
		"""Execute list.
		
		Purpose:
		    Provides execute list behavior for the Files workflow while preserving provider
		    request and response state.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'base_url', self.base_url )
			response = requests.get( url=f'{self.base_url.rstrip( "/" )}/files',
				headers=self.build_headers( ), params=self.request, timeout=self.timeout or 3600 )
			response.raise_for_status( )
			self.response = response.json( )
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'execute_list( self )'
			raise ex
	
	def build_retrieve_request( self ) -> Dict[ str, Any ]:
		"""Build retrieve request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'file_id', self.file_id )
			self.request = { }
			
			if self.team_id:
				self.request[ 'team_id' ] = self.team_id
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'build_retrieve_request( self )'
			raise ex
	
	def execute_retrieve( self ) -> Any:
		"""Execute retrieve.
		
		Purpose:
		    Provides execute retrieve behavior for the Files workflow while preserving provider
		    request and response state.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'base_url', self.base_url )
			throw_if( 'file_id', self.file_id )
			response = requests.get( url=f'{self.base_url.rstrip( "/" )}/files/{self.file_id}',
				headers=self.build_headers( ), params=self.request, timeout=self.timeout or 3600 )
			response.raise_for_status( )
			self.response = response.json( )
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'execute_retrieve( self )'
			raise ex
	
	def build_extract_request( self ) -> Dict[ str, Any ]:
		"""Build extract request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'file_id', self.file_id )
			self.request = { }
			
			if self.team_id:
				self.request[ 'team_id' ] = self.team_id
			
			if self.download_format:
				self.request[ 'format' ] = self.download_format
			
			if isinstance( self.page_number, int ) and self.page_number > 0:
				self.request[ 'page_number' ] = self.page_number
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'build_extract_request( self )'
			raise ex
	
	def execute_extract( self ) -> bytes | str | None:
		"""Execute extract.
		
		Purpose:
		    Provides execute extract behavior for the Files workflow while preserving provider
		    request and response state.
		
		Returns:
		    bytes | str | None: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'base_url', self.base_url )
			throw_if( 'file_id', self.file_id )
			response = requests.get(
				url=f'{self.base_url.rstrip( "/" )}/files/{self.file_id}/content',
				headers=self.build_headers( ), params=self.request, timeout=self.timeout or 3600 )
			response.raise_for_status( )
			content_type = str( response.headers.get( 'content-type', '' ) ).lower( )
			if 'application/json' in content_type:
				self.content = response.json( )
				return self.content
			
			if self.download_format == 'DOWNLOAD_FORMAT_TEXT' or 'text/' in content_type:
				self.content = response.text
				return self.content
			
			self.content = response.content
			return self.content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'execute_extract( self )'
			raise ex
	
	def build_delete_request( self ) -> Dict[ str, Any ]:
		"""Build delete request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'file_id', self.file_id )
			self.request = { }
			
			if self.team_id:
				self.request[ 'team_id' ] = self.team_id
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'build_delete_request( self )'
			raise ex
	
	def execute_delete( self ) -> Any:
		"""Execute delete.
		
		Purpose:
		    Provides execute delete behavior for the Files workflow while preserving provider
		    request and response state.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'base_url', self.base_url )
			throw_if( 'file_id', self.file_id )
			response = requests.delete( url=f'{self.base_url.rstrip( "/" )}/files/{self.file_id}',
				headers=self.build_headers( ), params=self.request, timeout=self.timeout or 3600 )
			response.raise_for_status( )
			if response.content:
				try:
					self.response = response.json( )
				except Exception:
					self.response = { 'id': self.file_id, 'deleted': True }
			else:
				self.response = { 'id': self.file_id, 'deleted': True }
			
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'execute_delete( self )'
			raise ex
	
	def build_file_input( self ) -> List[ Dict[ str, Any ] ]:
		"""Build file input.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    List[Dict[str, Any]]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'prompt', self.prompt )
			self.content = [ {
					'type': 'input_text',
					'text': self.prompt,
			}, ]
			
			for file_id in self.file_ids:
				if not isinstance( file_id, str ) or not file_id.strip( ):
					continue
				
				self.content.append( {
						'type': 'input_file',
						'file_id': file_id.strip( ),
				} )
			
			return self.content
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'build_file_input( self )'
			raise ex
	
	def build_file_response_request( self ) -> Dict[ str, Any ]:
		"""Build file response request.
		
		Purpose:
		    Builds normalized xAI request configuration from validated inputs and stores the
		    resulting state on the instance for provider execution.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'model', self.model )
			throw_if( 'prompt', self.prompt )
			if not isinstance( self.file_ids, list ) or len( self.file_ids ) == 0:
				raise ValueError( 'At least one file_id is required for a file-aware response.' )
			
			self.request = { 'model': self.model, 'input': [ {
					'role': 'user',
					'content': self.build_file_input( ),
			} ], }
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			if isinstance( self.max_output_tokens, int ) and self.max_output_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_output_tokens
			
			if self.temperature is not None:
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None:
				self.request[ 'top_p' ] = self.top_percent
			
			if self.frequency_penalty is not None:
				self.request[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None:
				self.request[ 'presence_penalty' ] = self.presence_penalty
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			if self.include:
				self.request[ 'include' ] = self.include
			
			if self.tools:
				self.request[ 'tools' ] = self.tools
			
			if self.tool_choice:
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.previous_id:
				self.request[ 'previous_response_id' ] = self.previous_id
			
			if self.conversation_id:
				self.request[ 'conversation' ] = self.conversation_id
			
			if self.response_format:
				self.request[ 'text' ] = { 'format': { 'type': self.response_format, } }
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'build_file_response_request( self )'
			raise ex
	
	def execute_file_response( self ) -> str | None:
		"""Execute file response.
		
		Purpose:
		    Provides execute file response behavior for the Files workflow while preserving
		    provider request and response state.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			self.initialize_client( )
			self.response = self.client.responses.create( **self.request )
			return self.get_output_text( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'execute_file_response( self )'
			raise ex
	
	def upload( self, filepath: str, filename: str = None, purpose: str = 'assistants',
			**kwargs: Any ) -> Any:
		"""Upload.
		
		Purpose:
		    Provides upload behavior for the Files workflow while preserving provider request and
		    response state.
		
		Args:
		    filepath (str): Filepath supplied to the xAI workflow.
		    filename (str): Filename supplied to the xAI workflow.
		    purpose (str): Purpose supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'filepath', filepath )
			self.file_path = filepath
			self.file_name = filename
			self.purpose = purpose or 'assistants'
			self.extra_kwargs = kwargs or { }
			self.build_upload_request( )
			return self.execute_upload( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'upload( self, filepath: str, filename: str=None )'
			raise ex
	
	def list( self, limit: int = None, next_token: str = None, order: str = None,
			sort_by: str = None, filter: str = None, team_id: str = None, **kwargs: Any ) -> Any:
		"""List.
		
		Purpose:
		    Provides list behavior for the Files workflow while preserving provider request and
		    response state.
		
		Args:
		    limit (int): Limit supplied to the xAI workflow.
		    next_token (str): Next token supplied to the xAI workflow.
		    order (str): Order supplied to the xAI workflow.
		    sort_by (str): Sort by supplied to the xAI workflow.
		    filter (str): Filter supplied to the xAI workflow.
		    team_id (str): Team id supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			self.limit = limit
			self.next_token = next_token
			self.order = order
			self.sort_by = sort_by
			self.filter = filter
			self.team_id = team_id
			self.extra_kwargs = kwargs or { }
			self.build_list_request( )
			return self.execute_list( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'list( self )'
			raise ex
	
	def list_files( self, limit: int = None, next_token: str = None, order: str = None,
			sort_by: str = None, filter: str = None, team_id: str = None, **kwargs: Any ) -> Any:
		"""List files.
		
		Purpose:
		    Lists xAI resources and returns normalized metadata for UI display or downstream
		    selection.
		
		Args:
		    limit (int): Limit supplied to the xAI workflow.
		    next_token (str): Next token supplied to the xAI workflow.
		    order (str): Order supplied to the xAI workflow.
		    sort_by (str): Sort by supplied to the xAI workflow.
		    filter (str): Filter supplied to the xAI workflow.
		    team_id (str): Team id supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			return self.list( limit=limit, next_token=next_token, order=order,
				sort_by=sort_by, filter=filter, team_id=team_id, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'list_files( self )'
			raise ex
	
	def retrieve( self, file_id: str, team_id: str = None, **kwargs: Any ) -> Any:
		"""Retrieve.
		
		Purpose:
		    Provides retrieve behavior for the Files workflow while preserving provider request
		    and response state.
		
		Args:
		    file_id (str): File id supplied to the xAI workflow.
		    team_id (str): Team id supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.team_id = team_id
			self.extra_kwargs = kwargs or { }
			self.build_retrieve_request( )
			return self.execute_retrieve( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'retrieve( self, file_id: str )'
			raise ex
	
	def extract( self, file_id: str, format: str = None, page_number: int = None,
			team_id: str = None, **kwargs: Any ) -> bytes | str | None:
		"""Extract.
		
		Purpose:
		    Provides extract behavior for the Files workflow while preserving provider request and
		    response state.
		
		Args:
		    file_id (str): File id supplied to the xAI workflow.
		    format (str): Format supplied to the xAI workflow.
		    page_number (int): Page number supplied to the xAI workflow.
		    team_id (str): Team id supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    bytes | str | None: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.download_format = format
			self.page_number = page_number
			self.team_id = team_id
			self.extra_kwargs = kwargs or { }
			self.build_extract_request( )
			return self.execute_extract( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'extract( self, file_id: str )'
			raise ex
	
	def download( self, file_id: str, format: str = None, page_number: int = None,
			team_id: str = None, **kwargs: Any ) -> bytes | str | None:
		"""Download.
		
		Purpose:
		    Provides download behavior for the Files workflow while preserving provider request
		    and response state.
		
		Args:
		    file_id (str): File id supplied to the xAI workflow.
		    format (str): Format supplied to the xAI workflow.
		    page_number (int): Page number supplied to the xAI workflow.
		    team_id (str): Team id supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    bytes | str | None: Result produced by the xAI workflow.
		"""
		try:
			return self.extract( file_id=file_id, format=format, page_number=page_number,
				team_id=team_id, **kwargs )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'download( self, file_id: str )'
			raise ex
	
	def delete( self, file_id: str, team_id: str = None, **kwargs: Any ) -> Any:
		"""Delete.
		
		Purpose:
		    Provides delete behavior for the Files workflow while preserving provider request and
		    response state.
		
		Args:
		    file_id (str): File id supplied to the xAI workflow.
		    team_id (str): Team id supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'file_id', file_id )
			self.file_id = file_id
			self.team_id = team_id
			self.extra_kwargs = kwargs or { }
			self.build_delete_request( )
			return self.execute_delete( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'delete( self, file_id: str )'
			raise ex
	
	def summarize( self, filepath: str = None, filename: str = None, prompt: str = None,
			file_id: str = None, model: str = 'grok-4.20-reasoning', temperature: float = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, store: bool = None, stream: bool = None,
			instruct: str = None, include: List[ str ] = None, tools: List[ Any ] = None,
			tool_choice: str = None, previous_id: str = None, conversation_id: str = None,
			purpose: str = 'assistants', **kwargs: Any ) -> str | None:
		"""Summarize.
		
		Purpose:
		    Provides summarize behavior for the Files workflow while preserving provider request
		    and response state.
		
		Args:
		    filepath (str): Filepath supplied to the xAI workflow.
		    filename (str): Filename supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    file_id (str): File id supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    include (List[str]): Include supplied to the xAI workflow.
		    tools (List[Any]): Tools supplied to the xAI workflow.
		    tool_choice (str): Tool choice supplied to the xAI workflow.
		    previous_id (str): Previous id supplied to the xAI workflow.
		    conversation_id (str): Conversation id supplied to the xAI workflow.
		    purpose (str): Purpose supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			self.instructions = instruct
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.include = include if include is not None else [ ]
			self.tools = tools if tools is not None else [ ]
			self.tool_choice = tool_choice
			self.previous_id = previous_id
			self.previous_response_id = previous_id
			self.conversation_id = conversation_id
			self.extra_kwargs = kwargs or { }
			self.file_ids = [ ]
			
			if file_id:
				self.file_id = file_id
				self.file_ids.append( file_id )
			
			if filepath:
				upload_response = self.upload( filepath=filepath, filename=filename,
					purpose=purpose )
				uploaded_id = self.normalize_file_id( upload_response )
				if uploaded_id and uploaded_id not in self.file_ids:
					self.file_ids.append( uploaded_id )
			
			self.build_file_response_request( )
			return self.execute_file_response( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'summarize( self, filepath: str, prompt: str )'
			raise ex
	
	def survey( self, filepaths: List[ str ], filenames: List[ str ], prompt: str,
			model: str = 'grok-4.20-reasoning', temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None,
			purpose: str = 'assistants', **kwargs: Any ) -> str | None:
		"""Survey.
		
		Purpose:
		    Provides survey behavior for the Files workflow while preserving provider request and
		    response state.
		
		Args:
		    filepaths (List[str]): Filepaths supplied to the xAI workflow.
		    filenames (List[str]): Filenames supplied to the xAI workflow.
		    prompt (str): Prompt supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		    temperature (float): Temperature supplied to the xAI workflow.
		    top_p (float): Top p supplied to the xAI workflow.
		    frequency (float): Frequency supplied to the xAI workflow.
		    presence (float): Presence supplied to the xAI workflow.
		    max_tokens (int): Max tokens supplied to the xAI workflow.
		    store (bool): Store supplied to the xAI workflow.
		    stream (bool): Stream supplied to the xAI workflow.
		    instruct (str): Instruct supplied to the xAI workflow.
		    purpose (str): Purpose supplied to the xAI workflow.
		    **kwargs: Additional keyword values supplied to the xAI workflow.
		
		Returns:
		    str | None: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'filepaths', filepaths )
			throw_if( 'filenames', filenames )
			throw_if( 'prompt', prompt )
			
			if len( filepaths ) != len( filenames ):
				raise ValueError( 'filepaths and filenames must have the same length.' )
			
			self.file_ids = [ ]
			for index, filepath in enumerate( filepaths ):
				upload_response = self.upload( filepath=filepath, filename=filenames[ index ],
					purpose=purpose )
				uploaded_id = self.normalize_file_id( upload_response )
				if uploaded_id and uploaded_id not in self.file_ids:
					self.file_ids.append( uploaded_id )
			
			self.prompt = prompt
			self.model = model
			self.instructions = instruct
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_output_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.extra_kwargs = kwargs or { }
			self.build_file_response_request( )
			return self.execute_file_response( )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'survey( self, filepaths: List[ str ], filenames: List[ str ] )'
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Provides dir behavior for the Files workflow while preserving provider request and
		    response state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'api_key',
				'base_url',
				'client',
				'file_path',
				'file_name',
				'file_id',
				'file_ids',
				'file_paths',
				'file_names',
				'purpose',
				'model',
				'prompt',
				'instructions',
				'response',
				'request',
				'content',
				'output_text',
				'documents',
				'response_format',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_output_tokens',
				'store',
				'stream',
				'include',
				'tools',
				'tool_choice',
				'previous_id',
				'previous_response_id',
				'conversation_id',
				'limit',
				'next_token',
				'order',
				'sort_by',
				'filter',
				'team_id',
				'download_format',
				'page_number',
				'extra_kwargs',
				'model_options',
				'purpose_options',
				'format_options',
				'tool_options',
				'include_options',
				'initialize_client',
				'build_headers',
				'build_json_headers',
				'normalize_file_id',
				'get_output_text',
				'build_upload_request',
				'execute_upload',
				'build_list_request',
				'execute_list',
				'build_retrieve_request',
				'execute_retrieve',
				'build_extract_request',
				'execute_extract',
				'build_delete_request',
				'execute_delete',
				'build_file_input',
				'build_file_response_request',
				'execute_file_response',
				'upload',
				'list',
				'list_files',
				'retrieve',
				'extract',
				'download',
				'delete',
				'summarize',
				'survey',
		]

class VectorStores( Grok ):
	"""VectorStores workflow wrapper.
	
	Purpose:
	    Manages xAI collection and vector-store style operations used to connect documents to
	    retrieval-enabled workflows.
	
	Attributes:
	    client: Runtime attribute used by the VectorStores workflow.
	    model: Runtime attribute used by the VectorStores workflow.
	    prompt: Runtime attribute used by the VectorStores workflow.
	    response_format: Runtime attribute used by the VectorStores workflow.
	    number: Runtime attribute used by the VectorStores workflow.
	    content: Runtime attribute used by the VectorStores workflow.
	    name: Runtime attribute used by the VectorStores workflow.
	    file_path: Runtime attribute used by the VectorStores workflow.
	    file_name: Runtime attribute used by the VectorStores workflow.
	    file_ids: Runtime attribute used by the VectorStores workflow.
	    store_ids: Runtime attribute used by the VectorStores workflow.
	    store_id: Runtime attribute used by the VectorStores workflow.
	    collection_ids: Runtime attribute used by the VectorStores workflow.
	    collection_id: Runtime attribute used by the VectorStores workflow.
	    documents: Runtime attribute used by the VectorStores workflow.
	    collections: Runtime attribute used by the VectorStores workflow.
	    response: Runtime attribute used by the VectorStores workflow.
	"""
	client: Optional[ Client ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	number: Optional[ int ]
	content: Optional[ str ]
	name: Optional[ str ]
	file_path: Optional[ str ]
	file_name: Optional[ str ]
	file_ids: Optional[ List[ str ] ]
	store_ids: Optional[ List[ str ] ]
	store_id: Optional[ str ]
	collection_ids: Optional[ List[ str ] ]
	collection_id: Optional[ str ]
	documents: Optional[ Dict[ str, str ] ]
	collections: Optional[ Dict[ str, str ] ]
	response: Optional[ Any ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
		    Initializes VectorStores state with default configuration values and runtime
		    attributes used by later xAI provider calls.
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.client = Client( api_key=cfg.XAI_API_KEY )
		self.model = None
		self.prompt = None
		self.response_format = None
		self.number = None
		self.content = None
		self.name = None
		self.response = None
		self.file_ids = [ ]
		self.store_ids = [ ]
		self.collection_ids = [ ]
		self.file_path = None
		self.file_name = None
		self.store_id = None
		self.collection_id = None
		default_collections = {
				'Federal Financial Regulations': 'collection_9195d847-03a1-443c-9240-294c64dd01e2',
				'Federal Financial Data': 'collection_e28cdcc2-a9e5-430a-bdf5-94fbaf44b6a4',
				'Explanatory Statements': 'collection_41dc3374-24d0-4692-819c-59e3d7b11b93',
				'Public Laws': 'collection_c1d0b83e-2f59-4f10-9cf7-51392b490fee'
		}
		configured_collections = getattr( cfg, 'GROK_COLLECTIONS', None )
		self.collections = configured_collections if isinstance( configured_collections,
			dict ) else default_collections
		
		default_documents = {
				'Outlays.csv': 'file_b0a448b3-904a-40c7-bae1-64df657fde1c',
				'Authority.csv': 'file_c6ad236f-0c52-45f4-8883-d3be032d07c2',
				'Balances.csv': 'file_0f63d120-406f-49e6-97e5-7855f2cb26b5'
		}
		configured_documents = getattr( cfg, 'GROK_DOCUMENTS', None )
		self.documents = configured_documents if isinstance( configured_documents,
			dict ) else default_documents
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Model options.
		
		Purpose:
		    Returns the configured option values exposed by the VectorStores workflow selector
		    without mutating provider state.
		
		Returns:
		    List[str]: Result produced by the xAI workflow.
		"""
		return [
				'grok-4',
				'grok-4-0709',
				'grok-4-latest',
				'grok-4-1-fast',
				'grok-4-1-fast-reasoning',
				'grok-4-1-fast-reasoning-latest',
				'grok-4-1-fast-non-reasoning',
				'grok-4-1-fast-non-reasoning-latest',
				'grok-4-fast',
				'grok-4-fast-reasoning',
				'grok-4-fast-reasoning-latest',
				'grok-4-fast-non-reasoning',
				'grok-4-fast-non-reasoning-latest',
				'grok-code-fast-1',
				'grok-3',
				'grok-3-latest',
				'grok-3-mini',
				'grok-3-fast',
				'grok-3-fast-latest',
				'grok-3-mini-fast',
				'grok-3-mini-fast-latest',
		]
	
	def get_collection_id( self, store_id: str ) -> str:
		"""Get collection id.
		
		Purpose:
		    Retrieves normalized xAI provider state or response data for display, reuse,
		    or downstream request construction.
		
		Args:
		    store_id (str): Store id supplied to the xAI workflow.
		
		Returns:
		    str: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'store_id', store_id )
			value = str( store_id ).strip( )
			
			if value in self.collections:
				return self.collections[ value ]
			
			if ' — ' in value:
				return value.split( ' — ' )[ -1 ].strip( )
			
			return value
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'get_collection_id( self, store_id: str ) -> str'
			raise ex
	
	def get_collection_rows( self ) -> List[ Dict[ str, Any ] ]:
		"""Get collection rows.
		
		Purpose:
		    Retrieves normalized xAI provider state or response data for display, reuse,
		    or downstream request construction.
		
		Returns:
		    List[Dict[str, Any]]: Result produced by the xAI workflow.
		"""
		try:
			rows = [ ]
			for name, collection_id in self.collections.items( ):
				rows.append(
					{
							'id': collection_id,
							'name': name,
							'display_name': name,
							'description': '',
							'status': 'configured',
							'file_counts': '',
							'usage_bytes': '',
							'collection_id': collection_id,
							'collection_name': name,
							'collection_description': '',
					}
				)
			
			return rows
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'get_collection_rows( self ) -> List[ Dict[ str, Any ] ]'
			raise ex
	
	def get_text_output( self, response: Any ) -> Any:
		"""Get text output.
		
		Purpose:
		    Retrieves normalized xAI provider state or response data for display, reuse,
		    or downstream request construction.
		
		Args:
		    response (Any): Response supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			if response is None:
				return None
			
			output_text = getattr( response, 'output_text', None )
			if output_text:
				return output_text
			
			text = getattr( response, 'text', None )
			if text:
				return text
			
			if isinstance( response, dict ):
				output_text = response.get( 'output_text' ) or response.get( 'text' )
				if output_text:
					return output_text
			
			return response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'get_text_output( self, response: Any ) -> Any'
			raise ex
	
	def raise_management_required( self, operation: str ) -> None:
		"""Raise management required.
		
		Purpose:
		    Provides raise management required behavior for the VectorStores workflow while
		    preserving provider request and response state.
		
		Args:
		    operation (str): Operation supplied to the xAI workflow.
		"""
		raise NotImplementedError(
			f'Grok VectorStores.{operation} requires xAI collection-management capability. '
			f'This wrapper is currently configured with XAI_API_KEY only. Use configured '
			f'collections for search, or add the required management credential/path before '
			f'enabling remote collection management.'
		)
	
	def create( self, name: str, model: str = None ) -> Any:
		"""Create.
		
		Purpose:
		    Provides create behavior for the VectorStores workflow while preserving provider
		    request and response state.
		
		Args:
		    name (str): Name supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'name', name )
			self.name = name
			self.file_name = name
			self.model = model
			self.raise_management_required( 'create' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'create( self, name: str, model: str=None ) -> Any'
			raise ex
	
	def list( self ) -> List[ Dict[ str, Any ] ]:
		"""List.
		
		Purpose:
		    Provides list behavior for the VectorStores workflow while preserving provider request
		    and response state.
		
		Returns:
		    List[Dict[str, Any]]: Result produced by the xAI workflow.
		"""
		try:
			self.response = self.get_collection_rows( )
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'list( self ) -> List[ Dict[ str, Any ] ]'
			raise ex
	
	def retrieve( self, store_id: str ) -> Dict[ str, Any ]:
		"""Retrieve.
		
		Purpose:
		    Provides retrieve behavior for the VectorStores workflow while preserving provider
		    request and response state.
		
		Args:
		    store_id (str): Store id supplied to the xAI workflow.
		
		Returns:
		    Dict[str, Any]: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.collection_id = self.get_collection_id( self.store_id )
			display_name = ''
			for name, collection_id in self.collections.items( ):
				if collection_id == self.collection_id:
					display_name = name
					break
			
			self.response = {
					'id': self.collection_id,
					'name': display_name or self.collection_id,
					'display_name': display_name or self.collection_id,
					'description': '',
					'status': 'configured',
					'file_counts': '',
					'usage_bytes': '',
					'collection_id': self.collection_id,
					'collection_name': display_name or self.collection_id,
					'collection_description': '',
			}
			return self.response
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'retrieve( self, store_id: str ) -> Dict[ str, Any ]'
			raise ex
	
	def search( self, prompt: str, store_id: str, model: str = 'grok-4-fast' ) -> Any:
		"""Search.
		
		Purpose:
		    Provides search behavior for the VectorStores workflow while preserving provider
		    request and response state.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    store_id (str): Store id supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_id', store_id )
			self.prompt = prompt
			self.model = model
			self.store_id = store_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.store_ids = [ self.collection_id ]
			self.collection_ids = [ self.collection_id ]
			self.response = self.client.collections.search( query=self.prompt,
				collection_ids=self.collection_ids )
			return self.get_text_output( self.response )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'search( self, prompt: str, store_id: str, model: str ) -> Any'
			raise ex
	
	def survey( self, prompt: str, store_ids: List[ str ], model: str = 'grok-4-fast' ) -> Any:
		"""Survey.
		
		Purpose:
		    Provides survey behavior for the VectorStores workflow while preserving provider
		    request and response state.
		
		Args:
		    prompt (str): Prompt supplied to the xAI workflow.
		    store_ids (List[str]): Store ids supplied to the xAI workflow.
		    model (str): Model supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_ids', store_ids )
			self.prompt = prompt
			self.model = model
			self.store_ids = store_ids
			self.collection_ids = [
					self.get_collection_id( store_id )
					for store_id in self.store_ids
			]
			self.response = self.client.collections.search( query=self.prompt,
				collection_ids=self.collection_ids )
			return self.get_text_output( self.response )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'survey( self, prompt: str, store_ids: List[ str ], model: str ) -> Any'
			raise ex
	
	def update( self, store_id: str, filepath: str = None, filename: str = None ) -> Any:
		"""Update.
		
		Purpose:
		    Provides update behavior for the VectorStores workflow while preserving provider
		    request and response state.
		
		Args:
		    store_id (str): Store id supplied to the xAI workflow.
		    filepath (str): Filepath supplied to the xAI workflow.
		    filename (str): Filename supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.file_path = filepath
			self.file_name = filename
			self.raise_management_required( 'update' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = ('update( self, store_id: str, filepath: str=None, filename: str=None ) -> '
			             'Any')
			raise ex
	
	def delete( self, store_id: str ) -> Any:
		"""Delete.
		
		Purpose:
		    Provides delete behavior for the VectorStores workflow while preserving provider
		    request and response state.
		
		Args:
		    store_id (str): Store id supplied to the xAI workflow.
		
		Returns:
		    Any: Result produced by the xAI workflow.
		"""
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.collection_id = self.get_collection_id( self.store_id )
			self.raise_management_required( 'delete' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'VectorStores'
			ex.method = 'delete( self, store_id: str ) -> Any'
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		Purpose:
		    Provides dir behavior for the VectorStores workflow while preserving provider request
		    and response state.
		
		Returns:
		    List[str] | None: Result produced by the xAI workflow.
		"""
		return [
				'client',
				'file_path',
				'file_name',
				'response',
				'model',
				'prompt',
				'response_format',
				'number',
				'content',
				'name',
				'file_ids',
				'store_ids',
				'store_id',
				'collection_ids',
				'collection_id',
				'documents',
				'collections',
				'model_options',
				'get_collection_id',
				'get_collection_rows',
				'get_text_output',
				'raise_management_required',
				'create',
				'list',
				'retrieve',
				'search',
				'survey',
				'update',
				'delete',
		]
