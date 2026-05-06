'''
  ******************************************************************************************
      Assembly:                Boo
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
from boogr import ErrorDialog, Error
import config as cfg
from openai import OpenAI
from xai_sdk.aio.image import ImageResponse
from xai_sdk import Client
from xai_sdk.chat import user, system, image, file

def encode_image( image_path: str ) -> str:
	"""Encodes a local image to a base64 string for vision API requests."""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

def throw_if( name: str, value: object ) -> None:
	"""
	
		Purpose:
		--------
		Validate that a required value is not empty.
		
		Parameters:
		-----------
		name (str): Name of the argument being validated.
		value (object): Value to validate.
		
		Returns:
		--------
		None
		
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be None.' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty.' )

class Grok( ):
	"""
	
		Purpose:
		--------
		Base class for xAI (Grok) REST API functionality.
	
		This class provides:
			- API key and base URL management
			- Common request headers
			- Shared HTTP helpers for JSON and streaming requests
	
		Notes:
		------
		xAI exposes an OpenAI-compatible REST surface at:
			https://api.x.ai/v1
	
		All child capability classes (Chat, Images, Embeddings, Files, etc.)
		are expected to route through the helpers defined here.
	
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
		"""
			
			Purpose:
			--------
			Initialize the Grok (xAI) API client.
	
			Parameters:
			-----------
			cfg : object
				Configuration object providing API credentials and options.
			
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
	"""
	
	    Purpose:
	    --------
	    Provides a wrapper around the xAI Responses API for text-generation,
	    retrieval-augmented, and tool-enabled Grok chat workflows.

	    Attributes:
	    -----------
	    include:
	        Optional Responses API include fields.

	    tool_choice:
	        Optional Responses API tool-choice policy.

	    previous_id:
	        Optional previous response identifier used for stateful Responses API calls.

	    conversation_id:
	        Optional Responses API conversation identifier.

	    parallel_tools:
	        Optional flag allowing parallel tool calls.

	    max_tools:
	        Optional maximum number of tool calls.

	    input:
	        Responses API input payload.

	    tools:
	        Normalized xAI Responses API tool definitions.

	    reasoning:
	        Optional xAI Responses API reasoning configuration.

	    allowed_domains:
	        Optional list of web-search allowed domains.

	    output_text:
	        Text output from the most recent response.

	    vector_store_ids:
	        xAI Collection identifiers used for collections/file search.

	    file_ids:
	        File identifiers retained for compatibility.

	    response:
	        Last Responses API response object.

	    Methods:
	    --------
	    generate_text:
	        Generates a text response through the xAI Responses API.

	    build_reasoning:
	        Builds a valid xAI Responses API reasoning object.

	    build_input:
	        Builds the Responses API input payload.

	    build_tools:
	        Builds valid built-in xAI Responses API tool objects.

	    build_tool_choice:
	        Builds a safe tool-choice value based on the final tool list.

	    build_include:
	        Filters include values to a conservative supported subset.

	    build_text_format:
	        Builds the Responses API text-format object.

	    build_request:
	        Builds the full Responses API request dictionary.

	    get_output_text:
	        Extracts output text from a completed response.

	    get_usage:
	        Returns usage metadata from the last response.

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
			top_p: float = None, presense: float = None, presence: float = None, store: bool = None,
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
		"""
		
			Purpose:
			--------
			Initialize a Grok Chat wrapper instance with optional Responses API defaults.

			Parameters:
			-----------
			model: str
				Default xAI model name.

			prompt: str
				Optional default user prompt.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			presense: float
				Backward-compatible misspelled presence penalty argument.

			presence: float
				Optional presence penalty value.

			store: bool
				Optional Responses API store flag.

			stream: bool
				Optional stream flag retained for compatibility.

			stops: List[ str ]
				Optional stop sequences retained for compatibility.

			response_format: Dict[ str, Any ]
				Optional Responses API text formatting object.

			number: int
				Optional number retained for compatibility.

			instruct: str
				Optional system/developer instructions.

			context: List[ Dict[ str, str ] ]
				Optional prior message context.

			allowed_domains: List[ str ]
				Optional web-search allowed-domain list.

			include: List[ str ]
				Optional include fields.

			tools: List[ Dict[ str, Any ] ]
				Optional tool definitions or selected tool-name dictionaries.

			max_tools: int
				Optional maximum tool-call count.

			tool_choice: str
				Optional tool-choice policy.

			file_path: str
				Optional file path retained for compatibility.

			background: bool
				Optional background flag retained for compatibility.

			is_parallel: bool
				Optional parallel tool-call flag.

			max_tokens: int
				Optional maximum output token count.

			frequency: float
				Optional frequency penalty value.

			input: List[ Dict[ str, Any ] ]
				Optional prebuilt Responses API input payload.

			file_ids: List[ str ]
				Optional file identifiers retained for compatibility.

			previous_id: str
				Optional previous response identifier.

			conversation_id: str
				Optional Responses API conversation identifier.

			reasoning: Dict[ str, str ] | str
				Optional reasoning configuration.

			output_text: str
				Optional output text retained for compatibility.

			max_search_results: int
				Optional maximum search-result count retained for compatibility.

			content: str
				Optional content retained for compatibility.

			vector_store_ids: List[ str ]
				Optional xAI Collection identifiers used by collections/file search.

			Returns:
			--------
			None
		
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
		'''
		
			Purpose:
			--------
			Return xAI text-capable model names used by the Text mode selector.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Model option names.

		'''
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
		'''
		
			Purpose:
			--------
			Return conservative xAI Responses API include options supported by Text mode.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Include option names.

		'''
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
		'''
		
			Purpose:
			--------
			Return built-in xAI tool options that Text mode can configure.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Tool option names.

		'''
		return [
				'web_search',
				'x_search',
				'collections_search',
				'code_execution',
		]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return supported tool-choice policies.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Tool-choice option names.

		'''
		return [ 'auto', 'required', 'none', ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return Text mode response-format options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Response-format names.

		'''
		return [
				'text',
				'json_object',
				'json_schema',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return xAI reasoning effort options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Reasoning effort names.

		'''
		return [
				'none',
				'low',
				'medium',
				'high',
				'xhigh',
		]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return modality options retained for Text UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Modality names.

		'''
		return [ 'text', ]
	
	@property
	def media_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return media-resolution options retained for Text UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Media-resolution option names.

		'''
		return [ 'auto', ]
	
	def build_reasoning( self, reasoning: str | Dict[ str, str ] = None ) -> Dict[
		                                                                         str, str ] | None:
		"""
		
			Purpose:
			--------
			Create a valid xAI Responses API reasoning object from a string or dictionary.

			Parameters:
			-----------
			reasoning: str | Dict[ str, str ]
				Reasoning effort string or prebuilt reasoning dictionary.

			Returns:
			--------
			Dict[ str, str ] | None:
				Reasoning object or None.

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
			raise exception
	
	def build_input( self, prompt: str, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""
		
			Purpose:
			--------
			Create the Responses API input payload for text-generation requests.

			Parameters:
			-----------
			prompt: str
				User prompt submitted to the Responses API.

			context: List[ Dict[ str, str ] ]
				Prior user/assistant/developer/system messages.

			input_data: List[ Dict[ str, Any ] ]
				Optional prebuilt Responses API input objects.

			Returns:
			--------
			List[ Dict[ str, Any ] ]:
				Responses API input payload.

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
			raise exception
	
	def build_tools( self, tools: List[ Any ] = None, allowed_domains: List[ str ] = None,
			vector_store_ids: List[ str ] = None ) -> List[ Dict[ str, Any ] ] | None:
		"""
		
			Purpose:
			--------
			Normalize supported built-in xAI Responses API tool objects for Text mode.

			Parameters:
			-----------
			tools: List[ Any ]
				Tool strings or dictionaries selected by the application UI.

			allowed_domains: List[ str ]
				Optional list of allowed domains for web_search.

			vector_store_ids: List[ str ]
				Optional xAI Collection IDs used by collections_search.

			Returns:
			--------
			List[ Dict[ str, Any ] ] | None:
				Normalized tool dictionaries or None.

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
			raise exception
	
	def build_tool_choice( self, tool_choice: str = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> str | None:
		"""
		
			Purpose:
			--------
			Build a safe tool-choice value based on the final normalized tool list.

			Parameters:
			-----------
			tool_choice: str
				Requested tool-choice policy.

			tools: List[ Dict[ str, Any ] ]
				Final normalized tool list.

			Returns:
			--------
			str | None:
				Tool-choice policy or None.

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
			raise exception
	
	def build_include( self, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Filter include values to a conservative subset supported by selected tools.

			Parameters:
			-----------
			include: List[ str ]
				Requested include values.

			tools: List[ Dict[ str, Any ] ]
				Final normalized tool list.

			Returns:
			--------
			List[ str ] | None:
				Filtered include values or None.

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
			raise exception
	
	def build_text_format( self, format: Dict[ str, Any ] | str = None,
			response_schema: Any = None ) -> Dict[ str, Any ] | None:
		"""
		
			Purpose:
			--------
			Build or validate a Responses API text-format object.

			Parameters:
			-----------
			format: Dict[ str, Any ] | str
				Response format dictionary or response format name.

			response_schema: Any
				Optional JSON schema for json_schema output.

			Returns:
			--------
			Dict[ str, Any ] | None:
				Responses API text-format object or None.

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
		"""
		
			Purpose:
			--------
			Create a normalized xAI Responses API request payload for text generation.
	
			Parameters:
			-----------
			prompt: str
				User prompt submitted to the model.
	
			model: str
				xAI model identifier.
	
			temperature: float
				Optional sampling temperature.
	
			format: Dict[ str, Any ]
				Optional Responses API text formatting object.
	
			top_p: float
				Optional nucleus sampling value.
	
			frequency: float
				Optional frequency penalty.
	
			max_tools: int
				Optional maximum number of tool calls.
	
			presence: float
				Optional presence penalty.
	
			max_tokens: int
				Optional maximum output token count.
	
			store: bool
				Optional flag controlling whether xAI stores the response.
	
			stream: bool
				Optional stream flag retained for compatibility. This non-streaming wrapper path
				does not send stream=True.
	
			instruct: str
				Optional system or developer instructions.
	
			background: bool
				Optional background flag retained for compatibility. This immediate wrapper path
				does not send background=True.
	
			reasoning: str
				Optional reasoning effort value.
	
			include: List[ str ]
				Optional Responses API include fields.
	
			tools: List[ Any ]
				Optional tool dictionaries or tool names.
	
			allowed_domains: List[ str ]
				Optional web_search allowed-domain filters.
	
			previous_id: str
				Optional previous response ID.
	
			tool_choice: str
				Optional tool-choice policy.
	
			is_parallel: bool
				Optional flag allowing parallel tool calls.
	
			context: List[ Dict[ str, str ] ]
				Optional conversation context.
	
			input_data: List[ Dict[ str, Any ] ]
				Optional prebuilt Responses API input items.
	
			vector_store_ids: List[ str ]
				Optional xAI Collection IDs for collections_search.
	
			conversation_id: str
				Optional Responses API conversation identifier.
	
			response_schema: Any
				Optional JSON schema for structured output.
	
			Returns:
			--------
			Dict[ str, Any ]:
				Responses API request dictionary.
	
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
			self.response_format = self.build_text_format( format, response_schema=response_schema )
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
			
			# Stream and background are retained on self for layout/UI parity. This path returns final text.
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
			raise exception
	
	def get_output_text( self ) -> str | None:
		"""
		
			Purpose:
			--------
			Return text output from the last completed Responses API call.

			Parameters:
			-----------
			None

			Returns:
			--------
			str | None:
				Output text when available.

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
			raise exception
	
	def get_usage( self ) -> Any:
		"""
		
			Purpose:
			--------
			Return usage metadata from the last Responses API call.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				Usage metadata when available.

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
		"""
		
			Purpose:
			--------
			Generate a text response through the xAI Responses API.

			Parameters:
			-----------
			prompt: str
				User prompt submitted to the Responses API.

			model: str
				xAI model name.

			temperature: float
				Optional sampling temperature.

			format: Dict[ str, Any ]
				Optional Responses API text formatting object.

			top_p: float
				Optional nucleus sampling value.

			top_k: int
				Optional top-k value retained for compatibility.

			frequency: float
				Optional frequency penalty value.

			max_tools: int
				Optional maximum number of tool calls.

			presence: float
				Optional presence penalty value.

			max_tokens: int
				Optional maximum output token value.

			store: bool
				Optional Responses API store flag.

			stream: bool
				Optional Responses API stream flag. This non-streaming wrapper path does
				not send stream=True.

			instruct: str
				Optional system or developer instructions.

			background: bool
				Optional background execution flag. This immediate wrapper path does not
				send background=True.

			reasoning: str
				Optional reasoning effort value.

			include: List[ str ]
				Optional include fields returned by the Responses API.

			tools: List[ Any ]
				Optional built-in tool names or definitions.

			allowed_domains: List[ str ]
				Optional web-search domain allowlist.

			previous_id: str
				Optional previous response identifier.

			tool_choice: str
				Optional tool-choice mode.

			is_parallel: bool
				Optional parallel tool-call flag.

			context: List[ Dict[ str, str ] ]
				Optional prior conversation context.

			input_data: List[ Dict[ str, Any ] ]
				Optional prebuilt Responses API input payload.

			vector_store_ids: List[ str ]
				Optional xAI Collection identifiers used by collections_search.

			conversation_id: str
				Optional Responses API conversation identifier.

			response_format: Dict[ str, Any ] | str
				Optional Boo UI response-format value.

			response_schema: Any
				Optional structured-output JSON schema.

			number: int
				Optional number retained for UI compatibility.

			modalities: List[ str ]
				Optional modalities retained for UI compatibility.

			media_resolution: str
				Optional media resolution retained for UI compatibility.

			content: str
				Optional content retained for UI compatibility.

			urls: List[ str ]
				Optional URLs retained for UI compatibility.

			max_urls: int
				Optional maximum URL count retained for UI compatibility.

			safety_profile: str
				Optional safety profile retained for UI compatibility.

			**kwargs: Any
				Additional provider-neutral UI arguments.

			Returns:
			--------
			str | None
				Assistant output text when available.

		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.client = OpenAI( api_key=self.api_key, base_url=self.base_url )
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
			raise exception
	
	def get_grounding_sources( self ) -> List[ Dict[ str, Any ] ]:
		"""
		
			Purpose:
			--------
			Return source/citation records from the current response when available.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ Dict[ str, Any ] ]:
				Source dictionaries.

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
			raise exception
	
	def answer_document( self, prompt: str, document_text: str, model: str,
			instructions: str = None, temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			store: bool = None, include: List[ str ] = None, tools: List[ str ] = None,
			tool_choice: str = None, reasoning: str = None,
			context: List[ Dict[ str, str ] ] = None,
			vector_store_ids: List[ str ] = None ) -> str | None:
		"""
		
			Purpose:
			--------
			Answer a user question against extracted document text using the Grok Chat
			Responses API wrapper.

			Parameters:
			-----------
			prompt: str
				User question about the document.

			document_text: str
				Extracted document text used as grounding context.

			model: str
				Grok model name.

			instructions: str
				Optional system or developer instructions.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			frequency: float
				Optional frequency penalty.

			presence: float
				Optional presence penalty.

			max_tokens: int
				Optional maximum output token count.

			store: bool
				Optional Responses API store flag.

			include: List[str]
				Optional include values.

			tools: List[str]
				Optional selected tool names.

			tool_choice: str
				Optional tool-choice policy.

			reasoning: str
				Optional reasoning effort.

			context: List[Dict[str, str]]
				Optional prior conversation context.

			vector_store_ids: List[str]
				Optional collection identifiers used by file-search tooling.

			Returns:
			--------
			str | None
				Assistant answer text when available.
		
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
			exception.method = 'answer_document( self, prompt: str, document_text: str, model: str ) -> str | None'
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return member names for inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Member names.

		'''
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
	"""
	
		Purpose:
		--------
		Provide text-to-speech audio generation using the xAI Text to Speech REST API.

		Attributes:
		-----------
		api_key:
			xAI API key used for authorization.

		base_url:
			xAI REST API base URL.

		model:
			Logical model/API label retained for UI compatibility.

		input_text:
			Text submitted for speech synthesis.

		voice:
			xAI TTS voice identifier.

		language:
			BCP-47 language code or auto.

		response_format:
			Audio codec/output format.

		sample_rate:
			Optional output sample rate.

		bit_rate:
			Optional MP3 bit rate.

		speed:
			Optional playback speed retained for UI compatibility.

		audio_path:
			Optional destination file path.

		audio_bytes:
			Generated audio bytes returned by the API.

		request:
			JSON request payload sent to the xAI TTS endpoint.

		response:
			Raw requests.Response object.

		Methods:
		--------
		create_speech:
			Generate speech audio from text.

		synthesize:
			Alias for create_speech.

		generate:
			Alias for create_speech.

		build_output_format:
			Build the xAI output_format object.

		build_request:
			Build the xAI TTS request payload.

		execute_request:
			Execute the xAI TTS REST request.

		extract_audio:
			Extract raw audio bytes from the response.

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
		"""
		
			Purpose:
			--------
			Initialize the Grok Text to Speech wrapper.

			Parameters:
			-----------
			model: str
				Logical TTS model/API label retained for UI compatibility.

			Returns:
			--------
			None
		
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
		"""
		
			Purpose:
			--------
			Return xAI Text to Speech model/API labels for the Audio UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				TTS model/API labels.
		
		"""
		return [
				'xai-tts',
		]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Return xAI Text to Speech voice identifiers.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Voice identifiers.
		
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
		"""
		
			Purpose:
			--------
			Return xAI Text to Speech language options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Language codes.
		
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
		"""
		
			Purpose:
			--------
			Return xAI Text to Speech output codec options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Output codec options.
		
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
		"""
		
			Purpose:
			--------
			Return xAI Text to Speech response format options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Response format options.
		
		"""
		return self.format_options
	
	@property
	def output_format_options( self ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Return xAI Text to Speech output format options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Output format options.
		
		"""
		return self.format_options
	
	@property
	def speed_options( self ) -> List[ float ] | None:
		"""
		
			Purpose:
			--------
			Return playback speed options retained for Audio UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[float] | None:
				Playback speed values.
		
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
		"""
		
			Purpose:
			--------
			Return supported xAI Text to Speech sample rates.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[int] | None:
				Sample rate options.
		
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
		"""
		
			Purpose:
			--------
			Return supported MP3 bit rates.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[int] | None:
				Bit rate options.
		
		"""
		return [
				32000,
				64000,
				96000,
				128000,
				192000,
		]
	
	def validate_voice( self, voice: str = None ) -> str:
		"""
		
			Purpose:
			--------
			Validate and normalize an xAI Text to Speech voice identifier.

			Parameters:
			-----------
			voice: str
				Requested voice identifier.

			Returns:
			--------
			str:
				Validated voice identifier.
		
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
		"""
		
			Purpose:
			--------
			Validate and normalize an xAI Text to Speech language code.

			Parameters:
			-----------
			language: str
				Requested language code.

			Returns:
			--------
			str:
				Validated language code.
		
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
		"""
		
			Purpose:
			--------
			Validate and normalize an xAI Text to Speech output format.

			Parameters:
			-----------
			format: str
				Requested output format.

			Returns:
			--------
			str:
				Validated output codec.
		
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
		"""
		
			Purpose:
			--------
			Validate an xAI Text to Speech sample rate.

			Parameters:
			-----------
			sample_rate: int
				Requested sample rate.

			Returns:
			--------
			int | None:
				Validated sample rate or None.
		
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
		"""
		
			Purpose:
			--------
			Validate an xAI Text to Speech MP3 bit rate.

			Parameters:
			-----------
			bit_rate: int
				Requested bit rate.

			Returns:
			--------
			int | None:
				Validated bit rate or None.
		
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
		"""
		
			Purpose:
			--------
			Validate playback speed for UI compatibility.

			Parameters:
			-----------
			speed: float
				Requested playback speed.

			Returns:
			--------
			float:
				Validated playback speed.
		
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
		"""
		
			Purpose:
			--------
			Build the xAI Text to Speech output_format object from assigned members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any] | None:
				Output format object or None.
		
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
		"""
		
			Purpose:
			--------
			Build the xAI Text to Speech request payload from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				TTS request payload.
		
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
		"""
		
			Purpose:
			--------
			Execute the xAI Text to Speech REST request using assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				Raw requests.Response object.
		
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
		"""
		
			Purpose:
			--------
			Extract raw audio bytes from the xAI Text to Speech response.

			Parameters:
			-----------
			None

			Returns:
			--------
			bytes | None:
				Generated audio bytes.
		
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
		"""
		
			Purpose:
			--------
			Generate speech audio from text using the xAI Text to Speech REST API.

			Parameters:
			-----------
			text: str
				Text input to synthesize.

			model: str
				Logical TTS model/API label retained for UI compatibility.

			format: str
				Requested output codec.

			speed: float
				Playback speed retained for UI compatibility.

			voice: str
				xAI voice identifier.

			instruct: str
				Optional instructions retained for UI compatibility. xAI TTS speech style
				should be expressed using inline speech tags in text.

			file_path: str
				Optional destination path for generated audio.

			language: str
				BCP-47 language code or auto.

			sample_rate: int
				Optional output sample rate.

			bit_rate: int
				Optional MP3 bit rate.

			optimize_streaming_latency: int
				Optional xAI latency optimization value.

			text_normalization: bool
				Optional xAI text normalization flag.

			**kwargs: Any
				Additional UI arguments retained on the wrapper.

			Returns:
			--------
			bytes | None:
				Generated audio bytes, or None if no bytes are produced.
		
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
		"""
		
			Purpose:
			--------
			Provider-neutral alias for text-to-speech generation.

			Parameters:
			-----------
			text: str
				Text input to synthesize.

			model: str
				Logical TTS model/API label retained for UI compatibility.

			format: str
				Requested output codec.

			speed: float
				Playback speed retained for UI compatibility.

			voice: str
				xAI voice identifier.

			instruct: str
				Optional instructions retained for UI compatibility.

			file_path: str
				Optional destination path for generated audio.

			language: str
				BCP-47 language code or auto.

			**kwargs: Any
				Additional UI arguments.

			Returns:
			--------
			bytes | None:
				Generated audio bytes.
		
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
		"""
		
			Purpose:
			--------
			Provider-neutral alias for text-to-speech generation.

			Parameters:
			-----------
			text: str
				Text input to synthesize.

			prompt: str
				Alias for text input.

			model: str
				Logical TTS model/API label retained for UI compatibility.

			format: str
				Requested output codec.

			speed: float
				Playback speed retained for UI compatibility.

			voice: str
				xAI voice identifier.

			instruct: str
				Optional instructions retained for UI compatibility.

			file_path: str
				Optional destination path for generated audio.

			language: str
				BCP-47 language code or auto.

			**kwargs: Any
				Additional UI arguments.

			Returns:
			--------
			bytes | None:
				Generated audio bytes.
		
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
		"""
		
			Purpose:
			--------
			Return member names for inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Member names.
		
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
	"""
	
		Purpose:
		--------
		Provide speech-to-text transcription using xAI chat/file attachment behavior.

		Attributes:
		-----------
		client:
			xAI SDK client instance.

		model:
			xAI model used for audio transcription.

		prompt:
			User or system-generated transcription instruction.

		language:
			Source language hint.

		file_path:
			Local audio file path.

		audio_file:
			Open audio file handle used during the request.

		messages:
			xAI SDK chat messages.

		response:
			Raw xAI SDK response object.

		transcript:
			Normalized transcript text.

		Methods:
		--------
		transcribe:
			Transcribe the provided audio file.

		build_prompt:
			Build the transcription prompt from assigned members.

		build_messages:
			Build xAI SDK chat messages from assigned members.

		build_request:
			Build the request dictionary from assigned members.

		execute_request:
			Execute the xAI SDK request from assigned members.

		extract_transcript:
			Extract transcript text from the response.

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
		"""
		
			Purpose:
			--------
			Initialize the Grok transcription wrapper.

			Parameters:
			-----------
			number: int
				Optional response count retained for UI compatibility.

			model: str
				xAI model used for transcription.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			frequency: float
				Optional frequency penalty retained for compatibility.

			presence: float
				Optional presence penalty retained for compatibility.

			max_tokens: int
				Optional maximum output token count.

			store: bool
				Optional storage flag retained for compatibility.

			stream: bool
				Optional stream flag retained for compatibility.

			language: str
				Source language hint.

			instruct: str
				Optional system instruction.

			Returns:
			--------
			None
		
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
		"""
		
			Purpose:
			--------
			Return xAI text-capable models usable for audio-file transcription workflows.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Model option names.
		
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
		"""
		
			Purpose:
			--------
			Return language options for the Audio UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Language option names.
		
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
		"""
		
			Purpose:
			--------
			Return supported audio input format labels for UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Audio format option names.
		
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
		"""
		
			Purpose:
			--------
			Return response format options retained for Audio UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Response format option names.
		
		"""
		return [
				'text',
				'json',
		]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return include options retained for Audio UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Include option names.
		
		"""
		return [ ]
	
	def build_prompt( self ) -> str:
		"""
		
			Purpose:
			--------
			Build the transcription instruction from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			str:
				Transcription prompt.
		
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
		"""
		
			Purpose:
			--------
			Build xAI SDK chat messages from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[Any]:
				xAI SDK chat messages.
		
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
		"""
		
			Purpose:
			--------
			Build the xAI SDK request dictionary from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				Request dictionary.
		
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
		"""
		
			Purpose:
			--------
			Execute the xAI SDK chat/file request using assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				Raw response object.
		
		"""
		try:
			throw_if( 'api_key', self.api_key )
			throw_if( 'file_path', self.file_path )
			self.client = Client( api_key=self.api_key )
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
		"""
		
			Purpose:
			--------
			Extract transcript text from the current response object.

			Parameters:
			-----------
			None

			Returns:
			--------
			str:
				Transcript text.
		
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
		"""
		
			Purpose:
			--------
			Transcribe a local audio file using the xAI SDK chat/file workflow.

			Parameters:
			-----------
			path: str
				Local audio file path.

			model: str
				xAI model name.

			language: str
				Source language hint.

			prompt: str
				Optional transcription instruction.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			frequency: float
				Optional frequency penalty retained on the wrapper.

			presence: float
				Optional presence penalty retained on the wrapper.

			max_tokens: int
				Optional maximum output token count.

			store: bool
				Optional storage flag retained on the wrapper.

			stream: bool
				Optional stream flag retained on the wrapper.

			instruct: str
				Optional system instruction.

			response_format: str
				Optional response format retained on the wrapper.

			include: List[str]
				Optional include values retained on the wrapper.

			mime_type: str
				Optional MIME type retained on the wrapper.

			start_time: float
				Optional start time retained on the wrapper.

			end_time: float
				Optional end time retained on the wrapper.

			**kwargs: Any
				Additional UI values retained on the wrapper.

			Returns:
			--------
			str:
				Transcript text.
		
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
			self.max_output_tokens = max_tokens if max_tokens is not None else self.max_output_tokens
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
		"""
		
			Purpose:
			--------
			Return member names for inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Member names.
		
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
	"""
	
		Purpose:
		--------
		Provide speech translation from an audio file using xAI chat/file attachment behavior.

		Attributes:
		-----------
		client:
			xAI SDK client instance.

		model:
			xAI model used for translation.

		prompt:
			Translation prompt.

		target_language:
			Requested output language.

		source_language:
			Optional source language hint.

		file_path:
			Local audio file path.

		audio_file:
			Open audio file handle used during the request.

		messages:
			xAI SDK chat messages.

		response:
			Raw xAI SDK response object.

		translation:
			Normalized translation text.

		Methods:
		--------
		translate:
			Translate spoken audio into target-language text.

		build_prompt:
			Build the translation prompt from assigned members.

		build_messages:
			Build xAI SDK chat messages from assigned members.

		build_request:
			Build the request dictionary from assigned members.

		execute_request:
			Execute the xAI SDK request from assigned members.

		extract_translation:
			Extract translation text from the response.

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
		"""
		
			Purpose:
			--------
			Initialize the Grok translation wrapper.

			Parameters:
			-----------
			model: str
				xAI model used for translation.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			frequency: float
				Optional frequency penalty retained for compatibility.

			presence: float
				Optional presence penalty retained for compatibility.

			max_tokens: int
				Optional maximum output token count.

			store: bool
				Optional storage flag retained for compatibility.

			stream: bool
				Optional stream flag retained for compatibility.

			instruct: str
				Optional system instruction.

			Returns:
			--------
			None
		
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
		"""
		
			Purpose:
			--------
			Return xAI text-capable models usable for audio translation workflows.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Model option names.
		
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
		"""
		
			Purpose:
			--------
			Return target language options for the Audio UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Language option names.
		
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
		"""
		
			Purpose:
			--------
			Return supported audio input format labels for UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Audio format option names.
		
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
		"""
		
			Purpose:
			--------
			Return response format options retained for Audio UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Response format option names.
		
		"""
		return [
				'text',
				'json',
		]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return include options retained for Audio UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Include option names.
		
		"""
		return [ ]
	
	def build_prompt( self ) -> str:
		"""
		
			Purpose:
			--------
			Build the translation instruction from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			str:
				Translation prompt.
		
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
						'Return only the translated text unless additional instructions require otherwise.'
				)
			
			return (
					f'{base_prompt} Translate the speech into {self.target_language}. '
					'Return only the translated text unless additional instructions require otherwise.'
			)
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Translation'
			ex.method = 'build_prompt( self )'
			raise ex
	
	def build_messages( self ) -> List[ Any ]:
		"""
		
			Purpose:
			--------
			Build xAI SDK chat messages from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[Any]:
				xAI SDK chat messages.
		
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
		"""
		
			Purpose:
			--------
			Build the xAI SDK request dictionary from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				Request dictionary.
		
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
		"""
		
			Purpose:
			--------
			Execute the xAI SDK chat/file request using assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				Raw response object.
		
		"""
		try:
			throw_if( 'api_key', self.api_key )
			throw_if( 'file_path', self.file_path )
			self.client = Client( api_key=self.api_key )
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
		"""
		
			Purpose:
			--------
			Extract translated text from the current response object.

			Parameters:
			-----------
			None

			Returns:
			--------
			str:
				Translated text.
		
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
		"""
		
			Purpose:
			--------
			Translate spoken audio into the requested target language.

			Parameters:
			-----------
			path: str
				Local audio file path.

			model: str
				xAI model name.

			language: str
				Target language.

			prompt: str
				Optional translation instruction.

			source_language: str
				Optional source language hint.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			frequency: float
				Optional frequency penalty retained on the wrapper.

			presence: float
				Optional presence penalty retained on the wrapper.

			max_tokens: int
				Optional maximum output token count.

			store: bool
				Optional storage flag retained on the wrapper.

			stream: bool
				Optional stream flag retained on the wrapper.

			instruct: str
				Optional system instruction.

			response_format: str
				Optional response format retained on the wrapper.

			include: List[str]
				Optional include values retained on the wrapper.

			mime_type: str
				Optional MIME type retained on the wrapper.

			**kwargs: Any
				Additional UI values retained on the wrapper.

			Returns:
			--------
			str:
				Translated text.
		
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
			self.max_output_tokens = max_tokens if max_tokens is not None else self.max_output_tokens
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
		"""
		
			Purpose:
			--------
			Return member names for inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Member names.
		
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
	"""
	
		Purpose:
		--------
		Provide image generation, image editing, and image analysis functionality using
		the xAI Images API and xAI-compatible Responses API.

		Attributes:
		-----------
		model:
			xAI model used for image generation, editing, or image understanding.

		prompt:
			User prompt or image instruction.

		aspect_ratio:
			Optional xAI image aspect ratio.

		resolution:
			Optional xAI image resolution.

		response_format:
			Optional image response format.

		client:
			OpenAI-compatible xAI client.

		image_path:
			Optional local image path used for edit or analysis workflows.

		image_url:
			Optional public URL or data URI used for edit or analysis workflows.

		detail:
			Optional image understanding detail level.

		response:
			Last API response object.

		request:
			Last request payload built by this wrapper.

		output:
			Last normalized image or text output.

		Methods:
		--------
		generate:
			Generate one or more images from a text prompt.

		create:
			Backward-compatible alias for image generation.

		edit:
			Edit an image using a local image path or image URL.

		analyze:
			Analyze an image using xAI-compatible Responses API input.

		vision:
			Alias for analyze.

		describe:
			Alias for analyze.

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
		"""
		
			Purpose:
			--------
			Initialize the Grok Images wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
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
		"""
		
			Purpose:
			--------
			Return supported xAI image-related models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				xAI image generation model names.
		
		"""
		return [ 'grok-imagine-image', 'grok-2-image-1212' ]
	
	@property
	def analysis_model_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI image-understanding model names.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				xAI vision-capable model names.
		
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
		"""
		
			Purpose:
			--------
			Return image-mode tool options retained for UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Tool option names.
		
		"""
		return [ ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Return image-mode include options retained for UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Include option names.
		
		"""
		return [ ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""
		
			Purpose:
			--------
			Return image-mode tool choice options retained for UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Tool choice option names.
		
		"""
		return [ 'auto', 'required', 'none' ]
	
	@property
	def aspect_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI image aspect ratios.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Aspect ratio values.
		
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
		"""
		
			Purpose:
			--------
			Return supported xAI image resolutions.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Image resolution values.
		
		"""
		return [ '1k', '2k' ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return image quality options retained for UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Image quality option values.
		
		"""
		return [ 'auto', 'low', 'medium', 'high' ]
	
	@property
	def style_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return style options retained for UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Style option values.
		
		"""
		return [ ]
	
	@property
	def backcolor_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return background options retained for UI compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Background option values.
		
		"""
		return [ ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported xAI image-understanding detail options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Detail option values.
		
		"""
		return [ 'auto', 'low', 'high' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return xAI image response format options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Response format option values.
		
		"""
		return [ 'url', 'b64_json' ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return output format options consumed by the Images UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Output option values.
		
		"""
		return [ 'url', 'b64_json' ]
	
	@property
	def output_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return output format options consumed by the Images UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Output option values.
		
		"""
		return [ 'url', 'b64_json' ]
	
	def initialize_client( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the OpenAI-compatible xAI client from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
		"""
		try:
			throw_if( 'api_key', self.api_key )
			throw_if( 'base_url', self.base_url )
			self.client = OpenAI( api_key=self.api_key, base_url=self.base_url )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Images'
			ex.method = 'initialize_client( self )'
			raise ex
	
	def normalize_resolution( self, value: str = None ) -> str | None:
		"""
		
			Purpose:
			--------
			Normalize UI resolution values to xAI-supported image resolution values.

			Parameters:
			-----------
			value: str
				Resolution value from the UI.

			Returns:
			--------
			str | None:
				Normalized resolution value or None.
		
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
		"""
		
			Purpose:
			--------
			Normalize UI response format values to xAI/OpenAI-compatible image format values.

			Parameters:
			-----------
			value: str
				Response format value from the UI.

			Returns:
			--------
			str | None:
				Normalized response format value or None.
		
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
		"""
		
			Purpose:
			--------
			Encode a local image path into a base64 data URI accepted by xAI image editing
			and image understanding requests.

			Parameters:
			-----------
			image_path: str
				Local image path.

			Returns:
			--------
			str:
				Base64 data URI.
		
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
		"""
		
			Purpose:
			--------
			Extract text output from the last xAI Responses API response.

			Parameters:
			-----------
			None

			Returns:
			--------
			str | None:
				Output text when available.
		
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
		"""
		
			Purpose:
			--------
			Normalize the last image API response into a renderable value for the Streamlit UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				URL, base64 JSON string, list of image values, text, or raw response.
		
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
		"""
		
			Purpose:
			--------
			Build the xAI image generation request from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				OpenAI-compatible image generation request payload.
		
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
		"""
		
			Purpose:
			--------
			Build the xAI JSON image-edit request from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				JSON image edit request payload.
		
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
		"""
		
			Purpose:
			--------
			Build the xAI Responses API image-understanding request from assigned wrapper
			members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				Responses API request payload.
		
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
			top_p: float = None, top_k: int = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, instruct: str = None, tools: List[ Any ] = None,
			tool_choice: str = None, include: List[ str ] = None,
			allowed_domains: List[ str ] = None,
			store: bool = None, stream: bool = None, is_parallel: bool = None,
			max_tools: int = None,
			max_searches: int = None, grounded: bool = False, image_search: bool = False,
			response_format: str = None, **kwargs: Any ) -> Any:
		"""
		
			Purpose:
			--------
			Generate one or more images from a text prompt using the xAI image generation
			endpoint.

			Parameters:
			-----------
			prompt: str
				Text prompt used to generate the image.

			model: str
				xAI image generation model.

			number: int
				Optional number of images to generate.

			size: str
				Optional xAI resolution value from the UI.

			quality: str
				Optional quality value retained for UI compatibility.

			style: str
				Optional style value retained for UI compatibility.

			fmt: str
				Optional response format alias.

			mime_type: str
				Optional response format alias.

			compression: float
				Optional compression value retained for UI compatibility.

			background: str
				Optional background value retained for UI compatibility.

			aspect_ratio: str
				Optional xAI aspect ratio value.

			response_modalities: str
				Optional response modality retained for UI compatibility.

			temperature: float
				Optional temperature retained for UI compatibility.

			top_p: float
				Optional top-p value retained for UI compatibility.

			top_k: int
				Optional top-k value retained for UI compatibility.

			frequency: float
				Optional frequency penalty retained for UI compatibility.

			presence: float
				Optional presence penalty retained for UI compatibility.

			max_tokens: int
				Optional maximum token value retained for UI compatibility.

			instruct: str
				Optional instructions retained for UI compatibility.

			tools: List[Any]
				Optional tools retained for UI compatibility.

			tool_choice: str
				Optional tool choice retained for UI compatibility.

			include: List[str]
				Optional include values retained for UI compatibility.

			allowed_domains: List[str]
				Optional allowed domains retained for UI compatibility.

			store: bool
				Optional store flag retained for UI compatibility.

			stream: bool
				Optional stream flag retained for UI compatibility.

			is_parallel: bool
				Optional parallel tool flag retained for UI compatibility.

			max_tools: int
				Optional max tools retained for UI compatibility.

			max_searches: int
				Optional max searches retained for UI compatibility.

			grounded: bool
				Optional grounding flag retained for UI compatibility.

			image_search: bool
				Optional image search flag retained for UI compatibility.

			response_format: str
				Optional response format value.

			**kwargs: Any
				Additional UI values retained for compatibility.

			Returns:
			--------
			Any:
				Renderable image output.
		
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
		"""
		
			Purpose:
			--------
			Provider-neutral alias for image generation.

			Parameters:
			-----------
			prompt: str
				Text prompt used to generate the image.

			model: str
				xAI image generation model.

			number: int
				Optional number of images to generate.

			size: str
				Optional xAI resolution value from the UI.

			quality: str
				Optional quality value retained for UI compatibility.

			style: str
				Optional style value retained for UI compatibility.

			fmt: str
				Optional response format alias.

			mime_type: str
				Optional response format alias.

			compression: float
				Optional compression value retained for UI compatibility.

			background: str
				Optional background value retained for UI compatibility.

			aspect_ratio: str
				Optional xAI aspect ratio value.

			response_modalities: str
				Optional response modality retained for UI compatibility.

			**kwargs: Any
				Additional UI values.

			Returns:
			--------
			Any:
				Renderable image output.
		
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
		"""
		
			Purpose:
			--------
			Backward-compatible alias for image generation.

			Parameters:
			-----------
			prompt: str
				Text prompt used to generate the image.

			model: str
				xAI image generation model.

			resolution: str
				Optional xAI resolution value.

			aspect_ratio: str
				Optional xAI aspect ratio value.

			format: str
				Optional response format value.

			number: int
				Optional number of images to generate.

			quality: str
				Optional quality value retained for UI compatibility.

			style: str
				Optional style value retained for UI compatibility.

			**kwargs: Any
				Additional UI values.

			Returns:
			--------
			Any:
				Renderable image output.
		
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
		"""
		
			Purpose:
			--------
			Backward-compatible alias for image generation.

			Parameters:
			-----------
			prompt: str
				Text prompt used to generate the image.

			model: str
				xAI image generation model.

			**kwargs: Any
				Additional UI values.

			Returns:
			--------
			Any:
				Renderable image output.
		
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
		"""
		
			Purpose:
			--------
			Edit an existing image using a text prompt and local image path or image URL.

			Parameters:
			-----------
			image_path: str
				Local image path or public image URL.

			prompt: str
				Text instruction used to edit the image.

			model: str
				xAI image editing model.

			aspect_ratio: str
				Optional xAI aspect ratio value.

			resolution: str
				Optional xAI resolution value.

			quality: str
				Optional quality value retained for UI compatibility.

			response_format: str
				Optional response format value.

			path: str
				Alias for image_path from the UI.

			mask_path: str
				Optional mask path retained for UI compatibility.

			mask: str
				Optional mask alias retained for UI compatibility.

			size: str
				Optional resolution alias.

			fmt: str
				Optional response format alias.

			mime_type: str
				Optional response format alias.

			**kwargs: Any
				Additional UI values retained for compatibility.

			Returns:
			--------
			Any:
				Renderable edited image output.
		
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
		"""
		
			Purpose:
			--------
			Provider-neutral alias for image editing.

			Parameters:
			-----------
			image_path: str
				Local image path or public image URL.

			prompt: str
				Text instruction used to edit the image.

			model: str
				xAI image editing model.

			**kwargs: Any
				Additional UI values.

			Returns:
			--------
			Any:
				Renderable edited image output.
		
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
		"""
		
			Purpose:
			--------
			Provider-neutral alias for image editing.

			Parameters:
			-----------
			image_path: str
				Local image path or public image URL.

			prompt: str
				Text instruction used to edit the image.

			model: str
				xAI image editing model.

			**kwargs: Any
				Additional UI values.

			Returns:
			--------
			Any:
				Renderable edited image output.
		
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
		"""
		
			Purpose:
			--------
			Provider-neutral alias for image editing.

			Parameters:
			-----------
			image_path: str
				Local image path or public image URL.

			prompt: str
				Text instruction used to edit the image.

			model: str
				xAI image editing model.

			**kwargs: Any
				Additional UI values.

			Returns:
			--------
			Any:
				Renderable edited image output.
		
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
		"""
		
			Purpose:
			--------
			Analyze an image using the xAI-compatible Responses API.

			Parameters:
			-----------
			prompt: str
				Text question or instruction about the image.

			image_url: str
				Public image URL or data URI.

			model: str
				xAI image-understanding model.

			max_output_tokens: int
				Optional maximum output tokens.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			detail: str
				Optional image detail value.

			image_path: str
				Optional local image path.

			path: str
				Alias for image_path from the UI.

			store: bool
				Optional response storage flag. Defaults to False for image requests.

			**kwargs: Any
				Additional UI values retained for compatibility.

			Returns:
			--------
			str | None:
				Image analysis text.
		
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
	
	def analyze_image( self, prompt: str, image_url: str = None, model: str = 'grok-4.20-reasoning',
			image_path: str = None, path: str = None, **kwargs: Any ) -> str | None:
		"""
		
			Purpose:
			--------
			Provider-neutral alias for image analysis.

			Parameters:
			-----------
			prompt: str
				Text question or instruction about the image.

			image_url: str
				Public image URL or data URI.

			model: str
				xAI image-understanding model.

			image_path: str
				Optional local image path.

			path: str
				Alias for image_path from the UI.

			**kwargs: Any
				Additional UI values.

			Returns:
			--------
			str | None:
				Image analysis text.
		
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
		"""
		
			Purpose:
			--------
			Provider-neutral alias for image analysis.

			Parameters:
			-----------
			prompt: str
				Text question or instruction about the image.

			image_url: str
				Public image URL or data URI.

			model: str
				xAI image-understanding model.

			image_path: str
				Optional local image path.

			path: str
				Alias for image_path from the UI.

			**kwargs: Any
				Additional UI values.

			Returns:
			--------
			str | None:
				Image analysis text.
		
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
		"""
		
			Purpose:
			--------
			Provider-neutral alias for image analysis.

			Parameters:
			-----------
			prompt: str
				Text question or instruction about the image.

			image_url: str
				Public image URL or data URI.

			model: str
				xAI image-understanding model.

			image_path: str
				Optional local image path.

			path: str
				Alias for image_path from the UI.

			**kwargs: Any
				Additional UI values.

			Returns:
			--------
			str | None:
				Image analysis text.
		
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
		"""
		
			Purpose:
			--------
			Return member names for inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Member names.
		
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
	"""
	
		Purpose:
		--------
		Provide xAI file upload, listing, retrieval, content download, deletion, and
		file-aware question answering through the OpenAI-compatible xAI API surface.

		Attributes:
		-----------
		client:
			OpenAI-compatible xAI client.

		api_key:
			xAI API key used for authorization.

		base_url:
			xAI API base URL.

		file_path:
			Local file path used for upload or file-aware requests.

		file_name:
			File name sent during upload.

		file_id:
			xAI uploaded file identifier.

		file_ids:
			List of uploaded file identifiers.

		purpose:
			File purpose value used by the OpenAI-compatible Files API.

		model:
			xAI model used for file-aware Responses API requests.

		prompt:
			User prompt for file-aware Responses API requests.

		request:
			Last request payload or request metadata.

		response:
			Last response object.

		content:
			Last downloaded file content.

		output_text:
			Last normalized text output.

		documents:
			Known file-name to file-id mapping retained for UI compatibility.

		Methods:
		--------
		upload:
			Upload a local file.

		list:
			List uploaded files.

		list_files:
			Alias for list.

		retrieve:
			Retrieve file metadata.

		extract:
			Download file content.

		content:
			Alias for extract.

		download:
			Alias for extract.

		delete:
			Delete an uploaded file.

		summarize:
			Ask a question about an uploaded file or newly uploaded file.

		survey:
			Alias for summarize supporting legacy plural arguments.

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
		"""
		
			Purpose:
			--------
			Initialize the xAI Files wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
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
		"""
		
			Purpose:
			--------
			Return xAI models appropriate for file-aware Responses API workflows.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				xAI model option names.
		
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
		"""
		
			Purpose:
			--------
			Return file purpose options consumed by the Files UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				File purpose option names.
		
		"""
		return [
				'assistants',
				'batch',
				'fine-tune',
				'user_data',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return file-aware response format options consumed by the Files UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Response format option names.
		
		"""
		return [
				'text',
				'json_object',
		]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return file-aware tool options consumed by the Files UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Tool option names.
		
		"""
		return [
				'code_interpreter',
		]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return file-aware include options consumed by the Files UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]:
				Include option names.
		
		"""
		return [
				'code_execution_call_output',
		]
	
	def initialize_client( self ) -> None:
		"""
		
			Purpose:
			--------
			Initialize the OpenAI-compatible xAI client from assigned members.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
		"""
		try:
			throw_if( 'api_key', self.api_key )
			throw_if( 'base_url', self.base_url )
			self.client = OpenAI( api_key=self.api_key, base_url=self.base_url )
		except Exception as e:
			ex = Error( e )
			ex.module = 'grok'
			ex.cause = 'Files'
			ex.method = 'initialize_client( self )'
			raise ex
	
	def build_headers( self ) -> Dict[ str, str ]:
		"""
		
			Purpose:
			--------
			Build authorization headers for xAI Files REST endpoints.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, str]:
				HTTP headers.
		
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
		"""
		
			Purpose:
			--------
			Build JSON authorization headers for xAI Files REST endpoints.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, str]:
				HTTP headers.
		
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
		"""
		
			Purpose:
			--------
			Extract a file identifier from a response object or dictionary.

			Parameters:
			-----------
			response: Any
				Response object or dictionary.

			Returns:
			--------
			str | None:
				File identifier when available.
		
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
		"""
		
			Purpose:
			--------
			Extract text output from the last Responses API response.

			Parameters:
			-----------
			None

			Returns:
			--------
			str | None:
				Text output when available.
		
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
		"""
		
			Purpose:
			--------
			Build upload request metadata from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				Upload request metadata.
		
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
		"""
		
			Purpose:
			--------
			Execute the file upload using assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				File upload response.
		
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
		"""
		
			Purpose:
			--------
			Build file-list query parameters from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				File-list query parameters.
		
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
		"""
		
			Purpose:
			--------
			Execute the file-list request using assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				File-list response.
		
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
		"""
		
			Purpose:
			--------
			Build file metadata retrieval query parameters from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				File metadata query parameters.
		
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
		"""
		
			Purpose:
			--------
			Execute the file metadata retrieval request using assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				File metadata response.
		
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
		"""
		
			Purpose:
			--------
			Build file content download query parameters from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				File content query parameters.
		
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
		"""
		
			Purpose:
			--------
			Execute the file content download request using assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			bytes | str | None:
				File bytes or extracted text.
		
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
		"""
		
			Purpose:
			--------
			Build file delete query parameters from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				Delete query parameters.
		
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
		"""
		
			Purpose:
			--------
			Execute the file delete request using assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Any:
				Delete response.
		
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
		"""
		
			Purpose:
			--------
			Build Responses API input content containing text and uploaded file references.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[Dict[str, Any]]:
				Responses API content blocks.
		
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
		"""
		
			Purpose:
			--------
			Build a file-aware Responses API request from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[str, Any]:
				Responses API request payload.
		
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
		"""
		
			Purpose:
			--------
			Execute a file-aware Responses API request from assigned wrapper members.

			Parameters:
			-----------
			None

			Returns:
			--------
			str | None:
				Model output text.
		
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
		"""
		
			Purpose:
			--------
			Upload a local file to xAI file storage.

			Parameters:
			-----------
			filepath: str
				Local file path to upload.

			filename: str
				Optional file name assigned to the upload.

			purpose: str
				File purpose value.

			**kwargs: Any
				Additional UI arguments retained on the wrapper.

			Returns:
			--------
			Any:
				Upload response.
		
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
		"""
		
			Purpose:
			--------
			List uploaded xAI files.

			Parameters:
			-----------
			limit: int
				Optional maximum number of files.

			next_token: str
				Optional pagination token.

			order: str
				Optional sort order.

			sort_by: str
				Optional sort field.

			filter: str
				Optional xAI filter expression.

			team_id: str
				Optional xAI team identifier.

			**kwargs: Any
				Additional UI arguments retained on the wrapper.

			Returns:
			--------
			Any:
				File-list response.
		
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
		"""
		
			Purpose:
			--------
			Alias for list.

			Parameters:
			-----------
			limit: int
				Optional maximum number of files.

			next_token: str
				Optional pagination token.

			order: str
				Optional sort order.

			sort_by: str
				Optional sort field.

			filter: str
				Optional xAI filter expression.

			team_id: str
				Optional xAI team identifier.

			**kwargs: Any
				Additional UI arguments.

			Returns:
			--------
			Any:
				File-list response.
		
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
		"""
		
			Purpose:
			--------
			Retrieve metadata for an uploaded xAI file.

			Parameters:
			-----------
			file_id: str
				xAI file identifier.

			team_id: str
				Optional xAI team identifier.

			**kwargs: Any
				Additional UI arguments retained on the wrapper.

			Returns:
			--------
			Any:
				File metadata response.
		
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
		"""
		
			Purpose:
			--------
			Download content for an uploaded xAI file.

			Parameters:
			-----------
			file_id: str
				xAI file identifier.

			format: str
				Optional download format.

			page_number: int
				Optional page number.

			team_id: str
				Optional xAI team identifier.

			**kwargs: Any
				Additional UI arguments retained on the wrapper.

			Returns:
			--------
			bytes | str | None:
				File bytes or extracted text.
		
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
		"""
		
			Purpose:
			--------
			Alias for extract.

			Parameters:
			-----------
			file_id: str
				xAI file identifier.

			format: str
				Optional download format.

			page_number: int
				Optional page number.

			team_id: str
				Optional xAI team identifier.

			**kwargs: Any
				Additional UI arguments.

			Returns:
			--------
			bytes | str | None:
				File bytes or extracted text.
		
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
		"""
		
			Purpose:
			--------
			Delete an uploaded xAI file.

			Parameters:
			-----------
			file_id: str
				xAI file identifier.

			team_id: str
				Optional xAI team identifier.

			**kwargs: Any
				Additional UI arguments retained on the wrapper.

			Returns:
			--------
			Any:
				Delete response.
		
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
		"""
		
			Purpose:
			--------
			Ask a question about an uploaded file or a local file uploaded before the request.

			Parameters:
			-----------
			filepath: str
				Optional local file path to upload.

			filename: str
				Optional file name for upload.

			prompt: str
				User question or instruction.

			file_id: str
				Optional existing xAI file identifier.

			model: str
				xAI model for file-aware Responses API request.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			frequency: float
				Optional frequency penalty.

			presence: float
				Optional presence penalty.

			max_tokens: int
				Optional maximum output tokens.

			store: bool
				Optional response storage flag.

			stream: bool
				Optional stream flag retained on the wrapper.

			instruct: str
				Optional instructions.

			include: List[str]
				Optional include fields.

			tools: List[Any]
				Optional Responses API tools.

			tool_choice: str
				Optional tool-choice value.

			previous_id: str
				Optional previous response identifier.

			conversation_id: str
				Optional conversation identifier.

			purpose: str
				File upload purpose when filepath is provided.

			**kwargs: Any
				Additional UI arguments retained on the wrapper.

			Returns:
			--------
			str | None:
				Model output text.
		
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
		"""
		
			Purpose:
			--------
			Ask a question about one or more local files after uploading them to xAI.

			Parameters:
			-----------
			filepaths: List[str]
				Local file paths to upload.

			filenames: List[str]
				File names corresponding to each path.

			prompt: str
				User question or instruction.

			model: str
				xAI model for file-aware Responses API request.

			temperature: float
				Optional sampling temperature.

			top_p: float
				Optional nucleus sampling value.

			frequency: float
				Optional frequency penalty.

			presence: float
				Optional presence penalty.

			max_tokens: int
				Optional maximum output tokens.

			store: bool
				Optional response storage flag.

			stream: bool
				Optional stream flag retained on the wrapper.

			instruct: str
				Optional instructions.

			purpose: str
				File upload purpose.

			**kwargs: Any
				Additional UI arguments.

			Returns:
			--------
			str | None:
				Model output text.
		
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
		"""
		
			Purpose:
			--------
			Return member names for inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None:
				Member names.
		
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
	"""
	
		Purpose:
		--------
		Provide xAI configured collection search behind the application's Vector Stores
		interface.

		This wrapper uses the configured XAI_API_KEY path. It supports listing and retrieving
		configured collection metadata locally and searching configured xAI collections through
		the xAI client. Remote collection creation, document upload, and deletion are not
		performed by this wrapper without collection-management capability.

		Attributes:
		-----------
		client:
			xAI client instance.

		model:
			Model used for collection search compatibility.

		prompt:
			Search prompt.

		response_format:
			Response format retained for compatibility.

		number:
			Number retained for compatibility.

		content:
			Last content value.

		name:
			Collection name.

		file_path:
			Local file path retained for compatibility.

		file_name:
			File name retained for compatibility.

		file_ids:
			File identifiers retained for compatibility.

		store_ids:
			UI-facing collection identifiers.

		store_id:
			UI-facing collection identifier.

		collection_ids:
			xAI collection identifiers.

		collection_id:
			xAI collection identifier.

		documents:
			Friendly document-name to file-id mapping.

		collections:
			Friendly collection-name to collection-id mapping.

		response:
			Last provider response.

		Methods:
		--------
		list:
			List configured collections.

		retrieve:
			Retrieve configured collection metadata.

		search:
			Search one configured collection.

		survey:
			Search multiple configured collections.

		create:
			Raise a clear collection-management error.

		update:
			Raise a clear collection-management error.

		delete:
			Raise a clear collection-management error.
	
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
		"""
		
			Purpose:
			--------
			Initialize the VectorStores wrapper.

			Parameters:
			-----------
			None

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.api_key = cfg.XAI_API_KEY
		self.base_url = cfg.XAI_BASE_URL
		self.client = Client( api_key=self.api_key )
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
		"""
		
			Purpose:
			--------
			Return list of efficient file interaction models.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
				Model names.
		
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
		"""
		
			Purpose:
			--------
			Resolve a configured collection name or collection identifier to a collection ID.

			Parameters:
			-----------
			store_id: str
				Collection name or collection identifier.

			Returns:
			--------
			str
				Collection identifier.
		
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
		"""
		
			Purpose:
			--------
			Return configured collection metadata rows for the Vector Stores UI.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[Dict[str, Any]]
				Configured collection rows.
		
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
		"""
		
			Purpose:
			--------
			Return response output text when available; otherwise return the response object.

			Parameters:
			-----------
			response: Any
				Provider response.

			Returns:
			--------
			Any
				Output text or provider response.
		
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
		"""
		
			Purpose:
			--------
			Raise a clear error when an operation requires collection-management capability.

			Parameters:
			-----------
			operation: str
				Operation name.

			Returns:
			--------
			None
		
		"""
		raise NotImplementedError(
			f'Grok VectorStores.{operation} requires xAI collection-management capability. '
			f'This wrapper is currently configured with XAI_API_KEY only. Use configured '
			f'collections for search, or add the required management credential/path before '
			f'enabling remote collection management.'
		)
	
	def create( self, name: str, model: str = None ) -> Any:
		"""
		
			Purpose:
			--------
			Block remote collection creation when collection-management capability is not
			configured.

			Parameters:
			-----------
			name: str
				Collection name.

			model: str
				Model name retained for interface compatibility.

			Returns:
			--------
			Any
				This method raises NotImplementedError.
		
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
		"""
		
			Purpose:
			--------
			List configured xAI collections.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[Dict[str, Any]]
				Configured collection rows.
		
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
		"""
		
			Purpose:
			--------
			Retrieve configured metadata for a specific collection.

			Parameters:
			-----------
			store_id: str
				Collection identifier or configured collection name.

			Returns:
			--------
			Dict[str, Any]
				Configured collection metadata.
		
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
		"""
		
			Purpose:
			--------
			Search a specific configured xAI collection.

			Parameters:
			-----------
			prompt: str
				Search prompt.

			store_id: str
				Collection identifier or configured collection name.

			model: str
				Model name retained for interface compatibility.

			Returns:
			--------
			Any
				Output text when available; otherwise provider response.
		
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
		"""
		
			Purpose:
			--------
			Search across multiple configured xAI collections.

			Parameters:
			-----------
			prompt: str
				Search prompt.

			store_ids: List[str]
				Collection identifiers or configured collection names.

			model: str
				Model name retained for interface compatibility.

			Returns:
			--------
			Any
				Output text when available; otherwise provider response.
		
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
		"""
		
			Purpose:
			--------
			Block document upload to a collection when collection-management capability is not
			configured.

			Parameters:
			-----------
			store_id: str
				Collection identifier or configured collection name.

			filepath: str
				Local file path.

			filename: str
				File name.

			Returns:
			--------
			Any
				This method raises NotImplementedError.
		
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
			ex.method = 'update( self, store_id: str, filepath: str=None, filename: str=None ) -> Any'
			raise ex
	
	def delete( self, store_id: str ) -> Any:
		"""
		
			Purpose:
			--------
			Block remote collection deletion when collection-management capability is not
			configured.

			Parameters:
			-----------
			store_id: str
				Collection identifier or configured collection name.

			Returns:
			--------
			Any
				This method raises NotImplementedError.
		
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
		"""
		
			Purpose:
			--------
			Return member names for inspection.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str] | None
				Member names.
		
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