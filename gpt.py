'''
	******************************************************************************************
	    Assembly:                Boo
	    Filename:                Boo.py
	    Author:                  Terry D. Eppler
	    Created:                 05-31-2022
	
	    Last Modified By:        Terry D. Eppler
	    Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="gpt.py" company="Terry D. Eppler">
	
	           Boo is a df analysis tool integrating various Generative GPT, GptText-Processing, and
	           Machine-Learning algorithms for federal analysts.
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
	  Boo.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations
import json
import os
from pathlib import Path
import tiktoken
from openai import OpenAI
from typing import Optional, List, Dict, Any
from openai.types.responses import Response
import base64
from openai.types import CreateEmbeddingResponse, VectorStore, FileObject
from boogr import Error
import config as cfg
import tempfile

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
		--------
		Encodes a local image to a base64 string for vision API requests.
		
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class GPT:
	'''
	
	    Purpose:
	    --------
	    Base class for OpenAI functionality.

    '''
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_tokens: Optional[ int ]
	stops: Optional[ List[ str ] ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	background: Optional[ bool ]
	number: Optional[ int ]
	response_format: Optional[ Dict[ str, str ] ]
	context: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	
	def __init__( self  ):
		self.api_key = cfg.OPENAI_API_KEY
		self.model = None
		self.client = None
		self.number = None
		self.stops = [ ]
		self.response_format = { }
		self.number = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.prompt = None
		self.store = None
		self.stream = None
		self.background = None
		self.instructions = None
		self.context = [ ]

class Chat( GPT ):
	"""
	
	    Purpose:
	    --------
	    Provides a wrapper around the OpenAI Responses API for text-generation,
	    retrieval-augmented, and tool-enabled chat workflows.

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
	        Normalized Responses API tool definitions.

	    reasoning:
	        Optional Responses API reasoning configuration.

	    allowed_domains:
	        Optional list of web-search allowed domains.

	    output_text:
	        Text output from the most recent response.

	    vector_store_ids:
	        Vector store identifiers used for file_search.

	    file_ids:
	        File identifiers retained for compatibility.

	    response:
	        Last Responses API response object.

	    Methods:
	    --------
	    completion:
	        Creates a Responses API call and returns the full response object.

	    generate_text:
	        Generates a text response through the OpenAI Responses API.

	    build_reasoning:
	        Builds a valid Responses API reasoning object.

	    build_input:
	        Builds the Responses API input payload.

	    build_tools:
	        Builds valid built-in Responses API tool objects.

	    build_tool_choice:
	        Builds a safe tool-choice value based on the final tool list.

	    build_include:
	        Filters include values to a conservative supported subset.

	    build_text_format:
	        Builds the Responses API text-format object.

	    build_prompt_template:
	        Builds a valid Responses API prompt template object.

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
	conversation_id: Optional[ str ]
	parallel_tools: Optional[ bool ]
	max_tools: Optional[ int ]
	input: Optional[ List[ Dict[ str, Any ] ] | str ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	image_url: Optional[ str ]
	image_path: Optional[ str ]
	file_url: Optional[ str ]
	file_path: Optional[ str ]
	allowed_domains: Optional[ List[ str ] ]
	max_search_results: Optional[ int ]
	output_text: Optional[ str ]
	vector_stores: Optional[ Dict[ str, str ] ]
	files: Optional[ Dict[ str, str ] ]
	content: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	response: Optional[ Response ]
	file: Optional[ FileObject ]
	purpose: Optional[ str ]
	
	def __init__( self, model: str = 'gpt-5-nano', prompt: str = None, temperature: float = None,
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
			Initialize a Chat wrapper instance with optional Responses API defaults.

			Parameters:
			-----------
			model: str
				Default OpenAI model name.

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
				Optional vector store identifiers used by file_search.

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
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
		self.max_tokens = max_tokens
		self.context = context if context is not None else [ ]
		self.stream = stream
		self.store = store
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
		self.conversation_id = conversation_id
		self.reasoning = reasoning
		self.parallel_tools = is_parallel
		self.tool_choice = tool_choice
		self.response = None
		self.file = None
		self.file_url = file_path
		self.file_path = file_path
		self.image_url = None
		self.content = content
		self.max_search_results = max_search_results
		self.purpose = None
		self.request = { }
		self.messages = [ ]
		self.built_tools = [ ]
		self.stream_requested = False
		self.background_requested = False
		self.prompt_template = None
		self.vector_stores = {
				'Guidance': 'vs_712r5W5833G6aLxIYIbuvVcK',
				'Appropriations': 'vs_8fEoYp1zVvk5D8atfWLbEupN',
		}
		self.files = {
				'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
				'SF133.csv': 'file-WT2h2F5SNxqK2CxyAMSDg6',
				'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
				'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn',
		}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return text-capable model names used by the Text mode selector.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Model option names.

        '''
		return [ 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'gpt-5.2', 'gpt-5.1',
		         'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-4.1-mini',
		         'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Return conservative Responses API include options supported by Text mode.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Include option names.

		'''
		return [
				'file_search_call.results',
				'web_search_call.results',
				'web_search_call.action.sources',
				'code_interpreter_call.outputs',
				'reasoning.encrypted_content',
				'message.output_text.logprobs',
		]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Return built-in tool options that Text mode can safely configure.

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
				'file_search',
				'code_interpreter',
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
	def purpose_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return file purpose options retained for compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				File purpose names.

		'''
		return [
				'assistants',
				'batch',
				'fine-tune',
				'vision',
				'user_data',
				'evals',
		]
	
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
			Return conservative reasoning effort options.

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
				'minimal',
				'low',
				'medium',
				'high',
		]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return modality options retained for compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Modality names.

		'''
		return [ 'text', ]
	
	def build_reasoning( self, reasoning: str | Dict[ str, str ]=None ) -> Dict[ str, str ] | None:
		"""
	
	        Purpose:
	        --------
	        Create a valid Responses API reasoning object from a string or dictionary.

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
			exception.module = 'gpt'
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
					
					self.messages.append( {
								'role': role,
								'content': [
										{
												'type': 'input_text',
												'text': content.strip( ),
										}, ],
						} )
			
			self.messages.append( {
						'role': 'user',
						'content': [
								{
										'type': 'input_text',
										'text': prompt,
								}, ],
				} )
			
			return self.messages
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_input( self, prompt, context, input_data )'
			raise exception
	
	def build_tools( self, tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None,
			vector_store_ids: List[ str ] = None ) -> List[ Dict[ str, Any ] ] | None:
		"""

			Purpose:
			--------
			Normalize supported built-in Responses API tool objects for Text and Chat mode.

			Parameters:
			-----------
			tools: List[ Dict[ str, Any ] ]
				Tool dictionaries selected by the application UI.

			allowed_domains: List[ str ]
				Optional list of allowed domains for web_search.

			vector_store_ids: List[ str ]
				Optional vector store IDs used by file_search.

			Returns:
			--------
			List[ Dict[ str, Any ] ] | None:
				Normalized tool dictionaries or None.

		"""
		try:
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			self.built_tools = [ ]
			
			if tools is None or len( tools ) == 0:
				return None
			
			for tool in tools:
				if not isinstance( tool, dict ):
					continue
				
				tool_type = str( tool.get( 'type', '' ) or '' ).strip( )
				if not tool_type:
					continue
				
				if tool_type == 'web_search':
					built_tool = { 'type': 'web_search' }
					
					filters = tool.get( 'filters' )
					if isinstance( filters, dict ) and len( filters ) > 0:
						built_tool[ 'filters' ] = filters
					elif len( self.allowed_domains ) > 0:
						built_tool[ 'filters' ] = { 'allowed_domains': self.allowed_domains }
					
					search_context_size = tool.get( 'search_context_size' )
					if isinstance( search_context_size, str ) and search_context_size.strip( ):
						built_tool[ 'search_context_size' ] = search_context_size.strip( )
					
					user_location = tool.get( 'user_location' )
					if isinstance( user_location, dict ) and len( user_location ) > 0:
						built_tool[ 'user_location' ] = user_location
					
					self.built_tools.append( built_tool )
					continue
				
				if tool_type == 'file_search':
					ids = tool.get( 'vector_store_ids' )
					if isinstance( ids, list ) and len( ids ) > 0:
						self.vector_store_ids = ids
					
					if len( self.vector_store_ids ) == 0:
						continue
					
					built_tool = { 'type': 'file_search',
							'vector_store_ids': self.vector_store_ids, }
					
					max_num_results = tool.get( 'max_num_results' )
					if isinstance( max_num_results, int ) and max_num_results > 0:
						built_tool[ 'max_num_results' ] = max_num_results
					
					filters = tool.get( 'filters' )
					if isinstance( filters, dict ) and len( filters ) > 0:
						built_tool[ 'filters' ] = filters
					
					self.built_tools.append( built_tool )
					continue
				
				if tool_type == 'code_interpreter':
					built_tool = { 'type': 'code_interpreter' }
					container = tool.get( 'container' )
					
					if isinstance( container, dict ) and len( container ) > 0:
						built_tool[ 'container' ] = container
					elif isinstance( container, str ) and container.strip( ):
						built_tool[ 'container' ] = container.strip( )
					else:
						built_tool[ 'container' ] = { 'type': 'auto' }
					
					self.built_tools.append( built_tool )
					continue
			
			return self.built_tools if len( self.built_tools ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
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
			exception.module = 'gpt'
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
				if name == 'reasoning.encrypted_content':
					allowed.append( name )
					continue
				
				if name == 'message.output_text.logprobs':
					allowed.append( name )
					continue
				
				if name.startswith( 'web_search_call.' ) and 'web_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'file_search_call.results' and 'file_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'code_interpreter_call.outputs' and 'code_interpreter' in tool_types:
					allowed.append( name )
					continue
			
			return allowed if len( allowed ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_include( self, include, tools )'
			raise exception
	
	def build_text_format( self, format: Dict[ str, Any ] | str = None ) -> Dict[ str, Any ] | None:
		"""
		
			Purpose:
			--------
			Build or validate a Responses API text-format object.

			Parameters:
			-----------
			format: Dict[ str, Any ] | str
				Response format dictionary or response format name.

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
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_text_format( self, format )'
			raise exception
	
	def build_prompt_template( self, prompt_id: str = None, prompt_version: str = None,
			prompt_variables: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""
		
			Purpose:
			--------
			Build a Responses API prompt-template reference object.

			Parameters:
			-----------
			prompt_id: str
				Prompt template identifier.

			prompt_version: str
				Optional prompt template version.

			prompt_variables: Dict[ str, Any ]
				Optional prompt template variables.

			Returns:
			--------
			Dict[ str, Any ] | None:
				Prompt template object or None.

		"""
		try:
			if not isinstance( prompt_id, str ) or not prompt_id.strip( ):
				return None
			
			template = { 'id': prompt_id.strip( ) }
			
			if isinstance( prompt_version, str ) and prompt_version.strip( ):
				template[ 'version' ] = prompt_version.strip( )
			
			if isinstance( prompt_variables, dict ) and len( prompt_variables ) > 0:
				template[ 'variables' ] = prompt_variables
			
			return template
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_prompt_template( self, prompt_id, prompt_version, prompt_variables )'
			raise exception
	
	def build_request( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None,
			background: bool = False, reasoning: str = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None, allowed_domains: List[ str ] = None,
			previous_id: str = None, tool_choice: str = None, is_parallel: bool = None,
			context: List[ Dict[ str, str ] ] = None, input_data: List[ Dict[ str, Any ] ] = None,
			vector_store_ids: List[ str ] = None, conversation_id: str = None,
			prompt_id: str = None, prompt_version: str = None,
			prompt_variables: Dict[ str, Any ] = None ) -> Dict[ str, Any ]:
		"""

			Purpose:
			--------
			Create a normalized Responses API request payload for text generation.

			Parameters:
			-----------
			prompt: str
				User prompt submitted to the model.

			model: str
				OpenAI model identifier.

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
				Optional flag controlling whether OpenAI stores the response.

			stream: bool
				Optional stream flag retained for compatibility.

			instruct: str
				Optional system or developer instructions.

			background: bool
				Optional background flag retained for compatibility.

			reasoning: str
				Optional reasoning effort value.

			include: List[ str ]
				Optional Responses API include fields.

			tools: List[ Dict[ str, Any ] ]
				Optional tool dictionaries.

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
				Optional vector store IDs for file_search.

			conversation_id: str
				Optional Responses API conversation identifier.

			prompt_id: str
				Optional OpenAI prompt template identifier.

			prompt_version: str
				Optional OpenAI prompt template version.

			prompt_variables: Dict[ str, Any ]
				Optional OpenAI prompt template variables.

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
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.background = background
			self.instructions = instruct
			self.response_format = self.build_text_format( format )
			self.max_tools = max_tools
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			self.previous_id = previous_id if isinstance( previous_id, str ) else None
			self.conversation_id = conversation_id if isinstance( conversation_id, str ) else None
			self.parallel_tools = is_parallel
			self.reasoning = self.build_reasoning( reasoning )
			self.prompt_template = self.build_prompt_template( prompt_id=prompt_id,
				prompt_version=prompt_version, prompt_variables=prompt_variables )
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
			
			if self.prompt_template is not None:
				self.request[ 'prompt' ] = self.prompt_template
			
			if self.reasoning is not None:
				self.request[ 'reasoning' ] = self.reasoning
			
			if isinstance( self.max_tokens, int ) and self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if self.temperature is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'top_p' ] = self.top_percent
			
			if self.frequency_penalty is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'presence_penalty' ] = self.presence_penalty
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			if self.background is not None:
				self.request[ 'background' ] = bool( self.background )
			
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
			exception.module = 'gpt'
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
			exception.module = 'gpt'
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
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'get_usage( self ) -> Any'
			raise exception
	
	def completion( self, prompt_id: str = None, prompt_version: str = None, model: str = None,
			user_input: str = None, temperature: float = None, format: Dict[ str, Any ] = None,
			top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, store: bool = None, stream: bool = None, instruct: str = None,
			background: bool = False, reasoning: str = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None, tool_choice: str = None,
			is_parallel: bool = None,
			previous_id: str = None, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None, allowed_domains: List[ str ] = None,
			vector_store_ids: List[ str ] = None, conversation_id: str = None,
			max_tools: int = None, prompt_variables: Dict[ str, Any ] = None ) -> Response | Any:
		"""
		
			Purpose:
			--------
			Create a Responses API completion and return the full response object.

			Parameters:
			-----------
			prompt_id: str
				Optional OpenAI prompt template identifier.

			prompt_version: str
				Optional OpenAI prompt template version.

			model: str
				OpenAI model identifier.

			user_input: str
				User input submitted to the model.

			temperature: float
				Optional sampling temperature.

			format: Dict[ str, Any ]
				Optional Responses API text format object.

			top_p: float
				Optional nucleus sampling value.

			frequency: float
				Optional frequency penalty.

			presence: float
				Optional presence penalty.

			max_tokens: int
				Optional maximum output token count.

			store: bool
				Optional response storage flag.

			stream: bool
				Optional stream flag retained for compatibility.

			instruct: str
				Optional system or developer instructions.

			background: bool
				Optional background execution flag.

			reasoning: str
				Optional reasoning effort.

			include: List[ str ]
				Optional include fields.

			tools: List[ Dict[ str, Any ] ]
				Optional Responses API tool dictionaries.

			tool_choice: str
				Optional tool-choice policy.

			is_parallel: bool
				Optional parallel tool-call flag.

			previous_id: str
				Optional previous response identifier.

			context: List[ Dict[ str, str ] ]
				Optional conversation context.

			input_data: List[ Dict[ str, Any ] ]
				Optional prebuilt Responses API input items.

			allowed_domains: List[ str ]
				Optional web-search allowed-domain list.

			vector_store_ids: List[ str ]
				Optional vector store identifiers.

			conversation_id: str
				Optional Responses API conversation identifier.

			max_tools: int
				Optional maximum tool-call count.

			prompt_variables: Dict[ str, Any ]
				Optional prompt template variables.

			Returns:
			--------
			Response | Any:
				Full Responses API response object.

		"""
		try:
			prompt = user_input if user_input is not None else self.prompt
			selected_model = model if model is not None else self.model
			throw_if( 'user_input', prompt )
			throw_if( 'model', selected_model )
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.stream_requested = bool( stream )
			self.background_requested = bool( background )
			self.request = self.build_request( prompt=prompt, model=selected_model,
				temperature=temperature, format=format, top_p=top_p, frequency=frequency,
				max_tools=max_tools, presence=presence, max_tokens=max_tokens, store=store,
				stream=False, instruct=instruct, background=background, reasoning=reasoning,
				include=include, tools=tools, allowed_domains=allowed_domains,
				previous_id=previous_id, tool_choice=tool_choice, is_parallel=is_parallel,
				context=context, input_data=input_data, vector_store_ids=vector_store_ids,
				conversation_id=conversation_id, prompt_id=prompt_id,
				prompt_version=prompt_version, prompt_variables=prompt_variables )
			
			self.response = self.client.responses.create( **self.request )
			self.previous_id = getattr( self.response, 'id', None )
			self.output_text = self.get_output_text( )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'completion( self, **kwargs ) -> Response | Any'
			raise exception
	
	def generate_text( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None, background: bool=False,
			reasoning: str = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None, previous_id: str = None, tool_choice: str = None,
			is_parallel: bool = None, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None, vector_store_ids: List[ str ] = None,
			conversation_id: str = None ) -> str | None:
		"""

			Purpose:
			--------
			Generate a text response through the OpenAI Responses API.

			Parameters:
			-----------
			prompt: str
				User prompt submitted to the Responses API.

			model: str
				OpenAI model name.

			temperature: float
				Optional sampling temperature.

			format: Dict[ str, Any ]
				Optional Responses API text formatting object.

			top_p: float
				Optional nucleus sampling value.

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

			tools: List[ Dict[ str, Any ] ]
				Optional built-in tool definitions.

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
				Optional vector store identifiers used by the file_search tool.

			conversation_id: str
				Optional Responses API conversation identifier.

			Returns:
			--------
			str | None
				Assistant output text when available.

		"""
		try:
			response = self.completion( model=model, user_input=prompt, temperature=temperature,
				format=format, top_p=top_p, frequency=frequency, presence=presence,
				max_tokens=max_tokens, store=store, stream=stream, instruct=instruct,
				background=False, reasoning=reasoning, include=include, tools=tools,
				tool_choice=tool_choice, is_parallel=is_parallel, previous_id=previous_id,
				context=context, input_data=input_data, allowed_domains=allowed_domains,
				vector_store_ids=vector_store_ids, conversation_id=conversation_id,
				max_tools=max_tools )
			
			if response is None:
				return None
			
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str ) -> str | None'
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
				'client',
				'model',
				'prompt',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'stops',
				'store',
				'stream',
				'background',
				'number',
				'response_format',
				'context',
				'instructions',
				'include',
				'tool_choice',
				'previous_id',
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
				'file',
				'purpose',
				'model_options',
				'include_options',
				'tool_options',
				'choice_options',
				'purpose_options',
				'format_options',
				'reasoning_options',
				'modality_options',
				'build_reasoning',
				'build_input',
				'build_tools',
				'build_tool_choice',
				'build_include',
				'build_text_format',
				'build_prompt_template',
				'build_request',
				'get_output_text',
				'get_usage',
				'completion',
				'generate_text',
		]

class Images( GPT ):
	"""
	
	    Purpose:
	    --------
	    Provides OpenAI image generation, image editing, and image analysis functionality.

	    Attributes:
	    -----------
	    api_key:
	        OpenAI API key loaded from config.py.

	    client:
	        OpenAI client instance.

	    model:
	        Image generation/editing model or vision analysis model.

	    prompt:
	        Prompt used for image generation or image editing.

	    input_text:
	        Prompt used for image analysis or image editing.

	    response:
	        Last OpenAI API response object.

	    number:
	        Number of images requested.

	    size:
	        Requested output image size.

	    quality:
	        Requested output image quality.

	    detail:
	        Vision detail level for image analysis.

	    response_format:
	        Requested image output format.

	    mime_format:
	        Requested MIME/output image format.

	    background:
	        Requested background behavior.

	    compression:
	        Requested compression value.

	    image_path:
	        Local image path used for analysis or editing.

	    image_url:
	        Image URL, when returned by the API.

	    file:
	        File object returned by the Files API for vision analysis.

	    Methods:
	    --------
	    generate:
	        Generates one or more images from a text prompt.

	    analyze:
	        Analyzes an uploaded image using a vision-capable Responses API model.

	    edit:
	        Edits one or more images from an uploaded source image and prompt.
	
	"""
	quality: Optional[ str ]
	detail: Optional[ str ]
	size: Optional[ str ]
	previous_id: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	parallel_tools: Optional[ bool ]
	input: Optional[ List[ Dict[ str, Any ] ] | str ]
	instructions: Optional[ str ]
	max_tools: Optional[ int ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	messages: Optional[ List[ Dict[ str, Any ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	image_url: Optional[ str ]
	image_path: Optional[ str ]
	file_url: Optional[ str ]
	file_path: Optional[ str ]
	style: Optional[ str ]
	allowed_domains: Optional[ List[ str ] ]
	response_format: Optional[ str ]
	mime_format: Optional[ str ]
	background: Optional[ str ]
	backcolor: Optional[ str ]
	compression: Optional[ float ]
	
	def __init__( self, prompt: str = None, model: str = 'gpt-image-1', temperature: float = None,
			top_p: float = None, presence: float = None, frequency: float = None,
			max_tokens: int = None, store: bool = None, stream: bool = False, backcolor: str = None,
			instruct: str = None, background: bool = None, number: int = None,
			response_format: str = None, path: str = None, image_url: str = None, size: str = None,
			quality: str = None, detail: str = None, style: str = None, compression: float = None ):
		"""
		
			Purpose:
			--------
			Initialize the OpenAI Images wrapper.

			Parameters:
			-----------
			prompt: str
				Optional prompt used for image generation or image editing.

			model: str
				Optional image model or vision model.

			temperature: float
				Optional sampling temperature for vision analysis.

			top_p: float
				Optional nucleus sampling value retained for compatibility.

			presence: float
				Optional presence penalty retained for compatibility.

			frequency: float
				Optional frequency penalty retained for compatibility.

			max_tokens: int
				Optional maximum output tokens for image analysis.

			store: bool
				Optional Responses API storage flag.

			stream: bool
				Optional stream flag retained for compatibility.

			backcolor: str
				Optional background behavior value.

			instruct: str
				Optional system or developer instructions.

			background: bool
				Optional background value retained for compatibility.

			number: int
				Optional number of images to generate or edit.

			response_format: str
				Optional output format or response format.

			path: str
				Optional local image path for analysis or editing.

			image_url: str
				Optional image URL for analysis.

			size: str
				Optional image output size.

			quality: str
				Optional image quality.

			detail: str
				Optional image analysis detail value.

			style: str
				Optional style value retained for compatibility.

			compression: float
				Optional output compression value from 0.0 to 1.0.

			Returns:
			--------
			None
		
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.prompt = prompt
		self.input_text = None
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.background = backcolor if backcolor is not None else background
		self.backcolor = backcolor
		self.number = number
		self.size = size
		self.quality = quality
		self.detail = detail
		self.style = style
		self.compression = compression
		self.response_format = response_format
		self.mime_format = response_format
		self.output_format = response_format
		self.output_compression = None
		self.image_path = path
		self.file_path = path
		self.image_url = image_url
		self.file_url = None
		self.response = None
		self.file = None
		self.data = None
		self.outputs = [ ]
		self.request = { }
		self.messages = [ ]
		self.include = [ ]
		self.tool_choice = None
		self.parallel_tools = None
		self.max_tools = None
		self.tools = [ ]
		self.reasoning = None
		self.allowed_domains = [ ]
		self.previous_id = None
		self.b64_json = None
		self.url = None
	
	@property
	def model_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return supported GPT image generation model names.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Image generation model names.

        '''
		return [
				'gpt-image-2',
				'gpt-image-1.5',
				'gpt-image-1',
				'gpt-image-1-mini',
				'dall-e-3',
				'dall-e-2',
		]
	
	@property
	def analysis_model_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return supported image analysis model names.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Vision-capable model names.

        '''
		return [
				'gpt-5.5',
				'gpt-5.4',
				'gpt-5.4-mini',
				'gpt-5.4-nano',
				'gpt-5.2',
				'gpt-5.1',
				'gpt-5',
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o',
				'gpt-4o-mini',
		]
	
	@property
	def edit_model_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return supported GPT image editing model names.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Image editing model names.

        '''
		return [
				'gpt-image-2',
				'gpt-image-1.5',
				'gpt-image-1',
				'gpt-image-1-mini',
				'chatgpt-image-latest',
				'dall-e-2',
		]
	
	@property
	def style_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return image style options retained for UI compatibility.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Style option names.

        '''
		return [
				'vivid',
				'natural',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return image output format options.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Output format names.

        '''
		return [
				'png',
				'jpeg',
				'webp',
		]
	
	@property
	def mime_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return MIME/output format options for Image mode controls.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            MIME/output format names.

        '''
		return [
				'png',
				'jpeg',
				'webp',
		]
	
	@property
	def size_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return supported image output sizes.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Image size names.

        '''
		return [
				'auto',
				'1024x1024',
				'1024x1536',
				'1536x1024',
				'1792x1024',
				'1024x1792',
				'512x512',
				'256x256',
		]
	
	@property
	def choice_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return tool choice options retained for Image mode compatibility.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Tool choice option names.

        '''
		return [
				'auto',
				'required',
				'none',
		]
	
	@property
	def backcolor_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        --------
	        Return supported background behavior options for image generation/editing.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Background option names.

        '''
		return [
				'auto',
				'transparent',
				'opaque',
		]
	
	@property
	def quality_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return supported GPT image quality options.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Image quality option names.

        '''
		return [
				'auto',
				'low',
				'medium',
				'high',
				'standard',
				'hd',
		]
	
	@property
	def detail_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Return supported vision detail options for image analysis.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ]:
	            Vision detail options.

        '''
		return [
				'auto',
				'low',
				'high',
				'original',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Return reasoning effort options retained for Image mode compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Reasoning effort names.

		'''
		return [
				'low',
				'medium',
				'high',
				'none',
				'minimal',
				'xhigh',
		]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return modality options retained for Image mode compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Modality option names.

		'''
		return [
				'text',
				'auto',
				'image',
				'audio',
		]
	
	def is_gpt_image_model( self, model: str ) -> bool:
		"""
		
			Purpose:
			--------
			Determine whether a model uses GPT image parameter semantics.

			Parameters:
			-----------
			model: str
				Image generation or editing model name.

			Returns:
			--------
			bool:
				True if the model is a GPT image model; otherwise, False.
		
		"""
		return isinstance( model, str ) and model in [
				'gpt-image-2',
				'gpt-image-1.5',
				'gpt-image-1',
				'gpt-image-1-mini',
				'chatgpt-image-latest',
		]
	
	def normalize_count( self, number: int = None ) -> int:
		"""
		
			Purpose:
			--------
			Normalize the requested image count to the OpenAI Images API range.

			Parameters:
			-----------
			number: int
				Requested image count.

			Returns:
			--------
			int:
				Normalized image count.
		
		"""
		try:
			if isinstance( number, int ) and number > 0:
				return max( 1, min( 10, number ) )
			
			return 1
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'normalize_count( self, number ) -> int'
			raise exception
	
	def normalize_output_format( self, fmt: str = None, background: str = None ) -> str:
		"""
		
			Purpose:
			--------
			Normalize image output format and prevent transparent JPEG requests.

			Parameters:
			-----------
			fmt: str
				Requested output format.

			background: str
				Requested background mode.

			Returns:
			--------
			str:
				Normalized output format.
		
		"""
		try:
			valid_formats = [ 'png', 'jpeg', 'webp' ]
			output_format = fmt if isinstance( fmt, str ) and fmt in valid_formats else 'jpeg'
			
			if background == 'transparent' and output_format == 'jpeg':
				output_format = 'png'
			
			return output_format
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'normalize_output_format( self, fmt, background ) -> str'
			raise exception
	
	def normalize_background( self, background: str = None, model: str = None ) -> str | None:
		"""
		
			Purpose:
			--------
			Normalize image background behavior for supported GPT image models.

			Parameters:
			-----------
			background: str
				Requested background mode.

			model: str
				Selected image model.

			Returns:
			--------
			str | None:
				Normalized background value or None.
		
		"""
		try:
			valid_backgrounds = [ 'auto', 'transparent', 'opaque' ]
			value = background if isinstance( background,
				str ) and background in valid_backgrounds else None
			
			if model == 'gpt-image-2' and value == 'transparent':
				return 'auto'
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'normalize_background( self, background, model ) -> str | None'
			raise exception
	
	def normalize_size( self, size: str = None, model: str = None ) -> str:
		"""
		
			Purpose:
			--------
			Normalize image output size by selected model family.

			Parameters:
			-----------
			size: str
				Requested image size.

			model: str
				Selected image model.

			Returns:
			--------
			str:
				Normalized image size.
		
		"""
		try:
			if model == 'dall-e-2':
				valid_sizes = [ '256x256', '512x512', '1024x1024' ]
				return size if isinstance( size, str ) and size in valid_sizes else '1024x1024'
			
			if model == 'dall-e-3':
				valid_sizes = [ '1024x1024', '1792x1024', '1024x1792' ]
				return size if isinstance( size, str ) and size in valid_sizes else '1024x1024'
			
			valid_sizes = [ 'auto', '1024x1024', '1024x1536', '1536x1024' ]
			return size if isinstance( size, str ) and size in valid_sizes else '1024x1024'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'normalize_size( self, size, model ) -> str'
			raise exception
	
	def normalize_quality( self, quality: str = None, model: str = None ) -> str | None:
		"""
		
			Purpose:
			--------
			Normalize image quality by selected model family.

			Parameters:
			-----------
			quality: str
				Requested image quality.

			model: str
				Selected image model.

			Returns:
			--------
			str | None:
				Normalized image quality or None.
		
		"""
		try:
			if model == 'dall-e-2':
				return None
			
			if model == 'dall-e-3':
				valid_qualities = [ 'standard', 'hd' ]
				return quality if isinstance( quality,
					str ) and quality in valid_qualities else 'standard'
			
			valid_qualities = [ 'auto', 'low', 'medium', 'high' ]
			return quality if isinstance( quality, str ) and quality in valid_qualities else 'auto'
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'normalize_quality( self, quality, model ) -> str | None'
			raise exception
	
	def normalize_compression( self, compression: float = None ) -> int | None:
		"""
		
			Purpose:
			--------
			Normalize a 0.0 to 1.0 compression value into OpenAI's 0 to 100 integer scale.

			Parameters:
			-----------
			compression: float
				Requested compression value.

			Returns:
			--------
			int | None:
				Normalized compression value or None.
		
		"""
		try:
			if compression is None:
				return None
			
			return max( 0, min( 100, int( round( float( compression ) * 100 ) ) ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'normalize_compression( self, compression ) -> int | None'
			raise exception
	
	def normalize_image_outputs( self ) -> str | bytes | list[ str | bytes ] | None:
		"""
		
			Purpose:
			--------
			Normalize OpenAI Images API response data into bytes, URLs, lists, or None.

			Parameters:
			-----------
			None

			Returns:
			--------
			str | bytes | list[ str | bytes ] | None:
				Image bytes, URL fallback, list of outputs, or None.
		
		"""
		try:
			self.data = getattr( self.response, 'data', None )
			self.outputs = [ ]
			
			if self.data and len( self.data ) > 0:
				for item in self.data:
					self.b64_json = getattr( item, 'b64_json', None )
					self.url = getattr( item, 'url', None )
					
					if self.b64_json:
						self.outputs.append( base64.b64decode( self.b64_json ) )
						continue
					
					if self.url:
						self.outputs.append( self.url )
						continue
				
				if len( self.outputs ) == 1:
					return self.outputs[ 0 ]
				
				if len( self.outputs ) > 1:
					return self.outputs
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'normalize_image_outputs( self ) -> str | bytes | list[ str | bytes ] | None'
			raise exception
	
	def generate( self, prompt: str, number: int = 1, model: str = 'gpt-image-1-mini',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = 'jpeg',
			compression: float = None, background: str = None ) -> str | bytes | list[ str | bytes ] | None:
		'''

			Purpose:
			--------
			Generates one or more images from a text prompt using the OpenAI Images API.

			Parameters:
			-----------
			prompt: str
				Text prompt used to generate the image.

			number: int
				Number of images to request.

			model: str
				GPT image model name.

			size: str
				Requested image size.

			quality: str
				Requested image quality.

			fmt: str
				Requested image output format.

			compression: float
				Optional compression value from 0.0 to 1.0 for jpeg and webp outputs.

			background: str
				Optional background mode.

			Returns:
			--------
			str | bytes | list[ str | bytes ] | None
				Generated image bytes, URL fallback, list of outputs, or None.

		'''
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model or 'gpt-image-1-mini'
			self.number = self.normalize_count( number )
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			
			valid_generation_models = [
					'gpt-image-2',
					'gpt-image-1.5',
					'gpt-image-1',
					'gpt-image-1-mini',
					'dall-e-3',
					'dall-e-2',
			]
			
			if self.model not in valid_generation_models:
				raise ValueError( f'Unsupported GPT image generation model: {self.model}' )
			
			self.size = self.normalize_size( size=size, model=self.model )
			self.quality = self.normalize_quality( quality=quality, model=self.model )
			self.background = self.normalize_background( background=background, model=self.model )
			self.output_format = self.normalize_output_format( fmt=fmt, background=self.background )
			self.request = {
					'model': self.model,
					'prompt': self.prompt,
					'n': self.number,
					'size': self.size,
			}
			
			if self.is_gpt_image_model( self.model ):
				self.request[ 'output_format' ] = self.output_format
				
				if self.quality:
					self.request[ 'quality' ] = self.quality
				
				if self.background:
					self.request[ 'background' ] = self.background
				
				if compression is not None and self.output_format in [ 'jpeg', 'webp' ]:
					self.output_compression = self.normalize_compression( compression )
					self.request[ 'output_compression' ] = self.output_compression
			else:
				self.request[ 'response_format' ] = 'b64_json'
				
				if self.quality:
					self.request[ 'quality' ] = self.quality
			
			self.response = self.client.images.generate( **self.request )
			return self.normalize_image_outputs( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'generate( self, prompt: str ) -> str | bytes | list[ str | bytes ] | None'
			raise exception
	
	def analyze( self, text: str, path: str = None, image_url: str = None,
			instruct: str = None, model: str = 'gpt-4.1-mini', max_tokens: int = None,
			temperature: float = None, include: List[ str ] = None, store: bool = None,
			stream: bool = False, detail: str = 'auto' ) -> str | None:
		'''
		
			Purpose:
			--------
			Analyze a local image or image URL using a vision-capable Responses API model.

			Parameters:
			-----------
			text: str
				Question or instruction for image analysis.

			path: str
				Optional local image path.

			image_url: str
				Optional public image URL or data URL.

			instruct: str
				Optional system or developer instructions.

			model: str
				Vision-capable OpenAI model name.

			max_tokens: int
				Optional maximum output token count.

			temperature: float
				Optional sampling temperature.

			include: List[ str ]
				Optional Responses API include fields.

			store: bool
				Optional Responses API store flag.

			stream: bool
				Optional stream flag retained for compatibility.

			detail: str
				Image input detail value.

			Returns:
			--------
			str | None
				Image analysis text when available.

		'''
		try:
			throw_if( 'text', text )
			self.input_text = text
			self.file_path = path
			self.image_path = path
			self.image_url = image_url
			self.model = model or 'gpt-4.1-mini'
			self.instructions = instruct
			self.max_tokens = max_tokens
			self.temperature = temperature
			self.include = include if include is not None else [ ]
			self.store = store
			self.stream = stream
			self.detail = detail if detail in self.detail_options else 'auto'
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			
			if self.image_url is None and self.image_path:
				encoded = encode_image( self.image_path )
				self.image_url = f'data:image/png;base64,{encoded}'
			
			throw_if( 'image_url', self.image_url )
			self.request = { 'model': self.model,
					'input': [ {
									'role': 'user',
									'content': [
											{
													'type': 'input_text',
													'text': self.input_text,
											},
											{
													'type': 'input_image',
													'image_url': self.image_url,
													'detail': self.detail,
											},
									],
							}, ], }
			
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			if isinstance( self.max_tokens, int ) and self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if self.temperature is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			if self.include:
				self.request[ 'include' ] = self.include
			
			self.response = self.client.responses.create( **self.request )
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
							output = getattr( block, 'text', None )
							if output:
								text_parts.append( output )
				
				if len( text_parts ) > 0:
					return ''.join( text_parts ).strip( )
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'analyze( self, text: str, path: str=None ) -> str | None'
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gpt-image-1',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = 'jpeg',
			compression: float = None, background: str = None,
			number: int = 1 ) -> str | bytes | list[ str | bytes ] | None:
		'''
		
			Purpose:
			--------
			Edit an image using the OpenAI Images API.

			Parameters:
			-----------
			prompt: str
				Text instruction used to edit the image.

			path: str
				Local image path.

			model: str
				OpenAI image editing model name.

			size: str
				Requested output image size.

			quality: str
				Requested image quality.

			fmt: str
				Requested output format.

			compression: float
				Optional compression value from 0.0 to 1.0 for jpeg and webp outputs.

			background: str
				Optional background mode.

			number: int
				Number of edited images to request.

			Returns:
			--------
			str | bytes | list[ str | bytes ] | None
				Edited image bytes, URL fallback, list of outputs, or None.

		'''
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.prompt = prompt
			self.input_text = prompt
			self.file_path = path
			self.image_path = path
			self.model = model or 'gpt-image-1'
			self.number = self.normalize_count( number )
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			
			valid_edit_models = [
					'gpt-image-2',
					'gpt-image-1.5',
					'gpt-image-1',
					'gpt-image-1-mini',
					'chatgpt-image-latest',
					'dall-e-2',
			]
			
			if self.model not in valid_edit_models:
				raise ValueError( f'Unsupported GPT image edit model: {self.model}' )
			
			self.size = self.normalize_size( size=size, model=self.model )
			self.quality = self.normalize_quality( quality=quality, model=self.model )
			self.background = self.normalize_background( background=background, model=self.model )
			self.output_format = self.normalize_output_format( fmt=fmt, background=self.background )
			self.request = {
					'model': self.model,
					'prompt': self.input_text,
					'size': self.size,
					'n': self.number,
			}
			
			if self.is_gpt_image_model( self.model ):
				self.request[ 'output_format' ] = self.output_format
				
				if self.quality:
					self.request[ 'quality' ] = self.quality
				
				if self.background:
					self.request[ 'background' ] = self.background
				
				if compression is not None and self.output_format in [ 'jpeg', 'webp' ]:
					self.output_compression = self.normalize_compression( compression )
					self.request[ 'output_compression' ] = self.output_compression
			else:
				self.request[ 'response_format' ] = 'b64_json'
			
			with open( self.file_path, 'rb' ) as source:
				self.response = self.client.images.edit( image=source, **self.request )
			
			return self.normalize_image_outputs( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'edit( self, **kwargs ) -> str | bytes | list[ str | bytes ] | None'
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method returns a list of strings representing members.

	        Parameters:
	        ----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Member names.

        '''
		return [
				'number',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_completion_tokens',
				'store',
				'stream',
				'modalities',
				'stops',
				'api_key',
				'client',
				'path',
				'input_text',
				'image_url',
				'size',
				'quality',
				'detail',
				'model',
				'style_options',
				'model_options',
				'analysis_model_options',
				'edit_model_options',
				'detail_options',
				'format_options',
				'mime_options',
				'size_options',
				'quality_options',
				'backcolor_options',
				'is_gpt_image_model',
				'normalize_count',
				'normalize_output_format',
				'normalize_background',
				'normalize_size',
				'normalize_quality',
				'normalize_compression',
				'normalize_image_outputs',
				'generate',
				'analyze',
				'edit',
		]

class TTS( ):
	"""

	    Purpose:
	    --------
	    Provides text-to-speech functionality through the OpenAI Audio Speech API.

	    Attributes:
	    -----------
	    api_key:
	        OpenAI API key loaded from config.py.

	    client:
	        OpenAI client instance.

	    speed:
	        Speech playback speed.

	    voice:
	        Voice name used for speech generation.

	    input:
	        Text input to synthesize.

	    instructions:
	        Optional voice/style instructions for supported models.

	    response:
	        Last OpenAI API response object.

	    response_format:
	        Audio output format.

	    file_path:
	        Optional destination path for generated audio.

	    model:
	        Text-to-speech model name.

	    audio_bytes:
	        Last generated audio byte output.

	    Methods:
	    --------
	    create_speech:
	        Generate speech audio from input text.

    """
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	speed: Optional[ float ]
	voice: Optional[ str ]
	input: Optional[ str ]
	instructions: Optional[ str ]
	response: Optional[ Any ]
	response_format: Optional[ str ]
	file_path: Optional[ str ]
	model: Optional[ str ]
	audio_bytes: Optional[ bytes ]
	request: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, input: str=None, model: str='gpt-4o-mini-tts', format: str=None,
			instruct: str=None, voice: str=None, speed: float=None, file_path: str=None ):
		"""

	        Purpose:
	        --------
	        Initialize a text-to-speech wrapper instance.

	        Parameters:
	        -----------
	        input: str
	            Optional text input to synthesize.

	        model: str
	            Optional text-to-speech model name.

	        format: str
	            Optional audio output format.

	        instruct: str
	            Optional speech instructions for supported models.

	        voice: str
	            Optional voice name.

	        speed: float
	            Optional speech speed.

	        file_path: str
	            Optional destination path for generated audio.

	        Returns:
	        --------
	        None

        """
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.input = input
		self.model = model
		self.instructions = instruct
		self.response_format = format
		self.voice = voice
		self.file_path = file_path
		self.speed = speed
		self.response = None
		self.audio_bytes = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported text-to-speech model names.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Text-to-speech model names.

        '''
		return [
				'gpt-4o-mini-tts',
				'gpt-4o-mini-tts-2025-12-15',
				'tts-1',
				'tts-1-hd',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported text-to-speech output formats.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Audio output format names.

        '''
		return [
				'mp3',
				'opus',
				'aac',
				'flac',
				'wav',
				'pcm',
		]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported text-to-speech voice names.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Voice names.

        '''
		return [
				'alloy',
				'ash',
				'ballad',
				'coral',
				'echo',
				'fable',
				'nova',
				'onyx',
				'sage',
				'shimmer',
				'verse',
				'marin',
				'cedar',
		]
	
	@property
	def speed_options( self ) -> List[ float ] | None:
		'''

	        Purpose:
	        --------
	        Return supported text-to-speech speed values.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ float ] | None:
	            Speech speed values.

        '''
		return [
				0.25,
				0.50,
				0.75,
				1.0,
				1.25,
				1.50,
				2.0,
				3.0,
				4.0,
		]
	
	def validate_model( self, model: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize the text-to-speech model name.

	        Parameters:
	        -----------
	        model: str
	            Requested text-to-speech model name.

	        Returns:
	        --------
	        str:
	            Valid text-to-speech model name.

        """
		try:
			value = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-mini-tts'
			value = value.strip( )
			if value not in self.model_options:
				raise ValueError( f'Unsupported TTS model: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'validate_model( self, model: str=None ) -> str'
			raise exception
	
	def validate_format( self, format: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize the text-to-speech output format.

	        Parameters:
	        -----------
	        format: str
	            Requested audio output format.

	        Returns:
	        --------
	        str:
	            Valid audio output format.

        """
		try:
			value = format if isinstance( format, str ) and format.strip( ) else 'mp3'
			value = value.strip( ).lower( )
			if value not in self.mime_options:
				raise ValueError( f'Unsupported TTS output format: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'validate_format( self, format: str=None ) -> str'
			raise exception
	
	def validate_voice( self, voice: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize the text-to-speech voice name.

	        Parameters:
	        -----------
	        voice: str
	            Requested voice name.

	        Returns:
	        --------
	        str:
	            Valid voice name.

        """
		try:
			value = voice if isinstance( voice, str ) and voice.strip( ) else 'alloy'
			value = value.strip( )
			if value not in self.voice_options:
				raise ValueError( f'Unsupported TTS voice: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'validate_voice( self, voice: str=None ) -> str'
			raise exception
	
	def validate_speed( self, speed: float=None ) -> float:
		"""

	        Purpose:
	        --------
	        Validate and normalize the text-to-speech speed value.

	        Parameters:
	        -----------
	        speed: float
	            Requested speech speed.

	        Returns:
	        --------
	        float:
	            Valid speech speed from 0.25 through 4.0.

        """
		try:
			value = 1.0 if speed is None else float( speed )
			if value < 0.25:
				return 0.25
			
			if value > 4.0:
				return 4.0
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'validate_speed( self, speed: float=None ) -> float'
			raise exception
	
	def create_speech( self, text: str, model: str='gpt-4o-mini-tts', format: str='mp3',
			speed: float=1.0, voice: str='alloy', instruct: str=None,
			file_path: str=None ) -> bytes | None:
		"""

	        Purpose:
	        --------
	        Generate speech audio from text and return the generated audio bytes.

	        Parameters:
	        -----------
	        text: str
	            Text input to synthesize.

	        model: str
	            Text-to-speech model name.

	        format: str
	            Audio output format.

	        speed: float
	            Speech speed from 0.25 through 4.0.

	        voice: str
	            Voice name.

	        instruct: str
	            Optional voice/style instructions for supported models.

	        file_path: str
	            Optional destination path for generated audio.

	        Returns:
	        --------
	        bytes | None:
	            Generated audio bytes, or None if no bytes are produced.

        """
		try:
			throw_if( 'text', text )
			self.input = text
			self.model = self.validate_model( model )
			self.response_format = self.validate_format( format )
			self.voice = self.validate_voice( voice )
			self.speed = self.validate_speed( speed )
			self.instructions = instruct
			self.file_path = file_path
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.response = None
			self.audio_bytes = None
			
			with tempfile.NamedTemporaryFile(
					suffix=f'.{self.response_format}', delete=False ) as tmp:
				temp_path = tmp.name
			
			try:
				self.request = {
						'model': self.model,
						'voice': self.voice,
						'input': self.input,
						'response_format': self.response_format,
						'speed': self.speed,
				}
				
				if self.instructions and self.model not in ('tts-1', 'tts-1-hd'):
					self.request[ 'instructions' ]=self.instructions
				
				with self.client.audio.speech.with_streaming_response.create(
						**self.request ) as response:
					self.response = response
					response.stream_to_file( temp_path )
				
				with open( temp_path, 'rb' ) as source:
					self.audio_bytes = source.read( )
				
				if self.file_path:
					with open( self.file_path, 'wb' ) as target:
						target.write( self.audio_bytes )
				
				return self.audio_bytes
			finally:
				try:
					if os.path.exists( temp_path ):
						os.remove( temp_path )
				except Exception:
					pass
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, text: str ) -> bytes | None'
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
				'input',
				'file_path',
				'voice',
				'client',
				'response_format',
				'speed',
				'model',
				'instructions',
				'response',
				'audio_bytes',
				'request',
				'model_options',
				'mime_options',
				'voice_options',
				'speed_options',
				'validate_model',
				'validate_format',
				'validate_voice',
				'validate_speed',
				'create_speech',
		]

class Transcription( GPT ):
	"""

	    Purpose:
	    --------
	    Provides audio transcription functionality through the OpenAI Audio
	    Transcriptions API.

	    Attributes:
	    -----------
	    client:
	        OpenAI client instance.

	    language:
	        Optional source-language hint.

	    instructions:
	        Optional prompt/instructions text.

	    response_format:
	        Requested transcription response format.

	    include:
	        Optional transcription include fields such as logprobs.

	    transcript:
	        Extracted transcript text.

	    normalized_result:
	        Normalized transcription output containing text, segments, and raw content.

	    Methods:
	    --------
	    transcribe:
	        Transcribe audio into text or structured transcription output.

    """
	client: Optional[ OpenAI ]
	language: Optional[ str ]
	instructions: Optional[ str ]
	include: Optional[ List[ str ] ]
	normalized_result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str='gpt-4o-transcribe', temperature: float=None,
			prompt: str=None, number: int=None, top_p: float=None, frequency: float=None,
			presence: float=None, max_tokens: int=None, stream: bool=None, store: bool=None,
			language: str=None, instruct: str=None, format: str=None, background: bool=None,
			messages: List[ Dict[ str, str ] ]=None, stops: List[ str ]=None,
			include: List[ str ]=None ):
		"""

	        Purpose:
	        --------
	        Initialize an audio transcription wrapper instance.

	        Parameters:
	        -----------
	        model: str
	            Optional transcription model name.

	        temperature: float
	            Optional transcription temperature.

	        prompt: str
	            Optional prompt text.

	        number: int
	            Optional number retained for compatibility.

	        top_p: float
	            Optional top-p value retained for compatibility.

	        frequency: float
	            Optional frequency penalty retained for compatibility.

	        presence: float
	            Optional presence penalty retained for compatibility.

	        max_tokens: int
	            Optional maximum token value retained for compatibility.

	        stream: bool
	            Optional stream flag retained for compatibility.

	        store: bool
	            Optional store flag retained for compatibility.

	        language: str
	            Optional source-language hint.

	        instruct: str
	            Optional instruction/prompt text.

	        format: str
	            Optional response format.

	        background: bool
	            Optional background flag retained for compatibility.

	        messages: List[ Dict[ str, str ] ]
	            Optional message list retained for compatibility.

	        stops: List[ str ]
	            Optional stop values retained for compatibility.

	        include: List[ str ]
	            Optional include fields.

	        Returns:
	        --------
	        None

        """
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.prompt = prompt
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.stream = stream
		self.response_format = format
		self.background = background
		self.message = messages
		self.stops = stops
		self.store = store
		self.language = language
		self.instructions = instruct
		self.model = model
		self.number = number
		self.input_text = None
		self.audio_file = None
		self.transcript = None
		self.response = None
		self.include = include if include is not None else [ ]
		self.normalized_result = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported transcription model names.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Transcription model names.

        '''
		return [
				'gpt-4o-transcribe',
				'gpt-4o-mini-transcribe',
				'gpt-4o-mini-transcribe-2025-12-15',
				'whisper-1',
				'gpt-4o-transcribe-diarize',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported input audio file formats.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Audio input format names.

        '''
		return [
				'flac',
				'mp3',
				'mp4',
				'mpeg',
				'mpga',
				'm4a',
				'ogg',
				'wav',
				'webm',
		]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return common ISO-639-1 language codes for transcription hints.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Language code values.

        '''
		return [
				'en',
				'es',
				'fr',
				'de',
				'it',
				'pt',
				'ru',
				'uk',
				'el',
				'he',
				'ar',
				'hi',
				'zh',
				'ja',
				'ko',
				'vi',
				'th',
		]
	
	@property
	def language_labels( self ) -> Dict[ str, str ] | None:
		'''

	        Purpose:
	        --------
	        Return human-readable labels for language code options.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        Dict[ str, str ] | None:
	            Mapping of language codes to labels.

        '''
		return {
				'en': 'English',
				'es': 'Spanish',
				'fr': 'French',
				'de': 'German',
				'it': 'Italian',
				'pt': 'Portuguese',
				'ru': 'Russian',
				'uk': 'Ukrainian',
				'el': 'Greek',
				'he': 'Hebrew',
				'ar': 'Arabic',
				'hi': 'Hindi',
				'zh': 'Chinese',
				'ja': 'Japanese',
				'ko': 'Korean',
				'vi': 'Vietnamese',
				'th': 'Thai',
		}
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return optional transcription include fields.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Include field names.

        '''
		return [
				'logprobs',
		]
	
	@property
	def response_format_options( self ) -> Dict[ str, List[ str ] ]:
		'''
		
			Purpose:
			--------
			Return transcription response formats by model.

			Parameters:
			-----------
			None

			Returns:
			--------
			Dict[ str, List[ str ] ]:
				Response formats keyed by model name.

		'''
		return {
				'whisper-1': [
						'json',
						'text',
						'srt',
						'verbose_json',
						'vtt',
				],
				'gpt-4o-transcribe': [
						'json',
				],
				'gpt-4o-mini-transcribe': [
						'json',
				],
				'gpt-4o-mini-transcribe-2025-12-15': [
						'json',
				],
				'gpt-4o-transcribe-diarize': [
						'json',
						'text',
						'diarized_json',
				],
		}
	
	def validate_model( self, model: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize the transcription model name.

	        Parameters:
	        -----------
	        model: str
	            Requested transcription model name.

	        Returns:
	        --------
	        str:
	            Valid transcription model name.

        """
		try:
			value = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-transcribe'
			value = value.strip( )
			if value not in self.model_options:
				raise ValueError( f'Unsupported transcription model: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'validate_model( self, model: str=None ) -> str'
			raise exception
	
	def validate_format( self, model: str, format: str=None ) -> str | None:
		"""

	        Purpose:
	        --------
	        Validate and normalize the transcription response format for a model.

	        Parameters:
	        -----------
	        model: str
	            Valid transcription model name.

	        format: str
	            Requested response format.

	        Returns:
	        --------
	        str | None:
	            Valid response format or None when omitted.

        """
		try:
			options = self.response_format_options.get( model, [ 'json' ] )
			if not isinstance( format, str ) or not format.strip( ):
				return options[ 0 ] if len( options ) > 0 else None
			
			value = format.strip( )
			if value not in options:
				return options[ 0 ] if len( options ) > 0 else None
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'validate_format( self, model: str, format: str=None ) -> str | None'
			raise exception
	
	def validate_include( self, model: str, include: List[ str ]=None ) -> List[ str ]:
		"""

	        Purpose:
	        --------
	        Validate optional transcription include fields for the selected model.

	        Parameters:
	        -----------
	        model: str
	            Valid transcription model name.

	        include: List[ str ]
	            Requested include fields.

	        Returns:
	        --------
	        List[ str ]:
	            Valid include fields.

        """
		try:
			if include is None or len( include ) == 0:
				return [ ]
			
			if model not in [ 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe',
			                  'gpt-4o-mini-transcribe-2025-12-15' ]:
				return [ ]
			
			values = [ ]
			for item in include:
				if isinstance( item, str ) and item.strip( ) in self.include_options:
					values.append( item.strip( ) )
			
			return values
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'validate_include( self, model: str, include: List[ str ]=None )'
			raise exception
	
	def normalize_response( self, response: Any ) -> Dict[ str, Any ]:
		"""

	        Purpose:
	        --------
	        Normalize transcription responses into a dictionary with text, segments,
	        language, duration, and raw content where available.

	        Parameters:
	        -----------
	        response: Any
	            OpenAI transcription response object or string.

	        Returns:
	        --------
	        Dict[ str, Any ]:
	            Normalized transcription result.

        """
		try:
			result: Dict[ str, Any ]={
					'text': '',
					'segments': [ ],
					'language': None,
					'duration': None,
					'raw': None,
			}
			
			if response is None:
				return result
			
			if isinstance( response, str ):
				result[ 'text' ]=response
				result[ 'raw' ]=response
				return result
			
			if hasattr( response, 'model_dump' ):
				try:
					result[ 'raw' ]=response.model_dump( )
				except Exception:
					result[ 'raw' ]=str( response )
			else:
				result[ 'raw' ]=str( response )
			
			text = getattr( response, 'text', None )
			if isinstance( text, str ):
				result[ 'text' ]=text
			
			segments = getattr( response, 'segments', None )
			if isinstance( segments, list ):
				normalized_segments = [ ]
				for segment in segments:
					if hasattr( segment, 'model_dump' ):
						normalized_segments.append( segment.model_dump( ) )
					elif isinstance( segment, dict ):
						normalized_segments.append( segment )
					else:
						normalized_segments.append( { 'text': str( segment ) } )
				
				result[ 'segments' ]=normalized_segments
			
			language = getattr( response, 'language', None )
			if language:
				result[ 'language' ]=language
			
			duration = getattr( response, 'duration', None )
			if duration:
				result[ 'duration' ]=duration
			
			if not result[ 'text' ] and len( result[ 'segments' ] ) > 0:
				parts = [ ]
				for segment in result[ 'segments' ]:
					if isinstance( segment, dict ) and segment.get( 'text' ):
						parts.append( str( segment.get( 'text' ) ) )
				
				result[ 'text' ]='\n'.join( parts ).strip( )
			
			if not result[ 'text' ]:
				result[ 'text' ]=str( response )
			
			return result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'normalize_response( self, response: Any ) -> Dict[ str, Any ]'
			raise exception
	
	def transcribe( self, path: str, model: str='gpt-4o-transcribe', language: str=None,
			prompt: str=None, format: str=None, temperature: float=None,
			include: List[ str ]=None ) -> str | None:
		"""

			Purpose:
			--------
			Transcribe an audio file into text or structured transcription output.

			Parameters:
			-----------
			path: str
				Local path to the audio file.

			model: str
				Transcription model name.

			language: str
				Optional source-language hint.

			prompt: str
				Optional transcription prompt.

			format: str
				Optional response format.

			temperature: float
				Optional transcription temperature.

			include: List[ str ]
				Optional transcription include fields such as logprobs.

			Returns:
			--------
			str | None:
				Extracted transcript text, or None if unavailable.

        """
		try:
			throw_if( 'path', path )
			self.model = self.validate_model( model )
			self.language = language if isinstance( language, str ) and language.strip( ) else None
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else None
			self.response_format = self.validate_format( self.model, format )
			self.temperature = temperature
			self.include = self.validate_include( self.model, include )
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = {
					'model': self.model,
			}
			
			if self.language:
				self.request[ 'language' ]=self.language
			
			if self.prompt:
				self.request[ 'prompt' ]=self.prompt
			
			if self.response_format:
				self.request[ 'response_format' ]=self.response_format
			
			if self.include:
				self.request[ 'include' ]=self.include
			
			if self.temperature is not None:
				if self.model == 'whisper-1':
					self.request[ 'temperature' ]=self.temperature
			
			with open( path, 'rb' ) as self.audio_file:
				self.response = self.client.audio.transcriptions.create(
					file=self.audio_file,
					**self.request )
			
			self.normalized_result = self.normalize_response( self.response )
			self.transcript = self.normalized_result.get( 'text' )
			return self.transcript
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path: str ) -> str | None'
			raise ex
	
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
				'number',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'store',
				'stream',
				'stops',
				'prompt',
				'response',
				'audio_file',
				'messages',
				'response_format',
				'api_key',
				'client',
				'input_text',
				'transcript',
				'language',
				'model',
				'include',
				'normalized_result',
				'model_options',
				'mime_options',
				'language_options',
				'language_labels',
				'include_options',
				'response_format_options',
				'validate_model',
				'validate_format',
				'validate_include',
				'normalize_response',
				'transcribe',
		]

class Translation( GPT ):
	"""

	    Purpose:
	    --------
	    Provides audio translation functionality through the OpenAI Audio
	    Translations API.

	    Notes:
	    ------
	    OpenAI audio translation translates non-English speech to English. The language
	    parameter is retained only as optional local/source-language context and is not
	    sent as a target-language control.

    """
	client: Optional[ OpenAI ]
	target_language: Optional[ str ]
	response_format: Optional[ str ]
	normalized_result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str='whisper-1', temperature: float=None, top_p: float=None,
			frequency: float=None, presence: float=None, max_tokens: int=None,
			store: bool=None,
			stream: bool=None, instruct: str=None, audio_file: str=None, format: str=None,
			language: str=None ):
		"""

	        Purpose:
	        --------
	        Initialize an audio translation wrapper instance.

	        Parameters:
	        -----------
	        model: str
	            Optional translation model name.

	        temperature: float
	            Optional translation temperature.

	        top_p: float
	            Optional top-p value retained for compatibility.

	        frequency: float
	            Optional frequency penalty retained for compatibility.

	        presence: float
	            Optional presence penalty retained for compatibility.

	        max_tokens: int
	            Optional maximum token value retained for compatibility.

	        store: bool
	            Optional store flag retained for compatibility.

	        stream: bool
	            Optional stream flag retained for compatibility.

	        instruct: str
	            Optional prompt/instruction text.

	        audio_file: str
	            Optional audio file path.

	        format: str
	            Optional response format.

	        language: str
	            Optional source-language context retained for compatibility.

	        Returns:
	        --------
	        None

        """
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.audio_file = audio_file
		self.response = None
		self.response_format = format
		self.target_language = language
		self.normalized_result = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported audio translation model names.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Translation model names.

        '''
		return [
				'whisper-1',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported input audio formats.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Audio input format names.

        '''
		return [
				'flac',
				'mp3',
				'mp4',
				'mpeg',
				'mpga',
				'm4a',
				'ogg',
				'wav',
				'webm',
		]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return common ISO-639-1 language codes retained for source-language context.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[ str ] | None:
	            Language code values.

        '''
		return [
				'en',
				'es',
				'fr',
				'de',
				'it',
				'pt',
				'ru',
				'uk',
				'el',
				'he',
				'ar',
				'hi',
				'zh',
				'ja',
				'ko',
				'vi',
				'th',
		]
	
	@property
	def language_labels( self ) -> Dict[ str, str ] | None:
		'''

	        Purpose:
	        --------
	        Return human-readable labels for source-language context codes.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        Dict[ str, str ] | None:
	            Mapping of language codes to labels.

        '''
		return {
				'en': 'English',
				'es': 'Spanish',
				'fr': 'French',
				'de': 'German',
				'it': 'Italian',
				'pt': 'Portuguese',
				'ru': 'Russian',
				'uk': 'Ukrainian',
				'el': 'Greek',
				'he': 'Hebrew',
				'ar': 'Arabic',
				'hi': 'Hindi',
				'zh': 'Chinese',
				'ja': 'Japanese',
				'ko': 'Korean',
				'vi': 'Vietnamese',
				'th': 'Thai',
		}
	
	@property
	def response_format_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return audio translation response format options.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Response format names.

		'''
		return [
				'json',
				'text',
				'srt',
				'verbose_json',
				'vtt',
		]
	
	def validate_model( self, model: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize the audio translation model name.

	        Parameters:
	        -----------
	        model: str
	            Requested translation model name.

	        Returns:
	        --------
	        str:
	            Valid translation model name.

        """
		try:
			value = model if isinstance( model, str ) and model.strip( ) else 'whisper-1'
			value = value.strip( )
			if value not in self.model_options:
				raise ValueError( f'Unsupported translation model: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Translation'
			exception.method = 'validate_model( self, model: str=None ) -> str'
			raise exception
	
	def validate_format( self, format: str=None ) -> str | None:
		"""

	        Purpose:
	        --------
	        Validate and normalize the audio translation response format.

	        Parameters:
	        -----------
	        format: str
	            Requested response format.

	        Returns:
	        --------
	        str | None:
	            Valid response format or None.

        """
		try:
			if not isinstance( format, str ) or not format.strip( ):
				return 'json'
			
			value = format.strip( )
			if value not in self.response_format_options:
				return 'json'
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Translation'
			exception.method = 'validate_format( self, format: str=None ) -> str | None'
			raise exception
	
	def normalize_response( self, response: Any ) -> Dict[ str, Any ]:
		"""

	        Purpose:
	        --------
	        Normalize audio translation responses into a dictionary with text,
	        segments, language, duration, and raw content where available.

	        Parameters:
	        -----------
	        response: Any
	            OpenAI translation response object or string.

	        Returns:
	        --------
	        Dict[ str, Any ]:
	            Normalized translation result.

        """
		try:
			result: Dict[ str, Any ]={
					'text': '',
					'segments': [ ],
					'language': None,
					'duration': None,
					'raw': None,
			}
			
			if response is None:
				return result
			
			if isinstance( response, str ):
				result[ 'text' ]=response
				result[ 'raw' ]=response
				return result
			
			if hasattr( response, 'model_dump' ):
				try:
					result[ 'raw' ]=response.model_dump( )
				except Exception:
					result[ 'raw' ]=str( response )
			else:
				result[ 'raw' ]=str( response )
			
			text = getattr( response, 'text', None )
			if isinstance( text, str ):
				result[ 'text' ]=text
			
			segments = getattr( response, 'segments', None )
			if isinstance( segments, list ):
				normalized_segments = [ ]
				for segment in segments:
					if hasattr( segment, 'model_dump' ):
						normalized_segments.append( segment.model_dump( ) )
					elif isinstance( segment, dict ):
						normalized_segments.append( segment )
					else:
						normalized_segments.append( { 'text': str( segment ) } )
				
				result[ 'segments' ]=normalized_segments
			
			language = getattr( response, 'language', None )
			if language:
				result[ 'language' ]=language
			
			duration = getattr( response, 'duration', None )
			if duration:
				result[ 'duration' ]=duration
			
			if not result[ 'text' ] and len( result[ 'segments' ] ) > 0:
				parts = [ ]
				for segment in result[ 'segments' ]:
					if isinstance( segment, dict ) and segment.get( 'text' ):
						parts.append( str( segment.get( 'text' ) ) )
				
				result[ 'text' ]='\n'.join( parts ).strip( )
			
			if not result[ 'text' ]:
				result[ 'text' ]=str( response )
			
			return result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Translation'
			exception.method = 'normalize_response( self, response: Any ) -> Dict[ str, Any ]'
			raise exception
	
	def translate( self, filepath: str, model: str='whisper-1', prompt: str=None,
			format: str=None, temperature: float=None, language: str=None ) -> str | None:
		"""

            Purpose:
            --------
            Translate non-English speech to English.

			Parameters:
			-----------
			filepath: str
				Local path to the audio file.

			model: str
				Translation model name.

			prompt: str
				Optional prompt text.

			format: str
				Optional response format.

			temperature: float
				Optional translation temperature.

			language: str
				Optional source-language context retained for compatibility. This is not
				sent as a target-language parameter.

			Returns:
			--------
			str | None:
				Translated English text, or None if unavailable.

        """
		try:
			throw_if( 'filepath', filepath )
			self.model = self.validate_model( model )
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else None
			self.response_format = self.validate_format( format )
			self.temperature = temperature
			self.target_language = language
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = {
					'model': self.model,
			}
			
			if self.prompt:
				self.request[ 'prompt' ]=self.prompt
			
			if self.response_format:
				self.request[ 'response_format' ]=self.response_format
			
			if self.temperature is not None:
				self.request[ 'temperature' ]=self.temperature
			
			with open( filepath, 'rb' ) as audio_file:
				self.response = self.client.audio.translations.create(
					file=audio_file,
					**self.request )
			
			self.normalized_result = self.normalize_response( self.response )
			return self.normalized_result.get( 'text' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Translation'
			ex.method = 'translate( self, filepath: str ) -> str | None'
			raise ex
	
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
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'store',
				'stream',
				'prompt',
				'response',
				'audio_file',
				'response_format',
				'api_key',
				'client',
				'model',
				'target_language',
				'normalized_result',
				'model_options',
				'mime_options',
				'language_options',
				'language_labels',
				'response_format_options',
				'validate_model',
				'validate_format',
				'normalize_response',
				'translate',
		]

class Embeddings( GPT ):
	"""
	
	    Purpose:
	    --------
	    Provides a wrapper around the OpenAI Embeddings API for creating vector
	    representations of text inputs.

	    Attributes:
	    -----------
	    api_key:
	        OpenAI API key loaded from config.py.

	    client:
	        OpenAI client instance.

	    model:
	        Embedding model name.

	    input:
	        Text input or list of text inputs submitted to the API.

	    encoding_format:
	        Embedding encoding format: float or base64.

	    dimensions:
	        Optional reduced embedding dimension for supported models.

	    user:
	        Optional end-user identifier.

	    response:
	        Last OpenAI embeddings response object.

	    embedding:
	        First embedding returned by the API.

	    embeddings:
	        All embeddings returned by the API.

	    usage:
	        Usage metadata returned by the API.

	    request:
	        Last OpenAI embeddings request dictionary.

	    Methods:
	    --------
	    create:
	        Create one or more embeddings from text input.

	    count_tokens:
	        Count tokens for one text string.

	    count_total_tokens:
	        Count tokens across one or more text inputs.

	    validate_model:
	        Validate and normalize the embedding model name.

	    validate_encoding_format:
	        Validate and normalize the embedding encoding format.

	    validate_dimensions:
	        Validate and normalize optional embedding dimensions.

	    validate_input:
	        Validate and normalize embedding input text.

	    get_default_dimensions:
	        Return default embedding dimensions for a model.

	    get_max_dimensions:
	        Return maximum supported dimensions for a model.

    """
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	input: Optional[ str | List[ str ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	user: Optional[ str ]
	response: Optional[ CreateEmbeddingResponse ]
	embedding: Optional[ List[ float ] | str ]
	embeddings: Optional[ List[ List[ float ] ] | List[ str ] ]
	usage: Optional[ Any ]
	request: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, text: str | List[ str ]=None, model: str='text-embedding-3-small',
			format: str='float', dimensions: int=None, user: str=None ):
		"""

	        Purpose:
	        --------
	        Initialize an Embeddings wrapper instance.

	        Parameters:
	        -----------
	        text: str | List[str]
	            Optional text input or list of text inputs.

	        model: str
	            Optional OpenAI embedding model name.

	        format: str
	            Optional embedding encoding format: float or base64.

	        dimensions: int
	            Optional embedding dimension for supported models.

	        user: str
	            Optional end-user identifier.

	        Returns:
	        --------
	        None

        """
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.input = text
		self.encoding_format = format
		self.dimensions = dimensions
		self.user = user
		self.response = None
		self.embedding = None
		self.embeddings = None
		self.usage = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported OpenAI embedding model names.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[str] | None:
	            Embedding model names.

        '''
		return [
				'text-embedding-3-small',
				'text-embedding-3-large',
				'text-embedding-ada-002',
		]
	
	@property
	def encoding_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported embedding encoding formats.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[str] | None:
	            Embedding encoding formats.

        '''
		return [
				'float',
				'base64',
		]
	
	@property
	def model_default_dimensions( self ) -> Dict[ str, int ]:
		'''

	        Purpose:
	        --------
	        Return default embedding dimensions by model.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        Dict[str, int]:
	            Default dimension values keyed by model name.

        '''
		return {
				'text-embedding-3-small': 1536,
				'text-embedding-3-large': 3072,
				'text-embedding-ada-002': 1536,
		}
	
	@property
	def model_max_dimensions( self ) -> Dict[ str, int ]:
		'''

	        Purpose:
	        --------
	        Return maximum supported embedding dimensions by model.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        Dict[str, int]:
	            Maximum dimension values keyed by model name.

        '''
		return {
				'text-embedding-3-small': 1536,
				'text-embedding-3-large': 3072,
				'text-embedding-ada-002': 1536,
		}
	
	@property
	def model_dimension_support( self ) -> Dict[ str, bool ]:
		'''

	        Purpose:
	        --------
	        Return whether each embedding model supports the dimensions parameter.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        Dict[str, bool]:
	            Dimension-parameter support keyed by model name.

        '''
		return {
				'text-embedding-3-small': True,
				'text-embedding-3-large': True,
				'text-embedding-ada-002': False,
		}
	
	def validate_model( self, model: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize the embedding model name.

	        Parameters:
	        -----------
	        model: str
	            Requested embedding model name.

	        Returns:
	        --------
	        str:
	            Valid embedding model name.

        """
		try:
			value = model if isinstance( model, str ) and model.strip( ) else \
				'text-embedding-3-small'
			
			value = value.strip( )
			if value not in self.model_options:
				raise ValueError( f'Unsupported embedding model: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_model( self, model: str=None ) -> str'
			raise exception
	
	def validate_encoding_format( self, format: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize the embedding encoding format.

	        Parameters:
	        -----------
	        format: str
	            Requested encoding format.

	        Returns:
	        --------
	        str:
	            Valid encoding format: float or base64.

        """
		try:
			value = format if isinstance( format, str ) and format.strip( ) else 'float'
			value = value.strip( ).lower( )
			if value not in self.encoding_options:
				raise ValueError( f'Unsupported embedding encoding format: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_encoding_format( self, format: str=None ) -> str'
			raise exception
	
	def validate_dimensions( self, model: str, dimensions: int=None ) -> int | None:
		"""

	        Purpose:
	        --------
	        Validate and normalize optional embedding dimensions for the selected model.

	        Parameters:
	        -----------
	        model: str
	            Valid embedding model name.

	        dimensions: int
	            Requested output dimensions.

	        Returns:
	        --------
	        int | None:
	            Valid dimensions value, or None when dimensions should be omitted.

        """
		try:
			if dimensions is None:
				return None
			
			try:
				value = int( dimensions )
			except Exception:
				return None
			
			if value <= 0:
				return None
			
			supports_dimensions = self.model_dimension_support.get( model, False )
			if not supports_dimensions:
				return None
			
			max_dimensions = self.get_max_dimensions( model )
			if value > max_dimensions:
				return max_dimensions
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_dimensions( self, model: str, dimensions: int=None )'
			raise exception
	
	def validate_input( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""

	        Purpose:
	        --------
	        Validate and normalize embedding input text.

	        Parameters:
	        -----------
	        text: str | List[str]
	            Text string or list of text strings to embed.

	        Returns:
	        --------
	        str | List[str]:
	            Clean embedding input.

        """
		try:
			throw_if( 'text', text )
			
			if isinstance( text, str ):
				value = text.strip( )
				throw_if( 'text', value )
				return value
			
			if isinstance( text, list ):
				values = [ ]
				for item in text:
					if not isinstance( item, str ):
						continue
					
					clean = item.strip( )
					if clean:
						values.append( clean )
				
				throw_if( 'text', values )
				return values
			
			raise ValueError( 'Embedding input must be a string or list of strings.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_input( self, text: str | List[ str ] )'
			raise exception
	
	def get_default_dimensions( self, model: str ) -> int:
		"""

	        Purpose:
	        --------
	        Return the default embedding dimensions for a model.

	        Parameters:
	        -----------
	        model: str
	            Embedding model name.

	        Returns:
	        --------
	        int:
	            Default embedding dimension count.

        """
		try:
			return int( self.model_default_dimensions.get( model, 1536 ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'get_default_dimensions( self, model: str ) -> int'
			raise exception
	
	def get_max_dimensions( self, model: str ) -> int:
		"""

	        Purpose:
	        --------
	        Return maximum supported embedding dimensions for a model.

	        Parameters:
	        -----------
	        model: str
	            Embedding model name.

	        Returns:
	        --------
	        int:
	            Maximum supported dimension count.

        """
		try:
			return int( self.model_max_dimensions.get( model, 1536 ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'get_max_dimensions( self, model: str ) -> int'
			raise exception
	
	def count_tokens( self, text: str, encoding_name: str='cl100k_base' ) -> int:
		"""
	
	        Purpose:
	        --------
	        Count tokens in a text string using tiktoken.

	        Parameters:
	        -----------
	        text: str
	            Text to count.

	        encoding_name: str
	            Tiktoken encoding name.

	        Returns:
	        --------
	        int:
	            Token count.

        """
		try:
			if not isinstance( text, str ) or not text:
				return 0
			
			encoding = tiktoken.get_encoding( encoding_name )
			return len( encoding.encode( text ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'count_tokens( self, text: str, encoding_name: str ) -> int'
			raise exception
	
	def count_total_tokens( self, text: str | List[ str ],
			encoding_name: str='cl100k_base' ) -> int:
		"""
	
	        Purpose:
	        --------
	        Count total tokens across one text string or a list of text strings.

	        Parameters:
	        -----------
	        text: str | List[str]
	            Text input or list of text inputs.

	        encoding_name: str
	            Tiktoken encoding name.

	        Returns:
	        --------
	        int:
	            Total token count.

        """
		try:
			if isinstance( text, str ):
				return self.count_tokens( text, encoding_name=encoding_name )
			
			if isinstance( text, list ):
				return sum( self.count_tokens( item, encoding_name=encoding_name )
				            for item in text if isinstance( item, str ) )
			
			return 0
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'count_total_tokens( self, text: str | List[ str ] ) -> int'
			raise exception
	
	def validate_token_limits( self, text: str | List[ str ],
			max_input_tokens: int=8192, max_total_tokens: int=300000 ) -> None:
		"""
	
	        Purpose:
	        --------
	        Validate per-input and total token limits before calling the Embeddings API.

	        Parameters:
	        -----------
	        text: str | List[str]
	            Text input or list of text inputs.

	        max_input_tokens: int
	            Maximum tokens allowed for a single input item.

	        max_total_tokens: int
	            Maximum total tokens allowed across the request.

	        Returns:
	        --------
	        None

        """
		try:
			values = text if isinstance( text, list ) else [ text ]
			for index, item in enumerate( values ):
				token_count = self.count_tokens( item )
				if token_count > max_input_tokens:
					raise ValueError(
						f'Embedding input item {index + 1} has {token_count} tokens, '
						f'which exceeds the {max_input_tokens} token per-input limit.' )
			
			total_tokens = self.count_total_tokens( text )
			if total_tokens > max_total_tokens:
				raise ValueError(
					f'Embedding request has {total_tokens} total tokens, which exceeds '
					f'the {max_total_tokens} token request limit.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_token_limits( self, text: str | List[ str ] )'
			raise exception
	
	def build_request( self, text: str | List[ str ], model: str='text-embedding-3-small',
			format: str='float', dimensions: int=None, user: str=None ) -> Dict[ str, Any ]:
		"""
	
	        Purpose:
	        --------
	        Build a validated OpenAI Embeddings API request dictionary.

	        Parameters:
	        -----------
	        text: str | List[str]
	            Text input or list of text inputs.

	        model: str
	            Embedding model name.

	        format: str
	            Encoding format: float or base64.

	        dimensions: int
	            Optional reduced dimensions for supported models.

	        user: str
	            Optional end-user identifier.

	        Returns:
	        --------
	        Dict[str, Any]:
	            Embeddings API request dictionary.

        """
		try:
			self.input = self.validate_input( text )
			self.model = self.validate_model( model )
			self.encoding_format = self.validate_encoding_format( format )
			self.dimensions = self.validate_dimensions( self.model, dimensions )
			self.user = user if isinstance( user, str ) and user.strip( ) else None
			
			self.validate_token_limits( self.input )
			
			self.request = {
					'model': self.model,
					'input': self.input,
					'encoding_format': self.encoding_format,
			}
			
			if self.dimensions is not None:
				self.request[ 'dimensions' ]=self.dimensions
			
			if self.user:
				self.request[ 'user' ]=self.user.strip( )
			
			return self.request
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'build_request( self, text: str | List[ str ], **kwargs )'
			raise exception
	
	def create( self, text: str | List[ str ], model: str='text-embedding-3-small',
			format: str='float', dimensions: int=None,
			user: str=None ) -> List[ float ] | List[ List[ float ] ] | str | List[ str ] | None:
		"""
	
	        Purpose:
	        --------
	        Create one or more embeddings from text input using the OpenAI Embeddings API.

	        Parameters:
	        -----------
	        text: str | List[str]
	            Text input or list of text inputs.

	        model: str
	            Embedding model name.

	        format: str
	            Encoding format: float or base64.

	        dimensions: int
	            Optional reduced dimensions for supported embedding models.

	        user: str
	            Optional end-user identifier.

	        Returns:
	        --------
	        List[float] | List[List[float]] | str | List[str] | None:
	            Single embedding, list of embeddings, base64 embedding string, list of
	            base64 strings, or None.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = self.build_request( text=text, model=model, format=format,
				dimensions=dimensions, user=user )
			
			self.response = self.client.embeddings.create( **self.request )
			self.usage = getattr( self.response, 'usage', None )
			self.data = getattr( self.response, 'data', None )
			self.embeddings = [ ]
			
			if self.data is None or len( self.data ) == 0:
				self.embedding = None
				return None
			
			for item in self.data:
				value = getattr( item, 'embedding', None )
				if value is not None:
					self.embeddings.append( value )
			
			if len( self.embeddings ) == 0:
				self.embedding = None
				return None
			
			self.embedding = self.embeddings[ 0 ]
			
			if isinstance( self.input, str ):
				return self.embedding
			
			return self.embeddings
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'create( self, text: str | List[ str ], **kwargs )'
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
	        List[str] | None:
	            Member names.

        '''
		return [
				'api_key',
				'client',
				'model',
				'input',
				'encoding_format',
				'dimensions',
				'user',
				'response',
				'embedding',
				'embeddings',
				'usage',
				'request',
				'model_options',
				'encoding_options',
				'model_default_dimensions',
				'model_max_dimensions',
				'model_dimension_support',
				'validate_model',
				'validate_encoding_format',
				'validate_dimensions',
				'validate_input',
				'get_default_dimensions',
				'get_max_dimensions',
				'count_tokens',
				'count_total_tokens',
				'validate_token_limits',
				'build_request',
				'create',
		]

class Files( GPT ):
	"""
	
	    Purpose:
	    --------
	    Provides a wrapper around the OpenAI Files API for file upload, listing,
	    retrieval, content extraction, deletion, and selected file analysis workflows.

	    Attributes:
	    -----------
	    api_key:
	        OpenAI API key loaded from config.py.

	    client:
	        OpenAI client instance.

	    file:
	        Last raw FileObject or file-related response object.

	    file_id:
	        Last selected or returned OpenAI file identifier.

	    filepath:
	        Local file path used for upload.

	    filename:
	        Filename associated with the current file.

	    purpose:
	        Upload or filter purpose.

	    response:
	        Last API response object.

	    content:
	        Last extracted file content.

	    files:
	        Last normalized list of files.

	    request:
	        Last request dictionary.

	    model:
	        Model used by optional file analysis workflows.

	    prompt:
	        Prompt used by optional file analysis workflows.

	    output_text:
	        Last text output from an optional file analysis workflow.

	    Methods:
	    --------
	    upload:
	        Upload a local file to OpenAI.

	    list:
	        List OpenAI files, optionally filtered by purpose.

	    retrieve:
	        Retrieve metadata for one OpenAI file.

	    extract:
	        Retrieve normalized content for one OpenAI file.

	    delete:
	        Delete one OpenAI file by ID.

	    summarize:
	        Summarize or analyze retrieved file content.

	    search:
	        Search retrieved file content with a user query.

	    survey:
	        Return metadata and content preview for one selected file.

    """
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	file: Optional[ Any ]
	file_id: Optional[ str ]
	filepath: Optional[ str ]
	filename: Optional[ str ]
	purpose: Optional[ str ]
	response: Optional[ Any ]
	content: Optional[ str | bytes | Dict[ str, Any ] ]
	files: Optional[ List[ Dict[ str, Any ] ] ]
	request: Optional[ Dict[ str, Any ] ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	output_text: Optional[ str ]
	
	def __init__( self, id: str=None, filepath: str=None, purpose: str='user_data',
			model: str='gpt-4o-mini', prompt: str=None ):
		"""

	        Purpose:
	        --------
	        Initialize a Files API wrapper instance.

	        Parameters:
	        -----------
	        id: str
	            Optional OpenAI file identifier.

	        filepath: str
	            Optional local file path used for upload.

	        purpose: str
	            Optional file upload or listing purpose.

	        model: str
	            Optional model used by file analysis workflows.

	        prompt: str
	            Optional prompt used by file analysis workflows.

	        Returns:
	        --------
	        None

        """
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.file = None
		self.file_id = id
		self.filepath = filepath
		self.filename = None
		self.purpose = purpose
		self.response = None
		self.content = None
		self.files = [ ]
		self.request = None
		self.model = model
		self.prompt = prompt
		self.output_text = None
	
	@property
	def upload_purpose_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return valid OpenAI file upload purposes.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[str] | None:
	            Upload purpose values.

        '''
		return [
				'assistants',
				'batch',
				'fine-tune',
				'vision',
				'user_data',
				'evals',
		]
	
	@property
	def file_purpose_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return known OpenAI file object purposes for filtering and display.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[str] | None:
	            File purpose values.

        '''
		return [
				'assistants',
				'assistants_output',
				'batch',
				'batch_output',
				'fine-tune',
				'fine-tune-results',
				'vision',
				'user_data',
				'evals',
		]
	
	@property
	def purpose_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return upload purpose options for backward compatibility.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[str] | None:
	            Upload purpose values.

        '''
		return self.upload_purpose_options
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return model options for optional file analysis workflows.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[str] | None:
	            Model names.

        '''
		return [
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o-mini',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Return conservative reasoning effort options.

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
				'minimal',
				'low',
				'medium',
				'high',
		]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Return conservative Responses API include options supported by Text mode.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Include option names.

		'''
		return [
				'file_search_call.results',
				'web_search_call.results',
				'web_search_call.action.sources',
				'code_interpreter_call.outputs',
				'reasoning.encrypted_content',
				'message.output_text.logprobs',
		]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Purpose:
			--------
			Return built-in tool options that Text mode can safely configure.

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
				'file_search',
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
	def modality_options( self ) -> List[ str ] | None:
		'''
		
			Purpose:
			--------
			Return modality options retained for compatibility.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[ str ] | None:
				Modality names.

		'''
		return [
				'text',
		]
	
	def validate_upload_purpose( self, purpose: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize an OpenAI file upload purpose.

	        Parameters:
	        -----------
	        purpose: str
	            Requested upload purpose.

	        Returns:
	        --------
	        str:
	            Valid upload purpose.

        """
		try:
			value = purpose if isinstance( purpose, str ) and purpose.strip( ) else 'user_data'
			value = value.strip( )
			
			if value not in self.upload_purpose_options:
				raise ValueError( f'Unsupported upload purpose: {value}' )
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'validate_upload_purpose( self, purpose: str=None ) -> str'
			raise exception
	
	def validate_file_id( self, id: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize an OpenAI file identifier.

	        Parameters:
	        -----------
	        id: str
	            OpenAI file identifier.

	        Returns:
	        --------
	        str:
	            Clean file identifier.

        """
		try:
			value = id if isinstance( id, str ) and id.strip( ) else self.file_id
			throw_if( 'id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'validate_file_id( self, id: str=None ) -> str'
			raise exception
	
	def normalize_file_object( self, file: Any ) -> Dict[ str, Any ]:
		"""

	        Purpose:
	        --------
	        Normalize an OpenAI FileObject or file-like response into a dictionary.

	        Parameters:
	        -----------
	        file: Any
	            OpenAI FileObject, dictionary, or file-like object.

	        Returns:
	        --------
	        Dict[str, Any]:
	            Normalized file metadata.

        """
		try:
			if file is None:
				return { }
			
			if isinstance( file, dict ):
				source = file
			elif hasattr( file, 'model_dump' ):
				source = file.model_dump( )
			else:
				source = {
						'id': getattr( file, 'id', None ),
						'bytes': getattr( file, 'bytes', None ),
						'created_at': getattr( file, 'created_at', None ),
						'expires_at': getattr( file, 'expires_at', None ),
						'filename': getattr( file, 'filename', None ),
						'object': getattr( file, 'object', None ),
						'purpose': getattr( file, 'purpose', None ),
						'status': getattr( file, 'status', None ),
						'status_details': getattr( file, 'status_details', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'filename': source.get( 'filename' ),
					'purpose': source.get( 'purpose' ),
					'bytes': source.get( 'bytes' ),
					'created_at': source.get( 'created_at' ),
					'expires_at': source.get( 'expires_at' ),
					'object': source.get( 'object' ),
					'status': source.get( 'status' ),
					'status_details': source.get( 'status_details' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'normalize_file_object( self, file: Any ) -> Dict[ str, Any ]'
			raise exception
	
	def normalize_file_list( self, response: Any, purpose: str=None ) -> List[ Dict[ str, Any ] ]:
		"""

	        Purpose:
	        --------
	        Normalize a list-files response into display-ready file metadata rows.

	        Parameters:
	        -----------
	        response: Any
	            OpenAI file list response, dictionary, or list.

	        purpose: str
	            Optional purpose filter.

	        Returns:
	        --------
	        List[Dict[str, Any]]:
	            Normalized file metadata rows.

        """
		try:
			if response is None:
				return [ ]
			
			if isinstance( response, list ):
				items = response
			elif isinstance( response, dict ):
				items = response.get( 'data', [ ] )
			else:
				items = getattr( response, 'data', [ ] )
			
			rows: List[ Dict[ str, Any ] ]=[ ]
			for item in items:
				row = self.normalize_file_object( item )
				
				if not row.get( 'id' ):
					continue
				
				if isinstance( purpose, str ) and purpose.strip( ):
					if row.get( 'purpose' ) != purpose.strip( ):
						continue
				
				rows.append( row )
			
			return rows
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'normalize_file_list( self, response: Any, purpose: str=None )'
			raise exception
	
	def normalize_file_content( self, content: Any ) -> str | bytes | Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Normalize retrieved file content into text, bytes, or a serializable dictionary.

	        Parameters:
	        -----------
	        content: Any
	            File content response returned by the OpenAI SDK.

	        Returns:
	        --------
	        str | bytes | Dict[str, Any] | None:
	            Normalized content.

        """
		try:
			if content is None:
				return None
			
			if isinstance( content, (str, bytes) ):
				return content
			
			if hasattr( content, 'read' ):
				value = content.read( )
				if isinstance( value, bytes ):
					try:
						return value.decode( 'utf-8' )
					except Exception:
						return value
				
				return value
			
			if hasattr( content, 'text' ):
				value = getattr( content, 'text' )
				if isinstance( value, str ):
					return value
			
			if hasattr( content, 'content' ):
				value = getattr( content, 'content' )
				if isinstance( value, bytes ):
					try:
						return value.decode( 'utf-8' )
					except Exception:
						return value
				
				return value
			
			if hasattr( content, 'model_dump' ):
				return content.model_dump( )
			
			return str( content )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'normalize_file_content( self, content: Any )'
			raise exception
	
	def upload( self, filepath: str, purpose: str='user_data' ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Upload a local file to the OpenAI Files API.

	        Parameters:
	        -----------
	        filepath: str
	            Local path to the file to upload.

	        purpose: str
	            OpenAI file upload purpose.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized uploaded file metadata, or None.

        """
		try:
			throw_if( 'filepath', filepath )
			
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.filepath = filepath
			self.purpose = self.validate_upload_purpose( purpose )
			self.request = {
					'file': filepath,
					'purpose': self.purpose,
			}
			
			with open( filepath, 'rb' ) as source:
				self.response = self.client.files.create(
					file=source,
					purpose=self.purpose )
			
			self.file = self.response
			metadata = self.normalize_file_object( self.response )
			self.file_id = metadata.get( 'id' )
			self.filename = metadata.get( 'filename' )
			return metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'upload( self, filepath: str, purpose: str )'
			raise exception
	
	def list( self, purpose: str=None ) -> List[ Dict[ str, Any ] ]:
		"""

	        Purpose:
	        --------
	        List OpenAI files, optionally filtered by purpose after retrieval.

	        Parameters:
	        -----------
	        purpose: str
	            Optional file purpose filter. If omitted, all returned files are listed.

	        Returns:
	        --------
	        List[Dict[str, Any]]:
	            Normalized file metadata rows.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.purpose = purpose if isinstance( purpose, str ) and purpose.strip( ) else None
			self.request = { }
			
			if self.purpose:
				self.request[ 'purpose_filter' ]=self.purpose
			
			self.response = self.client.files.list( )
			self.files = self.normalize_file_list( self.response, purpose=self.purpose )
			return self.files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'list( self, purpose: str=None ) -> List[ Dict[ str, Any ] ]'
			raise exception
	
	def retrieve( self, id: str ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Retrieve metadata for one OpenAI file.

	        Parameters:
	        -----------
	        id: str
	            OpenAI file identifier.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized file metadata, or None.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.file_id = self.validate_file_id( id )
			self.request = {
					'file_id': self.file_id,
			}
			
			self.response = self.client.files.retrieve( file_id=self.file_id )
			self.file = self.response
			metadata = self.normalize_file_object( self.response )
			self.filename = metadata.get( 'filename' )
			return metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'retrieve( self, id: str ) -> Dict[ str, Any ] | None'
			raise exception
	
	def extract( self, id: str ) -> str | bytes | Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Retrieve content for one OpenAI file.

	        Parameters:
	        -----------
	        id: str
	            OpenAI file identifier.

	        Returns:
	        --------
	        str | bytes | Dict[str, Any] | None:
	            Normalized file content.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.file_id = self.validate_file_id( id )
			self.request = {
					'file_id': self.file_id,
			}
			
			self.response = self.client.files.content( file_id=self.file_id )
			self.content = self.normalize_file_content( self.response )
			return self.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'extract( self, id: str )'
			raise exception
	
	def delete( self, id: str ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Delete one OpenAI file by file ID.

	        Parameters:
	        -----------
	        id: str
	            OpenAI file identifier.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized deletion result, or None.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.file_id = self.validate_file_id( id )
			self.request = {
					'file_id': self.file_id,
			}
			
			self.response = self.client.files.delete( file_id=self.file_id )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'id': getattr( self.response, 'id', self.file_id ),
					'deleted': getattr( self.response, 'deleted', None ),
					'object': getattr( self.response, 'object', None ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'delete( self, id: str ) -> Dict[ str, Any ] | None'
			raise exception
		
	def summarize( self, id: str, prompt: str = None, model: str = 'gpt-4o-mini',
			max_chars: int = 120000 ) -> str | None:
		"""

			Purpose:
			--------
			Summarize or analyze an uploaded OpenAI file using the Responses API input_file
			contract.

			Parameters:
			-----------
			id: str
				OpenAI file identifier.

			prompt: str
				Optional analysis prompt.

			model: str
				Model used for summarization or analysis.

			max_chars: int
				Retained for backward compatibility. The Responses API receives the file by
				file_id and does not need this local truncation value.

			Returns:
			--------
			str | None:
				Model output text, or None.

		"""
		try:
			self.file_id = self.validate_file_id( id )
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else \
				'Summarize the selected file content.'
			self.model = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-mini'
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = { 'model': self.model, 'input': [
							{
									'role': 'user',
									'content': [
											{
													'type': 'input_file',
													'file_id': self.file_id,
											},
											{
													'type': 'input_text',
													'text': self.prompt,
											}, ],
							}, ], }
			
			self.response = self.client.responses.create( **self.request )
			self.output_text = getattr( self.response, 'output_text', None )
			if self.output_text:
				return self.output_text
			
			if hasattr( self.response, 'output' ) and self.response.output:
				text_parts: List[ str ] = [ ]
				
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
			
			return str( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'summarize( self, id: str, prompt: str=None ) -> str | None'
			raise exception
	
	def search( self, id: str, query: str, model: str = 'gpt-4o-mini',
			max_chars: int = 120000 ) -> str | None:
		"""

	        Purpose:
	        --------
	        Search or question an uploaded OpenAI file using the Responses API input_file
	        contract.

	        Parameters:
	        -----------
	        id: str
	            OpenAI file identifier.

	        query: str
	            User question or search instruction.

	        model: str
	            Model used for analysis.

	        max_chars: int
	            Retained for backward compatibility.

	        Returns:
	        --------
	        str | None:
	            Model output text, or None.

        """
		try:
			throw_if( 'query', query )
			prompt = ( 'Answer the user question using the uploaded file when possible.\n\n'
					f'Question: {query}' )
			
			return self.summarize( id=id, prompt=prompt, model=model, max_chars=max_chars )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'search( self, id: str, query: str ) -> str | None'
			raise exception
	
	def survey( self, id: str, max_chars: int=4000 ) -> Dict[ str, Any ]:
		"""

	        Purpose:
	        --------
	        Return metadata and a preview of retrieved file content.

	        Parameters:
	        -----------
	        id: str
	            OpenAI file identifier.

	        max_chars: int
	            Maximum preview characters to return.

	        Returns:
	        --------
	        Dict[str, Any]:
	            Metadata and content preview.

        """
		try:
			self.file_id = self.validate_file_id( id )
			metadata = self.retrieve( self.file_id )
			content = self.extract( self.file_id )
			
			if isinstance( content, bytes ):
				try:
					content_text = content.decode( 'utf-8' )
				except Exception:
					content_text = str( content )
			elif isinstance( content, dict ):
				content_text = str( content )
			else:
				content_text = content if isinstance( content, str ) else ''
			
			preview = content_text[ :max_chars ] if isinstance( max_chars, int ) else content_text
			
			return {
					'metadata': metadata,
					'preview': preview,
					'file_id': self.file_id,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'survey( self, id: str ) -> Dict[ str, Any ]'
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
	        List[str] | None:
	            Member names.

        '''
		return [
				'api_key',
				'client',
				'file',
				'file_id',
				'filepath',
				'filename',
				'purpose',
				'response',
				'content',
				'files',
				'request',
				'model',
				'prompt',
				'output_text',
				'upload_purpose_options',
				'file_purpose_options',
				'purpose_options',
				'model_options',
				'validate_upload_purpose',
				'validate_file_id',
				'normalize_file_object',
				'normalize_file_list',
				'normalize_file_content',
				'upload',
				'list',
				'retrieve',
				'extract',
				'delete',
				'summarize',
				'search',
				'survey',
		]

class VectorStores( GPT ):
	"""
	
	    Purpose:
	    --------
	    Provides a wrapper around the OpenAI Vector Stores API, including vector store
	    management, vector store file management, file batches, native vector store search,
	    and Responses API file_search workflows.

	    Attributes:
	    -----------
	    api_key:
	        OpenAI API key loaded from config.py.

	    client:
	        OpenAI client instance.

	    name:
	        Vector store name.

	    description:
	        Optional vector store description.

	    store_id:
	        Selected or returned vector store identifier.

	    file_id:
	        Selected or returned OpenAI file identifier.

	    batch_id:
	        Selected or returned vector store file batch identifier.

	    response:
	        Last raw API response object.

	    vector_store:
	        Last normalized vector store metadata.

	    vector_stores:
	        Last normalized list of vector stores.

	    vector_file:
	        Last normalized vector store file metadata.

	    vector_files:
	        Last normalized list of vector store files.

	    file_batch:
	        Last normalized file batch metadata.

	    search_results:
	        Last normalized vector store search results.

	    output_text:
	        Last answer text generated through Responses API file_search.

	    request:
	        Last API request dictionary.

	    Methods:
	    --------
	    create:
	        Create a vector store.

	    list_stores:
	        List vector stores.

	    retrieve:
	        Retrieve vector store metadata.

	    update:
	        Update vector store metadata.

	    delete:
	        Delete a vector store.

	    attach_file:
	        Attach an OpenAI file to a vector store.

	    list:
	        Backward-compatible alias for listing vector store files.

	    list_files:
	        List files attached to a vector store.

	    retrieve_file:
	        Retrieve vector store file metadata.

	    update_file:
	        Update vector store file attributes.

	    delete_file:
	        Delete a file from a vector store.

	    retrieve_file_content:
	        Retrieve vector store file content.

	    create_file_batch:
	        Create a vector store file batch.

	    retrieve_file_batch:
	        Retrieve vector store file batch metadata.

	    list_file_batch_files:
	        List files in a vector store file batch.

	    cancel_file_batch:
	        Cancel a vector store file batch.

	    search:
	        Backward-compatible native vector store search method.

	    search_store:
	        Search a vector store using the native Vector Stores Search API.

	    answer_with_file_search:
	        Answer a question using Responses API file_search.

	    survey:
	        Run a Responses API file_search survey across one or more vector stores.

    """
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	name: Optional[ str ]
	description: Optional[ str ]
	store_id: Optional[ str ]
	file_id: Optional[ str ]
	batch_id: Optional[ str ]
	response: Optional[ Any ]
	vector_store: Optional[ Dict[ str, Any ] ]
	vector_stores: Optional[ List[ Dict[ str, Any ] ] ]
	vector_file: Optional[ Dict[ str, Any ] ]
	vector_files: Optional[ List[ Dict[ str, Any ] ] ]
	file_batch: Optional[ Dict[ str, Any ] ]
	search_results: Optional[ List[ Dict[ str, Any ] ] ]
	output_text: Optional[ str ]
	request: Optional[ Dict[ str, Any ] ]
	collections: Optional[ Dict[ str, str ] ]
	max_search_results: Optional[ int ]
	
	def __init__( self, name: str=None, store_id: str=None, file_id: str=None,
			model: str='gpt-4o-mini', max_search_results: int=10 ):
		"""

	        Purpose:
	        --------
	        Initialize a VectorStores wrapper instance.

	        Parameters:
	        -----------
	        name: str
	            Optional vector store name.

	        store_id: str
	            Optional vector store identifier.

	        file_id: str
	            Optional OpenAI file identifier.

	        model: str
	            Optional model used for Responses API file_search workflows.

	        max_search_results: int
	            Optional maximum search results value.

	        Returns:
	        --------
	        None

        """
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.name = name
		self.description = None
		self.store_id = store_id
		self.file_id = file_id
		self.batch_id = None
		self.model = model
		self.response = None
		self.vector_store = None
		self.vector_stores = [ ]
		self.vector_file = None
		self.vector_files = [ ]
		self.file_batch = None
		self.search_results = [ ]
		self.output_text = None
		self.request = None
		self.max_search_results = max_search_results
		self.collections = {
				'Guidance': 'vs_712r5W5833G6aLxIYIbuvVcK',
		}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return model options for Responses API file_search answer workflows.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[str] | None:
	            Model names.

        '''
		return [
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o-mini',
		]
	
	@property
	def ranker_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return vector store search ranker options.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[str] | None:
	            Ranker option values.

        '''
		return [
				'auto',
				'default-2024-11-15',
		]
	
	@property
	def chunking_strategy_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Return supported chunking strategy option names.

	        Parameters:
	        -----------
	        None

	        Returns:
	        --------
	        List[str] | None:
	            Chunking strategy option values.

        '''
		return [
				'auto',
				'static',
		]
	
	def validate_store_name( self, name: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize a vector store name.

	        Parameters:
	        -----------
	        name: str
	            Requested vector store name.

	        Returns:
	        --------
	        str:
	            Clean vector store name.

        """
		try:
			value = name if isinstance( name, str ) and name.strip( ) else self.name
			throw_if( 'name', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_store_name( self, name: str=None ) -> str'
			raise exception
	
	def validate_store_id( self, store_id: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize a vector store identifier.

	        Parameters:
	        -----------
	        store_id: str
	            Requested vector store identifier.

	        Returns:
	        --------
	        str:
	            Clean vector store identifier.

        """
		try:
			value = store_id if isinstance( store_id, str ) and store_id.strip( ) else self.store_id
			throw_if( 'store_id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_store_id( self, store_id: str=None ) -> str'
			raise exception
	
	def validate_file_id( self, file_id: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize an OpenAI file identifier.

	        Parameters:
	        -----------
	        file_id: str
	            Requested OpenAI file identifier.

	        Returns:
	        --------
	        str:
	            Clean OpenAI file identifier.

        """
		try:
			value = file_id if isinstance( file_id, str ) and file_id.strip( ) else self.file_id
			throw_if( 'file_id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_file_id( self, file_id: str=None ) -> str'
			raise exception
	
	def validate_batch_id( self, batch_id: str=None ) -> str:
		"""

	        Purpose:
	        --------
	        Validate and normalize a vector store file batch identifier.

	        Parameters:
	        -----------
	        batch_id: str
	            Requested vector store file batch identifier.

	        Returns:
	        --------
	        str:
	            Clean vector store file batch identifier.

        """
		try:
			value = batch_id if isinstance( batch_id, str ) and batch_id.strip( ) else self.batch_id
			throw_if( 'batch_id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_batch_id( self, batch_id: str=None ) -> str'
			raise exception
	
	def validate_file_ids( self, file_ids: List[ str ]=None ) -> List[ str ]:
		"""

	        Purpose:
	        --------
	        Validate and normalize a list of OpenAI file identifiers.

	        Parameters:
	        -----------
	        file_ids: List[str]
	            Requested OpenAI file identifiers.

	        Returns:
	        --------
	        List[str]:
	            Clean OpenAI file identifiers.

        """
		try:
			if file_ids is None:
				return [ ]
			
			values = [ ]
			for item in file_ids:
				if isinstance( item, str ) and item.strip( ):
					values.append( item.strip( ) )
			
			return values
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_file_ids( self, file_ids: List[ str ]=None )'
			raise exception
	
	def validate_max_num_results( self, max_num_results: int=None ) -> int:
		"""

	        Purpose:
	        --------
	        Validate and normalize a vector store search result limit.

	        Parameters:
	        -----------
	        max_num_results: int
	            Requested maximum search result count.

	        Returns:
	        --------
	        int:
	            Valid result count between 1 and 50.

        """
		try:
			value = self.max_search_results if max_num_results is None else int( max_num_results )
			
			if value < 1:
				return 1
			
			if value > 50:
				return 50
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_max_num_results( self, max_num_results: int=None )'
			raise exception
	
	def build_expires_after( self, anchor: str=None, days: int=None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Build an OpenAI vector store expiration policy dictionary.

	        Parameters:
	        -----------
	        anchor: str
	            Expiration anchor value.

	        days: int
	            Number of days after the anchor.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Expiration policy or None.

        """
		try:
			if days is None:
				return None
			
			value = int( days )
			if value <= 0:
				return None
			
			anchor_value = anchor if isinstance( anchor,
				str ) and anchor.strip( ) else 'last_active_at'
			
			return {
					'anchor': anchor_value.strip( ),
					'days': value,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'build_expires_after( self, anchor: str=None, days: int=None )'
			raise exception
	
	def build_chunking_strategy( self, strategy: str='auto', max_chunk_size_tokens: int=None,
			chunk_overlap_tokens: int=None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Build an OpenAI vector store chunking strategy dictionary.

	        Parameters:
	        -----------
	        strategy: str
	            Chunking strategy name: auto or static.

	        max_chunk_size_tokens: int
	            Maximum chunk size in tokens for static chunking.

	        chunk_overlap_tokens: int
	            Chunk overlap in tokens for static chunking.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Chunking strategy dictionary or None.

        """
		try:
			strategy_value = strategy if isinstance( strategy,
				str ) and strategy.strip( ) else 'auto'
			strategy_value = strategy_value.strip( )
			
			if strategy_value == 'auto':
				return { 'type': 'auto', }
			
			if strategy_value != 'static':
				return None
			
			max_value = 800 if max_chunk_size_tokens is None else int( max_chunk_size_tokens )
			overlap_value = 400 if chunk_overlap_tokens is None else int( chunk_overlap_tokens )
			
			if max_value < 100:
				max_value = 100
			
			if max_value > 4096:
				max_value = 4096
			
			if overlap_value < 0:
				overlap_value = 0
			
			if overlap_value > max_value // 2:
				overlap_value = max_value // 2
			
			return {
					'type': 'static',
					'static': {
							'max_chunk_size_tokens': max_value,
							'chunk_overlap_tokens': overlap_value,
					},
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'build_chunking_strategy( self, strategy: str, **kwargs )'
			raise exception
	
	def normalize_vector_store( self, store: Any ) -> Dict[ str, Any ]:
		"""

	        Purpose:
	        --------
	        Normalize a vector store response object into a dictionary.

	        Parameters:
	        -----------
	        store: Any
	            OpenAI vector store object, dictionary, or response.

	        Returns:
	        --------
	        Dict[str, Any]:
	            Normalized vector store metadata.

        """
		try:
			if store is None:
				return { }
			
			if isinstance( store, dict ):
				source = store
			elif hasattr( store, 'model_dump' ):
				source = store.model_dump( )
			else:
				source = {
						'id': getattr( store, 'id', None ),
						'name': getattr( store, 'name', None ),
						'description': getattr( store, 'description', None ),
						'created_at': getattr( store, 'created_at', None ),
						'object': getattr( store, 'object', None ),
						'usage_bytes': getattr( store, 'usage_bytes', None ),
						'file_counts': getattr( store, 'file_counts', None ),
						'status': getattr( store, 'status', None ),
						'expires_after': getattr( store, 'expires_after', None ),
						'expires_at': getattr( store, 'expires_at', None ),
						'last_active_at': getattr( store, 'last_active_at', None ),
						'metadata': getattr( store, 'metadata', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'name': source.get( 'name' ),
					'description': source.get( 'description' ),
					'created_at': source.get( 'created_at' ),
					'object': source.get( 'object' ),
					'usage_bytes': source.get( 'usage_bytes' ),
					'file_counts': source.get( 'file_counts' ),
					'status': source.get( 'status' ),
					'expires_after': source.get( 'expires_after' ),
					'expires_at': source.get( 'expires_at' ),
					'last_active_at': source.get( 'last_active_at' ),
					'metadata': source.get( 'metadata' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_vector_store( self, store: Any ) -> Dict[ str, Any ]'
			raise exception
	
	def normalize_vector_store_file( self, file: Any ) -> Dict[ str, Any ]:
		"""

	        Purpose:
	        --------
	        Normalize a vector store file response object into a dictionary.

	        Parameters:
	        -----------
	        file: Any
	            OpenAI vector store file object, dictionary, or response.

	        Returns:
	        --------
	        Dict[str, Any]:
	            Normalized vector store file metadata.

        """
		try:
			if file is None:
				return { }
			
			if isinstance( file, dict ):
				source = file
			elif hasattr( file, 'model_dump' ):
				source = file.model_dump( )
			else:
				source = {
						'id': getattr( file, 'id', None ),
						'object': getattr( file, 'object', None ),
						'created_at': getattr( file, 'created_at', None ),
						'vector_store_id': getattr( file, 'vector_store_id', None ),
						'status': getattr( file, 'status', None ),
						'last_error': getattr( file, 'last_error', None ),
						'chunking_strategy': getattr( file, 'chunking_strategy', None ),
						'attributes': getattr( file, 'attributes', None ),
						'usage_bytes': getattr( file, 'usage_bytes', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'object': source.get( 'object' ),
					'created_at': source.get( 'created_at' ),
					'vector_store_id': source.get( 'vector_store_id' ),
					'status': source.get( 'status' ),
					'last_error': source.get( 'last_error' ),
					'chunking_strategy': source.get( 'chunking_strategy' ),
					'attributes': source.get( 'attributes' ),
					'usage_bytes': source.get( 'usage_bytes' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_vector_store_file( self, file: Any )'
			raise exception
	
	def normalize_file_batch( self, batch: Any ) -> Dict[ str, Any ]:
		"""

	        Purpose:
	        --------
	        Normalize a vector store file batch response object into a dictionary.

	        Parameters:
	        -----------
	        batch: Any
	            OpenAI vector store file batch object, dictionary, or response.

	        Returns:
	        --------
	        Dict[str, Any]:
	            Normalized file batch metadata.

        """
		try:
			if batch is None:
				return { }
			
			if isinstance( batch, dict ):
				source = batch
			elif hasattr( batch, 'model_dump' ):
				source = batch.model_dump( )
			else:
				source = {
						'id': getattr( batch, 'id', None ),
						'object': getattr( batch, 'object', None ),
						'created_at': getattr( batch, 'created_at', None ),
						'vector_store_id': getattr( batch, 'vector_store_id', None ),
						'status': getattr( batch, 'status', None ),
						'file_counts': getattr( batch, 'file_counts', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'object': source.get( 'object' ),
					'created_at': source.get( 'created_at' ),
					'vector_store_id': source.get( 'vector_store_id' ),
					'status': source.get( 'status' ),
					'file_counts': source.get( 'file_counts' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_file_batch( self, batch: Any ) -> Dict[ str, Any ]'
			raise exception
	
	def normalize_search_results( self, response: Any ) -> List[ Dict[ str, Any ] ]:
		"""

	        Purpose:
	        --------
	        Normalize native vector store search results into dictionaries.

	        Parameters:
	        -----------
	        response: Any
	            OpenAI vector store search response.

	        Returns:
	        --------
	        List[Dict[str, Any]]:
	            Normalized search result rows.

        """
		try:
			if response is None:
				return [ ]
			
			if isinstance( response, dict ):
				items = response.get( 'data', [ ] )
			elif isinstance( response, list ):
				items = response
			else:
				items = getattr( response, 'data', [ ] )
			
			rows: List[ Dict[ str, Any ] ]=[ ]
			for item in items:
				if isinstance( item, dict ):
					source = item
				elif hasattr( item, 'model_dump' ):
					source = item.model_dump( )
				else:
					source = {
							'file_id': getattr( item, 'file_id', None ),
							'filename': getattr( item, 'filename', None ),
							'score': getattr( item, 'score', None ),
							'attributes': getattr( item, 'attributes', None ),
							'content': getattr( item, 'content', None ),
					}
				
				rows.append(
					{
							'file_id': source.get( 'file_id' ),
							'filename': source.get( 'filename' ),
							'score': source.get( 'score' ),
							'attributes': source.get( 'attributes' ),
							'content': source.get( 'content' ),
					} )
			
			return rows
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_search_results( self, response: Any )'
			raise exception
	
	def create( self, name: str, description: str=None, metadata: Dict[ str, Any ]=None,
			expires_after: Dict[ str, Any ]=None, file_ids: List[ str ]=None,
			chunking_strategy: Dict[ str, Any ]=None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Create an OpenAI vector store.

	        Parameters:
	        -----------
	        name: str
	            Vector store name.

	        description: str
	            Optional vector store description.

	        metadata: Dict[str, Any]
	            Optional vector store metadata.

	        expires_after: Dict[str, Any]
	            Optional expiration policy.

	        file_ids: List[str]
	            Optional OpenAI file IDs to attach on creation.

	        chunking_strategy: Dict[str, Any]
	            Optional chunking strategy.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized vector store metadata.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.name = self.validate_store_name( name )
			self.description = description if isinstance( description,
				str ) and description.strip( ) else None
			
			self.request = {
					'name': self.name,
			}
			
			if self.description:
				self.request[ 'description' ]=self.description
			
			if isinstance( metadata, dict ) and len( metadata ) > 0:
				self.request[ 'metadata' ]=metadata
			
			if isinstance( expires_after, dict ) and len( expires_after ) > 0:
				self.request[ 'expires_after' ]=expires_after
			
			clean_file_ids = self.validate_file_ids( file_ids )
			if len( clean_file_ids ) > 0:
				self.request[ 'file_ids' ]=clean_file_ids
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ]=chunking_strategy
			
			self.response = self.client.vector_stores.create( **self.request )
			self.vector_store = self.normalize_vector_store( self.response )
			self.store_id = self.vector_store.get( 'id' )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'create( self, name: str, **kwargs ) -> Dict[ str, Any ] | None'
			raise exception
	
	def list_stores( self, limit: int=100, order: str='desc',
			after: str=None, before: str=None ) -> List[ Dict[ str, Any ] ]:
		"""

	        Purpose:
	        --------
	        List OpenAI vector stores.

	        Parameters:
	        -----------
	        limit: int
	            Maximum number of vector stores to return.

	        order: str
	            Sort order.

	        after: str
	            Optional pagination cursor.

	        before: str
	            Optional pagination cursor.

	        Returns:
	        --------
	        List[Dict[str, Any]]:
	            Normalized vector store rows.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = {
					'limit': limit,
					'order': order,
			}
			
			if isinstance( after, str ) and after.strip( ):
				self.request[ 'after' ]=after.strip( )
			
			if isinstance( before, str ) and before.strip( ):
				self.request[ 'before' ]=before.strip( )
			
			self.response = self.client.vector_stores.list( **self.request )
			items = getattr( self.response, 'data', [ ] )
			self.vector_stores = [ self.normalize_vector_store( item ) for item in items ]
			return self.vector_stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_stores( self, limit: int=100 )'
			raise exception
	
	def retrieve( self, store_id: str ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Retrieve one OpenAI vector store by ID.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized vector store metadata.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.request = {
					'vector_store_id': self.store_id,
			}
			
			self.response = self.client.vector_stores.retrieve(
				vector_store_id=self.store_id )
			self.vector_store = self.normalize_vector_store( self.response )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve( self, store_id: str ) -> Dict[ str, Any ] | None'
			raise exception
	
	def update( self, store_id: str, name: str=None, description: str=None,
			metadata: Dict[ str, Any ]=None,
			expires_after: Dict[ str, Any ]=None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Update one OpenAI vector store.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        name: str
	            Optional new vector store name.

	        description: str
	            Optional new vector store description.

	        metadata: Dict[str, Any]
	            Optional metadata dictionary.

	        expires_after: Dict[str, Any]
	            Optional expiration policy.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized updated vector store metadata.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.request = { }
			
			if isinstance( name, str ) and name.strip( ):
				self.request[ 'name' ]=name.strip( )
			
			if isinstance( description, str ) and description.strip( ):
				self.request[ 'description' ]=description.strip( )
			
			if isinstance( metadata, dict ):
				self.request[ 'metadata' ]=metadata
			
			if isinstance( expires_after, dict ) and len( expires_after ) > 0:
				self.request[ 'expires_after' ]=expires_after
			
			if len( self.request ) == 0:
				return self.retrieve( self.store_id )
			
			self.response = self.client.vector_stores.update(
				vector_store_id=self.store_id,
				**self.request )
			
			self.vector_store = self.normalize_vector_store( self.response )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'update( self, store_id: str, **kwargs )'
			raise exception
	
	def delete( self, store_id: str ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Delete one OpenAI vector store by ID.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized delete result.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.request = {
					'vector_store_id': self.store_id,
			}
			
			self.response = self.client.vector_stores.delete(
				vector_store_id=self.store_id )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'id': getattr( self.response, 'id', self.store_id ),
					'deleted': getattr( self.response, 'deleted', None ),
					'object': getattr( self.response, 'object', None ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'delete( self, store_id: str ) -> Dict[ str, Any ] | None'
			raise exception
	
	def list( self, store_id: str, limit: int=100, order: str='desc' ) -> List[ Dict[ str, Any ] ]:
		"""

	        Purpose:
	        --------
	        Backward-compatible alias for listing vector store files.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        limit: int
	            Maximum number of vector store files to return.

	        order: str
	            Sort order.

	        Returns:
	        --------
	        List[Dict[str, Any]]:
	            Normalized vector store file rows.

        """
		try:
			return self.list_files( store_id=store_id, limit=limit, order=order )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list( self, store_id: str ) -> List[ Dict[ str, Any ] ]'
			raise exception
	
	def list_files( self, store_id: str, limit: int=100,
			order: str='desc' ) -> List[ Dict[ str, Any ] ]:
		"""

	        Purpose:
	        --------
	        List files attached to a vector store.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        limit: int
	            Maximum number of files to return.

	        order: str
	            Sort order.

	        Returns:
	        --------
	        List[Dict[str, Any]]:
	            Normalized vector store file rows.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.request = {
					'limit': limit,
					'order': order,
			}
			
			self.response = self.client.vector_stores.files.list(
				vector_store_id=self.store_id,
				**self.request )
			
			items = getattr( self.response, 'data', [ ] )
			self.vector_files = [ self.normalize_vector_store_file( item ) for item in items ]
			return self.vector_files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_files( self, store_id: str )'
			raise exception
	
	def retrieve_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Retrieve one vector store file metadata object.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        file_id: str
	            OpenAI file identifier.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized vector store file metadata.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			
			self.response = self.client.vector_stores.files.retrieve(
				vector_store_id=self.store_id,
				file_id=self.file_id )
			
			self.vector_file = self.normalize_vector_store_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file( self, store_id: str, file_id: str )'
			raise exception
		
	def attach_file( self, store_id: str = None, file_id: str = None,
			vector_store_id: str = None, attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""

			Purpose:
			--------
			Attach an existing OpenAI file to a vector store.

			Parameters:
			-----------
			store_id: str
				OpenAI vector store identifier.

			file_id: str
				OpenAI file identifier.

			vector_store_id: str
				OpenAI vector store identifier accepted for app.py compatibility.

			attributes: Dict[str, Any]
				Optional file attributes.

			chunking_strategy: Dict[str, Any]
				Optional file chunking strategy.

			Returns:
			--------
			Dict[str, Any] | None:
				Normalized vector store file metadata.

		"""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			selected_store_id = vector_store_id if isinstance(
				vector_store_id, str ) and vector_store_id.strip( ) else store_id
			self.store_id = self.validate_store_id( selected_store_id )
			self.file_id = self.validate_file_id( file_id )
			self.request = {
					'file_id': self.file_id,
			}
			
			if isinstance( attributes, dict ) and len( attributes ) > 0:
				self.request[ 'attributes' ] = attributes
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ] = chunking_strategy
			
			self.response = self.client.vector_stores.files.create(
				vector_store_id=self.store_id,
				**self.request )
			
			self.vector_file = self.normalize_vector_store_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'attach_file( self, store_id: str=None, file_id: str=None )'
			raise exception
	
	def upload_and_attach( self, store_id: str = None, vector_store_id: str = None,
			path: str = None, filepath: str = None, purpose: str = 'assistants',
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Upload a local file to OpenAI and attach the uploaded file to a vector store.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        vector_store_id: str
	            OpenAI vector store identifier accepted for app.py compatibility.

	        path: str
	            Local file path.

	        filepath: str
	            Local file path accepted for compatibility with file wrappers.

	        purpose: str
	            OpenAI file upload purpose.

	        attributes: Dict[str, Any]
	            Optional vector store file attributes.

	        chunking_strategy: Dict[str, Any]
	            Optional file chunking strategy.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Combined upload and vector store attachment result.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			selected_store_id = vector_store_id if isinstance(
				vector_store_id, str ) and vector_store_id.strip( ) else store_id
			selected_path = path if isinstance( path, str ) and path.strip( ) else filepath
			self.store_id = self.validate_store_id( selected_store_id )
			throw_if( 'path', selected_path )
			upload_purpose = purpose if isinstance( purpose, str ) and purpose.strip( ) else \
				'assistants'
			
			with open( selected_path, 'rb' ) as source:
				uploaded_file = self.client.files.create(
					file=source,
					purpose=upload_purpose )
			
			if hasattr( uploaded_file, 'model_dump' ):
				file_metadata = uploaded_file.model_dump( )
			else:
				file_metadata = {
						'id': getattr( uploaded_file, 'id', None ),
						'object': getattr( uploaded_file, 'object', None ),
						'bytes': getattr( uploaded_file, 'bytes', None ),
						'created_at': getattr( uploaded_file, 'created_at', None ),
						'filename': getattr( uploaded_file, 'filename', None ),
						'purpose': getattr( uploaded_file, 'purpose', upload_purpose ),
				}
			
			self.file_id = file_metadata.get( 'id' )
			self.vector_file = self.attach_file(
				store_id=self.store_id,
				file_id=self.file_id,
				attributes=attributes,
				chunking_strategy=chunking_strategy )
			
			return {
					'file': file_metadata,
					'vector_store_file': self.vector_file,
					'file_id': self.file_id,
					'vector_store_id': self.store_id,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'upload_and_attach( self, store_id: str=None, path: str=None )'
			raise exception
	
	def upload_file( self, store_id: str = None, vector_store_id: str = None,
			path: str = None, filepath: str = None, purpose: str = 'assistants',
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Compatibility alias for uploading and attaching a file to a vector store.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        vector_store_id: str
	            OpenAI vector store identifier accepted for app.py compatibility.

	        path: str
	            Local file path.

	        filepath: str
	            Local file path accepted for compatibility.

	        purpose: str
	            OpenAI file upload purpose.

	        attributes: Dict[str, Any]
	            Optional file attributes.

	        chunking_strategy: Dict[str, Any]
	            Optional chunking strategy.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Combined upload and vector store attachment result.

        """
		try:
			return self.upload_and_attach(
				store_id=store_id,
				vector_store_id=vector_store_id,
				path=path,
				filepath=filepath,
				purpose=purpose,
				attributes=attributes,
				chunking_strategy=chunking_strategy )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'upload_file( self, store_id: str=None, path: str=None )'
			raise exception
	
	def attach_upload( self, store_id: str = None, vector_store_id: str = None,
			path: str = None, filepath: str = None, purpose: str = 'assistants',
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Compatibility alias for uploading and attaching a file to a vector store.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        vector_store_id: str
	            OpenAI vector store identifier accepted for app.py compatibility.

	        path: str
	            Local file path.

	        filepath: str
	            Local file path accepted for compatibility.

	        purpose: str
	            OpenAI file upload purpose.

	        attributes: Dict[str, Any]
	            Optional file attributes.

	        chunking_strategy: Dict[str, Any]
	            Optional chunking strategy.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Combined upload and vector store attachment result.

        """
		try:
			return self.upload_and_attach(
				store_id=store_id,
				vector_store_id=vector_store_id,
				path=path,
				filepath=filepath,
				purpose=purpose,
				attributes=attributes,
				chunking_strategy=chunking_strategy )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'attach_upload( self, store_id: str=None, path: str=None )'
			raise exception
	
	def upload( self, store_id: str = None, vector_store_id: str = None,
			path: str = None, filepath: str = None, purpose: str = 'assistants',
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Compatibility alias for uploading and attaching a file to a vector store.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        vector_store_id: str
	            OpenAI vector store identifier accepted for app.py compatibility.

	        path: str
	            Local file path.

	        filepath: str
	            Local file path accepted for compatibility.

	        purpose: str
	            OpenAI file upload purpose.

	        attributes: Dict[str, Any]
	            Optional file attributes.

	        chunking_strategy: Dict[str, Any]
	            Optional chunking strategy.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Combined upload and vector store attachment result.

        """
		try:
			return self.upload_and_attach(
				store_id=store_id,
				vector_store_id=vector_store_id,
				path=path,
				filepath=filepath,
				purpose=purpose,
				attributes=attributes,
				chunking_strategy=chunking_strategy )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'upload( self, store_id: str=None, path: str=None )'
			raise exception
	
	def create_file_batch( self, store_id: str = None, file_ids: List[ str ] = None,
			vector_store_id: str = None, attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Create a vector store file batch.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        file_ids: List[str]
	            OpenAI file identifiers.

	        vector_store_id: str
	            OpenAI vector store identifier accepted for app.py compatibility.

	        attributes: Dict[str, Any]
	            Optional attributes applied to files.

	        chunking_strategy: Dict[str, Any]
	            Optional chunking strategy.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized file batch metadata.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			selected_store_id = vector_store_id if isinstance(
				vector_store_id, str ) and vector_store_id.strip( ) else store_id
			self.store_id = self.validate_store_id( selected_store_id )
			clean_file_ids = self.validate_file_ids( file_ids )
			throw_if( 'file_ids', clean_file_ids )
			
			if len( clean_file_ids ) > 2000:
				raise ValueError( 'Vector store file batches cannot exceed 2000 files.' )
			
			self.request = {
					'file_ids': clean_file_ids,
			}
			
			if isinstance( attributes, dict ) and len( attributes ) > 0:
				self.request[ 'attributes' ] = attributes
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ] = chunking_strategy
			
			self.response = self.client.vector_stores.file_batches.create(
				vector_store_id=self.store_id,
				**self.request )
			
			self.file_batch = self.normalize_file_batch( self.response )
			self.batch_id = self.file_batch.get( 'id' )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'create_file_batch( self, store_id: str=None, file_ids: List[ str ]=None )'
			raise exception
		
	def update_file( self, store_id: str, file_id: str,
			attributes: Dict[ str, Any ]=None ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Update vector store file attributes.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        file_id: str
	            OpenAI file identifier.

	        attributes: Dict[str, Any]
	            File attributes to apply.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized vector store file metadata.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			self.request = { }
			
			if isinstance( attributes, dict ):
				self.request[ 'attributes' ]=attributes
			
			self.response = self.client.vector_stores.files.update(
				vector_store_id=self.store_id,
				file_id=self.file_id,
				**self.request )
			
			self.vector_file = self.normalize_vector_store_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'update_file( self, store_id: str, file_id: str )'
			raise exception
	
	def delete_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Delete a file from a vector store.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        file_id: str
	            OpenAI file identifier.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized delete result.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			
			self.response = self.client.vector_stores.files.delete(
				vector_store_id=self.store_id,
				file_id=self.file_id )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'id': getattr( self.response, 'id', self.file_id ),
					'deleted': getattr( self.response, 'deleted', None ),
					'object': getattr( self.response, 'object', None ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'delete_file( self, store_id: str, file_id: str )'
			raise exception
	
	def retrieve_file_content( self, store_id: str, file_id: str ) -> Any:
		"""

	        Purpose:
	        --------
	        Retrieve content for a vector store file.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        file_id: str
	            OpenAI file identifier.

	        Returns:
	        --------
	        Any:
	            Vector store file content response.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			
			self.response = self.client.vector_stores.files.content(
				vector_store_id=self.store_id,
				file_id=self.file_id )
			
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file_content( self, store_id: str, file_id: str )'
			raise exception
	
	def retrieve_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Retrieve one vector store file batch.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        batch_id: str
	            Vector store file batch identifier.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized file batch metadata.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.batch_id = self.validate_batch_id( batch_id )
			
			self.response = self.client.vector_stores.file_batches.retrieve(
				vector_store_id=self.store_id,
				batch_id=self.batch_id )
			
			self.file_batch = self.normalize_file_batch( self.response )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file_batch( self, store_id: str, batch_id: str )'
			raise exception
	
	def list_file_batch_files( self, store_id: str, batch_id: str,
			limit: int=100 ) -> List[ Dict[ str, Any ] ]:
		"""

	        Purpose:
	        --------
	        List files in a vector store file batch.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        batch_id: str
	            Vector store file batch identifier.

	        limit: int
	            Maximum number of files to return.

	        Returns:
	        --------
	        List[Dict[str, Any]]:
	            Normalized vector store file rows.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.batch_id = self.validate_batch_id( batch_id )
			
			self.response = self.client.vector_stores.file_batches.files.list(
				vector_store_id=self.store_id,
				batch_id=self.batch_id,
				limit=limit )
			
			items = getattr( self.response, 'data', [ ] )
			self.vector_files = [ self.normalize_vector_store_file( item ) for item in items ]
			return self.vector_files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_file_batch_files( self, store_id: str, batch_id: str )'
			raise exception
	
	def cancel_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ] | None:
		"""

	        Purpose:
	        --------
	        Cancel a vector store file batch.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        batch_id: str
	            Vector store file batch identifier.

	        Returns:
	        --------
	        Dict[str, Any] | None:
	            Normalized file batch metadata.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.batch_id = self.validate_batch_id( batch_id )
			
			self.response = self.client.vector_stores.file_batches.cancel(
				vector_store_id=self.store_id,
				batch_id=self.batch_id )
			
			self.file_batch = self.normalize_file_batch( self.response )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'cancel_file_batch( self, store_id: str, batch_id: str )'
			raise exception
	
	def search( self, store_id: str, query: str, max_num_results: int=10,
			filters: Dict[ str, Any ]=None, ranking_options: Dict[ str, Any ]=None,
			rewrite_query: bool = None ) -> List[ Dict[ str, Any ] ]:
		"""

	        Purpose:
	        --------
	        Backward-compatible native vector store search method.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        query: str
	            Search query.

	        max_num_results: int
	            Maximum number of results.

	        filters: Dict[str, Any]
	            Optional attribute filters.

	        ranking_options: Dict[str, Any]
	            Optional ranking options.

	        rewrite_query: bool
	            Optional query rewriting flag.

	        Returns:
	        --------
	        List[Dict[str, Any]]:
	            Normalized search results.

        """
		try:
			return self.search_store( store_id=store_id, query=query, max_num_results=max_num_results,
				filters=filters, ranking_options=ranking_options, rewrite_query=rewrite_query )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search( self, store_id: str, query: str )'
			raise exception
	
	def search_store( self, store_id: str, query: str, max_num_results: int=10,
			filters: Dict[ str, Any ]=None, ranking_options: Dict[ str, Any ]=None,
			rewrite_query: bool = None ) -> List[ Dict[ str, Any ] ]:
		"""

	        Purpose:
	        --------
	        Search a vector store using the native OpenAI Vector Stores Search API.

	        Parameters:
	        -----------
	        store_id: str
	            OpenAI vector store identifier.

	        query: str
	            Search query.

	        max_num_results: int
	            Maximum number of results.

	        filters: Dict[str, Any]
	            Optional attribute filters.

	        ranking_options: Dict[str, Any]
	            Optional ranking options.

	        rewrite_query: bool
	            Optional query rewriting flag.

	        Returns:
	        --------
	        List[Dict[str, Any]]:
	            Normalized search results.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			throw_if( 'query', query )
			
			self.request = {
					'query': query.strip( ),
					'max_num_results': self.validate_max_num_results( max_num_results ),
			}
			
			if isinstance( filters, dict ) and len( filters ) > 0:
				self.request[ 'filters' ]=filters
			
			if isinstance( ranking_options, dict ) and len( ranking_options ) > 0:
				self.request[ 'ranking_options' ]=ranking_options
			
			if isinstance( rewrite_query, bool ):
				self.request[ 'rewrite_query' ]=rewrite_query
			
			self.response = self.client.vector_stores.search(
				vector_store_id=self.store_id,
				**self.request )
			
			self.search_results = self.normalize_search_results( self.response )
			return self.search_results
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search_store( self, store_id: str, query: str )'
			raise exception
	
	def answer_with_file_search( self, store_ids: List[ str ], prompt: str,
			model: str='gpt-4o-mini', max_num_results: int=10,
			instructions: str=None ) -> str | None:
		"""

	        Purpose:
	        --------
	        Answer a prompt using Responses API file_search over vector store IDs.

	        Parameters:
	        -----------
	        store_ids: List[str]
	            Vector store identifiers.

	        prompt: str
	            User prompt.

	        model: str
	            Model used by the Responses API.

	        max_num_results: int
	            Maximum file_search results.

	        instructions: str
	            Optional system/developer instructions.

	        Returns:
	        --------
	        str | None:
	            Response output text.

        """
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			clean_store_ids = [
					item.strip( ) for item in store_ids
					if isinstance( item, str ) and item.strip( )
			]
			
			throw_if( 'store_ids', clean_store_ids )
			throw_if( 'prompt', prompt )
			
			model_value = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-mini'
			
			input_items: List[ Dict[ str, Any ] ]=[ ]
			if isinstance( instructions, str ) and instructions.strip( ):
				input_items.append(
					{
							'role': 'developer',
							'content': [
									{
											'type': 'input_text',
											'text': instructions.strip( ),
									}, ],
					} )
			
			input_items.append(
				{
						'role': 'user',
						'content': [
								{
										'type': 'input_text',
										'text': prompt.strip( ),
								}, ],
				} )
			
			self.request = {
					'model': model_value,
					'input': input_items,
					'tools': [
							{
									'type': 'file_search',
									'vector_store_ids': clean_store_ids,
									'max_num_results': self.validate_max_num_results(
										max_num_results ),
							}, ],
			}
			
			self.response = self.client.responses.create( **self.request )
			self.output_text = getattr( self.response, 'output_text', None )
			
			if self.output_text:
				return self.output_text
			
			return str( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'answer_with_file_search( self, store_ids: List[ str ], prompt: str )'
			raise exception
	
	def survey( self, store_ids: List[ str ], prompt: str=None, model: str='gpt-4o-mini',
			max_num_results: int=10, instructions: str=None ) -> str | None:
		"""

	        Purpose:
	        --------
	        Run a Responses API file_search survey across one or more vector stores.

	        Parameters:
	        -----------
	        store_ids: List[str]
	            Vector store identifiers.

	        prompt: str
	            Optional survey prompt.

	        model: str
	            Model used by the Responses API.

	        max_num_results: int
	            Maximum file_search result count.

	        instructions: str
	            Optional system/developer instructions.

	        Returns:
	        --------
	        str | None:
	            Survey response text.

        """
		try:
			query = prompt if isinstance( prompt, str ) and prompt.strip( ) else \
				'Summarize the most relevant information available in the selected vector stores.'
			
			return self.answer_with_file_search(
				store_ids=store_ids,
				prompt=query,
				model=model,
				max_num_results=max_num_results,
				instructions=instructions )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'survey( self, store_ids: List[ str ], prompt: str=None )'
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
	        List[str] | None:
	            Member names.

        '''
		return [
				'api_key',
				'client',
				'name',
				'description',
				'store_id',
				'file_id',
				'batch_id',
				'model',
				'response',
				'vector_store',
				'vector_stores',
				'vector_file',
				'vector_files',
				'file_batch',
				'search_results',
				'output_text',
				'request',
				'collections',
				'max_search_results',
				'model_options',
				'ranker_options',
				'chunking_strategy_options',
				'validate_store_name',
				'validate_store_id',
				'validate_file_id',
				'validate_batch_id',
				'validate_file_ids',
				'validate_max_num_results',
				'build_expires_after',
				'build_chunking_strategy',
				'normalize_vector_store',
				'normalize_vector_store_file',
				'normalize_file_batch',
				'normalize_search_results',
				'create',
				'list_stores',
				'retrieve',
				'update',
				'delete',
				'attach_file',
				'list',
				'list_files',
				'retrieve_file',
				'update_file',
				'delete_file',
				'retrieve_file_content',
				'create_file_batch',
				'retrieve_file_batch',
				'list_file_batch_files',
				'cancel_file_batch',
				'search',
				'search_store',
				'answer_with_file_search',
				'survey',
		]
		