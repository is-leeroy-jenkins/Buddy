"""OpenAI provider wrapper for Buddy.


	Purpose:
	    Provides OpenAI-backed chat, image, audio, embedding, file, and vector-store workflows used by the Buddy Streamlit application and MkDocs API reference.

	Notes:
	    The module preserves the project provider-wrapper pattern, local validation helpers, OpenAI request builders, and wrapped exception handling used by the application.
"""
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
from boogr import Error, Logger
import config as cfg
import tempfile

def throw_if( name: str, value: object ) -> None:
	"""Throw if.
	
	
		Purpose:
		    Validates a required argument before provider request construction or local workflow execution.
	
		Args:
		    name: Resource, argument, or store name to validate or use.
		    value: Candidate value checked for required input validation.
	
		Raises:
		    ValueError: Raised when required input validation fails or unsupported provider options are supplied.
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
		    Reads a local image file and converts its bytes to a base64 string for vision-capable provider requests.
	
		Args:
		    image_path: Local image path read and encoded for provider requests.
	
		Returns:
		    str: Result produced by the provider workflow.
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class GPT:
	"""GPT provider wrapper.
	
	
		Purpose:
		    Stores shared OpenAI configuration, credential state, prompt parameters, and response settings used by the provider-specific workflow classes.
	
		Attributes:
		    api_key: OpenAI API key loaded from project configuration.
		    client: OpenAI client created for the current provider operation.
		    prompt: User prompt or task instruction used by the current request.
		    temperature: Sampling temperature value retained for compatible models.
		    top_percent: Nucleus sampling value retained for compatible models.
		    frequency_penalty: Frequency penalty value retained for compatible models.
		    presence_penalty: Presence penalty value retained for compatible models.
		    max_tokens: Maximum output-token value used by supported requests.
		    stops: Stop-sequence collection retained for compatible request types.
		    store: Response storage flag used by supported OpenAI requests.
		    stream: Streaming flag retained by UI workflows and request builders.
		    background: Background-execution flag retained by UI workflows and request builders.
		    number: Requested count for generated candidates or image outputs.
		    response_format: Response-format configuration used by text or media requests.
		    context: Prior conversation or document context supplied to request builders.
		    instructions: System or developer instructions supplied to model requests.
	
		Notes:
		    The wrapper stores request state on the instance so Streamlit callbacks, provider calls, and documentation-generated API pages expose consistent runtime behavior.
	"""
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
	
	def __init__( self ):
		"""Initialize GPT.
		
		
			Purpose:
			    Initializes GPT state by assigning configuration values, request defaults, cached outputs, and compatibility fields used by later methods.
		"""
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
	"""Chat provider wrapper.
	
	
		Purpose:
		    Builds and executes OpenAI Responses API chat requests, including tool configuration, retrieval inputs, prompt templates, response extraction, and usage metadata.
	
		Attributes:
		    include: Responses API include fields requested by the current workflow.
		    tool_choice: Tool-choice policy selected for the current request.
		    previous_id: Previous Responses API identifier used for stateful continuation.
		    conversation_id: Responses API conversation identifier used for continuation.
		    parallel_tools: Flag controlling parallel tool-call support when tools are active.
		    max_tools: Maximum tool-call count sent with supported Responses API requests.
		    input: Responses API input payload built for the current request.
		    tools: Tool definitions selected or built for the current request.
		    reasoning: Reasoning-effort configuration used by supported models.
		    image_url: Remote image URL used by image-analysis workflows.
		    image_path: Local image path used by image-analysis or image-editing workflows.
		    file_url: File URL retained for compatibility with document workflows.
		    file_path: Local file path retained for file-enabled workflows.
		    allowed_domains: Allowed-domain filters used by web-search tool configuration.
		    max_search_results: Maximum number of search results requested by supported tools.
		    output_text: Text extracted from the latest provider response.
		    vector_stores: Named OpenAI vector-store identifiers available to the application.
		    files: Named OpenAI file identifiers available to the application.
		    content: Supplemental content block retained for request construction.
		    vector_store_ids: Vector-store identifiers used by file-search tools.
		    file_ids: File identifiers retained for file-enabled workflows.
		    response: Latest provider response object returned by an API call.
		    file: Latest file object returned by an OpenAI file workflow.
		    purpose: OpenAI file purpose used by upload and file operations.
	
		Notes:
		    The wrapper stores request state on the instance so Streamlit callbacks, provider calls, and documentation-generated API pages expose consistent runtime behavior.
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
		"""Initialize Chat.
		
		
			Purpose:
			    Initializes Chat state by assigning configuration values, request defaults, cached outputs, and compatibility fields used by later methods.
		
			Args:
			    model: Provider model identifier selected for the operation.
			    prompt: User prompt or task instruction submitted to the provider.
			    temperature: Sampling temperature supplied to compatible model requests.
			    top_p: Nucleus sampling value supplied to compatible model requests.
			    presense: Backward-compatible misspelled presence-penalty argument.
			    presence: Presence penalty supplied to compatible model requests.
			    store: Response storage flag supplied to compatible provider requests.
			    stream: Streaming flag retained by the UI and compatible provider requests.
			    stops: Stop sequences retained for compatible provider requests.
			    response_format: Response-format configuration retained for compatible workflows.
			    number: Requested output count before provider-specific normalization.
			    instruct: System or developer instructions supplied to the provider.
			    context: Prior conversation context supplied to request builders.
			    allowed_domains: Allowed web-search domains supplied to tool configuration.
			    include: Requested provider include fields.
			    tools: Tool selections or provider tool dictionaries supplied by the UI.
			    max_tools: Maximum number of tool calls allowed for the request.
			    tool_choice: Tool-choice policy selected for the request.
			    file_path: Output or input file path used by the workflow.
			    background: Background transparency or execution option supplied by the caller.
			    is_parallel: Flag controlling parallel tool-call support when tools are active.
			    max_tokens: Maximum output-token value supplied to compatible requests.
			    frequency: Frequency penalty supplied to compatible model requests.
			    input: Prebuilt provider input payload supplied by the caller.
			    file_ids: OpenAI file identifiers attached to a vector store or batch.
			    previous_id: Previous Responses API identifier used for stateful continuation.
			    conversation_id: Conversation identifier used for stateful continuation.
			    reasoning: Reasoning-effort value or reasoning configuration supplied by the UI.
			    output_text: Previously extracted output text retained for compatibility.
			    max_search_results: Maximum search-result count retained for compatible tools.
			    content: Supplemental content block supplied to request construction.
			    vector_store_ids: Vector-store identifiers used by file-search tools.
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
		"""Model options.
		
		
			Purpose:
			    Returns model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Model option names exposed to the UI selector.
		"""
		return [ 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'gpt-5.2', 'gpt-5.1',
		         'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-4.1-mini',
		         'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		
			Purpose:
			    Returns include options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Include option names exposed to the UI selector.
		"""
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
		"""Tool options.
		
		
			Purpose:
			    Returns tool options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Tool option names exposed to the UI selector.
		"""
		return [
				'web_search',
				'file_search',
				'code_interpreter',
		]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		
			Purpose:
			    Returns choice options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Tool-choice option names exposed to the UI selector.
		"""
		return [ 'auto', 'required', 'none', ]
	
	@property
	def purpose_options( self ) -> List[ str ] | None:
		"""Purpose options.
		
		
			Purpose:
			    Returns purpose options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: File-purpose option names exposed to the UI selector.
		"""
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
		"""Format options.
		
		
			Purpose:
			    Returns format options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Response-format option names exposed to the UI selector.
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
			    Returns reasoning options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Reasoning-effort option names exposed to the UI selector.
		"""
		return [
				'none',
				'minimal',
				'low',
				'medium',
				'high',
		]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		
			Purpose:
			    Returns modality options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Modality option names exposed to the UI selector.
		"""
		return [ 'text', ]
	
	def build_reasoning( self, reasoning: str | Dict[ str, str ] = None ) -> Dict[
		                                                                         str, str ] | None:
		"""Build reasoning.
		
		
			Purpose:
			    Builds the reasoning structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    reasoning: Reasoning-effort value or reasoning configuration supplied by the UI.
		
			Returns:
			    Dict[ str, str ] | None: Provider reasoning configuration or None when reasoning is not active.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_input( self, prompt: str, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""Build input.
		
		
			Purpose:
			    Builds the input structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    prompt: User prompt or task instruction submitted to the provider.
			    context: Prior conversation context supplied to request builders.
			    input_data: Prebuilt provider input payload supplied by the caller.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Responses API input payload for the current request.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None,
			vector_store_ids: List[ str ] = None ) -> List[ Dict[ str, Any ] ] | None:
		"""Build tools.
		
		
			Purpose:
			    Builds the tools structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    tools: Tool selections or provider tool dictionaries supplied by the UI.
			    allowed_domains: Allowed web-search domains supplied to tool configuration.
			    vector_store_ids: Vector-store identifiers used by file-search tools.
		
			Returns:
			    List[ Dict[ str, Any ] ] | None: Normalized provider tool definitions or None when no supported tools are active.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_tool_choice( self, tool_choice: str = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> str | None:
		"""Build tool choice.
		
		
			Purpose:
			    Builds the tool choice structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    tool_choice: Tool-choice policy selected for the request.
			    tools: Tool selections or provider tool dictionaries supplied by the UI.
		
			Returns:
			    str | None: Validated tool-choice policy or None when not applicable.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_include( self, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> List[ str ] | None:
		"""Build include.
		
		
			Purpose:
			    Builds the include structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    include: Requested provider include fields.
			    tools: Tool selections or provider tool dictionaries supplied by the UI.
		
			Returns:
			    List[ str ] | None: Filtered include fields or None when no include values are valid.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_text_format( self, format: Dict[ str, Any ] | str = None ) -> Dict[ str, Any ] | None:
		"""Build text format.
		
		
			Purpose:
			    Builds the text format structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    format: Output or response format selected for the operation.
		
			Returns:
			    Dict[ str, Any ] | None: Provider text-format configuration or None when no format is active.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_prompt_template( self, prompt_id: str = None, prompt_version: str = None,
			prompt_variables: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Build prompt template.
		
		
			Purpose:
			    Builds the prompt template structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    prompt_id: OpenAI prompt-template identifier.
			    prompt_version: OpenAI prompt-template version.
			    prompt_variables: Variables supplied to an OpenAI prompt template.
		
			Returns:
			    Dict[ str, Any ] | None: Prompt-template reference dictionary or None when no template is selected.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
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
		"""Build request.
		
		
			Purpose:
			    Builds the request structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    prompt: User prompt or task instruction submitted to the provider.
			    model: Provider model identifier selected for the operation.
			    temperature: Sampling temperature supplied to compatible model requests.
			    format: Output or response format selected for the operation.
			    top_p: Nucleus sampling value supplied to compatible model requests.
			    frequency: Frequency penalty supplied to compatible model requests.
			    max_tools: Maximum number of tool calls allowed for the request.
			    presence: Presence penalty supplied to compatible model requests.
			    max_tokens: Maximum output-token value supplied to compatible requests.
			    store: Response storage flag supplied to compatible provider requests.
			    stream: Streaming flag retained by the UI and compatible provider requests.
			    instruct: System or developer instructions supplied to the provider.
			    background: Background transparency or execution option supplied by the caller.
			    reasoning: Reasoning-effort value or reasoning configuration supplied by the UI.
			    include: Requested provider include fields.
			    tools: Tool selections or provider tool dictionaries supplied by the UI.
			    allowed_domains: Allowed web-search domains supplied to tool configuration.
			    previous_id: Previous Responses API identifier used for stateful continuation.
			    tool_choice: Tool-choice policy selected for the request.
			    is_parallel: Flag controlling parallel tool-call support when tools are active.
			    context: Prior conversation context supplied to request builders.
			    input_data: Prebuilt provider input payload supplied by the caller.
			    vector_store_ids: Vector-store identifiers used by file-search tools.
			    conversation_id: Conversation identifier used for stateful continuation.
			    prompt_id: OpenAI prompt-template identifier.
			    prompt_version: OpenAI prompt-template version.
			    prompt_variables: Variables supplied to an OpenAI prompt template.
		
			Returns:
			    Dict[ str, Any ]: Normalized provider request dictionary.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str | None:
		"""Get output text.
		
		
			Purpose:
			    Gets output text from the current instance state or latest provider response.
		
			Returns:
			    str | None: Extracted response text or None when no text output is available.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def get_usage( self ) -> Any:
		"""Get usage.
		
		
			Purpose:
			    Gets usage from the current instance state or latest provider response.
		
			Returns:
			    Any: Usage metadata from the latest provider response.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
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
		"""Completion.
		
		
			Purpose:
			    Creates a Responses API completion call from prompt-template, tool, context, and model configuration state.
		
			Args:
			    prompt_id: OpenAI prompt-template identifier.
			    prompt_version: OpenAI prompt-template version.
			    model: Provider model identifier selected for the operation.
			    user_input: User text submitted through the prompt-template workflow.
			    temperature: Sampling temperature supplied to compatible model requests.
			    format: Output or response format selected for the operation.
			    top_p: Nucleus sampling value supplied to compatible model requests.
			    frequency: Frequency penalty supplied to compatible model requests.
			    presence: Presence penalty supplied to compatible model requests.
			    max_tokens: Maximum output-token value supplied to compatible requests.
			    store: Response storage flag supplied to compatible provider requests.
			    stream: Streaming flag retained by the UI and compatible provider requests.
			    instruct: System or developer instructions supplied to the provider.
			    background: Background transparency or execution option supplied by the caller.
			    reasoning: Reasoning-effort value or reasoning configuration supplied by the UI.
			    include: Requested provider include fields.
			    tools: Tool selections or provider tool dictionaries supplied by the UI.
			    tool_choice: Tool-choice policy selected for the request.
			    is_parallel: Flag controlling parallel tool-call support when tools are active.
			    previous_id: Previous Responses API identifier used for stateful continuation.
			    context: Prior conversation context supplied to request builders.
			    input_data: Prebuilt provider input payload supplied by the caller.
			    allowed_domains: Allowed web-search domains supplied to tool configuration.
			    vector_store_ids: Vector-store identifiers used by file-search tools.
			    conversation_id: Conversation identifier used for stateful continuation.
			    max_tools: Maximum number of tool calls allowed for the request.
			    prompt_variables: Variables supplied to an OpenAI prompt template.
		
			Returns:
			    Response | Any: Full Responses API response object returned by OpenAI.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None, background: bool = False,
			reasoning: str = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None, previous_id: str = None, tool_choice: str = None,
			is_parallel: bool = None, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None, vector_store_ids: List[ str ] = None,
			conversation_id: str = None ) -> str | None:
		"""Generate text.
		
		
			Purpose:
			    Generates text through the Responses API after validating prompt and model settings and building the provider request payload.
		
			Args:
			    prompt: User prompt or task instruction submitted to the provider.
			    model: Provider model identifier selected for the operation.
			    temperature: Sampling temperature supplied to compatible model requests.
			    format: Output or response format selected for the operation.
			    top_p: Nucleus sampling value supplied to compatible model requests.
			    frequency: Frequency penalty supplied to compatible model requests.
			    max_tools: Maximum number of tool calls allowed for the request.
			    presence: Presence penalty supplied to compatible model requests.
			    max_tokens: Maximum output-token value supplied to compatible requests.
			    store: Response storage flag supplied to compatible provider requests.
			    stream: Streaming flag retained by the UI and compatible provider requests.
			    instruct: System or developer instructions supplied to the provider.
			    background: Background transparency or execution option supplied by the caller.
			    reasoning: Reasoning-effort value or reasoning configuration supplied by the UI.
			    include: Requested provider include fields.
			    tools: Tool selections or provider tool dictionaries supplied by the UI.
			    allowed_domains: Allowed web-search domains supplied to tool configuration.
			    previous_id: Previous Responses API identifier used for stateful continuation.
			    tool_choice: Tool-choice policy selected for the request.
			    is_parallel: Flag controlling parallel tool-call support when tools are active.
			    context: Prior conversation context supplied to request builders.
			    input_data: Prebuilt provider input payload supplied by the caller.
			    vector_store_ids: Vector-store identifiers used by file-search tools.
			    conversation_id: Conversation identifier used for stateful continuation.
		
			Returns:
			    str | None: Generated text output or None when no output text is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		
			Purpose:
			    Returns the public Chat members displayed by interactive inspection and documentation tooling.
		
			Returns:
			    List[ str ] | None: Public member names exposed for interactive inspection.
		"""
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
	"""Images provider wrapper.
	
	
		Purpose:
		    Builds and executes OpenAI image-generation, image-analysis, and image-editing workflows while normalizing model-specific image options.
	
		Attributes:
		    quality: Image quality option selected for generation or editing.
		    detail: Image detail option selected for analysis.
		    size: Image size option selected for generation or editing.
		    previous_id: Previous Responses API identifier used for stateful continuation.
		    include: Responses API include fields requested by the current workflow.
		    tool_choice: Tool-choice policy selected for the current request.
		    parallel_tools: Flag controlling parallel tool-call support when tools are active.
		    input: Responses API input payload built for the current request.
		    instructions: System or developer instructions supplied to model requests.
		    max_tools: Maximum tool-call count sent with supported Responses API requests.
		    tools: Tool definitions selected or built for the current request.
		    messages: Message list built for Responses API input payloads.
		    reasoning: Reasoning-effort configuration used by supported models.
		    image_url: Remote image URL used by image-analysis workflows.
		    image_path: Local image path used by image-analysis or image-editing workflows.
		    file_url: File URL retained for compatibility with document workflows.
		    file_path: Local file path retained for file-enabled workflows.
		    style: Image style option retained for compatible image models.
		    allowed_domains: Allowed-domain filters used by web-search tool configuration.
		    response_format: Response-format configuration used by text or media requests.
		    mime_format: Mime format value retained by the Images workflow.
		    background: Background-execution flag retained by UI workflows and request builders.
		    backcolor: Background color option retained by image-generation workflows.
		    compression: Image compression option used when a provider supports it.
	
		Notes:
		    The wrapper stores request state on the instance so Streamlit callbacks, provider calls, and documentation-generated API pages expose consistent runtime behavior.
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
		"""Initialize Images.
		
		
			Purpose:
			    Initializes Images state by assigning configuration values, request defaults, cached outputs, and compatibility fields used by later methods.
		
			Args:
			    prompt: User prompt or task instruction submitted to the provider.
			    model: Provider model identifier selected for the operation.
			    temperature: Sampling temperature supplied to compatible model requests.
			    top_p: Nucleus sampling value supplied to compatible model requests.
			    presence: Presence penalty supplied to compatible model requests.
			    frequency: Frequency penalty supplied to compatible model requests.
			    max_tokens: Maximum output-token value supplied to compatible requests.
			    store: Response storage flag supplied to compatible provider requests.
			    stream: Streaming flag retained by the UI and compatible provider requests.
			    backcolor: Backcolor supplied to the init workflow.
			    instruct: System or developer instructions supplied to the provider.
			    background: Background transparency or execution option supplied by the caller.
			    number: Requested output count before provider-specific normalization.
			    response_format: Response-format configuration retained for compatible workflows.
			    path: Local file path supplied to image, audio, or vector-store workflows.
			    image_url: Remote image URL supplied to image-analysis workflows.
			    size: Image size option selected for generation or editing.
			    quality: Image quality option selected for generation or editing.
			    detail: Image detail option selected for image analysis.
			    style: Image style option retained for compatible image-generation models.
			    compression: Compression setting supplied to supported image-output workflows.
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
		"""Model options.
		
		
			Purpose:
			    Returns model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Model option names exposed to the UI selector.
		"""
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
		"""Analysis model options.
		
		
			Purpose:
			    Returns analysis model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Image-analysis model option names exposed to the UI selector.
		"""
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
		"""Edit model options.
		
		
			Purpose:
			    Returns edit model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Image-editing model option names exposed to the UI selector.
		"""
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
		"""Style options.
		
		
			Purpose:
			    Returns style options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Image style option names exposed to the UI selector.
		"""
		return [
				'vivid',
				'natural',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Format options.
		
		
			Purpose:
			    Returns format options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Response-format option names exposed to the UI selector.
		"""
		return [
				'png',
				'jpeg',
				'webp',
		]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Mime options.
		
		
			Purpose:
			    Returns mime options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: MIME type option names exposed to the UI selector.
		"""
		return [
				'png',
				'jpeg',
				'webp',
		]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Size options.
		
		
			Purpose:
			    Returns size options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Image size option names exposed to the UI selector.
		"""
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
		"""Choice options.
		
		
			Purpose:
			    Returns choice options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Tool-choice option names exposed to the UI selector.
		"""
		return [
				'auto',
				'required',
				'none',
		]
	
	@property
	def backcolor_options( self ) -> List[ str ]:
		"""Backcolor options.
		
		
			Purpose:
			    Returns backcolor options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Image background option names exposed to the UI selector.
		"""
		return [
				'auto',
				'transparent',
				'opaque',
		]
	
	@property
	def quality_options( self ) -> List[ str ]:
		"""Quality options.
		
		
			Purpose:
			    Returns quality options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Image quality option names exposed to the UI selector.
		"""
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
		"""Detail options.
		
		
			Purpose:
			    Returns detail options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ]: Image detail option names exposed to the UI selector.
		"""
		return [
				'auto',
				'low',
				'high',
				'original',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		
			Purpose:
			    Returns reasoning options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Reasoning-effort option names exposed to the UI selector.
		"""
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
		"""Modality options.
		
		
			Purpose:
			    Returns modality options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Modality option names exposed to the UI selector.
		"""
		return [
				'text',
				'auto',
				'image',
				'audio',
		]
	
	def is_gpt_image_model( self, model: str ) -> bool:
		"""Is gpt image model.
		
		
			Purpose:
			    Executes the is gpt image model workflow for the Images provider wrapper while preserving normalized instance state for downstream use.
		
			Args:
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    bool: True when the selected model uses the GPT image-generation endpoint.
		"""
		return isinstance( model, str ) and model in [
				'gpt-image-2',
				'gpt-image-1.5',
				'gpt-image-1',
				'gpt-image-1-mini',
				'chatgpt-image-latest',
		]
	
	def normalize_count( self, number: int = None ) -> int:
		"""Normalize count.
		
		
			Purpose:
			    Normalizes count data from provider-specific objects into application-ready Python values.
		
			Args:
			    number: Requested output count before provider-specific normalization.
		
			Returns:
			    int: Validated image count accepted by the selected endpoint.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_output_format( self, fmt: str = None, background: str = None ) -> str:
		"""Normalize output format.
		
		
			Purpose:
			    Normalizes output format data from provider-specific objects into application-ready Python values.
		
			Args:
			    fmt: Image output format before provider-specific normalization.
			    background: Background transparency or execution option supplied by the caller.
		
			Returns:
			    str: Image output format accepted by the selected endpoint.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_background( self, background: str = None, model: str = None ) -> str | None:
		"""Normalize background.
		
		
			Purpose:
			    Normalizes background data from provider-specific objects into application-ready Python values.
		
			Args:
			    background: Background transparency or execution option supplied by the caller.
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    str | None: Background option accepted by the selected image endpoint.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_size( self, size: str = None, model: str = None ) -> str:
		"""Normalize size.
		
		
			Purpose:
			    Normalizes size data from provider-specific objects into application-ready Python values.
		
			Args:
			    size: Image size option selected for generation or editing.
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    str: Image size accepted by the selected image endpoint.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_quality( self, quality: str = None, model: str = None ) -> str | None:
		"""Normalize quality.
		
		
			Purpose:
			    Normalizes quality data from provider-specific objects into application-ready Python values.
		
			Args:
			    quality: Image quality option selected for generation or editing.
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    str | None: Image quality accepted by the selected image endpoint.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_compression( self, compression: float = None ) -> int | None:
		"""Normalize compression.
		
		
			Purpose:
			    Normalizes compression data from provider-specific objects into application-ready Python values.
		
			Args:
			    compression: Compression setting supplied to supported image-output workflows.
		
			Returns:
			    int | None: Compression value accepted by the selected image endpoint.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_image_outputs( self ) -> str | bytes | list[ str | bytes ] | None:
		"""Normalize image outputs.
		
		
			Purpose:
			    Normalizes image outputs data from provider-specific objects into application-ready Python values.
		
			Returns:
			    str | bytes | list[ str | bytes ] | None: Image output payload normalized as URLs or binary image bytes.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, number: int = 1, model: str = 'gpt-image-1-mini',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = 'jpeg',
			compression: float = None, background: str = None ) -> str | bytes | list[
		str | bytes ] | None:
		"""Generate.
		
		
			Purpose:
			    Generates images through the OpenAI image API after normalizing image count, size, quality, format, background, and compression settings.
		
			Args:
			    prompt: User prompt or task instruction submitted to the provider.
			    number: Requested output count before provider-specific normalization.
			    model: Provider model identifier selected for the operation.
			    size: Image size option selected for generation or editing.
			    quality: Image quality option selected for generation or editing.
			    fmt: Image output format before provider-specific normalization.
			    compression: Compression setting supplied to supported image-output workflows.
			    background: Background transparency or execution option supplied by the caller.
		
			Returns:
			    str | bytes | list[ str | bytes ] | None: Generated image output normalized as URLs or binary image bytes.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, text: str, path: str = None, image_url: str = None,
			instruct: str = None, model: str = 'gpt-4.1-mini', max_tokens: int = None,
			temperature: float = None, include: List[ str ] = None, store: bool = None,
			stream: bool = False, detail: str = 'auto' ) -> str | None:
		"""Analyze.
		
		
			Purpose:
			    Analyzes an image with a vision-capable model and returns text extracted from the model response.
		
			Args:
			    text: Input text supplied to the operation.
			    path: Local file path supplied to image, audio, or vector-store workflows.
			    image_url: Remote image URL supplied to image-analysis workflows.
			    instruct: System or developer instructions supplied to the provider.
			    model: Provider model identifier selected for the operation.
			    max_tokens: Maximum output-token value supplied to compatible requests.
			    temperature: Sampling temperature supplied to compatible model requests.
			    include: Requested provider include fields.
			    store: Response storage flag supplied to compatible provider requests.
			    stream: Streaming flag retained by the UI and compatible provider requests.
			    detail: Image detail option selected for image analysis.
		
			Returns:
			    str | None: Image-analysis text output or None when no output text is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gpt-image-1',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = 'jpeg',
			compression: float = None, background: str = None,
			number: int = 1 ) -> str | bytes | list[ str | bytes ] | None:
		"""Edit.
		
		
			Purpose:
			    Edits an input image with the OpenAI image API and returns normalized URL or binary image output.
		
			Args:
			    prompt: User prompt or task instruction submitted to the provider.
			    path: Local file path supplied to image, audio, or vector-store workflows.
			    model: Provider model identifier selected for the operation.
			    size: Image size option selected for generation or editing.
			    quality: Image quality option selected for generation or editing.
			    fmt: Image output format before provider-specific normalization.
			    compression: Compression setting supplied to supported image-output workflows.
			    background: Background transparency or execution option supplied by the caller.
			    number: Requested output count before provider-specific normalization.
		
			Returns:
			    str | bytes | list[ str | bytes ] | None: Edited image output normalized as URLs or binary image bytes.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
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
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		
			Purpose:
			    Returns the public Images members displayed by interactive inspection and documentation tooling.
		
			Returns:
			    List[ str ] | None: Public member names exposed for interactive inspection.
		"""
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
	"""TTS provider wrapper.
	
	
		Purpose:
		    Builds and executes OpenAI text-to-speech requests and validates voice, speed, model, and output format selections.
	
		Attributes:
		    api_key: OpenAI API key loaded from project configuration.
		    client: OpenAI client created for the current provider operation.
		    speed: Speech speed value selected for text-to-speech output.
		    voice: Voice option selected for text-to-speech output.
		    input: Responses API input payload built for the current request.
		    instructions: System or developer instructions supplied to model requests.
		    response: Latest provider response object returned by an API call.
		    response_format: Response-format configuration used by text or media requests.
		    file_path: Local file path retained for file-enabled workflows.
		    model: Provider model identifier used by the current workflow.
		    audio_bytes: Audio bytes value retained by the TTS workflow.
		    request: Normalized request dictionary prepared for provider execution.
	
		Notes:
		    The wrapper stores request state on the instance so Streamlit callbacks, provider calls, and documentation-generated API pages expose consistent runtime behavior.
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
	
	def __init__( self, input: str = None, model: str = 'gpt-4o-mini-tts', format: str = None,
			instruct: str = None, voice: str = None, speed: float = None, file_path: str = None ):
		"""Initialize TTS.
		
		
			Purpose:
			    Initializes TTS state by assigning configuration values, request defaults, cached outputs, and compatibility fields used by later methods.
		
			Args:
			    input: Prebuilt provider input payload supplied by the caller.
			    model: Provider model identifier selected for the operation.
			    format: Output or response format selected for the operation.
			    instruct: System or developer instructions supplied to the provider.
			    voice: Text-to-speech voice selected for audio generation.
			    speed: Text-to-speech speed value selected by the caller.
			    file_path: Output or input file path used by the workflow.
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
		"""Model options.
		
		
			Purpose:
			    Returns model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Model option names exposed to the UI selector.
		"""
		return [
				'gpt-4o-mini-tts',
				'gpt-4o-mini-tts-2025-12-15',
				'tts-1',
				'tts-1-hd',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		
			Purpose:
			    Returns mime options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: MIME type option names exposed to the UI selector.
		"""
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
		"""Voice options.
		
		
			Purpose:
			    Returns voice options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Voice option names exposed to the UI selector.
		"""
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
		"""Speed options.
		
		
			Purpose:
			    Returns speed options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ float ] | None: Speech-speed options exposed to the UI selector.
		"""
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
	
	def validate_model( self, model: str = None ) -> str:
		"""Validate model.
		
		
			Purpose:
			    Validates model input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    str: Validated provider model identifier.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_format( self, format: str = None ) -> str:
		"""Validate format.
		
		
			Purpose:
			    Validates format input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    format: Output or response format selected for the operation.
		
			Returns:
			    str: Validated output format for the selected provider workflow.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_voice( self, voice: str = None ) -> str:
		"""Validate voice.
		
		
			Purpose:
			    Validates voice input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    voice: Text-to-speech voice selected for audio generation.
		
			Returns:
			    str: Validated text-to-speech voice name.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_speed( self, speed: float = None ) -> float:
		"""Validate speed.
		
		
			Purpose:
			    Validates speed input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    speed: Text-to-speech speed value selected by the caller.
		
			Returns:
			    float: Validated text-to-speech speed value.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def create_speech( self, text: str, model: str = 'gpt-4o-mini-tts', format: str = 'mp3',
			speed: float = 1.0, voice: str = 'alloy', instruct: str = None,
			file_path: str = None ) -> bytes | None:
		"""Create speech.
		
		
			Purpose:
			    Generates speech audio from text after validating text-to-speech model, output format, voice, and speed settings.
		
			Args:
			    text: Input text supplied to the operation.
			    model: Provider model identifier selected for the operation.
			    format: Output or response format selected for the operation.
			    speed: Text-to-speech speed value selected by the caller.
			    voice: Text-to-speech voice selected for audio generation.
			    instruct: System or developer instructions supplied to the provider.
			    file_path: Output or input file path used by the workflow.
		
			Returns:
			    bytes | None: Generated speech bytes or None when the provider returns no audio body.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
					self.request[ 'instructions' ] = self.instructions
				
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
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		
			Purpose:
			    Returns the public TTS members displayed by interactive inspection and documentation tooling.
		
			Returns:
			    List[ str ] | None: Public member names exposed for interactive inspection.
		"""
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
	"""Transcription provider wrapper.
	
	
		Purpose:
		    Builds and executes OpenAI audio transcription requests and normalizes transcription response payloads for the application UI.
	
		Attributes:
		    client: OpenAI client created for the current provider operation.
		    language: Language option selected for transcription or translation workflows.
		    instructions: System or developer instructions supplied to model requests.
		    include: Responses API include fields requested by the current workflow.
		    normalized_result: Normalized result value retained by the Transcription workflow.
	
		Notes:
		    The wrapper stores request state on the instance so Streamlit callbacks, provider calls, and documentation-generated API pages expose consistent runtime behavior.
	"""
	client: Optional[ OpenAI ]
	language: Optional[ str ]
	instructions: Optional[ str ]
	include: Optional[ List[ str ] ]
	normalized_result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'gpt-4o-transcribe', temperature: float = None,
			prompt: str = None, number: int = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, stream: bool = None, store: bool = None,
			language: str = None, instruct: str = None, format: str = None, background: bool = None,
			messages: List[ Dict[ str, str ] ] = None, stops: List[ str ] = None,
			include: List[ str ] = None ):
		"""Initialize Transcription.
		
		
			Purpose:
			    Initializes Transcription state by assigning configuration values, request defaults, cached outputs, and compatibility fields used by later methods.
		
			Args:
			    model: Provider model identifier selected for the operation.
			    temperature: Sampling temperature supplied to compatible model requests.
			    prompt: User prompt or task instruction submitted to the provider.
			    number: Requested output count before provider-specific normalization.
			    top_p: Nucleus sampling value supplied to compatible model requests.
			    frequency: Frequency penalty supplied to compatible model requests.
			    presence: Presence penalty supplied to compatible model requests.
			    max_tokens: Maximum output-token value supplied to compatible requests.
			    stream: Streaming flag retained by the UI and compatible provider requests.
			    store: Response storage flag supplied to compatible provider requests.
			    language: Language code selected for transcription or translation.
			    instruct: System or developer instructions supplied to the provider.
			    format: Output or response format selected for the operation.
			    background: Background transparency or execution option supplied by the caller.
			    messages: Message history retained for audio or chat workflows.
			    stops: Stop sequences retained for compatible provider requests.
			    include: Requested provider include fields.
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
		"""Model options.
		
		
			Purpose:
			    Returns model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Model option names exposed to the UI selector.
		"""
		return [
				'gpt-4o-transcribe',
				'gpt-4o-mini-transcribe',
				'gpt-4o-mini-transcribe-2025-12-15',
				'whisper-1',
				'gpt-4o-transcribe-diarize',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		
			Purpose:
			    Returns mime options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: MIME type option names exposed to the UI selector.
		"""
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
		"""Language options.
		
		
			Purpose:
			    Returns language options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Language option names exposed to the UI selector.
		"""
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
		"""Language labels.
		
		
			Purpose:
			    Returns language labels used by the Streamlit selectors and provider request builders.
		
			Returns:
			    Dict[ str, str ] | None: Mapping of language codes to display labels.
		"""
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
		"""Include options.
		
		
			Purpose:
			    Returns include options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Include option names exposed to the UI selector.
		"""
		return [
				'logprobs',
		]
	
	@property
	def response_format_options( self ) -> Dict[ str, List[ str ] ]:
		"""Response format options.
		
		
			Purpose:
			    Returns response format options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    Dict[ str, List[ str ] ]: Response-format options exposed for the selected audio workflow.
		"""
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
	
	def validate_model( self, model: str = None ) -> str:
		"""Validate model.
		
		
			Purpose:
			    Validates model input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    str: Validated provider model identifier.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_format( self, model: str, format: str = None ) -> str | None:
		"""Validate format.
		
		
			Purpose:
			    Validates format input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    model: Provider model identifier selected for the operation.
			    format: Output or response format selected for the operation.
		
			Returns:
			    str | None: Validated output format for the selected provider workflow.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_include( self, model: str, include: List[ str ] = None ) -> List[ str ]:
		"""Validate include.
		
		
			Purpose:
			    Validates include input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    model: Provider model identifier selected for the operation.
			    include: Requested provider include fields.
		
			Returns:
			    List[ str ]: Filtered include fields valid for the selected transcription model.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_response( self, response: Any ) -> Dict[ str, Any ]:
		"""Normalize response.
		
		
			Purpose:
			    Normalizes response data from provider-specific objects into application-ready Python values.
		
			Args:
			    response: Provider response object to normalize or inspect.
		
			Returns:
			    Dict[ str, Any ]: Normalized response metadata and output fields.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			result: Dict[ str, Any ] = {
					'text': '',
					'segments': [ ],
					'language': None,
					'duration': None,
					'raw': None,
			}
			
			if response is None:
				return result
			
			if isinstance( response, str ):
				result[ 'text' ] = response
				result[ 'raw' ] = response
				return result
			
			if hasattr( response, 'model_dump' ):
				try:
					result[ 'raw' ] = response.model_dump( )
				except Exception:
					result[ 'raw' ] = str( response )
			else:
				result[ 'raw' ] = str( response )
			
			text = getattr( response, 'text', None )
			if isinstance( text, str ):
				result[ 'text' ] = text
			
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
				
				result[ 'segments' ] = normalized_segments
			
			language = getattr( response, 'language', None )
			if language:
				result[ 'language' ] = language
			
			duration = getattr( response, 'duration', None )
			if duration:
				result[ 'duration' ] = duration
			
			if not result[ 'text' ] and len( result[ 'segments' ] ) > 0:
				parts = [ ]
				for segment in result[ 'segments' ]:
					if isinstance( segment, dict ) and segment.get( 'text' ):
						parts.append( str( segment.get( 'text' ) ) )
				
				result[ 'text' ] = '\n'.join( parts ).strip( )
			
			if not result[ 'text' ]:
				result[ 'text' ] = str( response )
			
			return result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'normalize_response( self, response: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, path: str, model: str = 'gpt-4o-transcribe', language: str = None,
			prompt: str = None, format: str = None, temperature: float = None,
			include: List[ str ] = None ) -> str | None:
		"""Transcribe.
		
		
			Purpose:
			    Transcribes an audio file through OpenAI after validating model, language, output format, and include settings.
		
			Args:
			    path: Local file path supplied to image, audio, or vector-store workflows.
			    model: Provider model identifier selected for the operation.
			    language: Language code selected for transcription or translation.
			    prompt: User prompt or task instruction submitted to the provider.
			    format: Output or response format selected for the operation.
			    temperature: Sampling temperature supplied to compatible model requests.
			    include: Requested provider include fields.
		
			Returns:
			    str | None: Transcribed text output or None when transcription returns no text.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
				self.request[ 'language' ] = self.language
			
			if self.prompt:
				self.request[ 'prompt' ] = self.prompt
			
			if self.response_format:
				self.request[ 'response_format' ] = self.response_format
			
			if self.include:
				self.request[ 'include' ] = self.include
			
			if self.temperature is not None:
				if self.model == 'whisper-1':
					self.request[ 'temperature' ] = self.temperature
			
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
			Logger( ).write( ex )
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		
			Purpose:
			    Returns the public Transcription members displayed by interactive inspection and documentation tooling.
		
			Returns:
			    List[ str ] | None: Public member names exposed for interactive inspection.
		"""
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
	"""Translation provider wrapper.
	
	
		Purpose:
		    Builds and executes OpenAI audio translation requests and normalizes translation response payloads for the application UI.
	
		Attributes:
		    client: OpenAI client created for the current provider operation.
		    target_language: Target language value retained by the Translation workflow.
		    response_format: Response-format configuration used by text or media requests.
		    normalized_result: Normalized result value retained by the Translation workflow.
	
		Notes:
		    The wrapper stores request state on the instance so Streamlit callbacks, provider calls, and documentation-generated API pages expose consistent runtime behavior.
	"""
	client: Optional[ OpenAI ]
	target_language: Optional[ str ]
	response_format: Optional[ str ]
	normalized_result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'whisper-1', temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			store: bool = None,
			stream: bool = None, instruct: str = None, audio_file: str = None, format: str = None,
			language: str = None ):
		"""Initialize Translation.
		
		
			Purpose:
			    Initializes Translation state by assigning configuration values, request defaults, cached outputs, and compatibility fields used by later methods.
		
			Args:
			    model: Provider model identifier selected for the operation.
			    temperature: Sampling temperature supplied to compatible model requests.
			    top_p: Nucleus sampling value supplied to compatible model requests.
			    frequency: Frequency penalty supplied to compatible model requests.
			    presence: Presence penalty supplied to compatible model requests.
			    max_tokens: Maximum output-token value supplied to compatible requests.
			    store: Response storage flag supplied to compatible provider requests.
			    stream: Streaming flag retained by the UI and compatible provider requests.
			    instruct: System or developer instructions supplied to the provider.
			    audio_file: Audio file supplied to the init workflow.
			    format: Output or response format selected for the operation.
			    language: Language code selected for transcription or translation.
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
		"""Model options.
		
		
			Purpose:
			    Returns model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Model option names exposed to the UI selector.
		"""
		return [
				'whisper-1',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Mime options.
		
		
			Purpose:
			    Returns mime options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: MIME type option names exposed to the UI selector.
		"""
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
		"""Language options.
		
		
			Purpose:
			    Returns language options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Language option names exposed to the UI selector.
		"""
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
		"""Language labels.
		
		
			Purpose:
			    Returns language labels used by the Streamlit selectors and provider request builders.
		
			Returns:
			    Dict[ str, str ] | None: Mapping of language codes to display labels.
		"""
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
		"""Response format options.
		
		
			Purpose:
			    Returns response format options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Response-format options exposed for the selected audio workflow.
		"""
		return [
				'json',
				'text',
				'srt',
				'verbose_json',
				'vtt',
		]
	
	def validate_model( self, model: str = None ) -> str:
		"""Validate model.
		
		
			Purpose:
			    Validates model input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    str: Validated provider model identifier.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_format( self, format: str = None ) -> str | None:
		"""Validate format.
		
		
			Purpose:
			    Validates format input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    format: Output or response format selected for the operation.
		
			Returns:
			    str | None: Validated output format for the selected provider workflow.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_response( self, response: Any ) -> Dict[ str, Any ]:
		"""Normalize response.
		
		
			Purpose:
			    Normalizes response data from provider-specific objects into application-ready Python values.
		
			Args:
			    response: Provider response object to normalize or inspect.
		
			Returns:
			    Dict[ str, Any ]: Normalized response metadata and output fields.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			result: Dict[ str, Any ] = {
					'text': '',
					'segments': [ ],
					'language': None,
					'duration': None,
					'raw': None,
			}
			
			if response is None:
				return result
			
			if isinstance( response, str ):
				result[ 'text' ] = response
				result[ 'raw' ] = response
				return result
			
			if hasattr( response, 'model_dump' ):
				try:
					result[ 'raw' ] = response.model_dump( )
				except Exception:
					result[ 'raw' ] = str( response )
			else:
				result[ 'raw' ] = str( response )
			
			text = getattr( response, 'text', None )
			if isinstance( text, str ):
				result[ 'text' ] = text
			
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
				
				result[ 'segments' ] = normalized_segments
			
			language = getattr( response, 'language', None )
			if language:
				result[ 'language' ] = language
			
			duration = getattr( response, 'duration', None )
			if duration:
				result[ 'duration' ] = duration
			
			if not result[ 'text' ] and len( result[ 'segments' ] ) > 0:
				parts = [ ]
				for segment in result[ 'segments' ]:
					if isinstance( segment, dict ) and segment.get( 'text' ):
						parts.append( str( segment.get( 'text' ) ) )
				
				result[ 'text' ] = '\n'.join( parts ).strip( )
			
			if not result[ 'text' ]:
				result[ 'text' ] = str( response )
			
			return result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Translation'
			exception.method = 'normalize_response( self, response: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def translate( self, filepath: str, model: str = 'whisper-1', prompt: str = None,
			format: str = None, temperature: float = None, language: str = None ) -> str | None:
		"""Translate.
		
		
			Purpose:
			    Translates an audio file through OpenAI after validating model, language, output format, and temperature settings.
		
			Args:
			    filepath: Local file path supplied to upload, transcription, translation, or vector-store workflows.
			    model: Provider model identifier selected for the operation.
			    prompt: User prompt or task instruction submitted to the provider.
			    format: Output or response format selected for the operation.
			    temperature: Sampling temperature supplied to compatible model requests.
			    language: Language code selected for transcription or translation.
		
			Returns:
			    str | None: Translated text output or None when translation returns no text.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
				self.request[ 'prompt' ] = self.prompt
			
			if self.response_format:
				self.request[ 'response_format' ] = self.response_format
			
			if self.temperature is not None:
				self.request[ 'temperature' ] = self.temperature
			
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
			Logger( ).write( ex )
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		
			Purpose:
			    Returns the public Translation members displayed by interactive inspection and documentation tooling.
		
			Returns:
			    List[ str ] | None: Public member names exposed for interactive inspection.
		"""
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
	"""Embeddings provider wrapper.
	
	
		Purpose:
		    Builds and executes OpenAI embedding requests, validates dimensional settings, and enforces model token limits for semantic-search workflows.
	
		Attributes:
		    api_key: OpenAI API key loaded from project configuration.
		    client: OpenAI client created for the current provider operation.
		    model: Provider model identifier used by the current workflow.
		    input: Responses API input payload built for the current request.
		    encoding_format: Encoding format value retained by the Embeddings workflow.
		    dimensions: Embedding dimensionality requested for supported models.
		    user: Optional user identifier sent with supported embedding requests.
		    response: Latest provider response object returned by an API call.
		    embedding: Embedding value retained by the Embeddings workflow.
		    embeddings: Embeddings value retained by the Embeddings workflow.
		    usage: Usage value retained by the Embeddings workflow.
		    request: Normalized request dictionary prepared for provider execution.
	
		Notes:
		    The wrapper stores request state on the instance so Streamlit callbacks, provider calls, and documentation-generated API pages expose consistent runtime behavior.
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
	
	def __init__( self, text: str | List[ str ] = None, model: str = 'text-embedding-3-small',
			format: str = 'float', dimensions: int = None, user: str = None ):
		"""Initialize Embeddings.
		
		
			Purpose:
			    Initializes Embeddings state by assigning configuration values, request defaults, cached outputs, and compatibility fields used by later methods.
		
			Args:
			    text: Input text supplied to the operation.
			    model: Provider model identifier selected for the operation.
			    format: Output or response format selected for the operation.
			    dimensions: Embedding dimensionality requested for supported models.
			    user: Optional user identifier supplied to embedding requests.
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
		"""Model options.
		
		
			Purpose:
			    Returns model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Model option names exposed to the UI selector.
		"""
		return [
				'text-embedding-3-small',
				'text-embedding-3-large',
				'text-embedding-ada-002',
		]
	
	@property
	def encoding_options( self ) -> List[ str ] | None:
		"""Encoding options.
		
		
			Purpose:
			    Returns encoding options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Embedding encoding format options exposed to the UI selector.
		"""
		return [
				'float',
				'base64',
		]
	
	@property
	def model_default_dimensions( self ) -> Dict[ str, int ]:
		"""Model default dimensions.
		
		
			Purpose:
			    Returns model default dimensions used by the Streamlit selectors and provider request builders.
		
			Returns:
			    Dict[ str, int ]: Default embedding dimensions keyed by model name.
		"""
		return {
				'text-embedding-3-small': 1536,
				'text-embedding-3-large': 3072,
				'text-embedding-ada-002': 1536,
		}
	
	@property
	def model_max_dimensions( self ) -> Dict[ str, int ]:
		"""Model max dimensions.
		
		
			Purpose:
			    Returns model max dimensions used by the Streamlit selectors and provider request builders.
		
			Returns:
			    Dict[ str, int ]: Maximum embedding dimensions keyed by model name.
		"""
		return {
				'text-embedding-3-small': 1536,
				'text-embedding-3-large': 3072,
				'text-embedding-ada-002': 1536,
		}
	
	@property
	def model_dimension_support( self ) -> Dict[ str, bool ]:
		"""Model dimension support.
		
		
			Purpose:
			    Returns model dimension support used by the Streamlit selectors and provider request builders.
		
			Returns:
			    Dict[ str, bool ]: Flags indicating whether models support custom dimensions.
		"""
		return {
				'text-embedding-3-small': True,
				'text-embedding-3-large': True,
				'text-embedding-ada-002': False,
		}
	
	def validate_model( self, model: str = None ) -> str:
		"""Validate model.
		
		
			Purpose:
			    Validates model input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    str: Validated provider model identifier.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_encoding_format( self, format: str = None ) -> str:
		"""Validate encoding format.
		
		
			Purpose:
			    Validates encoding format input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    format: Output or response format selected for the operation.
		
			Returns:
			    str: Validated embedding encoding format.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_dimensions( self, model: str, dimensions: int = None ) -> int | None:
		"""Validate dimensions.
		
		
			Purpose:
			    Validates dimensions input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    model: Provider model identifier selected for the operation.
			    dimensions: Embedding dimensionality requested for supported models.
		
			Returns:
			    int | None: Validated embedding dimension count or None when dimensions are not used.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_input( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""Validate input.
		
		
			Purpose:
			    Validates input input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    text: Input text supplied to the operation.
		
			Returns:
			    str | List[ str ]: Validated embedding input text or text collection.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def get_default_dimensions( self, model: str ) -> int:
		"""Get default dimensions.
		
		
			Purpose:
			    Gets default dimensions from the current instance state or latest provider response.
		
			Args:
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    int: Default embedding dimension count for the selected model.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			return int( self.model_default_dimensions.get( model, 1536 ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'get_default_dimensions( self, model: str ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def get_max_dimensions( self, model: str ) -> int:
		"""Get max dimensions.
		
		
			Purpose:
			    Gets max dimensions from the current instance state or latest provider response.
		
			Args:
			    model: Provider model identifier selected for the operation.
		
			Returns:
			    int: Maximum embedding dimension count for the selected model.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			return int( self.model_max_dimensions.get( model, 1536 ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'get_max_dimensions( self, model: str ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def count_tokens( self, text: str, encoding_name: str = 'cl100k_base' ) -> int:
		"""Count tokens.
		
		
			Purpose:
			    Counts tokens for one text value using the configured tokenizer encoding.
		
			Args:
			    text: Input text supplied to the operation.
			    encoding_name: Tokenizer encoding name used for token counting.
		
			Returns:
			    int: Token count for a single text value.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def count_total_tokens( self, text: str | List[ str ],
			encoding_name: str = 'cl100k_base' ) -> int:
		"""Count total tokens.
		
		
			Purpose:
			    Counts tokens across one text value or a collection of text values for embedding limit enforcement.
		
			Args:
			    text: Input text supplied to the operation.
			    encoding_name: Tokenizer encoding name used for token counting.
		
			Returns:
			    int: Combined token count for a text value or text collection.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_token_limits( self, text: str | List[ str ],
			max_input_tokens: int = 8192, max_total_tokens: int = 300000 ) -> None:
		"""Validate token limits.
		
		
			Purpose:
			    Validates token limits input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    text: Input text supplied to the operation.
			    max_input_tokens: Maximum tokens allowed for a single embedding input.
			    max_total_tokens: Maximum tokens allowed across an embedding input collection.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_request( self, text: str | List[ str ], model: str = 'text-embedding-3-small',
			format: str = 'float', dimensions: int = None, user: str = None ) -> Dict[ str, Any ]:
		"""Build request.
		
		
			Purpose:
			    Builds the request structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    text: Input text supplied to the operation.
			    model: Provider model identifier selected for the operation.
			    format: Output or response format selected for the operation.
			    dimensions: Embedding dimensionality requested for supported models.
			    user: Optional user identifier supplied to embedding requests.
		
			Returns:
			    Dict[ str, Any ]: Normalized provider request dictionary.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
				self.request[ 'dimensions' ] = self.dimensions
			
			if self.user:
				self.request[ 'user' ] = self.user.strip( )
			
			return self.request
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'build_request( self, text: str | List[ str ], **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def create( self, text: str | List[ str ], model: str = 'text-embedding-3-small',
			format: str = 'float', dimensions: int = None,
			user: str = None ) -> List[ float ] | List[ List[ float ] ] | str | List[ str ] | None:
		"""Create.
		
		
			Purpose:
			    Creates the provider resource represented by the current class after validation and request construction.
		
			Args:
			    text: Input text supplied to the operation.
			    model: Provider model identifier selected for the operation.
			    format: Output or response format selected for the operation.
			    dimensions: Embedding dimensionality requested for supported models.
			    user: Optional user identifier supplied to embedding requests.
		
			Returns:
			    List[ float ] | List[ List[ float ] ] | str | List[ str ] | None: Normalized created-resource metadata or None when the provider returns no object.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		
			Purpose:
			    Returns the public Embeddings members displayed by interactive inspection and documentation tooling.
		
			Returns:
			    List[ str ] | None: Public member names exposed for interactive inspection.
		"""
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
	"""Files provider wrapper.
	
	
		Purpose:
		    Builds and executes OpenAI file upload, retrieval, extraction, summarization, search, survey, and deletion workflows for document-enabled modes.
	
		Attributes:
		    api_key: OpenAI API key loaded from project configuration.
		    client: OpenAI client created for the current provider operation.
		    file: Latest file object returned by an OpenAI file workflow.
		    file_id: OpenAI file identifier used by file and vector-store operations.
		    filepath: Local file path supplied to upload, transcription, translation, or vector-store operations.
		    filename: Filename value retained by the Files workflow.
		    purpose: OpenAI file purpose used by upload and file operations.
		    response: Latest provider response object returned by an API call.
		    content: Supplemental content block retained for request construction.
		    files: Named OpenAI file identifiers available to the application.
		    request: Normalized request dictionary prepared for provider execution.
		    model: Provider model identifier used by the current workflow.
		    prompt: User prompt or task instruction used by the current request.
		    output_text: Text extracted from the latest provider response.
	
		Notes:
		    The wrapper stores request state on the instance so Streamlit callbacks, provider calls, and documentation-generated API pages expose consistent runtime behavior.
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
	
	def __init__( self, id: str = None, filepath: str = None, purpose: str = 'user_data',
			model: str = 'gpt-4o-mini', prompt: str = None ):
		"""Initialize Files.
		
		
			Purpose:
			    Initializes Files state by assigning configuration values, request defaults, cached outputs, and compatibility fields used by later methods.
		
			Args:
			    id: OpenAI file identifier used by file operations.
			    filepath: Local file path supplied to upload, transcription, translation, or vector-store workflows.
			    purpose: OpenAI file purpose used for upload or filtering.
			    model: Provider model identifier selected for the operation.
			    prompt: User prompt or task instruction submitted to the provider.
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
		"""Upload purpose options.
		
		
			Purpose:
			    Returns upload purpose options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Upload-purpose option names exposed to the UI selector.
		"""
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
		"""File purpose options.
		
		
			Purpose:
			    Returns file purpose options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: File-purpose option names exposed to the UI selector.
		"""
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
		"""Purpose options.
		
		
			Purpose:
			    Returns purpose options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: File-purpose option names exposed to the UI selector.
		"""
		return self.upload_purpose_options
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Model options.
		
		
			Purpose:
			    Returns model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Model option names exposed to the UI selector.
		"""
		return [
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o-mini',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Reasoning options.
		
		
			Purpose:
			    Returns reasoning options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Reasoning-effort option names exposed to the UI selector.
		"""
		return [
				'none',
				'minimal',
				'low',
				'medium',
				'high',
		]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Include options.
		
		
			Purpose:
			    Returns include options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Include option names exposed to the UI selector.
		"""
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
		"""Tool options.
		
		
			Purpose:
			    Returns tool options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Tool option names exposed to the UI selector.
		"""
		return [
				'web_search',
				'file_search',
		]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Choice options.
		
		
			Purpose:
			    Returns choice options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Tool-choice option names exposed to the UI selector.
		"""
		return [ 'auto', 'required', 'none', ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Modality options.
		
		
			Purpose:
			    Returns modality options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Modality option names exposed to the UI selector.
		"""
		return [
				'text',
		]
	
	def validate_upload_purpose( self, purpose: str = None ) -> str:
		"""Validate upload purpose.
		
		
			Purpose:
			    Validates upload purpose input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    purpose: OpenAI file purpose used for upload or filtering.
		
			Returns:
			    str: Validated file-upload purpose.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_file_id( self, id: str = None ) -> str:
		"""Validate file id.
		
		
			Purpose:
			    Validates file id input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    id: OpenAI file identifier used by file operations.
		
			Returns:
			    str: Validated OpenAI file identifier.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_object( self, file: Any ) -> Dict[ str, Any ]:
		"""Normalize file object.
		
		
			Purpose:
			    Normalizes file object data from provider-specific objects into application-ready Python values.
		
			Args:
			    file: Provider file object to normalize.
		
			Returns:
			    Dict[ str, Any ]: Normalized file metadata dictionary.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_list( self, response: Any, purpose: str = None ) -> List[ Dict[ str, Any ] ]:
		"""Normalize file list.
		
		
			Purpose:
			    Normalizes file list data from provider-specific objects into application-ready Python values.
		
			Args:
			    response: Provider response object to normalize or inspect.
			    purpose: OpenAI file purpose used for upload or filtering.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Normalized file metadata dictionaries returned by list operations.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			
			rows: List[ Dict[ str, Any ] ] = [ ]
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_content( self, content: Any ) -> str | bytes | Dict[ str, Any ] | None:
		"""Normalize file content.
		
		
			Purpose:
			    Normalizes file content data from provider-specific objects into application-ready Python values.
		
			Args:
			    content: Supplemental content block supplied to request construction.
		
			Returns:
			    str | bytes | Dict[ str, Any ] | None: Normalized file-content payload.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def upload( self, filepath: str, purpose: str = 'user_data' ) -> Dict[ str, Any ] | None:
		"""Upload.
		
		
			Purpose:
			    Uploads a local file to OpenAI and returns normalized file metadata for downstream workflows.
		
			Args:
			    filepath: Local file path supplied to upload, transcription, translation, or vector-store workflows.
			    purpose: OpenAI file purpose used for upload or filtering.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized uploaded-file metadata or None when upload fails to return a file.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def list( self, purpose: str = None ) -> List[ Dict[ str, Any ] ]:
		"""List.
		
		
			Purpose:
			    Lists provider resources and normalizes the returned collection for UI display or downstream processing.
		
			Args:
			    purpose: OpenAI file purpose used for upload or filtering.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Normalized provider object list.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.purpose = purpose if isinstance( purpose, str ) and purpose.strip( ) else None
			self.request = { }
			
			if self.purpose:
				self.request[ 'purpose_filter' ] = self.purpose
			
			self.response = self.client.files.list( )
			self.files = self.normalize_file_list( self.response, purpose=self.purpose )
			return self.files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'list( self, purpose: str=None ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve.
		
		
			Purpose:
			    Retrieves provider resource metadata by identifier and normalizes the response.
		
			Args:
			    id: OpenAI file identifier used by file operations.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized provider object metadata or None when the provider returns no object.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def extract( self, id: str ) -> str | bytes | Dict[ str, Any ] | None:
		"""Extract.
		
		
			Purpose:
			    Retrieves file content by identifier and normalizes text, bytes, or structured content for application use.
		
			Args:
			    id: OpenAI file identifier used by file operations.
		
			Returns:
			    str | bytes | Dict[ str, Any ] | None: Extracted file content or None when the provider returns no content.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def delete( self, id: str ) -> Dict[ str, Any ] | None:
		"""Delete.
		
		
			Purpose:
			    Deletes the selected provider resource and returns normalized deletion metadata.
		
			Args:
			    id: OpenAI file identifier used by file operations.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized deletion response or None when deletion returns no object.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def summarize( self, id: str, prompt: str = None, model: str = 'gpt-4o-mini',
			max_chars: int = 120000 ) -> str | None:
		"""Summarize.
		
		
			Purpose:
			    Summarizes file content by combining extracted document text with a model prompt.
		
			Args:
			    id: OpenAI file identifier used by file operations.
			    prompt: User prompt or task instruction submitted to the provider.
			    model: Provider model identifier selected for the operation.
			    max_chars: Maximum extracted characters included in a model prompt.
		
			Returns:
			    str | None: Generated file summary or None when no output text is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def search( self, id: str, query: str, model: str = 'gpt-4o-mini',
			max_chars: int = 120000 ) -> str | None:
		"""Search.
		
		
			Purpose:
			    Searches provider-managed files or vector stores and returns normalized search output for grounded workflows.
		
			Args:
			    id: OpenAI file identifier used by file operations.
			    query: Search query submitted to a file or vector-store workflow.
			    model: Provider model identifier selected for the operation.
			    max_chars: Maximum extracted characters included in a model prompt.
		
			Returns:
			    str | None: Search results or generated answer text returned by the provider workflow.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			throw_if( 'query', query )
			prompt = ('Answer the user question using the uploaded file when possible.\n\n'
			          f'Question: {query}')
			
			return self.summarize( id=id, prompt=prompt, model=model, max_chars=max_chars )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'search( self, id: str, query: str ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, id: str, max_chars: int = 4000 ) -> Dict[ str, Any ]:
		"""Survey.
		
		
			Purpose:
			    Surveys provider-managed file or vector-store content and returns structured descriptive output.
		
			Args:
			    id: OpenAI file identifier used by file operations.
			    max_chars: Maximum extracted characters included in a model prompt.
		
			Returns:
			    Dict[ str, Any ]: Structured file or vector-store survey output.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		
			Purpose:
			    Returns the public Files members displayed by interactive inspection and documentation tooling.
		
			Returns:
			    List[ str ] | None: Public member names exposed for interactive inspection.
		"""
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
	"""VectorStores provider wrapper.
	
	
		Purpose:
		    Builds and executes OpenAI vector-store management workflows for persistent file search, batch ingestion, search, and file-grounded answers.
	
		Attributes:
		    api_key: OpenAI API key loaded from project configuration.
		    client: OpenAI client created for the current provider operation.
		    name: Name value retained by the VectorStores workflow.
		    description: Description value retained by the VectorStores workflow.
		    store_id: Vector-store identifier used by store and file operations.
		    file_id: OpenAI file identifier used by file and vector-store operations.
		    batch_id: Batch id value retained by the VectorStores workflow.
		    response: Latest provider response object returned by an API call.
		    vector_store: Vector store value retained by the VectorStores workflow.
		    vector_stores: Named OpenAI vector-store identifiers available to the application.
		    vector_file: Vector file value retained by the VectorStores workflow.
		    vector_files: Vector files value retained by the VectorStores workflow.
		    file_batch: File batch value retained by the VectorStores workflow.
		    search_results: Search results value retained by the VectorStores workflow.
		    output_text: Text extracted from the latest provider response.
		    request: Normalized request dictionary prepared for provider execution.
		    collections: Collections value retained by the VectorStores workflow.
		    max_search_results: Maximum number of search results requested by supported tools.
	
		Notes:
		    The wrapper stores request state on the instance so Streamlit callbacks, provider calls, and documentation-generated API pages expose consistent runtime behavior.
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
	
	def __init__( self, name: str = None, store_id: str = None, file_id: str = None,
			model: str = 'gpt-4o-mini', max_search_results: int = 10 ):
		"""Initialize VectorStores.
		
		
			Purpose:
			    Initializes VectorStores state by assigning configuration values, request defaults, cached outputs, and compatibility fields used by later methods.
		
			Args:
			    name: Resource, argument, or store name to validate or use.
			    store_id: Vector-store identifier used by store and file operations.
			    file_id: File id supplied to the init workflow.
			    model: Provider model identifier selected for the operation.
			    max_search_results: Maximum search-result count retained for compatible tools.
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
		"""Model options.
		
		
			Purpose:
			    Returns model options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Model option names exposed to the UI selector.
		"""
		return [
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o-mini',
		]
	
	@property
	def ranker_options( self ) -> List[ str ] | None:
		"""Ranker options.
		
		
			Purpose:
			    Returns ranker options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Vector-store ranker option names exposed to the UI selector.
		"""
		return [
				'auto',
				'default-2024-11-15',
		]
	
	@property
	def chunking_strategy_options( self ) -> List[ str ] | None:
		"""Chunking strategy options.
		
		
			Purpose:
			    Returns chunking strategy options used by the Streamlit selectors and provider request builders.
		
			Returns:
			    List[ str ] | None: Chunking strategy option names exposed to the UI selector.
		"""
		return [
				'auto',
				'static',
		]
	
	def validate_store_name( self, name: str = None ) -> str:
		"""Validate store name.
		
		
			Purpose:
			    Validates store name input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    name: Resource, argument, or store name to validate or use.
		
			Returns:
			    str: Validated vector-store name.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_store_id( self, store_id: str = None ) -> str:
		"""Validate store id.
		
		
			Purpose:
			    Validates store id input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
		
			Returns:
			    str: Validated vector-store identifier.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_file_id( self, file_id: str = None ) -> str:
		"""Validate file id.
		
		
			Purpose:
			    Validates file id input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    file_id: File id supplied to the validate file id workflow.
		
			Returns:
			    str: Validated OpenAI file identifier.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_batch_id( self, batch_id: str = None ) -> str:
		"""Validate batch id.
		
		
			Purpose:
			    Validates batch id input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    batch_id: Vector-store file-batch identifier.
		
			Returns:
			    str: Validated vector-store batch identifier.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_file_ids( self, file_ids: List[ str ] = None ) -> List[ str ]:
		"""Validate file ids.
		
		
			Purpose:
			    Validates file ids input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    file_ids: OpenAI file identifiers attached to a vector store or batch.
		
			Returns:
			    List[ str ]: Validated file identifier list.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def validate_max_num_results( self, max_num_results: int = None ) -> int:
		"""Validate max num results.
		
		
			Purpose:
			    Validates max num results input, stores the normalized value on the instance when applicable, and blocks unsupported provider values before API execution.
		
			Args:
			    max_num_results: Max num results supplied to the validate max num results workflow.
		
			Returns:
			    int: Validated maximum search-result count.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_expires_after( self, anchor: str = None, days: int = None ) -> Dict[
		                                                                         str, Any ] | None:
		"""Build expires after.
		
		
			Purpose:
			    Builds the expires after structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    anchor: Expiration anchor used for vector-store retention policy.
			    days: Retention duration in days for vector-store expiration policy.
		
			Returns:
			    Dict[ str, Any ] | None: Vector-store expiration policy or None when no policy is selected.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def build_chunking_strategy( self, strategy: str = 'auto', max_chunk_size_tokens: int = None,
			chunk_overlap_tokens: int = None ) -> Dict[ str, Any ] | None:
		"""Build chunking strategy.
		
		
			Purpose:
			    Builds the chunking strategy structure required by the OpenAI workflow and stores the normalized request state on the instance.
		
			Args:
			    strategy: Chunking strategy name selected for vector-store ingestion.
			    max_chunk_size_tokens: Maximum token count per vector-store chunk.
			    chunk_overlap_tokens: Token overlap count between vector-store chunks.
		
			Returns:
			    Dict[ str, Any ] | None: Vector-store chunking strategy or None when default chunking is used.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_vector_store( self, store: Any ) -> Dict[ str, Any ]:
		"""Normalize vector store.
		
		
			Purpose:
			    Normalizes vector store data from provider-specific objects into application-ready Python values.
		
			Args:
			    store: Response storage flag supplied to compatible provider requests.
		
			Returns:
			    Dict[ str, Any ]: Normalized vector-store metadata dictionary.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_vector_store_file( self, file: Any ) -> Dict[ str, Any ]:
		"""Normalize vector store file.
		
		
			Purpose:
			    Normalizes vector store file data from provider-specific objects into application-ready Python values.
		
			Args:
			    file: Provider file object to normalize.
		
			Returns:
			    Dict[ str, Any ]: Normalized vector-store file metadata dictionary.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_batch( self, batch: Any ) -> Dict[ str, Any ]:
		"""Normalize file batch.
		
		
			Purpose:
			    Normalizes file batch data from provider-specific objects into application-ready Python values.
		
			Args:
			    batch: Batch supplied to the normalize file batch workflow.
		
			Returns:
			    Dict[ str, Any ]: Normalized vector-store file-batch metadata dictionary.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def normalize_search_results( self, response: Any ) -> List[ Dict[ str, Any ] ]:
		"""Normalize search results.
		
		
			Purpose:
			    Normalizes search results data from provider-specific objects into application-ready Python values.
		
			Args:
			    response: Provider response object to normalize or inspect.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Normalized vector-store search-result dictionaries.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			
			rows: List[ Dict[ str, Any ] ] = [ ]
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
			Logger( ).write( exception )
			raise exception
	
	def create( self, name: str, description: str = None, metadata: Dict[ str, Any ] = None,
			expires_after: Dict[ str, Any ] = None, file_ids: List[ str ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Create.
		
		
			Purpose:
			    Creates the provider resource represented by the current class after validation and request construction.
		
			Args:
			    name: Resource, argument, or store name to validate or use.
			    description: Description metadata supplied to a vector store.
			    metadata: Metadata dictionary supplied to a vector store or attached file.
			    expires_after: Expiration policy supplied to a vector store.
			    file_ids: OpenAI file identifiers attached to a vector store or batch.
			    chunking_strategy: Chunking strategy supplied to vector-store ingestion.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized created-resource metadata or None when the provider returns no object.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
				self.request[ 'description' ] = self.description
			
			if isinstance( metadata, dict ) and len( metadata ) > 0:
				self.request[ 'metadata' ] = metadata
			
			if isinstance( expires_after, dict ) and len( expires_after ) > 0:
				self.request[ 'expires_after' ] = expires_after
			
			clean_file_ids = self.validate_file_ids( file_ids )
			if len( clean_file_ids ) > 0:
				self.request[ 'file_ids' ] = clean_file_ids
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ] = chunking_strategy
			
			self.response = self.client.vector_stores.create( **self.request )
			self.vector_store = self.normalize_vector_store( self.response )
			self.store_id = self.vector_store.get( 'id' )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'create( self, name: str, **kwargs ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def list_stores( self, limit: int = 100, order: str = 'desc',
			after: str = None, before: str = None ) -> List[ Dict[ str, Any ] ]:
		"""List stores.
		
		
			Purpose:
			    Lists OpenAI vector stores and normalizes store metadata for UI display.
		
			Args:
			    limit: Maximum number of provider objects to return.
			    order: Sort order supplied to list operations.
			    after: Pagination cursor for results after a provider object.
			    before: Pagination cursor for results before a provider object.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Normalized vector-store metadata dictionaries.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = {
					'limit': limit,
					'order': order,
			}
			
			if isinstance( after, str ) and after.strip( ):
				self.request[ 'after' ] = after.strip( )
			
			if isinstance( before, str ) and before.strip( ):
				self.request[ 'before' ] = before.strip( )
			
			self.response = self.client.vector_stores.list( **self.request )
			items = getattr( self.response, 'data', [ ] )
			self.vector_stores = [ self.normalize_vector_store( item ) for item in items ]
			return self.vector_stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_stores( self, limit: int=100 )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve.
		
		
			Purpose:
			    Retrieves provider resource metadata by identifier and normalizes the response.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized provider object metadata or None when the provider returns no object.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def update( self, store_id: str, name: str = None, description: str = None,
			metadata: Dict[ str, Any ] = None,
			expires_after: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Update.
		
		
			Purpose:
			    Executes the update workflow for the VectorStores provider wrapper while preserving normalized instance state for downstream use.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    name: Resource, argument, or store name to validate or use.
			    description: Description metadata supplied to a vector store.
			    metadata: Metadata dictionary supplied to a vector store or attached file.
			    expires_after: Expiration policy supplied to a vector store.
		
			Returns:
			    Dict[ str, Any ] | None: Result produced by the provider workflow.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.request = { }
			
			if isinstance( name, str ) and name.strip( ):
				self.request[ 'name' ] = name.strip( )
			
			if isinstance( description, str ) and description.strip( ):
				self.request[ 'description' ] = description.strip( )
			
			if isinstance( metadata, dict ):
				self.request[ 'metadata' ] = metadata
			
			if isinstance( expires_after, dict ) and len( expires_after ) > 0:
				self.request[ 'expires_after' ] = expires_after
			
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
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str ) -> Dict[ str, Any ] | None:
		"""Delete.
		
		
			Purpose:
			    Deletes the selected provider resource and returns normalized deletion metadata.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized deletion response or None when deletion returns no object.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def list( self, store_id: str, limit: int = 100, order: str = 'desc' ) -> List[
		Dict[ str, Any ] ]:
		"""List.
		
		
			Purpose:
			    Lists provider resources and normalizes the returned collection for UI display or downstream processing.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    limit: Maximum number of provider objects to return.
			    order: Sort order supplied to list operations.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Normalized provider object list.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			return self.list_files( store_id=store_id, limit=limit, order=order )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list( self, store_id: str ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def list_files( self, store_id: str, limit: int = 100,
			order: str = 'desc' ) -> List[ Dict[ str, Any ] ]:
		"""List files.
		
		
			Purpose:
			    Lists files attached to a vector store and normalizes file metadata for UI display.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    limit: Maximum number of provider objects to return.
			    order: Sort order supplied to list operations.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Normalized vector-store file metadata dictionaries.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve file.
		
		
			Purpose:
			    Retrieves metadata for a file attached to a vector store.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    file_id: File id supplied to the retrieve file workflow.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized vector-store file metadata or None when no file is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def attach_file( self, store_id: str = None, file_id: str = None,
			vector_store_id: str = None, attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Attach file.
		
		
			Purpose:
			    Attaches an existing OpenAI file to a vector store and returns normalized attachment metadata.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    file_id: File id supplied to the attach file workflow.
			    vector_store_id: Alternate vector-store identifier retained for UI compatibility.
			    attributes: File attributes supplied to vector-store file operations.
			    chunking_strategy: Chunking strategy supplied to vector-store ingestion.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized attached-file metadata or None when attachment returns no object.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def upload_and_attach( self, store_id: str = None, vector_store_id: str = None,
			path: str = None, filepath: str = None, purpose: str = 'assistants',
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Upload and attach.
		
		
			Purpose:
			    Uploads a local file and attaches it to a vector store in one workflow.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    vector_store_id: Alternate vector-store identifier retained for UI compatibility.
			    path: Local file path supplied to image, audio, or vector-store workflows.
			    filepath: Local file path supplied to upload, transcription, translation, or vector-store workflows.
			    purpose: OpenAI file purpose used for upload or filtering.
			    attributes: File attributes supplied to vector-store file operations.
			    chunking_strategy: Chunking strategy supplied to vector-store ingestion.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized uploaded-and-attached file metadata or None when no object is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def upload_file( self, store_id: str = None, vector_store_id: str = None,
			path: str = None, filepath: str = None, purpose: str = 'assistants',
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Upload file.
		
		
			Purpose:
			    Uploads a local file for vector-store use and returns normalized file metadata.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    vector_store_id: Alternate vector-store identifier retained for UI compatibility.
			    path: Local file path supplied to image, audio, or vector-store workflows.
			    filepath: Local file path supplied to upload, transcription, translation, or vector-store workflows.
			    purpose: OpenAI file purpose used for upload or filtering.
			    attributes: File attributes supplied to vector-store file operations.
			    chunking_strategy: Chunking strategy supplied to vector-store ingestion.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized uploaded-file metadata or None when upload returns no object.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def attach_upload( self, store_id: str = None, vector_store_id: str = None,
			path: str = None, filepath: str = None, purpose: str = 'assistants',
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Attach upload.
		
		
			Purpose:
			    Uploads a local file and attaches it to a vector store in one compatibility workflow.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    vector_store_id: Alternate vector-store identifier retained for UI compatibility.
			    path: Local file path supplied to image, audio, or vector-store workflows.
			    filepath: Local file path supplied to upload, transcription, translation, or vector-store workflows.
			    purpose: OpenAI file purpose used for upload or filtering.
			    attributes: File attributes supplied to vector-store file operations.
			    chunking_strategy: Chunking strategy supplied to vector-store ingestion.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized uploaded-and-attached file metadata or None when no object is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def upload( self, store_id: str = None, vector_store_id: str = None,
			path: str = None, filepath: str = None, purpose: str = 'assistants',
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Upload.
		
		
			Purpose:
			    Uploads a local file to OpenAI and returns normalized file metadata for downstream workflows.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    vector_store_id: Alternate vector-store identifier retained for UI compatibility.
			    path: Local file path supplied to image, audio, or vector-store workflows.
			    filepath: Local file path supplied to upload, transcription, translation, or vector-store workflows.
			    purpose: OpenAI file purpose used for upload or filtering.
			    attributes: File attributes supplied to vector-store file operations.
			    chunking_strategy: Chunking strategy supplied to vector-store ingestion.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized uploaded-file metadata or None when upload fails to return a file.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def create_file_batch( self, store_id: str = None, file_ids: List[ str ] = None,
			vector_store_id: str = None, attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Create file batch.
		
		
			Purpose:
			    Creates a vector-store file batch from validated file identifiers.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    file_ids: OpenAI file identifiers attached to a vector store or batch.
			    vector_store_id: Alternate vector-store identifier retained for UI compatibility.
			    attributes: File attributes supplied to vector-store file operations.
			    chunking_strategy: Chunking strategy supplied to vector-store ingestion.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized file-batch metadata or None when no batch is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def update_file( self, store_id: str, file_id: str,
			attributes: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Update file.
		
		
			Purpose:
			    Updates vector-store file attributes and returns normalized file metadata.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    file_id: File id supplied to the update file workflow.
			    attributes: File attributes supplied to vector-store file operations.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized vector-store file metadata or None when no file is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			self.request = { }
			
			if isinstance( attributes, dict ):
				self.request[ 'attributes' ] = attributes
			
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
			Logger( ).write( exception )
			raise exception
	
	def delete_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ] | None:
		"""Delete file.
		
		
			Purpose:
			    Deletes a file attachment from a vector store and returns normalized deletion metadata.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    file_id: File id supplied to the delete file workflow.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized deletion response or None when deletion returns no object.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file_content( self, store_id: str, file_id: str ) -> Any:
		"""Retrieve file content.
		
		
			Purpose:
			    Retrieves provider content for a file attached to a vector store.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    file_id: File id supplied to the retrieve file content workflow.
		
			Returns:
			    Any: Provider file-content payload returned by the vector-store file operation.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve file batch.
		
		
			Purpose:
			    Retrieves vector-store file-batch metadata by batch identifier.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    batch_id: Vector-store file-batch identifier.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized file-batch metadata or None when no batch is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def list_file_batch_files( self, store_id: str, batch_id: str,
			limit: int = 100 ) -> List[ Dict[ str, Any ] ]:
		"""List file batch files.
		
		
			Purpose:
			    Lists files associated with a vector-store file batch.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    batch_id: Vector-store file-batch identifier.
			    limit: Maximum number of provider objects to return.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Normalized file metadata dictionaries for a file batch.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def cancel_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ] | None:
		"""Cancel file batch.
		
		
			Purpose:
			    Cancels an active vector-store file batch and returns normalized batch metadata.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    batch_id: Vector-store file-batch identifier.
		
			Returns:
			    Dict[ str, Any ] | None: Normalized cancelled-batch metadata or None when no batch is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def search( self, store_id: str, query: str, max_num_results: int = 10,
			filters: Dict[ str, Any ] = None, ranking_options: Dict[ str, Any ] = None,
			rewrite_query: bool = None ) -> List[ Dict[ str, Any ] ]:
		"""Search.
		
		
			Purpose:
			    Searches provider-managed files or vector stores and returns normalized search output for grounded workflows.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    query: Search query submitted to a file or vector-store workflow.
			    max_num_results: Max num results supplied to the search workflow.
			    filters: Search filters supplied to vector-store search.
			    ranking_options: Ranking options supplied to vector-store search.
			    rewrite_query: Flag controlling provider query rewriting.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Search results or generated answer text returned by the provider workflow.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
		"""
		try:
			return self.search_store( store_id=store_id, query=query,
				max_num_results=max_num_results,
				filters=filters, ranking_options=ranking_options, rewrite_query=rewrite_query )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search( self, store_id: str, query: str )'
			Logger( ).write( exception )
			raise exception
	
	def search_store( self, store_id: str, query: str, max_num_results: int = 10,
			filters: Dict[ str, Any ] = None, ranking_options: Dict[ str, Any ] = None,
			rewrite_query: bool = None ) -> List[ Dict[ str, Any ] ]:
		"""Search store.
		
		
			Purpose:
			    Searches a vector store and returns normalized search-result dictionaries.
		
			Args:
			    store_id: Vector-store identifier used by store and file operations.
			    query: Search query submitted to a file or vector-store workflow.
			    max_num_results: Max num results supplied to the search store workflow.
			    filters: Search filters supplied to vector-store search.
			    ranking_options: Ranking options supplied to vector-store search.
			    rewrite_query: Flag controlling provider query rewriting.
		
			Returns:
			    List[ Dict[ str, Any ] ]: Normalized vector-store search-result dictionaries.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
				self.request[ 'filters' ] = filters
			
			if isinstance( ranking_options, dict ) and len( ranking_options ) > 0:
				self.request[ 'ranking_options' ] = ranking_options
			
			if isinstance( rewrite_query, bool ):
				self.request[ 'rewrite_query' ] = rewrite_query
			
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
			Logger( ).write( exception )
			raise exception
	
	def answer_with_file_search( self, store_ids: List[ str ], prompt: str,
			model: str = 'gpt-4o-mini', max_num_results: int = 10,
			instructions: str = None ) -> str | None:
		"""Answer with file search.
		
		
			Purpose:
			    Generates a file-grounded answer using OpenAI file_search tool configuration.
		
			Args:
			    store_ids: Vector-store identifiers used for file-search answer generation.
			    prompt: User prompt or task instruction submitted to the provider.
			    model: Provider model identifier selected for the operation.
			    max_num_results: Max num results supplied to the answer with file search workflow.
			    instructions: System or developer instructions supplied to answer generation.
		
			Returns:
			    str | None: Generated answer text from file-search grounding or None when no output text is returned.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			
			input_items: List[ Dict[ str, Any ] ] = [ ]
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
			Logger( ).write( exception )
			raise exception
	
	def survey( self, store_ids: List[ str ], prompt: str = None, model: str = 'gpt-4o-mini',
			max_num_results: int = 10, instructions: str = None ) -> str | None:
		"""Survey.
		
		
			Purpose:
			    Surveys provider-managed file or vector-store content and returns structured descriptive output.
		
			Args:
			    store_ids: Vector-store identifiers used for file-search answer generation.
			    prompt: User prompt or task instruction submitted to the provider.
			    model: Provider model identifier selected for the operation.
			    max_num_results: Max num results supplied to the survey workflow.
			    instructions: System or developer instructions supplied to answer generation.
		
			Returns:
			    str | None: Structured file or vector-store survey output.
		
			Raises:
			    Error: Re-raised after provider validation, request construction, or API execution errors are wrapped and logged.
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
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Dir.
		
		
			Purpose:
			    Returns the public VectorStores members displayed by interactive inspection and documentation tooling.
		
			Returns:
			    List[ str ] | None: Public member names exposed for interactive inspection.
		"""
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
		