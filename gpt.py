"""OpenAI provider wrapper for Buddy.


	Purpose:
	    Provides OpenAI-backed chat, image, audio, embedding, file, and vector-store workflows
	    used by the Buddy Streamlit application and MkDocs API reference.

	Notes:
	    The module preserves the project provider-wrapper pattern, local validation helpers,
	    OpenAI request builders, and wrapped exception handling used by the application.
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
		Validates a required value before a provider or application operation proceeds. The
		function raises a ValueError when the supplied value is missing, blank, or empty.
	
	Args:
		name (str): Name value used by the operation.
		value (object): Value value used by the operation.
	
	Raises:
		ValueError: Raised when required input is missing or invalid.
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
		Reads a local image file and converts its bytes into a base64-encoded string. The
		encoded value is used by image and vision workflows that require inline image content.
	
	Args:
		image_path (str): Image path value used by the operation.
	
	Returns:
		Base64-encoded image content.
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class GPT:
	"""Provide GPT workflow support.
	
	Purpose:
		Provides the shared OpenAI wrapper base used by Gipity provider workflows. The class
		stores common model, prompt, request, response, and compatibility fields inherited by
		text, image, audio, embedding, file, and vector-store wrappers.
	
	Attributes:
		api_key (Optional[str]): Api key retained by the provider wrapper.
		client (Optional[OpenAI]): Client retained by the provider wrapper.
		prompt (Optional[str]): Prompt retained by the provider wrapper.
		temperature (Optional[float]): Temperature retained by the provider wrapper.
		top_percent (Optional[float]): Top percent retained by the provider wrapper.
		frequency_penalty (Optional[float]): Frequency penalty retained by the provider wrapper.
		presence_penalty (Optional[float]): Presence penalty retained by the provider wrapper.
		max_tokens (Optional[int]): Max tokens retained by the provider wrapper.
		stops (Optional[List[str]]): Stops retained by the provider wrapper.
		store (Optional[bool]): Store retained by the provider wrapper.
		stream (Optional[bool]): Stream retained by the provider wrapper.
		background (Optional[bool]): Background retained by the provider wrapper.
		number (Optional[int]): Number retained by the provider wrapper.
		response_format (Optional[Dict[str, str]]): Response format retained by the provider
			wrapper.
		context (Optional[List[Dict[str, str]]]): Context retained by the provider wrapper.
		instructions (Optional[str]): Instructions retained by the provider wrapper.
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
		"""Initialize instance.
		
		Purpose:
			Initializes the GPT object with default configuration, runtime state, provider
			settings,
			and compatibility fields. This constructor prepares the instance for later method calls
			without performing external work beyond local attribute assignment.
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
	"""Provide OpenAI Responses API text-generation support.
	
	Purpose:
		Provides the OpenAI Responses API implementation used by Text mode. The class stores
		request arguments as object members, constructs provider-specific input, tool, response-
		format, reasoning, and continuation payloads, executes synchronous or streaming requests,
		and exposes response text and usage information to the application.
	
	Attributes:
		include (List[str]): Additional response fields requested from the provider.
		tool_choice (str): Tool-selection behavior used by the request.
		previous_id (str): Previous response identifier used for continuation.
		conversation_id (str): Conversation identifier used for continuation.
		parallel_tools (bool): Indicates whether parallel tool calls are permitted.
		max_tools (int): Maximum number of built-in tool calls permitted.
		input (List[Dict[str, Any]]): Input messages sent to the provider.
		tools (List[Dict[str, Any]]): Provider-ready tool definitions.
		reasoning (Dict[str, str]): Provider-ready reasoning configuration.
		allowed_domains (List[str]): Domains allowed by the web-search tool.
		output_text (str): Text extracted from the latest response.
		vector_store_ids (List[str]): Vector store identifiers used by file search.
		response (Optional[Response]): Latest OpenAI response object.
	"""
	include: List[ str ]
	tool_choice: str
	previous_id: str
	conversation_id: str
	parallel_tools: bool
	max_tools: int
	input: List[ Dict[ str, Any ] ]
	tools: List[ Dict[ str, Any ] ]
	reasoning: Dict[ str, str ]
	allowed_domains: List[ str ]
	output_text: str
	vector_store_ids: List[ str ]
	response: Optional[ Response ]
	
	def __init__( self, model: str = 'gpt-5-nano', prompt: str = '', temperature: float = 0.0,
		top_p: float = 0.0, frequency: float = 0.0, presence: float = 0.0, max_tokens: int = 0,
		max_tools: int = 0, store: bool = False, stream: bool = False, background: bool = False,
		is_parallel: bool = False, instruct: str = '', tool_choice: str = '', previous_id: str = '',
		conversation_id: str = '', reasoning: str = '',
		response_format: Optional[ Dict[ str, Any ] ] = None,
		context: Optional[ List[ Dict[ str, Any ] ] ] = None,
		allowed_domains: Optional[ List[ str ] ] = None, include: Optional[ List[ str ] ] = None,
		tools: Optional[ List[ str | Dict[ str, Any ] ] ] = None,
		input_data: Optional[ List[ Dict[ str, Any ] ] ] = None,
		vector_store_ids: Optional[ List[ str ] ] = None ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes the OpenAI Chat wrapper with explicit defaults and provider-request
			state. The constructor performs local assignment only and does not execute an API
			request.
		
		Args:
			model (str): OpenAI model identifier.
			prompt (str): User prompt retained for a later request.
			temperature (float): Sampling temperature retained for supported models.
			top_p (float): Nucleus-sampling value retained for supported models.
			frequency (float): Frequency penalty retained for supported models.
			presence (float): Presence penalty retained for supported models.
			max_tokens (int): Maximum output-token count.
			max_tools (int): Maximum number of built-in tool calls.
			store (bool): Indicates whether the response should be stored.
			stream (bool): Indicates whether response events should be streamed.
			background (bool): Indicates whether the response should run in background mode.
			is_parallel (bool): Indicates whether parallel tool calls are permitted.
			instruct (str): System or developer instructions.
			tool_choice (str): Tool-selection behavior.
			previous_id (str): Previous response identifier.
			conversation_id (str): Conversation identifier.
			reasoning (str): Reasoning effort.
			response_format (Optional[Dict[str, Any]]): Text-format configuration.
			context (Optional[List[Dict[str, Any]]]): Prior application messages.
			allowed_domains (Optional[List[str]]): Domains allowed by web search.
			include (Optional[List[str]]): Additional response fields to include.
			tools (Optional[List[str | Dict[str, Any]]]): Selected provider tools.
			input_data (Optional[List[Dict[str, Any]]]): Prebuilt Responses API input items.
			vector_store_ids (Optional[List[str]]): Vector stores used by file search.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.prompt = prompt
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.max_tools = max_tools
		self.store = store
		self.stream = stream
		self.background = background
		self.parallel_tools = is_parallel
		self.instructions = instruct
		self.tool_choice = tool_choice
		self.previous_id = previous_id
		self.conversation_id = conversation_id
		self.reasoning_effort = reasoning
		self.reasoning = { }
		self.response_format = response_format if response_format is not None else { }
		self.context = context if context is not None else [ ]
		self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
		self.include = include if include is not None else [ ]
		self.selected_tools = tools if tools is not None else [ ]
		self.tools = [ ]
		self.input = input_data if input_data is not None else [ ]
		self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
		self.response = None
		self.output_text = ''
		self.request = { }
		self.messages = [ ]
		self.stream_events = [ ]
		self.response_stream = None
		self.requested_format = None
		self.effective_context = [ ]
		self.vector_stores = { 'Governance': 'vs_6a1850a9bdc08191912353eedf59aede',
			'Public Laws': 'vs_699506f7d5348191990e0557c717fa9d',
			'Explanatory Statements': 'vs_699505df9ac48191a525c0ecb86fef66',
			'Army Techniques Publications': 'vs_699356ef052c81918da14c4ed3bcea17',
			'Army Field Manuals': 'vs_69935542863481918d150c1e89c38633',
			'Army Regulations': 'vs_6993550488408191919cd70968ba8be8',
			'DoD Armory': 'vs_697f86ad98888191b967685ae558bfc0',
			'Army Style Guides': 'vs_68f4efd7d4c4819191458dd6cde6f2cc',
			'Apportionments': 'vs_68a34aaff93481918c3b3fef8c4e8fea',
			'Financial Regulations': 'vs_712r5W5833G6aLxIYIbuvVcK', }
		self.files = { 'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
			'SF133.csv': 'file-WT2h2F5SNxqK2CxyAMSDg6',
			'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
			'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn', }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get model options.
		
		Purpose:
			Returns model identifiers exposed to the application for OpenAI Text mode.
		
		Returns:
			List[str]: Available OpenAI text-generation models.
		"""
		return [ 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'gpt-5.1', 'gpt-5', 'gpt-5-mini',
			'gpt-5-nano', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Get include options.
		
		Purpose:
			Returns additional response fields supported by the Responses API workflow.
		
		Returns:
			List[str]: Supported include-path values.
		"""
		return [ 'file_search_call.results', 'web_search_call.action.sources',
			'code_interpreter_call.outputs', 'reasoning.encrypted_content',
			'message.output_text.logprobs', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Get tool options.
		
		Purpose:
			Returns built-in tools implemented by this wrapper.
		
		Returns:
			List[str]: Supported built-in tool names.
		"""
		return [ 'web_search', 'file_search', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Get tool-choice options.
		
		Purpose:
			Returns tool-selection values accepted by the Responses API workflow.
		
		Returns:
			List[str]: Supported tool-choice values.
		"""
		return [ 'auto', 'required', 'none', ]
	
	@property
	def purpose_options( self ) -> List[ str ]:
		"""Get file-purpose options.
		
		Purpose:
			Returns file-purpose values retained for compatibility with file workflows.
		
		Returns:
			List[str]: Supported file-purpose values.
		"""
		return [ 'assistants', 'batch', 'fine-tune', 'vision', 'user_data', 'evals', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get response-format options.
		
		Purpose:
			Returns text-format values implemented by the Responses API request builder.
		
		Returns:
			List[str]: Supported response-format values.
		"""
		return [ 'text', 'json_object', 'json_schema', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Get reasoning options.
		
		Purpose:
			Returns reasoning-effort values supported by current OpenAI reasoning models.
		
		Returns:
			List[str]: Supported reasoning-effort values.
		"""
		return [ 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Get modality options.
		
		Purpose:
			Returns the output modality implemented by the Text-mode wrapper.
		
		Returns:
			List[str]: Supported output modalities.
		"""
		return [ 'text' ]
	
	def supports_reasoning_model( self, model: str = '' ) -> bool:
		"""Determine reasoning-model support.
		
		Purpose:
			Determines whether the selected model accepts a Responses API reasoning object.
		
		Args:
			model (str): Model identifier to inspect.
		
		Returns:
			bool: True when the model supports reasoning configuration.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.model = model if model else self.model
			return self.model.startswith( 'gpt-5' ) or self.model.startswith( 'o' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Chat'
			ex.method = 'supports_reasoning_model( self, model: str = "" ) -> bool'
			Logger( ).write( ex )
			raise ex
	
	def build_reasoning( self, reasoning: str = '', model: str = '' ) -> Dict[ str, str ]:
		"""Build reasoning configuration.
		
		Purpose:
			Builds the provider-ready reasoning object for a supported model and effort value.
		
		Args:
			reasoning (str): Requested reasoning effort.
			model (str): OpenAI model identifier.
		
		Returns:
			Dict[str, str]: Provider-ready reasoning configuration or an empty dictionary.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.reasoning_effort = reasoning
			self.model = model if model else self.model
			self.reasoning = { }
			
			if not self.reasoning_effort:
				return self.reasoning
			
			if self.reasoning_effort == 'none':
				return self.reasoning
			
			if not self.supports_reasoning_model( self.model ):
				return self.reasoning
			
			if self.reasoning_effort not in self.reasoning_options:
				return self.reasoning
			
			if self.model.startswith( 'gpt-5.1' ):
				if self.reasoning_effort in [ 'minimal', 'xhigh' ]:
					return self.reasoning
			
			if self.reasoning_effort == 'xhigh':
				if not self.model.startswith( 'gpt-5.4' ):
					self.reasoning_effort = 'high'
			
			self.reasoning = { 'effort': self.reasoning_effort, }
			return self.reasoning
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Chat'
			ex.method = ('build_reasoning( self, reasoning: str = "", '
			             'model: str = "" ) -> Dict[ str, str ]')
			Logger( ).write( ex )
			raise ex
	
	def build_input( self, prompt: str, context: Optional[ List[ Dict[ str, Any ] ] ] = None,
		input_data: Optional[ List[ Dict[ str, Any ] ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""Build input messages.
		
		Purpose:
			Builds Responses API input items from prebuilt input data or application history and
			appends the current user prompt.
		
		Args:
			prompt (str): Current user prompt.
			context (Optional[List[Dict[str, Any]]]): Prior application messages.
			input_data (Optional[List[Dict[str, Any]]]): Prebuilt Responses API input items.
		
		Returns:
			List[Dict[str, Any]]: Provider-ready Responses API input items.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.context = context if context is not None else [ ]
			self.input = input_data if input_data is not None else [ ]
			self.messages = [ ]
			
			if self.input:
				self.messages.extend( self.input )
			else:
				for item in self.context:
					if not isinstance( item, dict ):
						continue
					
					self.message_role = item.get( 'role', '' )
					self.message_content = item.get( 'content', '' )
					
					if self.message_role not in [ 'user', 'assistant', 'system', 'developer', ]:
						continue
					
					if not self.message_content:
						continue
					
					self.messages.append( { 'role': self.message_role,
						'content': [ { 'type': 'input_text', 'text': self.message_content, }, ],
					} )
			
			self.messages.append( { 'role': 'user',
				'content': [ { 'type': 'input_text', 'text': self.prompt, }, ], } )
			self.input = self.messages
			return self.input
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Chat'
			ex.method = 'build_input( self, **kwargs ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( ex )
			raise ex
	
	def build_tools( self, tools: Optional[ List[ str | Dict[ str, Any ] ] ] = None,
		allowed_domains: Optional[ List[ str ] ] = None,
		vector_store_ids: Optional[ List[ str ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""Build tool definitions.
		
		Purpose:
			Builds OpenAI web-search and file-search tool definitions from application-selected
			tool names.
		
		Args:
			tools (Optional[List[str | Dict[str, Any]]]): Selected provider tools.
			allowed_domains (Optional[List[str]]): Domains permitted by web search.
			vector_store_ids (Optional[List[str]]): Vector stores used by file search.
		
		Returns:
			List[Dict[str, Any]]: Provider-ready OpenAI tool definitions.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.selected_tools = tools if tools is not None else [ ]
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.vector_store_ids = (vector_store_ids if vector_store_ids is not None else [ ])
			self.tools = [ ]
			
			for selected_tool in self.selected_tools:
				if isinstance( selected_tool, dict ):
					self.tool_name = selected_tool.get( 'type', '' )
				else:
					self.tool_name = selected_tool
				
				if self.tool_name in [ 'web_search', 'web_search_preview',
					'web_search_preview_2025_03_11', ]:
					self.web_search_tool = { 'type': 'web_search', }
					
					if self.allowed_domains:
						self.web_search_tool[ 'filters' ] = {
							'allowed_domains': self.allowed_domains, }
					
					self.tools.append( self.web_search_tool )
					continue
				
				if self.tool_name == 'file_search':
					throw_if( 'vector_store_ids', self.vector_store_ids )
					self.tools.append(
						{ 'type': 'file_search', 'vector_store_ids': self.vector_store_ids, } )
			
			return self.tools
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Chat'
			ex.method = 'build_tools( self, **kwargs ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( ex )
			raise ex
	
	def build_text_format( self, format: Optional[ Dict[ str, Any ] | str ] = None ) -> Dict[
		str, Any ]:
		"""Build text-format configuration.
		
		Purpose:
			Builds the Responses API text-format object from a supported format name or a
			complete provider-ready format dictionary.
		
		Args:
			format (Optional[Dict[str, Any] | str]): Requested response format.
		
		Returns:
			Dict[str, Any]: Provider-ready text configuration or an empty dictionary.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.requested_format = format
			self.response_format = { }
			
			if self.requested_format is None:
				return self.response_format
			
			if isinstance( self.requested_format, dict ):
				if 'format' in self.requested_format:
					self.response_format = self.requested_format
					return self.response_format
				
				if 'type' in self.requested_format:
					self.response_format = { 'format': self.requested_format, }
					return self.response_format
				
				return self.response_format
			
			if self.requested_format == 'text':
				self.response_format = { 'format': { 'type': 'text', }, }
				return self.response_format
			
			if self.requested_format == 'json_object':
				self.response_format = { 'format': { 'type': 'json_object', }, }
				return self.response_format
			
			return self.response_format
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Chat'
			ex.method = 'build_text_format( self, **kwargs| str ] = None ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def build_request( self, prompt: str, model: str, temperature: float = 0.0,
		format: Optional[ Dict[ str, Any ] | str ] = None, top_p: float = 0.0,
		frequency: float = 0.0, max_tools: int = 0, presence: float = 0.0, max_tokens: int = 0,
		store: bool = False, stream: bool = False, instruct: str = '', background: bool = False,
		reasoning: str = '', include: Optional[ List[ str ] ] = None,
		tools: Optional[ List[ str | Dict[ str, Any ] ] ] = None,
		allowed_domains: Optional[ List[ str ] ] = None, previous_id: str = '',
		tool_choice: str = '', is_parallel: bool = False,
		context: Optional[ List[ Dict[ str, Any ] ] ] = None,
		input_data: Optional[ List[ Dict[ str, Any ] ] ] = None,
		vector_store_ids: Optional[ List[ str ] ] = None, conversation_id: str = '' ) -> Dict[
		str, Any ]:
		"""Build request.
		
		Purpose:
			Builds the complete OpenAI Responses API request from values assigned to object
			members.
		
		Args:
			prompt (str): Current user prompt.
			model (str): OpenAI model identifier.
			temperature (float): Sampling temperature for supported models.
			format (Optional[Dict[str, Any] | str]): Text-format configuration.
			top_p (float): Nucleus-sampling value for supported models.
			frequency (float): Frequency penalty for supported models.
			max_tools (int): Maximum number of built-in tool calls.
			presence (float): Presence penalty for supported models.
			max_tokens (int): Maximum output-token count.
			store (bool): Indicates whether the response should be stored.
			stream (bool): Indicates whether response events should be streamed.
			instruct (str): System or developer instructions.
			background (bool): Indicates whether the response runs in background mode.
			reasoning (str): Reasoning effort.
			include (Optional[List[str]]): Additional response fields to include.
			tools (Optional[List[str | Dict[str, Any]]]): Selected provider tools.
			allowed_domains (Optional[List[str]]): Domains allowed by web search.
			previous_id (str): Previous response identifier.
			tool_choice (str): Tool-selection behavior.
			is_parallel (bool): Indicates whether parallel tool calls are permitted.
			context (Optional[List[Dict[str, Any]]]): Prior application messages.
			input_data (Optional[List[Dict[str, Any]]]): Prebuilt Responses API input items.
			vector_store_ids (Optional[List[str]]): Vector stores used by file search.
			conversation_id (str): Conversation identifier.
		
		Returns:
			Dict[str, Any]: Provider-ready Responses API request.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.prompt = prompt
			self.model = model
			self.temperature = temperature
			self.requested_format = format
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.max_tools = max_tools
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.instructions = instruct
			self.background = background
			self.reasoning_effort = reasoning
			self.include = include if include is not None else [ ]
			self.selected_tools = tools if tools is not None else [ ]
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.previous_id = previous_id
			self.tool_choice = tool_choice
			self.parallel_tools = is_parallel
			self.context = context if context is not None else [ ]
			self.input = input_data if input_data is not None else [ ]
			self.vector_store_ids = (vector_store_ids if vector_store_ids is not None else [ ])
			self.conversation_id = conversation_id
			self.reasoning = self.build_reasoning( self.reasoning_effort, self.model, )
			self.tools = self.build_tools( self.selected_tools, self.allowed_domains,
				self.vector_store_ids, )
			self.response_format = self.build_text_format( self.requested_format, )
			self.effective_context = ([ ] if self.conversation_id else self.context)
			self.input = self.build_input( self.prompt, self.effective_context, self.input, )
			self.request = { 'model': self.model, 'input': self.input, }
			
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			if self.reasoning:
				self.request[ 'reasoning' ] = self.reasoning
			
			if self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
				self.request[ 'top_p' ] = self.top_percent
				self.request[ 'frequency_penalty' ] = self.frequency_penalty
				self.request[ 'presence_penalty' ] = self.presence_penalty
			
			self.request[ 'store' ] = self.store
			self.request[ 'stream' ] = self.stream
			self.request[ 'background' ] = self.background
			
			if self.include:
				self.request[ 'include' ] = self.include
			
			if self.tools:
				self.request[ 'tools' ] = self.tools
				self.request[ 'parallel_tool_calls' ] = self.parallel_tools
				
				if self.max_tools > 0:
					self.request[ 'max_tool_calls' ] = self.max_tools
			
			if self.tool_choice:
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.previous_id:
				self.request[ 'previous_response_id' ] = self.previous_id
			
			if self.conversation_id:
				self.request[ 'conversation' ] = self.conversation_id
			
			if self.response_format:
				self.request[ 'text' ] = self.response_format
			
			return self.request
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Chat'
			ex.method = 'build_request( self, **kwargs ) -> Dict[ str, Any ]'
			Logger( ).write( ex )
			raise ex
	
	def get_output_text( self ) -> str:
		"""Get output text.
		
		Purpose:
			Extracts aggregated text from the latest synchronous or completed background response.
		
		Returns:
			str: Extracted response text or an empty string.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_text = ''
			if self.response is None:
				return self.output_text
			
			self.response_text = getattr( self.response, 'output_text', '' )
			if self.response_text:
				self.output_text = self.response_text
				return self.output_text
			
			self.text_parts = [ ]
			for item in getattr( self.response, 'output', [ ] ) or [ ]:
				if getattr( item, 'type', '' ) != 'message':
					continue
				
				for block in getattr( item, 'content', [ ] ) or [ ]:
					if getattr( block, 'type', '' ) != 'output_text':
						continue
					
					self.block_text = getattr( block, 'text', '' )
					if self.block_text:
						self.text_parts.append( self.block_text )
			
			self.output_text = ''.join( self.text_parts ).strip( )
			return self.output_text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Chat'
			ex.method = 'get_output_text( self ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def get_usage( self ) -> Any:
		"""Get response usage.
		
		Purpose:
			Returns token usage from the latest completed response.
		
		Returns:
			Any: Provider usage object or None when unavailable.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			if self.response is None:
				return None
			
			return getattr( self.response, 'usage', None )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Chat'
			ex.method = 'get_usage( self ) -> Any'
			Logger( ).write( ex )
			raise ex
	
	def generate_text( self, prompt: str, model: str, temperature: float = 0.0,
		format: Optional[ Dict[ str, Any ] | str ] = None, top_p: float = 0.0,
		frequency: float = 0.0, max_tools: int = 0, presence: float = 0.0, max_tokens: int = 0,
		store: bool = False, stream: bool = False, instruct: str = '', background: bool = False,
		reasoning: str = '', include: Optional[ List[ str ] ] = None,
		tools: Optional[ List[ str | Dict[ str, Any ] ] ] = None,
		allowed_domains: Optional[ List[ str ] ] = None, previous_id: str = '',
		tool_choice: str = '', is_parallel: bool = False,
		context: Optional[ List[ Dict[ str, Any ] ] ] = None,
		input_data: Optional[ List[ Dict[ str, Any ] ] ] = None,
		vector_store_ids: Optional[ List[ str ] ] = None, conversation_id: str = '' ) -> str:
		"""Generate text.
		
		Purpose:
			Executes a synchronous, streaming, or background OpenAI Responses API request using
			arguments assigned to wrapper members.
		
		Args:
			prompt (str): Current user prompt.
			model (str): OpenAI model identifier.
			temperature (float): Sampling temperature for supported models.
			format (Optional[Dict[str, Any] | str]): Text-format configuration.
			top_p (float): Nucleus-sampling value for supported models.
			frequency (float): Frequency penalty for supported models.
			max_tools (int): Maximum number of built-in tool calls.
			presence (float): Presence penalty for supported models.
			max_tokens (int): Maximum output-token count.
			store (bool): Indicates whether the response should be stored.
			stream (bool): Indicates whether response events should be streamed.
			instruct (str): System or developer instructions.
			background (bool): Indicates whether the response runs in background mode.
			reasoning (str): Reasoning effort.
			include (Optional[List[str]]): Additional response fields to include.
			tools (Optional[List[str | Dict[str, Any]]]): Selected provider tools.
			allowed_domains (Optional[List[str]]): Domains allowed by web search.
			previous_id (str): Previous response identifier.
			tool_choice (str): Tool-selection behavior.
			is_parallel (bool): Indicates whether parallel tool calls are permitted.
			context (Optional[List[Dict[str, Any]]]): Prior application messages.
			input_data (Optional[List[Dict[str, Any]]]): Prebuilt Responses API input items.
			vector_store_ids (Optional[List[str]]): Vector stores used by file search.
			conversation_id (str): Conversation identifier.
		
		Returns:
			str: Generated text, streamed text, or an empty string for an incomplete background
			response.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.prompt = prompt
			self.model = model
			self.temperature = temperature
			self.requested_format = format
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.max_tools = max_tools
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.instructions = instruct
			self.background = background
			self.reasoning_effort = reasoning
			self.include = include if include is not None else [ ]
			self.selected_tools = tools if tools is not None else [ ]
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			self.previous_id = previous_id
			self.tool_choice = tool_choice
			self.parallel_tools = is_parallel
			self.context = context if context is not None else [ ]
			self.input = input_data if input_data is not None else [ ]
			self.vector_store_ids = (vector_store_ids if vector_store_ids is not None else [ ])
			self.conversation_id = conversation_id
			self.client = OpenAI( api_key=self.api_key, )
			self.request = self.build_request( self.prompt, self.model, self.temperature,
				self.requested_format, self.top_percent, self.frequency_penalty, self.max_tools,
				self.presence_penalty, self.max_tokens, self.store, self.stream, self.instructions,
				self.background, self.reasoning_effort, self.include, self.selected_tools,
				self.allowed_domains, self.previous_id, self.tool_choice, self.parallel_tools,
				self.context, self.input, self.vector_store_ids, self.conversation_id, )
			
			if self.stream:
				self.stream_events = [ ]
				self.text_parts = [ ]
				self.response_stream = self.client.responses.create( **self.request )
				
				for event in self.response_stream:
					self.stream_events.append( event )
					self.event_type = getattr( event, 'type', '' )
					if self.event_type == 'response.output_text.delta':
						self.delta = getattr( event, 'delta', '' )
						if self.delta:
							self.text_parts.append( self.delta )
					
					elif self.event_type == 'response.completed':
						self.response = getattr( event, 'response', None )
				
				self.output_text = ''.join( self.text_parts ).strip( )
				if self.response is not None:
					self.previous_id = getattr( self.response, 'id', self.previous_id, )
				
				return self.output_text
			
			self.response = self.client.responses.create( **self.request )
			self.previous_id = getattr( self.response, 'id', self.previous_id, )
			self.output_text = self.get_output_text( )
			return self.output_text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Chat'
			ex.method = 'generate_text( self, **kwargs ) -> str'
			Logger( ).write( ex )
			raise ex
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the OpenAI Chat wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'client', 'model', 'prompt', 'temperature', 'top_percent',
			'frequency_penalty', 'presence_penalty', 'max_tokens', 'store', 'stream', 'background',
			'response_format', 'context', 'instructions', 'include', 'tool_choice', 'previous_id',
			'conversation_id', 'parallel_tools', 'max_tools', 'input', 'tools', 'reasoning',
			'allowed_domains', 'output_text', 'vector_store_ids', 'response', 'model_options',
			'include_options', 'tool_options', 'choice_options', 'purpose_options',
			'format_options', 'reasoning_options', 'modality_options', 'supports_reasoning_model',
			'build_reasoning', 'build_input', 'build_tools', 'build_text_format', 'build_request',
			'get_output_text', 'get_usage', 'generate_text', ]

class Images( GPT ):
	"""Provide OpenAI image workflow support.
	
	Purpose:
		Provides OpenAI image generation, image analysis, and image editing functionality.
		The class stores each accepted method argument as an object member before constructing
		and executing the corresponding Images API or Responses API request.
	
	Attributes:
		api_key (str): OpenAI API key used by the wrapper.
		client (Optional[OpenAI]): OpenAI client used by the wrapper.
		model (str): Model used by the current image operation.
		prompt (str): Prompt used by the current image operation.
		number (int): Number of images requested.
		size (str): Requested image dimensions.
		quality (str): Requested image quality.
		detail (str): Image-analysis detail level.
		background (str): Requested image background behavior.
		output_format (str): Requested image output format.
		output_compression (int): Requested image compression percentage.
		image_path (str): Local source-image path.
		mask_path (str): Local mask-image path.
		response (Any): Latest provider response.
		outputs (List[str | bytes]): Extracted image outputs.
		output_text (str): Extracted image-analysis text.
		request (Dict[str, Any]): Provider-ready request payload.
	"""
	api_key: str
	client: Optional[ OpenAI ]
	model: str
	prompt: str
	number: int
	size: str
	quality: str
	detail: str
	background: str
	output_format: str
	output_compression: int
	image_path: str
	mask_path: str
	response: Any
	outputs: List[ str | bytes ]
	output_text: str
	request: Dict[ str, Any ]
	
	def __init__( self, model: str = 'gpt-image-1-mini' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes OpenAI image-wrapper state without executing a provider request.
		
		Args:
			model (str): Default OpenAI image model.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.prompt = ''
		self.input_text = ''
		self.instructions = ''
		self.number = 1
		self.size = '1024x1024'
		self.quality = 'auto'
		self.detail = 'auto'
		self.background = 'auto'
		self.output_format = 'png'
		self.output_compression = 0
		self.image_path = ''
		self.mask_path = ''
		self.image_url = ''
		self.file = None
		self.file_id = ''
		self.response = None
		self.outputs = [ ]
		self.output_text = ''
		self.request = { }
		self.input = [ ]
		self.image_content = { }
		self.max_tokens = 0
		self.temperature = 0.0
		self.store = False
		self.stream = False
		self.include = [ ]
		self.data = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get generation-model options.
		
		Purpose:
			Returns the OpenAI image-generation models exposed by the wrapper.
		
		Returns:
			List[str]: Supported image-generation model identifiers.
		"""
		return [ 'gpt-image-2', 'gpt-image-1.5', 'gpt-image-1', 'gpt-image-1-mini', ]
	
	@property
	def analysis_model_options( self ) -> List[ str ]:
		"""Get analysis-model options.
		
		Purpose:
			Returns vision-capable text models exposed for image analysis.
		
		Returns:
			List[str]: Supported image-analysis model identifiers.
		"""
		return [ 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5', 'gpt-5-mini', 'gpt-4.1', 'gpt-4.1-mini',
			'gpt-4o', 'gpt-4o-mini', ]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Get image-size options.
		
		Purpose:
			Returns image sizes exposed for OpenAI image generation and editing.
		
		Returns:
			List[str]: Supported image-size values.
		"""
		return [ 'auto', '1024x1024', '1024x1536', '1536x1024', ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		"""Get image-quality options.
		
		Purpose:
			Returns image-quality values exposed by the wrapper.
		
		Returns:
			List[str]: Supported image-quality values.
		"""
		return [ 'auto', 'low', 'medium', 'high', ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		"""Get image-detail options.
		
		Purpose:
			Returns detail levels supported by image-analysis requests.
		
		Returns:
			List[str]: Supported image-analysis detail values.
		"""
		return [ 'auto', 'low', 'high', 'original', ]
	
	@property
	def backcolor_options( self ) -> List[ str ]:
		"""Get background options.
		
		Purpose:
			Returns background values exposed for image generation and editing.
		
		Returns:
			List[str]: Supported background values.
		"""
		return [ 'auto', 'transparent', 'opaque', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get output-format options.
		
		Purpose:
			Returns image output formats supported by the wrapper.
		
		Returns:
			List[str]: Supported output-format values.
		"""
		return [ 'png', 'jpeg', 'webp', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get MIME-format options.
		
		Purpose:
			Returns image format values used by the application selector.
		
		Returns:
			List[str]: Supported image format values.
		"""
		return [ 'png', 'jpeg', 'webp', ]
	
	@property
	def style_options( self ) -> List[ str ]:
		"""Get style options.
		
		Purpose:
			Returns legacy image-style options retained for application compatibility.
		
		Returns:
			List[str]: Available style values.
		"""
		return [ 'vivid', 'natural', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Get analysis-include options.
		
		Purpose:
			Returns additional response fields supported by image-analysis requests.
		
		Returns:
			List[str]: Supported Responses API include values.
		"""
		return [ 'message.input_image.image_url', 'message.output_text.logprobs', ]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Get image-tool options.
		
		Purpose:
			Returns tools applicable to the current OpenAI image wrapper.
		
		Returns:
			List[str]: Supported image-related tool names.
		"""
		return [ 'image_generation', ]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Get tool-choice options.
		
		Purpose:
			Returns tool-choice values retained for application selector compatibility.
		
		Returns:
			List[str]: Supported tool-choice values.
		"""
		return [ 'auto', 'required', 'none', ]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Get reasoning options.
		
		Purpose:
			Returns reasoning-effort options exposed for vision-capable reasoning models.
		
		Returns:
			List[str]: Supported reasoning-effort values.
		"""
		return [ 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', ]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Get modality options.
		
		Purpose:
			Returns modalities produced or consumed by image workflows.
		
		Returns:
			List[str]: Supported modality values.
		"""
		return [ 'text', 'image', ]
	
	def supports_original_detail( self, model: str ) -> bool:
		"""Determine original-detail support.
		
		Purpose:
			Determines whether a selected image-analysis model supports the original image-detail
			setting.
		
		Args:
			model (str): Image-analysis model identifier.
		
		Returns:
			bool: True when the selected model supports original detail.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'model', model )
			self.model = model
			return self.model.startswith( 'gpt-5.4' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'supports_original_detail( self, model: str ) -> bool'
			Logger( ).write( exception )
			raise exception
	
	def get_analysis_detail( self, detail: str, model: str ) -> str:
		"""Get effective analysis detail.
		
		Purpose:
			Returns the image-detail value permitted by the selected analysis model.
		
		Args:
			detail (str): Requested image-detail level.
			model (str): Image-analysis model identifier.
		
		Returns:
			str: Effective image-detail value.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'detail', detail )
			throw_if( 'model', model )
			self.detail = detail
			self.model = model
			if self.detail not in self.detail_options:
				self.detail = 'auto'
			
			if self.detail == 'original':
				if not self.supports_original_detail( self.model ):
					self.detail = 'high'
			
			return self.detail
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'get_analysis_detail( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_compression( self, compression: float, output_format: str ) -> int:
		"""Get effective output compression.
		
		Purpose:
			Returns an integer compression percentage for JPEG and WebP image output.
		
		Args:
			compression (float): Requested compression percentage.
			output_format (str): Requested image output format.
		
		Returns:
			int: Effective compression percentage or zero when compression is not applicable.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.compression = compression
			self.output_format = output_format
			self.output_compression = 0
			if self.output_format not in [ 'jpeg', 'webp' ]:
				return self.output_compression
			
			if self.compression <= 0:
				return self.output_compression
			
			if self.compression > 100:
				self.compression = 100
			
			self.output_compression = int( self.compression )
			return self.output_compression
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'get_output_compression( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def extract_image_outputs( self ) -> str | bytes | List[ str | bytes ] | None:
		"""Extract image outputs.
		
		Purpose:
			Extracts URLs or decoded base64 image bytes from the latest Images API response.
		
		Returns:
			str | bytes | List[str | bytes] | None: Extracted image output or outputs.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.outputs = [ ]
			if self.response is None:
				return None
			
			self.data = getattr( self.response, 'data', None )
			if not self.data:
				return None
			
			for item in self.data:
				self.image_url = getattr( item, 'url', '' )
				self.image_base64 = getattr( item, 'b64_json', '' )
				
				if self.image_url:
					self.outputs.append( self.image_url )
					continue
				
				if self.image_base64:
					self.outputs.append( base64.b64decode( self.image_base64 ) )
			
			if len( self.outputs ) == 0:
				return None
			
			if len( self.outputs ) == 1:
				return self.outputs[ 0 ]
			
			return self.outputs
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'extract_image_outputs( self )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str:
		"""Get analysis output text.
		
		Purpose:
			Extracts text from the latest image-analysis Responses API response.
		
		Returns:
			str: Extracted image-analysis text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.output_text = ''
			if self.response is None:
				return self.output_text
			
			self.response_text = getattr( self.response, 'output_text', '', )
			if self.response_text:
				self.output_text = self.response_text
				return self.output_text
			
			self.text_parts = [ ]
			for item in getattr( self.response, 'output', [ ] ) or [ ]:
				if getattr( item, 'type', '' ) != 'message':
					continue
				
				for block in getattr( item, 'content', [ ] ) or [ ]:
					if getattr( block, 'type', '' ) != 'output_text':
						continue
					
					self.block_text = getattr( block, 'text', '' )
					if self.block_text:
						self.text_parts.append( self.block_text )
			
			self.output_text = ''.join( self.text_parts ).strip( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'get_output_text( self ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, model: str, number: int = 1, size: str = '1024x1024',
		quality: str = 'auto', fmt: str = 'png', compression: float = 0.0,
		background: str = 'auto' ) -> str | bytes | List[ str | bytes ] | None:
		"""Generate images.
		
		Purpose:
			Generates one or more images from a required text prompt using the selected OpenAI
			image model.
		
		Args:
			prompt (str): Required image-generation prompt.
			model (str): Required OpenAI image model.
			number (int): Number of images requested.
			size (str): Requested image dimensions.
			quality (str): Requested image quality.
			fmt (str): Requested output format.
			compression (float): Requested JPEG or WebP compression percentage.
			background (str): Requested background behavior.
		
		Returns:
			str | bytes | List[str | bytes] | None: Generated image output or outputs.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'size', size )
			throw_if( 'quality', quality )
			throw_if( 'fmt', fmt )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.prompt = prompt
			self.model = model
			self.number = number
			self.size = size
			self.quality = quality
			self.output_format = fmt
			self.compression = compression
			self.background = background
			self.output_compression = self.get_output_compression( self.compression,
				self.output_format, )
			
			if self.number <= 0:
				self.number = 1
			
			if self.number > 10:
				self.number = 10
			
			if self.model == 'gpt-image-2':
				if self.background == 'transparent':
					self.background = 'auto'
			
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'model': self.model, 'prompt': self.prompt, 'n': self.number,
				'size': self.size, 'quality': self.quality, 'output_format': self.output_format, }
			
			if self.background:
				self.request[ 'background' ] = self.background
			
			if self.output_compression > 0:
				self.request[ 'output_compression' ] = self.output_compression
			
			self.response = self.client.images.generate( **self.request )
			return self.extract_image_outputs( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'generate( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, text: str, path: str, model: str, instruct: str = '', max_tokens: int = 0,
		temperature: float = 0.0, include: Optional[ List[ str ] ] = None, store: bool = False,
		stream: bool = False, detail: str = 'auto' ) -> str:
		"""Analyze an image.
		
		Purpose:
			Uploads a required local image and analyzes it with a required vision-capable model
			through the OpenAI Responses API.
		
		Args:
			text (str): Required question or instruction for image analysis.
			path (str): Required local image path.
			model (str): Required vision-capable OpenAI model.
			instruct (str): Optional system or developer instructions.
			max_tokens (int): Maximum output-token count.
			temperature (float): Sampling temperature for supported models.
			include (Optional[List[str]]): Additional response fields to include.
			store (bool): Indicates whether the response should be stored.
			stream (bool): Indicates whether response events should be streamed.
			detail (str): Requested image-detail level.
		
		Returns:
			str: Image-analysis response text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.input_text = text
			self.image_path = path
			self.model = model
			self.instructions = instruct
			self.max_tokens = max_tokens
			self.temperature = temperature
			self.include = include if include is not None else [ ]
			self.store = store
			self.stream = stream
			self.detail = self.get_analysis_detail( detail, self.model, )
			self.client = OpenAI( api_key=self.api_key, )
			
			with open( self.image_path, 'rb' ) as source:
				self.file = self.client.files.create( file=source, purpose='vision', )
			
			self.file_id = self.file.id
			self.image_content = { 'type': 'input_image', 'file_id': self.file_id,
				'detail': self.detail, }
			self.input = [ { 'role': 'user',
				'content': [ { 'type': 'input_text', 'text': self.input_text, },
					self.image_content, ], }, ]
			self.request = { 'model': self.model, 'input': self.input, 'store': self.store,
				'stream': self.stream, }
			
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			if self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
			
			if self.include:
				self.request[ 'include' ] = self.include
			
			self.response = self.client.responses.create( **self.request )
			if self.stream:
				self.text_parts = [ ]
				for event in self.response:
					self.event_type = getattr( event, 'type', '' )
					if self.event_type == 'response.output_text.delta':
						self.delta = getattr( event, 'delta', '' )
						if self.delta:
							self.text_parts.append( self.delta )
					
					elif self.event_type == 'response.completed':
						self.response = getattr( event, 'response', None )
				
				self.output_text = ''.join( self.text_parts ).strip( )
				return self.output_text
			
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'analyze( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str, number: int=1, size: str='1024x1024',
		quality: str='auto', fmt: str='png', compression: float=0.0, background: str='auto',
		mask_path: str = '' ) -> str | bytes | List[ str | bytes ] | None:
		"""Edit an image.
		
		Purpose:
			Edits a required local image using a required prompt and OpenAI image model.
		
		Args:
			prompt (str): Required image-editing instruction.
			path (str): Required local source-image path.
			model (str): Required OpenAI image model.
			number (int): Number of edited images requested.
			size (str): Requested image dimensions.
			quality (str): Requested image quality.
			fmt (str): Requested output format.
			compression (float): Requested JPEG or WebP compression percentage.
			background (str): Requested background behavior.
			mask_path (str): Optional local image-mask path.
		
		Returns:
			str | bytes | List[str | bytes] | None: Edited image output or outputs.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'size', size )
			throw_if( 'quality', quality )
			throw_if( 'fmt', fmt )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.prompt = prompt
			self.image_path = path
			self.model = model
			self.number = number
			self.size = size
			self.quality = quality
			self.output_format = fmt
			self.compression = compression
			self.background = background
			self.mask_path = mask_path
			self.output_compression = self.get_output_compression( self.compression,
				self.output_format, )
			
			if self.number <= 0:
				self.number = 1
			
			if self.number > 10:
				self.number = 10
			
			if self.model == 'gpt-image-2':
				if self.background == 'transparent':
					self.background = 'auto'
			
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'model': self.model, 'prompt': self.prompt, 'n': self.number,
				'size': self.size, 'quality': self.quality, 'output_format': self.output_format, }
			
			if self.background:
				self.request[ 'background' ] = self.background
			
			if self.output_compression > 0:
				self.request[ 'output_compression' ] = self.output_compression
			
			with open( self.image_path, 'rb' ) as source:
				if self.mask_path:
					with open( self.mask_path, 'rb' ) as mask:
						self.response = self.client.images.edit( image=source, mask=mask,
							**self.request )
				else:
					self.response = self.client.images.edit( image=source, **self.request )
			
			return self.extract_image_outputs( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'edit( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the OpenAI Images wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'client', 'model', 'prompt', 'input_text', 'instructions', 'number',
			'size', 'quality', 'detail', 'background', 'output_format', 'output_compression',
			'image_path', 'mask_path', 'image_url', 'file', 'file_id', 'response', 'outputs',
			'output_text', 'request', 'input', 'image_content', 'max_tokens', 'temperature',
			'store', 'stream', 'include', 'model_options', 'analysis_model_options',
			'size_options',
			'quality_options', 'detail_options', 'backcolor_options', 'format_options',
			'mime_options', 'style_options', 'include_options', 'tool_options', 'choice_options',
			'reasoning_options', 'modality_options', 'supports_original_detail',
			'get_analysis_detail', 'get_output_compression', 'extract_image_outputs',
			'get_output_text', 'generate', 'analyze', 'edit', ]

class TTS( GPT ):
	"""Provide OpenAI text-to-speech workflow support.
	
	Purpose:
		Provides text-to-speech generation through the OpenAI Audio Speech API. The class
		stores speech request arguments as object members, creates provider-ready requests from
		those members, streams generated audio to a temporary file, returns the resulting audio
		bytes, and optionally writes the audio to a caller-specified output path.
	
	Attributes:
		api_key (str): OpenAI API key used by the wrapper.
		client (Optional[OpenAI]): OpenAI client used by the wrapper.
		model (str): Text-to-speech model used by the current request.
		input (str): Text converted to speech.
		voice (str): Voice used to generate speech.
		response_format (str): Audio format returned by the provider.
		speed (float): Speech playback speed.
		instructions (str): Model instructions controlling speech delivery.
		file_path (str): Optional output path used to persist generated audio.
		response (Any): Latest streaming speech response.
		audio_bytes (bytes): Audio bytes produced by the latest request.
		request (Dict[str, Any]): Provider-ready speech request.
		temp_path (str): Temporary audio file path used during response streaming.
	"""
	api_key: str
	client: Optional[ OpenAI ]
	model: str
	input: str
	voice: str
	response_format: str
	speed: float
	instructions: str
	file_path: str
	response: Any
	audio_bytes: bytes
	request: Dict[ str, Any ]
	temp_path: str
	
	def __init__( self, model: str='gpt-4o-mini-tts', format: str = 'mp3', voice: str = 'alloy',
		speed: float = 1.0 ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes OpenAI text-to-speech configuration and request state without executing
			a provider request.
		
		Args:
			model (str): Default OpenAI text-to-speech model.
			format (str): Default audio response format.
			voice (str): Default speech voice.
			speed (float): Default speech speed.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.input = ''
		self.voice = voice
		self.response_format = format
		self.speed = speed
		self.instructions = ''
		self.file_path = ''
		self.response = None
		self.audio_bytes = b''
		self.request = { }
		self.temp_path = ''
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get text-to-speech model options.
		
		Purpose:
			Returns OpenAI models supported by the Audio Speech API workflow.
		
		Returns:
			List[str]: Supported text-to-speech model identifiers.
		"""
		return [ 'gpt-4o-mini-tts', 'gpt-4o-mini-tts-2025-12-15', 'tts-1', 'tts-1-hd', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get audio-format options.
		
		Purpose:
			Returns audio response formats supported by the OpenAI Speech API.
		
		Returns:
			List[str]: Supported audio response-format values.
		"""
		return [ 'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get audio-format options.
		
		Purpose:
			Returns audio response formats supported by the OpenAI Speech API.
		
		Returns:
			List[str]: Supported audio response-format values.
		"""
		return self.mime_options
	
	@property
	def voice_options( self ) -> List[ str ]:
		"""Get speech-voice options.
		
		Purpose:
			Returns built-in OpenAI voices exposed by the text-to-speech wrapper.
		
		Returns:
			List[str]: Supported built-in voice identifiers.
		"""
		return [ 'alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage',
			'shimmer', 'verse', 'marin', 'cedar', ]
	
	@property
	def speed_options( self ) -> List[ float ]:
		"""Get speech-speed options.
		
		Purpose:
			Returns speech-speed values exposed by the text-to-speech wrapper.
		
		Returns:
			List[float]: Supported speech-speed selections.
		"""
		return [ 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0, 4.0, ]
	
	def create_speech( self, text: str, model: str = 'gpt-4o-mini-tts', format: str = 'mp3',
		speed: float = 1.0, voice: str = 'alloy', instruct: str = '',
		file_path: str = '' ) -> bytes:
		"""Create speech.
		
		Purpose:
			Generates speech audio from required input text using the selected OpenAI speech
			model, voice, format, speed, and optional delivery instructions. The method streams
			the provider response to a temporary file, reads the generated audio bytes, and
			optionally persists those bytes to a caller-specified path.
		
		Args:
			text (str): Required text converted to speech.
			model (str): OpenAI text-to-speech model.
			format (str): Audio response format.
			speed (float): Speech playback speed.
			voice (str): Voice used to generate speech.
			instruct (str): Optional instructions controlling speech delivery.
			file_path (str): Optional destination path for generated audio.
		
		Returns:
			bytes: Generated speech audio.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'model', model )
			throw_if( 'format', format )
			throw_if( 'voice', voice )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.input = text
			self.model = model
			self.response_format = format
			self.speed = speed
			self.voice = voice
			self.instructions = instruct
			self.file_path = file_path
			self.client = OpenAI( api_key=self.api_key, )
			self.response = None
			self.audio_bytes = b''
			self.request = { 'model': self.model, 'input': self.input, 'voice': self.voice,
				'response_format': self.response_format, 'speed': self.speed, }
			
			if self.instructions:
				if self.model not in [ 'tts-1', 'tts-1-hd', ]:
					self.request[ 'instructions' ] = self.instructions
			
			with tempfile.NamedTemporaryFile( suffix=f'.{self.response_format}',
					delete=False, ) as temporary_file:
				self.temp_path = temporary_file.name
			
			try:
				with self.client.audio.speech.with_streaming_response.create(
						**self.request ) as response:
					self.response = response
					self.response.stream_to_file( self.temp_path, )
				
				with open( self.temp_path, 'rb' ) as source:
					self.audio_bytes = source.read( )
				
				throw_if( 'audio_bytes', self.audio_bytes )
				
				if self.file_path:
					with open( self.file_path, 'wb' ) as target:
						target.write( self.audio_bytes )
				
				return self.audio_bytes
			finally:
				if self.temp_path:
					if os.path.exists( self.temp_path ):
						os.remove( self.temp_path )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the OpenAI text-to-speech wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'client', 'model', 'input', 'voice', 'response_format', 'speed',
			'instructions', 'file_path', 'response', 'audio_bytes', 'request', 'temp_path',
			'model_options', 'mime_options', 'format_options', 'voice_options', 'speed_options',
			'create_speech', ]

class Transcription( GPT ):
	"""Provide OpenAI audio-transcription workflow support.
	
	Purpose:
		Provides audio transcription through the OpenAI Audio Transcriptions API. The class
		stores each accepted transcription argument as an object member before constructing and
		executing the provider request. It supports plain-text, JSON, verbose JSON, subtitle,
		and diarized transcription responses where supported by the selected model.
	
	Attributes:
		api_key (str): OpenAI API key used by the wrapper.
		client (Optional[OpenAI]): OpenAI client used by the wrapper.
		model (str): Transcription model used by the current request.
		audio_file (str): Local audio-file path used by the current request.
		language (str): Optional ISO-639-1 source-language hint.
		prompt (str): Optional transcription prompt.
		response_format (str): Requested transcription response format.
		temperature (float): Sampling temperature used by the transcription request.
		include (List[str]): Additional transcription response fields.
		timestamp_granularities (List[str]): Requested timestamp granularities.
		chunking_strategy (str): Diarization chunking strategy.
		response (Any): Latest provider transcription response.
		transcript (str): Text extracted from the latest response.
		result (Dict[str, Any]): Structured transcription result.
		request (Dict[str, Any]): Provider-ready transcription request.
	"""
	api_key: str
	client: Optional[ OpenAI ]
	model: str
	audio_file: str
	language: str
	prompt: str
	response_format: str
	temperature: float
	include: List[ str ]
	timestamp_granularities: List[ str ]
	chunking_strategy: str
	response: Any
	transcript: str
	result: Dict[ str, Any ]
	request: Dict[ str, Any ]
	
	def __init__( self, model: str = 'gpt-4o-transcribe', format: str = 'json',
		temperature: float = 0.0 ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes OpenAI transcription configuration and runtime state without executing
			a provider request.
		
		Args:
			model (str): Default OpenAI transcription model.
			format (str): Default transcription response format.
			temperature (float): Default transcription sampling temperature.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.audio_file = ''
		self.language = ''
		self.prompt = ''
		self.response_format = format
		self.temperature = temperature
		self.include = [ ]
		self.timestamp_granularities = [ ]
		self.chunking_strategy = 'auto'
		self.response = None
		self.transcript = ''
		self.result = { }
		self.request = { }
		self.segments = [ ]
		self.words = [ ]
		self.speakers = [ ]
		self.duration = 0.0
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get transcription-model options.
		
		Purpose:
			Returns OpenAI models implemented by the audio-transcription wrapper.
		
		Returns:
			List[str]: Supported transcription model identifiers.
		"""
		return [ 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe',
			'gpt-4o-mini-transcribe-2025-12-15',
			'gpt-4o-transcribe-diarize', 'whisper-1', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get supported audio-file extensions.
		
		Purpose:
			Returns audio formats accepted by the OpenAI transcription workflow.
		
		Returns:
			List[str]: Supported audio-file extensions.
		"""
		return [ 'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm', ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Get language options.
		
		Purpose:
			Returns ISO-639-1 source-language hints exposed by the transcription wrapper.
		
		Returns:
			List[str]: Supported source-language selections.
		"""
		return [ '', 'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'uk', 'tr', 'ar', 'hi',
			'ja', 'ko', 'zh', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get transcription-format options.
		
		Purpose:
			Returns transcription response formats supported by the wrapper.
		
		Returns:
			List[str]: Supported transcription response formats.
		"""
		return [ 'json', 'text', 'verbose_json', 'srt', 'vtt', 'diarized_json', ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Get transcription-include options.
		
		Purpose:
			Returns additional response fields supported by compatible transcription models.
		
		Returns:
			List[str]: Supported transcription include values.
		"""
		return [ 'logprobs', ]
	
	@property
	def timestamp_options( self ) -> List[ str ]:
		"""Get timestamp-granularity options.
		
		Purpose:
			Returns timestamp granularities supported by verbose Whisper transcription output.
		
		Returns:
			List[str]: Supported timestamp-granularity values.
		"""
		return [ 'word', 'segment', ]
	
	def build_result( self, response: Any ) -> Dict[ str, Any ]:
		"""Build transcription result.
		
		Purpose:
			Extracts text, language, duration, segment, word, and speaker information from the
			provider response and stores the resulting application-facing transcription record.
		
		Args:
			response (Any): Provider transcription response.
		
		Returns:
			Dict[str, Any]: Structured transcription result.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response', response )
			self.response = response
			self.transcript = ''
			self.segments = [ ]
			self.words = [ ]
			self.speakers = [ ]
			self.duration = 0.0
			self.result = { 'text': '', 'language': '', 'duration': 0.0, 'segments': [ ],
				'words': [ ], 'speakers': [ ], 'raw': None, }
			
			if isinstance( self.response, str ):
				self.transcript = self.response
				self.result[ 'text' ] = self.transcript
				self.result[ 'raw' ] = self.response
				return self.result
			
			self.transcript = getattr( self.response, 'text', '', )
			self.language = getattr( self.response, 'language', self.language, )
			self.duration = getattr( self.response, 'duration', 0.0, )
			self.response_segments = getattr( self.response, 'segments', [ ], ) or [ ]
			self.response_words = getattr( self.response, 'words', [ ], ) or [ ]
			
			for segment in self.response_segments:
				if hasattr( segment, 'model_dump' ):
					self.segments.append( segment.model_dump( ) )
				elif isinstance( segment, dict ):
					self.segments.append( segment )
				else:
					self.segments.append( { 'text': str( segment ), } )
			
			for word in self.response_words:
				if hasattr( word, 'model_dump' ):
					self.words.append( word.model_dump( ) )
				elif isinstance( word, dict ):
					self.words.append( word )
				else:
					self.words.append( { 'word': str( word ), } )
			
			for segment in self.segments:
				if not isinstance( segment, dict ):
					continue
				
				self.speaker = segment.get( 'speaker', '' )
				
				if self.speaker:
					if self.speaker not in self.speakers:
						self.speakers.append( self.speaker )
			
			if not self.transcript:
				self.text_parts = [ ]
				
				for segment in self.segments:
					if not isinstance( segment, dict ):
						continue
					
					self.segment_text = segment.get( 'text', '' )
					
					if self.segment_text:
						self.text_parts.append( self.segment_text )
				
				self.transcript = '\n'.join( self.text_parts ).strip( )
			
			if hasattr( self.response, 'model_dump' ):
				self.raw_response = self.response.model_dump( )
			else:
				self.raw_response = str( self.response )
			
			self.result = { 'text': self.transcript, 'language': self.language,
				'duration': self.duration, 'segments': self.segments, 'words': self.words,
				'speakers': self.speakers, 'raw': self.raw_response, }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'build_result( self, response: Any )'
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, path: str, model: str = 'gpt-4o-transcribe', language: str = '',
		prompt: str = '', format: str = 'json', temperature: float = 0.0,
		include: Optional[ List[ str ] ] = None,
		timestamp_granularities: Optional[ List[ str ] ] = None,
		chunking_strategy: str = 'auto' ) -> str:
		"""Transcribe audio.
		
		Purpose:
			Transcribes a required local audio file through the OpenAI Audio Transcriptions API
			using the selected model, source-language hint, prompt, response format, temperature,
			include fields, timestamp granularities, and diarization chunking strategy.
		
		Args:
			path (str): Required local audio-file path.
			model (str): OpenAI transcription model.
			language (str): Optional ISO-639-1 source-language hint.
			prompt (str): Optional transcription prompt.
			format (str): Transcription response format.
			temperature (float): Transcription sampling temperature.
			include (Optional[List[str]]): Additional response fields.
			timestamp_granularities (Optional[List[str]]): Requested timestamp granularities.
			chunking_strategy (str): Diarization chunking strategy.
		
		Returns:
			str: Extracted transcript text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'format', format )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.audio_file = path
			self.model = model
			self.language = language
			self.prompt = prompt
			self.response_format = format
			self.temperature = temperature
			self.include = include if include is not None else [ ]
			self.timestamp_granularities = (
				timestamp_granularities if timestamp_granularities is not None else [ ])
			self.chunking_strategy = chunking_strategy
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'model': self.model, 'response_format': self.response_format,
				'temperature': self.temperature, }
			
			if self.language:
				self.request[ 'language' ] = self.language
			
			if self.prompt:
				self.request[ 'prompt' ] = self.prompt
			
			if self.include:
				if self.model != 'whisper-1':
					self.request[ 'include' ] = self.include
			
			if self.timestamp_granularities:
				if self.model == 'whisper-1':
					if self.response_format == 'verbose_json':
						self.request[ 'timestamp_granularities' ] = (self.timestamp_granularities)
			
			if self.model == 'gpt-4o-transcribe-diarize':
				self.response_format = 'diarized_json'
				self.request[ 'response_format' ] = self.response_format
				self.request[ 'chunking_strategy' ] = self.chunking_strategy
			
			with open( self.audio_file, 'rb' ) as source:
				self.response = self.client.audio.transcriptions.create( file=source,
					**self.request )
			
			self.result = self.build_result( self.response )
			self.transcript = self.result.get( 'text', '', )
			return self.transcript
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'transcribe( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the OpenAI transcription wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'client', 'model', 'audio_file', 'language', 'prompt',
			'response_format', 'temperature', 'include', 'timestamp_granularities',
			'chunking_strategy', 'response', 'transcript', 'result', 'request', 'segments',
			'words',
			'speakers', 'duration', 'model_options', 'mime_options', 'language_options',
			'format_options', 'include_options', 'timestamp_options', 'build_result',
			'transcribe', ]

class Translation( GPT ):
	"""Provide OpenAI audio-translation workflow support.
	
	Purpose:
		Provides audio translation through the OpenAI Audio Translations API. The class stores
		each accepted translation argument as an object member before constructing and executing
		the provider request. OpenAI audio translation converts supported spoken audio into
		English.
	
	Attributes:
		api_key (str): OpenAI API key used by the wrapper.
		client (Optional[OpenAI]): OpenAI client used by the wrapper.
		model (str): Translation model used by the current request.
		audio_file (str): Local audio-file path used by the current request.
		prompt (str): Optional English prompt used to guide translation.
		response_format (str): Requested translation response format.
		temperature (float): Sampling temperature used by the translation request.
		response (Any): Latest provider translation response.
		translation (str): English text extracted from the latest response.
		result (Dict[str, Any]): Structured translation result.
		request (Dict[str, Any]): Provider-ready translation request.
	"""
	api_key: str
	client: Optional[ OpenAI ]
	model: str
	audio_file: str
	prompt: str
	response_format: str
	temperature: float
	response: Any
	translation: str
	result: Dict[ str, Any ]
	request: Dict[ str, Any ]
	
	def __init__( self, model: str = 'whisper-1',
		format: str = 'json', temperature: float = 0.0 ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes OpenAI audio-translation configuration and runtime state without
			executing a provider request.
		
		Args:
			model (str): Default OpenAI audio-translation model.
			format (str): Default translation response format.
			temperature (float): Default translation sampling temperature.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.audio_file = ''
		self.prompt = ''
		self.response_format = format
		self.temperature = temperature
		self.response = None
		self.translation = ''
		self.result = { }
		self.request = { }
		self.segments = [ ]
		self.language = 'English'
		self.duration = 0.0
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get translation-model options.
		
		Purpose:
			Returns OpenAI models supported by the Audio Translations API.
		
		Returns:
			List[str]: Supported translation model identifiers.
		"""
		return [ 'whisper-1', ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get supported audio-file extensions.
		
		Purpose:
			Returns audio formats accepted by the OpenAI Audio Translations API.
		
		Returns:
			List[str]: Supported audio-file extensions.
		"""
		return [ 'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm', ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get translation-format options.
		
		Purpose:
			Returns response formats supported by the OpenAI Audio Translations API.
		
		Returns:
			List[str]: Supported translation response formats.
		"""
		return [ 'json', 'text', 'srt', 'verbose_json', 'vtt', ]
	
	@property
	def language_options( self ) -> List[ str ]:
		"""Get target-language options.
		
		Purpose:
			Returns the only target language supported by the OpenAI Audio Translations API.
		
		Returns:
			List[str]: Supported target-language values.
		"""
		return [ 'English', ]
	
	def translate( self, path: str, model: str = 'whisper-1',
		prompt: str = '', format: str = 'json',
		temperature: float = 0.0 ) -> str:
		"""Translate audio.
		
		Purpose:
			Translates a required local audio file into English through the OpenAI Audio
			Translations API using the selected model, optional English prompt, response format,
			and sampling temperature.
		
		Args:
			path (str): Required local audio-file path.
			model (str): OpenAI audio-translation model.
			prompt (str): Optional English prompt used to guide translation.
			format (str): Translation response format.
			temperature (float): Translation sampling temperature.
		
		Returns:
			str: English translation extracted from the provider response.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'format', format )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.audio_file = path
			self.model = model
			self.prompt = prompt
			self.response_format = format
			self.temperature = temperature
			self.client = OpenAI( api_key=self.api_key, )
			self.response = None
			self.translation = ''
			self.result = { }
			self.segments = [ ]
			self.duration = 0.0
			self.request = {
				'model': self.model,
				'response_format': self.response_format,
				'temperature': self.temperature,
			}
			
			if self.prompt:
				self.request[ 'prompt' ] = self.prompt
			
			with open( self.audio_file, 'rb' ) as source:
				self.response = self.client.audio.translations.create( file=source,
					**self.request )
			
			throw_if( 'response', self.response )
			if isinstance( self.response, str ):
				self.translation = self.response
				self.result = { 'text': self.translation, 'language': self.language,
					'duration': self.duration, 'segments': self.segments, 'raw': self.response, }
				return self.translation
			
			self.translation = getattr( self.response, 'text', '', )
			self.duration = getattr( self.response, 'duration', 0.0, )
			self.response_segments = getattr( self.response, 'segments', [ ], ) or [ ]
			
			for segment in self.response_segments:
				if hasattr( segment, 'model_dump' ):
					self.segments.append( segment.model_dump( ) )
				elif isinstance( segment, dict ):
					self.segments.append( segment )
				else:
					self.segments.append( { 'text': str( segment ), } )
			
			if not self.translation:
				self.text_parts = [ ]
				for segment in self.segments:
					if not isinstance( segment, dict ):
						continue
					
					self.segment_text = segment.get( 'text', '', )
					if self.segment_text:
						self.text_parts.append( self.segment_text )
				
				self.translation = '\n'.join( self.text_parts ).strip( )
			
			throw_if( 'translation', self.translation )
			if hasattr( self.response, 'model_dump' ):
				self.raw_response = self.response.model_dump( )
			else:
				self.raw_response = str( self.response )
			
			self.result = { 'text': self.translation, 'language': self.language,
				'duration': self.duration, 'segments': self.segments, 'raw': self.raw_response, }
			return self.translation
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Translation'
			exception.method = 'translate( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the OpenAI audio-translation wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'client', 'model', 'audio_file', 'prompt', 'response_format',
			'temperature', 'response', 'translation', 'result', 'request', 'segments', 'language',
			'duration', 'model_options', 'mime_options', 'format_options', 'language_options',
			'translate', ]

class Embeddings( GPT ):
	"""Provide Embeddings workflow support.
	
	Purpose:
		Provides OpenAI embedding generation for text inputs. The class manages embedding model
		selection, encoding format, optional dimensions, usage metadata, and normalized single
		or batch embedding output.
	
	Attributes:
		api_key (Optional[str]): Api key retained by the provider wrapper.
		client (Optional[OpenAI]): Client retained by the provider wrapper.
		model (Optional[str]): Model retained by the provider wrapper.
		input (Optional[str | List[str]]): Input retained by the provider wrapper.
		encoding_format (Optional[str]): Encoding format retained by the provider wrapper.
		dimensions (Optional[int]): Dimensions retained by the provider wrapper.
		user (Optional[str]): User retained by the provider wrapper.
		response (Optional[CreateEmbeddingResponse]): Response retained by the provider wrapper.
		embedding (Optional[List[float] | str]): Embedding retained by the provider wrapper.
		embeddings (Optional[List[List[float]] | List[str]]): Embeddings retained by the provider
			wrapper.
		usage (Optional[Any]): Usage retained by the provider wrapper.
		request (Optional[Dict[str, Any]]): Request retained by the provider wrapper.
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
	
	def __init__( self, text: str | List[ str ] = None, model: str='text-embedding-3-small',
		format: str='float', dimensions: int=None, user: str=None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Embeddings object with default configuration, runtime state, provider
			settings, and compatibility fields. This constructor prepares the instance for later
			method calls without performing external work beyond local attribute assignment.
		
		Args:
			text (str | List[str]): Text value used by the operation.
			model (str): Model value used by the operation.
			format (str): Format value used by the operation.
			dimensions (int): Dimensions value used by the operation.
			user (str): User value used by the operation.
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
		"""Get model options.
		
		Purpose:
			Returns the model options exposed by the Embeddings wrapper. The property
			centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002', ]
	
	@property
	def encoding_options( self ) -> List[ str ] | None:
		"""Get encoding options.
		
		Purpose:
			Returns the encoding options exposed by the Embeddings wrapper. The property
			centralizes
			UI option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'float', 'base64', ]
	
	@property
	def model_default_dimensions( self ) -> Dict[ str, int ]:
		"""Get model default dimensions.
		
		Purpose:
			Returns the model default dimensions exposed by the Embeddings wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return { 'text-embedding-3-small': 1536, 'text-embedding-3-large': 3072,
			'text-embedding-ada-002': 1536, }
	
	@property
	def model_max_dimensions( self ) -> Dict[ str, int ]:
		"""Get model max dimensions.
		
		Purpose:
			Returns the model max dimensions exposed by the Embeddings wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return { 'text-embedding-3-small': 1536, 'text-embedding-3-large': 3072,
			'text-embedding-ada-002': 1536, }
	
	@property
	def model_dimension_support( self ) -> Dict[ str, bool ]:
		"""Get model dimension support.
		
		Purpose:
			Returns the model dimension support exposed by the Embeddings wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return { 'text-embedding-3-small': True, 'text-embedding-3-large': True,
			'text-embedding-ada-002': False, }
	
	def validate_input( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""Validate input.
		
		Purpose:
			Validates and normalizes the input value used for the Embeddings workflow. The method
			raises an application error when required input is missing and returns a clean value
			suitable for downstream provider calls.
		
		Args:
			text (str | List[str]): Text value used by the operation.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
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
	
	def validate_dimensions( self ) -> int | None:
		"""Validate dimensions.
		
		Purpose:
			Validates and normalizes the dimensions value used for the Embeddings workflow. The
			method raises an application error when required input is missing and returns a clean
			value suitable for downstream provider calls.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.dimensions is None:
				return None
			
			try:
				value = int( self.dimensions )
			except Exception as e:
				exception = Error( e )
				exception.module = 'gpt'
				exception.cause = 'Embeddings'
				exception.method = 'validate_dimensions( ... )'
				Logger( ).write( exception )
				return None
			
			if value <= 0:
				return None
			
			supports_dimensions = self.model_dimension_support.get( self.model, False )
			if not supports_dimensions:
				return None
			
			max_dimensions = self.get_max_dimensions( self.model )
			if value > max_dimensions:
				return max_dimensions
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_dimensions( self ) -> int | None'
			Logger( ).write( exception )
			raise exception
	
	def get_default_dimensions( self, model: str ) -> int:
		"""Get default dimensions.
		
		Purpose:
			Returns the default dimensions value for the active Embeddings request. The method
			inspects current runtime state and provides a safe application-facing result.
		
		Args:
			model (str): Model value used by the operation.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'model', model )
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
			Returns the max dimensions value for the active Embeddings request. The method inspects
			current runtime state and provides a safe application-facing result.
		
		Args:
			model (str): Model value used by the operation.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'model', model )
			return int( self.model_max_dimensions.get( model, 1536 ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'get_max_dimensions( self, model: str ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def build_request( self, text: str | List[ str ], model: str='text-embedding-3-small',
		format: str='float', dimensions: int=None, user: str=None ) -> Dict[ str, Any ]:
		"""Build request.
		
		Purpose:
			Builds the request payload used for the Embeddings workflow. The method validates
			caller
			input, applies compatibility defaults, and returns a provider-ready structure without
			executing the provider request.
		
		Args:
			text (str | List[str]): Text value used by the operation.
			model (str): Model value used by the operation.
			format (str): Format value used by the operation.
			dimensions (int): Dimensions value used by the operation.
			user (str): User value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'model', model )
			throw_if( 'format', format )
			self.input = self.validate_input( text )
			self.model = model
			self.encoding_format = format
			self.dimensions = dimensions
			self.dimensions = self.validate_dimensions( )
			self.user = user if isinstance( user, str ) and user.strip( ) else None
			self.request = { 'model': self.model, 'input': self.input,
				'encoding_format': self.encoding_format, }
			
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
	
	def create( self, text: str | List[ str ], model: str='text-embedding-3-small',
		format: str='float', dimensions: int=None, user: str=None ) -> List[ float ] | List[
		List[ float ] ] | str | List[ str ] | None:
		"""Create.
		
		Purpose:
			Creates provider resources or generated outputs for the Embeddings workflow using
			validated request state and provider-specific defaults.
		
		Args:
			text (str | List[str]): Text value used by the operation.
			model (str): Model value used by the operation.
			format (str): Format value used by the operation.
			dimensions (int): Dimensions value used by the operation.
			user (str): User value used by the operation.
		
		Returns:
			Single embedding, batch embeddings, base64 embedding content, or no value when no
			embeddings are returned.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
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
		"""Return member names.
		
		Purpose:
			Returns a stable list of public members exposed by the Embeddings object for
			interactive
			inspection, debugging, and application-level compatibility.
		
		Returns:
			Member names exposed for inspection.
		"""
		return [ 'api_key', 'client', 'model', 'input', 'encoding_format', 'dimensions', 'user',
			'response', 'embedding', 'embeddings', 'usage', 'request', 'model_options',
			'encoding_options', 'model_default_dimensions', 'model_max_dimensions',
			'model_dimension_support', 'validate_input', 'validate_dimensions',
			'get_default_dimensions', 'get_max_dimensions', 'build_request', 'create', ]

class Files( GPT ):
	"""Provide OpenAI Files API workflow support.
	
	Purpose:
		Provides OpenAI file upload, listing, retrieval, content extraction, deletion, summary,
		search, and survey operations. The class assigns each accepted method argument to an
		object member before constructing and executing the corresponding provider request.
	
	Attributes:
		api_key (str): OpenAI API key used by the wrapper.
		client (Optional[OpenAI]): OpenAI client used by the wrapper.
		file (Any): Latest OpenAI file object.
		file_id (str): File identifier used by the current operation.
		filepath (str): Local file path used by an upload operation.
		filename (str): Filename associated with the current file.
		purpose (str): OpenAI file purpose.
		response (Any): Latest provider response.
		content (str | bytes | Dict[str, Any] | None): Retrieved file content.
		files (List[Dict[str, Any]]): File metadata returned by the latest list operation.
		request (Dict[str, Any]): Provider-ready request values.
		model (str): OpenAI model used for file-content analysis.
		prompt (str): Prompt used for file-content analysis.
		output_text (str): Text returned by the latest Responses API request.
		max_chars (int): Maximum file-content characters included in analysis.
	"""
	api_key: str
	client: Optional[ OpenAI ]
	file: Any
	file_id: str
	filepath: str
	filename: str
	purpose: str
	response: Any
	content: str | bytes | Dict[ str, Any ] | None
	files: List[ Dict[ str, Any ] ]
	request: Dict[ str, Any ]
	model: str
	prompt: str
	output_text: str
	max_chars: int
	
	def __init__( self, id: str = '', filepath: str = '', purpose: str = 'user_data',
		model: str = 'gpt-4o-mini', prompt: str = '' ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes OpenAI Files API configuration and runtime state without executing a
			provider request.
		
		Args:
			id (str): Optional initial OpenAI file identifier.
			filepath (str): Optional initial local file path.
			purpose (str): Default OpenAI upload purpose.
			model (str): Default model used for file-content analysis.
			prompt (str): Optional initial file-analysis prompt.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.file = None
		self.file_id = id
		self.filepath = filepath
		self.filename = ''
		self.purpose = purpose
		self.response = None
		self.content = None
		self.files = [ ]
		self.request = { }
		self.model = model
		self.prompt = prompt
		self.output_text = ''
		self.max_chars = 0
		self.metadata = { }
		self.preview = ''
		self.file_data = [ ]
		self.source = { }
		self.content_text = ''
		self.input = [ ]
	
	@property
	def upload_purpose_options( self ) -> List[ str ]:
		"""Get upload-purpose options.
		
		Purpose:
			Returns purposes accepted when uploading files through the OpenAI Files API.
		
		Returns:
			List[str]: Supported upload-purpose values.
		"""
		return [ 'assistants', 'batch', 'fine-tune', 'vision', 'user_data', 'evals', ]
	
	@property
	def file_purpose_options( self ) -> List[ str ]:
		"""Get file-purpose options.
		
		Purpose:
			Returns file-purpose values that may appear in OpenAI file metadata.
		
		Returns:
			List[str]: Supported file-purpose metadata values.
		"""
		return [ 'assistants', 'assistants_output', 'batch', 'batch_output', 'fine-tune',
			'fine-tune-results', 'vision', 'user_data', 'evals', ]
	
	@property
	def purpose_options( self ) -> List[ str ]:
		"""Get purpose options.
		
		Purpose:
			Returns upload-purpose values exposed to the application.
		
		Returns:
			List[str]: Supported upload-purpose values.
		"""
		return self.upload_purpose_options
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get file-analysis model options.
		
		Purpose:
			Returns OpenAI models exposed for file-content summary and search operations.
		
		Returns:
			List[str]: Supported model identifiers.
		"""
		return [ 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o-mini', ]
	
	def get_file_metadata( self, file: Any ) -> Dict[ str, Any ]:
		"""Get file metadata.
		
		Purpose:
			Extracts application-facing metadata from a required OpenAI file object.
		
		Args:
			file (Any): Required OpenAI file object or file metadata dictionary.
		
		Returns:
			Dict[str, Any]: Application-facing file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'file', file )
			self.file = file
			if isinstance( self.file, dict ):
				self.source = self.file
			elif hasattr( self.file, 'model_dump' ):
				self.source = self.file.model_dump( )
			else:
				self.source = { 'id': getattr( self.file, 'id', '' ),
					'bytes': getattr( self.file, 'bytes', 0 ),
					'created_at': getattr( self.file, 'created_at', 0 ),
					'expires_at': getattr( self.file, 'expires_at', 0 ),
					'filename': getattr( self.file, 'filename', '' ),
					'object': getattr( self.file, 'object', '' ),
					'purpose': getattr( self.file, 'purpose', '' ),
					'status': getattr( self.file, 'status', '' ),
					'status_details': getattr( self.file, 'status_details', None, ), }
			
			self.metadata = { 'id': self.source.get( 'id', '' ),
				'filename': self.source.get( 'filename', '' ),
				'purpose': self.source.get( 'purpose', '' ), 'bytes': self.source.get( 'bytes', 0 ),
				'created_at': self.source.get( 'created_at', 0 ),
				'expires_at': self.source.get( 'expires_at', 0 ),
				'object': self.source.get( 'object', '' ),
				'status': self.source.get( 'status', '' ),
				'status_details': self.source.get( 'status_details', None, ), }
			return self.metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = ('get_file_metadata( self, file: Any ) -> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def get_file_content( self, response: Any ) -> str | bytes | Dict[ str, Any ]:
		"""Get file content.
		
		Purpose:
			Extracts text, bytes, or structured content from a required OpenAI file-content
			response.
		
		Args:
			response (Any): Required OpenAI file-content response.
		
		Returns:
			str | bytes | Dict[str, Any]: Extracted file content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response', response )
			self.response = response
			if isinstance( self.response, bytes ):
				self.content = self.response
				return self.content
			
			if isinstance( self.response, str ):
				self.content = self.response
				return self.content
			
			if isinstance( self.response, dict ):
				self.content = self.response
				return self.content
			
			if hasattr( self.response, 'text' ):
				self.response_text = self.response.text
				if callable( self.response_text ):
					self.content = self.response_text( )
				else:
					self.content = self.response_text
				
				if self.content is not None:
					return self.content
			
			if hasattr( self.response, 'content' ):
				self.response_content = self.response.content
				if callable( self.response_content ):
					self.content = self.response_content( )
				else:
					self.content = self.response_content
				
				if self.content is not None:
					return self.content
			
			if hasattr( self.response, 'read' ):
				self.content = self.response.read( )
				return self.content
			
			if hasattr( self.response, 'model_dump' ):
				self.content = self.response.model_dump( )
				return self.content
			
			self.content = str( self.response )
			return self.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'get_file_content( self, **kwargs) -> str | bytes | Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def get_content_text( self, content: str | bytes | Dict[ str, Any ] ) -> str:
		"""Get content text.
		
		Purpose:
			Converts retrieved file content into text suitable for a Responses API request.
		
		Args:
			content (str | bytes | Dict[str, Any]): Required retrieved file content.
		
		Returns:
			str: File content represented as text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'content', content )
			self.content = content
			self.content_text = ''
			if isinstance( self.content, str ):
				self.content_text = self.content
			elif isinstance( self.content, bytes ):
				self.content_text = self.content.decode( 'utf-8', errors='replace', )
			elif isinstance( self.content, dict ):
				self.content_text = json.dumps( self.content, ensure_ascii=False, indent=2,
					default=str, )
			else:
				self.content_text = str( self.content )
			
			throw_if( 'content_text', self.content_text )
			return self.content_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'get_content_text( self, **kwargs ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def upload( self, path: str, purpose: str = 'user_data' ) -> Dict[ str, Any ]:
		"""Upload a file.
		
		Purpose:
			Uploads a required local file to the OpenAI Files API using the selected purpose.
		
		Args:
			path (str): Required local file path.
			purpose (str): OpenAI upload purpose.
		
		Returns:
			Dict[str, Any]: Metadata for the uploaded file.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'purpose', purpose )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.filepath = path
			self.purpose = purpose
			self.filename = Path( self.filepath ).name
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'purpose': self.purpose, }
			with open( self.filepath, 'rb' ) as source:
				self.file = source
				self.response = self.client.files.create( file=self.file,
					purpose=self.request[ 'purpose' ], )
			
			self.file = self.response
			self.metadata = self.get_file_metadata( self.file )
			self.file_id = self.metadata.get( 'id', '', )
			self.filename = self.metadata.get( 'filename', self.filename, )
			return self.metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'upload( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list( self, purpose: str = '' ) -> List[ Dict[ str, Any ] ]:
		"""List files.
		
		Purpose:
			Lists files available through the OpenAI Files API and optionally limits the result
			to a selected file purpose.
		
		Args:
			purpose (str): Optional file-purpose filter.
		
		Returns:
			List[Dict[str, Any]]: Application-facing file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.purpose = purpose
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { }
			self.response = self.client.files.list( )
			self.file_data = getattr( self.response, 'data', [ ], ) or [ ]
			self.files = [ ]
			for item in self.file_data:
				self.metadata = self.get_file_metadata( item )
				
				if self.purpose:
					if self.metadata.get( 'purpose', '' ) != self.purpose:
						continue
				
				self.files.append( self.metadata )
			
			return self.files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'list( self, purpose: str = "" ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, id: str ) -> Dict[ str, Any ]:
		"""Retrieve file metadata.
		
		Purpose:
			Retrieves metadata for a required OpenAI file identifier.
		
		Args:
			id (str): Required OpenAI file identifier.
		
		Returns:
			Dict[str, Any]: Application-facing file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'id', id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.file_id = id
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'file_id': self.file_id, }
			self.response = self.client.files.retrieve( file_id=self.request[ 'file_id' ], )
			self.file = self.response
			self.metadata = self.get_file_metadata( self.file )
			self.filename = self.metadata.get( 'filename', '', )
			return self.metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'retrieve( self, id: str ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def extract( self, id: str ) -> str | bytes | Dict[ str, Any ]:
		"""Extract file content.
		
		Purpose:
			Retrieves content for a required OpenAI file identifier.
		
		Args:
			id (str): Required OpenAI file identifier.
		
		Returns:
			str | bytes | Dict[str, Any]: Retrieved file content.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'id', id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.file_id = id
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'file_id': self.file_id, }
			self.response = self.client.files.content( file_id=self.request[ 'file_id' ], )
			self.content = self.get_file_content( self.response )
			return self.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'extract( self, id: str ) -> str | bytes | Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, id: str ) -> Dict[ str, Any ]:
		"""Delete a file.
		
		Purpose:
			Deletes a required OpenAI file identifier and returns the provider deletion result.
		
		Args:
			id (str): Required OpenAI file identifier.
		
		Returns:
			Dict[str, Any]: File deletion result.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'id', id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.file_id = id
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'file_id': self.file_id, }
			self.response = self.client.files.delete( file_id=self.request[ 'file_id' ], )
			if isinstance( self.response, dict ):
				self.metadata = self.response
			elif hasattr( self.response, 'model_dump' ):
				self.metadata = self.response.model_dump( )
			else:
				self.metadata = { 'id': getattr( self.response, 'id', self.file_id, ),
					'deleted': getattr( self.response, 'deleted', False, ),
					'object': getattr( self.response, 'object', 'file', ), }
			
			return self.metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'delete( self, id: str ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def summarize( self, id: str, prompt: str = 'Summarize the selected file content.',
		model: str = 'gpt-4o-mini', max_chars: int = 120000 ) -> str:
		"""Summarize file content.
		
		Purpose:
			Retrieves a required file and summarizes or analyzes its content through the OpenAI
			Responses API.
		
		Args:
			id (str): Required OpenAI file identifier.
			prompt (str): File-summary or analysis instruction.
			model (str): OpenAI model used for analysis.
			max_chars (int): Maximum content characters included in the request.
		
		Returns:
			str: Generated file-content analysis.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'id', id )
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.file_id = id
			self.prompt = prompt
			self.model = model
			self.max_chars = max_chars
			self.content = self.extract( self.file_id )
			self.content_text = self.get_content_text( self.content )
			
			if self.max_chars > 0:
				self.content_text = self.content_text[ :self.max_chars ]
			_items = (f'{self.prompt}\n\n'
                                    f'File ID: {
                                    self.file_id}\n\n'
                                    f'{self.content_text}')
			self.input = [ { 'role': 'user',
				'content': [ { 'type': 'input_text', 'text': _items, }, ], }, ]
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'model': self.model, 'input': self.input, }
			self.response = self.client.responses.create( **self.request )
			self.output_text = getattr( self.response, 'output_text', '', )
			if self.output_text:
				return self.output_text
			
			self.output_text = str( self.response )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'summarize( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def search( self, id: str, query: str, model: str = 'gpt-4o-mini',
		max_chars: int = 120000 ) -> str:
		"""Search file content.
		
		Purpose:
			Answers a required question using content retrieved from a required OpenAI file.
		
		Args:
			id (str): Required OpenAI file identifier.
			query (str): Required question about the selected file.
			model (str): OpenAI model used for analysis.
			max_chars (int): Maximum content characters included in the request.
		
		Returns:
			str: Generated answer based on the selected file.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'id', id )
			throw_if( 'query', query )
			throw_if( 'model', model )
			self.file_id = id
			self.query = query
			self.model = model
			self.max_chars = max_chars
			self.prompt = ('Answer the user question using the selected file content. '
			               f'Question: {self.query}')
			self.output_text = self.summarize( self.file_id, self.prompt, self.model, self.max_chars, )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'search( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, id: str, max_chars: int = 4000 ) -> Dict[ str, Any ]:
		"""Survey a file.
		
		Purpose:
			Retrieves file metadata and a bounded content preview for a required OpenAI file.
		
		Args:
			id (str): Required OpenAI file identifier.
			max_chars (int): Maximum preview characters returned.
		
		Returns:
			Dict[str, Any]: File metadata, content preview, and file identifier.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'id', id )
			self.file_id = id
			self.max_chars = max_chars
			self.metadata = self.retrieve( self.file_id )
			self.content = self.extract( self.file_id )
			self.content_text = self.get_content_text( self.content )
			self.preview = self.content_text
			
			if self.max_chars > 0:
				self.preview = self.content_text[ :self.max_chars ]
			
			self.result = { 'metadata': self.metadata, 'preview': self.preview,
				'file_id': self.file_id, }
			return self.result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'survey( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the OpenAI Files wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'client', 'file', 'file_id', 'filepath', 'filename', 'purpose',
			'response', 'content', 'files', 'request', 'model', 'prompt', 'output_text',
			'max_chars', 'metadata', 'preview', 'upload_purpose_options', 'file_purpose_options',
			'purpose_options', 'model_options', 'get_file_metadata', 'get_file_content',
			'get_content_text', 'upload', 'list', 'retrieve', 'extract', 'delete', 'summarize',
			'search', 'survey', ]

class VectorStores( GPT ):
	"""Provide OpenAI Vector Stores API workflow support.
	
	Purpose:
		Provides vector-store management, attached-file management, file-batch operations,
		native vector-store search, and Responses API file-search workflows. Each public
		wrapper method checks required input with throw_if, assigns accepted arguments to
		object members, constructs provider requests from those members, and returns
		application-facing metadata or generated text.
	
	Attributes:
		api_key (str): OpenAI API key used by the wrapper.
		client (Optional[OpenAI]): OpenAI client used by the wrapper.
		name (str): Vector-store name used by the current operation.
		description (str): Vector-store description used by the current operation.
		store_id (str): Vector-store identifier used by the current operation.
		file_id (str): File identifier used by the current operation.
		file_ids (List[str]): File identifiers used by the current operation.
		batch_id (str): File-batch identifier used by the current operation.
		model (str): OpenAI model used by file-search answer workflows.
		query (str): Native vector-store search query.
		prompt (str): Responses API file-search prompt.
		instructions (str): Optional Responses API instructions.
		max_search_results (int): Maximum number of search results requested.
		response (Any): Latest provider response.
		vector_store (Dict[str, Any]): Latest vector-store metadata.
		vector_stores (List[Dict[str, Any]]): Latest vector-store collection.
		vector_file (Dict[str, Any]): Latest attached-file metadata.
		vector_files (List[Dict[str, Any]]): Latest attached-file collection.
		file_batch (Dict[str, Any]): Latest file-batch metadata.
		search_results (List[Dict[str, Any]]): Latest native search results.
		output_text (str): Latest Responses API file-search answer.
		request (Dict[str, Any]): Provider-ready request.
	"""
	api_key: str
	client: Optional[ OpenAI ]
	name: str
	description: str
	store_id: str
	file_id: str
	file_ids: List[ str ]
	batch_id: str
	model: str
	query: str
	prompt: str
	instructions: str
	max_search_results: int
	response: Any
	vector_store: Dict[ str, Any ]
	vector_stores: List[ Dict[ str, Any ] ]
	vector_file: Dict[ str, Any ]
	vector_files: List[ Dict[ str, Any ] ]
	file_batch: Dict[ str, Any ]
	search_results: List[ Dict[ str, Any ] ]
	output_text: str
	request: Dict[ str, Any ]
	
	def __init__( self, name: str = '', store_id: str = '', file_id: str = '',
		model: str = 'gpt-4o-mini', max_search_results: int = 10 ) -> None:
		"""Initialize instance.
		
		Purpose:
			Initializes OpenAI vector-store configuration and runtime state without executing a
			provider request.
		
		Args:
			name (str): Optional initial vector-store name.
			store_id (str): Optional initial vector-store identifier.
			file_id (str): Optional initial file identifier.
			model (str): Default model used by file-search answer workflows.
			max_search_results (int): Default maximum number of search results.
		
		Returns:
			None: This method initializes object state.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.name = name
		self.description = ''
		self.store_id = store_id
		self.file_id = file_id
		self.file_ids = [ ]
		self.batch_id = ''
		self.model = model
		self.query = ''
		self.prompt = ''
		self.instructions = ''
		self.max_search_results = max_search_results
		self.metadata = { }
		self.attributes = { }
		self.filters = { }
		self.ranking_options = { }
		self.expires_after = { }
		self.chunking_strategy = { }
		self.response = None
		self.vector_store = { }
		self.vector_stores = [ ]
		self.vector_file = { }
		self.vector_files = [ ]
		self.file_batch = { }
		self.search_results = [ ]
		self.output_text = ''
		self.request = { }
		self.input = [ ]
		self.limit = 100
		self.order = 'desc'
		self.after = ''
		self.before = ''
		self.rewrite_query = False
		self.collections = { 'Governance': 'vs_6a1850a9bdc08191912353eedf59aede',
			'Public Laws': 'vs_699506f7d5348191990e0557c717fa9d',
			'Explanatory Statements': 'vs_699505df9ac48191a525c0ecb86fef66',
			'Army Techniques Publications': 'vs_699356ef052c81918da14c4ed3bcea17',
			'Army Field Manuals': 'vs_69935542863481918d150c1e89c38633',
			'Army Regulations': 'vs_6993550488408191919cd70968ba8be8',
			'DoD Armory': 'vs_697f86ad98888191b967685ae558bfc0',
			'Army Style Guides': 'vs_68f4efd7d4c4819191458dd6cde6f2cc',
			'Apportionments': 'vs_68a34aaff93481918c3b3fef8c4e8fea',
			'Financial Regulations': 'vs_712r5W5833G6aLxIYIbuvVcK', }
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Get model options.
		
		Purpose:
			Returns OpenAI models exposed for Responses API file-search answer workflows.
		
		Returns:
			List[str]: Supported model identifiers.
		"""
		return [ 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o-mini', ]
	
	@property
	def ranker_options( self ) -> List[ str ]:
		"""Get ranker options.
		
		Purpose:
			Returns ranking algorithms exposed for native vector-store search.
		
		Returns:
			List[str]: Supported ranker values.
		"""
		return [ 'auto', 'default-2024-11-15', ]
	
	@property
	def chunking_strategy_options( self ) -> List[ str ]:
		"""Get chunking-strategy options.
		
		Purpose:
			Returns chunking strategies supported by vector-store file operations.
		
		Returns:
			List[str]: Supported chunking-strategy values.
		"""
		return [ 'auto', 'static', ]
	
	def get_vector_store( self, response: Any ) -> Dict[ str, Any ]:
		"""Get vector-store metadata.
		
		Purpose:
			Extracts application-facing metadata from a required vector-store response.
		
		Args:
			response (Any): Required provider vector-store response.
		
		Returns:
			Dict[str, Any]: Application-facing vector-store metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response', response )
			self.response = response
			
			if isinstance( self.response, dict ):
				self.source = self.response
			elif hasattr( self.response, 'model_dump' ):
				self.source = self.response.model_dump( )
			else:
				self.source = { 'id': getattr( self.response, 'id', '' ),
					'name': getattr( self.response, 'name', '' ),
					'description': getattr( self.response, 'description', '' ),
					'created_at': getattr( self.response, 'created_at', 0 ),
					'object': getattr( self.response, 'object', '' ),
					'usage_bytes': getattr( self.response, 'usage_bytes', 0 ),
					'file_counts': getattr( self.response, 'file_counts', None ),
					'status': getattr( self.response, 'status', '' ),
					'expires_after': getattr( self.response, 'expires_after', None ),
					'expires_at': getattr( self.response, 'expires_at', 0 ),
					'last_active_at': getattr( self.response, 'last_active_at', 0 ),
					'metadata': getattr( self.response, 'metadata', None ), }
			
			self.vector_store = { 'id': self.source.get( 'id', '' ),
				'name': self.source.get( 'name', '' ),
				'description': self.source.get( 'description', '' ),
				'created_at': self.source.get( 'created_at', 0 ),
				'object': self.source.get( 'object', '' ),
				'usage_bytes': self.source.get( 'usage_bytes', 0 ),
				'file_counts': self.source.get( 'file_counts', None ),
				'status': self.source.get( 'status', '' ),
				'expires_after': self.source.get( 'expires_after', None ),
				'expires_at': self.source.get( 'expires_at', 0 ),
				'last_active_at': self.source.get( 'last_active_at', 0 ),
				'metadata': self.source.get( 'metadata', None ), }
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'get_vector_store( self, response: Any )'
			Logger( ).write( exception )
			raise exception
	
	def get_vector_file( self, response: Any ) -> Dict[ str, Any ]:
		"""Get vector-store file metadata.
		
		Purpose:
			Extracts application-facing metadata from a required vector-store file response.
		
		Args:
			response (Any): Required provider vector-store file response.
		
		Returns:
			Dict[str, Any]: Application-facing attached-file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response', response )
			self.response = response
			
			if isinstance( self.response, dict ):
				self.source = self.response
			elif hasattr( self.response, 'model_dump' ):
				self.source = self.response.model_dump( )
			else:
				self.source = { 'id': getattr( self.response, 'id', '' ),
					'object': getattr( self.response, 'object', '' ),
					'created_at': getattr( self.response, 'created_at', 0 ),
					'vector_store_id': getattr( self.response, 'vector_store_id', '', ),
					'status': getattr( self.response, 'status', '' ),
					'last_error': getattr( self.response, 'last_error', None ),
					'chunking_strategy': getattr( self.response, 'chunking_strategy', None, ),
					'attributes': getattr( self.response, 'attributes', None ),
					'usage_bytes': getattr( self.response, 'usage_bytes', 0 ), }
			
			self.vector_file = { 'id': self.source.get( 'id', '' ),
				'object': self.source.get( 'object', '' ),
				'created_at': self.source.get( 'created_at', 0 ),
				'vector_store_id': self.source.get( 'vector_store_id', '' ),
				'status': self.source.get( 'status', '' ),
				'last_error': self.source.get( 'last_error', None ),
				'chunking_strategy': self.source.get( 'chunking_strategy', None ),
				'attributes': self.source.get( 'attributes', None ),
				'usage_bytes': self.source.get( 'usage_bytes', 0 ), }
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'get_vector_file( self, response: Any )'
			Logger( ).write( exception )
			raise exception
	
	def get_file_batch( self, response: Any ) -> Dict[ str, Any ]:
		"""Get file-batch metadata.
		
		Purpose:
			Extracts application-facing metadata from a required file-batch response.
		
		Args:
			response (Any): Required provider file-batch response.
		
		Returns:
			Dict[str, Any]: Application-facing file-batch metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response', response )
			self.response = response
			
			if isinstance( self.response, dict ):
				self.source = self.response
			elif hasattr( self.response, 'model_dump' ):
				self.source = self.response.model_dump( )
			else:
				self.source = { 'id': getattr( self.response, 'id', '' ),
					'object': getattr( self.response, 'object', '' ),
					'created_at': getattr( self.response, 'created_at', 0 ),
					'vector_store_id': getattr( self.response, 'vector_store_id', '', ),
					'status': getattr( self.response, 'status', '' ),
					'file_counts': getattr( self.response, 'file_counts', None, ), }
			
			self.file_batch = { 'id': self.source.get( 'id', '' ),
				'object': self.source.get( 'object', '' ),
				'created_at': self.source.get( 'created_at', 0 ),
				'vector_store_id': self.source.get( 'vector_store_id', '' ),
				'status': self.source.get( 'status', '' ),
				'file_counts': self.source.get( 'file_counts', None ), }
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'get_file_batch( self, response: Any )'
			Logger( ).write( exception )
			raise exception
	
	def get_search_results( self, response: Any ) -> List[ Dict[ str, Any ] ]:
		"""Get vector-store search results.
		
		Purpose:
			Extracts native vector-store search results from a required provider response.
		
		Args:
			response (Any): Required provider vector-store search response.
		
		Returns:
			List[Dict[str, Any]]: Application-facing search results.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'response', response )
			self.response = response
			self.items = getattr( self.response, 'data', [ ], ) or [ ]
			self.search_results = [ ]
			for item in self.items:
				if isinstance( item, dict ):
					self.source = item
				elif hasattr( item, 'model_dump' ):
					self.source = item.model_dump( )
				else:
					self.source = { 'file_id': getattr( item, 'file_id', '' ),
						'filename': getattr( item, 'filename', '' ),
						'score': getattr( item, 'score', 0.0 ),
						'attributes': getattr( item, 'attributes', None ),
						'content': getattr( item, 'content', [ ] ), }
				
				self.search_results.append( { 'file_id': self.source.get( 'file_id', '' ),
					'filename': self.source.get( 'filename', '' ),
					'score': self.source.get( 'score', 0.0 ),
					'attributes': self.source.get( 'attributes', None ),
					'content': self.source.get( 'content', [ ] ), } )
			
			return self.search_results
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'get_search_results( self, response: Any )'
			Logger( ).write( exception )
			raise exception
	
	def create( self, name: str, description: str = '',
		metadata: Optional[ Dict[ str, Any ] ] = None,
		expires_after: Optional[ Dict[ str, Any ] ] = None,
		file_ids: Optional[ List[ str ] ] = None,
		chunking_strategy: Optional[ Dict[ str, Any ] ] = None ) -> Dict[ str, Any ]:
		"""Create a vector store.
		
		Purpose:
			Creates a vector store with a required name and optional description, metadata,
			expiration policy, files, and chunking strategy.
		
		Args:
			name (str): Required vector-store name.
			description (str): Optional vector-store description.
			metadata (Optional[Dict[str, Any]]): Optional vector-store metadata.
			expires_after (Optional[Dict[str, Any]]): Optional expiration policy.
			file_ids (Optional[List[str]]): Optional files attached during creation.
			chunking_strategy (Optional[Dict[str, Any]]): Optional chunking strategy.
		
		Returns:
			Dict[str, Any]: Created vector-store metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'name', name )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.name = name
			self.description = description
			self.metadata = metadata if metadata is not None else { }
			self.expires_after = (expires_after if expires_after is not None else { })
			self.file_ids = file_ids if file_ids is not None else [ ]
			self.chunking_strategy = (chunking_strategy if chunking_strategy is not None else { })
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'name': self.name, }
			if self.description:
				self.request[ 'description' ] = self.description
			
			if self.metadata:
				self.request[ 'metadata' ] = self.metadata
			
			if self.expires_after:
				self.request[ 'expires_after' ] = self.expires_after
			
			if self.file_ids:
				self.request[ 'file_ids' ] = self.file_ids
			
			if self.chunking_strategy:
				self.request[ 'chunking_strategy' ] = self.chunking_strategy
			
			self.response = self.client.vector_stores.create( **self.request )
			self.vector_store = self.get_vector_store( self.response )
			self.store_id = self.vector_store.get( 'id', '', )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'create( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list_stores( self, limit: int = 100, order: str = 'desc', after: str = '',
		before: str = '' ) -> List[ Dict[ str, Any ] ]:
		"""List vector stores.
		
		Purpose:
			Lists vector stores using the selected result limit, order, and cursor values.
		
		Args:
			limit (int): Maximum number of vector stores returned.
			order (str): Result order.
			after (str): Optional cursor identifying the first result boundary.
			before (str): Optional cursor identifying the last result boundary.
		
		Returns:
			List[Dict[str, Any]]: Vector-store metadata rows.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.limit = limit
			self.order = order
			self.after = after
			self.before = before
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'limit': self.limit, 'order': self.order, }
			if self.after:
				self.request[ 'after' ] = self.after
			
			if self.before:
				self.request[ 'before' ] = self.before
			
			self.response = self.client.vector_stores.list( **self.request )
			self.items = getattr( self.response, 'data', [ ], ) or [ ]
			self.vector_stores = [ self.get_vector_store( item ) for item in self.items ]
			return self.vector_stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_stores( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list( self, limit: int=100, order: str='desc', after: str='',
		before: str='' ) -> List[ Dict[ str, Any ] ]:
		"""List vector stores.
		
		Purpose:
			Provides the application-compatible list alias for vector-store listing.
		
		Args:
			limit (int): Maximum number of vector stores returned.
			order (str): Result order.
			after (str): Optional after cursor.
			before (str): Optional before cursor.
		
		Returns:
			List[Dict[str, Any]]: Vector-store metadata rows.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.limit = limit
			self.order = order
			self.after = after
			self.before = before
			return self.list_stores( self.limit, self.order, self.after, self.before, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str ) -> Dict[ str, Any ]:
		"""Retrieve a vector store.
		
		Purpose:
			Retrieves metadata for a required vector-store identifier.
		
		Args:
			store_id (str): Required vector-store identifier.
		
		Returns:
			Dict[str, Any]: Retrieved vector-store metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.client = OpenAI( api_key=self.api_key, )
			self.response = self.client.vector_stores.retrieve( self.store_id )
			self.vector_store = self.get_vector_store( self.response )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = ('retrieve( self, store_id: str ) -> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def update( self, store_id: str, name: str = '', description: str = '',
		metadata: Optional[ Dict[ str, Any ] ] = None,
		expires_after: Optional[ Dict[ str, Any ] ] = None ) -> Dict[ str, Any ]:
		"""Update a vector store.
		
		Purpose:
			Updates a required vector store using supplied name, description, metadata, or
			expiration-policy values.
		
		Args:
			store_id (str): Required vector-store identifier.
			name (str): Optional updated vector-store name.
			description (str): Optional updated description.
			metadata (Optional[Dict[str, Any]]): Optional updated metadata.
			expires_after (Optional[Dict[str, Any]]): Optional updated expiration policy.
		
		Returns:
			Dict[str, Any]: Updated vector-store metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.name = name
			self.description = description
			self.metadata = metadata if metadata is not None else { }
			self.expires_after = (expires_after if expires_after is not None else { })
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { }
			if self.name:
				self.request[ 'name' ] = self.name
			
			if self.description:
				self.request[ 'description' ] = self.description
			
			if metadata is not None:
				self.request[ 'metadata' ] = self.metadata
			
			if self.expires_after:
				self.request[ 'expires_after' ] = self.expires_after
			
			if not self.request:
				return self.retrieve( self.store_id )
			
			self.response = self.client.vector_stores.update( self.store_id, **self.request )
			self.vector_store = self.get_vector_store( self.response )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'update( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str ) -> Dict[ str, Any ]:
		"""Delete a vector store.
		
		Purpose:
			Deletes a required vector-store identifier.
		
		Args:
			store_id (str): Required vector-store identifier.
		
		Returns:
			Dict[str, Any]: Provider deletion result.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.client = OpenAI( api_key=self.api_key, )
			self.response = self.client.vector_stores.delete( self.store_id )
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return { 'id': getattr( self.response, 'id', self.store_id ),
				'deleted': getattr( self.response, 'deleted', False ),
				'object': getattr( self.response, 'object', '' ), }
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = ('delete( self, store_id: str ) -> Dict[ str, Any ]')
			Logger( ).write( exception )
			raise exception
	
	def attach_file( self, store_id: str, file_id: str,
		attributes: Optional[ Dict[ str, Any ] ] = None,
		chunking_strategy: Optional[ Dict[ str, Any ] ] = None ) -> Dict[ str, Any ]:
		"""Attach a file.
		
		Purpose:
			Attaches a required OpenAI file to a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			file_id (str): Required OpenAI file identifier.
			attributes (Optional[Dict[str, Any]]): Optional attached-file attributes.
			chunking_strategy (Optional[Dict[str, Any]]): Optional chunking strategy.
		
		Returns:
			Dict[str, Any]: Attached-file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.file_id = file_id
			self.attributes = attributes if attributes is not None else { }
			self.chunking_strategy = (chunking_strategy if chunking_strategy is not None else { })
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'file_id': self.file_id, }
			if self.attributes:
				self.request[ 'attributes' ] = self.attributes
			
			if self.chunking_strategy:
				self.request[ 'chunking_strategy' ] = self.chunking_strategy
			
			self.response = self.client.vector_stores.files.create( self.store_id, **self.request )
			self.vector_file = self.get_vector_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'attach_file( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list_files( self, store_id: str, limit: int = 100, order: str = 'desc', after: str = '',
		before: str = '', filter: str = '' ) -> List[ Dict[ str, Any ] ]:
		"""List attached files.
		
		Purpose:
			Lists files attached to a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			limit (int): Maximum number of files returned.
			order (str): Result order.
			after (str): Optional after cursor.
			before (str): Optional before cursor.
			filter (str): Optional attached-file status filter.
		
		Returns:
			List[Dict[str, Any]]: Attached-file metadata rows.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.limit = limit
			self.order = order
			self.after = after
			self.before = before
			self.filter = filter
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'limit': self.limit, 'order': self.order, }
			if self.after:
				self.request[ 'after' ] = self.after
			
			if self.before:
				self.request[ 'before' ] = self.before
			
			if self.filter:
				self.request[ 'filter' ] = self.filter
			
			self.response = self.client.vector_stores.files.list( self.store_id, **self.request )
			self.items = getattr( self.response, 'data', [ ], ) or [ ]
			self.vector_files = [ self.get_vector_file( item ) for item in self.items ]
			return self.vector_files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_files( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ]:
		"""Retrieve an attached file.
		
		Purpose:
			Retrieves metadata for a required file attached to a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			file_id (str): Required attached-file identifier.
		
		Returns:
			Dict[str, Any]: Attached-file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.file_id = file_id
			self.client = OpenAI( api_key=self.api_key, )
			self.response = self.client.vector_stores.files.retrieve( self.file_id,
				vector_store_id=self.store_id, )
			self.vector_file = self.get_vector_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def update_file( self, store_id: str, file_id: str,
		attributes: Dict[ str, Any ] ) -> Dict[ str, Any ]:
		"""Update an attached file.
		
		Purpose:
			Updates attributes for a required file attached to a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			file_id (str): Required attached-file identifier.
			attributes (Dict[str, Any]): Required updated file attributes.
		
		Returns:
			Dict[str, Any]: Updated attached-file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			throw_if( 'attributes', attributes )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.file_id = file_id
			self.attributes = attributes
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'attributes': self.attributes, }
			self.response = self.client.vector_stores.files.update( self.file_id,
				vector_store_id=self.store_id, **self.request )
			self.vector_file = self.get_vector_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'update_file( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def delete_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ]:
		"""Delete an attached file.
		
		Purpose:
			Removes a required file from a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			file_id (str): Required attached-file identifier.
		
		Returns:
			Dict[str, Any]: Provider deletion result.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.file_id = file_id
			self.client = OpenAI( api_key=self.api_key, )
			self.response = self.client.vector_stores.files.delete( self.file_id,
				vector_store_id=self.store_id, )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return { 'id': getattr( self.response, 'id', self.file_id ),
				'deleted': getattr( self.response, 'deleted', False ),
				'object': getattr( self.response, 'object', '' ), }
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'delete_file( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file_content( self, store_id: str, file_id: str ) -> Any:
		"""Retrieve attached-file content.
		
		Purpose:
			Retrieves parsed content for a required file attached to a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			file_id (str): Required attached-file identifier.
		
		Returns:
			Any: Provider file-content response.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_id', file_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.file_id = file_id
			self.client = OpenAI( api_key=self.api_key, )
			self.response = self.client.vector_stores.files.content( self.file_id,
				vector_store_id=self.store_id, )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file_content( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def create_file_batch( self, store_id: str, file_ids: List[ str ],
		attributes: Optional[ Dict[ str, Any ] ] = None,
		chunking_strategy: Optional[ Dict[ str, Any ] ] = None ) -> Dict[ str, Any ]:
		"""Create a file batch.
		
		Purpose:
			Creates a file batch for required files in a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			file_ids (List[str]): Required OpenAI file identifiers.
			attributes (Optional[Dict[str, Any]]): Optional common file attributes.
			chunking_strategy (Optional[Dict[str, Any]]): Optional chunking strategy.
		
		Returns:
			Dict[str, Any]: Created file-batch metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'file_ids', file_ids )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.file_ids = file_ids
			self.attributes = attributes if attributes is not None else { }
			self.chunking_strategy = (chunking_strategy if chunking_strategy is not None else { })
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'file_ids': self.file_ids, }
			
			if self.attributes:
				self.request[ 'attributes' ] = self.attributes
			
			if self.chunking_strategy:
				self.request[ 'chunking_strategy' ] = self.chunking_strategy
			
			self.response = self.client.vector_stores.file_batches.create( self.store_id,
				**self.request )
			self.file_batch = self.get_file_batch( self.response )
			self.batch_id = self.file_batch.get( 'id', '', )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'create_file_batch( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ]:
		"""Retrieve a file batch.
		
		Purpose:
			Retrieves metadata for a required file batch in a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			batch_id (str): Required file-batch identifier.
		
		Returns:
			Dict[str, Any]: File-batch metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'batch_id', batch_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.batch_id = batch_id
			self.client = OpenAI( api_key=self.api_key, )
			self.response = self.client.vector_stores.file_batches.retrieve( self.batch_id,
				vector_store_id=self.store_id, )
			self.file_batch = self.get_file_batch( self.response )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file_batch( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def list_file_batch_files( self, store_id: str, batch_id: str, limit: int = 100,
		order: str = 'desc', after: str = '', before: str = '', filter: str = '' ) -> List[
		Dict[ str, Any ] ]:
		"""List file-batch files.
		
		Purpose:
			Lists files associated with a required file batch and vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			batch_id (str): Required file-batch identifier.
			limit (int): Maximum number of files returned.
			order (str): Result order.
			after (str): Optional after cursor.
			before (str): Optional before cursor.
			filter (str): Optional file-status filter.
		
		Returns:
			List[Dict[str, Any]]: File-batch attached-file metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'batch_id', batch_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.batch_id = batch_id
			self.limit = limit
			self.order = order
			self.after = after
			self.before = before
			self.filter = filter
			self.client = OpenAI( api_key=self.api_key, )
			self.request = { 'limit': self.limit, 'order': self.order, }
			
			if self.after:
				self.request[ 'after' ] = self.after
			
			if self.before:
				self.request[ 'before' ] = self.before
			
			if self.filter:
				self.request[ 'filter' ] = self.filter
			
			self.response = self.client.vector_stores.file_batches.list_files( self.batch_id,
				vector_store_id=self.store_id, **self.request )
			self.items = getattr( self.response, 'data', [ ], ) or [ ]
			self.vector_files = [ self.get_vector_file( item ) for item in self.items ]
			return self.vector_files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_file_batch_files( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def cancel_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ]:
		"""Cancel a file batch.
		
		Purpose:
			Cancels a required file batch in a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			batch_id (str): Required file-batch identifier.
		
		Returns:
			Dict[str, Any]: Updated file-batch metadata.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'batch_id', batch_id )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.batch_id = batch_id
			self.client = OpenAI( api_key=self.api_key, )
			self.response = self.client.vector_stores.file_batches.cancel( self.batch_id,
				vector_store_id=self.store_id, )
			self.file_batch = self.get_file_batch( self.response )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'cancel_file_batch( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def search( self, store_id: str, query: str, max_num_results: int = 10,
		filters: Optional[ Dict[ str, Any ] ] = None,
		ranking_options: Optional[ Dict[ str, Any ] ] = None, rewrite_query: bool = False ) -> (
			List)[
		Dict[ str, Any ] ]:
		"""Search a vector store.
		
		Purpose:
			Provides the application-compatible alias for native vector-store search.
		
		Args:
			store_id (str): Required vector-store identifier.
			query (str): Required semantic-search query.
			max_num_results (int): Maximum number of results.
			filters (Optional[Dict[str, Any]]): Optional attribute filters.
			ranking_options (Optional[Dict[str, Any]]): Optional ranking configuration.
			rewrite_query (bool): Indicates whether the provider may rewrite the query.
		
		Returns:
			List[Dict[str, Any]]: Native vector-store search results.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			self.store_id = store_id
			self.query = query
			self.max_search_results = max_num_results
			self.filters = filters if filters is not None else { }
			self.ranking_options = (ranking_options if ranking_options is not None else { })
			self.rewrite_query = rewrite_query
			return self.search_store( self.store_id, self.query, self.max_search_results,
				self.filters, self.ranking_options, self.rewrite_query, )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def search_store( self, store_id: str, query: str, max_num_results: int = 10,
		filters: Optional[ Dict[ str, Any ] ] = None,
		ranking_options: Optional[ Dict[ str, Any ] ] = None, rewrite_query: bool = False ) -> (
			List)[
		Dict[ str, Any ] ]:
		"""Search a vector store.
		
		Purpose:
			Executes native semantic search against a required vector store.
		
		Args:
			store_id (str): Required vector-store identifier.
			query (str): Required semantic-search query.
			max_num_results (int): Maximum number of results.
			filters (Optional[Dict[str, Any]]): Optional attribute filters.
			ranking_options (Optional[Dict[str, Any]]): Optional ranking configuration.
			rewrite_query (bool): Indicates whether the provider may rewrite the query.
		
		Returns:
			List[Dict[str, Any]]: Native vector-store search results.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'query', query )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_id = store_id
			self.query = query
			self.max_search_results = max_num_results
			self.filters = filters if filters is not None else { }
			self.ranking_options = (ranking_options if ranking_options is not None else { })
			self.rewrite_query = rewrite_query
			self.client = OpenAI( api_key=self.api_key, )
			
			if self.max_search_results < 1:
				self.max_search_results = 1
			
			if self.max_search_results > 50:
				self.max_search_results = 50
			
			self.request = { 'query': self.query, 'max_num_results': self.max_search_results,
				'rewrite_query': self.rewrite_query, }
			
			if self.filters:
				self.request[ 'filters' ] = self.filters
			
			if self.ranking_options:
				self.request[ 'ranking_options' ] = self.ranking_options
			
			self.response = self.client.vector_stores.search( self.store_id, **self.request )
			self.search_results = self.get_search_results( self.response )
			return self.search_results
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search_store( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def answer_with_file_search( self, store_ids: List[ str ], prompt: str,
		model: str = 'gpt-4o-mini', max_num_results: int = 10, instructions: str = '' ) -> str:
		"""Answer with file search.
		
		Purpose:
			Answers a required prompt using the Responses API file-search tool across required
			vector stores.
		
		Args:
			store_ids (List[str]): Required vector-store identifiers.
			prompt (str): Required user prompt.
			model (str): OpenAI model used to generate the answer.
			max_num_results (int): Maximum number of retrieved results.
			instructions (str): Optional system or developer instructions.
		
		Returns:
			str: Generated answer text.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_ids', store_ids )
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'OPENAI_API_KEY', self.api_key )
			self.store_ids = store_ids
			self.prompt = prompt
			self.model = model
			self.max_search_results = max_num_results
			self.instructions = instructions
			self.client = OpenAI( api_key=self.api_key, )
			
			if self.max_search_results < 1:
				self.max_search_results = 1
			
			if self.max_search_results > 50:
				self.max_search_results = 50
			
			self.input = [ { 'role': 'user',
				'content': [ { 'type': 'input_text', 'text': self.prompt, }, ], }, ]
			self.request = { 'model': self.model, 'input': self.input, 'tools': [
				{ 'type': 'file_search', 'vector_store_ids': self.store_ids,
					'max_num_results': self.max_search_results, }, ], }
			
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			self.response = self.client.responses.create( **self.request )
			self.output_text = getattr( self.response, 'output_text', '', )
			
			if self.output_text:
				return self.output_text
			
			self.output_text = str( self.response )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'answer_with_file_search( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, store_ids: List[ str ],
		prompt: str = ('Summarize the most relevant information available in the '
		               'selected vector stores.'), model: str = 'gpt-4o-mini',
		max_num_results: int = 10, instructions: str = '' ) -> str:
		"""Survey vector stores.
		
		Purpose:
			Generates a summary across required vector stores through the Responses API
			file-search tool.
		
		Args:
			store_ids (List[str]): Required vector-store identifiers.
			prompt (str): Summary prompt.
			model (str): OpenAI model used to generate the summary.
			max_num_results (int): Maximum number of retrieved results.
			instructions (str): Optional system or developer instructions.
		
		Returns:
			str: Generated vector-store summary.
		
		Raises:
			Error: Re-raised after the exception is logged.
		"""
		try:
			throw_if( 'store_ids', store_ids )
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.store_ids = store_ids
			self.prompt = prompt
			self.model = model
			self.max_search_results = max_num_results
			self.instructions = instructions
			self.output_text = self.answer_with_file_search( self.store_ids, self.prompt,
				self.model, self.max_search_results, self.instructions, )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'survey( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ]:
		"""Return member names.
		
		Purpose:
			Returns public members exposed by the OpenAI VectorStores wrapper.
		
		Returns:
			List[str]: Public member names.
		"""
		return [ 'api_key', 'client', 'name', 'description', 'store_id', 'file_id', 'file_ids',
			'batch_id', 'model', 'query', 'prompt', 'instructions', 'max_search_results',
			'response', 'vector_store', 'vector_stores', 'vector_file', 'vector_files',
			'file_batch', 'search_results', 'output_text', 'request', 'collections',
			'model_options', 'ranker_options', 'chunking_strategy_options', 'get_vector_store',
			'get_vector_file', 'get_file_batch', 'get_search_results', 'create', 'list_stores',
			'list', 'retrieve', 'update', 'delete', 'attach_file', 'list_files', 'retrieve_file',
			'update_file', 'delete_file', 'retrieve_file_content', 'create_file_batch',
			'retrieve_file_batch', 'list_file_batch_files', 'cancel_file_batch', 'search',
			'search_store', 'answer_with_file_search', 'survey', ]
