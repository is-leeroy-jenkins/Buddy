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

def throw_if( name: str, value: object ):
	if value is None:
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
	
	def __init__( self ):
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
	
	    Purpose
	    ___________
	    Class used for interacting with OpenAI's  ChatGPT API
	
	
	    Parameters
	    ------------
	    number: int=1
	    temperature: float=0.8
	    top_p: float=0.9
	    frequency: float=0.0
	    presence: float=0.0
	    max_tokens: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    number,
	    temperature,
	    top_percent,
	    frequency_penalty,
	    presence_penalty,
	    store,
	    stream,
	    maximum_completion_tokens,
	    api_key,
	    client,
	    model,
	    embedding,
	    response,
	    modalities,
	    stops,
	    content,
	    prompt,
	    response,
	    file_path,
	    path,
	    messages,
	    image_url,
	    response_format,
	    tools,
	    vector_store_ids
	    
	    Properties:
	    -----------
	    model_options - List[ str ]
	
	    Methods
	    ------------
	    generate_text( self, prompt: str ) -> str:
	    analyze_image( self, prompt: str, url: str ) -> str:
	    summarize_document( self, prompt: str, path: str ) -> str
	    search_web( self, prompt: str ) -> str
	    search_files( self, prompt: str ) -> str
	    dump( self ) -> str
	    get_data( self ) -> { }


    """
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	previous_id: Optional[ str ]
	parallel_tools: Optional[ bool ]
	max_tools = Optional[ int ]
	input: Optional[ List[ Dict[ str, str ] ] | str ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	reasoning_effort: Optional[ Dict[ str, str ] ]
	image_url: Optional[ str ]
	image_path: Optional[ str ]
	file_url: Optional[ str ]
	file_path: Optional[ str ]
	allowed_domains: Optional[ List[ str ] ]
	max_tools = Optional[ int ]
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
			top_p: float = None, presense: float = None, store: bool = None, stream: bool = None,
			stops: List[ str ] = None, response_format: Dict[ str, str ] = None, number: int = None,
			instruct: str = None, context: List[ Dict[ str, str ] ] = None,
			allowed_domains: List[ str ] = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None, max_tools: int = None,
			tool_choice: str = None, file_path: str = None, background: bool = None,
			is_parallel: bool = None, max_tokens: int = None, frequency: float = None,
			input: List[ Dict[ str, Any ] ] = None, file_ids: List[ str ] = None,
			previous_id: str = None, reasoning: Dict[ str, str ] | str = None,
			output_text: str = None, max_search_results: int = None, content: str = None,
			vector_store_ids: List[ str ] = None ):
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
		self.presence_penalty = presense
		self.max_tokens = max_tokens
		self.context = context if context is not None else [ ]
		self.stream = stream
		self.store = store
		self.instructions = instruct
		self.stops = stops if stops is not None else [ ]
		self.background = background
		self.conetxt = self.context
		self.input = input if input is not None else [ ]
		self.include = include if include is not None else [ ]
		self.output_text = output_text
		self.max_tools = max_tools
		self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
		self.file_ids = file_ids if file_ids is not None else [ ]
		self.tools = tools if tools is not None else [ ]
		self.previous_id = previous_id
		self.reasoning = reasoning
		self.parallel_tools = is_parallel
		self.tool_choice = tool_choice
		self.response = None
		self.file = None
		self.file_url = file_path
		self.image_url = None
		self.content = content
		self.output_text = output_text
		self.max_search_results = max_search_results
		self.purpose = None
		self.vector_stores = \
			{
					'Guidance': 'vs_712r5W5833G6aLxIYIbuvVcK',
					'Appropriations': 'vs_8fEoYp1zVvk5D8atfWLbEupN',
			}
		self.files = \
			{
					'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
					'SF133.csv': 'file-WT2h2F5SNxqK2CxyAMSDg6',
					'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
					'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn'
			}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [
				'gpt-5.4',
				'gpt-5.4-mini',
				'gpt-5.4-nano',
				'gpt-5',
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o',
				'gpt-4o-mini' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of the includeable options

		'''
		return [ 'file_search_call.results',
		         'web_search_call.results',
		         'web_search_call.action.sources',
		         'message.input_image.image_url',
		         'computer_call_output.output.image_url',
		         'code_interpreter_call.outputs',
		         'reasoning.encrypted_content',
		         'message.output_text.logprobs' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'web_search', 'image_generation', 'file_search',
		         'code_interpreter', 'computer_use_preview' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'auto', 'required', 'none' ]
	
	@property
	def purpose_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			--------
			A List[ str ] of file purposes

		'''
		return [ 'assistants', 'batch', 'fine-tune', 'vision',
		         'user_data', 'evals' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			--------
			A List[ str ] of file purposes

		'''
		return [ 'text', 'json_object', 'json_schema' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of reasoning effort options

		'''
		return [ 'low', 'medium', 'high', 'minimal', 'xhigh' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			--------
			A List[ str ] of file purposes

		'''
		return [ 'text', 'auto', 'image', 'audio' ]
	
	def _build_reasoning( self, reasoning: str | Dict[ str, str ] = None ) -> Dict[  str, str ] | None:
		"""
	
	        Purpose
	        _______
	        Creates a Responses API reasoning object
	        from a string or dictionary input
	
	
	        Parameters
	        ----------
	        reasoning: str | Dict[ str, str ]
	
	
	        Returns
	        -------
	        Dict[ str, str ] | None

        """
		try:
			if reasoning is None:
				return None
			
			if isinstance( reasoning, dict ):
				return reasoning if len( reasoning ) > 0 else None
			
			if isinstance( reasoning, str ) and reasoning.strip( ):
				return { 'effort': reasoning.strip( ) }
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = '_build_reasoning( self, reasoning: str | Dict[ str, str ] )'
			raise exception
	
	def _build_input( self, prompt: str, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""
	
	        Purpose
	        _______
	        Creates the Responses API input payload
	        for text generation requests
	
	
	        Parameters
	        ----------
	        prompt: str
	        context: List[ Dict[ str, str ] ]
	        input_data: List[ Dict[ str, Any ] ]
	
	
	        Returns
	        -------
	        List[ Dict[ str, Any ] ]

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
					
					self.messages.append( item )
			
			self.messages.append(
				{
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
			exception.method = ('_build_input( self, prompt: str, '
			                    'context: List[ Dict[ str, str ] ]=None, '
			                    'input_data: List[ Dict[ str, Any ] ]=None )')
			raise exception
	
	def _build_tools( self, tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None ) -> List[ Dict[ str, Any ] ] | None:
		"""
	
	        Purpose
	        _______
	        Normalizes built-in Responses API tool objects
	        for text generation requests
	
	
	        Parameters
	        ----------
	        tools: List[ Dict[ str, Any ] ]
	        allowed_domains: List[ str ]
	
	
	        Returns
	        -------
	        List[ Dict[ str, Any ] ] | None

        """
		try:
			self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
			if tools is None or len( tools ) == 0:
				return None
			
			self.built_tools = [ ]
			for tool in tools:
				if not isinstance( tool, dict ):
					continue
				
				self.tool = dict( tool )
				self.tool_type = self.tool.get( 'type' )
				if not self.tool_type:
					continue
				
				if self.tool_type == 'web_search' and len( self.allowed_domains ) > 0:
					self.filters = self.tool.get( 'filters' )
					if not isinstance( self.filters, dict ):
						self.filters = { }
					
					self.filters[ 'allowed_domains' ] = self.allowed_domains
					self.tool[ 'filters' ] = self.filters
				
				if self.tool_type == 'file_search' and len( self.vector_store_ids ) > 0:
					self.tool[ 'vector_store_ids' ] = self.vector_store_ids
				
				self.built_tools.append( self.tool )
			return self.built_tools if len( self.built_tools ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = ('_build_tools( self, tools: List[ Dict[ str, Any ] ]=None, '
			                    'allowed_domains: List[ str ]=None )')
			raise exception
	
	def _build_request( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None, store: bool = None,
			stream: bool = None, instruct: str = None, background: bool = False, reasoning: str = None,
			include: List[ str ] = None, tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None, previous_id: str = None,
			tool_choice: str = None, is_parallel: bool = None,
			context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> Dict[ str, Any ]:
		"""
	
	        Purpose
	        _______
	        Creates a normalized Responses API request payload
	        for text generation
	
	
	        Parameters
	        ----------
	        prompt: str
	        model: str
	
	
	        Returns
	        -------
	        Dict[ str, Any ]

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
			self.response_format = format if format is not None else { }
			self.max_tools = max_tools
			self.include = include if include is not None else [ ]
			self.previous_id = previous_id if isinstance( previous_id, str ) else None
			self.tool_choice = tool_choice.strip( ) if isinstance( tool_choice, str ) else None
			self.parallel_tools = is_parallel
			self.reasoning = self._build_reasoning( reasoning )
			self.tools = self._build_tools( tools=tools, allowed_domains=allowed_domains )
			self.input = self._build_input( prompt=prompt, context=context, input_data=input_data )
			self.request = \
				{
						'model': self.model,
						'input': self.input,
				}
			
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			if self.reasoning is not None:
				self.request[ 'reasoning' ] = self.reasoning
			
			if isinstance( self.max_tokens, int ) and self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if self.temperature is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'top_p' ] = self.top_percent
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			if self.stream is not None:
				self.request[ 'stream' ] = self.stream
			
			if self.background is not None:
				self.request[ 'background' ] = self.background
			
			if self.include is not None and len( self.include ) > 0:
				self.request[ 'include' ] = self.include
			
			if self.tools is not None and len( self.tools ) > 0:
				self.request[ 'tools' ] = self.tools
			
			if self.tool_choice:
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.parallel_tools is not None:
				self.request[ 'parallel_tool_calls' ] = self.parallel_tools
			
			if self.previous_id and self.previous_id.strip( ):
				self.request[ 'previous_response_id' ] = self.previous_id.strip( )
			
			if isinstance( self.max_tools, int ) and self.max_tools > 0:
				self.request[ 'max_tool_calls' ] = self.max_tools
			
			if self.response_format is not None and len( self.response_format ) > 0:
				self.request[ 'text' ] = self.response_format
			
			return self.request
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = ('_build_request( self, prompt: str, model: str, '
			                    'temperature: float=None, format: Dict[ str, Any ]=None )')
			raise exception
	
	def get_output_text( self ) -> str | None:
		"""
	
	        Purpose
	        _______
	        Returns the text output from the last
	        Responses API call
	
	
	        Parameters
	        ----------
	        self
	
	
	        Returns
	        -------
	        str | None

        """
		try:
			if self.response is None:
				return None
			
			self.output_text = getattr( self.response, 'output_text', None )
			if self.output_text:
				return self.output_text
			
			if hasattr( self.response, 'output' ) and self.response.output:
				for item in self.response.output:
					if getattr( item, 'type', None ) != 'message':
						continue
					
					if not hasattr( item, 'content' ) or item.content is None:
						continue
					
					for block in item.content:
						if getattr( block, 'type', None ) == 'output_text':
							self.output_text = getattr( block, 'text', None )
							if self.output_text:
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
	
	        Purpose
	        _______
	        Returns usage metadata from the last
	        Responses API call
	
	
	        Parameters
	        ----------
	        self
	
	
	        Returns
	        -------
	        Any

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
	
	def completion( self, prompt_id: str, prompt_version: str = None, model: str = None,
			user_input: str = None, temperature: float = None, format: Dict[ str, Any ] = None,
			top_p: float = None, frequency: float = None, max_tools: int = None,
			presence: float = None, max_tokens: int = None, store: bool = None,
			stream: bool = None, instruct: str = None, background: bool = False,
			reasoning: str = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None, allowed_domains: List[ str ] = None,
			previous_id: str = None, tool_choice: str = None,
			is_parallel: bool = None, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> Response | None:
		"""
		
				Purpose
				_______
				Creates a Responses API completion using a stored OpenAI platform
				prompt identified by prompt_id and optional prompt_version.
		
		
				Parameters
				----------
				prompt_id: str
				prompt_version: str=None
				model: str=None
				user_input: str=None
				temperature: float=None
				format: Dict[ str, Any ]=None
				top_p: float=None
				frequency: float=None
				max_tools: int=None
				presence: float=None
				max_tokens: int=None
				store: bool=None
				stream: bool=None
				instruct: str=None
				background: bool=False
				reasoning: str=None
				include: List[ str ]=None
				tools: List[ Dict[ str, Any ] ]=None
				allowed_domains: List[ str ]=None
				previous_id: str=None
				tool_choice: str=None
				is_parallel: bool=None
				context: List[ Dict[ str, str ] ]=None
				input_data: List[ Dict[ str, Any ] ]=None
		
		
				Returns
				-------
				Response | None
	
			"""
		try:
			throw_if( 'prompt_id', prompt_id )
			self.client = OpenAI( api_key=self.api_key )
			self.prompt_id = str( prompt_id ).strip( )
			self.prompt_version = str( prompt_version ).strip( ) \
				if isinstance( prompt_version, str ) and prompt_version.strip( ) else None
			self.model = model if isinstance( model, str ) and model.strip( ) else None
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.background = background
			self.instructions = instruct
			self.response_format = format if format is not None else { }
			self.max_tools = max_tools
			self.include = include if include is not None else [ ]
			self.previous_id = previous_id if isinstance( previous_id, str ) else None
			self.tool_choice = tool_choice.strip( ) if isinstance( tool_choice, str ) else None
			self.parallel_tools = is_parallel
			self.reasoning = self._build_reasoning( reasoning )
			self.tools = self._build_tools( tools=tools, allowed_domains=allowed_domains )
			self.request = {
					'prompt': {
							'id': self.prompt_id,
					},
			}
			
			if self.prompt_version:
				self.request[ 'prompt' ][ 'version' ] = self.prompt_version
			
			if self.model:
				self.request[ 'model' ] = self.model
			
			if user_input is not None and str( user_input ).strip( ):
				self.input = self._build_input( prompt=user_input, context=context,
					input_data=input_data )
				self.request[ 'input' ] = self.input
			elif input_data is not None and len( input_data ) > 0:
				self.request[ 'input' ] = input_data
			elif context is not None and len( context ) > 0:
				self.request[ 'input' ] = context
			
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			if self.reasoning is not None:
				self.request[ 'reasoning' ] = self.reasoning
			
			if isinstance( self.max_tokens, int ) and self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if self.temperature is not None and self.model is not None \
					and not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None and self.model is not None \
					and not self.model.startswith( 'gpt-5' ):
				self.request[ 'top_p' ] = self.top_percent
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			if self.stream is not None:
				self.request[ 'stream' ] = self.stream
			
			if self.background is not None:
				self.request[ 'background' ] = self.background
			
			if self.include is not None and len( self.include ) > 0:
				self.request[ 'include' ] = self.include
			
			if self.tools is not None and len( self.tools ) > 0:
				self.request[ 'tools' ] = self.tools
			
			if self.tool_choice:
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.parallel_tools is not None:
				self.request[ 'parallel_tool_calls' ] = self.parallel_tools
			
			if self.previous_id and self.previous_id.strip( ):
				self.request[ 'previous_response_id' ] = self.previous_id.strip( )
			
			if isinstance( self.max_tools, int ) and self.max_tools > 0:
				self.request[ 'max_tool_calls' ] = self.max_tools
			
			if self.response_format is not None and len( self.response_format ) > 0:
				self.request[ 'text' ] = self.response_format
			
			self.response = self.client.responses.create( **self.request )
			self.previous_id = getattr( self.response, 'id', None )
			self.output_text = self.get_output_text( )
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'completion( self, prompt_id: str, prompt_version: str=None )'
			raise exception
	
	def generate_text( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None, store: bool = None,
			stream: bool = None, instruct: str = None, background: bool = False, reasoning: str = None,
			include: List[ str ] = None, tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None, previous_id: str = None,
			tool_choice: str = None, is_parallel: bool = None,
			context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> str | None:
		"""
	
	        Purpose
	        _______
	        Generates a chat completion given a prompt
	
	
	        Parameters
	        ----------
	        prompt: str
	
	
	        Returns
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.client = OpenAI( api_key=self.api_key )
			self.request = self._build_request( prompt=prompt, model=model,
				temperature=temperature, format=format, top_p=top_p, frequency=frequency,
				max_tools=max_tools, presence=presence, max_tokens=max_tokens, store=store,
				stream=stream, instruct=instruct, background=background, reasoning=reasoning,
				include=include, tools=tools, allowed_domains=allowed_domains,
				previous_id=previous_id, tool_choice=tool_choice, is_parallel=is_parallel,
				context=context, input_data=input_data )
			self.response = self.client.responses.create( **self.request )
			self.previous_id = getattr( self.response, 'id', None )
			self.output_text = self.get_output_text( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			raise exception

class Images( GPT ):
	"""
	
	    Purpose
	    ___________
	    Class used for generating images OpenAI's Images API and dall-e-2
	
	
	    Parameters
	    ------------
	    n: int=1
	    temperature: float=0.8
	    top_p: float=0.9
	    frequency: float=0.0
	    presence: float=0.0
	    max_tokens: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    self.api_key, self.client, self.small_model,  self.embedding,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.prompt, self.response, self.completion, self.file, self.path,
	    self.messages, self.image_url, self.response_format,
	    self.tools, self.vector_store_ids, self.input_text, self.image_url
	
		Properties:
		----------
	    detail_options( self ) -> list[ str ]
	    format_options( self ) -> list[ str ]
	    size_options( self ) -> list[ str ]
	    model_options( self ) -> str
	    
	    Methods
	    ------------
	    generate( self, path: str ) -> str
	    analyze( self, path: str, text: str ) -> str

    """
	quality: Optional[ str ]
	detail: Optional[ str ]
	size: Optional[ str ]
	previous_id: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	parallel_tools: Optional[ bool ]
	input: Optional[ List[ Dict[ str, str ] ] | str ]
	instructions: Optional[ str ]
	max_tools: Optional[ int ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	messages: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	image_url: Optional[ str ]
	image_path: Optional[ str ]
	file_url: Optional[ str ]
	file_path: Optional[ str ]
	style: Optional[ str ]
	allowed_domains: Optional[ List[ str ] ]
	response_format: Optional[ str ]
	mime_format: Optional[ str ]
	background: Optional[ bool ]
	backcolor: Optional[ str ]
	compression: Optional[ float ]
	
	def __init__( self, prompt: str = None, model: str = 'gpt-image-1', temperature: float = None,
			top_p: float = None, presence: float = None, frequency: float = None,
			max_tokens: int = None, store: bool = None, stream: bool = False, backcolor: str = None,
			instruct: str = None, background: bool = None, number: int = None,
			image_format: str = None, include: List[ Dict[ str, str ] ] = None,
			tools: List[ Dict[ str, str ] ] = None, max_tools: int = None,
			respose_format: Dict[
				str, str ] = None, tool_choice: str = None, image_path: str = None,
			is_parallel: bool = None, input: List[
				Dict[ str, str ] ] = None, previous_id: str = None,
			reasoning: Dict[ str, str ] = None, input_text: str = None, image_url: str = None,
			content: List[ Dict[ str, str ] ] = None, quality: str = None, size: str = None,
			detail: str = None, style: str = None, compression: float = None ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.number = number
		self.previous_id = previous_id
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instruct = instruct
		self.max_tools = max_tools
		self.reasoning = reasoning
		self.tools = tools
		self.tool_choice = tool_choice
		self.input_text = input_text
		self.input = input
		self.content = content
		self.background = background
		self.backcolor = backcolor
		self.input_text = prompt
		self.image_path = image_path
		self.image_url = image_url
		self.include = include
		self.quality = quality
		self.detail = detail
		self.size = size
		self.style = style
		self.compression = compression
		self.response_format = respose_format
		self.mime_format = image_format
		self.parallel_tools = is_parallel
	
	@property
	def style_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Methods that returns a list of style options for dall-e-3

        '''
		return [ 'vivid',
		         'natural', ]
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [ 'gpt-image-1',
		         'gpt-image-1-mini',
		         'gpt-image-1.5',
		         'gpt-5',
		         'gpt-5.2',
		         'gpt-5-mini',
		         'gpt-5-nano',
		         'gpt-5-turbo',
		         'gpt-4.1',
		         'gpt-4.1-mini',
		         'gpt-4o',
		         'gpt-4o-mini',
		         "dall-e-2",
		         "dall-e-3",
		         "gpt-image-1",
		         "gpt-image-1-mini" ]
	
	@property
	def size_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        --------
	        Method that returns a  list of sizes

	        - For gpt-image-1, the size must be one of '1024x1024', '1536x1024' (landscape),
	        '1024x1536' (portrait), or 'auto' (default value).

	        - For dall-e-2, the size must be one of '256x256', '512x512', or '1024x1024'

	        - For dall-e-3, the sie must be one of '1024x1024', '1792x1024', or '1024x1792'

        '''
		return [ 'auto',
		         '256x256',
		         '512x512',
		         '1024x1024',
		         '1792x1024',
		         '1024x1792',
		         '1536x1024', '1024x1536' ]
	
	@property
	def format_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ________
	        Method that returns a  list of format options

        '''
		return [ 'url', 'b64_json' ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ________
	        Method that returns a  list of format options

        '''
		return [ 'png',
		         'jpeg',
		         'webp' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of the includeable options

		'''
		return [ 'file_search_call.results',
		         'web_search_call.results',
		         'web_search_call.action.sources',
		         'message.input_image.image_url',
		         'computer_call_output.output.image_url',
		         'code_interpreter_call.outputs',
		         'reasoning.encrypted_content',
		         'message.output_text.logprobs' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'web_search',
		         'image_generation',
		         'file_search',
		         'code_interpreter',
		         'computer_use_preview' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'auto', 'required', 'none' ]
	
	@property
	def backcolor_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ________
	        Method that returns a  list of format options

        '''
		return [ 'auto', 'transparent', 'opaque', ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Method that returns a  list of quality options

        '''
		return [ 'auto', 'low', 'medium', 'high', 'standard', 'hd' ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Method that returns a  list of detail options

        '''
		return [ 'auto', 'low', 'medium', 'high' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of reasoning effort options

		'''
		return [ 'low',
		         'medium',
		         'high',
		         'none',
		         'minimal',
		         'xhigh' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			--------
			A List[ str ] of file purposes

		'''
		return [ 'text', 'auto', 'image', 'audio' ]
	
	def generate( self, prompt: str, number: int = 1, model: str = 'gpt-image-1-mini',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = '.jpeg',
			compression: float = None, background: str = None ) -> str | bytes | None:
		'''
	
			Purpose
			_______
			Generates an image given a prompt
	
	
			Parameters
			----------
			prompt: str
	
	
			Returns
			-------
			str | bytes | None

		'''
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.number = number if isinstance( number, int ) and number > 0 else 1
			self.model = model
			self.size = size if isinstance( size, str ) and size.strip( ) else '1024x1024'
			self.quality = quality if isinstance( quality, str ) and quality.strip( ) else 'auto'
			self.response_format = fmt if isinstance( fmt, str ) and fmt.strip( ) else '.jpeg'
			self.client = OpenAI( api_key=self.api_key )
			self.output_format = self.response_format.lower( ).replace( '.', '' )
			if self.output_format not in [ 'png', 'jpeg', 'webp' ]:
				self.output_format = 'jpeg'
			
			self.request = \
				{
						'model': self.model,
						'prompt': self.prompt,
						'n': self.number,
						'size': self.size,
						'quality': self.quality,
						'output_format': self.output_format,
				}
			
			if isinstance( background, str ) and background.strip( ):
				self.request[ 'background' ] = background.strip( )
			
			if compression is not None and self.output_format in [ 'jpeg', 'webp' ]:
				self.output_compression = max( 0, min( 100, int( round( compression * 100 ) ) ) )
				self.request[ 'output_compression' ] = self.output_compression
			
			self.response = self.client.images.generate( **self.request )
			self.data = getattr( self.response, 'data', None )
			if self.data and len( self.data ) > 0:
				self.item = self.data[ 0 ]
				self.b64_json = getattr( self.item, 'b64_json', None )
				self.url = getattr( self.item, 'url', None )
				if self.b64_json:
					return base64.b64decode( self.b64_json )
				
				if self.url:
					return self.url
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Image'
			exception.method = ('generate( self, prompt: str, number: int=1, '
			                    'model: str="gpt-image-1-mini" ) -> str | bytes | None')
			raise exception
	
	def analyze( self, text: str, path: str, instruct: str = None,
			model: str = 'gpt-4o-mini' ) -> str | None:
		'''
	
	        Purpose:
	        ________
	        Method providing image analysis functionality given a prompt and path
	
	
	        Parameters:
	        ----------
	        text: str
	        path: str
	
	
	        Returns:
	        --------
	        str | None

        '''
		try:
			throw_if( 'text', text )
			throw_if( 'path', path )
			self.instructions = instruct if isinstance( instruct, str ) else ''
			self.input_text = text
			self.model = model
			self.file_path = path
			self.client = OpenAI( api_key=self.api_key )
			with open( self.file_path, 'rb' ) as source:
				self.file = self.client.files.create( file=source, purpose='vision' )
			
			self.input = \
				[ { 'role': 'user', 'content': [ { 'type': 'input_text', 'text': self.input_text },
				                                 { 'type': 'input_image',
				                                   'file_id': self.file.id } ] } ]
			
			self.request = \
				{
						'model': self.model,
						'input': self.input,
				}
			
			if self.instructions and self.instructions.strip( ):
				self.request[ 'instructions' ] = self.instructions.strip( )
			
			if isinstance( self.max_tokens, int ) and self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if self.temperature is not None:
				self.request[ 'temperature' ] = self.temperature
			
			if self.tool_choice is not None and str( self.tool_choice ).strip( ):
				self.request[ 'tool_choice' ] = str( self.tool_choice ).strip( )
			
			if self.include is not None and len( self.include ) > 0:
				self.request[ 'include' ] = self.include
			
			if self.stream is not None:
				self.request[ 'stream' ] = self.stream
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			self.response = self.client.responses.create( **self.request )
			return getattr( self.response, 'output_text', None )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Image'
			exception.method = 'analyze( self, text: str, path: str, instruct: str=None ) -> str | None'
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gpt-image-1-mini',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = '.jpeg',
			compression: float = None ) -> str | bytes | None:
		"""
	
	        Purpose
	        _______
	        Creates an edited image given a prompt and source image path
	
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	
	
	        Returns
	        -------
	        str | bytes | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			self.input_text = prompt
			self.file_path = path
			self.model = model
			self.size = size if isinstance( size, str ) and size.strip( ) else '1024x1024'
			self.quality = quality if isinstance( quality, str ) and quality.strip( ) else 'auto'
			self.response_format = fmt if isinstance( fmt, str ) and fmt.strip( ) else '.jpeg'
			self.client = OpenAI( api_key=self.api_key )
			self.output_format = self.response_format.lower( ).replace( '.', '' )
			if self.output_format not in [ 'png', 'jpeg', 'webp' ]:
				self.output_format = 'jpeg'
			
			self.request = \
				{
						'model': self.model,
						'prompt': self.input_text,
						'size': self.size,
						'quality': self.quality,
						'output_format': self.output_format,
						'n': self.number if isinstance( self.number, int ) and self.number > 0 else 1,
				}
			
			if compression is not None and self.output_format in [ 'jpeg', 'webp' ]:
				self.output_compression = max( 0, min( 100, int( round( compression * 100 ) ) ) )
				self.request[ 'output_compression' ] = self.output_compression
			
			with open( self.file_path, 'rb' ) as source:
				self.response = self.client.images.edit( image=source, **self.request )
			
			self.data = getattr( self.response, 'data', None )
			if self.data and len( self.data ) > 0:
				self.item = self.data[ 0 ]
				self.b64_json = getattr( self.item, 'b64_json', None )
				self.url = getattr( self.item, 'url', None )
				if self.b64_json:
					return base64.b64decode( self.b64_json )
				
				if self.url:
					return self.url
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Image'
			exception.method = 'edit( self, prompt: str, path: str, model: str="gpt-image-1-mini" )'
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method returns a list of strings representing members
	
	        Parameters:
	        ----------
	        self
	
	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [  # Attributes
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
				# Properties
				'style_options',
				'model_options',
				'detail_options',
				'format_options',
				'size_options',
				# Methods
				'generate',
				'analyze',
				'edit', ]

class TTS( ):
	"""
	
	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (TTS)
	
	
	    Parameters
	    ------------
	    input: str=None
	    model: str='gpt-4o-mini-tts'
	    format: str=None
	    instruct: str=None
	    voice: str=None
	    speed: float=None
	    file_path: str=None
	
	
	    Attributes
	    -----------
	    self.api_key
	    self.client
	    self.speed
	    self.voice
	    self.input
	    self.instructions
	    self.response
	    self.response_format
	    self.file_path
	    self.model
	
	
	    Methods
	    ------------
	    create_speech( self, text: str ) -> bytes | None

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
	
	def __init__( self, input: str = None, model: str = 'gpt-4o-mini-tts', format: str = None,
			instruct: str = None, voice: str = None, speed: float = None, file_path: str = None ):
		'''

	        Purpose:
	        --------
	        Constructor to create TTS objects

        '''
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
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of TTS model names

        '''
		return [ 'gpt-4o-mini-tts',
		         'gpt-4o-mini-tts-2025-12-15',
		         'tts-1',
		         'tts-1-hd' ]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of audio output formats

        '''
		return [ 'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm' ]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'alloy',
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
		         'cedar' ]
	
	@property
	def speed_options( self ) -> List[ float ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of floats
	        representing different audio speeds

        '''
		return [ 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0, 4.0 ]
	
	def create_speech( self, text: str, model: str = 'gpt-4o-mini-tts', format: str = 'mp3',
			speed: float = 1.0, voice: str = 'alloy', instruct: str = None,
			file_path: str = None ) -> bytes | None:
		"""
	
	        Purpose
	        _______
	        Generates audio from input text and returns
	        the audio bytes. Optionally writes them to a file.
	
	
	        Parameters
	        ----------
	        text: str
	        model: str='gpt-4o-mini-tts'
	        format: str='mp3'
	        speed: float=1.0
	        voice: str='alloy'
	        instruct: str=None
	        file_path: str=None
	
	
	        Returns
	        -------
	        bytes | None

        """
		try:
			throw_if( 'text', text )
			self.input = text
			self.model = model or 'gpt-4o-mini-tts'
			self.response_format = format or 'mp3'
			self.voice = voice or 'alloy'
			self.speed = speed if speed is not None else 1.0
			self.instructions = instruct
			self.file_path = file_path
			self.client = OpenAI( api_key=self.api_key )
			self.response = None
			
			with tempfile.NamedTemporaryFile( suffix=f'.{self.response_format}', delete=False ) as tmp:
				temp_path = tmp.name
			
			try:
				self.request = {
						'model': self.model,
						'voice': self.voice,
						'input': self.input,
						'response_format': self.response_format,
						'speed': self.speed, }
				
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
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method returns a list of strings representing members
	
	        Parameters:
	        ----------
	        self
	
	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ 'input',
		         'file_path',
		         'voice',
		         'client',
		         'response_format',
		         'speed',
		         'model',
		         'instructions',
		         'response',
		         'create_speech' ]

class Transcription( GPT ):
	"""
	
	    Purpose
	    ___________
	    Class used for interacting with OpenAI's transcription API
	
	
	    Parameters
	    ------------
	    model: str='gpt-4o-transcribe'
	    temperature: float=None
	    prompt: str=None
	    number: int=None
	    top_p: float=None
	    frequency: float=None
	    presence: float=None
	    max_tokens: int=None
	    stream: bool=None
	    store: bool=None
	    language: str=None
	    instruct: str=None
	    format: str=None
	    background: bool=None
	    messages: List[ Dict[ str, str ] ]=None
	    stops: List[ str ]=None
	
	
	    Attributes
	    -----------
	    self.api_key
	    self.prompt
	    self.temperature
	    self.top_percent
	    self.frequency_penalty
	    self.presence_penalty
	    self.max_tokens
	    self.stream
	    self.response_format
	    self.background
	    self.message
	    self.stops
	    self.store
	    self.language
	    self.instructions
	    self.model
	    self.number
	    self.input_text
	    self.audio_file
	    self.transcript
	    self.response

    """
	client: Optional[ OpenAI ]
	language: Optional[ str ]
	instructions: Optional[ str ]
	
	def __init__( self, model: str = 'gpt-4o-transcribe', temperature: float = None, prompt: str = None,
			number: int = None, top_p: float = None, frequency: float = None, presence: float = None,
			max_tokens: int = None, stream: bool = None, store: bool = None, language: str = None,
			instruct: str = None, format: str = None, background: bool = None,
			messages: List[ Dict[ str, str ] ] = None, stops: List[ str ] = None ):
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
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of transcription model names

        '''
		return [ 'gpt-4o-transcribe',
		         'gpt-4o-mini-transcribe',
		         'gpt-4o-mini-transcribe-2025-12-15',
		         'whisper-1',
		         'gpt-4o-transcribe-diarize' ]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of supported input audio formats

        '''
		return [ 'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of ISO-639-1 language codes
	        supported for UI selection and API submission

        '''
		return [ 'en',
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
		         'th' ]
	
	@property
	def language_labels( self ) -> Dict[ str, str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method that returns a mapping of ISO-639-1 language codes
	        to human-readable labels for UI display

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
				'th': 'Thai', }
	
	@property
	def response_format_options( self ) -> Dict[ str, List[ str ] ]:
		'''
		
			Returns:
			--------
			Dict[ str, List[ str ] ]

		'''
		return {
				'whisper-1': [ 'json', 'text', 'srt', 'verbose_json', 'vtt' ],
				'gpt-4o-transcribe': [ 'json', 'text' ],
				'gpt-4o-mini-transcribe': [ 'json', 'text' ],
				'gpt-4o-mini-transcribe-2025-12-15': [ 'json', 'text' ],
				'gpt-4o-transcribe-diarize': [ 'json', 'text', 'diarized_json' ], }
	
	def transcribe( self, path: str, model: str = 'gpt-4o-transcribe', language: str = None,
			prompt: str = None, format: str = None, temperature: float = None ) -> str | None:
		"""
		
			Purpose:
			----------
			Transcribes audio into the input language.
		
		
			Parameters:
			-----------
			path: str
			model: str='gpt-4o-transcribe'
			language: str=None
			prompt: str=None
			format: str=None
			temperature: float=None
		
		
			Returns:
			--------
			str | None
        
        """
		try:
			throw_if( 'path', path )
			self.model = model or 'gpt-4o-transcribe'
			self.language = language
			self.prompt = prompt
			self.response_format = format
			self.temperature = temperature
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = {
					'model': self.model, }
			
			if self.language:
				self.request[ 'language' ] = self.language
			
			if self.prompt:
				self.request[ 'prompt' ] = self.prompt
			
			if self.response_format:
				self.request[ 'response_format' ] = self.response_format
			
			if self.temperature is not None and self.model == 'whisper-1':
				self.request[ 'temperature' ] = self.temperature
			
			with open( path, 'rb' ) as self.audio_file:
				self.response = self.client.audio.transcriptions.create(
					file=self.audio_file,
					**self.request )
			
			self.transcript = getattr( self.response, 'text', None )
			if isinstance( self.response, str ):
				self.transcript = self.response
			
			if self.transcript is not None:
				return self.transcript
			
			if hasattr( self.response, 'text' ):
				return self.response.text
			
			return str( self.response )
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
	        Method returns a list of strings representing members
	
	        Parameters:
	        ----------
	        self
	
	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ 'number',
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
		         'model' ]

class Translation( GPT ):
	"""

	    Purpose
	    ___________
	    Class used for interacting with OpenAI's translation API
	
	
	    Parameters
	    ------------
	    model: str='whisper-1'
	    temperature: float=None
	    top_p: float=None
	    frequency: float=None
	    presence: float=None
	    max_tokens: int=None
	    store: bool=None
	    stream: bool=None
	    instruct: str=None
	    audio_file: str=None
	    format: str=None
	    language: str=None

    """
	client: Optional[ OpenAI ]
	target_language: Optional[ str ]
	response_format: Optional[ str ]
	
	def __init__( self, model: str = 'whisper-1', temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None, store: bool = None,
			stream: bool = None, instruct: str = None, audio_file: str = None, format: str = None,
			language: str = None ):
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
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of translation model names

        '''
		return [ 'whisper-1' ]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of supported input audio formats

        '''
		return [ 'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm' ]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of ISO-639-1 language codes
	        for UI capture of source-language context

        '''
		return [ 'en',
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
		         'th' ]
	
	@property
	def language_labels( self ) -> Dict[ str, str ] | None:
		'''
	
	        Purpose:
	        --------
	        Method that returns a mapping of ISO-639-1 language codes
	        to human-readable labels for UI display

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
				'th': 'Thai', }
	
	@property
	def response_format_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			--------
			List[ str ] | None

		'''
		return [ 'json', 'text', 'srt', 'verbose_json', 'vtt' ]
	
	def translate( self, filepath: str, model: str = 'whisper-1', prompt: str = None,
			format: str = None, temperature: float = None, language: str = None ) -> str | None:
		"""
		
            Translate non-English speech to English.
		
		
			Parameters:
			-----------
			filepath: str
			model: str='whisper-1'
			prompt: str=None
			format: str=None
			temperature: float=None
			language: str=None
		
		
			Returns:
			--------
			str | None
        
        """
		try:
			throw_if( 'filepath', filepath )
			self.model = model or 'whisper-1'
			self.prompt = prompt
			self.response_format = format
			self.temperature = temperature
			self.target_language = language
			self.client = OpenAI( api_key=self.api_key )
			self.request = {
					'model': self.model, }
			
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
			
			if hasattr( self.response, 'text' ):
				return self.response.text
			
			if isinstance( self.response, str ):
				return self.response
			
			return str( self.response )
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
	        Method returns a list of strings representing members
	
	        Parameters:
	        ----------
	        self
	
	        Returns:
	        ---------
	        List[ str ] | None

        '''
		return [ 'temperature',
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
		         'translate',
		         'model_options' ]

class Embeddings( GPT ):
	"""
	
	    Purpose
	    ___________
	    Class used for creating vectors using OpenAI's embedding models
	
	    Parameters
	    ------------
	    None
	
	    Attributes
	    -----------
	    api_key
	    client
	    model
	    embedding
	    response
	
	    Methods
	    ------------
	    create( self, text: str ) -> get_list[ float ]


    """
	client: Optional[ OpenAI ]
	response: Optional[ CreateEmbeddingResponse ]
	model: Optional[ str ]
	input_text: Optional[ str ]
	embedding: Optional[ List[ float ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	batch_size: Optional[ int ]
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
		self.dimensions = None
		self.input_text = None
		self.encoding_format = None
		self.model = None
		self.embedding = None
		self.response = None
	
	@property
	def model_options( self ) -> List[ str ]:
		'''
		
			Returns:
			--------
			List[ str ] of embedding models

		'''
		return [ 'text-embedding-ada-002',
		         'text-embedding-3-small',
		         'text-embedding-3-large' ]
	
	@property
	def encoding_options( self ) -> List[ str ]:
		'''
			
			Returns:
			--------
			List[ str ] of available format options

		'''
		return [ 'float', 'base64' ]
	
	def create( self, text: str | List[
		str ], model: str = 'text-embedding-3-large', format: str = 'float',
			dimensions: int = None ) -> List[ float ] | List[ List[ float ] ] | None:
		"""
	
	        Purpose
	        _______
	        Creates an embedding ginve a text
	
	
	        Parameters
	        ----------
	        text: str
	
	
	        Returns
	        -------
	        get_list[ float ]

        """
		try:
			throw_if( 'text', text )
			self.input_text = text
			self.model = model
			self.encoding_format = format
			self.dimensions = dimensions
			self.client = OpenAI( api_key=self.api_key )
			if self.model == 'text-embedding-3-large' and self.dimensions is not None:
				self.response = self.client.embeddings.create( input=self.input_text, model=self.model,
					encoding_format=self.encoding_format, dimensions=self.dimensions )
			else:
				self.response = self.client.embeddings.create( input=self.input_text, model=self.model,
					encoding_format=self.encoding_format )
			if isinstance( self.input_text, list ):
				return [ item.embedding for item in self.response.data ]
			else:
				return self.response.data[ 0 ].embedding
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embedding'
			exception.method = 'create( self, text: str, model: str ) -> List[ float ]'
			raise exception
	
	def count_tokens( self, text: str, coding: str = 'cl100k_base' ) -> int:
		'''

	        Purpose:
	        -------
	        Returns the num of words in a documents path.
	
	        Parameters:
	        -----------
	        text: str - The string that is tokenized
	        coding: str - The encoding to use for tokenizing
	
	        Returns:
	        --------
	        int - The number of words

        '''
		try:
			throw_if( 'text', text )
			throw_if( 'coding', coding )
			_encoding = tiktoken.get_encoding( coding )
			_tokens = len( _encoding.encode( text ) )
			return _tokens
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embedding'
			exception.method = 'count_tokens( self, text: str, coding: str ) -> int'
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		'''

		        Purpose:
		        --------
                Method returns a list of strings representing members

		        Parameters:
		        ----------
                self

		        Returns:
		        ---------
                List[ str ] | None

        '''
		return [
				'api_key',
				'client',
				'model',
				'count_tokens',
				'input_text',
				'model_options', ]

class Files( GPT ):
	'''
		
		Purpose:
		--------
		
		Attributes:
		----------
		
	'''
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	name: Optional[ str ]
	response_format: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	file_path: Optional[ str ]
	file_id: Optional[ str ]
	purpose: Optional[ str ]
	content: Optional[ List[ Dict[ str, Any ] ] ]
	file_ids: Optional[ List[ str ] ]
	documents: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = None
		self.prompt = None
		self.purpose = None
		self.response = None
		self.file_id = None
		self.file_path = None
		self.include = [ ]
		self.content = [ ]
		self.input = [ ]
		self.tools = [ ]
		self.documents = \
			{
					'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
					'SF133.csv': 'file-WT2h2F5SNxqK2CxyAMSDg6',
					'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
					'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn'
			}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [
				'gpt-5',
				'gpt-5.2',
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-5-turbo',
				'gpt-4.1',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o',
				'gpt-4o-mini' ]
	
	@property
	def mime_options( self ) -> List[ str ]:
		'''
	
	        Purpose:
	        ________
	        Method that returns a  list of format options

        '''
		return [ 'png',
		         'jpeg',
		         'webp' ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			--------
			A List[ str ] of file purposes

		'''
		return [ 'auto', 'text', 'image', 'audio' ]
	
	@property
	def purpose_options( self ) -> List[ str ]:
		"""
		
			Purpose:
			--------
			Return supported file purpose identifiers.

			Parameters:
			-----------
			None

			Returns:
			--------
			List[str]
		
		"""
		return [ 'assistants',
		         'batch',
		         'fine-tune',
		         'vision',
		         'user_data',
		         'evals' ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of the includeable options

		'''
		return [ 'file_search_call.results',
		         'web_search_call.results',
		         'web_search_call.action.sources',
		         'message.input_image.image_url',
		         'computer_call_output.output.image_url',
		         'code_interpreter_call.outputs',
		         'reasoning.encrypted_content',
		         'message.output_text.logprobs' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of reasoning effort options

		'''
		return [ 'low',
		         'medium',
		         'high',
		         'minimal',
		         'xhigh',
		         'none' ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'auto', 'required', 'none' ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of available tools options

		'''
		return [ 'web_search',
		         'image_generation',
		         'file_search',
		         'code_interpreter',
		         'computer_use_preview' ]
	
	def upload( self, filepath: str, purpose: str = 'user_data' ) -> str | None:
		"""
	
	        Purpose
	        _______
	        Method that summarizes a document given a
	        path prompt, and a path
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	
	        Returns
	        -------
	        id: str - the id of the uploaded file.

        """
		try:
			throw_if( 'filepath', filepath )
			self.filepath = filepath
			self.purpose = purpose
			self.client = OpenAI( api_key=self.api_key )
			self.file = self.client.files.create( file=open( file=filepath, mode='rb' ),
				purpose=self.purpose )
			return self.file.id
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'upload( self, filepath: str, purpose: str=user_data ) -> str'
			raise exception
	
	def list( self, purpose: str = 'user_data' ):
		try:
			self.purpose = purpose
			self.client = OpenAI( api_key=self.api_key )
			files = self.client.files.list( purpose=self.purpose )
			out = [ ]
			for f in files:
				if purpose and getattr( f, 'purpose', None ) != purpose:
					continue
				out.append( { 'id': str( getattr( f, 'id', '' ) ),
				              'filename': str( getattr( f, 'filename', '' ) ),
				              'purpose': str( getattr( f, 'purpose', '' ) ), } )
			return out
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'list( self, prompt: str ) -> str'
			raise exception
	
	def retrieve( self, id: str ) -> Any | None:
		"""
	
	        Purpose
	        _______
	        Method that retrieves a Vector Store
	
	        Parameters
	        ----------
	        id: str
	
	        Returns
	        -------
	        Any | None

        """
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_files = self.client.files.retrieve( file_id=id )
			return _files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'retrieve( self, id: str ) -> str'
			raise exception
	
	def summarize( self, prompt: str, filepath: str, model: str = 'gpt-4.1-nano-2025-04-14' ) -> str | None:
		"""
	
	        Purpose
	        _______
	        Method that summarizes a document using the Responses API given a path, prompt, and a model
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	        model: str
	
	        Returns
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'filepath', filepath )
			self.prompt = prompt
			self.file_path = filepath
			self.model = model
			self.client = OpenAI( api_key=self.api_key )
			self.file = self.client.files.create( file=open( file=self.file_path, mode='rb' ),
				purpose='user_data' )
			self.messages = [ { 'role': 'user',
			                    'content': [ { 'type': 'file',
			                                   'file': { 'file_id': self.file.id, }, },
			                                 { 'type': 'text', 'content': self.prompt, }, ], } ]
			
			self.response = self.client.responses.create( model=self.model, input=self.messages )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'summarize( self, prompt: str, path: str, model: str ) -> str'
			raise exception
	
	def search( self, prompt: str, store_id: str, model: str = 'gpt-4.1-nano-2025-04-14' ) -> str | None:
		"""

	        Purpose:
	        _______
	        Method that searches a Vector Store using the Repsonses API given a prompt, id, and model

	        Parameters:
	        ----------
	        prompt: str
	        store_id: str
	        model: str

	        Returns:
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_id', store_id )
			self.prompt = prompt
			self.model = model
			self.vector_store_ids = [ store_id ]
			self.tools = [
					{
							'text': 'file_search',
							'vector_store_ids': self.vector_store_ids,
							'max_num_results': self.max_search_results,
					} ]
			
			self.client = OpenAI( api_key=self.api_key )
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=self.prompt )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search( self, prompt: str, store_id: str, model: str ) -> str'
			raise exception
	
	def survey( self, prompt: str, store_ids: List[ str ], model: str='gpt-4.1-nano-2025-04-14' ) -> str | None:
		"""

	        Purpose:
	        _______
	        Method that searches a Vector Store given a prompt and model using the Responses API
	
	        Parameters:
	        ----------
	        prompt: str
	        model: str
	
	        Returns:
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_ids', store_ids )
			self.prompt = prompt
			self.model = model
			self.vector_store_ids = store_ids
			self.client = OpenAI( api_key=self.api_key )
			self.vector_store_ids = list( self.vector_stores.values( ) )
			self.tools = [ { 'text': 'file_search', 'vector_store_ids': self.vector_store_ids,
			                 'max_num_results': self.max_search_results, } ]
			
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=self.prompt )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'survey( self, prompt: str, store_ids: List[ str ], model: str ) -> str'
			raise exception
	
	def extract( self, id: str ) -> str | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_files = self.client.files.content( file_id=id )
			return _files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'extract( self, id: str ) -> str'
			raise exception
	
	def delete( self, id: str ) -> None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			self.client.files.delete( file_id=id )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'delete( self, id: str ) -> FileDeleted '
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		return [ 'client',
		         'file_path',
		         'response_format',
		         'name',
		         'purpose',
		         'content',
		         'file_id',
		         'documents',
		         'retrieve',
		         'list',
		         'extract',
		         'delete',
		         'upload', ]

class VectorStores( GPT ):
	'''
		
		Purpose:
		--------
		
		Attributes:
		----------
		
	'''
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	response_format: Optional[ str ]
	name: Optional[ str ]
	store_ids: Optional[ List[ str ] ]
	store_id: Optional[ str ]
	file_path: Optional[ str ]
	file_id: Optional[ str ]
	max_results: Optional[ int ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	documents: Optional[ Dict[ str, Any ] ]
	collections: Optional[ Dict[ str, Any ] ]
	
	def __init__( self ):
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = None
		self.name = None
		self.content = None
		self.response = None
		self.store_id = None
		self.file_id = None
		self.file_path = None
		self.max_results = None
		self.collections = \
			{
					'Financial Regulations': 'vs_712r5W5833G6aLxIYIbuvVcK',
					'Public Laws': 'vs_699506f7d5348191990e0557c717fa9d',
					'Explanatory Statements': 'vs_699505df9ac48191a525c0ecb86fef66',
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
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [
				'gpt-5',
				'gpt-5.2',
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-5-turbo',
				'gpt-4.1',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o',
				'gpt-4o-mini' ]
	
	def create( self, store_name: str ) -> Any | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'store_name', store_name )
			self.name = store_name
			self.client = OpenAI( api_key=self.api_key )
			_store = self.client.vector_stores.create( name=self.store_name )
			return _store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStore'
			exception.method = 'create( self, store_name: str ) -> str'
			raise exception
	
	def list( self, store_id: str ) -> List[ Any ] | None:
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.client = OpenAI( api_key=self.api_key )
			_stores = self.client.vector_stores.files.list( vector_store_id=self.store_id )
			return list( _stores )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStore'
			exception.method = 'list( self, store_id: str ) -> Any'
			raise exception
	
	def retrieve( self, store_id: str ) -> Any | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.client = OpenAI( api_key=self.api_key )
			vector_store = self.client.vector_stores.retrieve( vector_store_id=self.store_id )
			return vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStore'
			exception.method = 'retrieve( self, id: str ) -> Any'
			raise exception
	
	def search( self, prompt: str, store_id: str, model: str = 'gpt-4.1-nano-2025-04-14' ) -> str | None:
		"""

	        Purpose:
	        _______
	        Method that analyzeses an image given a prompt,

	        Parameters:
	        ----------
	        prompt: str
	        url: str

	        Returns:
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_id', store_id )
			self.prompt = prompt
			self.model = model
			self.vector_store_ids = [ store_id ]
			self.tools = [
					{
							'text': 'file_search',
							'vector_store_ids': self.vector_store_ids,
							'max_num_results': self.max_search_results,
					} ]
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=self.prompt )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStore'
			exception.method = ('search(self, prompt: str, store_id: str, '
			                    'model: str=gpt-4.1-nano) -> str')
			raise exception
	
	def survey( self, prompt: str, store_ids: List[ str ] = None, results: int = 10,
			model: str = 'gpt-4.1-nano' ) -> str | None:
		"""

	        Purpose:
	        _______
	        Method that analyzeses an image given a prompt,

	        Parameters:
	        ----------
	        prompt: str
	        url: str

	        Returns:
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'store_id', store_ids )
			self.prompt = prompt
			self.model = model
			self.vector_store_ids = store_ids
			self.max_results = results
			self.tools = [
					{
							'text': 'file_search',
							'vector_store_ids': self.vector_store_ids,
							'max_num_results': self.max_search_results,
					} ]
			self.client = OpenAI( api_key=self.api_key )
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=self.prompt )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStore'
			exception.method = ('survey( self, prompt: str, store_ids: List[ str ], '
			                    'results: int=10, model: str=gpt-4.1-nano )->str')
			raise exception
	
	def update( self, store_id: str, filename: str ) -> VectorStore | None:
		try:
			throw_if( 'store_id', store_id )
			throw_if( 'filename', filename )
			self.store_id = store_id
			self.name = filename
			self.client = OpenAI( api_key=self.api_key )
			vector_store = self.client.vector_stores.update( vector_store_id=self.store_id, name=self.name )
			return vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStore'
			exception.method = 'update( self, store_id: str, filename: str )'
			raise exception
	
	def delete( self, store_id: str ) -> None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'store_id', store_id )
			self.store_id = store_id
			self.client = OpenAI( api_key=self.api_key )
			self.client.vector_stores.delete( vector_store_id=self.store_id )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStore'
			exception.method = 'delete( self, id: str )'
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		return [ 'client',
		         'file_path',
		         'response_format',
		         'name',
		         'content',
		         'file_id',
		         'collections',
		         'retrieve',
		         'list',
		         'extract',
		         'delete',
		         'update', ]
		