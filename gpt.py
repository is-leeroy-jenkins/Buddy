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
from typing import Optional, List, Dict
from openai.types.responses import Response
from openai.types import CreateEmbeddingResponse, VectorStore
from boogr import Error
import config as cfg

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
	response_format: Optional[ str ]
	context: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	
	def __init__( self, model: str='gpt-5-nano', prompt: str=None, temperature: float=None,
			top_p: float=None, presence: float=None, store: bool=None, stream: bool=None,
			stops: List[ str ]=[ ], format: str=None, number: int=None, instruct: str=None,
			context: List[ Dict[ str, str ] ]=[ ], background: bool=None,
			max_tokens: int=None, frequency: float=None ):
		self.api_key = cfg.OPENAI_API_KEY
		self.model = model
		self.client = None
		self.number = number
		self.stops = stops
		self.response_format = format
		self.number = number
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.prompt = prompt
		self.store = store
		self.stream = stream
		self.background = background
		self.instructions = instruct
		self.context = context

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
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
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
	response: Optional[ openai.types.responses.Response ]
	file: Optional[ openai.types.file_object.FileObject ]
	purpose: Optional[ str ]
	domains: Optional[ str ]
	
	def __init__( self, model: str='gpt-5-nano', prompt: str=None, temperature: float=None,
			top_p: float=None, presense: float=None, store: bool=None, stream: bool=None,
			stops: List[ str ]=[ ], format: str=None, number: int=None,
			instruct: str=None, context: List[ Dict[ str, str ] ]=[ ], domains: List[ str ]=[ ],
			include: List[ Dict[ str, str ] ]=[ ], tools: List[ Dict[ str, str ] ]=[ ],
			max_tools: Optional[ int ]=None, tool_choice: Optional[ str ]=None, file_path: str=None,
			background: bool=None, is_parallel: bool=None, max_tokens: int=None, frequency: float=None,
			input: List[ Dict[ str, str ] ]=[ ], file_ids: List[ str ]=[ ], previous_id: str=None,
			reasoning: Dict[ str, str ]={}, output_text: str=None, max_search_results: Optional[ int ]=None,
			content: str=None, vector_store_ids: Optional[ List[ str ] ]=None ):
		super( ).__init__( model, prompt, temperature, top_p, presense, store, stream, stops,
			format, number, instruct, context, background, max_tokens, frequency )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.prompt = prompt
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presense
		self.max_tokens = max_tokens
		self.context = context
		self.stream = stream
		self.store = store
		self.instructions = instruct
		self.stops = stops
		self.background = background
		self.conetxt = context
		self.response_format = format
		self.input = input
		self.include = include
		self.output_text = output_text
		self.max_tools = max_tools
		self.allowed_domains = is_parallel
		self.vector_store_ids = vector_store_ids
		self.file_ids = file_ids
		self.tools = tools
		self.domains = domains
		self.previous_id = previous_id
		self.reasoning = reasoning
		self.parallel_tools = is_parallel
		self.tool_choice = tool_choice
		self.response = None
		self.file = None
		self.file_url = file_path
		self.image_url = None
		self.content = content
		self.output_text = None
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
	def purpose_options( self ) -> List[ str ] | None:
		'''
		
			Returns:
			--------
			A List[ str ] of file purposes

		'''
		return [ 'assistants',
		         'batch',
		         'fine-tune',
		         'vision',
		         'user_data',
		         'evals' ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		'''

			Returns:
			--------
			A List[ str ] of reasoning effort options

		'''
		return [ 'none',
		         'low',
		         'medium',
		         'high',
		         'minimal',
		         'xhigh' ]
	
	def generate_text( self, prompt: str, model: str, temperature: float=None,
			top_p: float=None, frequency: float=None, presence: float=None, max_tokens: int=None,
			store: bool=None, stream: bool=None, instruct: str=None, background: bool=False,
			reasoning: str=None, include: str=None  ) -> str | None:
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
			self.reasoning = {'effort': reasoning }
			self.input = self.prompt
			self.include = include
			self.client = OpenAI( api_key=self.api_key )
			if self.model.startswith( 'gpt-5' ):
				self.response = self.client.responses.create( model=self.model, input=self.input,
					reasoning=self.reasoning )
			else:
				self.response = self.client.responses.create( model=self.model, input=self.input,
					max_output_tokens=self.max_tokens, temperature=self.temperature,
					top_p=self.top_percent )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str )'
			raise  exception
	
	def generate_image( self, prompt: str, number: int=None, model: str=None,
			size: str=None, quality: str=None, fmt: str=None,  ) -> str | None:
		'''
	
	        Purpose
	        _______
	        Generates an image given a prompt
	
	
	        Parameters
	        ----------
	        prompt: str

	
	        Returns
	        -------
	        str | None

        '''
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.number = number
			self.model = model
			self.size = size
			self.quality = quality
			self.response_format = fmt
			self.client = OpenAI( api_key=self.api_key )
			self.response = self.client.images.generate( model=self.model, prompt=self.prompt,
				size=self.size, quality=self.quality, response_format=self.response_format,
				n=self.number )
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = ('generate_image( self, prompt: str ) -> str | None')
			raise exception
	
	def analyze_image( self, prompt: str, url: str ) -> str | None:
		"""

	        Purpose
	        _______
	        Analyze an image with a text instruction.
	
	        Parameters
	        ----------
	        prompt: str
	        url: str
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'url', url )
			self.prompt = prompt
			self.image_url = url
			self.input = [
			{
				'role': 'user',
				'content': [
				{
					'type': 'input_text',
					'text': self.prompt
				},
				{
					'type': 'input_image',
					'image_url': self.image_url
				}, ],
			} ]
			
			self.client = OpenAI( api_key=self.api_key )
			self.response = self.client.responses.create( model=self.model, input=self.input )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
	
	def edit_image( self, prompt: str, src_url: str, dest_path: str ) -> str | None:
		'''
			
			Purpose:
			--------
			
			
			Parameters:
			---------
			prompt: str - instructions guiding the LLM
			src_url: str - The path to the source image
			dest_path: str - name of the edited image
	
			Returns:
			----------
			
			
		'''
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'src_url', src_url )
			throw_if( 'dest_path', dest_path )
			self.prompt = prompt
			self.client = OpenAI( api_key=self.api_key )
			_source = src_url
			_url = dest_path
			
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'analyze_image( self, prompt: str, url: str )'
			error = ErrorDialog( exception )
			error.show( )
			
	def summarize_document( self, prompt: str, pdf_path: str) -> str | None:
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
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'pdf_path', pdf_path )
			self.prompt = prompt
			self.file_path = pdf_path
			self.file = self.client.files.create( file=open( file=self.file_path, mode='rb' ),
				purpose='user_data' )
			self.messages = [
			{
				'role': 'user',
				'content': [
				{
					'type': 'file',
					'file':
					{
						'file_id': self.file.id,
					},
				},
				{
					'type': 'text',
					'text': self.prompt,
				},],
			}]
			
			self.response = self.client.responses.create( model=self.model, input=self.messages )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'summarize_document( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def search_web( self, prompt: str, model: str='gpt-4.1-nano-2025-04-14',
			recency: int=30, max_results: int=100, ) -> str | None:
		"""

	        Purpose
	        _______
	        Method that analyzeses an image given a prompt,
	
	        Parameters
	        ----------
	        prompt: str
	
	        Returns
	        -------
	        str | None

        """
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.model = model
			self.search_recency = recency
			self.max_search_results = max_results
			self.web_options = { 'search_recency_days': self.search_recency,
			                     'max_search_results': self.max_search_results }
			self.messages = [
			{
				'role': 'user',
				'content': self.prompt,
			}]
			
			self.response = self.client.responses.create( model=self.model,
				web_search_options=self.web_options, input=self.messages )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = ('search_web( self, prompt: str, model: str, '
			                    'recency: int=30, max_results: int=8 ) -> str | None')
			error = ErrorDialog( exception )
			error.show( )

	def upload_file( self, filepath: str, purpose: str='user_data' ) -> str | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
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
			exception.cause = 'Chat'
			exception.method = 'upload_file( self, filepath: str, purpose: str=user_data ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def retrieve_file( self, id: str ) -> List[ str ] | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_files = self.client.files.retrieve( file_id=id )
			return _files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'retrieve_file( self, id: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def retrieve_files( self, purpose: str = "user_data" ):
		self.purpose = purpose
		self.client = OpenAI( api_key=self.api_key )
		page = self.client.files.list( )  # no purpose arg
		files = getattr( page, "data", None ) or page
		out = [ ]
		for f in files:
			if purpose and getattr( f, "purpose", None ) != purpose:
				continue
			out.append( {
					"id": str( getattr( f, "id", "" ) ),
					"filename": str( getattr( f, "filename", "" ) ),
					"purpose": str( getattr( f, "purpose", "" ) ),
			} )
		return out
	
	def retrieve_content( self, id: str ) -> str | None:
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
			exception.cause = 'Chat'
			exception.method = 'retrieve_file( self, id: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete_file( self, id: str ) -> bool | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_deleted = self.client.files.delete( file_id=id )
			return bool( _deleted )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'delete_file( self, id: str ) -> FileDeleted '
			error = ErrorDialog( exception )
			error.show( )
	
	def create_store( self, store_name: str ) -> VectorStore | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'store_name', store_name )
			self.client = OpenAI( api_key=self.api_key )
			_store = self.client.vector_stores.create( name=store_name )
			return _store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'create_store( self, id: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def retrieve_store( self, id: str ) -> VectorStore | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			vector_store = self.client.vector_stores.retrieve( vector_store_id=id )
			return vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'retrieve_store( self, purpose: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def delete_store( self, id: str ) -> bool | None:
		'''
			
			Returns:
			--------
			A List[ str ] of file_ids

		'''
		try:
			throw_if( 'id', id )
			self.client = OpenAI( api_key=self.api_key )
			_deleted = self.client.vector_stores.delete( vector_store_id=id )
			return bool( _deleted )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'retrieve_store( self, purpose: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
		
	def __dir__( self ) -> List[ str ] | None:
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'system_instructions',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'content',
		         'prompt',
		         'response',
		         'completion',
		         'file',
		         'path',
		         'messages',
		         'image_url',
		         'response_format',
		         'tools',
		         'vector_store_ids',
		         'name',
		         'id',
		         'description',
		         'get_format_options',
		         'get_model_options',
		         'reasoning_effort',
		         'purpose_options',
		         'input_text',
		         'metadata',
		         'get_files',
		         'get_data',
		         'dump',
		         'translate',
		         'transcribe',
		         'generate_text',
		         'generate_image',
		         'analyze_image',
		         'edit_image',
		         'summarize_document',
		         'search_web',
		         'search_files',
		         'retrieve_file',
		         'retrieve_files',
		         'retrieve_content',
		         'delete_file',
		         'upload_file', ]

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
	output_format: Optional[ str ]
	background: Optional[ bool ]
	backcolor: Optional[ str ]
	
	def __init__( self, prompt: str=None, model: str='gpt-image-1', temperature: float=None,
			top_p: float=None, presence: float=None, frequency: float=None,
			max_tokens: int=None, store: bool=None, stream: bool=False,  backcolor: str=None,
			instruct: str=None, background: bool=None, messages: List[ Dict[ str, str ] ]=[ ],
			format: str=None,  number: int=None, include: List[ Dict[ str, str ] ]=[ ],
			tools: List[ Dict[ str, str ] ]=[ ], max_tools: Optional[ int ]=None,
			tool_choice: Optional[ str ]=None, image_path: str=None, is_parallel: bool=None,
			input: List[ Dict[ str, str ] ]=[ ], previous_id: str=None,
			reasoning: Dict[ str, str ]=[ ],  input_text: str=None, image_url: str=None,
			content: List[ Dict[ str, str ] ]=[ ], quality: str=None, size: str=None,
			detail: str=None, style: str=None ):
		super( ).__init__( model, prompt, temperature, top_p, presence, store, stream,
			format, number, instruct, messages, background, max_tokens, frequency )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
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
		self.model = None
		self.size = size
		self.style = style
		self.response_format = format
		self.output_format = None
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
		         'gpt-4o-mini' ]
	
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
	def output_options( self ) -> List[ str ]:
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
		return [ 'transparent',
		         'opaque',
		         'auto' ]
	
	@property
	def quality_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Method that returns a  list of quality options

        '''
		return [ 'standard', 'hd' ]
	
	@property
	def detail_options( self ) -> List[ str ]:
		'''

	        Purpose:
	        ________
	        Method that returns a  list of detail options

        '''
		return [ 'auto',
		         'low',
		         'high' ]
	
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
	
	def generate( self, prompt: str, number: int=1, model: str ='dall-e-3',
			size: str='1024x1024', quality: str='standard', fmt: str = '.png' ) -> str | None:
		'''
	
	        Purpose
	        _______
	        Generates an image given a prompt
	
	
	        Parameters
	        ----------
	        prompt: str

	
	        Returns
	        -------
	        str | None

        '''
		try:
			throw_if( 'prompt', prompt )
			self.prompt = prompt
			self.number = number
			self.model = model
			self.size = size
			self.quality = quality
			self.response_format = fmt
			self.client = OpenAI( api_key=self.api_key )
			self.response = self.client.images.generate( model=self.model, prompt=self.prompt,
				size=self.size, quality=self.quality, response_format=self.response_format,
				n=self.number )
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = ('generate_image( self, prompt: str ) -> str | None')
			raise exception
	
	def analyze( self, text: str, path: str, model: str='gpt-4o-mini', ) -> str:
		'''
	
	        Purpose:
	        ________
	
	        Method providing image analysis functionality given a prompt and path
	
	        Parameters:
	        ----------
	        input: str
	        path: str
	
	        Returns:
	        --------
	        str | None

        '''
		try:
			throw_if( 'text', text )
			throw_if( 'path', path )
			self.input_text = text
			self.model = model
			self.file_path = path
			self.input = [
					{
							'role': 'user',
							'content': [
									{ 'type': 'input_text', 'text': self.input_text },
									{ 'type': 'input_image', 'image_url': self.file_path },
							],
					} ]
			
			self.client = OpenAI( api_key=self.api_key )
			self.response = self.client.responses.create( model=self.model, input=self.input,
				max_output_tokens=self.max_completion_tokens, temperature=self.temperature,
				tool_choice=self.tool_choice, stream=self.stream, store=self.store )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Image'
			exception.method = 'analyze( self, path: str, text: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
	def edit( self, prompt: str, path: str, size: str='1024x1024' ) -> str:
		"""

	        Purpose
	        _______
	        Method that analyzeses an image given a path prompt,
	
	        Parameters
	        ----------
	        prompt: str
	        url: str
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'input', prompt )
			throw_if( 'path', path )
			self.input_text = prompt
			self.file_path = path
			self.size = size
			self.client = OpenAI( api_key=self.api_key )
			self.response = self.client.images.edit( model=self.model,
				image=open( self.file_path, 'rb' ), prompt=self.input_text, n=self.number,
				size=self.size, )
			return self.response.data[ 0 ].url
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Image'
			exception.method = 'edit( self, text: str, path: str, size: str=1024x1024 ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
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

class TTS(  ):
	"""
	
	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (TTS)
	
	
	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    max: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    self.api_key, self.system_instructions, self.client, self.small_model, self.reasoning_effort,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.input_text, self.response, self.completion, self.file, self.path,
	    self.messages, self.image_url, self.response_format,
	    self.tools, self.vector_store_ids, self.descriptions, self.assistants
	
	    Methods
	    ------------
	    get_model_options( self ) -> str
	    create_small_embedding( self, prompt: str, path: str )

    """
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	speed: Optional[ float ]
	voice: Optional[ str ]
	input: Optional[ str ]
	instructions: Optional[ str ]
	response: Optional[ Response ]
	streamed_response: Optional[ SpeechWithStreamingResponse ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	response_format: Optional[ str ]
	file_path: Optional[ str ]
	
	def __init__( self, input: str=None, model: str='gpt-4o-mini-tts',  format: str=None,
			instruct: str=None, voice: str=None, speed: float=None, file_path: str=None  ):
		'''

	        Purpose:
	        --------
	        Constructor to  create_small_embedding TTS objects

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
	
	@property
	def model_options( self ) -> List[ str ] | None:
		'''
	
	        Purpose:
	        --------
	        Methods that returns a list of tts model names

        '''
		return [ 'gpt-4o-mini-tts',
		         'tts-1',
		         'tts-1-hd' ]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'alloy',
		         'verse',
		         'ballad',
		         'aria',
		         'sol',
		         'luna',
		         'nova',
		         'sage', ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of image formats

        '''
		return [ 'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm' ]
	
	@property
	def speed_options( self ) -> List[ float ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of floats
	        representing different audio speeds

        '''
		return [ 0.25, 1.0, 4.0 ]
	
	def create_speech( self, text: str, file_path: str, model: str='gpt-4o-tts', format: str=None,
			speed: float=None, voice: str=None, ):
		"""
	
	        Purpose
	        _______
	        Generates audio given a text prompt less than
	        4096 characters and a path to audio file
	
	
	        Parameters
	        ----------
	        prompt: str
	        path: str
	
	
	        Returns
	        -------
	        str

        """
		try:
			throw_if( 'text', text )
			throw_if( 'file_pat', file_path )
			
			self.input = text
			self.speed = speed
			self.model = model
			self.response_format = format
			self.voice = voice
			self.file_path = file_path
			self.client = OpenAI( api_key=self.api_key )
			with self.client.audio.speech.with_streaming_response.create( model=self.model,
					speed=self.speed, voice=self.voice, response_format=self.response_format,
					input=self.input ) as response:
				response.stream_to_file( self.file_path )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, prompt: str, path: str ) -> str'
			error = ErrorDialog( exception )
			error.show( )
	
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
		return [ 'input', 'file_path', 'voice', 'client', 'response_formaat',
		         'speed', 'model', 'instructions', 'create_speech' ]

class Transcription( GPT ):
	"""
	
	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (whisper-1)
	
	
	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    max: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    self.api_key, self.system_instructions, self.client, self.small_model, self.reasoning_effort,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.input_text, self.response, self.completion, self.audio_file, self.transcript
	
	
	    Methods
	    ------------
	    get_model_options( self ) -> str
	    create_small_embedding( self, path: str  ) -> str


    """
	client: Optional[ OpenAI ]
	speed: Optional[ float ]
	voice: Optional[ str ]
	language: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	
	def __init__( self, model: str='gpt-4o-transcribe', temperature: float=None, prompt: str=None,
			number: int=None, top_p: float=None, frequency: float=None, presence: float=None,
			max_tokens: int=None, stream: bool=None, store: bool=None, language: str=None,
			instruct: str=None, format: str=None,  background: bool=None,
			messages: List[ Dict[ str, str ] ]=None, stops: List[ str ]=None  ):
		super( ).__init__( model, prompt, temperature, top_p, presence, store, stream, stops,
			format, number, instruct, messages, background, max_tokens, frequency )
		self.api_key = cfg.OPENAI_API_KEY
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.store = store
		self.language = language
		self.instructions = instruct
		self.model = model
		self.input_text = None
		self.audio_file = None
		self.transcript = None
		self.response = None
	
	@property
	def model_options( self ) -> str:
		'''

	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [ 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'whisper-1',
		         'gpt-4o-transcribe-diarize', 'gpt-4o-transcribe-diarize' ]
	
	@property
	def output_options( self ) -> List[ str ] | None:
		'''

	        Purpose:
	        --------
	        Method that returns a list of image formats

        '''
		return [ 'mp3',
		         'wav',
		         'aac',
		         'flac',
		         'opus',
		         'pcm' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		'''
			
			Returns:
			-------
			List[ str ] output  format options
			
		'''
		return [ 'json',
		         'text',
		         'srt',
		         'verbose_json',
		         'vtt',
		         'diarized_json' ]
	
	@property
	def language_options( self ):
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of voice names

        '''
		return [ 'english',
		         'spanish',
		         'french',
		         'german',
		         'italian',
		         'portuguese',
		         'russian',
		         'ukrainian',
		         'greek',
		         'hebrew',
		         'arabic',
		         'hindi',
		         'chinese',
		         'japanese',
		         'korean',
		         'vietnamese',
		         'thai' ]
	
	def transcribe( self, path: str, model: str='gpt-4o-transcribe', language: str=None, ) -> str:
		"""
		
			Purpose:
			----------
            Transcribe audio with Whisper.
        
        """
		try:
			throw_if( 'path', path )
			self.model = model
			self.language = language
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			with open( path, 'rb' ) as self.audio_file:
				resp = self.client.audio.transcriptions.create( model=self.model,
					file=self.audio_file, language=self.language )
			return resp.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'boo'
			ex.cause = 'Transcription'
			ex.method = 'transcribe(self, path)'
			error = ErrorDialog( ex )
			error.show( )
	
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
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'prompt',
		         'response',
		         'audio_file',
		         'messages',
		         'response_format',
		         'api_key',
		         'client',
		         'input_text',
		         'transcript', ]

class Translation( GPT ):
	"""

	    Purpose
	    ___________
	    Class used for interacting with OpenAI's TTS API (whisper-1)
	
	
	    Parameters
	    ------------
	    num: int=1
	    temp: float=0.8
	    top: float=0.9
	    freq: float=0.0
	    pres: float=0.0
	    max: int=10000
	    store: bool=True
	    stream: bool=True
	
	    Attributes
	    -----------
	    self.api_key, self.system_instructions, self.client, self.small_model,  self.reasoning_effort,
	    self.response, self.num, self.temperature, self.top_percent,
	    self.frequency_penalty, self.presence_penalty, self.max_completion_tokens,
	    self.store, self.stream, self.modalities, self.stops, self.content,
	    self.input_text, self.response, self.completion, self.file, self.path,
	    self.messages, self.image_url, self.response_format,
	    self.tools, self.vector_store_ids, self.descriptions, self.assistants
	
	    Methods
	    ------------
	    create_small_embedding( self, prompt: str, path: str )

    """
	client: Optional[ OpenAI ]
	target_language: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	input: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	response_format: Optional[ str ]
	
	def __init__( self, model: str=None, temperature: float=None, top_p: float=None,
			frequency: float=None, presence: float=None, max_tokens: int=None, store: bool=None,
			stream: bool=None, instruct: str=None, audio_file: str=None, format: str=None,
			language: str=None ):
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
	def model_options( self ) -> str:
		'''
	
	        Purpose:
	        --------
	        Methods that returns a list of small_model names

        '''
		return [ 'gpt-5',
		         'gpt-5-mini',
		         'gpt-5-turbo',
		         'gpt-4o',
		         'gpt-4.1',
		         'gpt-4.1-mini',
		         'gpt-4.1-turbo',
		         'gpt-4o',
		         'gpt-4o-mini', ]
	
	@property
	def language_options( self ):
		'''
	
	        Purpose:
	        --------
	        Method that returns a list of translatable languages

        '''
		return [ 'english',
		         'spanish',
		         'french',
		         'german',
		         'italian',
		         'portuguese',
		         'russian',
		         'ukrainian',
		         'greek',
		         'hebrew',
		         'arabic',
		         'hindi',
		         'chinese',
		         'japanese',
		         'korean',
		         'vietnamese',
		         'thai' ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		'''
			
			Returns:
			-------
			List[ str ] output  format options
			
		'''
		return [ 'json',
		         'text',
		         'srt',
		         'verbose_json',
		         'vtt',
		         'diarized_json' ]
	
	def translate( self, prompt: str, filepath: str, model: str=None,
			temperature: float=None, top_p: float=None, frequency: float=None,
			presence: float=None, max_tokens: int=None, format: str=None, language: str=None,
			store: bool=True, stream: bool=True, instruct: str=None ) -> str | None:
		"""
		
            Translate non-English speech to English with Whisper.
        
        """
		try:
			throw_if( 'path', filepath )
			throw_if( 'prompt', prompt )
			self.model = model
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = max_tokens
			self.store = store
			self.stream = stream
			self.instruct = instruct
			self.response_format = format
			self.target_language = language
			self.client = OpenAI( api_key=self.api_key )
			with open( filepath, 'rb' ) as audio_file:
				self.response = self.client.audio.translations.create( model=self.model,
					file=audio_file )
			return resp.text
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Translation'
			ex.method = 'translate(self, path)'
			error = ErrorDialog( ex )
			error.show( )
	
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
		return [ 'num',
		         'temperature',
		         'top_percent',
		         'frequency_penalty',
		         'presence_penalty',
		         'max_completion_tokens',
		         'store',
		         'stream',
		         'modalities',
		         'stops',
		         'prompt',
		         'response',
		         'audio_path',
		         'path',
		         'messages',
		         'response_format',
		         'tools',
		         'api_key',
		         'client',
		         'model',
		         'translate',
		         'model_options', ]

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
	
	def create( self, text: str | List[ str ], model: str='text-embedding-3-large', format: str='float',
			dimensions: int=None ) -> List[ float ] | List[ List[ float ] ] | None:
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
	
	def count_tokens( self, text: str, coding: str='cl100k_base' ) -> int:
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
			error = ErrorDialog( exception )
			error.show( )
	
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
		return [ 'none',
		         'low',
		         'medium',
		         'high',
		         'minimal',
		         'xhigh' ]
	
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
	
	def upload( self, filepath: str, purpose: str='user_data' ) -> str | None:
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
			raise  exception
	
	def list( self, purpose: str='user_data' ):
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
	
	def summarize( self, prompt: str, filepath: str, model: str='gpt-4.1-nano-2025-04-14' ) -> str | None:
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
			self.file = self.client.files.create( file=open( file=self.file_path, mode='rb' ),
				purpose='user_data' )
			self.messages = [{'role': 'user',
			                  'content': [{ 'type': 'file',
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
	
	def search( self, prompt: str, store_id: str, model: str='gpt-4.1-nano-2025-04-14' ) -> str | None:
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
	
	def __init__( self  ):
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
			error = ErrorDialog( exception )
			error.show( )
	
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
			error = ErrorDialog( exception )
			error.show( )
	
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
			error = ErrorDialog( exception )
			error.show( )
	
	def search( self, prompt: str, store_id: str, model: str='gpt-4.1-nano-2025-04-14' ) -> str | None:
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
			error = ErrorDialog( exception )
			error.show( )
	
	def survey( self, prompt: str, store_ids: List[ str ]=None, results: int=10,
			model: str='gpt-4.1-nano' ) -> str | None:
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
			self.response = self.client.responses.create( model=self.model, tools=self.tools,
				input=self.prompt )
			return self.response.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStore'
			exception.method = ('survey( self, prompt: str, store_ids: List[ str ], '
			                    'results: int=10, model: str=gpt-4.1-nano )->str')
			error = ErrorDialog( exception )
			error.show( )
	
	def update( self, store_id: str, filename: str ) -> None:
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
			error = ErrorDialog( exception )
			error.show( )
	
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
			error = ErrorDialog( exception )
			error.show( )
	
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
		